"""
convert_hf_weights_to_openvla_prismatic.py

Utility script for converting OpenVLA weights saved in Hugging Face format (via the
`convert_openvla_weights_to_hf.py` script) back into the Prismatic training/run format.

This reconstructs a Prismatic-like run directory:
  <output_prismatic_run_dir>/
    - config.json                  (minimal stub; see notes below)
    - dataset_statistics.json      (copied from HF folder or reconstructed from HF config.norm_stats)
    - checkpoints/latest-checkpoint.pt
        -> checkpoint dict with:
           {
             "model": {
               "vision_backbone": { ...featurizer keys... },
               "projector": { ... MLP keys ... },
               "llm_backbone": { ... LLM keys ... },
               "downsampler": {}
             }
           }

Usage:
    python vla-scripts/extern/convert_hf_weights_to_openvla_prismatic.py \
        --hf_model_path_or_id <PATH OR HF HUB ID> \
        --output_prismatic_run_dir <OUTPUT RUN DIR> \
        [--base_vlm <PRISMATIC BASE VLM NAME>]
"""

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union, Optional

import draccus
import torch

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction


# Mapping used in the original conversion (Prismatic -> HF); we invert it here.
PROJECTOR_KEY_MAPPING = {
    "projector.0.weight": "projector.fc1.weight",
    "projector.0.bias": "projector.fc1.bias",
    "projector.2.weight": "projector.fc2.weight",
    "projector.2.bias": "projector.fc2.bias",
    "projector.4.weight": "projector.fc3.weight",
    "projector.4.bias": "projector.fc3.bias",
}
REVERSE_PROJECTOR_KEY_MAPPING = {v: k for k, v in PROJECTOR_KEY_MAPPING.items()}


@dataclass
class HFToPrismaticConvertConfig:
    # Path or HF Hub ID to load the huggingface-format OpenVLA model
    hf_model_path_or_id: Union[str, Path] = "hf-convert/openvla-7b"

    # Output run directory (Prismatic-style)
    output_prismatic_run_dir: Path = Path("runs/recovered-openvla-7b")

    # Optional: base_vlm string for Prismatic config.json. If None, we will use hf_config.arch_specifier.
    # If you know the original base_vlm (e.g., "prism-dinosiglip-224px+mx-oxe-magic-soup-plus+n8+b32+x7"),
    # pass it here for maximal compatibility with Prismatic's ModelConfig registry.
    base_vlm: Optional[str] = None

    # HF Hub token for gated models (optional).
    hf_token: Union[str, Path, None] = Path(".hf_token")

    def __post_init__(self) -> None:
        if isinstance(self.hf_token, Path):
            if self.hf_token.exists():
                self.hf_token = self.hf_token.read_text().strip()
            else:
                # If token file doesn't exist, set to None
                self.hf_token = None


def remap_state_dicts_from_hf(
    hf_state_dict: Dict[str, torch.Tensor],
    use_fused_vision_backbone: bool,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Reverse the HF mapping to recover Prismatic-format state dict sections:
      - vision_backbone
      - projector
      - llm_backbone
      - downsampler (empty)
    """
    prismatic_vision_backbone_sd: Dict[str, torch.Tensor] = {}
    prismatic_projector_sd: Dict[str, torch.Tensor] = {}
    prismatic_llm_sd: Dict[str, torch.Tensor] = {}

    for key, val in hf_state_dict.items():
        # Vision backbone
        if key.startswith("vision_backbone."):
            subkey = key[len("vision_backbone."):]  # e.g., "featurizer.blocks.0...." or "fused_featurizer...."

            if use_fused_vision_backbone:
                # HF: featurizer. -> Prismatic: dino_featurizer.
                # HF: fused_featurizer. -> Prismatic: siglip_featurizer.
                if subkey.startswith("featurizer."):
                    subkey = "dino_featurizer." + subkey[len("featurizer.") :]
                elif subkey.startswith("fused_featurizer."):
                    subkey = "siglip_featurizer." + subkey[len("fused_featurizer.") :]
            else:
                # Non-fused: keep as 'featurizer.'
                # Nothing else to do.
                pass

            # Convert LayerScale param name back from '.scale_factor' to '.gamma' for DINO (safe even if not present)
            if ".scale_factor" in subkey:
                subkey = subkey.replace(".scale_factor", ".gamma")

            prismatic_vision_backbone_sd[subkey] = val

        # Projector
        elif key.startswith("projector."):
            # Map HF keys back to Prismatic MLP keys
            if key not in REVERSE_PROJECTOR_KEY_MAPPING:
                raise KeyError(f"Unexpected projector key in HF checkpoint: {key}")
            prismatic_key = REVERSE_PROJECTOR_KEY_MAPPING[key]
            prismatic_projector_sd[prismatic_key] = val

        # LLM backbone
        elif key.startswith("language_model."):
            prismatic_key = "llm." + key[len("language_model.") :]
            prismatic_llm_sd[prismatic_key] = val

        # Ignore other keys (if any)
        else:
            continue

    return {
        "vision_backbone": prismatic_vision_backbone_sd,
        "projector": prismatic_projector_sd,
        "llm_backbone": prismatic_llm_sd,
        "downsampler": {},  # expected to be empty / unused
    }


def write_minimal_prismatic_config(
    out_dir: Path,
    hf_config: OpenVLAConfig,
    base_vlm_override: Optional[str] = None,
) -> Path:
    """
    Write a minimal Prismatic-style config.json sufficient for downstream tooling.
    If base_vlm_override is provided, it will be used as vla.base_vlm.
    Otherwise, hf_config.arch_specifier is used.
    """
    vla_config = {
        # NOTE: Prismatic conversion to HF uses vla.base_vlm to instantiate a ModelConfig.
        # If you know the original base_vlm string, pass it to preserve exact behavior.
        "base_vlm": base_vlm_override or hf_config.arch_specifier,
        # Helpful extra fields (not strictly required, but informative)
        "arch_specifier": hf_config.arch_specifier,
        "vision_backbone_id": hf_config.vision_backbone_id,
        "llm_backbone_id": hf_config.llm_backbone_id,
        "image_resize_strategy": hf_config.image_resize_strategy,
        "llm_max_length": hf_config.llm_max_length,
        "use_fused_vision_backbone": hf_config.use_fused_vision_backbone,
    }

    cfg = {
        "vla": vla_config,
        "_note": (
            "This config was reconstructed from a Hugging Face OpenVLA checkpoint. "
            "If prismatic tooling requires an exact base_vlm, supply it via --base_vlm when converting."
        ),
    }

    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return config_path


def write_dataset_statistics(out_dir: Path, hf_model_path_or_id: Union[str, Path], hf_config: OpenVLAConfig) -> Path:
    """
    Ensure dataset_statistics.json exists in the out_dir.
    Prefer copying from the HF model directory if present; otherwise, reconstruct from hf_config.norm_stats.
    """
    # Try to find local file if hf_model_path_or_id is a directory
    ds_out = out_dir / "dataset_statistics.json"
    if isinstance(hf_model_path_or_id, (str, Path)) and Path(hf_model_path_or_id).is_dir():
        src = Path(hf_model_path_or_id) / "dataset_statistics.json"
        if src.exists():
            shutil.copyfile(src, ds_out)
            return ds_out

    # Reconstruct from hf_config.norm_stats
    if getattr(hf_config, "norm_stats", None) is None:
        # If not present, write an empty stub (prismatic conversion to HF used this)
        norm_stats = {}
    else:
        norm_stats = hf_config.norm_stats

    with open(ds_out, "w") as f:
        json.dump(norm_stats, f, indent=2)
    return ds_out


@draccus.wrap()
def convert_hf_weights_to_openvla_prismatic(cfg: HFToPrismaticConvertConfig) -> None:
    print(f"[*] Converting HF OpenVLA Model `{cfg.hf_model_path_or_id}` back to Prismatic format")

    # Create output run directory structure
    run_dir = Path(cfg.output_prismatic_run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load HF config and model
    print("[*] Loading HF OpenVLAConfig and OpenVLAForActionPrediction")
    hf_config = OpenVLAConfig.from_pretrained(cfg.hf_model_path_or_id, token=cfg.hf_token)
    # Load on CPU, keep bf16 to reduce memory if available
    hf_model = OpenVLAForActionPrediction.from_pretrained(
        cfg.hf_model_path_or_id,
        token=cfg.hf_token,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    hf_model.eval()
    hf_state_dict = hf_model.state_dict()

    # Map HF state dict back to Prismatic layout
    print("[*] Remapping HF state_dict to Prismatic sub-dicts")
    prismatic_model_state = remap_state_dicts_from_hf(
        hf_state_dict=hf_state_dict, use_fused_vision_backbone=hf_config.use_fused_vision_backbone
    )

    # Sanity checks
    assert "vision_backbone" in prismatic_model_state and len(prismatic_model_state["vision_backbone"]) > 0
    assert "projector" in prismatic_model_state and len(prismatic_model_state["projector"]) > 0
    assert "llm_backbone" in prismatic_model_state and len(prismatic_model_state["llm_backbone"]) > 0

    # Build checkpoint payload compatible with Prismatic conversion script
    print("[*] Building Prismatic checkpoint payload")
    checkpoint = {
        "model": prismatic_model_state,
        # Keep minimal extras; downstream tools generally only need ["model"]
    }

    # Save checkpoint
    ckpt_path = ckpt_dir / "latest-checkpoint.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"[*] Saved checkpoint to: {ckpt_path}")

    # Write minimal Prismatic config.json
    config_path = write_minimal_prismatic_config(
        out_dir=run_dir,
        hf_config=hf_config,
        base_vlm_override=cfg.base_vlm,
    )
    print(f"[*] Wrote minimal config.json to: {config_path}")

    # Write or copy dataset_statistics.json
    ds_stats_path = write_dataset_statistics(out_dir=run_dir, hf_model_path_or_id=cfg.hf_model_path_or_id, hf_config=hf_config)
    print(f"[*] Wrote dataset_statistics.json to: {ds_stats_path}")

    print(f"[*] Conversion complete! Prismatic-style run dir: {run_dir}")


if __name__ == "__main__":
    convert_hf_weights_to_openvla_prismatic()