"""
verify_openvla.py

Given an HF-exported OpenVLA model, attempt to load via AutoClasses, and verify forward() and predict_action().
"""

import time
import json
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# === Verification Arguments
MODEL_PATH = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/ecot-lite/out_openvla_ecot_lora/ecot-openvla-7b-oxe+libero_lm_90+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--20000_chkpt"
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str) -> str:
    if "v01" in MODEL_PATH:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut: PLAN:"


@torch.inference_mode()
def verify_openvla() -> None:
    print(f"[*] Verifying OpenVLAForActionPrediction using Model `{MODEL_PATH}`")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load Processor & VLA
    print("[*] Instantiating Processor and Pretrained OpenVLA")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # === BFLOAT16 + FLASH-ATTN MODE ===
    print("[*] Loading in BF16 with Flash-Attention Enabled")
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    # === 8-BIT QUANTIZATION MODE (`pip install bitsandbytes`) :: [~9GB of VRAM Passive || 10GB of VRAM Active] ===
    # print("[*] Loading in 8-Bit Quantization Mode")
    # vla = AutoModelForVision2Seq.from_pretrained(
    #     MODEL_PATH,
    #     attn_implementation="flash_attention_2",
    #     torch_dtype=torch.float16,
    #     quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # )

    # === 4-BIT QUANTIZATION MODE (`pip install bitsandbytes`) :: [~6GB of VRAM Passive || 7GB of VRAM Active] ===
    # print("[*] Loading in 4-Bit Quantization Mode")
    # vla = AutoModelForVision2Seq.from_pretrained(
    #     MODEL_PATH,
    #     attn_implementation="flash_attention_2",
    #     torch_dtype=torch.float16,
    #     quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # )

    print("[*] Iterating Tasks")
    with open("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/ecot-lite/experiments/robot/libero/libero_90_tasks/tasks.jsonl", 'r') as file:
        tasks = [json.loads(line) for line in file]

    for record in tasks:
        prompt = get_openvla_prompt(record["task_description"])
        image = Image.open(f"/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/ecot-lite/experiments/robot/libero/libero_90_tasks/images/{record['task_id']}.png").convert("RGB")

        # === BFLOAT16 MODE ===
        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

        # === 8-BIT/4-BIT QUANTIZATION MODE ===
        # inputs = processor(prompt, image).to(device, dtype=torch.float16)

        # Run OpenVLA Inference
        start_time = time.time()
        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
        action, generated_ids = vla.predict_action(**inputs, unnorm_key="libero_lm_90", max_new_tokens=1024)
        generated_text = processor.batch_decode(generated_ids)[0]
        print(f"\t=>> Time: {time.time() - start_time:.4f} || Action: {action}")
        print(generated_text)


if __name__ == "__main__":
    verify_openvla()
