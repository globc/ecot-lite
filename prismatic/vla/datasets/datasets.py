"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.cot_utils import CotTag, abbreviate_tag
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

def reasoning_dropout(reasoning: str, dropout_prob: float) -> Tuple[str, str]:
    """Dropout reasoning tokens with probability `dropout_prob`."""
    if len(reasoning) == 0:
        return reasoning, ""

    reasoning_parts = reasoning.split("@")
    tags = [(reasoning_parts[i], reasoning_parts[i + 1]) for i in range(0, len(reasoning_parts), 2)]

    subset = np.random.rand(len(tags)) > dropout_prob

    subset_string = (
        "[" + ", ".join([abbreviate_tag(tag) for (tag, _), is_taken in zip(tags, subset) if is_taken]) + "]"
    )  # abbreviation

    excluded_tags = []

    if "EXCLUDE_TAGS" in os.environ:
        excluded_tags = os.environ["EXCLUDE_TAGS"].split(",")

    return (
        " ".join(
            [f"{tag[0]} {tag[1]}" for tag, is_taken in zip(tags, subset) if (is_taken and tag[0] not in excluded_tags)]
        ),
        subset_string,
    )



@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    image_window_size: int = 1
    use_wrist_image: bool = False
    print_prompt_limit: int = 20
    reasoning_dropout_prob: float = 0.0

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        reasoning, subset = reasoning_dropout(rlds_batch["reasoning"].decode(), dropout_prob=self.reasoning_dropout_prob)

        # either a single or multi image, depending on image_window_size
        if self.image_window_size == 1:
            img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
            if self.use_wrist_image:
                img = [img, Image.fromarray(rlds_batch["observation"]["image_wrist"][0])]
        else:
            img = [Image.fromarray(rlds_batch["observation"]["image_primary"][t]) for t in range(self.image_window_size)]
            if self.use_wrist_image:
                # wrist images are interleaved
                wrist_img = [
                    Image.fromarray(rlds_batch["observation"]["image_wrist"][t]) for t in range(self.image_window_size)
                ]
                img = [val for tup in zip(img, wrist_img) for val in tup]

        conversation = []

        # if there is no action horizon, remove it here.

        if self.action_tokenizer.required_future_horizon == 0:
            action = action[-1]
        else:
            # get the last FH + 1 actions (current action + future ones) if required
            action = action[-self.action_tokenizer.required_future_horizon - 1 :]

        tokenized_action = self.action_tokenizer(action)
        raw_action_tokens = self.base_tokenizer(tokenized_action)["input_ids"]

        conversation.extend(
            [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": f"{reasoning} {CotTag.ACTION.value} {tokenized_action}"},
            ]
        )
        num_answer_tokens = len(raw_action_tokens)

        # Construct Chat-based Prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        if self.print_prompt_limit > 0:
            print("Conversation:", conversation)
            p = prompt_builder.get_prompt()
            print("Prompt:", p)

            self.print_prompt_limit -= 1

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # critical, some tokenizers have different numbers of "end tokens".
        num_end_tokens = 1
        if isinstance(self.base_tokenizer, Qwen2TokenizerFast):
            # Qwen has <|im_end|><|endoftext|> for example
            num_end_tokens = 2

        # labels[: -(num_answer_tokens + num_end_tokens)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-num_end_tokens:] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name)


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        future_action_window_size: int = 0,
        image_window_size: int = 1,
        load_camera_views: tuple = ("primary",),
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=image_window_size,                        # If we wanted to feed / predict more than one step
                future_action_window_size=future_action_window_size,  # For action chunking
                skip_unlabeled=True,                                  # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                   # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
