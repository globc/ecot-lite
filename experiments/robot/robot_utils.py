"""Utils for evaluating robot policies in various environments."""

import math
import os
import random
import time
import cv2
import imageio
import ast

from prismatic.util.cot_utils import CotTag, get_cot_tags_list

import numpy as np
import torch

from experiments.robot.openvla_utils import (
    get_prismatic_vla,
    get_prismatic_vla_action,
    get_vla,
    get_vla_action,
)

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def round_to_n(x, n=1):
    # round scalar to n significant figure(s)
    return round(x, -int(math.floor(math.log10(abs(x))) + (n - 1)))


def hr_name(float_arg, fp=None):
    if fp is not None:
        float_arg = round_to_n(float_arg, n=fp)
    return str(float_arg).replace(".", "_")


def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg, wrap_diffusion_policy_for_droid=False):
    """Load model for evaluation."""
    if cfg.model_family == "prismatic":
        model = get_prismatic_vla(cfg)
    elif cfg.model_family == "openvla":
        model = get_vla(cfg)
    else:
        raise ValueError(f"Unexpected `model_family` found in config ({cfg.model_family}).")
    print(f"Loaded model: {type(model)}")
    return model


def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    if cfg.model_family == "prismatic":
        resize_size = 224
    elif cfg.model_family == "openvla":
        resize_size = 224
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


def get_action(cfg, model, obs, task_label, processor=None, task_id=None):
    """Queries the model to get an action."""
    if cfg.model_family == "prismatic":
        action, reasoning = get_prismatic_vla_action(
            model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop, task_id=task_id
        )
        # assert action.shape == (ACTION_DIM,)
    elif cfg.model_family == "openvla":
        action, reasoning = get_vla_action(
            model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop, task_id=task_id
        )
        # assert action.shape == (ACTION_DIM,)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action, reasoning


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action

def write_text(image, text, size, location, line_max_length):
    next_x, next_y = location

    for line in text:
        x, y = next_x, next_y

        for i in range(0, len(line), line_max_length):
            line_chunk = line[i : i + line_max_length]
            cv2.putText(image, line_chunk, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 1, cv2.LINE_AA)
            y += 18

        next_y = max(y, next_y + 50)


def split_reasoning(text, tags):
    new_parts = {None: text}

    for tag in tags:
        parts = new_parts
        new_parts = dict()

        for k, v in parts.items():
            if tag in v:
                s = v.split(tag)
                new_parts[k] = s[0]
                new_parts[tag] = s[1]
            else:
                new_parts[k] = v

    return new_parts


def get_metadata(reasoning):
    metadata = {"gripper": [[0, 0]], "bboxes": dict()}

    if f";\n{CotTag.GRIPPER_POSITION.value}" in reasoning:
        gripper_pos = reasoning[f";\n{CotTag.GRIPPER_POSITION.value}"]
        gripper_pos = gripper_pos.split("[")[-1]
        gripper_pos = gripper_pos.split("]")[0]
        gripper_pos = [int(x) for x in gripper_pos.split(",")]
        gripper_pos = [(gripper_pos[2 * i], gripper_pos[2 * i + 1]) for i in range(len(gripper_pos) // 2)]
        metadata["gripper"] = gripper_pos

    if f";\n{CotTag.VISIBLE_OBJECTS.value}" in reasoning:
        bboxes = ast.literal_eval(reasoning[f";\n{CotTag.VISIBLE_OBJECTS.value}"])
        metadata["bboxes"] = {k: [int(n) for pair in v for n in pair] for k, v in bboxes.items()}

    return metadata


def resize_pos(pos, img_size):
    return [(x * size) // 224 for x, size in zip(pos, img_size)]


def draw_gripper(img, pos_list, img_size=(640, 480)):
    for i, pos in enumerate(reversed(pos_list)):
        pos = (pos[1], pos[0])
        pos = resize_pos(pos, img_size)
        scale = 255 - int(255 * i / len(pos_list))
        cv2.circle(img, pos, 6, (0, 0, 0), -1)
        cv2.circle(img, pos, 5, (scale, scale, 255), -1)


def name_to_random_color(name):
    return [(hash(name) // (256**i)) % 256 for i in range(3)]


def draw_bboxes(img, bboxes, img_size=(640, 480)):
    for name, bbox in bboxes.items():
        show_name = name
        # show_name = f'{name}; {str(bbox)}'

        cv2.rectangle(
            img,
            resize_pos((bbox[1], bbox[0]), img_size),
            resize_pos((bbox[3], bbox[2]), img_size),
            name_to_random_color(name),
            2,
        )
        cv2.putText(
            img,
            show_name,
            resize_pos((bbox[1], bbox[0] + 6), img_size),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def make_reasoning_image(text):
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    tags = [f";\n{tag}" for tag in get_cot_tags_list()]
    reasoning = split_reasoning(text, tags)

    reasoning[f";\n{CotTag.PLAN.value}"] = text.split(f"assistant\n{CotTag.PLAN.value}")[-1].split(";\n")[0]
    text = [tag + reasoning[tag] for tag in tags[:-1] if tag in reasoning]
    write_text(image, text, 0.5, (10, 30), 70)

    return image, get_metadata(reasoning)