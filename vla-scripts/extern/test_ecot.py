"""
verify_openvla.py

Given an HF-exported OpenVLA model, attempt to load via AutoClasses, and verify forward() and predict_action().
"""

import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

import time
import numpy as np
import cv2
import textwrap
from PIL import Image, ImageDraw, ImageFont
import enum

import requests
from io import BytesIO
from PIL import Image
import json

import os

# === Verification Arguments
MODEL_PATH = "Embodied-CoT/ecot-openvla-7b-bridge"

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def get_openvla_prompt(instruction: str) -> str:
    return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT: TASK:"

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
                # print(tag, s)
            else:
                new_parts[k] = v

    return new_parts

class CotTag(enum.Enum):
    TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    ACTION = "ACTION:"


def get_cot_tags_list():
    return [
        CotTag.TASK.value,
        CotTag.PLAN.value,
        CotTag.VISIBLE_OBJECTS.value,
        CotTag.SUBTASK_REASONING.value,
        CotTag.SUBTASK.value,
        CotTag.MOVE_REASONING.value,
        CotTag.MOVE.value,
        CotTag.GRIPPER_POSITION.value,
        CotTag.ACTION.value,
    ]

def name_to_random_color(name):
    return [(hash(name) // (256**i)) % 256 for i in range(3)]

def draw_gripper(img, pos_list, img_size=(640, 480)):
    for i, pos in enumerate(reversed(pos_list)):
        pos = resize_pos(pos, img_size)
        scale = 255 - int(255 * i / len(pos_list))
        cv2.circle(img, pos, 6, (0, 0, 0), -1)
        cv2.circle(img, pos, 5, (scale, scale, 255), -1)

def get_metadata(reasoning):
    metadata = {"gripper": [[0, 0]], "bboxes": dict()}

    if f" {CotTag.GRIPPER_POSITION.value}" in reasoning:
        gripper_pos = reasoning[f" {CotTag.GRIPPER_POSITION.value}"]
        gripper_pos = gripper_pos.split("[")[-1]
        gripper_pos = gripper_pos.split("]")[0]
        gripper_pos = [int(x) for x in gripper_pos.split(",")]
        gripper_pos = [(gripper_pos[2 * i], gripper_pos[2 * i + 1]) for i in range(len(gripper_pos) // 2)]
        metadata["gripper"] = gripper_pos

    if f" {CotTag.VISIBLE_OBJECTS.value}" in reasoning:
        for sample in reasoning[f" {CotTag.VISIBLE_OBJECTS.value}"].split("]"):
            obj = sample.split("[")[0]
            if obj == "":
                continue
            coords = [int(n) for n in sample.split("[")[-1].split(",")]
            metadata["bboxes"][obj] = coords

    return metadata

def resize_pos(pos, img_size):
    return [(x * size) // 256 for x, size in zip(pos, img_size)]

def draw_bboxes(img, bboxes, img_size=(640, 480)):
    for name, bbox in bboxes.items():
        show_name = name
        # show_name = f'{name}; {str(bbox)}'

        cv2.rectangle(
            img,
            resize_pos((bbox[0], bbox[1]), img_size),
            resize_pos((bbox[2], bbox[3]), img_size),
            name_to_random_color(name),
            1,
        )
        cv2.putText(
            img,
            show_name,
            resize_pos((bbox[0], bbox[1] + 6), img_size),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

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
    with open("vla-scripts/extern/bridge_val_tasks.jsonl", 'r') as file:
        tasks = [json.loads(line) for line in file]

    for record in tasks:
        image = Image.open(f"vla-scripts/extern/bridge_images/9.png").convert("RGB") # Image.open(f"vla-scripts/extern/bridge_images/{record['id']}.png").convert("RGB")
        instruction = "put the fruit in the pot" # record["instruction"]
        prompt = get_openvla_prompt(instruction)

        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

        # Run OpenVLA Inference
        start_time = time.time()

        torch.manual_seed(0)
        action, generated_ids = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False, max_new_tokens=1024)
        generated_text = processor.batch_decode(generated_ids)[0]
        print(f"Time: {time.time() - start_time:.4f} || Action: {action}")
        
        image = image.resize((640, 480), resample=Image.BILINEAR)

        #@title Visualize reasoning and image
        tags = [f" {tag}" for tag in get_cot_tags_list()]
        reasoning = split_reasoning(generated_text, tags)
        text = [tag + reasoning[tag] for tag in [' TASK:',' PLAN:',' SUBTASK REASONING:',' SUBTASK:',
                                                ' MOVE REASONING:',' MOVE:', ' VISIBLE OBJECTS:', ' GRIPPER POSITION:'] if tag in reasoning]
        metadata = get_metadata(reasoning)
        bboxes = {}
        for k, v in metadata["bboxes"].items():
            if k[0] == ",":
                k = k[1:]
            bboxes[k.lstrip().rstrip()] = v

        caption = ""
        for t in text:
            wrapper = textwrap.TextWrapper(width=80, replace_whitespace=False)
            word_list = wrapper.wrap(text=t)
            caption_new = ''
            for ii in word_list[:-1]:
                caption_new = caption_new + ii + '\n      '
            caption_new += word_list[-1]

            caption += caption_new.lstrip() + "\n\n"

        base = Image.fromarray(np.ones((480, 640, 3), dtype=np.uint8) * 255)
        draw = ImageDraw.Draw(base)
        font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
        font = ImageFont.truetype(font_path, size=14)
        color = (0,0,0) # RGB
        draw.text((30, 30), caption, color, font=font)

        img_arr = np.array(image)
        draw_gripper(img_arr, metadata["gripper"])
        draw_bboxes(img_arr, bboxes)

        text_arr = np.array(base)

        reasoning_img = Image.fromarray(np.concatenate([img_arr, text_arr], axis=1))


        reasoning_img.save(f"vla-scripts/extern/bridge_reasonings_test/9.png") # reasoning_img.save(f"vla-scripts/extern/bridge_reasonings/{record['id']}.png")


if __name__ == "__main__":
    verify_openvla()
