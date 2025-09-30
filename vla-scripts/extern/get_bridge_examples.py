import tensorflow as tf
import tensorflow_datasets as tfds
import random
import os
import json
from PIL import Image
import numpy as np

# Load the validation split.
# For Bridge Orig, validation is defined as the last 5% of the training split.
DATA_DIR = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/ecot-lite/data/bridge_orig"

# Load validation split (assuming last 5% of the training split as validation)
builder = tfds.builder_from_directory("data/bridge_orig/1.0.0")
ds = builder.as_dataset(split='val')



# Extract all language instructions.
language_instructions = set()
with open("bridge_val_tasks.jsonl", "w", encoding="utf-8") as fout:
    for episode in ds:
        # In our dataset transform, language_instruction is assumed to be a string (or bytes).
        step = next(iter(episode['steps']))
        instr = step['language_instruction']

        if hasattr(instr, "numpy"):
            instr = instr.numpy()
        if isinstance(instr, (bytes, np.bytes_)):
            instr = instr.decode("utf-8")
        else:
            instr = str(instr)

        if instr not in language_instructions:
            img_array = step['observation']['image_0'].numpy()
            img = Image.fromarray(img_array)
            img.save(f"bridge_images/{len(language_instructions)}.png")

            fout.write(json.dumps({"id": len(language_instructions),"instruction": instr}, ensure_ascii=False) + "\n")
            language_instructions.add(instr)

            