import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm


def get_start_end_indexes(data_steps, caption):
    """Get the end of the first and second chunks by checking the gripper state.
    The logic is that the first chunk ends when the gripper closes and the second chunk
    ends when the gripper opens. Works pretty well in practice.

    data_steps: - Data of one item in the TFDS
    caption: str - Caption for the event
    """
    gripper_state = np.array(
        [i["observation"]["gripper_closed"].numpy() for i in data_steps]
    )
    threshold = 0.5

    # The entire video is a chunk
    if caption == "" or caption.split()[0] in ["open", "close"]:
        return [0, len(gripper_state) - 1]

    # first chunk -> till gripper_state > threshold
    first_chunk_end = np.argmax(gripper_state > threshold)

    # second chunk -> from first chunk, till gripper state < threshold or end of video
    second_chunk_offset = np.argmax(gripper_state[first_chunk_end:] < threshold)
    if second_chunk_offset == 0:
        second_chunk_end = len(gripper_state) - 1
    else:
        second_chunk_end = first_chunk_end + second_chunk_offset

    return [0, first_chunk_end, second_chunk_end]


root = "/data/sriram/rt1/"
builder = tfds.builder_from_directory(builder_dir=f"{root}/fractal20220817_data_0.1.0")
dataset = builder.as_dataset(split="train")
dataset = dataset.enumerate()
with open(f"{root}/chunked_captions.json") as f:
    captions = json.load(f)

out_dir = f"{root}/rt1_rgb_chunk"
os.makedirs(out_dir, exist_ok=True)

for idx, item in tqdm(dataset):
    os.makedirs(f"{out_dir}/{idx}", exist_ok=True)
    steps = [it for it in item["steps"]]
    caption = captions[idx]["original"]
    chunk_indexes = get_start_end_indexes(steps, caption)

    for c_idx in chunk_indexes:
        rgb = steps[c_idx]["observation"]["image"].numpy()
        plt.imsave(f"{out_dir}/{idx}/{str(c_idx).zfill(5)}.png", rgb)
