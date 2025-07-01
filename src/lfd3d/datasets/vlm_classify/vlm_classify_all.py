import argparse
import os
import random
import shutil

import numpy as np
import zarr
from classify_utils import (
    call_gemini_with_retry,
    generate_prompt_for_current_subtask,
    load_incontext_examples,
)
from google import genai
from lfd3d.utils.gemini_utils import setup_client
from PIL import Image
from tqdm import tqdm

TASK_SPEC = {
    "place_mug_on_table": (
        "place the mug on the table",
        ["grasp mug", "place mug on table"],
    )
}

EXAMPLES = load_incontext_examples()


def classify_demo(
    dataset, demo_name, subgoals, goal_text, client, model_name, output_dir
):
    global EXAMPLES
    demo = dataset[demo_name]

    if demo["gripper_pos"].shape[1] != 500:
        return  # Not robot demo. pass
    if "gemini_subgoals" in demo:
        return  # Already computed

    images = demo["raw/rgb/image_rect/img"]
    img_ts = np.asarray(demo["raw/rgb/image_rect/ts"])
    GRIPPER_MIN, GRIPPER_MAX = 0.01, 0.048  # Values for aloha
    joint_positions = demo["raw/follower_right/joint_states/pos"][:]
    joint_positions_ts = demo["raw/follower_right/joint_states/ts"][:]
    pred_goals = []
    for i, img in tqdm(enumerate(images), total=len(images)):
        joint_idx = np.searchsorted(joint_positions_ts, img_ts[i])
        gripper_state = joint_positions[joint_idx, 7]
        gripper_state_scaled = (gripper_state - GRIPPER_MIN) / GRIPPER_MAX

        pil_image = Image.fromarray(img)
        prompt = generate_prompt_for_current_subtask(
            goal_text, subgoals, pil_image, gripper_state_scaled, EXAMPLES
        )
        config = genai.types.GenerateContentConfig(temperature=0.0, candidate_count=1)
        pred = call_gemini_with_retry(client, model_name, prompt, config)
        pred_goals.append(pred)

    demo["gemini_subgoals"] = pred_goals


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VLM classification on zarr dataset."
    )
    parser.add_argument(
        "--root",
        default="/data/sriram/rpad_foxglove/pick_mug_all.zarr",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--task_name", default="place_mug_on_table", help="Task specified in dataset"
    )
    parser.add_argument(
        "--model_name", default="gemini-2.5-flash", help="Name of the Gemini model"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", default="vlm_classify_eval_results", help="Output directory"
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    goal_text, subgoals = TASK_SPEC[args.task_name]
    client = setup_client(os.environ.get("RPAD_GEMINI_API_KEY"))
    model_name = args.model_name
    dataset = zarr.group(args.root)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)

    pbar = tqdm(dataset)
    for demo_name in pbar:
        pbar.set_description(demo_name)
        classify_demo(
            dataset, demo_name, subgoals, goal_text, client, model_name, args.output_dir
        )


if __name__ == "__main__":
    main()
