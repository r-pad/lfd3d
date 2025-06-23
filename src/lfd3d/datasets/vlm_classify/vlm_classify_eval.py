import argparse
import os
import random
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
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


def eval_demo(dataset, demo_name, subgoals, goal_text, client, model_name, output_dir):
    global EXAMPLES
    demo = dataset[demo_name]
    num_subgoals = len(subgoals)
    events = demo["events/end"]
    assert len(events) == num_subgoals
    event_ts = [datetime.fromisoformat(i).timestamp() for i in events]
    img_ts = np.asarray(demo["raw/rgb/image_rect/ts"])
    event_idxs = np.array([np.searchsorted(img_ts, i) for i in event_ts])

    subgoal_idx = random.randint(0, num_subgoals - 1)
    if subgoal_idx == 0:
        image_idx = random.randint(0, event_idxs[subgoal_idx])
    else:
        image_idx = random.randint(event_idxs[subgoal_idx - 1], event_idxs[subgoal_idx])

    img = demo["raw/rgb/image_rect/img"][image_idx]
    pil_image = Image.fromarray(img)

    if demo["gripper_pos"].shape[1] == 500:
        prompt = generate_prompt_for_current_subtask(
            goal_text, subgoals, pil_image, EXAMPLES
        )
        config = genai.types.GenerateContentConfig(temperature=0.0, candidate_count=1)
        pred_goal = call_gemini_with_retry(client, model_name, prompt, config)

        gt_goal = subgoals[subgoal_idx]
        pil_image.save(
            f"{output_dir}/{demo_name}_idx={image_idx}_gt={gt_goal}_pred={pred_goal}.png"
        )
        return gt_goal, pred_goal  # Return for aggregation
    return None, None  # For non-robot demos


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
        "--model_name", default="gemini-2.5-pro", help="Name of the Gemini model"
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

    gt, pred = [], []
    pbar = tqdm(dataset)
    for demo_name in pbar:
        pbar.set_description(demo_name)
        gt_item, pred_item = eval_demo(
            dataset, demo_name, subgoals, goal_text, client, model_name, args.output_dir
        )
        if gt_item and pred_item:  # Only append for valid robot demos
            gt.append(gt_item)
            pred.append(pred_item)

    accuracy = sum([i == j for i, j in zip(gt, pred)]) / len(gt) if gt else 0
    confusion_matrix = pd.crosstab(
        gt, pred, rownames=["Actual"], colnames=["Predicted"]
    )
    confusion_matrix.to_csv(f"{args.output_dir}/confusion_matrix.csv")
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix saved to disk.")


if __name__ == "__main__":
    main()
