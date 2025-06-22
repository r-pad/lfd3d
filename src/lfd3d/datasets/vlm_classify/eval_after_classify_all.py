import json
from datetime import datetime

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

TASK_SPEC = {
    "place_mug_on_table": (
        "place the mug on the table",
        ["grasp mug", "place mug on table"],
    )
}


def get_filtered_pred_gt(demo, subgoals, pred_goals):
    events = demo["events/end"]
    event_ts = [datetime.fromisoformat(i).timestamp() for i in events]
    img_ts = np.asarray(demo["raw/rgb/image_rect/ts"])
    event_idxs = np.array([np.searchsorted(img_ts, i) for i in event_ts])

    gt_goals = []
    for i in range(event_idxs[-1]):
        if i < event_idxs[0]:
            gt_goals.append(subgoals[0])
        else:
            gt_goals.append(subgoals[1])
    pred_goals = pred_goals[: event_idxs[-1]]

    return pred_goals, gt_goals


def main():
    root = "/data/sriram/rpad_foxglove/pick_mug_all.zarr"
    task_name = "place_mug_on_table"
    json_split = "../rpad_foxglove/val.json"
    split = json.load(open(json_split))

    goal_text, subgoals = TASK_SPEC[task_name]
    dataset = zarr.group(root)
    pbar = tqdm(split)
    all_gt, all_preds = [], []
    for demo_name in pbar:
        demo = dataset[demo_name]
        if "gemini_subgoals" not in demo:
            continue

        pred = demo["gemini_subgoals"]
        pred, gt = get_filtered_pred_gt(demo, subgoals, pred)

        all_gt.extend(gt)
        all_preds.extend([str(i) for i in pred])

    accuracy = sum([i == j for i, j in zip(all_gt, all_preds)]) / len(all_gt)
    confusion_matrix = pd.crosstab(
        all_gt, all_preds, rownames=["Actual"], colnames=["Predicted"]
    )
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:", confusion_matrix)


if __name__ == "__main__":
    main()
