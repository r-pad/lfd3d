import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import zarr
from google.genai import types
from lfd3d.utils.gemini_utils import (
    TASK_SPEC,
    generate_prompt_with_subgoals,
    process_with_gemini,
    save_video,
    setup_client,
)
from tqdm import tqdm


def save_event_images(output_dir, images, ts, event_ts, goals):
    for event in zip(event_ts, goals):
        e_ts, goal = event
        e_ts = datetime.fromisoformat(e_ts).timestamp()
        idx = np.searchsorted(ts, e_ts)
        plt.imsave(f"{output_dir}/{goal}.png", images[idx])


def extract_events_with_gemini(
    images,
    ts,
    video_path,
    client,
    model_name,
    generate_content_config,
    goal_text,
    subgoals,
):
    fps = save_video(images, video_path, approx_duration=30)
    prompt = generate_prompt_with_subgoals(goal_text, subgoals)
    parsed_json = process_with_gemini(
        client, model_name, generate_content_config, video_path, prompt
    )
    ends, events = [], []
    for item in parsed_json:
        end_sec = float(item["timestamp"][3:])
        end_idx = int(end_sec * fps)
        end_ts = ts[end_idx]

        goal = item["subgoal"]
        end = datetime.fromtimestamp(end_ts).isoformat()

        events.append(goal)
        ends.append(end)
    return ends, events


def extract_events_with_gripper_pos(
    joint_states, subgoals, close_thresh=-0.4, open_thresh=0
):
    """
    First event ends when gripper closes,
    second events ends when gripper opens again.
    """
    assert len(subgoals) == 2, "designed only for 2 subgoals"
    joint_ts = np.asarray(joint_states["publish_ts"])
    gripper_pos = np.asarray(joint_states["pos"])[:, 6]

    close_gripper = np.where(gripper_pos < close_thresh)[0][0]
    open_gripper = (
        close_gripper + np.where(gripper_pos[close_gripper:] > open_thresh)[0][0]
    )

    ends = [
        datetime.fromtimestamp(joint_ts[i]).isoformat()
        for i in [close_gripper, open_gripper]
    ]
    return ends, subgoals


def main(args):
    """Main function to process the dataset and generate subgoal timestamps."""
    client = setup_client(os.environ.get("RPAD_GEMINI_API_KEY"))
    dataset = zarr.group(args.root)
    os.makedirs(args.output_dir, exist_ok=True)

    goal_text, subgoals = TASK_SPEC[args.task_spec]

    model_name = args.model_name
    generate_content_config = types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )

    pbar = tqdm(dataset)
    for demo_name in pbar:
        pbar.set_description(demo_name)
        demo = dataset[demo_name]
        os.makedirs(f"{args.output_dir}/{demo_name}", exist_ok=True)
        video_path = f"{args.output_dir}/{demo_name}/video.mp4"

        ts = np.asarray(demo["raw"]["rgb"]["image_rect"]["publish_ts"])
        images = demo["raw"]["rgb"]["image_rect"]["img"]

        if "follower_right" in demo["raw"] and not args.disable_gripper_pos:
            joint_states = demo["raw"]["follower_right"]["joint_states"]
            ends, events = extract_events_with_gripper_pos(joint_states, subgoals)
        else:
            ends, events = extract_events_with_gemini(
                images,
                ts,
                video_path,
                client,
                model_name,
                generate_content_config,
                goal_text,
                subgoals,
            )

        if "events" in demo:
            if len(demo["events"]["event"]) == 0:
                del demo["events"]
            else:
                print("skipping as events are already present")
                continue

        events_group = demo.create_group("events")
        ends = np.array(ends)
        events = np.array(events)
        events_group.create_dataset("end", data=ends)
        events_group.create_dataset("event", data=events)

        save_event_images(f"{args.output_dir}/{demo_name}", images, ts, ends, events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process RPAD-Foxglove dataset with Gemini API."
    )
    parser.add_argument(
        "--root",
        default="/data/sriram/rpad_foxglove/pick_mug_all.zarr",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--task_spec", default="place_mug_on_table", help="Task specified in dataset"
    )
    parser.add_argument(
        "--disable_gripper_pos",
        default=False,
        action="store_true",
        help="For robot demos, by default we use the gripper state to segment demos. With this arg, we still use Gemini.",
    )
    parser.add_argument(
        "--model_name",
        default="gemini-2.5-pro",
        help="Name of the Gemini model to use",
    )
    parser.add_argument(
        "--output_dir",
        default="event_viz",
    )
    args = parser.parse_args()
    main(args)
