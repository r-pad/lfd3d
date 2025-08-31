import argparse
import json
import os
from glob import glob

import imageio.v3 as iio
import matplotlib.pyplot as plt
from google.genai import types
from lfd3d.utils.gemini_utils import (
    TASK_SPEC,
    generate_prompt_with_subgoals,
    process_with_gemini,
    setup_client,
)
from tqdm import tqdm


def extract_events_with_gemini(
    video_path,
    fps,
    client,
    model_name,
    generate_content_config,
    goal_text,
    subgoals,
):
    prompt = generate_prompt_with_subgoals(goal_text, subgoals)
    parsed_json = process_with_gemini(
        client, model_name, generate_content_config, video_path, prompt
    )
    ends, events = [], []
    for item in parsed_json:
        end_sec = float(item["timestamp"][3:])
        end_idx = int(end_sec * fps)
        goal = item["subgoal"]

        events.append(goal)
        ends.append(end_idx)
    return ends, events


def main(args):
    """Main function to process the dataset and generate subgoal timestamps."""
    client = setup_client(os.environ.get("RPAD_GEMINI_API_KEY"))
    model_name = args.model_name
    generate_content_config = types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )

    # Get all kinect videos
    videos = sorted(
        glob(f"{args.root}/videos/*/observation.images.cam_azure_kinect.color/*")
    )

    with open(f"{args.root}/meta/info.json") as f:
        info_json = json.load(f)
    with open(f"{args.root}/meta/episodes.jsonl") as f:
        episode_info = [json.loads(line) for line in f]
    fps = info_json["fps"]

    os.makedirs(f"{args.output_dir}/gemini_events", exist_ok=True)
    viz = True

    pbar = tqdm(enumerate(videos), total=len(videos))
    for i, vid_name in pbar:
        pbar.set_description(os.path.basename(vid_name))
        goal_text = episode_info[i]["tasks"][0]
        subgoals = TASK_SPEC[goal_text]

        ends, events = extract_events_with_gemini(
            vid_name,
            fps,
            client,
            model_name,
            generate_content_config,
            goal_text,
            subgoals,
        )

        with open(
            f"{args.output_dir}/gemini_events/{os.path.basename(vid_name)}.json", "w"
        ) as f:
            json.dump({"event_idxs": ends, "events": events}, f)

        if viz:
            video_arr = iio.imread(vid_name)
            for i in range(len(ends)):
                save_path = f"{args.output_dir}/gemini_events/{os.path.basename(vid_name)}_{i}_{ends[i]}_{events[i]}.png"
                plt.imsave(save_path, video_arr[ends[i]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process LeRobotDataset with Gemini API."
    )
    parser.add_argument(
        "--root",
        default="/home/sriram/.cache/huggingface/lerobot/sriramsk/mug_on_platform_20250830_human/",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="/data/sriram/lerobot_extradata/sriramsk/mug_on_platform_20250830_human/",
    )
    parser.add_argument(
        "--model_name",
        default="gemini-2.5-pro",
        help="Name of the Gemini model to use",
    )
    args = parser.parse_args()
    main(args)
