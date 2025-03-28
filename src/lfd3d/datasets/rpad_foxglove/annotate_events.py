import argparse
import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import zarr
from google import genai
from google.genai import types
from tqdm import tqdm

from lfd3d.utils.gemini_utils import save_video, setup_client, upload_video

TASK_SPEC = {
    "place_mug_on_table": (
        "place the mug on the table",
        ["grasp mug", "place mug on table"],
    )
}


def generate_prompt_with_subgoals(goal_text, subgoals):
    """Generate the prompt for Gemini API."""
    return f"""
    # Task Analysis Request: Subgoal Completion Timestamping

    - **Goal**: "{goal_text}"
    - **Sub-Goals**: "{subgoals}"
    - **Video**: Human or Robot video demonstrating - {goal_text}

    ## Instructions (CRITICAL):
    1. **Analyze the video and identify when the provided *Sub-Goals* are completed.**
    3. **For each subgoal, generate the timestamp (MM:SS.s) at which the subgoal action is *clearly and visually completed*.**

        **Completion is defined as the moment the subgoal action reaches a visually observable and intended end state.** For example:
            * "grasp the cup":** Completion is when the robot's gripper/hand is fully closed and securely holding the cup.
            * "pour tea":** Completion is when the pouring motion visibly stops, or the tea flow ceases from the spout.
            * "put down the teapot":** Completion is when the teapot is resting stably on the surface and the robot's hand/gripper has just released it.

        **Choose the timestamp that best marks this visually clear point of completion.**
    5. **Return timestamps for all {len(subgoals)} subgoals.**
    6. **Make sure the timestamps are valid and not longer than the actual video.**

    ## Output Format:
    Return a JSON array with an entry for each identified subgoal:

    [
        {{
            "subgoal": "string",  // Subgoal
            "timestamp": "MM:SS.s",  // Timestamp in minutes, seconds and tenths of a second (moment of completion)
        }}
    ]

    ## Example (for "pour tea from a teapot" with the subgoals - ["grasp the teapot", "pour tea from teapot", "put down the teapot"]):
    [
      {{"subgoal": "grasp the teapot", "timestamp": "00:02.0"}},
      {{"subgoal": "pour tea from teapot", "timestamp": "00:05.3"}},
      {{"subgoal": "put down the teapot", "timestamp": "00:07.8"}}
    ]

    IMPORTANT: Provide ONLY valid JSON as your response, no explanation text.
    """


def process_with_gemini(
    client, model_name, generate_content_config, video_path, prompt
):
    """Process video with Gemini API and return parsed JSON response."""
    max_retries = 50
    retry_delay = 10  # seconds

    for attempt in range(max_retries):
        try:
            video_prompt = upload_video(client, video_path)
            response = client.models.generate_content(
                model=model_name,
                contents=[video_prompt, prompt],
                config=generate_content_config,
            )
            return json.loads(response.text.strip("`json\n"))
        except genai.errors.ServerError as e:
            if attempt < max_retries - 1:  # Don't wait after the last attempt
                print(
                    f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached. Final error: {e}")
                return None


def save_event_images(output_dir, images, ts, event_ts, goals):
    for event in zip(event_ts, goals):
        e_ts, goal = event
        e_ts = datetime.fromisoformat(e_ts).timestamp()
        idx = np.searchsorted(ts, e_ts)
        plt.imsave(f"{output_dir}/{goal}.png", images[idx])


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

    for demo_name in tqdm(dataset):
        demo = dataset[demo_name]
        os.makedirs(f"{args.output_dir}/{demo_name}", exist_ok=True)
        video_path = f"{args.output_dir}/{demo_name}/video.mp4"

        images = np.asarray(demo["_rgb_image_rect"]["img"])
        ts = np.asarray(demo["_rgb_image_rect"]["publish_ts"])
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

        if "events" in demo and len(demo["events"]["event"]) == 0:
            del demo["events"]
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
        "--model_name",
        default="gemini-2.5-pro-exp-03-25",
        help="Name of the Gemini model to use",
    )
    parser.add_argument(
        "--output_dir",
        default="event_viz",
    )
    args = parser.parse_args()
    main(args)
