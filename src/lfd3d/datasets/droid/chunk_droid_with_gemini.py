import argparse
import json
import os
import re
import time

import numpy as np
import tensorflow_datasets as tfds
from google import genai
from google.genai import types
from tqdm import tqdm

from lfd3d.utils.gemini_utils import save_video, setup_client, upload_video


def load_dataset(root):
    """Load the dataset from the specified root directory."""
    builder = tfds.builder_from_directory(builder_dir=root)
    return builder.as_dataset(split="train")


def get_valid_indices(dataset_size, split_num=None):
    """Determine valid indices based on dataset size and optional split number."""
    valid_idxs = range(dataset_size)
    if split_num is not None:
        assert 0 <= split_num < 10, "Split number must be between 0 and 9"
        start = split_num * (dataset_size // 10)
        end = (split_num + 1) * (dataset_size // 10)
        valid_idxs = valid_idxs[start:end]
    return valid_idxs


def search_annotations_json(
    file_path, droid_language_annotations, droid_language_annotations_keys
):
    query_p1 = file_path.split("/")[0]
    query_p2 = ".*"
    query_p3 = file_path.split("/")[2]
    query_p4 = file_path[-13:-5]
    try:
        hours, minutes, seconds = query_p4.split(":")
        query_p4 = f"-{hours}h-{minutes}m-{seconds}s"
    except:
        hours, minutes, seconds = query_p4.split("_")
        query_p4 = f"-{hours}h-{minutes}m-{seconds}s"
    query_key = query_p1 + query_p2 + query_p3 + query_p4

    regex = re.compile(query_key)

    match = None
    for index, key in enumerate(droid_language_annotations_keys):
        if regex.search(key):
            match = key
            droid_language_annotations_keys.pop(index)
            break

    if match is None:
        return None
    return droid_language_annotations[match]["language_instruction1"]


def filter_and_process_item(
    item, file_path, droid_language_annotations, droid_language_annotations_keys
):
    """Filter items based on metadata and return goal text and images."""
    if "failure" in file_path:
        return None, None

    steps = [i for i in item["steps"]]
    goal_text = steps[0]["language_instruction"].numpy().decode("utf-8")
    if not goal_text:
        # Try backup search in the json
        goal_text = search_annotations_json(
            file_path, droid_language_annotations, droid_language_annotations_keys
        )
        if not goal_text:
            return None, None

    images = np.array([i["observation"]["exterior_image_1_left"] for i in steps])
    return goal_text, images


def generate_prompt(goal_text):
    """Generate the prompt for Gemini API."""
    return f"""
    # Task Analysis Request: Robot Subgoal Completion Timestamping

    - **Robot Goal**: "{goal_text}"
    - **Video**: Robot video demonstrating - {goal_text}

    ## Instructions (CRITICAL):
    1. **Analyze the video and identify significant state changes that represent meaningful subgoals.** Focus on actions that clearly advance the robot towards achieving the **Robot Goal**.
    2. **A subgoal is meaningful if it represents a clear, distinct step and shows substantial progress towards the Robot Goal.** Examples of meaningful subgoals include: grasping an object, manipulating an object, picking up or placing an object, initiating a pouring action, etc. Avoid overly granular actions (e.g., "move fingers slightly").
    3. **For each identified meaningful subgoal, generate a concise, imperative caption and the timestamp (MM:SS) at which the subgoal action is *clearly and visually completed*.**

        **Completion is defined as the moment the subgoal action reaches a visually observable and intended end state.** For example:
            * **"grasp the cup":** Completion is when the robot's gripper/hand is fully closed and securely holding the cup.
            * **"pour tea":** Completion is when the pouring motion visibly stops, or the tea flow ceases from the spout.
            * **"put down the teapot":** Completion is when the teapot is resting stably on the surface and the robot's hand/gripper has released it.

        **Choose the timestamp that best marks this visually clear point of completion.**
    4. **Subgoal captions MUST use imperative tense (e.g., "grasp the cup").**
    5. **Aim for approximately 2-5 *key* subgoals that capture the most important steps in the video.** Focus on the major stages of the task.
    6. **Make sure the timestamps are valid and not longer than the video.**

    ## Output Format:
    Return a JSON array with an entry for each identified subgoal:

    [
        {{
            "subgoal": "string",  // Imperative caption
            "timestamp": "MM:SS",  // Timestamp in minutes and seconds (moment of completion)
        }}
    ]

    ## Example (for "pour tea from a teapot"):
    [
      {{"subgoal": "grasp the teapot", "timestamp": "00:02"}},
      {{"subgoal": "pour tea from teapot", "timestamp": "00:05"}},
      {{"subgoal": "put down the teapot", "timestamp": "00:07"}}
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
                raise


def main(args):
    """Main function to process the dataset and generate subgoal timestamps."""
    client = setup_client(os.environ.get("RPAD_GEMINI_API_KEY"))
    dataset = load_dataset(args.root)
    valid_idxs = get_valid_indices(len(dataset), args.split_num)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_name = args.model_name
    generate_content_config = types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )

    with open(f"{args.root}/../droid_language_annotations.json") as f:
        droid_language_annotations = json.load(f)
    droid_language_annotations_keys = list(droid_language_annotations.keys())
    with open("idx_to_fname_mapping.json") as f:
        idx_to_fname_mapping = json.load(f)

    for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
        if idx not in valid_idxs:
            continue

        file_path = idx_to_fname_mapping[idx]
        goal_text, images = filter_and_process_item(
            item, file_path, droid_language_annotations, droid_language_annotations_keys
        )
        if goal_text is None or images is None:
            continue

        if os.path.exists(f"{output_dir}/{idx}/subgoal.json"):
            continue

        os.makedirs(f"{output_dir}/{idx}", exist_ok=True)
        video_path = f"{output_dir}/{idx}/video.mp4"
        save_video(images, video_path, approx_duration=20)

        prompt = generate_prompt(goal_text)
        parsed_json = process_with_gemini(
            client, model_name, generate_content_config, video_path, prompt
        )

        with open(f"{output_dir}/{idx}/subgoal.json", "w") as f:
            json.dump(parsed_json, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process DROID dataset with Gemini API."
    )
    parser.add_argument(
        "--root",
        default="/data/sriram/DROID/droid",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="/data/sriram/DROID/droid_gemini_events",
        help="Output directory for results",
    )
    parser.add_argument(
        "--split_num",
        type=int,
        help="Optional split number (0-9) for processing a subset of the dataset",
    )
    parser.add_argument(
        "--model_name",
        default="gemini-2.0-flash",
        help="Name of the Gemini model to use",
    )

    args = parser.parse_args()
    main(args)
