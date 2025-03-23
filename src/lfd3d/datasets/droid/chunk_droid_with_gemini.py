import argparse
import json
import os
import time

import numpy as np
import tensorflow_datasets as tfds
from google import genai
from google.genai import types
from tqdm import tqdm

from lfd3d.utils.gemini_utils import save_video, upload_video


def setup_client(api_key):
    """Initialize and return a Gemini client."""
    return genai.Client(api_key=api_key)


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


def filter_and_process_item(item):
    """Filter items based on metadata and return goal text and images."""
    file_path = item["episode_metadata"]["file_path"].numpy().decode("utf-8")
    if "failure" in file_path:
        return None, None

    steps = [i for i in item["steps"]]
    goal_text = steps[0]["language_instruction"].numpy().decode("utf-8")
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
            video_prompt = upload_video(video_path)
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

    for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
        if idx not in valid_idxs:
            continue

        goal_text, images = filter_and_process_item(item)
        if goal_text is None or images is None:
            continue

        if os.path.exists(f"{output_dir}/{idx}/subgoal.json"):
            continue

        os.makedirs(f"{output_dir}/{idx}", exist_ok=True)
        video_path = f"{output_dir}/{idx}/video.mp4"
        save_video(images, video_path)

        prompt = generate_prompt(goal_text)
        parsed_json = process_with_gemini(
            client, model_name, generate_content_config, video_path, prompt
        )

        if parsed_json is not None:
            print(goal_text, parsed_json)
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
    # model_name = "gemini-2.0-pro-exp-02-05"

    args = parser.parse_args()
    main(args)
