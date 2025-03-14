import json
import os
import sys
import time

import imageio
import numpy as np
import tensorflow_datasets as tfds
from google import genai
from google.genai import types
from tqdm import tqdm


# Function to save a sequence of frames as a video file
def save_video(frames, output_path):
    """
    Save a sequence of frames as a video file

    Args:
        frames: NumPy array of shape (num_frames, height, width, channels)
        output_path: Path to save the video file
        fps: Frames per second for the video
    """
    # Gemini seems to struggle with longer videos, so adaptively set fps so that its always 20 sec video
    # We'll lose temporal resolution, but atleast our goals should be accurate
    fps = frames.shape[0] // 20
    height, width = frames[0].shape[:2]
    writer = imageio.get_writer(
        output_path,
        format="mp4",
        fps=fps,
        codec="h264",
        quality=7,
        pixelformat="yuv420p",
        macro_block_size=1,
        output_params=["-vf", f"scale={width}:{height}"],
    )

    for frame in frames:
        writer.append_data(frame)
    writer.close()


def upload_video(video_file_name):
    video_file = client.files.upload(file=video_file_name)

    while video_file.state == "PROCESSING":
        time.sleep(10)
        video_file = client.files.get(name=video_file.name)

    if video_file.state == "FAILED":
        raise ValueError(video_file.state)
    return video_file


root = "/data/sriram/DROID/droid"
builder = tfds.builder_from_directory(builder_dir=f"{root}")
dataset = builder.as_dataset(split="train")

client = genai.Client(
    api_key=os.environ.get("RPAD_GEMINI_API_KEY"),
)
model_name = "gemini-2.0-flash"
# model_name = "gemini-2.0-pro-exp-02-05"
generate_content_config = types.GenerateContentConfig(
    temperature=0.3,
    top_p=0.95,
    top_k=40,
    max_output_tokens=8192,
    response_mime_type="text/plain",
)

valid_idxs = range(len(dataset))
dataset_size = len(dataset)
if len(sys.argv) == 2:
    # Executing on one subset of the dataset
    split_num = int(sys.argv[1])
    assert split_num >= 0
    assert split_num < 10

    valid_idxs = valid_idxs[
        split_num * (dataset_size // 10) : (split_num + 1) * (dataset_size // 10)
    ]

output_dir = "/data/sriram/DROID/droid_gemini_events"
os.makedirs(output_dir, exist_ok=True)

for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
    if idx not in valid_idxs:
        continue

    if "failure" in item["episode_metadata"]["file_path"].numpy().decode("utf-8"):
        continue

    if os.path.exists(f"{output_dir}/{idx}/subgoal.json"):
        continue

    steps = [i for i in item["steps"]]
    goal_text = steps[0]["language_instruction"].numpy().decode("utf-8")
    if goal_text == "":
        continue

    images = np.array([i["observation"]["exterior_image_1_left"] for i in steps])
    # Save segment as video
    video_path = f"{output_dir}/{idx}/video.mp4"

    os.makedirs(f"{output_dir}/{idx}", exist_ok=True)
    save_video(images, video_path)

    # Add standard prompt text
    text_prompt = f"""
    # Task Analysis Request: Robot Subgoal Completion Timestamping

    - **Robot Goal**: "{goal_text}"
    - **Video**: Robot video demonstrating - {goal_text}

    ## Instructions (CRITICAL):
    1. **Analyze the video and identify significant state changes that represent meaningful subgoals.**  Focus on actions that clearly advance the robot towards achieving the **Robot Goal**.
    2. **A subgoal is meaningful if it represents a clear, distinct step and shows substantial progress towards the Robot Goal.**  Examples of meaningful subgoals include: grasping an object, manipulating an object, picking up or placing an object, initiating a pouring action, etc.  Avoid overly granular actions (e.g., "move fingers slightly").
    3. **For each identified meaningful subgoal, generate a concise, imperative caption and the timestamp (MM:SS) at which the subgoal action is *clearly and visually completed*.**

        **Completion is defined as the moment the subgoal action reaches a visually observable and intended end state.**  For example:
            * **"grasp the cup":** Completion is when the robot's gripper/hand is fully closed and securely holding the cup.
            * **"pour tea":** Completion is when the pouring motion visibly stops, or the tea flow ceases from the spout.
            * **"put down the teapot":** Completion is when the teapot is resting stably on the surface and the robot's hand/gripper has released it.

        **Choose the timestamp that best marks this visually clear point of completion.**
    4. **Subgoal captions MUST use imperative tense (e.g., "grasp the cup").**
    5. **Aim for approximately 2-5 *key* subgoals that capture the most important steps in the video.**  Focus on the major stages of the task.

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

    max_retries = 50
    retry_delay = 10  # seconds

    for attempt in range(max_retries):
        try:
            # Upload video segment
            video_prompt = upload_video(video_path)

            response = client.models.generate_content(
                model=model_name,
                contents=[video_prompt, text_prompt],
                config=generate_content_config,
            )
            parsed_json = json.loads(response.text.strip("`json\n"))
            print(goal_text, parsed_json)

            with open(f"{output_dir}/{idx}/subgoal.json", "w") as f:
                json.dump(parsed_json, f)
            break  # Exit the loop if successful
        except genai.errors.ServerError as e:
            if attempt < max_retries:  # Don't wait after the last attempt
                print(
                    f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached. Could not process {idx}. Final error: {e}")
