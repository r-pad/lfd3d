import json
import time

import imageio
from google import genai

TASK_SPEC = {
    "place_mug_on_table": (
        "place the mug on the table",
        ["grasp mug", "place mug on table"],
    ),
    "Grasp mug and place it on the platform.": (
        "Grasp mug and place it on the platform.",
        ["grasp mug", "place mug on platform."],
    ),
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


def upload_video(client, video_file_name):
    video_file = client.files.upload(file=video_file_name)

    while video_file.state == "PROCESSING":
        time.sleep(10)
        video_file = client.files.get(name=video_file.name)

    if video_file.state == "FAILED":
        raise ValueError(video_file.state)
    return video_file


def save_video(frames, output_path, approx_duration=20):
    """
    Save a sequence of frames as a video file

    Args:
        frames: NumPy array of shape (num_frames, height, width, channels)
        output_path: Path to save the video file
        duration: Approx length of video (in sec) [approx because we set fps to closest integer]
    """
    # Gemini seems to struggle with longer videos, so adaptively set fps so that its `approx_duration` secs
    # We'll lose temporal resolution, but atleast our goals should be accurate
    fps = frames.shape[0] // approx_duration
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
    return fps


def setup_client(api_key):
    """Initialize and return a Gemini client."""
    return genai.Client(api_key=api_key)
