import os
import time

from google import genai
from PIL import Image


def call_gemini_with_retry(
    client, model_name, contents, config, max_retries=50, retry_delay=10
):
    """Wrapper for Gemini API call with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            return response.text
        except genai.errors.ServerError as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached. Final error: {e}")
                return None


def generate_prompt_for_current_subtask(goal_text, subgoals, query_img, examples=[]):
    """
    Generates an expert-level prompt for Gemini to identify the SINGLE
    current sub-task being performed in a static image.

    Args:
        goal_text (str): The high-level description of the overall task.
        subgoals (list): An ordered list of strings representing the sub-goals.
        query_img (PIL.Image): Query image
        examples (list): List of (PIL.Image, ground_truth_subgoal) tuples for in-context learning.

    Returns:
        list(): The formatted prompt
    """
    # Convert the Python list of subgoals to a formatted string for the prompt
    subgoals_str = "\n".join([f"- {sg}" for sg in subgoals])
    prompt = []

    base_prompt = f"""
# AI Task: Current Robotic Sub-Task Identification

## Persona:
You are a highly perceptive AI specializing in robotic action recognition. Your function is to analyze a static image and instantly identify which specific step of a task a robot is currently performing.

## Core Objective:
Based on the provided image, identify which ONE of the **Ordered Sub-Goals** the robot is currently executing.

## Inputs You Will Be Given:
- **Overall Goal**: "{goal_text}"
- **Ordered Sub-Goals**:
{subgoals_str}
- **Image**: A single snapshot of the robot during the task.

## Critical Instructions:

1.  **Understand the Sequence**: First, read the **Overall Goal** and the list of **Ordered Sub-Goals** to understand the full workflow. The order is critical.

2.  **Identify the "In-Progress" Sub-Goal**: Your primary task is to find the sub-goal that describes the action the robot is physically performing in the image.

3.  **Tie-Breaker Rules (Extremely Important)**:
    *   **If the robot is between actions** (e.g., it has just finished one sub-goal but has not yet started the next), your answer should be the sub-goal that was **just completed**.
    *   **If the entire task is visibly complete**, your answer should be the **final sub-goal** in the list.
    *   **If the task has not yet started** (e.g., the robot is idle before the first action), your answer should be the **first sub-goal** in the list.

## Output Format:
- Your response MUST BE ONLY the string of the identified sub-goal.
- **DO NOT** include any other text, explanations, or formatting.

"""
    prompt.append(base_prompt)

    if examples:
        prompt.append("""
**Cross-Modal Understanding**: You will see example images from human demonstrations followed by a robot image. The same sub-goals apply to both human and robot executions of this task.
## Examples from human demonstrations:
        """)
    for i, (example_img, gt_subgoal) in enumerate(examples, 1):
        prompt.append(f"Example {i}:")
        prompt.append(example_img)
        prompt.append(f"Answer: {gt_subgoal}")

    prompt.append("Now analyze this robot image and identify the current sub-goal:")
    prompt.append(query_img)
    return prompt


def load_incontext_examples():
    """
    Handpicked examples from the dataset
    """
    icl_dir = "incontext_examples"
    examples = []
    for imgname in os.listdir(icl_dir):
        pil_img = Image.open(f"{icl_dir}/{imgname}").convert("RGB")
        # Name is expected to be of the pattern <demo_name>--<image_idx>--<subgoal>.png
        subgoal = imgname.split("--")[-1].split(".")[0]
        examples.append((pil_img, subgoal))
    return examples
