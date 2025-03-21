import json
import re

import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm


def process_caption(caption):
    """
    Rewrite the captions with some regexes
    """
    chunked_caption = []

    verb = caption.split(" ")[0]

    caption = caption.replace("rxbar chocolate", "chocolate bar")
    caption = caption.replace("rxbar blueberry", "blueberry bar")
    caption = caption.replace(" jalapeno ", " ")
    caption = caption.replace(" rice ", " ")

    found_match = False
    if verb == "pick":
        patterns = {
            r"pick (.*) from (.*) and place (.*)": lambda match: [
                f"grasp {match.group(1)} from {match.group(2)}",
                f"place {match.group(1)} {match.group(3)}",
            ],
            r"pick (.*) from (.*)": lambda match: [
                f"grasp {match.group(1)} from {match.group(2)}",
                f"pick up {match.group(1)}",
            ],
            r"pick (.*)": lambda match: [
                f"grasp {match.group(1)}",
                f"pick up {match.group(1)}",
            ],
        }
    elif verb == "place":
        patterns = {
            r"place (.*) into (.*)": lambda match: [
                f"grasp {match.group(1)}",
                f"place {match.group(1)} into {match.group(2)}",
            ],
            r"place (.*) upright": lambda match: [
                f"grasp {match.group(1)}",
                f"place {match.group(1)} upright",
            ],
        }
    elif verb == "move":
        patterns = {
            r"move (.*) near (.*)": lambda match: [
                f"grasp {match.group(1)}",
                f"move {match.group(1)} near {match.group(2)}",
            ],
        }
    elif verb == "knock":
        patterns = {
            r"knock (.*) over": lambda match: [
                f"grasp {match.group(1)}",
                f"knock {match.group(1)} over",
            ],
        }
    elif verb == "open" or verb == "close":
        patterns = {r"(.*)": lambda match: [f"{match.group(1)}"]}
    else:
        print("caption is empty")
        patterns = {r"(.*)": lambda match: ["no caption available"]}

    for pattern, transform in patterns.items():
        match = re.match(pattern, caption)
        if match:
            chunked_caption.extend(transform(match))
            found_match = True
            break

    if not found_match:
        raise NotImplementedError

    return chunked_caption


# Function to extract and decode the instruction
def extract_instruction(idx, item):
    step = next(iter(item["steps"]))
    caption = step["observation"]["natural_language_instruction"]
    return caption


if __name__ == "__main__":
    root = "/data/sriram/rt1/fractal20220817_data_0.1.0/"
    output_file = "chunked_captions.json"

    builder = tfds.builder_from_directory(builder_dir=root)
    dataset = builder.as_dataset(split="train")
    dataset = dataset.enumerate()

    processed_dataset = dataset.map(
        extract_instruction, num_parallel_calls=tf.data.AUTOTUNE
    )
    processed_dataset = processed_dataset.prefetch(tf.data.AUTOTUNE)

    caption_list = []
    for caption_tensor in tqdm(processed_dataset):
        caption_string = caption_tensor.numpy().decode("utf-8")
        caption_string = caption_string.strip()
        chunked_cap = process_caption(caption_string)
        caption_list.append({"original": caption_string, "chunked": chunked_cap})

    with open(output_file, "w") as f:
        json.dump(caption_list, f, indent=4)
