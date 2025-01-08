"""A script to check the GPU usage on the autobot cluster.

Usage:
    python autobot.py status
    python autobot.py all_available --gpu_type rtx2080
    python autobot.py available --gpu_type rtx2080 --num_gpus 2
"""

import logging
import subprocess
from enum import Enum
from io import StringIO
from typing import List, Optional

import pandas as pd
import typer

app = typer.Typer()

AUTOBOT_URL = "autobot.vision.cs.cmu.edu"

NODE_INFO = {
    "autobot-0-9": {"RTX_2080_Ti": 4},
    "autobot-0-11": {"RTX_2080_Ti": 4},
    "autobot-0-13": {"RTX_2080_Ti": 4},
    "autobot-0-15": {"RTX_2080_Ti": 4},
    "autobot-0-17": {"RTX_2080_Ti": 3},
    "autobot-0-19": {"RTX_2080_Ti": 4},
    "autobot-0-21": {"RTX_2080_Ti": 4},
    "autobot-0-23": {"RTX_2080_Ti": 4},
    "autobot-0-25": {"RTX_3090": 8},
    "autobot-0-29": {"RTX_3090": 8},
    "autobot-0-33": {"RTX_3090": 8},
    "autobot-0-37": {"RTX_3090": 8},
    "autobot-1-1": {"RTX_2080_Ti": 10},
    "autobot-1-10": {"RTX_3080_Ti": 8},
    "autobot-1-14": {"RTX_3080_Ti": 8},
    "autobot-1-18": {"RTX_A6000": 8},
}


# GPU type enum:
class GPUType(str, Enum):
    RTX_2080_Ti = "rtx2080"
    RTX_3090 = "rtx3090"
    RTX_3080_Ti = "rtx3080"
    RTX_A6000 = "rtxa6000"


NODE_TYPES = {
    GPUType.RTX_2080_Ti: {
        "autobot-0-9",
        "autobot-0-11",
        "autobot-0-13",
        "autobot-0-15",
        "autobot-0-19",
        "autobot-0-21",
        "autobot-0-23",
        "autobot-1-1",
    },
    GPUType.RTX_3080_Ti: {
        "autobot-1-10",
        "autobot-1-14",
    },
    GPUType.RTX_3090: {
        "autobot-0-25",
        "autobot-0-29",
        "autobot-0-33",
        "autobot-0-37",
    },
    GPUType.RTX_A6000: {
        "autobot-1-18",
    },
}

DEBUG = False

# Get the GPU current usage info.
USAGE_CMD = (
    r"nvidia-smi --query-gpu=index,name,memory.used,memory.total,gpu_uuid --format=csv"
)
# Get the PID and serial number.
PID_USAGE_CMD = r"nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv"
# get a list of all the pids for each.
# USER_LIST_CMD = r"nvidia-smi -q -x | grep pid | sed -e 's/<pid>//g' -e 's/<\/pid>//g' -e 's/^[[:space:]]*//' | xargs -r ps --no-headers -up"
# Adding the 'true' should swallow the error command, in case it fails...
USER_LIST_CMD = r"nvidia-smi -q -x | grep pid | sed -e 's/<pid>//g' -e 's/<\/pid>//g' -e 's/^[[:space:]]*//' | xargs -r ps --no-headers -up || true"


def parse_usage_cmd(response):
    """Returns a DF with the columns:
    - "index": The index for CUDA_VISIBLE_DEVICES purposes.
    - "name": Name of the GPU (e.g. RTX 2080)
    - "memory.used [MiB]"
    - "memory.total [MiB]"
    - "gpu_uuid": Serial number (primary key).
    Where each row is a unique GPU
    """
    df = pd.read_csv(StringIO(response), sep=", ", engine="python")
    df = df.rename(columns={"uuid": "gpu_uuid"})
    return df


def parse_pid_usage_cmd(response):
    """Returns a DF with the columns:
    - "pid": Process ID.
    - "gpu_uuid": Serial number (primary key).
    - "used_gpu_memory [MiB]"
    Where each row is a process consuming GPU memory.
    """
    return pd.read_csv(StringIO(response), sep=", ", engine="python")


def parse_user_list_cmd(response):
    """Returns a DF with the columns:
    - "username": Name of user.
    - "pid": Process ID
    Where each row is a process consuming GPU memory.
    """
    lines = response.split("\n")
    new_resp = []
    for line in lines:
        if line == "":
            continue
        # .split() will remove variable length spaces.
        new_resp.append(",".join(line.split()[:2]))
    new_resp_str = "\n".join(new_resp)
    return pd.read_csv(StringIO(new_resp_str), sep=",", names=["username", "pid"])


def safe_merge(usage_df, pid_usage_df, user_map_df):
    # Sometimes, we don't have a user for user_map_df - e.g. if ps doesn't return something coherent. For any process ID which exists in pid_usage_df (but not in user_map_df), make the username "unknown".

    # Merge everything.
    user_map_df = user_map_df.set_index("pid")
    pid_usage_df["username"] = pid_usage_df["pid"].apply(
        lambda pid: (
            user_map_df.loc[pid]["username"] if pid in user_map_df.index else "unknown"
        )
    )
    process_df = pid_usage_df
    process_df = pd.merge(usage_df, process_df, on="gpu_uuid")

    usage_df["processes"] = usage_df["gpu_uuid"].apply(
        lambda gpu_uuid: "\n".join(
            process_df[process_df["gpu_uuid"] == gpu_uuid].apply(
                lambda row: f"{row['username']} ({row['pid']}) [{row['memory.used [MiB]']}]",
                axis=1,
            )
        )
    )

    usage_df = usage_df.drop(columns=["gpu_uuid"])

    return usage_df


def execute_command_chain_on_node(node_name, commands, username=None, local=False):
    """Run a command chain on a node (e.g. to limit the number of SSH'es. But is brittle)."""
    split_token = "END_OF_COMMAND_OUTPUT"
    new_commands = []
    for command in commands:
        new_commands.append(command)
        new_commands.append(f"echo {split_token}")
    new_commands = new_commands[:-1]
    joined_cmd = " && ".join(new_commands)
    outputs = execute_command_on_node(
        node_name, joined_cmd, username=username, local=local
    )

    return outputs.split(split_token)


def execute_command_on_node(node_name, command, username=None, local=False):
    # If we're running this locally on autobot, no need to double-ssh.
    if local:
        cmd = ["ssh", node_name, command]
        if DEBUG:
            logging.info(" ".join(cmd))
        output = subprocess.check_output(cmd, text=True)
        return output
    else:
        if username is None:
            raise ValueError("username must be provided if local=True")
        cmd = ["ssh", f"{username}@{AUTOBOT_URL}", f'ssh {node_name} "{command}"']
        if DEBUG:
            logging.info(" ".join(cmd))
        output = subprocess.check_output(cmd, text=True)
        return output


def get_complete_usage_df(
    node_name: str, username: str = "sskrishn", local: bool = False
):
    raw_results = execute_command_chain_on_node(
        node_name=node_name,
        commands=[
            USAGE_CMD,
            PID_USAGE_CMD,
            USER_LIST_CMD,
        ],
        username=username,
        local=local,
    )

    usage_df = parse_usage_cmd(raw_results[0])
    pid_usage_df = parse_pid_usage_cmd(raw_results[1])
    user_map_df = parse_user_list_cmd(raw_results[2])

    merged_usage_df = safe_merge(usage_df, pid_usage_df, user_map_df)

    return merged_usage_df


def get_available_gpus(merged_usage_df) -> List[int]:
    # Get the index of GPUs where "processes" is empty.
    available_gpus = merged_usage_df[merged_usage_df["processes"] == ""]["index"]
    return list(available_gpus)


@app.command()
def status(
    nodes: Optional[List[str]] = None, username: str = "sskrishn", local: bool = False
):
    if not nodes:
        nodes = list(NODE_INFO.keys())
    logging.basicConfig(level=logging.INFO)
    for node in nodes:
        print(f"--------Node {node}----------")
        try:
            merged_usage_df = get_complete_usage_df(
                node, username=username, local=local
            )
            print(merged_usage_df)
        except Exception as e:
            logging.info(f"\t FAILED: {e}")


@app.command()
def all_available(gpu_type: GPUType, username: str = "sskrishn", local: bool = False):
    logging.basicConfig(level=logging.INFO)
    nodes = NODE_TYPES[gpu_type]

    total = 0
    for node in nodes:
        try:
            merged_usage_df = get_complete_usage_df(
                node, username=username, local=local
            )
            available_gpus = get_available_gpus(merged_usage_df)
            total += len(available_gpus)
            print(
                f"Node {node}: {len(available_gpus)} available GPUs: {available_gpus}"
            )
        except Exception as e:
            logging.info(f"\t FAILED: {e}")
    print(f"TOTAL available GPUs: {total}")


def get_first_available(
    gpu_type: GPUType, num_gpus: int, username: str = "sskrishn", local: bool = False
):
    nodes = NODE_TYPES[gpu_type]
    for node in nodes:
        try:
            merged_usage_df = get_complete_usage_df(
                node, username=username, local=local
            )
            available_gpus = get_available_gpus(merged_usage_df)
            if len(available_gpus) >= num_gpus:
                return node, available_gpus[:num_gpus]
        except Exception as e:
            logging.info(f"\t FAILED: {e}")
    return None, None


@app.command()
def available(
    gpu_type: GPUType,
    num_gpus: int,
    username: str = "sskrishn",
    local: bool = False,
    quiet: bool = False,
):
    node, gpus = get_first_available(gpu_type, num_gpus, username=username, local=local)

    # Machine-readable output, csv format.
    if quiet:
        if node is None:
            print(f"No available GPUs found for {gpu_type} on any node.")
            exit(1)
        else:
            print(f"{node}:{','.join(map(str, gpus))}")
        return

    if node is None:
        print(f"No available GPUs found for {gpu_type} on any node.")
    else:
        print(f"Found {num_gpus} available GPUs for {gpu_type} on node {node}: {gpus}")


if __name__ == "__main__":
    app()
