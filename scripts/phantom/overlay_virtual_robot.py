"""
Overlay a virtual robot on the real robot using joint data from a LeRobot dataset.

This script takes a source dataset, copies it to a destination, and then renders
a virtual robot overlay on each frame using the joint angles from the dataset.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import imageio.v3 as iio
import mujoco
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from phantom_utils import (
    ALOHA_REST_QPOS,
    convert_real_joints,
    render_rightArm_images,
    setup_camera,
    write_depth_video,
)
from robot_descriptions.loaders.mujoco import load_robot_description
from tqdm import tqdm

# Constants
WIDTH, HEIGHT = 1280, 720
FPS = 15


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Overlay virtual robot on real robot using joint data"
    )
    parser.add_argument(
        "--dset_path",
        type=str,
        default="/home/sriram/.cache/huggingface/lerobot/",
        help="Path to source LeRobot dataset directory",
    )
    parser.add_argument(
        "--source_dset",
        type=str,
        required=True,
        help="Source dataset id",
    )
    parser.add_argument(
        "--dest_dset",
        type=str,
        required=True,
        help="Destination dataset id (will be created)",
    )
    parser.add_argument(
        "--calib_json",
        type=str,
        default="../../src/lfd3d/datasets/aloha_calibration/multiview_calib.json",
        help="Path to calibration JSON with camera configurations",
    )
    return parser.parse_args()


def load_camera_configs(calib_json_path):
    """Load camera configurations from JSON file."""
    with open(calib_json_path) as f:
        calib_data = json.load(f)
    return calib_data["cameras"]


def setup_mujoco_model():
    """Setup MuJoCo model and data."""
    model = load_robot_description("aloha_mj_description")
    data = mujoco.MjData(model)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "teleoperator_pov")
    return model, data, cam_id


def render_virtual_robot_for_frame(model, data, renderer, joint_angles):
    """
    Render virtual robot for a single frame using joint angles.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        renderer: MuJoCo renderer
        joint_angles: (18,) array of joint angles in degrees (from observation.state)

    Returns:
        rgb: (H, W, 3) rendered RGB image
        depth: (H, W) rendered depth image
        seg: (H, W) segmentation mask
    """
    # Convert real robot joint angles (18-DOF, degrees) to sim joint angles (16-DOF, radians)
    sim_qpos = convert_real_joints(joint_angles).numpy()

    # Set the joint positions
    data.qpos = sim_qpos
    mujoco.mj_forward(model, data)

    # Render the robot
    rgb, depth, seg = render_rightArm_images(renderer, data)

    return rgb, depth, seg


def process_camera_videos(
    source_dataset_path,
    dest_dataset_path,
    camera_config,
    args,
    model,
    data,
    cam_id,
    lerobot_dataset,
):
    """Process all videos for a single camera view."""
    camera_key = camera_config["name"]

    # Load camera-specific calibration
    calib_dir = os.path.dirname(args.calib_json)
    cam_to_world = np.loadtxt(os.path.join(calib_dir, camera_config["extrinsics"]))
    K = np.loadtxt(os.path.join(calib_dir, camera_config["intrinsics"]))

    # Setup camera at lower resolution for faster rendering
    width = WIDTH // 4
    height = HEIGHT // 4
    K_scaled = K.copy()
    K_scaled[0, 0] /= 4
    K_scaled[0, 2] /= 4
    K_scaled[1, 1] /= 4
    K_scaled[1, 2] /= 4
    setup_camera(model, cam_id, cam_to_world, width, height, K_scaled)
    renderer = mujoco.Renderer(model, width=width, height=height)

    # Video directory paths in the dataset
    COLOR_VIDEO_DIR = f"observation.images.{camera_key}.color"
    DEPTH_VIDEO_DIR = f"observation.images.{camera_key}.transformed_depth"

    videos_dir = Path(dest_dataset_path) / "videos"

    print(f"Processing camera: {camera_key}")

    # Get number of episodes
    num_episodes = lerobot_dataset.num_episodes

    for episode_idx in tqdm(range(num_episodes), desc=f"Processing {camera_key}"):
        # Get episode bounds
        episode_start = lerobot_dataset.episode_data_index["from"][episode_idx].item()
        episode_end = lerobot_dataset.episode_data_index["to"][episode_idx].item()
        episode_length = episode_end - episode_start

        # Collect all frames for this episode
        rgb_frames = []
        depth_frames = []

        for frame_idx in tqdm(range(episode_length)):
            # Load frame from dataset
            frame = lerobot_dataset[episode_start + frame_idx]

            # Get joint angles (observation.state is in degrees)
            joint_angles = frame["observation.state"].numpy()

            # Render virtual robot at lower resolution
            virtual_rgb, virtual_depth, virtual_seg = render_virtual_robot_for_frame(
                model, data, renderer, joint_angles
            )

            # Get original RGB and depth from dataset
            original_rgb = (
                frame[COLOR_VIDEO_DIR].permute(1, 2, 0).numpy() * 255
            ).astype(np.uint8)
            original_depth = frame[DEPTH_VIDEO_DIR].numpy()[
                0
            ]  # Remove channel dimension

            # Resize virtual robot renders to full resolution
            virtual_seg_resized = cv2.resize(
                virtual_seg.astype(np.uint8), (WIDTH, HEIGHT)
            ).astype(bool)
            virtual_rgb_resized = cv2.resize(virtual_rgb, (WIDTH, HEIGHT))
            virtual_depth_resized = cv2.resize(
                virtual_depth, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST
            )

            # Create composite by overlaying virtual robot on real robot
            composite_rgb = original_rgb.copy()
            composite_rgb[virtual_seg_resized] = virtual_rgb_resized[
                virtual_seg_resized
            ]

            composite_depth = original_depth.copy()
            composite_depth[virtual_seg_resized] = virtual_depth_resized[
                virtual_seg_resized
            ]

            rgb_frames.append(composite_rgb)
            depth_frames.append(composite_depth)

        # Save videos back to the dataset directory
        # Video files are stored in videos/episode_{episode_idx:06d}/{COLOR_VIDEO_DIR}/chunk-000.mp4
        episode_dir = videos_dir / "chunk-000"

        # Save RGB video
        rgb_video_dir = episode_dir / COLOR_VIDEO_DIR
        rgb_video_path = rgb_video_dir / f"episode_{episode_idx:06d}.mp4"
        if rgb_video_path.exists():
            print(f"Overwriting RGB video: {rgb_video_path}")
            iio.imwrite(str(rgb_video_path), np.array(rgb_frames), fps=FPS)

        # Save depth video
        depth_video_dir = episode_dir / DEPTH_VIDEO_DIR
        depth_video_path = depth_video_dir / "chunk-000.mkv"
        if depth_video_path.exists():
            print(f"Overwriting depth video: {depth_video_path}")
            write_depth_video(str(depth_video_path), np.array(depth_frames), fps=FPS)


def main():
    """Main function."""
    args = parse_args()

    source_path = f"{args.dset_path}/{args.source_dset}"
    dest_path = f"{args.dset_path}/{args.dest_dset}"

    # Copy entire dataset from source to destination
    print(f"Copying dataset from {source_path} to {dest_path}...")
    assert not os.path.exists(dest_path), f"{dest_path} exists."

    shutil.copytree(source_path, dest_path)
    print("Dataset copied successfully.")

    # Load the dataset
    print(f"Loading dataset - {args.dest_dset}...")
    lerobot_dataset = LeRobotDataset(repo_id=args.dest_dset, tolerance_s=0.0004)

    # Load camera configs
    cameras = load_camera_configs(args.calib_json)

    # Setup MuJoCo
    model, data, cam_id = setup_mujoco_model()
    data.qpos = ALOHA_REST_QPOS

    # Process all cameras
    for camera_config in cameras:
        process_camera_videos(
            source_path,
            dest_path,
            camera_config,
            args,
            model,
            data,
            cam_id,
            lerobot_dataset,
        )

    print(f"\nDone! Virtual robot overlays saved in-place to: {args.dest_dset}")


if __name__ == "__main__":
    main()
