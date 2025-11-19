import argparse
import json
import os
from glob import glob

import cv2
import imageio.v3 as iio
import mink
import mujoco
import numpy as np
from lfd3d.utils.viz_utils import (
    annotate_video,
    plot_barchart_with_error,
)
from phantom_utils import (
    ALOHA_REST_QPOS,
    compute_handpose_error,
    compute_metric,
    read_depth_video,
    render_with_ik,
    retarget_human_pose,
    setup_camera,
    smooth_and_interpolate_pose,
    visualize_eef,
    write_depth_video,
)
from robot_descriptions.loaders.mujoco import load_robot_description
from tqdm import tqdm

# Constants
WIDTH, HEIGHT = 1280, 720
FPS = 15
HANDPOSE_DIR = "wilor_hand_pose"
VIZ_DIR = "phantom_viz_results"
OUTPUT_DIR = "phantom_retarget"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Phantom on Lerobot")
    parser.add_argument(
        "--calib_json",
        type=str,
        default="../../src/lfd3d/datasets/aloha_calibration/multiview_calib.json",
        help="Path to calibration JSON with camera configurations",
    )
    parser.add_argument(
        "--lerobot-extradata-path",
        type=str,
        default="/data/sriram/lerobot_extradata/sriramsk/human_mug_0718",
        help="Path to auxiliary data for LeRobot dataset",
    )
    parser.add_argument(
        "--interpolate_factor",
        type=int,
        default=1,
        help="Interpolation factor for for smoother IK",
    )
    parser.add_argument(
        "--visualize", default=False, action="store_true", help="Enable visualization"
    )
    return parser.parse_args()


def load_camera_configs(calib_json_path):
    """Load camera configurations from JSON file."""
    with open(calib_json_path) as f:
        calib_data = json.load(f)
    return calib_data["cameras"]


def setup_mujoco_model():
    """Setup MuJoCo model, data, and configuration."""
    model = load_robot_description("aloha_mj_description")
    data = mujoco.MjData(model)
    mink_config = mink.Configuration(model)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "teleoperator_pov")
    return model, data, mink_config, cam_id


def load_video_data(data_path, vid_name, inpaint_dir, depth_dir, mask_dir):
    """Load all video data for a single video."""
    inpainted_video = iio.imread(f"{data_path}/{inpaint_dir}/{vid_name}")
    depth_video = read_depth_video(
        f"{data_path}/{depth_dir}/{vid_name.replace('.mp4', '.mkv')}"
    )

    real_masks = np.array(
        [
            cv2.imread(i, -1)
            for i in sorted(glob(f"{data_path}/{mask_dir}/{vid_name}_masks/*"))
        ]
    ).astype(bool)
    depth_video[real_masks] = 0

    hand_pose = np.load(f"{data_path}/{HANDPOSE_DIR}/{vid_name}.npy")

    return inpainted_video, depth_video, hand_pose


def save_video_outputs(
    data_path,
    output_vid_name,
    composite_img,
    composite_depth,
    world_human_eef_pos,
    world_human_eef_rot,
    human_eef_artic,
    joint_state,
):
    """Save composite video, depth, and pose data."""
    output_dir = f"{data_path}/{OUTPUT_DIR}/{output_vid_name}"
    os.makedirs(output_dir, exist_ok=True)

    iio.imwrite(f"{output_dir}/{output_vid_name}", composite_img, fps=FPS)
    write_depth_video(
        f"{output_dir}/depth_{output_vid_name.replace('.mp4', '.mkv')}",
        composite_depth,
        fps=FPS,
    )
    np.savez(
        f"{output_dir}/{output_vid_name.replace('.mp4', '')}_eef.npz",
        eef_pos=world_human_eef_pos,
        eef_rot=world_human_eef_rot,
        eef_artic=human_eef_artic,
        joint_state=joint_state,
    )


def process_single_video(
    vid_name,
    camera_key,
    args,
    model,
    data,
    mink_config,
    renderer,
    cam_to_world,
    inpaint_dir,
    depth_dir,
    mask_dir,
    interpolate_factor,
):
    output_vid_name = f"{vid_name.replace('.mp4', '')}_{camera_key}.mp4"
    if os.path.exists(f"{args.lerobot_extradata_path}/{OUTPUT_DIR}/{output_vid_name}"):
        print(
            f"Phantom output already saved at {args.lerobot_extradata_path}/{OUTPUT_DIR}/{output_vid_name}. Skipping"
        )
        return

    """Process a single video file."""
    data.qpos = ALOHA_REST_QPOS

    # Load video data
    inpainted_video, depth_video, hand_pose = load_video_data(
        args.lerobot_extradata_path, vid_name, inpaint_dir, depth_dir, mask_dir
    )

    # NOTE: Hand pose is expected to be in world frame.

    # Retarget and smooth hand pose
    try:
        world_human_eef_pos, world_human_eef_rot, human_eef_artic = retarget_human_pose(
            hand_pose
        )
        n_human_original = world_human_eef_pos.shape[0]
        n_human_interpolate = n_human_original * interpolate_factor
        world_human_eef_pos, world_human_eef_rot, human_eef_artic = (
            smooth_and_interpolate_pose(
                world_human_eef_pos,
                world_human_eef_rot,
                human_eef_artic,
                N=n_human_interpolate,
            )
        )
    except Exception as e:
        print(f"Could not retarget {vid_name} due to {e}. Skipping.")
        return None

    if args.visualize:
        os.makedirs(
            f"{args.lerobot_extradata_path}/{VIZ_DIR}/{vid_name}/", exist_ok=True
        )
        visualize_eef(
            world_human_eef_pos,
            world_human_eef_rot,
            f"{args.lerobot_extradata_path}/{VIZ_DIR}/{vid_name}/world_human_retarget.ply",
        )

    # Render with IK
    (
        ik_human_render_img,
        ik_human_render_seg,
        ik_human_render_depth,
        world_human_actual_eef_pos,
        world_human_actual_eef_rot,
        world_human_actual_eef_artic,
        joint_state,
    ) = render_with_ik(
        model,
        mink_config,
        renderer,
        data,
        world_human_eef_pos,
        world_human_eef_rot,
        human_eef_artic,
        n=interpolate_factor,
    )

    if args.visualize:
        visualize_eef(
            world_human_eef_pos,
            world_human_eef_rot,
            f"{args.lerobot_extradata_path}/{VIZ_DIR}/{vid_name}/world_human_retarget.ply",
        )
        visualize_eef(
            world_human_actual_eef_pos,
            world_human_actual_eef_rot,
            f"{args.lerobot_extradata_path}/{VIZ_DIR}/{vid_name}/world_human_actual_retarget.ply",
        )
        iio.imwrite(
            f"{args.lerobot_extradata_path}/{VIZ_DIR}/{vid_name}/ik_human_img_{vid_name}",
            ik_human_render_img,
            fps=FPS,
        )
        iio.imwrite(
            f"{args.lerobot_extradata_path}/{VIZ_DIR}/{vid_name}/ik_human_seg_{vid_name}",
            ik_human_render_seg.astype(np.uint8) * 255,
            fps=FPS,
        )

    # Compute metrics
    eef_mse, rot_error, std_eef_mse, std_rot_error = compute_handpose_error(
        world_human_eef_pos,
        world_human_eef_rot,
        world_human_actual_eef_pos,
        world_human_actual_eef_rot,
        f"{args.lerobot_extradata_path}/{VIZ_DIR}/{vid_name}"
        if args.visualize
        else None,
    )

    # Resize renders to full resolution
    ik_human_render_seg = np.array(
        [cv2.resize(i, (WIDTH, HEIGHT)) for i in ik_human_render_seg.astype(np.uint8)]
    ).astype(bool)
    ik_human_render_img = np.array(
        [cv2.resize(i, (WIDTH, HEIGHT)) for i in ik_human_render_img]
    )
    ik_human_render_depth = np.array(
        [
            cv2.resize(i, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
            for i in ik_human_render_depth
        ]
    )

    # Create composites
    composite_img = inpainted_video.copy()
    composite_img[ik_human_render_seg] = ik_human_render_img[ik_human_render_seg]
    if args.visualize:
        annot_video = composite_img.copy()
        annot_video = annotate_video(
            annot_video, {"t(m)": eef_mse, "r(deg)": rot_error}
        )
        iio.imwrite(
            f"{args.lerobot_extradata_path}/{VIZ_DIR}/{vid_name}/annot_video_{vid_name}",
            annot_video,
            fps=FPS,
        )
    composite_depth = depth_video.copy()
    composite_depth[ik_human_render_seg] = ik_human_render_depth[ik_human_render_seg]

    # Save outputs
    save_video_outputs(
        args.lerobot_extradata_path,
        output_vid_name,
        composite_img,
        composite_depth,
        world_human_eef_pos,
        world_human_eef_rot,
        human_eef_artic,
        joint_state,
    )

    return {
        "eef_mse": np.mean(eef_mse),
        "rot_error": np.mean(rot_error),
        "std_eef_mse": std_eef_mse,
        "std_rot_error": std_rot_error,
        "num_samples": eef_mse.shape[0],
    }


def process_camera(
    camera_config,
    args,
    model,
    data,
    mink_config,
    cam_id,
    WIDTH,
    HEIGHT,
    interpolate_factor,
    FPS,
):
    """Process all videos for a single camera view."""
    camera_key = camera_config["name"]

    # Load camera-specific calibration
    calib_dir = os.path.dirname(args.calib_json)
    cam_to_world = np.loadtxt(os.path.join(calib_dir, camera_config["extrinsics"]))
    K = np.loadtxt(os.path.join(calib_dir, camera_config["intrinsics"]))

    # Setup camera
    width = WIDTH // 4
    height = HEIGHT // 4
    K[0, 0] /= 4
    K[0, 2] /= 4
    K[1, 1] /= 4
    K[1, 2] /= 4
    setup_camera(model, cam_id, cam_to_world, width, height, K)
    renderer = mujoco.Renderer(model, width=width, height=height)

    # Camera-specific directory paths
    INPAINT_VIDEO_DIR = f"e2fgvi_vid/{camera_key}"
    DEPTH_DIR = f"observation.images.{camera_key}.transformed_depth"
    MASK_DIR = f"gsam2_masks/{camera_key}"

    print(f"Processing camera: {camera_key}")
    videos = sorted(os.listdir(f"{args.lerobot_extradata_path}/{INPAINT_VIDEO_DIR}"))
    overal_metric = {
        "eef_mse": [],
        "rot_error": [],
        "std_eef_mse": [],
        "std_rot_error": [],
        "num_sample_each_eposide": [],
    }

    for vid_name in tqdm(videos, desc=f"{camera_key}"):
        result = process_single_video(
            vid_name,
            camera_key,
            args,
            model,
            data,
            mink_config,
            renderer,
            cam_to_world,
            INPAINT_VIDEO_DIR,
            DEPTH_DIR,
            MASK_DIR,
            interpolate_factor,
        )

        if result is not None:
            overal_metric["eef_mse"].append(result["eef_mse"])
            overal_metric["rot_error"].append(result["rot_error"])
            overal_metric["std_eef_mse"].append(result["std_eef_mse"])
            overal_metric["std_rot_error"].append(result["std_rot_error"])
            overal_metric["num_sample_each_eposide"].append(result["num_samples"])

    # Plot metrics for this camera
    plot_barchart_with_error(
        overal_metric["eef_mse"],
        overal_metric["std_eef_mse"],
        f"Overal MSE eef metric - {camera_key}",
        "Episode index",
        "MSE (m)",
        os.path.join(
            args.lerobot_extradata_path,
            OUTPUT_DIR,
            f"episodes_mean_std_mse_error_{camera_key}.png",
        ),
    )
    plot_barchart_with_error(
        overal_metric["rot_error"],
        overal_metric["std_rot_error"],
        f"Overal Rot error metric - {camera_key}",
        "Episode index",
        "Rot error (degree)",
        os.path.join(
            args.lerobot_extradata_path,
            OUTPUT_DIR,
            f"episodes_mean_std_rot_error_{camera_key}.png",
        ),
    )
    compute_metric(
        overal_metric,
        os.path.join(
            args.lerobot_extradata_path, OUTPUT_DIR, f"overal_metric_{camera_key}.json"
        ),
    )


def main():
    """Main function to process all cameras."""
    args = parse_args()
    cameras = load_camera_configs(args.calib_json)
    model, data, mink_config, cam_id = setup_mujoco_model()
    interpolate_factor = args.interpolate_factor

    # Process all cameras
    for camera_config in cameras:
        process_camera(
            camera_config,
            args,
            model,
            data,
            mink_config,
            cam_id,
            WIDTH,
            HEIGHT,
            interpolate_factor,
            FPS,
        )
    print("Done!")


if __name__ == "__main__":
    main()
