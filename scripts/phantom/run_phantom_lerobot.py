import argparse
import os
from glob import glob

import cv2
import imageio.v3 as iio
import mink
import mujoco
import numpy as np
from phantom_utils import (
    ALOHA_REST_QPOS,
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

parser = argparse.ArgumentParser(description="Run Phantom on Lerobot")
parser.add_argument(
    "--calib_file",
    type=str,
    default="../../src/lfd3d/datasets/aloha_calibration/T_world_from_camera_est_left_v6_0709.txt",
    help="Cam to world calibration file",
)
parser.add_argument(
    "--intrinsics_file",
    type=str,
    default="../../src/lfd3d/datasets/aloha_calibration/intrinsics.txt",
    help="Path to intrinsics file",
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

args = parser.parse_args()

WIDTH, HEIGHT = 1280, 720
interpolate_factor = args.interpolate_factor
cam_to_world = np.loadtxt(args.calib_file)
model = load_robot_description("aloha_mj_description")
data = mujoco.MjData(model)
mink_config = mink.Configuration(model)

cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "teleoperator_pov")
K = np.loadtxt(args.intrinsics_file)
width = WIDTH // 4
height = HEIGHT // 4
K[0, 0] /= 4
K[0, 2] /= 4
K[1, 1] /= 4
K[1, 2] /= 4
setup_camera(model, cam_id, cam_to_world, width, height, K)
renderer = mujoco.Renderer(model, width=width, height=height)

INPAINT_VIDEO_DIR = "e2fgvi_vid"
DEPTH_DIR = "observation.images.cam_azure_kinect.transformed_depth"
MASK_DIR = "gsam2_masks"
HANDPOSE_DIR = "wilor_hand_pose"
VIZ_DIR = "phantom_viz_results"
OUTPUT_DIR = "phantom_retarget"
FPS = 15

videos = sorted(os.listdir(f"{args.lerobot_extradata_path}/{INPAINT_VIDEO_DIR}"))

for vid_name in tqdm(videos):
    data.qpos = ALOHA_REST_QPOS
    inpainted_video = iio.imread(
        f"{args.lerobot_extradata_path}/{INPAINT_VIDEO_DIR}/{vid_name}"
    )

    depth_video = read_depth_video(
        f"{args.lerobot_extradata_path}/{DEPTH_DIR}/{vid_name.replace('.mp4', '.mkv')}"
    )
    real_masks = np.array(
        [
            cv2.imread(i, -1)
            for i in sorted(
                glob(f"{args.lerobot_extradata_path}/{MASK_DIR}/{vid_name}_masks/*")
            )
        ]
    ).astype(bool)
    depth_video[real_masks] = 0

    hand_pose = np.load(f"{args.lerobot_extradata_path}/{HANDPOSE_DIR}/{vid_name}.npy")

    try:
        cam_human_eef_pos, cam_human_eef_rot, human_eef_artic = retarget_human_pose(
            hand_pose
        )
        n_human_original = cam_human_eef_pos.shape[0]
        n_human_interpolate = n_human_original * interpolate_factor
        cam_human_eef_pos, cam_human_eef_rot, human_eef_artic = (
            smooth_and_interpolate_pose(
                cam_human_eef_pos,
                cam_human_eef_rot,
                human_eef_artic,
                N=n_human_interpolate,
            )
        )
    except Exception as e:
        print(f"Could not retarget {vid_name} due to {e}. Skipping.")
        continue

    if args.visualize:
        os.makedirs(
            f"{args.lerobot_extradata_path}/{VIZ_DIR}/{vid_name}/", exist_ok=True
        )
        visualize_eef(
            cam_human_eef_pos,
            cam_human_eef_rot,
            f"{args.lerobot_extradata_path}/{VIZ_DIR}/{vid_name}/cam_human_retarget.ply",
        )

    # Transform retargeted action from cam to world frame
    cam_human_eef_pos_hom = np.concatenate(
        [cam_human_eef_pos, np.ones((cam_human_eef_pos.shape[0], 1))], axis=1
    )
    cam_to_world_rot = cam_to_world[:3, :3]

    world_human_eef_pos_hom = (cam_to_world @ cam_human_eef_pos_hom.T).T
    world_human_eef_pos = world_human_eef_pos_hom[:, :3]
    world_human_eef_rot = cam_to_world_rot @ cam_human_eef_rot

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

    composite_img = inpainted_video.copy()
    composite_img[ik_human_render_seg] = ik_human_render_img[ik_human_render_seg]
    composite_depth = depth_video.copy()
    composite_depth[ik_human_render_seg] = ik_human_render_depth[ik_human_render_seg]

    os.makedirs(
        f"{args.lerobot_extradata_path}/{OUTPUT_DIR}/{vid_name}/", exist_ok=True
    )
    iio.imwrite(
        f"{args.lerobot_extradata_path}/{OUTPUT_DIR}/{vid_name}/{vid_name}",
        composite_img,
        fps=FPS,
    )
    write_depth_video(
        f"{args.lerobot_extradata_path}/{OUTPUT_DIR}/{vid_name}/depth_{vid_name.replace('.mp4', '.mkv')}",
        composite_depth,
        fps=FPS,
    )
    np.savez(
        f"{args.lerobot_extradata_path}/{OUTPUT_DIR}/{vid_name}/{vid_name}_eef.npz",
        eef_pos=world_human_eef_pos,
        eef_rot=world_human_eef_rot,
        eef_artic=human_eef_artic,
        joint_state=joint_state,
    )
