import numpy as np
import open3d as o3d
from lfd3d.datasets.lerobot.lerobot_dataset import RpadLeRobotDataset
from omegaconf import OmegaConf

cfg_dict = {
    "dataset": {
        "name": "rpadLerobot",
        "cache_dir": None,
        "cache_invalidation_rate": 0.05,
        "train_size": None,
        "val_size": None,
        "num_points": 8192,
        "max_depth": 1.5,
        "normalize": False,
        "augment_train": None,
        "augment_cfg": {
            "augment_prob": 0.75,
            "augment_transform": True,
            "pcd_sample": ["voxel", "fps"],
            "fps_num_points": [4096, 8192],
            "voxel_size": [0.01, 0.02],
        },
        "color_key": "observation.images.cam_azure_kinect.color",
        "depth_key": "observation.images.cam_azure_kinect.transformed_depth",
        "gripper_pcd_key": "observation.points.gripper_pcds",
        "data_name": "0627",
        "additional_img_dir": None,
        "data_sources": ["aloha"],
        "oversample_robot": True,
        "rgb_feat": False,
        "use_full_text": True,
        "use_intermediate_frames": False,
        "use_subgoals": False,
        # "repo_id": ["sriramsk/fold_onesie_20250831_subsampled_heatmapGoal",
        #             "sriramsk/fold_shirt_20250918_subsampled_heatmapGoal",
        #             "sriramsk/fold_towel_20250919_subsampled_heatmapGoal",
        #             "sriramsk/fold_bottoms_20250919_human_heatmapGoal"],
        "repo_id": "sriramsk/fold_bottoms_20250919_human_heatmapGoal",
    },
    "model": {
        "name": "articubot",
        "type": "cross_displacement",
        "in_channels": 4,
        "num_classes": 13,
        "keep_gripper_in_fps": False,
        "add_action_pcd_masked": True,
        "use_text_embedding": True,
        "use_rgb": False,
        "is_gmm": False,
        "fixed_variance": [0.01, 0.05, 0.1, 0.25, 0.5],
        "uniform_weights_coeff": 0.1,
    },
    "training": {
        "lr": 0.0001,
        "lr_warmup_steps": 100,
        "weight_decay": 1e-05,
        "epochs": 100,
        "precision": 32,
        "batch_size": 8,
        "val_batch_size": 8,
        "grad_clip_norm": 0.3,
        "num_training_steps": "None",
        "check_val_every_n_epochs": 10,
        "additional_train_logging_period": 100,
        "n_samples_wta": 5,
        "save_wta_to_disk": False,
    },
    "log_dir": "${hydra:runtime.cwd}/logs",
    "output_dir": "${hydra:runtime.output_dir}",
    "job_type": "${mode}_${dataset.name}",
    "lightning": {"checkpoint_dir": "${output_dir}/checkpoints"},
    "wandb": {
        "entity": "r-pad",
        "project": "lfd3d",
        "group": None,
        "save_dir": "${output_dir}",
        "artifact_dir": "${hydra:runtime.cwd}/wandb_artifacts",
    },
    "mode": "train",
    "seed": 42,
    "resources": {"num_workers": 0, "gpus": [0]},
    "load_checkpoint": False,
    "checkpoint": {
        "run_id": False,
        "type": "rmse",
        "reference": "${wandb.entity}/${wandb.project}/best_${checkpoint.type}_model-${checkpoint.run_id}:best",
    },
    "lora": {"enable": False, "rank": 4, "target_modules": "all", "dropout": 0.1},
}
cfg = OmegaConf.create(cfg_dict)
lr_dset = RpadLeRobotDataset(dataset_cfg=cfg.dataset, augment_train=None)
item = lr_dset[0]


def gripper_points_to_rotation(gripper_center, palm_point, finger_point):
    # Always use palm->gripper as primary axis (more stable)
    forward = gripper_center - palm_point
    x_axis = forward / np.linalg.norm(forward, axis=0, keepdims=True)

    # Use finger relative to the forward direction for secondary axis
    finger_vec = gripper_center - finger_point

    # Project finger vector onto plane perpendicular to forward
    finger_projected = (
        finger_vec - np.sum(finger_vec * x_axis, axis=0, keepdims=True) * x_axis
    )
    y_axis = finger_projected / np.linalg.norm(finger_projected, axis=0, keepdims=True)

    # Z completes the frame
    z_axis = np.cross(x_axis, y_axis)

    return np.stack([x_axis, y_axis, z_axis], axis=-1)


def get_gripper_pose(gripper_points):
    """
    Extract gripper 6DoF pose
    """
    gripper_pos = (gripper_points[0, :] + gripper_points[1, :]) / 2
    # eef pose, base, right finger
    gripper_rot = gripper_points_to_rotation(
        gripper_pos, gripper_points[2, :], gripper_points[0, :]
    )
    return gripper_pos, gripper_rot


action_pcd = o3d.geometry.PointCloud()
action_pcd.points = o3d.utility.Vector3dVector(item["action_pcd"].copy())
anchor_pcd = o3d.geometry.PointCloud()
anchor_pcd.points = o3d.utility.Vector3dVector(item["anchor_pcd"].copy())

gripper_points = item["action_pcd"][item["gripper_idx"]]
gripper_pos, gripper_rot = get_gripper_pose(gripper_points)
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
# transform it
T = np.eye(4)
T[:3, :3] = gripper_rot  # 3x3 rotation
T[:3, 3] = gripper_pos  # translation
frame.transform(T)

o3d.visualization.draw_geometries([action_pcd, anchor_pcd, frame])
