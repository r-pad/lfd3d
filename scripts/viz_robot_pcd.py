import numpy as np
import open3d as o3d
from lfd3d.datasets.lerobot.lerobot_dataset import RpadLeRobotDataset
from omegaconf import OmegaConf
from tqdm import tqdm


cfg_dict_robot = {
    "dataset": {
        "cache_dir": "/home/haotian/.cache",
        "cache_invalidation_rate": 0.1,
        "cameras": [ 
            # {
            #         "name": "",
            #         "color_key": "observation.images.cam_azure_kinect.color",
            #         "depth_key": "observation.images.cam_azure_kinect.transformed_depth",
            #         "extrinsics": "aloha_calibration/T_world_from_camera_est_v7_1013.txt",
            #         "intrinsics": "aloha_calibration/intrinsics_000077925212.txt",
            # },
            {
                    "name": "front",
                    "color_key": "observation.images.cam_azure_kinect_front.color",
                    "depth_key": "observation.images.cam_azure_kinect_front.transformed_depth",
                    "extrinsics": "aloha_calibration/T_world_from_camera_front_v1_1020.txt",
                    "intrinsics": "aloha_calibration/intrinsics_000259921812.txt"
            },
            # {
            #         "name": "back",
            #         "color_key": "observation.images.cam_azure_kinect_back.color",
            #         "depth_key": "observation.images.cam_azure_kinect_back.transformed_depth",
            #         "extrinsics": "aloha_calibration/T_world_from_camera_back_v1_1020.txt",
            #         "intrinsics": "aloha_calibration/intrinsics_000003493812.txt",
            # }
        ],
        "data_dir": None,
        "gripper_pcd_key": "observation.points.gripper_pcds",
        "max_depth": 2.0,
        "name": "rpadLerobot",
        "normalize": False,
        "num_points": 8192,
        "repo_id": [#"sriramsk/fold_bottoms_MV_20251031_ss_hg"
                    #"sriramsk/fold_towel_MV_20251030_ss_hg",
                    #"sriramsk/fold_towel_MV_20251030_ss_hg_debug_9",
                    #"sriramsk/fold_onesie_MV_20251025_ss_hg"
                    #"sriramsk/fold_towel_20250919_debug_0"
                    "sriramsk/fold_towel_MV_20251030_ss_hg_debug_3"
                    ],
        "rgb_feat": False,
        "train_size": None,
        "use_subgoals": False,
        "val_size": None
    },
    "job_type": "train_rpadLerobot",
    "lightning": {
        "checkpoint_dir": "/project_data/held/haotian/lfd3d/logs/train_rpadLerobot/2025-11-09/02-29-18/checkpoints"
    },
    "load_checkpoint": False,
    "log_dir": "/project_data/held/haotian/lfd3d/logs",
    "lora": {
        "dropout": 0.1,
        "enable": False,
        "rank": 4,
        "target_modules": "all"
    },
    "mode": "train",
    "model": {
        "dino_model": "facebook/dinov2-base",
        "dropout": 0.1,
        "fixed_variance": [
            0.01,
            0.05,
            0.1,
            0.25,
            0.5
        ],
        "fourier_include_input": True,
        "fourier_num_frequencies": 21,
        "is_gmm": True,
        "name": "dino_3dgp",
        "num_transformer_layers": 4,
        "type": "cross_displacement",
        "uniform_weights_coeff": 0.1,
        "use_fourier_pe": False,
        "use_gripper_token": True,
        "use_source_token": True,
        "use_text_embedding": True
    },
    "output_dir": "/project_data/held/haotian/lfd3d/logs/train_rpadLerobot/2025-11-09/02-29-18",
    "resources": {
        "gpus": -1,
        "num_workers": 4
    },
    "seed": 42,
    "training": {
        "additional_train_logging_period": 100,
        "augment_cfg": {
            "augment_prob": 0.75,
            "image": {
                "blur_kernel_size": 1,
                "blur_prob": 0.1,
                "brightness": 0.2,
                "color_jitter_prob": 0.3,
                "contrast": 0.2,
                "crop_resize_prob": 0.3,
                "crop_size_range": [
                    192,
                    224
                ],
                "grayscale_prob": 0.1,
                "hflip_prob": 0.5,
                "hue": 0.1,
                "rotate_prob": 0.3,
                "saturation": 0.2
            },
            "pcd": {
                "augment_transform": True,
                "fps_num_points": [
                    4096,
                    8192
                ],
                "pcd_sample": [
                    "voxel",
                    "fps"
                ],
                "voxel_size": [
                    0.01,
                    0.02
                ]
            }
        },
        "augment_train": "image_color_only",
        "batch_size": 4,
        "check_val_every_n_epochs": 5,
        "checkpoints": {
            "rmse": {
                "mode": "min",
                "monitor": "val/rmse"
            },
            "rmse_and_std_combi": {
                "mode": "min",
                "monitor": "val/rmse_and_std_combi"
            }
        },
        "epochs": 50,
        "grad_clip_norm": 0.3,
        "lr": 0.0001,
        "lr_warmup_steps": 100,
        "n_samples_wta": 5,
        "num_training_steps": 838850,
        "precision": 32,
        "save_wta_to_disk": False,
        "val_batch_size": 4,
        "weight_decay": 1e-05
    },
    "wandb": {
        "artifact_dir": "/project_data/held/haotian/lfd3d/wandb_artifacts",
        "entity": "hz2851-carnegie-mellon-university",
        "group": None,
        "name": "libero_130_hl_3d",
        "project": "lfd3d",
        "save_dir": "/project_data/held/haotian/lfd3d/logs/train_rpadLerobot/2025-11-09/02-29-18"
    }
}


cfg_robot = OmegaConf.create(cfg_dict_robot)

lr_dset_robot= RpadLeRobotDataset(dataset_cfg=cfg_robot.dataset, augment_train=None)


scene_pcds = {0:[]}
robot_pcds = {0:[]}

for i in range(len(lr_dset_robot)):
    lerobot_data = lr_dset_robot.lerobot_dataset[i]
    episode_idx = lerobot_data["episode_index"].item()
    if episode_idx == 1:
        break
    if episode_idx not in scene_pcds.keys():
        scene_pcds[episode_idx] = []
        robot_pcds[episode_idx] = []

    #scene_pcds[episode_idx].append(lr_dset_robot[i]["anchor_pcd"])
    robot_pcds[episode_idx].append(lerobot_data["observation.points.gripper_pcds"])

    print(i, episode_idx)

name = cfg_dict_robot["dataset"]["repo_id"][0].split("/")[-1]
# for k,v in scene_pcds.items():
#     np.save(f"scene_{name}_episode_{k}.npy", np.array(v))

for k,v in robot_pcds.items():
    np.save(f"robot_{name}_episode_{k}.npy", np.array(v))

# breakpoint()
# vis_points = []

# for point in robot_points:
#     action_pcd = o3d.geometry.PointCloud()
#     action_pcd.points = o3d.utility.Vector3dVector(point.copy())
#     action_pcd.paint_uniform_color([1, 0, 0])  # RGB: Red
#     vis_points.append(action_pcd)

# for point in phantom_points:
#     action_pcd = o3d.geometry.PointCloud()
#     action_pcd.points = o3d.utility.Vector3dVector(point.copy())
#     action_pcd.paint_uniform_color([0, 0, 1])  # RGB: Blue
#     vis_points.append(action_pcd)

# anchor_pcd = o3d.geometry.PointCloud()
# anchor_pcd.points = o3d.utility.Vector3dVector(scene_pcd)
# anchor_pcd.paint_uniform_color([0.5, 0.5, 0.5])
# vis_points.append(anchor_pcd)

# frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
# vis_points.append(frame)

# o3d.visualization.draw_geometries(vis_points)
# breakpoint()

# action_pcd = o3d.geometry.PointCloud()
# action_pcd.points = o3d.utility.Vector3dVector(item["action_pcd"].copy())
# anchor_pcd = o3d.geometry.PointCloud()
# anchor_pcd.points = o3d.utility.Vector3dVector(item["anchor_pcd"].copy())

# gripper_points = item["action_pcd"][item["gripper_idx"]]
# gripper_pos, gripper_rot = get_gripper_pose(gripper_points)
# frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

# o3d.visualization.draw_geometries([action_pcd, anchor_pcd, frame])


