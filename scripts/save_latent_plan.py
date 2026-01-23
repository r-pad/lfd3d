import argparse
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import wandb
import torch
import numpy as np
from torchvision import transforms
from lfd3d.models.dino_3dgp import Dino3DGPNetwork
from lfd3d.models.mimicplay import monkey_patch_mimicplay
from hydra.core.hydra_config import HydraConfig
from lfd3d.utils.script_utils import (
    load_checkpoint_config_from_wandb,
)
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf
import PIL
from PIL import Image
from tqdm import tqdm
from pytorch3d.transforms import matrix_to_rotation_6d

TARGET_SHAPE = 224
rgb_preprocess = transforms.Compose(
    [
        transforms.Resize(
            TARGET_SHAPE,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(TARGET_SHAPE),
    ]
)
depth_preprocess = transforms.Compose(
    [
        transforms.Resize(
            TARGET_SHAPE,
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.CenterCrop(TARGET_SHAPE),
    ]
)

GRIPPER_IDX = {
            "aloha": np.array([6, 197, 174]),
            "human": np.array([343, 763, 60]),
            "libero_franka": np.array(
                [1, 2, 0]
            ),  # gripper pcd in dataset: [left right top grasp-center] in agentview; (right gripper, left gripper, top, grasp-center)
        }

def _gripper_pcd_to_token(gripper_pcd):
        """
        Convert gripper point cloud (3 points) to gripper token (10-dim).
        Token format: [3 position, 6 rotation (6d), 1 gripper width]

        Args:
            gripper_pcd: (3, 3) numpy array with gripper points

        Returns:
            gripper_token: (10,) numpy array
        """
        # Gripper position (center of first two points - fingertips)
        gripper_pos = (gripper_pcd[0, :] + gripper_pcd[1, :]) / 2

        # Gripper width (distance between fingertips)
        gripper_width = np.linalg.norm(gripper_pcd[0, :] - gripper_pcd[1, :])

        # Gripper orientation from the three points
        # Use palm->center as primary axis
        forward = gripper_pos - gripper_pcd[2, :]
        x_axis = forward / np.linalg.norm(forward)

        # Use finger direction for secondary axis
        finger_vec = gripper_pos - gripper_pcd[0, :]

        # Project finger vector onto plane perpendicular to forward
        finger_projected = finger_vec - np.dot(finger_vec, x_axis) * x_axis
        y_axis = finger_projected / np.linalg.norm(finger_projected)

        # Z completes the frame
        z_axis = np.cross(x_axis, y_axis)

        # Create rotation matrix
        rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=-1)

        # Convert to 6D rotation representation
        rotation_matrix_torch = torch.from_numpy(rotation_matrix).float()
        rotation_6d = matrix_to_rotation_6d(rotation_matrix_torch).numpy()

        # Combine into token
        gripper_token = np.concatenate([gripper_pos, rotation_6d, [gripper_width]])

        return gripper_token

def _process_frame_data(original_frame, source_dataset, model, source_meta, max_depth):
    frame_data = {}

    # Define fields that LeRobot manages automatically
    AUTO_FIELDS = {"episode_index", "frame_index", "index", "task_index", "timestamp"}
    camera_names = []
    # Copy existing data
    for key in source_dataset.features.keys():
        if key not in AUTO_FIELDS:
            frame_data[key] = original_frame[key]
        if key.startswith("observation.images.") and key.endswith(".color"):
            camera_names.append(key.split(".")[2])  # Extract camera name

    # Generate latent plan using the model
    frame_data["task"] = source_meta.tasks[original_frame['task_index'].item()]
    frame_data["next_event_idx"] = frame_data["next_event_idx"].numpy().astype(np.int32).reshape(-1,)

    rgb_data, depth_data = [], []
    intrinsics_list, extrinsics_list = [], []
    for cam_name in camera_names:
        intrinsics_key = f"observation.{cam_name}.intrinsics"
        extrinsics_key = f"observation.{cam_name}.extrinsics"

        rgb = (frame_data[f"observation.images.{cam_name}.color"].permute(1,2,0) * 255).detach().cpu().numpy().astype(np.uint8)
        depth = (frame_data[f"observation.images.{cam_name}.transformed_depth"].permute(1,2,0) * 1000).detach().cpu().numpy().astype(np.uint16)
        rgb_data.append(rgb)
        depth_data.append(depth)
        intrinsics_list.append(frame_data[intrinsics_key].detach().cpu().numpy())
        extrinsics_list.append(frame_data[extrinsics_key].detach().cpu().numpy())

        frame_data[f"observation.images.{cam_name}.color"] = rgb
        frame_data[f"observation.images.{cam_name}.transformed_depth"] = depth

    gripper_token = _gripper_pcd_to_token(frame_data["observation.points.gripper_pcds"][GRIPPER_IDX[frame_data["embodiment"]]])

    latent_plan, _ = inference_mimicplay(
        model=model,
        rgbs=rgb_data,
        depths=depth_data,
        intrinsics_list=intrinsics_list,
        extrinsics_list=extrinsics_list,
        gripper_token=gripper_token,
        text=frame_data["task"],
        robot_type=frame_data["embodiment"],
        max_depth=max_depth,
        device=next(model.parameters()).device,
    )

    frame_data["latent_plan"] = latent_plan.squeeze(0).cpu().numpy()  # (896,)

    return frame_data

def upgrade_dataset(
    model,
    source_repo_id: str,
    target_repo_id: str, 
    max_depth: float,
    latent_dim: int = 896,
):
    tolerance_s = 0.0004
    source_dataset = LeRobotDataset(source_repo_id, tolerance_s=tolerance_s)
    source_meta = LeRobotDatasetMetadata(source_repo_id)

    expanded_features = dict(source_dataset.features)
    expanded_features['latent_plan'] = {
        'dtype': 'float32',
        'shape': (latent_dim,),
        'names': ['latent_plan'],
        'info': 'latent plan from mimicplay model'
    }

    target_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=source_dataset.fps,
        features=expanded_features,
    )

    for episode_idx in range(source_meta.info["total_episodes"]):
        print(f"Processing episode {episode_idx + 1}/{source_meta.info['total_episodes']}")

        # Get episode bounds
        episode_start = source_dataset.episode_data_index["from"][episode_idx].item()
        episode_end = source_dataset.episode_data_index["to"][episode_idx].item()
        episode_length = episode_end - episode_start

        # Process each frame in the episode
        for frame_idx in tqdm(range(episode_length)):
            original_frame = source_dataset[episode_start + frame_idx]

            frame_data = _process_frame_data(
                original_frame, source_dataset, model, source_meta, max_depth,
            )

            target_dataset.add_frame(frame_data)

        # Save episode
        target_dataset.save_episode()

    print(f"Upgrade complete! New dataset saved to: {target_dataset.root}")
    return target_dataset

def inference_mimicplay(model, rgbs, depths, intrinsics_list, extrinsics_list,
                        gripper_token, text, robot_type, max_depth, device):
    """
    Run Mimicplay model inference on RGB+depth and predict latent plan.

    Args:
        model (Dino3DGPNetwork): Trained model.
        rgbs (list): List of RGB images [(H, W, 3), ...], uint8, one per camera.
        depths (list): List of depth images [(H, W), ...], uint16 in mm, one per camera.
        intrinsics_list (list): List of camera intrinsics [(3, 3), ...], scaled to 224x224.
        extrinsics_list (list): List of camera extrinsics [(4, 4), ...], T_world_from_camera.
        gripper_token (np.ndarray): Optional gripper token (10,).
        text (str): Optional text caption
        robot_type (str): Robot type (e.g., "aloha", "robot").
        max_depth (float): Maximum depth threshold in meters.
        device (torch.device): Device for inference.

    Returns:
        latent_plan (torch.Tensor): Predicted latent plan (B, latent_dim).
        gmm_dist : GMM distribution over future waypoints
    """
    N = len(rgbs)  # Number of cameras

    # Preprocess all RGBs
    rgbs_processed = []
    for rgb in rgbs:
        rgb_ = np.asarray(rgb_preprocess(Image.fromarray(rgb))).copy()
        rgb_ = torch.from_numpy(rgb_).permute(2, 0, 1).float()  # (3, 224, 224)
        rgbs_processed.append(rgb_)

    # Stack into (1, N, 3, 224, 224)
    rgbs_tensor = torch.stack(rgbs_processed, dim=0).unsqueeze(0).to(device)

    # Preprocess all depths
    depths_processed = []
    for depth in depths:
        depth_ = (depth / 1000.0).squeeze().astype(np.float32)  # Convert mm to meters
        depth_ = PIL.Image.fromarray(depth_)
        depth_ = np.asarray(depth_preprocess(depth_)).copy()
        depth_[depth_ > max_depth] = 0  # Mask out far depths
        depth_ = torch.from_numpy(depth_).float()  # (224, 224)
        depths_processed.append(depth_)

    # Stack into (1, N, 224, 224)
    depths_tensor = torch.stack(depths_processed, dim=0).unsqueeze(0).to(device)

    # Stack intrinsics into (1, N, 3, 3)
    intrinsics_tensor = torch.stack([
        torch.from_numpy(K.astype(np.float32)) for K in intrinsics_list
    ], dim=0).unsqueeze(0).to(device)

    # Stack extrinsics into (1, N, 4, 4)
    extrinsics_tensor = torch.stack([
        torch.from_numpy(T.astype(np.float32)) for T in extrinsics_list
    ], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        # Convert gripper_token to torch if provided
        gripper_tok = None
        if gripper_token is not None:
            gripper_tok = torch.from_numpy(gripper_token.astype(np.float32)).unsqueeze(0).to(device)  # (1, 10)

        # Determine source
        source = [robot_type] if model.use_source_token else None

        # Forward through network
        latent_plan, gmm_dist = model.mimicplay_forward(
            image=rgbs_tensor,
            depth=depths_tensor,
            intrinsics=intrinsics_tensor,
            extrinsics=extrinsics_tensor,
            gripper_token=gripper_tok,
            text=text,
            source=source
        )
        
        return latent_plan, gmm_dist

def initialize_mimicplay_model(entity, project, checkpoint_type,
    run_id, dino_model, use_text_embedding, use_gripper_token, use_source_token,
    use_fourier_pe, fourier_num_frequencies, fourier_include_input,
    num_transformer_layers, dropout, device
):
    """Initialize Mimicplay model from wandb artifact"""

    # Simple config object to match what Dino3DGPNetwork expects
    class ModelConfig:
        def __init__(self, dino_model, use_text_embedding, use_gripper_token,
                     use_source_token, use_fourier_pe, fourier_num_frequencies,
                     fourier_include_input, num_transformer_layers, dropout):
            self.dino_model = dino_model
            self.use_text_embedding = use_text_embedding
            self.use_gripper_token = use_gripper_token
            self.use_source_token = use_source_token
            self.use_fourier_pe = use_fourier_pe
            self.fourier_num_frequencies = fourier_num_frequencies
            self.fourier_include_input = fourier_include_input
            self.num_transformer_layers = num_transformer_layers
            self.dropout = dropout
            self.image_token_dropout = False # We only do inference here.

    model_cfg = ModelConfig(
        dino_model, use_text_embedding, use_gripper_token,
        use_source_token, use_fourier_pe, fourier_num_frequencies,
        fourier_include_input, num_transformer_layers, dropout
    )
    model = monkey_patch_mimicplay(Dino3DGPNetwork(model_cfg))

    artifact_dir = "wandb"
    checkpoint_reference = f"{entity}/{project}/best_{checkpoint_type}_model-{run_id}:best"
    api = wandb.Api()
    artifact = api.artifact(checkpoint_reference, type="model")
    ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
    ckpt = torch.load(ckpt_file)
    # Remove the "network." prefix, since we're not using Lightning here.
    state_dict = {k.replace("network.",""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)

    model = model.eval()
    model = model.to(device)

    return model

def main():
    parser = argparse.ArgumentParser(description='Save Latent plan MIMICPLAY')
    parser.add_argument('--source_repo_id', type=str, required=True, help='Directory containing MIMICPLAY data')
    parser.add_argument('--target_repo_id', type=str, required=True, help='Directory to save processed latent plans')
    parser.add_argument("--hl_run_id", type=str, required=True, help="Checkpoint ID to use for latent plan generation")
    parser.add_argument('--hl_entity', type=str, default="r-pad")
    parser.add_argument('--hl_project', type=str, default="lfd3d")
    args = parser.parse_args()

    overrides = ["dataset=rpadLerobot", "model=mimicplay", f"dataset.repo_id={args.source_repo_id}"]  # or build from args

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=overrides)

    task_overrides = overrides
    cfg = load_checkpoint_config_from_wandb(
        cfg, task_overrides, args.hl_entity, args.hl_project, args.hl_run_id
    )

    max_depth = cfg.dataset.max_depth

    model = initialize_mimicplay_model(
        entity=args.hl_entity,
        project=args.hl_project,
        checkpoint_type=cfg.checkpoint.type,
        run_id=args.hl_run_id,
        dino_model=cfg.model.dino_model,
        use_text_embedding=cfg.model.use_text_embedding,
        use_gripper_token=cfg.model.use_gripper_token,
        use_source_token=cfg.model.use_source_token,
        use_fourier_pe=cfg.model.use_fourier_pe,
        fourier_num_frequencies=cfg.model.fourier_num_frequencies,
        fourier_include_input=cfg.model.fourier_include_input,
        num_transformer_layers=cfg.model.num_transformer_layers,
        dropout=cfg.model.dropout,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    upgrade_dataset(
    model,
    args.source_repo_id,
    args.target_repo_id,
    max_depth,
    )


if __name__ == "__main__":
    main()