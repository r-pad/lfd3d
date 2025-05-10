#!/usr/bin/env python3
import argparse
import json
import numpy as np
import roslibpy
from torchvision import transforms
import PIL
import torch
from pytorch3d.ops import sample_farthest_points
import wandb
from lfd3d.models.articubot import PointNet2_super
import threading
import time
import base64

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

def get_weighted_displacement(outputs):
    """
    Extract weighted displacement from network output
    """
    weights = outputs[:, :, -1]  # B, N
    outputs = outputs[:, :, :-1]  # B, N, 12

    # softmax the weights
    weights = torch.nn.functional.softmax(weights, dim=1)

    # sum the displacement of the predicted gripper point cloud according to the weights
    outputs = outputs * weights.unsqueeze(-1)
    outputs = outputs.sum(dim=1)
    return outputs.reshape(-1, 4, 3)

class HighLevelPolicy:
    def __init__(
        self,
        run_id,
        rosbridge_host="localhost",
        rosbridge_port=9090,
        max_depth=1.0,
        num_points=8192,
        in_channels=3,
        use_gripper_pcd=False,
    ):
        self.device = "cuda"
        self.max_depth = max_depth
        self.num_points = num_points
        self.use_gripper_pcd = use_gripper_pcd
        self.rng = np.random.default_rng()

        # Initialize model
        self.model = initialize_model(run_id, in_channels, self.device)

        # Initialize roslibpy client
        self.client = roslibpy.Ros(host=rosbridge_host, port=rosbridge_port)
        self.client.run()

        # Initialize subscribers
        self.rgb_sub = roslibpy.Topic(
            self.client, "/rgb/image_rect", "sensor_msgs/Image"
        )
        self.depth_sub = roslibpy.Topic(
            self.client, "/depth_registered/image_rect", "sensor_msgs/Image"
        )
        self.camera_info_sub = roslibpy.Topic(
            self.client, "/rgb/camera_info", "sensor_msgs/CameraInfo"
        )
        if use_gripper_pcd:
            self.gripper_pcd_sub = roslibpy.Topic(
                self.client, "/follower_right/gripper_pcd_dense", "sensor_msgs/PointCloud2"
            )

        # Initialize publisher
        self.goal_pub = roslibpy.Topic(
            self.client, "/goal_prediction", "sensor_msgs/PointCloud2"
        )

        # Data storage
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_gripper_pcd = None
        self.camera_info = None
        self.K = None

        # Subscribe to topics
        self.rgb_sub.subscribe(self.rgb_callback)
        self.depth_sub.subscribe(self.depth_callback)
        self.camera_info_sub.subscribe(self.camera_info_callback)
        if use_gripper_pcd:
            self.gripper_pcd_sub.subscribe(self.gripper_pcd_callback)

        print("HighLevelPolicy initialized.")

    def run_loop(self, interval=1.0):
        """Main loop that runs synchronously every `interval` seconds."""
        try:
            while True:
                self.timer_callback()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Interrupted by user. Shutting down.")
            self.shutdown()

    def rgb_callback(self, msg):
        self.latest_rgb = msg

    def depth_callback(self, msg):
        self.latest_depth = msg

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def gripper_pcd_callback(self, msg):
        self.latest_gripper_pcd = msg

    def timer_callback(self):
        if all([self.latest_rgb, self.latest_depth, self.camera_info]):
            if self.use_gripper_pcd and self.latest_gripper_pcd is None:
                return

            print(
                f"Processing RGB: {self.latest_rgb['width']}x{self.latest_rgb['height']}, "
                f"Timestamp: {self.latest_rgb['header']['stamp']['sec']}.{self.latest_rgb['header']['stamp']['nanosec']}"
            )
            print(
                f"Processing Depth: {self.latest_depth['width']}x{self.latest_depth['height']}, "
                f"Timestamp: {self.latest_depth['header']['stamp']['sec']}.{self.latest_depth['header']['stamp']['nanosec']}"
            )

            # Extract data from messages
            rgb, depth = self.extract_images_from_messages(self.latest_rgb, self.latest_depth)

            if self.K is None:
                K = np.asarray(self.camera_info["k"]).reshape(3, 3)
                self.K = get_scaled_intrinsics(
                    K, (self.latest_rgb["height"], self.latest_rgb["width"]), TARGET_SHAPE
                )

            pcd = compute_pcd(
                rgb,
                depth,
                self.K,
                rgb_preprocess,
                depth_preprocess,
                self.rng,
                self.num_points,
                self.max_depth,
            )
            pcd_xyz = pcd[:, :3]
            if self.use_gripper_pcd:
                gripper_pcd = self.extract_gripper_pcd(self.latest_gripper_pcd)
                pcd_xyz = concat_gripper_pcd(gripper_pcd, pcd_xyz)

            # Run inference
            goal_prediction = inference(self.model, pcd_xyz, self.device)

            # Publish goal prediction
            self.publish_msg(goal_prediction, header=self.latest_depth["header"])

        else:
            print("Waiting for images...")

    def extract_images_from_messages(self, rgb_msg, depth_msg):
        """
        Extracts and processes RGB and depth arrays from roslibpy messages.

        Args:
            rgb_msg (dict): roslibpy message for RGB image.
            depth_msg (dict): roslibpy message for depth image.

        Returns:
            tuple: (rgb_array, depth_array) as NumPy arrays.
        """
        assert rgb_msg["encoding"] == "bgra8"
        assert depth_msg["encoding"] == "16UC1"

        rgb_dtype, rgb_ch = np.uint8, 4
        depth_dtype, depth_ch = np.uint16, 1

        rgb_data = base64.b64decode(rgb_msg["data"])
        depth_data = base64.b64decode(depth_msg["data"])
        rgb = np.frombuffer(rgb_data, dtype=rgb_dtype).reshape(
            rgb_msg["height"], rgb_msg["width"], rgb_ch
        )
        rgb = rgb[:, :, :3][:, :, ::-1]  # Convert BGRA to RGB

        depth = np.frombuffer(depth_data, dtype=depth_dtype).reshape(
            depth_msg["height"], depth_msg["width"], depth_ch
        )
        return rgb, depth

    def extract_gripper_pcd(self, pcd_msg):
        """
        Extracts gripper point cloud from roslibpy PointCloud2 message.

        Args:
            pcd_msg (dict): roslibpy message for PointCloud2.

        Returns:
            np.ndarray: Gripper points (M, 3).
        """
        points = []
        point_step = pcd_msg["point_step"]
        data = base64.b64decode(pcd_msg["data"])
        for i in range(0, len(data), point_step):
            x = np.frombuffer(data[i : i + 4], dtype=np.float32)[0]
            y = np.frombuffer(data[i + 4 : i + 8], dtype=np.float32)[0]
            z = np.frombuffer(data[i + 8 : i + 12], dtype=np.float32)[0]
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                points.append([x, y, z])
        return np.array(points, dtype=np.float32)

    def publish_msg(self, goal_prediction, header):
        """
        Publishes goal prediction as a PointCloud2 message.

        Args:
            goal_prediction (np.ndarray): Predicted goal points (N, 3).
            header (dict): Header from depth message.
        """
        points = goal_prediction.astype(np.float32).tolist()
        fields = [
            {"name": "x", "offset": 0, "datatype": 7, "count": 1},
            {"name": "y", "offset": 4, "datatype": 7, "count": 1},
            {"name": "z", "offset": 8, "datatype": 7, "count": 1},
        ]
        point_step = 12
        data = []
        for pt in points:
            for coord in pt:
                data.extend(np.float32(coord).tobytes())

        msg = {
            "header": {
                "stamp": header["stamp"],
                "frame_id": "rgb_camera_link",
            },
            "height": 1,
            "width": len(points),
            "fields": fields,
            "is_bigendian": False,
            "point_step": point_step,
            "row_step": point_step * len(points),
            "data": data,
            "is_dense": True,
        }
        self.goal_pub.publish(msg)
        print(
            f"Published goal prediction "
            f"Timestamp: {msg['header']['stamp']['sec']}.{msg['header']['stamp']['nanosec']}"
        )

    def shutdown(self):
        """Closes the roslibpy client and unsubscribes from topics."""
        self.rgb_sub.unsubscribe()
        self.depth_sub.unsubscribe()
        self.camera_info_sub.unsubscribe()
        if self.use_gripper_pcd:
            self.gripper_pcd_sub.unsubscribe()
        self.goal_pub.unadvertise()
        self.client.terminate()
        
def initialize_model(run_id, in_channels, device):
    artifact_dir = "wandb"
    checkpoint_reference = f"r-pad/lfd3d/best_rmse_model-{run_id}:best"
    api = wandb.Api()
    artifact = api.artifact(checkpoint_reference, type="model")
    ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
    ckpt = torch.load(ckpt_file)
    state_dict = {k.replace("network.", ""): v for k, v in ckpt["state_dict"].items()}

    model = PointNet2_super(num_classes=13, input_channel=in_channels)
    model.load_state_dict(state_dict)
    model = model.eval().to(device)
    return model

def inference(model, pcd_xyz, device):
    with torch.no_grad():
        if len(pcd_xyz.shape) == 2:
            pcd_xyz = pcd_xyz.transpose(1, 0)[None]  # [1, 3, N]
        elif len(pcd_xyz.shape) == 3:
            pcd_xyz = pcd_xyz.transpose(0, 2, 1)
        pcd_xyz = torch.from_numpy(pcd_xyz.astype(np.float32)).to(device)
        outputs = model(pcd_xyz)  # [1, N, 13]
        goal_prediction = get_weighted_displacement(outputs).squeeze().cpu().numpy()  # [4, 3]
        return goal_prediction

def compute_pcd(rgb, depth, K, rgb_preprocess, depth_preprocess, rng, num_points, max_depth):
    rgb_ = PIL.Image.fromarray(rgb)
    rgb_ = np.asarray(rgb_preprocess(rgb_))

    depth_ = (depth / 1000.0).squeeze().astype(np.float32)
    depth_ = PIL.Image.fromarray(depth_)
    depth_ = np.asarray(depth_preprocess(depth_))

    height, width = depth_.shape
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = depth_.flatten()
    rgb_flat = rgb_.reshape(-1, 3)

    valid_depth = np.logical_and(z_flat > 0, z_flat < max_depth)
    x_flat = x_flat[valid_depth]
    y_flat = y_flat[valid_depth]
    z_flat = z_flat[valid_depth]
    rgb_flat = rgb_flat[valid_depth]

    pixels = np.stack([x_flat, y_flat, np.ones_like(x_flat)], axis=0)
    K_inv = np.linalg.inv(K)
    points = K_inv @ pixels
    points = points * z_flat
    points = points.T

    scene_pcd_pt3d = torch.from_numpy(points)
    scene_pcd_downsample, scene_points_idx = sample_farthest_points(
        scene_pcd_pt3d[None], K=num_points, random_start_point=False
    )
    scene_pcd = scene_pcd_downsample.squeeze().numpy()
    scene_rgb_pcd = rgb_flat[scene_points_idx.squeeze().numpy()]
    pcd = np.concatenate([scene_pcd, scene_rgb_pcd], axis=1)
    return pcd

def concat_gripper_pcd(gripper_pcd, pcd_xyz):
    gripper_pcd = np.concatenate(
        [gripper_pcd, np.ones((gripper_pcd.shape[0], 1))], axis=1
    )
    pcd_xyz = np.concatenate(
        [pcd_xyz, np.zeros((pcd_xyz.shape[0], 1))], axis=1
    )
    pcd_xyz = np.concatenate([gripper_pcd, pcd_xyz], axis=0)
    return pcd_xyz

def get_scaled_intrinsics(K, orig_shape, target_shape):
    K_ = K.copy()
    scale_factor = target_shape / min(orig_shape)
    K_[0, 0] *= scale_factor
    K_[1, 1] *= scale_factor
    K_[0, 2] *= scale_factor
    K_[1, 2] *= scale_factor
    crop_offset_x = (orig_shape[1] * scale_factor - target_shape) / 2
    crop_offset_y = (orig_shape[0] * scale_factor - target_shape) / 2
    K_[0, 2] -= crop_offset_x
    K_[1, 2] -= crop_offset_y
    return K_

def main():
    parser = argparse.ArgumentParser(description="HighLevelPolicy parameters")
    parser.add_argument("--run_id", type=str, required=True, help="WandB run ID (e.g., abc123)")
    parser.add_argument("--rosbridge_host", type=str, default="localhost", help="Rosbridge WebSocket host")
    parser.add_argument("--rosbridge_port", type=int, default=9090, help="Rosbridge WebSocket port")
    parser.add_argument("--max_depth", type=float, default=1.0, help="Maximum depth value")
    parser.add_argument("--num_points", type=int, default=8192, help="Number of points for point cloud sampling")
    parser.add_argument("--in_channels", type=int, default=3, help="Input channels -> 4 if use_gripper_pcd")
    parser.add_argument("--use_gripper_pcd", action="store_true", help="Use gripper point cloud")

    args = parser.parse_args()

    policy = HighLevelPolicy(
        run_id=args.run_id,
        rosbridge_host=args.rosbridge_host,
        rosbridge_port=args.rosbridge_port,
        max_depth=args.max_depth,
        num_points=args.num_points,
        in_channels=args.in_channels,
        use_gripper_pcd=args.use_gripper_pcd,
    )

    policy.run_loop()

if __name__ == "__main__":
    main()