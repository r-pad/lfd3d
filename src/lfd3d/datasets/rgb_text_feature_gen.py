"""
Extracts and processes image and text features using DINOv2 and SigLIP models.

Usage:
Run the script with command-line arguments specifying the dataset and input directory.

Example:
python rgb_text_feature_gen.py --dataset hoi4d --input_dir /path/to/hoi4d
"""
import argparse
import json
import os
import random
from glob import glob

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from moviepy.editor import VideoFileClip
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def get_dinov2_image_embedding(image, dinov2=None, device="cuda"):
    if dinov2 is None:
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").to(
            device
        )
    patch_size = 14
    target_shape = 224

    assert type(image) == Image.Image
    preprocess = transforms.Compose(
        [
            transforms.Resize(
                target_shape, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(target_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    inputs = preprocess(image).unsqueeze(0).to(device)

    # Forward pass to get features
    with torch.no_grad():
        outputs = dinov2.forward_features(inputs)

    # Extract the last hidden state as features
    patch_features = outputs["x_norm_patchtokens"].squeeze(0)
    num_patches = patch_features.shape[0]
    h = w = int(num_patches**0.5)
    patch_features_2d = patch_features.reshape(h, w, -1)

    # Permute to [C, H, W] for interpolation
    patch_features_2d = patch_features_2d.permute(2, 0, 1)

    # Upsample to match original image patch dimensions
    resized_features = F.interpolate(
        patch_features_2d.unsqueeze(0),
        size=(target_shape, target_shape),
        mode="bilinear",
        align_corners=False,
    )

    return resized_features.squeeze().permute(1, 2, 0).cpu().numpy()


def get_dinov2_image_embedding_from_file(image_path, dinov2=None):
    if dinov2 is None:
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").cuda()
    patch_size = 14
    target_shape = 224

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize(
                target_shape, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(target_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    inputs = preprocess(image).unsqueeze(0).to("cuda")

    # Forward pass to get features
    with torch.no_grad():
        outputs = dinov2.forward_features(inputs)

    # Extract the last hidden state as features
    patch_features = outputs["x_norm_patchtokens"].squeeze(0)
    num_patches = patch_features.shape[0]
    h = w = int(num_patches**0.5)
    patch_features_2d = patch_features.reshape(h, w, -1)

    # Permute to [C, H, W] for interpolation
    patch_features_2d = patch_features_2d.permute(2, 0, 1)

    # Upsample to match original image patch dimensions
    resized_features = F.interpolate(
        patch_features_2d.unsqueeze(0),
        size=(target_shape, target_shape),
        mode="bilinear",
        align_corners=False,
    )

    return resized_features.squeeze().permute(1, 2, 0).cpu().numpy()


def get_siglip_text_embedding(
    caption, siglip=None, siglip_processor=None, device="cuda"
):
    if siglip is None or siglip_processor is None:
        siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(
            device
        )
        siglip_processor = AutoProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384"
        )

    # Process text input
    inputs = siglip_processor(text=[caption], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate embeddings
    with torch.no_grad():
        text_embedding = siglip.get_text_features(**inputs)

    return text_embedding.cpu().squeeze().numpy()


def get_hoi4d_items(hoi4d_root_dir):
    with open("hoi4d/hoi4d_videos.json") as f:
        hoi4d_videos = json.load(f)
    hoi4d_videos = [f"{hoi4d_root_dir}/{i}" for i in hoi4d_videos]

    hoi4d_items = []
    for vid in tqdm(hoi4d_videos):
        dir_name = os.path.dirname(os.path.dirname(vid))
        action_annotation = json.load(open(f"{dir_name}/action/color.json"))
        if "events" in action_annotation:
            all_events = action_annotation["events"]
        else:
            all_events = action_annotation["markResult"]["marks"]

        objpose_fname = f"{dir_name}/objpose/0.json"
        if not os.path.exists(objpose_fname):
            objpose_fname = f"{dir_name}/objpose/00000.json"
        obj_name = json.load(open(objpose_fname))["dataList"][0]["label"]

        for event_idx, event in enumerate(all_events):
            save_name = f"{dir_name}/rgb_text_features/{event_idx}_compressed.npz"
            event_name = event["event"]
            caption = f"{event_name} {obj_name}"

            # 300 frames per video, 30 or 15 fps depending on length of video
            try:
                fps = 300 / action_annotation["info"]["duration"]
            except KeyError:
                fps = 300 / action_annotation["info"]["Duration"]

            # Convert timestamp in seconds to frame_idx
            try:
                event_start_idx = int(event["startTime"] * fps)
            except KeyError:
                event_start_idx = int(event["hdTimeStart"] * fps)

            img_path = f"{dir_name}/align_rgb/{str(event_start_idx).zfill(5)}.jpg"

            item = {
                "dir_name": dir_name,
                "save_name": save_name,
                "caption": caption,
                "img_path": img_path,
            }
            hoi4d_items.append(item)
    return hoi4d_items


def get_droid_items(droid_root_dir):
    droid_videos = glob(f"{droid_root_dir}/droid_gemini_events/*")
    base_save_dir = f"{droid_root_dir}/droid_rgb_text_features"

    droid_items = []
    pbar = tqdm(droid_videos)
    for vid in pbar:
        pbar.set_description(vid)
        dir_name = os.path.basename(vid)
        with open(f"{vid}/subgoal.json") as f:
            subgoals = json.load(f)

        if subgoals == []:
            print("No subgoals for: ", vid)
            continue

        video_clip = VideoFileClip(f"{vid}/video.mp4")
        duration = video_clip.duration

        timestamps = [i["timestamp"] for i in subgoals]
        timestamps_sec = [
            int(t[3:]) for t in timestamps
        ]  # Convert MM:SS to int (discard minutes)

        last_timestamp = timestamps_sec[-1]
        if last_timestamp >= duration - 0.1:  # small buffer
            print("Hallucinated timestamps for:", vid)
            continue

        last_frame = video_clip.get_frame(last_timestamp)
        # We need the first frame, but we don't need features for the last
        # frame where the final subgoal is completed.
        # Input to the model is when the goal *starts* and the json
        # describes when the goal *ends*.
        timestamps_sec = [0] + timestamps_sec[:-1]

        # save image for last frame separately, we only need the image, not features
        img_path = f"{vid}/{len(subgoals)}_{last_timestamp}.png"
        Image.fromarray(last_frame).save(img_path)

        for i, subgoal in enumerate(subgoals):
            save_name = f"{base_save_dir}/{dir_name}/{i}_compressed.npz"
            img_path = f"{vid}/{i}_{timestamps_sec[i]}.png"
            caption = subgoal["subgoal"]

            # Save image for DINOv2 embeddings
            image = video_clip.get_frame(timestamps_sec[i])
            Image.fromarray(image).save(img_path)

            item = {
                "dir_name": dir_name,
                "save_name": save_name,
                "caption": caption,
                "img_path": img_path,
            }
            droid_items.append(item)
        video_clip.close()
    return droid_items


def compress_features(pca_model, features, pca_n_components, downscale_by):
    """
    Compress features with PCA.
    """
    height, width, feat_dim = features.shape

    features_proc = features.transpose(2, 0, 1)[None]
    downsample_features = F.interpolate(
        torch.from_numpy(features_proc),
        scale_factor=downscale_by,
        mode="bilinear",
        align_corners=False,
    )
    _, _, down_height, down_width = downsample_features.shape

    downsample_features = (
        downsample_features.numpy().transpose(0, 2, 3, 1).reshape(-1, feat_dim)
    )
    compressed_features = pca_model.transform(downsample_features)
    compressed_features = compressed_features.reshape(
        down_height, down_width, pca_n_components
    )
    return compressed_features


def save_pca_model(dataset_items, pca_model_path, dino_feat_dim, pca_n_components):
    print("Fitting PCA model for features")
    # 10 random data points to fit PCA
    data_sample = random.sample(dataset_items, 10)
    pca_fit_features = []
    for item in data_sample:
        img_embedding = get_dinov2_image_embedding_from_file(item["img_path"], dinov2)
        pca_fit_features.append(img_embedding)

    pca_model = PCA(n_components=pca_n_components)
    features = np.array(pca_fit_features)
    num_samples, height, width, feat_dim = features.shape
    features = features.reshape(-1, feat_dim)
    pca_model.fit(features)
    joblib.dump(pca_model, pca_model_path)
    print(f"Saved PCA model at {pca_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 image and SigLIP text features from a dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["hoi4d", "droid"],
        help="Dataset to process",
    )
    parser.add_argument("--input_dir", type=str, help="input dir")
    args = parser.parse_args()

    # Load models
    torch.autograd.set_grad_enabled(False)

    siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to("cuda")
    siglip_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").to("cuda")
    dino_feat_dim = 1024
    pca_n_components = 256  # number of components for PCA
    downscale_by = 1 / 4  # downsample factor for feature image before saving

    if args.dataset == "hoi4d":
        pca_model_path = "hoi4d/pca_model.pkl"
        dataset_items = get_hoi4d_items(args.input_dir)
        if not os.path.exists(pca_model_path):
            save_pca_model(
                dataset_items, pca_model_path, dino_feat_dim, pca_n_components
            )
        pca_model = joblib.load(pca_model_path)
    elif args.dataset == "droid":
        pca_model_path = "droid/pca_model.pkl"
        dataset_items = get_droid_items(args.input_dir)
        if not os.path.exists(pca_model_path):
            save_pca_model(
                dataset_items, pca_model_path, dino_feat_dim, pca_n_components
            )
        pca_model = joblib.load(pca_model_path)
    else:
        raise NotImplementedError

    pbar = tqdm(dataset_items)
    for item in pbar:
        pbar.set_description(item["dir_name"])
        if args.dataset == "hoi4d":
            os.makedirs(f"{item['dir_name']}/rgb_text_features", exist_ok=True)
        elif args.dataset == "droid":
            os.makedirs(os.path.dirname(item["save_name"]), exist_ok=True)
        else:
            raise NotImplementedError()

        if os.path.exists(item["save_name"]):
            continue

        # Extract features
        img_embedding = get_dinov2_image_embedding_from_file(item["img_path"], dinov2)
        img_embedding_compressed = compress_features(
            pca_model, img_embedding, pca_n_components, downscale_by
        ).astype(np.float32)
        text_embedding = get_siglip_text_embedding(
            item["caption"], siglip, siglip_processor
        )

        np.savez_compressed(
            item["save_name"],
            rgb_embed=img_embedding_compressed.astype(np.float16),
            text_embed=text_embedding.astype(np.float16),
        )
