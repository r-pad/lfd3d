import argparse
import json
import os

import joblib
import numpy as np
import tensorflow_datasets as tfds
import torch
import torch.nn.functional as F
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def get_video_chunk_idxs(gripper_state, caption):
    threshold = 0.5

    if caption == "" or caption.split()[0] in ["open", "close"]:
        return [0]

    # first chunk -> till gripper_state > threshold
    first_chunk_end = np.argmax(gripper_state > threshold)

    # second chunk -> from first chunk, till gripper state < threshold or end of video
    second_chunk_offset = np.argmax(gripper_state[first_chunk_end:] < threshold)
    if second_chunk_offset == 0:
        second_chunk_end = -1
    else:
        second_chunk_end = first_chunk_end + second_chunk_offset

    return [0, first_chunk_end]


def compute_pixel_aligned_embedding(
    img, caption, mask_generator, siglip, siglip_processor, cosine_similarity
):
    """
    Adapted from ConceptFusion
    """
    masks = mask_generator.generate(img)

    # Process inputs
    inputs = siglip_processor(
        text=[caption], images=Image.fromarray(img), return_tensors="pt", padding=True
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    # Generate embeddings
    with torch.no_grad():
        outputs = siglip(**inputs)

    # Extract embeddings
    global_feat = outputs.image_embeds
    text_embeds = outputs.text_embeds

    global_feat = global_feat.half()
    global_feat = torch.nn.functional.normalize(global_feat, dim=-1)  # --> (1, 1024)

    feat_per_roi = []
    roi_nonzero_inds = []
    similarity_scores = []
    for maskidx in range(len(masks)):
        _x, _y, _w, _h = tuple(masks[maskidx]["bbox"])  # xywh bounding box
        seg = masks[maskidx]["segmentation"]
        nonzero_inds = torch.argwhere(torch.from_numpy(masks[maskidx]["segmentation"]))
        # Note: Image is (H, W, 3). In SAM output, y coords are along height, x along width
        img_roi = img[int(_y) : int(_y + _h), int(_x) : int(_x + _w), :]

        if (img_roi.shape[0] * img_roi.shape[1]) < (0.01 * IMG_HEIGHT * IMG_WIDTH):
            continue

        # Process inputs
        inputs = siglip_processor(
            images=Image.fromarray(img_roi), return_tensors="pt", padding=True
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        # Generate embeddings
        with torch.no_grad():
            roifeat = siglip.get_image_features(**inputs).half()

        feat_per_roi.append(roifeat)
        roi_nonzero_inds.append(nonzero_inds)
        _sim = cosine_similarity(global_feat, roifeat)
        similarity_scores.append(_sim)

    similarity_scores = torch.cat(similarity_scores)
    softmax_scores = torch.nn.functional.softmax(similarity_scores, dim=0)
    pix_align_embed = torch.zeros(IMG_HEIGHT, IMG_WIDTH, feat_dim, dtype=torch.half)
    for maskidx in range(len(feat_per_roi)):
        _weighted_feat = (
            softmax_scores[maskidx] * global_feat
            + (1 - softmax_scores[maskidx]) * feat_per_roi[maskidx]
        )
        _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
        pix_align_embed[
            roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]
        ] += (_weighted_feat[0].detach().cpu().half())
        pix_align_embed[
            roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]
        ] = torch.nn.functional.normalize(
            pix_align_embed[
                roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]
            ].float(),
            dim=-1,
        ).half()

    pix_align_embed = torch.nn.functional.normalize(pix_align_embed.float(), dim=-1)
    pix_align_embed = pix_align_embed.numpy()  # --> H, W, feat_dim
    return pix_align_embed, text_embeds.cpu().numpy()


def compress_features(pca_model, features):
    """
    Downsample image and compress features with PCA.
    """
    downscale_by = 1 / 2
    features_proc = features.transpose(2, 0, 1)[None]
    downsample_features = F.interpolate(
        torch.from_numpy(features_proc),
        scale_factor=downscale_by,
        mode="bilinear",
        align_corners=False,
    )
    downsample_features = (
        downsample_features.numpy().transpose(0, 2, 3, 1).reshape(-1, feat_dim)
    )

    compressed_features = pca_model.transform(downsample_features)
    return compressed_features


def measure_compression_error(pca_model, features, compressed_features):
    """
    Invert PCA and upsample to check error
    Around 5e-3
    """
    upscale_by = 2

    reconstructed_features = pca_model.inverse_transform(compressed_features).astype(
        np.float32
    )
    reconstructed_features = reconstructed_features.transpose(1, 0).reshape(
        1, feat_dim, IMG_HEIGHT // upscale_by, IMG_WIDTH // upscale_by
    )
    upsample_features = F.interpolate(
        torch.from_numpy(reconstructed_features),
        scale_factor=upscale_by,
        mode="bilinear",
        align_corners=False,
    )

    reconstruct_upsample_features = (
        upsample_features.numpy().transpose(0, 2, 3, 1).squeeze()
    )
    print(
        "Error:",
        np.abs(reconstruct_upsample_features.astype(np.float16) - features).mean(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate RGB/text features using SIGLIP/ConceptFusion."
    )
    parser.add_argument(
        "--split",
        default=-1,
        type=int,
        help="Integer in 1-10 identifying which tenth of the dataset to process. \
        Basically just a manual way to distribute work on the cluster.",
    )
    args = parser.parse_args()

    IMG_HEIGHT, IMG_WIDTH = 256, 320  # RT1 resolution
    feat_dim = 1152  # Siglip feature dim
    RT1_DIR = "/data/sriram/rt1/fractal20220817_data_0.1.0"
    SAVE_DIR = "/data/sriram/rt1/rt1_rgb_feat"

    # Load models

    torch.autograd.set_grad_enabled(False)

    # wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    sam = sam_model_registry["vit_h"](checkpoint="../sam_vit_h_4b8939.pth")
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=8,
        pred_iou_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )

    siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to("cuda")
    siglip_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    builder = tfds.builder_from_directory(builder_dir=RT1_DIR)
    dataset = builder.as_dataset(split="train")
    dataset_size = len(dataset)

    split_num = args.split
    valid_idxs = range(dataset_size)
    if split_num >= 0 and split_num < 10:
        valid_idxs = valid_idxs[
            split_num * (dataset_size // 10) : (split_num + 1) * (dataset_size // 10)
        ]

    # Prepare PCA
    if not os.path.exists("pca_model.pkl"):
        print("Collecting some features to fit PCA")
        pca_fit_features = []
        # Fit a PCA with first 10 images.
        for idx, item in tqdm(enumerate(dataset), total=10):
            steps = [i for i in item["steps"]]
            img = np.array(steps[0]["observation"]["image"])

            pix_align_embedding, _ = compute_pixel_aligned_embedding(
                img,
                "test caption",
                mask_generator,
                siglip,
                siglip_processor,
                cosine_similarity,
            )
            pca_fit_features.append(pix_align_embedding)

            if idx == 9:
                break

        pca_model = PCA(n_components=256)
        features_proc = np.array(pca_fit_features).transpose(0, 3, 1, 2)
        downsample_features = F.interpolate(
            torch.from_numpy(features_proc),
            scale_factor=1 / 2,
            mode="bilinear",
            align_corners=False,
        )
        downsample_features = (
            downsample_features.numpy().transpose(0, 2, 3, 1).reshape(-1, feat_dim)
        )
        pca_model.fit(downsample_features)

        joblib.dump(pca_model, "pca_model.pkl")

    pca_model = joblib.load("pca_model.pkl")
    with open(f"{RT1_DIR}/../chunked_captions.json", "r") as f:
        chunked_captions = json.load(f)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Process rt1 video
    for idx, item in tqdm(enumerate(dataset), total=dataset_size):
        if idx not in valid_idxs:
            continue

        steps = [i for i in item["steps"]]

        gripper_state = np.array(
            [i["observation"]["gripper_closed"].numpy() for i in steps]
        )
        overall_caption = (
            steps[0]["observation"]["natural_language_instruction"]
            .numpy()
            .decode("utf-8")
        )

        chunk_idxs = get_video_chunk_idxs(gripper_state, overall_caption)

        for i, chunk_idx in enumerate(chunk_idxs):
            save_name = f"{SAVE_DIR}/{idx}_{i}_compressed.npz"

            if os.path.exists(save_name):
                continue

            caption = chunked_captions[idx]["chunked"][i]
            img = np.array(steps[chunk_idx]["observation"]["image"])

            img_embedding, text_embedding = compute_pixel_aligned_embedding(
                img,
                caption,
                mask_generator,
                siglip,
                siglip_processor,
                cosine_similarity,
            )
            img_embedding_compressed = compress_features(
                pca_model, img_embedding
            ).astype(np.float32)
            # measure_compression_error(pca_model, img_embedding, img_embedding_compressed)

            np.savez_compressed(
                save_name,
                rgb_embed=img_embedding_compressed.astype(np.float16),
                text_embed=text_embedding.astype(np.float16),
            )
