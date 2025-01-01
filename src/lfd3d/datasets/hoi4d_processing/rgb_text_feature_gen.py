import glob
import json
import os

import cv2
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

IMG_HEIGHT, IMG_WIDTH = 270, 480  # HOI4D scaled down from 1920x1080
feat_dim = 1152  # Siglip feature dim
HOI4D_DIR = "/data/sriram/hoi4d/hoi4d_data/"


def compute_pixel_aligned_embedding(
    img_path, caption, mask_generator, siglip, siglip_processor, cosine_similarity
):
    """
    Adapted from ConceptFusion
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img).cuda()

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
        img_roi = img[_y : _y + _h, _x : _x + _w, :]

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


# Load models

torch.autograd.set_grad_enabled(False)

# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
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

# Prepare PCA
if not os.path.exists("pca_model.pkl"):
    print("Collecting some features to fit PCA")
    img_paths = glob.glob(
        f"{HOI4D_DIR}/**/00200.jpg", recursive=True
    )  # Random image just for computing PCA
    pca_fit_features = []
    # Fit a PCA with first 10 images.
    for idx, img_path in tqdm(enumerate(img_paths[:10]), total=10):
        pix_align_embedding, _ = compute_pixel_aligned_embedding(
            img_path,
            "test caption",
            mask_generator,
            siglip,
            siglip_processor,
            cosine_similarity,
        )
        pca_fit_features.append(pix_align_embedding)

    pca_model = PCA(n_components=256)
    features_proc = np.array(pca_fit_features).transpose(0, 3, 1, 2)
    downsample_features = F.interpolate(
        torch.from_numpy(features_proc),
        scale_factor=1 / 5,
        mode="bilinear",
        align_corners=False,
    )
    downsample_features = (
        downsample_features.numpy().transpose(0, 2, 3, 1).reshape(-1, feat_dim)
    )
    pca_model.fit(downsample_features)

    joblib.dump(pca_model, "pca_model.pkl")

pca_model = joblib.load("pca_model.pkl")

hoi4d_videos = sorted(glob.glob(f"{HOI4D_DIR}/**/image.mp4", recursive=True))
base_dirs = [os.path.dirname(os.path.dirname(f)) for f in hoi4d_videos]

# Process hoi4d video
for idx, dir_name in tqdm(enumerate(base_dirs), total=len(base_dirs)):
    os.makedirs(f"{dir_name}/rgb_text_features", exist_ok=True)

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

        if os.path.exists(save_name):
            continue

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

        img_embedding, text_embedding = compute_pixel_aligned_embedding(
            img_path,
            caption,
            mask_generator,
            siglip,
            siglip_processor,
            cosine_similarity,
        )
        img_embedding_compressed = compress_features(pca_model, img_embedding).astype(
            np.float32
        )
        # measure_compression_error(pca_model, img_embedding, img_embedding_compressed)

        np.savez_compressed(
            save_name,
            rgb_embed=img_embedding_compressed.astype(np.float16),
            text_embed=text_embedding.astype(np.float16),
        )
