#!/usr/bin/env python3
"""
Analyze and visualize latent representations from two datasets using t-SNE.
Also computes Wasserstein-2 distance between distributions.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
from sklearn.manifold import TSNE


def load_latents_from_path(dataset_path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Load latent tensors from episode*.pt files in the given path.

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        Tuple of (concatenated latents array of shape [total_frames, latent_dim], list of episode names)
    """
    episode_files = sorted(dataset_path.glob("episode*.pt"))
    all_latents, episode_names = [], []

    for episode_file in episode_files:
        latent_tensor = torch.load(episode_file)["latents"]
        latent_np = latent_tensor.cpu().numpy()
        all_latents.append(latent_np)
        episode_names.append(episode_file.name)

    concatenated_latents = np.concatenate(all_latents, axis=0)
    return concatenated_latents, episode_names


def compute_wasserstein2_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute the Wasserstein-2 distance between two point clouds.

    Args:
        X: First point cloud of shape [n_samples, n_features]
        Y: Second point cloud of shape [m_samples, n_features]

    Returns:
        Wasserstein-2 distance
    """
    # Uniform weights for both distributions
    a = np.ones(len(X)) / len(X)
    b = np.ones(len(Y)) / len(Y)

    # Compute cost matrix (squared Euclidean distance)
    M = ot.dist(X, Y, metric="sqeuclidean")

    # Compute Wasserstein distance squared using EMD
    w2_squared = ot.emd2(a, b, M)

    # Return Wasserstein-2 distance (square root)
    return np.sqrt(w2_squared)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze latents using t-SNE visualization"
    )
    parser.add_argument(
        "--dset1_path", type=str, required=True, help="Path to dataset 1"
    )
    parser.add_argument(
        "--dset2_path", type=str, required=True, help="Path to dataset 2"
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity parameter (default: 30)",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1000,
        help="Number of t-SNE iterations (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="latent_tsne.png",
        help="Output figure path (default: latent_tsne.png)",
    )

    args = parser.parse_args()

    dset1_path = Path(args.dset1_path)
    dset2_path = Path(args.dset2_path)

    dset1_latents, _ = load_latents_from_path(dset1_path)
    dset2_latents, _ = load_latents_from_path(dset2_path)

    # Compute Wasserstein-2 distance
    w2_dist = compute_wasserstein2_distance(dset1_latents, dset2_latents)
    print(f"\nWasserstein-2 distance: {w2_dist:.6f}\n")

    # Combine all latents
    all_latents = np.concatenate([dset1_latents, dset2_latents], axis=0)
    dset1_labels = np.zeros(len(dset1_latents))
    dset2_labels = np.ones(len(dset2_latents))
    all_labels = np.concatenate([dset1_labels, dset2_labels])

    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        random_state=42,
        verbose=1,
    )
    embeddings = tsne.fit_transform(all_latents)

    dset1_embeddings = embeddings[all_labels == 0]
    dset2_embeddings = embeddings[all_labels == 1]

    plt.figure(figsize=(12, 8))
    plt.scatter(
        dset1_embeddings[:, 0],
        dset1_embeddings[:, 1],
        c="blue",
        alpha=0.6,
        s=10,
        label=f"{args.dset1_path} (n={len(dset1_latents)})",
    )
    plt.scatter(
        dset2_embeddings[:, 0],
        dset2_embeddings[:, 1],
        c="red",
        alpha=0.6,
        s=10,
        label=f"{args.dset2_path} (n={len(dset2_latents)})",
    )
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.title(
        f"t-SNE Visualization of Latents (W2: {w2_dist:.4f})",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {args.output}")
    print("\nDone!")


if __name__ == "__main__":
    main()
