from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def run_tsne(
    latents: np.ndarray,
    perplexity: int = 10,
    random_state: int = 42,
    n_iter: int = 1000,
) -> np.ndarray:
    """Run t-SNE on latent vectors."""
    if len(latents) <= 1:
        raise ValueError("t-SNE requires at least two samples.")

    safe_perplexity = min(perplexity, max(1, len(latents) - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        random_state=random_state,
        max_iter=n_iter,
        init="pca",
        learning_rate="auto",
    )
    embedding = tsne.fit_transform(latents)
    return embedding.astype(np.float32)


def plot_tsne(
    embedding: np.ndarray,
    labels: Sequence[str],
    save_path: Path | str | None = None,
    show: bool = False,
) -> None:
    """Plot a labeled t-SNE embedding."""
    labels = np.asarray(labels).astype(str)
    unique_labels = sorted(set(labels))

    plt.figure(figsize=(9, 7))
    for label in unique_labels:
        mask = labels == label
        points = embedding[mask]
        plt.scatter(points[:, 0], points[:, 1], s=20, alpha=0.8, label=label)

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE of VAE Latent Vectors")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_latent_2d(
    latents: np.ndarray,
    labels: Sequence[str],
    save_path: Path | str | None = None,
    show: bool = False,
) -> None:
    """Plot a two-dimensional latent space directly."""
    if latents.shape[1] < 2:
        raise ValueError("plot_latent_2d requires at least two latent dimensions.")

    labels = np.asarray(labels).astype(str)
    unique_labels = sorted(set(labels))

    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        mask = labels == label
        plt.scatter(
            latents[mask, 0],
            latents[mask, 1],
            label=label,
            alpha=0.7,
            s=20,
        )

    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("2D Latent Space")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
