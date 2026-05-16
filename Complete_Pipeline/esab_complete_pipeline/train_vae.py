from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from .vae_model import ConvVAE, vae_loss


@dataclass(frozen=True)
class TrainResult:
    """VAE training result."""

    model: ConvVAE
    losses: List[Dict[str, float]]


def iterate_batches(
    X: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> Iterator[np.ndarray]:
    """Yield NumPy mini-batches."""
    n = len(X)
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield X[batch_idx]


def train_vae(
    X: np.ndarray,
    latent_dim: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    beta: float,
) -> TrainResult:
    """Train a convolutional VAE on scalograms."""
    if X.ndim != 4:
        raise ValueError(f"Expected X with shape [N, C, H, W], got {X.shape}.")

    if len(X) == 0:
        raise ValueError("Cannot train on an empty dataset.")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, in_channels, height, width = X.shape
    model = ConvVAE(
        in_channels=in_channels,
        height=height,
        width=width,
        latent_dim=latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: List[Dict[str, float]] = []

    steps_per_epoch = int(np.ceil(len(X) / batch_size))
    total_steps = steps_per_epoch * num_epochs

    print("\n========== VAE Training Start ==========")
    print(f"Device          : {device}")
    print(f"Input shape     : {X.shape}")
    print(f"Num samples     : {len(X)}")
    print(f"Batch size      : {batch_size}")
    print(f"Steps/epoch     : {steps_per_epoch}")
    print(f"Num epochs      : {num_epochs}")
    print(f"Total steps     : {total_steps}")
    print(f"Latent dim      : {latent_dim}")
    print(f"Learning rate   : {learning_rate}")
    print(f"Beta max        : {beta}")
    print("========================================\n")

    global_start_time = time.time()

    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        beta_now = beta * (epoch / num_epochs)

        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0
        total_n = 0

        mu_all: list[torch.Tensor] = []
        logvar_all: list[torch.Tensor] = []

        for batch_np in iterate_batches(X, batch_size=batch_size, shuffle=True):
            batch_x = torch.from_numpy(np.ascontiguousarray(batch_np)).to(device)

            optimizer.zero_grad()
            x_hat, mu, logvar = model(batch_x)
            loss, recon, kld = vae_loss(x_hat, batch_x, mu, logvar, beta=beta_now)
            loss.backward()
            optimizer.step()

            batch_size_actual = batch_x.size(0)
            total_loss += float(loss.item()) * batch_size_actual
            total_recon += float(recon.item()) * batch_size_actual
            total_kld += float(kld.item()) * batch_size_actual
            total_n += batch_size_actual

            mu_all.append(mu.detach().cpu())
            logvar_all.append(logvar.detach().cpu())

        epoch_loss = total_loss / total_n
        epoch_recon = total_recon / total_n
        epoch_kld = total_kld / total_n

        mu_tensor = torch.cat(mu_all, dim=0)
        logvar_tensor = torch.cat(logvar_all, dim=0)

        row = {
            "epoch": float(epoch),
            "beta": float(beta_now),
            "loss": float(epoch_loss),
            "recon": float(epoch_recon),
            "kld": float(epoch_kld),
            "mu_mean": float(mu_tensor.mean().item()),
            "mu_std": float(mu_tensor.std().item()),
            "logvar_mean": float(logvar_tensor.mean().item()),
            "logvar_std": float(logvar_tensor.std().item()),
            "epoch_seconds": float(time.time() - epoch_start_time),
            "elapsed_seconds": float(time.time() - global_start_time),
        }
        history.append(row)

        progress = 100.0 * epoch / num_epochs
        print(
            f"Epoch [{epoch:03d}/{num_epochs:03d}] "
            f"({progress:6.2f}%) | "
            f"Beta: {beta_now:.6f} | "
            f"Loss: {epoch_loss:.6f} | "
            f"Recon: {epoch_recon:.6f} | "
            f"KLD: {epoch_kld:.6f} | "
            f"mu_mean: {row['mu_mean']:.6f} | "
            f"mu_std: {row['mu_std']:.6f} | "
            f"logvar_mean: {row['logvar_mean']:.6f} | "
            f"logvar_std: {row['logvar_std']:.6f} | "
            f"Epoch time: {row['epoch_seconds']:.2f}s | "
            f"Elapsed: {row['elapsed_seconds']:.2f}s"
        )

    print("\n========== VAE Training Finished ==========")
    print(f"Total elapsed time: {time.time() - global_start_time:.2f}s")
    print("===========================================\n")

    return TrainResult(model=model, losses=history)


def save_training_history(
    losses: List[Dict[str, float]],
    output_path: Path | str,
) -> Path:
    """Save training history as CSV."""
    output_path = Path(output_path)

    if not losses:
        raise ValueError("No loss history to save.")

    fieldnames = list(losses[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(losses)

    return output_path


def plot_training_losses(
    losses: List[Dict[str, float]],
    save_path: Path | str | None = None,
    show: bool = False,
) -> None:
    """Plot total, reconstruction, and KLD loss curves."""
    if not losses:
        raise ValueError("No loss history to plot.")

    epochs = [int(item["epoch"]) for item in losses]
    total_loss = [item["loss"] for item in losses]
    recon_loss = [item["recon"] for item in losses]
    kld_loss = [item["kld"] for item in losses]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, total_loss, label="Total Loss", linewidth=2)
    plt.plot(epochs, recon_loss, label="Reconstruction Loss", linewidth=2)
    plt.plot(epochs, kld_loss, label="KLD Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("VAE Training Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


@torch.no_grad()
def extract_latents(
    model: ConvVAE,
    X: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract posterior mean vectors from a trained VAE."""
    device = next(model.parameters()).device
    model.eval()

    latents: list[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        batch_np = X[start : start + batch_size]
        batch_x = torch.from_numpy(np.ascontiguousarray(batch_np)).to(device)
        mu, _ = model.encode(batch_x)
        latents.append(mu.cpu().numpy())

    return np.concatenate(latents, axis=0)


def save_latents_csv(
    latents: np.ndarray,
    labels: np.ndarray,
    class_folders: np.ndarray,
    source_files: np.ndarray,
    clip_indices: np.ndarray,
    output_path: Path | str,
) -> Path:
    """Save latent vectors with sample metadata."""
    output_path = Path(output_path)
    columns = [f"z_{idx}" for idx in range(latents.shape[1])]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "class_folder", "source_file", "clip_index", *columns])

        for idx in range(len(latents)):
            writer.writerow(
                [
                    str(labels[idx]),
                    str(class_folders[idx]),
                    str(source_files[idx]),
                    int(clip_indices[idx]),
                    *[float(value) for value in latents[idx]],
                ]
            )

    return output_path
