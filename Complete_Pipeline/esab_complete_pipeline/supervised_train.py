from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .dataset_builder import BuiltDataset
from .supervised_models import build_supervised_model, clone_model_state
from .supervised_targets import SupervisedTargets, display_wfs_ts_label
from .utils import ensure_dir


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


class CWTSupervisedDataset(Dataset):
    """Torch dataset for CWT scalograms and supervised labels."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        class_folders: np.ndarray,
        clip_indices: np.ndarray,
        indices: np.ndarray,
        task_type: str,
        image_size: int = 224,
    ) -> None:
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y)
        self.class_folders = np.asarray(class_folders)
        self.clip_indices = np.asarray(clip_indices)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.task_type = task_type
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        x_tensor = torch.from_numpy(np.ascontiguousarray(self.x[idx])).float()
        if x_tensor.ndim != 3:
            raise ValueError(f"Expected one sample with shape [C, H, W], got {tuple(x_tensor.shape)}.")

        if self.image_size > 0 and (x_tensor.shape[-2] != self.image_size or x_tensor.shape[-1] != self.image_size):
            x_tensor = F.interpolate(
                x_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if self.task_type == "classification":
            y_tensor = torch.tensor(int(self.y[idx]), dtype=torch.long)
        elif self.task_type == "regression":
            y_tensor = torch.as_tensor(self.y[idx], dtype=torch.float32).reshape(-1)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        return x_tensor, y_tensor, str(self.class_folders[idx]), int(self.clip_indices[idx]), idx


def make_supervised_splits(
    targets: SupervisedTargets,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> SplitIndices:
    """Create train/validation/test splits."""
    indices = np.asarray(targets.sample_indices, dtype=np.int64)
    y = np.asarray(targets.y)

    if len(indices) < 3:
        raise ValueError("At least three supervised samples are required for train/validation/test splitting.")

    stratify = None
    if targets.task_type == "classification":
        labels = y.reshape(-1)
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) > 1 and np.all(counts >= 2):
            stratify = labels

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=float(test_size),
        random_state=int(random_state),
        shuffle=True,
        stratify=stratify,
    )

    val_fraction_of_train_val = float(validation_size) / max(1e-12, 1.0 - float(test_size))
    val_fraction_of_train_val = min(max(val_fraction_of_train_val, 0.0), 0.9)

    stratify_train_val = None
    if targets.task_type == "classification":
        label_lookup = {int(idx): int(label) for idx, label in zip(indices, y.reshape(-1))}
        train_val_labels = np.asarray([label_lookup[int(idx)] for idx in train_val_idx], dtype=np.int64)
        unique, counts = np.unique(train_val_labels, return_counts=True)
        if len(unique) > 1 and np.all(counts >= 2):
            stratify_train_val = train_val_labels

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction_of_train_val,
        random_state=int(random_state),
        shuffle=True,
        stratify=stratify_train_val,
    )

    return SplitIndices(
        train=np.asarray(train_idx, dtype=np.int64),
        validation=np.asarray(val_idx, dtype=np.int64),
        test=np.asarray(test_idx, dtype=np.int64),
    )


def _epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task_type: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_n = 0
    all_true: list[Any] = []
    all_pred: list[Any] = []
    all_folder: list[str] = []
    all_clip_index: list[int] = []
    all_sample_index: list[int] = []

    for x, y, folders, clip_indices, sample_indices in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        output = model(x)
        if task_type == "regression":
            output = output.reshape(y.shape)
        loss = criterion(output, y)

        if is_training:
            loss.backward()
            optimizer.step()

        batch_n = x.size(0)
        total_loss += float(loss.item()) * batch_n
        total_n += batch_n

        if task_type == "classification":
            pred = torch.argmax(output, dim=1)
            all_true.extend(y.detach().cpu().numpy().astype(int).tolist())
            all_pred.extend(pred.detach().cpu().numpy().astype(int).tolist())
        else:
            all_true.extend(y.detach().cpu().numpy().reshape(batch_n, -1).tolist())
            all_pred.extend(output.detach().cpu().numpy().reshape(batch_n, -1).tolist())

        all_folder.extend(list(folders))
        all_clip_index.extend([int(v) for v in clip_indices])
        all_sample_index.extend([int(v) for v in sample_indices])

    metrics: dict[str, Any] = {
        "loss": float(total_loss / max(total_n, 1)),
        "true": all_true,
        "pred": all_pred,
        "class_folder": all_folder,
        "clip_index": all_clip_index,
        "sample_index": all_sample_index,
    }

    if task_type == "classification":
        correct = sum(int(t == p) for t, p in zip(all_true, all_pred))
        metrics["accuracy"] = float(correct / max(len(all_true), 1))
    else:
        true_arr = np.asarray(all_true, dtype=np.float64)
        pred_arr = np.asarray(all_pred, dtype=np.float64)
        rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
        mae = float(np.mean(np.abs(pred_arr - true_arr)))
        if len(true_arr) >= 2:
            try:
                r2 = float(r2_score(true_arr, pred_arr, multioutput="uniform_average"))
            except ValueError:
                r2 = float("nan")
        else:
            r2 = float("nan")
        metrics.update({"rmse": rmse, "mae": mae, "r2": r2})

    return metrics


def plot_learning_curves(history: list[dict[str, float]], output_path: Path, task_type: str) -> None:
    """Save train/validation learning curves."""
    epochs = [int(row["epoch"]) for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="Train Loss", linewidth=2)
    plt.plot(epochs, [row["validation_loss"] for row in history], label="Validation Loss", linewidth=2)
    if task_type == "classification":
        plt.plot(epochs, [row["train_accuracy"] for row in history], label="Train Accuracy", linewidth=2)
        plt.plot(epochs, [row["validation_accuracy"] for row in history], label="Validation Accuracy", linewidth=2)
    else:
        plt.plot(epochs, [row["train_r2"] for row in history], label="Train R2", linewidth=2)
        plt.plot(epochs, [row["validation_r2"] for row in history], label="Validation R2", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title("Supervised Learning Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str,
    accuracy: float | None = None,
    normalize: bool = False,
) -> None:
    """Save a readable confusion matrix figure."""
    if normalize:
        cm_to_show = cm.astype(np.float64)
        row_sums = cm_to_show.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_to_show = cm_to_show / row_sums
    else:
        cm_to_show = cm.astype(np.float64)

    cmap = LinearSegmentedColormap.from_list(
        "stronger_blue",
        ["#f2f6fb", "#d9e8f5", "#9ecae1", "#4f9dd9", "#08519c"],
    )
    fig_w = max(8, 0.75 * len(class_names))
    fig_h = max(7, 0.65 * len(class_names))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = cm_to_show.max() if cm_to_show.size > 0 and cm_to_show.max() > 0 else 1.0
    im = ax.imshow(cm_to_show, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="left", rotation_mode="anchor", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xlabel("Predicted label", fontsize=12, labelpad=12)
    ax.set_ylabel("True label", fontsize=12)

    if accuracy is not None:
        ax.set_title(f"{title}\nOverall Accuracy = {accuracy * 100:.2f}%", fontsize=14, pad=18)
    else:
        ax.set_title(title, fontsize=14, pad=18)

    threshold = vmax * 0.45
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]}\n({cm_to_show[i, j]:.2f})" if normalize else f"{cm[i, j]}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm_to_show[i, j] > threshold else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_regression_scatter(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    output_path: Path,
    target_name: str = "width",
) -> None:
    """Save predicted-vs-true regression scatter plot."""
    true_flat = np.asarray(true_values, dtype=np.float64).reshape(-1)
    pred_flat = np.asarray(pred_values, dtype=np.float64).reshape(-1)

    plt.figure(figsize=(6.5, 6))
    plt.scatter(true_flat, pred_flat, alpha=0.75, s=22)
    mn = min(float(true_flat.min()), float(pred_flat.min()))
    mx = max(float(true_flat.max()), float(pred_flat.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.5)
    plt.xlabel(f"True {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"Regression Result for {target_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_json(data: dict[str, Any], output_path: Path) -> Path:
    """Save JSON with indentation."""
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    return output_path


def save_history_csv(history: list[dict[str, float]], output_path: Path) -> Path:
    """Save supervised training history."""
    if not history:
        raise ValueError("No supervised training history to save.")
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    return output_path


def save_predictions_csv(
    metrics: dict[str, Any],
    output_path: Path,
    task_type: str,
    class_names: list[str] | None,
    target_names: list[str],
) -> Path:
    """Save sample-level predictions."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        if task_type == "classification":
            fieldnames = [
                "sample_index",
                "class_folder",
                "clip_index",
                "true_label_idx",
                "true_label_name",
                "pred_label_idx",
                "pred_label_name",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            assert class_names is not None
            for sample_idx, folder, clip_idx, true_idx, pred_idx in zip(
                metrics["sample_index"],
                metrics["class_folder"],
                metrics["clip_index"],
                metrics["true"],
                metrics["pred"],
            ):
                writer.writerow(
                    {
                        "sample_index": int(sample_idx),
                        "class_folder": folder,
                        "clip_index": int(clip_idx),
                        "true_label_idx": int(true_idx),
                        "true_label_name": display_wfs_ts_label(class_names[int(true_idx)]),
                        "pred_label_idx": int(pred_idx),
                        "pred_label_name": display_wfs_ts_label(class_names[int(pred_idx)]),
                    }
                )
        else:
            fieldnames = ["sample_index", "class_folder", "clip_index"]
            for name in target_names:
                fieldnames.extend([f"true_{name}", f"pred_{name}", f"error_{name}"])
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            true_arr = np.asarray(metrics["true"], dtype=np.float64)
            pred_arr = np.asarray(metrics["pred"], dtype=np.float64)
            for row_idx, (sample_idx, folder, clip_idx) in enumerate(
                zip(metrics["sample_index"], metrics["class_folder"], metrics["clip_index"])
            ):
                row = {
                    "sample_index": int(sample_idx),
                    "class_folder": folder,
                    "clip_index": int(clip_idx),
                }
                for target_idx, name in enumerate(target_names):
                    true_value = float(true_arr[row_idx, target_idx])
                    pred_value = float(pred_arr[row_idx, target_idx])
                    row[f"true_{name}"] = true_value
                    row[f"pred_{name}"] = pred_value
                    row[f"error_{name}"] = pred_value - true_value
                writer.writerow(row)
    return output_path


def run_supervised_experiment(
    dataset: BuiltDataset,
    targets: SupervisedTargets,
    output_dir: Path | str,
    *,
    model_name: str = "resnet18",
    pretrained: bool = False,
    image_size: int = 224,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
    num_workers: int = 0,
    early_stopping_patience: int | None = None,
    channel_indices: list[int] | tuple[int, ...] | None = None,
    patch_vit_patch_size: tuple[int, int] = (79, 20),
    patch_vit_embed_dim: int = 192,
    patch_vit_depth: int = 4,
    patch_vit_num_heads: int = 6,
    patch_vit_mlp_ratio: float = 4.0,
    patch_vit_dropout: float = 0.1,
) -> dict[str, Any]:
    """Train and evaluate one supervised CWT model."""
    output_dir = ensure_dir(Path(output_dir))
    supervised_dir = ensure_dir(output_dir / f"supervised_{targets.mode.lower()}")

    if dataset.X.ndim != 4:
        raise ValueError(f"Expected dataset.X with shape [N, C, H, W], got {dataset.X.shape}.")

    if channel_indices is None:
        x = dataset.X
        selected_channels = list(range(dataset.X.shape[1]))
    else:
        selected_channels = [int(idx) for idx in channel_indices]
        if not selected_channels:
            raise ValueError("channel_indices cannot be empty. Use None to select all channels.")
        max_channel = dataset.X.shape[1] - 1
        invalid = [idx for idx in selected_channels if idx < 0 or idx > max_channel]
        if invalid:
            raise ValueError(
                f"Invalid channel index/indices {invalid}; available range is 0..{max_channel}."
            )
        x = dataset.X[:, selected_channels, :, :]

    input_height = int(x.shape[-2])
    input_width = int(x.shape[-1])
    model_input_height = int(image_size) if int(image_size) > 0 else input_height
    model_input_width = int(image_size) if int(image_size) > 0 else input_width

    splits = make_supervised_splits(
        targets=targets,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
    )

    y_full = np.asarray(targets.y)
    if len(y_full) != len(dataset.X):
        expanded = np.full((len(dataset.X),) + y_full.shape[1:], np.nan, dtype=np.float32)
        if targets.task_type == "classification":
            expanded = np.full(len(dataset.X), -1, dtype=np.int64)
        expanded[targets.sample_indices] = y_full
        y_full = expanded

    train_data = CWTSupervisedDataset(
        x=x,
        y=y_full,
        class_folders=dataset.class_folders,
        clip_indices=dataset.clip_indices,
        indices=splits.train,
        task_type=targets.task_type,
        image_size=image_size,
    )
    val_data = CWTSupervisedDataset(
        x=x,
        y=y_full,
        class_folders=dataset.class_folders,
        clip_indices=dataset.clip_indices,
        indices=splits.validation,
        task_type=targets.task_type,
        image_size=image_size,
    )
    test_data = CWTSupervisedDataset(
        x=x,
        y=y_full,
        class_folders=dataset.class_folders,
        clip_indices=dataset.clip_indices,
        indices=splits.test,
        task_type=targets.task_type,
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = len(targets.class_names) if targets.task_type == "classification" else len(targets.target_names)
    model = build_supervised_model(
        model_name=model_name,
        in_channels=int(x.shape[1]),
        output_dim=int(output_dim),
        pretrained=pretrained,
        image_height=model_input_height,
        image_width=model_input_width,
        patch_size=patch_vit_patch_size,
        patch_vit_embed_dim=int(patch_vit_embed_dim),
        patch_vit_depth=int(patch_vit_depth),
        patch_vit_num_heads=int(patch_vit_num_heads),
        patch_vit_mlp_ratio=float(patch_vit_mlp_ratio),
        patch_vit_dropout=float(patch_vit_dropout),
    ).to(device)

    criterion: nn.Module
    if targets.task_type == "classification":
        criterion = nn.CrossEntropyLoss()
        best_key = "validation_accuracy"
        best_mode = "max"
    else:
        criterion = nn.MSELoss()
        best_key = "validation_loss"
        best_mode = "min"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    history: list[dict[str, float]] = []
    best_value = -math.inf if best_mode == "max" else math.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    patience = None if early_stopping_patience is None else int(early_stopping_patience)

    print("\n========== Supervised Training Start ==========")
    print(f"Task            : {targets.task_type}")
    print(f"Target mode     : {targets.mode}")
    print(f"Model           : {model_name}")
    print(f"Input shape     : {x.shape}")
    print(f"Selected channels: {selected_channels}")
    print(f"Image size      : {image_size}")
    print(f"Train/Val/Test  : {len(splits.train)}/{len(splits.validation)}/{len(splits.test)}")
    print("==============================================\n")

    for epoch in range(1, int(num_epochs) + 1):
        train_metrics = _epoch_pass(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            task_type=targets.task_type,
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_metrics = _epoch_pass(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                task_type=targets.task_type,
                optimizer=None,
            )

        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": float(train_metrics["loss"]),
            "validation_loss": float(val_metrics["loss"]),
        }
        if targets.task_type == "classification":
            row["train_accuracy"] = float(train_metrics["accuracy"])
            row["validation_accuracy"] = float(val_metrics["accuracy"])
        else:
            row["train_rmse"] = float(train_metrics["rmse"])
            row["validation_rmse"] = float(val_metrics["rmse"])
            row["train_mae"] = float(train_metrics["mae"])
            row["validation_mae"] = float(val_metrics["mae"])
            row["train_r2"] = float(train_metrics["r2"])
            row["validation_r2"] = float(val_metrics["r2"])
        history.append(row)

        current_value = row[best_key]
        improved = current_value > best_value if best_mode == "max" else current_value < best_value
        if improved:
            best_value = current_value
            best_state = clone_model_state(model)
            patience_counter = 0
        else:
            patience_counter += 1

        if targets.task_type == "classification":
            print(
                f"Epoch [{epoch:03d}/{num_epochs:03d}] | "
                f"train_loss={row['train_loss']:.6f} | val_loss={row['validation_loss']:.6f} | "
                f"train_acc={row['train_accuracy']:.4f} | val_acc={row['validation_accuracy']:.4f}"
            )
        else:
            print(
                f"Epoch [{epoch:03d}/{num_epochs:03d}] | "
                f"train_loss={row['train_loss']:.6f} | val_loss={row['validation_loss']:.6f} | "
                f"train_r2={row['train_r2']:.4f} | val_r2={row['validation_r2']:.4f}"
            )

        if patience is not None and patience > 0 and patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        test_metrics = _epoch_pass(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            task_type=targets.task_type,
            optimizer=None,
        )

    save_history_csv(history, supervised_dir / "training_history.csv")
    plot_learning_curves(history, supervised_dir / "learning_curves.png", targets.task_type)
    save_predictions_csv(
        test_metrics,
        supervised_dir / "test_predictions.csv",
        targets.task_type,
        targets.class_names,
        targets.target_names,
    )

    split_info = {
        "target_mode": targets.mode,
        "task_type": targets.task_type,
        "train_idx": splits.train.tolist(),
        "val_idx": splits.validation.tolist(),
        "test_idx": splits.test.tolist(),
        "class_names": targets.class_names,
        "target_names": targets.target_names,
        "selected_channels": selected_channels,
    }
    save_json(split_info, supervised_dir / "split_info.json")

    checkpoint_path = supervised_dir / f"best_{model_name}_{targets.mode.lower()}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "task_type": targets.task_type,
            "target_mode": targets.mode,
            "image_size": int(image_size),
            "selected_channels": selected_channels,
            "in_channels": int(x.shape[1]),
            "input_height": int(input_height),
            "input_width": int(input_width),
            "model_input_height": int(model_input_height),
            "model_input_width": int(model_input_width),
            "output_dim": int(output_dim),
            "class_names": targets.class_names,
            "target_names": targets.target_names,
            "history": history,
        },
        checkpoint_path,
    )

    summary: dict[str, Any] = {
        "target_mode": targets.mode,
        "task_type": targets.task_type,
        "model_name": model_name,
        "selected_channels": selected_channels,
        "checkpoint": str(checkpoint_path),
        "num_train": int(len(splits.train)),
        "num_validation": int(len(splits.validation)),
        "num_test": int(len(splits.test)),
        "test_loss": float(test_metrics["loss"]),
    }

    if targets.task_type == "classification":
        assert targets.class_names is not None
        cm = confusion_matrix(
            test_metrics["true"],
            test_metrics["pred"],
            labels=list(range(len(targets.class_names))),
        )
        display_names = [display_wfs_ts_label(name) for name in targets.class_names]
        plot_confusion_matrix(
            cm=cm,
            class_names=display_names,
            output_path=supervised_dir / "confusion_matrix_test.png",
            title=f"{targets.mode} Confusion Matrix",
            accuracy=float(test_metrics["accuracy"]),
            normalize=False,
        )
        plot_confusion_matrix(
            cm=cm,
            class_names=display_names,
            output_path=supervised_dir / "confusion_matrix_test_normalized.png",
            title=f"{targets.mode} Confusion Matrix (Normalized)",
            accuracy=float(test_metrics["accuracy"]),
            normalize=True,
        )
        report = classification_report(
            test_metrics["true"],
            test_metrics["pred"],
            target_names=display_names,
            digits=4,
            output_dict=True,
            zero_division=0,
        )
        save_json(report, supervised_dir / "classification_report_test.json")
        summary["test_accuracy"] = float(test_metrics["accuracy"])
        summary["test_accuracy_percent"] = round(float(test_metrics["accuracy"]) * 100.0, 2)
    else:
        true_arr = np.asarray(test_metrics["true"], dtype=np.float64)
        pred_arr = np.asarray(test_metrics["pred"], dtype=np.float64)
        plot_regression_scatter(
            true_values=true_arr,
            pred_values=pred_arr,
            output_path=supervised_dir / "regression_true_vs_pred_test.png",
            target_name=targets.target_names[0],
        )
        summary["test_rmse"] = float(test_metrics["rmse"])
        summary["test_mae"] = float(test_metrics["mae"])
        summary["test_r2"] = float(test_metrics["r2"])

    save_json(summary, supervised_dir / "summary_test.json")

    print("\n========== Supervised Training Finished ==========")
    print(f"Saved supervised outputs to: {supervised_dir}")
    print("=================================================\n")

    return summary
