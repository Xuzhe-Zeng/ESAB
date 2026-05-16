from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from .clip_utils import clip_all_records
from .dataset_builder import (
    build_grouped_cwt_dataset,
    flatten_grouped_cwt_dataset,
    load_cwt_dataset_npz,
    save_grouped_dataset_npz,
)
from .io_utils import discover_synced_records
from .sync_utils import build_synced_cache, discover_raw_signal_bundles
from .supervised_targets import build_supervised_targets, normalize_target_mode
from .supervised_train import run_supervised_experiment
from .train_vae import (
    extract_latents,
    plot_training_losses,
    save_latents_csv,
    save_training_history,
    train_vae,
)
from .utils import ensure_dir, set_seed
from .visualize import plot_latent_2d, plot_tsne, run_tsne


def load_config(config_module: str | None = None) -> ModuleType:
    """Load the default or user-provided Python config module."""
    if config_module is None:
        return importlib.import_module("esab_complete_pipeline.config")
    return importlib.import_module(config_module)


def get_attr(config: ModuleType, name: str, default: Any) -> Any:
    """Return config attribute with fallback."""
    return getattr(config, name, default)


def _target_modes(config: ModuleType) -> list[str]:
    """Return the supervised target modes to run."""
    raw = get_attr(config, "SUPERVISED_TARGET_MODES", None)

    if raw is None:
        raw = [get_attr(config, "TARGET_MODE", "WFS_TS")]

    if isinstance(raw, str):
        raw = [raw]

    modes = [normalize_target_mode(mode) for mode in raw]

    deduped: list[str] = []
    for mode in modes:
        if mode not in deduped:
            deduped.append(mode)

    return deduped


def _lookup_target_setting(
    config: ModuleType,
    mapping_name: str,
    fallback_name: str,
    target_mode: str,
    default: Any,
) -> Any:
    """Get a target-specific setting, falling back to a global setting."""
    mapping = get_attr(config, mapping_name, None)

    if isinstance(mapping, dict):
        if target_mode in mapping:
            return mapping[target_mode]
        if target_mode.upper() in mapping:
            return mapping[target_mode.upper()]
        if target_mode.lower() in mapping:
            return mapping[target_mode.lower()]

    return get_attr(config, fallback_name, default)


def _normalize_channel_indices(value: Any) -> list[int] | None:
    """Normalize a config value into channel indices or None for all channels."""
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "all"}:
            return None
        return [int(part.strip()) for part in text.split(",") if part.strip()]

    if isinstance(value, int):
        return [int(value)]

    return [int(v) for v in value]


def _build_or_load_dataset(config: ModuleType, output_dir: Path):
    """Build the CWT dataset from signals or load an existing grouped CWT NPZ dataset."""
    dataset_source = str(get_attr(config, "DATASET_SOURCE", "build")).strip().lower()

    if dataset_source == "grouped_npz":
        dataset_path = Path(
            get_attr(
                config,
                "CWT_DATASET_PATH",
                output_dir / "cwt_dataset_grouped.npz",
            )
        )

        print(f"\n[1/4] Loading existing grouped CWT dataset: {dataset_path}")

        dataset = load_cwt_dataset_npz(
            dataset_path,
            dataset_source="grouped_npz",
            signal_names=get_attr(config, "SIGNALS_TO_USE", None),
        )

        print(f"Loaded dataset shape: {dataset.X.shape}")
        return dataset

    if dataset_source != "build":
        raise ValueError(
            "DATASET_SOURCE must be either 'build' or 'grouped_npz'."
        )

    synced_root = Path(config.ROOT_DIR)

    if not bool(config.USE_SYNC_STAGE):
        print("\n[1/4] Discovering raw signal bundles...")

        bundles = discover_raw_signal_bundles(
            root_dir=Path(config.ROOT_DIR),
            label_regex=config.DISPLAY_LABEL_REGEX,
        )

        print(f"Found {len(bundles)} raw bundle(s).")

        print("\n[2/4] Building synced cache...")

        synced_root = build_synced_cache(
            bundles=bundles,
            output_root=Path(config.SYNC_CACHE_DIR),
            audio_key=config.AUDIO_KEY,
            current_eps=float(config.CURRENT_EPS),
            current_zero_run=int(config.CURRENT_ZERO_RUN),
            current_active_window=int(config.CURRENT_ACTIVE_WINDOW),
            current_min_active_count=int(config.CURRENT_MIN_ACTIVE_COUNT),
            current_near_window=int(config.CURRENT_NEAR_WINDOW),
            current_near_min_active=int(config.CURRENT_NEAR_MIN_ACTIVE),
            trim_ratio=float(config.SYNC_TRIM_RATIO),
            target_fs=float(config.TARGET_FS),
        )

        print(f"Synced files saved to: {synced_root}")

    else:
        print("\n[1/4] Using existing synced data.")
        print(f"Synced root: {synced_root}")

    print("\n[2/4] Loading synced records and clipping signals...")

    records = discover_synced_records(
        root_dir=synced_root,
        sync_file_patterns=config.SYNC_FILE_PATTERNS,
        signals_to_use=config.SIGNALS_TO_USE,
        label_regex=config.DISPLAY_LABEL_REGEX,
    )

    print(f"Loaded {len(records)} synced record(s).")

    clips = clip_all_records(
        records=records,
        clip_seconds=float(config.CLIP_SECONDS),
        clip_interval_seconds=float(config.CLIP_INTERVAL_SECONDS),
        drop_last_incomplete=bool(config.DROP_LAST_INCOMPLETE_CLIP),
    )

    print(f"Generated {len(clips)} clip(s).")

    if not clips:
        raise RuntimeError(
            "No clips were generated. Check clip settings and input data."
        )

    print("\n[3/4] Building grouped CWT dataset...")

    dt = 1.0 / float(config.TARGET_FS)

    grouped_data = build_grouped_cwt_dataset(
        clips=clips,
        signal_names=config.SIGNALS_TO_USE,
        dt=dt,
        magnitude=bool(config.CWT_MAGNITUDE),
        log1p_transform=bool(config.CWT_LOG1P),
        normalize_per_channel=bool(config.CWT_NORMALIZE_PER_CHANNEL),
        wavelet_name=str(get_attr(config, "CWT_WAVELET_NAME", "gmw")),
    )

    if bool(get_attr(config, "SAVE_GROUPED_DATASET_NPZ", True)):
        grouped_path = save_grouped_dataset_npz(
            grouped_data,
            output_dir / "cwt_dataset_grouped.npz",
        )
        print(f"Saved grouped dataset: {grouped_path}")

    dataset = flatten_grouped_cwt_dataset(
        grouped_data,
        config.SIGNALS_TO_USE,
    )

    print(f"Flattened in-memory dataset shape: {dataset.X.shape}")

    return dataset


def _run_vae(config: ModuleType, dataset, output_dir: Path) -> None:
    """Run VAE training and latent-space visualization."""
    if not bool(get_attr(config, "RUN_VAE", get_attr(config, "RUN_TRAINING", True))):
        print("\n[3/4] VAE training is disabled.")
        return

    vae_output_dir = ensure_dir(
        output_dir / str(get_attr(config, "VAE_OUTPUT_SUBDIR", "vae_results"))
    )

    print("\n[3/4] Training VAE...")
    print(f"VAE output directory: {vae_output_dir}")

    result = train_vae(
        X=dataset.X,
        latent_dim=int(config.LATENT_DIM),
        batch_size=int(config.BATCH_SIZE),
        num_epochs=int(config.NUM_EPOCHS),
        learning_rate=float(config.LEARNING_RATE),
        beta=float(config.BETA),
    )

    history_path = save_training_history(
        result.losses,
        vae_output_dir / "vae_training_history.csv",
    )
    print(f"Saved training history: {history_path}")

    loss_fig_path = vae_output_dir / "vae_training_losses.png"
    plot_training_losses(
        result.losses,
        save_path=loss_fig_path,
        show=False,
    )
    print(f"Saved loss figure: {loss_fig_path}")

    if bool(get_attr(config, "SAVE_MODEL_WEIGHTS", True)):
        weights_path = vae_output_dir / "vae_model.pt"
        torch.save(result.model.state_dict(), weights_path)
        print(f"Saved model weights: {weights_path}")

    latents = extract_latents(
        result.model,
        dataset.X,
        batch_size=int(config.BATCH_SIZE),
    )

    if bool(get_attr(config, "SAVE_LATENTS_CSV", True)):
        latents_path = save_latents_csv(
            latents=latents,
            labels=dataset.labels,
            class_folders=dataset.class_folders,
            source_files=dataset.source_files,
            clip_indices=dataset.clip_indices,
            output_path=vae_output_dir / "vae_latents.csv",
        )
        print(f"Saved latent vectors: {latents_path}")

    if bool(get_attr(config, "RUN_TSNE", True)) and bool(
        get_attr(config, "SAVE_TSNE_FIG", True)
    ):
        embedding = run_tsne(
            latents,
            perplexity=int(config.TSNE_PERPLEXITY),
            random_state=int(config.TSNE_RANDOM_STATE),
            n_iter=int(config.TSNE_N_ITER),
        )

        tsne_path = vae_output_dir / "vae_latent_tsne.png"

        plot_tsne(
            embedding,
            dataset.labels,
            save_path=tsne_path,
            show=False,
        )

        print(f"Saved t-SNE figure: {tsne_path}")


def _run_supervised(config: ModuleType, dataset, output_dir: Path) -> None:
    """Run one or more supervised target modes."""
    if not bool(get_attr(config, "RUN_SUPERVISED", False)):
        print("\n[4/4] Supervised learning is disabled.")
        return

    print("\n[4/4] Running supervised learning...")

    summaries: list[dict[str, Any]] = []

    for target_mode in _target_modes(config):
        print(f"\n--- Supervised target: {target_mode} ---")

        targets = build_supervised_targets(
            dataset=dataset,
            target_mode=target_mode,
            width_excel_path=get_attr(config, "WIDTH_EXCEL_PATH", None),
            width_class_folder_column=str(
                get_attr(config, "WIDTH_CLASS_FOLDER_COLUMN", "class_folder")
            ),
            width_clip_index_column=str(
                get_attr(config, "WIDTH_CLIP_INDEX_COLUMN", "indices")
            ),
            width_target_column=str(
                get_attr(config, "WIDTH_TARGET_COLUMN", "width")
            ),
            width_missing_policy=str(
                get_attr(config, "WIDTH_MISSING_POLICY", "drop")
            ),
        )

        model_name = str(
            _lookup_target_setting(
                config,
                "SUPERVISED_MODEL_BY_TARGET",
                "SUPERVISED_MODEL_NAME",
                target_mode,
                "resnet18",
            )
        )

        image_size = int(
            _lookup_target_setting(
                config,
                "SUPERVISED_IMAGE_SIZE_BY_TARGET",
                "SUPERVISED_IMAGE_SIZE",
                target_mode,
                224,
            )
        )

        channels = _normalize_channel_indices(
            _lookup_target_setting(
                config,
                "SUPERVISED_CHANNELS_BY_TARGET",
                "SUPERVISED_CHANNELS",
                target_mode,
                None,
            )
        )

        summary = run_supervised_experiment(
            dataset=dataset,
            targets=targets,
            output_dir=output_dir,
            model_name=model_name,
            pretrained=bool(get_attr(config, "SUPERVISED_PRETRAINED", False)),
            image_size=image_size,
            batch_size=int(get_attr(config, "SUPERVISED_BATCH_SIZE", 32)),
            num_epochs=int(get_attr(config, "SUPERVISED_NUM_EPOCHS", 50)),
            learning_rate=float(get_attr(config, "SUPERVISED_LEARNING_RATE", 1e-4)),
            weight_decay=float(get_attr(config, "SUPERVISED_WEIGHT_DECAY", 1e-4)),
            test_size=float(get_attr(config, "SUPERVISED_TEST_SIZE", 0.2)),
            validation_size=float(
                get_attr(config, "SUPERVISED_VALIDATION_SIZE", 0.2)
            ),
            random_state=int(get_attr(config, "RANDOM_SEED", 42)),
            num_workers=int(get_attr(config, "SUPERVISED_NUM_WORKERS", 0)),
            early_stopping_patience=get_attr(
                config,
                "SUPERVISED_EARLY_STOPPING_PATIENCE",
                None,
            ),
            channel_indices=channels,
            patch_vit_patch_size=get_attr(
                config,
                "PATCH_VIT_PATCH_SIZE",
                (79, 20),
            ),
            patch_vit_embed_dim=int(
                get_attr(config, "PATCH_VIT_EMBED_DIM", 192)
            ),
            patch_vit_depth=int(
                get_attr(config, "PATCH_VIT_DEPTH", 4)
            ),
            patch_vit_num_heads=int(
                get_attr(config, "PATCH_VIT_NUM_HEADS", 6)
            ),
            patch_vit_mlp_ratio=float(
                get_attr(config, "PATCH_VIT_MLP_RATIO", 4.0)
            ),
            patch_vit_dropout=float(
                get_attr(config, "PATCH_VIT_DROPOUT", 0.1)
            ),
        )

        summaries.append(summary)

        print(f"Supervised summary for {target_mode}: {summary}")

    print(f"Completed {len(summaries)} supervised target(s).")


def run_pipeline(config: ModuleType) -> None:
    """Run dataset preparation, VAE analysis, and supervised prediction."""
    set_seed(int(config.RANDOM_SEED))

    output_dir = ensure_dir(Path(config.OUTPUT_DIR))

    print("========== ESAB Complete Pipeline ==========")
    print(f"Output directory: {output_dir}")

    dataset = _build_or_load_dataset(config, output_dir)

    _run_vae(config, dataset, output_dir)

    _run_supervised(config, dataset, output_dir)

    print("\nPipeline finished successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ESAB complete pipeline."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional import path to a Python config module, for example "
            "`examples.local_config_example`. If omitted, "
            "esab_complete_pipeline.config is used."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    run_pipeline(config)


if __name__ == "__main__":
    main()