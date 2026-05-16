from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from .clip_utils import SignalClip
from .cwt_utils import simple_cwt
from .utils import sanitize_npz_key


@dataclass(frozen=True)
class BuiltDataset:
    """Flattened in-memory CWT dataset and sample metadata.

    Note:
        The saved dataset format is grouped NPZ.
        This flattened structure is only used internally for model training,
        validation, evaluation, and visualization.
    """

    X: np.ndarray
    y: np.ndarray
    labels: np.ndarray
    class_folders: np.ndarray
    source_files: np.ndarray
    clip_indices: np.ndarray
    signal_names: list[str]


GroupedDataset = Dict[str, Dict[str, np.ndarray]]


def build_grouped_cwt_dataset(
    clips: Sequence[SignalClip],
    signal_names: Sequence[str],
    dt: float,
    magnitude: bool = True,
    log1p_transform: bool = True,
    normalize_per_channel: bool = True,
    wavelet_name: str = "gmw",
) -> GroupedDataset:
    """Build CWT scalograms grouped by display label.

    Output grouped format:
        {
            "WFS_10_TS_4": {
                "scalogram": ndarray,      # [N, C, H, W]
                "clip_indices": ndarray,   # [N]
                "class_folder": ndarray,   # [N]
                "source_file": ndarray,    # [N]
            },
            ...
        }
    """
    grouped_data: GroupedDataset = {}

    for clip in clips:
        channels: list[np.ndarray] = []
        valid = True

        for signal_name in signal_names:
            signal = clip.signals.get(signal_name)

            if signal is None or len(signal) < 4:
                valid = False
                break

            try:
                cwt_image = simple_cwt(
                    x=signal,
                    dt=dt,
                    magnitude=magnitude,
                    log1p_transform=log1p_transform,
                    normalize_per_channel=normalize_per_channel,
                    wavelet_name=wavelet_name,
                )
            except Exception as exc:
                print(
                    "[WARN] CWT failed for "
                    f"{clip.source_file}, clip={clip.clip_index}, "
                    f"signal={signal_name}: {exc}"
                )
                valid = False
                break

            channels.append(cwt_image)

        if not valid:
            continue

        sample = np.stack(channels, axis=0).astype(np.float32)
        label = clip.display_label

        if label not in grouped_data:
            grouped_data[label] = {
                "scalogram": [],
                "clip_indices": [],
                "class_folder": [],
                "source_file": [],
            }

        grouped_data[label]["scalogram"].append(sample)
        grouped_data[label]["clip_indices"].append(clip.clip_index)
        grouped_data[label]["class_folder"].append(clip.class_folder)
        grouped_data[label]["source_file"].append(clip.source_file)

    if not grouped_data:
        raise ValueError("No valid clips remained after CWT conversion.")

    for label, data in grouped_data.items():
        shapes = [array.shape for array in data["scalogram"]]
        print(f"\nLabel: {label}")
        print(f"num samples = {len(data['scalogram'])}")
        print(f"unique shapes = {Counter(shapes)}")

        data["scalogram"] = np.stack(data["scalogram"], axis=0).astype(np.float32)
        data["clip_indices"] = np.asarray(data["clip_indices"], dtype=np.int64)
        data["class_folder"] = np.asarray(data["class_folder"], dtype=str)
        data["source_file"] = np.asarray(data["source_file"], dtype=str)

    return grouped_data


def flatten_grouped_cwt_dataset(
    grouped_data: GroupedDataset,
    signal_names: Sequence[str],
) -> BuiltDataset:
    """Flatten a grouped CWT dataset into arrays for model training.

    The project saves grouped NPZ files, but most PyTorch datasets are easier
    to build from flat arrays. This function only converts the grouped data
    in memory.
    """
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    class_folder_list: list[np.ndarray] = []
    source_file_list: list[np.ndarray] = []
    clip_index_list: list[np.ndarray] = []

    label_names = sorted(grouped_data)
    label_to_id = {label: idx for idx, label in enumerate(label_names)}

    for label in label_names:
        data = grouped_data[label]
        X_label = np.asarray(data["scalogram"], dtype=np.float32)

        if X_label.ndim != 4:
            raise ValueError(
                f"{label}: expected scalogram shape [N, C, H, W], "
                f"got {X_label.shape}."
            )

        n = X_label.shape[0]

        clip_indices = np.asarray(data["clip_indices"], dtype=np.int64)
        class_folder = np.asarray(data["class_folder"], dtype=object)
        source_file = np.asarray(data["source_file"], dtype=object)

        if len(clip_indices) != n:
            raise ValueError(
                f"{label}: len(clip_indices)={len(clip_indices)} does not "
                f"match number of scalograms={n}."
            )

        if len(class_folder) != n:
            raise ValueError(
                f"{label}: len(class_folder)={len(class_folder)} does not "
                f"match number of scalograms={n}."
            )

        if len(source_file) != n:
            raise ValueError(
                f"{label}: len(source_file)={len(source_file)} does not "
                f"match number of scalograms={n}."
            )

        X_list.append(X_label)
        y_list.append(np.full(n, label_to_id[label], dtype=np.int64))
        labels_list.append(np.full(n, label, dtype=object))
        class_folder_list.append(class_folder)
        source_file_list.append(source_file)
        clip_index_list.append(clip_indices)

    return BuiltDataset(
        X=np.concatenate(X_list, axis=0).astype(np.float32),
        y=np.concatenate(y_list, axis=0).astype(np.int64),
        labels=np.concatenate(labels_list, axis=0),
        class_folders=np.concatenate(class_folder_list, axis=0),
        source_files=np.concatenate(source_file_list, axis=0),
        clip_indices=np.concatenate(clip_index_list, axis=0).astype(np.int64),
        signal_names=list(signal_names),
    )


def build_cwt_dataset(
    clips: Sequence[SignalClip],
    signal_names: Sequence[str],
    dt: float,
    magnitude: bool = True,
    log1p_transform: bool = True,
    normalize_per_channel: bool = True,
    wavelet_name: str = "gmw",
) -> BuiltDataset:
    """Build a CWT dataset from clips.

    This function returns the flattened in-memory representation because it is
    convenient for training. Use ``build_grouped_cwt_dataset`` and
    ``save_grouped_dataset_npz`` when saving the dataset to disk.
    """
    grouped_data = build_grouped_cwt_dataset(
        clips=clips,
        signal_names=signal_names,
        dt=dt,
        magnitude=magnitude,
        log1p_transform=log1p_transform,
        normalize_per_channel=normalize_per_channel,
        wavelet_name=wavelet_name,
    )
    return flatten_grouped_cwt_dataset(grouped_data, signal_names)


def save_grouped_dataset_npz(
    grouped_data: GroupedDataset,
    output_path: Path | str,
) -> Path:
    """Save a grouped CWT dataset to one compressed NPZ file.

    Saved format:
        WFS_10_TS_4__scalogram
        WFS_10_TS_4__clip_indices
        WFS_10_TS_4__class_folder
        WFS_10_TS_4__source_file
        labels
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}

    for label, data in grouped_data.items():
        key = sanitize_npz_key(label)

        arrays[f"{key}__scalogram"] = np.asarray(
            data["scalogram"],
            dtype=np.float32,
        )
        arrays[f"{key}__clip_indices"] = np.asarray(
            data["clip_indices"],
            dtype=np.int64,
        )
        arrays[f"{key}__class_folder"] = np.asarray(
            data["class_folder"],
            dtype=str,
        )
        arrays[f"{key}__source_file"] = np.asarray(
            data["source_file"],
            dtype=str,
        )

    arrays["labels"] = np.asarray(sorted(grouped_data), dtype=str)

    np.savez_compressed(output_path, **arrays)
    return output_path


def _read_npz_object_dict(value: np.ndarray | object, key_name: str) -> dict:
    """Read object-dict entries used by earlier grouped NPZ datasets.

    Earlier format:
        npz["WFS_10_TS_4"].item() -> {
            "scalogram": ...,
            "clip_indices": ...
        }
    """
    if isinstance(value, dict):
        return value

    if isinstance(value, np.ndarray) and value.dtype == object:
        item = value.item()
        if isinstance(item, dict):
            return item

    try:
        item = value.item()  # type: ignore[attr-defined]
        if isinstance(item, dict):
            return item
    except Exception as exc:
        raise TypeError(
            f"Could not read grouped NPZ entry {key_name!r} as a dict."
        ) from exc

    raise TypeError(f"Grouped NPZ entry {key_name!r} is not a dict-like object.")


def _as_int_array(value: object, n: int, default: np.ndarray | None = None) -> np.ndarray:
    """Convert a value to an integer array of length n."""
    if value is None:
        if default is not None:
            arr = np.asarray(default, dtype=np.int64)
        else:
            arr = np.arange(n, dtype=np.int64)
    else:
        arr = np.asarray(value, dtype=np.int64)

    if arr.ndim == 0:
        arr = np.full(n, int(arr), dtype=np.int64)

    arr = arr.reshape(-1)

    if len(arr) != n:
        raise ValueError(f"Expected integer array length {n}, got {len(arr)}.")

    return arr.astype(np.int64)


def _as_string_array(value: object, n: int, default_value: str) -> np.ndarray:
    """Convert a value to a string array of length n."""
    if value is None:
        arr = np.asarray([default_value] * n, dtype=str)
    else:
        arr = np.asarray(value, dtype=str)

    if arr.ndim == 0:
        arr = np.asarray([str(arr.item())] * n, dtype=str)

    arr = arr.reshape(-1)

    if len(arr) != n:
        raise ValueError(f"Expected string array length {n}, got {len(arr)}.")

    return arr.astype(str)


def load_grouped_dataset_npz(
    path: Path | str,
    signal_names: Sequence[str] | None = None,
) -> BuiltDataset:
    """Load a grouped CWT dataset.

    Supported grouped formats:

    1. Current package format:
        WFS_10_TS_4__scalogram
        WFS_10_TS_4__clip_indices
        WFS_10_TS_4__class_folder
        WFS_10_TS_4__source_file

    2. Earlier object-dict format:
        WFS_10_TS_4:
            scalogram
            clip_indices
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    keys = list(data.files)

    if "X" in keys:
        raise ValueError(
            "This pipeline expects grouped NPZ format only. "
            "Detected a flat NPZ with key 'X'. Please provide a grouped NPZ "
            "with folder keys containing 'scalogram' and 'clip_indices'."
        )

    grouped: GroupedDataset = {}

    suffix = "__scalogram"

    if any(key.endswith(suffix) for key in keys):
        labels = sorted(key[: -len(suffix)] for key in keys if key.endswith(suffix))

        for label in labels:
            scalogram_key = f"{label}__scalogram"
            clip_key = f"{label}__clip_indices"
            class_key = f"{label}__class_folder"
            source_key = f"{label}__source_file"

            scalograms = np.asarray(data[scalogram_key], dtype=np.float32)

            if scalograms.ndim != 4:
                raise ValueError(
                    f"{label}: expected scalogram shape [N, C, H, W], "
                    f"got {scalograms.shape}."
                )

            n = scalograms.shape[0]

            clip_indices = _as_int_array(
                data[clip_key] if clip_key in keys else None,
                n=n,
            )
            class_folder = _as_string_array(
                data[class_key] if class_key in keys else None,
                n=n,
                default_value=label,
            )
            source_file = _as_string_array(
                data[source_key] if source_key in keys else None,
                n=n,
                default_value="",
            )

            grouped[label] = {
                "scalogram": scalograms,
                "clip_indices": clip_indices,
                "class_folder": class_folder,
                "source_file": source_file,
            }

    else:
        for key in keys:
            if key in {"labels", "signal_names"}:
                continue

            folder_data = _read_npz_object_dict(data[key], key)

            if "scalogram" not in folder_data:
                continue

            scalograms = np.asarray(folder_data["scalogram"], dtype=np.float32)

            if scalograms.ndim != 4:
                raise ValueError(
                    f"{key}: expected scalogram shape [N, C, H, W], "
                    f"got {scalograms.shape}."
                )

            n = scalograms.shape[0]

            clip_indices = _as_int_array(
                folder_data.get("clip_indices"),
                n=n,
            )
            class_folder = _as_string_array(
                folder_data.get("class_folder"),
                n=n,
                default_value=str(key),
            )
            source_file = _as_string_array(
                folder_data.get("source_file"),
                n=n,
                default_value="",
            )

            grouped[str(key)] = {
                "scalogram": scalograms,
                "clip_indices": clip_indices,
                "class_folder": class_folder,
                "source_file": source_file,
            }

    if not grouped:
        raise ValueError(f"No grouped scalogram data found in: {path}")

    if signal_names is None:
        first_scalogram = next(iter(grouped.values()))["scalogram"]
        inferred_signal_names = [
            f"channel_{idx}" for idx in range(first_scalogram.shape[1])
        ]
    else:
        inferred_signal_names = list(signal_names)

    return flatten_grouped_cwt_dataset(grouped, inferred_signal_names)


def load_cwt_dataset_npz(
    path: Path | str,
    dataset_source: str = "grouped_npz",
    signal_names: Sequence[str] | None = None,
) -> BuiltDataset:
    """Load a grouped CWT dataset from NPZ format."""
    source = str(dataset_source).strip().lower()

    if source != "grouped_npz":
        raise ValueError(
            "dataset_source must be 'grouped_npz'. "
        )

    return load_grouped_dataset_npz(path, signal_names=signal_names)