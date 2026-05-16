from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

import numpy as np
import pandas as pd

from .dataset_builder import BuiltDataset

TargetMode = Literal["WIDTH", "WFS_TS"]


@dataclass(frozen=True)
class SupervisedTargets:
    """Target arrays and metadata for supervised learning."""

    mode: str
    task_type: str
    y: np.ndarray
    sample_indices: np.ndarray
    target_names: list[str]
    class_names: list[str] | None = None


def normalize_target_mode(mode: str) -> str:
    """Normalize and validate a supervised target mode."""
    normalized = str(mode).strip().upper()
    valid_modes = {"WIDTH", "WFS_TS"}
    if normalized not in valid_modes:
        raise ValueError(f"TARGET_MODE must be one of {sorted(valid_modes)}, got {mode!r}.")
    return normalized


def normalize_class_folder_name(value: object) -> str:
    """Normalize class folder labels such as WFS_10_TS_4."""
    text = str(value).strip()
    match = re.search(r"WFS_(\d+)_TS_(\d+)", text)
    if match:
        return f"WFS_{match.group(1)}_TS_{match.group(2)}"
    return text


def display_wfs_ts_label(value: object) -> str:
    """Convert WFS_10_TS_4 into a compact display label."""
    text = normalize_class_folder_name(value)
    match = re.search(r"WFS_(\d+)_TS_(\d+)", text)
    if match:
        return f"WFS={match.group(1)}, TS={match.group(2)}"
    return text


def safe_int(value: object) -> int:
    """Convert integer-like spreadsheet values into int."""
    if pd.isna(value):
        raise ValueError("Cannot convert NaN to int.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError("Cannot convert an empty string to int.")
    return int(float(text))


def _find_column_case_insensitive(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def _build_width_lookup(
    excel_path: Path | str,
    class_folder_column: str,
    clip_index_column: str,
    width_column: str,
) -> dict[tuple[str, int], float]:
    """Read width labels from an Excel/CSV file and build a lookup table."""
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Width label file does not exist: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(
            "WIDTH_EXCEL_PATH must point to an .xlsx, .xls, or .csv file. "
            f"Got: {path}"
        )

    columns = [str(column) for column in frame.columns]
    class_col = _find_column_case_insensitive(columns, [class_folder_column, "class_folder", "folder_name"])
    index_col = _find_column_case_insensitive(
        columns,
        [clip_index_column, "clip_indices", "clip_index", "indices", "index", "image_file"],
    )
    target_col = _find_column_case_insensitive(columns, [width_column, "width"])

    missing = []
    if class_col is None:
        missing.append(class_folder_column)
    if index_col is None:
        missing.append(clip_index_column)
    if target_col is None:
        missing.append(width_column)
    if missing:
        raise KeyError(
            "Could not find required width-label column(s): "
            f"{missing}. Available columns: {columns}"
        )

    lookup: dict[tuple[str, int], float] = {}
    for row in frame.itertuples(index=False):
        row_dict = dict(zip(frame.columns, row))
        class_folder = normalize_class_folder_name(row_dict[class_col])
        clip_index = safe_int(row_dict[index_col])
        width = row_dict[target_col]
        if pd.isna(width):
            continue
        lookup[(class_folder, clip_index)] = float(width)

    if not lookup:
        raise ValueError(f"No valid width labels were found in: {path}")

    return lookup


def build_width_targets(
    dataset: BuiltDataset,
    excel_path: Path | str,
    class_folder_column: str = "class_folder",
    clip_index_column: str = "indices",
    width_column: str = "width",
    missing_policy: str = "drop",
) -> SupervisedTargets:
    """Create regression targets by matching class_folder and clip_index to width labels."""
    policy = str(missing_policy).strip().lower()
    if policy not in {"drop", "raise"}:
        raise ValueError("missing_policy must be either 'drop' or 'raise'.")

    lookup = _build_width_lookup(
        excel_path=excel_path,
        class_folder_column=class_folder_column,
        clip_index_column=clip_index_column,
        width_column=width_column,
    )

    kept_indices: list[int] = []
    values: list[float] = []
    missing_keys: list[tuple[str, int]] = []

    for idx, (folder, clip_index) in enumerate(zip(dataset.class_folders, dataset.clip_indices)):
        key = (normalize_class_folder_name(folder), int(clip_index))
        if key not in lookup:
            missing_keys.append(key)
            continue
        kept_indices.append(idx)
        values.append(lookup[key])

    if missing_keys and policy == "raise":
        preview = missing_keys[:10]
        raise KeyError(
            f"Missing {len(missing_keys)} width label(s). First missing keys: {preview}"
        )

    if not kept_indices:
        raise ValueError(
            "No samples could be matched to width labels. Check class_folder and index columns."
        )

    if missing_keys:
        print(
            f"[WARN] Dropped {len(missing_keys)} sample(s) without matching width labels."
        )

    return SupervisedTargets(
        mode="WIDTH",
        task_type="regression",
        y=np.asarray(values, dtype=np.float32).reshape(-1, 1),
        sample_indices=np.asarray(kept_indices, dtype=np.int64),
        target_names=["width"],
        class_names=None,
    )


def build_wfs_ts_targets(dataset: BuiltDataset) -> SupervisedTargets:
    """Create classification targets from class_folder labels such as WFS_10_TS_4."""
    labels = np.asarray(
        [normalize_class_folder_name(value) for value in dataset.class_folders],
        dtype=object,
    )
    class_names = sorted(set(labels.tolist()))
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    y = np.asarray([class_to_id[label] for label in labels], dtype=np.int64)

    return SupervisedTargets(
        mode="WFS_TS",
        task_type="classification",
        y=y,
        sample_indices=np.arange(len(dataset.X), dtype=np.int64),
        target_names=["class"],
        class_names=class_names,
    )


def build_supervised_targets(
    dataset: BuiltDataset,
    target_mode: str,
    *,
    width_excel_path: Path | str | None = None,
    width_class_folder_column: str = "class_folder",
    width_clip_index_column: str = "indices",
    width_target_column: str = "width",
    width_missing_policy: str = "drop",
) -> SupervisedTargets:
    """Build supervised targets from the requested target mode."""
    mode = normalize_target_mode(target_mode)

    if mode == "WIDTH":
        if width_excel_path is None:
            raise ValueError("WIDTH mode requires WIDTH_EXCEL_PATH.")
        return build_width_targets(
            dataset=dataset,
            excel_path=width_excel_path,
            class_folder_column=width_class_folder_column,
            clip_index_column=width_clip_index_column,
            width_column=width_target_column,
            missing_policy=width_missing_policy,
        )

    if mode == "WFS_TS":
        return build_wfs_ts_targets(dataset)

    raise AssertionError(f"Unhandled target mode: {mode}")
