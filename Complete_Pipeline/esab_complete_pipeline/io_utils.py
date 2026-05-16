from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .utils import extract_display_label, is_hidden_path


@dataclass(frozen=True)
class SyncedRecord:
    """One synchronized welding record loaded from an NPZ file."""

    class_id: int
    class_folder: str
    display_label: str
    source_file: str
    time: np.ndarray
    signals: Dict[str, np.ndarray]


REQUIRED_TIME_KEY_CANDIDATES = ("synced_time", "time", "t")


def find_time_key(npz_keys: Sequence[str]) -> str:
    """Return the first supported time key present in an NPZ file."""
    for key in REQUIRED_TIME_KEY_CANDIDATES:
        if key in npz_keys:
            return key
    raise KeyError(
        "No time key found. Expected one of "
        f"{list(REQUIRED_TIME_KEY_CANDIDATES)}, got {list(npz_keys)}."
    )


def load_synced_npz(
    path: Path | str,
    signals_to_use: Sequence[str],
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load synchronized time and selected signals from an NPZ file."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        keys = list(data.keys())
        time_key = find_time_key(keys)
        time = np.asarray(data[time_key], dtype=np.float64).reshape(-1)

        signals: Dict[str, np.ndarray] = {}
        for signal_name in signals_to_use:
            if signal_name not in keys:
                raise KeyError(
                    f"Signal '{signal_name}' was not found in {path}. "
                    f"Available keys: {keys}"
                )

            signal = np.asarray(data[signal_name], dtype=np.float32).reshape(-1)
            if len(signal) != len(time):
                raise ValueError(
                    f"Length mismatch in {path}: len(time)={len(time)}, "
                    f"len({signal_name})={len(signal)}."
                )
            signals[signal_name] = signal

    return time.astype(np.float64), signals


def discover_synced_records(
    root_dir: Path | str,
    sync_file_patterns: Sequence[str],
    signals_to_use: Sequence[str],
    label_regex: str,
    *,
    strict: bool = False,
) -> List[SyncedRecord]:
    """Discover valid synced NPZ records under class folders."""
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    class_folders = [
        path
        for path in sorted(root_dir.iterdir())
        if path.is_dir() and not is_hidden_path(path)
    ]

    records: List[SyncedRecord] = []
    skipped: list[tuple[Path, str]] = []

    for class_id, class_folder in enumerate(class_folders):
        display_label = extract_display_label(class_folder.name, label_regex)
        candidates: list[Path] = []
        seen: set[Path] = set()

        for pattern in sync_file_patterns:
            for path in sorted(class_folder.rglob(pattern)):
                if path.suffix.lower() != ".npz" or path in seen:
                    continue
                seen.add(path)
                candidates.append(path)

        for path in candidates:
            try:
                time, signals = load_synced_npz(path, signals_to_use)
            except Exception as exc:
                if strict:
                    raise
                skipped.append((path, str(exc)))
                continue

            records.append(
                SyncedRecord(
                    class_id=class_id,
                    class_folder=class_folder.name,
                    display_label=display_label,
                    source_file=str(path),
                    time=time,
                    signals=signals,
                )
            )

    if not records:
        details = "\n".join(f"- {path}: {reason}" for path, reason in skipped[:10])
        message = (
            "No valid synced NPZ files were found. The loader expects files with "
            "a time key ('synced_time', 'time', or 't') and all selected signals."
        )
        if details:
            message += f"\nFirst skipped files:\n{details}"
        raise FileNotFoundError(message)

    if skipped:
        print(f"Skipped {len(skipped)} invalid or unrelated NPZ file(s).")

    return records
