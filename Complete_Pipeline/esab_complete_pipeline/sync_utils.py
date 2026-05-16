from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d

from .utils import extract_display_label, is_hidden_path


@dataclass(frozen=True)
class RawSignalBundle:
    """Paths to raw signals belonging to one welding class folder."""

    class_id: int
    class_folder: str
    display_label: str
    current_txt: str
    voltage_txt: str
    wire_feed_speed_txt: str
    gas_flow_txt: str
    audio_mat: str
    bundle_name: str


DEFAULT_FILE_HINTS = {
    "current": ("current",),
    "voltage": ("voltage",),
    "wire_feed_speed": ("wire",),
    "gas_flow": ("gas", "flow"),
    "audio": ("audio",),
}


def pick_first_matching_file(
    class_folder: Path,
    extensions: Sequence[str],
    keywords: Sequence[str],
) -> Path | None:
    """Pick the first file whose name contains all required keywords."""
    extensions = tuple(ext.lower() for ext in extensions)
    keywords_lower = tuple(keyword.lower() for keyword in keywords)

    candidates: list[Path] = []
    for path in class_folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        name = path.name.lower()
        if all(keyword in name for keyword in keywords_lower):
            candidates.append(path)

    if not candidates:
        return None

    return sorted(candidates)[0]


def discover_raw_signal_bundles(
    root_dir: Path | str,
    label_regex: str,
    file_hints: Dict[str, Sequence[str]] | None = None,
) -> list[RawSignalBundle]:
    """Discover raw current, voltage, wire, gas, and audio files."""
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    hints = file_hints or DEFAULT_FILE_HINTS
    class_folders = [
        path
        for path in sorted(root_dir.iterdir())
        if path.is_dir() and not is_hidden_path(path)
    ]

    bundles: list[RawSignalBundle] = []
    for class_id, class_folder in enumerate(class_folders):
        display_label = extract_display_label(class_folder.name, label_regex)

        current_txt = pick_first_matching_file(
            class_folder, (".txt", ".csv"), hints["current"]
        )
        voltage_txt = pick_first_matching_file(
            class_folder, (".txt", ".csv"), hints["voltage"]
        )
        wire_txt = pick_first_matching_file(
            class_folder, (".txt", ".csv"), hints["wire_feed_speed"]
        )
        gas_txt = pick_first_matching_file(
            class_folder, (".txt", ".csv"), hints["gas_flow"]
        )
        audio_mat = pick_first_matching_file(class_folder, (".mat",), hints["audio"])

        if None in (current_txt, voltage_txt, wire_txt, gas_txt, audio_mat):
            continue

        bundles.append(
            RawSignalBundle(
                class_id=class_id,
                class_folder=class_folder.name,
                display_label=display_label,
                current_txt=str(current_txt),
                voltage_txt=str(voltage_txt),
                wire_feed_speed_txt=str(wire_txt),
                gas_flow_txt=str(gas_txt),
                audio_mat=str(audio_mat),
                bundle_name=display_label,
            )
        )

    if not bundles:
        raise FileNotFoundError(
            "No complete raw signal bundles were found. Each class folder should "
            "contain current, voltage, wire-feed-speed, gas-flow text files and "
            "one audio MAT file."
        )

    return bundles


def load_txt_two_columns(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column text/csv file as time and value arrays."""
    path = Path(path)
    data = np.loadtxt(path)

    if data.ndim == 1:
        raise ValueError(f"{path} is one-dimensional; expected at least two columns.")

    if data.shape[1] < 2:
        raise ValueError(f"{path} has fewer than two columns.")

    time = data[:, 0].astype(float)
    values = data[:, 1].astype(float)
    return time, values


def load_audio_from_mat(mat_path: str | Path, key: str) -> np.ndarray:
    """Load a one-dimensional audio vector from a MATLAB file."""
    mat_path = Path(mat_path)
    mat = loadmat(mat_path)

    if key not in mat:
        raise KeyError(f"MAT key '{key}' was not found in {mat_path}.")

    audio = np.asarray(mat[key]).squeeze().astype(float)
    if audio.ndim != 1:
        raise ValueError(f"Audio data must be one-dimensional, got shape={audio.shape}.")

    return audio


def find_txt_start_end_raw_loose(
    y: np.ndarray,
    eps: float,
    zero_run: int,
    active_window: int,
    min_active_count: int,
    near_window: int,
    near_min_active: int,
) -> tuple[int, int]:
    """Find active start/end indices in a text signal with loose window rules."""
    y = np.asarray(y, dtype=float).reshape(-1)

    active_mask = (np.abs(y) > eps).astype(np.int64)
    inactive_mask = np.abs(y) <= eps
    n = len(y)
    prefix = np.concatenate([[0], np.cumsum(active_mask)])

    def active_count(left: int, right: int) -> int:
        left = max(0, left)
        right = min(n, right)
        return int(prefix[right] - prefix[left])

    start_idx: int | None = None
    for i in range(zero_run, n - active_window + 1):
        before_all_inactive = np.all(inactive_mask[i - zero_run : i])
        after_active_count = active_count(i, i + active_window)
        after_near_count = active_count(i, i + near_window)

        if (
            before_all_inactive
            and after_active_count >= min_active_count
            and after_near_count >= near_min_active
        ):
            local_active = np.flatnonzero(active_mask[i : i + near_window])
            if len(local_active) > 0:
                start_idx = i + int(local_active[0])
                break

    if start_idx is None:
        raise ValueError("Could not find a valid text-signal start index.")

    end_idx: int | None = None
    for j in range(start_idx + active_window, n - zero_run + 1):
        after_all_inactive = np.all(inactive_mask[j : j + zero_run])
        before_active_count = active_count(j - active_window, j)
        before_near_count = active_count(j - near_window, j)

        if (
            after_all_inactive
            and before_active_count >= min_active_count
            and before_near_count >= near_min_active
        ):
            left = max(0, j - near_window)
            local_active = np.flatnonzero(active_mask[left:j])
            if len(local_active) > 0:
                end_idx = left + int(local_active[-1])
            else:
                end_idx = j - 1
            break

    if end_idx is None:
        raise ValueError("Could not find a valid text-signal end index.")

    return start_idx, end_idx


def find_current_style_txt_start_end_raw_loose(
    y: np.ndarray,
    eps: float = 5.0,
    zero_run: int = 5000,
    active_window: int = 50000,
    min_active_count: int = 5000,
    near_window: int = 10000,
    near_min_active: int = 500,
) -> tuple[int, int]:
    """Find start/end indices using current-style large-window parameters."""
    return find_txt_start_end_raw_loose(
        y=y,
        eps=eps,
        zero_run=zero_run,
        active_window=active_window,
        min_active_count=min_active_count,
        near_window=near_window,
        near_min_active=near_min_active,
    )


def find_audio_start_end_raw_loose(
    y: np.ndarray,
    eps: float = 0.0,
    zero_run: int = 50,
    active_window: int = 20000,
    min_nonzero_count: int = 2000,
    near_window: int = 1000,
    near_min_nonzero: int = 50,
) -> tuple[int, int]:
    """Find audio active start/end indices using zero-run logic."""
    y = np.asarray(y, dtype=float).reshape(-1)

    zero_mask = np.abs(y) <= eps
    nonzero_mask = (np.abs(y) > eps).astype(np.int64)
    n = len(y)
    prefix = np.concatenate([[0], np.cumsum(nonzero_mask)])

    def nonzero_count(left: int, right: int) -> int:
        left = max(0, left)
        right = min(n, right)
        return int(prefix[right] - prefix[left])

    start_idx: int | None = None
    for i in range(zero_run, n - active_window + 1):
        before_all_zero = np.all(zero_mask[i - zero_run : i])
        after_active_count = nonzero_count(i, i + active_window)
        after_near_count = nonzero_count(i, i + near_window)

        if (
            before_all_zero
            and after_active_count >= min_nonzero_count
            and after_near_count >= near_min_nonzero
        ):
            local_nonzero = np.flatnonzero(nonzero_mask[i : i + near_window])
            if len(local_nonzero) > 0:
                start_idx = i + int(local_nonzero[0])
                break

    if start_idx is None:
        raise ValueError("Could not find a valid audio start index.")

    end_idx: int | None = None
    for j in range(start_idx + active_window, n - zero_run + 1):
        after_all_zero = np.all(zero_mask[j : j + zero_run])
        before_active_count = nonzero_count(j - active_window, j)
        before_near_count = nonzero_count(j - near_window, j)

        if (
            after_all_zero
            and before_active_count >= min_nonzero_count
            and before_near_count >= near_min_nonzero
        ):
            left = max(0, j - near_window)
            local_nonzero = np.flatnonzero(nonzero_mask[left:j])
            if len(local_nonzero) > 0:
                end_idx = left + int(local_nonzero[-1])
            else:
                end_idx = j - 1
            break

    if end_idx is None:
        raise ValueError("Could not find a valid audio end index.")

    return start_idx, end_idx


def crop_by_index(y: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    """Crop a signal by inclusive start/end indices."""
    return np.asarray(y)[start_idx : end_idx + 1]


def crop_by_time(
    time: np.ndarray,
    y: np.ndarray,
    start_time: float,
    end_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a signal by a time interval."""
    mask = (time >= start_time) & (time <= end_time)
    if not np.any(mask):
        raise ValueError("Time-based crop produced an empty signal.")
    return time[mask], y[mask]


def trim_edges(arr: np.ndarray, trim_ratio: float) -> tuple[np.ndarray, int, int]:
    """Trim the same fraction from both sides of a 1D array."""
    if trim_ratio < 0:
        raise ValueError(f"trim_ratio must be non-negative, got {trim_ratio}.")

    n = len(arr)
    cut = int(round(n * trim_ratio))

    if cut == 0:
        return np.asarray(arr), 0, n

    if cut * 2 >= n:
        raise ValueError("trim_ratio is too large; no samples would remain.")

    return np.asarray(arr)[cut : n - cut], cut, n - cut


def resample_by_time(
    t_old: np.ndarray,
    y_old: np.ndarray,
    t_new: np.ndarray,
    kind: str = "linear",
) -> np.ndarray:
    """Resample a signal from an old time axis to a new time axis."""
    t_old = np.asarray(t_old, dtype=float).reshape(-1)
    y_old = np.asarray(y_old, dtype=float).reshape(-1)
    t_new = np.asarray(t_new, dtype=float).reshape(-1)

    if len(t_old) == 0:
        raise ValueError("Original time axis is empty.")
    if len(t_old) != len(y_old):
        raise ValueError("Time axis and signal length do not match.")
    if len(t_old) == 1:
        return np.full(len(t_new), y_old[0], dtype=float)

    keep = np.concatenate([[True], np.diff(t_old) > 0])
    t_old = t_old[keep]
    y_old = y_old[keep]

    if len(t_old) == 1:
        return np.full(len(t_new), y_old[0], dtype=float)

    actual_kind = "linear" if kind == "cubic" and len(t_old) < 4 else kind

    interpolator = interp1d(
        t_old,
        y_old,
        kind=actual_kind,
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    return interpolator(t_new)


def sync_and_resample_signals(
    current_txt: str | Path,
    voltage_txt: str | Path,
    wire_feed_speed_txt: str | Path,
    gas_flow_txt: str | Path,
    audio_mat: str | Path,
    audio_key: str,
    current_eps: float = 5.0,
    current_zero_run: int = 5000,
    current_active_window: int = 50000,
    current_min_active_count: int = 5000,
    current_near_window: int = 10000,
    current_near_min_active: int = 500,
    trim_ratio: float = 0.1,
    target_fs: float = 20000.0,
) -> dict:
    """Synchronize raw signals and resample them to a common frequency.

    Current defines the synchronized time range. Voltage, wire-feed-speed, and
    gas-flow signals are cropped by this current time range. Audio is cropped
    using its own active-region detector and then linearly mapped to the same
    current-defined time range.
    """
    if target_fs <= 0:
        raise ValueError(f"target_fs must be positive, got {target_fs}.")

    t_current, y_current = load_txt_two_columns(current_txt)
    t_voltage, y_voltage = load_txt_two_columns(voltage_txt)
    t_wfs, y_wfs = load_txt_two_columns(wire_feed_speed_txt)
    t_gas, y_gas = load_txt_two_columns(gas_flow_txt)
    y_audio = load_audio_from_mat(audio_mat, audio_key)

    cur_start_idx, cur_end_idx = find_current_style_txt_start_end_raw_loose(
        y_current,
        eps=current_eps,
        zero_run=current_zero_run,
        active_window=current_active_window,
        min_active_count=current_min_active_count,
        near_window=current_near_window,
        near_min_active=current_near_min_active,
    )

    aud_start_idx, aud_end_idx = find_audio_start_end_raw_loose(y_audio)

    start_time = float(t_current[cur_start_idx])
    end_time = float(t_current[cur_end_idx])

    if end_time <= start_time:
        raise ValueError("Invalid synchronization time range.")

    t_current_crop, y_current_crop = crop_by_time(
        t_current, y_current, start_time, end_time
    )
    t_voltage_crop, y_voltage_crop = crop_by_time(
        t_voltage, y_voltage, start_time, end_time
    )
    t_wfs_crop, y_wfs_crop = crop_by_time(t_wfs, y_wfs, start_time, end_time)
    t_gas_crop, y_gas_crop = crop_by_time(t_gas, y_gas, start_time, end_time)
    y_audio_crop = crop_by_index(y_audio, aud_start_idx, aud_end_idx)

    duration = end_time - start_time
    target_len = int(np.floor(duration * target_fs)) + 1
    if target_len < 2:
        raise ValueError("Target length is too short.")

    synced_time = start_time + np.arange(target_len, dtype=float) / float(target_fs)
    synced_time = synced_time[synced_time <= end_time]

    if len(synced_time) < 2:
        raise ValueError("Generated synchronized time axis is too short.")

    current_sync = resample_by_time(t_current_crop, y_current_crop, synced_time)
    voltage_sync = resample_by_time(t_voltage_crop, y_voltage_crop, synced_time)
    wfs_sync = resample_by_time(t_wfs_crop, y_wfs_crop, synced_time)
    gas_sync = resample_by_time(t_gas_crop, y_gas_crop, synced_time)

    t_audio_crop = np.linspace(
        start_time,
        end_time,
        len(y_audio_crop),
        endpoint=True,
        dtype=float,
    )
    audio_sync = resample_by_time(t_audio_crop, y_audio_crop, synced_time)

    synced_time_trim, trim_start_idx, trim_end_idx = trim_edges(synced_time, trim_ratio)
    current_trim, _, _ = trim_edges(current_sync, trim_ratio)
    voltage_trim, _, _ = trim_edges(voltage_sync, trim_ratio)
    wfs_trim, _, _ = trim_edges(wfs_sync, trim_ratio)
    gas_trim, _, _ = trim_edges(gas_sync, trim_ratio)
    audio_trim, _, _ = trim_edges(audio_sync, trim_ratio)

    return {
        "synced_time": synced_time_trim.astype(np.float64),
        "current": current_trim.astype(np.float64),
        "voltage": voltage_trim.astype(np.float64),
        "wire_feed_speed": wfs_trim.astype(np.float64),
        "gas_flow": gas_trim.astype(np.float64),
        "audio": audio_trim.astype(np.float64),
        "meta": {
            "current_active_index_range": (int(cur_start_idx), int(cur_end_idx)),
            "audio_active_index_range": (int(aud_start_idx), int(aud_end_idx)),
            "sync_time_range_before_trim": (float(start_time), float(end_time)),
            "target_fs_hz": float(target_fs),
            "length_before_trim": int(len(synced_time)),
            "length_after_trim": int(len(synced_time_trim)),
            "trim_ratio_each_side": float(trim_ratio),
            "trim_index_range_on_synced_signal": (
                int(trim_start_idx),
                int(trim_end_idx - 1),
            ),
            "original_lengths_after_crop": {
                "current": int(len(y_current_crop)),
                "voltage": int(len(y_voltage_crop)),
                "wire_feed_speed": int(len(y_wfs_crop)),
                "gas_flow": int(len(y_gas_crop)),
                "audio": int(len(y_audio_crop)),
            },
            "current_detection_params": {
                "eps": float(current_eps),
                "zero_run": int(current_zero_run),
                "active_window": int(current_active_window),
                "min_active_count": int(current_min_active_count),
                "near_window": int(current_near_window),
                "near_min_active": int(current_near_min_active),
            },
        },
    }


# Backward-compatible alias for older scripts.
sync_and_upsample_signals = sync_and_resample_signals


def build_synced_cache(
    bundles: Sequence[RawSignalBundle],
    output_root: Path | str,
    audio_key: str,
    current_eps: float,
    current_zero_run: int,
    current_active_window: int,
    current_min_active_count: int,
    current_near_window: int,
    current_near_min_active: int,
    trim_ratio: float,
    target_fs: float = 20000.0,
) -> Path:
    """Create synced NPZ files for all discovered raw signal bundles."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for bundle in bundles:
        class_dir = output_root / bundle.class_folder
        class_dir.mkdir(parents=True, exist_ok=True)
        output_path = class_dir / f"{bundle.display_label}_synced.npz"

        synced = sync_and_resample_signals(
            current_txt=bundle.current_txt,
            voltage_txt=bundle.voltage_txt,
            wire_feed_speed_txt=bundle.wire_feed_speed_txt,
            gas_flow_txt=bundle.gas_flow_txt,
            audio_mat=bundle.audio_mat,
            audio_key=audio_key,
            current_eps=current_eps,
            current_zero_run=current_zero_run,
            current_active_window=current_active_window,
            current_min_active_count=current_min_active_count,
            current_near_window=current_near_window,
            current_near_min_active=current_near_min_active,
            trim_ratio=trim_ratio,
            target_fs=target_fs,
        )

        np.savez_compressed(output_path, **synced)

    return output_root
