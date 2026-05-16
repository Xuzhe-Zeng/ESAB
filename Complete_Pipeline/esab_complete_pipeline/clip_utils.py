from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .io_utils import SyncedRecord


@dataclass(frozen=True)
class SignalClip:
    """One fixed-length time clip extracted from a synchronized record."""

    class_id: int
    class_folder: str
    display_label: str
    source_file: str
    clip_index: int
    time: np.ndarray
    signals: Dict[str, np.ndarray]


def infer_expected_clip_length(
    time: np.ndarray,
    start_time: float,
    clip_seconds: float,
) -> int:
    """Infer the sample count of one clip from the time axis."""
    first_end = start_time + clip_seconds
    left = int(np.searchsorted(time, start_time, side="left"))
    right = int(np.searchsorted(time, first_end, side="left"))
    return right - left


def clip_record_by_time(
    record: SyncedRecord,
    clip_seconds: float,
    clip_interval_seconds: float,
    drop_last_incomplete: bool = True,
    expected_len: int | None = None,
) -> tuple[List[SignalClip], int | None]:
    """Split one synchronized record into fixed-length clips.

    ``clip_seconds`` controls the window length. ``clip_interval_seconds``
    controls the spacing between adjacent clip start times, so overlapping
    clips are supported.
    """
    if clip_seconds <= 0:
        raise ValueError(f"clip_seconds must be positive, got {clip_seconds}.")

    if clip_interval_seconds <= 0:
        raise ValueError(
            "clip_interval_seconds must be positive, "
            f"got {clip_interval_seconds}."
        )

    time = np.asarray(record.time).reshape(-1)
    if len(time) < 2:
        return [], expected_len

    start_time = float(time[0])
    end_time = float(time[-1])

    if expected_len is None:
        expected_len = infer_expected_clip_length(time, start_time, clip_seconds)
        if expected_len < 2:
            raise ValueError(
                "Unable to determine a valid global clip length from "
                f"the first record: {record.source_file}."
            )

    clips: List[SignalClip] = []
    clip_idx = 0
    clip_start = start_time
    eps = 1e-9

    while True:
        clip_end = clip_start + clip_seconds

        if drop_last_incomplete and clip_end > end_time + eps:
            break

        if (not drop_last_incomplete) and clip_start >= end_time:
            break

        left = int(np.searchsorted(time, clip_start, side="left"))
        right = left + expected_len

        if right > len(time):
            break

        clip_time = time[left:right].copy()
        clip_signals = {
            name: signal[left:right].copy()
            for name, signal in record.signals.items()
        }

        clips.append(
            SignalClip(
                class_id=record.class_id,
                class_folder=record.class_folder,
                display_label=record.display_label,
                source_file=record.source_file,
                clip_index=clip_idx,
                time=clip_time,
                signals=clip_signals,
            )
        )

        clip_idx += 1
        clip_start += clip_interval_seconds

    return clips, expected_len


def clip_all_records(
    records: List[SyncedRecord],
    clip_seconds: float,
    clip_interval_seconds: float,
    drop_last_incomplete: bool = True,
) -> List[SignalClip]:
    """Clip all synchronized records using one global sample count."""
    all_clips: List[SignalClip] = []
    global_expected_len: int | None = None

    for record in records:
        record_clips, global_expected_len = clip_record_by_time(
            record=record,
            clip_seconds=clip_seconds,
            clip_interval_seconds=clip_interval_seconds,
            drop_last_incomplete=drop_last_incomplete,
            expected_len=global_expected_len,
        )
        all_clips.extend(record_clips)

    if not all_clips:
        return all_clips

    all_clips = [
        clip for clip in all_clips if len(clip.time) == global_expected_len
    ]

    lengths = sorted({len(clip.time) for clip in all_clips})
    print(f"Unique clip lengths after clipping: {lengths}")

    return all_clips
