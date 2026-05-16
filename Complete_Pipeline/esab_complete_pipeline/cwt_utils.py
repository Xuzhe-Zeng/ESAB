from __future__ import annotations

import numpy as np



def _load_ssqueezepy():
    """Import ssqueezepy only when CWT generation is requested."""
    try:
        from ssqueezepy import Wavelet, cwt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ssqueezepy is required for CWT generation. Install it with "
            "`pip install ssqueezepy`."
        ) from exc
    return Wavelet, cwt


def simple_cwt(
    x: np.ndarray,
    dt: float,
    magnitude: bool = True,
    log1p_transform: bool = True,
    normalize_per_channel: bool = True,
    wavelet_name: str = "gmw",
) -> np.ndarray:
    """Convert a 1D signal into a CWT scalogram."""
    signal = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(signal) < 4:
        raise ValueError("Input signal is too short for CWT.")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}.")

    fs = 1.0 / dt

    signal = signal - signal.mean()
    signal_std = signal.std()
    if signal_std > 1e-8:
        signal = signal / signal_std

    Wavelet, cwt = _load_ssqueezepy()
    wavelet = Wavelet(wavelet_name)
    coeffs, _ = cwt(signal, wavelet=wavelet, fs=fs)

    if magnitude:
        cwt_map = np.abs(coeffs)
    else:
        cwt_map = np.real(coeffs)

    cwt_map = cwt_map.astype(np.float32)

    if log1p_transform:
        cwt_map = np.log1p(cwt_map)

    if normalize_per_channel:
        min_value = float(cwt_map.min())
        max_value = float(cwt_map.max())
        if max_value > min_value:
            cwt_map = (cwt_map - min_value) / (max_value - min_value)

    return cwt_map.astype(np.float32)
