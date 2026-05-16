from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds when PyTorch is available."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_display_label(folder_name: str, pattern: str) -> str:
    """Extract a display label from a folder name using a regular expression."""
    match = re.search(pattern, folder_name)
    if match:
        return match.group(0)
    return folder_name


def is_hidden_path(path: Path) -> bool:
    """Return True if any path component starts with a dot."""
    return any(part.startswith(".") for part in path.parts)


def sanitize_npz_key(name: str) -> str:
    """Create a stable key that can be safely used inside an NPZ archive."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
