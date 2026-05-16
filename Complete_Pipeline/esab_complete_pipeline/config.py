"""Default configuration for the ESAB complete pipeline.

The same package can:
1. synchronize raw ESAB signals,
2. build CWT scalogram datasets,
3. train a VAE for latent-space analysis,
4. train supervised models for WFS_TS classification and WIDTH regression.

For a local project, copy ``examples/local_config_example.py`` and edit paths there.
Run with:
    python -m esab_complete_pipeline.pipeline --config examples.local_config_example
"""

from pathlib import Path

# Reproducibility
RANDOM_SEED = 42

# ============================================================
# Data source
# ============================================================
# Options:
#   "build"       : build grouped CWT dataset from raw/synced signals.
#   "grouped_npz" : load an existing grouped CWT dataset.
DATASET_SOURCE = "grouped_npz"
CWT_DATASET_PATH = Path("outputs/cwt_dataset_grouped.npz")
SAVE_GROUPED_DATASET_NPZ = True

# ============================================================
# Paths for building from raw/synced signals
# ============================================================
# ROOT_DIR has two possible meanings:
# 1. If USE_SYNC_STAGE is True, ROOT_DIR should contain raw class folders.
# 2. If USE_SYNC_STAGE is False, ROOT_DIR should contain synced class folders.
ROOT_DIR = Path("D:\ESAB\Experimental Data\data")
SYNC_CACHE_DIR = Path("sync_data")
OUTPUT_DIR = Path("outputs")
SYNC_FILE_PATTERNS = ["*_synced.npz"]
# Whether to build synced NPZ files from raw txt/mat signals first.
USE_SYNC_STAGE = False

# Label extraction from folder names such as 4_14_2026_WFS_10_TS_4.
DISPLAY_LABEL_REGEX = r"WFS_(\d+)_TS_(\d+)"

# Raw audio MATLAB key.
AUDIO_KEY = "RODE_M3_recording"

# Current-based active-region detection.
CURRENT_EPS = 5.0
CURRENT_ZERO_RUN = 5000
CURRENT_ACTIVE_WINDOW = 50000
CURRENT_MIN_ACTIVE_COUNT = 5000
CURRENT_NEAR_WINDOW = 10000
CURRENT_NEAR_MIN_ACTIVE = 500

# Common sampling rate after synchronization.
TARGET_FS = 5000.0

# Fraction removed from each edge after synchronization.
SYNC_TRIM_RATIO = 0.0

# Signals to use as CWT input channels.
# Supported names: "current", "voltage", "wire_feed_speed", "gas_flow", "audio".
SIGNALS_TO_USE = [
    "current",
    "voltage",
]

# Clip settings.
CLIP_SECONDS = 0.2
CLIP_INTERVAL_SECONDS = 1.0 / 8.99
DROP_LAST_INCOMPLETE_CLIP = True

# CWT settings.
CWT_MAGNITUDE = True
CWT_LOG1P = True
CWT_NORMALIZE_PER_CHANNEL = True
CWT_WAVELET_NAME = "gmw"


# ============================================================
# VAE latent-space analysis
# ============================================================
RUN_VAE = True
LATENT_DIM = 4
BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
BETA = 0.01

# t-SNE settings.
RUN_TSNE = True
TSNE_PERPLEXITY = 10
TSNE_RANDOM_STATE = 42
TSNE_N_ITER = 1000
VAE_OUTPUT_SUBDIR = "vae_results"


# ============================================================
# Supervised prediction
# ============================================================
RUN_SUPERVISED = True
TARGET_MODE_WIDTH = "WIDTH"
TARGET_MODE_WFS_TS = "WFS_TS"

# Run one or both targets. The complete setting is:
#   ["WFS_TS", "WIDTH"]
# If WIDTH labels are not available yet, use ["WFS_TS"] first.
SUPERVISED_TARGET_MODES = [TARGET_MODE_WFS_TS, TARGET_MODE_WIDTH]

# Required only for WIDTH regression.
# The file may be .xlsx, .xls, or .csv and should contain columns like:
# class_folder, indices, width
WIDTH_EXCEL_PATH = Path("D:\ESAB\Experimental Data\data\image_width_results.xlsx")
WIDTH_CLASS_FOLDER_COLUMN = "class_folder"
WIDTH_CLIP_INDEX_COLUMN = "indices"
WIDTH_TARGET_COLUMN = "width"
WIDTH_MISSING_POLICY = "drop"  # "drop" or "raise"

# Select input channels by target.
# None means all CWT channels. [1] means only the second channel: scalogram[:, 1, :, :].
# For WIDTH, this matches the earlier width-regression script that used one selected channel.
SUPERVISED_CHANNELS_BY_TARGET = {
    TARGET_MODE_WFS_TS: None,
    TARGET_MODE_WIDTH: [1],
}

# Model options:
#   "cnn"       : lightweight CNN, good default for width regression.
#   "resnet18"  : torchvision ResNet-18.
#   "vit_b_16"  : torchvision ViT-B/16, usually with image resizing.
#   "patch_vit" : small non-square ViT that can use original scalogram size.
SUPERVISED_MODEL_NAME = "resnet18"
SUPERVISED_MODEL_BY_TARGET = {
    TARGET_MODE_WFS_TS: "resnet18",
    TARGET_MODE_WIDTH: "cnn",
}
SUPERVISED_PRETRAINED = False

# If > 0, supervised inputs are resized to image_size x image_size.
# Set to 0 to keep the original CWT size. For patch_vit, 0 is often preferred.
SUPERVISED_IMAGE_SIZE = 224
SUPERVISED_IMAGE_SIZE_BY_TARGET = {
    TARGET_MODE_WFS_TS: 224,
    TARGET_MODE_WIDTH: 0,
}

SUPERVISED_BATCH_SIZE = 32
SUPERVISED_NUM_EPOCHS = 50
SUPERVISED_LEARNING_RATE = 1e-4
SUPERVISED_WEIGHT_DECAY = 1e-4
SUPERVISED_TEST_SIZE = 0.2
SUPERVISED_VALIDATION_SIZE = 0.2
SUPERVISED_NUM_WORKERS = 0
SUPERVISED_EARLY_STOPPING_PATIENCE = 15

# patch_vit parameters, used only when SUPERVISED_MODEL_NAME="patch_vit".
PATCH_VIT_PATCH_SIZE = (79, 20)
PATCH_VIT_EMBED_DIM = 192
PATCH_VIT_DEPTH = 4
PATCH_VIT_NUM_HEADS = 6
PATCH_VIT_MLP_RATIO = 4.0
PATCH_VIT_DROPOUT = 0.1

# Save options.
SAVE_GROUPED_DATASET_NPZ = True
SAVE_FLAT_DATASET_NPZ = True
SAVE_MODEL_WEIGHTS = True
SAVE_LATENTS_CSV = True
SAVE_TSNE_FIG = True
