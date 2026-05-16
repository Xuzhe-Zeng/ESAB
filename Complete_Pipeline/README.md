# ESAB Complete Pipeline

This repository contains a Python pipeline for ESAB robotic welding signal analysis.

The pipeline supports:

```text
raw or synced signals
    -> CWT scalogram dataset
    -> VAE latent-space analysis
    -> WFS_TS classification
    -> WIDTH regression
```

The two supervised prediction targets are:

- `WFS_TS`: classification target, such as `WFS_10_TS_4`
- `WIDTH`: regression target from an Excel or CSV table

---

## Project Structure

```text
esab-complete-pipeline/
├── esab_complete_pipeline/
│   ├── config.py
│   ├── main.py
│   ├── dataset_builder.py
│   ├── sync_utils.py
│   ├── io_utils.py
│   ├── clip_utils.py
│   ├── cwt_utils.py
│   ├── train_vae.py
│   ├── vae_model.py
│   ├── visualize.py
│   ├── supervised_targets.py
│   ├── supervised_models.py
│   └── supervised_train.py
├── examples/
│   └── local_config_example.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Installation

From the project root folder:

```bash
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

---

## How to Run

Run the pipeline from the project root folder.

Correct:

```bash
python -m esab_complete_pipeline.main
```

With a custom config:

```bash
python -m esab_complete_pipeline.main --config examples.local_config_example
```

Do not run this inside the `esab_complete_pipeline/` folder:

```bash
python main.py
```

Running `python main.py` inside the package folder can cause relative import errors.

---

## Data Source

The pipeline supports two data sources.

### 1. Build CWT dataset from synced signals

Use this mode if you want to build the CWT scalogram dataset from synced signal `.npz` files.

```python
DATASET_SOURCE = "build"
USE_SYNC_STAGE = False
ROOT_DIR = Path(r"D:/ESAB/Experimental Data/sync_data_5kHz")
```

Expected synced data structure:

```text
sync_data_5kHz/
├── 4_14_2026_WFS_10_TS_4/
│   └── WFS_10_TS_4_synced.npz
├── 4_14_2026_WFS_12_TS_4/
│   └── WFS_12_TS_4_synced.npz
└── ...
```

Each synced `.npz` file should contain a time key:

```text
synced_time
```

or:

```text
time
t
```

and the selected signal arrays, for example:

```text
current
voltage
audio
wire_feed_speed
gas_flow
```

---

### 2. Load an existing grouped CWT dataset

Use this mode if you already have a grouped CWT `.npz` file.

```python
DATASET_SOURCE = "grouped_npz"
CWT_DATASET_PATH = Path(r"D:/ESAB/Experimental Data/output/cwt_dataset_grouped.npz")
```

Expected grouped format:

```text
WFS_10_TS_4
    ├── scalogram
    └── clip_indices

WFS_12_TS_4
    ├── scalogram
    └── clip_indices
```

This pipeline uses the grouped dataset format. It no longer uses the flat `X` dataset format.

---

## Config Example

Most settings are controlled in:

```text
esab_complete_pipeline/config.py
```

Example:

```python
from pathlib import Path

# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(r"D:/ESAB/Experimental Data/sync_data_5kHz")
OUTPUT_DIR = Path("outputs")

# ============================================================
# Data source
# ============================================================

# Options:
#   "build"       : build grouped CWT dataset from synced signals
#   "grouped_npz" : load an existing grouped CWT dataset
DATASET_SOURCE = "build"

CWT_DATASET_PATH = Path(r"D:/ESAB/Experimental Data/output/cwt_dataset_grouped.npz")

# ============================================================
# Signal settings
# ============================================================

SIGNALS_TO_USE = [
    "current",
    "voltage",
]

SYNC_FILE_PATTERNS = ["*_synced.npz"]

# ============================================================
# CWT dataset save option
# ============================================================

SAVE_GROUPED_DATASET_NPZ = True
```

---

## Choose Prediction Target

Use this switch in `config.py`:

```python
PREDICT_TARGET = "BOTH"
```

Available options:

```text
"WFS_TS" : run only WFS_TS classification
"WIDTH"  : run only WIDTH regression
"BOTH"   : run both tasks
```

Recommended config:

```python
TARGET_MODE_WFS_TS = "WFS_TS"
TARGET_MODE_WIDTH = "WIDTH"

PREDICT_TARGET = "BOTH"

if PREDICT_TARGET == TARGET_MODE_WFS_TS:
    SUPERVISED_TARGET_MODES = [TARGET_MODE_WFS_TS]
elif PREDICT_TARGET == TARGET_MODE_WIDTH:
    SUPERVISED_TARGET_MODES = [TARGET_MODE_WIDTH]
elif PREDICT_TARGET == "BOTH":
    SUPERVISED_TARGET_MODES = [TARGET_MODE_WFS_TS, TARGET_MODE_WIDTH]
else:
    raise ValueError("PREDICT_TARGET must be 'WFS_TS', 'WIDTH', or 'BOTH'.")
```

---

## WIDTH Excel File

The `WIDTH` regression task needs an Excel or CSV file with width labels.

Recommended columns:

```text
class_folder
indices
width
```

Example:

| class_folder | indices | width |
|---|---:|---:|
| WFS_10_TS_4 | 0 | 143 |
| WFS_10_TS_4 | 1 | 146 |
| WFS_12_TS_4 | 0 | 158 |

Config:

```python
WIDTH_EXCEL_PATH = Path(r"D:/ESAB/Experimental Data/data/image_width_results.xlsx")

WIDTH_CLASS_FOLDER_COLUMN = "class_folder"
WIDTH_CLIP_INDEX_COLUMN = "indices"
WIDTH_TARGET_COLUMN = "width"

WIDTH_MISSING_POLICY = "drop"
```

If your Excel file uses `image_file` instead of `indices`, use:

```python
WIDTH_CLIP_INDEX_COLUMN = "image_file"
```

`WIDTH_MISSING_POLICY` controls what happens when a sample cannot find a matching width value:

```python
WIDTH_MISSING_POLICY = "drop"
```

This skips samples without width labels.

```python
WIDTH_MISSING_POLICY = "raise"
```

This stops the program if any width label is missing.

---

## Channel Selection

Different targets can use different CWT input channels.

```python
SUPERVISED_CHANNELS_BY_TARGET = {
    "WFS_TS": None,
    "WIDTH": [1],
}
```

Meaning:

```text
None : use all CWT channels
[1]  : use only the second CWT channel, scalogram[:, 1, :, :]
```

Example:

If:

```python
SIGNALS_TO_USE = [
    "current",
    "voltage",
]
```

then:

```text
channel 0 = current
channel 1 = voltage
```

So:

```python
"WIDTH": [1]
```

means the width regression model uses only the voltage CWT channel.

If you want `WIDTH` to use all channels, use:

```python
SUPERVISED_CHANNELS_BY_TARGET = {
    "WFS_TS": None,
    "WIDTH": None,
}
```

---

## VAE

To run VAE:

```python
RUN_VAE = True
```

To skip VAE:

```python
RUN_VAE = False
```

VAE outputs are saved in:

```text
outputs/vae_results/
```

Typical outputs:

```text
vae_model.pt
vae_training_history.csv
vae_training_losses.png
vae_latents.csv
vae_latent_tsne.png
vae_latent_2d.png
```

### t-SNE vs 2D Latent Plot

The 2D latent plot directly shows the first two VAE latent dimensions.

The t-SNE plot uses all latent dimensions and maps them into 2D for visualization.

In simple terms:

```text
2D latent plot = direct plot of latent dimension 1 and 2
t-SNE plot     = nonlinear 2D visualization using all latent dimensions
```

If `LATENT_DIM = 2`, the 2D latent plot is the original latent space.

If `LATENT_DIM > 2`, the 2D latent plot only shows the first two dimensions, while t-SNE uses all latent dimensions.

---

## Supervised Learning

To enable supervised learning:

```python
RUN_SUPERVISED = True
```

To disable supervised learning:

```python
RUN_SUPERVISED = False
```

The supervised tasks are controlled by:

```python
SUPERVISED_TARGET_MODES = ["WFS_TS", "WIDTH"]
```

or by the `PREDICT_TARGET` switch shown above.

---

## WFS_TS Classification

The `WFS_TS` task predicts welding condition classes from folder names.

Example:

```text
4_14_2026_WFS_10_TS_4 -> WFS_10_TS_4
```

Typical outputs:

```text
outputs/supervised_wfs_ts/
├── best_model.pt
├── training_history.csv
├── learning_curves.png
├── test_predictions.csv
├── confusion_matrix_test.png
├── confusion_matrix_test_normalized.png
├── classification_report_test.json
├── split_info.json
└── summary_test.json
```

### Confusion Matrix

The regular confusion matrix shows raw sample counts.

The normalized confusion matrix shows row-wise percentages.

For example, if one true class has 100 samples and 80 are correctly predicted, the normalized value is:

```text
80 / 100 = 0.80
```

Each row in the normalized confusion matrix sums to approximately 1.

---

## WIDTH Regression

The `WIDTH` task predicts one continuous width value for each CWT sample.

It matches CWT samples to Excel labels using:

```text
dataset class_folder  <->  Excel class_folder
dataset clip_indices  <->  Excel indices
```

Typical outputs:

```text
outputs/supervised_width/
├── best_model.pt
├── training_history.csv
├── learning_curves.png
├── test_predictions.csv
├── regression_true_vs_pred_test.png
├── split_info.json
└── summary_test.json
```

---

## Expected Output Structure

Example:

```text
outputs/
├── cwt_dataset_grouped.npz
├── vae_results/
│   ├── vae_model.pt
│   ├── vae_training_history.csv
│   ├── vae_training_losses.png
│   ├── vae_latents.csv
│   ├── vae_latent_tsne.png
│   └── vae_latent_2d.png
├── supervised_wfs_ts/
│   ├── best_model.pt
│   ├── training_history.csv
│   ├── learning_curves.png
│   ├── test_predictions.csv
│   ├── confusion_matrix_test.png
│   ├── confusion_matrix_test_normalized.png
│   ├── classification_report_test.json
│   ├── split_info.json
│   └── summary_test.json
└── supervised_width/
    ├── best_model.pt
    ├── training_history.csv
    ├── learning_curves.png
    ├── test_predictions.csv
    ├── regression_true_vs_pred_test.png
    ├── split_info.json
    └── summary_test.json
```

---

## Train/Test Split Notes

The VAE uses the full dataset because it is unsupervised.

The supervised tasks use train, validation, and test splits.

The `WFS_TS` and `WIDTH` tasks may not always use exactly the same samples because the `WIDTH` task depends on whether a matching width label exists in the Excel file.

If `WIDTH_MISSING_POLICY = "drop"`, samples without width labels are skipped for the width regression task.

---

## Common Issues

### Relative import error

If you see:

```text
ImportError: attempted relative import with no known parent package
```

it usually means you ran:

```bash
python main.py
```

inside the package folder.

Run this from the project root instead:

```bash
python -m esab_complete_pipeline.main
```

---

### No synced NPZ files found

If you see:

```text
No valid synced NPZ files were found
```

check your data source.

If you already have a grouped CWT dataset, use:

```python
DATASET_SOURCE = "grouped_npz"
CWT_DATASET_PATH = Path(r".../cwt_dataset_grouped.npz")
```

If you want to build from synced signal files, use:

```python
DATASET_SOURCE = "build"
USE_SYNC_STAGE = False
ROOT_DIR = Path(r".../sync_data_5kHz")
```

If you want to build from raw signal files, use:

```python
DATASET_SOURCE = "build"
USE_SYNC_STAGE = True
ROOT_DIR = Path(r".../data")
SYNC_CACHE_DIR = Path(r".../sync_data_5kHz")
```

---

## Notes

This repository is intended for research and experimental analysis of ESAB robotic welding data.

The generated width Excel file may contain invalid or black images. These samples should be manually reviewed before being used for supervised width regression.