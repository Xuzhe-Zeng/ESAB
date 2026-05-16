# ESAB Camera Image Export and Width Extraction

This folder contains three files:

- `config.py`: stores the paths and shared settings.
- `img_save.py`: exports camera frames from `.mat` files to PNG images.
- `width_extract.py`: calculates the processed weld width from exported PNG images and saves the result to Excel.

## 1. Set the path

Open `config.py` and edit this path if needed:

```python
ROOT_DIR = Path(r"D:\ESAB\Experimental Data\data_rotate")
```

The Excel result will be saved by default to:

```python
OUTPUT_EXCEL = ROOT_DIR / "image_width_results.xlsx"
```

So the default output is:

```text
D:\ESAB\Experimental Data\data_rotate\image_width_results.xlsx
```

## 2. Expected folder structure

```text
D:\ESAB\Experimental Data\data_rotate
├── 4_14_2026_WFS_10_TS_4
│   ├── image
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   └── ...
└── ...
```

## 3. Export camera images

Run:

```bash
python img_save.py
```

This script searches under `ROOT_DIR` for `.mat` files with `camera` in the filename and saves PNG frames into an `image` folder under each experiment folder.

The rotation setting is also in `config.py`:

```python
ROTATE_CCW_90 = True
```

Set it to `False` if you do not want to rotate the exported images.

## 4. Extract processed width

Run:

```bash
python width_extract.py
```

This script reads all PNG images from each `image` folder, calculates the processed width using `imgBasics.Frame`, and saves the result to `OUTPUT_EXCEL`.

The Excel file contains:

- `class_folder`: class label such as `WFS_10_TS_4`
- `image_file`: image index, such as `0`, `1`, `2`
- `width`: processed width in pixels

## Required packages

```bash
pip install h5py numpy scipy pillow pandas openpyxl
```

`width_extract.py` also requires your local `imgBasics` module.
