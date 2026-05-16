# ESAB Camera Image and Width Extraction

This project contains simple scripts for extracting camera images from `.mat` files and calculating the processed image width from the extracted `.png` images.

## Project Structure

```text
ESAB_Project/
│
├── img_save.py
├── width_extract.py
├── config.py
└── README.md
```

## 1. Image Extraction

`img_save.py` reads camera `.mat` files and saves each frame as a `.png` image.

The extracted images will be saved into an `image/` folder under each experiment folder.

Example:

```text
4_14_2026_WFS_10_TS_4/
│
├── camera_xxx.mat
└── image/
    ├── 0.png
    ├── 1.png
    ├── 2.png
    └── ...
```

The main settings are controlled in `config.py`.

Example:

```python
ROOT_DIR = Path(r"D:\ESAB\Experimental Data\data_rotate")
DATASET_KEY = "bFLY_recording"
ROTATE_CCW_90 = True
```

## 2. Width Extraction

`width_extract.py` calculates the processed width from the extracted images.

The output Excel file contains three columns:

```text
class_folder | image_file | width
```

Example:

```text
WFS_10_TS_4 | 0 | 235
WFS_10_TS_4 | 1 | 238
WFS_10_TS_4 | 2 | None
```

## Important Note Before Training

The generated Excel file should **not** be used directly for model training.

Some extracted frames may be black or invalid images. These images may still appear in the generated Excel file, but they should not be treated as valid training samples.

Before training, the extracted images should be checked manually. Black or invalid images should be removed first.

After removing invalid images, the remaining valid images should be re-indexed from zero.

For example, if the original valid images are:

```text
3.png
4.png
5.png
8.png
9.png
```

They should be renamed as:

```text
0.png
1.png
2.png
3.png
4.png
```

Then, a new clean annotation file should be created based only on the valid images.

Only this cleaned Excel file should be used for training.

## Recommended Workflow

1. Run `img_save.py` to extract images from camera `.mat` files.
2. Manually check the extracted images.
3. Remove black or invalid images.
4. Re-index the remaining valid images from zero.
5. Run `width_extract.py` on the cleaned image set.
6. Use the cleaned Excel file for model training.

## Notes

- GitHub should mainly store source code, configuration files, and README files.
- Large raw data files such as `.mat`, `.npz`, `.xlsx`, and extracted image folders are not recommended to be uploaded to GitHub.
- Large experimental data should be stored locally or in cloud storage.
