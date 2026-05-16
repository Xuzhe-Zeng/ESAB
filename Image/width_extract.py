from pathlib import Path
import re
import pandas as pd

from config import OUTPUT_EXCEL, ROOT_DIR
from imgBasics import Frame


def extract_class_label(folder_name: str) -> str:
    """
    Example:
    4_14_2026_WFS_10_TS_4 -> WFS_10_TS_4
    """
    match = re.search(r"WFS_\d+_TS_\d+", folder_name)
    if match:
        return match.group(0)
    return folder_name


def get_image_index(png_path: Path) -> int:
    """
    Example:
    123.png -> 123
    """
    return int(png_path.stem)


def get_frame_processed_width(img_path: Path) -> int | None:
    """
    Use Frame preprocessing and thresholding to calculate the processed width.
    The width is calculated from visible_pixel_indices after Frame processing.
    """
    frame = Frame(file_path=str(img_path))

    pixels = frame.visible_pixel_indices  # [row, col]

    if pixels is None:
        return None

    if pixels.size == 0:
        return None

    if pixels[0, 0] is None:
        return None

    cols = pixels[:, 1].astype(int)
    width_px = cols.max() - cols.min() + 1

    return int(width_px)


def main():
    rows = []

    for class_folder in sorted(ROOT_DIR.iterdir()):
        if not class_folder.is_dir():
            continue

        class_label = extract_class_label(class_folder.name)

        image_dir = class_folder / "image"

        if not image_dir.exists() or not image_dir.is_dir():
            print(f"[SKIP] No image folder: {class_folder.name}")
            continue

        png_files = sorted(image_dir.glob("*.png"), key=lambda p: int(p.stem))

        if not png_files:
            print(f"[SKIP] No PNG files in: {image_dir}")
            continue

        for png_path in png_files:
            try:
                image_index = get_image_index(png_path)
                width = get_frame_processed_width(png_path)

                rows.append({
                    "class_folder": class_label,
                    "image_file": image_index,
                    "width": width,
                })

                print(f"[OK] {class_label} | {image_index} | width = {width}")

            except Exception as e:
                rows.append({
                    "class_folder": class_label,
                    "image_file": None,
                    "width": None,
                })

                print(f"[WARN] Failed: {png_path} | {e}")

    df = pd.DataFrame(rows)

    df.to_excel(OUTPUT_EXCEL, index=False)

    print("\nDone.")
    print(f"Saved to: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
