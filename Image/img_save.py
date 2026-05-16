import h5py
import numpy as np
from scipy.io import loadmat
from PIL import Image

from config import DATASET_KEY, ROOT_DIR, ROTATE_CCW_90


def normalize_to_uint8(img):
    img = np.array(img)
    img = np.squeeze(img)

    img = img.astype(np.float32)
    img = img - img.min()

    if img.max() > 0:
        img = img / img.max() * 255

    return img.astype(np.uint8)


def save_one_image(img, save_path, rotate_ccw_90: bool = False):
    img = normalize_to_uint8(img)

    if rotate_ccw_90:
        img = np.rot90(img, k=1)  # k=1 means 90 degrees counterclockwise.

    Image.fromarray(img).save(save_path)


def save_camera_images(mat_path):
    output_dir = mat_path.parent / "image"
    output_dir.mkdir(exist_ok=True)

    print("\n==============================")
    print(f"Reading: {mat_path}")
    print(f"Saving to: {output_dir}")
    print(f"Rotate CCW 90: {ROTATE_CCW_90}")

    try:
        # Try h5py first for MATLAB v7.3 files.
        with h5py.File(mat_path, "r") as f:
            print("Read mode: h5py")
            print("Keys:", list(f.keys()))

            if DATASET_KEY not in f:
                print(f"[SKIP] Cannot find key: {DATASET_KEY}")
                return

            frames = f[DATASET_KEY]
            print("Shape:", frames.shape)

            num_frames = frames.shape[0]

            for idx in range(num_frames):
                img = frames[idx, 0, :, :]
                save_path = output_dir / f"{idx}.png"

                save_one_image(
                    img,
                    save_path,
                    rotate_ccw_90=ROTATE_CCW_90,
                )

                if idx % 50 == 0:
                    print(f"Saved {idx}/{num_frames}")

    except OSError:
        # If h5py fails, use scipy.loadmat for standard MATLAB files.
        print("h5py failed, trying scipy.loadmat...")
        data = loadmat(mat_path)

        print("Read mode: scipy.loadmat")
        print("Keys:", [k for k in data.keys() if not k.startswith("__")])

        if DATASET_KEY not in data:
            print(f"[SKIP] Cannot find key: {DATASET_KEY}")
            return

        frames = data[DATASET_KEY]
        print("Shape:", frames.shape)

        # Expected shape: (530, 1, 1920, 1200)
        if frames.ndim == 4 and frames.shape[1] == 1:
            num_frames = frames.shape[0]

            for idx in range(num_frames):
                img = frames[idx, 0, :, :]
                save_path = output_dir / f"{idx}.png"

                save_one_image(
                    img,
                    save_path,
                    rotate_ccw_90=ROTATE_CCW_90,
                )

                if idx % 50 == 0:
                    print(f"Saved {idx}/{num_frames}")

        # Expected shape: (1920, 1200, 1, 530), or a similar MATLAB-style order.
        elif frames.ndim == 4 and frames.shape[-1] > 1:
            num_frames = frames.shape[-1]

            for idx in range(num_frames):
                img = frames[:, :, 0, idx]
                save_path = output_dir / f"{idx}.png"

                save_one_image(
                    img,
                    save_path,
                    rotate_ccw_90=ROTATE_CCW_90,
                )

                if idx % 50 == 0:
                    print(f"Saved {idx}/{num_frames}")

        else:
            print("[SKIP] Unsupported shape:", frames.shape)
            return

    print(f"Finished: {mat_path}")


def main():
    camera_files = list(ROOT_DIR.rglob("*camera*.mat"))

    print(f"Found {len(camera_files)} camera files.")
    print(f"ROTATE_CCW_90 = {ROTATE_CCW_90}")

    for mat_path in camera_files:
        save_camera_images(mat_path)

    print("\nAll done.")


if __name__ == "__main__":
    main()
