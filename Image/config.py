from pathlib import Path


# Root folder that contains all experiment folders.
# Example experiment folder name: 4_14_2026_WFS_10_TS_4
ROOT_DIR = Path(r"D:\ESAB\Experimental Data\data")

# Camera dataset key inside the .mat file.
DATASET_KEY = "bFLY_recording"

# True: rotate images 90 degrees counterclockwise before saving.
# False: keep the original image orientation.
ROTATE_CCW_90 = True

# Excel output path for width extraction results.
OUTPUT_EXCEL = ROOT_DIR / "image_width_results.xlsx"


import os
import json

################################################################################
CONFIG_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.json"
)  # The path of the config file.


################################################################################


def read_config(config_path: str) -> dict:
    """_summary_

    Args:
        config_path (str): _description_

    Returns:
        dict: _description_
    """

    with open(config_path) as json_file:
        config = json.load(json_file)
    json_file.close()

    return config


def update_config(config: dict, config_path: str):
    """_summary_

    Args:
        config (dict): _description_
        config_path (str): _description_
    """

    with open(config_path, 'w') as json_file:
        json.dump(config, json_file, indent=4)
    json_file.close()
