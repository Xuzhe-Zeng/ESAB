DIR_TO_TOP: str = "../.."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(2)

from typing import Tuple
import config
CONFIG = config.read_config(config.CONFIG_PATH)


################################################################################
################################################################################
################################################################################


IMG_RAW_DATA_DIR: str = (
    f"{CONFIG.get('data')}/" + 
    f"{CONFIG.get('raw_data_fname')}/" + 
    f"{CONFIG.get('HSI')}"
)
IMG_SEITON_DATA_DIR: str = (
    f"{CONFIG.get('data')}/" + 
    f"{CONFIG.get('processed_data_fname')}/" + 
    f"{CONFIG.get('STN')}"
)
LAYER_THETA_CSV_PATH: str = (
    f"{CONFIG.get('data')}/layer_orientation.csv"
)

HS_SR: float = 22.5e3 # 22.5e3; 22e3. 

HSI_RAW_EXT: str = 'png'

INTENSITY_THRESHOLD: Tuple[float, float] = (0.8, 1.0)
SIDEBAR_THRESHOLD: float = 0.05
SIDEBAR_COLUMNS: list[Tuple[int,int], Tuple[int,int]] = [(0,16), (495,511)] # For the collected meltpool dataset, images commonly have two bars of relatively-highlighted pixels located on the two sides of the image. 
PLUME_THRESHOLD: Tuple[float, float] = (0.05, 0.4)

# For image pixel clustering. 
DBSCAN_EPSILON: int = 2
DBSCAN_MIN_PTS: int = 5

# For melt pool principal axes determination. 
PC_NUM_FRAME: int = 2
PCA_MODE_FRAME: str = 'transpose'