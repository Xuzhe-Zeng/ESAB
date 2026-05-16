DIR_TO_TOP: str = "../.."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(2)

from typing import Optional

import config
CONFIG = config.read_config(config.CONFIG_PATH)
PM_TS = __import__(
    f"{CONFIG.get('PARAM')}.TS.TS", fromlist=['']
)
PM_IMG = __import__(
    f"{CONFIG.get('PARAM')}.IMG.IMG", fromlist=['']
)


################################################################################
################################################################################
################################################################################


CLIP_DIR: str = (
    f"{CONFIG.get('data')}/" + 
    f"{CONFIG.get('processed_data_fname')}/" + 
    f"{CONFIG.get('CLP')}"
) # Total directory of all clips. The subfolder would be categorized by different clip timespans. 

# Clipping param. -- Example: 400 TSpt = 90 IMG = 4 ms. 
T_CLIP_LENGTH: float = 4e-3 # Time span of one clip. At least 2 ms. Unit: s. 
T_CLIP_STRIDE: float = 2e-3 # Time span of stride between adjacent clips. At least 2 ms. Default: 2e-3. Unit: s. 
CLIP_NUM_EACHGROUP: Optional[int] = None # The number of clips after each synchronized anchor points (typically scanline starting points). 
CLIP_FIRST_N_SCANLINES: Optional[int] = None # The number of (beginning) scanlines used for clipping. Needed to be manually fine-tuned. 

# Scanline grouping -- Synchronize every a group of scanlines. 
SCANLINE_GROUP_SIZE: int = 2 # How many consecutive scanlines used to generate clip data. Better if this number is larger than 1 (including at least one inter-scanline dwelling sections). 
SCANLINE_GROUP_STRIDE: int = 1 # How many scanlines between two consecutive scanline groups. 

TS_CLIP_LENGTH: int = int(T_CLIP_LENGTH * PM_TS.TS_SR)
TS_CLIP_STRIDE: int = int(T_CLIP_STRIDE * PM_TS.TS_SR)

IMG_CLIP_LENGTH: int = int(T_CLIP_LENGTH * PM_IMG.HS_SR)
IMG_CLIP_STRIDE: int = int(T_CLIP_STRIDE * PM_IMG.HS_SR)

# Saving param. 
SAVE_DICT_EXT: str = 'pkl'

