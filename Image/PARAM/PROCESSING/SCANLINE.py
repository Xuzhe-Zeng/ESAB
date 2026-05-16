DIR_TO_TOP: str = "../.."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(2)

import numpy as np
from typing import Optional, Callable, Any

import config
CONFIG = config.read_config(config.CONFIG_PATH)
PM_IMG = __import__(
    f"{CONFIG.get('PARAM')}.IMG.IMG", fromlist=['']
)
PM_TS = __import__(
    f"{CONFIG.get('PARAM')}.TS.TS", fromlist=['']
)
PM_PR_PR = __import__(
    f"{CONFIG.get('PARAM')}.PROCESSING.PROCESS", fromlist=['']
)


################################################################################
################################################################################
################################################################################


PHO_RAW_DATA_DIR: str = PM_TS.PHO_RAW_DATA_DIR
ACC_RAW_DATA_DIR: str = PM_TS.ACC_RAW_DATA_DIR
IMG_RAW_DATA_DIR: str = PM_IMG.IMG_RAW_DATA_DIR

SCANLINE_DIR: str = (
    f"{CONFIG.get('data')}/" + 
    f"{CONFIG.get('processed_data_fname')}/" + 
    f"{CONFIG.get('SCL')}"
)

TS_SR: float = PM_TS.TS_SR # 100e3. 
HS_SR: float = PM_IMG.HS_SR # 22.5e3; 22e3. 

ACC_SYNC_OFFSET: int = int(PM_TS.ACC_LATENCY * TS_SR)

# Time-series scanline partition PARAM. Need to manually calibrate.
# Modality-specific partition mode. ['Avg', 'MinMaxCov', 'Callable', 'Manual']. 
PHO_PARTITION_MODE: str = 'Avg'
ACC_PARTITION_MODE: str = 'Manual' 
PHO_ACTIVE_TRSHLD: float = PM_TS.PHO_ACTIVE_TRSHLD # A constant threshold for photodiode data that indicates whether the laser is on. 
TS_PARSE_WINDOW_LENGTH: int = 50
TS_PARSE_WINDOW_STRIDE: int = 1
# Photodiode scanline partition -- For 'MinMaxCov' mode. 
MAX_MIN_TRSHLD: float = 0.002 # Min-Max threshold value must guarantee accurate capturing of the first laser-on point in the photodiode data. 
COV_TRSHLD: float = 0.018 # Covariance threshold value must guarantee accurate capturing of the first laser-on point in the photodiode data. 
# Photodiode scanline partition -- For 'Avg' mode (currently used). 
AVG_TRSHLD: float = 0.016 # Average threshold value must guarantee accurate capturing of the first laser-on point in the photodiode data.
NEWLINE_TRSHLD: float = 0.0015 # Threshold value that select the point where the next scanline starts. It must guarantee accurate capturing of the first laser-on point in the photodiode data. 
# Photodiode scanline partition -- For 'Callable' mode. 
PARSE_FUNC: Optional[Callable] = None
PARSE_ARGS: Optional[list[Any]] = None
# Highspeed image scanline partition PARAM. Need to manually calibrate. 
HS_TRACK_AREADEVIATE_LIM: float = 0.25 # The minimum tolerated area ratio of average history melt pol areas that could be recognized as a meltpool. Used to filter out spatters close to the scanline. 0.25. 
HS_TRACK_LOCATE_SEARCH_RANGE: float = 60. # Melt pool locating range. 80. 
HS_HATCH_SPACING_PXL: float = 10. # Hatch spacing between two adjacent scanlines in pixels. 

# High speed image data processing PARAM. 
SCALE_BAR_DPMM: float = 56.25 # Pix per mm. 56.25; 55.

# meltpool_area_trshld_begin = 50 # Need parametric search. 
MP_DIST_TRSHLD: float = 30. # Need parametric search. 

# Scanline match-up checking. Need to manually calibrate. 
SCANLINE_MATCHUP_TRSHLD: float = 1.0 # Unit: ms. Manually characterized to fit all cases (mainly between TS and HSI). Default: 0.5 ms (~10 hsi img & 50 ts pt)

# Kalman filter tracking (optional). 
IS_KALMAN_OPT: bool = True
F: np.ndarray[float] = np.array([
    [1., 0., 1./HS_SR, 0.],
    [0., 1., 0., 1./HS_SR],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.]
])
P_0: np.ndarray[float] = np.eye(4) * 0.01
Q: np.ndarray[float] = np.eye(4) * 0.01
R: np.ndarray[float] = np.eye(4) * 0.01
H: np.ndarray[float] = np.eye(4)

# Saving param. 
SAVE_DICT_EXT: str = 'pkl'
