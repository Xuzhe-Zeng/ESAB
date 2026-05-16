DIR_TO_TOP: str = "../.."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(2)

import numpy as np
from typing import Tuple, Optional

import config
CONFIG = config.read_config(config.CONFIG_PATH)
PM_TS = __import__(
    f"{CONFIG.get('PARAM')}.TS.TS", fromlist=['']
)
PM_IMG = __import__(
    f"{CONFIG.get('PARAM')}.IMG.IMG", fromlist=['']
)
PM_POR = __import__(
    f"{CONFIG.get('PARAM')}.POR.POR", fromlist=['']
)


################################################################################
################################################################################
################################################################################


PHO_RAW_DATA_DIR: str = PM_TS.PHO_RAW_DATA_DIR
ACC_RAW_DATA_DIR: str = PM_TS.ACC_RAW_DATA_DIR
HSI_RAW_DATA_DIR: str = PM_IMG.IMG_RAW_DATA_DIR
HSI_SEITON_DATA_DIR: str = PM_IMG.IMG_SEITON_DATA_DIR
POR_RAW_DATA_DIR: str = PM_POR.POR_RAW_DATA_DIR
PARAM_FILE_PATH: Optional[str] = (
    f"{CONFIG.get('data')}/" + 
    f"{CONFIG.get('raw_data_fname')}/" + 
    f"{CONFIG.get('PARAM_FILE')}"
)

ACC_LATENCY: float = PM_TS.ACC_LATENCY # Unit: s. Manually characterized. For new data: 0.00078. 

# Spectral. 
SPECTRAL_MODE: str = 'cwt' # ['cwt', 'stft']. 
# # Spectral -- CWT. 
CWT_MOTHER_WAVELET: str = 'morlet'
CWT_MORLET_MU: float = 6.2 # For Morlet mother wavelet. MATLAB value: 6.2. The higher this number, the higher the spectral resolution & the lower the temporal resolution. 
CWT_GMW_BETA: int = 60 # Default: 60.  
CWT_FREQ_RANGE: list[float] = [3e3, 50e3] # list[f_min, f_max]. Must satisfy the Nyquist-Shannon theorem. 
CWT_SCALETYPE: str = 'log' # 'log', 'log-piecewise', 'linear'. Default: 'log'. 
CWT_OCTNV: int = 32. # Number of voices (wavelets) per octave. Default : 32. 
CWT_SCALO_VAL_RANGE: dict[list[float]] = {
    CONFIG.get('PHO'): [0., 0.002],
    CONFIG.get('ACC'): [0., 0.02], 
} # Manually characterize. 
# # Spectral -- STFT. 

# Straighten. 
STRAIGHTEN_MODE: str = 'original' # ['original', 'thresheld']. 
STRAIGHTEN_DIRECT: np.ndarray[float] = np.array([0., 1.]) # Right direction of the highspeed image frame. 

# HS image average param. 
HSI_IMG_AVG_PRANGE: Tuple[float] = (0.4, 0.6)

# Data processing log. 
DATA_PROCESSING_SAVELOG: str = os.path.join(
    f"{CONFIG.get('data')}", 
    f"{CONFIG.get('processed_data_fname')}", 
    f"{CONFIG.get('process_log_name')}.log"
)
