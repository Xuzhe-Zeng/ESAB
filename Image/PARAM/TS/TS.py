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


################################################################################
################################################################################
################################################################################


PHO_RAW_DATA_DIR: str = (
    f"{CONFIG.get('data')}/{CONFIG.get('raw_data_fname')}/{CONFIG.get('PHO')}"
)
ACC_RAW_DATA_DIR: str = (
    f"{CONFIG.get('data')}/{CONFIG.get('raw_data_fname')}/{CONFIG.get('ACC')}"
)

PHO_RAW_EXT: str = 'npy' # Photodiode raw data file format (extension). 
ACC_RAW_EXT: str = 'npy' # Acoustic raw data file format (extension). 

PHO_CHANNEL_NO: Optional[int] = 0
ACC_CHANNEL_NO: Optional[int] = 1

TS_SR: float = 100e3

ACC_LATENCY: float = 0.00078
PHO_ACTIVE_TRSHLD: float = 0.017
