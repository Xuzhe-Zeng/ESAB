DIR_TO_TOP: str = "../.."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(2)

import numpy as np
import operator
import pandas as pd
from typing import Any

import config
CONFIG = config.read_config(config.CONFIG_PATH)
MyErr = __import__(
    f"{CONFIG.get('utils')}.exceptions", fromlist=['']
)


################################################################################
################################################################################
################################################################################


POR_RAW_DATA_DIR: str = (
    f"{CONFIG.get('data')}/{CONFIG.get('raw_data_fname')}/{CONFIG.get('POR')}"
)
COL_NAME: str = "Y (µm)" # Name of column of interest for clipping. Relative coordinates. Currently only for 0 degrees single-beads. 
PORE_FILTER_LIST: list[list[Any]] = [
    ["Voxel count", (4, operator.ge)], # Unit: 1. Might impact pore numbers. 
    ["Min Feret Diameter (µm)", (5.92, operator.ge)], # Unit: um. 
    ["Y (µm)", (0.+100., operator.ge)], # Unit: um. 
    ["Y (µm)", (5800.-100., operator.le)], # Unit: um. 
] # Applied when scanline is partitioned. [`cof_name`, (`threshold`, `operator_func`)]. Only one (threshold, judgment function) tuple is allowed for one list. For an INPUT of `cof_name`, operator_func(INPUT, `threshold`). 

POR_TYPES: list[str] = [
    'volume', 'count', 'line_volume', 'line_count'
] # Types of definition of porosity. Could be more than 1. 
POR_EXT: str = 'xlsx' # File format of porosity data. 

POR_DRIFTING_DIST: float = 50 # Pore drifting distance within melt pool. Only drifting backwards. Unit: um. Default: 0.

def POROSITY(clip_data: dict[Any], type: str) -> float:
    """
    Porosity density definition. 

    Args:
        clip_data (dict[Any]): _description_
        type (str): _description_

    Raises:
        MyErr.InvalidError: _description_

    Returns:
        float: _description_
    """
    
    clip_por_df: pd.DataFrame = clip_data['clip_data']
    if 'line' in type: travel_dist: float = clip_data['line_dist'] # Unit: um. 
    else: pass
    
    if type == 'volume':
        return np.sum(clip_por_df['Volume (µm³)'].values.astype(float)) # Unit: um^3. 
    elif type == 'count': return len(clip_por_df) # Unit: 1. 
    elif type == 'line_volume': # Linear volumetric density. 
        por_vol: float = np.sum(
            clip_por_df['Volume (µm³)'].values.astype(float)
        ) # Unit: um^3. 
        return por_vol / travel_dist # Unit: um^2. 
    elif type == 'line_count': # Linear number density. 
        por_num: float = len(clip_por_df) # Unit: 1. 
        return por_num / travel_dist # Unit: um^-1. 
    else: raise MyErr.InvalidError("Invalid porosity type. ")

