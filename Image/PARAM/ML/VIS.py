DIR_TO_TOP: str = "../.."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(2)

import numpy as np
from typing import Union, Tuple
from numpy import inf

import config
CONFIG = config.read_config(config.CONFIG_PATH)
MyErr = __import__(
    f"{CONFIG.get('utils')}.exceptions", fromlist=['']
)


################################################################################
################################################################################
################################################################################


LOSS_PLOT_NAME: str = "train_valid_loss.png" # Please include file extension.
LOSS_PLOT_YSCALE: str = 'log' # The yscale for loss plot. 'log' or 'linear'. 

FIG_SIZE: Tuple[float, float] = (20., 20.) # Size of the figure. 
FONT_SIZE: float = 35. # General font size (default value for params without specifications). 
LABEL_SIZE: float = 35. # Label font size. 
LEGEND_SIZE: float = 40. # Legend font size. 

PLOT_EPOCHLOSS_LW: float = 3. # Line width of epoch loss curves. 
PLOT_BATCHLOSS_LW: float = 10. # Line width of batch loss curves. 
PLOT_BATCHLOSS_ALPHA: float = 0.3 # Transparency param of batch loss curves. 
PLOT_TR_LOSS_C: Union[str, Tuple[float]] = 'blue' # Color of training loss curves. 
PLOT_EV_LOSS_C: Union[str, Tuple[float]] = 'orange' # Color of validation loss curves. 


def LOSS_PLOT_FUNC(
    loss_arr: np.ndarray[float], 
    scale: str = LOSS_PLOT_YSCALE, 
) -> np.ndarray[float]:
    """_summary_

    Args:
        loss_arr (np.ndarray[float]): _description_
        scale (str, optional): _description_. Defaults to LOSS_PLOT_YSCALE.

    Raises:
        MyErr.InvalidError: Invalid loss plot scale mode. 

    Returns:
        np.ndarray[float]: _description_
    """
    
    if scale == 'log': 
        # Skipping zero loss values when plotting. 
        loss_arr: np.ndarray[float] = np.asarray([
            np.log(loss_val) if loss_val != 0 else -inf
            for loss_val in loss_arr
        ]).astype(float).reshape(-1)
        # loss_arr: np.ndarray[float] = copy.deepcopy(np.log(loss_arr))
    elif scale == 'linear': pass
    else: raise MyErr.InvalidError("Invalid loss plot function scale. ")
    
    return loss_arr

