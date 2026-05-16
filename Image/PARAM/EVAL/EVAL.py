DIR_TO_TOP: str = "../.."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(2)

import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor
from typing import Tuple, Optional, Callable, Any

import config
CONFIG = config.read_config(config.CONFIG_PATH)
PM_ML = __import__(
    f"{CONFIG.get('PARAM')}.ML.ML", fromlist=['']
)
util = __import__(
    f"{CONFIG.get('utils')}.util", fromlist=['']
)
MyErr = __import__(
    f"{CONFIG.get('utils')}.exceptions", fromlist=['']
)


################################################################################
################################################################################
################################################################################


EVAL_SOTA_DIR: str = os.path.join(PM_ML.RESULT_DIR, CONFIG.get('eval_fname')) # Could be 'temp'. Accept user input. 
EVAL_DATA_DIR: str = PM_ML.TEST_DATA_DIR # Directory containing evaluation dataset. The dataset directory tree should follow the same structure as that of the training dataset. Accept user input. 
EVAL_MODEL_PATH: str = "OPTIMUM" # Token or path of a specific trained model file. Accept user input.
EVAL_LOG_NAME: str = PM_ML.LOG_NAME # The name of log file from the training result. Accept user input. 









# MODEL: str = PM_ML.MODEL
# DATASET: str = PM_ML.DATASET
# MAX_DATASET_SIZE: Optional[str] = None
# INPUT_IMG_SIZE: Optional[list[int,int]] = PM_ML.INPUT_IMG_SIZE # 2D size of the input scalogram image. (width - axis 1, height - axis 0). Could be None (no reshape needed). 
# INPUT_TRANSFORM: Compose = PM_ML.INPUT_TRANSFORM # Convert input data to "Tensor[torch.float32]". Could be None. 
# INPUT_LIST: list[Tuple[str]] = PM_ML.INPUT_LIST # Allow multiple (e.g., two) types of data. 
# INPUT_FEATURIZE_MODE: str = PM_ML.INPUT_FEATURIZE_MODE
# OUTPUT_LIST: list[str] = PM_ML.OUTPUT_LIST # Only one type of data is allowed, a.k.a no tuple is allowed. 
# OUTPUT_FEATURIZE_MODE: str = PM_ML.OUTPUT_FEATURIZE_MODE
