DIR_TO_TOP: str = "../.."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(2)

import copy
import glob
import numpy as np
import PIL
import PIL.Image
import shutil
import torch

from collections import defaultdict
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.transforms import Compose

from pathlib import Path
from typing import Tuple, Union, Any, Optional

import config
CONFIG = config.read_config(config.CONFIG_PATH)
PM_ML = __import__(
    f"{CONFIG.get('PARAM')}.ML.ML", fromlist=['']
)
util = __import__(
    f"{CONFIG.get('utils')}.util", fromlist=['']
)
uplog = __import__(
    f"{CONFIG.get('utils')}.process.savelog", fromlist=['']
)
MyErr = __import__(
    f"{CONFIG.get('utils')}.exceptions", fromlist=['']
)


################################################################################
################################################################################
################################################################################


