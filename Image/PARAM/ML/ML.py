DIR_TO_TOP: str = "../.."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(2)

import torch
import torch.nn as nn

from numpy.typing import ArrayLike, DTypeLike
from torchvision.transforms import Compose, Resize, ToTensor
from typing import Union, Tuple, Optional, Callable, Any

import config
CONFIG = config.read_config(config.CONFIG_PATH)
PM_PR_CL = __import__(
    f"{CONFIG.get('PARAM')}.PROCESSING.CLIP", fromlist=['']
)


################################################################################
################################################################################
################################################################################


# ======================= For neural network definition ====================== # 
MODEL: Union[str, Tuple[str, str]] = "cnn2d" # Mode of data-driven model. 'cnn2d' ('CNN2D') or 'resnet' ('ResNet'). Accept user input, in which the format is: Tuple[`module_path`, `module_name`]. If including more modes, remember to update the module `train/retrieve.py` and make sure the new modes are recognizable by the function. Accept user input. 
ACTIVATION_LAYER: Callable = nn.ReLU(inplace=True)
IS_ACTIVATION_LASTLAYER: bool = False # Whether the last layer of ResNet is an activation layer (e.g., ReLU()). This can be useful when the output is surely non-negative. Default: False. 
# ============================================================================ #


# ============================ For Data definition =========================== #
DATASET: Union[str, Tuple[str, str]] = "ToPore" # Mode (Type) of dataset. Example: 'ToPore'. The format is: Tuple[`module_path`, `module_name`]. To include more dataset modes in `utils/ML/dataset.py`, remember to update the module `train/retrieve.py` and make sure the new modes are recognizable by the function. Accept user input. 
ML_DATA_DIR: str = f"{CONFIG.get('data')}/{CONFIG.get('ml_data_fname')}" # Accept user input. 
TRAIN_DATA_DIR: str = os.path.join(ML_DATA_DIR, f"{CONFIG.get('train_data')}") # Directory for the training dataset. Note that it should have the same dirtree structure as that of the test dataset directory. Accept user input. 
VALID_DATA_DIR: Optional[str] = None # Directory for the validation dataset. Note that it should have the same dirtree structure as that of the training dataset directory. In most cases this is not necessarily needed. Accept user input. 
TEST_DATA_DIR: str = os.path.join(ML_DATA_DIR, f"{CONFIG.get('test_data')}") # Directory for the test dataset. Note that it should have the same dirtree structure as that of the training dataset directory. Accept user input. 

DATASET_SHUFFLE: bool = True
MAX_DATASET_SIZE: Optional[int] = None # Maximum dataset size, typically smaller than the total dataset size. If not applied, set it to None. Accept user input. 
TENSOR_DTYPE: DTypeLike = torch.float # Data type for tensors. As default settings, torch.float = torch.float32, which has been optimized for GPU computation. Specify as torch.float64 only when necessary. 

INPUT_TOKEN: Any = 'input' # The key to retrieve input data from the output of `__getitem__` of a Dataset object. 
OUTPUT_TOKEN: Any = 'output' # The key to retrieve input data from the output of `__getitem__` of a Dataset object. 
INPUT_LIST: list[Tuple[str, str]] = [
    (CONFIG.get('ACC'), 'cwt_pil', ), # Acoustic scalogram clip. 
    # (CONFIG.get('PHO'), 'cwt_pil', ), # Photodiode scalogram clip. 
    # (CONFIG.get('HSI'), 'avg_seiton', ), # High-speed image, straightened. 
    (CONFIG.get('POR'), 'V$100', ), # Laser scanning velocity. Default option.
    # (CONFIG.get('POR'), 'P$0.5', ), # Laser power. 
    # (CONFIG.get('POR'), 'P_sqrtV$8', ), # Normalized enthalpy. 
] # Allow multi-modal data in different channels. Tuple: (*TOKENS), layer-by-layer tokens. For the last token, if there is additional number after the $ sign for scalars, this means the feature is originally a multiplicable variable and it will be multiplied by the number (typically for PIL image conversion purposes). 
OUTPUT_LIST: list[str] = [
    CONFIG.get('POR'), 'line_count', 
] # Only one type of data is allowed, a.k.a. no tuple is allowed. Check `POR.py` for more info. 
INPUT_FEATURIZE_MODE: str = 'img' # 'scalar', 'arr' or 'img'. 
OUTPUT_FEATURIZE_MODE: str = 'log_scalar0.00001' # 'scalar', 'log_scalar0.00001' (for 'line_count'), 'log_scalar0.36788' (0.36788 ~ 1/e, for 'count'), '1log+1_scalar', '1000exp-1_scalar', or 'vector'. 
INPUT_IMG_SIZE: ArrayLike = [224, 224] # Height and width of the input image. (width - axis 1, height - axis 0). Note: it is recommended to be [224, 224] for ResNet. Default: [128, 128]. 
INPUT_TRANSFORM: Optional[Compose] = (
    Compose([Resize(INPUT_IMG_SIZE), ToTensor(),])
) # Convert input data to "Tensor[`TENSOR_DTYPE`]". Could be None (no reshape needed). 

DATA_EXT: str = PM_PR_CL.SAVE_DICT_EXT # File format extension for clip data dictionary files. 
EXCLUDE_ZERO_LABELS: bool = False # Excluding all zero porosity cases. Only for debugging purpose. This should always be False when the pipeline is running in the normal mode. Do not touch this and turn this on! 
# ============================================================================ #


# ============================= For training ================================= #
BATCH_SIZE: int = 64 # Batch size of the training process. Accept user input. 
LR: float = 1e-4 # Learning rate of the training process. Accept user input. 
NUM_EPOCHS: int = 200 # Number of epochs of the training process. Accept user input. 
LOSS_FUNC: Callable = nn.MSELoss(reduction='sum') # Loss function for gradient calculation. Allow self-customization from the user. 

OPTIMIZER: "torch.optim.Optimizer" = torch.optim.Adam # Default: Adam, which is preferred in general. 
OPT_FOREACH: bool = True # For training optimizer. 
WEIGHT_DECAY: float = 1e-5 # For regularization. Default: 1e-5. 
GRAD_CLIP: Optional[float] = None # Default: float = 1. None if grad clip is not applied. 

IS_LR_SCHEDULER: bool = True # Turn on/off the learning rate scheduler, which reduces LR as the validation loss of the training process temporarily plateaus.
LR_SCHEDULER: "torch.optim.lr_scheduler._LRScheduler" = (
    torch.optim.lr_scheduler.ReduceLROnPlateau
) # The function (class) of learning rate scheduler. 
LR_SCHD_MODE: str = 'min' # 'min' or 'max', depending on whether the evaluated quantities are increasing (e.g., most loss-related quantities) or decreasing (e.g., most accuracy-related quantities). Default: 'min'. 
LR_SCHD_FACTOR: float = 0.8 # The factor (0 < x < 1) multiplied when reducing learning rate. In short: lr_new = lr_prev * factor. Default: 0.8. 
LR_SCHD_MIN_LR: float = 1e-6 # Minimum possible learning rate. Default: 1e-6. 
LR_SCHD_EPS: float = 1e-8 # Minimum possible change of the learning rate. Default: 1e-8. 
# ============================================================================ #


# ============================== For validation ============================== #
VALID_BATCH_NUM: int = 20 # How many batch iter to check in test dataloader. Mainly for inloop validation. Based on `BATCH_SIZE`. 
VALID_INLOOP_EVERY: int = 10 # How many batch iters for an in-loop check. In number of total batch iterations.   
EVAL_SAMPLE_NUM: int = 20 # How many samples used for in-loop model validation. 
CHECKPOINT_EVERY: int = 20 # Saving intermediate model every `CHECKPOINT_EVERY` epochs. 
UPDATE_SOTA_EVERY: int = 10 # Updating the relative state-of-the-art model every `UPDATE_SOTA_EVERY` epochs. This is also the period of learning rates updating for the learning rate scheduler. Default: 10. 
# ============================================================================ #


# ============================== For save & log ============================== #
CLEAR_LOG_DIR: bool = True # Whether to clear the logging directory for a new round of training. Default: True. 
RESULT_DIR: str = f"{CONFIG.get('eval')}/{CONFIG.get('results_fname')}" # The directory to store temporary training results as well as prior archived SOTA results. 
LOGGING_DIR: str = os.path.join(RESULT_DIR, CONFIG.get('train_log_fname')) # The sub-directory to store temporary results. Accept user input. 
MODEL_ARXIV_FNAME: str = "model_checkpoints" # Folder name containing all checkpoints model (number of checkpoints is controlled by `CHECKPOINT_EVERY` -- see above. )
LOG_NAME: str = "training_log.log" # Name of the training log file. 
MODEL_EXT: str = 'pth' # File format extension for archived models/checkpoints. 
SOTA_MODEL_NAME: str = f"model_optimal.{MODEL_EXT}" # File name of the state-of-the-art trained model (i.e., with the minimum validation loss perEpochSample). 
EPOCH_TRAIN_LOSS_SAVE_TO: str = "epoch_loss_list_train.npy" # File name for saving epoch training loss. 
EPOCH_VALID_LOSS_SAVE_TO: str = "epoch_loss_list_valid.npy" # File name for saving epoch validation loss. 
BATCH_TRAIN_LOSS_SAVE_TO: str = "batch_loss_list_train.npy" # File name for saving batch training loss. 
BATCH_VALID_LOSS_SAVE_TO: str = "batch_loss_list_valid.npy" # File name for saving batch validation loss. 
LR_SCHEME_SAVE_TO: str = "lr_list.npy" # File name for saving learning rate scheme. In particular useful when the learning rate scheduler is on. 
# ============================================================================ #
