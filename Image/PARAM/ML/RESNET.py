DIR_TO_TOP: str = "../.."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(2)

import torch.nn as nn
from typing import Any, Callable


################################################################################
################################################################################
################################################################################


MODEL_NAME: str = "resnet34" # Model type of ResNet. 'resnet18', 'resnet34', or 'resnet50'. Default: 'resnet18'. 
IS_PRETRAINED: bool = False # Whether to use a pretrained model or train from scratch. Default to True. 
IS_FINETUNE: bool = True # Default: True. Only useful when `IS_PRETRAINED` is True. True for the fine-tuning mode, where all/some weights of the pretrained ResNet are updated (see `FINETUNE_FREEZE_LAYERS_LIST`); this is common for unfamiliar new dataset. False for the feature extraction mode, where only weights of the final fully-connected layers are updated; this is common for similar but small new dataset. 
FINETUNE_FREEZE_LAYERS_LIST: list[str] = [
    # 'conv1.weight', 'bn1.weight', 'bn1.bias', # Input layer. 
    # 'layer1', 'layer2', 'layer3', # Blocks. 
] # Useful only when both `IS_PRETRAINED` and `IS_FINETUNE` are True. List of block/layer names selected to be frozen for fine-tuning. Weights of any blocks/layers not with their names in this list will be optimized (i.e., equivalent to requires_grad = True). Might vary as `MODEL_NAME` or `MLP_STRUCT` changes. Could be [] (no parameteres are frozen). 

MLP_STRUCT: dict[Any] = {
    # The input layer of fc block should be `layer 0` (0: {Any}); 
    1: 128, 
    2: 16, # Optional. 
    # The output layer of fc block. 
} # Hidden structure (excluding the input and output layers) of the FC block (appended after the convolutional block). Default: [256, 64]. Starting from 1. Default to {}, meaning there is no hidden structure of fully-connected blocks. 

IS_DROPOUT: bool = False # Whether to include dropout layers. 
DROPOUT_LAYER: Callable = nn.Dropout(0.5)
