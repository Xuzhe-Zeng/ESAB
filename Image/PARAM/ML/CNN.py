DIR_TO_TOP: str = "../.."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(2)

import torch.nn as nn
from typing import Union, Tuple, Optional, Callable, Any

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


def _CONV_STRUCT_DICT(*args: Tuple[Any]) -> dict[Any]:
    """
    Input format (check more details in return). 

    Returns:
        dict[Any]: _description_
    """
    
    assert len(args) == 7, (
        "Missing input parameters for convolutional layer construction. "
    )
    
    return {
        'in_channels': args[0],
        'out_channels': args[1],
        'kernel_size': args[2],
        'stride': args[3] if args[3] is not None else 1,
        'padding': args[4] if args[4] is not None else 0,
        'dilation': args[5] if args[5] is not None else 1,
        'bias': args[6],
    } # Parameterized inputs. 

# Default conv layer parameters. 
FIRST_CONV_CHANNEL_N: int = 8 # Output channel number of the 1st convolutional layer (conv block `layer 0`)/input channel number of the 2nd convolutional layer (conv block `layer 1`). Default: 8. 
KERNEL: Union[Tuple[int,int], int] = (3,3) # Default: (3,3) or 3. (height -- axis 0(, weight -- axis 1)). Input single integer a = (a,a). 
STRIDE: Union[Tuple[int,int], int] = (1,1) # Default: (1,1) or 1. (height -- axis 0(, weight -- axis 1)). Input single integer a = (a,a). Could be None. 
PADDING: Union[Tuple[int,int], int] = (1,1) # Default: (1,1) or 1. (height -- axis 0(, weight -- axis 1)). Input single integer a = (a,a). Could be None. 
DILATION: Union[Tuple[int,int], int] = (1,1) # Default: (1,1) or 1. (height -- axis 0(, weight -- axis 1)). Input single integer a = (a,a). Could be None. 
BIAS: bool = True # On/off bias for each conv layer. Off when there is a batchnorm layer immediately after the conv layer (i.e., `IS_CONV_BATCHNORM` = True). 

CONV_STRUCT: dict[dict[Optional[Union[int, Tuple[int,int]]]]] = {
    # This does not include the first conv layer -- the input layer. The input layer of conv block should be interpreted as `layer 0` (0: {Any}), but is not supposed to appear here; 
    1: _CONV_STRUCT_DICT(
        FIRST_CONV_CHANNEL_N, 16, KERNEL, STRIDE, PADDING, DILATION, BIAS, 
    ), # The first hidden conv layer. 
    2: _CONV_STRUCT_DICT(
        16, 32, KERNEL, STRIDE, PADDING, DILATION, BIAS, 
    ), 
    3: _CONV_STRUCT_DICT(
        32, 64, KERNEL, STRIDE, PADDING, DILATION, BIAS, 
    ), 
    4: _CONV_STRUCT_DICT(
        64, 128, KERNEL, STRIDE, PADDING, DILATION, BIAS
    ), 
    # 5: _CONV_STRUCT_DICT(
    #     128, 256, KERNEL, STRIDE, PADDING, DILATION, BIAS
    # ), 
 } # Hidden structure (excluding the input layer -- layer 0) of Conv block. Change in/out channel number here. Tuple structure: (in_channel_n, out_channel_n, kernel, stride, padding, dilation, bias). Could be {}. Tuple format: (). Allow user-customization/hard coding. 
MLP_STRUCT: dict[Any] = {
    # The input layer of mlp block should be `layer 0` (0: {Any}); 
    1: 128, 
    2: 64, 
    # The output layer of mlp block. 
} # Hidden structure (excluding the input and output layers) of MLP block (appended after the convolutional block). Default: [256, 64]. Could be {}. 

IS_CONV_BATCHNORM: bool = True # nn.BatchNorm2d()
IS_MLP_BATCHNORM: bool = False # nn.BatchNorm1d()
IS_DROPOUT: bool = False # Whether to include dropout layers. 
DROPOUT_LAYER: Callable = nn.Dropout(0.5)
IS_CONV_POOLING: bool = True # Whether to apply pooling layer after conv layer. 
POOLING_LAYER: Callable = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # 2D pooling layer for conv block. No parameterization for the pooling layer -- keep it constant all the time. 
