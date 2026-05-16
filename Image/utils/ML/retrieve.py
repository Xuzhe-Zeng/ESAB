DIR_TO_TOP: str = "../.."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(2)

import torch.nn as nn
from torch.utils.data import Dataset
from typing import Union, Tuple, Optional

from .dataset import ScalogramToPorosityDataset

import config
CONFIG = config.read_config(config.CONFIG_PATH)
md_cnn2d = __import__(
    f"{CONFIG.get('training')}.{CONFIG.get('model_fname')}.CNN2d.cnn2d", 
    fromlist=['']
)
md_resnet = __import__(
    f"{CONFIG.get('training')}.{CONFIG.get('model_fname')}.ResNet.ResNet", 
    fromlist=['']
)
MyErr = __import__(
    f"{CONFIG.get('utils')}.exceptions", fromlist=['']
)


################################################################################
################################################################################
################################################################################


def get_NN_model(
    model_token: Union[str, Tuple[str, str]]
) -> Optional[Tuple[nn.Module, str]]:
    """
    Accept user input (tuple). 

    Args:
        model_token (Union[str, Tuple[str, str]]): _description_

    Raises:
        MyErr.UnrecognizableError: _description_

    Returns:
        Optional[Tuple[nn.Module, str]]: _description_
    """
    
    if isinstance(model_token, str): 
        if model_token in ['cnn2d', 'CNN2D']: 
            return (md_cnn2d.CNN2D, md_cnn2d.__file__)
        elif model_token in ['resnet', 'ResNet']: 
            return (md_resnet.ResNet, md_resnet.__file__)
        else: 
            raise MyErr.UnrecognizableError(
                "Unrecognizable neural network retrivation token. "
            )
    else: return None
        
def get_dataset(
    ds_token: Union[str, Tuple[str, str]]
) -> Optional[Tuple[Dataset, str]]:
    """
    Accept user input (tuple).

    Args:
        ds_token (Union[str, Tuple[str, str]]): _description_

    Raises:
        MyErr.UnrecognizableError: _description_

    Returns:
        Optional[Tuple[Dataset, str]]: _description_
    """
    
    if isinstance(ds_token, str): 
        if ds_token == 'ToPore': 
            dataset_module_path: str = os.path.abspath(
                sys.modules[ScalogramToPorosityDataset.__module__].__file__
            )
            return (ScalogramToPorosityDataset, dataset_module_path)
        else: 
            raise MyErr.UnrecognizableError(
                "Unrecognizable dataset retrivation token. "
            )
    else: return None


if __name__ == "__main__":
    """_summary_
    """
    
    pass

