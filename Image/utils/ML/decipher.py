DIR_TO_TOP: str = "../.."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(2)

import ast
from typing import Union, Tuple, Any, Optional


################################################################################
################################################################################
################################################################################


def log_decipher(
    log_filepath: str, 
    *args: Optional[Tuple[str]],
) -> dict[Any]:
    """
    Extract useful information (variables/parameters) from the training log file. 
    Used together with the corresponding `save_log` function in the training module. 

    Args:
        log_filepath (str): _description_

    Raises:
        FileNotFoundError: _description_

    Returns:
        dict[Any]: _description_
    """
    
    if not os.path.exists(log_filepath): 
        raise FileNotFoundError("Missing training log file for evaluation. ")
    else: pass
    
    with open(log_filepath, 'r') as f: lines = f.read().splitlines()
    
    return_dict: dict[Optional[Any]] = {}
    for keyword in [*args]:
        for line in lines: 
            if keyword in line: 
                if keyword in [
                    "Device", "Dataset module path",
                    "NN module path", "Input format", 
                    "Output format", "Training module type", 
                ]: 
                    val_temp: str = line.split(': ')[-1]
                elif keyword in ["Dataset type", "NN model type"]:
                    val_temp: Tuple[Union[str, Tuple[str, str]]] = (
                        ast.literal_eval(line.split(': ')[-1])[0]
                    )
                elif keyword == "Input dimension":
                    val_temp: Tuple[float] = (
                        ast.literal_eval(line.split(': ')[-1])
                    )
                elif keyword == "Input":
                    val_temp: list[Tuple[str]] = (
                        ast.literal_eval(line.split(': ')[-1])
                    )
                elif keyword in [
                    "Dataset size", "Output dimension", "Batch size", 
                    "Learning rate", "Epoch number", 
                ]:
                    val_temp: float = float(line.split(': ')[-1])
                else: val_temp = None
                return_dict[keyword] = val_temp
            else: continue
    
    return return_dict

