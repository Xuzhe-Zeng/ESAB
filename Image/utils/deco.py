DIR_TO_TOP: str = ".."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(1)

from datetime import datetime
from typing import Any, Callable


##################################################################################
##################################################################################
##################################################################################


def log_datetime(func: Callable) -> Callable:
    """_summary_

    Args:
        func (Callable): _description_

    Returns:
        Callable: _description_
    """
    
    now = datetime.now()
    now_string: str = now.strftime('%Y-%m-%d %H:%M:%S')
    
    def _func_wrapper(*args: tuple[str]) -> str:
        """_summary_

        Returns:
            str: _description_
        """
        
        ori_string: str = func(*args)
        with_datetime_string: str = (
            "\n" +
            "*************************\n" + 
            f"{ori_string}\n" + 
            f"Process datetime: {now_string}"
        )
        
        return with_datetime_string
    
    return _func_wrapper


