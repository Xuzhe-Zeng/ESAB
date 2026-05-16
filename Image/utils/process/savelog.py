DIR_TO_TOP: str = "../.."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(2)

import copy
import config

from pathlib import Path
from typing import Any
CONFIG = config.read_config(config.CONFIG_PATH)

util = __import__(
    f"{CONFIG.get('utils')}.util", fromlist=['']
)
udec = __import__(
    f"{CONFIG.get('utils')}.deco", fromlist=['']
)


################################################################################
################################################################################
################################################################################


class ProcessLog(object):
    """_summary_

    Args:
        object (_type_): _description_
    """
    
    def __init__(
        self: "ProcessLog", 
        log_path: str, 
        overwrite: bool = False, 
        **kwargs: dict[Any]
    ):
        """_summary_

        Args:
            self (ProcessLog): _description_
            log_path (str): _description_
            overwrite (bool, optional): _description_. Defaults to False.
        """
        
        self.kwargs: dict[Any] = kwargs
        
        self.log_path: str = log_path
        self._log_line_list: list[str] = [
            "Multi-modality data processing log"
        ]
        
        # Backbones. 
        util.set_dirtree(root_dir=str(Path(self.log_path).parent))
        if not overwrite: self._preload()
        else: pass

    def _preload(self: "ProcessLog"):
        """_summary_

        Args:
            self (ProcessLog): _description_
        """
        
        if not os.path.exists(self.log_path): pass
        else: 
            with open(self.log_path, 'rb') as f:
                self._log_line_list += f.read().splitlines()
            f.close()
        self._log_line_list += ['\n']
    
    @staticmethod
    @udec.log_datetime
    def NewModule(*args: tuple[str]) -> str:
        """
        Generate log header strings for a new processing module. 

        Returns:
            str: _description_
        """
        
        header_string: str = '\n'.join([header_line for header_line in args])
        return f"Process Module -- {header_string}"
    
    @staticmethod
    def EndModule(*args: tuple[str]) -> str:
        """_summary_

        Returns:
            str: _description_
        """
        
        suffix_list: list[str] = ["*************************"] * 2
        suffix_list[1:1] = copy.deepcopy([suffix_line for suffix_line in args])
        return '\n'.join(suffix_list)
    
    def update(self: "ProcessLog", *args: tuple[str]):
        """_summary_

        Args:
            self (ProcessLog): _description_
        """
        
        self._log_line_list += [str(line) for line in args]
        
    def dump(self: "ProcessLog"):
        """_summary_

        Args:
            self (ProcessLog): _description_
        """
        
        content: str = '\n'.join(self._log_line_list)
        with open(self.log_path, 'w') as f: f.write(content)
        f.close()
        

if __name__ == "__main__":
    """_summary_
    """
    
    pass

