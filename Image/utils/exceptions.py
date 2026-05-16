DIR_TO_TOP: str = ".."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(1)


# ======= Self-defined exceptions =======
# ---------------------------------------


class NoneError(Exception):
    """_summary_

    Exception when the object is None. 
    """
    
    pass

class DataProcessingError(Exception):
    """_summary_

    Exception when data processing is problematic. 
    """
    
    pass

class DataLoadingError(Exception):
    """_summary_

    Exception when data loading is problematic. 
    """
    
    pass

class DataSavingingError(Exception):
    """_summary_

    Exception when data saving is problematic. 
    """
    
    pass

class UnrecognizableError(Exception):
    """_summary_

    Exception when things are unrecognizable. 
    """
    
    pass

class InvalidError(Exception):
    """_summary_

    Exception when input is invalid. 
    """
    
    pass

class MissingFilesError(Exception):
    """_summary_

    Exception when files are missing. 
    """
    
    pass