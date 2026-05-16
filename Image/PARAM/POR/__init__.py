import os


filename_in_this_package = os.listdir(os.path.dirname(os.path.abspath(__file__)))

__all__ = [filename.split('.')[0] for filename in filename_in_this_package
           if filename.split('.')[-1] == 'py']