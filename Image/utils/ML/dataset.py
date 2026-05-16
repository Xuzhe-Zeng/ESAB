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
from numpy.typing import ArrayLike
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
MyErr = __import__(
    f"{CONFIG.get('utils')}.exceptions", fromlist=['']
)


################################################################################
################################################################################
################################################################################


class ScalogramToPorosityDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
        self: "ScalogramToPorosityDataset", 
        data_dir: Optional[str] = None, # Training and testing should have separate directories. If inputting data repo directly, set this to None. 
        shuffle: bool = PM_ML.DATASET_SHUFFLE, 
        max_dataset_size: Optional[int] = PM_ML.MAX_DATASET_SIZE,
        **kwargs: Optional[dict[Any]], 
    ): 
        """_summary_

        Args:
            self (ScalogramToPorosityDataset): _description_
            data_dir (Optional[str], optional): _description_. Defaults to None.
            max_dataset_size (Optional[int], optional): _description_. Defaults to PM_ML.MAX_DATASET_SIZE.
        """
        
        super(ScalogramToPorosityDataset, self).__init__()
        self.kwargs: Optional[dict[Any]] = kwargs
        self.__DEBUG: bool = self.kwargs.get('DEBUG', False)
        
        self.data_dir: Optional[str] = data_dir
        self.shuffle: bool = shuffle
        self.max_dataset_size: Optional[int] = max_dataset_size
        
        self._data_repo_dict: dict[Any] = self.kwargs.get('data_repo_dict', {}) # Allow direct input of data repo dict, if there is one. 
        self._data_size: int = len(self)
        self._data_ext: str = self.kwargs.get('data_ext', PM_ML.DATA_EXT)
        self._inputs: list[Tuple[str]] = (
            self.kwargs.get('inputs', PM_ML.INPUT_LIST)
        ) # Options in ['ACC', 'PHO']. Currently not used but could be useful for future. 
        self._outputs: list[str] = (
            self.kwargs.get('outputs', PM_ML.OUTPUT_LIST)
        ) # Options in ['POR']. Currently not used but could be useful for future. 
        self._input_transform: Optional[Compose] = (
            self.kwargs.get('input_transform', PM_ML.INPUT_TRANSFORM)
        )
        
        self._input_channel_n: Optional[int] = None
        self._output_dim: Optional[int] = None
        
        # Backbones. 
        if self._data_size == 0: # When there is no user input of `data_repo_dict`. 
            self._set_datarepo(
                shuffle=self.shuffle, 
                max_dataset_size=self.max_dataset_size
            )
        else: pass
        self._init()
    
    @classmethod
    def from_files(
        cls: "ScalogramToPorosityDataset", 
        clip_path_list: ArrayLike, 
        shuffle: bool = PM_ML.DATASET_SHUFFLE, 
    ) -> "ScalogramToPorosityDataset":
        """
        Alternative method to create a dataset from a list of clip filepaths. 
        Argument `max_dataset_size` is not available for this construction method.

        Args:
            cls (ScalogramToPorosityDataset): _description_
            clip_path_list (ArrayLike): _description_
            shuffle (bool, optional): _description_. Defaults to PM_ML.DATASET_SHUFFLE.

        Returns:
            ScalogramToPorosityDataset: _description_
        """
        
        if shuffle: np.random.shuffle(clip_path_list)
        else: pass
        
        _data_repo_dict: dict[Any] = defaultdict(util.default_dict)
        data_num: int = 0 # Set up in case there is data augmentation implemented. 
        for i, data_filepath in enumerate(clip_path_list):
            if PM_ML.EXCLUDE_ZERO_LABELS:
                por_prop: str = PM_ML.OUTPUT_LIST[-1]
                data_dict: dict[Any] = (
                    util.load_dict_from_file(data_filepath)
                )
                if data_dict[CONFIG['POR']][por_prop] == 0: continue
                else: pass
            else: pass
            _data_repo_dict[data_num] = copy.deepcopy(data_filepath)
            data_num += 1
        
        return cls(data_repo_dict=_data_repo_dict)
    
    def __len__(self: "ScalogramToPorosityDataset") -> int:
        """_summary_

        Args:
            self (ScalogramToPorosityDataset): _description_

        Returns:
            int: _description_
        """
        
        return len(self._data_repo_dict)
        
    def __getitem__(
        self: "ScalogramToPorosityDataset", 
        key: int
    ) -> Optional[dict[Any]]:
        """
        Assuming `data_repo_dict` has been established. 

        Args:
            self (ScalogramToPorosityDataset): _description_
            key (int): _description_

        Returns:
            Optional[dict[Any]]: _description_
        """
        
        assert isinstance(key, int) and key >= 0, (
            "Input data retrieving key must be an non-negative integer. "
        )
        
        try: 
            data_filepath: str = self.data_repo_dict[key] # Retrieving the specific clip data dict given the key. 
            data_dict: dict[Any] = copy.deepcopy(
                util.load_dict_from_file(data_filepath)
            ) # Retrieving the specific clip data dict given the key. 
            
            input_val: Tensor = (
                self._input_featurization(
                    data_dict=data_dict, 
                    process_mode=PM_ML.INPUT_FEATURIZE_MODE
                )
            )
            output_val: Tensor = (
                self._output_featurization(
                    data_dict=data_dict, 
                    process_mode=PM_ML.OUTPUT_FEATURIZE_MODE
                )
            )
            return {
                PM_ML.INPUT_TOKEN: input_val, 
                PM_ML.OUTPUT_TOKEN: output_val,
                'filename': Path(data_filepath).name,
            }
        except KeyError: return None
        
    def _init(self: "ScalogramToPorosityDataset"):
        """
        Assuming `data_repo_dict` has been established. 

        Args:
            self (ScalogramToPorosityDataset): _description_

        Returns:
            _type_: _description_
        """
        
        data_sample: dict[Any] = self[0]
        if data_sample is None: 
            if self.__DEBUG: print("This is an empty dataset. ")
            pass # Empty dataset. 
        else: 
            self._input_channel_n = data_sample['input'].size(0) # For torch.Tensor, the first dim would be channel_n. 
            self._output_dim = data_sample['output'].size(0)

    def _input_featurization(
        self: "ScalogramToPorosityDataset", 
        data_dict: dict[Any],
        process_mode: str = PM_ML.INPUT_FEATURIZE_MODE, 
        **addi_items: Optional[dict[Any]], 
    ) -> Any:
        """_summary_

        Args:
            self (ScalogramToPorosityDataset): _description_
            data_dict (dict[Any]): _description_
            process_mode (str, optional): _description_. Defaults to PM_ML.INPUT_FEATURIZE_MODE.

        Raises:
            MyErr.InvalidError: _description_

        Returns:
            Any: _description_
        """
        
        input_val_list: list[Any] = []
        for mod_item in self._inputs:
            val: dict[Any] = copy.deepcopy(data_dict)
            for key_id, key in enumerate(mod_item): 
                if key_id == len(mod_item) - 1 and '$' in key: # Only for scalar multiplication of the last key. 
                    true_key: str = key.split('$')[0]
                    scalar: float = float(key.split('$')[-1])
                    val = copy.deepcopy(val.get(true_key) * scalar)
                else: val = copy.deepcopy(val.get(key))
            input_val_list.append(val)
        
        # Process input data with specific modes. 
        processed_val_list: list[Any] = []
        if process_mode == 'img': # Convert all formats of data to PIL image. 
            for input_val in input_val_list:
                if isinstance(input_val, PIL.Image.Image): 
                    processed_val_list.append(np.asarray(
                        input_val.resize(PM_ML.INPUT_IMG_SIZE).convert('L')
                    ).astype(np.uint8))
                elif isinstance(input_val, (int, float)): # Convert scalar to a PIL image.
                    processed_val_list.append(np.asarray(
                        np.ones(tuple(PM_ML.INPUT_IMG_SIZE)) * input_val
                    ).astype(np.uint8))
                elif isinstance(input_val, (list, tuple, np.ndarray)): 
                    # Need to code the resize & interpolation part. 
                    pass
                else: raise MyErr.InvalidError("Invalid input value type. ")
            img_arr_tuple: Tuple[np.ndarray[np.uint8]] = (
                tuple(processed_val_list)
            )
            img_pil_stacked: PIL.Image.Image = PIL.Image.fromarray(
                np.squeeze(np.uint8(np.dstack(img_arr_tuple)))
            )
            if self._input_transform is not None: 
                input_val: "Tensor[PM_ML.TENSOR_DTYPE]" = (
                    copy.deepcopy(
                        self._input_transform(
                            img_pil_stacked
                        ).to(PM_ML.TENSOR_DTYPE)
                    )
                )
            else: input_val: PIL.Image.Image = copy.deepcopy(img_pil_stacked)
        else: raise MyErr.InvalidError("Invalid input processing mode. ")
        
        return input_val
    
    def _output_featurization(
        self: "ScalogramToPorosityDataset", 
        data_dict: dict[Any], 
        process_mode: str = PM_ML.OUTPUT_FEATURIZE_MODE
    ) -> Any:
        """_summary_

        Args:
            self (ScalogramToPorosityDataset): _description_
            data_dict (dict[Any]): _description_

        Returns:
            Any: _description_
        """
        
        val: dict[Any] = copy.deepcopy(data_dict)
        for key in self._outputs: val = copy.deepcopy(val.get(key))
        
        if process_mode == 'scalar':
            output_val: Tensor = copy.deepcopy(
                torch.from_numpy(np.asarray([val,]).astype(float))
            )
        elif 'log_scalar' in process_mode:
            # The suffix float number is for zero replacement. 
            replacing_zero: float = util.str2num(process_mode)[0]
            val: float = copy.deepcopy(
                np.log(val) if val != 0. else np.log(replacing_zero)
            )
            output_val: Tensor = copy.deepcopy(
                torch.from_numpy(np.asarray([val,]).astype(float))
            )
        elif 'log' in process_mode and '_scalar' in process_mode:
            # The first float number should be the scalar; the second number should be the phase shift.
            alpha, beta = util.str2num(process_mode)  
            val: float = copy.deepcopy(alpha*np.log(val+beta))
            output_val: Tensor = copy.deepcopy(
                torch.from_numpy(np.asarray([val,]).astype(float))
            )
        elif 'exp' in process_mode and '_scalar' in process_mode:
            # The first float number should be the scalar; the second number should be the phase shift.
            alpha, beta = util.str2num(process_mode)  
            val: float = copy.deepcopy(alpha*(np.exp(val)+beta))
            output_val: Tensor = copy.deepcopy(
                torch.from_numpy(np.asarray([val,]).astype(float))
            )
        elif process_mode == 'vector': 
            output_val: Tensor = copy.deepcopy(
                torch.from_numpy(val.astype(float))
            )
        else: raise MyErr.InvalidError("Invalid output processing mode. ")
        
        return output_val
    
    def _set_datarepo(
        self: "ScalogramToPorosityDataset", 
        shuffle: bool = True, 
        max_dataset_size: Optional[int] = None,
    ):
        """
        When there is no/empty input of `data_repo_dict`, load data information (clip filepaths) from the given repository directory.

        Args:
            self (ScalogramToPorosityDataset): _description_
            shuffle (bool, optional): _description_. Defaults to True.
            max_dataset_size (Optional[int], optional): _description_. Defaults to None.
        """
        
        # This is the only place where `self._data_repo_dict` can be changed. 
        del self.data_repo_dict # Invoke `deleter` of `self.data_repo_dict`. 
        self.data_repo_dict: dict[Any] = defaultdict(util.default_dict) # Invoke `setter` of `self.data_repo_dict`.
        
        # Reach to clip file level. 
        try: 
            data_file_allist: list[str] = util.get_paths_of_ext(
                dir_path=self.data_dir, file_ext=self._data_ext
            )
        except TypeError: data_file_allist: list[str] = []
        if shuffle: np.random.shuffle(data_file_allist)
        else: pass
        
        data_num: int = 0
        for i, data_filepath in enumerate(data_file_allist):
            if max_dataset_size is not None and data_num >= max_dataset_size: 
                break
            else: 
                if PM_ML.EXCLUDE_ZERO_LABELS:
                    por_prop: str = PM_ML.OUTPUT_LIST[-1]
                    data_dict: dict[Any] = (
                        util.load_dict_from_file(data_filepath)
                    )
                    if data_dict[CONFIG['POR']][por_prop] == 0: continue
                    else: pass
                else: pass
                self._data_repo_dict[data_num] = copy.deepcopy(data_filepath)
                data_num += 1

    @property
    def input_channel_n(self: "ScalogramToPorosityDataset") -> Optional[int]:
        """_summary_

        Args:
            self (ScalogramToPorosityDataset): _description_

        Raises:
            MyErr.InvalidError: _description_
            MyErr.InvalidError: _description_

        Returns:
            Optional[int]: _description_
        """
        
        return self._input_channel_n
    
    @property
    def output_dim(self: "ScalogramToPorosityDataset") -> Optional[int]:
        """_summary_

        Args:
            self (ScalogramToPorosityDataset): _description_

        Raises:
            MyErr.InvalidError: _description_
            MyErr.InvalidError: _description_

        Returns:
            Optional[int]: _description_
        """
        
        return self._output_dim
    
    @property
    def data_repo_dict(self: "ScalogramToPorosityDataset") -> dict[Any]:
        """_summary_

        Args:
            self (ScalogramToPorosityDataset): _description_

        Returns:
            dict[Any]: _description_
        """
        
        return self._data_repo_dict
    
    @data_repo_dict.setter
    def data_repo_dict(self: "ScalogramToPorosityDataset", val: Any):
        """_summary_

        Args:
            self (ScalogramToPorosityDataset): _description_
            val (Any): _description_
        """
        
        self._data_repo_dict: Any = copy.deepcopy(val)
        
    @data_repo_dict.deleter
    def data_repo_dict(self: "ScalogramToPorosityDataset"):
        """_summary_

        Args:
            self (ScalogramToPorosityDataset): _description_
        """
        
        if hasattr(self, "_data_repo_dict"): delattr(self, "_data_repo_dict")
        else: pass
    
def set_dataloader(
    dataset_obj: Dataset, 
    batch_size: int = 1, 
    indices_list: Optional[list[int]] = None, 
    **kwargs: Optional[dict[Any]]
) -> DataLoader:
    """_summary_

    Args:
        dataset_obj (Dataset): Dataset object. 
        batch_size (int, optional): _description_. Defaults to 1.
        indices_list (Optional[list[int]], optional): _description_. Defaults to None.

    Returns:
        DataLoader: _description_
    """
    
    user_sampler_func: Sampler = kwargs.get(
        'sampler_func', torch.utils.data.SubsetRandomSampler
    ) # `SubsetRandomSampler` randomly picks subsets (size=batch_size) of input indices. Input indices therefore does not need to be shuffled. 
    
    if indices_list is None or len(indices_list) == 0: 
        dataset_size: int = len(dataset_obj)
        indices_list: list[int] = list(range(dataset_size))
    else: pass
    data_sampler: Sampler = user_sampler_func(indices_list)
    
    return torch.utils.data.DataLoader(
        dataset_obj, batch_size=batch_size, sampler=data_sampler
    )

def dataset_partition(
    data_dir: str, 
    partition: Union[float, Tuple[float,float]],
    shuffle: bool = True, 
    ml_data_savedir: str = PM_ML.ML_DATA_DIR
) -> Tuple[str, str, str]:
    """
    Used to partition & generate train(, valid), and test subdatasets from a unified data directory. 

    Args:
        data_dir (str): The directory containing all data samples in some format. Non-related files should not be included. Just a pure, clean folder. 
        partition (Union[float, Tuple[float,float]]): Ratios of train/test or train/valid/test. 
        shuffle (bool, optional): Whether to shuffle the data before partition. Default: True. 
        ml_data_savedir (str, optional): The directory to create and save train(, valid) and test data subfolders. Default: PM_ML.ML_DATA_DIR
    """
    
    def _move_dataset_mapping(
        src_dir: str, 
        dataset_ind_list: list[int],
        dst_dir: str, 
    ):
        """_summary_

        Args:
            src_dir (str): _description_
            dataset_ind_list (list[int]): _description_
            dst_dir (str): _description_
        """
        
        if not os.path.isdir(dst_dir): util.set_dirtree(dst_dir)
        else: pass
        
        data_path_list: list[str] = glob.glob(os.path.join(src_dir, '*'))
        for ind in dataset_ind_list:
            path: str = data_path_list[ind]
            if not os.path.isdir(path): 
                shutil.copy(path, os.path.join(dst_dir, Path(path).name))
            else: 
                shutil.copytree(path, os.path.join(dst_dir, Path(path).name))
    
    if isinstance(partition, float):
        train_ratio: float = partition
        valid_ratio: float = 0.
    elif util.is_arraylike(partition):
        partition: Tuple[float] = copy.deepcopy(tuple(partition))
        assert len(partition) in [1,2], (
            "Number of input partition ratios must be 1 or 2. "
        )
        if len(partition) == 1:
            train_ratio: float = partition[0]
            valid_ratio: float = 0.
        else: # len(partition) == 2. 
            train_ratio, valid_ratio = partition
    else: 
        raise MyErr.InvalidError(
            "Invalid variable type for input argument `partition`. "
        )
    
    data_sample_num: int = len(glob.glob(os.path.join(data_dir, '*'))) # List all objects under the `data_dir` folder. 
    data_ind_list: list[int] = list(range(data_sample_num))
    if shuffle: np.random.shuffle(data_ind_list)
    else: pass
    
    train_ind: int = int(train_ratio*data_sample_num)
    valid_ind: int = int((train_ratio+valid_ratio)*data_sample_num)
    train_ind_list: list[int] = data_ind_list[:train_ind]
    valid_ind_list: list[int] = data_ind_list[train_ind:valid_ind] # Could be an empty list. 
    test_ind_list: list[int] = data_ind_list[valid_ind:]
    
    train_data_savedir: str = os.path.join(
        ml_data_savedir, CONFIG.get('train_data')
    )
    valid_data_savedir: str = os.path.join(ml_data_savedir, 'valid') # Could be omitted. 
    test_data_savedir: str = os.path.join(
        ml_data_savedir, CONFIG.get('test_data')
    )
    
    _move_dataset_mapping(data_dir, train_ind_list, train_data_savedir)
    _move_dataset_mapping(data_dir, valid_ind_list, valid_data_savedir)
    _move_dataset_mapping(data_dir, test_ind_list, test_data_savedir)
    
    return train_data_savedir, valid_data_savedir, test_data_savedir


if __name__ == "__main__":
    """_summary_
    """
    
    pass

