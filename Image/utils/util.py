DIR_TO_TOP: str = ".."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(1)

import copy
import collections
import importlib.util
import numpy as np
import matplotlib.figure
import matplotlib.axes
import pickle
import re
import scipy.io
import shutil

from collections import defaultdict
from pathlib import Path
from numpy.typing import ArrayLike, DTypeLike
from typing import Union, Tuple, Any, Callable, Optional

from .exceptions import (
    DataLoadingError, DataSavingingError, DataProcessingError, InvalidError
)


################################################################################
################################################################################
################################################################################


def clr_dir(directory: str):
    """_summary_

    Args:
        directory (str): _description_
    """
    
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        if os.path.isdir(path): shutil.rmtree(path)
        else: os.remove(path)
        
def new_folder(dir: str, clear: bool = False):
    """_summary_

    Args:
        dir (str): _description_
        clear (bool, optional): _description_. Defaults to False.
    """
    
    if not os.path.isdir(dir): os.mkdir(dir)
    if clear: clr_dir(dir)

def set_dirtree(
    root_dir: str, 
    leaf_dir: Optional[Union[str, list[str]]] = None
):
    """_summary_

    Args:
        root_dir (str): _description_
        leaf_dir (Union[str, list[str], None], optional): _description_. Defaults to None.
    """
    
    assert (
        type(root_dir) == str and (
            leaf_dir is None or 
            type(leaf_dir) == str or 
            type(leaf_dir) == list
        )
    ), "Unrecognized input format. Try string or list of strings. "
    
    if leaf_dir is None: 
        os.makedirs(root_dir, exist_ok=True)
    else:
        if type(leaf_dir) == str:
            os.makedirs(os.path.join(root_dir, leaf_dir), exist_ok=True)
        else: 
            for leaf in leaf_dir:
                full_dir_temp = os.path.join(root_dir, leaf)
                os.makedirs(full_dir_temp, exist_ok=True)

def normalize_1d(
    data: Union[np.ndarray[float], list[float]], 
    avoid_zero: bool = False
) -> np.ndarray[float]:
    """_summary_

    Args:
        data (Union[np.ndarray[float], list[float]]): _description_
        avoid_zero (bool, optional): _description_. Defaults to False.

    Returns:
        np.ndarray[float]: _description_
    """
    
    assert is_subscriptable(data), (
        "Wrong type of 1d data input. Try either 1D Numpy array or 1D list. "
    )
    
    if type(data) != np.ndarray: 
        array: np.ndarray[Any] = np.asarray(data).reshape(-1)
    else: array: np.ndarray[Any] = data.reshape(-1)
    
    max_val, min_val = np.max(array), np.min(array)
    
    if avoid_zero: 
        return ((array-9.*min_val/10.)/(max_val-9.*min_val/10.)).reshape(-1) # Assuming there is no -min_val/10.
    else: return ((array-min_val)/(max_val-min_val)).reshape(-1)

def default_dict() -> defaultdict:
    """Callable (pickable) parameterized dictionary structure. 

    Returns:
        defaultdict: _description_
    """
    
    return defaultdict(dict)

def shifting(
    data_matrix: np.ndarray[float], 
    axis: np.ndarray[float], 
    origin: np.ndarray[float], 
    target: np.ndarray[float]
) -> np.ndarray[float]:
    """_summary_

    Args:
        data_matrix (np.ndarray[float]): _description_
        axis (np.ndarray[float]): _description_
        origin (np.ndarray[float]): _description_
        target (np.ndarray[float]): _description_

    Returns:
        np.ndarray[float]: _description_
    """
    
    if axis == 0: data_new = copy.deepcopy(data_matrix)
    elif axis == 1: data_new = copy.deepcopy(data_matrix.T)
    else: data_new = copy.deepcopy(origin - target)

    data_new += (target - origin)

    if axis == 1: return data_new.T
    else: return data_new

def normalization(
    data_matrix: ArrayLike, 
    axis: int, 
    mode: str ='unitize'
) -> ArrayLike:
    """
    """
    
    if mode == 'unitize':
        if axis == 1: data_matrix_normalized = copy.deepcopy(data_matrix.T)
        else: data_matrix_normalized = copy.deepcopy(data_matrix)

        norm_vect = np.linalg.norm(data_matrix_normalized, axis=0)
        data_matrix_normalized /= norm_vect

        if axis == 1: return data_matrix_normalized.T
        else: return data_matrix_normalized

    else: return data_matrix

def standardization(matrix, axis, mean_vect, std_vect):
    """
    `mean_vect` and `std_vect` must be 1d array. 
    """

    std_vect = np.where(std_vect==0., 1e-5, std_vect) # Replace 0 with a small number to avoid numerical issue. 
    
    if axis == 1: matrix_standardized = copy.deepcopy(matrix.T)
    else: matrix_standardized = copy.deepcopy(matrix)

    matrix_standardized = (matrix_standardized - mean_vect) / std_vect

    if axis == 1: return matrix_standardized.T
    else: return matrix_standardized

def de_standardization(matrix, axis, mean_vect, std_vect):
    """
    `mean_vect` and `std_vect` must be 1d array. 
    """

    if axis == 1: matrix_de_standardized = copy.deepcopy(matrix.T)
    else: matrix_de_standardized = copy.deepcopy(matrix)

    matrix_de_standardized = matrix_de_standardized * std_vect + mean_vect

    if axis == 1: return matrix_de_standardized.T
    else: return matrix_de_standardized
    
def angle_2vect(
    vect_1: np.ndarray[float], 
    vect_2: np.ndarray[float],
    direct: bool = True
) -> float:
    """
    Both shape = (-1,).

    Args:
        vect_1 (np.ndarray[float]): The anchor vector. 
        vect_2 (np.ndarray[float]): The movable vector. 
        direct (bool, optional): 
            Whether the result is a signed float, the sign of which indicates the direction from `vect_2` to `vect_1`. 
            Assuming counter-clockwise is positive. Defaults to True.

    Returns:
        float: Return the value in (signed, if direct=True) degrees.
    """

    cos_theta: float = (
        np.dot(vect_1, vect_2) / 
        (np.linalg.norm(vect_1)*np.linalg.norm(vect_2))
    )
    theta_rad: float = np.arccos(cos_theta)
    
    if direct: 
        wedge: float = np.cross(vect_2, vect_1)
        theta_rad = np.sign(wedge) * theta_rad
    else: pass

    return theta_rad * 180. / np.pi # [0., 180.], in degree. 

def A_diff_B_array(A, B) -> np.ndarray[float]:
    """
    A, B must be axis-0-dominant matrices. 
    """

    return np.array([
        data for data in (
            set(tuple(pt_A) for pt_A in A) - 
            set(tuple(pt_B) for pt_B in B)
        )
    ])

def A_intersect_B_array(A, B):
    """
    A, B must be axis-0-dominant matrices. 
    """

    return np.array([
        data for data in (
            set(tuple(pt_A) for pt_A in A) & 
            set(tuple(pt_B) for pt_B in B)
        )
    ])

def A_and_B_array(A, B):
    """
    A, B must be axis-0-dominant matrices. 
    """

    return np.array([
        data for data in (
            set(tuple(pt_A) for pt_A in A) | 
            set(tuple(pt_B) for pt_B in B)
        )
    ])

def A_symm_diff_B_array(A, B):
    """
    A, B must be axis-0-dominant matrices. 
    """

    return np.array([
        data for data in (
            set(tuple(pt_A) for pt_A in A) ^ 
            set(tuple(pt_B) for pt_B in B)
        )
    ])

def fit_simple_line(
    x_arr: np.ndarray[float], 
    y_arr: np.ndarray[float], 
    **kwargs: dict[Any]
) -> np.ndarray[float]:
    """Least square line fitting for scalar functions. 

    Args:
        x_arr (np.ndarray[float]): [_description_]
        y_arr (np.ndarray[float]): _description_

    Returns:
        np.ndarray[float]: _description_
    """
    
    n_sample: Union[int, None] = kwargs.get('n_sample', None)
    n_feature: Union[int, None] = kwargs.get('n_feature', None)
    a_point_on_line: bool = kwargs.get('a_point_on_line', False)
    on_line_pt_ind: int = kwargs.get('on_line_pt_ind', 0)
    
    assert type(x_arr) == np.ndarray and type(y_arr) == np.ndarray
    if y_arr.ndim != 2: y_arr = y_arr.reshape(-1,1) 
    if n_sample: 
        x_arr = x_arr.reshape(n_sample,-1)
        y_arr = y_arr.reshape(n_sample,1)
    if n_feature: x_arr = x_arr.reshape(-1, n_feature)
    if x_arr.ndim != 2: x_arr = x_arr.reshape(-1,1)
    assert x_arr.shape[0] == y_arr.shape[0]
    
    if not a_point_on_line: 
        x_intercept = np.ones(shape=(x_arr.shape[0],1))
        x_arr = np.hstack((x_arr, x_intercept)) # Last column corresponds to the 'b' param. 
        try: w = np.linalg.pinv(x_arr.T@x_arr)@x_arr.T@y_arr # [b,k] or [k,b] depending on the construction of x_arr. 
        except np.linalg.LinAlgError: w = np.asarray(np.mean(x_arr, axis=0)) # In case of non-invertible x.T@x -- return x=k.
    else: 
        anchor_x: np.ndarray[int] = x_arr[on_line_pt_ind,:].reshape(-1)
        anchor_y: np.ndarray[int] = y_arr[on_line_pt_ind,:].reshape(-1)
        x_arr = copy.deepcopy(x_arr-anchor_x)
        y_arr = copy.deepcopy(y_arr-anchor_y)
        try: 
            w_shifted = (
                np.linalg.pinv(x_arr.T@x_arr)@x_arr.T@y_arr
            ).reshape(-1)
            w = np.hstack((
                w_shifted,
                -(anchor_x@w_shifted).reshape(-1)+anchor_y
            ))
        except np.linalg.LinAlgError: w = np.asarray(anchor_x) # In case of non-invertible/scalar x.T@x -- return x=k.
    
    return w.reshape(-1)
    
def kalman_filter(
    x: np.ndarray[float], 
    P: np.ndarray[float], 
    y: np.ndarray[float], 
    transition_model: np.ndarray[float], 
    predict_noise: np.ndarray[float], 
    measurement_model: np.ndarray[float], 
    measurement_noise: np.ndarray[float], 
    force: Union[np.ndarray[float], None] = None, 
    control_model: Union[np.ndarray[float], None] = None
) -> Tuple[
    np.ndarray[float],
    np.ndarray[float],
    float
]:
    """Linear Kalman filter implementation. 
    Mind the dimensions. 
    
    x: input state variable.
    P: state estimate covariance. 
    y: Measurement.  
    transition_model: F. 
    predict_noise: Q. 
    measurement_model: H. 
    measurement_noise: R. 
    force: u. 
    control_model: B. 

    Args:
        x (np.ndarray[float]): _description_
        P (np.ndarray[float]): _description_
        y (np.ndarray[float]): _description_
        transition_model (np.ndarray[float]): _description_
        predict_noise (np.ndarray[float]): _description_
        measurement_model (np.ndarray[float]): _description_
        measurement_noise (np.ndarray[float]): _description_
        force (Union[np.ndarray[float], None], optional): _description_. Defaults to None.
        control_model (Union[np.ndarray[float], None], optional): _description_. Defaults to None.

    Returns:
        Tuple[ np.ndarray[float], np.ndarray[float], float ]: _description_
    """
    
    # Predict. 
    if force is not None and control_model is not None: 
        x_pred = transition_model @ x + control_model @ force
    else: 
        x_pred = transition_model @ x
    
    P_pred = (
        transition_model @ P @np.transpose(transition_model) + predict_noise
    )
    
    # Update. 
    residual = y - measurement_model @ x_pred
    innov_cov = (
        measurement_model @ P_pred @ np.transpose(measurement_model) + 
        measurement_noise
    )
    kalman_gain = (
        P_pred @ np.transpose(measurement_model) @ np.linalg.pinv(innov_cov)
    )
    
    # Estimate. 
    x_est = x_pred + kalman_gain @ residual
    P_est = (
        np.eye(P_pred.shape[0]) - kalman_gain @ measurement_model
    ) @ P_pred
    residual_est = y - measurement_model @ x_est
    
    return x_est, P_est, residual_est

def save_fig_to_pkl(fig: matplotlib.figure.Figure, save_path: str):
    """_summary_

    Args:
        fig (matplotlib.figure.Figure): _description_
        save_path (str): _description_
    """
    
    pickle.dump(fig, open(save_path, 'wb'))

def load_pkl_to_fig(load_path: str) -> matplotlib.figure.Figure:
    """_summary_

    Args:
        load_path (str): _description_

    Returns:
        matplotlib.figure.Figure: _description_
    """
    
    return pickle.load(open(load_path, 'rb'))

def fig2data(fig: matplotlib.figure.Figure) -> np.ndarray[np.uint8]:
    """_summary_

    Args:
        fig (matplotlib.figure.Figure): _description_

    Returns:
        np.ndarray[np.uint8]: _description_
    """
    
    # draw the renderer
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data

def save_dict_to_file(
    data_dict: dict, 
    save_name: str, 
    save_ext: Union[str, None] = 'pkl'
):
    """_summary_

    Args:
        data_dict (dict): _description_
        save_name (str): _description_
        save_ext (Union[str, None], optional): _description_. Defaults to 'pkl'.

    Raises:
        DataSavingingError: _description_
    """
    
    if save_ext is not None: 
        save_path = f"{save_name}.{save_ext}"
    else: 
        save_path: str = save_name
        save_ext: str = Path(save_name).suffix
    
    # scipy.io.savemat(save_dir, data_dict)
    if save_ext == 'pkl' or save_ext == '.pkl':
        with open(save_path, 'wb') as f:
            pickle.dump(copy.deepcopy(data_dict), f)
        f.close()
    elif save_ext == '.mat' or save_ext == '.mat':
        scipy.io.savemat(save_path, copy.deepcopy(data_dict))
    else: raise DataSavingingError("Invalid dictionary saving file format. ")
    
def load_dict_from_file(file_path: str) -> dict[Any]:
    """_summary_

    Args:
        file_path (str): _description_

    Returns:
        dict[Any]: _description_
    """
    
    assert file_path is not None and type(file_path) == str, (
        "Invalid input for \'file_path\'. Try giving a path string of a file. "
    )
    
    file_ext: str = Path(file_path).suffix
    
    if file_ext == '.pkl': 
        with open(file_path, 'rb') as f: data_dict = pickle.load(f)
        f.close()
    elif file_ext == '.mat':
        data_dict = scipy.io.loadmat(file_path)
    else: raise DataLoadingError("Unsupported dictionary file format. ")
    
    return data_dict

def y_plot_lim_given_arr(
    y_array: np.ndarray[float], 
    all_posi: bool = False
) -> tuple[float]:
    """_summary_

    Args:
        y_array (np.ndarray[float]): _description_
        all_posi (bool, optional): _description_. Defaults to False.

    Returns:
        tuple[float, float]: _description_
    """
    
    y_plot_min = np.min(y_array) - 0.5*(np.max(y_array)-np.min(y_array))
    if all_posi: y_plot_min = max(0., y_plot_min)
    else: pass
    y_plot_max = np.max(y_array) + 0.5*(np.max(y_array)-np.min(y_array))
    
    return (y_plot_min, y_plot_max)

def is_iterable(obj: Any) -> bool:
    """_summary_

    Args:
        obj (Any): Any variable. 

    Returns:
        bool: _description_
    """
    
    try: return bool(iter(obj))
    except TypeError: return False
    
def is_arraylike(obj: Any) -> bool:
    """_summary_

    Args:
        obj (Any): _description_

    Returns:
        bool: _description_
    """
    
    return isinstance(obj, (list, tuple, np.ndarray))

def is_sequence(obj: Any) -> bool:
    """_summary_

    Args:
        obj (Any): _description_

    Returns:
        bool: _description_
    """
    
    return isinstance(obj, (np.ndarray, collections.Sequence))

def is_subscriptable(obj: Any) -> bool:
    """_summary_

    Args:
        obj (Any): _description_

    Returns:
        bool: _description_
    """
    
    return hasattr(obj, "__getitem__")

def is_same_size(size_func: Callable, *args: tuple[ArrayLike]) -> bool:
    """_summary_

    Args:
        size_func (Callable): A Callable that returns the size (int) of the input variable. 

    Returns:
        bool: _description_
    """
    
    size_list: list[int] = [size_func(arg) for arg in args]
    return min(size_list) == max(size_list)

def empty_entity(dtype: type, n: int = 1) -> Tuple[Any]:
    """_summary_

    Args:
        dtype (type): _description_
        n (int, optional): _description_. Defaults to 1.

    Returns:
        Tuple[Any]: _description_
    """
    
    assert type(n) == int and n > 0, "`n` must be a positive integer. "
    
    if dtype == list: 
        empty_var: list = copy.deepcopy([])
    elif dtype == np.ndarray: 
        empty_var: np.ndarray = copy.deepcopy(np.array([]))
    elif dtype == tuple: 
        empty_var: tuple = copy.deepcopy(())
    elif dtype == dict: 
        empty_var: dict = copy.deepcopy({})
    else: raise ValueError("Unrecognized data type. ")
        
    return tuple([empty_var for _ in range(n)])

def read_csv2arr(
    file_path: str, skip_header: int = 1, dtype: Any = float
) -> ArrayLike:
    """_summary_

    Args:
        file_path (str): _description_
        skip_header (int, optional): _description_. Defaults to 1.
        dtype (Any, optional): _description_. Defaults to float.

    Returns:
        ArrayLike: _description_
    """
    
    if Path(file_path).suffix != '.csv':
        file_path: str = f"{file_path}.csv"
    else: pass
    
    return np.genfromtxt(
        file_path, delimiter=',', 
        skip_header=skip_header, dtype=dtype
    )

def get_paths_of_ext(dir_path: str, file_ext: str):
    """
    Gets all paths of files with a specific extension in a directory.

    Args:
        dir_path: The directory path to search.
        file_ext: The file extension to look for (e.g., ".txt").

    Returns:
        A list of file paths.
    """

    file_paths = []

    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(file_ext):
                file_path: str = os.path.join(root, file)
                file_paths.append(file_path)
            else: pass

    return file_paths

def str2num(string: str, type: DTypeLike = float) -> list[Union[int,float]]:
    """_summary_

    Args:
        string (str): _description_
        type (DTypeLike, optional): _description_. Defaults to float.

    Returns:
        list[Union[int, float]]: _description_
    """
    
    if type == float: 
        return np.asarray(
            re.findall(r"[-+]?(?:\d*\.*\d+)", string)
        ).astype(float)
    elif type == int: return np.asarray(re.findall(r'\d+', string)).astype(int)
    else: raise InvalidError("Invalid extracted number type. ")

def import_module_frompath(
    module_path: str, 
    var_name: str, 
    module_name: str = 'mymodule'
) -> Union[Callable, Any]:
    """_summary_

    Args:
        module_path (str): _description_
        var_name (str): _description_
        module_name (str, optional): _description_. Defaults to 'mymodule'.

    Returns:
        Union[Callable, Any]: _description_
    """
    
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mymodule = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mymodule
    spec.loader.exec_module(mymodule)
    
    return getattr(mymodule, var_name)

class IndexOfVal(object):
    """
    Extract a specific index of an input value given an 1d ArrayLike structure. 

    Args:
        object (_type_): _description_
    """
    
    def __init__(self, val: Any, **kwargs: Optional[dict[Any]]):
        """_summary_

        Args:
            val (Any): _description_
        """
        
        self.kwargs: Optional[dict[Any]] = kwargs
        self.val: Any = val
        
        self._ordinal_token: Optional[Union[int, str]] = None
        self._index_func: Optional[Callable] = None
    
    def At(self, ith: Union[int, str]) -> "IndexOfVal":
        """_summary_

        Args:
            ith (Union[int, str]): _description_

        Returns:
            IndexOfVal: _description_
        """
        
        returned_self: "IndexOfVal" = copy.deepcopy(self)
        returned_self._ordinal_token = copy.deepcopy(ith)
        
        if isinstance(returned_self._ordinal_token, str):
            if returned_self._ordinal_token == 'last': 
                def _index_func(d_list: list[Any]) -> int:
                    """_summary_

                    Args:
                        d_list (list[Any]): _description_

                    Returns:
                        int: _description_
                    """
                    return (
                        len(d_list) - 1 - 
                        d_list[::-1].index(returned_self.val)
                    )
            elif returned_self._ordinal_token == 'first': 
                def _index_func(d_list: list[Any]) -> int:
                    """_summary_

                    Args:
                        d_list (list[Any]): _description_

                    Returns:
                        int: _description_
                    """
                    return d_list.index(returned_self.val)
            else: raise InvalidError("Invalid ordinal input. ")
        elif isinstance(returned_self._ordinal_token, int): 
            assert returned_self._ordinal_token >= 0, (
                "Input ordinal integer must be non-negative. "
            )
            def _index_func(d_list: list[Any]) -> int:
                """_summary_

                Args:
                    d_list (list[Any]): _description_

                Returns:
                    int: _description_
                """
                index_arr: np.ndarray[int] = np.where(
                    np.asarray(d_list).reshape(-1) == returned_self.val
                )[0].reshape(-1)
                try: return index_arr[returned_self._ordinal_token]
                except IndexError: 
                    raise IndexError("Input ordinal number out of bounds. ")
        else: 
            try:
                returned_self._ordinal_token = (
                    int(returned_self._ordinal_token)
                )
                returned_self = self.at(returned_self._ordinal_token)
                _index_func = returned_self._index_func
            except (TypeError, ValueError): 
                raise DataProcessingError(
                    "Input ordinal token must be either integer or string. "
                )
        
        returned_self._index_func = _index_func
        return returned_self
    
    def In(self, d_arr: ArrayLike) -> int:
        """_summary_

        Args:
            d_arr (ArrayLike): _description_

        Returns:
            int: _description_
        """
        
        d_arr: np.ndarray[Any] = np.asarray(d_arr).reshape(-1)
        return self._index_func(list(d_arr))


if __name__ == "__main__":
    """_summary_
    """
    
    pass

