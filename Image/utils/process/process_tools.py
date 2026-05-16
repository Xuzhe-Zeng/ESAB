DIR_TO_TOP: str = "../.."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(2)

import copy
import numpy as np

from collections import defaultdict
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
from typing import Union

from .util import default_dict

import config
CONFIG = config.read_config(config.CONFIG_PATH)
PM_PR_SC = __import__(
    f"{CONFIG.get('PARAM')}.PROCESSING.SCANLINE", fromlist=['']
)


##################################################################################
##################################################################################
##################################################################################


def photodiode_scanlines_parsing(
    photodiode_data: np.ndarray[float], 
    slide_window_length: int, 
    slide_window_stride: int = 1, 
    COV_trshld: float = 0.05, 
    max_min_trshld: float = 0.05, 
    prescribed_parsing_indices: Union[list[int], None] = None
) -> dict:
    """_summary_

    Args:
        photodiode_data (np.ndarray[float]): _description_
        slide_window_length (int): _description_
        slide_window_stride (int, optional): _description_. Defaults to 1.
        COV_trshld (float, optional): _description_. Defaults to 0.05.
        max_min_trshld (float, optional): _description_. Defaults to 0.05.
        prescribed_parsing_indices (Union[list[int], None], optional): _description_. Defaults to None.

    Returns:
        dict: _description_
    """
    
    scanlines_photo_dict = defaultdict(default_dict)
    photo_data_ind_all = [i for i in range(len(photodiode_data))]
    laser_on_off_flag = np.ones(
        shape=(len(photodiode_data,))
    ).astype(int).reshape(-1)

    if prescribed_parsing_indices is None: 
        clips_mat = sliding_window_view(
            photodiode_data, slide_window_length
        )[::slide_window_stride,:]
        clips_num = clips_mat.shape[0]
    
        laser_off_indices = []
        for i in range(clips_num):
            clip_start_ind_temp = int(i*slide_window_stride)
            clip_end_ind_temp = int(i*slide_window_stride+slide_window_length)
            photo_data_ind_temp = (
                photo_data_ind_all[clip_start_ind_temp:clip_end_ind_temp]
            )

            clip_temp = clips_mat[i]
            clip_max_temp, clip_min_temp = (
                np.max(clip_temp), np.min(clip_temp)
            )
            clip_mean_temp, clip_std_temp = (
                np.mean(clip_temp), np.std(clip_temp)
            )
            clip_COV_temp = clip_std_temp / clip_mean_temp

            if ((clip_max_temp - clip_min_temp) < max_min_trshld and 
                clip_COV_temp < COV_trshld):
                laser_off_indices += photo_data_ind_temp # Laser-off datapoints. 
        
        laser_off_indices = list(set(laser_off_indices)) # Singularize and sort the laser_off indices. 
        laser_on_off_flag[laser_off_indices] = 0 # Put laser_off pts as 0. 
        laser_impinging_pts_arr = np.where(
            np.diff(laser_on_off_flag) == 1
        )[0] + 1 # Extract the beginning indices of each scanline. 
        
    else: laser_impinging_pts_arr = copy.deepcopy(prescribed_parsing_indices)
    
    scanlines_indices_list = np.split(
        photo_data_ind_all, laser_impinging_pts_arr
    ) # Separate scanlines based on laser impinging pts - scanline indices. 
    scanlines_photodata_list = np.split(
        photodiode_data, laser_impinging_pts_arr
    ) # Separate scanlines based on laser impinging pts - scanline photodiode data.
    scanlines_onoff_list = np.split(
        laser_on_off_flag, laser_impinging_pts_arr
    ) # Separate scanlines based on laser impinging pts - scanline on/off flags.
    
    scanlines_photo_dict['parse_indices'] = laser_impinging_pts_arr
    for i, scanline in enumerate(scanlines_indices_list):
        i_str = str(i).zfill(3)
        scanlines_photo_dict[i_str]['beginat'] = scanline[0]
        scanlines_photo_dict[i_str]['indices'] = scanline
        scanlines_photo_dict[i_str]['photodata'] = scanlines_photodata_list[i]
        scanlines_photo_dict[i_str]['onoff'] = scanlines_onoff_list[i]
    
    return scanlines_photo_dict


def get_meltpool_area_thrshld(P: float, V: float):
    """_summary_

    Args:
        P (_type_): _description_
        V (_type_): _description_
    """
    
    if V >= 1000: return 30
    else: return 50
    

def scanline_angle(theta: float, scanline_No: int):
    """
    Use to calculate speed direction. 
    Return in rad. 
    """
    
    if scanline_No % 2 == 0: 
        if theta == 0: return 90*np.pi/180
        elif theta == 45: return -45*np.pi/180
        elif theta == 90: return 180*np.pi/180
        elif theta == 135: return -135*np.pi/180
        else: raise ValueError("Theta unrecognizable. ")
    else:
        if theta == 0: return -90*np.pi/180
        elif theta == 45: return 135*np.pi/180
        elif theta == 90: return 0*np.pi/180
        elif theta == 135: return 45*np.pi/180
        else: raise ValueError("Theta unrecognizable. ")
        

def straighten_angle(theta: float, scanline_No: int):
    """
    Use to calculate rotation angle. 
    Return in degrees. 
    """
    
    if scanline_No % 2 == 0: 
        if theta == 0: return 0
        elif theta == 45: return 135
        elif theta == 90: return -90
        elif theta == 135: return -135
        else: raise ValueError("Theta unrecognizable. ")
    else:
        if theta == 0: return 180
        elif theta == 45: return -45
        elif theta == 90: return 90
        elif theta == 135: return 45
        else: raise ValueError("Theta unrecognizable. ")
        
        
if __name__ == "__main__":
    """_summary_
    """
    
    pass

