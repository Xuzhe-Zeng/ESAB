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
import matplotlib.image as mig
import PIL
import PIL.Image

from scipy.ndimage import rotate
from numpy.typing import ArrayLike
from typing import Union, Any

from clustering import dbscan

import config
CONFIG = config.read_config(config.CONFIG_PATH)
PM_IMG = __import__(
    f"{CONFIG.get('PARAM')}.IMG.IMG", fromlist=['']
)
util = __import__(
    f"{CONFIG.get('utils')}.util", fromlist=['']
)
# udim = __import__(
#     f"{CONFIG.get('utils')}.dimensionality", fromlist=['']
# )


##################################################################################
##################################################################################
##################################################################################


def rotate_2D(data_matrix, theta, center_pt):
    """
    data_matrix: shape=(n_points, n_features=2)
    theta: in degrees. 
    center_pt: shape=(n_features=2,)
    """

    theta_rad = theta * np.pi / 180.
    rot_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])

    vector_matrix = data_matrix - center_pt
    vector_matrix_rotated = copy.deepcopy(rot_matrix @ vector_matrix.T).T

    return vector_matrix_rotated + center_pt


def projection_along_axis(pixel_index_array, origin_pt, axis_vect):
    """
    `pixel_index_array`: shape=(n_sample, n_feature). 
    `origin_pt`: shape=(n_feature,). 
    `axis_vect`: shape=(n_feature,). 
    """
    
    vect_array = copy.deepcopy(pixel_index_array - origin_pt)
    inner_product_array = copy.deepcopy(vect_array @ axis_vect.reshape(-1,1)).reshape(-1)
    
    return inner_product_array


class Image(object):
    def __init__(
        self,
        file_path: Union[str, None] = None,
        image_matrix: Union[np.ndarray, None] = None,
    ):
        self.file_path = file_path

        if image_matrix is not None:
            self.image_matrix = image_matrix
        elif self.file_path is not None:
            self.image_matrix = mig.imread(self.file_path)
        else:
            raise ValueError("Either file_path or image_matrix must be provided.")

        # 如果 PNG 是 RGB / RGBA，转成灰度图
        if self.image_matrix.ndim == 3:
            self.image_matrix = self.image_matrix[..., :3].mean(axis=2)

        if self.image_matrix.ndim != 2:
            raise ValueError(
                f"Expected 2D image matrix, got shape {self.image_matrix.shape}"
            )

        self.image_length, self.image_width = self.image_matrix.shape
        self._default_background_value = 0.

    
    def pixelNum(self):
        """
        """

        return int(self.image_length*self.image_width)


    def length(self):
        """
        """
        
        return self.image_length

    
    def width(self):
        """
        """
        
        return self.image_width

    
    def blank_version(self):
        """
        """

        return np.ones(shape=(self.image_length, self.image_width)) * self._default_background_value
    

    def refine_asper_threshold(self, image_matrix, threshold):
        """
        """
        
        image_matrix_refined = copy.deepcopy(image_matrix)
        image_matrix_refined[image_matrix_refined < threshold] = self._default_background_value
        
        return image_matrix_refined


class Frame(Image):
    def __init__(
        self,
        file_path: Union[str, None] = None,
        image_matrix: Union[np.ndarray, None] = None,
        intensity_threshold: tuple[float, float] = PM_IMG.INTENSITY_THRESHOLD,
        sidebar_threshold: float = PM_IMG.SIDEBAR_THRESHOLD,
        sidebar_columns: list[tuple, tuple] = PM_IMG.SIDEBAR_COLUMNS,
    ):
        """_summary_

        Args:
            file_path (Union[str, None], optional): _description_. Defaults to None.
            image_matrix (np.ndarray[Any], optional): _description_. Defaults to np.array([None]).
            intensity_threshold (tuple[float, float], optional): _description_. Defaults to PM_IMG.INTENSITY_THRESHOLD.
            sidebar_threshold (float, optional): _description_. Defaults to PM_IMG.SIDEBAR_THRESHOLD.
            sidebar_columns (list[tuple, tuple], optional): _description_. Defaults to PM_IMG.SIDEBAR_COLUMNS.
        """

        super(Frame, self).__init__(file_path, image_matrix)
        
        self.intensity_threshold: tuple[float] = intensity_threshold # Tuple of Floats. Two constant intensity thresholds that filter out the bright melt pool and spatters. 
        self.sidebar_threshold: float = sidebar_threshold # Float. A constant intensity threshold for sidebar filtering. Default: 0.05. 
        self.sidebar_columns: list[tuple[int]] = sidebar_columns # List of tuples. Two tuples of int that prescribe the range of side bar. 
        # self.plume_threshold = plume_threshold # Tuple of Floats. Two constant intensity thresholds that filter out the plumes. 
        self._isEmpty: bool = False # Boolean. True if no pixels satisfy the intensity threshold. 
        
        # Original. 
        self.original_image_matrix: np.ndarray[float] = (
            copy.deepcopy(self.image_matrix)
        ) # 2D array of Float. (512, 512). 0.-1. Save a copy of original unprocessed image matrix. 
        self.original_pixel_index_array = None
        self.original_pixel_intensity_array = None
        
        # Total. 
        self._bright_pixel_index_array: np.ndarray = np.array([[None, None]])
        self._bright_pixel_intensity_array: np.ndarray = np.array([None]) # Same order as `self._bright_pixel_index_array`. 
        self.visible_image: np.ndarray = np.array([None])
        self.mass_center_pt: np.ndarray = np.array([None]) # 1D Int array. A pixel. 
        self._total_area: Union[float, None] = None

        self._pixel_clusters: Union[dbscan, None] = None

        self._preprocess() # Frame image preprocessing. 
        self._thresholding() # Image thresholding. 
        self._pixel_clustering()
    

    @property
    def total_area(self) -> int:
        """_summary_

        Returns:
            int: _description_
        """

        return self._total_area


    @property
    def visible_pixel_indices(self) -> np.ndarray[int]:
        """Get the indices of pixels with intensity values fallen in between the range defined by `self.intensity_threshold`. 

        Returns:
            np.ndarray[int]: _description_
        """

        return self._bright_pixel_index_array

    
    @property
    def frame(self) -> np.ndarray[float]:
        """Return the original, unprocessed image matrix. 

        Returns:
            np.ndarray[float]: _description_
        """
        
        return self.original_image_matrix


    @property
    def thresheld_image(self) -> np.ndarray[float]:
        """_summary_

        Returns:
            np.ndarray[float]: _description_
        """
        
        return self.visible_image

    
    @property
    def all_clusters(self) -> dbscan:
        """_summary_

        Returns:
            dbscan: _description_
        """

        return self._pixel_clusters

    
    def _preprocess(self):
        """
        Remove background noise locating at the two visible bright bars on both sides of the image frame. 
        """
        
        preprocessed_matrix = copy.deepcopy(self.original_image_matrix)
        
        for sidebar_col_range in self.sidebar_columns:
            left, right = sidebar_col_range[0], sidebar_col_range[1]+1
            preprocessed_matrix[:,left:right][
                np.where(
                    preprocessed_matrix[:,left:right] < 
                    self.sidebar_threshold
                )
            ] = self._default_background_value
        
        self.image_matrix = copy.deepcopy(preprocessed_matrix)
        
        # Save pixels and intensity values of the original image. 
        original_indices_array_row_col = np.where(
            np.logical_and(self.image_matrix >= 0., self.image_matrix <= 1.)
        )
        self.original_pixel_index_array = np.vstack((
            original_indices_array_row_col[0], 
            original_indices_array_row_col[1]
        )).astype(int).T # [row | col]; [Axis-0 | Axis-1].
        self.original_pixel_intensity_array = copy.deepcopy(
            self.image_matrix.reshape(-1)
        )


    def _pixel_clustering(self):
        """_summary_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        
        if not self._isEmpty:
            if self._pixel_clusters is None: 
                if self._bright_pixel_index_array is None: 
                    self._thresholding()
                else: pass
                self._pixel_clusters = dbscan(
                    self._bright_pixel_index_array, 
                    PM_IMG.DBSCAN_EPSILON, 
                    PM_IMG.DBSCAN_MIN_PTS
                )
            else: pass
        else: pass
    

    @staticmethod
    def _specify_image(
        image: np.ndarray[float], 
        pixel_index_array: np.ndarray[int], 
        img_val_array: np.ndarray[float]
    ) -> np.ndarray[float]:
        """
        `pixel_index_array`: shape=(n_sample, n_feature)
        `img_val_array`: shape=(n_sample,), following the same order. 

        Args:
            image (np.ndarray[float]): _description_
            pixel_index_array (np.ndarray[int]): _description_
            img_val_array (np.ndarray[float]): _description_

        Returns:
            np.ndarray[float]: _description_
        """

        image_dim_0, image_dim_1 = image.shape
        for i, pixel_indices in enumerate(pixel_index_array):
            row_ind_temp, col_ind_temp = pixel_indices.astype(int)
            if (
                0 <= row_ind_temp < image_dim_0 and
                0 <= col_ind_temp < image_dim_1
            ): # Make sure that the indices is within the new image's frame range. Discard the outliers. The rest of unfilled new image region will be autofilled with 0 (black background). 
                image[row_ind_temp,col_ind_temp] = img_val_array[i]
            else: pass
        
        return image


    def _filter_image(
        self, pixel_index_array: np.ndarray[int], 
        mode: str = 'as_is'
    ) -> np.ndarray[float]:
        """_summary_

        Args:
            pixel_index_array (np.ndarray[int]): _description_
            mode (str, optional): _description_. Defaults to 'as_is'.

        Raises:
            ValueError: _description_

        Returns:
            np.ndarray[float]: _description_
        """

        image_matrix = copy.deepcopy(self.image_matrix)
        filtered_image_matrix = self.blank_version()
        
        for i in range(pixel_index_array.shape[0]):
            row_ind_temp, col_ind_temp = pixel_index_array[i,:].astype(int)
            
            if mode == 'as_is': 
                filtered_image_matrix[row_ind_temp,col_ind_temp] = (
                    image_matrix[row_ind_temp,col_ind_temp]
                )
            elif mode == 'binary':
                filtered_image_matrix[row_ind_temp,col_ind_temp] = 1.
            else: raise ValueError("Invalid image filtering mode. ")
        
        return copy.deepcopy(filtered_image_matrix)


    def binarize(self, 
        image_matrix: np.ndarray[float], 
        threshold: Union[float, None] = None
    ) -> np.ndarray[float]:
        """Binarize image by filtering out all background pixels.

        Args:
            image_matrix (np.ndarray[float]): _description_
            threshold (Union[float, None], optional): _description_. Defaults to None.

        Returns:
            np.ndarray[float]: _description_
        """

        if threshold is None: 
            return (image_matrix > self._default_background_value).astype(float)
        else: return (image_matrix > threshold).astype(float)


    def _centering(self, 
        pixel_index_array: np.ndarray[int], 
        center_pt: np.ndarray[float]
    ) -> np.ndarray[float]:
        """
        Shifting `pixel_index_array`, the distance of which is determined by the center of the image frame and `center_pt` (a label point in `pixel_index_array`). 

        center_pt (np.ndarray[float]): The label point of `pixel_index_array`, which is going to be centered at the center of the image frame. 

        Args:
            pixel_index_array (np.ndarray[int]): _description_
            center_pt (np.ndarray[float]): _description_

        Returns:
            np.ndarray[float]: _description_
        """

        center_pt = copy.deepcopy(center_pt.astype(int).reshape(-1))
        center_pt_blank_version = np.array(
            [self.length()/2., self.width()/2.]
        ).astype(int).reshape(-1)
        
        pixel_index_array_translated = util.shifting(
            pixel_index_array, 
            axis=0, 
            origin=center_pt, 
            target=center_pt_blank_version
        ).astype(int)

        return pixel_index_array_translated
    

    def _thresholding(self):
        """
        Use thresholds to filter different regions of frame of interest. 
        """
        
        # Bright pixels. 
        bright_indices_array_row_col = np.where(
            np.logical_and(
                self.image_matrix >= self.intensity_threshold[0], 
                self.image_matrix <= self.intensity_threshold[1]
            )
        )
        self._bright_pixel_index_array = np.vstack((
            bright_indices_array_row_col[0],
            bright_indices_array_row_col[1]
        )).astype(int).T # [row | col]; [Axis-0 | Axis-1].
        self._total_area = self._bright_pixel_index_array.shape[0]

        if self._total_area != 0:
            self._isEmpty = False
            self._bright_pixel_intensity_array = copy.deepcopy(
                np.array([
                    self.image_matrix[i,j] 
                    for i,j in self._bright_pixel_index_array
                ]).reshape(-1)
            )
            self.visible_image = self._filter_image(
                self._bright_pixel_index_array
            )
            self.mass_center_pt = self._get_center_of(
                self._bright_pixel_index_array
            )

        else:
            self._isEmpty = True
            self._bright_pixel_index_array = np.array([[None, None]])
            self._bright_pixel_intensity_array = np.array([None])
            self.visible_image = self.blank_version()
            self.mass_center_pt = np.array([
                self.length() / 2., self.width() / 2.
            ]).astype(int).reshape(-1)
    
    
    def _center_rotate_image(self, 
        pixel_index_array: np.ndarray[int], 
        pixel_val_array: np.ndarray[float], 
        center_pt: np.ndarray[float], 
        rotate_angle: float, 
        resize_scale: float
    ) -> PIL.Image.Image:
        """Commonly operating on the meltpool center.

        Args:
            pixel_index_array (np.ndarray[int]): _description_
            pixel_val_array (np.ndarray[float]): _description_
            center_pt (np.ndarray[float]): _description_
            rotate_angle (float): _description_
            resize_scale (float): _description_

        Returns:
            PIL.Image.Image: _description_
        """
        
        pixel_index_array_centered: np.ndarray[float] = (
            self._centering(pixel_index_array, center_pt)
        )
        centered_version: np.ndarray[float] = (
            self._specify_image(
                self.blank_version(), 
                pixel_index_array_centered, 
                pixel_val_array
            )
        )
        image_obj: PIL.Image.Image = (
            PIL.Image.fromarray(np.uint8(centered_version*255))
        )
        width, height = image_obj.size
        image_obj = image_obj.rotate(
            rotate_angle, 
            resample=PIL.Image.Resampling.BICUBIC, 
            expand=False, 
            fillcolor='black' # zero intensity val. for unfilled region. 
        )
        image_obj = image_obj.resize(
            (int(width*resize_scale), int(height*resize_scale))
        )
        
        return image_obj
    
    
    def straighten_explicit(self, 
        center_pt: np.ndarray[int], 
        rotation_angle: float, 
        ROI_keyword: str = 'original',
        resize_scale: float = 1.
    ) -> PIL.Image.Image:
        """
        Straighten the `ROI_keyword` image by centering the label point (`center_pt`) of the image and rotate the image by `rotation_angle`. `resize_scale` is applied to zoom in/out the content within the image frame. 

        Args:
            center_pt (np.ndarray[int]): _description_
            rotation_angle (float): _description_
            ROI_keyword (str, optional): _description_. Defaults to 'original'.
            resize_scale (float, optional): _description_. Defaults to 1.

        Raises:
            ValueError: _description_

        Returns:
            PIL.Image.Image: _description_
        """
        
        if not self._isEmpty:            
            if ROI_keyword == 'original':
                straightened_image_obj = self._center_rotate_image(
                    self.original_pixel_index_array, 
                    self.original_pixel_intensity_array, 
                    center_pt, 
                    rotation_angle, 
                    resize_scale
                )
            elif ROI_keyword == 'thresheld': 
                straightened_image_obj = self._center_rotate_image(
                    self._bright_pixel_index_array, 
                    self._bright_pixel_intensity_array, 
                    center_pt, 
                    rotation_angle, 
                    resize_scale
                )
            else: raise ValueError("ROI keyword unrecognizable. ")
        else: 
            straightened_image_obj = PIL.Image.fromarray(
                np.uint8(self.blank_version()*255)
            )

        return straightened_image_obj

    
    def straighten_implicit(self):
        """_summary_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        pass
    
    
    @staticmethod
    def _get_center_of(pixel_index_array: np.ndarray[float]):
        """
        """

        return np.mean(pixel_index_array, axis=0).astype(int)

#
# if __name__ == "__main__":
#
#     pass