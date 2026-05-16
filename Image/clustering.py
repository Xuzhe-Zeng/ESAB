DIR_TO_TOP: str = "../.."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(2)

import numpy as np
import sklearn.cluster as skc

from typing import Union, Tuple, Any


##################################################################################
##################################################################################
##################################################################################


class dbscan(object):
    """
    """

    def __init__(self, 
        sample_pts_array: np.ndarray[float], 
        epsilon: float, 
        minPts: int
    ):
        """_summary_

        Args:
            sample_pts_array (np.ndarray[float]): _description_
            epsilon (float): _description_
            minPts (int): _description_
        """

        self.sample_pts_array: np.ndarray[float] = sample_pts_array
        self.epsilon: float = epsilon
        self.minPts: int = minPts

        self._sample_pts_label_array: Union[np.ndarray[int], None] = None
        self._cluster_label_list: Union[list[int], None] = None # 1D array of labels of all clusters. 
        self._cluster_dict: dict[Any] = {}

        self._clustering()
        self._set_cluster_dict()
        
    
    def _clustering(self):
        """_summary_
        """

        clustering = skc.DBSCAN(
            eps=self.epsilon, min_samples=self.minPts
        ).fit(self.sample_pts_array)
        self._sample_pts_label_array = clustering.labels_ # Integer labels. 
        self._cluster_label_list = list(set(self._sample_pts_label_array))


    def _set_cluster_dict(self):
        """_summary_
        """

        for cluster_label in self._cluster_label_list:
            sample_pts_indices_thisCluster = np.where(
                self._sample_pts_label_array==cluster_label
            )[0].reshape(-1)
            sample_pts_thisCluster = (
                self.sample_pts_array[sample_pts_indices_thisCluster,:]
            )
            self._cluster_dict[cluster_label] = sample_pts_thisCluster
            

    def cluster_of(self, label: int) -> np.ndarray[int]:
        """Use label to retrieve the cluster. 

        Args:
            label (int): _description_

        Returns:
            np.ndarray[int]: _description_
        """
        
        return self._cluster_dict[label]

    
    def cluster_of_center(self, 
        center_pt: np.ndarray[int]
    ) -> Tuple[int, np.ndarray[int], None]:
        """Use center point to retrieve the cluster. Obtain the cluster closest to the given center point.

        Args:
            center_pt (np.ndarray[int]): _description_

        Returns:
            Tuple[int, np.ndarray[int]]: _description_
        """
        
        closest_cluster_label, smallest_center_dist = -1, 1e5
        closest_cluster = None
        
        for key, val in self._cluster_dict.items():
            if key == -1: continue # Skip the noise cluster.
            
            mean_pt = np.mean(val, axis=0)
            if np.linalg.norm(mean_pt-center_pt) <= smallest_center_dist: 
                closest_cluster_label = key
                closest_cluster = val
                smallest_center_dist = np.linalg.norm(mean_pt-center_pt)
            else: continue
        
        return closest_cluster_label, closest_cluster
    
    
    def clusters_within(self, 
        anchor_pt: np.ndarray[int], 
        dist: float
    ) -> dict[Any]:
        """_summary_

        Args:
            anchor_pt (np.ndarray[int]): _description_
            dist (float): _description_

        Raises:
            ValueError: _description_

        Returns:
            dict[Any]: _description_
        """
        
        within_dict: dict[Any] = {}
        
        for key, val in self._cluster_dict.items():
            if key == -1: continue # Skip the noise cluster. 
            center_pt_temp = self.center_pt(val).reshape(-1)
            dist_temp: float = np.linalg.norm(center_pt_temp - anchor_pt)
            if dist_temp <= dist: within_dict[key] = val
        
        return within_dict
    
    
    @staticmethod
    def _largest_of(cluster_dict: dict[Any]) -> Tuple[
        int, np.ndarray[int], None
    ]:
        """_summary_

        Args:
            cluster_dict (dict[Any]): _description_

        Returns:
            Tuple[int, np.ndarray[int]]: _description_
        """
        
        largest_cluster_label, largest_cluster_size = -1, 0
        largest_cluster = None

        for key, val in cluster_dict.items():
            if val.shape[0] >= largest_cluster_size: 
                largest_cluster_label = key
                largest_cluster = val
                largest_cluster_size = largest_cluster.shape[0]
            else: continue
        
        return largest_cluster_label, largest_cluster
        
    
    def largest_cluster(self) -> Tuple[int, np.ndarray[int]]:
        """_summary_

        Returns:
            Tuple[int, np.ndarray[int]]: _description_
        """
        
        cluster_dict: dict[Any] = {}
        for key, val in self._cluster_dict.items():
            if key == -1: continue # Skip the noise cluster.
            else: cluster_dict[key] = val
        
        return self._largest_of(cluster_dict)
    
    
    @property
    def noise_cluster(self) -> np.ndarray[int]:
        """_summary_

        Returns:
            np.ndarray[int]: _description_
        """

        return self._cluster_dict[-1]

    
    def labels_and_clusters(self, 
        sort: str ='as_is', 
        include_noise: bool = False, 
        exclude: Union[list[int], None] = None
    ) -> Tuple[list[int], list[np.ndarray[int]]]:
        """
        Return: two lists of labels and clusters, respectively. 
        `exclude`: a list of the labels of clusters to be excluded. 

        Args:
            sort (str, optional): _description_. Defaults to 'as_is'.
            include_noise (bool, optional): _description_. Defaults to False.
            exclude (Union[list[int], None], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            Tuple[list[int], list[np.ndarray[int]]]: _description_
        """

        # Collect all labels and clusters into separate lists. 
        label_list = [key for key, _ in self._cluster_dict.items()]
        cluster_list = [val for _, val in self._cluster_dict.items()]

        if not include_noise and -1 in label_list: # Remove noise(-1) cluster if `include_noise` is True. 
            del cluster_list[label_list.index(-1)]
            label_list.remove(-1)
        
        if exclude is not None: # Remove clusters as per the given excluding list. 
            for label_ex in exclude:
                if label_ex in label_list:
                    del cluster_list[label_list.index(label_ex)]
                    label_list.remove(label_ex)
                else: pass

        if sort == 'as_is': return label_list, cluster_list
        elif sort == 'size_descend':
            cluster_size_list = [cluster.shape[0] for cluster in cluster_list]
            zipped = list(zip(cluster_size_list, label_list, cluster_list))
            zipped.sort(key=lambda x: x[0], reverse=True)
            _, label_list, cluster_list = list(zip(*zipped))
            return list(label_list), list(cluster_list)
        elif sort == 'size_ascend':
            cluster_size_list = [cluster.shape[0] for cluster in cluster_list]
            zipped = list(zip(cluster_size_list, label_list, cluster_list))
            zipped.sort(key=lambda x: x[0], reverse=False)
            _, label_list, cluster_list = list(zip(*zipped))
            return list(label_list), list(cluster_list)
        elif sort == 'asper_label': # Sort as per the order of values of labels (ascend). 
            zipped = list(zip(label_list, cluster_list))
            zipped.sort(key=lambda x: x[0], reverse=False)
            label_list, cluster_list = list(zip(*zipped))
            return list(label_list), list(cluster_list)
        else: 
            raise ValueError(
                "Incorrect argument for `sort`. " + 
                "Try using one of the following: " + 
                "\'as_is\', \'asper_label\', " + 
                "\'size_ascend\', \'size_descend\'. "
            )


    @staticmethod
    def center_pt(cluster: np.ndarray[int]) -> np.ndarray[float]:
        """_summary_

        Args:
            cluster (np.ndarray[int]): _description_

        Returns:
            np.ndarray[float]: _description_
        """

        return np.mean(cluster, axis=0)


class kmeans(object):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(self, 
        data_matrix: np.ndarray[float], 
        n_clusters: int, 
        random_state: int = 0
    ):
        """_summary_

        Args:
            data_matrix (np.ndarray[float]): _description_
            n_clusters (int): _description_
            random_state (int, optional): _description_. Defaults to 0.
        """

        self.data_matrix: np.ndarray[float] = data_matrix
        self.n_clusters: int = n_clusters
        self.random_state: int = random_state

        self._kmeans: Union[skc.KMeans, None] = None
        self._labels: list[int] = None
        self._cluster_centers: np.ndarray[float] = None

    
    @property
    def labels(self) -> list[int]:
        """_summary_

        Returns:
            list[int]: _description_
        """

        return self._labels
    

    @property
    def cluster_centers(self) -> np.ndarray[float]:
        """_summary_

        Returns:
            np.ndarray[float]: _description_
        """

        return self._cluster_centers


    def clustering(self):
        """
        """

        self._kmeans = skc.KMeans(
            self.n_clusters, random_state=self.random_state
        ).fit(self.data_matrix)
        
        self._labels = self._kmeans.labels_
        self._cluster_centers = self._kmeans.cluster_centers


if __name__ == "__main__":
    """_summary_
    """
    
    pass