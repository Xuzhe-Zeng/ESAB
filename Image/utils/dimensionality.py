DIR_TO_TOP: str = ".."

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.abspath(CURRENT_DIR+f'/{DIR_TO_TOP}/'))

# import os
# import totop
# totop._from(os.path.dirname(os.path.abspath(__file__)))._by(1)

import copy
import numpy as np

from typing import Union, Tuple


from .util import shifting, normalization


class PCA(object):
    """
    """

    def __init__(self, 
                 training_matrix: np.ndarray[float], 
                 PC_num: int, 
                 mode: str = 'normal'):
        """
        training_matrix with Axis-0 be the sample axis. 
        mode: 'normal' (default) or 'transpose'. 
        'normal' (default): decompose on sample axis. 
        'transpose': decompose on feature axis. 

        Args:
            training_matrix (np.ndarray[float]): _description_
            PC_num (int): _description_
            mode (str, optional): _description_. Defaults to 'normal'.
        """

        self.mode = mode
        self.PC_num = PC_num

        if self.mode == 'normal': self.matrix = training_matrix
        elif self.mode == 'transpose': self.matrix = training_matrix.T
        else: self.matrix = np.eye(training_matrix.shape[0])
        
        self._mean_vect: Union[np.ndarray[float], None] = None
        self._eigFace_matrix: Union[np.ndarray[float], None] = None
        self._weights: Union[np.ndarray[float], None] = None
        self._eigFace_matrix_full: Union[np.ndarray[float], None] = None
        self._weights_full: Union[np.ndarray[float], None] = None
        self._eigVal_full_sorted: Union[np.ndarray[float], None] = None

        self._set_encoder_decoder()

    
    @property
    def eigVals(self) -> np.ndarray[float]:
        """_summary_

        Returns:
            np.ndarray[float]: _description_
        """
        
        return self._eigVal_full_sorted
    

    def _meanshifting(self
                      ) -> Tuple[np.ndarray[float],
                                 np.ndarray[float],
                                 np.ndarray[float],
                                 np.ndarray[float]]:
        """_summary_

        Returns:
            Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]: _description_
        """

        if self.mode == 'transpose': axis = 1
        else: axis = 0

        self._mean_vect = np.mean(self.matrix, axis=axis).reshape(-1)

        return shifting(self.matrix, axis=axis, origin=self._mean_vect, 
                        target=np.zeros(shape=self._mean_vect.shape))


    @staticmethod
    def _eigendecomposing(matrix: np.ndarray[float]
                          ) -> Tuple[np.ndarray[float], np.ndarray[float]]:
        """_summary_

        Args:
            matrix (np.ndarray[float]): _description_

        Returns:
            Tuple[np.ndarray[float], np.ndarray[float]]: _description_
        """

        cov_matrix = copy.deepcopy(matrix @ matrix.T)
        return eigenDecomposition(cov_matrix)

    
    def _eigSorting(self, eigVal: np.ndarray[float], 
                    eigVect: np.ndarray[float]) -> Tuple[np.ndarray[float], 
                                                         np.ndarray[float]]:
        """_summary_

        Args:
            eigVal (np.ndarray[float]): _description_
            eigVect (np.ndarray[float]): _description_

        Returns:
            Tuple[np.ndarray[float], np.ndarray[float]]: _description_
        """

        eigFace_num = eigVal.shape[0]
        eigVal_sorted = np.zeros(shape=eigVal.shape, dtype=complex)
        eigVect_sorted = np.zeros(shape=eigVect.shape, dtype=complex)

        eigVal_sorted_indices = np.argsort(np.real(eigVal))
        eigVal_PC_indices = eigVal_sorted_indices[-1:-(eigFace_num+1):-1] # Pick PC_num indices of largest principal eigenvalues
        
        for i, index in enumerate(eigVal_PC_indices): # From biggest to smallest
            eigVal_sorted[i] = eigVal[index] # Pick PC_num principal eigenvalues. Sorted. 
            eigVect_sorted[:,i] = eigVect[:,index] # Pick PC_num principal eigenvectors. Sorted. 

        return np.real(eigVect_sorted), np.real(eigVal_sorted)


    def _update_eigFaces_weights(self):
        """
        """
        
        self._eigFace_matrix = copy.deepcopy(self._eigFace_matrix_full[:,:self.PC_num])
        self._weights = copy.deepcopy(self._weights_full[:,:self.PC_num])


    def _set_encoder_decoder(self):
        """
        """

        matrix_meanshifted = self._meanshifting()
        eigVal, eigVect = self._eigendecomposing(matrix_meanshifted)

        eigVect_sorted, self._eigVal_full_sorted = self._eigSorting(eigVal, eigVect) # Real matrix. 

        if self.mode == 'normal':
            eigFace_matrix = copy.deepcopy(self.matrix.T @ eigVect_sorted)
            self._eigFace_matrix_full = normalization(eigFace_matrix, axis=0)
            self._weights_full = self.matrix @ self._eigFace_matrix_full # sample_num * PC_num. (max_PC_num=sample_num)
        elif self.mode == 'transpose':
            self._eigFace_matrix_full = copy.deepcopy(eigVect_sorted)
            self._weights_full = self.matrix.T @ self._eigFace_matrix_full # sample_num * PC_num. . (max_PC_num=feature_num)
        else: 
            self._eigFace_matrix_full = np.zeros(shape=eigVect.shape)
            self._weights_full = np.zeros(shape=eigVect.shape)

        self._update_eigFaces_weights()


    def eigFaces(self):
        """
        """

        return self._eigFace_matrix


    def weights(self):
        """
        """

        return self._weights


    def change_PC_num_to(self, new_PC_num):
        """
        Merely change the `PC_num`. 
        """

        self.PC_num = new_PC_num
        self._update_eigFaces_weights()

    
    def encoding(self, input_matrix):
        """
        input_matrix with Axis-0 be the sample axis. 
        """

        input_matrix_meanshifted = shifting(input_matrix, axis=0, origin=self._mean_vect, 
                                            target=np.zeros(shape=self._mean_vect.shape))
        return copy.deepcopy(input_matrix_meanshifted @ self._eigFace_matrix) # sample_num_test # weight_num
    

    def reconstructing(self, weight_matrix):
        """
        """

        data_matrix_reconstructed = copy.deepcopy(weight_matrix @ self._eigFace_matrix.T)
        return copy.deepcopy(shifting(data_matrix_reconstructed, axis=0, 
                                      origin=np.zeros(shape=self._mean_vect.shape), 
                                      target=self._mean_vect))


def eigenDecomposition(square_matrix: np.ndarray[float]
                       ) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    """_summary_

    Args:
        square_matrix (np.ndarray[float]): _description_

    Returns:
        Tuple[np.ndarray[float], np.ndarray[float]]: _description_
    """
    
    return np.linalg.eig(square_matrix)