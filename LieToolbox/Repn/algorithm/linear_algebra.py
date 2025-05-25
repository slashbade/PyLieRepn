import numpy as np

from scipy.linalg import null_space

TOL = 1e-7

# Linear algebra functions
class LinearAlgebra:
    @staticmethod
    def dual_basis(basis: np.ndarray) -> np.ndarray:
        return np.linalg.inv(basis.T)

    @staticmethod
    def change_basis(v: np.ndarray, basis: np.ndarray) -> np.ndarray:
        """Basis change function.

        Args:
            v (NDArray): vector in the original basis.
            basis (NDArray): new basis represented by original basis.

        Returns:
            NDArray: new vector in the new basis.
        """
        return v @ basis
    
    @staticmethod
    def find_complement(subspace_basis1: np.ndarray, subspace_basis2: np.ndarray) -> np.ndarray:
        return null_space(subspace_basis1 @ subspace_basis2.T).T

    @staticmethod
    def is_in_subspace(subspace_basis: np.ndarray, new_coord: np.ndarray) -> bool:
        matrix_rank = np.linalg.matrix_rank(subspace_basis, tol=TOL)
        ext_matrix = np.concatenate([subspace_basis, new_coord.reshape([1, -1])], axis=0)
        ext_matrix_rank = np.linalg.matrix_rank(ext_matrix)
        if matrix_rank >= ext_matrix_rank:
            return True
        return False

    @staticmethod
    def embed_array(v: np.ndarray, dim: int) -> np.ndarray:
        assert v.shape[0] <= dim
        return np.concatenate([v, np.zeros(dim - v.shape[0])])
    
    @staticmethod
    def restrict_array(v: np.ndarray, dim: int) -> np.ndarray:
        assert v.shape[0] >= dim
        return v[:dim]

    @staticmethod
    def embed_basis(basis: np.ndarray, dim: int) -> np.ndarray:
        assert basis.shape[1] <= dim
        return np.concatenate([basis, np.zeros((basis.shape[0], dim - basis.shape[1]))], axis=1)

    @staticmethod
    def restrict_basis(basis: np.ndarray, dim: int) -> np.ndarray:
        assert basis.shape[1] >= dim
        return basis[:, :dim]


