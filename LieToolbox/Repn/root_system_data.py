import numpy as np
from numpy.typing import NDArray
from ..PyCox import chv1r6180 as pycox

def simple_root_data(typ: str, rank: int, format: str = "gap") -> NDArray:
    if typ == "A":
        mat = np.zeros((rank, rank + 1))
        for i in range(rank):
            mat[i, i : i + 2] = [1, -1]
    elif typ == "B":
        mat = np.zeros((rank, rank))
        for i in range(rank - 1):
            mat[i, i : i + 2] = [1, -1]
        mat[rank - 1, rank - 1] = 1
    elif typ == "C":
        mat = np.zeros((rank, rank))
        for i in range(rank - 1):
            mat[i, i : i + 2] = [1, -1]
        mat[rank - 1, rank - 1] = 2
    elif typ == "D":
        mat = np.zeros((rank, rank))
        for i in range(rank - 1):
            mat[i, i : i + 2] = [1, -1]
        mat[rank - 1, rank - 2 : rank] = [1, 1]
    elif typ == "E":
        if rank == 6:
            mat = np.zeros((rank, 8))
            mat[0, :] = [0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5]
            mat[1, :2] = [1, 1]
            mat[2, 0:2] = [-1, 1]
            mat[3, 1:3] = [-1, 1]
            mat[4, 2:4] = [-1, 1]
            mat[5, 3:5] = [-1, 1]
        elif rank == 7:
            mat = np.zeros((rank, 8))
            mat[0, :] = [0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5]
            mat[1, :2] = [1, 1]
            mat[2, 0:2] = [-1, 1]
            mat[3, 1:3] = [-1, 1]
            mat[4, 2:4] = [-1, 1]
            mat[5, 3:5] = [-1, 1]
            mat[6, 4:6] = [-1, 1]
        elif rank == 8:
            mat = np.zeros((rank, 8))
            mat[0, :] = [0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5]
            mat[1, :2] = [1, 1]
            mat[2, 0:2] = [-1, 1]
            mat[3, 1:3] = [-1, 1]
            mat[4, 2:4] = [-1, 1]
            mat[5, 3:5] = [-1, 1]
            mat[6, 4:6] = [-1, 1]
            mat[7, 5:7] = [-1, 1]
    elif typ == "F" and rank == 4:
        mat = np.zeros((rank, rank))
        mat[0, 1:3] = [1, -1]
        mat[1, 2:4] = [1, -1]
        mat[2, 3] = 1
        mat[3, :] = [0.5, -0.5, -0.5, -0.5]
    elif typ == "G" and rank == 2:
        mat = np.zeros((rank, 3))
        mat[1, :] = [1, -1, 0]
        mat[0, :] = [-2, 1, 1]

    if format == "bourbaki":
        return mat
    elif format == "gap":
        if typ in ["B", "C", "D"]:
            return mat[::-1]
        else:
            return mat


def fundamental_weight_data(typ: str, rank: int, format: str = "bourbaki") -> NDArray:
    """Fundamental weights of the Lie algebra in terms of the simple roots.

    Returns:
        NDArray: each line is a fundamental weight.
    """
    if typ == "A":
        mat = np.zeros((rank, rank))
        for i in range(rank):
            for j in range(i):
                mat[i, j] = (rank - i) * (j + 1) / (rank + 1)
            for j in range(i, rank):
                mat[i, j] = (i + 1) * (rank - j) / (rank + 1)
    elif typ == "B":
        mat = np.zeros((rank, rank))
        for i in range(rank - 1):
            for j in range(i):
                mat[i, j] = j + 1
            for j in range(i, rank):
                mat[i, j] = i + 1
        mat[rank - 1, :] = [0.5 * (j + 1) for j in range(rank)]
    elif typ == "C":
        mat = np.zeros((rank, rank))
        for i in range(rank):
            for j in range(i):
                mat[i, j] = j + 1
            for j in range(i, rank - 1):
                mat[i, j] = i + 1
            mat[i, rank - 1] = 0.5 * (i + 1)
    elif typ == "D":
        mat = np.zeros((rank, rank))
        for i in range(rank - 2):
            for j in range(i):
                mat[i, j] = j + 1
            for j in range(i, rank - 2):
                mat[i, j] = i + 1
            mat[i, rank - 2] = 0.5 * (i + 1)
            mat[i, rank - 1] = 0.5 * (i + 1)
        for j in range(rank - 2):
            mat[rank - 2, j] = 0.5 * (j + 1)
            mat[rank - 1, j] = 0.5 * (j + 1)
        mat[rank - 2, rank - 2] = 0.25 * rank
        mat[rank - 2, rank - 1] = 0.25 * (rank - 2)
        mat[rank - 1, rank - 2] = 0.25 * (rank - 2)
        mat[rank - 1, rank - 1] = 0.25 * rank
    elif typ == "E" and rank == 6:
        mat = np.zeros((rank, rank))
        mat[0, :] = np.array([4, 3, 5, 6, 4, 2]) / 3
        mat[1, :] = np.array([1, 2, 2, 3, 2, 1])
        mat[2, :] = np.array([5, 6, 10, 12, 8, 4]) / 3
        mat[3, :] = np.array([2, 3, 4, 6, 4, 2])
        mat[4, :] = np.array([4, 6, 8, 12, 10, 5]) / 3
        mat[5, :] = np.array([2, 3, 4, 6, 5, 4]) / 3
    elif typ == "E" and rank == 7:
        mat = np.zeros((rank, rank))
        mat[0, :] = np.array([2, 2, 3, 4, 3, 2, 1])
        mat[1, :] = np.array([4, 7, 8, 12, 9, 6, 3]) / 2
        mat[2, :] = np.array([3, 4, 6, 8, 6, 4, 2])
        mat[3, :] = np.array([4, 6, 8, 12, 9, 6, 3])
        mat[4, :] = np.array([6, 9, 12, 18, 15, 10, 5]) / 2
        mat[5, :] = np.array([2, 3, 4, 6, 5, 4, 2])
        mat[6, :] = np.array([2, 3, 4, 6, 5, 4, 3]) / 2
    elif typ == "E" and rank == 8:
        mat = np.zeros((rank, rank))
        mat[0, :] = np.array([4, 5, 7, 10, 8, 6, 4, 2])
        mat[1, :] = np.array([5, 8, 10, 15, 12, 9, 6, 3])
        mat[2, :] = np.array([7, 10, 14, 20, 16, 12, 8, 4])
        mat[3, :] = np.array([10, 15, 20, 30, 24, 18, 12, 6])
        mat[4, :] = np.array([8, 12, 16, 24, 20, 15, 10, 5])
        mat[5, :] = np.array([6, 9, 12, 18, 15, 12, 8, 4])
        mat[6, :] = np.array([4, 6, 8, 12, 10, 8, 6, 3])
        mat[7, :] = np.array([5, 8, 10, 15, 12, 9, 6, 3])
    elif typ == "F" and rank == 4:
        mat = np.zeros((rank, rank))
        mat[0, :] = np.array([2, 3, 4, 2])
        mat[1, :] = np.array([3, 6, 8, 4])
        mat[2, :] = np.array([2, 4, 6, 3])
        mat[3, :] = np.array([1, 2, 3, 2])
    elif typ == "G" and rank == 2:
        mat = np.zeros((rank, rank))
        mat[0, :] = np.array([2, 1])
        mat[1, :] = np.array([3, 2])

    if format == "bourbaki":
        return mat
    elif format == "gap":
        if typ in ["A", "B", "C", "D"]:
            return mat[::-1, ::-1]
        else:
            return mat

def num_positive_roots_data(typ: str, rank: int) -> int:
    if typ == "A":
        return rank * (rank + 1) // 2
    elif typ == "B":
        return rank ** 2
    elif typ == "C":
        return rank ** 2
    elif typ == "D":
        return rank * (rank - 1)
    elif typ == "E":
        if rank == 6:
            return 36
        elif rank == 7:
            return 63
        elif rank == 8:
            return 120
    elif typ == "F":
        return 24
    elif typ == "G":
        return 6


def roots_pycox(typ, rank):
    return np.array(pycox.roots(pycox.cartanmat(typ, rank))[0])

def cartan_matrix_pycox(typ, rank):
    return np.array(pycox.cartanmat(typ, rank))
