"""\
This file contains information about the roots of the Lie algebras. 
It follows the Bourbaki conventions if not specified otherwise.
Vectors are represented as row vectors.
"""

from ..PyCox import chv1r6180 as pycox
from .weight import *
import numpy as np
import cProfile
from typing import Iterable, List, Tuple, Callable, Any
from numpy.typing import NDArray, ArrayLike
import networkx as nx

TOL = 1e-7

# Essential Algorithms
def is_integer(x: float, tol: float = TOL) -> bool:
    """Justify whether a value is an integer at a precision of tol

    Args:
        x (float): a float value
        tol (float, optional): tolerance. Defaults to 1e-7.
    """
    return np.abs(x - np.round(x)) < tol

def is_half_integer(x: float, tol: float = TOL) -> bool:
    return is_integer(2 * x, tol)

def partition_equivalence(
    l: Iterable[Any], r: Callable[[Any, Any], bool]
) -> Tuple[List[List[Any]], List[int]]:
    """Partition a list into equivalence classes.

    Args:
        l (Iterable[Any]): list to partition.
        r (Callable[[Any, Any], bool]): equivalence relation.

    Returns:
        Tuple[List[List[Any]], List[int]]: partitions and the indices of
        the elements in the original list.
    """
    partitions = []
    ind_partitions = []
    for index, elem in enumerate(l):
        found = False
        for i, p in zip(ind_partitions, partitions):
            if r(p[0], elem):
                p.append(elem)
                i.append(index)
                found = True
                break
        if not found:
            partitions.append([elem])
            ind_partitions.append([index])
    return partitions, ind_partitions


# Linear algebra functions
def dual_basis(basis: np.ndarray) -> np.ndarray:
    return np.linalg.inv(basis.T)


def change_basis(v: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Basis change function.

    Args:
        v (np.ndarray): vector in the original basis.
        basis (np.ndarray): new basis represented by original basis.

    Returns:
        np.ndarray: new vector in the new basis.
    """
    return v @ basis

def root_data(typ: str, rank: int, format: str = "bourbaki") -> np.ndarray:
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
        mat[0, :] = [1, -1, 0]
        mat[1, :] = [-2, 1, 1]

    if format == "bourbaki":
        return mat
    elif format == "gap":
        if typ in ["A", "B", "C", "D"]:
            return mat[::-1]
        else:
            return mat

def fundamental_weight_data(
    typ: str, rank: int, format: str = "bourbaki"
) -> np.ndarray:
    """Fundamental weights of the Lie algebra in terms of the simple roots.

    Returns:
        np.ndarray: each line is a fundamental weight.
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


def is_root_system(vectors: NDArray) -> bool:
    pass

def half_positive_sum(typ: str, rank: int) -> NDArray:
    simple_roots = root_data(typ, rank, 'gap')
    roots_ = np.array(pycox.roots(pycox.cartanmat(typ, rank))[0])
    positive_roots_ = roots_[:roots_.shape[0] // 2]
    half_positive_sum_ = np.sum(positive_roots_, axis=0) / 2
    # print(half_positive_sum_[np.newaxis, :])
    return half_positive_sum_[np.newaxis, :] @ simple_roots


def as_coord(v: np.ndarray, typ: str, rank: int) -> np.ndarray:
    """Converts a vector in the root basis to the coordinate representation."""
    simple_roots = root_data(typ, rank)
    return change_basis(v, simple_roots)


def cartan_matrix_(simple_roots: np.ndarray) -> np.ndarray:
    return 2 * simple_roots @ simple_roots.T / np.sum(simple_roots**2, axis=1)


def antidominant(
    typ: str, rank: int, weight: np.ndarray, weyl: list = []
) -> np.ndarray:
    """A fast recursive algorithm to compute the antidominant weight of a given weight.

    Args:
        typ (str): type of the Lie algebra.
        rank (int): rank of the Lie algebra.
        weight (np.ndarray): weight to compute the antidominant weight,
        represented in the fundamental weight basis.
        weyl (list, optional): weyl group element. Defaults to [].

    Returns:
        np.ndarray: the antidominant weight, represented in the fundamental weight basis.
    """
    if np.all(weight <= 0):
        return weyl, weight
    else:
        max_pos_index = np.argwhere(weight > 0)[-1][0]
        new_weyl = weyl + [max_pos_index]
        new_weight = act_on_weight(typ, rank, max_pos_index, weight)
        # print(f"weight: {weight}, maximum positive index: {max_pos_index + 1}")
        # print(f"{max_pos_index + 1}th simple root action, new weight: {new_weight}")
        return antidominant(typ, rank, new_weight, new_weyl)


def act_on_weight(
    typ: str, rank: int, root_index: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """Compute the result of the action of the simple root indexed by root_index on the weight.

    Args:
        typ (str): type of the Lie algebra.
        rank (int): rank of the Lie algebra.
        root_index (np.ndarray): index of the simple root.
        weight (np.ndarray): weight to act on, represented in the fundamental weight basis.

    Returns:
        np.ndarray: the new weight, represented in the fundamental weight basis.
    """
    cmat = np.array(pycox.cartanmat(typ, rank))
    return weight - weight[root_index] * cmat[root_index]

def weight_partition(typ: str, rank: int, weight: np.ndarray):
    congruence = lambda a, b : is_integer(a - b) or is_integer(a + b)
    weights, _ = partition_equivalence(weight, congruence)
    return weights
    
def integral_root_system(typ: str, rank: int, weight: np.ndarray) -> Tuple[np.ndarray, NDArray[np.intp]]:
    simple_roots = root_data(typ, rank, "gap")
    roots = np.array(pycox.roots(pycox.coxeter(typ, rank).cartan)[0]) @ simple_roots
    # print((roots @ weight)[12])
    roots_weight_ind = np.argwhere(is_integer(2 * roots @ weight / np.sum(roots**2, axis=1))).ravel()
    return roots[roots_weight_ind], roots_weight_ind

def root_system_decomposition(roots: NDArray) -> np.ndarray:
    positive_roots = roots[:roots.shape[0]//2]
    mat = np.abs(positive_roots @ positive_roots.T) > TOL
    for i in mat.shape[0]:
        mat[i, i] = False
    root_graph = nx.from_numpy_array(mat)
    decomposed = [np.array([positive_roots[i] for i in c])
                  for c in nx.connected_components(root_graph)]
    return decomposed

def simple_roots_of_positive_roots(positive_roots: NDArray) -> NDArray:
    pass

def cartan_type(roots: NDArray) -> Tuple[str, int, NDArray]:
    pass

if __name__ == "__main__":
    np.set_printoptions(
        precision=3,
        suppress=True,
    )
    typ, rank = "E", 7
    W = pycox.coxeter(typ=typ, rank=rank)
    v = pycox.lpol([1], 1, "v")
    # cProfile.run("pycox.klcells(W, 1, v)", sort='cumtime', filename='klcells.profile')
    # replm = pycox.klcells(W, 1, v)
    # print(replm)
    # print(np.array(pycox.roots(W.cartan)[0]))
    # simple_roots = root_data('F', 4)
    # cmat = cartan_matrix_(simple_roots)
    # print(cmat)
    roots = np.array(pycox.roots(W.cartan)[0]) @ root_data(typ, rank, format="gap")
    simple_roots = roots[:rank]
    fundamental_weights = fundamental_weight_data(typ, rank, "gap") @ root_data(
        typ, rank, "gap"
    )
    ttt = 2 * fundamental_weights @ simple_roots.T / np.sum(simple_roots**2, axis=1)
    # print(fundamental_weight_data(typ, rank, 'bourbaki') @ root_data(typ, rank, 'bourbaki'))
    # print(np.round(ttt, 8) + 0)
    # lbd = Weight([2, 1, 1.1, 3, 0.9, 1.9, 4, 2.1], 'D')
    lbd_1 = np.array([2, -1, 0, -5, -6, -8, 12, -12])
    lbd_2 = np.array([1, 1, 1, 1, 1, 1, -1, 1])
    lbd_3 = np.array([-1, 2, 0, -5, -6, -8, 12, -12])
    lbd_1_new_coord = 2 * lbd_1 @ simple_roots.T / np.sum(simple_roots**2, axis=1)
    lbd_2_new_coord = 2 * lbd_2 @ simple_roots.T / np.sum(simple_roots**2, axis=1)
    lbd_3_new_coord = 2 * lbd_3 @ simple_roots.T / np.sum(simple_roots**2, axis=1)
    print(lbd_2_new_coord)
    weyl, antidominant_lbd = antidominant(typ, rank, lbd_2_new_coord, [])
    print(f"anti-dominant: {antidominant_lbd}\nweyl: {weyl}")
    print(f"reduced word: {W.reducedword(weyl, W)}")

    def relation(a, b):
        return is_integer(a - b) or is_integer(a + b)
    
    p, ip = partition_equivalence([2, 1, 1.1, 3, 0.9, 1.9, 4, 2.1, 0.3, 1.7, 0.6, 1.6, 0.5], relation)
    print(p)
    # lbd.decomposition().show()
    rt = integral_root_system('D', 8, np.array([2, 1, 1.1, 3, 0.9, 1.9, 4, 2.1]))
    # print(rt.shape)
    # pts, ipts = direct_sum_decomposition(rt)
    print(half_positive_sum('A', 5))
    
    