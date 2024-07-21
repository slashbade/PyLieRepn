"""\
This file contains information about the roots of the Lie algebras. 
It follows the Bourbaki conventions if not specified otherwise.
Vectors are represented as row vectors.
"""

from PyCox import chv1r6180 as pycox

# from .weight import *
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
def dual_basis(basis: NDArray) -> NDArray:
    return np.linalg.inv(basis.T)


def change_basis(v: NDArray, basis: NDArray) -> NDArray:
    """Basis change function.

    Args:
        v (NDArray): vector in the original basis.
        basis (NDArray): new basis represented by original basis.

    Returns:
        NDArray: new vector in the new basis.
    """
    return v @ basis

def is_in_subspace(subspace_basis: NDArray, new_coord: NDArray):
    matrix_rank = np.linalg.matrix_rank(subspace_basis, tol=TOL)
    ext_matrix = np.concatenate([subspace_basis, new_coord.reshape([1, -1])], axis=0)
    ext_matrix_rank = np.linalg.matrix_rank(ext_matrix)
    if matrix_rank >= ext_matrix_rank:
        return True
    return False


def root_data(typ: str, rank: int, format: str = "bourbaki") -> NDArray:
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


def is_root_system(vectors: NDArray) -> bool:
    pass


def half_positive_sum(typ: str, rank: int) -> NDArray:
    simple_roots = root_data(typ, rank, "gap")
    roots_ = np.array(pycox.roots(pycox.cartanmat(typ, rank))[0])
    positive_roots_ = roots_[: roots_.shape[0] // 2]
    half_positive_sum_ = np.sum(positive_roots_, axis=0) / 2
    # print(half_positive_sum_[np.newaxis, :])
    return half_positive_sum_[np.newaxis, :] @ simple_roots


def as_coord(v: NDArray, typ: str, rank: int) -> NDArray:
    """Converts a vector in the root basis to the coordinate representation."""
    simple_roots = root_data(typ, rank)
    return change_basis(v, simple_roots)


def cartan_matrix_(simple_roots: NDArray) -> NDArray:
    return 2 * simple_roots @ simple_roots.T / np.sum(simple_roots**2, axis=1)


def antidominant(typ: str, rank: int, weight: NDArray, weyl: list = []) -> NDArray:
    """A fast recursive algorithm to compute the antidominant weight of a given weight.

    Args:
        typ (str): type of the Lie algebra.
        rank (int): rank of the Lie algebra.
        weight (NDArray): weight to compute the antidominant weight,
        represented in the fundamental weight basis.
        weyl (list, optional): weyl group element. Defaults to [].

    Returns:
        NDArray: the antidominant weight, represented in the fundamental weight basis.
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


def act_on_weight(typ: str, rank: int, root_index: NDArray, weight: NDArray) -> NDArray:
    """Compute the result of the action of the simple root indexed by root_index on the weight.

    Args:
        typ (str): type of the Lie algebra.
        rank (int): rank of the Lie algebra.
        root_index (NDArray): index of the simple root.
        weight (NDArray): weight to act on, represented in the fundamental weight basis.

    Returns:
        NDArray: the new weight, represented in the fundamental weight basis.
    """
    cmat = np.array(pycox.cartanmat(typ, rank))
    return weight - weight[root_index] * cmat[root_index]


def weight_partition(typ: str, rank: int, weight: NDArray):
    congruence = lambda a, b: is_integer(a - b) or is_integer(a + b)
    weights, _ = partition_equivalence(weight, congruence)
    return weights

def root_system(typ: str, rank: int) -> Tuple[NDArray, NDArray]:
    """Silimar to PyCox, representing all roots as absolute coordinates in root space,
    where first rank-th roots are simple roots and first half are positive roots.

    Args:
        typ (str): Lie type
        rank (int): rank

    Returns:
        NDArray: root system as required
    """
    simple_roots = root_data(typ, rank, "gap")
    roots = np.array(pycox.roots(pycox.coxeter(typ, rank).cartan)[0]) @ simple_roots
    return simple_roots, roots

def integral_root_system(
    typ: str, rank: int, weight: NDArray
) -> Tuple[NDArray, NDArray[np.intp]]:
    """Give a integral root system by a highest weight. Note that this func gives all roots

    Args:
        typ (str): type
        rank (int): rank
        weight (NDArray): Highest weight as np array

    Returns:
        Tuple[NDArray, NDArray[np.intp]]: roots and indices
    """ 
    simple_roots = root_data(typ, rank, "gap")
    # Trigger PyCox root enumerating, but with a transformation
    roots = np.array(pycox.roots(pycox.coxeter(typ, rank).cartan)[0]) @ simple_roots
    # print((roots @ weight)[12])
    roots_weight_ind = np.argwhere(
        is_integer(2 * roots @ weight / np.sum(roots**2, axis=1))
    ).ravel()
    return roots[roots_weight_ind], roots_weight_ind

def integral_root_system_compl(
    typ: str, rank: int, weight: NDArray
) -> Tuple[NDArray, NDArray]:
    """Compute a complement root system of integral root system

    Args:
        typ (str): ttype
        rank (int): rank
        weight (NDArray): weight
    """
    simple_roots = root_data(typ, rank, "gap")
    
    # Trigger PyCox root enumerating, but with a transformation
    roots = np.array(pycox.roots(pycox.coxeter(typ, rank).cartan)[0]) @ simple_roots
    
    # Compute integral roots
    integral_roots, integral_root_ind = integral_root_system(typ, rank, weight)
    
    # Get Simple root for basis
    decomposed_intl_rts = root_system_decomposition(integral_roots)
    decomposed_spl_rts = []
    for comp_posi_rts in decomposed_intl_rts[0]:
        comp_sp_rts, _ = simple_roots_of_positive_roots(comp_posi_rts)
        decomposed_spl_rts.append(comp_sp_rts)
    combine_basis = np.concatenate(decomposed_spl_rts, axis=0)
        
    # Compute list diff
    integral_root_compl_ind = []
    for ind in range(len(roots)):
        if ind not in integral_root_ind:
            integral_root_compl_ind.append(ind)
    
    # Find roots that cannot be spanned by proposed system basis
    intl_rt_compl_basis_ind = []
    for ind in integral_root_compl_ind:
        # this works by linear independecy
        if not is_in_subspace(combine_basis, roots[ind]):
            intl_rt_compl_basis_ind.append(ind)
    
            
    
    return roots[intl_rt_compl_basis_ind], intl_rt_compl_basis_ind

def root_system_decomposition(roots: NDArray) -> Tuple[list[NDArray], list[list]]:
    """Decomposition is carried out on positive roots

    Args:
        roots (NDArray): root

    Returns:
        Tuple[list[NDArray], list[list]]: decomposed positive systems with their index
        in orginal system.
    """
    positive_roots = roots[: roots.shape[0] // 2]
    mat = np.abs(positive_roots @ positive_roots.T) > TOL
    for i in range(mat.shape[0]):
        mat[i, i] = False
    root_graph = nx.from_numpy_array(mat)
    decomposed = [
        np.array([positive_roots[i] for i in c])
        for c in nx.connected_components(root_graph)
    ]
    decomposed_ind = [
        [i for i in c] for c in nx.connected_components(root_graph)
    ]
    return decomposed, decomposed_ind


def simple_roots_of_positive_roots(positive_roots: NDArray) -> Tuple[NDArray, NDArray]:

    half_positive_sum = np.sum(positive_roots, axis=0) / 2
    aprod = 2 * half_positive_sum @ positive_roots.T / np.sum(positive_roots**2, axis=1)
    simple_roots = positive_roots[aprod == 1]
    simple_roots_ind = np.argwhere(aprod == 1).ravel()
    return simple_roots, simple_roots_ind



def cartan_type(positive_roots: NDArray, simple_roots: NDArray) -> Tuple[str, int]:
    rank = len(simple_roots)
    # Check root lengths
    root_lengths = np.linalg.norm(positive_roots, axis=1)
    root_unique_lengths = np.unique(root_lengths)
    root_length_num = root_unique_lengths.shape[0]
    if root_length_num == 1:
        # same root lengths -> case: A D E
        if len(positive_roots) == 36 and rank == 6:
            return "E", 6
        elif len(positive_roots) == 63 and rank == 7:
            return "E", 7
        elif len(positive_roots) == 120 and rank == 8:
            return "E", 8
        elif rank * (rank + 1) // 2 == len(positive_roots):
            return "A", rank
        elif rank * (rank - 1) == len(positive_roots):
            return "D", rank
        else:
            raise ValueError

    elif root_length_num == 2:
        # long roots and short roots -> case: B C F G
        if len(positive_roots) == 24 and rank == 4:
            return "F", 4
        elif len(positive_roots) == 6 and rank == 2:
            return "G", 2
        elif len(positive_roots) == rank**2:
            short_length = np.min(root_unique_lengths)
            long_length = np.max(root_unique_lengths)
            if len(positive_roots[root_lengths == short_length]) == rank and rank >= 2:
                return "B", rank
            elif len(positive_roots[root_lengths == long_length]) == rank and rank >= 3:
                return "C", rank
            else:
                raise ValueError
        # elif len(posi)
        else:
            raise ValueError
    else:
        raise ValueError


if __name__ == "__main__":
    np.set_printoptions(
        precision=3,
        suppress=True,
    )
    typ, rank = "E", 7
    W = pycox.coxeter(typ=typ, rank=rank)
    v = pycox.lpol([1], 1, "v")
    rt1, rt_ind1 = integral_root_system(
        "D", 8, np.array([2, 1, 1.1, 3, 0.9, 1.9, 4, 2.1])
    )
    rt2, rt_ind2 = integral_root_system("F", 4, np.array([4, 5, 3 / 2, 1 / 2]))
    rt3, rt_ind3 = integral_root_system(
        "E", 8, np.array([1 / 2, -3 / 2, -3, -2, -1, -4, -5, -19])
    )
    rt4, rt_ind4 = integral_root_system("F", 4, np.array([7 / 4, 1 / 4, 5 / 4, -3 / 4]))
    rt5, rt_ind5 = integral_root_system(
        "E", 6, np.array([1, 2, 1, 4, 4.5, 0.5, 0.5, -0.5])
    )
    rt6, rt_ind6 = integral_root_system(
        "E", 7, np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4, -3 / 4, -1, 1])
    )
    rt7, rt_ind7 = integral_root_system("E", 8, np.array([1, 5, 9, 13, 9, 1, 5, 9]) / 4)
    rt8, rt_ind8 = integral_root_system(
        "E", 7, np.array([1, 3, 5, -7, -9, -11, -1 / 2, 1 / 2])
    )
    rt9, rt_ind9 = integral_root_system(
        "E", 8, np.array([1, 1, 1, 1, 1, 1, 1 / 2, 5 / 2])
    )
    rt10, rt_ind10 = integral_root_system(
        'F', 4, np.array([7/4, 1/4, 5/4, -3/4])
    )
    
    rt1_compl, rt1_compl_ind = integral_root_system_compl(
        "D", 8, np.array([2, 1, 1.1, 3, 0.9, 1.9, 4, 2.1])
    )
    prts, prtsi = root_system_decomposition(rt1_compl)
    result = []
    for prt in prts:
        srt, srti = simple_roots_of_positive_roots(prt)
        print(cartan_type(prt, srt))
        result.append(cartan_type(prt, srt))
    print(result)


    # print(half_positive_sum('A', 5))
    # print(simple_roots_of_positive_roots(pts[0]))
