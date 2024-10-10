import numpy as np
from scipy.linalg import null_space
from typing import Iterable, List, Tuple, Callable, Any
from numpy.typing import NDArray, ArrayLike

TOL = 1e-7
# Essential Algorithms
def is_integer(x: float, tol: float = TOL) -> bool:
    """Justify whether a value is an integer at a precision of tol

    Args:
        x (float): a float value
        tol (float, optional): tolerance. Defaults to 1e-7.
    """
    return np.abs(x - np.round(x)) < tol

def is_integer_array(xl: NDArray, tol: float = TOL) -> bool:
    return np.all(np.abs(xl - np.round(xl)) < tol)

def is_zero(x: float, tol: float = TOL) -> bool:
    return np.abs(x) < tol

def is_one(x: float, tol: float = TOL) -> bool:
    return np.abs(x - 1) < tol


def is_half_integer(x: float, tol: float = TOL) -> bool:
    return is_integer(2 * x, tol)


def round2_one(x: float) -> int | float:
    if is_integer(x):
        return int(np.round(x))
    elif is_half_integer(x):
        return int(np.round(2 * x)) / 2
    else:
        return x

def round2(xl: NDArray) -> NDArray:
    return np.array([round2_one(x) for x in xl])


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

def find_complement(subspace_basis1: NDArray, subspace_basis2: NDArray) -> NDArray:
    return null_space(subspace_basis1 @ subspace_basis2.T).T

def is_in_subspace(subspace_basis: NDArray, new_coord: NDArray):
    matrix_rank = np.linalg.matrix_rank(subspace_basis, tol=TOL)
    ext_matrix = np.concatenate([subspace_basis, new_coord.reshape([1, -1])], axis=0)
    ext_matrix_rank = np.linalg.matrix_rank(ext_matrix)
    if matrix_rank >= ext_matrix_rank:
        return True
    return False


def embed_array(v, dim):
    assert v.shape[0] <= dim
    return np.concatenate([v, np.zeros(dim - v.shape[0])])

def restrict_array(v, dim):
    assert v.shape[0] >= dim
    return v[:dim]

def embed_basis(basis, dim):
    assert basis.shape[1] <= dim
    return np.concatenate([basis, np.zeros((basis.shape[0], dim - basis.shape[1]))], axis=1)

def restrict_basis(basis, dim):
    assert basis.shape[1] >= dim
    return basis[:, :dim]


def pretty_print_array(array: NDArray, symbol='\\epsilon') -> str:
    lst = []
    is_first = True
    for i in range(array.shape[0]):
        if is_zero(array[i]):
            continue
        elif is_one(array[i]):
            coo = f"{symbol}_{i+1}"
        elif is_one(-array[i]):
            coo = f"-{symbol}_{i+1}"
        elif is_integer(array[i]):
            coo = f"{int(np.round(array[i]))}{symbol}_{i+1}"
        elif is_half_integer(array[i]):
            if array[i] > 0:
                coo = f"\\frac{{{int(np.round(2 * array[i]))}}}{{2}}{symbol}_{i+1}"
            else:
                coo = f"-\\frac{{{int(np.round(-2 * array[i]))}}}{{2}}{symbol}_{i+1}"
        else:
            coo = f"{array[i]:.2}{symbol}_{i+1}"
        
        if array[i] > 0 and not is_first:
            lst.append('+' + coo)
            # print(i)
        else:
            lst.append(coo)
        is_first = False
    if is_first:
        lst.append('0')
    return "".join(lst)


def pretty_print_basis(basis: NDArray) -> str:
    if basis.shape[0] == 0:
        return '\emptyset'
    return '\{' + ', '.join([pretty_print_array(basis[i]) for i in range(basis.shape[0])]) + '\}'

def pretty_print_basises(basis: NDArray) -> str:
    return ' \\times '.join([pretty_print_basis(b) for b in basis])

def pretty_print_weight(weight: NDArray) -> str:
    lst = []
    for i in range(weight.shape[0]):
        if is_integer(weight[i]):
            lst.append(f"{int(np.round(weight[i]))}")
        elif is_half_integer(weight[i]):
            if weight[i] > 0:
                lst.append(f"\\frac{{{int(np.round(2 * weight[i]))}}}{{2}}")
            else:
                lst.append(f"-\\frac{{{int(np.round(-2 * weight[i]))}}}{{2}}")
        else:
            lst.append(f"{weight[i]:.4}")
    return '(' + ', '.join(lst) + ')'

def pretty_print_weight_(weight: NDArray) -> str:
    return pretty_print_array(weight, '\\omega')

def pretty_print_lietype(typ: str, rank: int) -> str:
    return f"{typ}_{rank}"

def pretty_print_lietypes(lietypes: List[Tuple[str, int]]) -> str:
    return ' \\times '.join([pretty_print_lietype(*lt) for lt in lietypes])

def parse_float(x):
    if is_integer(x):
        return int(np.round(x))
    else:
        return np.round(x, 3)

def pretty_print_matrix(matrix: NDArray) -> str:
    return '\\begin{pmatrix}' + '\\\\'.join(
        [' & '.join([f"{parse_float(matrix[i, j])}" 
                     for j in range(matrix.shape[1])]) for i in range(matrix.shape[0])]) + '\\end{pmatrix}'

def pretty_print_character(character: str) -> str:
    def parse_ch(c):
        return c.replace('phi', '\\phi')
    return parse_ch(character)