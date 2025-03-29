"""\
This file contains information about the roots of the Lie algebras. 
It follows the Bourbaki conventions if not specified otherwise.
Vectors are represented as row vectors.

Usually, roots and weights are represented as the orthonormal basis. 
If roots are represented in the simple root basis, it will have a 
suffix "_". If weights are represented in the fundamental weight basis,
it will have a suffix "_".

"""
import sys
sys.path.append("..")
from .root_system_data import simple_root_data, roots_pycox
from .utils import *
import numpy as np
from numpy.typing import NDArray
import networkx as nx
from dataclasses import dataclass


LieType = tuple[str, int]


def as_fundamental_weight_basis(v: NDArray, simple_roots: NDArray) -> NDArray:
    """Converts a vector in the root basis to the fundamental weight basis."""
    return change_basis(v, np.linalg.inv(cartan_matrix_(simple_roots)))


def compute_fundamental_weights_(typ: str, rank: int, simple_roots: NDArray) -> NDArray:
    """Compute the fundamental weights of the Lie algebra.

    Args:
        typ (str): type of the Lie algebra.
        rank (int): rank of the Lie algebra.
        simple_root (NDArray): simple roots of the Lie algebra.

    Returns:
        NDArray: fundamental weights of the Lie algebra.
    """
    cmat = cartan_matrix_(simple_roots)
    return np.linalg.inv(cmat)

def compute_fundamental_weights(simple_roots: NDArray) -> NDArray:
    return np.linalg.inv(cartan_matrix_(simple_roots)) @ simple_roots

@dataclass
class RootSystem:
    roots: NDArray
    positive_roots: NDArray
    simple_roots: NDArray
    

def is_root_system(vectors: NDArray) -> bool:
    pass


def half_positive_sum(typ: str, rank: int) -> NDArray:
    simple_roots = simple_root_data(typ, rank, "gap")
    roots_ = roots_pycox(typ, rank)
    positive_roots_ = roots_[: roots_.shape[0] // 2]
    half_positive_sum_ = np.sum(positive_roots_, axis=0) / 2
    # print(half_positive_sum_[np.newaxis, :])
    return half_positive_sum_[np.newaxis, :] @ simple_roots


def as_coord(v: NDArray, typ: str, rank: int) -> NDArray:
    """Converts a vector in the root basis to the coordinate representation."""
    simple_roots = simple_root_data(typ, rank)
    return change_basis(v, simple_roots)


def cartan_matrix_(simple_roots: NDArray) -> NDArray:
    return 2 * simple_roots @ simple_roots.T / np.sum(simple_roots**2, axis=1)




def root_system(typ: str, rank: int) -> tuple[NDArray, NDArray]:
    """Silimar to PyCox, representing all roots as absolute coordinates in root space,
    where first rank-th roots are simple roots and first half are positive roots.

    Args:
        typ (str): Lie type
        rank (int): rank

    Returns:
        NDArray: root system as required
    """
    simple_roots = simple_root_data(typ, rank, "gap")
    roots = roots_pycox(typ, rank) @ simple_roots
    return simple_roots, roots

def integral_root_system(
    typ: str, rank: int, weight: NDArray
) -> tuple[NDArray, NDArray[np.intp]]:
    """Give a integral root system by a highest weight. Note that this func gives all roots

    Args:
        typ (str): type
        rank (int): rank
        weight (NDArray): Highest weight as np array

    Returns:
        Tuple[NDArray, NDArray[np.intp]]: roots and indices
    """ 
    simple_roots = simple_root_data(typ, rank, "gap")
    # Trigger PyCox root enumerating, but with a transformation
    roots = roots_pycox(typ, rank) @ simple_roots
    roots_weight_ind = np.argwhere(
        is_integer(2 * roots @ weight / np.sum(roots**2, axis=1))
    ).ravel()
    # print(roots)
    # print(2 * roots @ weight / np.sum(roots**2, axis=1))
    return roots[roots_weight_ind], roots_weight_ind


def root_system_decomposition(roots: NDArray) -> tuple[list[NDArray], list[list]]:
    """Decomposition is carried out on positive roots

    Args:
        roots (NDArray): root

    Returns:
        Tuple[list[NDArray], list[list]]: decomposed positive systems with their index
        in orginal system.
    """
    # This is a bit tricky, but we can use the fact that the positive roots are symmetric
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


def simple_roots_from_irreducible(positive_roots: NDArray) -> tuple[NDArray, NDArray]:
    """Get simple roots from positive roots

    Returns:
        Tuple[NDArray, NDArray]: simple roots and their indices
    """
    half_positive_sum = np.sum(positive_roots, axis=0) / 2
    aprod = 2 * half_positive_sum @ positive_roots.T / np.sum(positive_roots**2, axis=1)
    simple_roots = positive_roots[aprod == 1]
    simple_roots_ind = np.argwhere(aprod == 1).ravel()
    return simple_roots, simple_roots_ind


def simple_roots(roots: NDArray) -> NDArray:
    """Get simple roots from the root system.

    Args:
        roots (NDArray): root system.

    Returns:
        NDArray: simple roots.
    """
    decomposed, _ = root_system_decomposition(roots)
    simple_roots = []
    for comp_posi_rts in decomposed:
        comp_sp_rts, _ = simple_roots_from_irreducible(comp_posi_rts)
        simple_roots.append(comp_sp_rts)
    return simple_roots


def cartan_type_from_irreducible(positive_roots: NDArray, simple_roots: NDArray) -> tuple[str, int]:
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


def get_cartan_type(roots: NDArray) -> tuple[list[LieType], list[NDArray]]:
    """Get the Cartan type of the the root system.

    Args:
        roots (NDArray): root system.

    Returns:
        Tuple[str, int]: type and rank of the Lie algebra.
    """
    decomposed, _ = root_system_decomposition(roots)
    cartan_type : list[LieType] = []
    simple_roots = []
    for comp_posi_rts in decomposed:
        comp_sp_rts, _ = simple_roots_from_irreducible(comp_posi_rts)
        typ, rank = cartan_type_from_irreducible(comp_posi_rts, comp_sp_rts)
        cartan_type.append((typ, rank))
        simple_roots.append(comp_sp_rts)
    return cartan_type, simple_roots


def get_dynkin_diagram(simple_roots: NDArray) -> nx.Graph:
    """Generate the Dynkin diagram of the Lie algebra.

    Args:
        simple_roots (NDArray): simple roots of the Lie algebra.

    Returns:
        nx.Graph: Dynkin diagram of the Lie algebra.
    """
    G = nx.Graph()
    for i, simple_root in enumerate(simple_roots):
        G.add_node(i, simple_root=simple_root)
    
    cmat = cartan_matrix_(simple_roots)
    graph_mat = np.zeros_like(cmat)
    for i in range(cmat.shape[0]):
        for j in range(i+1, cmat.shape[1]):
            num_edges = np.round(cmat[i, j] * cmat[j, i])
            if num_edges > 0:
                G.add_edge(i, j, weight=num_edges, cij=cmat[i, j], cji=cmat[j, i])
    return G
    

def reorder_simple_roots(simple_roots: NDArray, typ: str, rank: int) -> NDArray:
    """Reorder the simple roots according to the Bourbaki ordering.

    Args:
        simple_roots (NDArray): simple roots.
        typ (str): type of the Lie algebra.
        rank (int): rank of the Lie algebra.

    Returns:
        NDArray: reordered simple roots.
    """
    G = get_dynkin_diagram(simple_roots)
    G_canonical = get_dynkin_diagram(simple_root_data(typ, rank))
    matcher = nx.algorithms.isomorphism.GraphMatcher(G, G_canonical, edge_match=lambda x, y: x["weight"] == y["weight"])
    if not matcher.is_isomorphic():
        raise ValueError("Two Dynkin diagrams are not isomorphic.")
    mapping = matcher.mapping
    perm = [0] * len(simple_roots)
    for k, v in mapping.items():
        perm[v] = k
    # print(perm)
    return simple_roots[perm]
    # print(mapping)
    



if __name__ == "__main__":
    np.set_printoptions(
        precision=3,
        suppress=True,
    )
    typ, rank = "E", 7
    # W = pycox.coxeter(typ=typ, rank=rank)
    # v = pycox.lpol([1], 1, "v")
    test_cases = [
        (('D', 8, np.array([2, 1, 1.1, 3, 0.9, 1.9, 4, 2.1])), 
        [('A', 3), ('D', 4)]),
        (('F', 4, np.array([4, 5, 3/2, 1/2])),
        [('C', 4)]),
        (('F', 4, np.array([7/4, 1/4, 5/4, -3/4])),
        [('B', 3), ('A', 1)]),
        (('E', 6, np.array([1, 2, 1, 4, 4.5, 0.5, 0.5, -0.5])),
        [('D', 5)]),
        (('E', 7, np.array([1/4, 1/4, 1/4, 1/4, 1/4, -3/4, -1, 1])),
        [('A', 7)]),
        (('E', 8, np.array([1, 5, 9, 13, 9, 1, 5, 9])/4),
        [('D', 8)]),
        (('E', 8, np.array([1/2, -3/2, -3, -2, -1, -4, -5, -19])),
        [('E', 7), ('A', 1)]),
        (('E', 7, np.array([1, 3, 5, -7, -9, -11, -1/2, 1/2])),
        [('D', 6), ('A', 1)]),
        (('E', 8, np.array([1, 1, 1, 1, 1, 1, 1/2, 5/2])),
        [('E', 7), ('A', 1)])
    ]
    test_case = test_cases[3]
    weight = test_case[0][2]
    typ = test_case[0][0]
    rank = test_case[0][1]
    
    dim_ambient = weight.shape[0]
    rt, _ = integral_root_system(typ, rank, weight)
    cts, sps = get_cartan_type(rt) # [('A', 3), ('D', 4)]

    sps = [reorder_simple_roots(sp, *ct) for ct, sp in zip(cts, sps)]
    print(sps)
    # sps[0] = sps[0][[0,1,2,4,3]]
    print(sps)
    # print(sps)
    sp_basis = np.concatenate(sps)
    cpl_basis = find_complement(np.concatenate(sps), np.eye(dim_ambient))
    all_basis = np.concatenate([sp_basis, cpl_basis])
    sps_new = sps + [cpl_basis]
    
    embedded = np.concatenate(
        [embed_basis(simple_root_data(*ct), dim_ambient) 
         for ct in cts]
        )
    embedded = np.concatenate(
        [embedded, np.zeros((dim_ambient - embedded.shape[0], dim_ambient))])
    isomap = embedded.T @ np.linalg.inv(all_basis).T
    
    transformed_weights = []
    transformed_weights_ = []
    for ct, sp in zip(cts, sps):
        cananical_sp = simple_root_data(*ct)
        dim_sp = cananical_sp.shape[1]
        print(cartan_matrix_(sp), cartan_matrix_(cananical_sp))
        # embeded_canonical_sp = embed_basis(cananical_sp, dim_ambient)
        
        fundamental_weights = compute_fundamental_weights(sp)
        transformed_fundamental_weights = isomap @ fundamental_weights.T
        weight_ = (2 * weight @ sp.T / np.sum(sp**2, axis=1))
        transformed_weight = weight_ @ transformed_fundamental_weights.T
        # print(transformed_weight)
        transformed_weight = restrict_array(transformed_weight, dim_sp)
        transformed_weight_ = (2 * transformed_weight @ cananical_sp.T / np.sum(cananical_sp**2, axis=1))
        
        transformed_weights.append(transformed_weight)
        transformed_weights_.append(transformed_weight_)
        
    print(transformed_weights)
    print(transformed_weights_)
    print(cts)
    from GK_dimension import a_value_integral
    print(a_value_integral(*cts[0], np.round(transformed_weights_[0])))
    
    # test fundamental weights
    
