import sys
sys.path.append("..")
sys.path.append(".")
import numpy as np
from numpy.typing import NDArray
from LieToolbox.Repn.utils import *
from LieToolbox.Repn.roots import integral_root_system, simple_root_data, get_cartan_type, reorder_simple_roots, compute_fundamental_weights
from LieToolbox.Repn.root_system_data import cartan_matrix_pycox, num_positive_roots_data
from LieToolbox.Repn.PyCox import chv1r6180 as pycox
from LieToolbox.Repn.weight import HighestWeightModule, Weight

def antidominant(typ: str, rank: int, weight_: NDArray, weyl: list = []) -> NDArray:
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
    for i in range(weight_.shape[0]):
        if is_zero(weight_[i]):
            weight_[i] = 0
    if np.all(weight_ <= 0):
        return weyl, weight_
    else:
        max_pos_index = np.argwhere(weight_ > 0)[-1][0]
        new_weyl = weyl + [max_pos_index]
        new_weight = act_on_weight(typ, rank, max_pos_index, weight_)
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
    cmat = cartan_matrix_pycox(typ, rank)
    return weight - weight[root_index] * cmat[root_index]


def weight_partition(typ: str, rank: int, weight: NDArray):
    congruence = lambda a, b: is_integer(a - b) or is_integer(a + b)
    weights, _ = partition_equivalence(weight, congruence)
    return weights


def a_value_integral(typ, rank, weight):
    if typ in ["A", "B", "C", "D"]:
        lbd = Weight(weight.tolist(), typ)
        L = HighestWeightModule(lbd)
        obtinfo = L.nilpotentOrbitInfo()
        if typ == "D" and obtinfo['isVeryEven']:
            obt = f"{obtinfo['Orbit']}, {obtinfo['veryEvenType']}"
        else:
            obt = f"{obtinfo['Orbit']}"
        return L.a_value_integral(), obt
    else:
        cananical_sp = simple_root_data(typ, rank)
        weight_ = (2 * weight @ cananical_sp.T / np.sum(cananical_sp**2, axis=1))
        W = pycox.coxeter(typ, rank)
        
        w, adw = antidominant(typ, rank, weight_)
        cell_repm = pycox.klcellrepelm(W, w)
        # print('input weight fundam:', np.round(weight_,2))
        # print('weyl:', w)
        # print('antidom weight', np.round(adw))
        # print('repm:', cell_repm)
        character = cell_repm['character']
        return cell_repm['a'], f"{character[0][0]}, {character[0][1]}"


def GK_dimension(typ, rank, weight: NDArray) -> int:
    """Compute the dimension of the Gelfand-Kirillov dimension of a weight.

    Args:
        typ (str): type of the Lie algebra.
        rank (int): rank of the Lie algebra.
        weight (NDArray): weight to compute the Gelfand-Kirillov dimension,
        represented in the orthonormal basis.

    Returns:
        int: the Gelfand-Kirillov dimension.
    """
    dim_ambient = weight.shape[0]
    
    # Integral root system decomposition
    rt, _ = integral_root_system(typ, rank, weight)
    cts, sps = get_cartan_type(rt)
    
    # Reorder simple roots to match the cananical order
    sps = [reorder_simple_roots(sp, *ct) for ct, sp in zip(cts, sps)]
    sp_basis = np.concatenate(sps)
    cpl_basis = find_complement(np.concatenate(sps), np.eye(dim_ambient))
    all_basis = np.concatenate([sp_basis, cpl_basis])
    
    # Get the isomorphism map from the decomposed system to cananical root system
    embedded = np.concatenate(
        [embed_basis(simple_root_data(*ct), dim_ambient) 
         for ct in cts]
        )
    # isomap1 = embedded.T @ np.linalg.inv(all_basis).T
    embedded = np.concatenate(
        [embedded, np.zeros((dim_ambient - embedded.shape[0], dim_ambient))])
    isomap = embedded.T @ np.linalg.inv(all_basis).T
    
    weights_ = []
    transformed_weights = []
    transformed_weights_ = []
    a_values = []
    characters = []
    for ct, sp in zip(cts, sps):
        # Compute the weight in each cananical simple root basis
        cananical_sp = simple_root_data(*ct)
        dim_sp = cananical_sp.shape[1]
        fundamental_weights = compute_fundamental_weights(sp)
        transformed_fundamental_weights = isomap @ fundamental_weights.T
        weight_ = (2 * weight @ sp.T / np.sum(sp**2, axis=1))
        transformed_weight = weight_ @ transformed_fundamental_weights.T
        transformed_weight = restrict_array(transformed_weight, dim_sp)
        # print(f"transformed weight: {transformed_weight}")
        transformed_weight_ = (2 * transformed_weight @ cananical_sp.T / np.sum(cananical_sp**2, axis=1))
        # Compute the Gelfand-Kirillov dimension
        transformed_weight_ = np.round(transformed_weight_)
        a_value, character = a_value_integral(*ct, transformed_weight)
        # print(weight, weight_, sp)
        weights_.append(weight_)
        transformed_weights.append(transformed_weight)
        transformed_weights_.append(transformed_weight_)
        a_values.append(a_value)
        characters.append(character)
    
    total_a_value = sum(a_values)
    num_postive_roots = num_positive_roots_data(typ, rank)
    gk_dim = num_postive_roots - total_a_value
    
    info = {
        "cartan_type": pretty_print_lietype(typ, rank),
        "simple_roots_weight": pretty_print_basis(simple_root_data(typ, rank)),
        "weight": pretty_print_weight(weight),
        "integral_roots": pretty_print_basis(rt),
        "cartan_types": [pretty_print_lietype(*ct) for ct in cts],
        "pretty_cartan_types": pretty_print_lietypes(cts),
        "simple_roots": [pretty_print_basis(sp) for sp in sps],
        "pretty_simple_roots": pretty_print_basises(sps),
        "cananical_simple_roots": [pretty_print_basis(simple_root_data(*ct)) for ct in cts],
        "pretty_cananical_simple_roots": pretty_print_basises([simple_root_data(*ct) for ct in cts]),
        "complement_basis": pretty_print_basis(cpl_basis),
        "isomap": pretty_print_matrix(isomap),
        "weights_": [pretty_print_weight_(weight_) for weight_ in weights_],
        "transformed_weights": [pretty_print_weight(transformed_weight) for transformed_weight in transformed_weights],
        "transformed_weights_": [pretty_print_weight_(transformed_weight_) for transformed_weight_ in transformed_weights_],
        "a_values": a_values,
        "characters": characters,
        "num_positive_roots": num_postive_roots,
        "total_a_value": total_a_value,
        "GK_dimension": gk_dim
    }
    return gk_dim, info
    
    
    


if __name__ == "__main__":
    # typ, rank, weight_ = "B", 3, np.array([2, -1, 2])
    # typ, rank, weight_ = "E", 7, np.array([-1, 1, -3, 1, -5, -1, -2])
    # typ, rank, weight_ = "A", 1, np.array([-1])
    # W = pycox.coxeter(typ, rank)
    # w, adw = antidominant(typ, rank, weight_)
    # print(pycox.klcellrepelm(W, w)['a'])
    test_cases = [
        (
            ('D', 8, np.array([2, 1, 1.1, 3, 0.9, 1.9, 4, 2.1])), 
            ([('A', 3), ('D', 4)], 5, 51)
        ), (
            ('F', 4, np.array([4, 5, 3/2, 1/2])),
            ([('C', 4)], 9, 15)
        ), (
            ('F', 4, np.array([7/4, 1/4, 5/4, -3/4])),
            ([('B', 3), ('A', 1)], 5, 19)
        ), (
            ('E', 6, np.array([1, 2, 1, 4, 4.5, 0.5, 0.5, -0.5])), 
            ([('D', 5)], 7, 29)
        ), (
            ('E', 7, np.array([1/4, 1/4, 1/4, 1/4, 1/4, -3/4, -1, 1])), 
            ([('A', 7)], 3, 60)
        ), (
            ('E', 8, np.array([1, 5, 9, 13, 9, 1, 5, 9])/4), 
            ([('D', 8)], 17, 103)
        ), (
            ('E', 8, np.array([1/2, -3/2, -3, -2, -1, -4, -5, -19])),
            ([('E', 7), ('A', 1)], 3, 117)
        ), (
            ('E', 7, np.array([1, 3, 5, -7, -9, -11, -1/2, 1/2])), 
            ([('D', 6), ('A', 1)], None, None)
        ), (
            ('E', 8, np.array([1, 1, 1, 1, 1, 1, 1/2, 5/2])), 
            ([('E', 7), ('A', 1)], 7, 113)
        ), (
            ('E', 7, np.array([2.1, 1.1, -0.1, 2.1, 2, 4, 2, 0.9])),
            ([('D', 6), ('A', 1)], None, None)
        )
    ]
    for test_case in test_cases:
        typ, rank, weight = test_case[0]
        gk_dim, info = GK_dimension(typ, rank, weight)
        if test_case[1][2] is not None:
            assert gk_dim == test_case[1][2]
        print(f"finish test case {weight}")
    print("All test cases passed.")
    test_case = test_cases[8]
    # print(test_case)
    typ, rank, weight = test_case[0]
    # print(GK_dimension(typ, rank, weight))