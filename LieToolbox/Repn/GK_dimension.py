import numpy as np
import traceback
from .utils import *
from .roots import (
    integral_root_system, simple_root_data, get_cartan_type, 
    reorder_simple_roots, compute_fundamental_weights)
from .root_system_data import cartan_matrix_pycox, num_positive_roots_data
from ..PyCox import chv1r6180 as pycox
from .weight import HighestWeightModule, Weight, NilpotentOrbit
from .orbit import (BalaCarterOrbit, from_orbit_string, from_alvis_notation, 
                    from_partition_dual, get_mark_from_diagram)

from .neutral_elements import get_chosen_neutral_elements

def antidominant(typ: str, rank: int, weight_: np.ndarray, weyl: list = []) -> tuple[list, np.ndarray]:
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


def act_on_weight(typ: str, rank: int, root_index: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Compute the result of the action of the simple root indexed by root_index on the weight.

    Args:
        typ (str): type of the Lie algebra.
        rank (int): rank of the Lie algebra.
        root_index (np.ndarray): index of the simple root.
        weight (np.ndarray): weight to act on, represented in the fundamental weight basis.

    Returns:
        np.ndarray: the new weight, represented in the fundamental weight basis.
    """
    cmat = cartan_matrix_pycox(typ, rank)
    return weight - weight[root_index] * cmat[root_index]


def weight_partition(typ: str, rank: int, weight: np.ndarray):
    congruence = lambda a, b: is_integer(a - b) or is_integer(a + b)
    weights, _ = partition_equivalence(weight, congruence)
    return weights


def a_value_integral_classical(typ, rank, weight):
    weight = round2(weight)
    lbd = Weight(weight.tolist(), typ)
    L = HighestWeightModule(lbd)
    obtinfo = L.nilpotentOrbitInfo()
    orbit = L.nilpotentOrbit()
    character = orbit.convert2Symbol()
    # print(typ, rank, weight)
    return L.a_value_integral(), character, orbit

def a_value_integral_exceptional(typ, rank, weight):
    cananical_sp = simple_root_data(typ, rank)
    weight_ = (2 * weight @ cananical_sp.T / np.sum(cananical_sp**2, axis=1))
    W = pycox.coxeter(typ, rank)
    
    w, adw = antidominant(typ, rank, weight_)
    cell_repm = pycox.klcellrepelm(W, w)
    # print('input weight fundam:', np.round(weight_,2))
    # print('weyl:', w)
    # print('antidom weight', np.round(adw))
    # print('repm:', cell_repm)
    # character = cell_repm['character']
    # character = cell_repm['special']
    # orbit = find_orbit_from_character(typ, rank, cell_repm['special'])
    bl_orbit = from_alvis_notation(cell_repm['special'], (typ, rank))
    return cell_repm['a'], cell_repm['special'], bl_orbit

def get_neutral_element_sum(simple_roots: np.ndarray, bl_orbit: BalaCarterOrbit, orbit: NilpotentOrbit | BalaCarterOrbit) -> np.ndarray:
    """ Get neutral element (Only implemented for summation of type A)
    """
    ranks = []
    ranks_D = []
    for (typ, rank, _), mult in bl_orbit.orbits.items():
        if typ == 'D':
            ranks_D.extend([rank] * mult)
        elif typ == 'A':
            ranks.extend([rank] * mult)
        else:
            raise ValueError(f"Unknown type {typ} in the orbit.")
    if len(ranks_D) > 1:
        raise ValueError(f"Only one rank D is allowed in the orbit.")
    to_cover_id, drop_id = None, None
    if isinstance(orbit, NilpotentOrbit) and orbit.lieType == 'D':
        if ranks_D and ranks_D[0] == 2:
            pass

        if orbit.veryEvenType == 'I':
            to_cover_id, drop_id = 1, 0
        elif orbit.veryEvenType == 'II':
            to_cover_id, drop_id = 0, 1
    chosen_ids = get_chosen_neutral_elements(simple_roots, ranks, to_cover_id, drop_id)
    if not chosen_ids:
        return np.zeros(simple_roots.shape[1])
    chosen_roots = simple_roots[chosen_ids]
    # print("chosen ids: ", chosen_ids)
    # print("chosen roots: ", chosen_roots)
    sum_of_chosen_roots = np.sum(chosen_roots, axis=0)
    # print("sum of chosen roots: ", sum_of_chosen_roots)
    return sum_of_chosen_roots

def get_diagram(typ, rank, neutral: np.ndarray) -> list[int]:
    print(neutral)
    weight_ = neutral @ simple_root_data(typ, rank).T
    antidom, w = antidominant(typ, rank, weight_)
    w = (-1) * w
    sps = simple_root_data(typ, rank)
    # print(sps @ w)
    print('diagram:', round2(w).tolist())
    return round2(w).tolist() # type: ignore

def need_to_decide_mark(orbit: BalaCarterOrbit) -> bool:
    return all([typ=='A' or (typ, rank) == ('D', 2) or (typ, rank) == ('D', 3)
        for (typ, rank, _) in orbit.orbits.keys()])

def GK_dimension(typ, rank, weight: np.ndarray) -> tuple[str, dict]:
    """Compute the dimension of the Gelfand-Kirillov dimension of a weight.

    Args:
        typ (str): type of the Lie algebra.
        rank (int): rank of the Lie algebra.
        weight (np.ndarray): weight to compute the Gelfand-Kirillov dimension,
        represented in the orthonormal basis.

    Returns:
        int: the Gelfand-Kirillov dimension.
    """
    # print(weight)
    is_integral_weight = is_integer_array(weight)
    simple_root_data0 = simple_root_data(typ, rank)
    weight0_ = (2 * weight @ simple_root_data0.T / np.sum(simple_root_data0**2, axis=1))
    dim_ambient = weight.shape[0]
    # print(typ, rank, weight)
    # Integral root system decomposition
    rt, _ = integral_root_system(typ, rank, weight)
    # print(rt)
    cts, sps = get_cartan_type(rt)
    
    # Find a sufficiently large ambiant space
    dim_spaces = [simple_root_data0.shape[1]]
    for ct in cts:
        dim_spaces.append(simple_root_data(*ct).shape[1])
    dim_ambient = max(dim_spaces)

    # Reorder simple roots to match the cananical order
    sps = [reorder_simple_roots(sp, *ct) for ct, sp in zip(cts, sps)]
    if sps:
        sp_basis = embed_basis(np.concatenate(sps), dim_ambient)
        cpl_basis = find_complement(sp_basis, np.eye(dim_ambient))
        all_basis = np.concatenate([sp_basis, cpl_basis])
        
        # Get the isomorphism map from the decomposed system to cananical root system
        embedded = np.concatenate(
            [embed_basis(simple_root_data(*ct), dim_ambient) 
            for ct in cts]
            )
        embedded = np.concatenate(
            [embedded, np.zeros((dim_ambient - embedded.shape[0], dim_ambient))])
    else:
        embedded = np.eye(dim_ambient)
        cpl_basis = np.zeros((dim_ambient, dim_ambient))
        all_basis = np.eye(dim_ambient)
    isomap = embedded.T @ np.linalg.inv(all_basis).T
    
    weights = []
    weights_ = []
    transformed_weights = []
    transformed_weights_ = []
    a_values = []
    characters = []
    orbits = []
    orbit_duals = []
    result_bl_orbit = BalaCarterOrbit()
    result_bl_orbit.lie_type = (typ, rank)
    for ct, sp in zip(cts, sps):
        # Compute the weight in each cananical simple root basis
        cananical_sp = simple_root_data(*ct)
        dim_sp = cananical_sp.shape[1]
        fundamental_weights = compute_fundamental_weights(sp)
        transformed_fundamental_weights = isomap @ embed_basis(fundamental_weights, dim_ambient).T
        weight_ = (2 * weight @ sp.T / np.sum(sp**2, axis=1))
        weight1 = weight_ @ fundamental_weights
        transformed_weight = weight_ @ transformed_fundamental_weights.T
        transformed_weight = restrict_array(transformed_weight, dim_sp)
        transformed_weight_ = (2 * transformed_weight @ cananical_sp.T / np.sum(cananical_sp**2, axis=1))
        transformed_weight_ = np.round(transformed_weight_)
        
        if ct[0] in ['A', 'B', 'C', 'D']:
            a_value, character, orbit = a_value_integral_classical(*ct, transformed_weight)
        else:
            a_value, character, orbit = a_value_integral_exceptional(*ct, transformed_weight)

        weights.append(weight1)
        weights_.append(weight_)
        transformed_weights.append(transformed_weight)
        transformed_weights_.append(transformed_weight_)
        a_values.append(a_value)
        characters.append(character)
        orbits.append(orbit)
        
        if is_integral_weight:
            orbit_duals.append(orbit)
        else:
            if isinstance(orbit, NilpotentOrbit):
                dual_orbit = orbit.dual()
                try:
                    bl_orbit_dual = from_partition_dual(dual_orbit.lieType, dual_orbit.entry)
                except ValueError as e:
                    print("err:", e)
                    bl_orbit_dual = from_orbit_string('0')
            elif isinstance(orbit, BalaCarterOrbit):
                try:
                    bl_orbit_dual = orbit.ls_dual()
                except ValueError as e:
                    print("err:", e)
                    bl_orbit_dual = from_orbit_string('0')
            else:
                raise TypeError(f"Unknown orbit type {type(orbit)}.")
            orbit_duals.append(bl_orbit_dual)
            result_bl_orbit = result_bl_orbit + bl_orbit_dual
            print("orbit", orbit)
    print('Decomposed root system: ', cts)
    print('Summed dual orbits: ', result_bl_orbit)
    try:
    # Compute mark ' or " for the orbit
        if need_to_decide_mark(result_bl_orbit):
            neutral_element_all = np.zeros(dim_ambient)
            for _, sp, orb, od in zip(cts, sps, orbits, orbit_duals):
                # print(neutral_element_all, get_neutral_element(typ, rank, sp, od))
                
                neutral_element_all += get_neutral_element_sum(sp, od, orb)
            diagram = get_diagram(typ, rank, neutral_element_all)
            result_bl_orbit = get_mark_from_diagram(result_bl_orbit, diagram) 
        # dual = result_bl_orbit.sommers_dual()
        dual = result_bl_orbit.dual()
    except Exception as e:
        print(traceback.format_exc())
        dual = "N/A"
    
    total_a_value = sum(a_values)
    num_postive_roots = num_positive_roots_data(typ, rank)
    gk_dim = num_postive_roots - total_a_value

    info = {
        "cartan_type": pretty_print_lietype(typ, rank),
        "simple_roots_weight": pretty_print_basis(simple_root_data0),
        "weight": pretty_print_weight(weight),
        "weight_": pretty_print_weight_(weight0_),
        "integral_roots": pretty_print_basis(rt),
        "cartan_types": [pretty_print_lietype(*ct) for ct in cts],
        "pretty_cartan_types": pretty_print_lietypes(cts),
        "simple_roots": [pretty_print_basis(sp) for sp in sps],
        "pretty_simple_roots": pretty_print_basises(sps),
        "cananical_simple_roots": [pretty_print_basis(simple_root_data(*ct)) for ct in cts],
        "pretty_cananical_simple_roots": pretty_print_basises([simple_root_data(*ct) for ct in cts]),
        "complement_basis": pretty_print_basis(cpl_basis),
        "isomap": pretty_print_matrix(isomap),
        "weights": [pretty_print_weight(weight) for weight in weights],
        "weights_": [pretty_print_weight_(weight_) for weight_ in weights_],
        "transformed_weights": [pretty_print_weight(transformed_weight) 
            for transformed_weight in transformed_weights],
        "transformed_weights_": [pretty_print_weight_(transformed_weight_) 
            for transformed_weight_ in transformed_weights_],
        "a_values": a_values,
        "characters": [pretty_print_character(str(character)) for character in characters],
        "orbits": [str(o) for o in orbits],
        "orbit_duals": orbit_duals,
        "num_positive_roots": num_postive_roots,
        "total_a_value": total_a_value,
        "GK_dimension": gk_dim,
        "dual": dual
    }
    return str(gk_dim), info
    
    
    


if __name__ == "__main__":
    # typ, rank, weight_ = "B", 3, np.array([2, -1, 2])
    # typ, rank, weight_ = "E", 7, np.array([-1, 1, -3, 1, -5, -1, -2])
    # typ, rank, weight_ = "A", 1, np.array([-1])
    # W = pycox.coxeter(typ, rank)
    # w, adw = antidominant(typ, rank, weight_)
    # print(pycox.klcellrepelm(W, w)['a'])
    # GK_dimension("E", 8, np.array([1, 1, 1, 1, 1, 1, 1/2, 5/2]))
    GK_dimension("E", 8, np.array([1, 1, 1, 1, 1, 3, 7, 13])/8)
    # GK_dimension("F", 4, np.array([4, 5, 3/2, 1/2]))
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
            ('E', 7, np.array([1, 3, -5, -7, -9, -11, -1/2, 1/2])), 
            ([('D', 6), ('A', 1)], 7, 60)
        ), (
            ('E', 8, np.array([1, 1, 1, 1, 1, 1, 1/2, 5/2])), 
            ([('E', 7), ('A', 1)], 7, 113)
        ), (
            ('E', 7, np.array([2.1, 1.1, -0.1, 2.1, 2, 4, 2, 0.9])),
            ([('D', 6), ('A', 1)], None, None)
        ), (
            ('E', 7, np.array([-1/4, 1/4, 1/4, 1/4, 1/4, 1/4, -5/4, 5/4])),
            ([('D', 6), ('A', 1)], None, 59)
        ), (
            ('E', 8, np.array([1/8, 1/8, 1/8, 1/8, 1/8, 3/8, 7/8, 13/8])),
            ([('D', 6), ('A', 1)], None, None)
        )
    ]
    for i, test_case in enumerate(test_cases):
        print(f"start test case {i}", test_case[0])
        if i in [0]:
            continue
        typ, rank, weight = test_case[0]
        gk_dim, info = GK_dimension(typ, rank, weight)
        print(gk_dim)
        if test_case[1][2] is not None:
            assert eval(gk_dim) == test_case[1][2], f"{gk_dim}"
        print(f"finish test case {weight}")
    print("All test cases passed.")

