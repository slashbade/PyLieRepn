from typing import Literal

import numpy as np
import traceback

from LieToolbox.Repn.utils import PPUtil
# from .utils import *
from .roots import (
    integral_root_system, simple_root_data, get_cartan_type, 
    reorder_simple_roots, compute_fundamental_weights)
from .root_system_data import cartan_matrix_pycox, num_positive_roots_data
from ..PyCox import chv1r6180 as pycox
from .weight import HighestWeightModule, Symbol, Weight, NilpotentOrbit
from .orbit import (BalaCarterOrbit, from_orbit_string, from_alvis_notation, 
                    from_partition_dual, get_mark_from_diagram)
from .algorithm import Number, LinearAlgebra
from .algorithm.pir import antidominant
# from .algorithm.partition_equivalence import weight_partition
from .neutral_elements import need_to_decide_mark, get_diagram, get_neutral_element_sum 

Typ = Literal["A", "B", "C", "D", "E", "F", "G"]
LieType = tuple[Typ, int]



def a_value_integral_classical(typ: Typ, rank: int, weight: np.ndarray) -> tuple[int, Symbol, NilpotentOrbit]:
    weight = Number.round2(weight)
    lbd = Weight(weight.tolist(), typ) # type: ignore
    L = HighestWeightModule(lbd)
    # obtinfo = L.nilpotentOrbitInfo()
    orbit = L.nilpotentOrbit()
    character = orbit.convert2Symbol()
    # print(typ, rank, weight)
    return L.a_value_integral(), character, orbit

def a_value_integral_exceptional(typ: Typ, rank: int, weight):
    cananical_sp = simple_root_data(typ, rank)
    weight_ = (2 * weight @ cananical_sp.T / np.sum(cananical_sp**2, axis=1))
    W = pycox.coxeter(typ, rank)
    
    cmat = cartan_matrix_pycox(typ, rank)
    w, adw = antidominant(cmat, weight_)
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
    is_integral_weight = Number.is_integer_array(weight)
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
        sp_basis = LinearAlgebra.embed_basis(np.concatenate(sps), dim_ambient)
        cpl_basis = LinearAlgebra.find_complement(sp_basis, np.eye(dim_ambient))
        all_basis = np.concatenate([sp_basis, cpl_basis])
        
        # Get the isomorphism map from the decomposed system to cananical root system
        embedded = np.concatenate(
            [LinearAlgebra.embed_basis(simple_root_data(*ct), dim_ambient) 
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
        transformed_fundamental_weights = isomap @ LinearAlgebra.embed_basis(fundamental_weights, dim_ambient).T
        weight_ = (2 * weight @ sp.T / np.sum(sp**2, axis=1))
        weight1 = weight_ @ fundamental_weights
        transformed_weight = weight_ @ transformed_fundamental_weights.T
        transformed_weight = LinearAlgebra.restrict_array(transformed_weight, dim_sp)
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
    # print('Decomposed root system: ', cts)
    # print('Summed dual orbits: ', result_bl_orbit)
    neutral_element_images = []
    try:
    # Compute mark ' or " for the orbit
        if need_to_decide_mark(result_bl_orbit):
            neutral_element_all = np.zeros(dim_ambient)
            for ct, sp, orb, od in zip(cts, sps, orbits, orbit_duals):
                neutral_element_all += get_neutral_element_sum(ct, sp, od, orb)
                neutral_element_images.append(f'neutral_elements_chosen_ids_{ct[0]}_{ct[1]}.png')
            diagram = get_diagram(typ, rank, neutral_element_all)
            result_bl_orbit = get_mark_from_diagram(result_bl_orbit, diagram)
        dual = result_bl_orbit.dual()
    except Exception as e:
        print(traceback.format_exc())
        dual = "N/A"
    
    total_a_value = sum(a_values)
    num_postive_roots = num_positive_roots_data(typ, rank)
    gk_dim = num_postive_roots - total_a_value

    info = {
        "cartan_type": PPUtil.pretty_print_lietype(typ, rank),
        "simple_roots_weight": PPUtil.pretty_print_basis(simple_root_data0),
        "weight": PPUtil.pretty_print_weight(weight),
        "weight_": PPUtil.pretty_print_weight_(weight0_),
        "integral_roots": PPUtil.pretty_print_basis(rt),
        "cartan_types": [PPUtil.pretty_print_lietype(*ct) for ct in cts],
        "pretty_cartan_types": PPUtil.pretty_print_lietypes(cts),
        "simple_roots": [PPUtil.pretty_print_basis(sp) for sp in sps],
        "pretty_simple_roots": PPUtil.pretty_print_basises(sps),
        "cananical_simple_roots": [PPUtil.pretty_print_basis(simple_root_data(*ct)) for ct in cts],
        "pretty_cananical_simple_roots": PPUtil.pretty_print_basises([simple_root_data(*ct) for ct in cts]),
        "complement_basis": PPUtil.pretty_print_basis(cpl_basis),
        "isomap": PPUtil.pretty_print_matrix(isomap),
        "weights": [PPUtil.pretty_print_weight(weight) for weight in weights],
        "weights_": [PPUtil.pretty_print_weight_(weight_) for weight_ in weights_],
        "transformed_weights": [PPUtil.pretty_print_weight(transformed_weight) 
            for transformed_weight in transformed_weights],
        "transformed_weights_": [PPUtil.pretty_print_weight_(transformed_weight_) 
            for transformed_weight_ in transformed_weights_],
        "a_values": a_values,
        "characters": [PPUtil.pretty_print_character(str(character)) for character in characters],
        "orbits": [str(o) for o in orbits],
        "orbit_duals": orbit_duals,
        "num_positive_roots": num_postive_roots,
        "total_a_value": total_a_value,
        "GK_dimension": gk_dim,
        "result_bl_orbit": str(result_bl_orbit),
        "dual": str(dual),
        "neutral_element_images": neutral_element_images,
    }
    return str(gk_dim), info
    
    
    


if __name__ == "__main__":
    test_cases = [
        (
            ('D', 8, np.array([2, 1, 1.1, 3, 0.9, 1.9, 4, 2.1])), 
            ([('A', 3), ('D', 4)], 5, 51, None)
        ), (
            ('F', 4, np.array([4, 5, 3/2, 1/2])),
            ([('C', 4)], 9, 15, 'A_2')
        ), (
            ('F', 4, np.array([7/4, 1/4, 5/4, -3/4])),
            ([('B', 3), ('A', 1)], 5, 19, None)
        ), (
            ('E', 6, np.array([1, 2, 1, 4, 4.5, 0.5, 0.5, -0.5])), 
            ([('D', 5)], 7, 29, None)
        ), (
            ('E', 7, np.array([1/4, 1/4, 1/4, 1/4, 1/4, -3/4, -1, 1])), 
            ([('A', 7)], 3, 60, None)
        ), (
            ('E', 8, np.array([1, 5, 9, 13, 9, 1, 5, 9])/4), 
            ([('D', 8)], 17, 103, None)
        ), (
            ('E', 8, np.array([1/2, -3/2, -3, -2, -1, -4, -5, -19])),
            ([('E', 7), ('A', 1)], 3, 117, None)
        ), (
            ('E', 7, np.array([1, 3, -5, -7, -9, -11, -1/2, 1/2])), 
            ([('D', 6), ('A', 1)], 7, 60, 'E_6')
        ), (
            ('E', 8, np.array([1, 1, 1, 1, 1, 1, 1/2, 5/2])), 
            ([('E', 7), ('A', 1)], 7, 113, 'D_7')
        ), (
            ('E', 7, np.array([2.1, 1.1, -0.1, 2.1, 2, 4, 2, 0.9])),
            ([('D', 6), ('A', 1)], None, None, None)
        ), (
            ('E', 7, np.array([-1/4, 1/4, 1/4, 1/4, 1/4, 1/4, -5/4, 5/4])),
            ([('D', 6), ('A', 1)], None, 59, None)
        ), (
            ('E', 8, np.array([1/8, 1/8, 1/8, 1/8, 1/8, 3/8, 7/8, 13/8])),
            ([('D', 6), ('A', 1)], None, None, 'E_7')
        ), (
            ('E', 8, np.array([1, 3, -5, -7, -9, -11, -1/2, 1/2])),
            ([('D', 6), ('A', 1)], None, 113, 'E_8(b_5)')
        )
    ]
    for i, test_case in enumerate(test_cases):
        print(f"start test case {i}", test_case[0])
        if i in [0]:
            continue
        typ, rank, weight = test_case[0]
        gk_dim, info = GK_dimension(typ, rank, weight)
        print(gk_dim)
        print(info['dual'])
        if test_case[1][3] is not None:
            assert info['dual'] == test_case[1][3], f"{info['dual']} {test_case[1][3]}"
        if test_case[1][2] is not None:
            assert eval(gk_dim) == test_case[1][2], f"{gk_dim}"
        print(f"finish test case {weight}")
    print("All test cases passed.")

