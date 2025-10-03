import numpy as np
import traceback

from LieToolbox.Repn.utils import PPUtil
from .roots import (
    integral_root_system, simple_root_data, get_cartan_type, 
    reorder_simple_roots, compute_fundamental_weights)
from .root_system_data import cartan_matrix_pycox, num_positive_roots_data
from ..PyCox import chv1r6180 as pycox
from .weight import HighestWeightModule, Weight, NilpotentOrbit
from .orbit import (BalaCarterOrbit, from_orbit_string, from_alvis_notation, 
                    from_partition_dual, get_mark_from_diagram)
from .algorithm import Number, LinearAlgebra
from .algorithm.pir import antidominant
from .neutral_elements import need_to_decide_mark, get_diagram, get_neutral_element_sum 
from .structs import LieType, Typ

BiPartition = tuple[list, list]
Character = BiPartition | str

def a_value_integral_classical(typ: Typ, rank: int, weight: np.ndarray) -> tuple[int, BiPartition, NilpotentOrbit]:
    weight = Number.round_half(weight)
    lbd = Weight(weight.tolist(), typ) # type: ignore
    L = HighestWeightModule(lbd)
    orbit = L.nilpotentOrbit()
    character = orbit.convert_to_bi_partition()
    return L.a_value_integral(), character, orbit

def a_value_integral_exceptional(typ: Typ, rank: int, weight) -> tuple[int, str, BalaCarterOrbit]:
    cananical_sp = simple_root_data(typ, rank)
    weight_ = (2 * weight @ cananical_sp.T / np.sum(cananical_sp**2, axis=1))
    W = pycox.coxeter(typ, rank)
    
    cmat = cartan_matrix_pycox(typ, rank)
    w, _ = antidominant(cmat, weight_)
    cell_repm = pycox.klcellrepelm(W, w)
    bl_orbit = from_alvis_notation(cell_repm['special'], (typ, rank))
    return cell_repm['a'], cell_repm['special'], bl_orbit

def GK_dimension(typ: Typ, rank: int, weight: np.ndarray) -> tuple[str, dict]:
    """Compute the dimension of the Gelfand-Kirillov dimension of a weight.

    Args:
        typ (str): type of the Lie algebra.
        rank (int): rank of the Lie algebra.
        weight (np.ndarray): weight to compute the Gelfand-Kirillov dimension,
        represented in the orthonormal basis.

    Returns:
        int: the Gelfand-Kirillov dimension.
    """
    is_integral_weight = Number.is_integer_array(weight)
    simple_root_data0 = simple_root_data(typ, rank)
    weight0_ = (2 * weight @ simple_root_data0.T / np.sum(simple_root_data0**2, axis=1))
    dim_ambient = weight.shape[0]
    
    # Integral root system decomposition
    rt, _ = integral_root_system(typ, rank, weight)
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
            orbit_duals.append(orbit.dual())
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
    
    if is_integral_weight:
        neutral_element_images = None
        result_bl_orbit = orbit_duals[0]
        dual = orbits[0]
    else:
        neutral_element_images = []
        try:
            # Compute mark ' or " for the orbit
            if need_to_decide_mark(result_bl_orbit):
                neutral_element_all = np.zeros(dim_ambient)
                for ct, sp, orb, od in zip(cts, sps, orbits, orbit_duals):
                    sp = LinearAlgebra.embed_basis(sp, dim_ambient)
                    if od.lie_type[0] not in ['A', 'B', 'C', 'D']:
                        k = 0
                        while True:
                            _neutral_element, img = get_neutral_element_sum(ct, sp, od, orb, k)
                            _diagram = get_diagram(ct, _neutral_element, sp)
                            _marked = get_mark_from_diagram(od, _diagram)
                            if _marked.mark == od.mark:
                                break
                            k += 1
                    else:
                        _neutral_element, img = get_neutral_element_sum(ct, sp, od, orb, 0)
                    neutral_element_all += _neutral_element
                    neutral_element_images.append(img)
                
                diagram = get_diagram((typ, rank), neutral_element_all, LinearAlgebra.embed_basis(simple_root_data0, dim_ambient))
                result_bl_orbit = get_mark_from_diagram(result_bl_orbit, diagram)
            dual = result_bl_orbit.dual()
        except Exception as e:
            print(traceback.format_exc())
            dual = "N/A"
    
    total_a_value = sum(a_values)
    num_postive_roots = num_positive_roots_data(typ, rank)
    gk_dim = num_postive_roots - total_a_value
    
    info = {
        "cartan_type": PPUtil.pretty_print_lietype((typ, rank)),
        "simple_roots_weight": PPUtil.pretty_print_basis(simple_root_data0),
        "weight": PPUtil.pretty_print_weight(weight),
        "weight_": PPUtil.pretty_print_weight_(weight0_),
        "integral_roots": PPUtil.pretty_print_basis(rt),
        "cartan_types": [PPUtil.pretty_print_lietype(ct) for ct in cts],
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

