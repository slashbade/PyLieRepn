from collections import deque
from functools import reduce
from typing import Optional
import typing

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from .root_system_data import simple_root_data, cartan_matrix_pycox
from .roots import get_dynkin_diagram, cartan_matrix_
from .orbit import BalaCarterOrbit
from .weight import NilpotentOrbit
from .algorithm import Number
from .algorithm.pir import antidominant
from .structs import LieType
from .utils import PPUtil

def get_feasible_placements(diagram: nx.Graph, rank: int) -> list[list[int]]:
    assert rank > 0
    placements = []
    # print("current nodes left", diagram.nodes)
    for start_node in diagram.nodes:
        queue = deque([(start_node, [start_node], 1)])
        while queue:
            node, path, length = queue.popleft()
            if length == rank:
                placements.append(path)
                continue
            for neighbor in diagram.neighbors(node):
                if neighbor not in path:
                    queue.append((neighbor, path + [neighbor], length + 1))
    return placements


def get_first_feasible_placements_at(diagram: nx.Graph, rank: int, start_node: int) -> list[int]:
    assert rank > 0
    placements = []
    # print("current nodes left", diagram.nodes)
    queue = deque([(start_node, [start_node], 1)])
    # rank -= 1
    while queue:
        node, path, length = queue.popleft()
        if length == rank:
            return path
        for neighbor in diagram.neighbors(node):
            if neighbor not in path:
                print("neighbor", neighbor, "path", path, "length", length)
                queue.append((neighbor, path + [neighbor], length + 1))
    return placements

def place_ranks(diagram: nx.Graph, ranks: list[int]) -> list[list[tuple[int, list[int]]]]:
    """
    This try to place A-x or D-x orbit in the diagram, and return
    the all feasible choices.
    """
    all_chosens: list[list[tuple[int, list[int]]]] = []
    queue = deque([(diagram, ranks, [])])
    while queue:
        diagram, ranks, chosen = queue.popleft()
        if ranks == []:
            all_chosens.append(chosen)
            continue
            # return chosen
        placements = get_feasible_placements(diagram, ranks[0])
        if not placements:
            continue
        for placement in placements:
            adjoints = reduce(set.union, map(lambda x : set(diagram.neighbors(x)), placement))
            covered = set(placement) | adjoints
            # print("current_chosen", chosen)
            # print("current_nodes", diagram.nodes)
            # print("placement", placement)
            # print("adjoints", adjoints)
            new_diagram = diagram.copy()
            new_diagram.remove_nodes_from(covered)
            # print("left nodes", new_diagram.nodes, "\n\n")
            queue.append((new_diagram, ranks[1:], chosen + [(ranks[0],placement)]))
    # print(queue)
    if all_chosens:
        return all_chosens
    else:
        raise ValueError('No feasible placements')

def place_ranks_for_type_D_very_even(
    diagram: nx.Graph, 
    ranks: list[int], 
    chosen_id: int, 
    drop_id: int, 
    k: int
) -> list[tuple[int, list[int]]]:
    """
    Place the ranks for type D of very even type.
    """
    # chosen = (1, [chosen_id])
    # find a rank 1 and pop it
    ranks = sorted(ranks)
    # print("ranks after sort", ranks, chosen_id, drop_id)
    placement = get_first_feasible_placements_at(diagram, ranks[0], chosen_id)
    # print("placement found", placement)
    prechosen = (ranks[0], placement)
    # placement += chosen[1] # get chosen id
    adjoints = reduce(set.union, map(lambda x : set(diagram.neighbors(x)), placement))
    covered = set(placement) | adjoints | {drop_id}
    new_diagram = diagram.copy()
    new_diagram.remove_nodes_from(covered)
    ranks = ranks[1:]
    left_chosen = place_ranks(new_diagram, ranks)[k]
    return [prechosen] + left_chosen

def place_ranks_for_type_D_D2_D3(
    diagram: nx.Graph, 
    ranks: list[int], 
    ranks_D: list[int], 
    k: int
) -> list[tuple[int, list[int]]]:
    """
    Place the ranks for type D2 and D3.
    """
    if not len(ranks_D) == 1:
        raise ValueError('len D ranks not 1, No feasible placements')
    rank_D = ranks_D[0]
    new_diagram = diagram.copy()
    if rank_D == 2:
        pre_chosens = [(1, [0]), (1, [1])]
    else: # rank_D == 3
        pre_chosens = [(3, [0, 2, 1])]
    # pre_chosen_ids = pre_chosen[1]
    new_diagram.remove_nodes_from(list(range(rank_D + 1)))
    chosen = place_ranks(new_diagram, ranks)[k]
    chosen = pre_chosens + chosen
    return chosen

def flatten_grouped_chosen(chosen: list[tuple[int, list[int]]]) -> list[int]:
    flatten_chosen_ids = []
    for _, ids in chosen:
        flatten_chosen_ids.extend(ids)
    return flatten_chosen_ids

def get_chosen_neutral_elements(
    simple_roots, 
    ranks: list[int], 
    ranks_D: list[int], 
    chosen_id: typing.Optional[int], 
    drop_id: typing.Optional[int], 
    ct: LieType, 
    k: int
) -> tuple[list[tuple[int, list[int]]], str]:
    # print(f"Getting neutral elements for {ct} with ranks {ranks} and ranks_D {ranks_D}, chosen_id {chosen_id}, drop_id {drop_id}, k {k}")
    diagram = get_dynkin_diagram(simple_roots)
    if not ranks_D == []:
        chosen = place_ranks_for_type_D_D2_D3(diagram, ranks, ranks_D, k)
    elif chosen_id is not None and drop_id is not None:
        chosen = place_ranks_for_type_D_very_even(diagram, ranks, chosen_id, drop_id, k)
    else:
        chosen = place_ranks(diagram, ranks)[k]
    flattened_chosen_ids = flatten_grouped_chosen(chosen)
    filename = visualize_chosen_ids(ct[0], ct[1], diagram, flattened_chosen_ids, save=True)
    assert filename is not None
    return chosen, filename

def get_neutral_element_sum_for_rank(
    rank: int, 
    chosen_ids: list[int], 
    simple_roots: np.ndarray
) -> np.ndarray:
    """
    to compute `h` for one A_k rank
    """
    assert rank == len(chosen_ids)
    chosen_roots = simple_roots[chosen_ids]
    k = rank
    s = np.zeros(simple_roots.shape[1])
    for j in range(k):
        s += sum(k - 2 * i for i in range(j+1)) * chosen_roots[j]
    return s



def get_neutral_element_sum(
    ct: LieType, 
    simple_roots: np.ndarray, 
    bl_orbit: BalaCarterOrbit, 
    orbit: NilpotentOrbit | BalaCarterOrbit, 
    k: int
) -> tuple[np.ndarray, str]:
    """ 
    Get neutral element (Only implemented for summation of type A)
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
    # print(f"ranks: {ranks}, ranks_D: {ranks_D}")
    if len(ranks_D) > 1:
        raise ValueError(f"Only one rank D is allowed in the orbit.")
    to_cover_id, drop_id = None, None
    if isinstance(orbit, NilpotentOrbit) and orbit.lieType == 'D':
        if orbit.veryEvenType == 'I':
            to_cover_id, drop_id = 1, 0
        elif orbit.veryEvenType == 'II':
            to_cover_id, drop_id = 0, 1
    # print(f"ranks: {ranks_D}")
    chosens, filename = get_chosen_neutral_elements(simple_roots, ranks, ranks_D, to_cover_id, drop_id, ct, k)
    if not chosens:
        return np.zeros(simple_roots.shape[1]), filename
    
    # Compute the sum of chosen roots
    sum_of_chosen_roots = np.zeros(simple_roots.shape[1])
    # print(chosens)
    for rank, chosen_ids in chosens:
        # rank, corresponding_ids
        sum_of_chosen_roots += get_neutral_element_sum_for_rank(rank, chosen_ids, simple_roots)
    # chosen_roots = simple_roots[chosen_ids]
    # print("chosen ids: ", chosen_ids)
    # print("chosen roots: ", chosen_roots)
    # sum_of_chosen_roots = np.sum(chosen_roots, axis=0)
    # print("sum of chosen roots: ", sum_of_chosen_roots)
    return sum_of_chosen_roots, filename

def get_diagram(typ, rank, neutral: np.ndarray, simple_roots: np.ndarray | None) -> list[int]:
    if simple_roots is None:
        weight_ = neutral @ simple_root_data(typ, rank).T
        cmat = cartan_matrix_pycox(typ, rank)
    else:
        weight_ = neutral @ simple_roots.T
        cmat = cartan_matrix_(simple_roots)
    
    antidom, w = antidominant(cmat, weight_)
    w = (-1) * w
    # sps = simple_root_data(typ, rank)
    # print(sps @ w)
    # print('diagram:', round2(w).tolist())
    return Number.round_half(w).tolist() # type: ignore

def need_to_decide_mark(orbit: BalaCarterOrbit) -> bool:
    return all([typ=='A' or (typ, rank) == ('D', 2) or (typ, rank) == ('D', 3)
        for (typ, rank, _) in orbit.orbits.keys()])

def chose_neutral_elements_test(
    ct: LieType, 
    ranks: list[int], 
    ranks_D: list[int], 
    chosen_id: Optional[int]=None, 
    drop_id: Optional[int]=None
) -> list[tuple[int, list[int]]]:
    chosen, _ = get_chosen_neutral_elements(simple_root_data(typ, rank), ranks, ranks_D, chosen_id, drop_id, ct, 0)
    return chosen

def visualize_chosen_ids(
        typ, rank, diagram: nx.Graph, chosen: list[int], save: bool=False) -> str | None:
    import uuid
    print(f"Chosen ids for {typ}_{rank}: {chosen}")
    print(diagram.nodes)
    node_colors = ['lightcoral' if node in chosen else '#7bbce6' for node in diagram.nodes]
    print(node_colors)
    pos = nx.spring_layout(diagram, seed=42)

    # Get edge weights as labels
    edge_labels = nx.get_edge_attributes(diagram, 'weight')
    node_labels = nx.get_node_attributes(diagram, 'simple_root')
    pp_node_labels = {k: f'${PPUtil.pretty_print_array(v)}$' for k, v in node_labels.items()}

    node_label_pos = {k: (v[0], v[1] - 0.1) for k, v in pos.items()}
    edge_label_pos = {k: (v[0], v[1] - 0.1) for k, v in pos.items()}
    # Draw edge labels
    _, ax = plt.subplots(figsize=(5, 4), facecolor='#CAE6F6')
    # nx.draw_networkx_edge_labels(diagram, edge_label_pos, edge_labels=edge_labels, ax=ax)

    nx.draw_networkx_labels(diagram, node_label_pos, labels=pp_node_labels, font_size=10, font_weight='bold', ax=ax)

    nx.draw(diagram, pos, node_color=node_colors, with_labels=True, node_size=2000, font_size=10, font_weight='bold', ax=ax)
    # ax.set_facecolor('#CAE6F6')
    plt.title(f'Chosen Neutral Elements for ${typ}_{rank}$')
    if save:
        filename = f'neutral_elements_chosen_ids_{typ}_{rank}_{uuid.uuid4()}.png'
        plt.savefig(f'LieToolbox/static/images/{filename}', format='png', facecolor='#dce7f0')
        plt.close()
        return filename
    else:
        plt.show()

if __name__ == '__main__':

    typ, rank = 'D', 6
    ranks = [1, 1, 1]
    ranks_D = []
    chosen_id, drop_id = 1, 0
    diagram = get_dynkin_diagram(simple_root_data(typ, rank))
    chosen = chose_neutral_elements_test((typ, rank), ranks, ranks_D, chosen_id, drop_id)
    flattened_chosen_ids = flatten_grouped_chosen(chosen)
    visualize_chosen_ids(typ, rank, diagram, flattened_chosen_ids)

    typ, rank = 'D', 6
    ranks = [1]
    ranks_D = [3]
    chosen_id, drop_id = None, None
    diagram = get_dynkin_diagram(simple_root_data(typ, rank))
    chosen = chose_neutral_elements_test((typ, rank), ranks, ranks_D, chosen_id, drop_id)
    flattened_chosen_ids = flatten_grouped_chosen(chosen)
    visualize_chosen_ids(typ, rank, diagram, flattened_chosen_ids)
    
