from collections import deque
from functools import reduce
from typing import Optional
import typing

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from .root_system_data import simple_root_data, cartan_matrix_pycox
from .roots import get_dynkin_diagram
from .orbit import BalaCarterOrbit
from .weight import NilpotentOrbit
from .algorithm import Number
from .algorithm.pir import antidominant

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
    while queue:
        node, path, length = queue.popleft()
        if length == rank:
            return path
        for neighbor in diagram.neighbors(node):
            if neighbor not in path:
                queue.append((neighbor, path + [neighbor], length + 1))
    return placements

def place_ranks(diagram: nx.Graph, ranks: list[int]) -> list[int]:
    """
    This try to place A-x or D-x orbit in the diagram, and return
    the first feasible choice.
    """
    chosen = []
    queue = deque([(diagram, ranks, chosen)])
    while queue:
        diagram, ranks, chosen = queue.popleft()
        if ranks == []:
            return chosen
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
            queue.append((new_diagram, ranks[1:], chosen + placement))
    # print(queue)
    raise ValueError('No feasible placements')

def place_ranks_all_replacement(diagram: nx.Graph, ranks: list[int]) -> list[list[int]]:
    return [[]]

def place_ranks_for_type_D_very_even(diagram: nx.Graph, ranks: list[int], chosen_id: int, drop_id: int) -> list[int]:
    """
    Place the ranks for type D of very even type.
    """
    chosen = [chosen_id]
    # find a rank 1 and pop it
    ranks = sorted(ranks)
    placement = get_first_feasible_placements_at(diagram, ranks[0], chosen_id)
    placement += chosen
    adjoints = reduce(set.union, map(lambda x : set(diagram.neighbors(x)), placement))
    covered = set(placement) | adjoints | {drop_id}
    new_diagram = diagram.copy()
    new_diagram.remove_nodes_from(covered)
    ranks = ranks[1:]
    left_chosen = place_ranks(new_diagram, ranks)
    return chosen + left_chosen

def place_ranks_for_type_D_D2_D3(diagram: nx.Graph, ranks: list[int], ranks_D: list[int]) -> list[int]:
    """
    Place the ranks for type D2 and D3.
    """
    if not len(ranks_D) == 1:
        raise ValueError('len D ranks not 1, No feasible placements')
    rank_D = ranks_D[0]
    new_diagram = diagram.copy()
    pre_chosen_ids = list(range(rank_D))
    new_diagram.remove_nodes_from(list(range(rank_D + 1)))
    chosen = place_ranks(new_diagram, ranks)
    chosen = pre_chosen_ids + chosen
    return chosen

def get_chosen_neutral_elements(simple_roots, ranks: list[int], ranks_D: list[int], chosen_id: typing.Optional[int], drop_id: typing.Optional[int], ct) -> list[int]:
    diagram = get_dynkin_diagram(simple_roots)
    if not ranks_D == []:
        chosen = place_ranks_for_type_D_D2_D3(diagram, ranks, ranks_D)
    elif chosen_id is not None and drop_id is not None:
        chosen = place_ranks_for_type_D_very_even(diagram, ranks, chosen_id, drop_id)
    else:
        chosen = place_ranks(diagram, ranks)
    visualize_chosen_ids(ct[0], ct[1], diagram, chosen, save=True)
    return chosen

def get_neutral_element_sum(ct: tuple[str, int], simple_roots: np.ndarray, bl_orbit: BalaCarterOrbit, orbit: NilpotentOrbit | BalaCarterOrbit) -> np.ndarray:
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
    if len(ranks_D) > 1:
        raise ValueError(f"Only one rank D is allowed in the orbit.")
    to_cover_id, drop_id = None, None
    if isinstance(orbit, NilpotentOrbit) and orbit.lieType == 'D':
        if orbit.veryEvenType == 'I':
            to_cover_id, drop_id = 1, 0
        elif orbit.veryEvenType == 'II':
            to_cover_id, drop_id = 0, 1
    chosen_ids = get_chosen_neutral_elements(simple_roots, ranks, ranks_D, to_cover_id, drop_id, ct)
    if not chosen_ids:
        return np.zeros(simple_roots.shape[1])
    chosen_roots = simple_roots[chosen_ids]
    # print("chosen ids: ", chosen_ids)
    # print("chosen roots: ", chosen_roots)
    sum_of_chosen_roots = np.sum(chosen_roots, axis=0)
    # print("sum of chosen roots: ", sum_of_chosen_roots)
    return sum_of_chosen_roots

def get_diagram(typ, rank, neutral: np.ndarray) -> list[int]:
    # print(neutral)
    weight_ = neutral @ simple_root_data(typ, rank).T
    cmat = cartan_matrix_pycox(typ, rank)
    antidom, w = antidominant(cmat, weight_)
    w = (-1) * w
    # sps = simple_root_data(typ, rank)
    # print(sps @ w)
    # print('diagram:', round2(w).tolist())
    return Number.round_half(w).tolist() # type: ignore

def need_to_decide_mark(orbit: BalaCarterOrbit) -> bool:
    return all([typ=='A' or (typ, rank) == ('D', 2) or (typ, rank) == ('D', 3)
        for (typ, rank, _) in orbit.orbits.keys()])

def chose_neutral_elements_test(typ: str, rank: int, ranks: list[int], ranks_D: list[int], chosen_id: Optional[int]=None, drop_id: Optional[int]=None) -> list[int]:
    chosen = get_chosen_neutral_elements(simple_root_data(typ, rank), ranks, ranks_D, chosen_id, drop_id, (typ, rank))
    return chosen

def visualize_chosen_ids(typ, rank, diagram: nx.Graph, chosen: list[int], save: bool=False) -> None:
    node_colors = ['lightcoral' if node in chosen else '#7bbce6' for node in diagram.nodes]

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
        plt.savefig(f'LieToolbox/static/images/neutral_elements_chosen_ids_{typ}_{rank}.png', format='png', facecolor='#dce7f0')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':

    typ, rank = 'D', 6
    ranks = [1, 1, 1]
    ranks_D = []
    chosen_id, drop_id = 1, 0
    diagram = get_dynkin_diagram(simple_root_data(typ, rank))
    chosen = chose_neutral_elements_test(typ, rank, ranks, ranks_D, chosen_id, drop_id)
    visualize_chosen_ids(typ, rank, diagram, chosen)

    typ, rank = 'D', 6
    ranks = [1]
    ranks_D = [3]
    chosen_id, drop_id = None, None
    diagram = get_dynkin_diagram(simple_root_data(typ, rank))
    chosen = chose_neutral_elements_test(typ, rank, ranks, ranks_D, chosen_id, drop_id)
    visualize_chosen_ids(typ, rank, diagram, chosen)
    
