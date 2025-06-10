import numpy as np
import networkx as nx

from LieToolbox.Repn.algorithm import Number
from LieToolbox.Repn.algorithm.pir import antidominant
from LieToolbox.Repn.root_system_data import cartan_matrix_pycox, simple_root_data
from LieToolbox.Repn.weight import NilpotentOrbit
from orbit import BalaCarterOrbit, LieType, Typ, get_mark_from_diagram

from functools import reduce
from dataclasses import dataclass
from collections import deque
from typing import Callable, Literal, Optional, TypeVar

T = TypeVar('T')

def bfs(start: T, 
        neighbors: Callable[[T], list[T]]) -> list[T]:
    visited = set()
    queue = deque([start])
    visited.add(start)
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result

def bfs_find(start: T, 
             neighbors: Callable[[T], list[T]], 
             predicate: Callable[[T], bool]) -> Optional[T]:
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()

        if predicate(node):
            return node  # Found node satisfying the predicate

        for neighbor in neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return None  # If no matching node is found

def bfs_find_all(start: T,
                 neighbors: Callable[[T], list[T]],
                 predicate: Callable[[T], bool]) -> list[T]:
    visited = set()
    queue = deque([start])
    visited.add(start)
    results = []
    while queue:
        node = queue.popleft()

        if predicate(node):
            results.append(node)  # Found node satisfying the predicate

        for neighbor in neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return results  # Return all matching nodes found

@dataclass
class PlaceRootState:
    """
    Record the placed roots and the left ranks in the system.
    """
    placed_path: list[int]
    left_num_roots: int

    def __hash__(self) -> int:
        return hash((tuple(self.placed_path), self.left_num_roots))
    
    @staticmethod
    def place_next_root(p: "PlaceRootState", g: nx.Graph) -> list["PlaceRootState"]:
        """
        This is the function to place one root of this rank.
        """
        path, rank = p.placed_path, p.left_num_roots
        if rank < 1:
            return []
        next_placements = []
        for nb in g.neighbors(path[-1]):
            if nb not in path:
                new_path = path + [nb]
                new_rank = rank - 1
                next_placements.append(PlaceRootState(new_path, new_rank))
        return next_placements
    
    @staticmethod
    def is_place_root_end(p: "PlaceRootState") -> bool:
        return p.left_num_roots == 0


@dataclass
class PlaceRankState:
    """
    Record the placed roots and the left ranks in the system.
    """
    placed_roots: list[int] # placed ids in the diagram
    removed_roots: list[int] # ids that should be removed from the diagram
    left_ranks: list[int] # the left ranks to be placed
    
    def __hash__(self) -> int:
        return hash((tuple(self.placed_roots), self.left_ranks))

    @staticmethod
    def _remove_nodes_closure_from(g: nx.Graph, nodes: list[int]) -> nx.Graph:
        """ Remove nodes and all the neighbors """
        adjoints = reduce(set.union, map(lambda x : set(g.neighbors(x)), nodes))
        covered = set(nodes) | adjoints 
        new_diagram = g.copy()
        new_diagram.remove_nodes_from(covered)
        return new_diagram

    @staticmethod
    def _get_all_feasible_rank_place(g: nx.Graph, rank: int) -> list[list[int]]:
        """
        Get all the feasible placements of the rank in the graph.
        """
        def neighbors(p: PlaceRootState) -> list[PlaceRootState]:
            return PlaceRootState.place_next_root(p, g)

        def predicate(p: PlaceRootState) -> bool:
            return PlaceRootState.is_place_root_end(p)

        states = set()
        for start_node in g.nodes:
            start_state = PlaceRootState([start_node], rank)
            for state in bfs_find_all(start_state, neighbors, predicate):
                states.add(state)
        return [state.placed_path for state in states]

    @staticmethod
    def place_next_rank(p: "PlaceRankState", g: nx.Graph) -> list["PlaceRankState"]:
        feasible_graph = PlaceRankState._remove_nodes_closure_from(g, p.placed_roots + p.removed_roots)
        feasible_rank_places = PlaceRankState._get_all_feasible_rank_place(feasible_graph, p.left_ranks[0])
        next_states = []
        for placement in feasible_rank_places:
            new_placed_roots = p.placed_roots + placement
            new_left_ranks = p.left_ranks[1:]
            next_states.append(PlaceRankState(new_placed_roots, [], new_left_ranks))
        return next_states

    @staticmethod
    def is_place_rank_end(p: "PlaceRankState") -> bool:
        return len(p.left_ranks) == 0

def get_weighted_diagram_from_neutral(lie_type: LieType, neutral: np.ndarray) -> list[int]:
    typ, rank = lie_type
    weight_ = neutral @ simple_root_data(typ, rank).T
    cmat = cartan_matrix_pycox(typ, rank)
    _, w = antidominant(cmat, weight_)
    w = (-1) * w
    return Number.round_half(w).tolist() # type: ignore


def _validate_place_rank_state(p: PlaceRankState, simple_roots: np.ndarray, bl_orbit: BalaCarterOrbit) -> bool:
    chosen_roots = simple_roots[p.placed_roots]
    neutral_element = np.sum(chosen_roots, axis=0)
    weighted_diagram = get_weighted_diagram_from_neutral(bl_orbit.lie_type, neutral_element)
    mark = get_mark_from_diagram(bl_orbit, weighted_diagram)
    return mark == bl_orbit.mark

def get_neutral_element_indices(ct: LieType, simple_roots: np.ndarray, bl_orbit: BalaCarterOrbit, orbit: NilpotentOrbit | BalaCarterOrbit) -> list[int]:
    pass 

