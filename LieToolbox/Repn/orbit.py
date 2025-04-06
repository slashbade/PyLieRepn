from typing import Literal, Optional
import re
import json
import copy
from pathlib import Path

OrbitType = Optional[Literal["\'", "\""]]
Typ = Literal["A", "B", "C", "D", "E", "F", "G"]
LieType = tuple[Typ, int]

# A class to handle Bala-Carter labels
class BalaCarterOrbit:
    def __init__(self):
        self.orbits = {}
        self.mark: OrbitType = None
        self.lie_type: LieType = ('A', 1)

    def add_orbit(self, typ, rank, parameter=None, multiplicity=1):
        # Key uniquely identifies an orbit by its type, rank, and optional parameter
        if rank == 0:
            return
        key = (typ, rank, parameter)
        if key in self.orbits:
            self.orbits[key] += multiplicity
        else:
            self.orbits[key] = multiplicity

    def __add__(self, other):
        if not isinstance(other, BalaCarterOrbit):
            raise TypeError("Can only add BalaCarterOrbit objects")
        result = BalaCarterOrbit()
        result.orbits = self.orbits.copy()
        result.mark = self.mark
        result.lie_type = self.lie_type
        for key, mult in other.orbits.items():
            (_, rank, _) = key
            if rank == 0:
                continue
            result.orbits[key] = result.orbits.get(key, 0) + mult
        return result

    def __str__(self):
        if len(self.orbits.keys()) == 0: #and list(self.orbits.keys())[0][1] == 0:
            return "0"
        sorted_keys = sorted(
            self.orbits.keys(),
            key=lambda x: (x[0], x[1], x[2] or "")  # Parameter `None` is treated as an empty string
        )
        # Build the string representation
        components = []
        for lie_type, rank, parameter in sorted_keys:
            multiplicity = self.orbits[(lie_type, rank, parameter)]
            part = f"{multiplicity if multiplicity > 1 else ''}{lie_type}_{rank}"
            if parameter:
                part += f"({parameter})"
            components.append(part)
        inner_orbit = " + ".join(components)
        if self.mark:
            return f"({inner_orbit}){self.mark}"
        else:
            return inner_orbit

    def ls_dual(self) -> "BalaCarterOrbit":
        orbit_string = str(self)
        root = Path(__file__).parent
        with open(root / "data" / "ls_dual" / f"{self.lie_type[0]}{self.lie_type[1]}.json", "r") as f:
            data = json.load(f)
        for d in data:
            if d["orbit"] == orbit_string:
                return from_orbit_string(d["dual"], self.lie_type)
        raise ValueError(f"Orbit {orbit_string} not found in ls_dual/{self.lie_type[0]}{self.lie_type[1]}.json")
    
    def sommers_dual(self) -> "BalaCarterOrbit":
        orbit_string = str(self)
        root = Path(__file__).parent
        with open(root / "data" / "sommers_dual" / f"{self.lie_type[0]}{self.lie_type[1]}.json", "r") as f:
            data = json.load(f)
        for d in data:
            if d["orbit"] == orbit_string:
                return from_orbit_string(d["dual"], self.lie_type)
            if d["dual"] == orbit_string:
                return from_orbit_string(d["orbit"], self.lie_type)
        raise ValueError(f"Orbit {orbit_string} not found in sommers_dual/{self.lie_type[0]}{self.lie_type[1]}.json")
    
    def dual(self) -> "BalaCarterOrbit":
        orbit_string = str(self)
        root = Path(__file__).parent
        with open(root / "data" / "dual" / f"{self.lie_type[0]}{self.lie_type[1]}.json", "r") as f:
            data = json.load(f)
        for d in data:
            if d["orbit"] == orbit_string:
                return from_orbit_string(d["dual"], self.lie_type)
        raise ValueError(f"Orbit {orbit_string} not found in ls_dual/{self.lie_type[0]}{self.lie_type[1]}.json")



def set_mark(bl: BalaCarterOrbit, mark: OrbitType):
    bl.mark = mark
    return copy.copy(bl)


def get_mark_from_diagram(bl: BalaCarterOrbit, diagram: list) -> BalaCarterOrbit:
    assert bl.mark is None
    root = Path(__file__).parent

    with open(root / "data" / "dual" / f"{bl.lie_type[0]}{bl.lie_type[1]}.json", "r") as f:
    # with open(root / "data" / "sommers_dual" / f"{bl.lie_type[0]}{bl.lie_type[1]}.json", "r") as f:
        data = json.load(f)
    candidates = [d for d in data if d['diagram'] == diagram]
    for cand in candidates:
        if str(set_mark(bl, '\'')) == cand['orbit']:
            return set_mark(bl, '\'')
        if str(set_mark(bl, '\"')) == cand['orbit']:
            return set_mark(bl, '\"')
        if str(set_mark(bl, None)) == cand['orbit']:
            return set_mark(bl, None)
    
    print(f'Orbit {bl} with diagram {diagram} not found \
                      in sommers_dual/{bl.lie_type[0]}{bl.lie_type[1]}.json')
    return bl

def parse_orbit_singleton(orbit_string):
    if orbit_string == "":
        return ("A", 0, None), 1
    if orbit_string == "1":
        return ("1", 0, None), 1
    if orbit_string == "0":
        return ("1", 0, None), 1
    orbit_pattern = re.compile(
        r"(\d+)?(\\tilde{[A-Z]}|[A-Z])_(\d+)(?:\(([a-z]_\d+)\))?"
    )
    match = orbit_pattern.match(orbit_string)
    if match:
        multiplicity = int(match.group(1)) if match.group(1) else 1
        lie_type = match.group(2)
        rank = int(match.group(3))
        parameter = match.group(4) if match.group(4) else None
    else:
        raise ValueError(f"Invalid orbit string: {orbit_string}")
    return (lie_type, rank, parameter), multiplicity

def from_orbit_string(orbit_string, lie_typ: LieType = ('A', 1)) -> BalaCarterOrbit:
    mark_pattern = re.compile(
        r"^\((.*?)\)(['\"])?$"
    )
    
    # Initialize an empty Orbit object
    result_orbit = BalaCarterOrbit()
    result_orbit.lie_type = lie_typ

    # Check if the entire string is wrapped in parentheses with a type marker
    match = mark_pattern.match(orbit_string.strip())
    
    if match:
        orbit_string = match.group(1)  # Content inside the parentheses
        mark = match.group(2)  # Type marker: ' or "
        # print(f"Type-marked orbit: {orbit_string}, Type: {mark}")  # Debug
        if mark in ["\'", "\""]:
            result_orbit.mark = mark # type: ignore
        else:
            raise ValueError(f"Invalid type marker: {mark}")
    else:
        orbit_string = orbit_string.strip()
        mark = None
    components = orbit_string.split('+')
    for component in components:
        ((typ, rank, parameter), multiplicity) = parse_orbit_singleton(component.strip())
        # print(component)
        result_orbit.add_orbit(typ, rank, parameter, multiplicity)    
    return result_orbit
    

def from_alvis_notation(alvis: str, lie_type: LieType) -> BalaCarterOrbit:
    root = Path(__file__).parent
    with open(root / "data" / "notation" / f"{lie_type[0]}{lie_type[1]}.json", "r") as f:
        data = json.load(f)
    for d in data:
        if d["alvis"] == alvis:
            return from_orbit_string(d["sommers"], lie_type)
    raise ValueError(f"Alvis notation {alvis} not found in notation/{lie_type[0]}{lie_type[1]}.json")


def from_partition_dual_singleton(typ: str, orbit: list[int]) -> BalaCarterOrbit:
    if typ == 'A':
        if len(orbit) == 1:
            if orbit[0] == 1:
                orbit_string = ""
            else:
                orbit_string = f"A_{orbit[0]-1}"
        else:
            raise ValueError(f"Invalid orbit {orbit}.")
    elif typ == 'B':
        if orbit == [5, 3, 1]:
            orbit_string = "B_4(a_2)"
        elif orbit == [1, 1]:
            orbit_string = ""
        elif len(orbit) == 2 and orbit[0] == orbit[1]:
            orbit_string = f"A_{orbit[0]-1}"
        elif len(orbit) == 1 and (orbit[0]-1) % 2 == 0:
            orbit_string = f"B_{(orbit[0]-1)//2}"
        else:
            raise ValueError(f"Invalid orbit {orbit}.")
    elif typ == 'C':
        if orbit == [2, 2, 2]:
            orbit_string = r"A_1+\tilde{A}_1"
        elif orbit == [1, 1]:
            orbit_string = ""
        elif orbit == [4, 2]:
            orbit_string = "C_3(a_1)"
        elif len(orbit) == 2 and orbit[0] == orbit[1]:
            orbit_string = rf"\tilde{{A}}_{orbit[0]-1}"
        elif len(orbit) == 1:
            assert (orbit[0]) % 2 == 0
            orbit_string = f"C_{orbit[0]//2}"
        else:
            raise ValueError(f"Invalid orbit {orbit}.")
    elif typ == 'D':
        if orbit == [7, 5, 3, 1]:
            orbit_string = "D_8(a_5)"
        elif orbit == [5, 3, 3, 1]:
            orbit_string = "A_3+A_2"
        elif orbit == [1, 1]:
            orbit_string = ""
        elif orbit == [5, 1]:
            orbit_string = "A_3"
        elif orbit == [3, 1]:
            orbit_string = "2A_1"
        elif len(orbit) == 2 and orbit[0] == orbit[1]:
            orbit_string = f"A_{orbit[0]-1}"
        elif len(orbit) == 2 and orbit[1] == 1 and (orbit[0]+1) % 2 == 0:
            orbit_string = f"D_{(orbit[0]+1)//2}"
        elif len(orbit) == 2 and (orbit[0]+orbit[1]) % 2 == 0 and orbit[1] % 2 == 1:
            orbit_string = f"D_{(orbit[0]+orbit[1])//2}(a_{(orbit[1]-1)//2})"
        else:
            raise ValueError(f"Invalid orbit {orbit}.")
    else:
        raise ValueError(f"Invalid Lie type {typ}.") 
    return from_orbit_string(orbit_string, (typ, len(orbit)))


def from_partition_dual(typ: Typ, orbit: list[int]) -> BalaCarterOrbit:
    """ Determine Bala-Carter label for a given nilpotent orbit represented by partition.
    """
    if not orbit:
        bl_orbit = BalaCarterOrbit()
        bl_orbit.lie_type = (typ, 1)
        return BalaCarterOrbit()
    for match_len in range(4, 0, -1):
        try:
            return (from_partition_dual_singleton(typ, orbit[-match_len:]) + 
                    from_partition_dual(typ, orbit[:-match_len]))
        except ValueError as e:
            continue
    raise ValueError(f"Invalid orbit {orbit}.")




if __name__ == "__main__":
    # typ, rank, char = 'E', 8, "1400_x'"
    # orb = find_orbit_from_character(typ, rank, char)
    # print(orb)
    print(from_orbit_string("\\tilde{A}_1+D_4", ('F', 4)))
    print(from_alvis_notation("phi_{1,6}", ('G', 2)))
    print(from_partition_dual_singleton('D', [5, 3, 3, 1]))
    print(from_partition_dual('D', [7, 5, 2, 2, 1, 1]))
    print(from_partition_dual('B', [7, 1, 1]))
    bl = from_orbit_string('4A_1', ('E', 8))
    bl_marked = get_mark_from_diagram(bl, "00000020")
    print(bl_marked)
    print(bl_marked.sommers_dual())
