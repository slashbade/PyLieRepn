import json
from pathlib import Path
from orbit import BalaCarterOrbit, from_orbit_string

E6_data = r"""
0 E_6
A_1 E_6(a_1)
2A_1 D_5
3A_1 E_6(a_3)
A_2 E_6(a_3)
A_2+A_1 D_5(a_1)
2A_2 D_4
A_2+2A_1 A_4+A_1
A_3 A_4
2A_2+A_1 D_4(a_1)
A_3+A_1 D_4(a_1)
D_4(a_1) D_4(a_1)
A_4 A_3
D_4 2A_2
A_4+A_1 A_2+2A_1
A_5 A_2
D_5(a_1) A_2+A_1
E_6(a_3) A_2
D_5 2A_1
E_6(a_1) A_1
E_6 0
"""

E7_data = r"""
0 E_7
A_1 E_7(a_1)
2A_1 E_7(a_2)
(3A_1)" E_6
(3A_1)' E_7(a_3)
A_2 E_7(a_3)
4A_1 E_6(a_1)
A_2+A_1 E_6(a_1)
A_2+2A_1 E_7(a_4)
A_3 D_6(a_1)
2A_2 D_5+A_1
A_2+3A_1 A_6
(A_3+A_1)" D_5
2A_2+A_1 E_7(a_5)
(A_3+A_1)' E_7(a_5)
D_4(a_1) E_7(a_5)
A_3+2A_1 E_6(a_3)
D_4 (A_5)"
D_4(a_1)+A_1 E_6(a_3)
A_3+A_2 D_5(a_1)+A_1
A_4 D_5(a_1)
A_3+A_2+A_1 A_4+A_2 
(A_5)" D_4
D_4+A_1 A_4 
A_4+A_1 A_4+A_1
D_5(a_1) A_4
A_4+A_2 A_3+A_2+A_1
(A_5)' D_4(a_1)+A_1
A_5+A_1 D_4(a_1)
D_5(a_1)+A_1 A_3+A_2
D_6(a_2) D_4(a_1)
E_6(a_3) D_4(a_1)+A_1
D_5 (A_3+A_1)"
E_7(a_5) D_4(a_1)
A_6 A_2+3A_1
D_5+A_1 2A_2
D_6(a_1) A_3
E_7(a_4) A_2+2A_1
D_6 A_2
E_6(a_1) A_2+A_1
E_6 (3A_1)"
E_7(a_3) A_2
E_7(a_2) 2A_1
E_7(a_1) A_1
E_7 0
"""

def data_to_dict(data: str) -> dict:
    d = {}
    for s in data.split("\n"):
        if not s:
            continue
        s = s.replace("$", "").replace("\\", "").replace("hline", "").split(" ")
        d.update({s[0]: s[1]})
    return d

def save_parsed_data(data: str, name: str, root: Path = Path("LieToolbox/Repn/data")) -> None:
    import json
    d = []
    for s in data.split("\n"):
        if not s:
            continue
        s = s.replace("$", "").replace("\\", "").replace("hline", "").split(" ")
        d.append({"orbit": from_orbit_string(s[0]).__str__(), "dual": from_orbit_string(s[1]).__str__()})
    data_path = root / "ls_dual"
    with open(data_path / f"{name}.json", "w") as f:
        json.dump(d, f, indent=4)
    

def get_dual_orbit_exceptional(typ: str, rank: int, orbit: str) -> str:
    if typ == 'E' and rank == 6:
        data = E6_data
    elif typ == 'E' and rank == 7:
        data = E7_data
    else:
        raise ValueError(f"Incorrect Lie type {typ}{rank}.")
    d = data_to_dict(data)
    return d.get(orbit, None)


def get_bala_carter_label_from_dual(typ: str, orbit: list[int]):
    if typ == 'A':
        assert len(orbit) == 1
        return f"A_{orbit[0]-1}"
    elif typ == 'B':
        if len(orbit) == 1:
            assert (orbit[0]-1) % 2 == 0
            return f"B_{(orbit[0]-1)//2}"
        elif orbit == [5, 3, 1]:
            return "B_4(a_2)"
        elif len(orbit) == 2:
            assert orbit[0] == orbit[1]
            return f"A_{orbit[0]-1}"
        else:
            raise ValueError(f"Invalid orbit {orbit}.")
    elif typ == 'C':
        if len(orbit) == 1:
            assert (orbit[0]) % 2 == 0
            return f"C_{orbit[0]//2}"
        elif len(orbit) == 2:
            assert orbit[0] == orbit[1]
            return rf"\tilde{{A}}_{orbit[0]-1}"
        elif orbit == [2, 2, 2]:
            return r"A_1+\tilde{A}_1"
        else:
            raise ValueError(f"Invalid orbit {orbit}.")
    elif typ == 'D':
        if orbit == [7, 5, 3, 1]:
            return "D_8(a_5)"
        elif orbit == [9, 3]:
            return "D_6(a_1)"
        elif orbit == [7, 5]:
            return "D_6(a_2)"
        elif orbit == [7, 3]:
            return "D_5(a_1)"
        elif orbit == [5, 3]:
            return "D_4(a_1)"
        elif orbit == [5, 1]:
            return "A_3"
        elif orbit == [5, 3, 3, 1]:
            return "A_3+A_2"
        elif orbit == [3, 1]:
            return "2A_1"
        elif len(orbit) == 2 and orbit[0] == orbit[1]:
            return f"A_{orbit[0]-1}"
        elif len(orbit) == 2 and orbit[1] == 1 and (orbit[0]+1) % 2 == 0:
            return f"D_{(orbit[0]+1)//2}"
        elif len(orbit) == 2 and (orbit[0]+orbit[1]) % 2 == 0 and orbit[1] % 2 == 1:
            return f"D_{(orbit[0]+orbit[1])//2}(a_{(orbit[1]-1)//2})"
        else:
            raise ValueError(f"Invalid orbit {orbit}.")
    else:
        raise ValueError(f"Invalid Lie type {typ}.") 


if __name__ == "__main__":
    # print(get_dual_orbit_exceptional('E', 6, 'A_3+A_1'))
    save_parsed_data(E6_data, "E6")
    save_parsed_data(E7_data, "E7")