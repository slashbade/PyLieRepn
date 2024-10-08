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

def get_dual_orbit_exceptional(typ: str, rank: int, orbit: str) -> str:
    if typ == 'E' and rank == 6:
        data = E6_data
    elif typ == 'E' and rank == 7:
        data = E7_data
    else:
        raise ValueError(f"Incorrect Lie type {typ}{rank}.")
    d = data_to_dict(data)
    return d.get(orbit, None)

if __name__ == "__main__":
    print(get_dual_orbit_exceptional('E', 6, 'A_3+A_1'))