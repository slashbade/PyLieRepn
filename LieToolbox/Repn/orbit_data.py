from typing import Literal, Optional
import re


def get_orbit_dual_string(orbit: str):
    pass

OrbitType = Optional[Literal["\'", "\""]]

class Orbit:
    def __init__(self):
        self.orbits = {}
        self.mark: OrbitType = None

    def add_orbit(self, lie_type, rank, parameter=None, multiplicity=1):
        # Key uniquely identifies an orbit by its type, rank, and optional parameter
        key = (lie_type, rank, parameter)
        if key in self.orbits:
            self.orbits[key] += multiplicity
        else:
            self.orbits[key] = multiplicity

    def __add__(self, other):
        if not isinstance(other, Orbit):
            raise TypeError("Can only add NilpotentOrbit objects")
        result = Orbit()
        result.orbits = self.orbits.copy()
        for key, mult in other.orbits.items():
            result.orbits[key] = result.orbits.get(key, 0) + mult
        return result

    def __str__(self):
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

    
    def dual(self):
        return

def parse_orbit_singleton(orbit_string):
    if orbit_string == "1":
        return ("1", 0, None), 1
    orbit_pattern = re.compile(
        r"(\d+)?(\\tilde{[A-Z]}|[A-Z])_(\d+)(?:\((a_\d+)\))?"
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

def parse_orbit_string(orbit_string) -> Orbit:
    mark_pattern = re.compile(
        r"^\((.*?)\)(['\"])?$"
    )
    
    # Initialize an empty Orbit object
    result_orbit = Orbit()
    
    # Check if the entire string is wrapped in parentheses with a type marker
    match = mark_pattern.match(orbit_string.strip())
    
    if match:
        orbit_string = match.group(1)  # Content inside the parentheses
        mark = match.group(2)  # Type marker: ' or "
        # print(f"Type-marked orbit: {orbit_string}, Type: {mark}")  # Debug
        if mark in ["\'", "\""]:
            result_orbit.mark = mark
        else:
            raise ValueError(f"Invalid type marker: {mark}")
    else:
        orbit_string = orbit_string.strip()
        mark = None
    components = orbit_string.split('+')
    for component in components:
        ((lie_type, rank, parameter), multiplicity) = parse_orbit_singleton(component.strip())
        # print(component)
        result_orbit.add_orbit(lie_type, rank, parameter, multiplicity)    
    return result_orbit
    

def from_alvis_notation(alvis: str) -> None:
    pass

def from_partition_notation(partition: list[int]) -> None:
    pass


E6_data = r"""
$1_p$& $E_6$& $72$\\\hline
$6_p$& $E_6(a_1)$& $70$\\\hline
$20_p$& $D_5$& $68$\\\hline
$30_p$& $E_6(a_3)$& $66$\\\hline
$64_p$& $D_5(a_1)$& $64$\\\hline 
$60_p$& $A_4+A_1$& $62$\\\hline
$81_p$& $A_4$& $60$\\
$24_p$& $D_4$& $60$\\\hline
$80_s$& $D_4(a_1)$& $58$\\\hline
$81_p'$& $A_3$& $52$\\\hline
$60_p'$& $A_2+2A_1$& $50$\\\hline
$24_p'$& $2A_2$& $48$\\\hline
$64_p'$& $A_2+A_1$& $46$\\\hline
$30_p'$& $A_2$& $42$\\\hline
$20_p'$& $2A_1$& $32$\\\hline
$6_p'$& $A_1$& $22$\\\hline
$1_p'$& $1$& $0$\\\hline
"""

E7_data = r"""
$1_a$& $E_7$& $126$\\\hline
$7_a'$& $E_7(a_1)$& $124$\\\hline
$27_a$& $E_7(a_2)$& $122$\\\hline
$21_b'$& $E_6$& $120$\\
$56_a'$& $E_7(a_3)$& $120$\\\hline
$120_a$& $E_6$& $118$\\\hline
$189_b'$& $E_7(a_4)$& $116$\\\hline
$168_a$& $D_5+A_1$& $114$\\
$105_b$& $A_6$& $114$\\
$210_a$& $D_6(a_1)$& $114$\\\hline
$189_c'$& $D_5$& $112$\\
$315_a'$& $E_7(a_5)$& $112$\\\hline
$405_a$& $E_6(a_3)$& $110$\\\hline
$378_a'$& $D_5(a_1)+A_1$& $108$\\\hline
$420_a$& $D_5(a_1)$& $106$\\
$210_b$& $A_4+A_2$& $106$\\\hline
$512_a'$& $A_4+A_1$& $104$\\\hline
$105_c$& $(A_5)''$& $102$\\\hline
$210_b'$& $A_3+A_2+A_1$& $100$\\
$420_a'$& $A_4$& $100$\\\hline
$378_a$& $A_3+A_2$& $98$\\\hline
$405_a'$& $D_4(a_1)+A_1$& $96$\\
$105_c'$& $D_4$& $96$\\\hline
$315_a$& $D_4(a_1)$& $94$\\\hline
$189_c$& $(A_3+A_1)''$& $86$\\\hline
$168_a'$& $2A_2$& $84$\\
$210_a'$& $A_3$& $84$\\
$105_b'$& $A_3+3A_1$& $84$\\\hline
$189_b$& $A_2+2A_1$& $82$\\\hline
$120_a'$& $A_2+A_1$& $76$\\\hline
$56_a$& $A_2$& $66$\\\hline
$21_b$& $(3A_1)''$& $54$\\\hline
$21_a'$& $2A_1$& $52$\\\hline
$7_a$& $A_1$& $34$\\\hline
$1_a'$& $1$& $0$\\\hline
"""

E8_data = r"""
$1_x$& $E_8$& $240$\\\hline
$8_z$& $E_8(a_1)$& $238$\\\hline
$35_x$& $E_8(a_2)$& $236$\\\hline
$112_z$& $E_8(a_3)$& $234$\\\hline
$210_x$& $E_8(a_4)$& $232$\\\hline
$560_z$& $E_8(b_4)$& $230$\\\hline
$567_x$& $E_7(a_1)$& $228$\\
$700_x$& $E_8(a_5)$& $228$\\\hline
$1400_z$& $E_8(b_5)$& $226$\\\hline
$1400_x$& $E_8(a_6)$& $224$\\\hline
$3240_z$& $D_7(a_1)$& $222$\\\hline
$2268_x$& $E_7(a_3)$& $220$\\
$2240_x$& $E_8(b_6)$& $220$\\\hline
$4096_z$& $E_6(a_1)+A_1$& $218$\\\hline
$525_x$& $E_6$& $216$\\
$4200_x$& $D_7(a_2)$& $216$\\\hline
$4536_z$& $D_5+A_2$& $214$\\
$2800_z$& $E_6(a_1)$& $214$\\\hline
$2835_x$& $A_6+A_1$& $212$\\
$6075_x$& $E_7(a_4)$& $212$\\\hline
$4200_z$& $A_6$& $210$\\
$5600_z$& $D_6(a_1)$& $210$\\\hline
$4480_y$& $E_8(a_7)$& $208$\\\hline
$2100_y$& $D_5$& $200$\\\hline
$4200_z'$& $D_6+A_2$& $198$\\
$5600_z'$& $E_6(a_3)$& $198$\\\hline
$6075_x'$& $D_5(a_1)+A_1$& $196$\\
$2835_x'$& $A_4+A_2+A_1$& $196$\\\hline
$4536_z'$& $A_4+A_2$& $194$\\\hline
$4200_x'$& $A_4+2A_1$& $192$\\\hline
$2800_z'$& $D_5(a_1)$& $190$\\\hline
$4096_x'$& $A_4+A_1$& $188$\\\hline
$2240_x'$& $D_4(a_1)+A_2$& $184$\\\hline
$2268_x'$& $A_4$& $180$\\\hline
$3240_z'$& $A_3+A_2$& $178$\\\hline
$1400_x'$& $D_4(a_1)+A_1$& $176$\\\hline
$525_x'$& $D_4$& $168$\\\hline
$1400_z'$& $D_4(a_1)$& $166$\\\hline
$700_x'$& $2A_2$& $156$\\\hline
$567_x'$& $A_3$& $148$\\\hline
$560_z'$& $A_2+2A_1$& $146$\\\hline
$210_x'$& $A_2+A_1$& $136$\\\hline
$112_z'$& $A_2$& $114$\\\hline
$35_x'$& $2A_1$& $92$\\\hline
$8_z'$& $A_1$& $58$\\\hline
$1_x'$& $1$& $0$\\\hline
"""

F4_data = r"""
$1_1$& $F_4$& $48$\\\hline
$4_2$& $F_4(a_1)$& $46$\\\hline
$9_1$& $F_4(a_2)$& $44$\\\hline
$8_1$& $B_3$& $42$\\
$8_3$& $C_3$& $42$\\\hline
$12$& $F_4(a_3)$& $40$\\\hline
$8_2$& $\tilde{A}_2$& $30$\\
$8_4$& $A_2$& $30$\\\hline
$9_4$& $A_1+\tilde{A}_1$& $28$\\\hline
$4_5$& $\tilde{A}_1$& $22$\\\hline
$1_4$& $1$& $0$\\\hline
"""

G2_data = r"""
$phi_{1,6}$& $1$& $6$\\
$phi_{1,3}''$& $A_1$& $3$\\
$phi_{2,2}$& $\tilde{A}_1$& $2$\\
$phi_{2,1}$& $G_2(a_1)$& $1$\\
$phi_{1,3}'$& $G_2(a_1)$& $1$\\
$phi_{1,0}$& $G_2$& $0$\\
"""


def find_orbit_from_character(typ, rank, char):
    if typ == "E" and rank == 6:
        data = E6_data
    elif typ == "E" and rank == 7:
        data = E7_data
    elif typ == "E" and rank == 8:
        data = E8_data
    elif typ == "F" and rank == 4:
        data = F4_data
    elif typ == "G" and rank == 2:
        data = G2_data
    else:
        raise ValueError(f"Orbit data for {typ}_{rank} not found.")
    char_to_orbit = data_to_dict(data)
    return char_to_orbit.get(char, None)

def data_to_dict(data):
    d = {}
    for s in data.split("\n"):
        if not s:
            continue
        s = s.replace("$", "").replace("\\", "").replace("hline", "").split("& ")
        d.update({s[0]: s[1]})
    return d

if __name__ == "__main__":
    typ, rank, char = 'E', 8, "1400_x'"
    orb = find_orbit_from_character(typ, rank, char)
    print(orb)
    print(parse_orbit_string("\\tilde{A}_1+D_4"))