E6_data = r"""
$1_p$& $E_6$& $72$\\\hline
$6_p$& $E_{6}(a_1)$& $70$\\\hline
$20_p$& $D_5$& $68$\\\hline
$30_p$& $E_{6}(a_3)$& $66$\\\hline
$64_p$& $D_{5}(a_1)$& $64$\\\hline 
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
$7_a'$& $E_{7}(a_1)$& $124$\\\hline
$27_a$& $E_{7}(a_2)$& $122$\\\hline
$21_b'$& $E_{6}$& $120$\\
$56_a'$& $E_{7}(a_3)$& $120$\\\hline
$120_a$& $E_{6}$& $118$\\\hline
$189_b'$& $E_{7}(a_4)$& $116$\\\hline
$168_a$& $D_5+A_1$& $114$\\
$105_b$& $A_6$& $114$\\
$210_a$& $D_6(a_1)$& $114$\\\hline
$189_c'$  &  $D_5$  & $112$ \\
$315_a'$  &  $E_7(a_5)$  & $112$ \\\hline
$405_a$  &  $E_6(a_3)$  & $110$ \\\hline
$378_a'$  &  $D_5(a_1)+A_1$  & $108$ \\\hline
$420_a$  &  $D_5(a_1)$  & $106$ \\
$210_b$  &  $A_4+A_2$  & $106$ \\\hline
$512_a'$  &  $A_4+A_1$  & $104$ \\\hline
$105_c$  &  $(A_5)''$  & $102$ \\\hline
$210_b'$  &  $A_3+A_2+A_1$  & $100$ \\
$420_a'$  &  $A_4$  & $100$ \\\hline
$378_a$  &  $A_3+A_2$  & $98$ \\\hline
$405_a'$  &  $D_4(a_1)+A_1$  & $96$ \\
$105_c'$  &  $D_4$  & $96$ \\\hline
$315_a$  &  $D_4(a_1)$  & $94$ \\\hline
$189_c$  &  $(A_3+A_1)''$  & $86$ \\\hline
$168_a'$  &  $2A_2$  & $84$ \\
$210_a'$  &  $A_3$  & $84$ \\
$105_b'$  &  $A_3+3A_1$  & $84$ \\\hline
$189_b$  &  $A_2+2A_1$  & $82$ \\\hline
$120_a'$  &  $A_2+A_1$  & $76$ \\\hline
$56_a$  &  $A_2$  & $66$ \\\hline
$21_b$  &  $(3A_1)''$  & $54$ \\\hline
$21_a'$  &  $2A_1$  & $52$ \\\hline
$7_a$  &  $A_1$  & $34$ \\\hline
$1_a'$  &  $1$  & $0$ \\\hline
"""


def find_orbit_from_character(typ, rank, char):
    if typ == "E" and rank == 6:
        data = E6_data
    elif typ == "E" and rank == 7:
        data = E7_data
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
