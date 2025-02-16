import json
from pathlib import Path
import re
import sys
sys.path.append('.')
from LieToolbox.Repn.orbit import from_orbit_string

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

G2_sommers_dual_data = r"""
\weightDynkinG{.2}{2}{0} & * $G_{2}(a_1)$ & 1 & $S_3$ & $G_{2}(a_1)$ \\ 
 & * $A_1 + \Tilde{A_{1}}$ & 2 & & $\Tilde{A_{1}}$ \\ 
 & * $A_2$ & 3 & & $A_{1}$ \\ \hline
"""

F4_sommers_dual_data = r"""
\weightDynkinF{.2}{0}{0}{0}{1} & * $\Tilde A_{1}$ & $1$ & $S_2$ & $F_{4}(a_{1})$ \\ 
    & $2A_{1}$ & 2 && $F_{4}(a_{2})$   \\ \hline

\weightDynkinF{.2}{2}{0}{0}{0} & * $A_{2}$ & 3 & $1$ & $B_3$ \\ 
    & $2A_{1} + \Tilde{A_{1}}$ & 3 & & $B_3$ \\ \hline

\weightDynkinF{.2}{2}{0}{0}{1} &  $B_{2}$ & 4 & $S_2$ & $F_{4}(a_{3})$ \\ 
  & $A_3$ & 6 & & $B_2$ \\ \hline

\weightDynkinF{.2}{1}{0}{1}{0} & $C_{3}(a_{1})$ & 4 &  $S_2$ & $F_{4}(a_{3})$  \\ 
   & $A_{1} + B_{2}$ & 5 & & $C_3(a_1)$ \\ \hline

\weightDynkinF{.2}{0}{2}{0}{0} & * $F_{4}(a_{3})$ & 4 &  $S_4$ & $F_{4}(a_{3})$ \\ 
        & * $A_{1} + C_{3}(a_{1})$ & 5 & & $C_{3}(a_{1})$\\  
        & * $A_{2} + \Tilde{A_{2}}$ & 6 & &$A_{1} + \Tilde{A_{2}}$\\ 
        & * $B_{4}(a_{2})$ & 6 & & $B_{2}$ \\ 
        & * $A_{3} + \Tilde{A_{1}}$ &  7 &  & $A_{2} + \Tilde{A_{1}}$ \\ \hline

\weightDynkinF{.2}{0}{2}{0}{2} & * $F_{4}(a_{2})$ & 10 & $1$ & $A_{1} + \Tilde{A_{1}}$ \\ 
                & $A_{1} + C_{3}$ & 10 & &  $A_{1} + \Tilde{A_{1}}$ \\ \hline 

\weightDynkinF{.2}{2}{2}{0}{2} & * $F_{4}(a_{1})$ & 13 & $S_2$ & $\Tilde{A_{1}}$ \\ 
                & * $B_{4}$ & 16 &  & $A_1$ \\ \hline
"""

E6_sommers_dual_data = r"""
\weightDynkinEsix{.18}{0}{0}{0}{0}{0}{2} & * $A_{2}$ & 3 & $S_2$ & $E_{6}(a_{3})$ \\ 
                 & * $4 A_{1}$ & 4 & & $A_{5}$\\ \hline

\weightDynkinEsix{.18}{0}{0}{2}{0}{0}{0} & * $D_{4}(a_{1})$  & 7 &  $S_3$ & $D_{4}(a_{1})$  \\ 
                 & * $A_{3} + 2A_{1}$ & 8 & &  $A_3 + A_1$ \\
                 & * $3A_{2}$ & 9 & & $2A_2 + A_1$ \\ \hline

\weightDynkinEsix{.18}{2}{0}{2}{0}{2}{0} & * $E_{6}(a_{3}) $ & 15  & $S_2$ & $A_{2}$ \\ 
                 & * $A_{5} + A_{1}$ & 16 & & $3A_1$ \\ \hline
"""

E7_sommers_dual_data = r"""
\weightDynkinEseven{.18}{2}{0}{0}{0}{0}{0}{0} & * $A_{2}$ & 3 & $S_2$ & $E_7(a_3)$ \\ 
                 & * $(4 A_{1})'$ & 4 & & $D_6$ \\ \hline

\weightDynkinEseven{.18}{1}{0}{0}{0}{1}{0}{0} & * $A_{2}+A_{1}$ & 4 & $S_2$ & $E_6(a_1)$\\ 
                 & $5 A_{1}$ & 5 & & $E_7(a_4)$  \\ \hline

\weightDynkinEseven{.18}{0}{2}{0}{0}{0}{0}{0} & * $D_{4}(a_{1})$  & 7 & $S_3$ & $E_7(a_5)$\\ 
                 & * $3A_{2}$ & 9 & & $A_5+A_1$ \\ 
                 & * $(A_{3} + 2A_{1})'$ & 8 & & $D_6(a_2)$ \\ \hline

\weightDynkinEseven{.18}{0}{1}{0}{0}{0}{1}{1} & * $D_{4}(a_{1})+A_{1} $ & 8 & $S_2$ & $E_6(a_3)$ \\ 
                 & * $A_{3} + 3A_{1}$ & 9 & & $(A_5)'$ \\ \hline

\weightDynkinEseven{.18}{0}{0}{1}{0}{1}{0}{0} & * $A_{3}+A_{2} $ & 9 & $1$ & $D_5(a_1)+A_1$ \\ 
                 & $D_{4}(a_{1}) + 2A_{1}$ & 9 & & $D_5(a_1)+A_1$ \\ \hline

\weightDynkinEseven{.18}{2}{0}{0}{0}{2}{0}{0} & * $A_{4}$ & 10 & $S_2$ & $D_5(a_1)$  \\ 
                 & * $2A_{3}$ & 12 & &  $D_4+A_1$\\ \hline

\weightDynkinEseven{.18}{1}{0}{1}{0}{1}{0}{0} & * $A_{4} + A_{1}$ & 11 & $S_2$ & $A_4+A_1$ \\ 
                 & $A_{1}+2A_{3}$ & 13 & & $A_3+A_2+A_1$ \\ \hline

\weightDynkinEseven{.18}{2}{0}{1}{0}{1}{0}{0} & * $D_{5}(a_{1})$ & 13 & $S_2$ & $A_4$\\ 
                 & $D_{4} + 2A_{1}$ & 14 & & $A_3+A_2$ \\ \hline

\weightDynkinEseven{.18}{0}{2}{0}{0}{2}{0}{0} & * $E_{6}(a_{3})$ & 15 &  $S_2$ & $D_4(a_1)+A_1$  \\
                 & * $(A_{5} + A_{1})'$ & 16 & & $A_3+2A_1$ \\ \hline

\weightDynkinEseven{.18}{0}{0}{2}{0}{0}{2}{0} & * $E_{7}(a_{5})$ & 16 & $S_3$ & $D_4(a_1)$ \\ 
                 & * $A_{5}+A_{2}$ & 18 &  & $2A_2+A_1$ \\ 
                 & * $A_{1}+D_{6}(a_{2})$ & 17 & & $(A_3+A_1)'$ \\ \hline

\weightDynkinEseven{.18}{2}{0}{2}{0}{0}{2}{0} & * $E_{7}(a_{4})$ & 22 & $1$ & $A_2+2A_1$ \\ 
                 & $A_{1}+D_{6}(a_{1})$ &  22 &  & $A_2+2A_1$ \\ \hline

\weightDynkinEseven{.18}{2}{0}{2}{0}{2}{0}{0} & * $E_{6}(a_{1})$ & 25 & $S_2$ & $A_2+A_1$\\ 
                 &  * $A_{7}$ & 28 & & $4A_1$ \\ \hline

\weightDynkinEseven{.18}{2}{0}{2}{0}{2}{2}{0} & * $E_{7}(a_{3})$ & 30 & $S_2$ & $A_2$ \\ 
                 &  * $A_{1}+D_{6}$ & 31 & &  $(3A_1)'$ \\ \hline
"""

E8_sommers_dual_data = r"""
\weightDynkinEeight{.18}{0}{0}{0}{0}{0}{0}{2}{0} & * $A_{2}$ & 3 & $S_2$ & $E_8(a_3)$\\ 
                 & * $(4 A_{1})''$ & 4 & & $E_7$ \\ \hline

\weightDynkinEeight{.18}{1}{0}{0}{0}{0}{0}{1}{0} & * $A_{2}+A_{1}$ & 4  & $S_2$& $E_8(a_4)$ \\ 
                 & $5 A_{1}$ & 5 & & $E_8(b_4)$ \\ \hline

\weightDynkinEeight{.18}{2}{0}{0}{0}{0}{0}{0}{0} & * $2A_{2}$ & 6  & $S_2$& $E_8(a_5)$ \\ 
                 & * $A_{2}+4 A_{1}$ & 7 & & $D_7$ \\ \hline

\weightDynkinEeight{.18}{0}{0}{0}{0}{0}{2}{0}{0} & * $D_{4}(a_{1})$  & 7  & $S_3$& $E_8(b_5)$ \\ 
                 &  * $3A_{2}$ & 9 & & $E_6+A_1$\\ 
                 &  * $(A_{3} + 2A_{1})''$ & 8 & & $E_7(a_2)$ \\ \hline

\weightDynkinEeight{.18}{0}{0}{0}{0}{0}{1}{0}{1} & *  $D_{4}(a_{1})+A_{1} $ & 8 & $S_3$ & $E_8(a_6)$\\ 
                 &  $3A_{2} + A_{1}$ & 10 & & $E_8(b_6)$ \\ 
                 &  $A_{3} + 3A_{1}$ & 9 & & $D_7(a_1)$ \\ \hline

\weightDynkinEeight{.18}{1}{0}{0}{0}{1}{0}{0}{0} & * $A_{3}+A_{2}$ & 9 & $1$ & $D_7(a_1)$ \\ 
                 &  $D_{4}(a_{1}) + 2A_{1}$ & 9 &  & $D_7(a_1)$ \\ \hline

\weightDynkinEeight{.18}{0}{0}{0}{0}{0}{0}{0}{2} & * $D_{4}(a_{1}) + A_{2}$ & 10 & $S_2$ & $E_8(b_6)$ \\ 
                 &  * $A_{3}+A_{2}+2A_{1}$ & 11 & & $A_7$ \\ \hline

\weightDynkinEeight{.18}{2}{0}{0}{0}{0}{0}{2}{0} & *  $A_{4}$ & 10 & $S_2$ & $E_7(a_3)$ \\ 
                 &  * $(2A_{3})''$ & 12 & & $D_6$  \\ \hline

\weightDynkinEeight{.18}{1}{0}{0}{0}{1}{0}{1}{0} &  * $A_{4} + A_{1}$ & 11 & $S_2$ & $E_6(a_1)+A_1$ \\ 
                 &   $A_{1}+2A_{3}$ & 13 & & $D_5+A_2$ \\ \hline

\weightDynkinEeight{.18}{0}{0}{1}{0}{0}{0}{1}{0} & * $A_{4} + 2A_{1}$ & 12 & $S_2$& $D_7(a_2)$ \\ 
                 &   $D_{4}(a_{1}) + A_{3}$ & 13 & & $D_5+A_2$ \\ \hline

\weightDynkinEeight{.18}{1}{0}{0}{0}{1}{0}{2}{0} & * $D_{5}(a_{1})$ & 13  & $S_2$& $E_6(a_1)$ \\ 
                 &   $D_{4} + 2A_{1}$ & 14 & & $E_7(a_4)$ \\ \hline
 
\weightDynkinEeight{.18}{0}{0}{0}{0}{0}{0}{2}{2} & * $D_{4}+A_{2}$ & 15 & $1$ & $A_6$ \\ 
                 &   $D_{5}(a_{1}) + 2A_{1}$ & 15 & & $A_6$ \\ \hline

\weightDynkinEeight{.18}{2}{0}{0}{0}{0}{2}{0}{0} & * $E_{6}(a_{3})$ & 15 & $S_2$ & $D_6(a_1)$  \\ 
                &  *  $(A_{5} + A_{1})''$ & 16 & & $D_5+A_1$ \\ \hline

\weightDynkinEeight{.18}{0}{1}{0}{0}{0}{1}{0}{1} &  $D_{6}(a_{2})$ & 16 & $S_2$& $E_8(a_7)$ \\ 
                 &  $D_{4}+A_{3}$ & 18 & &  $D_{6}(a_{2})$\\ \hline

\weightDynkinEeight{.18}{1}{0}{0}{1}{0}{1}{0}{0} & $E_{6}(a_{3})+A_{1}$ & 16 & $S_2$ & $E_8(a_7)$ \\ 
                 &  $A_{5} + 2 A_{1}$ & 17 & &  $E_7(a_5)$ \\ \hline

\weightDynkinEeight{.18}{0}{0}{1}{0}{1}{0}{0}{0} & $E_{7}(a_{5})$  & 16 & $S_3$& $E_8(a_7)$ \\ 
                 &  $A_{5}+A_{2}$ & 18 & & $E_6(a_3)+A_1$ \\ 
                 &  $A_{1}+D_{6}(a_{2})$ & 17 & & $E_7(a_5)$ \\ \hline

\weightDynkinEeight{.18}{0}{0}{0}{2}{0}{0}{0}{0} & * $E_{8}(a_{7})$  & 16 & $S_5$ & $E_8(a_7)$ \\ 
                 &  * $A_{5}+A_{2}+A_{1}$ & 19 &  &$A_5+A_1$  \\ 
                 &  * $2A_{4}$ & 20 &  & $A_4+A_3$ \\ 
                 &  * $D_{5}(a_{1}) + A_{3}$ & 19  &  & $D_5(a_1)+A_2$  \\
                 &  * $D_{8}(a_{5})$ & 18  & & $D_6(a_2)$ \\
                 &  * $E_{7}(a_{5})+A_{1}$ & 17  && $E_7(a_5)$ \\
                 &  * $E_{6}(a_{3})+A_{2}$ & 18  &&   $E_6(a_3)+A_1$ \\ \hline

\weightDynkinEeight{.18}{0}{1}{0}{0}{0}{1}{2}{1} & * $D_{6}(a_{1})$ & 21 & $S_2$& $E_6(a_3)$ \\ 
                 &  * $D_{5}+2A_{1}$ & 22 & & $A_5$ \\ \hline

\weightDynkinEeight{.18}{0}{0}{1}{0}{1}{0}{2}{0} &  * $E_{7}(a_{4})$ & 22 & $1$ & $D_5(a_1)+A_1$ \\ 
                 &  $A_{1}+D_{6}(a_{1})$ &  22 &  &$D_5(a_1)+A_1$ \\ \hline

\weightDynkinEeight{.18}{0}{0}{0}{2}{0}{0}{2}{0} & * $D_{5}+A_{2}$ & 23 &  $1$ &$A_4+A_2$ \\
                 &  $E_{7}(a_{4})+A_{1}$ & 23 &   &$A_4+A_2$ \\ \hline

\weightDynkinEeight{.18}{1}{0}{1}{0}{1}{0}{1}{0}  & * $D_{7}(a_{2})$ & 24 &  $S_2$& $A_4+2A_1$ \\ 
                 &  * $D_{5}+A_{3}$ & 26 & & $2A_3$ \\ \hline

\weightDynkinEeight{.18}{2}{0}{0}{0}{2}{0}{2}{0} &  * $E_{6}(a_{1})$ & 25 & $S_2$& $D_5(a_1)$ \\ 
                 &  * $(A_{7})''$ & 28 &  &$D_4+A_1$ \\ \hline

\weightDynkinEeight{.18}{1}{0}{1}{0}{1}{0}{2}{0} & * $E_{6}(a_{1})+A_{1}$ & 26 & $S_2$& $A_4+A_1$ \\ 
                 &  $A_{7} + A_{1}$ & 29 &  &$A_3+A_2+A_1$ \\ \hline

\weightDynkinEeight{.18}{0}{0}{2}{0}{0}{0}{2}{0} & * $E_{8}(b_{6})$  & 28 & $S_2$& $D_4(a_1) +A_2$ \\ 
                 &  $E_{6}(a_{1})+A_{2}$ & 28 & & $D_4(a_1) +A_2$ \\ 
                 &  * $D_{8}(a_{3})$ & 29 & & $A_3+A_2+A_1$ \\ \hline

\weightDynkinEeight{.18}{2}{0}{1}{0}{1}{0}{2}{0} & * $E_{7}(a_{3})$ & 30 & $S_2$& $A_4$ \\ 
                 &  $A_{1}+D_{6}$ & 31 & & $A_3+A_2$ \\ \hline

\weightDynkinEeight{.18}{2}{0}{0}{2}{0}{0}{2}{0} & * $D_{7}(a_{1})$ & 31 &  $1$ & $A_3+A_2$ \\ 
                 &  $E_{7}(a_{3})+A_{1}$ & 31 & & $A_3+A_2$ \\ \hline

\weightDynkinEeight{.18}{0}{0}{2}{0}{0}{2}{0}{0} & * $E_{8}(a_{6})$  & 32 &  $S_3$ & $D_4(a_1)+A_1$ \\ 
                 &  * $A_{8}$ & 36 & &$2A_2+2A_1$ \\ 
                 &  * $D_{8}(a_{2})$ & 34 & & $A_3+2A_1$ \\ \hline

\weightDynkinEeight{.18}{0}{0}{2}{0}{0}{2}{2}{0} & * $E_{8}(b_{5})$  & 37 &  $S_3$&$D_4(a_1)$ \\ 
                 &  * $E_{6}+A_{2}$ & 39 & & $2A_2+A_1$\\ 
                 &  * $E_{7}(a_{2})+A_{1}$ & 38 &  & $A_3+A_1$ \\ \hline

\weightDynkinEeight{.18}{2}{0}{2}{0}{0}{2}{0}{0} & * $E_{8}(a_{5})$ & 42 & $S_2$& $2A_2$ \\ 
                 &  * $D_{8}(a_{1})$ & 43 &  & $A_2+3A_1$ \\ \hline

\weightDynkinEeight{.18}{2}{0}{2}{0}{0}{2}{2}{0} & * $E_{8}(b_{4})$ & 47 &  $1$ & $A_2+2A_1$ \\ 
                 &  $E_{7}(a_{1}) + A_{1}$ & 47 &  & $A_2+2A_1$ \\ \hline

\weightDynkinEeight{.18}{2}{0}{2}{0}{2}{0}{2}{0} & * $E_{8}(a_{4})$ & 52 &  $S_2$& $A_2+A_1$ \\ 
                 &  * $D_{8}$ & 56 &&  $4A_1$ \\ \hline

\weightDynkinEeight{.18}{2}{0}{2}{0}{2}{2}{2}{0} & * $E_{8}(a_{3})$ & 63 &  $S_2$& $A_2$  \\ 
                 &  * $E_{7}+A_{1}$ & 64 & & $3A_1$ \\ \hline
"""


G2_ls_dual_data = r"""
Name   Diagram Dim Spec Dual   A:Abar
0      00      0   1    G2     1
A1     01      6   0    G2(a1) 1
A1s    10      8   0    G2(a1) 1
G2(a1) 02      10  1    G2(a1) S3
G2     22      12  1    0      1"""

F4_ls_dual_data = r"""
Name   Diagram Dim Spec Dual   A:Abar
0      0000    0   1    F4     1
A1     1000    16  0    F4(a1) 1
A1s    0001    22  1    F4(a1) S2
A1+A1s 0100    28  1    F4(a2) 1
A2     2000    30  1    B3     S2:1
A2s    0002    30  1    C3     1
A2+A1s 0010    34  0    F4(a3) 1
B2     2001    36  0    F4(a3) S2
A2s+A1 0101    36  0    F4(a3) 1
C3(a1) 1010    38  0    F4(a3) S2
F4(a3) 0200    40  1    F4(a3) S4
C3     1012    42  1    A2s    1
B3     2200    42  1    A2     1
F4(a2) 0202    44  1    A1+A1s S2:1
F4(a1) 2202    46  1    A1s    S2
F4     2222    48  1    0      1"""

E6_ls_dual_data = r"""
Name   Diagram Dim Spec Dual   A:Abar
0      000000  0   1    E6     1
A1     010000  22  1    E6(a1) 1
2A1    100001  32  1    D5     1
3A1    000100  40  0    E6(a3) 1
A2     020000  42  1    E6(a3) S2
A2+A1  110001  46  1    D5(a1) 1
2A2    200002  48  1    D4     Z3
A2+2A1 001010  50  1    A4+A1  1
A3     120001  52  1    A4     1
2A2+A1 100101  54  0    D4(a1) Z3
A3+A1  011010  56  0    D4(a1) 1
D4(a1) 000200  58  1    D4(a1) S3
A4     220002  60  1    A3     1
D4     020200  60  1    2A2    1
A4+A1  111011  62  1    A2+2A1 1
A5     211012  64  0    A2     Z3
D5(a1) 121011  64  1    A2+A1  1
E6(a3) 200202  66  1    A2     S2Z3
D5     220202  68  1    2A1    1
E6(a1) 222022  70  1    A1     Z3
E6     222222  72  1    0      Z3"""

E7_ls_dual_data = r"""
Name      Diagram Dim Spec Dual      A:Abar
0         0000000 0   1    E7        1
A1        1000000 34  1    E7(a1)    1
2A1       0000010 52  1    E7(a2)    1
(3A1)"    0000002 54  1    E6        Z2
(3A1)'    0010000 64  0    E7(a3)    1
A2        2000000 66  1    E7(a3)    S2
4A1       0100001 70  0    E6(a1)    1
A2+A1     1000010 76  1    E6(a1)    S2
A2+2A1    0001000 82  1    E7(a4)    1
A3        2000010 84  1    D6(a1)    1
2A2       0000020 84  1    D5+A1     1
A2+3A1    0200000 84  1    A6        Z2
(A3+A1)"  2000002 86  1    D5        Z2
2A2+A1    0010010 90  0    E7(a5)    1
(A3+A1)'  1001000 92  0    E7(a5)    1
D4(a1)    0020000 94  1    E7(a5)    S3
A3+2A1    1000101 94  0    E6(a3)    Z2
D4        2020000 96  1    (A5)"     1
D4(a1)+A1 0110001 96  1    E6(a3)    S2Z2
A3+A2     0001010 98  1    D5(a1)+A1 S2:1
A4        2000020 100 1    D5(a1)    S2
A3+A2+A1  0000200 100 1    A4+A2     Z2
(A5)"     2000022 102 1    D4        1
D4+A1     2110001 102 0    A4        Z2
A4+A1     1001010 104 1    A4+A1     S2
D5(a1)    2001010 106 1    A4        S2
A4+A2     0002000 106 1    A3+A2+A1  1
(A5)'     1001020 108 0    D4(a1)+A1 1
A5+A1     1001012 108 0    D4(a1)    Z2
D5(a1)+A1 2000200 108 1    A3+A2     Z2
D6(a2)    0110102 110 0    D4(a1)    Z2
E6(a3)    0020020 110 1    D4(a1)+A1 S2
D5        2020020 112 1    (A3+A1)"  1
E7(a5)    0002002 112 1    D4(a1)    S3Z2
A6        0002020 114 1    A2+3A1    1
D5+A1     2110110 114 1    2A2       Z2
D6(a1)    2110102 114 1    A3        Z2
E7(a4)    2002002 116 1    A2+2A1    S2Z2:Z2
D6        2110122 118 0    A2        Z2
E6(a1)    2002020 118 1    A2+A1     S2
E6        2022020 120 1    (3A1)"    1
E7(a3)    2002022 120 1    A2        S2Z2
E7(a2)    2220202 122 1    2A1       Z2
E7(a1)    2220222 124 1    A1        Z2
E7        2222222 126 1    0         Z2"""

E8_ls_dual_data = r"""
Name      Diagram  Dim Spec Dual      A:Abar
0         00000000 0   1    E8        1
A1        00000001 58  1    E8(a1)    1
2A1       10000000 92  1    E8(a2)    1
3A1       00000010 112 0    E8(a3)    1
A2        00000002 114 1    E8(a3)    S2
4A1       01000000 128 0    E8(a4)    1
A2+A1     10000001 136 1    E8(a4)    S2
A2+2A1    00000100 146 1    E8(b4)    1
A3        10000002 148 1    E7(a1)    1
A2+3A1    00100000 154 0    E8(a5)    1
2A2       20000000 156 1    E8(a5)    S2
2A2+A1    10000010 162 0    E8(b5)    1
A3+A1     00000101 164 0    E8(b5)    1
D4(a1)    00000020 166 1    E8(b5)    S3
D4        00000022 168 1    E6        1
2A2+2A1   00001000 168 0    E8(a6)    1
A3+2A1    00100001 172 0    E8(a6)    1
D4(a1)+A1 01000010 176 1    E8(a6)    S3
A3+A2     10000100 178 1    D7(a1)    S2:1
A4        20000002 180 1    E7(a3)    S2
A3+A2+A1  00010000 182 0    E8(b6)    1
D4+A1     01000012 184 0    E6(a1)    1
D4(a1)+A2 02000000 184 1    E8(b6)    S2
A4+A1     10000101 188 1    E6(a1)+A1 S2
2A3       10001000 188 0    D7(a2)    1
D5(a1)    10000102 190 1    E6(a1)    S2
A4+2A1    00010001 192 1    D7(a2)    S2
A4+A2     00000200 194 1    D5+A2     1
A5        20000101 196 0    D6(a1)    1
D5(a1)+A1 00010002 196 1    E7(a4)    1
A4+A2+A1  00100100 196 1    A6+A1     1
D4+A2     02000002 198 1    A6        S2:1
E6(a3)    20000020 198 1    D6(a1)    S2
D5        20000022 200 1    D5        1
A4+A3     00010010 200 0    E8(a7)    1
A5+A1     10010001 202 0    E8(a7)    1
D5(a1)+A2 00100101 202 0    E8(a7)    1
D6(a2)    01100010 204 0    E8(a7)    S2
E6(a3)+A1 10001010 204 0    E8(a7)    S2
E7(a5)    00010100 206 0    E8(a7)    S3
D5+A1     10001012 208 0    E6(a3)    1
E8(a7)    00002000 208 1    E8(a7)    S5
A6        20000200 210 1    D4+A2     1
D6(a1)    01100012 210 1    E6(a3)    S2
A6+A1     10010100 212 1    A4+A2+A1  1
E7(a4)    00010102 212 1    D5(a1)+A1 S2:1
E6(a1)    20000202 214 1    D5(a1)    S2
D5+A2     00002002 214 1    A4+A2     S2:1
D6        21100012 216 0    A4        1
E6        20000222 216 1    D4        1
D7(a2)    10010101 216 1    A4+2A1    S2
A7        10010110 218 0    D4(a1)+A2 1
E6(a1)+A1 10010102 218 1    A4+A1     S2
E7(a3)    20010102 220 1    A4        S2
E8(b6)    00020002 220 1    D4(a1)+A2 S3:S2
D7(a1)    20002002 222 1    A3+A2     S2:1
E6+A1     10010122 222 0    D4(a1)    1
E7(a2)    01101022 224 0    D4(a1)    1
E8(a6)    00020020 224 1    D4(a1)+A1 S3
D7        21101101 226 0    2A2       1
E8(b5)    00020022 226 1    D4(a1)    S3
E7(a1)    21101022 228 1    A3        1
E8(a5)    20020020 228 1    2A2       S2
E8(b4)    20020022 230 1    A2+2A1    S2:1
E7        21101222 232 0    A2        1
E8(a4)    20020202 232 1    A2+A1     S2
E8(a3)    20020222 234 1    A2        S2
E8(a2)    22202022 236 1    2A1       1
E8(a1)    22202222 238 1    A1        1
E8        22222222 240 1    0         1"""



root = Path(__file__).parent.parent / "data"

def notation_data_to_list(data: str, name: str, root: Path) -> None:
    l = []
    for s in data.split("\n"):
        if not s:
            continue
        s = s.replace("$", "").replace("\\\\", "").replace("hline", "").replace("''", "\"").split("& ")
        sommers = from_orbit_string(s[1]).__str__()
        l.append({"alvis": s[0], "sommers": sommers})
    data_path = root / "notation"
    data_path.mkdir(exist_ok=True)
    with open(data_path / f"{name}.json", "w") as f:
        json.dump(l, f, indent=4)

def sommers_dual_data_to_list(data: str, name: str, root: Path) -> None:
    # def process_diagram(s: str) -> str:
    #     print(s)
        
    #     s.replace("{", "").replace("}", "")
    #     return s
    
    l = []
    count = 0
    current_diagram = None
    for s in data.strip().replace("\\hline", "").split("\\\\"):
        s = re.sub(r"\\weightDynkin\w+\{\.\d+\}", "", s)
        s = s.replace("{", "").replace("}", "").replace("$", "").replace("*", "").replace(" ", "").replace("''", "\"")
        # Handling tilde things
        s = s.replace("Tilde", "tilde")
        s = re.sub(r"\\tilde([A-Z])_(\d)", r"\\tilde{\1}_\2", s)
        s = s.split("&")
        s = [si.strip() for si in s]
        
        if len(s) == 5:
            count += 1
            if s[0].strip() != "":
                current_diagram = s[0].strip()
            assert current_diagram is not None
            diagram = current_diagram
            print(diagram)
            orbit = from_orbit_string(s[1]).__str__()
            AAbar = s[3].strip()
            tildeb = s[2].strip()
            dual = from_orbit_string(s[4]).__str__()
            l.append({
                "orbit": orbit,
                "diagram": diagram,
                "tildeb": tildeb,
                "dual": dual,
                "A:Abar": AAbar
                })
    assert count == len(data.strip().split("\\\\"))-1, len(data.strip().split("\\\\"))
    data_path = root / "sommers_dual"
    data_path.mkdir(exist_ok=True)
    with open(data_path / f"{name}.json", "w") as f:
        json.dump(l, f, indent=4)

def ls_dual_data_to_list(data: str, name: str, root: Path) -> None:
    def preprocess_orbit_string(s: str) -> str:
        s = re.sub(r"([A-Z,a-z])(\d)", r"\1_\2", s)
        s = re.sub(r"([A-Z,a-z])_(\d)s", r"\\tilde{\1}_\2", s)
        return s
    l = []
    count = 0
    for s in data.strip().split("\n"):
        if not s or s.startswith("Name"):
            continue
        s = s.replace("''", "\"").split()
        if len(s) == 6:
            count += 1
            # print(count)
            # print(preprocess_orbit_string(s[0]), preprocess_orbit_string(s[4]))
            orbit = from_orbit_string(preprocess_orbit_string(s[0])).__str__()
            dual = from_orbit_string(preprocess_orbit_string(s[4])).__str__()
            l.append({
                "orbit": orbit,
                "diagram": s[1],
                "dim": s[2],
                "spec": s[3],
                "dual": dual,
                "A:Abar": s[5]})
    assert count == len(data.strip().split("\n"))-1, len(data.strip().split("\n"))
        
    data_path = root / "ls_dual"
    data_path.mkdir(exist_ok=True)
    with open(data_path / f"{name}.json", "w") as f:
        json.dump(l, f, indent=4)

# notation_data_to_list(G2_data, "G2", root)
# notation_data_to_list(F4_data, "F4", root)
# notation_data_to_list(E6_data, "E6", root)
# notation_data_to_list(E7_data, "E7", root)
# notation_data_to_list(E8_data, "E8", root)


sommers_dual_data_to_list(G2_sommers_dual_data, "G2", root)
sommers_dual_data_to_list(F4_sommers_dual_data, "F4", root)
sommers_dual_data_to_list(E6_sommers_dual_data, "E6", root)
sommers_dual_data_to_list(E7_sommers_dual_data, "E7", root)
sommers_dual_data_to_list(E8_sommers_dual_data, "E8", root)

# ls_dual_data_to_list(G2_ls_dual_data, "G2", root)
# ls_dual_data_to_list(F4_ls_dual_data, "F4", root)
# ls_dual_data_to_list(E6_ls_dual_data, "E6", root)
# ls_dual_data_to_list(E7_ls_dual_data, "E7", root)
# ls_dual_data_to_list(E8_ls_dual_data, "E8", root)