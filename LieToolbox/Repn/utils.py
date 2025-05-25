import numpy as np

from .algorithm import Number

class PPUtil:
    @staticmethod
    def pretty_print_array(array: np.ndarray, symbol='\\epsilon') -> str:
        lst = []
        is_first = True
        for i in range(array.shape[0]):
            if Number.is_zero(array[i]):
                continue
            elif Number.is_one(array[i]):
                coo = f"{symbol}_{i+1}"
            elif Number.is_one(-array[i]):
                coo = f"-{symbol}_{i+1}"
            elif Number.is_integer(array[i]):
                coo = f"{int(np.round(array[i]))}{symbol}_{i+1}"
            elif Number.is_half_integer(array[i]):
                if array[i] > 0:
                    coo = f"\\frac{{{int(np.round(2 * array[i]))}}}{{2}}{symbol}_{i+1}"
                else:
                    coo = f"-\\frac{{{int(np.round(-2 * array[i]))}}}{{2}}{symbol}_{i+1}"
            else:
                coo = f"{array[i]:.2}{symbol}_{i+1}"
            
            if array[i] > 0 and not is_first:
                lst.append('+' + coo)
                # print(i)
            else:
                lst.append(coo)
            is_first = False
        if is_first:
            lst.append('0')
        return "".join(lst)
    
    @staticmethod
    def pretty_print_basis(basis: np.ndarray) -> str:
        if basis.shape[0] == 0:
            return '\\emptyset'
        return '\\{' + ', '.join([PPUtil.pretty_print_array(basis[i]) for i in range(basis.shape[0])]) + '\\}'
    
    @staticmethod
    def pretty_print_basises(basis: list[np.ndarray]) -> str:
        return ' \\times '.join([PPUtil.pretty_print_basis(b) for b in basis])
    
    @staticmethod
    def pretty_print_weight(weight: np.ndarray) -> str:
        lst = []
        for i in range(weight.shape[0]):
            if Number.is_integer(weight[i]):
                lst.append(f"{int(np.round(weight[i]))}")
            elif Number.is_half_integer(weight[i]):
                if weight[i] > 0:
                    lst.append(f"\\frac{{{int(np.round(2 * weight[i]))}}}{{2}}")
                else:
                    lst.append(f"-\\frac{{{int(np.round(-2 * weight[i]))}}}{{2}}")
            else:
                lst.append(f"{weight[i]:.4}")
        return '(' + ', '.join(lst) + ')'
    
    @staticmethod
    def pretty_print_weight_(weight: np.ndarray) -> str:
        return PPUtil.pretty_print_array(weight, '\\omega')
    
    @staticmethod
    def pretty_print_lietype(typ: str, rank: int) -> str:
        return f"{typ}_{rank}"
    
    @staticmethod
    def pretty_print_lietypes(lietypes: list[tuple[str, int]]) -> str:
        return ' \\times '.join([PPUtil.pretty_print_lietype(*lt) for lt in lietypes])
    
    @staticmethod
    def _parse_float(x):
        if Number.is_integer(x):
            return int(np.round(x))
        else:
            return np.round(x, 3)
    
    @staticmethod
    def pretty_print_matrix(matrix: np.ndarray) -> str:
        return '\\begin{pmatrix}' + '\\\\'.join(
            [' & '.join([f"{PPUtil._parse_float(matrix[i, j])}" 
                         for j in range(matrix.shape[1])]) for i in range(matrix.shape[0])]) + '\\end{pmatrix}'
    
    @staticmethod
    def pretty_print_character(character: str) -> str:
        def parse_ch(c):
            return c.replace('phi', '\\phi')
        return parse_ch(character)

