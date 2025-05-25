import numpy as np

TOL = 1e-7

class Number:
    # Essential Algorithms
    @staticmethod
    def is_integer(x: float, tol: float = TOL) -> bool:
        """Justify whether a value is an integer at a precision of tol

        Args:
            x (float): a float value
            tol (float, optional): tolerance. Defaults to 1e-7.
        """
        return np.abs(x - np.round(x)) < tol
    
    @staticmethod
    def is_integer_array(xl: np.ndarray, tol: float = TOL) -> np.bool_:
        return np.all(np.abs(xl - np.round(xl)) < tol)
    
    @staticmethod
    def is_zero(x: float, tol: float = TOL) -> bool:
        return np.abs(x) < tol
    
    @staticmethod
    def is_one(x: float, tol: float = TOL) -> bool:
        return np.abs(x - 1) < tol

    @staticmethod
    def is_half_integer(x: float, tol: float = TOL) -> bool:
        return Number.is_integer(2 * x, tol)

    @staticmethod
    def round2_one(x: float) -> int | float:
        if Number.is_integer(x):
            return int(np.round(x))
        elif Number.is_half_integer(x):
            return int(np.round(2 * x)) / 2
        else:
            return x

    @staticmethod
    def round2(xl: np.ndarray) -> np.ndarray:
        return np.array([Number.round2_one(x) for x in xl])


