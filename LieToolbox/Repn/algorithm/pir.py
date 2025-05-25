import numpy as np

from .number import Number
# from ..root_system_data import cartan_matrix_pycox

def act_on_weight(cmat: np.ndarray, root_index: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Compute the result of the action of the simple root indexed by root_index on the weight.

    Args:
        cmat (np.ndarray): cartan matrix (usually in PyCox order)
        root_index (np.ndarray): index of the simple root.
        weight (np.ndarray): weight to act on, represented in the fundamental weight basis.

    Returns:
        np.ndarray: the new weight, represented in the fundamental weight basis.
    """
    # cmat = cartan_matrix_pycox(typ, rank)
    return weight - weight[root_index] * cmat[root_index]

def antidominant(cmat: np.ndarray, weight_: np.ndarray, weyl: list = []) -> tuple[list, np.ndarray]:
    """A fast recursive algorithm to compute the antidominant weight of a given weight.

    Args:
        cmat (np.ndarray): cartan matrix (usually in PyCox order)
        weight (np.ndarray): weight to compute the antidominant weight,
        represented in the fundamental weight basis.
        weyl (list, optional): weyl group element. Defaults to [].

    Returns:
        np.ndarray: the antidominant weight, represented in the fundamental weight basis.
    """
    for i in range(weight_.shape[0]):
        if Number.is_zero(weight_[i]):
            weight_[i] = 0
    if np.all(weight_ <= 0):
        return weyl, weight_
    else:
        max_pos_index = np.argwhere(weight_ > 0)[-1][0]
        new_weyl = weyl + [max_pos_index]
        new_weight = act_on_weight(cmat, max_pos_index, weight_)
        # print(f"weight: {weight}, maximum positive index: {max_pos_index + 1}")
        # print(f"{max_pos_index + 1}th simple root action, new weight: {new_weight}")
        return antidominant(cmat, new_weight, new_weyl)


