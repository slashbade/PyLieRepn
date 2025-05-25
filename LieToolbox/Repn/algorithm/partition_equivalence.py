from typing import Iterable, Callable, Any

def partition_equivalence(
    l: Iterable[Any], r: Callable[[Any, Any], bool]
) -> tuple[list[list[Any]], list[int]]:
    """Partition a list into equivalence classes.

    Args:
        l (Iterable[Any]): list to partition.
        r (Callable[[Any, Any], bool]): equivalence relation.

    Returns:
        Tuple[List[List[Any]], List[int]]: partitions and the indices of
        the elements in the original list.
    """
    partitions = []
    ind_partitions = []
    for index, elem in enumerate(l):
        found = False
        for i, p in zip(ind_partitions, partitions):
            if r(p[0], elem):
                p.append(elem)
                i.append(index)
                found = True
                break
        if not found:
            partitions.append([elem])
            ind_partitions.append([index])
    return partitions, ind_partitions


