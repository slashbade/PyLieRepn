import json
import numpy as np

def get_permutation_from_sommers(typ, rank) -> list[int]:
    match (typ, rank):
        case ('E', 6):
            return [0, 5, 1, 2, 3, 4]
        case ('E', 7):
            return [0, 6, 1, 2, 3, 4, 5]
        case ('E', 8):
            return [0, 7, 1, 2, 3, 4, 5, 6]
        case _:
            return list(range(rank))

def permute_index_list(l: list, perm: list[int]) -> list:
    return [l[j] for j in perm]

def permute_index(arr: np.ndarray, perm: list[int]) -> np.ndarray:
    new_arr = np.zeros_like(arr)
    for i, j in enumerate(perm):
        new_arr[i] = arr[j]
    return new_arr

def parse_diagram_string(dstr: str) -> list[int]:
    return list(map(eval, list(dstr)))

def change_sommers_diagram():
    change_list = [('E', 6), ('E', 7), ('E', 8)]
    for typ, rank in change_list:
        with open(f'LieToolbox/Repn/data/sommers_dual/{typ}{rank}.json') as fp:
            data = json.load(fp)
        perm = get_permutation_from_sommers(typ, rank)
        for d in data:
            d['diagram'] = permute_index_list(parse_diagram_string(d['diagram']), perm)
        print(data[:10])
        with open(f'LieToolbox/Repn/data/sommers_dual/{typ}{rank}.json', 'w') as fp:
            json.dump(data, fp, indent=4)

def change_sommers_diagram_F4_G2():
    change_list = [('F', 4), ('G', 2)]
    for typ, rank in change_list:
        with open(f'LieToolbox/Repn/data/sommers_dual/{typ}{rank}.json') as fp:
            data = json.load(fp)
        for d in data:
            d['diagram'] = parse_diagram_string(d['diagram'])
        print(data[:10])
        with open(f'LieToolbox/Repn/data/sommers_dual/{typ}{rank}.json', 'w') as fp:
            json.dump(data, fp, indent=4)

def change_ls_diagram():
    change_list = [('E', 6), ('E', 7), ('E', 8), ('F', 4), ('G', 2)]
    for typ, rank in change_list:
        with open(f'LieToolbox/Repn/data/ls_dual/{typ}{rank}.json') as fp:
            data = json.load(fp)
        perm = get_permutation_from_sommers(typ, rank)
        for d in data:
            d['diagram'] = permute_index_list(parse_diagram_string(d['diagram']), perm)
        print(data[:10])
        with open(f'LieToolbox/Repn/data/ls_dual/{typ}{rank}.json', 'w') as fp:
            json.dump(data, fp, indent=4)
 
if __name__ == '__main__':
    change_sommers_diagram_F4_G2()
    change_ls_diagram()
    
