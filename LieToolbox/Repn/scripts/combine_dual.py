import json

combine_list =[('E', 6), ('E', 7), ('E', 8), ('F', 4), ('G', 2)]

def combine_dict(duals1, duals2):
    global_orbit = set()
    combined = []
    for duals in [duals1, duals2]:
        for d in duals:
            if d["orbit"] not in global_orbit:
                global_orbit.add(d["orbit"])
                combined.append(dict(
                    orbit=d["orbit"],
                    dual=d["dual"],
                    diagram=d["diagram"]
                ))
    return combined

if __name__=='__main__':
    for typ, rank in combine_list:
        with open(f"LieToolbox/Repn/data/sommers_dual/{typ}{rank}.json") as f:
            duals1 = json.load(f)
        with open(f"LieToolbox/Repn/data/ls_dual/{typ}{rank}.json") as f:
            duals2 = json.load(f)
        combined = combine_dict(duals1, duals2)
        with open(f"LieToolbox/Repn/data/dual/{typ}{rank}.json", "w") as f:
            json.dump(combined, f, indent=4)
