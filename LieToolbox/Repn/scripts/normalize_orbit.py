import json
import sys
sys.path.append('.')
from LieToolbox.Repn.orbit import from_orbit_string
def process_one(data: dict[str, str]) -> dict[str, str]:
    data['orbit'] = from_orbit_string(data['orbit']).__str__()
    data['dual'] = from_orbit_string(data['dual']).__str__()
    return data

def process_all(data: list[dict[str, str]]) -> list[dict[str, str]]:
    return [process_one(d) for d in data]

def main():
    with open('LieToolbox/Repn/data/ls_dual/E7.json', 'r') as f:
        data = json.load(f)
    data = process_all(data)
    with open('LieToolbox/Repn/data/ls_dual/E7.json', 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    main()