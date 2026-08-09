"""
Generates a labels.json mapping each CriminalMinds submission path to an integer group ID shared by all submissions derived from the same original (oN) solution.
"""

import argparse
import json
import re
from pathlib import Path

ORIGIN_PATTERN = re.compile(r'^o(\d+)(?:-|$)')


def generate_labels(root_dirs: list[str]) -> dict[str, int]:
    # scanning: walk every root dir the same way pipeline.py does (sorted, absolute paths, skip .DS_Store)
    submission_paths = []
    for root in root_dirs:
        root_path = Path(root)
        if root_path.is_dir():
            for child in root_path.iterdir():
                if child.name != ".DS_Store":
                    submission_paths.append(str(child.absolute()))
    submission_paths.sort(key=lambda p: Path(p).name)

    # parsing: extract the delimiter-bounded oN token from each submission's name, guarding against o1/o10 substring collisions
    origin_by_path = {}
    for sub_path in submission_paths:
        name = Path(sub_path).name
        match = ORIGIN_PATTERN.match(name)
        if not match:
            raise ValueError(f"Could not parse origin token from submission name: {name}")
        origin_by_path[sub_path] = int(match.group(1))

    # grouping: assign a stable integer group ID to each distinct oN value, in first-seen order
    group_id_by_origin = {}
    next_group_id = 0
    for sub_path in submission_paths:
        origin = origin_by_path[sub_path]
        if origin not in group_id_by_origin:
            group_id_by_origin[origin] = next_group_id
            next_group_id += 1

    # labeling: build the final path -> group_id mapping
    labels = {sub_path: group_id_by_origin[origin_by_path[sub_path]] for sub_path in submission_paths}

    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate labels.json for CriminalMinds submissions based on oN origin prefixes.")
    parser.add_argument('root_dirs', nargs='+', help="One or more root directories containing submissions.")
    parser.add_argument('--output', default='labels.json', help="Path to write the resulting labels.json (default: labels.json).")
    args = parser.parse_args()

    labels = generate_labels(args.root_dirs)

    with open(args.output, 'w') as f:
        json.dump(labels, f, indent=2)

    group_counts = {}
    for group_id in labels.values():
        group_counts[group_id] = group_counts.get(group_id, 0) + 1
    num_singletons = sum(1 for count in group_counts.values() if count == 1)

    print(f"Labeled {len(labels)} submissions.")
    print(f"Distinct groups: {len(group_counts)}")
    print(f"Singleton groups: {num_singletons}")
