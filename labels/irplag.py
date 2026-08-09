"""
Generates a labels.json mapping each IR-Plag submission path to an integer group ID, where each case's baseline and its plagiarized derivatives share one group and every non-plagiarized file is its own singleton.
"""

import argparse
import json
import re
from pathlib import Path

PLAGIARIZED_PATTERN = re.compile(r'^student\d+-plagiarized-L\d+\.java$')
NON_PLAGIARIZED_PATTERN = re.compile(r'^student\d+-non-plagiarized\.java$')
BASELINE_NAME = 'original-baseline.java'


def generate_labels(root_dirs: list[str]) -> dict[str, int]:
    # scanning: treat each entry in root_dirs as one case directory containing submission files
    # directly, matching how pipeline.py scans root_dirs (one level deep, no case-NN discovery)
    case_dirs = sorted(
        (Path(root) for root in root_dirs if Path(root).is_dir() and Path(root).name != ".DS_Store"),
        key=lambda p: p.name,
    )

    # labeling: for each case, assign one shared group ID to the baseline and its plagiarized derivatives, and a fresh singleton group ID to every non-plagiarized file
    labels = {}
    next_group_id = 0
    for case_dir in case_dirs:
        case_group_id = next_group_id
        next_group_id += 1

        files = sorted(
            (child for child in case_dir.iterdir() if child.is_file() and child.name != ".DS_Store"),
            key=lambda p: p.name,
        )
        for file_path in files:
            name = file_path.name
            sub_path = str(file_path.absolute())
            if name == BASELINE_NAME or PLAGIARIZED_PATTERN.match(name):
                labels[sub_path] = case_group_id
            elif NON_PLAGIARIZED_PATTERN.match(name):
                labels[sub_path] = next_group_id
                next_group_id += 1
            else:
                raise ValueError(f"Unrecognized submission filename in {case_dir}: {name}")

    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate labels.json for IR-Plag submissions based on case baselines and plagiarism levels.")
    parser.add_argument('root_dirs', nargs='+', help="One or more case directories, each containing that case's submission files directly.")
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
