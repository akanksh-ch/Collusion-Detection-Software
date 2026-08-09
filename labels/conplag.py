"""
Generates a sparse list of (path_a, path_b, verdict) labeled pairs for ConPlag submissions by resolving labels.csv hash pairs against the flattened root_dirs corpus, optionally filtered to a train/test split.
"""

import argparse
import csv
from pathlib import Path


def _load_pairs_csv(path: str) -> list[tuple[str, str, int]]:
    # loading: read sub1,sub2,problem,verdict rows, tolerating an optional problem column
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row['sub1'], row['sub2'], int(row['verdict'])))
    return rows


def _load_split_ids(path: str) -> set[str]:
    # loading: read a train/test split file of sub1_sub2 pair-ID strings, one per line (optionally with a header)
    ids = set()
    with open(path) as f:
        for line in f:
            token = line.strip()
            if not token or token.lower() in ('id', 'pair', 'pair_id', 'sub1_sub2'):
                continue
            ids.add(token)
    return ids


def generate_pairwise_labels(
    root_dirs: list[str],
    labels_csv: str,
    split: str = 'all',
    train_pairs_csv: str | None = None,
    test_pairs_csv: str | None = None,
) -> list[tuple[str, str, int]]:
    # scanning: walk every root dir the same way pipeline.py does (one level deep), indexing each submission's
    # file stem (the hash) to its resolved absolute path, restricted to whatever root_dirs were actually passed in
    path_by_hash: dict[str, str] = {}
    for root in root_dirs:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for child in root_path.iterdir():
            if child.is_file() and child.name != ".DS_Store":
                path_by_hash[child.stem] = str(child.absolute())

    # splitting: optionally restrict to a train/test subset of pair IDs, formatted as sub1_sub2
    split_ids = None
    if split == 'train':
        if not train_pairs_csv:
            raise ValueError("split='train' requires train_pairs_csv")
        split_ids = _load_split_ids(train_pairs_csv)
    elif split == 'test':
        if not test_pairs_csv:
            raise ValueError("split='test' requires test_pairs_csv")
        split_ids = _load_split_ids(test_pairs_csv)
    elif split != 'all':
        raise ValueError(f"Unknown split: {split!r} (expected 'train', 'test', or 'all')")

    # filtering: keep only labels.csv rows where both hashes resolve under root_dirs, and (if a split was
    # requested) whose pair ID appears in the split file under either sub1_sub2 or sub2_sub1 ordering
    pairs: list[tuple[str, str, int]] = []
    skipped_missing = 0
    skipped_split = 0
    for sub1, sub2, verdict in _load_pairs_csv(labels_csv):
        if sub1 not in path_by_hash or sub2 not in path_by_hash:
            skipped_missing += 1
            continue
        if split_ids is not None:
            forward = f"{sub1}_{sub2}"
            backward = f"{sub2}_{sub1}"
            if forward not in split_ids and backward not in split_ids:
                skipped_split += 1
                continue
        pairs.append((path_by_hash[sub1], path_by_hash[sub2], verdict))

    if skipped_missing:
        print(f"Skipped {skipped_missing} labels.csv pairs with a hash not found under root_dirs.")
    if split_ids is not None and skipped_split:
        print(f"Skipped {skipped_split} pairs not present in the '{split}' split file.")

    return pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate sparse (path_a, path_b, verdict) labels for ConPlag submissions from labels.csv.")
    parser.add_argument('root_dirs', nargs='+', help="One or more problem directories containing flattened submission files.")
    parser.add_argument('--labels-csv', required=True, help="Path to labels.csv (columns: sub1,sub2,problem,verdict).")
    parser.add_argument('--split', choices=['train', 'test', 'all'], default='all', help="Restrict to train_pairs.csv, test_pairs.csv, or all of labels.csv (default: all).")
    parser.add_argument('--train-pairs-csv', default=None, help="Path to train_pairs.csv (required if --split train).")
    parser.add_argument('--test-pairs-csv', default=None, help="Path to test_pairs.csv (required if --split test).")
    parser.add_argument('--output', default='pairwise_labels.json', help="Path to write the resulting pairs as JSON (default: pairwise_labels.json).")
    args = parser.parse_args()

    pairs = generate_pairwise_labels(
        args.root_dirs,
        args.labels_csv,
        split=args.split,
        train_pairs_csv=args.train_pairs_csv,
        test_pairs_csv=args.test_pairs_csv,
    )

    import json
    with open(args.output, 'w') as f:
        json.dump(pairs, f, indent=2)

    num_positive = sum(1 for _, _, v in pairs if v == 1)
    print(f"Labeled {len(pairs)} pairs.")
    print(f"Positive (colluding): {num_positive}")
    print(f"Negative (non-colluding): {len(pairs) - num_positive}")
