"""
Runs the full collusion-detection pipeline over one or more submission root dirs, builds ground-truth labels for the chosen dataset, and prints/saves the resulting metrics.
"""

import argparse
import json
from pathlib import Path
from platformdirs import user_cache_dir

from pipeline import run_pipeline
from metrics import compute_metrics, compute_pairwise_metrics
from labels import generate_labels_criminalminds, generate_labels_irplag, generate_pairwise_labels_conplag

CACHE_DIR = Path(user_cache_dir('cds', ensure_exists=True))


def main(root_dirs: list[str], dataset: str, output: str | None, labels_csv: str | None = None,
         split: str = 'all', train_pairs_csv: str | None = None, test_pairs_csv: str | None = None) -> dict:
    # running: execute the full SNF + HDBSCAN/Leiden pipeline, which writes paths.json, S_fused.npy, and both cluster files under CACHE_DIR
    hdbscan_clusters, leiden_clusters = run_pipeline(root_dirs)

    # loading: reload the ordered submission paths and fused similarity matrix that run_pipeline just wrote to CACHE_DIR
    import numpy as np
    with open(CACHE_DIR / "paths.json") as f:
        submission_paths = json.load(f)
    fused_matrix = np.load(CACHE_DIR / "S_fused.npy")
    similarity_matrices = {"fused": fused_matrix}

    if dataset == 'conplag':
        # labeling: ConPlag's labels.csv is a sparse curated subset of pairs, not an exhaustive per-submission
        # grouping, so it takes the sparse pairwise scoring path instead of compute_metrics' dense one
        if not labels_csv:
            raise ValueError("--labels-csv is required when --dataset conplag")
        pairs = generate_pairwise_labels_conplag(
            root_dirs, labels_csv, split=split,
            train_pairs_csv=train_pairs_csv, test_pairs_csv=test_pairs_csv,
        )
        metrics = compute_pairwise_metrics(pairs, similarity_matrices, submission_paths)
    else:
        # labeling: build the ground-truth path -> group_id mapping with whichever generator matches this dataset's naming convention
        if dataset == 'criminalminds':
            ground_truth = generate_labels_criminalminds(root_dirs)
        else:
            ground_truth = generate_labels_irplag(root_dirs)

        # scoring: compare the fused matrix and both cluster assignments against ground truth
        predicted_clusters = {"hdbscan": hdbscan_clusters, "leiden": leiden_clusters}
        metrics = compute_metrics(submission_paths, ground_truth, predicted_clusters, similarity_matrices)

    print(json.dumps(metrics, indent=4))

    if output:
        with open(output, "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the collusion-detection pipeline and score it against ground truth")
    parser.add_argument("root_dirs", nargs='+', help="One or more root directories containing submissions")
    parser.add_argument("--dataset", choices=['criminalminds', 'irplag', 'conplag'], required=True, help="Which ground-truth convention to use")
    parser.add_argument("--output", default=None, help="Optional path to write the resulting metrics as JSON")
    parser.add_argument("--labels-csv", default=None, help="Path to labels.csv (required if --dataset conplag)")
    parser.add_argument("--split", choices=['train', 'test', 'all'], default='all', help="For --dataset conplag: restrict to train_pairs.csv, test_pairs.csv, or all of labels.csv")
    parser.add_argument("--train-pairs-csv", default=None, help="Path to train_pairs.csv (required if --split train)")
    parser.add_argument("--test-pairs-csv", default=None, help="Path to test_pairs.csv (required if --split test)")
    args = parser.parse_args()

    main(args.root_dirs, args.dataset, args.output, labels_csv=args.labels_csv,
         split=args.split, train_pairs_csv=args.train_pairs_csv, test_pairs_csv=args.test_pairs_csv)
