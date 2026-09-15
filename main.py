"""
Runs the full collusion-detection pipeline over one or more submission root dirs. Default (no --dataset): blind mode, producing archive.jplag with no ground truth needed. With --dataset: scores the run against that dataset's known labels instead.
"""

import argparse
import json
from pathlib import Path
from platformdirs import user_cache_dir

from pipeline import run_pipeline
from metrics import compute_metrics, compute_pairwise_metrics
from labels import generate_labels_criminalminds, generate_labels_irplag, generate_labels_progpedia19, generate_pairwise_labels_conplag
from generate_report import build_jplag_archive

CACHE_DIR = Path(user_cache_dir('cds', ensure_exists=True))


def main(root_dirs: list[str], dataset: str | None, output: str | None, labels_csv: str | None = None,
         split: str = 'all', train_pairs_csv: str | None = None, test_pairs_csv: str | None = None,
         fusion_method: str = 'snf', agglo_distance_threshold: float = 0.5, agglo_linkage: str = 'average',
         hdbscan_min_cluster_size: int = 2, hdbscan_min_samples: int | None = None,
         leiden_resolution: float = 0.1, leiden_threshold: float = 0.5, auto_tune: bool | None = None,
         cluster_method: str = 'hdbscan') -> dict | str:
    # tuning default: blind mode has no ground truth to hand-tune against, so auto-tune defaults on there; --dataset mode keeps defaulting off (fixed params, comparable across runs) unless --auto-tune is passed explicitly
    if auto_tune is None:
        auto_tune = dataset is None

    # running: execute the full fusion + HDBSCAN/Leiden/Agglomerative pipeline, which writes paths.json, S_fused.npy, and all three cluster files under CACHE_DIR
    hdbscan_clusters, leiden_clusters, agglomerative_clusters = run_pipeline(
        root_dirs, fusion_method=fusion_method, agglo_distance_threshold=agglo_distance_threshold, agglo_linkage=agglo_linkage,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size, hdbscan_min_samples=hdbscan_min_samples,
        leiden_resolution=leiden_resolution, leiden_threshold=leiden_threshold, auto_tune=auto_tune
    )
    predicted_clusters = {"hdbscan": hdbscan_clusters, "leiden": leiden_clusters, "agglomerative": agglomerative_clusters}

    # loading: reload the ordered submission paths and fused similarity matrix that run_pipeline just wrote to CACHE_DIR
    import numpy as np
    with open(CACHE_DIR / "paths.json") as f:
        submission_paths = json.load(f)
    fused_matrix = np.load(CACHE_DIR / "S_fused.npy")

    if dataset is None:
        # blind: no ground truth given, so the default outcome is a JPlag-viewer-compatible archive instead of scored metrics
        archive_path = output or str(CACHE_DIR / "archive.jplag")
        build_jplag_archive(archive_path, submission_paths, fused_matrix, predicted_clusters[cluster_method], root_dirs)
        print(f"Wrote {archive_path}")
        return archive_path

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
        elif dataset == 'progpedia19':
            ground_truth = generate_labels_progpedia19(root_dirs)
        else:
            ground_truth = generate_labels_irplag(root_dirs)

        # scoring: compare the fused matrix and all three cluster assignments against ground truth
        metrics = compute_metrics(submission_paths, ground_truth, predicted_clusters, similarity_matrices)

    print(json.dumps(metrics, indent=4))

    if output:
        with open(output, "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the collusion-detection pipeline. Defaults to blind mode (archive.jplag, no ground truth); pass --dataset to score against known labels instead")
    parser.add_argument("root_dirs", nargs='+', help="One or more root directories containing submissions")
    parser.add_argument("--dataset", choices=['criminalminds', 'irplag', 'progpedia19', 'conplag'], default=None, help="Which ground-truth convention to score against. Omit for blind mode (default): writes archive.jplag instead of metrics")
    parser.add_argument("--output", default=None, help="Optional output path: metrics JSON if --dataset is set, archive.jplag path otherwise")
    parser.add_argument("--labels-csv", default=None, help="Path to labels.csv (required if --dataset conplag)")
    parser.add_argument("--split", choices=['train', 'test', 'all'], default='all', help="For --dataset conplag: restrict to train_pairs.csv, test_pairs.csv, or all of labels.csv")
    parser.add_argument("--train-pairs-csv", default=None, help="Path to train_pairs.csv (required if --split train)")
    parser.add_argument("--test-pairs-csv", default=None, help="Path to test_pairs.csv (required if --split test)")
    parser.add_argument("--fusion-method", choices=['snf', 'noisy_or'], default='snf', help="Similarity fusion method: SNF cross-diffusion (default) or noisy-OR independent-evidence combination")
    parser.add_argument("--agglo-distance-threshold", type=float, default=0.5, help="Distance threshold for agglomerative clustering (ignored if --auto-tune is set)")
    parser.add_argument("--agglo-linkage", default="average", choices=["average", "complete", "single"], help="Linkage criterion for agglomerative clustering")
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=2, help="HDBSCAN min_cluster_size (ignored if --auto-tune is set)")
    parser.add_argument("--hdbscan-min-samples", type=int, default=None, help="HDBSCAN min_samples (ignored if --auto-tune is set)")
    parser.add_argument("--leiden-resolution", type=float, default=0.1, help="Leiden/CPM resolution_parameter (ignored if --auto-tune is set)")
    parser.add_argument("--leiden-threshold", type=float, default=0.5, help="Leiden similarity threshold below which edges are dropped")
    parser.add_argument("--auto-tune", action="store_true", default=None, help="Sweep and pick HDBSCAN/Leiden/Agglomerative params via DBCV instead of using the fixed values above. Defaults on in blind mode (no --dataset) since there's no ground truth to hand-tune against, and off in --dataset mode; pass this flag to force it on either way")
    parser.add_argument("--cluster-method", choices=['hdbscan', 'leiden', 'agglomerative'], default='hdbscan', help="Which clustering to bake into archive.jplag's cluster.json in blind mode (ignored if --dataset is set)")
    args = parser.parse_args()

    main(args.root_dirs, args.dataset, args.output, labels_csv=args.labels_csv,
         split=args.split, train_pairs_csv=args.train_pairs_csv, test_pairs_csv=args.test_pairs_csv,
         fusion_method=args.fusion_method, agglo_distance_threshold=args.agglo_distance_threshold, agglo_linkage=args.agglo_linkage,
         hdbscan_min_cluster_size=args.hdbscan_min_cluster_size, hdbscan_min_samples=args.hdbscan_min_samples,
         leiden_resolution=args.leiden_resolution, leiden_threshold=args.leiden_threshold, auto_tune=args.auto_tune,
         cluster_method=args.cluster_method)
