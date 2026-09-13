"""
Runs HDBSCAN density-based clustering on a fused similarity matrix to group submissions into collusion clusters, leaving non-colluding submissions unlabeled as noise, with min_cluster_size/min_samples/cluster_selection_epsilon picked by sweeping DBCV instead of ground truth.
"""

import numpy as np
import hdbscan
from hdbscan.validity import validity_index

def run_hdbscan(fused_matrix: np.ndarray, submission_paths: list[str], min_cluster_size: int = 2, min_samples: int = None, cluster_selection_epsilon: float = 0.0) -> dict[str, int]:

    # converting: HDBSCAN clusters on distance, not similarity, so invert the fused similarity matrix and zero its self-distance diagonal
    distance_matrix = 1.0 - fused_matrix
    np.fill_diagonal(distance_matrix, 0.0)

    # clustering: run HDBSCAN over the precomputed distance matrix, letting cluster density adapt locally instead of applying one global threshold
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon
    )
    labels = clusterer.fit_predict(distance_matrix)

    # mapping: attach the raw absolute paths to their cluster labels, keeping HDBSCAN's -1 "noise" label for submissions that aren't part of any cluster
    clusters = {path: int(label) for path, label in zip(submission_paths, labels)}

    return clusters

def tune_params(fused_matrix: np.ndarray, submission_paths: list[str], min_cluster_sizes: list[int] = None, min_samples_list: list[int] = None) -> dict:
    """
    Grid-sweeps min_cluster_size x min_samples and scores each candidate with DBCV (Moulavi et al., SDM
    2014) -- HDBSCAN's own authors' recommended validity index, already shipped as
    hdbscan.validity.validity_index -- instead of ARI/NMI against ground truth, so this stays runnable on
    unlabeled production data. DBCV's `d` parameter (ambient feature dimension) has no natural meaning for
    a fused similarity network rather than raw coordinates; it's fixed at a constant here so it rescales
    every candidate's score identically and doesn't affect which candidate wins, only the score's magnitude.
    """
    if min_cluster_sizes is None:
        min_cluster_sizes = [2, 3, 4, 5, 7, 10]
    if min_samples_list is None:
        min_samples_list = [None, 1, 2, 3]

    distance_matrix = 1.0 - fused_matrix
    np.fill_diagonal(distance_matrix, 0.0)
    d = distance_matrix.shape[0]

    # scoring: DBCV is undefined for a single cluster or an all-noise labeling, so those candidates are skipped rather than assigned a misleading score
    sweep = []
    best = None
    for mcs in min_cluster_sizes:
        for ms in min_samples_list:
            clusters = run_hdbscan(fused_matrix, submission_paths, min_cluster_size=mcs, min_samples=ms)
            labels = np.array([clusters[p] for p in submission_paths])
            n_clusters = len(set(labels.tolist()) - {-1})
            n_noise = int((labels == -1).sum())
            entry = {"min_cluster_size": mcs, "min_samples": ms, "dbcv": None, "n_clusters": n_clusters, "n_noise": n_noise}
            if n_clusters < 2:
                sweep.append(entry)
                continue
            try:
                entry["dbcv"] = float(validity_index(distance_matrix, labels, metric="precomputed", d=d))
            except (ValueError, ZeroDivisionError, AssertionError):
                pass
            sweep.append(entry)
            if entry["dbcv"] is not None and (best is None or entry["dbcv"] > best["dbcv"]):
                best = entry

    return {"best": best, "sweep": sweep}

if __name__ == '__main__':
    import argparse
    import json

    # testing: cluster a dummy random fused matrix to independently verify the module's execution
    parser = argparse.ArgumentParser(description="Run HDBSCAN clustering (or DBCV parameter sweeping) on a fused similarity matrix")
    parser.add_argument("--size", type=int, default=45, help="Number of dummy submissions to simulate")
    parser.add_argument("--min-cluster-size", type=int, default=2, help="Smallest group size considered a real cluster")
    parser.add_argument("--min-samples", type=int, default=None, help="Conservativeness of noise assignment, defaults to min_cluster_size")
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.0, help="Distance threshold below which nearby clusters are merged")
    parser.add_argument("--tune", action="store_true", help="Sweep min_cluster_size/min_samples by DBCV instead of clustering once")
    args = parser.parse_args()

    dummy_raw = np.random.rand(args.size, args.size)
    dummy_matrix = np.clip((dummy_raw + dummy_raw.T) / 2, 0, 1)  # symmetric, like a real cosine/SNF/GST-derived similarity matrix
    np.fill_diagonal(dummy_matrix, 1.0)
    dummy_paths = [f"submission_{i}" for i in range(args.size)]

    if args.tune:
        result = tune_params(dummy_matrix, dummy_paths)
        print(json.dumps(result, indent=4))
    else:
        result = run_hdbscan(dummy_matrix, dummy_paths, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples, cluster_selection_epsilon=args.cluster_selection_epsilon)
        n_noise = sum(1 for v in result.values() if v == -1)
        print(f"Found {len(set(result.values())) - (1 if n_noise else 0)} clusters and {n_noise} noise points across {args.size} submissions")
