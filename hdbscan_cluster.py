"""HDBSCAN density-based clustering on a fused similarity matrix, parameters tuned by DBCV."""

import numpy as np
import hdbscan
from hdbscan.validity import validity_index


def run_hdbscan(fused_matrix: np.ndarray, submission_paths: list[str], min_cluster_size: int = 2, min_samples: int = None, cluster_selection_epsilon: float = 0.0) -> dict[str, int]:
    distance_matrix = 1.0 - fused_matrix
    np.fill_diagonal(distance_matrix, 0.0)

    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon
    )
    labels = clusterer.fit_predict(distance_matrix)

    return {path: int(label) for path, label in zip(submission_paths, labels)}


def tune_params(fused_matrix: np.ndarray, submission_paths: list[str], min_cluster_sizes: list[int] = None, min_samples_list: list[int] = None, epsilons: list[float] = None) -> dict:
    """Grid-sweeps min_cluster_size x min_samples x cluster_selection_epsilon, scored by DBCV."""
    if min_cluster_sizes is None:
        min_cluster_sizes = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
    if min_samples_list is None:
        min_samples_list = [None, 1, 2, 3, 4, 5]
    if epsilons is None:
        epsilons = [0.0, 0.02, 0.05, 0.1]

    distance_matrix = 1.0 - fused_matrix
    np.fill_diagonal(distance_matrix, 0.0)
    d = 2  # DBCV's ambient dimension exponent, fixed constant -- not n_submissions

    sweep = []
    best = None
    for mcs in min_cluster_sizes:
        for ms in min_samples_list:
            for eps in epsilons:
                clusters = run_hdbscan(fused_matrix, submission_paths, min_cluster_size=mcs, min_samples=ms, cluster_selection_epsilon=eps)
                labels = np.array([clusters[p] for p in submission_paths])
                n_clusters = len(set(labels.tolist()) - {-1})
                n_noise = int((labels == -1).sum())
                entry = {"min_cluster_size": mcs, "min_samples": ms, "cluster_selection_epsilon": eps, "dbcv": None, "n_clusters": n_clusters, "n_noise": n_noise}
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

    parser = argparse.ArgumentParser(description="Run HDBSCAN clustering (or DBCV parameter sweeping)")
    parser.add_argument("--size", type=int, default=45)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.0)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    dummy_raw = np.random.rand(args.size, args.size)
    dummy_matrix = np.clip((dummy_raw + dummy_raw.T) / 2, 0, 1)
    np.fill_diagonal(dummy_matrix, 1.0)
    dummy_paths = [f"submission_{i}" for i in range(args.size)]

    if args.tune:
        result = tune_params(dummy_matrix, dummy_paths)
        print(json.dumps(result, indent=4))
    else:
        result = run_hdbscan(dummy_matrix, dummy_paths, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples, cluster_selection_epsilon=args.cluster_selection_epsilon)
        n_noise = sum(1 for v in result.values() if v == -1)
        print(f"Found {len(set(result.values())) - (1 if n_noise else 0)} clusters and {n_noise} noise points across {args.size} submissions")
