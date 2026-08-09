"""
Runs HDBSCAN density-based clustering on a fused similarity matrix to group submissions into collusion clusters, leaving non-colluding submissions unlabeled as noise.
"""

import numpy as np
import hdbscan

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

if __name__ == '__main__':
    import argparse

    # testing: cluster a dummy random fused matrix to independently verify the module's execution
    parser = argparse.ArgumentParser(description="Run HDBSCAN clustering on a fused similarity matrix")
    parser.add_argument("--size", type=int, default=45, help="Number of dummy submissions to simulate")
    parser.add_argument("--min-cluster-size", type=int, default=2, help="Smallest group size considered a real cluster")
    parser.add_argument("--min-samples", type=int, default=None, help="Conservativeness of noise assignment, defaults to min_cluster_size")
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.0, help="Distance threshold below which nearby clusters are merged")
    args = parser.parse_args()

    dummy_matrix = np.clip(np.random.rand(args.size, args.size), 0, 1)
    np.fill_diagonal(dummy_matrix, 1.0)
    dummy_paths = [f"submission_{i}" for i in range(args.size)]

    result = run_hdbscan(dummy_matrix, dummy_paths, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples, cluster_selection_epsilon=args.cluster_selection_epsilon)
    n_noise = sum(1 for v in result.values() if v == -1)
    print(f"Found {len(set(result.values())) - (1 if n_noise else 0)} clusters and {n_noise} noise points across {args.size} submissions")
