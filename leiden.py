"""Leiden/CPM community detection on a fused similarity matrix, resolution tuned by DBCV."""

import numpy as np
import igraph as ig
import leidenalg
from hdbscan.validity import validity_index


def run_leiden(fused_matrix: np.ndarray, submission_paths: list[str], resolution: float = 0.1, threshold: float = 0.5, seed: int = 0) -> dict[str, int]:
    adj_matrix = np.where(fused_matrix >= threshold, fused_matrix, 0)
    graph = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), mode="undirected", loops=False)
    graph.vs["name"] = submission_paths

    partition = leidenalg.find_partition(
        graph,
        leidenalg.CPMVertexPartition,
        weights=graph.es["weight"],
        resolution_parameter=resolution,
        seed=seed
    )

    return {v["name"]: cluster_id for v, cluster_id in zip(graph.vs, partition.membership)}


def tune_resolution(fused_matrix: np.ndarray, submission_paths: list[str], resolutions: list[float] = None, threshold: float = 0.5, seed: int = 0) -> dict:
    """Sweeps CPM resolution, scored by DBCV on the shared 1-fused_matrix distance matrix."""
    if resolutions is None:
        resolutions = np.round(np.geomspace(0.01, 1.0, 12), 3).tolist()

    distance_matrix = 1.0 - fused_matrix
    np.fill_diagonal(distance_matrix, 0.0)
    d = 2  # DBCV's ambient dimension exponent, fixed constant -- not n_submissions

    sweep = []
    best = None
    for r in resolutions:
        clusters = run_leiden(fused_matrix, submission_paths, resolution=r, threshold=threshold, seed=seed)
        labels = np.array([clusters[p] for p in submission_paths])
        counts = np.bincount(labels)
        scoring_labels = np.where(counts[labels] < 2, -1, labels)
        n_clusters = len(set(scoring_labels.tolist()) - {-1})
        entry = {"resolution": r, "dbcv": None, "n_clusters": n_clusters}
        if n_clusters < 2:
            sweep.append(entry)
            continue
        try:
            entry["dbcv"] = float(validity_index(distance_matrix, scoring_labels, metric="precomputed", d=d))
        except (ValueError, ZeroDivisionError, AssertionError):
            pass
        sweep.append(entry)
        if entry["dbcv"] is not None and (best is None or entry["dbcv"] > best["dbcv"]):
            best = entry

    return {"best": best, "sweep": sweep}


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run Leiden/CPM community detection (or DBCV resolution sweeping)")
    parser.add_argument("--size", type=int, default=45)
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    dummy_raw = np.random.rand(args.size, args.size)
    dummy_matrix = np.clip((dummy_raw + dummy_raw.T) / 2, 0, 1)
    np.fill_diagonal(dummy_matrix, 1.0)
    dummy_paths = [f"submission_{i}" for i in range(args.size)]

    if args.tune:
        result = tune_resolution(dummy_matrix, dummy_paths, threshold=args.threshold)
        print(json.dumps(result, indent=4))
    else:
        result = run_leiden(dummy_matrix, dummy_paths, resolution=args.resolution, threshold=args.threshold)
        print(f"Found {len(set(result.values()))} clusters across {args.size} submissions")
