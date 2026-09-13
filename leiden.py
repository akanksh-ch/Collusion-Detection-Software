"""
Executes Leiden community detection on a fused similarity matrix to group submissions into distinct collusion clusters, with resolution picked by sweeping DBCV instead of ground truth (CPM's own quality() isn't comparable across different resolution values, since resolution changes the objective itself).
"""

import numpy as np
import igraph as ig
import leidenalg
from hdbscan.validity import validity_index

def run_leiden(fused_matrix: np.ndarray, submission_paths: list[str], resolution: float = 0.1, threshold: float = 0.5, seed: int = 0) -> dict[str, int]:

    # Threshold the dense similarity matrix to remove weak connections and convert it into a weighted undirected igraph network.
    adj_matrix = np.where(fused_matrix >= threshold, fused_matrix, 0)
    graph = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), mode="undirected", loops=False)

    # Assign the raw absolute paths directly to the graph vertices.
    graph.vs["name"] = submission_paths

    # Execute Leiden under CPM rather than modularity/RBConfiguration: CPM's resolution parameter has a
    # direct interpretation as expected within-community edge density and, unlike modularity, is not
    # subject to the resolution limit that merges genuinely distinct small communities on small graphs.
    partition = leidenalg.find_partition(
        graph,
        leidenalg.CPMVertexPartition,
        weights=graph.es["weight"],
        resolution_parameter=resolution,
        seed=seed
    )

    # Map the resulting community integer assignments back using the graph's internal path names.
    clusters = {v["name"]: cluster_id for v, cluster_id in zip(graph.vs, partition.membership)}

    return clusters

def tune_resolution(fused_matrix: np.ndarray, submission_paths: list[str], resolutions: list[float] = None, threshold: float = 0.5, seed: int = 0) -> dict:
    """
    Sweeps the CPM resolution_parameter and scores each candidate partition with DBCV (Moulavi et al.,
    SDM 2014) computed on the same 1-fused_matrix distance matrix used by HDBSCAN/agglomerative, so all
    three clusterers' sweeps are scored on a shared, label-free index. CPM's own quality() is deliberately
    not used here: resolution changes the CPM objective itself, so quality values at different resolutions
    aren't on a comparable scale.
    """
    if resolutions is None:
        resolutions = np.round(np.geomspace(0.01, 1.0, 12), 3).tolist()

    distance_matrix = 1.0 - fused_matrix
    np.fill_diagonal(distance_matrix, 0.0)
    d = distance_matrix.shape[0]

    # scoring: DBCV is undefined for a single cluster, so resolutions collapsing everything into one
    # community are skipped rather than assigned a misleading score. Leiden has no noise concept either,
    # so isolated points become singleton communities -- relabeled -1 for scoring only, same fix as
    # agglomerative.py's tune_threshold
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

    # testing: cluster a dummy random fused matrix to independently verify the module's execution
    parser = argparse.ArgumentParser(description="Run Leiden/CPM community detection (or DBCV resolution sweeping) on a fused similarity matrix")
    parser.add_argument("--size", type=int, default=45, help="Number of dummy submissions to simulate")
    parser.add_argument("--resolution", type=float, default=0.1, help="CPM resolution parameter")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold below which edges are dropped")
    parser.add_argument("--tune", action="store_true", help="Sweep resolution by DBCV instead of clustering once")
    args = parser.parse_args()

    dummy_raw = np.random.rand(args.size, args.size)
    dummy_matrix = np.clip((dummy_raw + dummy_raw.T) / 2, 0, 1)  # symmetric, like a real cosine/SNF/GST-derived similarity matrix
    np.fill_diagonal(dummy_matrix, 1.0)
    dummy_paths = [f"submission_{i}" for i in range(args.size)]

    if args.tune:
        result = tune_resolution(dummy_matrix, dummy_paths, threshold=args.threshold)
        print(json.dumps(result, indent=4))
    else:
        result = run_leiden(dummy_matrix, dummy_paths, resolution=args.resolution, threshold=args.threshold)
        print(f"Found {len(set(result.values()))} clusters across {args.size} submissions")
