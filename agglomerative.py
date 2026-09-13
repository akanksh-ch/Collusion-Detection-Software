"""
Runs agglomerative clustering with a distance threshold (not a fixed k) on a fused similarity matrix to group submissions into collusion clusters, with the threshold picked by sweeping DBCV (a label-free density validity index) instead of ground truth, since production runs won't have labels.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from hdbscan.validity import validity_index

def run_agglomerative(fused_matrix: np.ndarray, submission_paths: list[str], distance_threshold: float = 0.5, linkage: str = "average") -> dict[str, int]:

    # converting: agglomerative clustering merges by distance, not similarity, so invert the fused similarity matrix and zero its self-distance diagonal, matching hdbscan_cluster.py's convention
    distance_matrix = 1.0 - fused_matrix
    np.fill_diagonal(distance_matrix, 0.0)

    # clustering: merge bottom-up until the nearest remaining pair of clusters exceeds distance_threshold, letting the
    # number of clusters emerge from the data instead of being fixed in advance via n_clusters
    clusterer = AgglomerativeClustering(
        metric="precomputed",
        linkage=linkage,
        distance_threshold=distance_threshold,
        n_clusters=None
    )
    labels = clusterer.fit_predict(distance_matrix)

    # mapping: attach the raw absolute paths to their cluster labels, consistent with leiden.py/hdbscan_cluster.py's output shape
    clusters = {path: int(label) for path, label in zip(submission_paths, labels)}

    return clusters

def tune_threshold(fused_matrix: np.ndarray, submission_paths: list[str], thresholds: list[float] = None, linkage: str = "average") -> dict:
    """
    Sweeps distance_threshold and scores each candidate with DBCV (Moulavi et al., SDM 2014), a
    density-based relative validity index computed purely from the distance matrix and the resulting
    labels -- no ground truth required, so this is safe to run at deployment time on unlabeled batches,
    unlike an ARI/NMI sweep against known groups.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2).tolist()

    distance_matrix = 1.0 - fused_matrix
    np.fill_diagonal(distance_matrix, 0.0)
    d = distance_matrix.shape[0]

    # scoring: DBCV is undefined for a single cluster or an all-noise labeling (no within/between-cluster
    # density to compare), so those thresholds are skipped rather than assigned a misleading score. Unlike
    # HDBSCAN, agglomerative clustering has no noise concept, so an isolated point becomes its own
    # singleton cluster -- DBCV's internal MST needs >=2 points per cluster, so singletons are relabeled
    # -1 (noise) for scoring purposes only, the same way HDBSCAN's own noise points are excluded
    sweep = []
    best = None
    for t in thresholds:
        clusters = run_agglomerative(fused_matrix, submission_paths, distance_threshold=t, linkage=linkage)
        labels = np.array([clusters[p] for p in submission_paths])
        counts = np.bincount(labels)
        scoring_labels = np.where(counts[labels] < 2, -1, labels)
        n_clusters = len(set(scoring_labels.tolist()) - {-1})
        if n_clusters < 2:
            sweep.append({"distance_threshold": t, "dbcv": None, "n_clusters": n_clusters})
            continue
        try:
            score = float(validity_index(distance_matrix, scoring_labels, metric="precomputed", d=d))
        except (ValueError, ZeroDivisionError, AssertionError):
            score = None
        sweep.append({"distance_threshold": t, "dbcv": score, "n_clusters": n_clusters})
        if score is not None and (best is None or score > best["dbcv"]):
            best = {"distance_threshold": t, "dbcv": score, "n_clusters": n_clusters}

    return {"best": best, "sweep": sweep}

if __name__ == '__main__':
    import argparse
    import json

    # testing: cluster a dummy random fused matrix to independently verify the module's execution
    parser = argparse.ArgumentParser(description="Run agglomerative clustering (or DBCV threshold sweeping) on a fused similarity matrix")
    parser.add_argument("--size", type=int, default=45, help="Number of dummy submissions to simulate")
    parser.add_argument("--distance-threshold", type=float, default=0.5, help="Distance above which clusters are not merged")
    parser.add_argument("--linkage", default="average", choices=["average", "complete", "single"], help="Linkage criterion")
    parser.add_argument("--tune", action="store_true", help="Sweep thresholds by DBCV instead of clustering once")
    args = parser.parse_args()

    dummy_raw = np.random.rand(args.size, args.size)
    dummy_matrix = np.clip((dummy_raw + dummy_raw.T) / 2, 0, 1)  # symmetric, like a real cosine/SNF/GST-derived similarity matrix
    np.fill_diagonal(dummy_matrix, 1.0)
    dummy_paths = [f"submission_{i}" for i in range(args.size)]

    if args.tune:
        result = tune_threshold(dummy_matrix, dummy_paths, linkage=args.linkage)
        print(json.dumps(result, indent=4))
    else:
        result = run_agglomerative(dummy_matrix, dummy_paths, distance_threshold=args.distance_threshold, linkage=args.linkage)
        print(f"Found {len(set(result.values()))} clusters across {args.size} submissions")
