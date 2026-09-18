"""Agglomerative clustering on a fused similarity matrix, distance_threshold/linkage tuned by DBCV."""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from hdbscan.validity import validity_index

def run_agglomerative(fused_matrix: np.ndarray, submission_paths: list[str], distance_threshold: float = 0.5, linkage: str = "average") -> dict[str, int]:
    distance_matrix = 1.0 - fused_matrix
    np.fill_diagonal(distance_matrix, 0.0)

    clusterer = AgglomerativeClustering(
        metric="precomputed",
        linkage=linkage,
        distance_threshold=distance_threshold,
        n_clusters=None
    )
    labels = clusterer.fit_predict(distance_matrix)

    return {path: int(label) for path, label in zip(submission_paths, labels)}

def tune_threshold(fused_matrix: np.ndarray, submission_paths: list[str], thresholds: list[float] = None, linkage: str = "average", linkages: list[str] = None) -> dict:
    """Sweeps distance_threshold (and linkage, if given), scored by DBCV."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2).tolist()
    if linkages is None:
        linkages = [linkage]

    distance_matrix = 1.0 - fused_matrix
    np.fill_diagonal(distance_matrix, 0.0)
    d = 2  # DBCV's ambient dimension exponent, fixed constant -- not n_submissions

    sweep = []
    best = None
    for lk in linkages:
        for t in thresholds:
            clusters = run_agglomerative(fused_matrix, submission_paths, distance_threshold=t, linkage=lk)
            labels = np.array([clusters[p] for p in submission_paths])
            counts = np.bincount(labels)
            scoring_labels = np.where(counts[labels] < 2, -1, labels)
            n_clusters = len(set(scoring_labels.tolist()) - {-1})
            if n_clusters < 2:
                sweep.append({"distance_threshold": t, "linkage": lk, "dbcv": None, "n_clusters": n_clusters})
                continue
            try:
                score = float(validity_index(distance_matrix, scoring_labels, metric="precomputed", d=d))
            except (ValueError, ZeroDivisionError, AssertionError):
                score = None
            sweep.append({"distance_threshold": t, "linkage": lk, "dbcv": score, "n_clusters": n_clusters})
            if score is not None and (best is None or score > best["dbcv"]):
                best = {"distance_threshold": t, "linkage": lk, "dbcv": score, "n_clusters": n_clusters}

    return {"best": best, "sweep": sweep}

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run agglomerative clustering (or DBCV threshold sweeping)")
    parser.add_argument("--size", type=int, default=45)
    parser.add_argument("--distance-threshold", type=float, default=0.5)
    parser.add_argument("--linkage", default="average", choices=["average", "complete", "single"])
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    dummy_raw = np.random.rand(args.size, args.size)
    dummy_matrix = np.clip((dummy_raw + dummy_raw.T) / 2, 0, 1)
    np.fill_diagonal(dummy_matrix, 1.0)
    dummy_paths = [f"submission_{i}" for i in range(args.size)]

    if args.tune:
        result = tune_threshold(dummy_matrix, dummy_paths, linkage=args.linkage)
        print(json.dumps(result, indent=4))
    else:
        result = run_agglomerative(dummy_matrix, dummy_paths, distance_threshold=args.distance_threshold, linkage=args.linkage)
        print(f"Found {len(set(result.values()))} clusters across {args.size} submissions")
