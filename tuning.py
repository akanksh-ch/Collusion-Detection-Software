"""Selects HDBSCAN/Leiden/Agglomerative hyperparameters via DBCV sweeps, no ground truth required."""

import numpy as np

import leiden
import hdbscan_cluster
import agglomerative

def select_cluster_params(fused_matrix: np.ndarray, submission_paths: list[str], leiden_threshold: float = 0.5, agglo_linkage: str = 'average', sweep_agglo_linkage: bool = True) -> dict:
    hdbscan_report = hdbscan_cluster.tune_params(fused_matrix, submission_paths)
    leiden_report = leiden.tune_resolution(fused_matrix, submission_paths, threshold=leiden_threshold)
    agglo_linkages = ["average", "complete", "single"] if sweep_agglo_linkage else [agglo_linkage]
    agglo_thresholds = np.round(np.arange(0.05, 0.96, 0.02), 2).tolist()
    agglo_report = agglomerative.tune_threshold(fused_matrix, submission_paths, thresholds=agglo_thresholds, linkage=agglo_linkage, linkages=agglo_linkages)

    params = {
        "hdbscan_min_cluster_size": hdbscan_report["best"]["min_cluster_size"] if hdbscan_report["best"] else 2,
        "hdbscan_min_samples": hdbscan_report["best"]["min_samples"] if hdbscan_report["best"] else None,
        "hdbscan_cluster_selection_epsilon": hdbscan_report["best"]["cluster_selection_epsilon"] if hdbscan_report["best"] else 0.0,
        "leiden_resolution": leiden_report["best"]["resolution"] if leiden_report["best"] else 0.1,
        "agglo_distance_threshold": agglo_report["best"]["distance_threshold"] if agglo_report["best"] else 0.5,
        "agglo_linkage": agglo_report["best"]["linkage"] if agglo_report["best"] else agglo_linkage,
    }
    report = {"hdbscan": hdbscan_report, "leiden": leiden_report, "agglomerative": agglo_report}

    return {"params": params, "report": report}

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Sweep HDBSCAN/Leiden/Agglomerative params by DBCV on a fused similarity matrix")
    parser.add_argument("--size", type=int, default=45)
    args = parser.parse_args()

    dummy_raw = np.random.rand(args.size, args.size)
    dummy_matrix = np.clip((dummy_raw + dummy_raw.T) / 2, 0, 1)
    np.fill_diagonal(dummy_matrix, 1.0)
    dummy_paths = [f"submission_{i}" for i in range(args.size)]

    result = select_cluster_params(dummy_matrix, dummy_paths)
    print(json.dumps(result["params"], indent=4))
