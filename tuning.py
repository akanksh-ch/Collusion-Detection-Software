"""
Picks HDBSCAN/Leiden/Agglomerative's hyperparameters by sweeping each clusterer's own DBCV-scored search space, so the pipeline can select parameters without any ground truth at run time.
"""

import numpy as np

import leiden
import hdbscan_cluster
import agglomerative

def select_cluster_params(fused_matrix: np.ndarray, submission_paths: list[str], leiden_threshold: float = 0.5, agglo_linkage: str = 'average') -> dict:

    # sweeping: each clusterer's own module owns its search grid and DBCV scoring (see agglomerative.py/
    # hdbscan_cluster.py/leiden.py's tune_* docstrings for why each is scored this way instead of e.g.
    # Leiden's own resolution-dependent CPM quality), this function just collects the three winners
    hdbscan_report = hdbscan_cluster.tune_params(fused_matrix, submission_paths)
    leiden_report = leiden.tune_resolution(fused_matrix, submission_paths, threshold=leiden_threshold)
    agglo_report = agglomerative.tune_threshold(fused_matrix, submission_paths, linkage=agglo_linkage)

    # defaulting: fall back to each module's own run_* default if every candidate in its sweep was
    # undefined (e.g. a degenerate fused matrix), rather than passing None through to the clusterer
    params = {
        "hdbscan_min_cluster_size": hdbscan_report["best"]["min_cluster_size"] if hdbscan_report["best"] else 2,
        "hdbscan_min_samples": hdbscan_report["best"]["min_samples"] if hdbscan_report["best"] else None,
        "leiden_resolution": leiden_report["best"]["resolution"] if leiden_report["best"] else 0.1,
        "agglo_distance_threshold": agglo_report["best"]["distance_threshold"] if agglo_report["best"] else 0.5,
    }
    report = {"hdbscan": hdbscan_report, "leiden": leiden_report, "agglomerative": agglo_report}

    return {"params": params, "report": report}

if __name__ == '__main__':
    import argparse
    import json

    # testing: sweep a dummy random fused matrix to independently verify the module's execution
    parser = argparse.ArgumentParser(description="Sweep HDBSCAN/Leiden/Agglomerative params by DBCV on a fused similarity matrix")
    parser.add_argument("--size", type=int, default=45, help="Number of dummy submissions to simulate")
    args = parser.parse_args()

    dummy_raw = np.random.rand(args.size, args.size)
    dummy_matrix = np.clip((dummy_raw + dummy_raw.T) / 2, 0, 1)  # symmetric, like a real cosine/SNF/GST-derived similarity matrix
    np.fill_diagonal(dummy_matrix, 1.0)
    dummy_paths = [f"submission_{i}" for i in range(args.size)]

    result = select_cluster_params(dummy_matrix, dummy_paths)
    print(json.dumps(result["params"], indent=4))
