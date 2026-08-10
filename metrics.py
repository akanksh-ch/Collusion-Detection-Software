"""
Scores predicted clusters and fused/per-signal similarity matrices against known ground-truth collusion groups using ARI, NMI, and pairwise ROC/PR-AUC.
"""

import json
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
    roc_auc_score,
    average_precision_score,
)

def compute_metrics(submission_paths: list[str], ground_truth: dict[str, int], predicted_clusters: dict[str, dict[str, int]], similarity_matrices: dict[str, np.ndarray]) -> dict:

    # aligning: build the ground-truth label array in the same order as submission_paths, raising rather than
    # silently skipping if any submission is missing a label, consistent with the pipeline's no-silent-drop policy
    missing = [p for p in submission_paths if p not in ground_truth]
    if missing:
        raise ValueError(f"{len(missing)} submissions missing from ground_truth: {missing}")
    y_group = np.array([ground_truth[p] for p in submission_paths])

    # pairwise: build the upper-triangle pair mask and binary same-group labels once, shared by every similarity matrix's AUC
    n = len(submission_paths)
    iu, ju = np.triu_indices(n, k=1)
    y_true_pairs = (y_group[iu] == y_group[ju]).astype(int)

    results = {"n_submissions": n, "n_known_groups": len(set(y_group.tolist())), "similarity_auc": {}, "clustering": {}}

    # scoring similarity: for each named similarity matrix (fused or per-signal), score its pairwise ROC-AUC and PR-AUC against ground truth
    for name, matrix in similarity_matrices.items():
        y_score_pairs = matrix[iu, ju]
        results["similarity_auc"][name] = {
            "roc_auc": float(roc_auc_score(y_true_pairs, y_score_pairs)),
            "pr_auc": float(average_precision_score(y_true_pairs, y_score_pairs)),
        }

    # scoring clusters: for each named predicted clustering (e.g. hdbscan, leiden), align its labels and score against ground truth
    for name, clusters in predicted_clusters.items():
        y_pred = np.array([clusters[p] for p in submission_paths])
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_group, y_pred)
        results["clustering"][name] = {
            "ari": float(adjusted_rand_score(y_group, y_pred)),
            "nmi": float(normalized_mutual_info_score(y_group, y_pred)),
            "homogeneity": float(homogeneity),
            "completeness": float(completeness),
            "v_measure": float(v_measure),
            "n_predicted_clusters": len(set(y_pred.tolist()) - {-1}),
            "n_noise": int((y_pred == -1).sum()),
        }

    return results

def compute_pairwise_metrics(pairs: list[tuple[str, str, int]], similarity_matrices: dict[str, np.ndarray], submission_paths: list[str]) -> dict:

    # aligning: build a path -> index lookup once, raising rather than silently skipping if a labeled pair
    # references a submission not present in submission_paths, consistent with the pipeline's no-silent-drop policy
    index_by_path = {p: i for i, p in enumerate(submission_paths)}
    missing = sorted({p for a, b, _ in pairs for p in (a, b) if p not in index_by_path})
    if missing:
        raise ValueError(f"{len(missing)} labeled-pair submissions missing from submission_paths: {missing}")

    # pairwise: resolve each explicit (path_a, path_b, verdict) triple to matrix indices and a binary label array,
    # scoring ONLY these pairs rather than the full C(n,2) corpus, since labels.csv is a sparse curated subset
    ia = np.array([index_by_path[a] for a, b, v in pairs])
    ib = np.array([index_by_path[b] for a, b, v in pairs])
    y_true_pairs = np.array([v for a, b, v in pairs])

    results = {"n_pairs": len(pairs), "n_positive": int(y_true_pairs.sum()), "similarity_auc": {}}

    # scoring similarity: for each named similarity matrix (fused or per-signal), score its pairwise ROC-AUC and
    # PR-AUC against the sparse ground truth; no clustering metrics, since a sparse label set can't be scored
    # against a full partition
    for name, matrix in similarity_matrices.items():
        y_score_pairs = matrix[ia, ib]
        results["similarity_auc"][name] = {
            "roc_auc": float(roc_auc_score(y_true_pairs, y_score_pairs)),
            "pr_auc": float(average_precision_score(y_true_pairs, y_score_pairs)),
        }

    return results

if __name__ == '__main__':
    import argparse

    # loading: read the ordered path list, ground-truth labels, fused/per-signal matrices, and any number of named cluster result files
    parser = argparse.ArgumentParser(description="Score predicted clusters and similarity matrices against ground truth")
    parser.add_argument("--paths", required=True, help="Path to paths.json (ordered submission ID list)")
    parser.add_argument("--ground-truth", default=None, help="Path to labels.json (submission path -> ground-truth group id). Mutually exclusive with --conplag-labels-csv.")
    parser.add_argument("--conplag-labels-csv", default=None, help="Path to ConPlag's labels.csv (sub1,sub2,problem,verdict). Mutually exclusive with --ground-truth; scores sparse pairwise AUC only, no clustering metrics.")
    parser.add_argument("--fused", required=True, help="Path to S_fused.npy")
    parser.add_argument("--signal", nargs=2, action="append", default=[], metavar=("NAME", "PATH"), help="Extra named similarity/embedding matrix, e.g. --signal gst gst_coverage.npy")
    parser.add_argument("--clusters", nargs=2, action="append", default=[], metavar=("NAME", "PATH"), help="Named predicted cluster file, e.g. --clusters hdbscan clusters_hdbscan.json. Ignored with --conplag-labels-csv.")
    parser.add_argument("--output", default=None, help="Optional path to write the resulting metrics as JSON")
    args = parser.parse_args()

    if bool(args.ground_truth) == bool(args.conplag_labels_csv):
        parser.error("specify exactly one of --ground-truth or --conplag-labels-csv")
    if args.ground_truth and not args.clusters:
        parser.error("--clusters is required with --ground-truth")

    with open(args.paths) as f:
        submission_paths = json.load(f)

    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrices = {"fused": np.load(args.fused)}
    for name, path in args.signal:
        mat = np.load(path)
        similarity_matrices[name] = cosine_similarity(mat) if mat.shape[0] != mat.shape[1] else mat

    if args.conplag_labels_csv:
        # resolving: match ConPlag's sub1/sub2 hash IDs against the filename stems already present
        # in submission_paths, avoiding a need for the raw root_dirs just to rebuild that mapping
        import csv
        path_by_hash = {p.split("/")[-1].rsplit(".", 1)[0]: p for p in submission_paths}
        pairs = []
        skipped = 0
        with open(args.conplag_labels_csv, newline="") as f:
            for row in csv.DictReader(f):
                a, b, v = row["sub1"], row["sub2"], int(row["verdict"])
                if a not in path_by_hash or b not in path_by_hash:
                    skipped += 1
                    continue
                pairs.append((path_by_hash[a], path_by_hash[b], v))
        if skipped:
            print(f"Skipped {skipped} labels.csv pairs with a hash not found in paths.json.")

        metrics = compute_pairwise_metrics(pairs, similarity_matrices, submission_paths)
    else:
        with open(args.ground_truth) as f:
            ground_truth = json.load(f)

        predicted_clusters = {}
        for name, path in args.clusters:
            with open(path) as f:
                predicted_clusters[name] = json.load(f)

        metrics = compute_metrics(submission_paths, ground_truth, predicted_clusters, similarity_matrices)

    print(json.dumps(metrics, indent=4))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=4)
