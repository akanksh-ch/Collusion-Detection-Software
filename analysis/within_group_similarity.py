import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from labels import generate_labels_criminalminds, generate_labels_progpedia19

LABEL_GENERATORS = {
    "criminalminds": generate_labels_criminalminds,
    "progpedia19": generate_labels_progpedia19,
}


def _scale_free_stats(values: np.ndarray) -> dict:
    # summarizing: mean/std/min/max/IQR plus coefficient of variation (std/mean), which is what
    # actually lets us compare "how heterogeneous is within-group similarity" across two datasets
    # whose fused matrices were independently min-max-rescaled and so aren't on a shared raw scale
    mean = float(np.mean(values))
    std = float(np.std(values))
    q1, q3 = np.percentile(values, [25, 75])
    return {
        "mean": mean,
        "std": std,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "iqr": float(q3 - q1),
        "coefficient_of_variation": (std / mean) if mean != 0 else None,
    }


def analyze(fused_matrix: np.ndarray, submission_paths: list[str], labels: dict[str, int]) -> dict:
    n = fused_matrix.shape[0]

    # sanity: the fused matrix must be the symmetric affinity fusion.py produces, diagonal 1.0
    assert fused_matrix.shape == (n, n), f"fused_matrix is not square: {fused_matrix.shape}"
    assert np.allclose(fused_matrix, fused_matrix.T, atol=1e-8), "fused_matrix is not symmetric"
    assert np.allclose(np.diag(fused_matrix), 1.0, atol=1e-8), "fused_matrix diagonal is not 1.0"

    # distance: identical conversion to agglomerative.run_agglomerative -- distance = 1 - similarity,
    # diagonal zeroed, so the bottleneck/nearest-outsider distances below are on the same scale the
    # global-threshold clusterer itself thresholds on
    distance_matrix = 1.0 - fused_matrix
    np.fill_diagonal(distance_matrix, 0.0)

    label_by_idx = [labels[p] for p in submission_paths]
    groups: dict[int, list[int]] = {}
    for idx, gid in enumerate(label_by_idx):
        groups.setdefault(gid, []).append(idx)

    n_groups_total = len(groups)
    scored_groups = {gid: idxs for gid, idxs in groups.items() if len(idxs) >= 2}

    # 1. per-group mean pairwise fused similarity
    group_mean_similarity = {}
    for gid, idxs in scored_groups.items():
        idxs_arr = np.array(idxs)
        iu, ju = np.triu_indices(len(idxs_arr), k=1)
        pair_sims = fused_matrix[idxs_arr[iu], idxs_arr[ju]]
        group_mean_similarity[gid] = float(np.mean(pair_sims))
    similarity_values = np.array(list(group_mean_similarity.values()))
    similarity_stats = _scale_free_stats(similarity_values) if len(similarity_values) else None

    # 2. global-threshold separability
    all_idx = np.arange(n)
    separable_flags = {}
    group_bottlenecks = {}
    group_nearest_outsider = {}
    for gid, idxs in scored_groups.items():
        idxs_arr = np.array(idxs)
        outsiders = np.setdiff1d(all_idx, idxs_arr)

        # bottleneck: largest edge on the MST of the group's internal distances -- the smallest
        # distance_threshold an agglomerative run would need to still fully connect this group
        sub_dist = distance_matrix[np.ix_(idxs_arr, idxs_arr)]
        mst = minimum_spanning_tree(sub_dist).toarray()
        bottleneck = float(mst.max()) if mst.size and len(idxs_arr) > 1 else 0.0

        # nearest_outsider: smallest distance from any group member to any non-member
        nearest_outsider = float(distance_matrix[np.ix_(idxs_arr, outsiders)].min()) if len(outsiders) else float("inf")

        group_bottlenecks[gid] = bottleneck
        group_nearest_outsider[gid] = nearest_outsider
        separable_flags[gid] = bottleneck < nearest_outsider

    n_scored = len(scored_groups)
    n_separable = sum(separable_flags.values())
    separable_fraction = (n_separable / n_scored) if n_scored else None
    non_separable_groups = [
        {"group_id": gid, "size": len(scored_groups[gid]), "bottleneck": group_bottlenecks[gid],
         "nearest_outsider": group_nearest_outsider[gid]}
        for gid in scored_groups if not separable_flags[gid]
    ]

    # 3. group size distribution (all groups, including singletons, for context)
    group_sizes = sorted(len(idxs) for idxs in groups.values())

    return {
        "n_submissions": n,
        "n_groups": n_groups_total,
        "n_groups_scored": n_scored,
        "group_mean_similarity": {
            "per_group": group_mean_similarity,
            "stats": similarity_stats,
        },
        "global_threshold_separability": {
            "separable_fraction": separable_fraction,
            "n_separable": n_separable,
            "n_scored": n_scored,
            "non_separable_groups": non_separable_groups,
        },
        "group_sizes": {
            "all": group_sizes,
            "stats": _scale_free_stats(np.array(group_sizes)) if group_sizes else None,
        },
    }


def _fmt(x, digits=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def write_markdown_comparison(results_by_dataset: dict[str, dict], output_path: str) -> None:
    rows = [
        ("n_submissions", lambda r: r["n_submissions"]),
        ("n_groups", lambda r: r["n_groups"]),
        ("n_groups scored (size>=2)", lambda r: r["n_groups_scored"]),
        ("mean-similarity: mean", lambda r: _fmt(r["group_mean_similarity"]["stats"]["mean"])),
        ("mean-similarity: std", lambda r: _fmt(r["group_mean_similarity"]["stats"]["std"])),
        ("mean-similarity: min", lambda r: _fmt(r["group_mean_similarity"]["stats"]["min"])),
        ("mean-similarity: max", lambda r: _fmt(r["group_mean_similarity"]["stats"]["max"])),
        ("mean-similarity: IQR", lambda r: _fmt(r["group_mean_similarity"]["stats"]["iqr"])),
        ("mean-similarity: CoV", lambda r: _fmt(r["group_mean_similarity"]["stats"]["coefficient_of_variation"])),
        ("separable fraction", lambda r: _fmt(r["global_threshold_separability"]["separable_fraction"])),
        ("n separable / n scored", lambda r: f"{r['global_threshold_separability']['n_separable']}/{r['global_threshold_separability']['n_scored']}"),
        ("group size: mean", lambda r: _fmt(r["group_sizes"]["stats"]["mean"])),
        ("group size: std", lambda r: _fmt(r["group_sizes"]["stats"]["std"])),
        ("group size: min", lambda r: _fmt(r["group_sizes"]["stats"]["min"])),
        ("group size: max", lambda r: _fmt(r["group_sizes"]["stats"]["max"])),
    ]
    datasets = list(results_by_dataset.keys())
    lines = ["| statistic | " + " | ".join(datasets) + " |", "|---|" + "---|" * len(datasets)]
    for label, fn in rows:
        lines.append("| " + label + " | " + " | ".join(_fmt(fn(results_by_dataset[d])) if not isinstance(fn(results_by_dataset[d]), str) else fn(results_by_dataset[d]) for d in datasets) + " |")

    lines.append("")
    lines.append("Non-separable groups (bottleneck >= nearest-outsider distance):")
    for d in datasets:
        groups = results_by_dataset[d]["global_threshold_separability"]["non_separable_groups"]
        if not groups:
            lines.append(f"- {d}: none")
        else:
            listing = ", ".join(f"group {g['group_id']} (size {g['size']})" for g in groups)
            lines.append(f"- {d}: {listing}")

    Path(output_path).write_text("\n".join(lines) + "\n")


def main(results_dir: str, dataset: str, root_dirs: list[str], output_json: str = None) -> dict:
    results_path = Path(results_dir)

    # loading: same reload pattern as generate_report.py/main.py -- paths.json + S_fused.npy that
    # run_pipeline() already wrote to results_dir, plus the dataset's own ground-truth generator
    with open(results_path / "paths.json") as f:
        submission_paths = json.load(f)
    fused_matrix = np.load(results_path / "S_fused.npy")

    if dataset not in LABEL_GENERATORS:
        raise ValueError(f"Unknown dataset: {dataset!r} (expected one of {list(LABEL_GENERATORS)})")
    labels = LABEL_GENERATORS[dataset](root_dirs)

    result = analyze(fused_matrix, submission_paths, labels)

    print(f"[{dataset}] n_submissions={result['n_submissions']} n_groups={result['n_groups']}")

    out_path = output_json or str(results_path / f"within_group_similarity_{dataset}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Wrote {out_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose within-group fused-similarity heterogeneity and global-threshold separability for a dataset (read-only, does not modify the pipeline)")
    parser.add_argument("results_dir", help="Directory containing paths.json and S_fused.npy already written by run_pipeline() for this dataset")
    parser.add_argument("--dataset", required=True, choices=list(LABEL_GENERATORS), help="Which ground-truth label generator to use")
    parser.add_argument("root_dirs", nargs='+', help="One or more root directories containing this dataset's submissions (same roots used to run the pipeline)")
    parser.add_argument("--output", default=None, help="Where to write the per-dataset JSON (default: <results_dir>/within_group_similarity_<dataset>.json)")
    args = parser.parse_args()

    main(args.results_dir, args.dataset, args.root_dirs, output_json=args.output)
