"""
Converts a JPlag .jplag result archive into the submission_paths / ground_truth /
predicted_clusters / similarity_matrices shapes metrics.compute_metrics expects, then scores
JPlag itself as a baseline the same way the pipeline's own signals are scored. Ground truth is
derived from the submission naming convention of the given --dataset (see ORIGIN_PATTERNS).
"""

import json
import re
import zipfile
from pathlib import Path

import numpy as np

from metrics import compute_metrics

# One entry per dataset's origin-token convention, mirrored from the matching labels/*.py script,
# since JPlag's own submissionIds mapping uses the same submission folder names the pipeline does.
ORIGIN_PATTERNS = {
    "criminalminds": re.compile(r'^o(\d+)(?:-|$)'),   # labels/criminalminds.py
    "progpedia19": re.compile(r'subm(\d+)$'),         # labels/progpedia19.py
}


def load_jplag_archive(jplag_path: str) -> dict:

    # unzipping: pull the handful of JSON files we need straight out of the .jplag zip without extracting to disk
    with zipfile.ZipFile(jplag_path) as zf:
        submission_ids = json.loads(zf.read("submissionMappings.json"))["submissionIds"]
        cluster_list = json.loads(zf.read("cluster.json"))
        comparison_names = [n for n in zf.namelist() if n.startswith("comparisons/") and n.endswith(".json")]
        comparisons = [json.loads(zf.read(n)) for n in comparison_names]

    return {"submission_ids": submission_ids, "cluster_list": cluster_list, "comparisons": comparisons}


def build_ground_truth(submission_ids: dict[str, str], dataset: str = "criminalminds") -> dict[str, int]:

    # deriving: same rule as the matching labels/*.py script (origin = the oN / subm<N> token in the
    # submission's own name), applied to JPlag's internal ids directly since ARI/NMI/pairwise-AUC are
    # invariant to group-id relabeling
    pattern = ORIGIN_PATTERNS[dataset]
    ground_truth = {}
    for internal_id, mapped_name in submission_ids.items():
        leaf = mapped_name.split("/")[-1]
        match = pattern.search(leaf)
        if not match:
            raise ValueError(f"Could not parse origin token from JPlag submission name: {leaf}")
        ground_truth[internal_id] = int(match.group(1))
    return ground_truth


def build_similarity_matrices(submission_paths: list[str], comparisons: list[dict]) -> dict[str, np.ndarray]:

    # indexing: map each submission id to its row/column position once, shared by both AVG and MAX matrices
    index_by_id = {p: i for i, p in enumerate(submission_paths)}
    n = len(submission_paths)
    s_avg = np.eye(n)
    s_max = np.eye(n)

    # filling: each comparisons/*.json is one unordered pair; JPlag's AVG/MAX similarities are symmetric by definition
    for comp in comparisons:
        i = index_by_id[comp["firstSubmissionId"]]
        j = index_by_id[comp["secondSubmissionId"]]
        s_avg[i, j] = s_avg[j, i] = comp["similarities"]["AVG"]
        s_max[i, j] = s_max[j, i] = comp["similarities"]["MAX"]

    return {"jplag_avg": s_avg, "jplag_max": s_max}


def build_predicted_clusters(submission_paths: list[str], cluster_list: list[dict]) -> dict[str, int]:

    # assigning: JPlag's spectral clusters only list members that made a cluster, so anything absent
    # is noise (-1), matching how the pipeline already treats HDBSCAN/Leiden noise points
    predicted = {p: -1 for p in submission_paths}
    for cluster_id, cluster in enumerate(cluster_list):
        for member in cluster["members"]:
            predicted[member] = cluster_id
    return predicted


def score_jplag_archive(jplag_path: str, dataset: str = "criminalminds") -> dict:
    archive = load_jplag_archive(jplag_path)

    submission_paths = sorted(archive["submission_ids"].keys())
    ground_truth = build_ground_truth(archive["submission_ids"], dataset)
    similarity_matrices = build_similarity_matrices(submission_paths, archive["comparisons"])
    predicted_clusters = {"jplag_spectral": build_predicted_clusters(submission_paths, archive["cluster_list"])}

    return compute_metrics(submission_paths, ground_truth, predicted_clusters, similarity_matrices)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Score a JPlag .jplag archive against a dataset's origin-token ground truth")
    parser.add_argument("jplag_archive", help="Path to the .jplag zip archive")
    parser.add_argument("--dataset", choices=sorted(ORIGIN_PATTERNS), default="criminalminds",
                         help="Which origin-token naming convention to parse ground truth with (default: criminalminds)")
    parser.add_argument("--output", default=None, help="Optional path to write the resulting metrics as JSON")
    args = parser.parse_args()

    metrics = score_jplag_archive(args.jplag_archive, args.dataset)
    print(json.dumps(metrics, indent=4))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=4)
