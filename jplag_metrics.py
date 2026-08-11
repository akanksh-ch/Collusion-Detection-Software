"""
Converts a JPlag .jplag result archive (Criminal Minds run) into the submission_paths /
ground_truth / predicted_clusters / similarity_matrices shapes metrics.compute_metrics expects,
then scores JPlag itself as a baseline the same way the pipeline's own signals are scored.
"""

import json
import re
import zipfile
from pathlib import Path

import numpy as np

from metrics import compute_metrics

ORIGIN_PATTERN = re.compile(r'^o(\d+)(?:-|$)')


def load_jplag_archive(jplag_path: str) -> dict:

    # unzipping: pull the handful of JSON files we need straight out of the .jplag zip without extracting to disk
    with zipfile.ZipFile(jplag_path) as zf:
        submission_ids = json.loads(zf.read("submissionMappings.json"))["submissionIds"]
        cluster_list = json.loads(zf.read("cluster.json"))
        comparison_names = [n for n in zf.namelist() if n.startswith("comparisons/") and n.endswith(".json")]
        comparisons = [json.loads(zf.read(n)) for n in comparison_names]

    return {"submission_ids": submission_ids, "cluster_list": cluster_list, "comparisons": comparisons}


def build_ground_truth(submission_ids: dict[str, str]) -> dict[str, int]:

    # deriving: same rule as labels/criminalminds.py (origin = the oN token in the submission's own name),
    # applied to JPlag's internal ids directly since ARI/NMI/pairwise-AUC are invariant to group-id relabeling
    ground_truth = {}
    for internal_id, mapped_name in submission_ids.items():
        leaf = mapped_name.split("/")[-1]
        match = ORIGIN_PATTERN.match(leaf)
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


def score_jplag_archive(jplag_path: str) -> dict:
    archive = load_jplag_archive(jplag_path)

    submission_paths = sorted(archive["submission_ids"].keys())
    ground_truth = build_ground_truth(archive["submission_ids"])
    similarity_matrices = build_similarity_matrices(submission_paths, archive["comparisons"])
    predicted_clusters = {"jplag_spectral": build_predicted_clusters(submission_paths, archive["cluster_list"])}

    return compute_metrics(submission_paths, ground_truth, predicted_clusters, similarity_matrices)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Score a JPlag .jplag archive against Criminal Minds oN ground truth")
    parser.add_argument("jplag_archive", help="Path to the .jplag zip archive")
    parser.add_argument("--output", default=None, help="Optional path to write the resulting metrics as JSON")
    args = parser.parse_args()

    metrics = score_jplag_archive(args.jplag_archive)
    print(json.dumps(metrics, indent=4))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=4)
