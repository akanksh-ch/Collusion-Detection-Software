#!/usr/bin/env python3
"""Compare pipeline vs JPlag: pairwise ROC-AUC/PR-AUC and clustering quality.

Ground truth: a submission is "plagiarized-family" if its filename does not
contain "non-plagiarized" (i.e. original-baseline + any *-plagiarized-L*).
A pair is a positive (true collusion/derivation pair) iff both members are
in the plagiarized family.

Usage:
    python evaluate_pipeline.py \
        --pipeline-csv result-all.csv --pipeline-cluster cluster.json \
        --jplag-csv jplag_results.csv --jplag-cluster jplag_cluster.json
"""

import argparse
import csv
import json
import os
import re
import sys

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def strip_ext(name: str) -> str:
    """Strip a trailing file extension, if any (works for any language,
    not just .java) — leaves bare identifiers/folder names untouched."""
    root, ext = os.path.splitext(name)
    return root if ext else name


# Overridable at the CLI (--positive-keyword / --negative-pattern /
# --baseline-keyword) for corpora that don't use "plagiar..." naming.
# Default "plag" (not "plagiar") so short folder names like Zenodo's
# orig/plag split still match — "plagiar" can never match inside the
# shorter string "plag", which silently mislabels everything negative.
POSITIVE_KEYWORD = "plag"
NEGATIVE_PATTERN = r"non[-_]?plag|(?:^|[/_-])orig(?:inal)?(?:[/_-]|$)"
BASELINE_KEYWORD = "baseline"  # e.g. "original-baseline" — counts as positive

# Populated from --manifest (JSON: submission-stem -> bool) when given.
# Some corpora (e.g. the "Criminal Minds" Zenodo dataset) encode the
# plag/orig label ONLY via which --src directory a submission lives in
# (random pseudonyms like "o1-A-uni1-drcfg" carry no plag/orig substring
# at all) — JPlag drops that folder context when assigning submission
# IDs, so no filename heuristic can recover it after the fact. The
# manifest is built straight from the source folders (see
# run_pipeline.sh) before that information is lost, and takes priority
# over the keyword heuristic below whenever a submission is present in
# it. Submissions absent from the manifest (e.g. flat, filename-encoded
# corpora like IRPlag) fall back to the keyword heuristic unchanged.
MANIFEST: dict[str, bool] = {}


def is_plag_family(name: str) -> bool:
    """A submission counts as ground-truth positive ("plagiarized
    family") if:
      1. It appears in MANIFEST (built from actual source-folder
         layout) — that label wins outright, since it's ground truth
         from the filesystem rather than a guess from the name.
      2. Otherwise, POSITIVE_KEYWORD or BASELINE_KEYWORD appears
         anywhere in its name or path (case-insensitive) and it doesn't
         match NEGATIVE_PATTERN. Covers folder-based layouts (e.g.
         case01/plag/L3/Submission.java, zenodo/orig/xxx) as well as
         filename-encoded ones (student01-plagiarized-L3.java), across
         any language/extension.
    """
    if MANIFEST:
        stem = strip_ext(name)
        if stem in MANIFEST:
            return MANIFEST[stem]
        if name in MANIFEST:
            return MANIFEST[name]

    lowered = name.lower()
    if BASELINE_KEYWORD and BASELINE_KEYWORD in lowered:
        return True
    if re.search(NEGATIVE_PATTERN, lowered, re.IGNORECASE):
        return False
    return POSITIVE_KEYWORD in lowered


# Overridable via --family-pattern. Extracts the origin-solution family a
# submission was derived from (e.g. "o4" out of "plag_o4-B-uni1-ijsyb" or
# "orig_o4-odiyc"). This matters because "both submissions are somewhere
# in the plag family" is NOT the same claim as "these two submissions are
# actually related" — plag_o1-... and plag_o5-... share no code lineage
# just because both are plagiarized copies of *something*. Scoring pairwise
# AUC or clustering ARI/NMI against the coarse plag/non-plag label when the
# corpus actually has multiple independent origin-families will make a
# correctly-discriminating similarity score look like noise, since same-
# family and cross-family "positives" get lumped together. Set to "" to
# disable and fall back to the old plag/non-plag-only definition (e.g. for
# corpora with a single origin, like IRPlag).
FAMILY_PATTERN = r"o(\d+)(?![a-z])"


def extract_family(name: str) -> str | None:
    """Returns a family key (e.g. "o4") if FAMILY_PATTERN matches, else None."""
    if not FAMILY_PATTERN:
        return None
    m = re.search(FAMILY_PATTERN, name, re.IGNORECASE)
    return f"o{m.group(1)}" if m else None


def is_true_pair(a: str, b: str) -> bool:
    """Ground truth for a pair: same origin family if both names resolve
    to one (the precise, lineage-aware definition); otherwise falls back
    to "both in the plag family" for corpora without a family/origin
    identifier in their naming."""
    fa, fb = extract_family(a), extract_family(b)
    if fa is not None and fb is not None:
        return fa == fb
    return is_plag_family(a) and is_plag_family(b)


def load_pairs_csv(path: str, name_a_key: str, name_b_key: str, score_keys: list[str]):
    """Returns y_true (list[int]) and dict[score_key -> list[float]]."""
    y_true = []
    scores = {k: [] for k in score_keys}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = strip_ext(row[name_a_key])
            b = strip_ext(row[name_b_key])
            y_true.append(1 if is_true_pair(a, b) else 0)
            for k in score_keys:
                scores[k].append(float(row[k]))

    return y_true, scores


def print_pair_metrics(label: str, y_true, scores: dict):
    n = len(y_true)
    n_pos = sum(y_true)
    print(f"\n[{label}] pairwise evaluation — {n} pairs "
          f"({n_pos} positive / {n - n_pos} negative, "
          f"base rate {n_pos / n:.3f})")
    for score_name, values in scores.items():
        auc = roc_auc_score(y_true, values)
        pr = average_precision_score(y_true, values)
        print(f"  {score_name:20s} ROC-AUC={auc:.4f}  PR-AUC={pr:.4f}")


def load_cluster_labels(path: str) -> dict[str, int]:
    """Maps submission name -> cluster id. Members appearing in no cluster
    are not included here; caller assigns them singleton ids."""
    with open(path) as f:
        clusters = json.load(f)

    labels = {}
    for cluster_id, cluster in enumerate(clusters):
        for member in cluster["members"]:
            labels[strip_ext(member)] = cluster_id
    return labels


def evaluate_clustering(label: str, cluster_path: str, all_names: set[str]):
    cluster_labels = load_cluster_labels(cluster_path)

    # Ground truth: if every submission resolves to an origin family (see
    # extract_family), use that multi-class label — it's the actual
    # structure a good clustering should recover (one cluster per origin
    # solution). Falls back to the coarse binary plag/non-plag label only
    # for corpora without a family/origin identifier in their names.
    families = {name: extract_family(name) for name in all_names}
    use_family_gt = all(f is not None for f in families.values())

    # Any submission absent from every cluster is its own singleton cluster
    # (treat "unclustered" as maximally conservative / not-grouped-with-anyone).
    next_id = max(cluster_labels.values(), default=-1) + 1
    pred_labels = []
    true_labels = []
    for name in sorted(all_names):
        if name in cluster_labels:
            pred_labels.append(cluster_labels[name])
        else:
            pred_labels.append(next_id)
            next_id += 1
        true_labels.append(families[name] if use_family_gt else (1 if is_plag_family(name) else 0))

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    gt_desc = "origin-family" if use_family_gt else "plag/non-plag"
    n_clusters = len(set(pred_labels))
    print(f"\n[{label}] clustering evaluation — {len(all_names)} submissions, "
          f"{n_clusters} resolved clusters (incl. singletons)")
    print(f"  Adjusted Rand Index (vs {gt_desc} ground truth): {ari:.4f}")
    print(f"  Normalized Mutual Info:                          {nmi:.4f}")

    # Purity per real cluster (non-singleton, size > 1): fraction of the
    # majority ground-truth class within each resolved cluster.
    from collections import defaultdict, Counter
    members_by_cluster = defaultdict(list)
    for name, cid in zip(sorted(all_names), pred_labels):
        members_by_cluster[cid].append(name)

    real_clusters = {cid: m for cid, m in members_by_cluster.items() if len(m) > 1}
    if real_clusters:
        total_correct = 0
        total_members = 0
        for cid, members in real_clusters.items():
            if use_family_gt:
                counts = Counter(families[m] for m in members)
                majority = counts.most_common(1)[0][1]
            else:
                plag_count = sum(1 for m in members if is_plag_family(m))
                majority = max(plag_count, len(members) - plag_count)
            total_correct += majority
            total_members += len(members)
        purity = total_correct / total_members
        print(f"  Purity over {len(real_clusters)} multi-member clusters "
              f"({total_members} submissions): {purity:.4f}")


def main():
    global POSITIVE_KEYWORD, NEGATIVE_PATTERN, BASELINE_KEYWORD, FAMILY_PATTERN

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipeline-csv", required=True)
    ap.add_argument("--pipeline-cluster", required=True)
    ap.add_argument("--jplag-csv", default=None,
                     help="Optional — omit to evaluate the pipeline alone.")
    ap.add_argument("--jplag-cluster", default=None,
                     help="Optional — omit to evaluate the pipeline alone.")
    ap.add_argument(
        "--positive-keyword", default=POSITIVE_KEYWORD,
        help='Substring (case-insensitive) marking a submission as ground-truth '
             'positive, e.g. "plagiar" (default) or "colluded".',
    )
    ap.add_argument(
        "--negative-pattern", default=NEGATIVE_PATTERN,
        help='Regex (case-insensitive) that overrides positive-keyword to negative, '
             r'e.g. "non[-_]?plagiari[sz]ed" (default).',
    )
    ap.add_argument(
        "--baseline-keyword", default=BASELINE_KEYWORD,
        help='Substring marking an origin/seed file as ground-truth positive '
             '(e.g. "baseline", default) even without the positive keyword. '
             'Pass "" to disable.',
    )
    ap.add_argument(
        "--family-pattern", default=FAMILY_PATTERN,
        help='Regex whose first capture group is the origin-solution family id '
             '(default r"o(\\d+)(?![a-z])", matches "o4" in "plag_o4-B-uni1-x" '
             'and "orig_o4-odiyc"). A true pair = same family, which is the '
             'precise ground truth for corpora with multiple independent '
             'origin solutions. Pass "" to disable and fall back to the '
             'coarser "both in plag family" definition (e.g. for IRPlag, '
             'which has a single origin).',
    )
    args = ap.parse_args()

    POSITIVE_KEYWORD = args.positive_keyword.lower()
    NEGATIVE_PATTERN = args.negative_pattern
    BASELINE_KEYWORD = args.baseline_keyword.lower()
    FAMILY_PATTERN = args.family_pattern

    # --- pairwise: your pipeline ---
    y_true_pipe, scores_pipe = load_pairs_csv(
        args.pipeline_csv, "student_a", "student_b",
        ["similarity", "sim_structural", "sim_lexical", "gst_coverage"],
    )
    print_pair_metrics("Pipeline", y_true_pipe, scores_pipe)

    # --- pairwise: jplag (optional) ---
    if args.jplag_csv:
        y_true_jplag, scores_jplag = load_pairs_csv(
            args.jplag_csv, "submissionName1", "submissionName2",
            ["averageSimilarity", "maxSimilarity"],
        )
        print_pair_metrics("JPlag", y_true_jplag, scores_jplag)

    # --- clustering: both, against the same submission universe ---
    all_names = set()
    with open(args.pipeline_csv, newline="") as f:
        for row in csv.DictReader(f):
            all_names.add(strip_ext(row["student_a"]))
            all_names.add(strip_ext(row["student_b"]))

    evaluate_clustering("Pipeline", args.pipeline_cluster, all_names)
    if args.jplag_cluster:
        evaluate_clustering("JPlag", args.jplag_cluster, all_names)

    print()


if __name__ == "__main__":
    sys.exit(main())
