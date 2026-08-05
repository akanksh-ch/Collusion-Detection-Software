#!/usr/bin/env python3
"""HDBSCAN clustering using GST coverage as the distance metric, instead
of (or blended with) embedding cosine distance.

Reads gst_coverage per pair from the pipeline's eval CSV (must be a
complete all-pairs CSV, i.e. --sim-floor 0.0) and builds a full pairwise
distance matrix as 1 - gst_coverage. Optionally blends in embedding
cosine distance from an exported .npz.

Usage:
    # GST distance only
    python gst_hdbscan_cluster.py output/result-all-eval.csv \
        --out output/gst_hdbscan_cluster.json

    # Blend GST distance with v_program cosine distance (50/50)
    python gst_hdbscan_cluster.py output/result-all-eval.csv \
        --embeddings output/embeddings.npz --space v_program --alpha 0.5 \
        --out output/gst_blend_hdbscan_cluster.json
"""

import argparse
import csv
import json
import os
import warnings

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def strip_ext(name: str) -> str:
    root, ext = os.path.splitext(name)
    return root if ext else name


def load_gst_distance(csv_path: str):
    """Returns (names, dist_matrix) built from 1 - gst_coverage per pair.
    Requires the CSV to cover every pair (i.e. an all-pairs run)."""
    pairs = {}
    names = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            a = strip_ext(row["student_a"])
            b = strip_ext(row["student_b"])
            names.add(a)
            names.add(b)
            pairs[frozenset((a, b))] = float(row["gst_coverage"])

    names = sorted(names)
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}
    expected_pairs = n * (n - 1) // 2

    missing = expected_pairs - len(pairs)
    if missing > 0:
        print(f"[warn] {missing} of {expected_pairs} pairs missing from {csv_path} "
              f"(was this run with --sim-floor 0.0? missing pairs default to distance 1.0)")

    dist = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    for key, coverage in pairs.items():
        a, b = tuple(key)
        dist[idx[a], idx[b]] = 1.0 - coverage
        dist[idx[b], idx[a]] = 1.0 - coverage

    return names, dist


def load_embedding_distance(npz_path: str, space: str, names: list[str]):
    """Returns a distance matrix aligned to `names` order."""
    data = np.load(npz_path, allow_pickle=True)
    if space not in data:
        available = [k for k in data.files if k != "student_ids"]
        raise SystemExit(f"'{space}' not found in {npz_path} (available: {available})")

    # v_topo/v_text/v_program cover every submission by construction and
    # only ever have "student_ids". v_g2v/v_stylo can have partial
    # coverage (empty CPGs / empty stripped source get dropped), in
    # which case main.py's export writes a "<space>_ids" sidecar array
    # instead — use that when present so vectors don't get silently
    # misaligned to the wrong names.
    id_key = f"{space}_ids" if f"{space}_ids" in data.files else "student_ids"
    npz_names = [str(n) for n in data[id_key]]
    npz_idx = {name: i for i, name in enumerate(npz_names)}

    missing = [n for n in names if n not in npz_idx]
    if missing:
        raise SystemExit(f"{len(missing)} submissions from the CSV not found in "
                          f"'{id_key}' of {npz_path} (space='{space}'), e.g. {missing[:5]}. "
                          f"This is expected if '{space}' has partial coverage — "
                          f"pass a filtered CSV/name list, or use a fully-covered space.")

    order = [npz_idx[n] for n in names]
    vectors = data[space][order].astype(np.float64)
    dist = cosine_distances(vectors)
    np.fill_diagonal(dist, 0.0)
    return np.clip(dist, 0.0, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", help="Pipeline eval CSV (needs a gst_coverage column, all-pairs)")
    ap.add_argument("--out", default="gst_hdbscan_cluster.json")
    ap.add_argument("--embeddings", default=None,
                     help="Optional .npz to blend in embedding cosine distance")
    ap.add_argument("--space", default="v_program",
                     help="Embedding space to use if --embeddings is given")
    ap.add_argument("--alpha", type=float, default=0.5,
                     help="Blend weight for GST distance vs embedding distance "
                          "(1.0 = GST only, 0.0 = embedding only, default 0.5)")
    ap.add_argument("--min-cluster-size", type=int, default=2)
    ap.add_argument("--min-samples", type=int, default=None)
    args = ap.parse_args()

    names, gst_dist = load_gst_distance(args.csv_path)
    print(f"Loaded GST distance for {len(names)} submissions from {args.csv_path}")

    if args.embeddings:
        emb_dist = load_embedding_distance(args.embeddings, args.space, names)
        dist = args.alpha * gst_dist + (1 - args.alpha) * emb_dist
        print(f"Blending GST distance ({args.alpha:.2f}) with '{args.space}' "
              f"cosine distance ({1 - args.alpha:.2f})")
    else:
        dist = gst_dist
        print("Using GST distance only")

    clusterer = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="precomputed",
    )
    labels = clusterer.fit_predict(dist)

    clusters: dict[int, list[str]] = {}
    noise_count = 0
    for name, label in zip(names, labels):
        if label == -1:
            noise_count += 1
            continue
        clusters.setdefault(int(label), []).append(name)

    cluster_list = [{"members": sorted(members)} for members in clusters.values()]

    print(f"  {len(cluster_list)} clusters formed, {noise_count} submissions "
          f"labeled noise (treated as singletons downstream)")
    for i, c in enumerate(cluster_list):
        print(f"    cluster {i}: {len(c['members'])} members — {c['members']}")

    with open(args.out, "w") as f:
        json.dump(cluster_list, f, indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
