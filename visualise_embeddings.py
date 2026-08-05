#!/usr/bin/env python3
"""Visualize exported submission embeddings via t-SNE and UMAP.

Reads the .npz produced by `main.py --export-embeddings PATH`, projects
each requested vector space (v_topo / v_text / v_program) to 2D, and
colors points by ground-truth label (same detection logic as
evaluate_pipeline.py, so labels are consistent across both scripts).

Usage:
    python visualize_embeddings.py --embeddings embeddings.npz --out plots/
"""

import argparse
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

try:
    import umap
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False


POSITIVE_KEYWORD = "plag"
NEGATIVE_PATTERN = r"non[-_]?plag|(?:^|[/_-])orig(?:inal)?(?:[/_-]|$)"
BASELINE_KEYWORD = "baseline"


def is_plag_family(name: str) -> bool:
    lowered = name.lower()
    if BASELINE_KEYWORD and BASELINE_KEYWORD in lowered:
        return True
    if re.search(NEGATIVE_PATTERN, lowered, re.IGNORECASE):
        return False
    return POSITIVE_KEYWORD in lowered


def plot_projection(coords: np.ndarray, labels: np.ndarray, names: np.ndarray,
                     title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 7))

    for label, color, marker, text in (
        (1, "#d62728", "o", "plagiarized family"),
        (0, "#1f77b4", "^", "non-plagiarized"),
    ):
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, marker=marker, s=40, alpha=0.75,
                   edgecolors="white", linewidths=0.5, label=text)

    ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    global POSITIVE_KEYWORD, NEGATIVE_PATTERN, BASELINE_KEYWORD

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("embeddings_pos", nargs="?", default=None,
                     help="Path to .npz (positional shorthand for --embeddings)")
    ap.add_argument("--embeddings", default=None, help="Path to .npz from --export-embeddings")
    ap.add_argument("--out", default="./embedding_plots", help="Output directory for PNGs")
    ap.add_argument(
        "--spaces", nargs="+", default=["v_topo", "v_text", "v_program"],
        help="Which vector spaces to plot (must exist as keys in the .npz).",
    )
    ap.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--positive-keyword", default=POSITIVE_KEYWORD,
        help='Substring marking ground-truth positive, default "plagiar".',
    )
    ap.add_argument("--negative-pattern", default=NEGATIVE_PATTERN)
    ap.add_argument("--baseline-keyword", default=BASELINE_KEYWORD)
    args = ap.parse_args()

    embeddings_path = args.embeddings or args.embeddings_pos
    if not embeddings_path:
        ap.error("provide the .npz path either positionally or via --embeddings")

    POSITIVE_KEYWORD = args.positive_keyword.lower()
    NEGATIVE_PATTERN = args.negative_pattern
    BASELINE_KEYWORD = args.baseline_keyword.lower()

    os.makedirs(args.out, exist_ok=True)

    data = np.load(embeddings_path, allow_pickle=True)
    names = data["student_ids"]
    labels = np.array([1 if is_plag_family(str(n)) else 0 for n in names])

    n_pos = labels.sum()
    print(f"{len(names)} submissions loaded — {n_pos} plagiarized-family / "
          f"{len(names) - n_pos} non-plagiarized")

    for space in args.spaces:
        if space not in data:
            print(f"[skip] '{space}' not found in {embeddings_path} "
                  f"(available: {[k for k in data.files if k != 'student_ids']})")
            continue

        vectors = data[space]
        n_samples = vectors.shape[0]

        # perplexity must be < n_samples; clamp for small cohorts
        perplexity = min(args.perplexity, max(5.0, (n_samples - 1) / 3))

        print(f"\n[{space}] {n_samples} vectors, dim={vectors.shape[1]}")

        print(f"  running t-SNE (perplexity={perplexity:.1f})...")
        tsne_coords = TSNE(
            n_components=2, perplexity=perplexity, random_state=args.seed,
            init="pca", metric="cosine",
        ).fit_transform(vectors)
        plot_projection(
            tsne_coords, labels, names,
            title=f"t-SNE — {space}",
            out_path=os.path.join(args.out, f"tsne_{space}.png"),
        )

        if _UMAP_AVAILABLE:
            print("  running UMAP...")
            umap_coords = umap.UMAP(
                n_components=2, random_state=args.seed, metric="cosine",
            ).fit_transform(vectors)
            plot_projection(
                umap_coords, labels, names,
                title=f"UMAP — {space}",
                out_path=os.path.join(args.out, f"umap_{space}.png"),
            )
        else:
            print("  [skip] umap-learn not installed (`pip install umap-learn`) "
                  "— t-SNE distorts global distances, UMAP is worth adding "
                  "as a cross-check before trusting apparent clusters.")


if __name__ == "__main__":
    main()
