"""
File which generates embeddings from Graphs.
"""

import hashlib

import networkx as nx
import numpy as np
from scipy.linalg import eigh


EMBEDDING_DIM = 256 + 250
WL_BINS = 256
NETLSD_DIM = 250


def _stable_hash_to_bin(value: str, bins: int = WL_BINS) -> int:
    """Map a WL label to a reproducible histogram bin."""
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % bins


def _node_type(graph: nx.Graph, node) -> str:
    """Return Joern's node label/type, with a safe fallback for unlabeled graphs."""
    return str(graph.nodes[node].get("label", graph.nodes[node].get("type", "")))


def _edge_type(attrs: dict) -> str:
    """Return Joern's edge label/type, with a safe fallback."""
    return str(attrs.get("label", attrs.get("type", "")))


def _wl_histogram(graph: nx.Graph, bins: int = WL_BINS) -> np.ndarray:
    """
    Compute a one-pass Weisfeiler-Lehman-style histogram.

    Each node is relabeled from its original node type plus the multiset of
    (edge_type, neighbor_type) pairs. Every edge incident to the node is kept,
    including edges whose endpoints have the same original node type. The
    resulting labels are hashed into a fixed-size histogram.
    """
    histogram = np.zeros(bins, dtype=np.float64)

    for node in graph.nodes:
        node_type = _node_type(graph, node)
        neighborhood = []

        # MultiGraph/MultiDiGraph-safe: iterate over every incident edge so
        # parallel Joern edges are retained rather than collapsed.
        if graph.is_directed():
            incident = list(graph.out_edges(node, keys=True, data=True))
            incident += list(graph.in_edges(node, keys=True, data=True))
            for src, dst, _key, attrs in incident:
                neighbor = dst if src == node else src
                neighborhood.append((_edge_type(attrs), _node_type(graph, neighbor)))
        else:
            for src, dst, _key, attrs in graph.edges(node, keys=True, data=True):
                neighbor = dst if src == node else src
                neighborhood.append((_edge_type(attrs), _node_type(graph, neighbor)))

        neighborhood.sort()
        relabel = f"{node_type}|{repr(neighborhood)}"
        histogram[_stable_hash_to_bin(relabel, bins)] += 1.0

    return histogram


def generate_embedding(
    graph: nx.Graph,
    scale_min: float = -2.0,
    scale_max: float = 2.0,
    scale_steps: int = NETLSD_DIM,
) -> np.ndarray:
    """
    Generate the graph embedding by concatenating two independently
    L2-normalized channels:

    1. NetLSD heat-trace signature (250 dimensions).
    2. One-pass WL neighborhood-label histogram (256 dimensions).

    Normalizing each channel before concatenation prevents the much larger
    NetLSD heat-trace values from numerically dominating the WL channel.
    """
    # NetLSD heat-trace channel.
    L = nx.normalized_laplacian_matrix(graph).toarray().astype(np.float64)
    eigenvalues = eigh(L, eigvals_only=True)

    scales = np.logspace(scale_min, scale_max, scale_steps)
    heat_traces = np.array(
        [np.sum(np.exp(-t * eigenvalues)) for t in scales],
        dtype=np.float64,
    )

    # Structural WL channel.
    wl_hist = _wl_histogram(graph, WL_BINS)

    # Normalize channels independently so their concatenation gives both
    # channels comparable influence under cosine similarity.
    heat_norm = np.linalg.norm(heat_traces)
    if heat_norm > 0:
        heat_traces = heat_traces / heat_norm

    wl_norm = np.linalg.norm(wl_hist)
    if wl_norm > 0:
        wl_hist = wl_hist / wl_norm

    return np.concatenate([heat_traces, wl_hist])


__all__ = ["generate_embedding", "EMBEDDING_DIM", "NETLSD_DIM", "WL_BINS"]
