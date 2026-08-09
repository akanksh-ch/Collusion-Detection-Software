"""
Fuses graph, lexical, and greedy string tiling similarity matrices into a single consolidated network for collusion detection.
"""

import numpy as np
import snf
from sklearn.metrics.pairwise import cosine_similarity

def fuse_similarity_networks(matrices: list[np.ndarray], k_neighbors: int = 10, t_iterations: int = 20) -> np.ndarray:

    # parsing: convert raw embeddings to non-negative cosine similarity matrices if they are not already square affinity matrices
    affinities = []
    for mat in matrices:
        sim = cosine_similarity(mat) if mat.shape[0] != mat.shape[1] else mat
        sim = np.clip(sim, 0, None)
        np.fill_diagonal(sim, 1.0)
        affinities.append(sim)

    # combining: apply similarity network fusion cross-diffusion to build the single robust similarity graph
    fused_matrix = snf.snf(affinities, K=k_neighbors, t=t_iterations)

    # rescaling: SNF's converged matrix is row-normalized/diffused (rows sum to ~1), so off-diagonal
    # entries sit around 1/N regardless of true similarity, a different scale to the [0,1] cosine/GST
    # matrices that fed it. Min-max rescale the off-diagonal entries back onto [0,1] so a downstream
    # absolute threshold stays meaningful instead of zeroing out almost every edge in the graph
    n = fused_matrix.shape[0]
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_vals = fused_matrix[off_diag_mask]
    lo, hi = off_vals.min(), off_vals.max()
    fused_matrix = (fused_matrix - lo) / (hi - lo + 1e-12)
    np.fill_diagonal(fused_matrix, 1.0)

    return fused_matrix

if __name__ == '__main__':
    import argparse

    # testing: generate dummy matrices to independently verify the fusion module's execution
    parser = argparse.ArgumentParser(description="Fuse similarity networks using SNF")
    parser.add_argument("--size", type=int, default=45, help="Number of dummy submissions to simulate")
    args = parser.parse_args()

    dummy_v_topo = np.random.rand(args.size, 128)
    dummy_v_lex = np.random.rand(args.size, 100)
    dummy_gst = np.clip(np.random.rand(args.size, args.size), 0, 1)
    np.fill_diagonal(dummy_gst, 1.0)

    result = fuse_similarity_networks([dummy_v_topo, dummy_v_lex, dummy_gst])
    print(f"Fused matrix shape: {result.shape}")
