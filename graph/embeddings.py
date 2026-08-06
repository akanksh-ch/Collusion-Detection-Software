"""
File which generates embeddings from Graphs
"""

def generate_embedding(
    graph: nx.Graph,
    scale_min: float = -2.0,
    scale_max: float = 2.0,
    scale_steps: int = 250,
) -> np.ndarray:
    """
    Minimal NetLSD heat-trace signature (Tsitsulin et al., KDD 2018).
    Exact eigendecomposition — fine for CPG-sized graphs.
    """
    L = nx.normalized_laplacian_matrix(graph).toarray().astype(np.float64)
    eigenvalues = eigh(L, eigvals_only=True)  # ascending, in [0, 2]

    scales = np.logspace(scale_min, scale_max, scale_steps)
    heat_traces = np.array([np.sum(np.exp(-t * eigenvalues)) for t in scales])
    return heat_traces
