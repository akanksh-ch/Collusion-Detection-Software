"""
Executes Leiden community detection on a fused similarity matrix to group submissions into distinct collusion clusters.
"""

import numpy as np
import igraph as ig
import leidenalg

def run_leiden(fused_matrix: np.ndarray, submission_paths: list[str], resolution: float = 1.0, threshold: float = 0.5, seed: int = 0) -> dict[str, int]:
    
    # Threshold the dense similarity matrix to remove weak connections and convert it into a weighted undirected igraph network[cite: 20].
    adj_matrix = np.where(fused_matrix >= threshold, fused_matrix, 0)[cite: 20]
    graph = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), mode="undirected", loops=False)[cite: 20]
    
    # Assign the raw absolute paths directly to the graph vertices.
    graph.vs["name"] = submission_paths
    
    # Execute the Leiden algorithm with a fixed random seed and resolution parameter to ensure deterministic community detection[cite: 20].
    partition = leidenalg.find_partition(
        graph, 
        leidenalg.RBConfigurationVertexPartition, 
        weights=graph.es["weight"], 
        resolution_parameter=resolution, 
        seed=seed
    )[cite: 20]
    
    # Map the resulting community integer assignments back using the graph's internal path names.
    clusters = {v["name"]: cluster_id for v, cluster_id in zip(graph.vs, partition.membership)}
    
    return clusters
