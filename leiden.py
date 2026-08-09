"""
Executes Leiden community detection on a fused similarity matrix to group submissions into distinct collusion clusters.
"""

import numpy as np
import igraph as ig
import leidenalg

def run_leiden(fused_matrix: np.ndarray, submission_paths: list[str], resolution: float = 0.1, threshold: float = 0.5, seed: int = 0) -> dict[str, int]:

    # Threshold the dense similarity matrix to remove weak connections and convert it into a weighted undirected igraph network.
    adj_matrix = np.where(fused_matrix >= threshold, fused_matrix, 0)
    graph = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), mode="undirected", loops=False)

    # Assign the raw absolute paths directly to the graph vertices.
    graph.vs["name"] = submission_paths

    # Execute Leiden under CPM rather than modularity/RBConfiguration: CPM's resolution parameter has a
    # direct interpretation as expected within-community edge density and, unlike modularity, is not
    # subject to the resolution limit that merges genuinely distinct small communities on small graphs.
    partition = leidenalg.find_partition(
        graph,
        leidenalg.CPMVertexPartition,
        weights=graph.es["weight"],
        resolution_parameter=resolution,
        seed=seed
    )

    # Map the resulting community integer assignments back using the graph's internal path names.
    clusters = {v["name"]: cluster_id for v, cluster_id in zip(graph.vs, partition.membership)}

    return clusters

if __name__ == '__main__':
    import argparse

    # testing: cluster a dummy random fused matrix to independently verify the module's execution
    parser = argparse.ArgumentParser(description="Run Leiden/CPM community detection on a fused similarity matrix")
    parser.add_argument("--size", type=int, default=45, help="Number of dummy submissions to simulate")
    parser.add_argument("--resolution", type=float, default=0.1, help="CPM resolution parameter")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold below which edges are dropped")
    args = parser.parse_args()

    dummy_matrix = np.clip(np.random.rand(args.size, args.size), 0, 1)
    np.fill_diagonal(dummy_matrix, 1.0)
    dummy_paths = [f"submission_{i}" for i in range(args.size)]

    result = run_leiden(dummy_matrix, dummy_paths, resolution=args.resolution, threshold=args.threshold)
    print(f"Found {len(set(result.values()))} clusters across {args.size} submissions")
