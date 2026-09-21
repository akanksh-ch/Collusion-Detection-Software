"""
Turns Code Property Graphs into fixed-length, cosine-comparable embeddings via Graph2Vec
(Narayanan et al., MLG 2017): WL-subtree "documents" per graph, doc2vec-trained across the
whole submission batch at once -- structure- and node-type-aware, size-invariant, no training
labels needed.
"""

import networkx as nx
import numpy as np
from karateclub import Graph2Vec

DIMENSIONS = 128  # pipeline.py's zero-vector fallback keys off this
WL_ITERATIONS = 2


def _feature_labeled(graph: nx.Graph) -> nx.Graph:
    # relabeling: Graph2Vec's attributed mode reads node type off a "feature" attr, and needs
    # contiguous integer node ids -- Joern's export.dot "label" attr is the raw CPG node type
    g = nx.convert_node_labels_to_integers(graph)
    for node, attrs in g.nodes(data=True):
        g.nodes[node]["feature"] = attrs.get("label") or "<none>"
    return g


def generate_embeddings(graphs: list[nx.Graph]) -> np.ndarray:
    # fitting: one Doc2Vec model across the entire submission batch, so every submission's
    # embedding lives in the same learned space and is directly cosine-comparable
    model = Graph2Vec(dimensions=DIMENSIONS, wl_iterations=WL_ITERATIONS, attributed=True, workers=1, seed=0) # set seed and single worker, check https://github.com/piskvorky/gensim/issues/641
    model.fit([_feature_labeled(g) for g in graphs])
    return model.get_embedding()


if __name__ == '__main__':
    import argparse

    # testing: embed a couple of dummy random graphs to independently verify the module's execution
    parser = argparse.ArgumentParser(description="Generate Graph2Vec embeddings for a batch of graphs")
    parser.add_argument("--size", type=int, default=10, help="Number of dummy graphs to simulate")
    parser.add_argument("--nodes", type=int, default=30, help="Nodes per dummy graph")
    args = parser.parse_args()

    dummy_graphs = [nx.gnm_random_graph(args.nodes, args.nodes * 2, seed=i) for i in range(args.size)]
    for g in dummy_graphs:
        nx.set_node_attributes(g, {n: str(g.degree(n) % 4) for n in g.nodes()}, "label")

    result = generate_embeddings(dummy_graphs)
    print(f"Embedded {len(dummy_graphs)} graphs into shape {result.shape}")
