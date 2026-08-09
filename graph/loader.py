"""
This file takes in multiple submission root directories and spits out submission paths. Files or Directories.
"""

import glob
import os
import threading
import networkx as nx

# pydot's dot parser isn't thread-safe, and pipeline.py calls load_graph()
# from multiple threads. This lock makes sure only one thread parses a
# .dot file at a time, so parses don't corrupt each other.
_DOT_PARSE_LOCK = threading.Lock()


def load_graph(path: str) -> nx.MultiGraph:

    dot_files = sorted(glob.glob(os.path.join(path, "**", "export.dot"), recursive=True))
    if not dot_files:
        raise FileNotFoundError(f"No export.dot files found under {path}")

    combined = nx.MultiGraph()
    node_offset = 0

    for dot_file in dot_files:
        with _DOT_PARSE_LOCK:
            g = nx.drawing.nx_pydot.read_dot(dot_file)

        if not isinstance(g, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            raise TypeError(f"{dot_file} returned unexpected type {type(g)}")

        if g.number_of_nodes() == 0:
            # empty method graph (e.g. an implicit _init_ with no body) —
            # legitimately contributes nothing, not a parse failure
            continue

        sorted_nodes = sorted(g.nodes())
        mapping = {old: node_offset + i for i, old in enumerate(sorted_nodes)}
        g = nx.relabel_nodes(g, mapping, copy=True)

        # force to MultiGraph explicitly — to_undirected() on a MultiDiGraph
        # stays a MultiGraph too, but being explicit avoids relying on that
        ug = nx.MultiGraph(g)

        combined.add_nodes_from(sorted(ug.nodes(data=True)))
        combined.add_edges_from(ug.edges(data=True))

        node_offset += g.number_of_nodes()

    return combined
