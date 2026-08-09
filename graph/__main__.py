"""
Runs the graph sub-pipeline end-to-end on a single source file:
source -> generate_cpg (joern) -> load_graph (loader) -> generate_embedding (embeddings).

Usage:
    python -m graph path/to/source_file
"""

import argparse

from .joern import generate_cpg
from .loader import load_graph
from .embeddings import generate_embedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a netlsd graph embedding from source")
    parser.add_argument("path", help="Source file to run through joern -> loader -> embeddings")
    parser.add_argument("--cache-dir", default=None,
                         help="Directory to write the .bin/_export CPG artifacts into (default: next to the source path)")
    args = parser.parse_args()

    export_dir = generate_cpg(args.path, out_dir=args.cache_dir)
    graph = load_graph(export_dir)
    signature = generate_embedding(graph)

    print(signature.shape if hasattr(signature, "shape") else signature)
