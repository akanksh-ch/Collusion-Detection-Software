"""
Orchestrator script to run the collusion detection pipeline.
"""

import json
import numpy as np
from pathlib import Path
from platformdirs import user_cache_dir
from concurrent.futures import ThreadPoolExecutor, as_completed

import util
import graph
import lexical
import gst
import snf
import leiden

CACHE_DIR = Path(user_cache_dir('cds', ensure_exists=True))

def process_graph_for_submission(sub_path_str: str, ram_mb: int) -> tuple[str, np.ndarray, bool]:
    export_dir = f"{sub_path_str}_export"
    try:
        # Generate the CPG and export to dot files if they do not already exist.
        if not Path(export_dir).exists():
            graph.generate_cpg(sub_path_str, ram_mb)
        
        # Load the dot files into a NetworkX graph and generate the NetLSD embedding.
        nx_graph = graph.load_graph(export_dir)
        embedding = graph.generate_embedding(nx_graph)
        return sub_path_str, embedding, True
    except Exception as e:
        print(f"Graph pipeline failed for {sub_path_str}: {e}")
        return sub_path_str, np.zeros(250), False 

def run_pipeline(root_dirs: list[str]):
    # Calculate hardware allocation safely to prevent out-of-memory errors.
    max_workers = util.get_cpu()
    total_ram = util.get_ram()
    per_worker_ram = int((total_ram - 1024) // max_workers)
    
    print(f"Using {max_workers} threads. Allocating {per_worker_ram}MB RAM per Joern worker.")
    print(f"Using cache directory: {CACHE_DIR}")
    
    # Scan multiple root directories JPlag-style to build a list of absolute submission paths.
    submission_paths = []
    for root in root_dirs:
        util.validate_path(root)
        root_path = Path(root)
        if root_path.is_dir():
            for child in root_path.iterdir():
                if child.name != ".DS_Store":
                    submission_paths.append(str(child.absolute()))
                    
    # Sort strictly by the basename to ensure determinism across different roots.
    submission_paths.sort(key=lambda p: Path(p).name)
    
    with open(CACHE_DIR / "paths.json", "w") as f:
        json.dump(submission_paths, f)
        
    print(f"Found {len(submission_paths)} submissions.")

    # Execute the structural graph pipeline concurrently across all submission paths.
    print("Running structural graph pipeline...")
    graph_embeddings_dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_graph_for_submission, path_str, per_worker_ram): path_str 
            for path_str in submission_paths
        }
        for future in as_completed(futures):
            path_str, emb, success = future.result()
            graph_embeddings_dict[path_str] = emb

    v_topo = np.array([graph_embeddings_dict[p] for p in submission_paths])
    np.save(CACHE_DIR / "v_topo.npy", v_topo)

    # Compute and save the lexical TF-IDF embeddings.
    print("Computing lexical embeddings...")
    v_lex = lexical.compute_lexical_embeddings(submission_paths)
    np.save(CACHE_DIR / "v_lex.npy", v_lex)

    # Compute and save the GST pairwise coverage matrix.
    print("Computing GST coverage...")
    gst_coverage = gst.compute_gst_coverage(submission_paths)
    np.save(CACHE_DIR / "gst_coverage.npy", gst_coverage)

    # Fuse the topological, lexical, and GST similarity networks into a single matrix.
    print("Fusing similarity networks...")
    fused_matrix = snf.fuse_similarity_networks([v_topo, v_lex, gst_coverage])
    np.save(CACHE_DIR / "S_fused.npy", fused_matrix)

    # Run Leiden community detection on the fused network to generate the final clusters.
    print("Clustering fused network...")
    clusters = leiden.run_leiden(fused_matrix, submission_paths)

    # Save the raw clusters dictionary to disk for later parsing.
    output_path = CACHE_DIR / "clusters.json"
    with open(output_path, "w") as f:
        json.dump(clusters, f, indent=4)
        
    print(f"Pipeline complete! Raw clusters saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the full SNF + Leiden pipeline across multiple roots")
    parser.add_argument("root_dirs", nargs='+', help="One or more root directories containing submissions")
    args = parser.parse_args()
    
    run_pipeline(args.root_dirs)
