"""
Orchestrates the full collusion detection pipeline end-to-end: scans submissions, computes graph/lexical/GST signals, fuses them with SNF, and clusters the result with HDBSCAN (primary) and Leiden/CPM (comparison).
"""

import json
import logging
import numpy as np
from pathlib import Path
from platformdirs import user_cache_dir
from concurrent.futures import ThreadPoolExecutor, as_completed

import util
import graph
import lexical
import strmatch
import fusion
import leiden
import hdbscan_cluster

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path(user_cache_dir('cds', ensure_exists=True))

# each signal gets its own subdirectory under CACHE_DIR, so CPG binaries/exports, lexical
# embeddings, and GST coverage never touch submissions/ and never collide with each other
CPG_CACHE_DIR = CACHE_DIR / "cpg"
LEXICAL_CACHE_DIR = CACHE_DIR / "lexical"
GST_CACHE_DIR = CACHE_DIR / "gst"


def run_pipeline(root_dirs: list[str], fusion_method: str = 'snf', output_dir: str = None):
    # results: paths.json/S_fused.npy/clusters_*.json/graph_failures.json are the run's actual
    # deliverables, so they go to output_dir if given (e.g. a bind-mounted host directory that
    # has to exist ahead of time) instead of CACHE_DIR, which stays reserved for the expensive-
    # to-recompute, don't-need-to-persist-outside-the-container CPG/lexical/GST artifacts
    results_dir = Path(output_dir) if output_dir else CACHE_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    # hardware: size the thread pool and per-worker RAM budget safely, guarding against 0 workers or negative RAM on constrained/single-core machines
    max_workers = max(1, util.get_cpu())
    total_ram = util.get_ram()
    per_worker_ram = max(128, int((total_ram - 1024) // max_workers))
    logger.info(f"Using {max_workers} threads. Allocating {per_worker_ram}MB RAM per Joern worker.")
    logger.info(f"Using cache directory: {CACHE_DIR}")
    logger.info(f"Using output directory: {results_dir}")

    for sub_cache_dir in (CPG_CACHE_DIR, LEXICAL_CACHE_DIR, GST_CACHE_DIR):
        sub_cache_dir.mkdir(parents=True, exist_ok=True)

    # scanning: walk every root dir JPlag-style and build one sorted, deterministic list of absolute submission paths shared by every downstream signal
    submission_paths = []
    for root in root_dirs:
        util.validate_path(root)
        root_path = Path(root)
        if root_path.is_dir():
            for child in root_path.iterdir():
                if child.name != ".DS_Store":
                    submission_paths.append(str(child.absolute()))
    if not submission_paths:
        raise ValueError(f"No submissions found under root_dirs: {root_dirs}")
    submission_paths.sort(key=lambda p: Path(p).name)
    with open(results_dir / "paths.json", "w") as f:
        json.dump(submission_paths, f)
    logger.info(f"Found {len(submission_paths)} submissions.")

    # graph: run Joern parse -> load -> NetLSD embedding concurrently across submissions, filling failures with a zero-vector instead of silently dropping them
    def process_graph_for_submission(sub_path_str: str) -> tuple[str, np.ndarray, bool]:
        # give each submission its own folder under CPG_CACHE_DIR (name + path hash) so the
        # .bin/_export artifacts never land in submissions/ and never collide across roots
        sub_cpg_dir = CPG_CACHE_DIR / util.cache_key(sub_path_str)
        export_dir = sub_cpg_dir / f"{Path(sub_path_str).name}_export"
        try:
            if not export_dir.exists():
                export_dir = Path(graph.generate_cpg(sub_path_str, per_worker_ram, out_dir=sub_cpg_dir))
            nx_graph = graph.load_graph(str(export_dir))
            embedding = graph.generate_embedding(nx_graph)
            return sub_path_str, embedding, True
        except Exception as e:
            logger.warning(f"Graph pipeline failed for {sub_path_str}: {e}")
            return sub_path_str, np.zeros(graph.EMBEDDING_DIM), False

    graph_embeddings_dict = {}
    failed_submissions = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_graph_for_submission, p): p for p in submission_paths}
        for future in as_completed(futures):
            path_str, emb, success = future.result()
            graph_embeddings_dict[path_str] = emb
            if not success:
                failed_submissions.append(path_str)
    if failed_submissions:
        logger.warning(
            f"{len(failed_submissions)}/{len(submission_paths)} submissions failed graph "
            f"processing and were filled with zero-vectors: {failed_submissions}"
        )
        with open(results_dir / "graph_failures.json", "w") as f:
            json.dump(failed_submissions, f, indent=4)
    v_topo = np.array([graph_embeddings_dict[p] for p in submission_paths])
    np.save(CPG_CACHE_DIR / "v_topo.npy", v_topo)

    # lexical: compute the shared-vocabulary TF-IDF char n-gram embedding matrix over the same ordered submission list
    logger.info("Computing lexical embeddings...")
    v_lex = lexical.generate_embeddings(submission_paths)
    np.save(LEXICAL_CACHE_DIR / "v_lex.npy", v_lex)

    # gst: compute the pairwise Greedy String Tiling coverage matrix, the strongest single signal
    logger.info("Computing GST coverage...")
    gst_coverage = strmatch.compute_gst_coverage(submission_paths)
    np.save(GST_CACHE_DIR / "gst_coverage.npy", gst_coverage)

    # fusion: combine the topological, lexical, and GST similarity networks into one fused matrix, via
    # either SNF's cross-diffusion or noisy-OR's independent-evidence combination (see fusion.py docstrings
    # for the trade-off: SNF averages/smooths across channels, noisy-OR preserves and boosts the strongest
    # signal but is more sensitive to correlated channel noise on unrelated pairs)
    logger.info(f"Fusing similarity networks ({fusion_method})...")
    if fusion_method == 'snf':
        fused_matrix = fusion.fuse_similarity_networks([v_topo, v_lex, gst_coverage])
    elif fusion_method == 'noisy_or':
        fused_matrix = fusion.fuse_noisy_or([v_topo, v_lex, gst_coverage])
    else:
        raise ValueError(f"Unknown fusion_method: {fusion_method!r} (expected 'snf' or 'noisy_or')")
    np.save(results_dir / "S_fused.npy", fused_matrix)

    # clustering: run HDBSCAN as the primary clustering method, since its density adapts locally across the
    # graph and it can leave non-colluding submissions unlabeled instead of forcing them into a cluster
    logger.info("Clustering fused network with HDBSCAN...")
    hdbscan_clusters = hdbscan_cluster.run_hdbscan(fused_matrix, submission_paths)
    hdbscan_output_path = results_dir / "clusters_hdbscan.json"
    with open(hdbscan_output_path, "w") as f:
        json.dump(hdbscan_clusters, f, indent=4)

    # comparison: also run Leiden/CPM as a comparison point against HDBSCAN, per the dissertation's cluster quality benchmarking
    logger.info("Clustering fused network with Leiden/CPM...")
    leiden_clusters = leiden.run_leiden(fused_matrix, submission_paths)
    leiden_output_path = results_dir / "clusters_leiden.json"
    with open(leiden_output_path, "w") as f:
        json.dump(leiden_clusters, f, indent=4)

    logger.info(f"Pipeline complete! HDBSCAN clusters saved to {hdbscan_output_path}")
    logger.info(f"Pipeline complete! Leiden/CPM clusters saved to {leiden_output_path}")

    return hdbscan_clusters, leiden_clusters


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the full SNF + HDBSCAN/Leiden pipeline across multiple roots")
    parser.add_argument("root_dirs", nargs='+', help="One or more root directories containing submissions")
    parser.add_argument("--fusion-method", choices=['snf', 'noisy_or'], default='snf', help="Similarity fusion method: SNF cross-diffusion (default) or noisy-OR independent-evidence combination")
    parser.add_argument("--output-dir", default=None, help="Where to write paths.json/S_fused.npy/clusters_*.json/graph_failures.json (e.g. a pre-created bind-mount target). Defaults to the CACHE_DIR used for CPG/lexical/GST caching if omitted.")
    args = parser.parse_args()

    run_pipeline(args.root_dirs, fusion_method=args.fusion_method, output_dir=args.output_dir)
