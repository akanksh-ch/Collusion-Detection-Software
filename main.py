import logging
import argparse
import multiprocessing
from parse import JoernAutomationParser
from loader import CPGDataLoader
from analyser import CohortAnalyzer
from generate_report import HTMLReportGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Enforce safe multiprocessing start methods across Linux/macOS environments
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Enterprise Scale CPG Collusion Pipeline Architecture")
    parser.add_argument("--src", type=str, default="./submissions", help="Folder containing raw code files")
    parser.add_argument("--out", type=str, default="output/", help="Target output destination directory")
    parser.add_argument("--skip-parse", action="store_true", help="Skip Joern step and run analytics on graph cache")
    args = parser.parse_args()

    workspace_dir = "./joern_workspace_exports"

    print("\n=== Launching High-Performance Concurrency Detection Pipeline ===\n")

    if not args.skip_parse:
        logging.info("Execution Step 1: Launching Multiprocessed Joern Extraction Engines...")
        j_parser = JoernAutomationParser()
        parse_success = j_parser.process_submission_folder(source_dir=args.src, output_dir=workspace_dir)
    else:
        logging.info("Execution Step 1: [SKIPPED] Ingesting structural graph cache.")
        parse_success = True

    logging.info("Execution Step 2: Running parallel schema discovery and streaming pass...")
    loader = CPGDataLoader(input_dir=workspace_dir)
    loader.discover_vocabularies()  
    cohort_graphs = loader.load_cohort(executed_successfully=parse_success)

    logging.info("Execution Step 3: Extracting Weisfeiler-Lehman topological color fingerprints...")
    analyzer = CohortAnalyzer(
        cohort_data=cohort_graphs,
        node_vocab_size=len(loader.node_vocab),
        edge_vocab_size=len(loader.edge_vocab)
    )
    analyzer.generate_all_embeddings()

    logging.info("Execution Step 4: Applying precomputed HDBSCAN clustering distance maps...")
    families = analyzer.extract_solution_families(min_cluster_size=2)

    logging.info("Execution Step 5: Constructing family-aware suspicion density tables...")
    flagged_pairs = analyzer.compute_suspicion_scores(families)

    logging.info(f"Execution Step 6: Exporting comprehensive anomaly dashboard to layout: {args.out}")
    HTMLReportGenerator.write_report(flagged_pairs, families, output_path=args.out)
    
    print("\n=== Pipeline Execution Finalized Successfully ===")
    logging.info(f"Assembled successfully. Inspect target path '{args.out}' for results.")

if __name__ == "__main__":
    main()
