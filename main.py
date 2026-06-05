import logging
import argparse
import multiprocessing
import os
from parse import JoernAutomationParser
from loader import CPGDataLoader
from analyser import CohortAnalyzer
from generate_report import HTMLReportGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Enforce safe multiprocessing start methods across varying OS environments
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="End-to-End CPG Collusion Detection Pipeline")
    parser.add_argument("--src", type=str, default="./submissions", help="Folder containing raw student source code")
    parser.add_argument("--out", type=str, default="output/", help="Target directory for the HTML dashboard")
    parser.add_argument("--skip-parse", action="store_true", help="Skip Joern compilation and use existing GraphML exports")
    
    # Feature blending and pipeline toggles for configuration control
    parser.add_argument("--alpha", type=float, default=0.5, 
                        help="Weight balance between Topology (1.0) and Style (0.0)")
    parser.add_argument("--no-slice", action="store_true", 
                        help="Completely disable the backward dependency slicing pass")
    args = parser.parse_args()

    # Bound alpha strictly between 0.0 and 1.0
    args.alpha = max(0.0, min(1.0, args.alpha))
    workspace_dir = "./joern_workspace_exports"

    print("\n=== Launching Collusion Detection Pipeline ===")
    print(f"[CONFIG] Structural Topology Weight (Alpha): {args.alpha:.2f}")
    print(f"[CONFIG] Lexical Stylometry Weight: {1.0 - args.alpha:.2f}")
    print(f"[CONFIG] Backward Slicing Enabled: {not args.no_slice}\n")

    # Step 1: Parse raw source code into GraphML if requested
    if not args.skip_parse:
        logging.info("Execution Step 1: Parsing source code into monolithic GraphML representations...")
        j_parser = JoernAutomationParser()
        parse_success = j_parser.process_submission_folder(source_dir=args.src, output_dir=workspace_dir)
    else:
        logging.info("Execution Step 1: Skipping parsing pass. Using existing GraphML cache.")
        parse_success = True

    # Step 2: Dynamic vocabulary discovery and graph loading
    logging.info("Execution Step 2: Extracting structural tokens and loading code graphs...")
    loader = CPGDataLoader(input_dir=workspace_dir)
    loader.discover_vocabularies()  
    cohort_graphs = loader.load_cohort(executed_successfully=parse_success, bypass_slicing=args.no_slice)

    # Step 3: Compute unit-normalized structural/textual fingerprints
    logging.info("Execution Step 3: Generating hybrid vector embeddings...")
    analyzer = CohortAnalyzer(
        cohort_data=cohort_graphs,
        node_vocab_size=len(loader.node_vocab),
        edge_vocab_size=len(loader.edge_vocab)
    )
    analyzer.generate_all_embeddings(alpha=args.alpha)

    # Step 4: Group vector space profiles into solution groups via HDBSCAN
    logging.info("Execution Step 4: Clustering cohort embeddings into solution families...")
    families = analyzer.extract_solution_families(min_cluster_size=2)

    # Step 5: Isolate localized similarities within clusters
    logging.info("Execution Step 5: Computing high-risk pair suspicion metrics...")
    flagged_pairs = analyzer.compute_suspicion_scores(families)

    # Step 6: Render results into the final static instructor dashboard
    logging.info(f"Execution Step 6: Exporting evaluation data to dashboard: {args.out}")
    HTMLReportGenerator.write_report(flagged_pairs, families, output_path=args.out)
    
    print("\n=== Pipeline Execution Finalized Successfully ===")

if __name__ == "__main__":
    main()
