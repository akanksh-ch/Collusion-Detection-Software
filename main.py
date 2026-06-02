import logging
import argparse
from parse import JoernAutomationParser
from loader import CPGDataLoader
from analyser import CohortAnalyzer
from generate_report import HTMLReportGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Core Framework - End-to-End CPG Collusion Detection Pipeline")
    parser.add_argument("--src", type=str, default="./submissions", help="Folder containing raw student source code")
    parser.add_argument("--out", type=str, default="output/", help="Target filename or directory path for the HTML dashboard")
    parser.add_argument("--skip-parse", action="store_true", help="Skip Joern compilation and use existing GraphML exports")
    args = parser.parse_args()

    workspace_dir = "./joern_workspace_exports"

    print("\n=== Launching Collusion Detection System Pipeline ===\n")

    if not args.skip_parse:
        logging.info(f"Execution Step 1: Invoking Joern parser on source directory: {args.src}")
        j_parser = JoernAutomationParser()
        parse_success = j_parser.process_submission_folder(source_dir=args.src, output_dir=workspace_dir)
    else:
        logging.info("Execution Step 1: [SKIPPED] Ingesting pre-existing structural cache archives.")
        parse_success = True

    # Step 2: Run discovery pass to dynamically configure dimensions before loading
    logging.info("Execution Step 2: Ingesting GraphML elements into PyG tensor arrays...")
    loader = CPGDataLoader(input_dir=workspace_dir)
    
    # --- CRITICAL PASS ACTION ADDED ---
    loader.discover_vocabularies() 
    
    cohort_graphs = loader.load_cohort(executed_successfully=parse_success)

    # Step 3: Instantiate analyzer using dynamic vocabulary array shapes
    logging.info("Execution Step 3: Extracting hybrid multi-relational embeddings...")
    analyzer = CohortAnalyzer(
        cohort_data=cohort_graphs,
        node_vocab_size=len(loader.node_vocab),
        edge_vocab_size=len(loader.edge_vocab)
    )
    analyzer.generate_all_embeddings()

    # Step 4: Group the entire student cohort into structural solution families using HDBSCAN
    logging.info("Execution Step 4: Applying hierarchical HDBSCAN logic to isolate copy lineages...")
    families = analyzer.extract_solution_families(min_cluster_size=2)

    # Step 5: Perform local risk scoring relative to the isolated family baselines
    logging.info("Execution Step 5: Executing family-aware suspicion metrics calculation...")
    flagged_pairs = analyzer.compute_suspicion_scores(families)

    # Step 6: Render results into your final visual instructor dashboard
    logging.info(f"Execution Step 6: Exporting localized anomaly structures to report layout: {args.out}")
    HTMLReportGenerator.write_report(flagged_pairs, families, output_path=args.out)
    
    print("\n=== Pipeline Execution Finalized Successfully ===")
    logging.info(f"Dashboard assembly complete. Verify directory path '{args.out}' for outputs.")

if __name__ == "__main__":
    main()
