"""End-to-end pipeline orchestrator.

Wires up all five modules with correct data handoff:
  1. parse.py   — comment stripping + Joern compilation
  2. loader.py  — template masking, slicing, casing, tensorisation
  3. gnn.py     — dual-track encoding + Gated Multimodal Fusion
  4. analyser.py — HNSW search, disaggregation, GST, Leiden
  5. generate_report.py — D3.js interactive forensic dashboard

CLI flags:
  --src             Raw submission directory
  --out             Output path for the HTML report
  --skeleton        Optional instructor boilerplate GraphML for masking
  --skip-parse      Skip Joern compilation, use existing GraphML
  --no-slice        Disable backward dependency slicing
  --sim-floor       Global cosine similarity floor (default 0.85)
  --leiden-res      Leiden resolution parameter (default 1.4)
  --gst-min-tile    GST minimum match length in tokens (default 8)

Environment variable override:
  GLOBAL_SIM_FLOOR  Overrides --sim-floor if set
"""

import logging
import argparse
import multiprocessing
import os

from parse import JoernAutomationParser
from loader import CPGDataLoader
from analyser import CohortAnalyzer
from generate_report import HTMLReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    # Enforce safe multiprocessing start methods across varying OS environments
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Multi-Modal Collusion Detection Pipeline"
    )
    parser.add_argument(
        "--src", type=str, default="./submissions",
        help="Folder containing raw student source code",
    )
    parser.add_argument(
        "--out", type=str, default="output/",
        help="Target path for the HTML dashboard",
    )
    parser.add_argument(
        "--skeleton", type=str, default=None,
        help="Optional instructor boilerplate GraphML for template masking",
    )
    parser.add_argument(
        "--skip-parse", action="store_true",
        help="Skip Joern compilation and use existing GraphML exports",
    )
    parser.add_argument(
        "--no-slice", action="store_true",
        help="Disable backward dependency slicing",
    )
    parser.add_argument(
        "--sim-floor", type=float, default=0.85,
        help="Global cosine similarity floor (default: 0.85)",
    )
    parser.add_argument(
        "--leiden-res", type=float, default=1.4,
        help="Leiden resolution parameter (default: 1.4)",
    )
    parser.add_argument(
        "--gst-min-tile", type=int, default=8,
        help="GST minimum match length in tokens (default: 8)",
    )
    args = parser.parse_args()

    # Environment variable override for sim floor
    env_floor = os.environ.get("GLOBAL_SIM_FLOOR")
    if env_floor is not None:
        try:
            args.sim_floor = float(env_floor)
        except ValueError:
            pass

    args.sim_floor = max(0.0, min(1.0, args.sim_floor))

    workspace_dir = "./joern_workspace_exports"

    print("\n=== Launching Multi-Modal Collusion Detection Pipeline ===")
    print(f"[CONFIG] Similarity Floor:   {args.sim_floor:.2f}")
    print(f"[CONFIG] Leiden Resolution:  {args.leiden_res:.2f}")
    print(f"[CONFIG] GST Min Tile:       {args.gst_min_tile} tokens")
    print(f"[CONFIG] Skeleton Masking:   {'ON → ' + args.skeleton if args.skeleton else 'OFF'}")
    print(f"[CONFIG] Backward Slicing:   {'DISABLED' if args.no_slice else 'ENABLED'}\n")

    # ── Step 1: Preprocess (comment stripping) + Joern Compilation ────
    j_parser = JoernAutomationParser()

    if not args.skip_parse:
        logging.info("Step 1a: Stripping comments from source files...")
        j_parser.preprocess_submissions(args.src, workspace_dir)

        logging.info("Step 1b: Parsing stripped source into GraphML via Joern...")
        stripped_dir = JoernAutomationParser.stripped_source_dir(workspace_dir)
        parse_success = j_parser.process_submission_folder(
            source_dir=stripped_dir, output_dir=workspace_dir
        )
    else:
        logging.info("Step 1: Skipping parse — using existing GraphML cache.")
        # Still run comment stripping if sidecar files don't exist yet
        comments_dir = JoernAutomationParser.comments_dir(workspace_dir)
        if not os.path.isdir(comments_dir) or not os.listdir(comments_dir):
            logging.info("Step 1a: Generating comment sidecar files...")
            j_parser.preprocess_submissions(args.src, workspace_dir)
        parse_success = True

    # ── Step 2: Vocabulary discovery + graph loading ──────────────────
    logging.info("Step 2: Loading and processing code property graphs...")
    loader = CPGDataLoader(
        input_dir=workspace_dir,
        skeleton_path=args.skeleton,
    )
    loader.discover_vocabularies()
    cohort_graphs = loader.load_cohort(
        executed_successfully=parse_success,
        bypass_slicing=args.no_slice,
    )

    logging.info(f"  → Loaded {len(cohort_graphs)} student graphs.")

    # ── Step 3: Generate multi-modal embeddings ───────────────────────
    logging.info("Step 3: Generating dual-track embeddings + Gated Multimodal Fusion...")
    analyzer = CohortAnalyzer(
        cohort_data=cohort_graphs,
        node_vocab_size=len(loader.node_vocab),
        edge_vocab_size=len(loader.edge_vocab),
        source_dir=args.src,
        workspace_dir=workspace_dir,
    )
    analyzer.generate_all_embeddings()

    # ── Step 4: HNSW search + disaggregation + GST + Leiden ──────────
    logging.info("Step 4: HNSW indexing, forensic disaggregation, GST evidence, Leiden clustering...")
    families = analyzer.extract_solution_families(
        sim_floor=args.sim_floor,
        leiden_resolution=args.leiden_res,
        gst_min_match=args.gst_min_tile,
    )

    # ── Step 5: Compute suspicion scores ──────────────────────────────
    logging.info("Step 5: Computing ranked suspicion scores...")
    flagged_pairs = analyzer.compute_suspicion_scores(families)

    # ── Step 6: Collect source texts for the diff sidebar ─────────────
    logging.info("Step 6: Collecting source texts for interactive report...")
    source_texts = {}
    all_student_ids = set()
    for row in flagged_pairs:
        all_student_ids.add(row["student_a"])
        all_student_ids.add(row["student_b"])

    for sid in all_student_ids:
        src = analyzer._read_stripped_source(sid)
        if src:
            source_texts[sid] = src

    # ── Step 7: Generate interactive D3.js report ─────────────────────
    logging.info(f"Step 7: Generating interactive forensic dashboard → {args.out}")
    HTMLReportGenerator.write_report(
        report_data=flagged_pairs,
        families=families,
        source_texts=source_texts,
        output_path=args.out,
    )

    n_critical = sum(1 for p in flagged_pairs if p["risk_level"] == "CRITICAL")
    n_high = sum(1 for p in flagged_pairs if p["risk_level"] == "HIGH")
    n_suspicious = sum(1 for p in flagged_pairs if p["risk_level"] == "SUSPICIOUS")
    n_families = sum(1 for f, m in families.items() if len(m) > 1)

    print("\n=== Pipeline Execution Complete ===")
    print(f"  Flagged pairs:    {len(flagged_pairs)}")
    print(f"  → CRITICAL:       {n_critical}")
    print(f"  → HIGH:           {n_high}")
    print(f"  → SUSPICIOUS:     {n_suspicious}")
    print(f"  Solution families (≥2 members): {n_families}")
    print(f"  Report saved to:  {args.out}\n")


if __name__ == "__main__":
    main()
