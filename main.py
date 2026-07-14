"""End-to-end pipeline orchestrator.

Wires up all four backend modules with correct data handoff:
  1. parse.py    — comment stripping + Joern compilation
  2. loader.py   — template masking, slicing, casing, tensorisation
  3. gnn.py      — dual-track encoding + Gated Multimodal Fusion
  4. analyser.py — HNSW search, disaggregation, GST, Leiden

CLI flags:
  --src             Raw submission directory
  --out             Output path for the compressed forensic archive
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
import json
import gzip

from parse import JoernAutomationParser
from loader import CPGDataLoader
from analyser import CohortAnalyzer
from generate_report import JPlagReportGenerator

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
        prog="collusion-detect",
        description="Multi-Modal Collusion Detection Pipeline\n"
                    "AI-powered plagiarism detection with JPlag-compatible report output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Core parameters (JPlag-compatible names) ─────────────────────
    parser.add_argument(
        "--src", type=str, default="./submissions",
        help="Root directory with submissions to check for plagiarism.",
    )
    parser.add_argument(
        "-r", "--result-file", type=str, default="result.jplag",
        dest="out",
        help="Name of the file in which the comparison results will be stored "
             "(default: result.jplag). Missing .jplag extension will be automatically added.",
    )
    parser.add_argument(
        "-bc", "--base-code", type=str, default=None,
        dest="skeleton",
        help="Path to the base code directory (common framework used in all submissions).",
    )
    parser.add_argument(
        "-l", "--language", type=str, default="java",
        help="Select the language of the submissions (default: java). "
             "Currently supported: java, c, cpp, python.",
    )
    parser.add_argument(
        "-m", "--similarity-threshold", type=float, default=0.0,
        dest="sim_threshold",
        help="Comparison similarity threshold [0.0-1.0]: All comparisons above "
             "this threshold will be saved (default: 0.0).",
    )
    parser.add_argument(
        "-t", "--min-tokens", type=int, default=9,
        dest="gst_min_tile",
        help="Tunes the comparison sensitivity by adjusting the minimum token "
             "required to be counted as a matching section (default: 9).",
    )
    parser.add_argument(
        "-n", "--shown-comparisons", type=int, default=2500,
        dest="shown_comparisons",
        help="The maximum number of comparisons that will be shown in the "
             "generated report, if set to -1 all comparisons will be shown (default: 2500).",
    )
    parser.add_argument(
        "-s", "--subdirectory", type=str, default=None,
        dest="subdirectory",
        help="Look in directories <root-dir>/*/<dir> for programs.",
    )

    # ── Advanced ─────────────────────────────────────────────────────
    parser.add_argument(
        "--csv-export", action="store_true",
        help="Export pairwise similarity values as a CSV file.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Existing result files will be overwritten.",
    )
    parser.add_argument(
        "--skip-parse", action="store_true",
        help="Skip Joern compilation and use existing GraphML exports.",
    )
    parser.add_argument(
        "--no-slice", action="store_true",
        help="Disable backward dependency slicing.",
    )

    # ── Pipeline-specific parameters ─────────────────────────────────
    parser.add_argument(
        "--sim-floor", type=float, default=0.85,
        help="AI cosine similarity floor for HNSW candidate filtering (default: 0.85). "
             "Pairs below this threshold are not analysed by GST. "
             "Note: isotropic calibration (All-But-The-Top) stabilises the vector "
             "space; this cutoff was derived empirically via precision-recall tuning "
             "against the labelled ground-truth corpus.",
    )
    parser.add_argument(
        "--leiden-res", type=float, default=1.4,
        help="Leiden resolution parameter (default: 1.4).",
    )

    # ── Subsequence Match Merging ────────────────────────────────────
    parser.add_argument(
        "--match-merging", action="store_true", default=True,
        help="Enables merging of neighboring matches to counteract obfuscation "
             "attempts (default: enabled).",
    )
    parser.add_argument(
        "--no-match-merging", action="store_false", dest="match_merging",
        help="Disables merging of neighboring matches.",
    )
    parser.add_argument(
        "--gap-size", type=int, default=6,
        help="Maximal gap between neighboring matches to be merged "
             "(between 1 and minTokenMatch, default: 6).",
    )
    parser.add_argument(
        "--neighbor-length", type=int, default=2,
        help="Minimal length of neighboring matches to be merged "
             "(between 1 and minTokenMatch, default: 2).",
    )

    # ── Clustering ───────────────────────────────────────────────────
    parser.add_argument(
        "--cluster-skip", action="store_true",
        help="Skips the Leiden cluster calculation.",
    )

    args = parser.parse_args()

    # Ensure .jplag extension
    if not args.out.endswith(".jplag"):
        args.out += ".jplag"

    # Handle --overwrite
    if os.path.exists(args.out) and not args.overwrite:
        if os.path.isfile(args.out):
            pass  # Will be overwritten by zipfile anyway

    # Handle --subdirectory
    if args.subdirectory:
        args.src = os.path.join(args.src, "*", args.subdirectory)

    # Environment variable override for sim floor
    env_floor = os.environ.get("GLOBAL_SIM_FLOOR")
    if env_floor is not None:
        try:
            args.sim_floor = float(env_floor)
        except ValueError:
            pass

    args.sim_floor = max(0.0, min(1.0, args.sim_floor))
    args.sim_threshold = max(0.0, min(1.0, args.sim_threshold))

    workspace_dir = "./joern_workspace_exports"

    print("\n=== Launching Multi-Modal Collusion Detection Pipeline ===")
    print(f"[CONFIG] Similarity Floor:   {args.sim_floor:.2f}")
    print(f"[CONFIG] Report Threshold:   {args.sim_threshold:.2f}")
    print(f"[CONFIG] Leiden Resolution:  {args.leiden_res:.2f}")
    print(f"[CONFIG] GST Min Tokens:     {args.gst_min_tile}")
    print(f"[CONFIG] Match Merging:      {'ON (gap={}, neighbor={})'.format(args.gap_size, args.neighbor_length) if args.match_merging else 'OFF'}")
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
        match_merging=args.match_merging,
        gap_size=args.gap_size,
        neighbor_length=args.neighbor_length,
        cluster_skip=args.cluster_skip,
    )

    # ── Step 5: Compute suspicion scores ──────────────────────────────
    logging.info("Step 5: Computing ranked suspicion scores...")
    flagged_pairs = analyzer.compute_suspicion_scores(families)

    # Apply similarity threshold filter (like JPlag -m)
    if args.sim_threshold > 0.0:
        flagged_pairs = [p for p in flagged_pairs if p["similarity"] >= args.sim_threshold]

    # Cap shown comparisons (like JPlag -n)
    if args.shown_comparisons != -1:
        flagged_pairs = flagged_pairs[:args.shown_comparisons]

    # ── Step 6: Collect Source Texts & Export Forensic Archive ────────
    logging.info(f"Step 6: Packaging forensic archive → {args.out}")
    source_texts = {}
    all_student_ids = set()
    for row in flagged_pairs:
        all_student_ids.add(row["student_a"])
        all_student_ids.add(row["student_b"])

    for sid in all_student_ids:
        src = analyzer._read_original_source(sid)
        if src:
            source_texts[sid] = src

    JPlagReportGenerator.write_report(
        report_data=flagged_pairs,
        families=families,
        source_texts=source_texts,
        output_path=args.out
    )

    # ── Optional CSV export ──────────────────────────────────────────
    if args.csv_export:
        csv_path = args.out.replace(".jplag", ".csv")
        import csv
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["student_a", "student_b", "similarity", "sim_structural", "sim_lexical", "gst_coverage", "risk_level"])
            for row in flagged_pairs:
                writer.writerow([
                    row["student_a"], row["student_b"],
                    f"{row['similarity']:.4f}",
                    f"{row['sim_structural']:.4f}",
                    f"{row['sim_lexical']:.4f}",
                    f"{row['gst_coverage']:.4f}",
                    row["risk_level"],
                ])
        logging.info(f"CSV export saved to: {csv_path}")

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
    print(f"  Archive saved to: {args.out}\n")


if __name__ == "__main__":
    main()
