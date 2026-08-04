"""End-to-end pipeline orchestrator.

Wires up all four backend modules with correct data handoff:
  1. parse.py    — comment stripping + Joern compilation
  2. loader.py   — template masking, slicing, casing, tensorisation
  3. gnn.py       — dual-track encoding + Gated Multimodal Fusion
  4. analyser.py — HNSW search, disaggregation, GST, Leiden
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
        "--src", type=str, action="append", default=None,
        help="Root directory with submissions to check for plagiarism. Specify multiple times for multiple roots.",
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
        help="Select the language of the submissions (default: java).",
    )
    parser.add_argument(
        "-m", "--similarity-threshold", type=float, default=0.0,
        dest="sim_threshold",
        help="Comparison similarity threshold [0.0-1.0].",
    )
    parser.add_argument(
        "-t", "--min-tokens", type=int, default=9,
        dest="gst_min_tile",
        help="Tunes the comparison sensitivity.",
    )
    parser.add_argument(
        "-n", "--shown-comparisons", type=int, default=2500,
        dest="shown_comparisons",
        help="The maximum number of comparisons shown in report.",
    )
    parser.add_argument(
        "-s", "--subdirectory", type=str, default=None,
        dest="subdirectory",
        help="Look in directories <root-dir>/*/<dir> for programs.",
    )

    # ── Advanced ─────────────────────────────────────────────────────
    parser.add_argument("--csv-export", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-parse", action="store_true")
    parser.add_argument("--no-slice", action="store_true")
    parser.add_argument("--sim-floor", type=float, default=0.85)
    parser.add_argument("--leiden-res", type=float, default=1.4)
    parser.add_argument("--match-merging", action="store_true", default=True)
    parser.add_argument("--no-match-merging", action="store_false", dest="match_merging")
    parser.add_argument("--gap-size", type=int, default=6)
    parser.add_argument("--neighbor-length", type=int, default=2)
    parser.add_argument("--cluster-skip", action="store_true")
    parser.add_argument(
        "--cluster-method", type=str, default="leiden",
        choices=["leiden", "hdbscan"],
        help="Family/community assignment method (default: leiden). "
             "'hdbscan' clusters graph2vec + TF-IDF stylometric "
             "embeddings instead of the fused GST/embedding graph. "
             "GST computation and the pairwise report are IDENTICAL "
             "either way — only family assignment changes. Requires "
             "karateclub to be installed.",
    )
    parser.add_argument("--hdbscan-structural-weight", type=float, default=0.7,
        help="Weight for graph2vec vectors when --cluster-method=hdbscan (default 0.7).")
    parser.add_argument("--hdbscan-stylometric-weight", type=float, default=0.3,
        help="Weight for TF-IDF stylometric vectors when --cluster-method=hdbscan (default 0.3).")
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=2,
        help="min_cluster_size for HDBSCAN when --cluster-method=hdbscan (default 2).")
    parser.add_argument(
        "--export-embeddings", type=str, default=None,
        metavar="PATH",
        help="Dump per-submission v_topo/v_text/v_program/v_g2v/v_stylo vectors "
             "to PATH (.npz) for offline analysis (e.g. HDBSCAN, t-SNE/UMAP).",
    )
    parser.add_argument(
        "--no-structural-stylo", action="store_true",
        help="Skip graph2vec + TF-IDF embedding generation even when "
             "--export-embeddings is set (they're otherwise generated "
             "automatically alongside v_topo/v_text/v_program). Useful "
             "if karateclub isn't installed or you just want the "
             "original three spaces faster.",
    )

    args = parser.parse_args()

    # Fallback to defaults if no custom paths are appended
    if not args.src:
        args.src = ["./submissions"]

    if not args.out.endswith(".jplag"):
        args.out += ".jplag"

    if os.path.exists(args.out) and not args.overwrite:
        if os.path.isfile(args.out):
            pass

    if args.subdirectory:
        args.src = [os.path.join(path, "*", args.subdirectory) for path in args.src]

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
        comments_dir = JoernAutomationParser.comments_dir(workspace_dir)
        if not os.path.isdir(comments_dir) or not os.listdir(comments_dir):
            logging.info("Step 1a: Generating comment sidecar files...")
            j_parser.preprocess_submissions(args.src, workspace_dir)
        parse_success = True

    # ── Step 2: Vocabulary discovery + graph loading ──────────────────
    logging.info("Step 2: Loading and processing code property graphs...")

    skeleton_graphml_path = args.skeleton
    if args.skeleton:
        if args.skeleton.endswith(".graphml"):
            skeleton_graphml_path = args.skeleton
        else:
            base_name = os.path.basename(args.skeleton.rstrip("/\\"))
            stem, _ = os.path.splitext(base_name)
            candidate_graphml = os.path.join(workspace_dir, f"{stem}.graphml")

            if os.path.exists(candidate_graphml):
                logging.info(f"[MASKING] Found pre-compiled GraphML for skeleton: {candidate_graphml}")
                skeleton_graphml_path = candidate_graphml
            else:
                logging.info(f"[MASKING] Raw skeleton detected: '{args.skeleton}'. Compiling on the fly...")
                skel_stripped_dir = os.path.join(workspace_dir, "_stripped_skeleton")
                skel_comments_dir = os.path.join(workspace_dir, "_comments_skeleton")
                os.makedirs(skel_stripped_dir, exist_ok=True)
                os.makedirs(skel_comments_dir, exist_ok=True)

                from parse import _preprocess_single_submission_worker, _parse_single_submission

                prep_args = (args.skeleton, skel_stripped_dir, skel_comments_dir, "skeleton")
                stem_id, prep_status = _preprocess_single_submission_worker(prep_args)

                if prep_status == "OK":
                    stripped_skel_path = os.path.join(skel_stripped_dir, "skeleton")

                    parse_args = (
                        stripped_skel_path,
                        workspace_dir,
                        j_parser.parse_bin,
                        j_parser.export_bin
                    )
                    skel_id, parse_status = _parse_single_submission(parse_args)

                    if parse_status in ("SUCCESS", "CACHED"):
                        skeleton_graphml_path = os.path.join(workspace_dir, f"{skel_id}.graphml")
                        logging.info(f"[MASKING] Successfully compiled skeleton to: {skeleton_graphml_path}")
                    else:
                        logging.error(f"[MASKING] Joern failed to compile skeleton: {parse_status}")
                        skeleton_graphml_path = None
                else:
                    logging.error(f"[MASKING] Preprocessor failed to strip skeleton comments: {prep_status}")
                    skeleton_graphml_path = None

    loader = CPGDataLoader(
        input_dir=workspace_dir,
        skeleton_path=skeleton_graphml_path,
    )
    loader.discover_vocabularies()
    cohort_graphs = loader.load_cohort(
        executed_successfully=parse_success,
        bypass_slicing=args.no_slice,
    )

    logging.info(f"  → Loaded {len(cohort_graphs)} student graphs.")

    # ── Step 3: Generate multi-modal embeddings ───────────────────────
    # analyzer is bound here, and must stay bound before anything below
    # this point references it — the embeddings export block (which
    # calls analyzer.generate_structural_stylometric_embeddings()) has
    # to live AFTER this assignment, never before it in the function
    # body, or Python raises UnboundLocalError at call time.
    logging.info("Step 3: Generating dual-track embeddings + Gated Multimodal Fusion...")
    analyzer = CohortAnalyzer(
        cohort_data=cohort_graphs,
        node_vocab_size=len(loader.node_vocab),
        edge_vocab_size=len(loader.edge_vocab),
        source_dir=args.src,  # Pass the entire list of roots directly
        workspace_dir=workspace_dir,
    )
    analyzer.generate_all_embeddings()

    if args.export_embeddings:
        import numpy as np

        if not args.no_structural_stylo:
            logging.info("Step 3b: Generating graph2vec + TF-IDF stylometric embeddings...")
            try:
                analyzer.generate_structural_stylometric_embeddings()
            except ImportError as e:
                logging.warning(
                    f"[EXPORT] Skipping structural/stylometric embeddings — "
                    f"missing dependency ({e}). v_topo/v_text/v_program will "
                    f"still be exported. Install karateclub to enable this."
                )
            except Exception as e:
                logging.error(
                    f"[EXPORT] generate_structural_stylometric_embeddings() failed "
                    f"({e}) — continuing with v_topo/v_text/v_program only."
                )

        student_ids = analyzer.student_ids
        export_payload = {"student_ids": np.array(student_ids, dtype=object)}
        for space, attr in (
            ("v_topo", "vectors_topo"),
            ("v_text", "vectors_text"),
            ("v_program", "vectors_program"),
            ("v_g2v", "vectors_g2v"),
            ("v_stylo", "vectors_stylo"),
        ):
            vec_dict = getattr(analyzer, attr, None)
            if not vec_dict:
                continue

            # v_g2v / v_stylo can legitimately cover fewer submissions
            # than student_ids (empty CPGs, empty stripped source) —
            # unlike v_topo/v_text/v_program, which every submission
            # gets by construction. Only emit a companion "<space>_ids"
            # array when coverage is partial, so hdbscan_cluster.py's
            # loader (which falls back to "student_ids" when no
            # "<space>_ids" key exists) keeps working unchanged for the
            # always-complete spaces.
            ids_present = [sid for sid in student_ids if sid in vec_dict]
            if len(ids_present) < len(student_ids):
                logging.warning(
                    f"[EXPORT] '{space}' only covers {len(ids_present)}/"
                    f"{len(student_ids)} submissions — see prior warnings "
                    f"for which ones were dropped and why"
                )
                export_payload[f"{space}_ids"] = np.array(ids_present, dtype=object)

            if ids_present:
                export_payload[space] = np.stack([vec_dict[sid] for sid in ids_present])

        np.savez_compressed(args.export_embeddings, **export_payload)
        logging.info(
            f"[EXPORT] Saved {len(student_ids)} submission embeddings "
            f"→ {args.export_embeddings}"
        )

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
        cluster_method=args.cluster_method,
        hdbscan_structural_weight=args.hdbscan_structural_weight,
        hdbscan_stylometric_weight=args.hdbscan_stylometric_weight,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
    )

    # ── Step 5: Compute suspicion scores ──────────────────────────────
    logging.info("Step 5: Computing ranked suspicion scores...")
    flagged_pairs = analyzer.compute_suspicion_scores(families)

    if args.sim_threshold > 0.0:
        flagged_pairs = [p for p in flagged_pairs if p["similarity"] >= args.sim_threshold]

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
