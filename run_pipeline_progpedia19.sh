#!/usr/bin/env bash
# End-to-end pipeline run for the ProgPedia19 corpus: encode -> cluster ->
# evaluate -> visualize.
#
# DIFFERENCES FROM THE ZENODO ("Criminal Minds") run_pipeline.sh:
#   1. ProgPedia19 is a single flat directory (gpt5-subm95/, plag-subm136/,
#      subm183/, ... all siblings) rather than Zenodo's split orig/plag
#      dirs, so --src / -new below use one root instead of two.
#   2. Ground truth lineage is encoded as "subm<N>" in every submission
#      name (original, traditionally-plagiarized, and GPT-obfuscated
#      derivatives all share the same N) — so evaluate_pipeline.py is
#      called with --family-pattern 'subm(\d+)' instead of its default
#      Zenodo-shaped 'o(\d+)(?![a-z])'. This is a CLI override only;
#      no code changes to evaluate_pipeline.py are needed, since
#      extract_family()/use_family_gt already generalizes to any corpus
#      where every name resolves to a lineage id.
#   3. --positive-keyword / --baseline-keyword / --negative-pattern are
#      left at their defaults but are effectively unused here, since
#      every ProgPedia19 name resolves via --family-pattern and
#      use_family_gt short-circuits before the keyword heuristic ever
#      runs. Only relevant if some ProgPedia19 submissions turn out
#      NOT to contain "subm<N>" at all — see the note at the bottom.
#
# Run 1 uses sim_floor=0.0 so every pair is scored — this is the run
# AUC/PR-AUC and the embedding export must come from, since any nonzero
# floor silently drops pairs before compute_suspicion_scores ever sees
# them, biasing the metric.
#
# Run 2 uses a real sim_floor + leiden_res tuned for clustering quality,
# not evaluation fairness — its cluster.json is for the ARI/NMI/purity
# numbers and the write-up figure, not for AUC.
set -euo pipefail

IMAGE="cds:latest"
SUB_DIR="$(pwd)/submissions"
OUT_DIR="$(pwd)/output_progpedia19"
# joern_workspace_exports holds the parsed GraphML cache (main.py's
# workspace_dir, default "./joern_workspace_exports" relative to /app
# inside the container). Every docker_run() call below is --rm, so
# without mounting this too, each container starts with an EMPTY
# workspace dir and --skip-parse has nothing to find — this is why
# --skip-parse silently failed to load the cache last time. Mounting it
# here makes the parse survive across the separate docker_run() calls
# in this script.
WORKSPACE_DIR="$(pwd)/joern_workspace_exports_progpedia19"
mkdir -p "$OUT_DIR" "$WORKSPACE_DIR"

# Adjust this if ProgPedia19 lives under a different folder name inside
# submissions/ — must be a single directory containing all the
# gpt5-subm*/plag-subm*/subm* siblings directly.
CORPUS_SRC="submissions/"

docker_run() {
    docker run --rm \
        -v "${SUB_DIR}:/app/submissions:z" \
        -v "${OUT_DIR}:/app/output:z" \
        -v "${WORKSPACE_DIR}:/app/joern_workspace_exports:z" \
        "${IMAGE}" "$@"
}

echo "=== [0/6] Rebuilding image ==="
docker build -t "${IMAGE}" .

echo "=== [1/6] Encoding + all-pairs scoring (sim-floor=0.0, --no-slice) ==="
# --shown-comparisons -1: main.py's -n/--shown-comparisons defaults to
# 2500 (it's a REPORT-DISPLAY cap — "max comparisons shown in report" —
# not an evaluation setting) and, left unset, silently truncates
# flagged_pairs to the top-2500-by-score BEFORE the CSV is written.
# That's what caused the 2500-pair Pipeline eval vs 8911-pair JPlag eval
# mismatch: JPlag's own C(134,2)=8911 was correct, ours was a biased
# top-scoring subsample. -1 is the sentinel that disables the cap
# entirely (see main.py: `if args.shown_comparisons != -1`) — same
# category of bug as the --sim-floor fix, just on the report-count knob
# instead of the score-floor knob.
# --no-slice: testing whether backward-slicing was stripping the
# idiosyncratic noise (dead code, unused vars, odd unreachable
# branches) that graph2vec needs to distinguish "copied-then-edited"
# from "independently convergent" submissions. NOTE: this flag is
# global to loader.load_cohort(), so it also changes what v_topo/
# v_text/v_program (the GNN fusion track) see, not just graph2vec —
# it is NOT an isolated graph2vec-only test. If you want to isolate
# graph2vec specifically while keeping the GNN track sliced as before,
# that needs a second unsliced cohort loaded separately inside
# generate_structural_stylometric_embeddings() instead of this flag —
# ask if you want that written instead.
docker_run python main.py \
    --src="${CORPUS_SRC}" \
    -r=output/result-all-eval.jplag \
    --sim-floor 0.0 -t 9 --leiden-res 0.8 --overwrite --csv-export \
    --no-slice --shown-comparisons -1 \
    --export-embeddings output/embeddings.npz

echo "=== [2/6] HDBSCAN clustering (graph2vec + stylometric, sim-floor=0.6, --no-slice, --skip-parse) ==="
# --skip-parse: reuse the GraphML cache from Step [1/6] (now persisted
# via WORKSPACE_DIR mount above) instead of re-running Joern. --no-slice
# must match Step [1/6] — mixing sliced/unsliced across runs against the
# same cache would silently produce inconsistent cohorts.
docker_run python main.py \
    --src="${CORPUS_SRC}" \
    -r=output/result-all-cluster.jplag \
    --sim-floor 0.6 -t 9 --overwrite --leiden-res 0.8 --csv-export \
    --no-slice --skip-parse \
    --cluster-method hdbscan

echo "=== Extracting cluster.json from the .jplag reports ==="
for tag in eval cluster; do
    archive="${OUT_DIR}/result-all-${tag}.jplag"
    extract_dir="${OUT_DIR}/_extract_${tag}"
    rm -rf "${extract_dir}"
    mkdir -p "${extract_dir}"
    unzip -q "${archive}" -d "${extract_dir}"
    found=$(find "${extract_dir}" -iname "*cluster*.json" | head -n1)
    if [ -n "${found}" ]; then
        cp "${found}" "${OUT_DIR}/result-all-${tag}_cluster.json"
        echo "  ${tag}: cluster.json -> result-all-${tag}_cluster.json"
    else
        echo "  ${tag}: WARNING — no *cluster*.json found inside ${archive}"
    fi
done

echo "=== [3/6] HDBSCAN baseline clustering (GST + v_program blend, alpha=0.5) ==="
# Reads embeddings.npz from Step [1/6] (already --no-slice) — no
# main.py invocation here, so no --skip-parse/--no-slice flags needed.
docker_run python hdbscan_cluster.py \
    output/result-all-eval.csv \
    --embeddings output/embeddings.npz --space v_program --alpha 0.5 \
    --out output/hdbscan_cluster.json

echo "=== [3b/6] HDBSCAN — graph2vec structural embeddings only ==="
docker_run python hdbscan_cluster.py \
    output/result-all-eval.csv \
    --embeddings output/embeddings.npz --space v_g2v --alpha 0.0 \
    --out output/hdbscan_g2v_cluster.json

echo "=== [3c/6] HDBSCAN — TF-IDF stylometric embeddings only ==="
docker_run python hdbscan_cluster.py \
    output/result-all-eval.csv \
    --embeddings output/embeddings.npz --space v_stylo --alpha 0.0 \
    --out output/hdbscan_stylo_cluster.json

echo "=== [4/6] Running real JPlag (v6.3.0, JDK 25) ==="
# -new takes a SINGLE root here (one flat corpus dir), not two comma-
# separated roots like the Zenodo orig/plag split.
docker_run /opt/temurin25/bin/java -jar /opt/jplag/jplag.jar \
    --mode run \
    -new="${CORPUS_SRC}" \
    -l java \
    -r output/jplag_result \
    -t 9 --match-merging --gap-size 6 --neighbor-length 2 \
    --csv-export --overwrite

echo "=== Extracting cluster.json from the JPlag report ==="
archive="${OUT_DIR}/jplag_result.jplag"
extract_dir="${OUT_DIR}/_extract_jplag"
rm -rf "${extract_dir}"
mkdir -p "${extract_dir}"
unzip -q "${archive}" -d "${extract_dir}"
found=$(find "${extract_dir}" -iname "*cluster*.json" | head -n1)
if [ -n "${found}" ]; then
    cp "${found}" "${OUT_DIR}/jplag_result_cluster.json"
    echo "  jplag: cluster.json -> jplag_result_cluster.json"
else
    echo "  jplag: WARNING — no *cluster*.json found inside ${archive}"
fi

echo "=== Locating JPlag's --csv-export output ==="
jplag_csv=$(find "${OUT_DIR}/jplag_result" -maxdepth 1 -iname "*.csv" | head -n1)
if [ -z "${jplag_csv}" ]; then
    echo "ERROR: no .csv found in ${OUT_DIR}/jplag_result — check what "
    echo "  --csv-export actually wrote (ls output_progpedia19/jplag_result/) "
    echo "  and adjust the path/glob above to match."
    exit 1
fi
jplag_csv_rel="output/jplag_result/$(basename "${jplag_csv}")"
echo "  using ${jplag_csv_rel}"

echo "=== [5/6] AUC / PR-AUC / clustering evaluation — HDBSCAN(g2v+stylo, via main.py --cluster-method hdbscan) vs JPlag ==="
# NOTE: this reads result-all-cluster_cluster.json, produced by Step [2/6]
# above, which now runs --cluster-method hdbscan (graph2vec + TF-IDF
# stylometric weighted combo, NOT Leiden — update this label if you ever
# switch Step [2/6] back).
docker_run python evaluate_pipeline.py \
    --pipeline-csv output/result-all-eval.csv \
    --pipeline-cluster output/result-all-cluster_cluster.json \
    --jplag-csv "${jplag_csv_rel}" \
    --jplag-cluster output/jplag_result_cluster.json \
    --family-pattern 'subm(\d+)'

echo "=== [6/6] Clustering evaluation — HDBSCAN(GST-distance + v_program blend, alpha=0.5) vs JPlag ==="
# NOTE: this is the ORIGINAL hdbscan_cluster.py baseline from Step [3/6]
# (GST pairwise distance blended 50/50 with v_program cosine distance) —
# it has nothing to do with graph2vec or TF-IDF. See [6b/6]/[6c/6] below
# for the graph2vec-only and stylo-only comparisons.
docker_run python evaluate_pipeline.py \
    --pipeline-csv output/result-all-eval.csv \
    --pipeline-cluster output/hdbscan_cluster.json \
    --jplag-cluster output/jplag_result_cluster.json \
    --family-pattern 'subm(\d+)'

echo "=== [6b/6] Clustering evaluation — HDBSCAN(graph2vec only, alpha=0.0) vs JPlag ==="
docker_run python evaluate_pipeline.py \
    --pipeline-csv output/result-all-eval.csv \
    --pipeline-cluster output/hdbscan_g2v_cluster.json \
    --jplag-cluster output/jplag_result_cluster.json \
    --family-pattern 'subm(\d+)'

echo "=== [6c/6] Clustering evaluation — HDBSCAN(TF-IDF stylometric only, alpha=0.0) vs JPlag ==="
docker_run python evaluate_pipeline.py \
    --pipeline-csv output/result-all-eval.csv \
    --pipeline-cluster output/hdbscan_stylo_cluster.json \
    --jplag-cluster output/jplag_result_cluster.json \
    --family-pattern 'subm(\d+)'

echo "=== Visualization: t-SNE / UMAP ==="
docker_run python visualise_embeddings.py output/embeddings.npz --out output/plots/

echo ""
echo "Done. Outputs in ${OUT_DIR}:"
echo "  result-all-eval.csv        — all-pairs scores (AUC/PR-AUC source)"
echo "  result-all-cluster.csv     — floor-restricted scores (clustering source)"
echo "  result-all-cluster_cluster.json — HDBSCAN(g2v+stylo) families [Step 2/6, --cluster-method hdbscan]"
echo "  hdbscan_cluster.json       — HDBSCAN(GST-distance + v_program blend, alpha=0.5) baseline [Step 3/6]"
echo "  hdbscan_g2v_cluster.json   — HDBSCAN(graph2vec only, alpha=0.0) [Step 3b/6]"
echo "  hdbscan_stylo_cluster.json — HDBSCAN(TF-IDF stylometric only, alpha=0.0) [Step 3c/6]"
echo "  jplag_result.jplag         — real JPlag report archive"
echo "  jplag_result_cluster.json  — JPlag's own clustering"
echo "  ${jplag_csv_rel##*/}       — JPlag pairwise scores (used as-is, no conversion)"
echo "  embeddings.npz             — v_topo / v_text / v_program / v_g2v / v_stylo vectors"
echo "  plots/                     — t-SNE + UMAP PNGs"
echo ""
echo "NOTE: if any ProgPedia19 submission name does NOT contain 'subm<N>',"
echo "  it will fail to resolve a family id, use_family_gt will become False"
echo "  for the WHOLE corpus (it requires every name to resolve), and ground"
echo "  truth silently falls back to the plag/non-plag keyword heuristic —"
echo "  check for a 'clustering evaluation' line saying 'plag/non-plag' "
echo "  instead of 'origin-family' in the output above if numbers look off."
