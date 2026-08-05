#!/usr/bin/env bash
# End-to-end pipeline run: encode -> cluster -> evaluate -> visualize.
#
# Run 1 uses sim_floor=0.0 so every pair is scored — this is the run
# AUC/PR-AUC and the embedding export must come from, since any nonzero
# floor silently drops pairs before compute_suspicion_scores ever sees
# them, biasing the metric (see prior discussion).
#
# Run 2 uses a real sim_floor + leiden_res tuned for clustering quality,
# not evaluation fairness — its cluster.json is for the ARI/NMI/purity
# numbers and the write-up figure, not for AUC.
set -euo pipefail

IMAGE="cds:latest"
SUB_DIR="$(pwd)/submissions"
OUT_DIR="$(pwd)/output"
mkdir -p "$OUT_DIR"

docker_run() {
    docker run --rm \
        -v "${SUB_DIR}:/app/submissions:z" \
        -v "${OUT_DIR}:/app/output:z" \
        "${IMAGE}" "$@"
}

echo "=== [0/6] Rebuilding image ==="
# Any script in the repo root (analyser.py, hdbscan_cluster.py, etc.) only
# exists inside the container if it was present at the last build. If you
# add/edit a script and don't rebuild, you get a confusing "can't open
# file" error from inside the container instead of a clear "stale image"
# message — so rebuild every run rather than assuming the image is current.
docker build -t "${IMAGE}" .

echo "=== [1/6] Encoding + all-pairs scoring (sim-floor=0.0) ==="
docker_run python main.py \
    --src=submissions/orig --src=submissions/plag \
    -r=output/result-all-eval.jplag \
    --sim-floor 0.0 -t 9 --leiden-res 0.8 --overwrite --csv-export \
    --export-embeddings output/embeddings.npz

echo "=== [2/6] Leiden clustering (sim-floor=0.6, leiden-res=1.4) ==="
docker_run python main.py \
    --src=submissions/orig --src=submissions/plag \
    -r=output/result-all-cluster.jplag \
    --sim-floor 0.6 -t 9 --overwrite --leiden-res 0.8 --csv-export \

echo "=== Extracting cluster.json from the .jplag reports ==="
# .jplag files are zip archives; the report viewer's cluster overview is
# bundled inside. Adjust the glob below if generate_report.py names it
# differently in your build.
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

echo "=== [3/6] HDBSCAN baseline clustering (embedding-only, no sim-floor) ==="
# hdbscan_cluster.py shares gst_hdbscan_cluster.py's CLI shape: positional
# arg is the all-pairs eval CSV (used to enumerate the submission/name
# universe), embeddings come in via --embeddings, and --alpha 0.0 forces
# pure embedding-based clustering (0.0 = embedding only, per its own
# docstring convention) — matching the "HDBSCAN (v_program only)" row
# from the original comparison table.
docker_run python hdbscan_cluster.py \
    output/result-all-eval.csv \
    --embeddings output/embeddings.npz --space v_program --alpha 0.5 \
    --out output/hdbscan_cluster.json

echo "=== [4/6] Running real JPlag (v6.3.0, JDK 25) ==="
# --mode run is NOT optional here: without it JPlag launches its bundled
# report-viewer as a local web server and blocks forever waiting for a
# browser, which just hangs a headless container. -new merges both source
# roots into one comparison universe (matches main.py comparing orig+plag
# together above). -l/--language defaults to java — CHANGE THIS if your
# submissions aren't Java, JPlag won't auto-detect it.
docker_run /opt/temurin25/bin/java -jar /opt/jplag/jplag.jar \
    --mode run \
    -new=submissions/orig,submissions/plag \
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
# JPlag 6.3.0 confirmed behavior: --csv-export with -r output/jplag_result
# creates a DIRECTORY at output/jplag_result/ (not a flat file next to the
# .jplag archive), containing results.csv — and that file's header is
# already submissionName1,submissionName2,averageSimilarity,maxSimilarity,
# the exact schema evaluate_pipeline.py's --jplag-csv expects. So no
# conversion step is needed (convert_jplag_csv.py existed only for the
# case where JPlag wrote a similarity matrix instead). We still resolve
# the path via find rather than hardcoding it a second time, so this
# keeps working if a future JPlag version renames results.csv.
jplag_csv=$(find "${OUT_DIR}/jplag_result" -maxdepth 1 -iname "*.csv" | head -n1)
if [ -z "${jplag_csv}" ]; then
    echo "ERROR: no .csv found in ${OUT_DIR}/jplag_result — check what "
    echo "  --csv-export actually wrote (ls output/jplag_result/) and "
    echo "  adjust the path/glob above to match."
    exit 1
fi
jplag_csv_rel="output/jplag_result/$(basename "${jplag_csv}")"
echo "  using ${jplag_csv_rel}"

echo "=== [5/6] AUC / PR-AUC / clustering evaluation — Leiden vs JPlag ==="
docker_run python evaluate_pipeline.py \
    --pipeline-csv output/result-all-eval.csv \
    --pipeline-cluster output/result-all-cluster_cluster.json \
    --jplag-csv "${jplag_csv_rel}" \
    --jplag-cluster output/jplag_result_cluster.json

echo "=== [6/6] Clustering evaluation — HDBSCAN(+GST embeddings) vs JPlag ==="
# --pipeline-cluster is the thing being evaluated (HDBSCAN here), and
# --jplag-cluster is JPlag's own clustering — these were swapped in an
# earlier ad-hoc run, which silently compared Leiden against HDBSCAN
# instead of HDBSCAN against JPlag. Pairwise AUC is skipped here
# (--jplag-csv omitted) since it would just reprint identical numbers to
# the call above — same source CSV, same scores.
docker_run python evaluate_pipeline.py \
    --pipeline-csv output/result-all-eval.csv \
    --pipeline-cluster output/hdbscan_cluster.json \
    --jplag-cluster output/jplag_result_cluster.json

echo "=== Visualization: t-SNE / UMAP ==="
docker_run python visualise_embeddings.py output/embeddings.npz --out output/plots/

echo ""
echo "Done. Outputs in ${OUT_DIR}:"
echo "  result-all-eval.csv        — all-pairs scores (AUC/PR-AUC source)"
echo "  result-all-cluster.csv     — floor-restricted scores (clustering source)"
echo "  result-all-cluster_cluster.json — Leiden communities"
echo "  hdbscan_cluster.json       — HDBSCAN embedding-only baseline communities"
echo "  jplag_result.jplag         — real JPlag report archive"
echo "  jplag_result_cluster.json  — JPlag's own clustering"
echo "  ${jplag_csv_rel##*/}       — JPlag pairwise scores (used as-is, no conversion)"
echo "  embeddings.npz             — v_topo / v_text / v_program vectors"
echo "  plots/                     — t-SNE + UMAP PNGs"
