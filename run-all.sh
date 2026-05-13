#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if the source directory argument is provided
if [ -z "$1" ]; then
    echo "Error: No source directory provided."
    echo "Usage: $0 <path_to_student_code_folder>"
    exit 1
fi

# Convert the input to an absolute path for reliability
TARGET_DIR=$(realpath "$1")

# Identify the root directory of the project (where this script resides)
ROOT_DIR=$(dirname "$(realpath "$0")")

echo "------------------------------------------------"
echo "Starting Collusion Detection Pipeline"
echo "Target Directory: $TARGET_DIR"
echo "------------------------------------------------"

# 1. Generate Code Property Graphs
echo "[1/4] Generating CPGs..."
bash "$ROOT_DIR/convert-to-cpg/generate-c2cpg.sh" "$TARGET_DIR"

# 2. Run Bag-of-Nodes Shell Normalization
echo "[2/4] Normalizing nodes (Shell)..."
bash "$ROOT_DIR/normalisation/bag-of-nodes.sh" "$TARGET_DIR"

# 3. Run Bag-of-Nodes Python Adapter
# Note: Assumes the 'adapters' folder exists in your project root
echo "[3/4] Adapting data for clustering (Python)..."
python3 "$ROOT_DIR/adapters/bag-of-nodes.py" "$TARGET_DIR"

# 4. Run DBSCAN Clustering
# Note: Based on your tree, dbscan.py is likely inside clustering/dbscan/
echo "[4/4] Running DBSCAN clustering..."
python3 "$ROOT_DIR/clustering/dbscan/dbscan.py" "$TARGET_DIR"

echo "------------------------------------------------"
echo "Pipeline complete. Results generated for: $TARGET_DIR"
echo "------------------------------------------------"