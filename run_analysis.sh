#!/bin/bash

# 1. Activate the virtual environment
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Error: .venv/bin/activate not found. Please ensure your virtual environment is set up."
    exit 1
fi

# Ensure the final output directory exists
mkdir -p output

# 2. Dynamically extract all unique assignment prefixes (e.g., o1, o2, o10)
# It scans submissions/orig/, grabs everything before the first dash, and keeps unique IDs.
assignments=$(find submissions/orig -maxdepth 1 -mindepth 1 -type d -exec basename {} \; | cut -d'-' -f1 | sort -u)

if [ -z "$assignments" ]; then
    echo "Error: No assignments discovered inside submissions/orig/"
    exit 1
fi

for assignment in $assignments; do
    echo "=================================================="
    echo "Processing Assignment Group: $assignment"
    echo "=================================================="

    # 3. Create an isolated temporary staging directory for this specific assignment run
    STAGE_DIR="output/stage_${assignment}"
    mkdir -p "$STAGE_DIR/orig" "$STAGE_DIR/plag"

    # Use symbolic links to instantly stage matching folders into the runner sandbox
    # This keeps 'orig/' and 'plag/' as the immediate parent folders for relative identity tracking
    cp -as "$(pwd)/submissions/orig/${assignment}-"* "$STAGE_DIR/orig/" 2>/dev/null
    cp -as "$(pwd)/submissions/plag/${assignment}-"* "$STAGE_DIR/plag/" 2>/dev/null

    # 4. Dynamically locate the exact base-code folder inside the staged orig directory
    base_code_dir=$(find "$STAGE_DIR/orig" -maxdepth 1 -mindepth 1 -type d | head -n 1)

    if [ -z "$base_code_dir" ]; then
        echo "Warning: No original base-code folder found for $assignment. Skipping run."
        rm -rf "$STAGE_DIR"
        continue
    fi

    # 5. Run the multi-modal detection tool on the isolated environment
    python main.py \
        --src="$STAGE_DIR" \
        -r="output/result-${assignment}.jplag" \
        --sim-floor 0.8 \
        -t 9 \
        --overwrite \
        --base-code="$base_code_dir"

    echo -e "Finished processing $assignment. Cleaning up temporary stage...\n"
    
    # Clean up the symbolic link stage to keep your workspace pristine
    rm -rf "$STAGE_DIR"
done

echo "All JPlag pipeline runs completed successfully!"
