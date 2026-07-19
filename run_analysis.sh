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
assignments=$(find submissions/orig -maxdepth 1 -mindepth 1 -type d -exec basename {} \; | cut -d'-' -f1 | sort -u)

if [ -z "$assignments" ]; then
    echo "Error: No assignments discovered inside submissions/orig/"
    exit 1
fi

for assignment in $assignments; do
    echo "=================================================="
    echo "Processing Assignment Group: $assignment"
    echo "=================================================="

    # 3. Dynamically locate the correct original base-code folder for THIS specific assignment iteration
    base_code_dir=$(find submissions/orig -maxdepth 1 -mindepth 1 -type d -name "${assignment}-*" | head -n 1)

    if [ -z "$base_code_dir" ]; then
        echo "Warning: No original base-code folder found for $assignment. Skipping run."
        continue
    fi

    echo "Using Base Code: $base_code_dir"

    # 4. Build the multi-src argument array dynamically using shell expansion
    # This targets only directories belonging to the current assignment (e.g., o1-*)
    src_args=()
    for dir in submissions/orig/${assignment}-*/ submissions/plag/${assignment}-*/; do
        if [ -d "$dir" ]; then
            src_args+=("--src=$dir")
        fi
    done

    # 5. Run the multi-modal detection tool directly on the original folder structure
    python main.py \
        "${src_args[@]}" \
        -r="output/result-${assignment}.jplag" \
        --sim-floor 0.8 \
        -t 9 \
        --overwrite \
        --base-code="$base_code_dir"

    echo -e "Finished processing $assignment\n"
done

echo "All JPlag pipeline runs completed successfully!"
