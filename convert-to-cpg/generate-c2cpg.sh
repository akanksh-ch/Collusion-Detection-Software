#!/bin/bash

# Use the first argument as the target directory, or default to current directory
TARGET_DIR="${1:-.}"

# Find all .c files recursively and process them
find "$TARGET_DIR" -type f -name "*.c" | while read -r src_file; do
    # Generate the output path in the same location (e.g., path/file.c -> path/file.bin)
    out_file="${src_file%.c}.bin"
    
    echo "Generating CPG: $src_file -> $out_file"
    
    # Execute joern-c2cpg (assumes it is in your PATH)
    c2cpg.sh "$src_file" --output "$out_file"
done

echo "Done. All individual CPGs are ready for analysis."