#!/bin/bash

# Check if a directory was provided, otherwise use current directory
TARGET_DIR="${1:-.}"

# Find all .c files recursively
find "$TARGET_DIR" -type f -name "*.c" | while read -r src_file; do
    # Generate the output path by replacing .c with .ll
    out_file="${src_file%.c}.ll"
    
    echo "Compiling: $src_file -> $out_file"
    
    # Run the Clang command
    clang -S -emit-llvm "$src_file" -o "$out_file"
done

echo "Done."
