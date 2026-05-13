#!/bin/bash

# Target the root folder of your solutions
BASE_DIR="fibonacci_c_solutions"

# Find all .c files recursively and process them
find "$BASE_DIR" -type f -name "*.c" | while read -r src_file; do
    # Define the output path (e.g., binet/fib_binet_1.c -> binet/fib_binet_1.bin)
    out_file="${src_file%.c}.bin"
    
    echo "Processing: $src_file"
    
    # Direct invocation since it's in your PATH
    joern-c2cpg "$src_file" --output "$out_file"
done

echo "CPG generation complete."
