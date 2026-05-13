#!/bin/bash

# Use the first argument as the target directory, or default to current directory
TARGET_DIR="${1:-.}"

# Find all .c files recursively and process them
find "$TARGET_DIR" -type f -name "*.c" | while read -r src_file; do
    # 1. Define paths
    # path/file.c -> path/file.bin (temp) -> path/file.bag-of-nodes
    temp_bin="${src_file%.c}.bin"
    out_json="${src_file%.c}.bag-of-nodes"
    
    echo "Processing: $src_file"
    
    # 2. Generate the CPG binary
    # We need the binary first because Joern's analysis engine operates on it.
    c2cpg.sh "$src_file" --output "$temp_bin" > /dev/null 2>&1
    
    # 3. Use Joern to export node counts to JSON
    # We use --execute to run a one-line Scala command without opening the interactive shell
    joern --execute "importCpg(\"$temp_bin\"); val counts = cpg.node.label.groupCount.toJson; os.write.over(os.Path(\"$out_json\"), counts); close" > /dev/null 2>&1
    
    # 4. Clean up the .bin file to save space
    rm "$temp_bin"
    
    echo "Successfully exported: $out_json"
done

echo "Done. All Fibonacci samples have been converted to Bag of Nodes format."