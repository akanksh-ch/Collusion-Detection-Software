#!/bin/bash

# Use the first argument as the target directory, or default to current directory
TARGET_DIR="${1:-.}"

# Find all .c files recursively
find "$TARGET_DIR" -type f -name "*.c" | while read -r src_file; do
    # Define output path: xyz/abc.c -> xyz/abc.json
    out_json="${src_file%.c}.bag-of-nodes.json"
    temp_script="temp_extract.sc"
    
    echo "------------------------------------------------"
    echo "Direct Normalization: $src_file"
    
    # 1. Create a script that imports the C code directly into Joern
    # importCode handles the c2cpg conversion internally
    cat <<EOF > "$temp_script"
@main def main() = {
    // importCode creates the CPG in-memory/workspace
    importCode("$src_file")
    
    // Generate the Bag of Nodes
    val counts = cpg.all.label.groupCount
    println(ujson.write(counts))
    
    // Delete the project from the workspace to keep it clean
    delete
}
EOF

    # 2. Run Joern and capture only the JSON result
    json_data=$(joern --script "$temp_script" 2>/dev/null | grep '^{.*}$')
    
    if [ -n "$json_data" ]; then
        echo "$json_data" > "$out_json"
        echo "Exported: $out_json"
    else
        echo "Error: Failed to process $src_file"
    fi
    
    # 3. Only cleanup the tiny temporary script
    rm -f "$temp_script"
done

echo "------------------------------------------------"
echo "Done. Bag of Nodes generated"