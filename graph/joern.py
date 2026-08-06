"""
This file contains code to generate CPGs using Joern
"""

import subprocess
from 

# joern-parse -J-Xmx2048M student01-non-plagiarized.java -o cpg.bin
# joern-export -J-Xmx2048M --repr cpg cpg.bin -o test

def generate_cpg(path: str):
    parse_result = subprocess.run(
            ['joern-parse', '-J-Xmx2048M', path, '-o', f"{path}.bin"],
        stderr=subprocess.STDOUT
        text=True  # Returns strings instead of bytes
    )

    export_result = subprocess.run(
            ['joern-export', '-J-Xmx2048M', '--repr', 'cpg', f"{path}.bin", '-o', f"{path}_export"]
        stderr=subprocess.STDOUT
        text=True  # Returns strings instead of bytes
    )
