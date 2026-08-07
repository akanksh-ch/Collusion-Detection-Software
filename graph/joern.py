"""
This file contains code to generate CPGs using Joern
"""

import subprocess
from sys import stdout

def generate_cpg(path: str, ram_mb: int):
    # Pass the calculated RAM down to the JVM, with a hard minimum of 128MB just in case
    ram_flag = f'-J-Xmx{max(128, ram_mb)}M' 
    
    parse_result = subprocess.run(
        ['joern-parse', ram_flag, path, '-o', f"{path}.bin"],
        stderr=stdout,
        text=True
    )

    export_result = subprocess.run(
        ['joern-export', ram_flag, '--repr', 'cpg', f"{path}.bin", '-o', f"{path}_export"],
        stderr=stdout,
        text=True
    )
