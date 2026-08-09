"""
This file contains code to generate CPGs using Joern
"""

import subprocess
from pathlib import Path
from sys import stdout


def generate_cpg(path: str, ram_mb: int = 128, out_dir: str | None = None) -> str:
    # setup: pass the calculated RAM down to the JVM (hard minimum of 128MB just in case), and
    # resolve where the .bin/_export artifacts land — a caller-supplied cache dir if given,
    # otherwise fall back to the submission's own parent dir for standalone/backwards-compat use
    ram_flag = f'-J-Xmx{max(128, ram_mb)}M'
    base_dir = Path(out_dir) if out_dir else Path(path).parent
    base_dir.mkdir(parents=True, exist_ok=True)

    name = Path(path).name
    bin_path = base_dir / f"{name}.bin"
    export_dir = base_dir / f"{name}_export"

    # parsing: joern-parse turns the raw submission into a CPG binary
    subprocess.run(
        ['joern-parse', ram_flag, path, '-o', str(bin_path)],
        stderr=stdout,
        text=True
    )

    # exporting: joern-export flattens the CPG binary into .dot files under export_dir
    subprocess.run(
        ['joern-export', ram_flag, '--repr', 'cpg', str(bin_path), '-o', str(export_dir)],
        stderr=stdout,
        text=True
    )

    return str(export_dir)
