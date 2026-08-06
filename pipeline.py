"""
Orchestrator script to run the whole pipeline
"""

from platformdirs import user_cache_dir # Storing CPGs and temp stuff in userdirs
from concurrent.futures import ThreadPoolExecutor
import subprocess

from graph import generate_graph

CACHE_DIR = user_cache_dir('cds', ensure_exists=True)

"""
commands = []

with ThreadPoolExecutor(max_workers=2) as executor:
futures = {
    executor.submit(
        subprocess.run,
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    ): cmd
    for cmd in commands
}

for future in as_completed(futures):
    cmd = futures[future]
    completed = future.result()

    if completed.returncode == 0:
        print(f"{cmd} succeeded")
    else:
        print(f"{cmd} failed ({completed.returncode})")
        print(completed.stderr)
"""

def run_pipeline():
    print("Hello from pipeline")

    # generate_graph_embeddings(cache_dir)
    # generate_tfidfvectors(cache_dir)
    # generate_gst_coverage(cache_dir?)

    # snfpy(graph, tfidf, gst)
    # leiden(snfpy)
