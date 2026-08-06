import networkx as nx
from platformdirs import user_cache_dir # Storing CPGs and temp stuff in userdirs
import subprocess
from pathlib import Path
from os import access as perm
from os import R_OK
from concurrent.futures import ThreadPoolExecutor

CACHE_DIR = user_cache_dir('cds', ensure_exists=True)

def generate_graph(path: str): # file or directory
    
    # Checking if path is valid
    if not Path(path).exists():
        raise FileNotFoundError(f"File or Directory not found: {str(Path(path).resolve())}")

    # Checking if it's readable
    if not perm(Path(path), R_OK):
        raise PermissionError(f"Could not read: {str(Path(path).resolve())}")
