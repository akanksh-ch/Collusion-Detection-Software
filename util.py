import subprocess
from pathlib import Path
from os import access as perm
from os import R_OK

def get_ram():
    # Get ram info (joern is ram-heavy)
    ram_avail = 0

    with open('/proc/meminfo', 'read') as meminfo:
    for line in meminfo:
        if line.startswith('MemAvailable:'):
            ram_avail = int(line.split()[1]) # Returns in kB
            ram_avail = ram_avail / 1024 # Returns in mb which can be used such as -Xmx5120M

    return ram_avail

def get_cpu():
    # Cpu (core) count
    return int(subprocess.check_output(["nproc", "--all"]).decode().strip()) - 1 # C'mon man poor CPU

def validate_path(path):
   # Checking if path is valid
    if not Path(path).exists():
        raise FileNotFoundError(f"File or Directory not found: {str(Path(path).resolve())}")

    # Checking if it's readable
    if not perm(Path(path), R_OK):
        raise PermissionError(f"Could not read: {str(Path(path).resolve())}")

