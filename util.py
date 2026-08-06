import subprocess

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
