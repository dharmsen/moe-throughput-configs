import re
import sys
import numpy as np

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <log_file>")
    sys.exit(1)

log_file = sys.argv[1]

# Regex patterns
tflops_pattern = re.compile(r"throughput per GPU \(TFLOP/s/GPU\): ([0-9.]+)")
container_pattern = re.compile(r"Using container:\s*(\S+)")
nodes_pattern = re.compile(r"SLURM_NNODES:\s*(\d+)")
elapsed_pattern = re.compile(r"elapsed time per iteration \(ms\): ([0-9.]+)")

try:
    with open(log_file, "r") as f:
        content = f.read()
except FileNotFoundError:
    print(f"Error: File not found: {log_file}")
    sys.exit(1)

# Extract data
tflops = [float(x) for x in tflops_pattern.findall(content)]
elapsed_times = [float(x) for x in elapsed_pattern.findall(content)]

container_match = container_pattern.search(content)
nodes_match = nodes_pattern.search(content)

container = container_match.group(1) if container_match else "Not found"
nodes = int(nodes_match.group(1)) if nodes_match else "Not found"

# Print results
print(f"Container: {container}")
print(f"SLURM_NNODES: {nodes}")

if tflops:
    arr = np.array(tflops)
    print(f"\nTFLOP/s/GPU statistics:")
    print(f"  Count: {len(arr)}")
    print(f"  Mean: {arr.mean():.3f}")
    print(f"  Median: {np.median(arr):.3f}")
    print(f"  Min: {arr.min():.3f}")
    print(f"  Max: {arr.max():.3f}")
    print(f"  Std Dev: {arr.std(ddof=1):.3f}")
else:
    print("\nNo TFLOP/s/GPU data found.")

if elapsed_times:
    et = np.array(elapsed_times)
    print(f"\nElapsed time per iteration (ms):")
    print(f"  Count: {len(et)}")
    print(f"  Average: {et.mean():.3f}")
else:
    print("\nNo elapsed time per iteration data found.")
