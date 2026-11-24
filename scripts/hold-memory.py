import argparse
import os
import time

import numpy as np

# Simulates a machine with less free memory: holds incompressible memory and keeps it in use,
# so the OS cannot keep large weight files in its file cache. Stop with Ctrl-C or kill.
parser = argparse.ArgumentParser()
parser.add_argument("--gb", type=float, required=True)
args = parser.parse_args()

page = os.sysconf("SC_PAGE_SIZE")
# Random bytes, so macOS memory compression cannot shrink them.
block = np.random.default_rng(0).integers(0, 256, int(args.gb * 1e9), dtype=np.uint8)
print(f"holding {block.nbytes / 1e9:.1f} GB", flush=True)
while True:
    # Read one byte per page so the pages stay recently used.
    int(block[::page].sum())
    time.sleep(5)
