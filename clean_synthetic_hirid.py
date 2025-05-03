"""
clean_synthetic_hirid.py

Utility script to clean up the output directories used during synthetic HiRID dataset
generation for the TRACE model. This script removes any previously generated data
to ensure a clean environment before running data generation again.

Directories removed:
- synthetic_hirid_dataset: stores `.pt` files (train/val tensors and labels)
- DONTCOMMITdata/hirid_numpy: stores `.npy` files used by TRACE training pipeline
"""

import shutil
import os

# List of directories to remove before regeneration
dirs_to_clear = ["synthetic_hirid_dataset", "DONTCOMMITdata/hirid_numpy", "DONTCOMMITplots/HiRID", "ckpt/HiRID/"]

# Remove each directory if it exists
for path in dirs_to_clear:
    if os.path.exists(path):
        print(f"🧹 Removing directory: {path}")
        shutil.rmtree(path)

# Optionally print confirmation
print("✅ Cleanup complete. Removed:", dirs_to_clear)
