import os
from pathlib import Path

# Path to the folder containing the explanation masks
MASK_DIR = Path("/DATA/hdhanu/GNN/Explainability/w5_g1000_m0/explanations")

total_bytes = 0
file_count = 0
for f in MASK_DIR.glob("*.npz"):
    try:
        total_bytes += f.stat().st_size
        file_count += 1
    except FileNotFoundError:
        pass  # in case file is removed during iteration

total_mb = total_bytes / (1024 ** 2)
total_gb = total_bytes / (1024 ** 3)

print(f"Folder: {MASK_DIR}")
print(f"Number of mask files: {file_count}")
print(f"Total size: {total_mb:.2f} MB ({total_gb:.2f} GB)")
