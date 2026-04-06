"""
merge_results.py
Merges all POCA output files for a single geometry (one per seed) into a
single accumulated result. Launched automatically by SLURM via --dependency=afterok.
"""

import numpy as np
import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Merge POCA results from all seeds for a given geometry.")
parser.add_argument("--namefile",          required=True,  help="Geometry name (without extension).")
parser.add_argument("--n_jobs",            required=True,  type=int, help="Number of seeds to merge.")
parser.add_argument("--npx",              required=True,  type=int)
parser.add_argument("--npy",              required=True,  type=int)
parser.add_argument("--npz",              required=True,  type=int)
parser.add_argument("--path_poca_output", required=True,  help="Directory with per-seed POCA .npy files.")
parser.add_argument("--output",           required=True,  help="Output .npy file for the merged result.")
args = parser.parse_args()


print(f"[INFO] Merging {args.n_jobs} POCA files for: {args.namefile}")

grid_N_total      = np.zeros((args.npy, args.npx, args.npz))
grid_sum_total    = np.zeros((args.npy, args.npx, args.npz))
grid_sum_sq_total = np.zeros((args.npy, args.npx, args.npz))

missing = []
for seed in range(args.n_jobs):
    filepath = os.path.join(args.path_poca_output, f"POCA_{args.namefile}_seed{seed}.npy")

    if not os.path.exists(filepath):
        print(f"[WARNING] Missing file for seed={seed}: {filepath}")
        missing.append(seed)
        continue

    data = np.load(filepath, allow_pickle=True).item()

    if data["n_events"].shape != (args.npy, args.npx, args.npz):
        print(f"[ERROR] Shape mismatch at seed={seed}: {data['n_events'].shape} != {(args.npy, args.npx, args.npz)}")
        sys.exit(1)

    grid_N_total      += data["n_events"]
    grid_sum_total    += data["sum_theta"]
    grid_sum_sq_total += data["sum_theta_sq"]

if missing:
    print(f"[WARNING] {len(missing)} files missing: seeds {missing}")
else:
    print(f"[CORRECT] All {args.n_jobs} files merged successfully.")

print(f"[INFO] Total muon events accumulated: {int(grid_N_total.sum()):,}")

np.save(args.output, {
    "n_events":     grid_N_total,
    "sum_theta":    grid_sum_total,
    "sum_theta_sq": grid_sum_sq_total
})
print(f"[CORRECT] Merged result saved to: {args.output}")