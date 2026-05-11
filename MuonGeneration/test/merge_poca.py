# -*- coding: utf-8 -*-
"""
merge_results_1.py — Merges POCA output .root files and computes 2D channel maps.

Called by loop_configuration_files1.py with:
    --namefile         {namefile}
    --n_jobs           {n_jobs_per_geometry}
    --npx              {npx}
    --npy              {npy}
    --npz              {npz}
    --dimension        {dimension}
    --path_poca_output {PATH_poca_output}
    --output           {out_merged}

INPUT:  .root files at {path_poca_output}/POCA_{namefile}_seed*.root
        produced by POCA1.py, each containing a TTree "events" with branches:
        theta, poca_x, poca_y, poca_z, is_parallel

OUTPUT: .npy file (path given by --output) with a dict of 2D channel maps
        (shape: npy, npx, 1):

    CHANNEL 1 — counts_all_2d     : all events per XY cell (MTR proxy — body shape)
    CHANNEL 2 — counts_scat_2d    : scattered events per XY cell
    CHANNEL 3 — mean_theta_sq_2d  : mean(theta²) of scattered events per XY cell
    CHANNEL 4 — top3_theta_sq_2d  : mean of top-3 theta² per XY cell (radiation length proxy)
    CHANNEL 5 — std_z_2d          : std(poca_z) of scattered events per XY cell (thickness proxy)
"""

print("[INFO] Iniciando merge_results_1...")
import os
import sys
import argparse
import numpy as np
import glob

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import ROOT
ROOT.gROOT.SetBatch(True)
print("[CORRECT] Libraries imported.")

# ---------------------------------------------------------------------------
# Argument parsing — matches loop_configuration_files1.py exactly
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Merge POCA .root files and compute 2D channel maps.")
parser.add_argument("--namefile",         required=True,  help="Geometry name (used to glob POCA files).")
parser.add_argument("--n_jobs",           required=True,  type=int, help="Expected number of POCA jobs (for validation).")
parser.add_argument("--npx",              type=int,   default=128,   help="Number of voxels in X.")
parser.add_argument("--npy",              type=int,   default=128,   help="Number of voxels in Y.")
parser.add_argument("--npz",              type=int,   default=128,   help="Number of voxels in Z (unused in 2D, kept for compatibility).")
parser.add_argument("--Lpx",              type=float, default=128.0, help="Physical length in X [cm].")
parser.add_argument("--Lpy",              type=float, default=128.0, help="Physical length in Y [cm].")
parser.add_argument("--Lpz",              type=float, default=128.0, help="Physical length in Z [cm].")
parser.add_argument("--dimension",        type=str,   default="2D",  choices=["2D", "3D"], help="Dimensionality.")
parser.add_argument("--path_poca_output", required=True,  help="Directory where POCA .root files are stored.")
parser.add_argument("--output",           required=True,  help="Full path for the output .npy file.")
args = parser.parse_args()

X_LIM = args.Lpx / 2.0
Y_LIM = args.Lpy / 2.0
Z_LIM = args.Lpz / 2.0
NX, NY = args.npx, args.npy

print(f"[INFO] Geometry  : {args.namefile}")
print(f"[INFO] Grid      : {NX} x {NY}")
print(f"[INFO] Volume XY : [{-X_LIM},{X_LIM}] x [{-Y_LIM},{Y_LIM}] cm")
print(f"[INFO] Dimension : {args.dimension}")

# ---------------------------------------------------------------------------
# Find and validate input .root files
# ---------------------------------------------------------------------------
pattern = os.path.join(args.path_poca_output, f"POCA_{args.namefile}_seed*.root")
input_files = sorted(glob.glob(pattern))

if len(input_files) == 0:
    print(f"[ERROR] No POCA .root files found matching: {pattern}")
    sys.exit(1)

if len(input_files) != args.n_jobs:
    print(f"[WARNING] Expected {args.n_jobs} POCA files, found {len(input_files)}. Proceeding anyway.")
else:
    print(f"[CORRECT] Found {len(input_files)} POCA files (matches n_jobs={args.n_jobs}).")

# ---------------------------------------------------------------------------
# Load all .root files into a single RDataFrame
# ---------------------------------------------------------------------------
df = ROOT.RDataFrame("events", input_files)
total_events = df.Count().GetValue()
print(f"[INFO] Total events loaded: {total_events:,}")

if total_events == 0:
    print("[ERROR] No events found in input files. Aborting.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Compute XY voxel indices (projection: integrate over Z)
# ---------------------------------------------------------------------------
df = df.Define("voxel_x", f"int(fmin({NX}-1, fmax(0, (poca_x + {X_LIM}) / (2*{X_LIM} / {NX}))))") \
       .Define("voxel_y", f"int(fmin({NY}-1, fmax(0, (poca_y + {Y_LIM}) / (2*{Y_LIM} / {NY}))))")

# ---------------------------------------------------------------------------
# Export to numpy
# ---------------------------------------------------------------------------
print("[INFO] Extracting arrays from RDataFrame...")
res = df.AsNumpy(columns=["theta", "poca_z", "voxel_x", "voxel_y", "is_parallel"])

v_x      = res["voxel_x"].astype(int)
v_y      = res["voxel_y"].astype(int)
theta    = res["theta"].astype(np.float32)
poca_z   = res["poca_z"].astype(np.float32)
is_par   = res["is_parallel"].astype(bool)
theta_sq = theta ** 2

mask_scat = ~is_par
mask_par  =  is_par

print(f"[INFO] Scattered events : {mask_scat.sum():,} ({100*mask_scat.mean():.1f}%)")
print(f"[INFO] Parallel events  : {mask_par.sum():,}  ({100*mask_par.mean():.1f}%)")

# ---------------------------------------------------------------------------
# Initialize accumulators
# ---------------------------------------------------------------------------
m_counts_all   = np.zeros((NY, NX), dtype=np.float32)  # CH1
m_counts_scat  = np.zeros((NY, NX), dtype=np.float32)  # CH2
m_sum_theta_sq = np.zeros((NY, NX), dtype=np.float32)  # → CH3
m_sum_z        = np.zeros((NY, NX), dtype=np.float32)  # → CH5 mean
m_sum_z_sq     = np.zeros((NY, NX), dtype=np.float32)  # → CH5 variance

# CH1 — all events
np.add.at(m_counts_all, (v_y, v_x), 1)

# CH2, CH3, CH5 — scattered only
np.add.at(m_counts_scat,  (v_y[mask_scat], v_x[mask_scat]), 1)
np.add.at(m_sum_theta_sq, (v_y[mask_scat], v_x[mask_scat]), theta_sq[mask_scat])
np.add.at(m_sum_z,        (v_y[mask_scat], v_x[mask_scat]), poca_z[mask_scat])
np.add.at(m_sum_z_sq,     (v_y[mask_scat], v_x[mask_scat]), poca_z[mask_scat] ** 2)

# ---------------------------------------------------------------------------
# Derived channels
# ---------------------------------------------------------------------------

# CH3 — mean(theta²) per XY cell
with np.errstate(invalid="ignore"):
    m_mean_theta_sq = np.where(m_counts_scat > 0, m_sum_theta_sq / m_counts_scat, 0.0).astype(np.float32)

# CH4 — mean of top-3 theta² per XY cell (robust radiation length proxy)
print("[INFO] Computing top-3 theta² per XY cell...")
m_top3_theta_sq = np.zeros((NY, NX), dtype=np.float32)

cell_idx        = v_y[mask_scat] * NX + v_x[mask_scat]
order           = np.argsort(cell_idx)
cell_idx_sorted = cell_idx[order]
theta_sq_sorted = theta_sq[mask_scat][order]

unique_cells, cell_starts = np.unique(cell_idx_sorted, return_index=True)
cell_ends = np.append(cell_starts[1:], len(cell_idx_sorted))

for uid, start, end in zip(unique_cells, cell_starts, cell_ends):
    vals = theta_sq_sorted[start:end]
    top3 = np.sort(vals)[-3:]
    iy, ix = divmod(int(uid), NX)
    m_top3_theta_sq[iy, ix] = top3.mean()

# CH5 — std(poca_z) per XY cell
with np.errstate(invalid="ignore"):
    mean_z = np.where(m_counts_scat > 0, m_sum_z / m_counts_scat, 0.0)
    var_z  = np.where(m_counts_scat > 0, m_sum_z_sq / m_counts_scat - mean_z ** 2, 0.0)
    var_z  = np.maximum(var_z, 0.0)
    m_std_z = np.sqrt(var_z).astype(np.float32)

# ---------------------------------------------------------------------------
# Pack into output dict  (shape: NY, NX, 1)
# ---------------------------------------------------------------------------
def to_channel(m):
    return m[:, :, np.newaxis].astype(np.float32)

output_dict = {
    "counts_all_2d"    : to_channel(m_counts_all),    # CH1 — body shape (MTR)
    "counts_scat_2d"   : to_channel(m_counts_scat),   # CH2 — scattered muon density
    "mean_theta_sq_2d" : to_channel(m_mean_theta_sq), # CH3 — scattering density
    "top3_theta_sq_2d" : to_channel(m_top3_theta_sq), # CH4 — radiation length proxy
    "std_z_2d"         : to_channel(m_std_z),          # CH5 — material thickness proxy
}

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
np.save(args.output, output_dict)
print(f"[CORRECT] Channels saved to: {args.output}")

for key, val in output_dict.items():
    print(f"  {key:25s} shape={val.shape}  max={val.max():.4f}  nonzero={np.count_nonzero(val)}")

print("[CORRECT] merge_results_1 finished.")