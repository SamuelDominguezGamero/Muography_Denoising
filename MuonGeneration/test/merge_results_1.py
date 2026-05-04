# -*- coding: utf-8 -*-
"""
merge_script.py — Merges POCA output .root files and computes 2D channel maps.

INPUT:  one or more .root files produced by POCA1.py, each containing a TTree
        "events" with branches: theta, poca_x, poca_y, poca_z, is_parallel

OUTPUT: .npy file with a dict containing 2D channel maps (shape: npy, npx, 1):

    CHANNEL 1 — N_counts_all     : number of ALL events per XY cell (MTR proxy — body shape)
    CHANNEL 2 — N_counts_scat    : number of SCATTERED events per XY cell
    CHANNEL 3 — mean_theta_sq    : mean(theta²) of scattered events per XY cell (scattering density)
    CHANNEL 4 — top3_theta_sq    : mean of top-3 theta² per XY cell (radiation length proxy, robust vs outliers)
    CHANNEL 5 — std_z            : std(poca_z) of scattered events per XY cell (material thickness proxy)

"""

print("[INFO] Iniciando merge_script...")
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
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Merge POCA .root files and compute 2D channel maps.")
parser.add_argument("--input",   required=True, help="Input .root file(s). Accepts wildcards, e.g. 'poca_*.root'.")
parser.add_argument("--output",  default="merged_channels", help="Output .npy file (without extension).")
parser.add_argument("--Lpx", type=float, default=128.0, help="Physical length in X [cm].")
parser.add_argument("--Lpy", type=float, default=128.0, help="Physical length in Y [cm].")
parser.add_argument("--Lpz", type=float, default=128.0, help="Physical length in Z [cm].")
parser.add_argument("--npx", type=int, default=128, help="Number of voxels in X.")
parser.add_argument("--npy", type=int, default=128, help="Number of voxels in Y.")
parser.add_argument("--visualization", action="store_true", default=False, help="If set, plot all 2D channel maps.")
args = parser.parse_args()

X_LIM = args.Lpx / 2.0
Y_LIM = args.Lpy / 2.0
Z_LIM = args.Lpz / 2.0
NX, NY = args.npx, args.npy

print(f"[INFO] Grid: {NX} x {NY} | Volume XY: [{-X_LIM},{X_LIM}] x [{-Y_LIM},{Y_LIM}] cm")

# ---------------------------------------------------------------------------
# Load all input files
# ---------------------------------------------------------------------------
input_files = sorted(glob.glob(args.input))
if len(input_files) == 0:
    print(f"[ERROR] No files found matching: {args.input}")
    sys.exit(1)
print(f"[INFO] Found {len(input_files)} input file(s).")

df = ROOT.RDataFrame("events", input_files)
total_events = df.Count().GetValue()
print(f"[INFO] Total events loaded: {total_events}")

# ---------------------------------------------------------------------------
# Compute voxel indices in XY (no Z — we project everything onto XY)
# ---------------------------------------------------------------------------
df = df.Define("voxel_x", f"int(fmin({NX}-1, fmax(0, (poca_x + {X_LIM}) / (2*{X_LIM} / {NX}))))") \
       .Define("voxel_y", f"int(fmin({NY}-1, fmax(0, (poca_y + {Y_LIM}) / (2*{Y_LIM} / {NY}))))")

# ---------------------------------------------------------------------------
# Export to numpy
# ---------------------------------------------------------------------------
print("[INFO] Extracting arrays from RDataFrame...")
res = df.AsNumpy(columns=["theta", "poca_z", "voxel_x", "voxel_y", "is_parallel"])

v_x       = res["voxel_x"].astype(int)
v_y       = res["voxel_y"].astype(int)
theta     = res["theta"].astype(float)
poca_z    = res["poca_z"].astype(float)
is_par    = res["is_parallel"].astype(bool)
theta_sq  = theta ** 2

mask_scat = ~is_par   # scattered muons (denom != 0, real POCA)
mask_par  =  is_par   # parallel muons  (denom ~ 0, fallback t=0)

print(f"[INFO] Scattered events : {mask_scat.sum()} ({100*mask_scat.mean():.1f}%)")
print(f"[INFO] Parallel events  : {mask_par.sum()}  ({100*mask_par.mean():.1f}%)")

# ---------------------------------------------------------------------------
# Initialize 2D matrices  (shape: NY, NX)
# ---------------------------------------------------------------------------
m_counts_all   = np.zeros((NY, NX), dtype=np.float32)  # CH1: all events
m_counts_scat  = np.zeros((NY, NX), dtype=np.float32)  # CH2: scattered only
m_sum_theta_sq = np.zeros((NY, NX), dtype=np.float32)  # for CH3
m_sum_z        = np.zeros((NY, NX), dtype=np.float32)  # for CH5 (mean z)
m_sum_z_sq     = np.zeros((NY, NX), dtype=np.float32)  # for CH5 (std z)

# CH1 — all events
np.add.at(m_counts_all,  (v_y, v_x), 1)

# CH2, CH3, CH5 — scattered only
np.add.at(m_counts_scat,  (v_y[mask_scat], v_x[mask_scat]), 1)
np.add.at(m_sum_theta_sq, (v_y[mask_scat], v_x[mask_scat]), theta_sq[mask_scat])
np.add.at(m_sum_z,        (v_y[mask_scat], v_x[mask_scat]), poca_z[mask_scat])
np.add.at(m_sum_z_sq,     (v_y[mask_scat], v_x[mask_scat]), poca_z[mask_scat]**2)

# ---------------------------------------------------------------------------
# Derived channels
# ---------------------------------------------------------------------------

# CH3 — mean(theta²) per XY cell, scattered only
with np.errstate(invalid="ignore"):
    m_mean_theta_sq = np.where(m_counts_scat > 0, m_sum_theta_sq / m_counts_scat, 0.0)

# CH4 — mean of top-3 theta² per XY cell (robust radiation length proxy)
# Build per-cell list of theta² values, extract top-3
print("[INFO] Computing top-3 theta² per XY cell...")
m_top3_theta_sq = np.zeros((NY, NX), dtype=np.float32)

# Group theta² by (iy, ix) cell efficiently
cell_idx = v_y[mask_scat] * NX + v_x[mask_scat]  # flatten 2D index
order    = np.argsort(cell_idx)                    # sort by cell
cell_idx_sorted  = cell_idx[order]
theta_sq_sorted  = theta_sq[mask_scat][order]

# Iterate over unique cells
unique_cells, cell_starts = np.unique(cell_idx_sorted, return_index=True)
cell_ends = np.append(cell_starts[1:], len(cell_idx_sorted))

for uid, start, end in zip(unique_cells, cell_starts, cell_ends):
    vals   = theta_sq_sorted[start:end]
    top3   = np.sort(vals)[-3:]          # up to 3 largest
    iy, ix = divmod(int(uid), NX)
    m_top3_theta_sq[iy, ix] = top3.mean()

# CH5 — std(poca_z) per XY cell, scattered only
with np.errstate(invalid="ignore"):
    mean_z = np.where(m_counts_scat > 0, m_sum_z / m_counts_scat, 0.0)
    var_z  = np.where(m_counts_scat > 0, m_sum_z_sq / m_counts_scat - mean_z**2, 0.0)
    var_z  = np.maximum(var_z, 0.0)   # numerical safety
    m_std_z = np.sqrt(var_z)

# ---------------------------------------------------------------------------
# Add channel axis → shape (NY, NX, 1) for consistency with UNet input
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
out_path = f"{args.output}.npy"
np.save(out_path, output_dict)
print(f"[CORRECT] Channels saved to: {out_path}")

# Summary
for key, val in output_dict.items():
    print(f"  {key:25s} shape={val.shape}  max={val.max():.4f}  nonzero={np.count_nonzero(val)}")

# ---------------------------------------------------------------------------
# Visualization (optional)
# ---------------------------------------------------------------------------
if args.visualization:
    import matplotlib
    matplotlib.use("Agg")  # sin interfaz gráfica, para cluster
    import matplotlib.pyplot as plt

    channel_labels = {
        "counts_all_2d"    : "CH1 — N counts all (MTR, body shape)",
        "counts_scat_2d"   : "CH2 — N counts scattered",
        "mean_theta_sq_2d" : "CH3 — mean(θ²) scattered",
        "top3_theta_sq_2d" : "CH4 — mean top-3 θ² (radiation length proxy)",
        "std_z_2d"         : "CH5 — std(z) scattered (thickness proxy)",
    }

    n_channels = len(output_dict)
    fig, axes = plt.subplots(1, n_channels, figsize=(5 * n_channels, 5))

    extent = [-X_LIM, X_LIM, -Y_LIM, Y_LIM]

    for ax, (key, val) in zip(axes, output_dict.items()):
        data = val[:, :, 0]  # remove channel axis → (NY, NX)
        im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            cmap="viridis",
            aspect="equal",
        )
        ax.set_title(channel_labels[key], fontsize=9)
        ax.set_xlabel("X [cm]")
        ax.set_ylabel("Y [cm]")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Merge result — {total_events} total events", fontsize=11)
    plt.tight_layout()

    plot_path = f"{args.output}_channels.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[CORRECT] Visualization saved to: {plot_path}")
    plt.close()