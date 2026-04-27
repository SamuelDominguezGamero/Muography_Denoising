"""
merge_results_1.py
Merges all POCA1 output files for a single geometry (one per seed) into a
single accumulated result with 4 training channels for UNET1_2D.

INPUT FROM POCA1:
  Per seed: counts, sum_theta², sum(z), sum(z²), sum(z*theta²), top-3 theta² values


OUTPUT: 4 Training Channels
  - Channel 0: log(1+N(xy)) - log of event counts per XY cell
  - Channel 1: weighted_mean(z, weights=θ²) - z position weighted by scattering strength
  - Channel 2: std(z) - Z-coordinate std dev, thickness proxy per XY cell
  - Channel 3: top-20 theta² statistic - maximum scattering power indicator

Shape: (128, 128, 4) suitable for UNET1_2D.py input
"""

import numpy as np
import argparse
import os
import sys
import glob

parser = argparse.ArgumentParser(description="Merge POCA results from all seeds for a given geometry.")
parser.add_argument("--namefile",         required=True,  help="Geometry name (without extension).")
parser.add_argument("--n_jobs",           required=True,  type=int, help="Number of seeds to merge.")
parser.add_argument("--npx",              required=True,  type=int)
parser.add_argument("--npy",              required=True,  type=int)
parser.add_argument("--npz",              required=True,  type=int)
parser.add_argument("--path_poca_output", required=True,  help="Directory with per-seed POCA .npy files.")
parser.add_argument("--output",           required=True,  help="Output .npy file for the merged result.")
parser.add_argument("--dimension",        required=False, default="2D", choices=["2D", "3D"], help="Whether the POCA results are 2D (XY) or 3D (XYZ).")
args = parser.parse_args()


print(f"[INFO] Merging POCA files for: {args.namefile}")

# Find all matching POCA files dynamically (handles random seeds)
poca_files = sorted(glob.glob(os.path.join(args.path_poca_output, f"POCA_{args.namefile}_seed*.npy")))

if len(poca_files) != args.n_jobs:
    print(f"[ERROR] Expected {args.n_jobs} POCA files but found {len(poca_files)}")
    print(f"[ERROR] Search directory: {args.path_poca_output}")
    print(f"[ERROR] Pattern: POCA_{args.namefile}_seed*.npy")
    if poca_files:
        print(f"[ERROR] Found files: {[os.path.basename(f) for f in poca_files]}")
    else:
        print(f"[ERROR] No POCA files found.")
    sys.exit(1)

# Check that all files exist before starting the merge
missing_files = [f for f in poca_files if not os.path.exists(f)]

if missing_files:
    print(f"[ERROR] Missing {len(missing_files)} POCA files. Cannot proceed with merge.")
    for f in missing_files:
        print(f"         {f}")
    sys.exit(1)

if args.dimension == "2D":

    # Initialize accumulators for all raw statistics from POCA1
    m_counts_2d       = np.zeros((args.npy, args.npx, 1))
    m_sum_theta_2d    = np.zeros((args.npy, args.npx, 1))
    m_sum_theta_sq_2d = np.zeros((args.npy, args.npx, 1))
    m_sum_poca_z_2d   = np.zeros((args.npy, args.npx, 1))
    m_sum_poca_z_sq_2d = np.zeros((args.npy, args.npx, 1))
    m_sum_z_theta_sq_2d = np.zeros((args.npy, args.npx, 1))
    
    # Accumulator for top-20 theta² values per cell (will be filled from top-3 per seed)
    # Store as list of arrays to dynamically accumulate, then extract top-20
    m_theta_sq_pool_2d = [[[] for _ in range(args.npx)] for _ in range(args.npy)]

    print(f"[INFO] Processing {len(poca_files)} POCA1 files...")
    for idx, filepath in enumerate(poca_files):
        if not os.path.exists(filepath):
            print(f"[ERROR] File not found: {filepath}")
            sys.exit(1)
            
        data = np.load(filepath, allow_pickle=True).item()

        # Validate required fields from POCA1
        expected_shape = (args.npy, args.npx, 1)
        required_keys = ["counts_2d", "sum_theta_2d", "sum_theta_sq_2d", "sum_poca_z_2d", "sum_poca_z_sq_2d", "sum_z_theta_sq_2d", "top3_theta_sq_2d"]
        
        for key in required_keys:
            if key not in data:
                print(f"[ERROR] Missing key '{key}' in {filepath}")
                sys.exit(1)
            if key != "top3_theta_sq_2d" and data[key].shape != expected_shape:
                print(f"[ERROR] Shape mismatch in '{key}': {data[key].shape} != {expected_shape}")
                sys.exit(1)
            if key == "top3_theta_sq_2d" and data[key].shape != (args.npy, args.npx, 3):
                print(f"[ERROR] Shape mismatch in 'top3_theta_sq_2d': {data[key].shape} != ({args.npy}, {args.npx}, 3)")
                sys.exit(1)

        # Accumulate statistics across all seeds
        m_counts_2d       += data["counts_2d"]
        m_sum_theta_2d    += data["sum_theta_2d"]
        m_sum_theta_sq_2d += data["sum_theta_sq_2d"]
        m_sum_poca_z_2d   += data["sum_poca_z_2d"]
        m_sum_poca_z_sq_2d += data["sum_poca_z_sq_2d"]
        m_sum_z_theta_sq_2d += data["sum_z_theta_sq_2d"]
        
        # Accumulate top-3 theta² values for later extraction of top-20
        top3 = data["top3_theta_sq_2d"]
        for iy in range(args.npy):
            for ix in range(args.npx):
                for val in top3[iy, ix]:
                    if val > 0:  # Only add non-zero values
                        m_theta_sq_pool_2d[iy][ix].append(val)
        
        if (idx + 1) % max(1, len(poca_files) // 10) == 0 or idx == 0:
            print(f"  [{idx + 1}/{len(poca_files)}] processed")

    print(f"[CORRECT] All {len(poca_files)} files merged successfully.\n")

    
    # ===========================================================================
    # CHANNEL CALCULATION: Derive 4 input channels for UNET1_2D
    # ===========================================================================

    # Masks for safe division
    mask_exists = m_counts_2d > 0
    mask_stat = m_counts_2d > 1  # At least 2 events for sample variance

    print("[CHANNELS] Computing 4-channel input for UNET1_2D...\n")

    # ========== CHANNEL 0: Log(1 + counts) ==========
    channel_0 = np.log1p(m_counts_2d)
    print(f"[CH0] log(1 + N) - event counts per XY cell")
    print(f"      Shape: {channel_0.shape}, min={channel_0.min():.4f}, max={channel_0.max():.4f}")
    print(f"      Non-zero cells: {mask_exists.sum()}/{channel_0.size}\n")

    # ========== CHANNEL 1: Weighted mean of Z (weighted by theta²) per XY cell ==========
    # This is a weighted average: mean_weighted(z, weights=θ²) = sum(z*θ²) / sum(θ²)
    # High theta² events pull the Z average toward them, identifying strong scattering regions
    channel_1 = np.zeros_like(m_counts_2d, dtype=np.float32)
    mask_has_theta_sq = m_sum_theta_sq_2d > 0
    channel_1[mask_has_theta_sq] = m_sum_z_theta_sq_2d[mask_has_theta_sq] / m_sum_theta_sq_2d[mask_has_theta_sq]
    print(f"[CH1] weighted_mean(z, weights=θ²) - z position weighted by scattering strength")
    print(f"      Shape: {channel_1.shape}, min={channel_1.min():.8f}, max={channel_1.max():.8f}")
    print(f"      Mean (where events exist): {channel_1[mask_exists].mean():.8f}\n")

    # ========== CHANNEL 2: Standard deviation of Z per XY cell ==========
    # This is a proxy for material thickness: larger std = thicker material
    channel_2 = np.zeros_like(m_counts_2d, dtype=np.float32)
    
    # E[Z] per cell
    mean_poca_z = np.zeros_like(m_counts_2d)
    mean_poca_z[mask_exists] = m_sum_poca_z_2d[mask_exists] / m_counts_2d[mask_exists]
    
    # E[Z²] per cell
    mean_poca_z_sq = np.zeros_like(m_counts_2d)
    mean_poca_z_sq[mask_exists] = m_sum_poca_z_sq_2d[mask_exists] / m_counts_2d[mask_exists]
    
    # Var(Z) = E[Z²] - (E[Z])² [population variance]
    var_poca_z_pop = mean_poca_z_sq - mean_poca_z**2
    var_poca_z_pop = np.maximum(0, var_poca_z_pop)
    
    # Apply Bessel correction for sample variance where N > 1
    channel_2[mask_stat] = var_poca_z_pop[mask_stat] * (m_counts_2d[mask_stat] / (m_counts_2d[mask_stat] - 1))
    
    # Take square root to get standard deviation (not variance)
    channel_2 = np.sqrt(channel_2)
    
    print(f"[CH2] std(Z) - Z-coordinate std dev, thickness proxy")
    print(f"      Shape: {channel_2.shape}, min={channel_2.min():.8f}, max={channel_2.max():.8f}")
    print(f"      Mean (where events exist): {channel_2[mask_exists].mean():.8f}\n")

    # ========== CHANNEL 3: Top-20 theta² statistic per XY cell ==========
    # Extract the 20th largest theta² value (or mean of top-20 if not enough values)
    # This identifies regions with high scattering power
    channel_3 = np.zeros_like(m_counts_2d, dtype=np.float32)
    
    for iy in range(args.npy):
        for ix in range(args.npx):
            theta_sq_list = m_theta_sq_pool_2d[iy][ix]
            if len(theta_sq_list) > 0:
                # Sort and take top-20 (or all if fewer than 20)
                top_vals = np.sort(theta_sq_list)[-20:]  # Top 20 values
                # Use the 20th percentile of these top values as channel (or mean if < 20 values)
                channel_3[iy, ix, 0] = np.mean(top_vals)
    
    print(f"[CH3] mean(top-20 θ²) - maximum scattering power indicator")
    print(f"      Shape: {channel_3.shape}, min={channel_3.min():.8f}, max={channel_3.max():.8f}")
    print(f"      Mean (where events exist): {channel_3[mask_exists].mean():.8f}\n")



    # ===========================================================================
    # SAVE 4-CHANNEL OUTPUT (npy, npx, 4)
    # ===========================================================================
    
    # Stack 4 channels: (npy, npx, 1) x 4 -> (npy, npx, 4)
    training_channels = np.concatenate([
        channel_0,  # Channel 0: log(1+N)
        channel_1,  # Channel 1: mean(z * θ²)
        channel_2,  # Channel 2: std(Z) - thickness proxy
        channel_3   # Channel 3: mean(top-20 θ²) - scattering power
    ], axis=2)
    
    print(f"[OUTPUT] Stacking 4 channels...")
    print(f"         Final shape: {training_channels.shape}")
    assert training_channels.shape == (args.npy, args.npx, 4), \
        f"Shape mismatch: expected ({args.npy}, {args.npx}, 4) but got {training_channels.shape}"

    # Save in dict format (allows pickle loading with allow_pickle=True)
    # Load as: data = np.load("file.npy", allow_pickle=True).item()
    #          training = data["training_channels"]
    output_dict = {
        "training_channels": training_channels,  # Shape (npy, npx, 4) ready for UNET1_2D
        "channel_names": ["log(1+N)", "weighted_mean(z, θ²)", "std(Z)", "mean(top-20 θ²)"],
        "channel_0_log_counts": channel_0,
        "channel_1_weighted_mean_z": channel_1,
        "channel_2_std_z": channel_2,
        "channel_3_top20_theta_sq": channel_3,
        "metadata": {
            "total_events": m_counts_2d.sum(),
            "n_geometries": len(poca_files),
            "geometry_name": args.namefile
        }
    }
    np.save(args.output, output_dict)
    
    print(f"\n[CORRECT] Merged result saved to: {args.output}")
    print(f"[INFO] Total accumulated events: {m_counts_2d.sum():.0f}")
    print(f"[INFO] Load with: data = np.load('{args.output}', allow_pickle=True).item()")
    print(f"[INFO] Access training data: training = data['training_channels']")


else:   
    print(f"[ERROR] ----- Dimension {args.dimension} not implemented yet.")