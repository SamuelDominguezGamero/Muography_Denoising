"""
merge_results.py
Merges all POCA output files for a single geometry (one per seed) into a
single accumulated result. Launched automatically by SLURM via --dependency=afterok.
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

    m_counts_2d      = np.zeros((args.npy, args.npx, 1))
    m_sum_theta_2d    = np.zeros((args.npy, args.npx, 1))
    m_sum_theta_sq_2d = np.zeros((args.npy, args.npx, 1))

    missing = []

    for filepath in poca_files:
        if not os.path.exists(filepath):
            print(f"[ERROR] File not found: {filepath}")
            sys.exit(1)
            
        data = np.load(filepath, allow_pickle=True).item()

        # Validate shapes: all 2D arrays should be (npy, npx, 1)
        expected_shape = (args.npy, args.npx, 1)
        for key in ["counts_2d", "sum_theta_2d", "sum_theta_sq_2d"]:
            if data[key].shape != expected_shape:
                print(f"[ERROR] Shape mismatch at seed={seed}, key '{key}': {data[key].shape} != {expected_shape}")
                sys.exit(1)

        m_counts_2d      += data["counts_2d"]
        m_sum_theta_2d    += data["sum_theta_2d"]
        m_sum_theta_sq_2d += data["sum_theta_sq_2d"]

    print(f"[CORRECT] All {len(poca_files)} files merged successfully.")

    
    # CALCULATIONS: CHANNELS FOR THE UNET

    ### Statistics regarding the scattering angle per XY cell
    mask_exists = m_counts_2d > 0
    mask_stat = m_counts_2d > 1  # Para varianza muestral, necesitamos al menos 2 eventos

    # <theta>_z per XY cell, using the formula mean = sum / N, careful with DIVISION BY ZERO
    matrix_mean_theta_z = np.zeros_like(m_counts_2d)
    matrix_mean_theta_z[mask_exists] = m_sum_theta_2d[mask_exists] / m_counts_2d[mask_exists]

    # <theta^2>_z per XY cell, using the formula mean = sum / N, careful with DIVISION BY ZERO
    matrix_mean_theta_sq_z = np.zeros_like(m_counts_2d)
    matrix_mean_theta_sq_z[mask_exists] = m_sum_theta_sq_2d[mask_exists] / m_counts_2d[mask_exists]       
    
    # variance of theta per XY cell, using the formula Var(X) = (N-1)^(-1) * sum ((x_i - mean)^2)
    matrix_var_theta_z = np.zeros_like(m_counts_2d)
    # poblational var: E[X^2] - (E[X])^2
    var_poblacional = matrix_mean_theta_sq_z - matrix_mean_theta_z**2
    
    matrix_var_theta_z[mask_stat] = var_poblacional[mask_stat] * (m_counts_2d[mask_stat] / (m_counts_2d[mask_stat] - 1)) # Bessel correctio for sample variance
    
    # Limpieza de posibles negativos ínfimos por precisión
    matrix_var_theta_z = np.maximum(0, matrix_var_theta_z)       


    ### Variance of Z per XY cell
    ###################################################
    # completar -> tendremos que sacar los poca points 
    ###################################################


    # Log(N+1) per XY cell -> statistical liability of the cell, we add 1 to avoid log(0)
    matrix_log_counts = np.log1p(m_counts_2d)



    # this form of saving then recquires allow_pickle=True when loading, but it's more compact and faster
    # then load as: data = np.load("merged_result.npy", allow_pickle=True).item() and access data["n_events"], etc.
    output_dict = {
        "mean_theta_sq_z": matrix_mean_theta_sq_z, # <theta^2>_z per XY cell
        "var_theta_z": matrix_var_theta_z, # Var(theta)_z per XY cell
        "log_counts": matrix_log_counts # log(N+1) per XY cell
    }
    np.save(args.output, output_dict)
    print(f"[CORRECT] Merged result saved to: {args.output}")
    # Remember to load with: data = np.load('XXXX.npy', allow_pickle=True).item() and access data['n_events'], data['sum_theta'], data['sum_theta_sq'], data['theta_rms']


else:   
    print(f"[ERROR] ----- Dimension {args.dimension} not implemented yet.")