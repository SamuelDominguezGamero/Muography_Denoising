"""
extract_2D.py

Extracts central 2D slices from 3D geometry files for U-Net training.
Processes both ground truth density and POCA theta_rms data.

Input:  3D arrays (npy, npx, npz) from:
        - create_geometry.py → [npy, npx, npz] ground_truth_density.npy
        - merge_results.py → theta_rms key in MERGED_*.npy dict

Output: 2D slices (npy, npx) saved as 
        - {config}_2DXY_ground_truth_density.npy: normalized GT [0, 1]
        - MERGED_{config}_2DXY_theta_rms.npy: normalized POCA [0, 1]

IMPORTANT: Both outputs normalized to [0, 1] per-sample for U-Net training.

Conventional shape indexing (inherited from POCA/create_geometry):
  3D: [iy, ix, iz] = [height, width, depth] = [npy, npx, npz]
  2D: [iy, ix] = [npy, npx] (extracted at z_center = npz // 2)
"""

import numpy as np
import argparse
import os
import sys
import glob
import json
import matplotlib.pyplot as plt


# ===========================================================================
# CONTROL FLAGS
# ===========================================================================
dibujar = False
environment = "local"  # "cluster" o "local"
verbose = True
save_stats = True  # Save statistics to JSON for reproducibility

# Material properties for validation
MATERIAL_DENSITY_THRESHOLD = 1e-3  # Separate lead (~11.35) from air (~0.0012)
MIN_MATERIAL_VOXELS = 10  # Minimum voxels for valid correlation


# ===========================================================================
# PATHS (same as loop_configuration_files.py)
# ===========================================================================
if environment == "cluster":
    PATH_geometry_files = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/geometric_configurations_json"
    PATH_density_files  = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/ground_truth_data"
    PATH_output_raw     = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_raw"
    PATH_preprocessed   = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_preprocessed"
    PATH_poca_output    = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/post_POCA_data"
    PATH_merged_output  = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/merged_poca_data"
    PATH_png_comparisons = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/png_comparisons"
    PATH_2D_GT          = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/2D_ground_truth"
    PATH_2D_POCA        = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/2D_poca_theta_rms"

elif environment == "local":
    PATH_geometry_files = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/geometric_configurations_json"
    PATH_density_files  = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/ground_truth_data"
    PATH_output_raw     = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/data_raw"
    PATH_preprocessed   = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/data_preprocessed"
    PATH_poca_output    = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/post_POCA_data"
    PATH_merged_output  = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/merged_poca_data"
    PATH_png_comparisons = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/png_comparisons"
    PATH_2D_GT          = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/2D_ground_truth"
    PATH_2D_POCA        = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/2D_poca_theta_rms"

else:
    sys.exit("[ERROR] environment must be 'local' or 'cluster'.")

# Create output directories
os.makedirs(PATH_2D_GT, exist_ok=True)
os.makedirs(PATH_2D_POCA, exist_ok=True)

# Stats collection for reporting
stats_all = []

if verbose:
    print(f"[INFO] Environment: {environment}")
    print(f"[INFO] Ground truth 2D output: {PATH_2D_GT}")
    print(f"[INFO] POCA 2D output:        {PATH_2D_POCA}")
    print(f"[INFO] Material threshold: {MATERIAL_DENSITY_THRESHOLD}")
    print(f"[INFO] Min material voxels for correlation: {MIN_MATERIAL_VOXELS}\n")


# ===========================================================================
# HELPER: Normalize array to [0, 1] per-sample
# ===========================================================================
def normalize_to_01(data):
    """
    Min-max normalization to [0, 1].
    Handles case where all values are identical.
    """
    data_min = np.min(data)
    data_max = np.max(data)
    
    if np.isclose(data_max, data_min):  # All values identical
        return np.ones_like(data) * 0.5  # Return middle value
    
    return (data - data_min) / (data_max - data_min)


# ===========================================================================
# STEP 1: Extract 2D Ground Truth Density
# ===========================================================================
print("[INFO] ========== EXTRACTING GROUND TRUTH 2D SLICES ==========")

gt_files = sorted(glob.glob(os.path.join(PATH_density_files, "*_ground_truth_density.npy")))

if not gt_files:
    print("[WARNING] No ground truth files found.")
else:
    print(f"[INFO] Found {len(gt_files)} ground truth files.\n")
    
    for i, gt_file in enumerate(gt_files, 1):
        try:
            # Load 3D density
            density_3d = np.load(gt_file)
            npy, npx, npz = density_3d.shape
            
            # Extract central slice (z=npz//2)
            z_center = npz // 2
            density_2d = density_3d[:, :, z_center]
            
            # Normalize to [0, 1]
            density_2d_norm = normalize_to_01(density_2d)
            
            # Construct output filename: replace _ground_truth_density with _2DXY_ground_truth_density
            basename = os.path.basename(gt_file)
            basename_2d = basename.replace("_ground_truth_density.npy", "_2DXY_ground_truth_density.npy")
            output_path = os.path.join(PATH_2D_GT, basename_2d)
            
            # Save 2D slice (normalized)
            np.save(output_path, density_2d_norm)
            
            if verbose:
                print(f"[CORRECT] [{i}/{len(gt_files)}] Extracted GT: {basename}")
                print(f"            Input 3D shape: {density_3d.shape} → 2D shape: {density_2d.shape} at z={z_center}")
                print(f"            Value range (original): [{density_3d.min():.6f}, {density_3d.max():.6f}]")
                print(f"            Value range (normalized): [{density_2d_norm.min():.6f}, {density_2d_norm.max():.6f}]\n")
            
            # Store stats
            stats_all.append({
                "geometry": basename_2d.replace("_2DXY_ground_truth_density.npy", ""),
                "data_type": "GT",
                "shape_3d": list(density_3d.shape),
                "shape_2d": list(density_2d.shape),
                "z_slice": int(z_center),
                "value_min": float(density_3d.min()),
                "value_max": float(density_3d.max()),
            })
        
        except Exception as e:
            print(f"[ERROR] Failed to process {gt_file}: {e}\n")

print("[CORRECT] Ground truth 2D extraction complete.\n")


# ===========================================================================
# STEP 2: Extract 2D POCA theta_rms
# ===========================================================================
print("[INFO] ========== EXTRACTING POCA 2D SLICES ==========")

poca_files = sorted(glob.glob(os.path.join(PATH_merged_output, "MERGED_*.npy")))

if not poca_files:
    print("[WARNING] No POCA merged files found.")
else:
    print(f"[INFO] Found {len(poca_files)} POCA merged files.\n")
    
    for i, poca_file in enumerate(poca_files, 1):
        try:
            # Load POCA merged data (contains dict with theta_rms)
            poca_data = np.load(poca_file, allow_pickle=True).item()
            
            # Validate that theta_rms key exists
            if "theta_rms" not in poca_data:
                print(f"[ERROR] Key 'theta_rms' not found in {os.path.basename(poca_file)}")
                print(f"        Available keys: {list(poca_data.keys())}")
                continue
            
            theta_rms_3d = poca_data["theta_rms"]
            npy, npx, npz = theta_rms_3d.shape
            
            # Extract central slice (z=npz//2)
            z_center = npz // 2
            theta_rms_2d = theta_rms_3d[:, :, z_center]
            
            # Normalize to [0, 1]
            theta_rms_2d_norm = normalize_to_01(theta_rms_2d)
            
            # Construct output filename
            basename = os.path.basename(poca_file)
            namefile = basename.replace("MERGED_", "").replace(".npy", "")
            basename_2d = f"MERGED_{namefile}_2DXY_theta_rms.npy"
            output_path = os.path.join(PATH_2D_POCA, basename_2d)
            
            # Save 2D slice (normalized)
            np.save(output_path, theta_rms_2d_norm)
            
            if verbose:
                print(f"[CORRECT] [{i}/{len(poca_files)}] Extracted POCA: {basename}")
                print(f"            Input 3D shape: {theta_rms_3d.shape} → 2D shape: {theta_rms_2d.shape} at z={z_center}")
                print(f"            Value range (original): [{theta_rms_3d.min():.6f}, {theta_rms_3d.max():.6f}] rad")
                print(f"            Value range (normalized): [{theta_rms_2d_norm.min():.6f}, {theta_rms_2d_norm.max():.6f}]\n")
            
            # Store stats
            stats_all.append({
                "geometry": namefile,
                "data_type": "POCA",
                "shape_3d": list(theta_rms_3d.shape),
                "shape_2d": list(theta_rms_2d.shape),
                "z_slice": int(z_center),
                "value_min": float(theta_rms_3d.min()),
                "value_max": float(theta_rms_3d.max()),
            })
        
        except KeyError as e:
            print(f"[ERROR] Key error in {poca_file}: {e}\n")
        except Exception as e:
            print(f"[ERROR] Failed to process {poca_file}: {e}\n")

print("[CORRECT] POCA 2D extraction complete.\n")


# ===========================================================================
# STEP 3: Verification, validation, and statistics
# ===========================================================================
print("[INFO] ========== VERIFICATION AND STATISTICS ==========\n")

# Get all pairs of files
gt_2d_files = sorted(glob.glob(os.path.join(PATH_2D_GT, "*_2DXY_ground_truth_density.npy")))
poca_2d_files = sorted(glob.glob(os.path.join(PATH_2D_POCA, "*_2DXY_theta_rms.npy")))

if not gt_2d_files or not poca_2d_files:
    print("[WARNING] No 2D files found for verification.")
    matched_pairs = []
else:
    # Extract geometry names from filenames for matching
    def extract_geom_name(filename):
        """Extract geometry name (everything between prefixes and suffixes)."""
        basename = os.path.basename(filename)
        if "_2DXY_ground_truth_density" in basename:
            return basename.replace("_2DXY_ground_truth_density.npy", "")
        elif "_2DXY_theta_rms" in basename:
            return basename.replace("MERGED_", "").replace("_2DXY_theta_rms.npy", "")
        return basename
    
    # Match GT and POCA files
    gt_dict = {extract_geom_name(f): f for f in gt_2d_files}
    poca_dict = {extract_geom_name(f): f for f in poca_2d_files}
    
    matched_pairs = []
    unmatched_gt = []
    unmatched_poca = []
    
    for geom_name in gt_dict.keys():
        if geom_name in poca_dict:
            matched_pairs.append((geom_name, gt_dict[geom_name], poca_dict[geom_name]))
        else:
            unmatched_gt.append(geom_name)
    
    for geom_name in poca_dict.keys():
        if geom_name not in gt_dict:
            unmatched_poca.append(geom_name)
    
    if unmatched_gt:
        print(f"[WARNING] Unmatched GT files (no POCA): {len(unmatched_gt)}")
    if unmatched_poca:
        print(f"[WARNING] Unmatched POCA files (no GT): {len(unmatched_poca)}")
    
    if not matched_pairs:
        print("[ERROR] Could not match any GT-POCA file pairs. Check naming convention.")
    else:
        print(f"[INFO] Found {len(matched_pairs)} matched GT-POCA pairs.\n")
        print("[INFO] Computing statistics...\n")
        
        paired_stats = []
        
        # Compute statistics for all pairs
        for geom_name, gt_file, poca_file in matched_pairs:
            try:
                gt_data = np.load(gt_file)
                poca_data = np.load(poca_file)
                
                # VALIDATION: Check shapes match
                if gt_data.shape != poca_data.shape:
                    print(f"[ERROR] Shape mismatch for {geom_name}:")
                    print(f"        GT: {gt_data.shape} vs POCA: {poca_data.shape}")
                    continue
                
                # Create masks: GT has material (>0) vs air (≈0)
                # Note: data is normalized to [0, 1], so threshold needs adjustment
                # Original: material ~11.35, air ~0.0012 → normalized: material ~1.0, air ~0.0
                material_mask = gt_data > 0.5  # Conservative threshold after normalization
                air_mask = ~material_mask
                
                n_material = material_mask.sum()
                n_air = air_mask.sum()
                
                print(f"[INFO] Geometry: {geom_name}")
                print(f"       Shape (both): {gt_data.shape}")
                print(f"       Material voxels: {n_material} ({100*n_material/gt_data.size:.1f}%)")
                print(f"       Air voxels: {n_air} ({100*n_air/gt_data.size:.1f}%)")
                
                # Statistics in material
                if n_material > 0:
                    poca_in_material = poca_data[material_mask]
                    print(f"       POCA in material: mean={poca_in_material.mean():.6f}, std={poca_in_material.std():.6f}")
                    print(f"                         min={poca_in_material.min():.6f}, max={poca_in_material.max():.6f}")
                else:
                    poca_in_material = np.array([])
                    print(f"       POCA in material: NO MATERIAL VOXELS (geometry issue?)")
                
                # Statistics in air
                if n_air > 0:
                    poca_in_air = poca_data[air_mask]
                    print(f"       POCA in air:      mean={poca_in_air.mean():.6f}, std={poca_in_air.std():.6f}")
                    print(f"                         min={poca_in_air.min():.6f}, max={poca_in_air.max():.6f}")
                else:
                    poca_in_air = np.array([])
                    print(f"       POCA in air:      NO AIR VOXELS (geometry issue?)")
                
                # Compute correlation in material regions (FIXED)
                correlation = np.nan
                if n_material >= MIN_MATERIAL_VOXELS:
                    gt_material = gt_data[material_mask]
                    # Check if data has variance (not all identical)
                    if np.std(gt_material) > 1e-10 and np.std(poca_in_material) > 1e-10:
                        try:
                            corr_matrix = np.corrcoef(gt_material.flatten(), poca_in_material.flatten())
                            correlation = corr_matrix[0, 1]
                            print(f"       Correlation (material, normalized): {correlation:.4f}")
                        except Exception as corr_err:
                            print(f"       Correlation: Could not compute ({corr_err})")
                    else:
                        print(f"       Correlation: Cannot compute (zero variance in one array)")
                else:
                    print(f"       Correlation: Not enough material voxels (need {MIN_MATERIAL_VOXELS}, have {n_material})")
                
                # Signal-to-Noise ratio
                if len(poca_in_material) > 0 and len(poca_in_air) > 0:
                    snr = poca_in_material.mean() / (poca_in_air.mean() + 1e-10)
                    print(f"       Signal-to-Noise: {snr:.2f}x")
                
                print()
                
                # Store paired stats
                paired_stats.append({
                    "geometry": geom_name,
                    "shape": list(gt_data.shape),
                    "n_material": int(n_material),
                    "n_air": int(n_air),
                    "poca_material_mean": float(poca_in_material.mean()) if len(poca_in_material) > 0 else None,
                    "poca_material_std": float(poca_in_material.std()) if len(poca_in_material) > 0 else None,
                    "poca_air_mean": float(poca_in_air.mean()) if len(poca_in_air) > 0 else None,
                    "poca_air_std": float(poca_in_air.std()) if len(poca_in_air) > 0 else None,
                    "correlation": float(correlation) if not np.isnan(correlation) else None,
                })
            
            except Exception as e:
                print(f"[ERROR] Failed to compute statistics for {geom_name}: {e}\n")
        
        # Save statistics if requested
        if save_stats:
            stats_file = os.path.join(PATH_2D_GT, "extraction_statistics.json")
            stats_output = {
                "extraction_info": {
                    "total_geometries": len(matched_pairs),
                    "material_threshold": MATERIAL_DENSITY_THRESHOLD,
                    "min_material_voxels": MIN_MATERIAL_VOXELS,
                },
                "paired_statistics": paired_stats,
            }
            with open(stats_file, 'w') as f:
                json.dump(stats_output, f, indent=2)
            print(f"[CORRECT] Statistics saved to: {stats_file}\n")


# ===========================================================================
# STEP 4: Visual comparison (if dibujar=True)
# ===========================================================================
if dibujar and matched_pairs:
    print("[INFO] ========== GENERATING VISUALIZATIONS ==========\n")
    
    # Process first 3 geometries for visualization
    for idx, (geom_name, gt_file, poca_file) in enumerate(matched_pairs[:3], 1):
        try:
            gt_data = np.load(gt_file)
            poca_data = np.load(poca_file)
            
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(18, 12))
            
            # 1. Ground truth heatmap
            ax1 = plt.subplot(2, 3, 1)
            im1 = ax1.imshow(gt_data, origin='lower', cmap='viridis')
            ax1.set_title('Ground Truth Density (Normalized)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('X (voxels)')
            ax1.set_ylabel('Y (voxels)')
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('Normalized [0, 1]')
            
            # 2. POCA theta_rms heatmap
            ax2 = plt.subplot(2, 3, 2)
            im2 = ax2.imshow(poca_data, origin='lower', cmap='hot')
            ax2.set_title('POCA θ_rms (Normalized)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('X (voxels)')
            ax2.set_ylabel('Y (voxels)')
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('Normalized [0, 1]')
            
            # 3. Overlay: GT regions with POCA intensity
            ax3 = plt.subplot(2, 3, 3)
            material_mask = gt_data > 0.5
            overlay = np.zeros_like(poca_data)
            overlay[material_mask] = poca_data[material_mask]
            im3 = ax3.imshow(overlay, origin='lower', cmap='plasma')
            ax3.set_title('POCA Intensity in Material Only', fontsize=12, fontweight='bold')
            ax3.set_xlabel('X (voxels)')
            ax3.set_ylabel('Y (voxels)')
            cbar3 = plt.colorbar(im3, ax=ax3)
            cbar3.set_label('Normalized [0, 1]')
            
            # 4. Histograms: POCA distribution (FIXED - safe against empty arrays)
            ax4 = plt.subplot(2, 3, 4)
            poca_material = poca_data[material_mask] if material_mask.sum() > 0 else np.array([])
            poca_air = poca_data[~material_mask] if (~material_mask).sum() > 0 else np.array([])
            
            if len(poca_material) > 0:
                ax4.hist(poca_material, bins=30, alpha=0.7, label=f'In Material (n={len(poca_material)})', 
                        color='red', edgecolor='black')
            if len(poca_air) > 0:
                ax4.hist(poca_air, bins=30, alpha=0.7, label=f'In Air (n={len(poca_air)})', 
                        color='blue', edgecolor='black')
            
            ax4.set_xlabel('POCA θ_rms (normalized)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('POCA Distribution by Region', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(alpha=0.3)
            
            # 5. Scatter: GT vs POCA (material regions, FIXED - safe against small samples)
            ax5 = plt.subplot(2, 3, 5)
            if len(poca_material) >= 10:
                gt_material = gt_data[material_mask]
                scatter = ax5.scatter(gt_material, poca_material, alpha=0.5, s=10, c=poca_material, cmap='cool')
                ax5.set_xlabel('GT Density (normalized)')
                ax5.set_ylabel('POCA θ_rms (normalized)')
                ax5.set_title(f'Correlation in Material (n={len(poca_material)})', fontsize=12, fontweight='bold')
                plt.colorbar(scatter, ax=ax5, label='POCA θ_rms')
                ax5.grid(alpha=0.3)
            else:
                ax5.text(0.5, 0.5, f'Not enough material voxels\n(n={len(poca_material)})', 
                        ha='center', va='center', transform=ax5.transAxes, fontsize=12)
                ax5.set_title('Correlation (Unavailable)', fontsize=12, fontweight='bold')
            
            # 6. Statistics box
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('off')
            poca_mat_mean = poca_material.mean() if len(poca_material) > 0 else np.nan
            poca_air_mean = poca_air.mean() if len(poca_air) > 0 else np.nan
            snr = (poca_mat_mean / (poca_air_mean + 1e-10)) if (len(poca_material) > 0 and len(poca_air) > 0) else np.nan
            
            stats_text = f"""
GEOMETRY: {geom_name}

Ground Truth (normalized):
  Shape: {gt_data.shape}
  Material voxels: {material_mask.sum()}
  Air voxels: {(~material_mask).sum()}

POCA Statistics (normalized):
  Overall - Mean: {poca_data.mean():.6f}
  Overall - Std:  {poca_data.std():.6f}
  
  In Material:
    Mean: {poca_mat_mean:.6f}
    Std:  {poca_material.std() if len(poca_material) > 0 else np.nan:.6f}
    Max:  {poca_material.max() if len(poca_material) > 0 else np.nan:.6f}
  
  In Air:
    Mean: {poca_air_mean:.6f}
    Std:  {poca_air.std() if len(poca_air) > 0 else np.nan:.6f}
    Max:  {poca_air.max() if len(poca_air) > 0 else np.nan:.6f}

Signal-to-Noise:
  Ratio (material/air): {snr:.2f}x
            """
            ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.suptitle(f'Geometry {idx}/{min(3, len(matched_pairs))}: {geom_name}', 
                        fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            print(f"[ERROR] Failed to visualize {geom_name}: {e}\n")
elif dibujar and not matched_pairs:
    print("[WARNING] Cannot generate visualizations: no matched pairs found.\n")

print("[CORRECT] ========== EXTRACTION COMPLETE ==========")
