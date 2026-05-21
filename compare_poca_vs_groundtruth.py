#!/usr/bin/env python3
"""
Compare POCA data projections with ground truth geometry projections.
Generates side-by-side visualizations to verify consistency.

Usage:
    python3 compare_poca_vs_groundtruth.py --root_file file.root --tensor_3d tensor.npy
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

try:
    import ROOT
except ImportError:
    print("[ERROR] ROOT not installed")
    sys.exit(1)

def load_poca_data(root_file, world_size=128.0):
    """Load POCA points from ROOT file."""
    print(f"[INFO] Loading POCA data from: {root_file}")
    
    df = ROOT.RDataFrame("events", root_file)
    df = df.Filter("abs(theta) > 0.0001")
    res = df.AsNumpy(columns=["poca_x", "poca_y", "poca_z"])
    
    x = res["poca_x"]
    y = res["poca_y"]
    z = res["poca_z"]
    
    print(f"[INFO] POCA points loaded: {len(x)}")
    print(f"[INFO] X range: [{x.min():.1f}, {x.max():.1f}]")
    print(f"[INFO] Y range: [{y.min():.1f}, {y.max():.1f}]")
    print(f"[INFO] Z range: [{z.min():.1f}, {z.max():.1f}]")
    
    return x, y, z, world_size

def load_ground_truth(tensor_3d_path):
    """Load ground truth tensor."""
    print(f"[INFO] Loading ground truth tensor from: {tensor_3d_path}")
    
    tensor = np.load(tensor_3d_path)
    print(f"[INFO] Tensor shape: {tensor.shape}")
    print(f"[INFO] Filled voxels: {int(np.sum(tensor))}")
    
    return tensor

def create_comparison_plot(x_poca, y_poca, z_poca, tensor_3d, world_size, output_path=None):
    """Create side-by-side comparison of POCA and ground truth projections."""
    
    # Calculate ground truth projections
    proj_xy_gt = np.max(tensor_3d, axis=2)  # (Y, X)
    proj_xz_gt = np.max(tensor_3d, axis=0).T  # (Z, X)
    proj_yz_gt = np.max(tensor_3d, axis=1).T  # (Z, Y)
    
    # Create figure with 6 subplots: 3 POCA (left) and 3 ground truth (right)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('POCA Data vs Ground Truth Comparison', fontsize=16, fontweight='bold')
    
    binning = 64
    world_min = -world_size / 2.0
    world_max = world_size / 2.0
    
    # Row 0: POCA histograms
    # ========================
    
    # XY POCA
    axes[0, 0].hist2d(x_poca, y_poca, bins=binning, cmap='viridis')
    axes[0, 0].set_title('POCA XY (Data)', fontweight='bold')
    axes[0, 0].set_xlabel('X (cm)')
    axes[0, 0].set_ylabel('Y (cm)')
    axes[0, 0].set_xlim(world_min, world_max)
    axes[0, 0].set_ylim(world_min, world_max)
    axes[0, 0].grid(True, alpha=0.3)
    
    # XZ POCA
    axes[0, 1].hist2d(x_poca, z_poca, bins=binning, cmap='viridis')
    axes[0, 1].set_title('POCA XZ (Data)', fontweight='bold')
    axes[0, 1].set_xlabel('X (cm)')
    axes[0, 1].set_ylabel('Z (cm)')
    axes[0, 1].set_xlim(world_min, world_max)
    axes[0, 1].set_ylim(world_min, world_max)
    axes[0, 1].grid(True, alpha=0.3)
    
    # YZ POCA
    axes[0, 2].hist2d(y_poca, z_poca, bins=binning, cmap='viridis')
    axes[0, 2].set_title('POCA YZ (Data)', fontweight='bold')
    axes[0, 2].set_xlabel('Y (cm)')
    axes[0, 2].set_ylabel('Z (cm)')
    axes[0, 2].set_xlim(world_min, world_max)
    axes[0, 2].set_ylim(world_min, world_max)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 1: Ground truth projections
    # ================================
    
    # XY GT
    im_xy = axes[1, 0].imshow(proj_xy_gt, origin='lower', cmap='viridis',
                               extent=[world_min, world_max, world_min, world_max])
    axes[1, 0].set_title('Ground Truth XY (Geometry)', fontweight='bold', color='green')
    axes[1, 0].set_xlabel('X (cm)')
    axes[1, 0].set_ylabel('Y (cm)')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(im_xy, ax=axes[1, 0], label='Occupancy')
    
    # XZ GT
    im_xz = axes[1, 1].imshow(proj_xz_gt, origin='lower', cmap='viridis',
                               extent=[world_min, world_max, world_min, world_max])
    axes[1, 1].set_title('Ground Truth XZ (Geometry)', fontweight='bold', color='green')
    axes[1, 1].set_xlabel('X (cm)')
    axes[1, 1].set_ylabel('Z (cm)')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(im_xz, ax=axes[1, 1], label='Occupancy')
    
    # YZ GT
    im_yz = axes[1, 2].imshow(proj_yz_gt, origin='lower', cmap='viridis',
                               extent=[world_min, world_max, world_min, world_max])
    axes[1, 2].set_title('Ground Truth YZ (Geometry)', fontweight='bold', color='green')
    axes[1, 2].set_xlabel('Y (cm)')
    axes[1, 2].set_ylabel('Z (cm)')
    axes[1, 2].grid(True, alpha=0.3)
    plt.colorbar(im_yz, ax=axes[1, 2], label='Occupancy')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Comparison plot saved: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare POCA data with ground truth geometry projections"
    )
    parser.add_argument("--root_file", type=str, required=True,
                       help="Path to POCA ROOT file")
    parser.add_argument("--tensor_3d", type=str, required=True,
                       help="Path to 3D ground truth tensor (.npy)")
    parser.add_argument("--world_size", type=float, default=128.0,
                       help="World size in cm (default: 128.0)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for comparison plot PNG")
    
    args = parser.parse_args()
    
    # Verify files exist
    if not Path(args.root_file).exists():
        sys.exit(f"[ERROR] ROOT file not found: {args.root_file}")
    if not Path(args.tensor_3d).exists():
        sys.exit(f"[ERROR] Tensor file not found: {args.tensor_3d}")
    
    # Load data
    x_poca, y_poca, z_poca, world_size = load_poca_data(args.root_file, args.world_size)
    tensor_3d = load_ground_truth(args.tensor_3d)
    
    # Create comparison
    print("\n" + "=" * 75)
    print("GENERATING COMPARISON PLOT")
    print("=" * 75)
    
    create_comparison_plot(x_poca, y_poca, z_poca, tensor_3d, world_size, args.output)
    
    print("\n" + "=" * 75)
    print("✅ COMPARISON COMPLETE")
    print("=" * 75)
    print("\nVisualize the plots above:")
    print("  • Top row: POCA data histograms (actual detector measurements)")
    print("  • Bottom row: Ground truth projections (geometry-based)")
    print("\nThe patterns should match in orientation and spatial extent!")
    print("=" * 75)
