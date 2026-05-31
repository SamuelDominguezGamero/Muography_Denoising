#!/usr/bin/env python3
"""
Extract POCA projections from .tar file containing .root files.

Converts POCA data to 128×128×3 numpy tensors (XY, XZ, YZ projections).

LOGIC:
    1. Extract all .root files from .tar
    2. For each .root file:
       - Load POCA points (x, y, z, theta)
       - Filter by abs(theta) > 0.0001
       - Create 3 histograms (XY, XZ, YZ) with 128 bins
       - Stack into 128×128×3 array
       - Save as .npy with metadata in filename

USAGE:
    python3 extract_poca_projections_from_tar.py \
        --tar_file /path/to/archivos.tar \
        --output_dir ./POCA_projections \
        --world_size 128 \
        --verbose
"""

import argparse
import ROOT
import numpy as np
import tarfile
import tempfile
from pathlib import Path
import os
import sys
from datetime import datetime

# ===========================================================================
# ARGUMENT PARSING
# ===========================================================================
parser = argparse.ArgumentParser(
    description="Extract POCA projections from .tar file containing .root files"
)
parser.add_argument("--tar_file", type=str, required=True,
    help="Path to .tar file containing .root files")
parser.add_argument("--output_dir", type=str, default="./POCA_projections",
    help="Output directory for .npy files (default: ./POCA_projections)")
parser.add_argument("--world_size", type=float, default=128.0,
    help="World size in cm for histogram binning (default: 128.0)")
parser.add_argument("--verbose", action="store_true",
    help="Print detailed processing information")

args = parser.parse_args()

# ===========================================================================
# SETUP
# ===========================================================================

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("POCA PROJECTIONS EXTRACTION FROM TAR")
print("=" * 80)
print(f"[INFO] TAR file: {args.tar_file}")
print(f"[INFO] Output directory: {output_dir}")
print(f"[INFO] World size: {args.world_size} cm")
print("=" * 80)

# ===========================================================================
# FUNCTIONS
# ===========================================================================

def extract_root_files_from_tar(tar_path):
    """Extract all .root files from .tar to temporary directory.
    
    Returns:
        tuple: (list of .root file paths, temp directory path)
    """
    print(f"\n[INFO] Opening TAR file...")
    
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        root_members = [m for m in members if m.name.endswith('.root')]
        
        print(f"[INFO] Found {len(root_members)} .root files in TAR")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="poca_tar_extract_")
        print(f"[INFO] Extracting to temporary directory: {temp_dir}")
        
        # Extract only .root files
        tar.extractall(path=temp_dir, members=root_members)
        
        # Get full paths to extracted files
        root_files = []
        for member in root_members:
            full_path = Path(temp_dir) / member.name
            if full_path.exists():
                root_files.append(full_path)
        
        print(f"[INFO] Extracted {len(root_files)} .root files")
        
        return root_files, temp_dir


def load_poca_data(root_file):
    """Load POCA points from .root file.
    
    Returns:
        tuple: (x, y, z, theta) as numpy arrays
        
    Returns None if loading fails
    """
    try:
        df = ROOT.RDataFrame("events", str(root_file))
        df = df.Filter("abs(theta) > 0.00001")
        res = df.AsNumpy(columns=["poca_x", "poca_y", "poca_z", "theta"])
        
        x = res["poca_x"]
        y = res["poca_y"]
        z = res["poca_z"]
        theta = res["theta"]
        
        return x, y, z, theta
    except Exception as e:
        print(f"[ERROR] Failed to load {root_file}: {e}")
        return None


def create_projection_tensor(x, y, z, world_size, bins=128):
    """Create 128×128×3 tensor with XY, XZ, YZ projections.
    
    Uses numpy.histogram2d to create density histograms.
    
    Returns:
        np.ndarray of shape (128, 128, 3) with uint8 values
    """
    world_min = -world_size / 2.0
    world_max = +world_size / 2.0
    
    # Define binning edges
    edges = np.linspace(world_min, world_max, bins + 1)
    
    # Create 3 projections using histogram2d
    # histogram2d returns (H, x_edges, y_edges), we need H transposed
    
    # XY projection (looking down on Z)
    xy_hist, _, _ = np.histogram2d(x, y, bins=edges)
    xy_proj = xy_hist.T  # Shape (128, 128)
    
    # XZ projection (looking from Y side)
    xz_hist, _, _ = np.histogram2d(x, z, bins=edges)
    xz_proj = xz_hist.T  # Shape (128, 128)
    
    # YZ projection (looking from X side)
    yz_hist, _, _ = np.histogram2d(y, z, bins=edges)
    yz_proj = yz_hist.T  # Shape (128, 128)
    
    # Normalize to uint8 (0-255) ----> NO hacer, porque entonces perdemos la noción de flujo alto y bajo
    #     # def normalize_projection(proj):
    #     proj = proj.astype(np.float32)
    #     max_val = np.max(proj)
    #     if max_val > 0:
    #         proj = (proj / max_val) * 255.0
    #     return np.uint8(proj)
    
    # xy_proj = normalize_projection(xy_proj)
    # xz_proj = normalize_projection(xz_proj)
    # yz_proj = normalize_projection(yz_proj)
    
    # Stack into 128×128×3 tensor
    tensor = np.stack([xy_proj, xz_proj, yz_proj], axis=2)
    
    return tensor


def extract_metadata_from_root_filename(root_path):
    """Extract metadata from .root filename for output naming.
    
    Example:
        POCA_merged__Lpx128_...depthZ20.root → Lpx128_...depthZ20
    """
    filename = Path(root_path).stem  # Remove .root extension
    
    # If it starts with POCA_merged__, remove that prefix
    if filename.startswith("POCA_merged__"):
        metadata = filename.replace("POCA_merged__", "")
    else:
        metadata = filename
    
    return metadata


# ===========================================================================
# MAIN PROCESSING
# ===========================================================================

try:
    # Step 1: Extract .root files from TAR
    root_files, temp_dir = extract_root_files_from_tar(args.tar_file)
    
    if not root_files:
        sys.exit("[ERROR] No .root files found in TAR")
    
    # Step 2: Process each .root file
    print(f"\n[INFO] Processing {len(root_files)} .root files...")
    print("-" * 80)
    
    success_count = 0
    error_count = 0
    
    for idx, root_file in enumerate(root_files, 1):
        root_filename = Path(root_file).name
        print(f"\n[{idx}/{len(root_files)}] {root_filename}")
        
        # Load POCA data
        poca_data = load_poca_data(root_file)
        if poca_data is None:
            error_count += 1
            continue
        
        x, y, z, theta = poca_data
        
        if args.verbose:
            print(f"    POCA points: {len(x)}")
            print(f"    X range: [{x.min():.1f}, {x.max():.1f}] cm")
            print(f"    Y range: [{y.min():.1f}, {y.max():.1f}] cm")
            print(f"    Z range: [{z.min():.1f}, {z.max():.1f}] cm")
        
        # Create projection tensor
        tensor = create_projection_tensor(x, y, z, args.world_size, bins=128)
        
        # Generate output filename with metadata
        metadata = extract_metadata_from_root_filename(root_file)
        output_filename = f"tensor_2D_POCA_{metadata}.npy"
        output_path = output_dir / output_filename
        
        # Save tensor
        np.save(output_path, tensor)
        
        print(f"    ✓ Saved: {output_filename}")
        print(f"      Shape: {tensor.shape}")
        print(f"      Data type: {tensor.dtype}")
        print(f"      Value range: [{tensor.min()}, {tensor.max()}]")
        
        success_count += 1
    
    # Step 3: Cleanup temporary directory
    import shutil
    shutil.rmtree(temp_dir)
    print(f"\n[INFO] Cleaned up temporary directory")
    
    # Summary
    print("\n" + "=" * 80)
    print(f"PROCESSING COMPLETE")
    print("=" * 80)
    print(f"[SUMMARY] Successful: {success_count}/{len(root_files)}")
    if error_count > 0:
        print(f"[WARNING] Errors: {error_count}/{len(root_files)}")
    print(f"[INFO] Output files saved to: {output_dir}")
    print("=" * 80)

except KeyboardInterrupt:
    print("\n[INFO] Processing interrupted by user")
    sys.exit(0)
except Exception as e:
    print(f"\n[ERROR] Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
