#!/usr/bin/env python3
"""
Visualize Geant4 geometry from JSON file in three orthogonal projections.
Shows XY, XZ, and YZ slices with material distribution.

Usage:
    python3 visualize_json.py --json geometry_file.json [--save output.png]
"""

import json
import numpy as np

# Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend (works in most environments)
import matplotlib.pyplot as plt

import argparse
import sys
from pathlib import Path


def load_geometry_json(json_file):
    """Load geometry from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def build_3d_array(data):
    """
    Reconstruct 3D geometry array from voxel list.
    Maps individual voxel positions to 3D array indices.
    Returns:
        array: 3D numpy array with voxel materials
        voxel_config: dict with configuration info
    """
    # Get world dimensions to establish coordinate mapping
    world_config = data.get('theWorld', {})
    xSizeWorld = world_config.get('xSizeWorld', 128.0)
    ySizeWorld = world_config.get('ySizeWorld', 128.0)
    zSizeWorld = world_config.get('zSizeWorld', 128.0)
    
    print(f"[DEBUG] World size: X={xSizeWorld}, Y={ySizeWorld}, Z={zSizeWorld}")
    
    voxel_config = data.get('VoxelConfig', {})
    
    # Determine grid resolution from first voxel size
    voxel_list = data.get('TheVoxels', [])
    if voxel_list:
        first_voxel = voxel_list[0]
        voxel_size = first_voxel.get('xSizeVoxel', 2.0)
    else:
        voxel_size = 2.0
    
    # Calculate number of voxels
    npx = int(xSizeWorld / voxel_size)
    npy = int(ySizeWorld / voxel_size)
    npz = int(zSizeWorld / voxel_size)
    
    print(f"[DEBUG] Grid resolution: npx={npx}, npy={npy}, npz={npz}")
    print(f"[DEBUG] Voxel size: {voxel_size}")
    
    # Initialize 3D array with zeros (vacuum)
    geometry_3d = np.zeros((npz, npy, npx), dtype=np.uint8)
    
    # Material name to ID mapping
    material_map = {
        'lead': 1,
        'scintillator': 2,
        'air': 0,
        'vacuum': 0
    }
    
    # Coordinate offset (world origin at center)
    x_offset = xSizeWorld / 2.0
    y_offset = ySizeWorld / 2.0
    z_offset = zSizeWorld / 2.0
    
    # Fill voxels from TheVoxels list
    count = 0
    for voxel_entry in voxel_list:
        if isinstance(voxel_entry, dict):
            x_pos = voxel_entry.get('xPosVoxel', 0.0)
            y_pos = voxel_entry.get('yPosVoxel', 0.0)
            z_pos = voxel_entry.get('zPosVoxel', 0.0)
            material_name = voxel_entry.get('materialVoxel', 'air').lower()
            material_id = material_map.get(material_name, 0)
            
            # Convert position to array indices
            # Position is from -(size/2) to +(size/2), map to [0, n]
            x_idx = int((x_pos + x_offset) / voxel_size)
            y_idx = int((y_pos + y_offset) / voxel_size)
            z_idx = int((z_pos + z_offset) / voxel_size)
            
            # Check bounds and fill
            if 0 <= x_idx < npx and 0 <= y_idx < npy and 0 <= z_idx < npz:
                geometry_3d[z_idx, y_idx, x_idx] = material_id
                count += 1
    
    print(f"[DEBUG] Filled {count} voxels out of {len(voxel_list)} total")
    print(f"[DEBUG] Non-zero voxels in array: {np.count_nonzero(geometry_3d)}")
    
    return geometry_3d, voxel_config


def create_projections(geometry_3d):
    """
    Create three orthogonal projections from 3D geometry.
    
    Returns:
        xy_proj: projection on XY plane (Z dimension summed)
        xz_proj: projection on XZ plane (Y dimension summed)
        yz_proj: projection on YZ plane (X dimension summed)
    """
    # Sum along different axes to get projections
    xy_proj = np.max(geometry_3d, axis=0)  # Project along Z
    xz_proj = np.max(geometry_3d, axis=1)  # Project along Y
    yz_proj = np.max(geometry_3d, axis=2)  # Project along X
    
    return xy_proj, xz_proj, yz_proj


def visualize_geometry(json_file, save_path=None, figsize=(16, 5)):
    """
    Load geometry and create visualization with three projections.
    
    Args:
        json_file: path to JSON geometry file
        save_path: optional path to save figure
        figsize: figure size in inches
    """
    print(f"[INFO] Loading geometry from: {json_file}")
    
    # Load and validate file
    if not Path(json_file).exists():
        print(f"[ERROR] File not found: {json_file}")
        return False
    
    # Load geometry
    data = load_geometry_json(json_file)
    geometry_3d, voxel_config = build_3d_array(data)
    
    print(f"[INFO] Geometry shape: {geometry_3d.shape}")
    print(f"[INFO] Voxel config: {voxel_config}")
    print(f"[INFO] Matplotlib backend: {matplotlib.get_backend()}")
    
    # Create projections
    xy_proj, xz_proj, yz_proj = create_projections(geometry_3d)
    
    # Ensure projections have actual data
    print(f"[DEBUG] XY projection stats: min={xy_proj.min()}, max={xy_proj.max()}, nonzero={np.count_nonzero(xy_proj)}")
    print(f"[DEBUG] XZ projection stats: min={xz_proj.min()}, max={xz_proj.max()}, nonzero={np.count_nonzero(xz_proj)}")
    print(f"[DEBUG] YZ projection stats: min={yz_proj.min()}, max={yz_proj.max()}, nonzero={np.count_nonzero(yz_proj)}")
    
    # Extract geometry name from file
    geom_name = Path(json_file).stem
    
    # Create figure with three subplots
    print(f"[INFO] Creating figure...")
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'Geometry Projections: {geom_name}', fontsize=14, fontweight='bold')
    
    # Use a better colormap and vmin/vmax
    vmin, vmax = 0, 1
    
    # XY projection (top view)
    print(f"[INFO] Rendering XY projection...")
    im1 = axes[0].imshow(xy_proj, cmap='binary', origin='lower', interpolation='nearest', vmin=0, vmax=1)
    axes[0].set_title('XY Projection (Top View)', fontweight='bold')
    axes[0].set_xlabel('X [voxels]')
    axes[0].set_ylabel('Y [voxels]')
    cbar1 = plt.colorbar(im1, ax=axes[0], label='Material')
    
    # XZ projection (front view)
    print(f"[INFO] Rendering XZ projection...")
    im2 = axes[1].imshow(xz_proj, cmap='binary', origin='lower', interpolation='nearest', vmin=0, vmax=1)
    axes[1].set_title('XZ Projection (Front View)', fontweight='bold')
    axes[1].set_xlabel('X [voxels]')
    axes[1].set_ylabel('Z [voxels]')
    cbar2 = plt.colorbar(im2, ax=axes[1], label='Material')
    
    # YZ projection (side view)
    print(f"[INFO] Rendering YZ projection...")
    im3 = axes[2].imshow(yz_proj, cmap='binary', origin='lower', interpolation='nearest', vmin=0, vmax=1)
    axes[2].set_title('YZ Projection (Side View)', fontweight='bold')
    axes[2].set_xlabel('Y [voxels]')
    axes[2].set_ylabel('Z [voxels]')
    cbar3 = plt.colorbar(im3, ax=axes[2], label='Material')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if save_path:
        print(f"[INFO] Saving to: {save_path}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[CORRECT] Visualization saved to: {save_path}")
        plt.close()
    else:
        # If no save path, auto-save to /tmp and try to display
        auto_save_path = f'/tmp/geometry_{geom_name}.png'
        print(f"[INFO] No save path specified, auto-saving to {auto_save_path}")
        plt.savefig(auto_save_path, dpi=150, bbox_inches='tight')
        print(f"[CORRECT] Visualization saved to: {auto_save_path}")
        
        # Try to display the window
        print(f"[INFO] Attempting to display window (matplotlib backend: {matplotlib.get_backend()})...")
        try:
            plt.show()
            print(f"[CORRECT] Visualization window displayed!")
        except Exception as e:
            print(f"[INFO] Could not open interactive window: {e}")
            print(f"[INFO] Visualization has been saved to {auto_save_path}")
            print(f"[INFO] You can open it with: xdg-open {auto_save_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Geant4 geometry from JSON file in three orthogonal projections.'
    )
    parser.add_argument(
        '--json', 
        type=str, 
        required=True,
        help='Path to geometry JSON file'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save the visualization (PNG/PDF). If not provided, shows interactive window.'
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[16, 5],
        help='Figure size in inches (width height)'
    )
    
    args = parser.parse_args()
    
    # Validate JSON file
    json_path = Path(args.json)
    if not json_path.exists():
        print(f"[ERROR] File not found: {args.json}")
        sys.exit(1)
    
    # Create visualization
    figsize = tuple(args.figsize) if args.figsize else (16, 5)
    success = visualize_geometry(args.json, save_path=args.save, figsize=figsize)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
