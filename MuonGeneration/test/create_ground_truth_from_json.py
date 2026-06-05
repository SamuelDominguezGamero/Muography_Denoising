"""
Convert Geant4 JSON geometries → 128×128×3 numpy tensors (XY, XZ, YZ projections)

Usage:
  # Single or multiple JSON files:
  python create_ground_truth_from_json.py --json file.json
  python create_ground_truth_from_json.py --json a.json b.json

  # All JSONs in a folder:
  python create_ground_truth_from_json.py --folder /path/to/jsons/

  # Custom output directory:
  python create_ground_truth_from_json.py --folder /path/to/jsons/ --output_path /path/to/output/

  # Optional: --world_size (default 128.0 cm), --voxel_size (default 1.0 cm)
"""

import json
import argparse
from pathlib import Path
import numpy as np
import sys

# Detect environment and set paths
if Path("/gpfs/projects/cms/dominguezs/data/ground_truth_data").exists():
    OUT_DIR_3D = Path("/gpfs/projects/cms/dominguezs/data/ground_truth_data/3Dimensions")
    OUT_DIR_2D = Path("/gpfs/projects/cms/dominguezs/data/ground_truth_data/2Dimensions")
    JSON_DIR = Path("/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/geometric_configurations_json")
else:
    OUT_DIR_3D = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration/data/ground_truth_data/3Dimensions")
    OUT_DIR_2D = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration/data/ground_truth_data/2Dimensions")
    JSON_DIR = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration/data/geometric_configurations_json")
    # JSON_DIR = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration/data/geometric_configurations_jsons_not_letters")

OUT_DIR_3D.mkdir(parents=True, exist_ok=True)
OUT_DIR_2D.mkdir(parents=True, exist_ok=True)


# Process each JSON
def process_json(args_json, args_world_size=128.0, args_voxel_size=1.0, output_path=None):

    out_dir_2d = Path(output_path) if output_path is not None else OUT_DIR_2D
    out_dir_2d.mkdir(parents=True, exist_ok=True)

    for json_name in args_json:
        json_path = JSON_DIR / json_name if "/" not in json_name else Path(json_name)

        if not json_path.exists():
            print(f"[ERROR] {json_path} not found")
            continue

        print(f"\nProcessing: {json_path.name}...")

        # Check if tensors already exist
        out_path_3d = OUT_DIR_3D / f"tensor_3D_{json_path.stem}.npy"
        out_path_2d = out_dir_2d / f"tensor_2D_{json_path.stem}.npy"

        if out_path_3d.exists() and out_path_2d.exists():
            print(f"  ⊘ Already processed, skipping")
            continue

        # Load geometry
        with open(json_path) as f:
            geometry = json.load(f)

        # Create 3D tensor
        n = int(args_world_size / args_voxel_size)
        tensor_3d = np.zeros((n, n, n), dtype=np.uint8)
        world_min = -args_world_size / 2.0

        # Fill tensor from voxels
        for voxel in geometry.get("TheVoxels", []):
            if voxel.get("materialVoxel", "air").lower() == "air":
                continue

            # Get voxel boundaries in cm
            x_min = voxel["xPosVoxel"] - voxel["xSizeVoxel"] / 2
            x_max = voxel["xPosVoxel"] + voxel["xSizeVoxel"] / 2
            y_min = voxel["yPosVoxel"] - voxel["ySizeVoxel"] / 2
            y_max = voxel["yPosVoxel"] + voxel["ySizeVoxel"] / 2
            z_min = voxel["zPosVoxel"] - voxel["zSizeVoxel"] / 2
            z_max = voxel["zPosVoxel"] + voxel["zSizeVoxel"] / 2

            # Convert to tensor indices
            xi_min = max(0, min(int((x_min - world_min) / args.voxel_size), n))
            xi_max = max(0, min(int((x_max - world_min) / args.voxel_size), n))
            yi_min = max(0, min(int((y_min - world_min) / args.voxel_size), n))
            yi_max = max(0, min(int((y_max - world_min) / args.voxel_size), n))
            zi_min = max(0, min(int((z_min - world_min) / args.voxel_size), n))
            zi_max = max(0, min(int((z_max - world_min) / args.voxel_size), n))

            tensor_3d[yi_min:yi_max, xi_min:xi_max, zi_min:zi_max] = 1

        # Generate 2D projections (max along each axis)
        xy = np.max(tensor_3d, axis=2)         # max over Z
        xz = np.max(tensor_3d, axis=0).T       # max over Y, transpose
        yz = np.max(tensor_3d, axis=1).T       # max over X, transpose

        # Stack into 128×128×3 tensor
        tensor_2d = np.stack([xy, xz, yz], axis=2)

        # Save 3D tensor
        # np.save(out_path_3d, tensor_3d)

        # Save 2D tensor
        np.save(out_path_2d, tensor_2d)

        occupancy = 100.0 * np.sum(tensor_3d) / (n ** 3)
        print(f"  ✓ 3D: {out_path_3d.name} (shape: {tensor_3d.shape})")
        print(f"  ✓ 2D: {out_path_2d.name} (shape: {tensor_2d.shape}, occupancy: {occupancy:.2f}%)")

    print("\n[Done]")
    print("Saved 2D GT map at:", out_path_2d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON → 128×128×3 tensor (XY, XZ, YZ projections)")
    parser.add_argument("--json", type=str, nargs='+', default=[], help="JSON filename(s)")
    parser.add_argument("--folder", type=str, default=None, help="Folder path: glob all *.json files inside it")
    parser.add_argument("--output_path", type=str, default=None, help="Output directory for 2D ground truth tensors. Defaults to the auto-detected OUT_DIR_2D.")
    parser.add_argument("--world_size", type=float, default=128.0, help="World size in cm")
    parser.add_argument("--voxel_size", type=float, default=1.0, help="Voxel size in cm")
    args = parser.parse_args()

    json_files = list(args.json)
    if args.folder:
        folder_path = Path(args.folder)
        folder_jsons = [str(p) for p in sorted(folder_path.glob("*.json"))]
        print(f"Found {len(folder_jsons)} JSON files in {folder_path}")
        json_files.extend(folder_jsons)

    if not json_files:
        parser.error("Provide at least one JSON file via --json or a folder via --folder")

    process_json(json_files, args.world_size, args.voxel_size, args.output_path)
