"""
================================================================================
create_h5_dataset.py

Converts MERGED_*.npy (POCA reconstructions) + ground_truth_density2D.npy 
into HDF5 datasets for TensorFlow UNET training.

WORKFLOW:
  1. Scans merged_poca_data/ for MERGED_*.npy files
  2. Parses geometric hyperparameters from filenames
  3. Loads corresponding ground_truth_density2D.npy
  4. Groups by image resolution (e.g., 128x128x3.h5)
  5. Creates separate HDF5 file per resolution with:
     - /training/images, /training/labels
     - /validation/images, /validation/labels
     - /test/images, /test/labels
     - /metadata/* (hyperparameters per sample)
  6. Dataset structure allows flexible appending (new data can be added later)
  7. Uses chunking + compression for efficient I/O on cluster

OUTPUT STRUCTURE:
  data/h5_datasets/
  ├── 128x128x3.h5
  ├── 256x256x3.h5
  └── ...
  Each H5 contains:
    /training/
      ├── images       (N_train, H, W, 3) POCA channels: [log_counts, mean_theta_sq_z, var_theta_z]
      ├── labels       (N_train, H, W, 1) Ground truth density
      └── metadata/*   Individual sample hyperparameters
    /validation/
      ├── images, labels, metadata/
    /test/
      ├── images, labels, metadata/

GEOMETRIC HYPERPARAMETERS (extracted from filename):
  _Lpx{value}        : X dimension of physical box
  _Lpy{value}        : Y dimension of physical box
  _Lpz{value}        : Z dimension of physical box
  _npx{value}        : Grid points X
  _npy{value}        : Grid points Y
  _npz{value}        : Grid points Z
  _zTop{value}       : Top detector position
  _zBot{value}       : Bottom detector position
  _spacing{value}    : Detector spacing
  _ratio{value}      : Scattering ratio parameter
  _FontX{value}      : Font size X
  _FontY{value}      : Font size Y
  _mat{material}     : Material type (iron, lead, etc)
  _word{word}        : Word/geometry type (MUON, MUNO, etc)
  _stroke{value}     : Stroke width

SIMULATION METADATA (critical for noise analysis):
  n_muons_total      : Total number of muons in simulation (stored per sample + global)
                       IMPORTANT: Higher n_muons → lower noise, better statistics
================================================================================
"""

import numpy as np
import h5py
import os
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import re


# ==============================================================================
# CONFIGURATION
# ==============================================================================

ENVIRONMENT = "cluster"  # Options: "cluster" | "local"

if ENVIRONMENT == "cluster":
    PATH_MERGED = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/merged_poca_data"
    PATH_GT_2D = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/ground_truth_data/2Dimensions"
    PATH_OUTPUT = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/h5_datasets"

elif ENVIRONMENT == "local":
    PATH_MERGED = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/merged_poca_data"
    PATH_GT_2D = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/ground_truth_data/2Dimensions"
    PATH_OUTPUT = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/h5_datasets"

# Dataset split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# HDF5 compression parameters
COMPRESSION = 'gzip'
COMPRESSION_OPTS = 4  # 0-9: 4 is good balance between compression & speed

# Random seed for reproducibility
RANDOM_SEED = 42

# === CRITICAL SIMULATION PARAMETER ===
# Total number of muons used in ALL simulations
# This is a critical metadata: directly affects noise level (more muons → less noise)
# IMPORTANT: Update this value when running simulations with different muon counts!
TOTAL_MUONS_PER_SIMULATION = 1_000_000  # 1 million muons per geometry


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def parse_geometric_hyperparameters(filename: str) -> Dict[str, any]:
    """
    Extract geometric hyperparameters from filename.
    
    Filename format:
      _Lpx{Lpx}_Lpy{Lpy}_Lpz{Lpz}_npx{npx}_npy{npy}_npz{npz}_zTop{zTop}_zBot{zBot}
      _spacing{spacing}_ratio{ratio}_FontX{FontX}_FontY{FontY}
      _mat{material}_word{word}_stroke{stroke}
    
    Args:
        filename: base filename (without extension)
    
    Returns:
        Dictionary with extracted parameters
    """
    params = {}
    
    # Define pattern for each parameter
    patterns = {
        'Lpx': r'_Lpx(\d+)',
        'Lpy': r'_Lpy(\d+)',
        'Lpz': r'_Lpz(\d+)',
        'npx': r'_npx(\d+)',
        'npy': r'_npy(\d+)',
        'npz': r'_npz(\d+)',
        'zTop': r'_zTop(-?\d+)',
        'zBot': r'_zBot(-?\d+)',
        'spacing': r'_spacing(\d+)',
        'ratio': r'_ratio(\d+)',
        'FontX': r'_FontX(\d+)',
        'FontY': r'_FontY(\d+)',
        'material': r'_mat(\w+)',
        'word': r'_word(\w+)',
        'stroke': r'_stroke(\d+)',
        'n_muons': r'_Muons_(\d+(?:_\d+)*)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, filename)
        if match:
            value = match.group(1)
            # Convert numeric values to int, keep strings as-is
            try:
                # Remove underscores from numeric strings (e.g., "1_000_000" -> "1000000")
                clean_value = value.replace('_', '')
                params[key] = int(clean_value)
            except ValueError:
                params[key] = value
        else:
            # Only warn for critical parameters
            if key in ['npx', 'npy', 'n_muons']:
                print(f"[WARNING] Could not extract '{key}' from: {filename}")
    
    return params


def get_poca_channels(poca_dict: dict) -> np.ndarray:
    """
    Extract 3 POCA channels from merged data dictionary.
    
    Channels:
      0: log_counts       - Logarithmic muon counts
      1: mean_theta_sq_z  - Mean scattering angle squared
      2: var_theta_z      - Variance of scattering angle
    
    Args:
        poca_dict: Dictionary from MERGED_*.npy (loaded with allow_pickle=True)
    
    Returns:
        Array of shape (H, W, 3) with dtype float32
    """
    log_counts = poca_dict["log_counts"][:, :, 0]          # (H, W)
    mean_theta_sq = poca_dict["mean_theta_sq_z"][:, :, 0]  # (H, W)
    var_theta = poca_dict["var_theta_z"][:, :, 0]          # (H, W)
    
    # Stack into 3-channel image
    channels = np.stack([log_counts, mean_theta_sq, var_theta], axis=-1)
    return channels.astype(np.float32)


def load_sample_pair(merged_path: str, gt_path: str, geom_name: str) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Load POCA image and ground truth label pair.
    
    Args:
        merged_path: Full path to MERGED_*.npy file
        gt_path: Directory containing ground_truth_density2D.npy files
        geom_name: Geometry name (used to construct GT filename)
    
    Returns:
        (image, label, success) tuple
        - image: (H, W, 3) float32 POCA channels
        - label: (H, W, 1) float32 ground truth density
        - success: bool indicating if loading succeeded
    """
    try:
        # Load POCA
        poca_data = np.load(merged_path, allow_pickle=True).item()
        image = get_poca_channels(poca_data)
        
        # Load ground truth
        gt_file = Path(gt_path) / f"{geom_name}_ground_truth_density2D.npy"
        
        if not gt_file.exists():
            print(f"  [SKIP] Ground truth not found: {gt_file.name}")
            return None, None, False
        
        label = np.load(str(gt_file), allow_pickle=True).astype(np.float32)
        
        # Ensure label is (H, W, 1)
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        elif label.ndim != 3 or label.shape[2] != 1:
            print(f"  [SKIP] Label shape invalid: {label.shape}")
            return None, None, False
        
        # Validate shapes match
        if image.shape[:2] != label.shape[:2]:
            print(f"  [SKIP] Shape mismatch: image {image.shape} vs label {label.shape}")
            return None, None, False
        
        return image, label, True
    
    except Exception as e:
        print(f"  [ERROR] {str(e)}")
        return None, None, False


def create_flexible_h5_split(h5_file: h5py.File, split_name: str, 
                             images: List[np.ndarray], labels: List[np.ndarray], 
                             hyperparams: List[Dict]) -> int:
    """
    Create HDF5 groups for a dataset split (train/val/test).
    Uses chunking to allow appending new data later.
    
    Args:
        h5_file: Open HDF5 file
        split_name: Split name ("training", "validation", "test")
        images: List of image arrays (H, W, 3)
        labels: List of label arrays (H, W, 1)
        hyperparams: List of hyperparameter dicts
    
    Returns:
        Number of samples added
    """
    if len(images) == 0:
        print(f"  [{split_name}] Skipping (no data)")
        return 0
    
    split_group = h5_file.create_group(split_name)
    
    # Get shapes from first sample
    H, W, C = images[0].shape
    N = len(images)
    
    # Create image dataset with chunking (allows append later)
    img_shape = (N, H, W, C)
    chunk_shape = (1, H, W, C)  # Chunk by individual sample for easy appending
    
    img_dset = split_group.create_dataset(
        'images',
        shape=img_shape,
        dtype=np.float32,
        chunks=chunk_shape,
        compression=COMPRESSION,
        compression_opts=COMPRESSION_OPTS,
        maxshape=(None, H, W, C)  # Allow growing along N dimension
    )
    
    # Create label dataset (same structure)
    lbl_shape = (N, H, W, 1)
    lbl_dset = split_group.create_dataset(
        'labels',
        shape=lbl_shape,
        dtype=np.float32,
        chunks=(1, H, W, 1),
        compression=COMPRESSION,
        compression_opts=COMPRESSION_OPTS,
        maxshape=(None, H, W, 1)
    )
    
    # Populate datasets
    for i, (img, lbl) in enumerate(zip(images, labels)):
        img_dset[i] = img
        lbl_dset[i] = lbl
    
    # Create metadata subgroup for hyperparameters
    meta_group = split_group.create_group('metadata')
    
    for i, params in enumerate(hyperparams):
        sample_meta = meta_group.create_group(f'sample_{i:06d}')
        for key, value in params.items():
            sample_meta.attrs[key] = value
    
    # Add split-level attributes
    split_group.attrs['n_samples'] = N
    split_group.attrs['image_shape'] = (H, W, C)
    split_group.attrs['label_shape'] = (H, W, 1)
    
    print(f"  [{split_name}] {N} samples ({100*N/(N+0):0.0f}%)")  # Will be corrected after totals
    return N


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

def main():
    print("\n" + "="*80)
    print("CREATING HDF5 DATASETS FOR UNET TRAINING")
    print("="*80 + "\n")
    
    # Validate paths
    for path_name, path in [("MERGED", PATH_MERGED), ("GT_2D", PATH_GT_2D)]:
        if not os.path.exists(path):
            print(f"[ERROR] {path_name} directory not found: {path}")
            sys.exit(1)
    
    # Create output directory
    Path(PATH_OUTPUT).mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # STEP 1: Scan and collect data
    # ============================================================
    print("[STEP 1] Scanning for MERGED_*.npy files...")
    
    merged_files = sorted(glob.glob(os.path.join(PATH_MERGED, "MERGED_*.npy")))
    print(f"  Found {len(merged_files)} files\n")
    
    if len(merged_files) == 0:
        print("[ERROR] No MERGED files found. Run loop_configuration_files.py first.")
        sys.exit(1)
    
    # ============================================================
    # STEP 2: Load and organize by resolution
    # ============================================================
    print("[STEP 2] Loading data and organizing by resolution...")
    
    # Dictionary: resolution_key -> { 'images': [], 'labels': [], 'hyperparams': [] }
    resolution_groups = {}
    
    for idx, merged_file in enumerate(merged_files, 1):
        filename = Path(merged_file).stem  # Remove extension
        
        # Extract geometry name (remove MERGED_ prefix and 2D/3D suffix)
        geom_name = re.sub(r'^MERGED_', '', filename)
        geom_name = re.sub(r'_(2D|3D)$', '', geom_name)
        
        print(f"  [{idx}/{len(merged_files)}] {geom_name[:60]}...", end=" ", flush=True)
        
        # Parse hyperparameters
        hyperparams = parse_geometric_hyperparameters(geom_name)
        
        if 'npx' not in hyperparams or 'npy' not in hyperparams:
            print("[SKIP] Could not extract resolution")
            continue
        
        npx = hyperparams['npx']
        npy = hyperparams['npy']
        resolution_key = f"{npx}x{npy}x3"  # Assuming 3 channels (POCA)
        
        # Load sample pair
        image, label, success = load_sample_pair(merged_file, PATH_GT_2D, geom_name)
        
        if not success:
            print("[FAIL]")
            continue
        
        # Validate resolution matches filename
        if image.shape != (npy, npx, 3):
            print(f"[SKIP] Shape mismatch: expected ({npy}, {npx}, 3), got {image.shape}")
            continue
        
        # Add to appropriate resolution group
        if resolution_key not in resolution_groups:
            resolution_groups[resolution_key] = {
                'images': [],
                'labels': [],
                'hyperparams': [],
                'geom_names': []
            }
        
        # Add critical metadata: total muons in simulation
        hyperparams['n_muons_total'] = TOTAL_MUONS_PER_SIMULATION
        
        resolution_groups[resolution_key]['images'].append(image)
        resolution_groups[resolution_key]['labels'].append(label)
        resolution_groups[resolution_key]['hyperparams'].append(hyperparams)
        resolution_groups[resolution_key]['geom_names'].append(geom_name)
        
        print("[OK]")
    
    print(f"\n  Total resolutions found: {len(resolution_groups)}\n")
    
    if not resolution_groups:
        print("[ERROR] No valid data loaded.")
        sys.exit(1)
    
    # ============================================================
    # STEP 3: Create HDF5 file per resolution
    # ============================================================
    print("[STEP 3] Creating HDF5 files per resolution...\n")
    
    for resolution_key, group_data in sorted(resolution_groups.items()):
        print(f"  Resolution: {resolution_key}")
        print(f"    Total samples: {len(group_data['images'])}")
        
        # Split data into train/val/test
        np.random.seed(RANDOM_SEED)
        n_total = len(group_data['images'])
        
        indices = np.arange(n_total)
        np.random.shuffle(indices)
        
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        # Prepare data for each split
        splits = {
            'training': train_idx,
            'validation': val_idx,
            'test': test_idx
        }
        
        # Create HDF5 file
        h5_filename = f"{resolution_key}.h5"
        h5_filepath = os.path.join(PATH_OUTPUT, h5_filename)
        
        print(f"    Output: {h5_filename}")
        
        with h5py.File(h5_filepath, 'w') as h5file:
            total_samples = 0
            
            for split_name, split_indices in splits.items():
                split_images = [group_data['images'][i] for i in split_indices]
                split_labels = [group_data['labels'][i] for i in split_indices]
                split_hyperparams = [group_data['hyperparams'][i] for i in split_indices]
                
                n_samples = create_flexible_h5_split(
                    h5file, split_name,
                    split_images, split_labels, split_hyperparams
                )
                total_samples += n_samples
            
            # Add global metadata
            h5file.attrs['resolution'] = resolution_key
            h5file.attrs['total_samples'] = total_samples
            h5file.attrs['train_ratio'] = TRAIN_RATIO
            h5file.attrs['val_ratio'] = VAL_RATIO
            h5file.attrs['test_ratio'] = TEST_RATIO
            h5file.attrs['compression'] = COMPRESSION
            h5file.attrs['compression_opts'] = COMPRESSION_OPTS
            h5file.attrs['random_seed'] = RANDOM_SEED
            h5file.attrs['n_muons_per_simulation'] = TOTAL_MUONS_PER_SIMULATION
            
            # Store geometry names as dataset
            h5file.create_dataset(
                'geometry_names',
                data=[n.encode('utf-8') for n in group_data['geom_names']]
            )
        
        file_size_mb = os.path.getsize(h5_filepath) / 1e6
        print(f"    File size: {file_size_mb:.1f} MB\n")
    
    # ============================================================
    # STEP 4: Summary and validation
    # ============================================================
    print("[STEP 4] Validation and Summary\n")
    
    h5_files = glob.glob(os.path.join(PATH_OUTPUT, "*.h5"))
    
    for h5_path in sorted(h5_files):
        filename = Path(h5_path).name
        print(f"  {filename}:")
        
        with h5py.File(h5_path, 'r') as h5file:
            for split in ['training', 'validation', 'test']:
                if split in h5file:
                    n_samples = h5file[split].attrs['n_samples']
                    img_shape = tuple(h5file[split].attrs['image_shape'])
                    print(f"    /{split}: {n_samples} samples, shape {img_shape}")
    
    print("\n" + "="*80)
    print("✓ SUCCESSFULLY CREATED HDF5 DATASETS")
    print("="*80 + "\n")
    
    print(f"Output directory: {PATH_OUTPUT}\n")
    
    print("NEXT STEPS:")
    print("  1. Train UNET model (see UNET0_2D.py)")
    print("  2. To append more data later:")
    print("     - Use h5py to resize datasets: dataset.resize(new_size)")
    print("     - Add new samples incrementally\n")
    
    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Dataset creation stopped by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
