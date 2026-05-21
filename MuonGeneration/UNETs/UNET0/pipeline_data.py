#!/usr/bin/env python3
"""
Pipeline principal: TAR → POCA_projections → JSON → GT → H5 dataset
Output: dataset0.h5 listo para entrenar UNET


dataset0.h5 (Archivo Raíz)
│
├── train (Grupo)
│   ├── poca (Dataset) ──> [N_train, 128, 128, 3] | uint8 | Chunks: (1, 128, 128, 3)
│   └── gt   (Dataset) ──> [N_train, 128, 128, 3] | uint8 | Chunks: (1, 128, 128, 3)
│
├── val   (Grupo)
│   ├── poca (Dataset) ──> [N_val,   128, 128, 3] | uint8 | Chunks: (1, 128, 128, 3)
│   └── gt   (Dataset) ──> [N_val,   128, 128, 3] | uint8 | Chunks: (1, 128, 128, 3)
│
└── test  (Grupo)
    ├── poca (Dataset) ──> [N_test,  128, 128, 3] | uint8 | Chunks: (1, 128, 128, 3)
    └── gt   (Dataset) ──> [N_test,  128, 128, 3] | uint8 | Chunks: (1, 128, 128, 3)

"""

import sys
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from data_utils import (
    extract_poca_from_source,
    create_gt_from_json,
    auto_match_datasets,
    create_h5_dataset
)


# ===========================================================================
# CONTROL
# ===========================================================================

obtain_ground_truth_from_jsons = True  # to be done
match_poca_with_gt_            = True
create_h5_dataset_             = True  # to be done 


# ===========================================================================
# HELPERS
# ===========================================================================

def print_phase(phase_num, phase_name):
    """Print phase header with timestamp"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{ts} [PHASE {phase_num}] {phase_name}")
    print("-" * 80)

def print_progress(current, total, label=""):
    """Print progress line with percentage"""
    pct = 100.0 * current / total if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    extra = f" {label}" if label else ""
    print(f"  [{bar}] {current:6d}/{total:6d} ({pct:5.1f}%){extra}", end='\r')

# ===========================================================================
# PATHS
# ===========================================================================

BASE        = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration")
DATA        = BASE / "data"

SOURCE_ROOT = DATA / "simulation_data/run0.tar"
JSON_DIR    = DATA / "geometric_configurations_json"
DATASETS    = DATA / "datasets"

POCA        = DATA / "simulation_data/run0_POCA_projections"
GT_3D       = DATA / "ground_truth_data/3Dimensions"
GT_2D       = DATA / "ground_truth_data/2Dimensions"
H5_FILE     = DATASETS / "dataset0.h5"

WORLD_SIZE = 128.0
VOXEL_SIZE = 1.0

# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def main():
    pipeline_start = time.time()
    
    print("\n" + "=" * 80)
    print("UNET0 DATA PIPELINE - DATASET GENERATION")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {H5_FILE}")
    print("=" * 80)
    
    # Validate inputs
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{ts} [VALIDATION] Checking inputs...")
    if not SOURCE_ROOT.exists():
        sys.exit(f"[ERROR] Source not found: {SOURCE_ROOT}")
    if not JSON_DIR.exists():
        sys.exit(f"[ERROR] JSON dir not found: {JSON_DIR}")
    
    json_files = list(JSON_DIR.glob("*.json"))
    if not json_files:
        sys.exit(f"[ERROR] No JSON files found in {JSON_DIR}")
    
    print(f"  ✓ Source valid: {SOURCE_ROOT.name}")
    print(f"  ✓ Found {len(json_files)} JSON geometries")
    
    #########################################################
    # PHASE 1: Extract POCA projections from TAR or directory
    print_phase(1, "EXTRACT POCA PROJECTIONS FROM TAR")
    phase1_start = time.time()

    poca_files = extract_poca_from_source(SOURCE_ROOT, POCA, WORLD_SIZE)
    if not poca_files:
        sys.exit("[ERROR] No POCA files generated")
        print(f"\n  ✓ Generated {len(poca_files)} POCA projection files")

    phase1_time = time.time() - phase1_start
    print(f"  ⏱  Time: {phase1_time:.1f}s")

    #########################################################
    # PHASE 2: Create GT from JSON
    print_phase(2, "CREATE GROUND TRUTH FROM JSON GEOMETRIES")
    phase2_start = time.time()
    
    gt_files_2d = []
    errors = 0
    
    if obtain_ground_truth_from_jsons:
        for idx, json_path in enumerate(json_files, 1):
            try:
                _, path_2d, metadata = create_gt_from_json(
                    json_path, GT_3D, GT_2D, WORLD_SIZE, VOXEL_SIZE
                )
                gt_files_2d.append(path_2d)
                
                # Smooth progress bar
                elapsed = time.time() - phase2_start
                rate = idx / elapsed if elapsed > 0 else 0
                remaining = (len(json_files) - idx) / rate if rate > 0 else 0
                
                print_progress(idx, len(json_files), 
                            f"({rate:.1f} files/s, ETA: {remaining:.0f}s)")
            
            except Exception as e:
                errors += 1
                if errors <= 3:  # Print first 3 errors only
                    print(f"\n  [WARN] {json_path.name}: {str(e)[:60]}")
    
        phase2_time = time.time() - phase2_start
        
        if not gt_files_2d:
            sys.exit("[ERROR] No GT files generated")
        
        print(f"\n  ✓ Created {len(gt_files_2d)} GT files ({errors} errors)")
        print(f"  ⏱  Time: {phase2_time:.1f}s ({phase2_time/len(json_files):.3f}s/file)")
    else:
        print("[INFO] ----- SKIPPED obtaining GT from JSON files")

    #########################################################
    
    # PHASE 3: Match POCA with GT
    if match_poca_with_gt_:
        print_phase(3, "MATCH POCA WITH GROUND TRUTH")
        phase3_start = time.time()
        pairs = auto_match_datasets(POCA, GT_2D)
        phase3_time = time.time() - phase3_start
        
        if not pairs:
            sys.exit("[ERROR] No matching pairs found")
        
        print(f"  Input POCA files:     {len(poca_files):6d}")
        print(f"  Input GT files:       {len(gt_files_2d):6d}")
        print(f"  Matched pairs:        {len(pairs):6d} ({100*len(pairs)/len(poca_files):.1f}%)")
        print(f"  ⏱  Time: {phase3_time:.2f}s")
    else:
        print("[INFO] ----- SKIPPED matching POCA with GT")

    #########################################################
    # PHASE 4: Create H5 dataset
    #########################################################
    
    print_phase(4, "CREATE H5 DATASET WITH TRAIN/VAL/TEST SPLITS")
    phase4_start = time.time()

    # Ejecuta la función optimizada con buffers y chunking
    create_h5_dataset(pairs, H5_FILE)
    phase4_time = time.time() - phase4_start

    print(f"  ⏱  Time: {phase4_time:.1f}s")


    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE ✓")
    print("=" * 80)
    print(f"\nOutput file: {H5_FILE}")
    print(f"Size: {H5_FILE.stat().st_size / (1024**3):.2f} GB")
    print(f"\nNext step:")
    print(f"  1. Visualize samples: python3 checkear_instancias_fromh5.py")
    print(f"  2. Train UNET:        python3 train_unet.py")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
