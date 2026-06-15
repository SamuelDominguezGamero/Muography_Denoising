#!/usr/bin/env python3
"""
Generate a perfectly balanced run2 dataset: rectangular blocks (filled only).

Strategy:
- Shape: rectangle_filled only (no hollow variants)
- Materials: all 7 (iron, steel, aluminium, silicon, water, lead, uranium) - no grouping
- Target: ~7371 files (1053 per material)
- Balance: equal distribution across materials, sizes, depths, positions
- Depths: diverse [4,6,8,10,12,14,16,18,20,22,24,26,28] to avoid old concentration

Filename convention (run2):
  shape_rectangle_filled_Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_ratio1_sx{sx}_sy{sy}_cx{cx}_cy{cy}_mat{mat}_dz{dz}.json
  
Ground truth: tensor_2D_POCA_..._mat{mat}_dz{dz}.npy
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


# ===========================================================================
# CONFIGURATION
# ===========================================================================

CREATOR_SCRIPT = "/home/samuel/Work/Muography_Denoising/MuonGeneration/test/create_other_geometries_not_words.py"
OUTPUT_JSON_DIR = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration/data/geometric_configurations_blocks")
OUTPUT_NPY_DIR = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration/data/ground_truth_data/run2_blocks")

# Ensure output directories exist
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_NPY_DIR.mkdir(parents=True, exist_ok=True)

# Shape: only rectangle_filled (no variants)
SHAPE = "rectangle_filled"

# Materials: all 7 (no grouping for run2 - only blocks)
MATERIALS = ["iron", "steel", "aluminium", "silicon", "water", "lead", "uranium"]

# Depth distribution: diverse values to avoid old {5, 10, 20} concentration
DEPTHS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]

# Sizes: 5 values for reasonable dataset size
SIZES_X = [14, 18, 22, 26, 30]
SIZES_Y = [14, 18, 22, 26, 30]

# Positions: 2 main positions (center and offset)
POSITIONS_X = [0, -10]
POSITIONS_Y = [0, -10]

# World dimensions (from Geant4)
WORLD_SIZE = 128  # cm (total world is 128x128x128)
DETECTOR_ZTOP = 54
DETECTOR_ZBOT = -54

# ===========================================================================
# MAIN GENERATION LOGIC
# ===========================================================================

def main():
    """Main generation pipeline."""
    
    start_time = datetime.now()
    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*80)
    print("GENERATE BALANCED RUN2: Rectangular blocks (filled only)")
    print("="*80)
    print(f"Start: {start_str}\n")
    
    print("Target:")
    print(f"  Total files: ~{len(MATERIALS) * len(SIZES_X) * len(SIZES_Y) * len(DEPTHS)}")
    print(f"  Shape: 1 ({SHAPE})")
    print(f"  Materials: {len(MATERIALS)}")
    print(f"  Combos: {len(MATERIALS)} (materials)")
    print(f"  Files per material: ~{len(SIZES_X) * len(SIZES_Y) * len(DEPTHS)}\n")
    
    print("Parameters:")
    print(f"  Depths: {DEPTHS} ({len(DEPTHS)} values)")
    print(f"  Sizes X: {SIZES_X} ({len(SIZES_X)} values)")
    print(f"  Sizes Y: {SIZES_Y} ({len(SIZES_Y)} values)")
    print(f"  Positions X: {POSITIONS_X}")
    print(f"  Positions Y: {POSITIONS_Y}\n")
    
    # =========================================================================
    # PHASE 1: BUILD COMBO LIST
    # =========================================================================
    print("="*80)
    print("PHASE 1: BUILD COMBO LIST")
    print("="*80 + "\n")
    
    all_combos = []
    
    for material in MATERIALS:
        mat_combos = 0
        
        # Iterate through all parameter combinations
        for sx in SIZES_X:
            for sy in SIZES_Y:
                for dz in DEPTHS:
                    # Cycle through positions
                    for cx in POSITIONS_X:
                        for cy in POSITIONS_Y:
                            all_combos.append((SHAPE, sx, sy, dz, cx, cy, material))
                            mat_combos += 1
        
        print(f"  {material:15s} → {mat_combos} files")
    
    total_combos = len(all_combos)
    print(f"\n  {'='*50}")
    print(f"  Total combos to generate: {total_combos}")
    print(f"  Files per material: {total_combos // len(MATERIALS)}")
    
    # =========================================================================
    # PHASE 2: FILE GENERATION
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 2: FILE GENERATION (subprocess calls)")
    print("="*80 + "\n")
    
    success_count = 0
    fail_count = 0
    failed_combos = []
    
    for i, (shape, sx, sy, dz, cx, cy, mat) in enumerate(all_combos, start=1):
        # Build filename
        json_filename = f"shape_{shape}_Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_ratio1_sx{sx}_sy{sy}_cx{cx}_cy{cy}_mat{mat}_dz{dz}.json"
        npy_filename = f"tensor_2D_POCA_mat{mat}_dz{dz}.npy"
        
        json_path = OUTPUT_JSON_DIR / json_filename
        npy_path = OUTPUT_NPY_DIR / npy_filename
        
        # Build subprocess command
        command = [
            "python3", str(CREATOR_SCRIPT),
            "--shape", shape,
            "--size_x", str(sx),
            "--size_y", str(sy),
            "--depth_z_cm", str(dz),
            "--center_x", str(cx),
            "--center_y", str(cy),
            "--material", mat,
            "--dimensions", "2D",
            "--output_json", str(json_path),
            "--output2D_density", str(npy_path),
        ]
        
        # Execute subprocess
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                success_count += 1
            else:
                fail_count += 1
                failed_combos.append((shape, mat, sx, sy, dz, cx, cy, result.stderr))
        except subprocess.TimeoutExpired:
            fail_count += 1
            failed_combos.append((shape, mat, sx, sy, dz, cx, cy, "TIMEOUT"))
        except Exception as e:
            fail_count += 1
            failed_combos.append((shape, mat, sx, sy, dz, cx, cy, str(e)))
        
        # Progress checkpoint every 500 files
        if i % 500 == 0 or i == total_combos:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = i / elapsed if elapsed > 0 else 0
            eta_seconds = (total_combos - i) / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
            print(f"  [{i:5d}/{total_combos}] OK ({rate:.1f} files/s, ETA {eta_minutes:.1f} min)")
    
    # =========================================================================
    # PHASE 3: VERIFY GENERATION
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 3: VERIFY GENERATION")
    print("="*80 + "\n")
    
    # Count actual files in directory
    actual_files = list(OUTPUT_JSON_DIR.glob("shape_*.json"))
    actual_count = len(actual_files)
    
    print(f"Generation complete!")
    print(f"  Time elapsed: {(datetime.now() - start_time).total_seconds() / 60:.1f} minutes\n")
    
    print(f"Statistics:")
    print(f"  ✓ Generated: {success_count}")
    print(f"  ✗ Failed: {fail_count}")
    print(f"  ⊘ Skipped (existed): {actual_count - success_count}\n")
    
    print(f"Actual files in directory: {actual_count}")
    print(f"Target: {total_combos} ± 100\n")
    
    if fail_count > 0:
        print(f"⚠️  FAILED COMBOS (first 10):")
        for combo in failed_combos[:10]:
            shape, mat, sx, sy, dz, cx, cy, err = combo
            print(f"  {shape} + {mat} (sx={sx}, sy={sy}, dz={dz}, cx={cx}, cy={cy})")
            print(f"    Error: {err[:80]}")
    
    if actual_count >= total_combos - 100:
        print(f"\nStatus: ✅ SUCCESS")
    else:
        print(f"\nStatus: ⚠️  PARTIAL (expected ~{total_combos}, got {actual_count})")
    
    # Save log
    log_path = Path("/tmp/generate_balanced_run2.log")
    with open(log_path, "w") as f:
        f.write(f"Generated: {actual_count}\n")
        f.write(f"Success: {success_count}\n")
        f.write(f"Failed: {fail_count}\n")
    
    print(f"\nLog saved to: {log_path}")
    print("="*80 + "\n")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
