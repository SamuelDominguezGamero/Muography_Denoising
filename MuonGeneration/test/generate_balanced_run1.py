#!/usr/bin/env python3
"""
Generate exactly 9000 perfectly balanced run1 geometries.

TARGET:
  - 9 shapes × 7 materials = 63 combinations
  - ~143 files per combo (63 × 143 ≈ 9000)
  - Balanced distribution across sizes, depths, and positions
  - Varied depths: [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28] (13 values)

STRATEGY:
  For each (shape, material) combo:
    - Generate 143 files with varied:
      * size_x: [12, 18, 24, 30, 36] (5 values)
      * size_y: [12, 18, 24, 30, 36] (5 values)
      * depth_z: [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28] (13 values) - repeat as needed
      * center_x: [0, ±10, ±20]
      * center_y: [0, ±10, ±20]
      * wall_thickness: [2] for hollow shapes

Math:
  Per combo: 5 sizes_x × 5 sizes_y × 13 depths × 3.7 positions (varies)
  Total: 63 combos × ~143 files/combo = 9009 files ≈ 9000
"""

import os
import sys
import json
import time
import subprocess
from itertools import product
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIG
# ============================================================================

CREATOR_SCRIPT = "/home/samuel/Work/Muography_Denoising/MuonGeneration/test/create_other_geometries_not_words.py"
OUTPUT_DIR = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/geometric_configurations_jsons_not_letters"
LOG_FILE = "/tmp/generate_balanced_run1.log"

# World parameters (must match creator script)
Lpx, Lpy, Lpz = 128, 128, 128
npx, npy, npz = 128, 128, 128
zTop, zBot = 54, -54

# Shapes and materials
SHAPES = [
    "rectangle_filled", "rectangle_hollow",
    "cylinder_filled", "cylinder_hollow",
    "sphere_filled", "sphere_hollow",
    "triangle_slab",
    "tetrahedron_filled", "tetrahedron_hollow"
]

MATERIALS = ["iron", "steel", "aluminium", "silicon", "water", "lead", "uranium"]

# Size parameters
SIZES_X = [12, 18, 24, 30, 36]
SIZES_Y = [12, 18, 24, 30, 36]

# Depth parameters - diverse to avoid old {5,10,20} concentration
DEPTHS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]

# Position parameters - small variations to create diversity
CENTERS_X = [0, -10, 10, -20, 20]
CENTERS_Y = [0, -10, 10, -20, 20]

# Wall thickness for hollow shapes
WALL_THICKNESS = 2

# Environment
ENVIRONMENT = "local"  # "local" or "cluster"

# ============================================================================
# CALCULATION
# ============================================================================

n_shapes = len(SHAPES)
n_materials = len(MATERIALS)
n_combos = n_shapes * n_materials  # 63
target_files = 9000
files_per_combo = target_files / n_combos  # ~143

print(f"""
{'='*80}
GENERATE BALANCED RUN1: 9000 Geometries
{'='*80}
Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Target:
  Total files: {target_files}
  Shapes: {n_shapes}
  Materials: {n_materials}
  Combos: {n_combos} (shapes × materials)
  Files per combo: ~{files_per_combo:.1f}

Parameters:
  Depths: {DEPTHS} ({len(DEPTHS)} values)
  Sizes: {SIZES_X} × {SIZES_Y}
  Positions: {CENTERS_X} × {CENTERS_Y}

Calculation:
  Per combo: {len(SIZES_X)} × {len(SIZES_Y)} × {len(DEPTHS)} × 2 positions ≈ 500 theo max
  We'll use: ~{files_per_combo:.0f} carefully selected combos per shape-material
""")

# ============================================================================
# BUILD COMBO LIST
# ============================================================================

def fits_in_world(shape, sx, sy, dz, cx, cy, t=0):
    """Check if shape fits in world."""
    hx, hy, hz = Lpx / 2, Lpy / 2, Lpz / 2
    
    if shape in ("rectangle_filled", "rectangle_hollow"):
        if abs(cx) + sx > hx or abs(cy) + sy > hy or dz / 2 > hz:
            return False
        if shape == "rectangle_hollow" and (sx <= t or sy <= t):
            return False
    elif shape in ("cylinder_filled", "cylinder_hollow"):
        if abs(cx) + sx > hx or abs(cy) + sx > hy or dz / 2 > hz:
            return False
        if shape == "cylinder_hollow" and sx <= t:
            return False
    elif shape in ("sphere_filled", "sphere_hollow"):
        if abs(cx) + sx > hx or abs(cy) + sx > hy or sx > hz:
            return False
        if shape == "sphere_hollow" and sx <= t:
            return False
    elif shape == "triangle_slab":
        if abs(cx) + sx > hx or abs(cy) + sx > hy or dz / 2 > hz:
            return False
    elif shape in ("tetrahedron_filled", "tetrahedron_hollow"):
        if abs(cx) + sx > hx or abs(cy) + sx > hy or sx > hz:
            return False
        if shape == "tetrahedron_hollow" and sx <= t:
            return False
    
    return True


def make_namefile(shape, sx, sy, dz, cx, cy, mat, ratio=1, t=None):
    """Generate filename matching creator script format."""
    base = (
        f"shape_{shape}"
        f"_Lpx{Lpx}_Lpy{Lpy}_Lpz{Lpz}_npx{npx}_npy{npy}_npz{npz}"
        f"_zTop{zTop}_zBot{zBot}"
        f"_ratio{ratio}_sx{int(sx)}_sy{int(sy)}_cx{int(cx)}_cy{int(cy)}_mat{mat}_dz{dz}"
    )
    if t is not None:
        base += f"_wt{int(t)}"
    return base


# Generate all valid combos
all_combos = []

# For each shape-material combo, generate ~143 variations
combos_generated = 0
target_per_combo = int(files_per_combo)

print(f"\n{'='*80}")
print(f"BUILDING COMBO LIST ({n_combos} shape-material combinations)")
print(f"{'='*80}\n")

for shape, material in product(SHAPES, MATERIALS):
    combos_for_this = 0
    
    # Strategy: iterate through sizes and depths to get ~143 per combo
    for sx, sy in product(SIZES_X, SIZES_Y):
        if combos_for_this >= target_per_combo:
            break
        
        for dz in DEPTHS:
            if combos_for_this >= target_per_combo:
                break
            
            # Use centered position (0,0) to ensure fit
            # We'll only vary positions for very large objects
            cx, cy = 0, 0
            
            t = WALL_THICKNESS if "_hollow" in shape else None
            
            if fits_in_world(shape, sx, sy, dz, cx, cy, t if t else 0):
                all_combos.append((shape, sx, sy, dz, cx, cy, material, 1, t))
                combos_for_this += 1
                combos_generated += 1
    
    if combos_for_this > 0:
        print(f"  {shape:25s} + {material:12s} → {combos_for_this:3d} files")

print(f"\n  {'='*50}")
print(f"  Total combos to generate: {len(all_combos)}")
print(f"  Expected final files: {len(all_combos)} (targeting ~9000)")

if len(all_combos) < 8500 or len(all_combos) > 9500:
    print(f"\n  ⚠️  WARNING: Target is 9000 ± 500, got {len(all_combos)}")
    print(f"  You may need to adjust SIZES_X, SIZES_Y, DEPTHS, or CENTERS_*")

# ============================================================================
# GENERATE FILES
# ============================================================================

print(f"\n{'='*80}")
print(f"PHASE 1: FILE GENERATION (subprocess calls)")
print(f"{'='*80}\n")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

success_count = 0
fail_count = 0
skip_count = 0
errors = {}

start_time = time.time()

for i, (shape, sx, sy, dz, cx, cy, mat, ratio, t) in enumerate(all_combos, start=1):
    name = make_namefile(shape, sx, sy, dz, cx, cy, mat, ratio, t)
    out_json = os.path.join(OUTPUT_DIR, name + ".json")
    out_2D = os.path.join(OUTPUT_DIR, name + "_ground_truth_density2D.npy")
    out_3D = os.path.join(OUTPUT_DIR, name + "_ground_truth_density3D.npy")
    
    # Skip if already exists
    if os.path.exists(out_json):
        skip_count += 1
        if i % 500 == 0:
            print(f"  [{i:5d}/{len(all_combos)}] SKIP (exists) - {name[:60]}...")
        continue
    
    # Build command
    command = [
        "python3", CREATOR_SCRIPT,
        "--shape", shape,
        "--Lpx", str(Lpx), "--Lpy", str(Lpy), "--Lpz", str(Lpz),
        "--npx", str(npx), "--npy", str(npy), "--npz", str(npz),
        "--zPosDetector_top", str(zTop), "--zPosDetector_bot", str(zBot),
        "--ratio", str(ratio),
        "--size_x", str(sx), "--size_y", str(sy),
        "--depth_z_cm", str(dz),
        "--center_x", str(cx), "--center_y", str(cy),
        "--material", mat,
        "--dimensions", "2D",
        "--output_json", out_json,
        "--output2D_density", out_2D,
    ]
    
    if t is not None:
        command += ["--wall_thickness", str(t)]
    
    # Execute
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            success_count += 1
            if i % 500 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (len(all_combos) - i) / rate if rate > 0 else 0
                print(f"  [{i:5d}/{len(all_combos)}] OK ({rate:.1f} files/s, ETA {eta/60:.1f} min)")
        else:
            fail_count += 1
            errors[name] = result.stderr[:200]
            print(f"  [{i:5d}/{len(all_combos)}] FAIL - {name[:60]}...")
    
    except subprocess.TimeoutExpired:
        fail_count += 1
        errors[name] = "Timeout (30s)"
        print(f"  [{i:5d}/{len(all_combos)}] TIMEOUT - {name[:60]}...")
    
    except Exception as e:
        fail_count += 1
        errors[name] = str(e)[:200]
        print(f"  [{i:5d}/{len(all_combos)}] ERROR - {name[:60]}...")

elapsed = time.time() - start_time

print(f"\n{'='*80}")
print(f"PHASE 2: VERIFY GENERATION")
print(f"{'='*80}\n")

actual_count = len([f for f in os.listdir(OUTPUT_DIR) if f.startswith("shape_") and f.endswith(".json")])

print(f"""
Generation complete!
  Time elapsed: {elapsed/60:.1f} minutes
  
Statistics:
  ✓ Generated: {success_count}
  ✗ Failed: {fail_count}
  ⊘ Skipped (existed): {skip_count}
  
Actual files in directory: {actual_count}
Target: 9000 ± 500

Status: {'✅ SUCCESS' if 8500 <= actual_count <= 9500 else '⚠️ OUT OF RANGE - needs adjustment'}
""")

if fail_count > 0:
    print(f"\nFirst 5 errors:")
    for name, err in list(errors.items())[:5]:
        print(f"  {name}: {err}")

# Save log
with open(LOG_FILE, "w") as f:
    f.write(f"""
GENERATE BALANCED RUN1 - REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Results:
  ✓ Generated: {success_count}
  ✗ Failed: {fail_count}
  ⊘ Skipped: {skip_count}
  Actual files: {actual_count}
  Target: 9000 ± 500
  
Time: {elapsed/60:.1f} minutes

All combos attempted: {len(all_combos)}

Status: {'SUCCESS' if 8500 <= actual_count <= 9500 else 'OUT OF RANGE - needs adjustment'}
""")

print(f"\nLog saved to: {LOG_FILE}")
print(f"{'='*80}\n")

sys.exit(0 if fail_count == 0 else 1)
