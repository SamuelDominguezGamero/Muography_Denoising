#!/usr/bin/env python3
"""
Analyze the balanced run1 dataset after generation.
Reports on shape, material, and depth distributions.
"""

import os
import re
from collections import Counter
from pathlib import Path

OUTPUT_DIR = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/geometric_configurations_jsons_not_letters"

def extract_metadata(filename):
    """Parse shape_* filename to extract shape, material, sx, sy, cx, cy, dz, wt, hollow."""
    # Pattern: shape_{shape}_Lpx..._{params}_mat{mat}_dz{dz}[_wt{wt}]
    pattern = (
        r"shape_(?P<shape>[a-z_]+)"
        r"_Lpx\d+_Lpy\d+_Lpz\d+_npx\d+_npy\d+_npz\d+"
        r"_zTop\d+_zBot-?\d+"
        r"_ratio\d+"
        r"_sx(?P<sx>\d+)_sy(?P<sy>\d+)"
        r"_cx(?P<cx>-?\d+)_cy(?P<cy>-?\d+)"
        r"_mat(?P<mat>[a-z]+)"
        r"_dz(?P<dz>\d+(?:\.\d+)?)"
        r"(?:_wt(?P<wt>\d+))?"
    )
    
    match = re.match(pattern, filename)
    if not match:
        return None
    
    data = match.groupdict()
    shape = data["shape"]
    material = data["mat"]
    hollow = "_hollow" in shape
    
    # Group materials as in the balanced dataset
    if material in ("iron", "steel"):
        mat_group = "iron_steel"
    elif material in ("aluminium", "silicon"):
        mat_group = "aluminium_silicon"
    else:
        mat_group = material
    
    return {
        "filename": filename,
        "shape": shape,
        "material": material,
        "mat_group": mat_group,
        "sx": int(data["sx"]),
        "sy": int(data["sy"]),
        "cx": int(data["cx"]),
        "cy": int(data["cy"]),
        "dz": float(data["dz"]),
        "hollow": hollow,
        "wt": int(data["wt"]) if data["wt"] else None,
    }


# Read all files
print(f"\n{'='*80}")
print(f"ANALYZE BALANCED RUN1 DATASET")
print(f"{'='*80}\n")

json_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("shape_") and f.endswith(".json")]
print(f"Found {len(json_files)} JSON files in {OUTPUT_DIR}\n")

if len(json_files) == 0:
    print("No files found yet. Generation may still be in progress.")
    exit(1)

# Parse metadata
all_data = []
for filename in json_files:
    metadata = extract_metadata(filename)
    if metadata:
        all_data.append(metadata)

print(f"Successfully parsed: {len(all_data)} / {len(json_files)}")

if len(all_data) == 0:
    print("Failed to parse any filenames. Check filename pattern.")
    exit(1)

# ============================================================================
# SHAPE DISTRIBUTION
# ============================================================================

shapes_counter = Counter(d["shape"] for d in all_data)
total = len(all_data)
target_per_shape = total / 9

print(f"\n{'='*80}")
print(f"SHAPE DISTRIBUTION (target: {target_per_shape:.0f} per shape, {100/9:.1f}%)")
print(f"{'='*80}\n")

shape_summary = []
for shape in sorted(shapes_counter.keys()):
    count = shapes_counter[shape]
    pct = 100 * count / total
    status = "✓" if abs(pct - 100/9) < 3 else "⚠" if abs(pct - 100/9) < 5 else "❌"
    shape_summary.append((shape, count, pct, status))
    print(f"  {shape:25s}  {count:5d}  ({pct:5.1f}%)  {status}")

shape_ratio = max(shapes_counter.values()) / min(shapes_counter.values())
print(f"\n  Ratio (max/min): {shape_ratio:.2f}x (goal < 1.3x)")

# ============================================================================
# MATERIAL DISTRIBUTION
# ============================================================================

mat_counter = Counter(d["mat_group"] for d in all_data)
target_per_mat = total / 5

print(f"\n{'='*80}")
print(f"MATERIAL GROUP DISTRIBUTION (target: {target_per_mat:.0f} per group, 20%)")
print(f"{'='*80}\n")

mat_summary = []
for mat in sorted(mat_counter.keys()):
    count = mat_counter[mat]
    pct = 100 * count / total
    status = "✓" if abs(pct - 20) < 3 else "⚠" if abs(pct - 20) < 5 else "❌"
    mat_summary.append((mat, count, pct, status))
    print(f"  {mat:20s}  {count:5d}  ({pct:5.1f}%)  {status}")

mat_ratio = max(mat_counter.values()) / min(mat_counter.values())
print(f"\n  Ratio (max/min): {mat_ratio:.2f}x (goal < 1.3x)")

# ============================================================================
# DEPTH DISTRIBUTION
# ============================================================================

depth_counter = Counter(d["dz"] for d in all_data)
print(f"\n{'='*80}")
print(f"DEPTH DISTRIBUTION ({len(depth_counter)} unique depths)")
print(f"{'='*80}\n")

depths_sorted = sorted(depth_counter.items())
old_depths = {5.0, 10.0, 20.0}
old_depth_count = sum(c for d, c in depths_sorted if d in old_depths)
old_depth_pct = 100 * old_depth_count / total

print(f"  Depth      Count      %      Status")
print(f"  " + "-" * 40)
for dz, count in depths_sorted:
    pct = 100 * count / total
    is_old = "OLD" if dz in old_depths else "NEW"
    status = "(old concentration)" if dz in old_depths else ""
    print(f"  dz={dz:5.1f}    {count:5d}  {pct:5.1f}%  {is_old} {status}")

print(f"\n  Total in {{5, 10, 20}}: {old_depth_count} ({old_depth_pct:.1f}%) - GOAL: < 50%")
print(f"  Unique depths: {len(depth_counter)} - GOAL: > 10")

# ============================================================================
# SHAPE × MATERIAL BALANCE CHECK
# ============================================================================

print(f"\n{'='*80}")
print(f"SHAPE × MATERIAL COMBINATIONS")
print(f"{'='*80}\n")

shape_mat_counter = Counter((d["shape"], d["mat_group"]) for d in all_data)
combos_present = len(shape_mat_counter)
expected_combos = 9 * 5  # 9 shapes × 5 material groups

print(f"  Combos present: {combos_present} / {expected_combos}")
print(f"  Coverage: {100*combos_present/expected_combos:.1f}%\n")

# Check if any combo is missing
all_combos = [(s, m) for s in sorted(set(d["shape"] for d in all_data))
              for m in sorted(set(d["mat_group"] for d in all_data))]
missing_combos = [c for c in all_combos if shape_mat_counter[c] == 0]

if missing_combos:
    print(f"  ⚠️  Missing {len(missing_combos)} combinations:")
    for shape, mat in missing_combos[:10]:
        print(f"      {shape} + {mat}")
else:
    print(f"  ✓ All combinations present!")

# ============================================================================
# SUMMARY & METRICS
# ============================================================================

print(f"\n{'='*80}")
print(f"BALANCE METRICS")
print(f"{'='*80}\n")

print(f"  Total files: {total}")
print(f"  Shape ratio (max/min): {shape_ratio:.2f}x (goal < 1.3x) {'✓' if shape_ratio < 1.3 else '⚠' if shape_ratio < 1.5 else '❌'}")
print(f"  Material ratio (max/min): {mat_ratio:.2f}x (goal < 1.3x) {'✓' if mat_ratio < 1.3 else '⚠' if mat_ratio < 1.5 else '❌'}")
print(f"  Old depth concentration: {old_depth_pct:.1f}% in {{5,10,20}} (goal < 50%) {'✓' if old_depth_pct < 50 else '⚠' if old_depth_pct < 60 else '❌'}")
print(f"  Unique depths: {len(depth_counter)} (goal ≥ 10) {'✓' if len(depth_counter) >= 10 else '❌'}")
print(f"  Shape×Material coverage: {combos_present}/{expected_combos} {'✓' if combos_present == expected_combos else '❌'}")

overall_status = (
    shape_ratio < 1.3 and
    mat_ratio < 1.3 and
    old_depth_pct < 50 and
    len(depth_counter) >= 10 and
    combos_present == expected_combos
)

print(f"\n  Overall: {'✅ EXCELLENT BALANCE' if overall_status else '⚠️ ACCEPTABLE' if combos_present > 0.8 * expected_combos else '❌ NEEDS WORK'}\n")

print(f"{'='*80}\n")
