#!/usr/bin/env python3
"""
Analyze balance metrics for run2 balanced dataset (rectangular blocks).

Metrics:
- Material distribution (target: 1053 files per material, 14.3%)
- Depth distribution (target: uniform across 13 values)
- Size distribution (sx, sy)
- Position distribution
- Concentration in old-depth values {5, 10, 20}
"""

from pathlib import Path
from collections import Counter
import re


OUTPUT_DIR = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration/data/geometric_configurations_blocks")


def extract_metadata(filename):
    """
    Extract metadata from run2 JSON filename.
    
    Format: shape_rectangle_filled_Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_ratio1_sx{sx}_sy{sy}_cx{cx}_cy{cy}_mat{mat}_dz{dz}.json
    """
    try:
        sx_match = re.search(r"_sx(-?\d+(?:\.\d+)?)", filename)
        sy_match = re.search(r"_sy(-?\d+(?:\.\d+)?)", filename)
        mat_match = re.search(r"_mat([a-z]+)_dz", filename)
        dz_match = re.search(r"_dz(-?\d+(?:\.\d+)?)", filename)
        cx_match = re.search(r"_cx(-?\d+(?:\.\d+)?)", filename)
        cy_match = re.search(r"_cy(-?\d+(?:\.\d+)?)", filename)
        
        sx = float(sx_match.group(1)) if sx_match else None
        sy = float(sy_match.group(1)) if sy_match else None
        mat = mat_match.group(1) if mat_match else None
        dz = float(dz_match.group(1)) if dz_match else None
        cx = float(cx_match.group(1)) if cx_match else None
        cy = float(cy_match.group(1)) if cy_match else None
        
        return sx, sy, mat, dz, cx, cy
    except Exception:
        return None, None, None, None, None, None


def main():
    """Main analysis."""
    
    # Get all JSON files
    json_files = list(OUTPUT_DIR.glob("shape_*.json"))
    total = len(json_files)
    
    print("\n" + "="*80)
    print("ANALYZE BALANCED RUN2 DATASET")
    print("="*80 + "\n")
    
    print(f"Found {total} JSON files in {OUTPUT_DIR}\n")
    
    # Extract metadata from all files
    materials = []
    depths = []
    sizes_x = []
    sizes_y = []
    positions_x = []
    positions_y = []
    parsed_count = 0
    failed_count = 0
    
    for json_file in json_files:
        sx, sy, mat, dz, cx, cy = extract_metadata(json_file.name)
        if mat is None:
            failed_count += 1
            continue
        parsed_count += 1
        materials.append(mat)
        depths.append(dz)
        sizes_x.append(sx)
        sizes_y.append(sy)
        positions_x.append(cx)
        positions_y.append(cy)
    
    print(f"Successfully parsed: {parsed_count} / {total}")
    if failed_count > 0:
        print(f"Failed to parse: {failed_count}\n")
    
    # =========================================================================
    # MATERIAL DISTRIBUTION
    # =========================================================================
    print("\n" + "="*80)
    print("MATERIAL DISTRIBUTION (target: ~1053 per material, 14.3%)")
    print("="*80 + "\n")
    
    mat_counts = Counter(materials)
    target_per_mat = parsed_count / len(mat_counts) if mat_counts else 0
    
    for mat in sorted(mat_counts.keys()):
        count = mat_counts[mat]
        pct = 100 * count / parsed_count if parsed_count > 0 else 0
        status = "✓" if abs(count - target_per_mat) < 50 else "⚠️"
        print(f"  {mat:15s}  {count:5d}  ({pct:5.1f}%)  {status}")
    
    if len(mat_counts) > 1:
        max_mat = max(mat_counts.values())
        min_mat = min(mat_counts.values())
        ratio = max_mat / min_mat
        print(f"\n  Ratio (max/min): {ratio:.2f}x (goal < 1.3x)")
        if ratio < 1.3:
            print(f"  ✓ Materials are well-balanced")
        else:
            print(f"  ❌ Materials are imbalanced")
    
    # =========================================================================
    # DEPTH DISTRIBUTION
    # =========================================================================
    print("\n" + "="*80)
    print("DEPTH DISTRIBUTION (13 unique depths)")
    print("="*80 + "\n")
    
    dz_counts = Counter(depths)
    target_depth = parsed_count / len(dz_counts) if dz_counts else 0
    
    print("  Depth      Count      %      Status")
    print("  " + "-"*40)
    
    old_depth_count = 0
    for dz in sorted(dz_counts.keys()):
        count = dz_counts[dz]
        pct = 100 * count / parsed_count if parsed_count > 0 else 0
        
        # Mark old concentrations
        if dz in [5, 10, 20]:
            status = "OLD (old concentration)"
            old_depth_count += count
        else:
            status = "NEW"
        
        print(f"  dz={dz:5.1f}      {count:5d}  {pct:5.1f}%  {status}")
    
    old_depth_pct = 100 * old_depth_count / parsed_count if parsed_count > 0 else 0
    print(f"\n  Total in {{5, 10, 20}}: {old_depth_count} ({old_depth_pct:.1f}%) - GOAL: < 50%")
    print(f"  Unique depths: {len(dz_counts)} - GOAL: > 10")
    
    if old_depth_pct < 50:
        print(f"  ✓ Depth distribution is excellent (low old-depth concentration)")
    
    # =========================================================================
    # SIZE DISTRIBUTION
    # =========================================================================
    print("\n" + "="*80)
    print("SIZE DISTRIBUTION")
    print("="*80 + "\n")
    
    sx_counts = Counter(sizes_x)
    sy_counts = Counter(sizes_y)
    
    print("  Size X (sx):")
    for size in sorted(sx_counts.keys()):
        count = sx_counts[size]
        pct = 100 * count / parsed_count if parsed_count > 0 else 0
        print(f"    sx={size:6.1f} cm : {count:5d}  ({pct:5.1f}%)")
    
    print(f"\n  Size Y (sy):")
    for size in sorted(sy_counts.keys()):
        count = sy_counts[size]
        pct = 100 * count / parsed_count if parsed_count > 0 else 0
        print(f"    sy={size:6.1f} cm : {count:5d}  ({pct:5.1f}%)")
    
    # =========================================================================
    # POSITION DISTRIBUTION
    # =========================================================================
    print("\n" + "="*80)
    print("POSITION DISTRIBUTION")
    print("="*80 + "\n")
    
    cx_counts = Counter(positions_x)
    cy_counts = Counter(positions_y)
    
    print("  Position X (cx):")
    for pos in sorted(cx_counts.keys()):
        count = cx_counts[pos]
        pct = 100 * count / parsed_count if parsed_count > 0 else 0
        print(f"    cx={pos:6.1f} : {count:5d}  ({pct:5.1f}%)")
    
    print(f"\n  Position Y (cy):")
    for pos in sorted(cy_counts.keys()):
        count = cy_counts[pos]
        pct = 100 * count / parsed_count if parsed_count > 0 else 0
        print(f"    cy={pos:6.1f} : {count:5d}  ({pct:5.1f}%)")
    
    # =========================================================================
    # FINAL METRICS
    # =========================================================================
    print("\n" + "="*80)
    print("BALANCE METRICS")
    print("="*80 + "\n")
    
    print(f"  Total files: {parsed_count}")
    
    if len(mat_counts) > 1:
        max_mat = max(mat_counts.values())
        min_mat = min(mat_counts.values())
        ratio = max_mat / min_mat
        print(f"  Material ratio (max/min): {ratio:.2f}x (goal < 1.3x)", end="")
        print(" ✓" if ratio < 1.3 else " ❌")
    
    print(f"  Old depth concentration: {old_depth_pct:.1f}% in {{5,10,20}} (goal < 50%)", end="")
    print(" ✓" if old_depth_pct < 50 else " ❌")
    
    print(f"  Unique depths: {len(dz_counts)} (goal ≥ 10)", end="")
    print(" ✓" if len(dz_counts) >= 10 else " ❌")
    
    print(f"  Unique sizes X: {len(sx_counts)}, Unique sizes Y: {len(sy_counts)}")
    print(f"  Unique positions X: {len(cx_counts)}, Unique positions Y: {len(cy_counts)}")
    
    print(f"\n  Overall: {'✅ EXCELLENT' if ratio < 1.3 and old_depth_pct < 50 else '⚠️ ACCEPTABLE'}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
