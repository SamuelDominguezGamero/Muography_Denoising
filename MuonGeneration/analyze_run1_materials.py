#!/usr/bin/env python3
"""
Run1 Material Balance Analysis
Analyzes the distribution of 5 material categories across all shapes.
Detects shape-specific biases.
"""

import re
from pathlib import Path
from collections import Counter

# ===========================================================================
# PATHS
# ===========================================================================
home_path = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration")
folder_1_json = home_path / "data" / "geometric_configurations_jsons_not_letters"  # Run1

# ===========================================================================
# METADATA EXTRACTION
# ===========================================================================
def extract_shape_metadata(filename):
    """
    Extract shape, material, size, depth, wall thickness, center from run1 JSON filename.
    
    Format: shape_{shape}_{variant}_..._sx{sx}_sy{sy}_cx{cx}_cy{cy}_mat{mat}_dz{dz}[_wt{wt}].json
    Example: shape_cylinder_filled_Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_ratio1_sx10_sy10_cx0_cy0_mataluminium_dz10.json
    """
    try:
        # Extract shape (everything after "shape_" and before the variant)
        shape_match = re.search(r"shape_([a-z_]+?)_(?:filled|hollow)", filename)
        if not shape_match:
            shape_match = re.search(r"shape_([a-z_]+?)_Lpx", filename)
        shape = shape_match.group(1) if shape_match else None
        
        # Check if hollow or filled
        hollow = "_hollow_" in filename or "_wt" in filename
        
        # Extract material
        if "mataluminium" in filename:
            material = "aluminium"
        elif "matiron" in filename:
            material = "iron"
        elif "matlead" in filename:
            material = "lead"
        elif "maturanium" in filename:
            material = "uranium"
        elif "matwater" in filename:
            material = "water"
        elif "matsilicon" in filename:
            material = "silicon"
        elif "matsteel" in filename:
            material = "steel"
        else:
            material = None
        
        # Extract size X
        sx_match = re.search(r"_sx([0-9.]+)_", filename)
        sx = float(sx_match.group(1)) if sx_match else None
        
        # Extract size Y
        sy_match = re.search(r"_sy([0-9.]+)_", filename)
        sy = float(sy_match.group(1)) if sy_match else None
        
        # Extract depth Z
        dz_match = re.search(r"_dz([0-9.]+)", filename)
        dz = float(dz_match.group(1)) if dz_match else None
        
        # Extract wall thickness (for hollow shapes)
        wt_match = re.search(r"_wt([0-9.]+)", filename)
        wt = float(wt_match.group(1)) if wt_match else None
        
        # Extract center X
        cx_match = re.search(r"_cx([0-9.]+)_", filename)
        cx = float(cx_match.group(1)) if cx_match else None
        
        # Extract center Y
        cy_match = re.search(r"_cy([0-9.]+)_", filename)
        cy = float(cy_match.group(1)) if cy_match else None
        
        return shape, material, sx, sy, dz, wt, cx, cy, hollow
    except Exception as e:
        print(f"[ERROR] extracting from {filename}: {e}")
        return None, None, None, None, None, None, None, None, None

# ===========================================================================
# ANALYSIS
# ===========================================================================
def analyze_run1():
    """Analyze run1 material balance by shape."""
    
    print("\n" + "="*80)
    print("RUN1 MATERIAL BALANCE ANALYSIS")
    print("="*80 + "\n")
    
    shapes    = []
    materials = []
    shape_mat_pairs = []
    depth_zs  = []
    sizes_x   = []
    hollow_count = 0
    total = 0
    
    # Material grouping: 5 categories (20% each)
    mat_categories = {
        "iron+steel": 0,
        "silicon+aluminium": 0,
        "water": 0,
        "lead": 0,
        "uranium": 0,
    }
    
    def map_to_category(mat):
        """Map individual material to category."""
        if mat in ("iron", "steel"):
            return "iron+steel"
        if mat in ("silicon", "aluminium"):
            return "silicon+aluminium"
        if mat in ("water", "lead", "uranium"):
            return mat
        return None
    
    shape_mat_category_pairs = []

    # Load all JSON files
    for json_file in folder_1_json.glob("*.json"):
        shape, mat, sx, sy, dz, wt, cx, cy, hollow = extract_shape_metadata(json_file.name)
        if shape is None:
            continue
        total += 1
        shapes.append(shape)
        if mat:
            materials.append(mat)
            category = map_to_category(mat)
            if category:
                mat_categories[category] += 1
            shape_mat_category_pairs.append((shape, category))
        shape_mat_pairs.append((shape, mat))
        if dz is not None:
            depth_zs.append(dz)
        if sx is not None:
            sizes_x.append(sx)
        if hollow:
            hollow_count += 1

    shape_counts   = Counter(shapes)
    mat_counts     = Counter(materials)
    pair_counts    = Counter(shape_mat_pairs)
    cat_pair_counts = Counter(shape_mat_category_pairs)

    # ===== OVERVIEW =====
    print(f"  Total JSON files : {total}")
    print(f"  Unique shapes    : {len(shape_counts)}")
    print(f"  Unique materials : {len(mat_counts)}")
    print(f"  Hollow variants  : {hollow_count}  ({100*hollow_count/total:.1f}%)")
    print(f"  Filled variants  : {total-hollow_count}  ({100*(total-hollow_count)/total:.1f}%)\n")

    # ===== SHAPE DISTRIBUTION =====
    print(f"  Shape distribution:")
    for s, n in shape_counts.most_common():
        print(f"    {s:35s}: {n:5d}  ({100*n/total:5.1f}%)")

    # ===== INDIVIDUAL MATERIAL DISTRIBUTION =====
    print(f"\n  Material distribution (individual):")
    for m, n in mat_counts.most_common():
        print(f"    {m:20s}: {n:5d}  ({100*n/total:5.1f}%)")

    # ===== MATERIAL CATEGORY BALANCE (MAIN FOCUS) =====
    print(f"\n  🎯 MATERIAL CATEGORY BALANCE (5 categories, target 20% each):")
    print(f"     ════════════════════════════════════════════════════════════════════")
    target = 20.0
    for category in ["iron+steel", "silicon+aluminium", "water", "lead", "uranium"]:
        count = mat_categories[category]
        pct = 100 * count / total if total > 0 else 0
        deviation = pct - target
        status = "✓" if abs(deviation) < 2 else "⚠️"
        bar_len = int(pct / 2)  # 50% of width
        bar = "█" * bar_len
        print(f"    {status} {category:25s}: {pct:5.1f}%  {bar:25s} ({count:5d})")
        if abs(deviation) > 3:
            print(f"        ⚠️  Deviation: {deviation:+.1f}% (need adjustment)")
    
    # ===== SHAPE-SPECIFIC MATERIAL DISTRIBUTION =====
    print(f"\n  🔍 SHAPE-SPECIFIC MATERIAL DISTRIBUTION:")
    print(f"     (Should have ~20% of each material category within each shape)")
    print(f"     ════════════════════════════════════════════════════════════════════")
    
    shape_category_dist = {}
    for shape in shape_counts.keys():
        shape_category_dist[shape] = {cat: 0 for cat in mat_categories.keys()}
        for (s, cat), count in cat_pair_counts.items():
            if s == shape:
                shape_category_dist[shape][cat] = count
    
    for shape in sorted(shape_counts.keys()):
        total_for_shape = shape_counts[shape]
        print(f"\n     {shape} ({total_for_shape} files):")
        dist = shape_category_dist[shape]
        has_bias = False
        for cat in ["iron+steel", "silicon+aluminium", "water", "lead", "uranium"]:
            count = dist[cat]
            pct = 100 * count / total_for_shape if total_for_shape > 0 else 0
            ideal = 20.0
            status = "✓" if abs(pct - ideal) < 5 else "⚠️"
            if abs(pct - ideal) >= 5:
                has_bias = True
            bar_len = int(pct / 2)
            bar = "█" * bar_len
            print(f"       {status} {cat:25s}: {pct:5.1f}%  {bar:25s}", end="")
            if count > 0:
                print(f"  ({count:3d})")
            else:
                print()
        if has_bias:
            print(f"       ⚠️  This shape has material bias!")
    
    # ===== TOP SHAPE+MATERIAL PAIRS =====
    print(f"\n  Top 15 shape+material combinations:")
    for (s, m), n in pair_counts.most_common(15):
        print(f"    {s:35s} + {m:15s}: {n:5d}  ({100*n/total:5.1f}%)")

    # ===== SIZE AND DEPTH INFO =====
    if sizes_x:
        print(f"\n  Size (sx) range: min={min(sizes_x):.1f} cm, max={max(sizes_x):.1f} cm")
        size_counts = Counter(sizes_x)
        for sz, n in sorted(size_counts.items()):
            print(f"    sx={sz:6.1f} cm : {n:5d}")

    if depth_zs:
        dz_counts = Counter(depth_zs)
        print(f"\n  Depth (dz) distribution:")
        for dz, n in sorted(dz_counts.items()):
            print(f"    dz={dz:6.1f} cm : {n:5d}")

    # ===== IMBALANCE SUMMARY =====
    print(f"\n  📊 IMBALANCE CHECKS:")
    if len(mat_counts) > 1:
        max_m = max(mat_counts.values())
        min_m = min(mat_counts.values())
        ratio = max_m / min_m
        print(f"    Material imbalance: max={max_m}, min={min_m}, ratio={ratio:.1f}x")
        if ratio > 2:
            print(f"    ⚠️  HIGH material imbalance detected.")
        else:
            print(f"    ✓ Material imbalance within acceptable range.")

    if len(shape_counts) > 1:
        max_s = max(shape_counts.values())
        min_s = min(shape_counts.values())
        ratio_s = max_s / min_s
        print(f"    Shape imbalance  : max={max_s}, min={min_s}, ratio={ratio_s:.1f}x")
        if ratio_s > 2:
            print(f"    ⚠️  HIGH shape imbalance detected.")
        else:
            print(f"    ✓ Shape imbalance within acceptable range.")

    print(f"\n" + "="*80 + "\n")

if __name__ == "__main__":
    analyze_run1()
