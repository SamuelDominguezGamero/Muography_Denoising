"""
Iterative balancing script for run1 geometries.

Goal: Rebalance 9000+ JSONs by:
1. Analyzing current distribution
2. Generating new geometries for underrepresented combinations
3. Randomly deleting overrepresented combinations
4. Repeating 3 times
5. Final report

Material groups (user preference):
  - iron + steel      (balanced together)
  - aluminium + silicon (balanced together)
  - water             (separate)
  - lead              (separate)
  - uranium           (separate)
"""

import os
import sys
import json
import re
import random
from pathlib import Path
from collections import Counter
import subprocess

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
CREATE_SCRIPT = SCRIPT_DIR / "create_other_geometries_not_words.py"
JSON_FOLDER = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration/data/geometric_configurations_jsons_not_letters")
TARGET_TOTAL = 9000
ITERATIONS = 3

# Material groups
MAT_GROUPS = {
    "iron_steel": ["iron", "steel"],
    "aluminium_silicon": ["aluminium", "silicon"],
    "water": ["water"],
    "lead": ["lead"],
    "uranium": ["uranium"],
}

# All shapes
ALL_SHAPES = [
    "rectangle_filled", "rectangle_hollow",
    "cylinder_filled", "cylinder_hollow",
    "sphere_filled", "sphere_hollow",
    "triangle_slab",
    "tetrahedron_filled", "tetrahedron_hollow",
]

# ============================================================================
# PARSING & ANALYSIS
# ============================================================================

def parse_json_filename(filename):
    """Extract metadata from run1 JSON filename."""
    try:
        shape_match = re.search(r"shape_([a-z_]+)_Lpx", filename)
        mat_match = re.search(r"_mat([a-z]+)_dz", filename)
        sx_match = re.search(r"_sx(-?\d+(?:\.\d+)?)", filename)
        sy_match = re.search(r"_sy(-?\d+(?:\.\d+)?)", filename)
        dz_match = re.search(r"_dz(-?\d+(?:\.\d+)?)", filename)
        wt_match = re.search(r"_wt(\d+(?:\.\d+)?)", filename)

        shape = shape_match.group(1) if shape_match else None
        mat = mat_match.group(1) if mat_match else None
        sx = float(sx_match.group(1)) if sx_match else None
        sy = float(sy_match.group(1)) if sy_match else None
        dz = float(dz_match.group(1)) if dz_match else None
        wt = float(wt_match.group(1)) if wt_match else None
        hollow = wt is not None

        return shape, mat, sx, sy, dz, wt, hollow
    except:
        return None, None, None, None, None, None, None


def get_mat_group(mat):
    """Get the group name for a material."""
    for group, mats in MAT_GROUPS.items():
        if mat in mats:
            return group
    return None


def analyze_jsons(folder):
    """Analyze current JSON distribution."""
    all_files = list(folder.glob("*.json"))
    print(f"\n  📊 Total JSONs: {len(all_files)}")
    
    # Parse all
    parsed = []
    for f in all_files:
        shape, mat, sx, sy, dz, wt, hollow = parse_json_filename(f.name)
        if shape and mat:
            mat_group = get_mat_group(mat)
            parsed.append({
                "file": f,
                "shape": shape,
                "mat": mat,
                "mat_group": mat_group,
                "sx": sx,
                "sy": sy,
                "dz": dz,
                "hollow": hollow,
            })
    
    print(f"  ✓ Parsed: {len(parsed)}\n")
    
    # Counters
    shape_counts = Counter(p["shape"] for p in parsed)
    mat_group_counts = Counter(p["mat_group"] for p in parsed)
    hollow_counts = Counter(p["hollow"] for p in parsed)
    shape_mat_pairs = Counter((p["shape"], p["mat_group"]) for p in parsed)
    dz_counts = Counter(p["dz"] for p in parsed)
    
    print(f"  Shape distribution:")
    total = sum(shape_counts.values())
    for s, n in shape_counts.most_common():
        pct = 100*n/total
        print(f"    {s:35s}: {n:4d}  ({pct:5.1f}%)")
    
    print(f"\n  Material group distribution:")
    for g, n in mat_group_counts.most_common():
        pct = 100*n/total
        print(f"    {g:25s}: {n:4d}  ({pct:5.1f}%)")
    
    print(f"\n  Hollow vs Filled:")
    for hollow, n in sorted(hollow_counts.items()):
        pct = 100*n/total
        label = "Hollow" if hollow else "Filled"
        print(f"    {label:10s}: {n:4d}  ({pct:5.1f}%)")
    
    print(f"\n  Depth distribution:")
    for dz, n in sorted(dz_counts.items()):
        pct = 100*n/total
        print(f"    dz={dz:5.1f} cm: {n:4d}  ({pct:5.1f}%)")
    
    print(f"\n  Shape + Material Group pairs (top 20):")
    for (s, g), n in shape_mat_pairs.most_common(20):
        pct = 100*n/total
        print(f"    {s:35s} + {g:25s}: {n:4d}  ({pct:5.1f}%)")
    
    # Imbalance metrics
    if len(shape_counts) > 1:
        max_s = max(shape_counts.values())
        min_s = min(shape_counts.values())
        shape_ratio = max_s / min_s
        print(f"\n  📈 Shape imbalance ratio: {shape_ratio:.2f}x (max={max_s}, min={min_s})")
    
    if len(mat_group_counts) > 1:
        max_m = max(mat_group_counts.values())
        min_m = min(mat_group_counts.values())
        mat_ratio = max_m / min_m
        print(f"  📈 Material group imbalance ratio: {mat_ratio:.2f}x (max={max_m}, min={min_m})")
    
    return parsed, shape_counts, mat_group_counts, shape_mat_pairs, dz_counts


def generate_selective(iteration):
    """
    Generate new geometries selectively.
    Focus on: underrepresented shapes, varied depths, balanced materials.
    """
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration}: GENERATION")
    print(f"{'='*70}")
    
    # Analyze current
    parsed, shape_counts, mat_group_counts, _, _ = analyze_jsons(JSON_FOLDER)
    current_total = len(parsed)
    
    # What to generate
    print(f"\n  🎯 Strategy:")
    print(f"    - Generate new combos for underrepresented shapes")
    print(f"    - Emphasize varied depths (6, 10, 14, 18, 22, 26, 30 cm)")
    print(f"    - Balance material groups: iron_steel, aluminium_silicon < water, lead, uranium")
    print(f"    - Target: add ~300-400 files this iteration\n")
    
    # Identify underrepresented shapes
    avg_shape = current_total / len(shape_counts)
    underrep_shapes = [s for s, n in shape_counts.items() if n < avg_shape * 0.9]
    
    print(f"  Underrepresented shapes (< {avg_shape*0.9:.0f}):")
    for s in underrep_shapes:
        print(f"    - {s}: {shape_counts[s]}")
    
    # Identify underrepresented material groups
    avg_mat = current_total / len(mat_group_counts)
    underrep_mats = [m for m, n in mat_group_counts.items() if n < avg_mat * 0.9]
    
    print(f"\n  Underrepresented material groups (< {avg_mat*0.9:.0f}):")
    for m in underrep_mats:
        print(f"    - {m}: {mat_group_counts[m]}")
    
    # Generate combos: prioritize underrep shapes × underrep mats
    combos_to_gen = []
    
    # Primary: underrep shapes with varied depths
    new_depths = [6, 10, 14, 18, 22, 26, 30]  # More variety!
    
    for shape in (underrep_shapes if underrep_shapes else ALL_SHAPES[:3]):
        for mat_group in (underrep_mats if underrep_mats else list(MAT_GROUPS.keys())[:2]):
            for mat in MAT_GROUPS[mat_group]:
                for dz in new_depths[:4]:  # 4 depths per combo
                    combos_to_gen.append((shape, mat, dz))
    
    # Also generate some standard combos with new depths
    for shape in ALL_SHAPES[::2]:  # Every other shape
        for mat in ["lead", "uranium"]:  # Emphasize these
            for dz in new_depths:
                combos_to_gen.append((shape, mat, dz))
    
    # Remove duplicates already present
    existing = {(p["shape"], p["mat"], p["dz"]) for p in parsed}
    combos_to_gen = [c for c in combos_to_gen if c not in existing]
    
    print(f"\n  Planning to generate: {len(combos_to_gen)} new combos")
    
    # Generate in batches
    generated = 0
    failed = 0
    
    for i, (shape, mat, dz) in enumerate(combos_to_gen[:400], 1):  # Cap at 400 per iteration
        sx = sy = 20  # Fixed sizes for now
        cx = cy = 0
        
        name = (
            f"shape_{shape}"
            f"_Lpx128_Lpy128_Lpz128_npx128_npy128_npz128"
            f"_zTop54_zBot-54_ratio1"
            f"_sx{sx}_sy{sy}_cx{cx}_cy{cy}_mat{mat}_dz{dz}"
        )
        
        out_json = JSON_FOLDER / f"{name}.json"
        
        if out_json.exists():
            continue
        
        # Check if hollow should have wall thickness
        wall_thickness = 2 if "_hollow" in shape else None
        
        command = [
            sys.executable, str(CREATE_SCRIPT),
            "--shape", shape,
            "--size_x", str(sx),
            "--size_y", str(sy),
            "--depth_z_cm", str(dz),
            "--material", mat,
            "--dimensions", "2D",
            "--output_json", str(out_json),
        ]
        if wall_thickness:
            command += ["--wall_thickness", str(wall_thickness)]
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            generated += 1
            if i % 50 == 0:
                print(f"    Generated {i}/{len(combos_to_gen[:400])}")
        else:
            failed += 1
    
    print(f"\n  ✓ Generated: {generated} new JSONs")
    print(f"  ✗ Failed: {failed}")
    
    return generated


def delete_overrep(iteration):
    """
    Delete some JSONs from overrepresented categories.
    """
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration}: SELECTIVE DELETION")
    print(f"{'='*70}")
    
    # Analyze current
    parsed, shape_counts, mat_group_counts, shape_mat_pairs, dz_counts = analyze_jsons(JSON_FOLDER)
    current_total = len(parsed)
    
    print(f"\n  🎯 Strategy:")
    print(f"    - Delete some JSONs from overrepresented combinations")
    print(f"    - Try to maintain ~9000 total files\n")
    
    target_delete = max(0, current_total - TARGET_TOTAL)
    
    if target_delete <= 0:
        print(f"  ℹ️  Current total ({current_total}) ≤ target ({TARGET_TOTAL})")
        print(f"  No deletion needed.\n")
        return 0
    
    print(f"  Need to delete ~{target_delete} files")
    
    # Find overrepresented combinations
    avg_pair = current_total / len(shape_mat_pairs)
    overrep_pairs = [
        (pair, count)
        for pair, count in shape_mat_pairs.most_common()
        if count > avg_pair * 1.3
    ]
    
    print(f"\n  Overrepresented pairs (> {avg_pair*1.3:.0f}):")
    for (s, g), n in overrep_pairs[:10]:
        print(f"    {s:35s} + {g:25s}: {n}")
    
    # Delete from overrep
    deleted = 0
    files_to_consider = [p for p in parsed if (p["shape"], p["mat_group"]) in {p[0] for p in overrep_pairs}]
    
    # Randomly delete
    random.shuffle(files_to_consider)
    for p in files_to_consider:
        if deleted >= target_delete:
            break
        
        try:
            p["file"].unlink()
            deleted += 1
        except:
            pass
    
    print(f"\n  🗑️  Deleted: {deleted} files")
    
    return deleted


def report_final():
    """Generate final report."""
    print(f"\n{'='*70}")
    print(f"FINAL REPORT")
    print(f"{'='*70}")
    
    parsed, shape_counts, mat_group_counts, shape_mat_pairs, dz_counts = analyze_jsons(JSON_FOLDER)
    
    print(f"\n  ✅ FINAL STATE:")
    print(f"    Total files: {len(parsed)}")
    print(f"    Target was: {TARGET_TOTAL}")
    print(f"    Difference: {len(parsed) - TARGET_TOTAL:+d}")
    
    # Check balances
    total = len(parsed)
    
    print(f"\n  📊 Shape balance (target: {100/len(shape_counts):.1f}% each):")
    for s in sorted(shape_counts.keys()):
        n = shape_counts[s]
        pct = 100*n/total
        expected = 100/len(shape_counts)
        delta = pct - expected
        status = "✓" if abs(delta) < 2 else "⚠" if abs(delta) < 5 else "❌"
        print(f"    {status} {s:35s}: {n:4d}  ({pct:5.1f}%)  [Δ {delta:+5.1f}%]")
    
    print(f"\n  📊 Material group balance (target: 20% iron_steel, 20% aluminium_silicon, 20% water, 20% lead, 20% uranium):")
    targets = {"iron_steel": 20, "aluminium_silicon": 20, "water": 20, "lead": 20, "uranium": 20}
    for g in sorted(mat_group_counts.keys()):
        n = mat_group_counts[g]
        pct = 100*n/total
        expected = targets.get(g, 20)
        delta = pct - expected
        status = "✓" if abs(delta) < 3 else "⚠" if abs(delta) < 8 else "❌"
        print(f"    {status} {g:25s}: {n:4d}  ({pct:5.1f}%)  [Δ {delta:+5.1f}%]")
    
    # Imbalance ratios
    if len(shape_counts) > 1:
        max_s = max(shape_counts.values())
        min_s = min(shape_counts.values())
        shape_ratio = max_s / min_s
        print(f"\n  📈 Shape imbalance ratio: {shape_ratio:.2f}x")
        print(f"     Goal: < 1.5x | Status: {'✓ OK' if shape_ratio < 1.5 else '⚠ NEEDS WORK'}")
    
    if len(mat_group_counts) > 1:
        max_m = max(mat_group_counts.values())
        min_m = min(mat_group_counts.values())
        mat_ratio = max_m / min_m
        print(f"\n  📈 Material group imbalance ratio: {mat_ratio:.2f}x")
        print(f"     Goal: < 1.3x | Status: {'✓ OK' if mat_ratio < 1.3 else '⚠ NEEDS WORK'}")
    
    print(f"\n  📈 Depth variety:")
    print(f"     Unique depths: {len(dz_counts)}")
    print(f"     Range: {min(dz_counts.keys()):.1f} - {max(dz_counts.keys()):.1f} cm")
    for dz in sorted(dz_counts.keys()):
        n = dz_counts[dz]
        pct = 100*n/total
        print(f"       dz={dz:5.1f} cm: {n:4d}  ({pct:5.1f}%)")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"ITERATIVE REBALANCING: run1 geometries")
    print(f"{'='*70}")
    print(f"Target: {TARGET_TOTAL} files")
    print(f"Iterations: {ITERATIONS}")
    print(f"Material groups: iron_steel, aluminium_silicon, water, lead, uranium")
    
    for it in range(1, ITERATIONS + 1):
        print(f"\n\n{'#'*70}")
        print(f"# ITERATION {it}/{ITERATIONS}")
        print(f"{'#'*70}")
        
        # Generate
        gen = generate_selective(it)
        
        # Analyze & delete overrep
        del_count = delete_overrep(it)
        
        # Brief check
        parsed, _, _, _, _ = analyze_jsons(JSON_FOLDER)
        print(f"\n  Current total: {len(parsed)} files")
    
    # Final report
    report_final()
    
    print(f"\n{'='*70}")
    print(f"✅ PROCESS COMPLETE")
    print(f"{'='*70}\n")
