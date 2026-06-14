"""
Dataset creation and analysis pipeline.

Steps:
1. Count available .npy files per run (POCA projections)
2. Count available .npy files per run (ground truth)
3. Count available .json files per run (geometry configs)
4. Test metadata extraction from filenames
5. Analyze material balance across runs
6. Analyze word diversity and overfitting risk in run0
"""

from pathlib import Path
from collections import Counter
import sys
import re
import numpy as np


# ===========================================================================
# PATHS & CONFIGURATION
# ===========================================================================
home_path = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration")

# POCA projections (simulation data)
folder_0_G4 = home_path / "data" / "simulation_data" / "run0_definitive_letters"
folder_1_G4 = home_path / "data" / "simulation_data" / "run1_definitive_forms"
folder_2_G4 = home_path / "data" / "simulation_data" / "run2_definitive_blocks"

# Ground truth maps
folder_0_gt = home_path / "data" / "ground_truth_data" / "run0_letters"
folder_1_gt = home_path / "data" / "ground_truth_data" / "run1_geometry_variety"
folder_2_gt = home_path / "data" / "ground_truth_data" / "run2_blocks"

# JSON geometry configs
folder_0_json = home_path / "data" / "geometric_configurations_json"
folder_1_json = home_path / "data" / "geometric_configurations_jsons_not_letters"
folder_2_json = home_path / "data" / "geometric_configurations_blocks"



# ===========================================================================
# MAIN PIPELINE
# ===========================================================================
def main():
    """Main analysis pipeline."""

    print("\n" + "="*70)
    print("DATASET ANALYSIS PIPELINE")
    print("="*70)

    # Step 1: Count available files
    print("\n" + 60 * "-")
    count_npy_files()
    count_ground_truth_files()
    count_json_files()
    print(60 * "-")

    # Step 2: Test metadata extraction
    test_metadata_extraction()

    # Step 3: Analyze material balance
    print(60 * "-"); print("ANALYSIS OF .npy files")
    analyze_material_balance()
    print(60 * "-")


    # Step 4: Analyze diversity per dataset
    print(60 * "-"); print("ANALYSIS OF .json files")
    # analyze_run0_jsons(folder_0_json); print("-"*70, flush=True)
    analyze_run1_jsons(folder_1_json); print("-"*70, flush=True)
    # analyze_run2_jsons(folder_2_json); print("-"*70, flush=True)


    # Para cuando creemos el dataset: una de las palabras que tiene que aparecer en el split de test es la palabra MUON. La palabra MUON no puede aparecer en ninguno de los otros dos splits. Es una forma visual de hacer tests, con una palabra reconocida para el lector

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70 + "\n")















# ===========================================================================
# STEP FUNCTIONS
# ===========================================================================

def count_npy_files():
    """Count POCA projection files per run."""
    print("\n[STEP 1] Counting POCA projection files...\n")
    counts = [
        ("run0", folder_0_G4),
        ("run1", folder_1_G4),
        ("run2", folder_2_G4),
    ]
    total = 0
    for name, folder in counts:
        count = len(list(folder.glob("*.npy")))
        total += count
        print(f"  {name:30s}: {count:5d}")
    print(f"  {'TOTAL':30s}: {total:5d}")


def count_ground_truth_files():
    """Count ground truth files per run."""
    print("\n[STEP 2] Counting ground truth files...\n")
    counts = [
        ("run0", folder_0_gt),
        ("run1", folder_1_gt),
        ("run2", folder_2_gt),
    ]
    for name, folder in counts:
        count = len(list(folder.glob("*.npy")))
        print(f"  {name:30s}: {count:5d}")


def count_json_files():
    """Count JSON geometry config files per run."""
    print("\n[STEP 3] Counting JSON geometry configs...\n")
    counts = [
        ("run0 (letters)", folder_0_json),
        ("run1 (other geometries)", folder_1_json),
        ("run2 (blocks)", folder_2_json),
    ]
    for name, folder in counts:
        count = len(list(folder.glob("*.json")))
        print(f"  {name:30s}: {count:5d}")


def test_metadata_extraction():
    """Test metadata extraction from filenames."""
    print("\n[STEP 4] Testing metadata extraction (first 2 files per run)...\n")
    
    i = 0
    errors = 0
    for folder in [folder_0_G4, folder_1_G4, folder_2_G4]:
        for npy_file in folder.glob("*.npy"):
            i += 1
            try:
                material, depthZ = extract_metadata_from_filename(npy_file.name)
            except Exception as e:
                errors += 1
                print(f"  [ERROR] {npy_file.name}: {e}")
                continue
            if i <= 2:
                print(f"  {npy_file.name}")
                print(f"    → Material: {material}, Depth Z: {depthZ}\n")
    
    print(f"Processed {i} files with {errors} errors.")
    if errors == 0:
        print("[✓] All files processed successfully.\n")


def analyze_material_balance():
    """Analyze material distribution across runs."""
    print("\n[STEP 5] Material balance per run...\n")
    
    groups = ["iron+steel", "silicon+aluminium", "water", "lead", "uranium"]
    
    def group_material(mat):
        if mat in ("iron", "steel"):
            return "iron+steel"
        if mat in ("silicon", "aluminium"):
            return "silicon+aluminium"
        return mat
    
    for run_name, folder in [("run0", folder_0_G4), ("run1", folder_1_G4), ("run2", folder_2_G4)]:
        counts = {g: 0 for g in groups}
        total = 0
        for npy_file in folder.glob("*.npy"):
            try:
                material, _ = extract_metadata_from_filename(npy_file.name)
                counts[group_material(material)] += 1
                total += 1
            except Exception:
                pass
        
        print(f"  {run_name} ({total} files):")
        for g, n in counts.items():
            pct = 100 * n / total if total > 0 else 0
            print(f"    {g:25s}: {n:5d}  ({pct:5.1f}%)")
        print()



























# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def extract_shape_metadata_run1(filename):
    """
    Extract shape, material, size_x, size_y, depth_z, wall_thickness from run1 JSON filename.
    
    Format: shape_{shape}_Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_ratio1_sx{sx}_sy{sy}_cx{cx}_cy{cy}_mat{mat}_dz{dz}[_wt{t}].json
    
    Example: shape_cylinder_filled_Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_ratio1_sx10_sy10_cx0_cy0_mataluminium_dz10.json
    """
    try:
        # shape: everything between "shape_" and the first "_Lpx"
        shape_match = re.search(r"shape_([a-z_]+)_Lpx", filename)
        
        # material: everything between "_mat" and "_dz" (e.g., mataluminium, matwater)
        mat_match   = re.search(r"_mat([a-z]+)_dz",   filename)
        
        # sizes and positions: numeric values (can be negative)
        sx_match    = re.search(r"_sx(-?\d+(?:\.\d+)?)", filename)
        sy_match    = re.search(r"_sy(-?\d+(?:\.\d+)?)", filename)
        cx_match    = re.search(r"_cx(-?\d+(?:\.\d+)?)", filename)
        cy_match    = re.search(r"_cy(-?\d+(?:\.\d+)?)", filename)
        dz_match    = re.search(r"_dz(-?\d+(?:\.\d+)?)", filename)
        
        # wall thickness (only for hollow shapes)
        wt_match    = re.search(r"_wt(\d+(?:\.\d+)?)", filename)

        shape   = shape_match.group(1) if shape_match else None
        mat     = mat_match.group(1)   if mat_match   else None
        sx      = float(sx_match.group(1)) if sx_match else None
        sy      = float(sy_match.group(1)) if sy_match else None
        cx      = float(cx_match.group(1)) if cx_match else None
        cy      = float(cy_match.group(1)) if cy_match else None
        dz      = float(dz_match.group(1)) if dz_match else None
        wt      = float(wt_match.group(1)) if wt_match else None
        hollow  = (wt is not None)

        return shape, mat, sx, sy, cx, cy, dz, wt, hollow
    except Exception as e:
        return None, None, None, None, None, None, None, None, None


def _analyze_shape_jsons(folder, run_label):
    """
    Shared analysis logic for run1 and run2 JSON folders.
    Reports shape and material balance, size distribution, hollow/filled ratio.
    """
    shapes    = []
    materials = []
    shape_mat_pairs = []
    depth_zs  = []
    sizes_x   = []
    hollow_count = 0
    total = 0

    for json_file in folder.glob("*.json"):
        shape, mat, sx, sy, dz, wt, cx, cy, hollow = extract_shape_metadata(json_file.name)
        if shape is None:
            continue
        total += 1
        shapes.append(shape)
        if mat:
            materials.append(mat)
        shape_mat_pairs.append((shape, mat))
        if dz is not None:
            depth_zs.append(dz)
        if sx is not None:
            sizes_x.append(sx)
        if hollow:
            hollow_count += 1

    if total == 0:
        print(f"  [WARNING] No JSON files found in {run_label}\n")
        return

    shape_counts   = Counter(shapes)
    mat_counts     = Counter(materials)
    pair_counts    = Counter(shape_mat_pairs)

    print(f"  Total JSON files : {total}")
    print(f"  Unique shapes    : {len(shape_counts)}")
    print(f"  Unique materials : {len(mat_counts)}")
    print(f"  Hollow variants  : {hollow_count}  ({100*hollow_count/total:.1f}%)")
    print(f"  Filled variants  : {total-hollow_count}  ({100*(total-hollow_count)/total:.1f}%)\n")

    print(f"  Shape distribution:")
    for s, n in shape_counts.most_common():
        print(f"    {s:35s}: {n:5d}  ({100*n/total:5.1f}%)")

    print(f"\n  Material distribution:")
    for m, n in mat_counts.most_common():
        print(f"    {m:20s}: {n:5d}  ({100*n/total:5.1f}%)")

    print(f"\n  Top 15 shape+material combinations (overfitting risk):")
    for (s, m), n in pair_counts.most_common(15):
        print(f"    {s:35s} + {m:15s}: {n:5d}  ({100*n/total:5.1f}%)")

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

    # Imbalance check on materials
    if len(mat_counts) > 1:
        max_m = max(mat_counts.values())
        min_m = min(mat_counts.values())
        ratio = max_m / min_m
        print(f"\n  Material imbalance: max={max_m}, min={min_m}, ratio={ratio:.1f}x")
        if ratio > 2:
            print(f"  ⚠️  MATERIAL IMBALANCE detected.")

    # Imbalance check on shapes
    if len(shape_counts) > 1:
        max_s = max(shape_counts.values())
        min_s = min(shape_counts.values())
        ratio_s = max_s / min_s
        print(f"  Shape imbalance  : max={max_s}, min={min_s}, ratio={ratio_s:.1f}x")
        if ratio_s > 2:
            print(f"  ⚠️  SHAPE IMBALANCE detected.")
    print()


def analyze_run1_jsons(folder):
    """Analyze shape/material balance and overfitting risk in run1 (geometric shapes with multiple materials)."""
    print("\n[STEP 5a] Run1 JSON analysis (geometric shapes)...\n")
    
    shapes    = []
    materials = []
    shape_mat_pairs = []
    depth_zs  = []
    sizes_x   = []
    sizes_y   = []
    hollow_count = 0
    total = 0
    failed = 0

    for json_file in folder.glob("*.json"):
        shape, mat, sx, sy, cx, cy, dz, wt, hollow = extract_shape_metadata_run1(json_file.name)
        
        if shape is None or mat is None:
            failed += 1
            continue
            
        total += 1
        shapes.append(shape)
        materials.append(mat)
        shape_mat_pairs.append((shape, mat))
        if dz is not None:
            depth_zs.append(dz)
        if sx is not None:
            sizes_x.append(sx)
        if sy is not None:
            sizes_y.append(sy)
        if hollow:
            hollow_count += 1

    if total == 0:
        print(f"  ⚠️  WARNING: No valid JSON files found in {folder}")
        print(f"     Failed to parse: {failed} files\n")
        return

    shape_counts   = Counter(shapes)
    mat_counts     = Counter(materials)
    pair_counts    = Counter(shape_mat_pairs)

    print(f"  Total JSON files parsed : {total}")
    if failed > 0:
        print(f"  Files failed to parse   : {failed}")
    print(f"  Unique shapes           : {len(shape_counts)}")
    print(f"  Unique materials        : {len(mat_counts)}")
    print(f"  Hollow variants         : {hollow_count}  ({100*hollow_count/total:.1f}%)")
    print(f"  Filled variants         : {total-hollow_count}  ({100*(total-hollow_count)/total:.1f}%)\n")

    # Shape distribution
    print(f"  Shape distribution (should be ~balanced):")
    for s, n in shape_counts.most_common():
        pct = 100*n/total
        print(f"    {s:35s}: {n:5d}  ({pct:5.1f}%)")

    # Material distribution
    print(f"\n  Material distribution (should be ~balanced):")
    for m, n in mat_counts.most_common():
        pct = 100*n/total
        print(f"    {m:20s}: {n:5d}  ({pct:5.1f}%)")

    # Shape+material pairs (check for overfitting risk / bias)
    print(f"\n  Top 20 shape+material combinations (check for bias):")
    for (s, m), n in pair_counts.most_common(20):
        pct = 100*n/total
        status = "🔴" if pct > 1.5 else "🟡" if pct > 1.0 else "🟢"
        print(f"    {status} {s:35s} + {m:15s}: {n:5d}  ({pct:5.1f}%)")

    # Size distribution
    if sizes_x:
        sx_counts = Counter(sizes_x)
        print(f"\n  Size (sx) distribution (min={min(sizes_x):.1f}, max={max(sizes_x):.1f}):")
        for sz, n in sorted(sx_counts.items()):
            pct = 100*n/total
            print(f"    sx={sz:6.1f} cm : {n:5d}  ({pct:5.1f}%)")

    if sizes_y:
        sy_counts = Counter(sizes_y)
        print(f"\n  Size (sy) distribution (min={min(sizes_y):.1f}, max={max(sizes_y):.1f}):")
        for sz, n in sorted(sy_counts.items()):
            pct = 100*n/total
            print(f"    sy={sz:6.1f} cm : {n:5d}  ({pct:5.1f}%)")

    # Depth distribution
    if depth_zs:
        dz_counts = Counter(depth_zs)
        print(f"\n  Depth (dz) distribution (min={min(depth_zs):.1f}, max={max(depth_zs):.1f}):")
        for dz, n in sorted(dz_counts.items()):
            pct = 100*n/total
            print(f"    dz={dz:6.1f} cm : {n:5d}  ({pct:5.1f}%)")

    # Imbalance metrics
    if len(mat_counts) > 1:
        max_m = max(mat_counts.values())
        min_m = min(mat_counts.values())
        ratio_m = max_m / min_m
        print(f"\n  Material imbalance: max={max_m}, min={min_m}, ratio={ratio_m:.2f}x")
        if ratio_m > 1.5:
            print(f"  ⚠️  MATERIAL IMBALANCE detected (ratio > 1.5x)")

    if len(shape_counts) > 1:
        max_s = max(shape_counts.values())
        min_s = min(shape_counts.values())
        ratio_s = max_s / min_s
        print(f"  Shape imbalance  : max={max_s}, min={min_s}, ratio={ratio_s:.2f}x")
        if ratio_s > 1.5:
            print(f"  ⚠️  SHAPE IMBALANCE detected (ratio > 1.5x)")

    print()


def analyze_run2_jsons(folder):
    """Analyze shape/material balance and overfitting risk in run2 (solid blocks)."""
    print("\n[STEP 5b] Run2 JSON analysis (solid blocks)...\n")
    _analyze_shape_jsons(folder, "run2")


def extract_metadata_from_filename(filename):
    """
    Extract material and depth from POCA projection filename.
    
    Supports two formats:
    run0: poca_abs_..._mat{mat}_word{word}_stroke{s}_depthZ{d}.npy
    run1/2: tensor_2D_POCA_..._mat{mat}_dz{d}.npy (or _dz{d}_wt{t}.npy for hollow)
    """
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
        raise ValueError(f"Unknown material in filename: {filename}")
    
    # Extract depth Z
    if "depthZ" in filename:
        # run0: depthZ20
        depth_str = filename.split("depthZ")[-1].split(".npy")[0]
        depthZ = float(depth_str)
    elif "dz" in filename:
        # run1/2: dz5 or dz10_wt4 (hollow shapes)
        depth_str = filename.split("dz")[-1].split(".npy")[0].split("_wt")[0]
        depthZ = float(depth_str)
    else:
        raise ValueError(f"Unknown depth Z in filename: {filename}")
    
    return material, depthZ


def extract_run0_metadata(filename):
    """
    Extract word, material, font size, and depth from run0 JSON filename.
    
    Format: ..._FontX{size}_FontY{size}_mat{material}_word{word}_stroke{s}_depthZ{d}.json
    """
    try:
        mat_match = re.search(r"_mat([a-z]+)_word", filename)
        word_match = re.search(r"_word([A-Z0-9]+)_stroke", filename)
        font_match = re.search(r"_FontX(\d+)_", filename)
        depth_match = re.search(r"_depthZ(\d+)", filename)
        
        material = mat_match.group(1) if mat_match else None
        word = word_match.group(1) if word_match else None
        fontsize = int(font_match.group(1)) if font_match else None
        depth = int(depth_match.group(1)) if depth_match else None
        
        return material, word, fontsize, depth
    except Exception:
        return None, None, None, None


def analyze_run0_jsons(folder):
    """Analyze word distribution and overfitting risk in run0."""
    print("\n[STEP 6] Run0 JSON analysis (geometry diversity & overfitting risk)...\n")
    
    words = []
    materials = []
    word_mat_pairs = []
    
    for json_file in folder.glob("*.json"):
        material, word, fontsize, depth = extract_run0_metadata(json_file.name)
        if word:
            words.append(word)
            if material:
                materials.append(material)
            word_mat_pairs.append((word, material))
    
    if not words:
        print("  [WARNING] No JSON files found in run0\n")
        return
    
    word_counts = Counter(words)
    mat_counts = Counter(materials)
    pair_counts = Counter(word_mat_pairs)
    
    # Summary
    print(f"  Total JSON files: {len(words)}")
    print(f"  Unique words: {len(word_counts)}")
    print(f"  Unique materials: {len(mat_counts)}\n")
    
    # Top words
    print(f"  Top 15 most frequent words:")
    for word, count in word_counts.most_common(15):
        pct = 100 * count / len(words)
        print(f"    {word:15s}: {count:4d}  ({pct:5.1f}%)")
    
    # Top word+material pairs (OVERFITTING RISK)
    print(f"\n  Top 20 word+material combinations (⚠️  OVERFITTING RISK):")
    for (word, mat), count in pair_counts.most_common(20):
        pct = 100 * count / len(words)
        print(f"    {word:15s} + {mat:15s}: {count:4d}  ({pct:5.1f}%)")
    
    # Imbalance metric
    max_word = max(word_counts.values())
    min_word = min(word_counts.values())
    ratio = max_word / min_word if min_word > 0 else float('inf')
    print(f"\n  Word frequency imbalance: max={max_word}, min={min_word}, ratio={ratio:.1f}x")
    if ratio > 2:
        print(f"  ⚠️  HIGH IMBALANCE! Recommend resampling words for balanced dataset.")
    
    # DETAILED: Show ALL words with statistics
    print(f"\n  📊 ALL {len(word_counts)} WORDS (sorted by frequency):")
    total_words = len(words)
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Group by frequency category
    high_freq = [w for w, c in sorted_words if c > 100]
    med_freq = [w for w, c in sorted_words if 50 <= c <= 100]
    low_freq = [w for w, c in sorted_words if c < 50]
    
    print(f"\n    🔴 HIGH FREQUENCY (>100): {len(high_freq)} words")
    for word, count in [(w, word_counts[w]) for w in high_freq]:
        pct = 100 * count / total_words
        print(f"       {word:15s} : {count:4d} ({pct:5.2f}%)")
    
    print(f"\n    🟡 MEDIUM FREQUENCY (50-100): {len(med_freq)} words")
    for word, count in [(w, word_counts[w]) for w in med_freq]:
        pct = 100 * count / total_words
        print(f"       {word:15s} : {count:4d} ({pct:5.2f}%)")
    
    print(f"\n    🟢 LOW FREQUENCY (<50): {len(low_freq)} words")
    for word, count in [(w, word_counts[w]) for w in low_freq]:
        pct = 100 * count / total_words
        print(f"       {word:15s} : {count:4d} ({pct:5.2f}%)")
    
    # DETAILED: Show top 40 word+material pairs
    print(f"\n  📊 TOP 40 WORD+MATERIAL PAIRS (balance analysis):")
    for idx, ((word, mat), count) in enumerate(pair_counts.most_common(40), 1):
        pct = 100 * count / total_words
        status = "🔴" if pct > 1.5 else "🟡" if pct > 0.8 else "🟢"
        print(f"    {status} {idx:2d}. {word:10s} + {mat:12s} : {count:4d} ({pct:5.2f}%)")


# ===========================================================================
# EXECUTION
# ===========================================================================
if __name__ == "__main__":
    main()

