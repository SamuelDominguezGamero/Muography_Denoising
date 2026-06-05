"""
Creates all JSON geometry files for letter-based training data (run0).

USAGE: Edit the GEOMETRY VARIATIONS section below, then run:
  python create_letters_geometries.py

Customize:
  - fontsizes      : 8, 10, 12, 14, 16 (warnings shown if combo doesn't fit)
  - strokes        : 1, 2, 3
  - depth_z_cm_list: word thickness in cm (single-slab method)
  - materials      : lead, iron, uranium, water, etc.
  - words_geometry : any combination of M, U, O, N
  - max_geometries : limit for quick tests (np.inf = all)
"""

import subprocess
import os
import sys
import time
import json
import random
import numpy as np
from itertools import product as iproduct

# ===========================================================================
# CONTROL FLAGS
# ===========================================================================
create_geometries = True
max_geometries    = np.inf   # set to a small number for quick tests
SKIP_JSON_CREATION = True   # Randomly skip ~70% of combos when True
SKIP_THRESHOLD = 0.3        # Keep ~30% of combos when SKIP_JSON_CREATION=True


# ===========================================================================
# ERROR TRACKING
# ===========================================================================
ERROR_LOG_FILE = os.path.join(os.path.dirname(__file__), ".geometry_errors.json")

def load_error_log():
    if os.path.exists(ERROR_LOG_FILE):
        try:
            with open(ERROR_LOG_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def record_geometry_error(namefile, error_message):
    error_log = load_error_log()
    error_log[namefile] = {
        "error": error_message.strip(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(ERROR_LOG_FILE, "w") as f:
        json.dump(error_log, f, indent=2)
    print(f"[ERROR RECORDED] {namefile} → {ERROR_LOG_FILE}")


# ===========================================================================
# PATHS
# ===========================================================================
SCRIPT_DIR             = os.path.dirname(os.path.abspath(__file__))
CREATE_GEOMETRY_SCRIPT = os.path.join(SCRIPT_DIR, "create_geometry.py")
PATH_geometry_files    = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/geometric_configurations_json"

os.makedirs(PATH_geometry_files, exist_ok=True)
print(f"[INFO] Output JSON folder: {PATH_geometry_files}")


# ===========================================================================
# GEOMETRY PARAMETERS (fixed)
# ===========================================================================
Lpx = 128
Lpy = 128
Lpz = 128
npx = 128
npy = 128
npz = 128
zPosDetector_top =  54
zPosDetector_bot = -54


# ===========================================================================
# GEOMETRY VARIATIONS
# ===========================================================================
# IMPORTANT: All combinations are now valid. Check bitmaps_letters.py BITMAP_DATA:
#   - Sizes 8, 10, 12, 14, 16: Stroke 1, 2, 3 all available
#   - Only letters available: M, U, O, N

spacings       = [1] 
ratios         = [1]
fontsizes      = [10, 12, 14]
strokes        = [3]
depth_z_cm_list = [9.0, 20.0]

# Single position (fast)
x_offsets_cm   = [0.0]
y_offsets_cm   = [0.0]
z_offsets_cm   = [0.0]

# ALL materials - diversificar cada palabra
materials      = ["iron", "steel", "silicon", "aluminium", "uranium", "lead", "water"]


# Generate 80+ rare words in ALL materials for balance
words_possible = [
    # 1 letra
    "M", "U", "O", "N",
    # 2 letras
    "MM", "MU", "MO", "MN", "UM", "UU", "UO", "UN", "OM", "OU", "OO", "ON", "NM", "NU", "NO", "NN",
    # 3 letras
    "MMM", "MMU", "MMO", "MMN", "MUM", "MUU", "MUO", "MUN", "MOM", "MOU", "MOO", "MON", "MNM", "MNU", "MNO", "MNN",
    "UMM", "UMU", "UMO", "UMN", "UUM", "UUU", "UUO", "UUN", "UOM", "UOU", "UOO", "UON", "UNM", "UNU", "UNO", "UNN",
    "OMM", "OMU", "OMO", "OMN", "OUM", "OUU", "OUO", "OUN", "OOM", "OOU", "OOO", "OON", "ONM", "ONU", "ONO", "ONN",
    "NMM", "NMU", "NMO", "NMN", "NUM", "NUU", "NUO", "NUN", "NOM", "NOU", "NOO", "NON", "NNM", "NNU", "NNO", "NNN",
    # 4 letras
    "MMMM", "MMMU", "MMMO", "MMMN", "MMUM", "MMUU", "MMUO", "MMUN", "MMOM", "MMOU", "MMOO", "MMON", "MMNM", "MMNU", "MMNO", "MMNN",
    "MUMM", "MUMU", "MUMO", "MUMN", "MUUM", "MUUU", "MUUO", "MUUN", "MUOM", "MUOU", "MUOO", "MUON", "MUNM", "MUNU", "MUNO", "MUNN",
    "MOMM", "MOMU", "MOMO", "MOMN", "MOUM", "MOUU", "MOUO", "MOUN", "MOOM", "MOOU", "MOOO", "MOON", "MONM", "MONU", "MONO", "MONN",
    "MNMM", "MNMU", "MNMO", "MNMN", "MNUM", "MNUU", "MNUO", "MNUN", "MNOM", "MNOU", "MNOO", "MNON", "MNNM", "MNNU", "MNNO", "MNNN",
    "UMMM", "UMMU", "UMMO", "UMMN", "UMUM", "UMUU", "UMUO", "UMUN", "UMOM", "UMOU", "UMOO", "UMON", "UMNM", "UMNU", "UMNO", "UMNN",
    "UUMM", "UUMU", "UUMO", "UUMN", "UUUM", "UUUU", "UUUO", "UUUN", "UUOM", "UUOU", "UUOO", "UUON", "UUNM", "UUNU", "UUNO", "UUNN",
    "UOMM", "UOMU", "UOMO", "UOMN", "UOUM", "UOUU", "UOUO", "UOUN", "UOOM", "UOOU", "UOOO", "UOON", "UONM", "UONU", "UONO", "UONN",
    "UNMM", "UNMU", "UNMO", "UNMN", "UNUM", "UNUU", "UNUO", "UNUN", "UNOM", "UNOU", "UNOO", "UNON", "UNNM", "UNNU", "UNNO", "UNNN",
    "OMMM", "OMMU", "OMMO", "OMMN", "OMUM", "OMUU", "OMUO", "OMUN", "OMOM", "OMOU", "OMOO", "OMON", "OMNM", "OMNU", "OMNO", "OMNN",
    "OUMM", "OUMU", "OUMO", "OUMN", "OUUM", "OUUU", "OUUO", "OUUN", "OUOM", "OUOU", "OUOO", "OUON", "OUNM", "OUNU", "OUNO", "OUNN",
    "OOMM", "OOMU", "OOMO", "OOMN", "OOUM", "OOUU", "OOUO", "OOUN", "OOOM", "OOOU", "OOOO", "OOON", "OONM", "OONU", "OONO", "OONN",
    "ONMM", "ONMU", "ONMO", "ONMN", "ONUM", "ONUU", "ONUO", "ONUN", "ONOM", "ONOU", "ONOO", "ONON", "ONNM", "ONNU", "ONNO", "ONNN",
    "NMMM", "NMMU", "NMMO", "NMMN", "NMUM", "NMUU", "NMUO", "NMUN", "NMOM", "NMOU", "NMOO", "NMON", "NMNM", "NMNU", "NMNO", "NMNN",
    "NUMM", "NUMU", "NUMO", "NUMN", "NUUM", "NUUU", "NUUO", "NUUN", "NUOM", "NUOU", "NUOO", "NUON", "NUNM", "NUNU", "NUNO", "NUNN",
    "NOMM", "NOMU", "NOMO", "NOMN", "NOUM", "NOUU", "NOUO", "NOUN", "NOOM", "NOOU", "NOOO", "NOON", "NONM", "NONU", "NONO", "NONN",
    "NNMM", "NNMU", "NNMO", "NNMN", "NNUM", "NNUU", "NNUO", "NNUN", "NNOM", "NNOU", "NNOO", "NNON", "NNNM", "NNNU", "NNNO", "NNNN"
]

words_geometry = ["N", "M", "U", "O"]


# ===========================================================================
# GEOMETRY CREATION
# ===========================================================================
all_combos = list(iproduct(
    spacings, ratios, fontsizes, materials, words_geometry,
    strokes, depth_z_cm_list, x_offsets_cm, y_offsets_cm, z_offsets_cm
))
total_geometries = len(all_combos)
print(f"[INFO] Total geometries to generate: {total_geometries}")

if not create_geometries:
    print("[INFO] create_geometries=False → nothing to do.")
    sys.exit(0)

print("\n" + "="*60)
print("GEOMETRY CREATION")
print("="*60)

created = skipped = errors = 0

for i, (spacing, ratio, fontsize, material, word,
        stroke, depth_z_cm, x_offset_cm, y_offset_cm, z_offset_cm) in enumerate(all_combos, start=1):

    if i > max_geometries:
        print(f"[INFO] Reached max_geometries={max_geometries}. Stopping.")
        break

    # Offset suffix: only added when any offset is non-zero.
    # Files without this suffix have offset=(0,0,0) by convention.
    if x_offset_cm != 0.0 or y_offset_cm != 0.0 or z_offset_cm != 0.0:
        offset_str = f"_xoff{x_offset_cm:g}_yoff{y_offset_cm:g}_zoff{z_offset_cm:g}"
    else:
        offset_str = ""

    namefile = (
        f"_Lpx{Lpx}_Lpy{Lpy}_Lpz{Lpz}"
        f"_npx{npx}_npy{npy}_npz{npz}"
        f"_zTop{zPosDetector_top}_zBot{zPosDetector_bot}"
        f"_spacing{spacing}_ratio{ratio}"
        f"_FontX{fontsize}_FontY{fontsize}"
        f"_mat{material}_word{word}_stroke{stroke}"
        f"_depthZ{int(depth_z_cm)}"
        f"{offset_str}"
    )
    output_json = os.path.join(PATH_geometry_files, namefile + ".json")

    if os.path.exists(output_json):
        print(f"[SKIP {i}/{total_geometries}] Already exists: {namefile}")
        skipped += 1
        continue

    # Random skip for diversity (only when SKIP_JSON_CREATION=True)
    if SKIP_JSON_CREATION and random.random() > SKIP_THRESHOLD:
        skipped += 1
        continue

    command = [
        "python3", CREATE_GEOMETRY_SCRIPT,
        "--Lpx", str(Lpx), "--Lpy", str(Lpy), "--Lpz", str(Lpz),
        "--npx", str(npx), "--npy", str(npy), "--npz", str(npz),
        "--zPosDetector_top", str(zPosDetector_top),
        "--zPosDetector_bot", str(zPosDetector_bot),
        "--spacing",          str(spacing),
        "--ratio",            str(ratio),
        "--depth_z_cm",       str(depth_z_cm),
        "--x_offset_cm",      str(x_offset_cm),
        "--y_offset_cm",      str(y_offset_cm),
        "--z_offset_cm",      str(z_offset_cm),
        "--FontSizeX",        str(fontsize),
        "--FontSizeY",        str(fontsize),
        "--material",         material,
        "--word_geometry",    word,
        "--StrokeWidth",      str(stroke),
        "--output_json",      output_json,
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = result.stderr or result.stdout
        errors += 1
        if "cabe" in error_msg.lower() or "fit" in error_msg.lower():
            print(f"[TOO LARGE {i}/{total_geometries}] {namefile}")
        else:
            print(f"[ERROR {i}/{total_geometries}] {namefile}")
            print(f"   {error_msg[:200]}")
        record_geometry_error(namefile, error_msg)
    else:
        print(f"[OK {i}/{total_geometries}] {namefile}")
        created += 1


print("\n" + "="*60)
print(f"Done. Created: {created}  |  Skipped: {skipped}  |  Errors: {errors}")
print("="*60)
