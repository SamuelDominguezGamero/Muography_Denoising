"""
makeConfigurationFileMUON.py
────────────────────────────
Generates a Geant4 geometry configuration JSON that spells the word "MUON" using cubic lead voxels placed between the two detector planes (z = 0).

The script defines a 5×5 pixel font for each letter, computes the (x, y) position of every filled pixel, and writes the result to confMUON.json.


Output:
    Muography_Denoising/Muon_Generation/data/confMUON.json

Key parameters (edit the CONFIG block below):
    VOXEL_SIZE      Side of each cubic voxel [cm].
    LETTER_GAP      Number of empty voxel columns between consecutive letters.
    Z_POS           z coordinate shared by all voxels [cm].
                    Must be inside the gap between the two detectors,
                    i.e. between z = -42.5 cm and z = +42.5 cm.
    MATERIAL        String key of the voxel material. Must match a key in
                    DetectorConstruction::ConstructMaterials()
                    (e.g. "lead", "iron", "aluminium", "silicon").
"""

import json
import os

# ─── Pixel font: 5 columns × 5 rows ──────────────────────────────────────────
# Each letter is stored top-to-bottom (row 0 = top, row 4 = bottom).
# 1 = filled voxel (lead), 0 = empty (air).
# Visualisation:
#
#   M:  █ . . . █       U:  █ . . . █
#       █ █ . █ █           █ . . . █
#       █ . █ . █           █ . . . █
#       █ . . . █           █ . . . █
#       █ . . . █           . █ █ █ .
#
#   O:  . █ █ █ .       N:  █ . . . █
#       █ . . . █           █ █ . . █
#       █ . . . █           █ . █ . █
#       █ . . . █           █ . . █ █
#       . █ █ █ .           █ . . . █

LETTERS = {
    'M': [
        [1, 0, 0, 0, 1],   # row 0 – top
        [1, 1, 0, 1, 1],   # row 1 – shoulders of the M
        [1, 0, 1, 0, 1],   # row 2 – centre notch
        [1, 0, 0, 0, 1],   # row 3
        [1, 0, 0, 0, 1],   # row 4 – bottom
    ],
    'U': [
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],   # row 4 – curved bottom of U
    ],
    'O': [
        [0, 1, 1, 1, 0],   # row 0 – top arch
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],   # row 4 – bottom arch
    ],
    'N': [
        [1, 0, 0, 0, 1],
        [1, 1, 0, 0, 1],   # row 1 – diagonal starts top-left
        [1, 0, 1, 0, 1],   # row 2 – diagonal crosses the middle
        [1, 0, 0, 1, 1],   # row 3 – diagonal reaches bottom-right
        [1, 0, 0, 0, 1],
    ],
}

# ─── CONFIG ───────────────────────────────────────────────────────────────────

WORD       = "MUON"   # word to write; must only contain letters defined above

VOXEL_SIZE = 4.0      # [cm] side of each cubic voxel
                      # 4 cm → total width 92 cm, fits inside 110×110 cm detectors
                      # 10 cm → total width 230 cm, extends beyond detector acceptance

LETTER_GAP = 1        # number of empty voxel columns between letters

Z_POS      = 0.0      # [cm] z position shared by all voxels
                      # 0 = midpoint between the two detector planes (z = ±75 cm)

MATERIAL   = "lead"   # string key in DetectorConstruction::ConstructMaterials()

N_COLS     = 5        # columns per letter (matches font definition above)
N_ROWS     = 5        # rows    per letter (matches font definition above)

# ─── Output path relative to this script location ────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH   = os.path.join(SCRIPT_DIR, "..", "data", "confMUON.json")

# ─── Compute layout ───────────────────────────────────────────────────────────

n_letters   = len(WORD)

# Total number of voxel columns occupied by the word (letters + gaps between them)
total_cols  = n_letters * N_COLS + (n_letters - 1) * LETTER_GAP
total_width = total_cols * VOXEL_SIZE   # [cm]

# x coordinate of the centre of the leftmost voxel column, so that the whole
# word is centred at x = 0.
x_origin = -total_width / 2.0 + VOXEL_SIZE / 2.0

# y coordinate of the centre of the topmost voxel row, so that the whole word
# is centred at y = 0.
total_height = N_ROWS * VOXEL_SIZE      # [cm]
y_origin     = total_height / 2.0 - VOXEL_SIZE / 2.0

# ─── Build voxel list ────────────────────────────────────────────────────────

voxels = []

for i_letter, char in enumerate(WORD):

    if char not in LETTERS:
        raise ValueError(f"Letter '{char}' not defined in the pixel font.")

    grid  = LETTERS[char]

    # x offset for the start (left edge) of this letter
    x_off = i_letter * (N_COLS + LETTER_GAP) * VOXEL_SIZE

    for row_idx, row in enumerate(grid):        # row_idx 0 = top of the letter
        for col_idx, filled in enumerate(row):  # col_idx 0 = left column

            if not filled:
                continue  # empty pixel – no voxel placed here

            # Compute world (x, y) coordinates of the voxel centre.
            # x increases to the right; y increases upward (row 0 → max y).
            x = x_origin + x_off + col_idx * VOXEL_SIZE
            y = y_origin - row_idx * VOXEL_SIZE

            voxels.append({
                "xPosVoxel": round(x, 3),
                "yPosVoxel": round(y, 3)
            })

# ─── Assemble the full JSON configuration ────────────────────────────────────
# The detector geometry is copied verbatim from confExample.json so this file
# can replace it directly in a simulation run.

config = {
    "theWorld": {
        "xSizeWorld": 300,
        "ySizeWorld":  300,
        "zSizeWorld":  300,
        "sizeBoxCRY":  300,
        "zOffsetCRY":  150
    },
    "Detectors": [
        {
            "xPosDetector": 0, "yPosDetector": 0, "zPosDetector":  110,
            "xDirDetector": 0, "yDirDetector": 0, "zDirDetector":   0,
            "xSizeDetector": 150, "ySizeDetector": 150, "zSizeDetector": 1,
            "Layers": [
                {"xPosLayer": 0, "yPosLayer": 0, "zPosLayer":  0,
                 "xDirLayer": 0, "yDirLayer": 0, "zDirLayer":   0,
                 "xSizeLayer": 150, "ySizeLayer": 150, "zSizeLayer": 1}, # LAYER 1
                {"xPosLayer": 0, "yPosLayer": 0, "zPosLayer": 20,
                 "xDirLayer": 0, "yDirLayer": 0, "zDirLayer":   0,
                 "xSizeLayer": 150, "ySizeLayer": 150, "zSizeLayer": 1}, # LAYER 2
            ]
        },
        {
            "xPosDetector": 0, "yPosDetector": 0, "zPosDetector": -110,
            "xDirDetector": 0, "yDirDetector": 0, "zDirDetector":   0,
            "xSizeDetector": 150, "ySizeDetector": 150, "zSizeDetector": 1,
            "Layers": [
                {"xPosLayer": 0, "yPosLayer": 0, "zPosLayer":  0,
                 "xDirLayer": 0, "yDirLayer": 0, "zDirLayer":   0,
                 "xSizeLayer": 150, "ySizeLayer": 150, "zSizeLayer": 1}, # LAYER 3
                {"xPosLayer": 0, "yPosLayer": 0, "zPosLayer": -20,
                 "xDirLayer": 0, "yDirLayer": 0, "zDirLayer":   0,
                 "xSizeLayer": 150, "ySizeLayer": 150, "zSizeLayer": 1}, # LAYER 4
            ]
        }
    ],
    # ── Voxel shared configuration ─────────────────────────────────────────
    "VoxelConfig": {
        "VoxelSize": VOXEL_SIZE,   # [cm] side of each cubic voxel
        "zPosVoxel": Z_POS,        # [cm] z coordinate shared by all voxels
        "material":  MATERIAL      # material key from DetectorConstruction
    },
    # ── One entry per filled pixel of the word ─────────────────────────────
    "TheVoxels": voxels
}

# ─── Write JSON ───────────────────────────────────────────────────────────────

with open(OUT_PATH, "w") as f:
    json.dump(config, f, indent=4)

# ─── Print summary ────────────────────────────────────────────────────────────

x_min = x_origin
x_max = x_origin + total_width - VOXEL_SIZE
y_min = y_origin - (N_ROWS - 1) * VOXEL_SIZE
y_max = y_origin

print("=" * 50)
print(f"  Word          : {WORD}")
print(f"  Total voxels  : {len(voxels)}")
print(f"  Voxel size    : {VOXEL_SIZE} cm")
print(f"  Total width   : {total_width} cm  (x: [{x_min:.1f}, {x_max:.1f}] cm)")
print(f"  Total height  : {total_height} cm  (y: [{y_min:.1f}, {y_max:.1f}] cm)")
print(f"  z position    : {Z_POS} cm")
print(f"  Material      : {MATERIAL}")
print(f"  Output file   : {os.path.abspath(OUT_PATH)}")
print("=" * 50)

# ─── ASCII preview of the geometry ───────────────────────────────────────────
# Print the word as it will appear in the (x, y) plane of Geant4

print("\n  ASCII preview (top = +y, right = +x):\n")
gap_col = ["."] * LETTER_GAP
for row_idx in range(N_ROWS):
    line = []
    for i_letter, char in enumerate(WORD):
        if i_letter > 0:
            line += gap_col          # empty column(s) between letters
        row = LETTERS[char][row_idx]
        line += ["█" if v else "." for v in row]
    print("    " + " ".join(line))
print()
