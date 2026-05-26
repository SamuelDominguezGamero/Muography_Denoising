# Pipeline: JSON Geometry Creation and Ground Truth

This document describes the full pipeline for generating Geant4 geometry JSON files and their
associated 2D ground truth arrays. It covers both **word/letter** geometries and
**non-letter geometric shape** geometries.

---

## Overview

```
STEP 1 ─ Create geometry JSON (Geant4 input) + 2D ground truth (.npy)
STEP 2 ─ Run Geant4 Monte Carlo simulation (cluster only)
STEP 3 ─ POCA reconstruction (cluster only)
STEP 4 ─ Merge simulation results into .h5 dataset
STEP 5 ─ (Optional) Reconstruct full 3D/2D ground truth from JSON
```

This document covers **STEP 1** and **STEP 5** in detail.

---

## STEP 1A — Word / Letter Geometries

### Script: `create_geometry.py`

**Purpose:** Creates one Geant4 JSON and one 2D ground truth `.npy` for a single
word/letter geometry configuration.

**Key parameters:**

| Argument | Default | Description |
|---|---|---|
| `--word_geometry` | `MUON` | Word to embed (letters: M, U, O, N) |
| `--FontSizeX/Y` | `12` | Font size in G4 voxels (8, 10, 12, 14, 16) |
| `--StrokeWidth` | `1` | Stroke width in G4 voxels (1, 2, 3) |
| `--spacing` | `1` | Inter-letter spacing in G4 voxels |
| `--depth_z_cm` | `2.0` | Slab thickness in cm (Z direction) |
| `--material` | `lead` | Voxel material |
| `--x_offset_cm` | `0.0` | X offset of word center from world center (cm) |
| `--y_offset_cm` | `0.0` | Y offset of word center from world center (cm) |
| `--z_offset_cm` | `0.0` | Z position of slab center (cm) |
| `--ratio` | `2` | G4-to-POCA voxel ratio |
| `--output_json` | — | Output JSON path |
| `--output2D_density` | — | Output ground truth `.npy` path |

**Method:** `embed_word_in_geometry_efficient` — single slab, one voxel per active XY pixel
with `zSizeVoxel = depth_z_cm` centered at `z_offset_cm`.

**Ground truth format:** `(ny, nx)` int array (0/1) = XY footprint of the word at G4
resolution. Built directly from `word_matrix` placement — independent of Z offset.

---

### Script: `create_multiple_geometries.py`

**Purpose:** Orchestrator. Loops over all parameter combinations and calls
`create_geometry.py` once per combination. On the cluster, each call is submitted as
an independent SLURM job.

**Geometry variation parameters** (edit the top section):

```python
spacings       = [1, 2, 3]
ratios         = [1, 2]
fontsizes      = [8, 10, 12, 14, 16]
strokes        = [1, 2, 3]
depth_z_cm_list = [1.0, 2.0, ..., 35.0]
x_offsets_cm   = [0.0]   # add non-zero values for offset variants
y_offsets_cm   = [0.0]
z_offsets_cm   = [0.0]
materials      = ["lead", "iron", "uranium", "aluminium", "silicon", "steel"]
words_geometry = [...]   # all 1-4 letter combinations of M, U, O, N
```

**File naming convention:**

```
_Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54
_spacing{s}_ratio{r}_FontX{fx}_FontY{fy}
_mat{material}_word{word}_stroke{sw}_depthZ{dz}
[_xoff{x}_yoff{y}_zoff{z}]   ← only appended when any offset != 0
```

**Backwards compatibility rule:** Files **without** the `_xoff/_yoff/_zoff` suffix were
generated with offset = (0, 0, 0). This allows ~4600 existing simulations to remain valid
without any re-processing.

**Output directories (local):**
- JSONs: `data/geometric_configurations_json/`
- 2D GT:  `data/ground_truth_data/2Dimensions/UNET1/`
- 3D GT:  `data/ground_truth_data/3Dimensions/`

---

## STEP 1B — Non-Letter Geometric Shapes

### Script: `create_other_geometries_not_words.py`

**Purpose:** Dual-mode script.
- **Orchestrator mode** (no `--shape` flag): loops over all shape/material/size
  combinations and calls itself in creator mode via subprocess.
- **Creator mode** (`--shape` flag): builds one JSON + 2D GT for a single shape
  configuration.

**Implemented shapes:**

| Shape | Type | Method |
|---|---|---|
| `rectangle_filled` / `rectangle_hollow` | Extruded slab | Vectorized XY mask |
| `cylinder_filled` / `cylinder_hollow` | Extruded slab | Vectorized XY mask |
| `triangle_slab` | Extruded equilateral triangle | CCW cross-product test |
| `sphere_filled` / `sphere_hollow` | Full 3D | Per-G4-voxel |
| `tetrahedron_filled` / `tetrahedron_hollow` | Full 3D | Barycentric coordinates |

**File naming convention:**

```
shape_{type}_Lpx128_..._ratio{r}_sx{sx}_sy{sy}_cx{cx}_cy{cy}_mat{m}_dz{dz}
```

**Output directory (local):**
- JSONs: `data/geometric_configurations_jsons_not_letters/`
- 2D GT:  `data/ground_truth_data/2Dimensions/UNET1/`

---

## STEP 5 — Reconstruct Ground Truth from JSON

### Script: `create_ground_truth_from_json.py`

**Purpose:** Reconstructs the ground truth 2D projection tensor directly from a JSON
geometry file. This is the **authoritative** ground truth source used to build the `.h5`
training datasets, and it handles any geometry type (words, shapes, offset or centered).

**Method:**
1. Read `TheVoxels` list from the JSON.
2. Fill a 3D binary tensor `(128, 128, 128)` from voxel positions and sizes.
3. Compute three **max-projections**:
   - `xy = max(tensor, axis=Z)` — top view
   - `xz = max(tensor, axis=Y).T` — front view
   - `yz = max(tensor, axis=X).T` — side view
4. Stack into `(128, 128, 3)` tensor and save.

**Why this handles offsets correctly:** max-projection accumulates any non-air voxel
regardless of its Z position. A word at z=+30 cm will appear correctly in XZ/YZ
projections.

**Output:**
- `data/ground_truth_data/3Dimensions/tensor_3D_{stem}.npy` — shape `(128, 128, 128)`
- `data/ground_truth_data/2Dimensions/tensor_2D_{stem}.npy` — shape `(128, 128, 3)`

---

## Data Flow Diagram

```
create_geometry.py / create_other_geometries_not_words.py
          │
          ├─→ JSON (data/geometric_configurations_json*/)
          │          │
          │          └─→ create_ground_truth_from_json.py
          │                        │
          │                        └─→ tensor_2D_*.npy  (128×128×3)  ← used in dataset
          │                        └─→ tensor_3D_*.npy  (128×128×128)
          │
          └─→ *_ground_truth_density2D.npy  (128×128, XY only)  ← legacy, not used by dataset
```

> **Note:** The `*_ground_truth_density2D.npy` files saved by `create_geometry.py` directly
> are only the XY footprint at G4 resolution. The training `.h5` datasets use the 3-channel
> `tensor_2D_*.npy` from `create_ground_truth_from_json.py`.

---

## Traceability: Parameter Encoding in Filenames

All parameters needed to reproduce a simulation are encoded in the filename.

### Letter geometry example:
```
_Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing2_ratio1_FontX12_FontY12_matlead_wordMUON_stroke2_depthZ10
```
→ offset = (0, 0, 0) by convention (no `_xoff` tag).

### Letter geometry with offset:
```
_Lpx128_..._matlead_wordMUON_stroke2_depthZ10_xoff20_yoff-10_zoff5
```
→ word center displaced +20 cm in X, -10 cm in Y, slab at z=+5 cm.

### Shape geometry example:
```
shape_cylinder_hollow_Lpx128_..._ratio1_sx20_sy20_cx0_cy0_matiron_dz15
```

---

## Adding New Offset Combinations

To generate word geometries at non-zero offsets:

1. Edit `create_multiple_geometries.py`, section `GEOMETRY VARIATIONS`:
   ```python
   x_offsets_cm = [0.0, 20.0, -20.0]
   y_offsets_cm = [0.0, 20.0, -20.0]
   z_offsets_cm = [0.0, 10.0, -10.0]
   ```
2. Run locally or submit to cluster (`environment = "cluster"`).
3. After simulation, run `create_ground_truth_from_json.py` on the new JSONs.

Files with offset=(0,0,0) are **not** regenerated (skipped by the `os.path.exists` check).

---

## Allowed Materials

Defined in Geant4 `DetectorConstruction.cc` and validated in `create_geometry.py`:

| Material | Density (g/cm³) |
|---|---|
| lead | 11.35 |
| iron | 7.874 |
| uranium | 18.95 |
| aluminium | 2.699 |
| steel | 8.00 |
| silicon | 2.33 |
| argon | 0.001639 |
| air | 0.00120479 |

Reference: https://geant4-userdoc.web.cern.ch/UsersGuides/ForApplicationDeveloper/html/Appendix/materialNames.html
