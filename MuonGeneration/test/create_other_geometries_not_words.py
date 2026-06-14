"""
Generates geometry JSON files for non-word shapes for UNET training.

════════════════════════════════════════════════════════════════════════════════
MODES
════════════════════════════════════════════════════════════════════════════════

  ORCHESTRATOR (default, no --shape argument)
  ─────────────────────────────────────────────
  Iterates all shape / parameter combinations defined in the VARIATIONS section
  and spawns itself as a subprocess in creator mode for each combination.

    python create_other_geometries_not_words.py

  Key knobs (edit at the top of the ORCHESTRATOR section):
    environment    – "local" | "cluster"  (selects output paths in PATHS dict)
    dimension      – "2D"   | "3D"        (output ground-truth format)
    create         – True / False         (actually run creator subprocesses)
    force_recreate – True / False         (overwrite existing files)
    max_geometries – int | float("inf")   (cap for quick tests)

  CREATOR (activated by passing --shape)
  ─────────────────────────────────────────────
  Creates a single geometry JSON (Geant4-compatible) plus a ground-truth
  density .npy array.

    python create_other_geometries_not_words.py \
        --shape     cylinder_filled \
        --size_x    30.0 \
        --depth_z_cm 20.0 \
        --material  lead \
        --dimensions 2D \
        --output_json        /path/to/output.json \
        --output2D_density   /path/to/gt_2d.npy

════════════════════════════════════════════════════════════════════════════════
CREATOR ARGUMENTS
════════════════════════════════════════════════════════════════════════════════

  World / grid
    --Lpx / --Lpy / --Lpz         World half-size in each axis (cm). Default 128.
    --npx / --npy / --npz         Number of G4 voxels per axis. Default 128.
    --ratio                       G4voxel / POCAvoxel integer ratio. Default 1.
    --zPosDetector_top / _bot     Z position of top/bottom detectors (cm).
                                  Defaults: +54 / -54.

  Shape (required: --shape; optional geometry parameters)
    --shape          One of the shapes listed below (required).
    --size_x         Half-width (rect) or circumradius / radius (cyl / sphere /
                     tri / tetra) in cm. Default 20.
    --size_y         Half-height; meaningful only for rectangle_* shapes. Default 20.
    --depth_z_cm     Z thickness for extruded shapes (rectangle, cylinder,
                     triangle_slab). Ignored by sphere / tetrahedron. Default 10.
    --center_x       X offset of shape centre (cm). Default 0.
    --center_y       Y offset of shape centre (cm). Default 0.
    --wall_thickness Wall thickness for hollow shapes (cm). Default 2.
    --material       Material name (see list below). Default "lead".

  Outputs
    --dimensions        "2D" (central XY slice) or "3D" (full volume). Default "2D".
    --output_json       Path for the Geant4 JSON file (required).
    --output2D_density  Path for the 2-D ground-truth .npy (used when --dimensions 2D).
    --output3D_density  Path for the 3-D ground-truth .npy (used when --dimensions 3D).

════════════════════════════════════════════════════════════════════════════════
AVAILABLE SHAPES
════════════════════════════════════════════════════════════════════════════════

  rectangle_filled / rectangle_hollow   – axis-aligned box extruded in Z
  cylinder_filled  / cylinder_hollow    – circular cross-section extruded in Z
  sphere_filled    / sphere_hollow      – 3-D sphere (ignores depth_z_cm)
  triangle_slab                         – equilateral-triangle XY section extruded in Z
  tetrahedron_filled / tetrahedron_hollow – regular tetrahedron (ignores depth_z_cm)

════════════════════════════════════════════════════════════════════════════════
AVAILABLE MATERIALS
════════════════════════════════════════════════════════════════════════════════

  lead, iron, uranium, aluminium, steel, silicon, argon, air

════════════════════════════════════════════════════════════════════════════════
EXAMPLES
════════════════════════════════════════════════════════════════════════════════

  # Orchestrator: generate all combinations locally in 2-D mode
  python create_other_geometries_not_words.py

  # Creator: solid lead sphere, radius 40 cm, 3-D ground truth
  python create_other_geometries_not_words.py \
      --shape sphere_filled --size_x 40 \
      --material lead --dimensions 3D \
      --output_json /tmp/sphere.json \
      --output3D_density /tmp/sphere_gt.npy

  # Creator: hollow iron cylinder, radius 30 cm, wall 3 cm, slab depth 20 cm
  python create_other_geometries_not_words.py \
      --shape cylinder_hollow --size_x 30 --wall_thickness 3 --depth_z_cm 20 \
      --material iron --dimensions 2D \
      --output_json /tmp/cyl_hollow.json \
      --output2D_density /tmp/cyl_hollow_gt.npy
"""

import sys, os

# ============================================================================
# CREATOR MODE  (activated when --shape is present in sys.argv)
# ============================================================================
if "--shape" in sys.argv:
    import argparse, json, math
    import numpy as np

    parser = argparse.ArgumentParser(
        description="Create a single non-word geometry JSON + ground-truth for Geant4."
    )
    # World / grid
    parser.add_argument("--Lpx",  type=float, default=128.0)
    parser.add_argument("--Lpy",  type=float, default=128.0)
    parser.add_argument("--Lpz",  type=float, default=128.0)
    parser.add_argument("--npx",  type=int,   default=128)
    parser.add_argument("--npy",  type=int,   default=128)
    parser.add_argument("--npz",  type=int,   default=128)
    parser.add_argument("--ratio",type=int,   default=1,
        help="G4voxel/POCAvoxel ratio (integer >= 1)")
    parser.add_argument("--zPosDetector_top", type=float, default=54.0)
    parser.add_argument("--zPosDetector_bot", type=float, default=-54.0)
    # Shape
    parser.add_argument("--shape",          type=str,   required=True)
    parser.add_argument("--size_x",         type=float, default=20.0,
        help="Half-width (rect) or radius/circumradius (cyl/sphere/tri/tetra) in cm")
    parser.add_argument("--size_y",         type=float, default=20.0,
        help="Half-height for rectangle shapes only (cm)")
    parser.add_argument("--depth_z_cm",     type=float, default=10.0,
        help="Z thickness for extruded shapes (cm). Ignored by sphere/tetrahedron.")
    parser.add_argument("--center_x",       type=float, default=0.0)
    parser.add_argument("--center_y",       type=float, default=0.0)
    parser.add_argument("--wall_thickness", type=float, default=2.0,
        help="Wall thickness for hollow shapes (cm)")
    parser.add_argument("--material",       type=str,   default="lead")
    parser.add_argument("--dimensions",     type=str,   default="2D",
        help="'2D' saves central XY slice; '3D' saves full density volume")
    # Outputs
    parser.add_argument("--output_json",       type=str, required=True)
    parser.add_argument("--output2D_density",  type=str, default=None)
    parser.add_argument("--output3D_density",  type=str, default=None)

    args = parser.parse_args()

    ALLOWED_MATERIALS = {
        "lead", "iron", "uranium", "aluminium", "argon", "silicon", "steel", "air", "water"
    }
    if args.material not in ALLOWED_MATERIALS:
        sys.exit(f"[ERROR] Material '{args.material}' not in {ALLOWED_MATERIALS}")

    # ------ G4 grid setup ------
    Lpx, Lpy, Lpz = args.Lpx, args.Lpy, args.Lpz
    npx, npy, npz = args.npx, args.npy, args.npz
    ratio = args.ratio
    nx, ny, nz = npx // ratio, npy // ratio, npz // ratio
    Lx, Ly, Lz = Lpx, Lpy, Lpz
    SV_x = Lx / nx
    SV_y = Ly / ny
    SV_z = Lz / nz

    xc = -Lx / 2 + (np.arange(nx) + 0.5) * SV_x   # (nx,)
    yc = -Ly / 2 + (np.arange(ny) + 0.5) * SV_y   # (ny,)
    zc = -Lz / 2 + (np.arange(nz) + 0.5) * SV_z   # (nz,)
    Y3, X3, Z3 = np.meshgrid(yc, xc, zc, indexing="ij")  # each (ny, nx, nz)

    shape    = args.shape
    sx       = args.size_x
    sy       = args.size_y
    dz       = args.depth_z_cm
    cx       = args.center_x
    cy       = args.center_y
    t        = args.wall_thickness
    material = args.material

    # ------ Build boolean mask (ny, nx, nz) ------
    hz   = dz / 2.0
    in_z = np.abs(Z3) <= hz   # Z slab for extruded shapes

    if shape == "rectangle_filled":
        mask = (np.abs(X3 - cx) <= sx) & (np.abs(Y3 - cy) <= sy) & in_z

    elif shape == "rectangle_hollow":
        outer = (np.abs(X3 - cx) <= sx) & (np.abs(Y3 - cy) <= sy)
        inner = (np.abs(X3 - cx) <= max(sx - t, 0.0)) & (np.abs(Y3 - cy) <= max(sy - t, 0.0))
        mask  = outer & ~inner & in_z

    elif shape == "cylinder_filled":
        mask = ((X3 - cx)**2 + (Y3 - cy)**2 <= sx**2) & in_z

    elif shape == "cylinder_hollow":
        outer = (X3 - cx)**2 + (Y3 - cy)**2 <= sx**2
        inner = (X3 - cx)**2 + (Y3 - cy)**2 <= max(sx - t, 0.0)**2
        mask  = outer & ~inner & in_z

    elif shape == "sphere_filled":
        mask = (X3 - cx)**2 + (Y3 - cy)**2 + Z3**2 <= sx**2

    elif shape == "sphere_hollow":
        outer = (X3 - cx)**2 + (Y3 - cy)**2 + Z3**2 <= sx**2
        inner = (X3 - cx)**2 + (Y3 - cy)**2 + Z3**2 <= max(sx - t, 0.0)**2
        mask  = outer & ~inner

    elif shape == "triangle_slab":
        # Equilateral triangle in XY: circumradius sx, apex pointing +Y, CCW vertices.
        # V0 -> V1 -> V2 is CCW; inside iff cross(edge, P-vertex) >= 0 for all 3 edges.
        R  = sx
        V0 = np.array([cx,                          cy + R])
        V1 = np.array([cx - R * math.sqrt(3) / 2,  cy - R / 2])
        V2 = np.array([cx + R * math.sqrt(3) / 2,  cy - R / 2])

        def _c2d(ex, ey, dx, dy):
            """2-D cross product of edge (ex,ey) and delta (dx,dy)."""
            return ex * dy - ey * dx

        e01x, e01y = float(V1[0] - V0[0]), float(V1[1] - V0[1])
        e12x, e12y = float(V2[0] - V1[0]), float(V2[1] - V1[1])
        e20x, e20y = float(V0[0] - V2[0]), float(V0[1] - V2[1])

        in_tri = (
            (_c2d(e01x, e01y, X3 - V0[0], Y3 - V0[1]) >= 0) &
            (_c2d(e12x, e12y, X3 - V1[0], Y3 - V1[1]) >= 0) &
            (_c2d(e20x, e20y, X3 - V2[0], Y3 - V2[1]) >= 0)
        )
        mask = in_tri & in_z

    elif shape in ("tetrahedron_filled", "tetrahedron_hollow"):
        # Regular tetrahedron: circumsphere radius R_c = sx, centroid at (cx, cy, 0).
        # Base equilateral triangle at z = -R_c/3, apex at z = R_c.
        # All 4 faces are equilateral triangles with edge length a = 2*sqrt(6)/3 * R_c.
        R_c    = sx
        R_base = 2.0 * math.sqrt(2.0) * R_c / 3.0   # circumradius of base triangle
        z_base = -R_c / 3.0
        z_apex =  R_c

        # Vertices (4, 3): base CCW from above, then apex
        V = np.array([
            [cx,                                cy + R_base,        z_base],
            [cx - R_base * math.sqrt(3.0) / 2, cy - R_base / 2.0,  z_base],
            [cx + R_base * math.sqrt(3.0) / 2, cy - R_base / 2.0,  z_base],
            [cx,                                cy,                 z_apex],
        ])

        def _tetra_inside(verts):
            """Boolean (ny,nx,nz): True where voxel center is inside tetrahedron."""
            T    = (verts[1:] - verts[0]).T           # (3, 3)
            pts  = np.stack([
                X3.ravel() - verts[0, 0],
                Y3.ravel() - verts[0, 1],
                Z3.ravel() - verts[0, 2],
            ], axis=1)                                 # (N, 3)
            try:
                bary = np.linalg.solve(T, pts.T).T    # (N, 3)
            except np.linalg.LinAlgError:
                return np.zeros(X3.shape, dtype=bool)
            bary4 = np.column_stack([1.0 - bary.sum(axis=1), bary])  # (N, 4)
            return np.all(bary4 >= -1e-10, axis=1).reshape(X3.shape)

        outer_mask = _tetra_inside(V)

        if shape == "tetrahedron_filled":
            mask = outer_mask
        else:
            # Hollow: scale all vertices toward centroid by (R_c - t) / R_c
            scale    = max((R_c - t) / R_c, 0.0)
            centroid = V.mean(axis=0)
            V_inner  = centroid + scale * (V - centroid)
            mask     = outer_mask & ~_tetra_inside(V_inner)

    else:
        sys.exit(
            f"[ERROR] Unknown shape '{shape}'. Allowed: "
            "rectangle_filled, rectangle_hollow, cylinder_filled, cylinder_hollow, "
            "sphere_filled, sphere_hollow, triangle_slab, "
            "tetrahedron_filled, tetrahedron_hollow"
        )

    if not mask.any():
        sys.exit(
            f"[WARN] Shape '{shape}' produced 0 filled voxels with these parameters "
            f"(sx={sx}, sy={sy}, dz={dz}, cx={cx}, cy={cy}, t={t})."
        )

    print(f"[INFO] Shape '{shape}': {int(mask.sum())} G4 voxels filled.")

    # ------ Convert mask to Geant4 voxel list ------
    # Extruded shapes  → efficient single-slab per XY position (zPosVoxel=0, zSizeVoxel=dz)
    # Volumetric shapes → one entry per G4 voxel with actual voxel size
    EXTRUDED = {
        "rectangle_filled", "rectangle_hollow",
        "cylinder_filled",  "cylinder_hollow",
        "triangle_slab",
    }

    voxel_list = []
    iy_arr, ix_arr, iz_arr = np.where(mask)

    if shape in EXTRUDED:
        seen = set()
        for iy, ix in zip(iy_arr, ix_arr):
            if (iy, ix) not in seen:
                seen.add((iy, ix))
                voxel_list.append({
                    "xPosVoxel":    float(X3[iy, ix, 0]),
                    "yPosVoxel":    float(Y3[iy, ix, 0]),
                    "zPosVoxel":    0.0,
                    "xSizeVoxel":   float(SV_x),
                    "ySizeVoxel":   float(SV_y),
                    "zSizeVoxel":   float(dz),
                    "materialVoxel": material,
                })
    else:
        for iy, ix, iz in zip(iy_arr, ix_arr, iz_arr):
            voxel_list.append({
                "xPosVoxel":    float(X3[iy, ix, iz]),
                "yPosVoxel":    float(Y3[iy, ix, iz]),
                "zPosVoxel":    float(Z3[iy, ix, iz]),
                "xSizeVoxel":   float(SV_x),
                "ySizeVoxel":   float(SV_y),
                "zSizeVoxel":   float(SV_z),
                "materialVoxel": material,
            })

    print(f"[INFO] Voxel list: {len(voxel_list)} entries in JSON.")

    # ------ Build global JSON dictionary ------
    def _make_detector(z_pos, Lx, Ly):
        layer_z2 = -10 if z_pos > 0 else 10
        return {
            "xPosDetector": 0, "yPosDetector": 0, "zPosDetector": z_pos,
            "xDirDetector": 0, "yDirDetector": 0, "zDirDetector": 0,
            "xSizeDetector": Lx, "ySizeDetector": Ly, "zSizeDetector": 20,
            "Layers": [
                {
                    "xPosLayer": 0, "yPosLayer": 0, "zPosLayer": 0,
                    "xDirLayer": 0, "yDirLayer": 0, "zDirLayer": 0,
                    "xSizeLayer": Lx, "ySizeLayer": Ly, "zSizeLayer": 1,
                },
                {
                    "xPosLayer": 0, "yPosLayer": 0, "zPosLayer": layer_z2,
                    "xDirLayer": 0, "yDirLayer": 0, "zDirLayer": 0,
                    "xSizeLayer": Lx, "ySizeLayer": Ly, "zSizeLayer": 1,
                },
            ],
        }

    global_dict = {
        "theWorld": {
            "xSizeWorld": Lx, "ySizeWorld": Ly, "zSizeWorld": Lz,
            "sizeBoxCRY": Lx, "zOffsetCRY": Lz / 2.0,
        },
        "Detectors": [
            _make_detector(args.zPosDetector_top, Lx, Ly),
            _make_detector(args.zPosDetector_bot, Lx, Ly),
        ],
        "VoxelConfig": {},
        "TheVoxels":   voxel_list,
    }

    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(global_dict, f, indent=4)
    print(f"[CORRECT] JSON saved: {args.output_json}")

    # ------ Ground-truth density ------
    DENSITY = {
        "lead": 11.35, "air": 0.00120479, "iron": 7.874, "uranium": 18.95,
        "aluminium": 2.699, "argon": 0.001639, "silicon": 2.33, "steel": 8.00,
    }
    density_val           = DENSITY.get(material, 1.0)
    MatrixGeometryBoolean = mask.astype(int)
    MatrixGeometryDensity = density_val * MatrixGeometryBoolean

    if ratio == 1:
        MGB_POCA = MatrixGeometryBoolean
        MGD_POCA = MatrixGeometryDensity
    else:
        r = ratio
        MGB_POCA = np.repeat(np.repeat(np.repeat(MatrixGeometryBoolean, r, 0), r, 1), r, 2)
        MGD_POCA = np.repeat(np.repeat(np.repeat(MatrixGeometryDensity,  r, 0), r, 1), r, 2)

    if args.dimensions == "2D" and args.output2D_density:
        xy_slice = MGB_POCA[:, :, nz // 2]   # central-Z slice → (npy, npx)
        os.makedirs(os.path.dirname(os.path.abspath(args.output2D_density)), exist_ok=True)
        np.save(args.output2D_density, xy_slice)
        print(f"[CORRECT] 2D ground-truth saved: {args.output2D_density}")
    elif args.dimensions == "3D" and args.output3D_density:
        os.makedirs(os.path.dirname(os.path.abspath(args.output3D_density)), exist_ok=True)
        np.save(args.output3D_density, MGD_POCA)
        print(f"[CORRECT] 3D ground-truth saved: {args.output3D_density}")

    sys.exit(0)


# ============================================================================
# ORCHESTRATOR MODE  (no --shape argument → generate all combinations)
# ============================================================================
import subprocess, time, json
from itertools import product

# ---------------------------------------------------------------------------
# CONTROL FLAGS
# ---------------------------------------------------------------------------
environment    = "local"       # "local" | "cluster"
dimension      = "2D"          # "2D"   | "3D"
create         = True
force_recreate = False         # CHANGED BACK: only create if not exists
max_geometries = float("inf")  # set to e.g. 50 to limit for a quick test

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
CREATE_SCRIPT = os.path.abspath(__file__)   # script calls itself in creator mode
ERROR_LOG     = os.path.join(SCRIPT_DIR, ".shape_errors.json")


#       "json":  "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/geometric_configurations_jsons_not_letters",
PATHS = {
    "local": {
        "json":  "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/geometric_configurations_jsons_not_letters",
        "den2D": "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/ground_truth_data/2Dimensions/UNET1",
        "den3D": "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/ground_truth_data/3Dimensions",
        "logs":  "/home/samuel/Work/Muography_Denoising/MuonGeneration/logs",
        "setup": None, "user": None,
    },
    "cluster": {
        "json":  "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/geometric_configurations_json",
        "den2D": "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/ground_truth_data/2Dimensions/UNET1",
        "den3D": "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/ground_truth_data/3Dimensions",
        "logs":  "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs",
        "setup": "/gpfs/users/dominguezs/Muography_Denoising/setup.sh",
        "user":  "dominguezs",
    },
}

if environment not in PATHS:
    sys.exit(f"[ERROR] environment must be one of {list(PATHS.keys())}")

P = PATHS[environment]
for key in ("json", "den2D", "den3D", "logs"):
    os.makedirs(P[key], exist_ok=True)

# ---------------------------------------------------------------------------
# WORLD / DETECTOR  (fixed)
# ---------------------------------------------------------------------------
Lpx, Lpy, Lpz = 128, 128, 128
npx, npy, npz = 128, 128, 128
zTop, zBot    =  54, -54

# ---------------------------------------------------------------------------
# VARIATIONS  <--- EDIT THIS SECTION
# ---------------------------------------------------------------------------
# GOAL: Balanced dataset with:
#   - All 7 materials represented equally
#   - All 9 shapes represented equally
#   - No material-shape bias (each material appears with each shape)
#   - Multiple size and depth variations
#   - Rough 50/50 hollow vs filled split
#
# Strategy: Generate 9 shapes × 7 materials × 5 sizes_x × 5 sizes_y × 4 depths_z
#           = 6300 total combinations (~800 per material, ~700 per shape)
#
# Material groups (user wants these balanced):
#   - iron + steel          (2 materiales, balanced together)
#   - aluminium + silicon   (2 materiales, balanced together)
#   - water                 (1 material)
#   - lead                  (1 material)
#   - uranium               (1 material)

shapes           = ["rectangle_filled", "rectangle_hollow",
                    "cylinder_filled", "cylinder_hollow",
                    "sphere_filled", "sphere_hollow",
                    "triangle_slab",
                    "tetrahedron_filled", "tetrahedron_hollow"]

materials        = ["iron", "steel", "aluminium", "silicon", "water", "lead", "uranium"]

sizes_x          = [12, 18, 24, 30, 36]
sizes_y          = [12, 18, 24, 30, 36]
depth_z_list     = [6, 12, 18, 24]

center_x_list    = [0]
center_y_list    = [0]
ratios           = [1]
wall_thicknesses = [2]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def fits_in_world(shape, sx, sy, dz, cx, cy, t=0):
    """Returns (ok: bool, reason: str). t = wall_thickness (cm)."""
    hx, hy, hz = Lpx / 2, Lpy / 2, Lpz / 2
    problems = []

    if shape in ("rectangle_filled", "rectangle_hollow"):
        if abs(cx) + sx > hx:
            problems.append(f"X |{cx}|+{sx} > {hx}")
        if abs(cy) + sy > hy:
            problems.append(f"Y |{cy}|+{sy} > {hy}")
        if dz / 2 > hz:
            problems.append(f"dz/2={dz/2} > hz={hz}")
        if shape == "rectangle_hollow" and (sx <= t or sy <= t):
            problems.append(f"wall_thickness {t} >= one side ({sx},{sy})")

    elif shape in ("cylinder_filled", "cylinder_hollow"):
        if abs(cx) + sx > hx:
            problems.append(f"X |{cx}|+{sx} > {hx}")
        if abs(cy) + sx > hy:
            problems.append(f"Y |{cy}|+{sx} > {hy}")
        if dz / 2 > hz:
            problems.append(f"dz/2={dz/2} > hz={hz}")
        if shape == "cylinder_hollow" and sx <= t:
            problems.append(f"wall_thickness {t} >= radius {sx}")

    elif shape in ("sphere_filled", "sphere_hollow"):
        if abs(cx) + sx > hx:
            problems.append(f"X |{cx}|+{sx} > {hx}")
        if abs(cy) + sx > hy:
            problems.append(f"Y |{cy}|+{sx} > {hy}")
        if sx > hz:
            problems.append(f"radius {sx} > hz={hz}")
        if shape == "sphere_hollow" and sx <= t:
            problems.append(f"wall_thickness {t} >= radius {sx}")

    elif shape == "triangle_slab":
        if abs(cx) + sx > hx:
            problems.append(f"X |{cx}|+{sx} > {hx}")
        if abs(cy) + sx > hy:
            problems.append(f"Y |{cy}|+{sx} > {hy}")
        if dz / 2 > hz:
            problems.append(f"dz/2={dz/2} > hz={hz}")

    elif shape in ("tetrahedron_filled", "tetrahedron_hollow"):
        # Circumsphere radius = sx; apex at +sx, base at -sx/3 in Z
        if abs(cx) + sx > hx:
            problems.append(f"X |{cx}|+{sx} > {hx}")
        if abs(cy) + sx > hy:
            problems.append(f"Y |{cy}|+{sx} > {hy}")
        if sx > hz:
            problems.append(f"circumradius {sx} > hz={hz}")
        if shape == "tetrahedron_hollow" and sx <= t:
            problems.append(f"wall_thickness {t} >= circumradius {sx}")

    return (not problems), "; ".join(problems)


def make_namefile(shape, sx, sy, dz, cx, cy, mat, ratio, t=None):
    base = (
        f"shape_{shape}"
        f"_Lpx{Lpx}_Lpy{Lpy}_Lpz{Lpz}_npx{npx}_npy{npy}_npz{npz}"
        f"_zTop{zTop}_zBot{zBot}"
        f"_ratio{ratio}_sx{sx}_sy{sy}_cx{cx}_cy{cy}_mat{mat}_dz{dz}"
    )
    if t is not None:
        base += f"_wt{t}"
    return base


def log_error(name, msg):
    log = json.load(open(ERROR_LOG)) if os.path.exists(ERROR_LOG) else {}
    log[name] = {"error": msg.strip(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(log, open(ERROR_LOG, "w"), indent=2)


def submit(command, i):
    if environment == "local":
        return subprocess.run(command, capture_output=True, text=True)
    MAX_JOBS, SLEEP = 2000, 10
    while True:
        n = len([l for l in subprocess.run(
            ["squeue", "-u", P["user"], "-h", "--format=%i"],
            capture_output=True, text=True).stdout.strip().split("\n") if l])
        if n < MAX_JOBS:
            break
        print(f"[THROTTLE] {n}/{MAX_JOBS} jobs. Waiting {SLEEP}s...")
        time.sleep(SLEEP)
    wrap = f"source {P['setup']} && {' '.join(command)}"
    return subprocess.run([
        "sbatch", f"--job-name=shape_{i}", "--time=01:00:00", "--mem=8G",
        "--cpus-per-task=2", f"--output={P['logs']}/log_shape_{i}.out",
        f"--chdir={P['logs']}", f"--wrap={wrap}", "--partition=wncompute_ifca",
    ], capture_output=True, text=True)


# ---------------------------------------------------------------------------
# BUILD COMBO LIST  (hollow shapes get a wall_thickness dimension)
# ---------------------------------------------------------------------------
all_combos = []
for shape, sx, sy, dz, cx, cy, mat, ratio in product(
        shapes, sizes_x, sizes_y, depth_z_list,
        center_x_list, center_y_list, materials, ratios):
    if "_hollow" in shape:
        for t in wall_thicknesses:
            all_combos.append((shape, sx, sy, dz, cx, cy, mat, ratio, t))
    else:
        all_combos.append((shape, sx, sy, dz, cx, cy, mat, ratio, None))

valid = [
    (s, sx, sy, dz, cx, cy, mat, r, t)
    for s, sx, sy, dz, cx, cy, mat, r, t in all_combos
    if fits_in_world(s, sx, sy, dz, cx, cy, t if t is not None else 0)[0]
]
invalid = len(all_combos) - len(valid)

print(f"[INFO] Total combos: {len(all_combos)} | Valid: {len(valid)} | Skipped: {invalid}")

if not create:
    sys.exit("[INFO] create=False. Set create=True to generate files.")

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
for i, (shape, sx, sy, dz, cx, cy, mat, ratio, t) in enumerate(valid, start=1):
    if i > max_geometries:
        print(f"[INFO] Reached max_geometries={max_geometries}. Stopping.")
        break

    name     = make_namefile(shape, sx, sy, dz, cx, cy, mat, ratio, t)
    out_json = os.path.join(P["json"],  name + ".json")
    out_2D   = os.path.join(P["den2D"], name + "_ground_truth_density2D.npy")
    out_3D   = os.path.join(P["den3D"], name + "_ground_truth_density3D.npy")

    if not force_recreate and os.path.exists(out_json):
        print(f"[SKIP | EXISTS] {i}/{len(valid)} {name}")
        continue

    command = [
        "python3", CREATE_SCRIPT,
        "--shape", shape,
        "--Lpx", str(Lpx), "--Lpy", str(Lpy), "--Lpz", str(Lpz),
        "--npx", str(npx), "--npy", str(npy), "--npz", str(npz),
        "--zPosDetector_top", str(zTop), "--zPosDetector_bot", str(zBot),
        "--ratio", str(ratio),
        "--size_x", str(sx), "--size_y", str(sy),
        "--depth_z_cm", str(dz),
        "--center_x", str(cx), "--center_y", str(cy),
        "--material", mat,
        "--dimensions", dimension,
        "--output_json",      out_json,
        "--output2D_density", out_2D,
        "--output3D_density", out_3D,
    ]
    if t is not None:
        command += ["--wall_thickness", str(t)]

    result = submit(command, i)

    if result.returncode != 0:
        err = result.stderr or result.stdout
        print(f"[ERROR] {i}/{len(valid)} FAILED: {name}\n        {err[:200]}")
        log_error(name, err)
    else:
        print(f"[OK] {i}/{len(valid)} {name}")
