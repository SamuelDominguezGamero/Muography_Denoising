"""
Creates the geometry json file to be parsed by Geant4.
Elements:
- detectors: measurement layers
- voxels: geometry with material, position, size

Efficient method: Word thickness via variable zSizeVoxel (single slab, not repeated layers)
Example: --depth_z_cm 10.0 creates 10cm thick word without layer repetition.
"""



import json
import math 
import sys
import numpy as np
import argparse
import os
from bitmaps_letters import get_word, get_letter, dimensions_test

# Debugging: 
dimensions_test() # should stop the whole program if the dimensions are not correct for the defined font sizes and stroke widths. This is crucial to ensure that the generated word matrices fit properly in the Geant4 geometry.


parser = argparse.ArgumentParser(description="Creation of geometry from POCA resolution ---> key parameter = ratio.")
parser.add_argument("--Lpx", type=float, default=128, help="Dimension X (cm).")
parser.add_argument("--Lpy", type=float, default=128, help="Dimension Y (cm).")
parser.add_argument("--Lpz", type=float, default=128, help="Dimension Z (cm).")
parser.add_argument("--npx", type=int, default=128, help="Number of voxels (X) Geant4.")
parser.add_argument("--npy", type=int, default=128, help="Number of voxels (Y) Geant4.")
parser.add_argument("--npz", type=int, default=128, help="Number of voxels (Z) Geant4.")
parser.add_argument("--ratio", type=int, default=2, help="SizeVoxelGeant4 / SizeVoxelPOCA: (natural >= 1). Keep in mind that the number of voxels in POCA should be greater than or equal to those in Geant4. The resolution of POCA is the resolution of the image that will be given to the neural network.")
parser.add_argument("--zPosDetector_top", type=float, default=118.0,
    help="Z position of the TOP detector (cm), above the geometry.")
parser.add_argument("--zPosDetector_bot", type=float, default=-118.0,
    help="Z position of the BOTTOM detector (cm), below the geometry.")


parser.add_argument("--word_geometry", type=str, default="MUON", help="Word to be embedded in the geometry. Default is 'MUON'.")
parser.add_argument("--FontSizeX", type=int, default=12, help="Font size in X for the word geometry. Measured in G4 voxels. Valid sizes are 8, 10, 12, 14, 16.")
parser.add_argument("--FontSizeY", type=int, default=12, help="Font size in Y for the word geometry. Measured in G4 voxels. Valid sizes are 8, 10, 12, 14, 16.")
parser.add_argument("--StrokeWidth", type=int, default=1, help="Stroke width for the word geometry. Measured in G4 voxels. Valid sizes are 1, 2, 3.")
parser.add_argument("--spacing", type=int, default=1, help="Spacing between letters in the word geometry. Measured in G4 voxels.")
# EFFICIENT VARIABLE THICKNESS: Instead of repeating word across multiple Z layers,
# use a single slab with configurable thickness via zSizeVoxel. Much more efficient!
parser.add_argument("--depth_z_cm", type=float, default=2.0, help="Word thickness in centimeters. Efficient single slab method (variable zSizeVoxel).")

parser.add_argument("--material", type=str, default="lead", help="Material for the word geometry. Default is 'lead'.")


script_dir = os.path.dirname(os.path.abspath(__file__))
default_json_path = os.path.join(script_dir, "default_geometry.json")

# outputs
parser.add_argument("--output_json", type=str, default=default_json_path, help="Output JSON file name for the Geant4 geometry configuration.")

# plotting
parser.add_argument("--visual_testing_XY_slice", action="store_false",
    help="If True, plots the XY slice of the geometry at the central Z voxel using matplotlib.")

# XY and Z offsets (default 0 = centered). Convention: files without _xoff/_yoff/_zoff in the
# filename have offset=(0,0,0). This keeps all legacy files valid without re-processing.
parser.add_argument("--x_offset_cm", type=float, default=0.0,
    help="X offset of the word center from the world center (cm). Default 0 = centered.")
parser.add_argument("--y_offset_cm", type=float, default=0.0,
    help="Y offset of the word center from the world center (cm). Default 0 = centered.")
parser.add_argument("--z_offset_cm", type=float, default=0.0,
    help="Z position of the slab center (cm). Default 0 = world center.")

# save all the arguments:
args = parser.parse_args()



# ALL-SPACE GEOMETRY CREATION

# ====== POCA ====== 
Lpx = args.Lpx
Lpy = args.Lpy
Lpz = args.Lpz
# resolución de poca: 
npx = args.npx # number of voxels (X)
npy = args.npy # number of voxels (Y)
npz = args.npz # number of voxels (Z)

size_voxel_poca_x = Lpx/npx 
size_voxel_poca_y = Lpy/npy 
size_voxel_poca_z = Lpz/npz 

sizes = [size_voxel_poca_x, size_voxel_poca_y, size_voxel_poca_z]

if all(s % 1 == 0 for s in sizes):
    print("============================================================")
    print("[CORRECT] ----- Valid voxel dimensions (integers)  [CORRECT]")
    print("============================================================")
else:
    sys.exit("[ERROR] ----- Voxel dimensions are not integers. Please adjust Lpx, Lpy, Lpz or npx, npy, npz to ensure integer voxel sizes. Remember that Lpx, Lpy, Lpz should be DIVISIBLE by npx, npy, npz respectively to get integer voxel sizes.")







# ====== GEANT4 ======
# apply same logic for geant4 
# ratio = npx/nx = nyp/ny = npz/nz -> ratio between both resolutions  
# usually, there should be more voxels in poca than in geant4
#                          """"""""""""""""""""""""""""""""""


ratio = args.ratio # must be a natural number >= 1

if npx % ratio != 0 or npy % ratio != 0 or npz % ratio != 0:
    sys.exit(f"[ERROR] ----- ratio ({ratio}) must be an exact divisor of the voxels in POCA ({npx}, {npy}, {npz}). "
             f"Otherwise, the cells in Geant4 and POCA will not align spatially.")

# once tests are passed: 
# numbers of voxels in geant4 (real geometry):
nx = npx // ratio # should be an integer
ny = npy // ratio 
nz = npz // ratio # afterwards, we will remove layers of voxels in Z that are not interesting for the POCA

Lx = Lpx # cm
Ly = Lpy
Lz = Lpz 

SizeG4Voxel_x = Lx/nx
SizeG4Voxel_y = Ly/ny
SizeG4Voxel_z = Lz/nz








######## DEFINING A CUSTOM GEOMETRY ########
# Generation of letters and words --> M, U, O, N --> script bitmaps_letters.py --> fixed matrixes and resolutions


def embed_word_in_geometry_efficient(word_matrix, voxel_list, start_vox,
                                      depth_z_cm, SizeG4Voxel_x, SizeG4Voxel_y, SizeG4Voxel_z,
                                      nx, ny, nz, Lx, Ly, Lz, material, z_offset_cm=0.0):
    """
    EFFICIENT: Create word voxels with variable Z thickness (single slab, not multiple layers).
    Each letter voxel gets thickness=depth_z_cm, centered at z_offset_cm.
    Memory efficient: no layer repetition, just one slab.

    Parameters:
    - start_vox: (x_start, y_start) in G4 voxel units, or None to center in XY.
    - z_offset_cm: Z position of the slab center in cm (default 0 = world center).
    """
    ny_word, nx_word = word_matrix.shape

    # Center in XY if not specified
    if start_vox is None:
        x_start = (nx - nx_word) // 2
        y_start = (ny - ny_word) // 2
    else:
        x_start, y_start = start_vox

    # Verify word fits in XY plane
    if x_start < 0 or y_start < 0 or (x_start + nx_word > nx) or (y_start + ny_word > ny):
        print(f"[ERROR] ----- Palabra ({nx_word}x{ny_word}) "
              f"no cabe en ({nx}x{ny}) en XY "
              f"desde posición ({x_start}, {y_start})")
        sys.exit(1)

    print(f"[INFO] Geometry slab center in Z: z={z_offset_cm:.1f} cm (thickness={depth_z_cm:.1f} cm)")
    print(f"Inserted word ({nx_word}x{ny_word}) with thickness {depth_z_cm:.1f}cm in XY: [{y_start}:{y_start+ny_word}, {x_start}:{x_start+nx_word}]")

    # Create one voxel per active pixel in word_matrix
    for iy in range(ny_word):
        for ix in range(nx_word):
            if word_matrix[iy, ix] == 1:
                global_ix = x_start + ix
                global_iy = y_start + iy

                # Physical coordinates
                x_pos = -Lx/2.0 + (global_ix + 0.5) * SizeG4Voxel_x
                y_pos = -Ly/2.0 + (global_iy + 0.5) * SizeG4Voxel_y
                z_pos = z_offset_cm  # Configurable Z center

                voxel_dict = {
                    "xPosVoxel": float(x_pos),
                    "yPosVoxel": float(y_pos),
                    "zPosVoxel": float(z_pos),
                    "xSizeVoxel": float(SizeG4Voxel_x),
                    "ySizeVoxel": float(SizeG4Voxel_y),
                    "zSizeVoxel": float(depth_z_cm),  # VARIABLE THICKNESS in cm
                    "materialVoxel": material
                }
                voxel_list.append(voxel_dict)

    return voxel_list


def embed_word_in_geometry(word_matrix, boolean_matrix, start_vox: tuple, depth_z: int):
    """
    DEPRECATED: Old inefficient method. Kept for backwards compatibility.
    Use embed_word_in_geometry_efficient instead.
    
    Inserts the word MUON into the real 3D geometry of Geant4, represented as a boolean matrix.

    Parameters:
    - word_matrix: 2D numpy array (ny_word, nx_word) with 1s where the letter is and 0s elsewhere. Shape is (height, width).
    - boolean_matrix: 3D numpy array (ny_world, nx_world, nz_world) representing the Geant4 world. We will modify this in-place to insert the word. Shape is (height, width, depth).
    - start_vox: Tuple (x0, y0, z0) indicating the starting voxel coordinates in the boolean_matrix where the top-left corner of the word will be placed. Coordinates are in the order (X, Y, Z).
    - depth_z: Integer indicating how many voxels in the Z direction the word should occupy (thickness of the word in Z). The word will be extruded in Z for this many voxels, starting from z0.
    """
    # word_matrix.shape es (ny_word, nx_word) = (height, width) = (nrows, ncols)
    ny_word, nx_word = word_matrix.shape
    ny_world, nx_world, nz_world = boolean_matrix.shape # shape is (ny, nx, nz)
    
    if start_vox is None: # Default: insert in the center of the world
        # For even/odd combinations, exact z=0 may not coincide with a voxel center.
        # This choice minimizes the offset of the inserted slab center from z=0.
        z_start = int(np.floor((nz_world - depth_z) / 2.0 + 0.5))
        x_start = (nx_world - nx_word) // 2
        y_start = (ny_world - ny_word) // 2
        start_vox = (x_start, y_start, z_start)
    
    x0, y0, z0 = start_vox


    

    # Debugging: verify that the word fits in the world at the specified location and depth
    if (
        x0 < 0 or y0 < 0 or z0 < 0 or
        (x0 + nx_word > nx_world) or
        (y0 + ny_word > ny_world) or
        (z0 + depth_z > nz_world)
    ):
        print(f"[ERROR] ----- Palabra ({nx_word}x{ny_word}x{depth_z}) "
            f"no cabe en ({nx_world}x{ny_world}x{nz_world}) "
            f"desde posición ({x0}, {y0}, {z0})")
        print("===========================================================")   
        sys.exit(1)

    z_idx_center = z0 + depth_z / 2.0
    z_center_cm = -Lz / 2.0 + z_idx_center * SizeG4Voxel_z
    print(f"[INFO] Geometry slab center in Z: z={z_center_cm:.3f} cm (target z=0)")

    print(f"Inserted word ({nx_word}x{ny_word}) in: [{y0}:{y0+ny_word}, {x0}:{x0+nx_word}, {z0}:{z0+depth_z}]")

    # Insert word into the 3D matrix.
    # `word_matrix` uses (ny, nx) = (height, width), matching boolean_matrix[y, x, z].
    for z in range(z0, z0 + depth_z):
        boolean_matrix[y0 : y0 + ny_word, x0 : x0 + nx_word, z] = word_matrix

    return boolean_matrix



#### EXECUTION OF THE PROGRAM ####

word_matrix, shape_word_YX = get_word(
    word_string=args.word_geometry,
    res_x_list=args.FontSizeX,
    res_y_list=args.FontSizeY,
    stroke_list=args.StrokeWidth,
    spacing=args.spacing
)

if word_matrix is None:
    print("[ERROR] Failed to create word matrix. Check bitmap templates.")
    sys.exit(1)

print("[CORRECT] ----- Word matrix created successfully." )


# Compute XY placement with optional offset (default 0 = centered, compatible with legacy files)
ny_word, nx_word = word_matrix.shape
x_offset_vox = int(round(args.x_offset_cm / SizeG4Voxel_x))
y_offset_vox = int(round(args.y_offset_cm / SizeG4Voxel_y))
x_start = (nx - nx_word) // 2 + x_offset_vox
y_start = (ny - ny_word) // 2 + y_offset_vox

# Use efficient method: create voxels directly with variable Z thickness
voxels_word_list = []
voxels_word_list = embed_word_in_geometry_efficient(
    word_matrix=word_matrix,
    voxel_list=voxels_word_list,
    start_vox=(x_start, y_start),
    depth_z_cm=args.depth_z_cm,
    SizeG4Voxel_x=SizeG4Voxel_x,
    SizeG4Voxel_y=SizeG4Voxel_y,
    SizeG4Voxel_z=SizeG4Voxel_z,
    nx=nx, ny=ny, nz=nz,
    Lx=Lx, Ly=Ly, Lz=Lz,
    material=args.material,
    z_offset_cm=args.z_offset_cm
)

print("[CORRECT] ----- Word voxels created successfully with efficient method (variable Z thickness).")


# Validate material (must match Geant4 DetectorConstruction.cc)
# https://geant4-userdoc.web.cern.ch/UsersGuides/ForApplicationDeveloper/html/Appendix/materialNames.html
possible_materials = ["lead", "air", "iron", "uranium", "aluminium", "argon", "silicon", "steel", "water"]
if args.material not in possible_materials:
    sys.exit(f"[ERROR] Invalid material: '{args.material}'. Allowed: {possible_materials}")








### CREATING THE JSON FILE FOR GEANT4 ###

# loop through each voxel. The one that is filled, we assign all its features (material, size, center) to the dictionary that will be exported to json. The one that is empty, we can ignore it (or assign it as air, depending on how we want to represent the geometry in Geant4). 

global_dictionary = {
    "theWorld": {
    "xSizeWorld": Lx,
    "ySizeWorld": Ly,
    "zSizeWorld": Lz,
    "sizeBoxCRY": Lx,
    "zOffsetCRY": Lz / 2.0,
    },
    "Detectors": [], # fill with detector_dictionaries
    "VoxelConfig": {}, # usually should remain empty
    "TheVoxels": [], # fill with voxel_dictionaries
}


# Add word voxels (created with efficient variable-thickness method)
global_dictionary["TheVoxels"].extend(voxels_word_list)

print(f"[INFO] Added {len(voxels_word_list)} voxels from word geometry (efficient method)")

print("[CORRECT] ----- Voxel dictionaries created successfully." )


# detectors:

global_dictionary["Detectors"] = [ # should be replaced with something more modular, pending
        {
            "xPosDetector": 0,
            "yPosDetector": 0,
            "zPosDetector": args.zPosDetector_top,
            "xDirDetector": 0,
            "yDirDetector": 0,
            "zDirDetector": 0,
            "xSizeDetector": Lx,
            "ySizeDetector": Ly,
            "zSizeDetector": 20,
            "Layers": [
                {
                    "xPosLayer": 0,
                    "yPosLayer": 0,
                    "zPosLayer": 0,
                    "xDirLayer": 0,
                    "yDirLayer": 0,
                    "zDirLayer": 0,
                    "xSizeLayer": Lx,
                    "ySizeLayer": Ly,
                    "zSizeLayer": 1
                },
                {
                    "xPosLayer": 0,
                    "yPosLayer": 0,
                    "zPosLayer": -10,
                    "xDirLayer": 0,
                    "yDirLayer": 0,
                    "zDirLayer": 0,
                    "xSizeLayer": Lx,
                    "ySizeLayer": Ly,
                    "zSizeLayer": 1
                }
            ]
        },
        {
            "xPosDetector": 0,
            "yPosDetector": 0,
            "zPosDetector": args.zPosDetector_bot,
            "xDirDetector": 0,
            "yDirDetector": 0,
            "zDirDetector": 0,
            "xSizeDetector": Lx,
            "ySizeDetector": Ly,
            "zSizeDetector": 20,
            "Layers": [
                {
                    "xPosLayer": 0,
                    "yPosLayer": 0,
                    "zPosLayer": 0,
                    "xDirLayer": 0,
                    "yDirLayer": 0,
                    "zDirLayer": 0,
                    "xSizeLayer": Lx,
                    "ySizeLayer": Ly,
                    "zSizeLayer": 1
                },
                {
                    "xPosLayer": 0,
                    "yPosLayer": 0,
                    "zPosLayer": 10,
                    "xDirLayer": 0,
                    "yDirLayer": 0,
                    "zDirLayer": 0,
                    "xSizeLayer": Lx,
                    "ySizeLayer": Ly,
                    "zSizeLayer": 1
                }
            ]
        }
]


print("[CORRECT] ----- Full json file information created successfully." )

# translate global_dictionary to a full json file
with open(args.output_json, 'w') as f:
    json.dump(global_dictionary, f, indent=4)

print("[CORRECT] ----- Json file created successfully, available at: " + args.output_json )


# XY footprint used for visual testing
xy_gt = np.zeros((ny, nx), dtype=int)
xy_gt[y_start:y_start + ny_word, x_start:x_start + nx_word] = word_matrix




######################
### VISUAL TESTING ###
######################
if args.visual_testing_XY_slice:
    print("[INFO] ----- Visual testing enabled. Plotting XY footprint of the word geometry...")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(
        xy_gt,
        origin="lower",
        cmap="Greys",
        interpolation="nearest",
        extent=[-Lx/2, Lx/2, -Ly/2, Ly/2]
    )
    ax.set_title(
        f"XY footprint | slab center z={args.z_offset_cm:.1f} cm, depth={args.depth_z_cm:.1f} cm\n"
        f"Word: '{args.word_geometry}' | FontSize: {args.FontSizeX}x{args.FontSizeY} | "
        f"Stroke: {args.StrokeWidth} | G4 voxel size: {SizeG4Voxel_x:.1f} cm"
    )
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")

    material_patch = mpatches.Patch(color="black", label=args.material)
    air_patch = mpatches.Patch(color="white", label="air")
    ax.legend(handles=[material_patch, air_patch], loc="upper right")

    plt.tight_layout()
    plot_path = os.path.join(script_dir, "geometry_xy_slice.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[CORRECT] ----- Visualization saved at: {plot_path}")
    plt.close()
else:
    print("[INFO] ----- Visual testing disabled. To enable, use the flag --visual_testing_XY_slice when running the script.")