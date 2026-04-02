import json
import math 
import sys
import numpy as np
import pandas as pd
import argparse
from bitmaps_letters import get_word, get_letter, dimensions_test

# Debugging: 
dimensions_test() # should stop the whole program if the dimensions are not correct for the defined font sizes and stroke widths. This is crucial to ensure that the generated word matrices fit properly in the Geant4 geometry.


parser = argparse.ArgumentParser(description="Creation of geometry from POCA resolution ---> key parameter = ratio.")
parser.add_argument("--Lpx", type=float, default=512.0, help="Dimension X (cm).")
parser.add_argument("--Lpy", type=float, default=512.0, help="Dimension Y (cm).")
parser.add_argument("--Lpz", type=float, default=512.0, help="Dimension Z (cm).")
parser.add_argument("--npx", type=int, default=128, help="Number of voxels (X) Geant4.")
parser.add_argument("--npy", type=int, default=128, help="NNumber of voxels (Y) Geant4.")
parser.add_argument("--npz", type=int, default=64, help="NNumber of voxels (Z) Geant4.")
parser.add_argument("--ratio", type=int, default=2, help="SizeVoxelGeant4 / SizeVoxelPOCA: (natural >= 1). Keep in mind that the number of voxels in POCA should be greater than or equal to those in Geant4. The resolution of POCA is the resolution of the image that will be given to the neural network.")

parser.add_argument("--word_geometry", type=str, default="MUON", help="Word to be embedded in the geometry. Default is 'MUON'.")
parser.add_argument("--FontSizeX", type=int, default=16, help="Font size in X for the word geometry. Measured in G4 voxels. Valid sizes are 8, 10, 12, 14, 16.")
parser.add_argument("--FontSizeY", type=int, default=16, help="Font size in Y for the word geometry. Measured in G4 voxels. Valid sizes are 8, 10, 12, 14, 16.")
parser.add_argument("--StrokeWidth", type=int, default=3, help="Stroke width for the word geometry. Measured in G4 voxels. Valid sizes are 1, 2, 3.")
parser.add_argument("--spacing", type=int, default=2, help="Spacing between letters in the word geometry. Measured in G4 voxels. Valid sizes are 0, 1, 2, 3.")

parser.add_argument("--material", type=str, default="lead", help="Material for the word geometry. Default is 'lead'.")


parser.add_argument("--output_ground_truth_density", type=str, default="ground_truth_density.npy", help="Output npy file (Tensor) for the ground truth density of the geometry, assigns the geometry of the material for each of the voxels (0 for air).")
parser.add_argument("--output_json", type=str, default="geometry.json", help="Output JSON file name for the Geant4 geometry configuration.")

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


size_vec = np.array(sizes)
left_down_corner = np.array([-Lpx/2.0, -Lpy/2.0, -Lpz/2.0])
first_voxel = left_down_corner + size_vec/2.0


voxels_centers = []
test_print = False
for ix in range(npx): 
    for iy in range(npy): 
        for iz in range(npz): 
            desplazamiento = np.array([ix * size_vec[0], iy * size_vec[1], iz * size_vec[2]])
            voxel_center = first_voxel + desplazamiento
            voxels_centers.append(voxel_center)
            if test_print: # debugging
                print(f"Voxel (POCA) center at: {voxel_center}")
            




# ====== GEANT4 ======
# apply same logic for geant4 
# ratio = nxp/nx = nyp/ny = npz/nz -> ratio between both resolutions  
# usually, there should be more voxels in poca than in geant4
#                          """"""""""""""""""""""""""""""""""


ratio: int = args.ratio # must be a natural number >= 1
# --- DEBUGGING --- 
if ratio < 1 or not isinstance(ratio, int):
    sys.exit("[ERROR] ----- Ratio must be a natural number greater than or equal to 1.")
elif ratio != npx/nx or ratio != npy/ny or ratio != npz/nz:
    sys.exit("[ERROR] ----- Ratio does not match the number of voxels in POCA and Geant4.")
# ---           ---    


if npx % ratio != 0 or npy % ratio != 0 or npz % ratio != 0:
    sys.exit(f"[ERROR] ----- ratio ({ratio}) must be an exact divisor of the voxels in POCA ({npx}, {npy}, {npz}). "
             f"Otherwise, the cells in Geant4 and POCA will not align spatially.")

# once tests are passed: 
nx = npx // ratio # should be an integer
ny = npy // ratio 
nz = npz // ratio

Lx = Lpx # cm
Ly = Lpy
Lz = Lpz

SizeG4Voxel_x = Lx/nx
SizeG4Voxel_y = Ly/ny
SizeG4Voxel_z = Lz/nz




voxel_centers_G4 = []
MatrixGeometryMaterials = np.zeros((nx, ny, nz), dtype=object)  # This will hold the material of each voxel
MatrixGeometryBoolean = np.zeros((nx, ny, nz), dtype=int)  # This will hold 1 for filled voxels and 0 for empty voxels
MatrixGeometryIndex = np.zeros((nx, ny, nz), dtype=object)  # This will hold the index of each voxel (for reference)
MatrixGeometryCenter = np.zeros((nx, ny, nz, 3))  # This will hold the center coordinates of each voxel
MatrixGeometryDensity = np.zeros((nx, ny, nz))  # This will hold the density of each voxel (for reference)

for ix in range(nx):
    for iy in range(ny):
        for iz in range(nz):
            voxel_center_g4 = np.array([-Lx/2.0 + (ix + 0.5) * SizeG4Voxel_x, 
                                        -Ly/2.0 + (iy + 0.5) * SizeG4Voxel_y, 
                                        -Lz/2.0 + (iz + 0.5) * SizeG4Voxel_z])
            voxel_centers_G4.append(voxel_center_g4)
            MatrixGeometryCenter[ix, iy, iz] = voxel_center_g4
            if test_print: # debugging
                print(f"Voxel (GEANT4) center at: {voxel_center_g4}")




######## DEFINING A CUSTOM GEOMETRY ########
# Generation of letters and words --> M, U, O, N --> script bitmaps_letters.py --> fixed matrixes and resolutions


def embed_word_in_geometry(word_matrix, boolean_matrix, start_vox: tuple, depth_z: int):
    """
    Inserts the word MUON into the real 3D geometry of Geant4, represented as a boolean matrix.    
    
    Parameters:
    - word_matrix: 2D numpy array (ny_word, nx_word) with 1s where the letter is and 0s elsewhere. Shape is (height, width).
    - boolean_matrix: 3D numpy array (nx_world, ny_world, nz_world) representing the Geant4 world. We will modify this in-place to insert the word.
    - start_vox: Tuple (x0, y0, z0) indicating the starting voxel coordinates in the boolean_matrix where the top-left corner of the word will be placed. Coordinates are in the order (X, Y, Z) corresponding to (width, height, depth) of the world.
    - depth_z: Integer indicating how many voxels in the Z direction the word should occupy (thickness of the word in Z). The word will be extruded in Z for this many voxels, starting from z0.
    """
    # word_matrix.shape es (ny_word, nx_word) -> (height, width)
    ny_word, nx_word = word_matrix.shape 
    nx_world, ny_world, nz_world = boolean_matrix.shape # careful interpreting dimensions
    
    if start_vox == None: # Default: insert in the center of the world
        z_start = (nz_world // 2) - 2
        x_start = (nx_world // 2) - (nx_word // 2)
        y_start = (ny_world // 2) - (ny_word // 2)
        start_vox = (x_start, y_start, z_start)  
    
    x0, y0, z0 = start_vox


    

    # Debugging: verify that the word fits in the world at the specified location and depth
    if (x0 + nx_word > nx_world) or (y0 + ny_word > ny_world) or (z0 + depth_z > nz_world):
        print(f"Error de límites: Palabra {nx_word}x{ny_word} se sale de {nx_world}x{ny_world}")
        sys.exit(1)

    # NumPy counts rows from top to bottom, but the physics (Y) counts from bottom to top.
    # By flipping vertically, we map the coordinates correctly.
    word_flipped = np.flipud(word_matrix)

    print(f"Inserted word ({nx_word}x{ny_word}) in: [{x0}:{x0+nx_word}, {y0}:{y0+ny_word}, {z0}:{z0+depth_z}]")

    # Inserción mediante slicing
    # Usamos transpuesta (.T) para mapear (ny, nx) -> (nx, ny) del mundo
    # boolean_matrix[X, Y, Z]
    for z in range(z0, z0 + depth_z):
        boolean_matrix[x0 : x0 + nx_word, y0 : y0 + ny_word, z] = word_flipped.T


    
    return boolean_matrix



#### EXECUTION OF THE PROGRAM ####

word_matrix, shape_word_YX = get_word(word=args.word_geometry, font_size_x=args.FontSizeX, font_size_y=args.FontSizeY, stroke_width=args.StrokeWidth, spacing=args.spacing) 


 # 3. Insertar la palabra en el centro del mundo con un grosor de 5 vóxeles en Z
# Calculamos posición central


MatrixGeometryBoolean = embed_word_in_geometry(
    word_matrix = word_matrix,
    boolean_matrix = MatrixGeometryBoolean,
    start_vox = None, # Default: insert in the center of the world
    depth_z = 5
)



MatrixGeometryMaterials[MatrixGeometryBoolean == 1] = args.material
MatrixGeometryMaterials[MatrixGeometryBoolean == 0] = "air"


possible_materials = ["lead", "air"]

# ---- Debugging ---- 
# verify that only allowed materials are present in the geometry
for element in np.unique(MatrixGeometryMaterials):
    if element not in possible_materials:
        sys.exit(f"[ERROR] ----- Invalid material found in MatrixGeometryMaterials. Allowed materials are: {possible_materials}")
# ----           ----


density_dictionary = {"lead": 1, "air": 0} # not realistic, just for testing purposes, SHOULD BE CHANGED!!
MatrixGeometryDensity = density_dictionary[args.material] * MatrixGeometryBoolean # assign density based on the material of each voxel
np.save(args.output_ground_truth_density, MatrixGeometryDensity)


### CREATING THE JSON FILE FOR GEANT4 ###
# print para los detectores: 
nDetectors = 2
nLayers = 4

 
# #We take the structure from this basic json file and adapt the dictionary
# with open('../data/confExample.json', 'r') as f:
#     data_ = json.load(f)

# theWorld = data_['theWorld']
# detector = data_['Detectors'][0]
# layer = detector['Layers'][0]
# sensor = layer['Sensors'][0]

# sensors = []
# for isensor in range(0, nSensors):
#     copysens = sensor.copy()
#     sensors.append(copysens)
# layer['Sensors'] = sensors

# layers = []
# for ilayer in range(0, nLayers):
#     copylayer = layer.copy()
#     layers.append(copylayer)
# detector['Layers'] = layers

# detectors = []
# for idetector in range(0, nDetectors):
#     copydetector = detector.copy()
#     detectors.append(copydetector)


# data = {} 
# data['theWorld'] = theWorld
# data['Detectors'] = detectors




# stepxp = stepx * ratio 
# stepyp = stepy * ratio 
# stepzp = stepy * ratio 

# nxp = math.floor(nx / ratio) 
# nyp = math.floor(ny / ratio) 
# nzp = math.floor(nz / ratio) 

# print('Separacion---------------------------') 

# for ix in range(nxp): 
# for iy in range(nyp): 
# for iz in range(nzp): 
# print('There is a voxel at', -Lx/2.0 + ix * stepxp, -Ly/2.0 + iy * stepyp, -Lz/2.0 + iz * stepzp)