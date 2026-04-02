import json
import math 
import sys
import numpy as np


############################################################################################################### 
############################################################################################################### 
# Lx, Ly, Lz, nx, ny, nz hacen referencia al plot de POCA en general con más voxeles 
# ratio es el cociente entre el tamaño del voxel de geant4 entre el de poca, siempre un número natural >= 1 
############################################################################################################### 
############################################################################################################### 


# VOXELIZACIÓN DE TODO EL ESPACIO

# ====== POCA ====== 
Lpx = Lpy = Lpz = 200 # POCA world dimensions in cm
# resolución de poca: 
npx = 4 # number of voxels (X)
npy = 4 # number of voxels (Y)
npz = 4 # number of voxels (Z)

size_voxel_poca_x = Lpx/npx 
size_voxel_poca_y = Lpy/npy 
size_voxel_poca_z = Lpz/npz 
sizes = [size_voxel_poca_x, size_voxel_poca_y, size_voxel_poca_z]

if all(s % 1 == 0 for s in sizes):
    print("=================================")
    print("=================================")
    print("Valid voxel dimensions (integers)")
    print("=================================")
    print("=================================")
else:
    sys.exit()


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
            if test_print:
                print(f"Voxel (POCA) center at: {voxel_center}")
            
# CORRECT: voxels centers for POCA are successfully calculated and printed



# ====== GEANT4 ======
# apply same logic for geant4 
# ratio = nxp/nx = nyp/ny = npz/nz -> ratio between both resolutions  
# usually, there should be more voxels in poca than in geant4
#                          """"""""""""""""""""""""""""""""""


ratio: int = 2 # must be a natural number >= 1


nx = math.floor(npx / ratio) 
ny = math.floor(npy / ratio)
nz = math.floor(npz / ratio)

Lx = Lpx
Ly = Lpy
Lz = Lpz

SizeG4Voxel_x = Lx/nx
SizeG4Voxel_y = Ly/ny
SizeG4Voxel_z = Lz/nz

if ratio < 1 or not isinstance(ratio, int):
    sys.exit("Error: Ratio must be a natural number greater than or equal to 1.")
elif ratio != npx/nx or ratio != npy/ny or ratio != npz/nz:
    sys.exit("Error: Ratio does not match the number of voxels in POCA and Geant4.")


voxel_centers_G4 = []

for ix in range(nx):
    for iy in range(ny):
        for iz in range(nz):
            voxel_center_g4 = np.array([-Lx/2.0 + (ix + 0.5) * SizeG4Voxel_x, 
                                        -Ly/2.0 + (iy + 0.5) * SizeG4Voxel_y, 
                                        -Lz/2.0 + (iz + 0.5) * SizeG4Voxel_z])
            voxel_centers_G4.append(voxel_center_g4)
            if test_print:
                print(f"Voxel (GEANT4) center at: {voxel_center_g4}")




######## DEFINING A CUSTOM GEOMETRY

#  need to define a nx · ny · nz boolean tensor, where 1 means filled voxel, and 0 means empty voxel (air).
# another tensor (same dimensions) contains the material of each voxel: "lead", "air", "iron", etc.

# Voxelized Phantom: few filled voxels, and the rest will be empty (air)
# For example, we can create a 4x4x4 grid where the central 2x2x2 block is filled with "lead" and the rest is "air".

def create_phantom_rectangle(nx, ny, nz, material_filled="lead", material_empty="air", ratio=2, verbose=True):
    phantom = np.full((nx, ny, nz), material_empty, dtype=object)  # Start with all voxels as empty
    
    # filled voxels: central 2x2x2 block
    x_filled = slice(nx//2 - 1, nx//2 + 1)  # Central 2 voxels in X
    y_filled = slice(ny//2 - 1, ny//2 + 1)  # Central 2 voxels in Y
    z_filled = slice(nz//2 - 1, nz//2 + 1)  # Central 2 voxels in Z

    phantom[x_filled, y_filled, z_filled] = material_filled  # Fill the central block with the specified material
    
    if verbose:
        print("Phantom structure (4x4x4):")
        print(phantom)
    
    return phantom

phantom_test = create_phantom_rectangle(4, 4, 4, verbose=False)


# Now, we define labels: X -> Y -> Z is the standard order
# we used a nested loop in the following order: for ix in range(nx): for iy in range(ny): for iz in range(nz): ----> WE MUST STICK TO THIS ORDER ALWAYS 

# Generation of letters  -> MUON --> script bitmaps_letters.py


# ///////////////////////// test (begin) /////////////////////////
from bitmaps_letters import get_letter, get_word, dimensions_test

dimensions_test() 
do_print_letters = False
plot_letters = False
resol = 16
M  = get_letter(resol, resol, 'M', stroke=1)
MOUN, shape = get_word('MOUN', [16,10,16,16], [16,10,16,16], [3,2,3,3], spacing=4)
print("Shape of MOUN:", shape)
if do_print_letters:
    print(M)
    print(MOUN)
if plot_letters:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 3)) # Ajustado para palabras horizontales
    plt.imshow(MOUN, cmap='binary', interpolation='nearest')
    plt.show()
# ///////////////////////// test (final) /////////////////////////






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