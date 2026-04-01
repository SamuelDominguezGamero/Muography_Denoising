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

###  Disertación de ideas iniciales: 
# conviene que el tamaño del voxel de poca sea un múltiplo del tamaño del voxel de geant4


# === POCA === 
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
test_print = True
for ix in range(npx): 
    for iy in range(npy): 
        for iz in range(npz): 
            desplazamiento = np.array([ix * size_vec[0], iy * size_vec[1], iz * size_vec[2]])
            voxel_center = first_voxel + desplazamiento
            voxels_centers.append(voxel_center)
            print(f"Voxel center at: {voxel_center}")
            
# CORRECT: voxels centers for POCA are successfully calculated and printed



# === GEANT4 ===
# apply same logic for geant4, 
# there has to be a ratio betwween number of voxels in each geomtetry

nx
ny
nz

Lx
Ly
Lz 

if npx % nLayers != 0 or npy % nLayers != 0 or npz % nLayers != 0:
    sys.exit("Error: The number of voxels in POCA must be divisible by the number of layers in Geant4.")





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
p

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