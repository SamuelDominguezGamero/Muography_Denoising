import json
import math 




############################################################################################################### 
############################################################################################################### 
# Lx, Ly, Lz, nx, ny, nz hacen referencia al plot de POCA en general con más voxeles 
# ratio es el cociente entre el tamaño del voxel de geant4 entre el de poca, siempre un número natural >= 1 
############################################################################################################### 
############################################################################################################### 

###  Disertación de ideas iniciales: 
# conviene que el tamaño del voxel de poca sea un múltiplo del tamaño del voxel de geant4
# 

# === POCA === 
Lpx = Lpy = Lpz = 200 # medidas del mundo de POCA en cm
# resolución de poca: 
npx = 20 # número de voxeles eje X
npy = 20 # número de voxeles eje Y
npz = 20 # número de voxeles eje Z
 
tamaño_voxel_poca_x = Lpx/npx # tamaño del voxel de poca en cm
tamaño_voxel_poca_y = Lpy/npy # tamaño del voxel de poca en cm
tamaño_voxel_poca_z = Lpz/npz # tamaño del voxel de poca en cm


# lista de voxeles de poca:
for ix in range(npx): 
    for iy in range(npy): 
        for iz in range(npz): 
            print('There is a voxel at', -Lpx/2.0 + ix * tamaño_voxel_poca_x, -Lpy/2.0 + iy * tamaño_voxel_poca_y, -Lpz/2.0 + iz * tamaño_voxel_poca_z) 







# === GEANT4 ===




# print para los detectores: 
nDetectors = 2
nLayers = 4

 
#We take the structure from this basic json file and adapt the dictionary
with open('../data/confExample.json', 'r') as f:
data_ = json.load(f)

theWorld = data_['theWorld']
detector = data_['Detectors'][0]
layer = detector['Layers'][0]
sensor = layer['Sensors'][0]

sensors = []
for isensor in range(0, nSensors):
    copysens = sensor.copy()
    sensors.append(copysens)
layer['Sensors'] = sensors

layers = []
for ilayer in range(0, nLayers):
    copylayer = layer.copy()
    layers.append(copylayer)
detector['Layers'] = layers

detectors = []
for idetector in range(0, nDetectors):
    copydetector = detector.copy()
    detectors.append(copydetector)


data = {} 
data['theWorld'] = theWorld
data['Detectors'] = detectors




stepxp = stepx * ratio 
stepyp = stepy * ratio 
stepzp = stepy * ratio 

nxp = math.floor(nx / ratio) 
nyp = math.floor(ny / ratio) 
nzp = math.floor(nz / ratio) 

print('Separacion---------------------------') 

for ix in range(nxp): 
for iy in range(nyp): 
for iz in range(nzp): 
print('There is a voxel at', -Lx/2.0 + ix * stepxp, -Ly/2.0 + iy * stepyp, -Lz/2.0 + iz * stepzp)