import json
import math 
import sys
import numpy as np
import pandas as pd

############################################################################################################### 
############################################################################################################### 
# Lx, Ly, Lz, nx, ny, nz hacen referencia al plot de POCA en general con más voxeles 
# ratio es el cociente entre el tamaño del voxel de geant4 entre el de poca, siempre un número natural >= 1 
############################################################################################################### 
############################################################################################################### 


# VOXELIZACIÓN DE TODO EL ESPACIO

# ====== POCA ====== 
Lpx = Lpy = Lpz = 512 # POCA world dimensions in cm
# resolución de poca: 
npx = 256 # number of voxels (X)
npy = 256 # number of voxels (Y)
npz = 128 # number of voxels (Z)

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
    sys.exit("Error: Voxel dimensions are not integers. Please adjust Lpx, Lpy, Lpz or npx, npy, npz to ensure integer voxel sizes.")


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
MatrixGeometryMaterials = np.zeros((nx, ny, nz), dtype=object)  # This will hold the material of each voxel
MatrixGeometryBoolean = np.zeros((nx, ny, nz), dtype=int)  # This will hold 1 for filled voxels and 0 for empty voxels
MatrixGeometryIndex = np.zeros((nx, ny, nz), dtype=object)  # This will hold the index of each voxel (for reference)
MatrixGeometryCenter = np.zeros((nx, ny, nz, 3))  # This will hold the center coordinates of each voxel

for ix in range(nx):
    for iy in range(ny):
        for iz in range(nz):
            voxel_center_g4 = np.array([-Lx/2.0 + (ix + 0.5) * SizeG4Voxel_x, 
                                        -Ly/2.0 + (iy + 0.5) * SizeG4Voxel_y, 
                                        -Lz/2.0 + (iz + 0.5) * SizeG4Voxel_z])
            voxel_centers_G4.append(voxel_center_g4)
            MatrixGeometryCenter[ix, iy, iz] = voxel_center_g4
            if test_print:
                print(f"Voxel (GEANT4) center at: {voxel_center_g4}")

print("MatrixGeometryCenter:")
print(MatrixGeometryCenter)




######## DEFINING A CUSTOM GEOMETRY ########

#  need to define a nx · ny · nz boolean tensor, where 1 means filled voxel, and 0 means empty voxel (air).
# another tensor (same dimensions) contains the material of each voxel: "lead", "air", "iron", etc.

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


## insert the word MUON into the real geometry:

def embed_word_in_geometry(word_matrix, boolean_matrix, start_vox: tuple, depth_z: int):
    """
    Inserta la matriz 2D en el volumen 3D asegurando orientación cartesiana correcta.
    """
    # word_matrix.shape es (ny_word, nx_word) -> (alto, ancho)
    ny_word, nx_word = word_matrix.shape 
    nx_world, ny_world, nz_world = boolean_matrix.shape
    
    x0, y0, z0 = start_vox

    # 1. Verificación de límites
    if (x0 + nx_word > nx_world) or (y0 + ny_word > ny_world) or (z0 + depth_z > nz_world):
        print(f"Error de límites: Palabra {nx_word}x{ny_word} se sale de {nx_world}x{ny_world}")
        sys.exit(1)

    # 2. CORRECCIÓN CLAVE: Inversión vertical para Geant4/Cartesiano
    # NumPy cuenta filas de arriba-abajo, pero la física (Y) de abajo-arriba.
    # Al flipear verticalmente, mapeamos correctamente las coordenadas.
    word_flipped = np.flipud(word_matrix)

    print(f"Insertando word ({nx_word}x{ny_word}) en: [{x0}:{x0+nx_word}, {y0}:{y0+ny_word}, {z0}:{z0+depth_z}]")

    # 3. Inserción mediante slicing
    # Usamos transpuesta (.T) para mapear (ny, nx) -> (nx, ny) de tu mundo
    # boolean_matrix[X, Y, Z]
    for z in range(z0, z0 + depth_z):
        boolean_matrix[x0 : x0 + nx_word, y0 : y0 + ny_word, z] = word_flipped.T
        
    return boolean_matrix


# /// aplicación de ejemplo ///


# 2. Generar la palabra con resoluciones mixtas
MOUN_2D, grid_size = get_word('MUON', 16, 16, 3, spacing=2) # Ejemplo, sintaxis = palabra, resol_x, resol_y, stroke, spacing

 
# 3. Insertar la palabra en el centro del mundo con un grosor de 5 vóxeles en Z
# Calculamos posición central
z_start = (nz // 2) - 2
x_start = (nx // 2) - (grid_size[1] // 2)
y_start = (ny // 2) - (grid_size[0] // 2)

MatrixGeometryBoolean = embed_word_in_geometry(
    word_matrix = MOUN_2D,
    boolean_matrix = MatrixGeometryBoolean,
    start_vox = (x_start, y_start, z_start),
    depth_z = 5
)

# 4. Asignar material (Plomo) solo donde Boolean == 1
# Esto es mucho más eficiente que iterar
MatrixGeometryMaterials[MatrixGeometryBoolean == 1] = "G4_Pb"
MatrixGeometryMaterials[MatrixGeometryBoolean == 0] = "G4_AIR"


# print("MatrixGeometryBoolean (slices):")
# print(MatrixGeometryBoolean[:, :, z_start])  # Slice central en Z
# print(MatrixGeometryBoolean[:, :, z_start + 1])  # Slice siguiente en Z

# print("MatrixGeometryMaterials (slices):")
# print(MatrixGeometryMaterials[:, :, z_start])  # Slice central en Z
# print(MatrixGeometryMaterials[:, :, z_start + 1])  # Slice siguiente en Z


plot_geometry = True
if plot_geometry:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    
    # 1. Extraemos el plano XY del mundo Geant4 [X, Y, nz]
    slice_world_xy = MatrixGeometryBoolean[:, :, z_start]
    
    # 2. Trasponemos para visualización estándar de imshow [X, Y] -> [Y, X]
    # (imshow requiere filas, columnas)
    slice_plot = slice_world_xy.T
    
    # 3. Dibujamos con origin='lower' para que el voxel [0,0] esté abajo.
    # Esto asegura que el eje X sea horizontal y crezca a la derecha,
    # y el eje Y sea vertical y crezca hacia arriba (como en la física).
    plt.imshow(slice_plot, cmap='binary', interpolation='nearest', origin='lower')
    
    # Etiquetas cartesianas
    plt.xlabel("Eje X (Voxels) - Ancho Mundo")
    plt.ylabel("Eje Y (Voxels) - Alto Mundo")
    plt.title(f"Corte Geometría (Slice Z={z_start}) - 'MUON' Derecha")
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.show()


def verificar_geometria_csv(boolean_matrix, z_slice, filename="verificacion_geometria.csv"):
    """
    Extrae un plano XY y lo guarda en un CSV para inspección manual.
    """
    # 1. Extraemos el plano (X, Y)
    # slice_2d.shape -> (nx, ny)
    slice_2d = boolean_matrix[:, :, z_slice]
    
    # 2. Para que sea legible en un editor de texto (donde las filas son Y):
    # Trasponemos para que las columnas del CSV sean el eje X 
    # y las filas del CSV sean el eje Y.
    # Usamos flipud para que la coordenada Y más alta esté arriba en el archivo.
    df = pd.DataFrame(np.flipud(slice_2d.T))
    
    # 3. Guardar sin índices para limpieza
    df.to_csv(filename, index=False, header=False)
    print(f"Archivo de verificación guardado en: {filename}")

# --- USO ---
# Suponiendo que ya ejecutaste embed_word_in_geometry
verificar_geometria_csv(MatrixGeometryBoolean, z_start)


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