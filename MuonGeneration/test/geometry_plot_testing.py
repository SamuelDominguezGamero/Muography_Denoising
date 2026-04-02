import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection

# ==========================================
# PARTE 1: Configuración de Geometrías (Tu código optimizado)
# ==========================================

# === POCA === 
Lpx = Lpy = Lpz = 200.0 # Dimensiones del mundo POCA en cm
npx = npy = npz = 4     # Número de vóxeles (POCA)

sizes_poca = np.array([Lpx/npx, Lpy/npy, Lpz/npz])

# Validación simple
if not np.all(sizes_poca % 1 == 0):
    sys.exit("Error: Las dimensiones del vóxel POCA deben ser enteros.")

print(f"Dimensiones Vóxel POCA: {sizes_poca} cm")

# === GEANT4 ===
ratio: int = 2  # Relación nxpoca / nxgeant4

if ratio < 1 or not isinstance(ratio, int):
    sys.exit("Error: Ratio debe ser un número natural >= 1.")

nx = math.floor(npx / ratio) 
ny = math.floor(npy / ratio)
nz = math.floor(npz / ratio)

# Validar que el ratio sea exacto
if ratio != npx/nx or ratio != npy/ny or ratio != npz/nz:
    sys.exit(f"Error: El ratio {ratio} no es compatible con las subdivisiones ({npx}/{nx}).")

Lx, Ly, Lz = Lpx, Lpy, Lpz # Mismo mundo
sizes_g4 = np.array([Lx/nx, Ly/ny, Lz/nz])

print(f"Dimensiones Vóxel Geant4: {sizes_g4} cm")
print(f"Subdivisiones Geant4: [{nx}, {ny}, {nz}]")


# ==========================================
# PARTE 2: Generación de Coordenadas de Aristas (Paredes)
# ==========================================

def get_grid_lines_2d_plane(L, n, size_voxel, plane='xy', offset=0.0):
    """
    Genera las líneas de una cuadrícula 2D en un plano 3D específico.
    L: Dimensiones del mundo [Lx, Ly] en el plano
    n: Número de vóxeles [nx, ny] en el plano
    """
    lines = []
    
    # Bordes en el primer eje (ej. líneas constantes en X)
    # n[0]+1 líneas para cerrar la cuadrícula
    x_coords = np.linspace(-L[0]/2.0, L[0]/2.0, n[0]+1)
    y_range = np.array([-L[1]/2.0, L[1]/2.0])
    
    for x in x_coords:
        if plane == 'xy':
            lines.append([(x, y_range[0], offset), (x, y_range[1], offset)])
        elif plane == 'xz':
            lines.append([(x, offset, y_range[0]), (x, offset, y_range[1])])
            
    # Bordes en el segundo eje (ej. líneas constantes en Y)
    y_coords = np.linspace(-L[1]/2.0, L[1]/2.0, n[1]+1)
    x_range = np.array([-L[0]/2.0, L[0]/2.0])
    
    for y in y_coords:
        if plane == 'xy':
            lines.append([(x_range[0], y, offset), (x_range[1], y, offset)])
        elif plane == 'xz':
            lines.append([(offset, x_range[0], y), (offset, x_range[1], y)])
            
    return lines

# Definimos el plano de visualización (ej. un corte en Z=0)
z_slice_offset = 0.0

# Obtener líneas para POCA (Rojo)
lines_poca = get_grid_lines_2d_plane([Lpx, Lpy], [npx, npy], sizes_poca, plane='xy', offset=z_slice_offset)

# Obtener líneas para Geant4 (Azul)
lines_g4 = get_grid_lines_2d_plane([Lx, Ly], [nx, ny], sizes_g4, plane='xy', offset=z_slice_offset)


# ==========================================
# PARTE 3: Visualización Matplotlib 3D
# ==========================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 1. Dibujar la Malla POCA (Línea fina roja)
# Usamos Line3DCollection para eficiencia
from mpl_toolkits.mplot3d.art3d import Line3DCollection

lc_poca = Line3DCollection(lines_poca, colors='red', linewidths=0.5, label='POCA Grid')
ax.add_collection3d(lc_poca)

# 2. Dibujar la Malla GEANT4 (Línea gruesa azul, ligeramente desplazada en Z para visibilidad)
# Un pequeño desplazamiento 'epsilon' evita problemas de renderizado (z-fighting)
epsilon = 0.1 
lines_g4_offset = [[(p[0], p[1], p[2] + epsilon) for p in line] for line in lines_g4]

lc_g4 = Line3DCollection(lines_g4_offset, colors='blue', linewidths=2.5, label='Geant4 Grid')
ax.add_collection3d(lc_g4)


# Configuración de los ejes
ax.set_xlim(-Lpx/2.0, Lpx/2.0)
ax.set_ylim(-Lpy/2.0, Lpy/2.0)
ax.set_zlim(-Lpz/2.0, Lpz/2.0)

ax.set_xlabel('X (cm)')
ax.set_ylabel('Y (cm)')
ax.set_zlabel('Z (cm)')

ax.set_title(f'Superposición de Mallas (Ratio={ratio})\nCorte en Z={z_slice_offset}')

# Leyenda personalizada (Line3DCollection no auto-añade leyenda fácilmente)
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='red', lw=0.5),
                Line2D([0], [0], color='blue', lw=2.5)]
ax.legend(custom_lines, ['POCA (Fina)', 'Geant4 (Gruesa)'])

plt.show()