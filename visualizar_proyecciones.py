"""
Receives either a .root file or a .npy file 
"""

import argparse
import ROOT
import numpy as np
import scienceplots
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import block_reduce

# PARSER
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--xSizeWorld", default=128)
parser.add_argument("--ySizeWorld", default=128)
parser.add_argument("--zSizeWorld", default=128)
parser.add_argument("--resolution", default=128)
args = parser.parse_args()


# Carga de datos
file = Path(args.input)
extension = file.suffix.lower()
binning = int(args.resolution)
xSizeWorld = args.xSizeWorld
ySizeWorld = args.ySizeWorld
zSizeWorld = args.zSizeWorld

# ==========================================
# CONFIGURACIÓN DE ESTILO ACADÉMICO (LaTeX)
# ==========================================
plt.style.use('default')
plt.style.use(['science'])
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,      # Habilita renderizado real de LaTeX (requiere LaTeX instalado en el sistema)
    "font.size": 11,
    "axes.titlesize": 11,     # Mismo tamaño para homogeneidad o 12 si prefieres destacar ligeramente
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,        # Calidad de publicación médica/científica
    "axes.grid": False        # <-- APAGA LA REJILLA PRINCIPAL
    })



##################################################################
# Version 1: la entrada es un .root (hace las proyecciones y todo)
##################################################################

if extension == ".root":
    df = ROOT.RDataFrame("events", args.input)
    df = df.Filter("abs(theta) > 0.0001")
    res = df.AsNumpy(columns=["poca_x", "poca_y", "poca_z", "theta"])

    x = res["poca_x"]
    y = res["poca_y"]
    z = res["poca_z"]
    theta = res["theta"]

    print(f"[INFO] POCA points loaded: {len(x)}")

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.5))
    
    # 1. Proyección XY
    # Desempaquetamos el cuarto argumento (el objeto QuadMesh) directamente como 'im0'
    _, _, _, im0 = axes[0].hist2d(x, y, bins=binning, cmap='viridis', rasterized=True)
    axes[0].set_title(r"Proyección $XY$")
    axes[0].set_xlabel(r"$x$ (cm)")
    axes[0].set_ylabel(r"$y$ (cm)")
    axes[0].set_xlim(left=-xSizeWorld/2, right=+xSizeWorld/2)
    axes[0].set_ylim(bottom=-ySizeWorld/2, top=+ySizeWorld/2)
    
    # Añadimos la barra de color y su respectiva etiqueta de conteo
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.ax.tick_params(labelsize=8)
    
    # 2. Proyección XZ
    _, _, _, im1 = axes[1].hist2d(x, z, bins=binning, cmap='viridis', rasterized=True)
    axes[1].set_title(r"Proyección $XZ$")
    axes[1].set_xlabel(r"$x$ (cm)")
    axes[1].set_ylabel(r"$z$ (cm)")
    axes[1].set_xlim(left=-xSizeWorld/2, right=+xSizeWorld/2)
    axes[1].set_ylim(bottom=-zSizeWorld/2, top=+zSizeWorld/2)
    
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=8)
    
    # 3. Proyección YZ
    _, _, _, im2 = axes[2].hist2d(y, z, bins=binning, cmap='viridis', rasterized=True)
    axes[2].set_title(r"Proyección $YZ$")
    axes[2].set_xlabel(r"$y$ (cm)")
    axes[2].set_ylabel(r"$z$ (cm)")
    axes[2].set_xlim(left=-ySizeWorld/2, right=+ySizeWorld/2)
    axes[2].set_ylim(bottom=-zSizeWorld/2, top=+zSizeWorld/2)
    
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=8)
    
    
    # Ajuste de márgenes y guardado
    plt.tight_layout()
    plt.savefig('figura_conteos_poca.pdf', dpi=300, bbox_inches='tight')
    plt.show()





###########################################################
# Version 2: la entrada es un .npy (proyecciones ya hechas)
###########################################################
elif extension == ".npy":

    reduction = xSizeWorld / binning # should not change nothing if bot coincide
    numpy_file = np.load(file)
    print("[INFO] ----- These are not normalized files, but absolute flux files")
    print("Tamaño del archivo cargado (.npy):")
    print(numpy_file.shape)

    print("Máximo valor que encontramos:")
    print(int(np.max(numpy_file)))

    print("Total puntos POCA (flux):")
    print(int(np.sum(numpy_file[:,:,1])))

    

    physical_limits = [-64, 64, 64, -64]
    # channel0 = XY
    # channel0 = XZ
    # channel0 = YZ
    channels = numpy_file 
    channels_binned = canal_binned = block_reduce(channels, block_size=(2, 2, 1), func=np.mean)


    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3) 
    ax1.imshow(channels_binned[:,:,0], extent = physical_limits)
    ax1.set_title("Channel 0 (XY projection)")
    ax1.set_xlabel("Posición X (cm)")  
    ax1.set_ylabel("Posición Y (cm)")

    ax2.imshow(channels_binned[:,:,1], extent = physical_limits)
    ax2.set_title("Channel 1 (XZ projection)")
    ax2.set_xlabel("Posición X (cm)")  
    ax2.set_ylabel("Posición Z (cm)")
    
    ax3.imshow(channels_binned[:,:,2], extent = physical_limits)
    ax3.set_title("Channel 0 (YZ projection)")
    ax3.set_xlabel("Posición Y (cm)")  
    ax3.set_ylabel("Posición Z (cm)")
    

    plt.tight_layout()
    plt.show()
