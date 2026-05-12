import argparse
import ROOT
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--xSizeWorld", default=128)
parser.add_argument("--ySizeWorld", default=128)
parser.add_argument("--zSizeWorld", default=128)
args = parser.parse_args()

# Carga de datos
df = ROOT.RDataFrame("events", args.input)
df = df.Filter("abs(theta) > 0.0001")
res = df.AsNumpy(columns=["poca_x", "poca_y", "poca_z", "theta"])


xSizeWorld = args.xSizeWorld
ySizeWorld = args.ySizeWorld
zSizeWorld = args.zSizeWorld


x = res["poca_x"]
y = res["poca_y"]
z = res["poca_z"]
theta = res["theta"]

print(f"[INFO] POCA points loaded: {len(x)}")

# Selección de binning para la visualización (ej. el fino)
binning = 64

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# 1. Proyección XY (Densidad)
axes[0].hist2d(x, y, bins=binning, cmap='viridis')
axes[0].set_title("Densidad XY")
axes[0].set_xlabel("poca_x"); axes[0].set_ylabel("poca_y"); 
axes[0].set_xlim(left=-xSizeWorld/2, right=+xSizeWorld/2)
axes[0].set_ylim(bottom=-ySizeWorld/2, top=+ySizeWorld/2)

# 2. Proyección XZ (Densidad)
axes[1].hist2d(x, z, bins=binning, cmap='viridis')
axes[1].set_title("Densidad XZ")
axes[1].set_xlabel("poca_x"); axes[1].set_ylabel("poca_z"); 
axes[1].set_xlim(left=-xSizeWorld/2, right=+xSizeWorld/2)
axes[1].set_ylim(bottom=-zSizeWorld/2, top=+zSizeWorld/2)

# 3. Proyección YZ (Densidad)
axes[2].hist2d(y, z, bins=binning, cmap='viridis')
axes[2].set_title("Densidad YZ")
axes[2].set_xlabel("poca_y"); axes[2].set_ylabel("poca_z")
axes[2].set_xlim(left=-ySizeWorld/2, right=+ySizeWorld/2)
axes[2].set_ylim(bottom=-zSizeWorld/2, top=+zSizeWorld/2)

# 4. Mapa de Calor XY ponderado por theta^2
# Usamos weights para que el color represente la intensidad de la desviación
h = axes[3].hist2d(x, y, bins=binning, weights=theta**2, cmap='viridis')
axes[3].set_title(r"Intensidad $\theta^2$ (XY)")
axes[3].set_xlabel("poca_x"); axes[3].set_ylabel("poca_y")
axes[3].set_xlim(left=-xSizeWorld/2, right=+xSizeWorld/2)
axes[3].set_ylim(bottom=-ySizeWorld/2, top=+ySizeWorld/2)

# Añadimos una barra de color específica para el mapa de theta^2
fig.colorbar(h[3], ax=axes[3], label=r'$\sum \theta^2$')

plt.tight_layout()
plt.show()
