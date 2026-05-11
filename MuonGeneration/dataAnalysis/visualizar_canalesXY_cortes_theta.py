import argparse
import ROOT
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
args = parser.parse_args()

# Carga de datos base
df_base = ROOT.RDataFrame("events", args.input)

# Definimos los cortes de Theta (canales de información)
# Ejemplo: [0.1 a 0.2], [0.2 a 0.3], [0.3 a 0.5] rad
cortes = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)]
n_canales = len(cortes)

binning = 32
fig, axes = plt.subplots(1, n_canales, figsize=(5 * n_canales, 5))

# Lista para guardar los arrays si luego quieres pasarlos a la U-Net
canales_xy = []

for i, (t_min, t_max) in enumerate(cortes):
    # Aplicamos el filtro para este canal específico
    condicion = f"theta >= {t_min} && theta < {t_max}"
    df_canal = df_base.Filter(condicion)
    
    # Extraemos a Numpy
    res = df_canal.AsNumpy(columns=["poca_x", "poca_y"])
    x_c, y_c = res["poca_x"], res["poca_y"]
    
    # Generamos el histograma 2D para este canal
    # Nota: Si x_c está vacío, hist2d podría fallar; manejamos el plot:
    counts, xedges, yedges, im = axes[i].hist2d(
        x_c, y_c, 
        bins=binning, 
        range=[[-50, 50], [-50, 50]], # Ajusta según las dimensiones de tu detector
        cmap='viridis'
    )
    
    canales_xy.append(counts)
    
    axes[i].set_title(f"Canal XY: {t_min} < $\\theta$ < {t_max}")
    axes[i].set_xlabel("poca_x")
    axes[i].set_ylabel("poca_y")
    fig.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.show()

# Estructura para la U-Net: (Canales, H, W)
input_tensor = np.stack(canales_xy, axis=0)
print(f"[INFO] Tensor de entrada creado con forma: {input_tensor.shape}")
