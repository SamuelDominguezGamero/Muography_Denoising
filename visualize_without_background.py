import argparse
import ROOT
import numpy as np
import matplotlib.pyplot as plt

def get_data(file_path):
    df = ROOT.RDataFrame("events", file_path)
    df = df.Filter("theta > 0.01")
    res = df.AsNumpy(columns=["poca_x", "poca_y", "poca_z", "theta"])
    return res

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Archivo ROOT de la simulación con objeto")
parser.add_argument("--background", required=True, help="Archivo ROOT del fondo (EMPTY)")
args = parser.parse_args()

# Carga de datos
res_obj = get_data(args.input)
res_bkg = get_data(args.background)

print(f"[INFO] POCA objeto: {len(res_obj['poca_x'])}")
print(f"[INFO] POCA fondo:   {len(res_bkg['poca_x'])}")

# Factor de escala para normalizar la estadística del fondo a la de la señal
scale = len(res_obj['poca_x']) / len(res_bkg['poca_x']) if len(res_bkg['poca_x']) > 0 else 1.0
print(f"[INFO] Factor de escala aplicado: {scale:.4f}")

binning = 60
# Definimos el rango común para que los histogramas coincidan pixel a pixel
# Ajusta estos límites según las dimensiones de tu detector (e.g., -50 a 50)
h_range = [[-50, 50], [-50, 50]] 

def subtract_hists(data_obj, data_bkg, val_obj, val_bkg, weights=None):
    # Generar histograma de objeto
    h_obj, x_e, y_e = np.histogram2d(data_obj[0], data_obj[1], bins=binning, range=h_range, weights=val_obj)
    # Generar histograma de fondo
    h_bkg, _, _ = np.histogram2d(data_bkg[0], data_bkg[1], bins=binning, range=h_range, weights=val_bkg)
    # Restar y limpiar
    h_sub = h_obj - (scale * h_bkg)
    return np.clip(h_sub, 0, None), x_e, y_e

# 1. Preparar la figura
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# --- PROYECCIÓN XY (Densidad) ---
h_xy, x_e, y_e = subtract_hists([res_obj["poca_x"], res_obj["poca_y"]], 
                                 [res_bkg["poca_x"], res_bkg["poca_y"]], None, None)
im0 = axes[0].imshow(h_xy.T, origin='lower', extent=[*h_range[0], *h_range[1]], cmap='viridis')
axes[0].set_title("XY (Limpio)"); axes[0].set_xlabel("poca_x"); axes[0].set_ylabel("poca_y")
fig.colorbar(im0, ax=axes[0])

# --- PROYECCIÓN XZ (Densidad) ---
# Usamos range diferente para Z si es necesario, aquí asumo h_range para simplificar
h_xz, x_e, z_e = subtract_hists([res_obj["poca_x"], res_obj["poca_z"]], 
                                 [res_bkg["poca_x"], res_bkg["poca_z"]], None, None)
im1 = axes[1].imshow(h_xz.T, origin='lower', extent=[*h_range[0], *h_range[1]], cmap='viridis')
axes[1].set_title("XZ (Limpio)"); axes[1].set_xlabel("poca_x"); axes[1].set_ylabel("poca_z")
fig.colorbar(im1, ax=axes[1])

# --- PROYECCIÓN YZ (Densidad) ---
h_yz, y_e, z_e = subtract_hists([res_obj["poca_y"], res_obj["poca_z"]], 
                                 [res_bkg["poca_y"], res_bkg["poca_z"]], None, None)
im2 = axes[2].imshow(h_yz.T, origin='lower', extent=[*h_range[0], *h_range[1]], cmap='viridis')
axes[2].set_title("YZ (Limpio)"); axes[2].set_xlabel("poca_y"); axes[2].set_ylabel("poca_z")
fig.colorbar(im2, ax=axes[2])

# --- MAPA THETA^2 (XY) ---
h_th2, x_e, y_e = subtract_hists([res_obj["poca_x"], res_obj["poca_y"]], 
                                  [res_bkg["poca_x"], res_bkg["poca_y"]], 
                                  res_obj["theta"]**2, res_bkg["theta"]**2)
im3 = axes[3].imshow(h_th2.T, origin='lower', extent=[*h_range[0], *h_range[1]], cmap='viridis')
axes[3].set_title(r"Intensidad $\theta^2$ (XY) - Sub")
axes[3].set_xlabel("poca_x"); axes[3].set_ylabel("poca_y")
fig.colorbar(im3, ax=axes[3], label=r'$\sum \theta^2$')

plt.tight_layout()
plt.show()
