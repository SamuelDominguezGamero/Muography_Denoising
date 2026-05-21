#!/usr/bin/env python3
"""
Script de evaluación visual para UNET2D (POCA Denoising)
Carga el mejor modelo entrenado, toma muestras aleatorias de TEST y grafica las comparativas.
"""

import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("[ERROR] TensorFlow es requerido. Instálalo con: mamba install tensorflow")
    exit(1)

try:
    import h5py
except ImportError:
    print("[ERROR] h5py es requerido. Instálalo con: mamba install h5py")
    exit(1)

# ===========================================================================
# CONFIGURACIÓN DE RUTAS
# ===========================================================================
BASE = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration")
DATA = BASE / "data"
DATASETS = DATA / "datasets"
MODELS = DATA / "models"

H5_FILE = DATASETS / "dataset0.h5"
MODEL_PATH =Path("/home/samuel/Work/Muography_Denoising/MuonGeneration/data/models_XY/checkpoints/best_model_0_XY.keras")
OUTPUT_PLOT_DIR = BASE / "UNETs/UNET0/visualizaciones_test"

NUM_SAMPLES_TO_PLOT = 3  # Número de instancias independientes a visualizar


# ===========================================================================
# PIPELINE DE EVALUACIÓN VISUAL
# ===========================================================================
def main():
    print("\n" + "=" * 80)
    print("EVALUACIÓN VISUAL DE LA UNET - CONJUNTO DE TEST")
    print("=" * 80)

    # 1. Verificar archivos esenciales
    if not H5_FILE.exists():
        print(f"[ERROR] No se encuentra el archivo de datos: {H5_FILE}")
        return
    if not MODEL_PATH.exists():
        print(f"[ERROR] No se encuentra el modelo entrenado en: {MODEL_PATH}")
        return

    # 2. Cargar el mejor modelo entrenado
    print(f"[INFO] Cargando pesos óptimos desde {MODEL_PATH.name}...")
    try:
        model = keras.models.load_model(str(MODEL_PATH))
        print("  ✓ Modelo cargado con éxito.")
    except Exception as e:
        print(f"[ERROR] Fallo crítico al reconstruir la red: {e}")
        return

    # 3. Extraer muestras aleatorias de TEST sin colapsar la RAM
    print(f"[INFO] Abriendo {H5_FILE.name} para extraer muestras de test...")
    with h5py.File(H5_FILE, 'r') as f:
        test_poca_ds = f['test/poca']
        test_gt_ds = f['test/gt']
        total_test_samples = test_poca_ds.shape[0]
        
        print(f"  - Muestras totales disponibles en TEST: {total_test_samples}")
        
        # Seleccionar índices aleatorios
        random.seed(42) # Semilla fija para reproducibilidad al evaluar las mismas imágenes
        chosen_indices = random.sample(range(total_test_samples), min(NUM_SAMPLES_TO_PLOT, total_test_samples))
        print(f"  - Índices seleccionados para graficar: {chosen_indices}")
        
        # Cargar únicamente las matrices indexadas y escalar a [0, 1]
        x_samples = np.array([test_poca_ds[i] for i in chosen_indices]).astype(np.float32) / 255.0
        y_samples = np.array([test_gt_ds[i] for i in chosen_indices]).astype(np.float32) / 255.0

    # 4. Realizar la inferencia con la GPU
    print(f"\n[INFO] Ejecutando *forward pass* (Inferencia UNet) sobre las {len(chosen_indices)} muestras...")
    predictions = model.predict(x_samples, batch_size=len(chosen_indices))

    # 5. Graficar y guardar resultados por canal (XY, XZ, YZ)
    OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    channel_names = ["Proyección XY (Canal 0)", "Proyección XZ (Canal 1)", "Proyección YZ (Canal 2)"]

    for idx, sample_idx in enumerate(chosen_indices):
        print(f"  ↳ Generando composición para la muestra de Test index: {sample_idx}...")
        
        # Creamos una cuadrícula de 3 filas (una por canal de proyección) y 3 columnas (POCA, Pred, GT)
        fig, axes = plt.subplots(3, 3, figsize=(11, 10))
        fig.suptitle(f"Evaluación UNet - Muestra de Test #{sample_idx}", fontsize=14, fontweight='bold', y=0.98)
        
        for channel in range(3):
            # Fila correspondiente al canal actual
            ax_row = axes[channel]
            
            # Datos de entrada, predicción de la red y ground truth para el canal mapeado
            img_poca = x_samples[idx, :, :, channel]
            img_pred = predictions[idx, :, :, channel]
            img_gt = y_samples[idx, :, :, channel]
            
            # Columna 1: Entrada POCA (Ruidosa)
            im0 = ax_row[0].imshow(img_poca, cmap='viridis', origin='lower')
            ax_row[0].set_title(f"{channel_names[channel]} - Entrada POCA", fontsize=10)
            fig.colorbar(im0, ax=ax_row[0], fraction=0.046, pad=0.04)
            
            # Columna 2: Salida UNet (Denoised)
            im1 = ax_row[1].imshow(img_pred, cmap='viridis', origin='lower')
            ax_row[1].set_title("Salida Denoised (UNet)", fontsize=10, fontweight='bold', color='darkgreen')
            fig.colorbar(im1, ax=ax_row[1], fraction=0.046, pad=0.04)
            
            # Columna 3: Ground Truth (Objetivo)
            im2 = ax_row[2].imshow(img_gt, cmap='viridis', origin='lower')
            ax_row[2].set_title("Ground Truth (Ideal)", fontsize=10)
            fig.colorbar(im2, ax=ax_row[2], fraction=0.046, pad=0.04)
            
            # Limpieza de ejes para mejor presentación visual
            for ax in ax_row:
                ax.axis('off')
        
        plt.tight_layout()
        save_path = OUTPUT_PLOT_DIR / f"test_sample_{sample_idx}_denoising.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Gráfica guardada en: {save_path.relative_to(BASE)}")

    print("\n" + "=" * 80)
    print("PROCESO DE VISUALIZACIÓN COMPLETADO")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
