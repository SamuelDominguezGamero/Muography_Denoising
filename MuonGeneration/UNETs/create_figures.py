from tensorflow import keras
from pathlib import Path

# 1. Configurar rutas
FIGURAS_DIR = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration/Figuras_TFM")
FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

model_path = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/models_XY_1_perc/checkpoints/best_model_0_XY_1_perc.keras"

# 2. Objetos de carga
_CUSTOM_OBJECTS = {
    'charbonnier_loss': lambda y_t, y_p: y_p,
    'psnr_metric':      lambda y_t, y_p: y_p,
    'ssim_metric':      lambda y_t, y_p: y_p,
}

model = keras.models.load_model(model_path, custom_objects=_CUSTOM_OBJECTS)

# 3. Generar diagrama horizontal y limpio
keras.utils.plot_model(
    model,
    to_file=str(FIGURAS_DIR / "arquitectura_unet_horizontal.png"),
    show_shapes=True,         # Muestra las dimensiones (ej. 128x128x1)
    show_layer_names=False,   # Desactiva nombres largos para reducir el ruido visual
    rankdir="LR",             # ¡CRÍTICO! Cambia el flujo a Horizontal (Left to Right)
    dpi=250                   # Máxima nitidez para el documento final
)

print("Diagrama horizontal guardado en Figuras_TFM/arquitectura_unet_horizontal.png")