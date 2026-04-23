"""
================================================================================
                        UNET0_2D.py - 2D U-NET Training Pipeline
================================================================================

¿QUÉ HACE ESTE SCRIPT?
``````````````````````
Este script entrena un modelo 2D U-NET para reconstruir geometría a partir de 
datos de muografía con ruido. El objetivo es eliminar el ruido y recuperar la 
máscara booleana 2D de la geometría original usando pocos muones.

El modelo procesa 3 canales de entrada (log_counts, mean_theta_sq_z, var_theta_z)
y produce una máscara de probabilidad de geometría como salida.


INSTRUCCIONES DE USO
````````````````````
1. Edita las variables de CONTROL VARIABLES (líneas ~45-70) según necesites:
   - device: "CPU" o "GPU"
   - environment: "local" o "cluster"
   - training: True (entrenar) o False (cargar modelo existente)
   - visualization: True para mostrar gráficos (desactiva en cluster)
   - only_use_channel_0: True para usar solo canal 0 (entrenamiento más rápido)
   - BATCH_SIZE, EPOCHS, LEARNING_RATE: hiperparámetros de entrenamiento

2. Ejecución:
   $ python UNET0_2D.py
   
   - Si training=True: entrena modelo nuevo y lo guarda
   - Si training=False: carga modelo existente e infiere
   - Si create_augmented_data=True: crea automáticamente datos aumentados


ARCHIVOS DE INPUT (¿Desde dónde se toman?)
```````````````````````````````````````````
1. HDF5 Original: 128x128x3.h5
   Ubicación LOCAL:   /home/samuel/Work/Muography_Denoising/MuonGeneration/data/h5_datasets/128x128x3.h5
   Ubicación CLUSTER: /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/h5_datasets/128x128x3.h5
   
   Estructura:
   - training/images (N, 128, 128, 3)   [Datos de entrenamiento]
   - training/labels (N, 128, 128, 1)   [Etiquetas ground truth]
   - validation/images (M, 128, 128, 3) [Datos de validación]
   - validation/labels (M, 128, 128, 1)
   - test/images (K, 128, 128, 3)       [Datos de test]
   - test/labels (K, 128, 128, 1)

2. HDF5 Aumentado: 128x128x3_augmented.h5 (GENERADO AUTOMÁTICAMENTE si no existe)
   Se crea en la MISMA carpeta que 128x128x3.h5
   Contiene: 16x más samples (4 rotaciones × 4 flip modes) + metadatos de transformación


ARCHIVOS DE OUTPUT (¿Hacia dónde se guardan?)
``````````````````````````````````````````````
1. Modelo Entrenado:
   Ubicación: ./models/UNET2D_{channels}_bs{batch}_ep{epochs}_lr{lr}/
   
   Ejemplo con defaults (16 batch, 100 epochs, 0.001 lr, 3 channels):
   ./models/UNET2D_3ch_bs16_ep100_lr0.001/
   
   Ejemplo con only_use_channel_0=True:
   ./models/UNET2D_1ch_channel0_bs16_ep100_lr0.001/
   
   Archivos generados:
   - model.keras          [Modelo entrenado en formato Keras 3.x]
   - hyperparameters.json [Hiperparámetros usados + timestamp]

2. Visualizaciones (se muestran en pantalla, sin guardar archivos):
   - Predicciones vs Ground Truth en test set
   - Canales normalizados (antes/después)
   - Samples aumentadas con metadatos de transformación
   - Augmentation preview (rotaciones)


NOTA SOBRE LOS CANALES
``````````````````````
- Canal 0 (log_counts): Información clara para reconstruir geometría (PRIMARIO)
- Canal 1 (mean_theta_sq_z): Alto ruido, bajo SNR (AUXILIAR)
- Canal 2 (var_theta_z): Alto ruido, bajo SNR (AUXILIAR)

Solución: Normalización por canal (z-score independiente) + opción only_use_channel_0
para usar solo el canal más informativo si es necesario.


FLUJO TÍPICO DE EJECUCIÓN
`````````````````````````
1. Script inicia y carga configuración
2. Si create_augmented_data=True:
   - Lee 128x128x3.h5
   - Genera versión aumentada (128x128x3_augmented.h5)
   - Guarda en mismo directorio que original
3. Si training=True:
   - Lee datos aumentados (o original si no existe aumentado)
   - Aplica normalización por canal on-the-fly
   - Entrena modelo por N epochs
   - Guarda modelo y hiperparámetros
4. Si visualization=True:
   - Muestra predicciones en test set
   - Muestra efecto de normalización de canales
   - Muestra samples augmentadas con metadatos


LOCAL vs CLUSTER
``````````````````````````
- "local":   Archivos en /home/samuel/Work/Muography_Denoising/...
- "cluster": Archivos en /gpfs/users/dominguezs/Muography_Denoising/...


TROUBLESHOOTING
```````````````
- Error "HDF5 file not found": Verifica ruta en variable H5_FILE
- No se crea augmented.h5: Asegúrate que create_augmented_data=True
- Visualizaciones no aparecen: Desactiva visualization=True en cluster
- Memoria insuficiente: Reduce BATCH_SIZE o activa only_use_channel_0=True
- Modelo no se carga: Verifica que hiperparámetros coincidan con modelo guardado

================================================================================
"""

import time
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers


##################################################################
##################################################################
# 		              CONTROL VARIABLES
##################################################################
##################################################################

device = "CPU" # "GPU" or "CPU"
see_dataset_images = True
environment = "local"  # "local" o "cluster"

explore_hdf5 = True
visualization = True
training = True

# Data augmentation
create_augmented_data = True  # Si True, crea/verifica H5 augmentado automáticamente
force_augmentation_rewrite = True  # Si True, reescribe H5 augmentado aunque exista

# Channel selection
# Note: "the only useful channel here is channel 0, related with the number of counts, 
#        provided the channels 1 and 2 have to low signal to noise ratio"
only_use_channel_0 = True  # If True, use only channel 0 for faster training

# Unet hyperparameters
# Hyperparameters (to be optimized)
BATCH_SIZE        = 16
EPOCHS            = 100
LEARNING_RATE     = 1e-3
VAL_SPLIT         = 0.15

# Dynamic image size based on channel selection
SIZE_IMAGES       = (128, 128, 1) if only_use_channel_0 else (128, 128, 3)

N_FILTERS         = 32
FILTER_SIZE       = 3
N_LEVELS          = 4 # as in the original paper

show_summary      = True

#################################


if device == "CPU":
    print("Device set to CPU. Training may be slow.")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
CHECK_GPU = True if device == "GPU" else False



if environment == "local":
    H5_FILE = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration/data/h5_datasets/128x128x3.h5")
    OUTPUT_DIR = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration/UNETs/results")
elif environment == "cluster":
    H5_FILE = Path("/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/h5_datasets/128x128x3.h5")
    OUTPUT_DIR = Path("/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/UNETs/results")

# Construct augmented HDF5 path: same location as original, with _augmented suffix
H5_FILE_AUGMENTED = H5_FILE.parent / f"{H5_FILE.stem}_augmented{H5_FILE.suffix}"

# Original HDF5 path (will be augmented to 16x samples during augmentation)
print(f"\n[INFO] ----- Environment set to {environment}")

# check GPU availability (optional, for performance)
if CHECK_GPU == False:
    print("Not looking for GPU, maybe everything is running on CPU. Don't worry if you are testing, careful if you are willing to get results.")
else:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nGPUs detected: {len(gpus)}\n")
        print(70 * '-')
        for gpu in gpus:
            print(f"  - {gpu}")
        print(70 * '-')                
        continue_ = input("Continue with GPU training? (y/n): ")
        print(60 * '-')
        if continue_.lower() != 'y':
            print("Exiting. Please configure your environment and run again.")
            sys.exit(0)
    else:
        print("No GPUs detected. Training may be slow on CPU.")





##################################################################
##################################################################
# 		       DIRECTORIES AND DATA
##################################################################
##################################################################

# ===========================================================================
# HDF5 EXPLORATION
# ===========================================================================

def explore_h5_structure(filepath):
    """Explora y muestra la estructura de un archivo .h5"""

    if not os.path.exists(filepath):
        print(f"[ERROR] ----- Archivo no encontrado: {filepath}")
        return
    
    print(f"\n[DIR] Explorando: {filepath}")
    print("=" * 80)
    
    with h5py.File(filepath, 'r') as f:
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}[DATASET] {os.path.basename(name)}")
                print(f"{indent}   └─ Shape: {obj.shape}, Dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}[DIR] {os.path.basename(name)}/")
        
        f.visititems(print_structure)
        
        if f.attrs:
            print("\n" + "=" * 80)
            print("Atributos globales:")
            for key, value in f.attrs.items():
                print(f"   {key}: {value}")




def explore_dataset_from_h5_file(h5_file):
    """
    Solo lee metadatos (shapes y tamaños) del H5, sin cargar datos en RAM.
    Devuelve un dict con shapes y número de samples por split.
    """
    required_paths = [
        "training/images", "training/labels",
        "validation/images", "validation/labels",
        "test/images", "test/labels",
    ]
    with h5py.File(h5_file, 'r') as f:
        missing = [p for p in required_paths if p not in f]
        if missing:
            raise KeyError(
                "El archivo HDF5 no tiene el formato esperado. "
                f"Faltan rutas: {missing}"
            )

        info = {
            'img_shape':  tuple(f["training/images"].shape[1:]),   # (H, W, 3)
            'label_shape':  tuple(f["training/labels"].shape[1:]),   # (H, W, 1)
            'n_train':    f["training/images"].shape[0],
            'n_val':      f["validation/images"].shape[0],
            'n_test':     f["test/images"].shape[0],
        }

    # Validaciones mínimas de consistencia
    assert len(info['img_shape']) == 3 and info['img_shape'][-1] == 3, \
        f"Se esperan imágenes (H,W,3), shape encontrado: {info['img_shape']}"
    assert len(info['label_shape']) == 3 and info['label_shape'][-1] == 1, \
        f"Se esperan labels (H,W,1), shape encontrado: {info['label_shape']}"

    print(f"  Split training:   {info['n_train']} samples  — shape img {info['img_shape']}, lbl {info['label_shape']}")
    print(f"  Split validation: {info['n_val']} samples")
    print(f"  Split test:       {info['n_test']} samples")

    return info




##################################################################
##################################################################
#               DATASET VISUALIZATION (EXPLORATION)
##################################################################
##################################################################

def plot_images_dataset(h5_file, n_samples=4):
    """
    Plotea algunos samples del split training para inspección visual.
    Lee solo los primeros n_samples directamente del H5 (sin cargar todo).
    """
    if not see_dataset_images or environment == "cluster":
        return

    with h5py.File(h5_file, 'r') as f:
        X = f["training/images"][:n_samples].astype(np.float32)
        Y = f["training/labels"][:n_samples].astype(np.float32)

    fig, axes = plt.subplots(nrows=n_samples, ncols=4, figsize=(20, 10 * n_samples))

    for i in range(n_samples):
        axes[i, 0].imshow(X[i, :, :, 0], cmap='viridis')
        axes[i, 0].set_title(f"Sample {i} - Input (log_counts)")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(X[i, :, :, 1], cmap='viridis')
        axes[i, 1].set_title(f"Sample {i} - Input (mean_theta_sq_z)")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(X[i, :, :, 2], cmap='viridis')
        axes[i, 2].set_title(f"Sample {i} - Input (var_theta_z)")
        axes[i, 2].axis('off')

        axes[i, 3].imshow(Y[i, :, :, 0], cmap='hot')
        axes[i, 3].set_title(f"Sample {i} - Ground Truth")
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.show()


# EXECUTION: 
# exploration
explore_h5_structure(H5_FILE)
time.sleep(3)
explore_dataset_from_h5_file(H5_FILE)
time.sleep(3)




plot_images_dataset(H5_FILE)


##################################################################
##################################################################
##################################################################
# 		               DATA MANIPULATION
##################################################################
##################################################################
##################################################################


##################################################################
# 		               PREPROCESSING
##################################################################

def normalize_channels(image):
    """
    Normaliza cada canal por separado usando z-score (media=0, std=1).
    Si only_use_channel_0, retorna solo canal 0. Si no, retorna todos los canales.
    """
    channels = range(1) if only_use_channel_0 else range(image.shape[-1])
    n_ch = len(list(channels))
    normalized = np.zeros((image.shape[0], image.shape[1], n_ch), dtype=np.float32)
    
    for out_ch, ch in enumerate(channels):
        data = image[:, :, ch]
        std = np.std(data)
        normalized[:, :, out_ch] = (data - np.mean(data)) / std if std > 0 else data - np.mean(data)
    
    return normalized


# NOTA SOBRE CANALES RUIDOSOS:
# Canal 0 (log_counts): información clara para reconstruir geometría
# Canal 1 y 2 (mean_theta_sq_z, var_theta_z): ruidosos pero potencialmente útiles
#
# Solución implementada: normalización por canal
# - Esto escala cada canal independientemente
# - Los canales 1 y 2 no dominarán por tener magnitudes altas
# - El modelo aprenderá si son útiles o no durante el entrenamiento




##################################################################
# 		               DATA AUGMENTATION
##################################################################

def create_augmented_h5(h5_file_original, h5_file_augmented, force_rewrite=False):
    """
    Crea HDF5 con datos aumentados (16x): 4 rotaciones × 4 flip modes + metadatos.
    """
    if h5_file_augmented.exists() and not force_rewrite:
        print(f"\n[AUGMENT] ✓ HDF5 augmentado ya existe: {h5_file_augmented}")
        return
    
    print(f"\n[AUGMENT] Creando HDF5 aumentado con metadatos de transformación...")
    print(f"  De: {h5_file_original}\n  A:  {h5_file_augmented}")
    
    flip_names = {0: "none", 1: "vertical", 2: "horizontal", 3: "both"}
    flip_ops = [
        lambda x: x,
        lambda x: np.flipud(x),
        lambda x: np.fliplr(x),
        lambda x: np.flipud(np.fliplr(x))
    ]
    
    with h5py.File(h5_file_original, 'r') as f_in:
        with h5py.File(h5_file_augmented, 'w') as f_out:
            for split in ['training', 'validation', 'test']:
                print(f"  Procesando {split}...")
                images = f_in[f"{split}/images"][:].astype(np.float32)
                labels = f_in[f"{split}/labels"][:].astype(np.float32)
                n_samples = images.shape[0]
                
                aug_images = np.zeros((n_samples * 16, *images.shape[1:]), dtype=np.float32)
                aug_labels = np.zeros((n_samples * 16, *labels.shape[1:]), dtype=np.float32)
                aug_metadata = []
                
                idx = 0
                for rot in range(4):
                    for flip_mode in range(4):
                        for i in range(n_samples):
                            img_aug = flip_ops[flip_mode](np.rot90(images[i], k=rot, axes=(0, 1)))
                            lbl_aug = flip_ops[flip_mode](np.rot90(labels[i], k=rot, axes=(0, 1)))
                            aug_images[idx] = img_aug
                            aug_labels[idx] = lbl_aug
                            aug_metadata.append(f"orig_idx={i}, rotation={rot*90}°, flip={flip_names[flip_mode]}")
                            idx += 1
                
                f_out.create_dataset(f"{split}/images", data=aug_images, compression='gzip')
                f_out.create_dataset(f"{split}/labels", data=aug_labels, compression='gzip')
                metadata_ds = f_out.create_dataset(f"{split}/augmentation_metadata", (len(aug_metadata),), dtype=h5py.string_dtype(encoding='utf-8'))
                for idx, meta in enumerate(aug_metadata):
                    metadata_ds[idx] = meta
                print(f"    Samples: {n_samples} → {n_samples * 16} (augmented)")
    
    print(f"\n[AUGMENT] ✓ Completado con metadatos de seguimiento!")




##################################################################
# DATA VISUALIZATION: Augmentation Preview
##################################################################

def visualize_augmented_data(h5_file, n_samples=2):
    """Visualiza augmentation: original + rotaciones + ground truth"""
    if not visualization or environment == "cluster": return
    print(f"\n[VIZ] Visualizando augmentation (rotaciones)...")
    
    with h5py.File(h5_file, 'r') as f:
        X = f["training/images"][:n_samples].astype(np.float32)
        Y = f["training/labels"][:n_samples].astype(np.float32)
    
    fig, axes = plt.subplots(nrows=n_samples, ncols=5, figsize=(20, 4*n_samples))
    if n_samples == 1: axes = axes.reshape(1, -1)
    
    titles = ["Original (0°)", "Rotación 90°", "Rotación 180°", "Rotación 270°", "Ground Truth"]
    for i in range(n_samples):
        for j in range(4):
            img = X[i] if j == 0 else np.rot90(X[i], k=j, axes=(0, 1))
            axes[i, j].imshow(img[:, :, 0], cmap='viridis')
            axes[i, j].set_title(f"Sample {i} - {titles[j]}")
            axes[i, j].axis('off')
        axes[i, 4].imshow(Y[i, :, :, 0], cmap='hot')
        axes[i, 4].set_title(f"Sample {i} - {titles[4]}")
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    plt.show()
    print(f"[VIZ] Visualización completada!")


def visualize_normalized_channels(h5_file, n_samples=2):
    """Visualiza canales antes/después normalización"""
    if not visualization or environment == "cluster": return
    print(f"\n[VIZ] Visualizando canales normalizados...")
    
    with h5py.File(h5_file, 'r') as f:
        X = f["training/images"][:n_samples].astype(np.float32)
    
    if only_use_channel_0:
        # Mostrar solo canal 0
        fig, axes = plt.subplots(nrows=n_samples, ncols=2, figsize=(10, 4*n_samples))
        if n_samples == 1: axes = axes.reshape(1, -1)
        for i in range(n_samples):
            axes[i, 0].imshow(X[i, :, :, 0], cmap='viridis')
            axes[i, 0].set_title(f"Sample {i} - Ch0 (original)")
            axes[i, 0].axis('off')
            X_norm = normalize_channels(X[i])
            axes[i, 1].imshow(X_norm[:, :, 0], cmap='RdBu_r', vmin=-2, vmax=2)
            axes[i, 1].set_title(f"Sample {i} - Ch0 (normalized)")
            axes[i, 1].axis('off')
    else:
        # Mostrar todos los 3 canales
        fig, axes = plt.subplots(nrows=n_samples, ncols=6, figsize=(18, 4*n_samples))
        if n_samples == 1: axes = axes.reshape(1, -1)
        for i in range(n_samples):
            X_norm = normalize_channels(X[i])
            for ch in range(3):
                axes[i, ch].imshow(X[i, :, :, ch], cmap='viridis')
                axes[i, ch].set_title(f"Sample {i} - Ch{ch} (original)")
                axes[i, ch].axis('off')
                axes[i, 3+ch].imshow(X_norm[:, :, ch], cmap='RdBu_r', vmin=-2, vmax=2)
                axes[i, 3+ch].set_title(f"Sample {i} - Ch{ch} (normalized)")
                axes[i, 3+ch].axis('off')
    
    plt.tight_layout()
    plt.show()
    print(f"[VIZ] Visualización de canales completada!")


def visualize_augmented_samples_with_metadata(h5_file, n_augmented_samples=4):
    """Visualiza samples augmentadas con metadatos de transformación"""
    if not visualization or environment == "cluster": return
    print(f"\n[VIZ] Visualizando samples augmentadas con metadatos...")
    
    with h5py.File(h5_file, 'r') as f:
        if "training/augmentation_metadata" not in f:
            print("[VIZ] No metadata found"); return
        images = f["training/images"][:n_augmented_samples].astype(np.float32)
        metadata = [f["training/augmentation_metadata"][i].decode('utf-8') for i in range(n_augmented_samples)]
    
    fig, axes = plt.subplots(nrows=n_augmented_samples, ncols=2, figsize=(12, 4*n_augmented_samples))
    if n_augmented_samples == 1: axes = axes.reshape(1, -1)
    
    for i in range(n_augmented_samples):
        axes[i, 0].imshow(images[i, :, :, 0], cmap='viridis')
        axes[i, 0].set_title(f"Sample {i}")
        axes[i, 0].axis('off')
        axes[i, 1].text(0.5, 0.5, metadata[i], ha='center', va='center', fontsize=10, family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f"Metadata")
    
    plt.tight_layout()
    plt.show()
    print(f"[VIZ] Visualización completada!")





# UNET architecture

def conv_block(input_tensor, n_filters, name, repeat_conv = 2):
    """
    Convolutional block with Batch Normalization
    Conv2D: convolution + bias term [LINEAR]
    |
    BatchNorm [LINEAR]
    |
    ReLU [NON-LINEAR]
    
    Usually, the conv_block is repeated twice in each level of the UNET. 
    This can be tunned by indicating repeat_conv = 1 or 3, for example.
    (Recommended: 2, as in the original UNET paper)
    """
    x = input_tensor
    for i in range(1, repeat_conv + 1):
        x = layers.Conv2D(n_filters, (FILTER_SIZE, FILTER_SIZE), padding='same', 
                          use_bias=False, name=f"{name}_conv{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.Activation('relu', name=f"{name}_relu{i}")(x)
    return x



def unet_dynamic(input_shape=SIZE_IMAGES, n_levels=N_LEVELS, n_filters=N_FILTERS):
    """
    Complete UNET architecture with dynamic number of levels and filters.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Listas para almacenar salidas del encoder para las skip connections
    encoder_outputs = []
    
    # --- ENCODER ---
    x = inputs
    for i in range(n_levels):
        filters = n_filters * (2 ** i)
        x = conv_block(x, filters, name=f"Enc{i+1}")
        encoder_outputs.append(x)
        x = layers.MaxPooling2D((2, 2), name=f"Pool{i+1}")(x)

    # --- BOTTLENECK ---
    x = conv_block(x, n_filters * (2 ** n_levels), name="Bottleneck")

    # --- DECODER ---
    # Iteramos hacia atrás desde n_levels hasta 1
    for i in range(n_levels, 0, -1):
        filters = n_filters * (2 ** (i - 1))
        # Upsampling
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name=f"Up{i}")(x)
        # Skip connection
        concat = layers.concatenate([x, encoder_outputs[i-1]], name=f"Concat{i}")
        # Bloque convolucional
        x = conv_block(concat, filters, name=f"Dec{i}")

    # Salida
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', name="Output")(x)

    return Model(inputs, outputs)


# Instanciar
model = unet_dynamic()
if show_summary:   
    model.summary()



# ===========================================================================
# TRAINING
# ===========================================================================

def get_dataset_sizes(h5_file):
    """Obtiene tamaño de cada split del dataset"""
    with h5py.File(h5_file, 'r') as f:
        return {split: f[f"{split}/images"].shape[0] for split in ["training", "validation", "test"]}

def train_unet(model, h5_file=H5_FILE,
               batch_size=BATCH_SIZE,
               epochs=EPOCHS, 
               learning_rate=LEARNING_RATE,
               apply_normalization=True):
    """
    Entrena UNET con carga de batches desde HDF5 (lazy loading).
    Cada batch se carga en RAM, se procesa y se libera.
    """
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['mse'])
    
    sizes = get_dataset_sizes(h5_file)
    n_train, n_val = sizes["training"], sizes["validation"]
    
    print(f"\n[TRAINING] Iniciando entrenamiento...")
    print(f"  - Total samples training: {n_train}")
    print(f"  - Total samples validation: {n_val}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Normalización de canales: {apply_normalization}\n")
    
    # TRAINING
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        n_batches = int(np.ceil(n_train / batch_size))
        
        train_loss = 0.0
        train_mse = 0.0
        
        # Iterar por batches
        for batch_idx in range(n_batches):
            # Calcular índices del batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_train)
            
            # CARGAR batch en RAM
            with h5py.File(h5_file, 'r') as f:
                X_batch = f["training/images"][start_idx:end_idx].astype(np.float32)
                Y_batch = f["training/labels"][start_idx:end_idx].astype(np.float32)
            
            # APLICAR NORMALIZACIÓN a cada imagen del batch (en RAM, no modifica H5)
            if apply_normalization:
                X_batch = np.array([normalize_channels(X_batch[i]) for i in range(X_batch.shape[0])])
            
            # Entrenar el modelo con este batch
            loss, mse = model.train_on_batch(X_batch, Y_batch)
            train_loss += loss
            train_mse += mse
            
            # LIBERAR batch (Python lo hará automáticamente)
            del X_batch, Y_batch
            
            # Mostrar progreso
            if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
                print(f"  Batch {batch_idx + 1}/{n_batches}")
        
        # Promediar métricas de training
        train_loss /= n_batches
        train_mse /= n_batches
        
        # =================== VALIDATION ===================
        n_val_batches = int(np.ceil(n_val / batch_size))
        
        val_loss = 0.0
        val_mse = 0.0
        
        # Iterar por batches de validación
        for batch_idx in range(n_val_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_val)
            
            # CARGAR batch en RAM
            with h5py.File(h5_file, 'r') as f:
                X_val = f["validation/images"][start_idx:end_idx].astype(np.float32)
                Y_val = f["validation/labels"][start_idx:end_idx].astype(np.float32)
            
            # APLICAR NORMALIZACIÓN
            if apply_normalization:
                X_val = np.array([normalize_channels(X_val[i]) for i in range(X_val.shape[0])])
            
            # Evaluar
            loss, mse = model.evaluate(X_val, Y_val, verbose=0)
            val_loss += loss
            val_mse += mse
            
            # LIBERAR batch
            del X_val, Y_val
        
        # Promediar métricas de validación
        val_loss /= n_val_batches
        val_mse /= n_val_batches
        
        # Mostrar resultados de la época
        print(f"  train_loss: {train_loss:.6f} | train_mse: {train_mse:.6f}")
        print(f"  val_loss: {val_loss:.6f} | val_mse: {val_mse:.6f}")
    
    print(f"\n[TRAINING] Entrenamiento completado!")
    return model


# ===========================================================================
# SAVE / LOAD MODEL
# ===========================================================================

def get_model_folder(batch_size, epochs, learning_rate):
    """Retorna la carpeta donde guardar el modelo con nombre descriptivo"""
    script_dir = Path(__file__).parent
    # Nombre descriptivo: incluir "channel_0" si está activado
    channels_str = "1ch_channel0" if only_use_channel_0 else "3ch"
    model_name = f"UNET2D_{channels_str}_bs{batch_size}_ep{epochs}_lr{learning_rate}"
    return script_dir / "models" / model_name


def save_model(model, batch_size, epochs, learning_rate):
    """Guarda el modelo entrenado en una carpeta con sus hiperparámetros"""
    # Crear carpeta de modelos
    model_folder = get_model_folder(batch_size, epochs, learning_rate)
    model_folder.mkdir(parents=True, exist_ok=True)
    
    # Rutas
    model_path = model_folder / "model.keras"
    hyperparams_path = model_folder / "hyperparameters.json"
    
    # Guardar el modelo
    model.save(str(model_path))
    print(f"\n[SAVE] Modelo guardado en: {model_path}")
    
    # Guardar hiperparámetros
    hyperparams = {
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'n_filters': N_FILTERS,
        'filter_size': FILTER_SIZE,
        'n_levels': N_LEVELS,
        'input_shape': SIZE_IMAGES,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    print(f"[SAVE] Carpeta del modelo: {model_folder}")
    print(f"[SAVE] Hiperparámetros guardados en: {hyperparams_path}")


def load_trained_model(batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE):
    """Carga el modelo entrenado si existe"""
    model_folder = get_model_folder(batch_size, epochs, learning_rate)
    model_path = model_folder / "model.keras"
    hyperparams_path = model_folder / "hyperparameters.json"
    
    if not model_folder.exists():
        print(f"[LOAD] No se encontró carpeta de modelo en: {model_folder}")
        print(f"[LOAD] Asegúrate de que los hiperparámetros coinciden con el modelo guardado.")
        return None
    
    try:
        model = keras.models.load_model(str(model_path))
        print(f"\n[LOAD] Modelo cargado desde: {model_path}")
        print(f"[LOAD] Carpeta: {model_folder}")
        
        # Mostrar hiperparámetros
        if hyperparams_path.exists():
            with open(hyperparams_path, 'r') as f:
                hyperparams = json.load(f)
            print(f"[LOAD] Hiperparámetros:")
            for key, value in hyperparams.items():
                print(f"       - {key}: {value}")
        
        return model
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo: {e}")
        return None


# ===========================================================================
# VISUALIZATION: Predictions vs Ground Truth
# ===========================================================================

def visualize_predictions(model, h5_file=H5_FILE, n_samples=4):
    """Compara predicciones vs ground truth en test set"""
    if not visualization or environment == "cluster": return
    print(f"\n[VIZ] Generando visualización de predicciones...")
    
    with h5py.File(h5_file, 'r') as f:
        X_test = f["test/images"][:n_samples].astype(np.float32)
        Y_test = f["test/labels"][:n_samples].astype(np.float32)
    
    # Normalizar datos igual que en entrenamiento
    X_test = np.array([normalize_channels(X_test[i]) for i in range(X_test.shape[0])])
    Y_pred = model.predict(X_test, verbose=0)
    
    fig, axes = plt.subplots(nrows=n_samples, ncols=3, figsize=(15, 5*n_samples))
    if n_samples == 1: axes = axes.reshape(1, -1)
    
    titles = ["Input (log_counts)", "Predicción", "Ground Truth"]
    for i in range(n_samples):
        for j, (img, title) in enumerate([(X_test[i, :, :, 0], titles[0]), (Y_pred[i, :, :, 0], titles[1]), (Y_test[i, :, :, 0], titles[2])]):
            cmap = 'viridis' if j == 0 else 'hot'
            axes[i, j].imshow(img, cmap=cmap)
            axes[i, j].set_title(f"Sample {i} - {title}")
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()
    print(f"[VIZ] Visualización completada!")

# ===========================================================================
# MAIN EXECUTION: Automatic augmentation + training/inference
# ===========================================================================

if create_augmented_data:
    # Automatically create augmented dataset if missing
    create_augmented_h5(H5_FILE, H5_FILE_AUGMENTED, force_rewrite=force_augmentation_rewrite)

# Select dataset for training
H5_FOR_TRAINING = H5_FILE_AUGMENTED if H5_FILE_AUGMENTED.exists() else H5_FILE

if training:
    print("\n" + "=" * 80)
    print(f"[MAIN] Usando H5: {H5_FOR_TRAINING}")
    print("[MAIN] Iniciando entrenamiento...")
    print("=" * 80)
    
    # Train with augmented data + channel normalization
    model = train_unet(model, h5_file=H5_FOR_TRAINING, apply_normalization=True)
    
    # Save trained model with hyperparameters
    save_model(model, BATCH_SIZE, EPOCHS, LEARNING_RATE)
else:
    print("\n" + "=" * 80)
    print("[MAIN] Training = False, cargando modelo entrenado...")
    # Load pre-trained model from disk
    model = load_trained_model(BATCH_SIZE, EPOCHS, LEARNING_RATE)
    
    if model is None:
        print("[ERROR] No se pudo cargar el modelo. Por favor, entrena primero.")
        sys.exit(1)

# Visualizaciones
if visualization:
    print("\n" + "=" * 80)
    print("[MAIN] Generando visualizaciones...")
    visualize_predictions(model, h5_file=H5_FILE, n_samples=4)
    visualize_normalized_channels(H5_FOR_TRAINING, n_samples=2)
    if H5_FILE_AUGMENTED.exists():
        visualize_augmented_samples_with_metadata(H5_FOR_TRAINING, n_augmented_samples=6)

