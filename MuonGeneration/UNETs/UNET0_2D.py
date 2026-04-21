"""
UNET0_2D.py

First model of 2D UNET
``````````````````````
Target: eliminate noise --> reconstruct perfectly the geometry (2D boolean mask)
First idea: several channels obtained from POCA signal
Dataset: 2D projected quantities. Different geometries, with different number of muons

Ideal objective: reproduce the geometry with few muons.

Script structure
````````````````
- Libraries and control variables
- Exploration of directories and data
- Data loading (lazy) for optimum (V)RAM usage
- Visualisation of data
- Preprocessing
- UNET training
"""

##################################################################
##################################################################
# 		       LIBRARIES AND CONTROL VARIABLES
##################################################################
##################################################################

import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

# CONTROL VARIABLES (¡¡ADJUST!!)
#################################
device = "CPU" # "GPU" or "CPU"
see_dataset_images = True
environment = "local"  # "local" o "cluster"
#################################


if device == "CPU":
    print("Device set to CPU. Training may be slow.")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
CHECK_GPU = True if device == "GPU" else False

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ===========================================================================
# CONFIGURATION
# ===========================================================================

environment = "local"  # "local" o "cluster"
if environment == "local":
    H5_FILE = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration/data/h5_datasets/128x128x3.h5")
    OUTPUT_DIR = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration/UNETs/results")
elif environment == "cluster":
    H5_FILE = Path("/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/h5_datasets/128x128x3.h5")
    OUTPUT_DIR = Path("/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/UNETs/results")
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

def explore_hdf5(filepath):
    """Explora y muestra la estructura de un archivo HDF5"""
    
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
            print("🏷️  Atributos globales:")
            for key, value in f.attrs.items():
                print(f"   {key}: {value}")


# ===========================================================================
# DATA LOADING — LAZY (sin cargar todo en RAM)
# ===========================================================================

def get_h5_info(h5_file):
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
            'lbl_shape':  tuple(f["training/labels"].shape[1:]),   # (H, W, 1)
            'n_train':    f["training/images"].shape[0],
            'n_val':      f["validation/images"].shape[0],
            'n_test':     f["test/images"].shape[0],
        }

    # Validaciones mínimas de consistencia
    assert len(info['img_shape']) == 3 and info['img_shape'][-1] == 3, \
        f"Se esperan imágenes (H,W,3), shape encontrado: {info['img_shape']}"
    assert len(info['lbl_shape']) == 3 and info['lbl_shape'][-1] == 1, \
        f"Se esperan labels (H,W,1), shape encontrado: {info['lbl_shape']}"

    print(f"  Split training:   {info['n_train']} samples  — shape img {info['img_shape']}, lbl {info['lbl_shape']}")
    print(f"  Split validation: {info['n_val']} samples")
    print(f"  Split test:       {info['n_test']} samples")

    return info


def _normalize_batch(X, Y):
    """Normaliza un batch a [0,1] por sample."""
    X_min = np.min(X, axis=(1, 2, 3), keepdims=True)
    X_max = np.max(X, axis=(1, 2, 3), keepdims=True)
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)

    Y_min = np.min(Y, axis=(1, 2, 3), keepdims=True)
    Y_max = np.max(Y, axis=(1, 2, 3), keepdims=True)
    Y_norm = (Y - Y_min) / (Y_max - Y_min + 1e-8)

    return X_norm.astype(np.float32), Y_norm.astype(np.float32)


def _h5_generator(h5_file, split, batch_size, shuffle):
    """
    Generador Python que lee mini-batches del H5 uno a uno.
    El archivo permanece abierto solo mientras se itera; nunca se carga
    el dataset completo en RAM.
    """
    with h5py.File(h5_file, 'r', swmr=True) as f:
        images = f[f"{split}/images"]
        labels = f[f"{split}/labels"]
        n = images.shape[0]

        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            # HDF5 requiere índices ordenados para fancy indexing
            batch_idx = np.sort(indices[start : start + batch_size])
            X = images[batch_idx].astype(np.float32)
            Y = labels[batch_idx].astype(np.float32)
            X, Y = _normalize_batch(X, Y)
            yield X, Y


def make_tf_dataset(h5_file, split, batch_size, shuffle=True):
    """
    Devuelve un tf.data.Dataset lazy que lee del H5 en mini-batches.
    Los datos se normalizan [0,1] en el generador, nunca se carga
    el split completo en RAM.
    """
    info = get_h5_info(h5_file)

    output_signature = (
        tf.TensorSpec(shape=(None, *info['img_shape']), dtype=tf.float32),
        tf.TensorSpec(shape=(None, *info['lbl_shape']), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(
        generator=lambda: _h5_generator(h5_file, split, batch_size, shuffle),
        output_signature=output_signature,
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


# ===========================================================================
# DATASET VISUALIZATION  (usa solo los primeros N samples del H5)
# ===========================================================================

def plot_images_dataset(h5_file, n_samples=10):
    """
    Plotea algunos samples del split training para inspección visual.
    Lee solo los primeros n_samples directamente del H5 (sin cargar todo).
    """
    if not see_dataset_images or environment == "cluster":
        return

    with h5py.File(h5_file, 'r') as f:
        X = f["training/images"][:n_samples].astype(np.float32)
        Y = f["training/labels"][:n_samples].astype(np.float32)

    fig, axes = plt.subplots(nrows=n_samples, ncols=4, figsize=5*n_cols, 4 * n_samples))

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


########################################################################
########################################################################
# PRE-TRAINING: EXPLORATION + VISUALIZATION

explore_hdf5(H5_FILE)
control_variable = input("Dataset inspeccionado. ¿Continuar con visualización? (y/n): ")
if control_variable.lower() != 'y':
    print("Visualización cancelada.")
    sys.exit(0)
else:
    print("Continuando con visualización...")

plot_images_dataset(H5_FILE, n_samples=10)

print(60 * "-")
print(60 * "-")
train_ = input("Continue to training? (y/n): ")
print(60 * "-")
print(60 * "-")
if train_.lower() != 'y':
    print("Training canceled.")
    sys.exit(0)
else:
    print("Continuing to training...")


##################################################################
##################################################################
# 		       DATA AUGMENTATION
##################################################################
##################################################################



##################################################################
##################################################################
# 		       DIRECTORIES AND DATA
##################################################################
##################################################################


# UNET architecture






# Hyperparameters (to be optimized later)
BATCH_SIZE        = 16
EPOCHS            = 50
LEARNING_RATE     = 1e-3
VAL_SPLIT         = 0.15



# ===========================================================================
# MODEL: UNET 2D
# ===========================================================================

def create_unet(input_shape=(128, 128, 3), n_filters_base=32):
    """
    UNET 2D
    - Encoder: 4 niveles de downsampling
    - Decoder: 4 niveles de upsampling
    - Skip connections
    """
    inputs = keras.Input(shape=input_shape)

    # ===== ENCODER =====
    c1 = layers.Conv2D(n_filters_base, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(n_filters_base, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    p1 = layers.Dropout(0.1)(p1)

    c2 = layers.Conv2D(n_filters_base*2, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(n_filters_base*2, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    p2 = layers.Dropout(0.1)(p2)

    c3 = layers.Conv2D(n_filters_base*4, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(n_filters_base*4, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D(2)(c3)
    p3 = layers.Dropout(0.2)(p3)

    c4 = layers.Conv2D(n_filters_base*8, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(n_filters_base*8, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(2)(c4)
    p4 = layers.Dropout(0.2)(p4)

    # ===== BOTTLENECK =====
    b = layers.Conv2D(n_filters_base*16, 3, activation='relu', padding='same')(p4)
    b = layers.Conv2D(n_filters_base*16, 3, activation='relu', padding='same')(b)

    # ===== DECODER =====
    u1 = layers.UpSampling2D(2)(b)
    u1 = layers.Conv2D(n_filters_base*8, 3, activation='relu', padding='same')(u1)
    u1 = layers.Concatenate()([u1, c4])
    u1 = layers.Conv2D(n_filters_base*8, 3, activation='relu', padding='same')(u1)
    u1 = layers.Conv2D(n_filters_base*8, 3, activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D(2)(u1)
    u2 = layers.Conv2D(n_filters_base*4, 3, activation='relu', padding='same')(u2)
    u2 = layers.Concatenate()([u2, c3])
    u2 = layers.Conv2D(n_filters_base*4, 3, activation='relu', padding='same')(u2)
    u2 = layers.Conv2D(n_filters_base*4, 3, activation='relu', padding='same')(u2)

    u3 = layers.UpSampling2D(2)(u2)
    u3 = layers.Conv2D(n_filters_base*2, 3, activation='relu', padding='same')(u3)
    u3 = layers.Concatenate()([u3, c2])
    u3 = layers.Conv2D(n_filters_base*2, 3, activation='relu', padding='same')(u3)
    u3 = layers.Conv2D(n_filters_base*2, 3, activation='relu', padding='same')(u3)

    u4 = layers.UpSampling2D(2)(u3)
    u4 = layers.Conv2D(n_filters_base, 3, activation='relu', padding='same')(u4)
    u4 = layers.Concatenate()([u4, c1])
    u4 = layers.Conv2D(n_filters_base, 3, activation='relu', padding='same')(u4)
    u4 = layers.Conv2D(n_filters_base, 3, activation='relu', padding='same')(u4)

    outputs = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(u4)

    model = keras.Model(inputs=inputs, outputs=outputs, name='UNET_2D')
    return model


# ===========================================================================
# TRAINING
# ===========================================================================

def train_unet():
    """Entrena la UNET con carga lazy desde HDF5."""

    print("=" * 70)
    print("ENTRENAMIENTO UNET 2D")
    print("=" * 70)

    # ===== INFO DEL DATASET (sin cargar en RAM) =====
    print("[INFO] Leyendo metadatos del H5...")
    info = get_h5_info(H5_FILE)

    steps_per_epoch  = info['n_train'] // BATCH_SIZE
    validation_steps = info['n_val']   // BATCH_SIZE
    test_steps       = info['n_test']  // BATCH_SIZE

    print(f"  steps_per_epoch:  {steps_per_epoch}")
    print(f"  validation_steps: {validation_steps}")

    # ===== DATASETS LAZY =====
    print("[INFO] Creando tf.data.Datasets lazy...")
    train_ds = make_tf_dataset(H5_FILE, "training",   BATCH_SIZE, shuffle=True)
    val_ds   = make_tf_dataset(H5_FILE, "validation", BATCH_SIZE, shuffle=False)
    test_ds  = make_tf_dataset(H5_FILE, "test",       BATCH_SIZE, shuffle=False)

    # ===== CREAR MODELO =====
    print("[INFO] Creando modelo UNET...")
    model = create_unet(input_shape=info['img_shape'], n_filters_base=32)
    model.summary()

    # ===== COMPILAR =====
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']
    )

    # ===== CALLBACKS =====
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = OUTPUT_DIR / "checkpoint_epoch{epoch:02d}.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        str(checkpoint_path),
        save_best_only=False,
        monitor='val_loss',
        verbose=1,
        save_freq='epoch'
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # ===== ENTRENAR =====
    print()
    print("[INFO] Iniciando entrenamiento...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs:     {EPOCHS}")
    print(f"  LR:         {LEARNING_RATE}")
    print()

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=[checkpoint_callback, early_stop],
        verbose=1
    )

    # ===== EVALUAR EN TEST =====
    print()
    print("[INFO] Evaluando en TEST...")
    test_loss, test_mae, test_mse = model.evaluate(test_ds, steps=test_steps, verbose=0)
    print(f"  Test MSE: {test_mse:.6f}")
    print(f"  Test MAE: {test_mae:.6f}")

    # ===== GUARDAR MODELO =====
    model_path = OUTPUT_DIR / "UNET0_2D_final.h5"
    model.save(model_path)
    print(f"\n[SUCCESS] Modelo guardado: {model_path}")

    # ===== PLOTEAR HISTORY =====
    print("[INFO] Guardando gráficos...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Training MAE')
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_history.png", dpi=100)
    print(f"  Gráfico: {OUTPUT_DIR / 'training_history.png'}")

    # ===== PREDICCIONES VISUALES =====
    # Leemos solo n_samples directamente del H5 para no romper el flujo lazy
    print("[INFO] Generando predicciones visuales...")
    n_samples = 5

    with h5py.File(H5_FILE, 'r') as f:
        X_vis = f["test/images"][:n_samples].astype(np.float32)
        Y_vis = f["test/labels"][:n_samples].astype(np.float32)
    X_vis, Y_vis = _normalize_batch(X_vis, Y_vis)

    predictions = model.predict(X_vis, verbose=0)

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

    for i in range(n_samples):
        axes[i, 0].imshow(X_vis[i, :, :, 0], cmap='viridis')
        axes[i, 0].set_title(f"Sample {i} - Input (log_counts)")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(X_vis[i, :, :, 1], cmap='plasma')
        axes[i, 1].set_title(f"Sample {i} - Input (mean_theta_sq)")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(Y_vis[i, :, :, 0], cmap='hot')
        axes[i, 2].set_title(f"Sample {i} - Ground Truth")
        axes[i, 2].axis('off')

        axes[i, 3].imshow(predictions[i, :, :, 0], cmap='hot')
        axes[i, 3].set_title(f"Sample {i} - Prediction")
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions_samples.png", dpi=100)
    print(f"  Predicciones: {OUTPUT_DIR / 'predictions_samples.png'}")

    print()
    print("=" * 70)
    print("[SUCCESS] Entrenamiento completado")
    print("=" * 70)


if __name__ == "__main__":
    train_unet()