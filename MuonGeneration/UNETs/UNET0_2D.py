#!/usr/bin/env python3
"""
UNET0_2D.py

UNET 2D para reconstrucción de densidad a partir de POCA data.
Entrada: 3 canales POCA (128, 128, 3)
Salida: Densidad 2D reconstruida (128, 128, 1)

Training con validación, checkpoints, etc.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

# ===========================================================================
# CONFIGURATION
# ===========================================================================
H5_FILE = Path("/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/training_dataset.h5")
OUTPUT_DIR = Path("/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/UNETs/results")

# Hiperparámetros
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15

# ===========================================================================
# DATA LOADING
# ===========================================================================
def load_h5_data(h5_file):
    """Carga datos del .h5"""
    print("[INFO] Cargando datos del .h5...")
    
    with h5py.File(h5_file, 'r') as f:
        X = f['datasets/images'][:]
        Y = f['datasets/targets'][:]
        train_idx = f['splits/train_idx'][:]
        val_idx = f['splits/val_idx'][:]
        test_idx = f['splits/test_idx'][:]
    
    print(f"  X: {X.shape}, Y: {Y.shape}")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return (X[train_idx], Y[train_idx], 
            X[val_idx], Y[val_idx], 
            X[test_idx], Y[test_idx])

def normalize_data(X, Y):
    """Normaliza inputs a [0, 1]"""
    # Para cada sample, normalizar independently
    X_min = np.min(X, axis=(1, 2, 3), keepdims=True)
    X_max = np.max(X, axis=(1, 2, 3), keepdims=True)
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)
    
    Y_min = np.min(Y, axis=(1, 2, 3), keepdims=True)
    Y_max = np.max(Y, axis=(1, 2, 3), keepdims=True)
    Y_norm = (Y - Y_min) / (Y_max - Y_min + 1e-8)
    
    return X_norm, Y_norm, (X_min, X_max), (Y_min, Y_max)

# ===========================================================================
# MODEL: UNET 2D
# ===========================================================================
def create_unet(input_shape=(128, 128, 3), n_filters_base=32):
    """
    UNET 2D simétrica
    - Encoder: 4 niveles de downsampling
    - Decoder: 4 niveles de upsampling
    - Skip connections
    """
    inputs = keras.Input(shape=input_shape)
    
    # ===== ENCODER =====
    # Conv block 1 (128 -> 64)
    c1 = layers.Conv2D(n_filters_base, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(n_filters_base, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    p1 = layers.Dropout(0.1)(p1)
    
    # Conv block 2 (64 -> 32)
    c2 = layers.Conv2D(n_filters_base*2, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(n_filters_base*2, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    p2 = layers.Dropout(0.1)(p2)
    
    # Conv block 3 (32 -> 16)
    c3 = layers.Conv2D(n_filters_base*4, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(n_filters_base*4, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D(2)(c3)
    p3 = layers.Dropout(0.2)(p3)
    
    # Conv block 4 (16 -> 8, bottleneck)
    c4 = layers.Conv2D(n_filters_base*8, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(n_filters_base*8, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(2)(c4)
    p4 = layers.Dropout(0.2)(p4)
    
    # ===== BOTTLENECK =====
    b = layers.Conv2D(n_filters_base*16, 3, activation='relu', padding='same')(p4)
    b = layers.Conv2D(n_filters_base*16, 3, activation='relu', padding='same')(b)
    
    # ===== DECODER =====
    # Upconv 1 (8 -> 16)
    u1 = layers.UpSampling2D(2)(b)
    u1 = layers.Conv2D(n_filters_base*8, 3, activation='relu', padding='same')(u1)
    u1 = layers.Concatenate()([u1, c4])  # Skip connection
    u1 = layers.Conv2D(n_filters_base*8, 3, activation='relu', padding='same')(u1)
    u1 = layers.Conv2D(n_filters_base*8, 3, activation='relu', padding='same')(u1)
    
    # Upconv 2 (16 -> 32)
    u2 = layers.UpSampling2D(2)(u1)
    u2 = layers.Conv2D(n_filters_base*4, 3, activation='relu', padding='same')(u2)
    u2 = layers.Concatenate()([u2, c3])  # Skip connection
    u2 = layers.Conv2D(n_filters_base*4, 3, activation='relu', padding='same')(u2)
    u2 = layers.Conv2D(n_filters_base*4, 3, activation='relu', padding='same')(u2)
    
    # Upconv 3 (32 -> 64)
    u3 = layers.UpSampling2D(2)(u2)
    u3 = layers.Conv2D(n_filters_base*2, 3, activation='relu', padding='same')(u3)
    u3 = layers.Concatenate()([u3, c2])  # Skip connection
    u3 = layers.Conv2D(n_filters_base*2, 3, activation='relu', padding='same')(u3)
    u3 = layers.Conv2D(n_filters_base*2, 3, activation='relu', padding='same')(u3)
    
    # Upconv 4 (64 -> 128)
    u4 = layers.UpSampling2D(2)(u3)
    u4 = layers.Conv2D(n_filters_base, 3, activation='relu', padding='same')(u4)
    u4 = layers.Concatenate()([u4, c1])  # Skip connection
    u4 = layers.Conv2D(n_filters_base, 3, activation='relu', padding='same')(u4)
    u4 = layers.Conv2D(n_filters_base, 3, activation='relu', padding='same')(u4)
    
    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(u4)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='UNET_2D')
    return model

# ===========================================================================
# TRAINING
# ===========================================================================
def train_unet():
    """Entrena la UNET"""
    
    print("="*70)
    print("ENTRENAMIENTO UNET 2D")
    print("="*70)
    print()
    
    # ===== CARGAR DATOS =====
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_h5_data(H5_FILE)
    
    # ===== NORMALIZAR =====
    print("[INFO] Normalizando datos...")
    X_train_norm, Y_train_norm, _, _ = normalize_data(X_train, Y_train)
    X_val_norm, Y_val_norm, _, _ = normalize_data(X_val, Y_val)
    X_test_norm, Y_test_norm, _, _ = normalize_data(X_test, Y_test)
    
    # ===== CREAR MODELO =====
    print("[INFO] Creando modelo UNET...")
    model = create_unet(input_shape=(128, 128, 3), n_filters_base=32)
    model.summary()
    
    # ===== COMPILAR =====
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='mse',  # MSE para regresión
        metrics=['mae', 'mse']
    )
    
    # ===== CALLBACKS =====
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = OUTPUT_DIR / f"checkpoint_epoch{{epoch:02d}}.h5"
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
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print()
    
    history = model.fit(
        X_train_norm, Y_train_norm,
        validation_data=(X_val_norm, Y_val_norm),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint_callback, early_stop],
        verbose=1
    )
    
    # ===== EVALUAR EN TEST =====
    print()
    print("[INFO] Evaluando en TEST...")
    test_loss, test_mae, test_mse = model.evaluate(X_test_norm, Y_test_norm, verbose=0)
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
    print("[INFO] Generando predicciones visuales...")
    
    n_samples = 5
    predictions = model.predict(X_test_norm[:n_samples], verbose=0)
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    
    for i in range(n_samples):
        # Input (canal 0: log_counts)
        axes[i, 0].imshow(X_test_norm[i, :, :, 0], cmap='viridis')
        axes[i, 0].set_title(f"Sample {i} - Input (log_counts)")
        axes[i, 0].axis('off')
        
        # Input (canal 1: mean_theta_sq)
        axes[i, 1].imshow(X_test_norm[i, :, :, 1], cmap='plasma')
        axes[i, 1].set_title(f"Sample {i} - Input (mean_theta_sq)")
        axes[i, 1].axis('off')
        
        # Target
        axes[i, 2].imshow(Y_test_norm[i, :, :, 0], cmap='hot')
        axes[i, 2].set_title(f"Sample {i} - Ground Truth")
        axes[i, 2].axis('off')
        
        # Prediction
        axes[i, 3].imshow(predictions[i, :, :, 0], cmap='hot')
        axes[i, 3].set_title(f"Sample {i} - Prediction")
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions_samples.png", dpi=100)
    print(f"  Predicciones: {OUTPUT_DIR / 'predictions_samples.png'}")
    
    print()
    print("="*70)
    print("[SUCCESS] Entrenamiento completado")
    print("="*70)

if __name__ == "__main__":
    train_unet()
