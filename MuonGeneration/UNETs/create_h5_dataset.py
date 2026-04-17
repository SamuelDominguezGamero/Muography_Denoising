#!/usr/bin/env python3
"""
create_h5_dataset.py

Convierte los archivos MERGED_*.npy + ground_truth_density2D.npy a un archivo .h5
para entrenar la UNET.

Estructura del .h5:
  /datasets/images       (N, 128, 128, 3) - Canales POCA
  /datasets/targets      (N, 128, 128, 1) - Ground truth
  /splits/train_idx      - Índices training
  /splits/val_idx        - Índices validación
  /splits/test_idx       - Índices test
  /metadata/geometry_names - Nombres geometría (para trazabilidad)
"""

import numpy as np
import h5py
import os
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

# ===========================================================================
# CONFIGURATION
# ===========================================================================
MERGED_PATH = Path("/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/merged_poca_data")
GROUND_TRUTH_2D_PATH = Path("/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/ground_truth_data/2Dimensions")

H5_OUTPUT = Path("/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/training_dataset.h5")

# Train/Val/Test split
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Grid size
npx, npy = 128, 128

# ===========================================================================
# STEP 1: ESCANEAR Y CARGAR DATOS
# ===========================================================================
print("="*70)
print("CREANDO DATASET .H5 PARA UNET")
print("="*70)
print()

# Buscar archivos MERGED
merged_files = sorted(glob.glob(str(MERGED_PATH / "MERGED_*.npy")))
print(f"[INFO] Encontrados {len(merged_files)} archivos MERGED")

if len(merged_files) == 0:
    print("[ERROR] No hay archivos MERGED. Ejecuta loop_configuration_files.py primero.")
    sys.exit(1)

# ===========================================================================
# STEP 2: CARGAR Y VALIDAR DATOS
# ===========================================================================
images_list = []
targets_list = []
geometry_names = []
failed_files = []

for i, merged_file in enumerate(merged_files, 1):
    filename = Path(merged_file).stem  # Ej: "MERGED_..._2D"
    
    # Extraer nombre geometría (quitar "MERGED_" y "_2D")
    geom_name = filename.replace("MERGED_", "").replace("_2D", "").replace("_3D", "")
    
    print(f"[{i}/{len(merged_files)}] Procesando: {geom_name}...", end=" ", flush=True)
    
    try:
        # ===== CARGAR POCA =====
        poca_data = np.load(merged_file, allow_pickle=True).item()
        
        # Extraer los 3 canales
        log_counts = poca_data["log_counts"][:, :, 0]  # (128, 128)
        mean_theta_sq = poca_data["mean_theta_sq_z"][:, :, 0]
        var_theta = poca_data["var_theta_z"][:, :, 0]
        
        # Stack en 3 canales: (128, 128, 3)
        image = np.stack([log_counts, mean_theta_sq, var_theta], axis=-1).astype(np.float32)
        
        # ===== CARGAR GROUND TRUTH =====
        gt_file = GROUND_TRUTH_2D_PATH / f"{geom_name}_ground_truth_density2D.npy"
        
        if not gt_file.exists():
            print(f"✗ Ground truth no encontrado: {gt_file.name}")
            failed_files.append(geom_name)
            continue
        
        target = np.load(gt_file).astype(np.float32)
        
        # Asegurar que sea (128, 128, 1)
        if target.ndim == 2:
            target = target[:, :, np.newaxis]
        elif target.ndim == 3 and target.shape[2] != 1:
            print(f"✗ Forma inesperada de target: {target.shape}")
            failed_files.append(geom_name)
            continue
        
        # Validar shapes
        if image.shape != (npy, npx, 3):
            print(f"✗ Imagen shape incorrecto: {image.shape}")
            failed_files.append(geom_name)
            continue
        
        if target.shape != (npy, npx, 1):
            print(f"✗ Target shape incorrecto: {target.shape}")
            failed_files.append(geom_name)
            continue
        
        images_list.append(image)
        targets_list.append(target)
        geometry_names.append(geom_name)
        print("✓")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        failed_files.append(geom_name)

print()
print(f"[SUCCESS] Cargados {len(images_list)}/{len(merged_files)} pares")
if failed_files:
    print(f"[WARNING] {len(failed_files)} archivos fallaron:")
    for f in failed_files[:5]:
        print(f"  - {f}")
    if len(failed_files) > 5:
        print(f"  ... y {len(failed_files) - 5} más")

if len(images_list) == 0:
    print("[ERROR] No se cargó ningún dato.")
    sys.exit(1)

# Stack en arrays
X = np.array(images_list)  # (N, 128, 128, 3)
Y = np.array(targets_list)  # (N, 128, 128, 1)
N = X.shape[0]

print()
print(f"[INFO] Dataset final:")
print(f"  X shape: {X.shape} (N, H, W, C)")
print(f"  Y shape: {Y.shape}")
print(f"  X dtype: {X.dtype} | Y dtype: {Y.dtype}")
print(f"  X range: [{np.min(X):.4f}, {np.max(X):.4f}]")
print(f"  Y range: [{np.min(Y):.4f}, {np.max(Y):.4f}]")

# ===========================================================================
# STEP 3: CREAR SPLITS TRAIN/VAL/TEST
# ===========================================================================
print()
print("[INFO] Creando splits...")

indices = np.arange(N)

# Train/Test
train_idx, test_idx = train_test_split(
    indices, test_size=(test_ratio + val_ratio), random_state=42
)

# Val/Test (de lo que quedó)
val_idx, test_idx = train_test_split(
    test_idx, test_size=test_ratio / (test_ratio + val_ratio), random_state=42
)

print(f"  Train: {len(train_idx)} ({100*len(train_idx)/N:.1f}%)")
print(f"  Val:   {len(val_idx)} ({100*len(val_idx)/N:.1f}%)")
print(f"  Test:  {len(test_idx)} ({100*len(test_idx)/N:.1f}%)")

# ===========================================================================
# STEP 4: GUARDAR EN .H5
# ===========================================================================
print()
print(f"[INFO] Guardando en: {H5_OUTPUT}")

H5_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with h5py.File(H5_OUTPUT, 'w') as f:
    # Datos
    f.create_dataset('datasets/images', data=X, compression='gzip', compression_opts=4)
    f.create_dataset('datasets/targets', data=Y, compression='gzip', compression_opts=4)
    
    # Splits
    f.create_dataset('splits/train_idx', data=train_idx)
    f.create_dataset('splits/val_idx', data=val_idx)
    f.create_dataset('splits/test_idx', data=test_idx)
    
    # Metadata
    geom_names_bytes = [g.encode('utf-8') for g in geometry_names]
    f.create_dataset('metadata/geometry_names', data=geom_names_bytes)
    
    # Atributos
    f.attrs['n_samples'] = N
    f.attrs['image_height'] = npy
    f.attrs['image_width'] = npx
    f.attrs['n_channels'] = 3
    f.attrs['split_ratio'] = f"{train_ratio:.2f}/{val_ratio:.2f}/{test_ratio:.2f}"

print(f"[SUCCESS] Archivo guardado: {H5_OUTPUT.stat().st_size / 1e6:.1f} MB")

# ===========================================================================
# STEP 5: VALIDAR LECTURA
# ===========================================================================
print()
print("[INFO] Validando lectura...")

with h5py.File(H5_OUTPUT, 'r') as f:
    print(f"  Claves: {list(f.keys())}")
    print(f"  X_train shape: {f['datasets/images'][f['splits/train_idx']][:5].shape}")
    print(f"  Y_train shape: {f['datasets/targets'][f['splits/train_idx']][:5].shape}")

print()
print("="*70)
print("[SUCCESS] Dataset .H5 creado correctamente")
print("="*70)
