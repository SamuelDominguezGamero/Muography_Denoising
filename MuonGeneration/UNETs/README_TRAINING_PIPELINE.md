# UNET Training Pipeline - Guía Completa

## Visión General

Pipeline completo para entrenar una UNET 2D con datos POCA:

```
MERGED_*.npy (POCA data) + ground_truth_density2D.npy
              ↓
        create_h5_dataset.py  ← Convierte a .h5
              ↓
      training_dataset.h5
              ↓
         UNET0_2D.py  ← Entrena modelo
              ↓
    UNET0_2D_final.h5 + gráficos de training
```

## Paso 1: Generar datos POCA

Ejecutar en el cluster:

```bash
cd /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/test

# Configurar cuántas geometrías quieres
# Editar loop_configuration_files.py:
#   max_geometries = 20
#   max_geometries_simulated = 20

python3 loop_configuration_files.py
```

Esto genera:
- Archivos MERGED en `/gpfs/.../merged_poca_data/MERGED_*.npy`
- Cada uno contiene un diccionario con 3 canales:
  - `log_counts` (128, 128, 1)
  - `mean_theta_sq_z` (128, 128, 1)
  - `var_theta_z` (128, 128, 1)

### Esperar a que terminen los jobs

```bash
watch -n 1 'squeue -u dominguezs | wc -l'
```

## Paso 2: Crear Dataset .H5

Una vez que haya archivos MERGED en el cluster:

```bash
cd /home/samuel/Work/Muography_Denoising/MuonGeneration/UNETs

python3 create_h5_dataset.py
```

**¿Qué hace?**
1. Lee todos los `MERGED_*.npy` del cluster (vía NFS)
2. Carga los correspondientes `ground_truth_density2D.npy`
3. Apila en (N, 128, 128, 3) para inputs
4. Apila en (N, 128, 128, 1) para targets
5. Crea splits: 70% train, 15% val, 15% test
6. Guarda todo en `training_dataset.h5`

**Salida esperada:**
```
[INFO] Encontrados 20 archivos MERGED
[SUCCESS] Cargados 20/20 pares
[INFO] Dataset final:
  X shape: (20, 128, 128, 3)
  Y shape: (20, 128, 128, 1)
  X range: [0.0000, 1.0000]
  Y range: [0.0000, 1.0000]
[SUCCESS] Archivo guardado: training_dataset.h5 (XXX MB)
```

### Estructura del .H5

```
training_dataset.h5
├── /datasets/
│   ├── images       (N, 128, 128, 3)    # Inputs (3 canales POCA)
│   └── targets      (N, 128, 128, 1)    # Targets (ground truth density)
├── /splits/
│   ├── train_idx    # Índices training
│   ├── val_idx      # Índices validación
│   └── test_idx     # Índices test
└── /metadata/
    └── geometry_names  # Nombres de geometrías
```

## Paso 3: Entrenar UNET

```bash
cd /home/samuel/Work/Muography_Denoising/MuonGeneration/UNETs

python3 UNET0_2D.py
```

**Hiperparámetros** (editar en `UNET0_2D.py`):
```python
BATCH_SIZE = 16      # Aumentar si hay VRAM
EPOCHS = 50          # Parar antes si overfitting
LEARNING_RATE = 1e-3 # Ajustar según resultados
```

**¿Qué hace?**
1. Carga datos del .h5
2. Normaliza cada sample a [0, 1]
3. Crea modelo UNET:
   - 4 niveles encoder (downsampling)
   - 4 niveles decoder (upsampling)
   - Skip connections
   - ~7.8M parámetros

4. Entrena con:
   - Optimizer: Adam
   - Loss: MSE (regresión)
   - Early stopping si val_loss no mejora 10 epochs
   - Checkpoints cada epoch

5. Guarda:
   - Modelo final: `UNET0_2D_final.h5`
   - Gráficos: `training_history.png`, `predictions_samples.png`

**Salida esperada:**
```
[INFO] Cargando datos del .h5...
  X: (20, 128, 128, 3), Y: (20, 128, 128, 1)
  Train: 14, Val: 3, Test: 3

[INFO] Creando modelo UNET...
Model: "UNET_2D"
... (arquitectura)
Total params: 7,765,569

Epoch 1/50
... training ...

[INFO] Evaluando en TEST...
  Test MSE: 0.012345
  Test MAE: 0.054321
  
[SUCCESS] Modelo guardado: results/UNET0_2D_final.h5
```

## Estructura de Directorios

```
MuonGeneration/
├── data/
│   ├── merged_poca_data/          ← MERGED_*.npy (salida simulation)
│   ├── ground_truth_data/2Dimensions/  ← ground_truth_density2D.npy
│   └── training_dataset.h5        ← Entrada a UNET (crear con create_h5_dataset.py)
└── UNETs/
    ├── UNET0_2D.py                ← Modelo y training
    ├── create_h5_dataset.py       ← Crea dataset .h5
    └── results/                   ← Salida training
        ├── UNET0_2D_final.h5      ← Modelo entrenado
        ├── training_history.png   ← Gráficos loss/mae
        ├── predictions_samples.png ← Ejemplos predicciones
        └── checkpoint_epoch*.h5   ← Checkpoints
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'tensorflow'"
```bash
pip install tensorflow h5py scikit-learn
```

### "No hay archivos MERGED"
1. Verifica que `loop_configuration_files.py` completó
2. Mira logs: `cat /gpfs/.../logs/log_merge_*.out`

### "OOM Error" en training
Reduce `BATCH_SIZE` en `UNET0_2D.py`

### Modelo no converge
- Baja `LEARNING_RATE` (ej: 1e-4)
- Aumenta `EPOCHS`
- Verifica que datos están normalizados

## Próximos Pasos

1. ✅ **Este paso**: Pipeline básica 2D
2. **Optimizar**: Data augmentation, dropout, batch norm
3. **3D**: Extender a UNET3D (128, 128, 128, 3)
4. **Inference**: Script para predicciones en nuevos datos
5. **Métricas**: SSIM, PSNR, análisis residuales

## Tips

- **Primero small**: Prueba con 5-10 geometrías antes de escalar
- **Monitor**: Abre `training_history.png` durante training
- **GPU**: Si tienes GPU, TensorFlow debería detectarla automáticamente
- **Reproducibilidad**: Random seeds están fijos (42) para reproducibilidad
