# UNET0 Data Pipeline & Training

Pipeline simplificada para generar `dataset0.h5` y entrenar UNET2D en denoising de proyecciones POCA.

## Arquitectura

```
┌─ data_utils.py              (Funciones compartidas)
│  ├─ extract_poca_from_source()    [TAR o directory]
│  ├─ create_gt_from_json()
│  ├─ match_poca_with_gt()
│  └─ create_h5_dataset()
│
├─ pipeline_data.py            (Orquestador: POCA → GT → H5)
├─ check_consistency.py         (Validación de estructura)
├─ checkear_instancias_fromh5.py (Visualización de muestras)
└─ train_unet.py               (Training UNET2D)
```

## Flujo de Datos

```
archivos.tar (o carpeta con .root files)
    ↓                     
.root files (POCA)       → extract_poca_from_source() → tensor_2D_POCA_*.npy
                           (flexible: TAR o directory)  (128×128×3 XY, XZ, YZ)

geometric_configurations_json/
    ↓
*.json geometries        → create_gt_from_json()    → tensor_2D_*.npy
                                                      (128×128×3 XY, XZ, YZ)
                                                      tensor_3D_*.npy
                                                      (128×128×128)

tensor_2D_POCA_*.npy + tensor_2D_*.npy → match_poca_with_gt() → pairs

pairs                    → create_h5_dataset()      → dataset0.h5
                                                      ├─ train/poca, train/gt
                                                      ├─ val/poca, val/gt
                                                      └─ test/poca, test/gt
```

## Ejecución

### 1. Generar Dataset H5

```bash
cd /home/samuel/Work/Muography_Denoising/MuonGeneration
python3 pipeline_data.py
```

**Output**: `output/dataset0.h5`

**Lo que hace**:
- Extrae POCA del TAR → `output/POCA_projections/tensor_2D_POCA_*.npy`
- Genera GT desde JSON → `output/ground_truth_2D/tensor_2D_*.npy`
- Emparejaambos por metadata
- Crea H5 con split train/val/test (80/10/10)

### 2. Verificar Dataset (Antes de Entrenar)

```bash
# Inspeccionar estructura y visualizar samples aleatorios
python3 checkear_instancias_fromh5.py

# Opciones
python3 checkear_instancias_fromh5.py --split train --num 5
python3 checkear_instancias_fromh5.py --split val --sample 0
python3 checkear_instancias_fromh5.py --save ./visualizations/
```

**Lo que hace**:
- Imprime estructura del H5 (número de samples, shapes, ranges)
- Visualiza muestras aleatorias (POCA vs GT lado a lado)
- Estadísticas de cada canal (XY, XZ, YZ)

### 3. Entrenar UNET2D

```bash
python3 train_unet.py
```

**Output**: 
- `output/models/best_model.h5` (mejor según val_loss)
- `output/models/model_final.h5` (modelo final)

**Hiperparámetros** (editar en `train_unet.py`):
- `BATCH_SIZE = 16`
- `LEARNING_RATE = 1e-3`
- `EPOCHS = 50`
- `N_FILTERS = 32`
- `N_LEVELS = 4`

## Configuración de Paths

Todos los scripts usan paths absolutos desde:
```
BASE = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration")
```

**Directorios esperados**:
```
BASE/
├─ data/
│  ├─ datasets/archivos.tar  (OR carpeta con .root files)
│  └─ geometric_configurations_json/  (JSON geometries)
└─ output/                   (Generado automáticamente)
   ├─ POCA_projections/
   ├─ ground_truth_2D/
   ├─ ground_truth_3D/
   ├─ dataset0.h5
   └─ models/
```

### Para usar carpeta local en lugar de TAR:

Edita en `pipeline_data.py`:
```python
# En lugar de:
SOURCE_ROOT = BASE / "data/datasets/archivos.tar"

# Usa:
SOURCE_ROOT = BASE / "data/datasets/root_files"  # Carpeta con .root files
```

La función `extract_poca_from_source()` detecta automáticamente si es TAR o directorio.

**Ventajas de usar carpeta local**:
- Más rápido (sin extraer)
- Más simple (no necesita lógica de TAR)
- Mejor para desarrollo iterativo

## Parámetros Globales

Editar en `pipeline_data.py`:
```python
WORLD_SIZE = 128.0    # Tamaño del mundo en cm
VOXEL_SIZE = 1.0      # Tamaño de cada voxel en cm
```

Resultado: tensor 128×128 (128.0 / 1.0 = 128 bins)

## Troubleshooting

**"No POCA files generated"**
→ Verificar que la source (TAR o carpeta) contiene .root files válidos con columnas `poca_x, poca_y, poca_z, theta`
→ Si usas carpeta, asegúrate que `SOURCE_ROOT` apunta a la carpeta correcta

**"No matching pairs found"**
→ Los metadatos en filenames POCA y GT no coinciden
- POCA: `tensor_2D_POCA_{metadata}.npy`
- GT: `tensor_2D_{metadata}.npy`
→ Deben tener el mismo `{metadata}` antes de `_spacing`

**"h5py not installed"**
```bash
pip install h5py
```

**"TensorFlow not found"**
```bash
pip install tensorflow
```

**"PyROOT not found" (al extraer POCA)**
→ Si necesitas extraer POCA desde .root, instala:
```bash
pip install uproot  # Alternativa más simple (no requiere ROOT)
# O instala Geant4 con soporte PyROOT
```

## Notes

- Los scripts son idempotentes: si ejecutas `pipeline_data.py` dos veces, detecta archivos ya creados y los salta
- H5 usa compresión gzip (compression_opts=4) para economizar espacio
- Train/val/test split es 80/10/10 (configurable en `create_h5_dataset()`)
- POCA y GT se normalizan a [0, 1] en training (dividir por 255)

## Scripts Eliminados

Ya no necesitas:
- `test/extract_poca_projections_from_tar.py` (en `data_utils.py`)
- `test/create_ground_truth_from_json.py` (en `data_utils.py`)
- `UNETs/UNET0.py` (reemplazado por `pipeline_data.py` + `train_unet.py`)

Todo consolidado en 3 scripts principales + 1 de utilidades.
