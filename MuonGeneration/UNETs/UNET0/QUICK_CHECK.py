#!/usr/bin/env python3
"""
RESUMEN EJECUTIVO: Validación rápida de la pipeline
Ejecutar primero para verificar que todo está bien
"""

import sys
from pathlib import Path
import json

print("\n" + "=" * 80)
print("VALIDACIÓN RÁPIDA - UNET0 PIPELINE")
print("=" * 80)

# Paths
BASE = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration")
DATA = BASE / "data"

paths = {
    "BASE": BASE,
    "DATA": DATA,
    "JSON_DIR": DATA / "geometric_configurations_json",
    "SOURCE_POCA": DATA / "simulation_data/run0.tar",
    "POCA_OUT": DATA / "simulation_data/run0_POCA_projections",
    "GT_3D": DATA / "ground_truth_data/3Dimensions",
    "GT_2D": DATA / "ground_truth_data/2Dimensions",
    "DATASETS": DATA / "datasets",
    "H5_FILE": DATA / "datasets/dataset0.h5",
    "MODELS": DATA / "models",
}

print("\n[PATHS DEFINIDOS]")
for name, path in paths.items():
    exists = "✓" if path.exists() else "✗"
    status = "exists" if path.exists() else "will create"
    print(f"  {exists} {name:15} → {path.relative_to(BASE)}")

print("\n[INPUT FILES]")
json_files = list((DATA / "geometric_configurations_json").glob("*.json")) if (DATA / "geometric_configurations_json").exists() else []
print(f"  JSON geometries: {len(json_files)} files")

if (DATA / "simulation_data/run0.tar").exists():
    print(f"  POCA source: TAR file (run0.tar)")
elif (DATA / "simulation_data/root_files").exists():
    root_count = len(list((DATA / "simulation_data/root_files").glob("*.root")))
    print(f"  POCA source: Directory ({root_count} .root files)")
else:
    print(f"  POCA source: ⚠️  No TAR or root_files directory found")

print("\n[SCRIPTS PRESENTES]")
script_dir = Path(__file__).parent
scripts = ["data_utils.py", "pipeline_data.py", "check_consistency.py", 
           "checkear_instancias_fromh5.py", "train_unet.py"]

for script in scripts:
    path = script_dir / script
    exists = "✓" if path.exists() else "✗"
    print(f"  {exists} {script}")

print("\n[PRÓXIMOS PASOS]")
print("""
  1. Ejecutar chequeo completo:
     python3 check_consistency.py
  
  2. Si chequeo pasa, generar dataset:
     python3 pipeline_data.py
  
  3. Visualizar muestras (antes de entrenar):
     python3 checkear_instancias_fromh5.py
  
  4. Entrenar modelo:
     python3 train_unet.py
""")

print("=" * 80 + "\n")
