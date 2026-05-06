"""
SUBSAMPLEAR INSTANCIAS DE ENTRENAMIENTO
- Carga archivos .npy mergeados (ya proyectados)
- Subsamplea a N muones aleatorios
- Guarda con nuevo nombre sin sobrescribir originales
"""

import os
import sys
import glob
import numpy as np
from pathlib import Path
import argparse

# ===========================================================================
# ARGUMENTOS DE LÍNEA DE COMANDOS
# ===========================================================================
parser = argparse.ArgumentParser(
    description="Subsample merged POCA data files"
)
parser.add_argument(
    "--path_merged_output",
    type=str,
    required=True,
    help="Path to directory containing MERGED_*.npy files"
)
parser.add_argument(
    "--target_muons",
    type=int,
    required=True,
    help="Target number of muons per file (e.g., 1000000)"
)
parser.add_argument(
    "--dimension",
    type=str,
    default="2D",
    choices=["2D", "3D"],
    help="Dimension of data (2D or 3D)"
)
parser.add_argument(
    "--source_muons",
    type=int,
    default=6000000,
    help="Original muon count in source files (default: 6000000)"
)
parser.add_argument(
    "--pattern",
    type=str,
    default=None,
    help="Optional: process only files matching this pattern (e.g., 'empty_config')"
)

args = parser.parse_args()

# ===========================================================================
# VALIDACIÓN
# ===========================================================================
if not os.path.isdir(args.path_merged_output):
    sys.exit(f"[ERROR] Directory not found: {args.path_merged_output}")

if args.target_muons <= 0:
    sys.exit(f"[ERROR] target_muons must be positive, got {args.target_muons}")

if args.source_muons <= 0:
    sys.exit(f"[ERROR] source_muons must be positive, got {args.source_muons}")

# ===========================================================================
# BÚSQUEDA DE ARCHIVOS
# ===========================================================================
print(f"[INFO] Searching for merged files in: {args.path_merged_output}")
print(f"[INFO] Looking for: MERGED_*_Muons_{args.source_muons}_{args.dimension}.npy")

all_merged_files = glob.glob(
    os.path.join(args.path_merged_output, f"MERGED_*_Muons_{args.source_muons}_{args.dimension}.npy")
)

if args.pattern:
    all_merged_files = [f for f in all_merged_files if args.pattern in f]
    print(f"[INFO] Filtered by pattern '{args.pattern}': {len(all_merged_files)} files")
else:
    print(f"[INFO] Found {len(all_merged_files)} files to subsample")

if len(all_merged_files) == 0:
    print("[WARNING] No files found matching criteria. Exiting.")
    sys.exit(0)

# ===========================================================================
# SUBSAMPLEAR CADA ARCHIVO
# ===========================================================================
files_processed = 0
files_skipped = 0
total_size_original_gb = 0
total_size_subsampled_gb = 0

for filepath in sorted(all_merged_files):
    filename = os.path.basename(filepath)
    namefile = filename.replace(f"MERGED_", "").replace(f"_Muons_{args.source_muons}_{args.dimension}.npy", "")
    
    print(f"\n[INFO] Processing: {namefile}")
    
    try:
        # Cargar datos
        print(f"  ├─ Loading {filename}...")
        data = np.load(filepath)
        original_size = data.shape[0]
        original_size_gb = (data.nbytes / 1e9)
        print(f"  ├─ Original shape: {data.shape}")
        print(f"  ├─ Original size: {original_size_gb:.2f} GB")
        
        total_size_original_gb += original_size_gb
        
        # Subsamplear
        if original_size > args.target_muons:
            print(f"  ├─ Subsampling to {args.target_muons:,} muons...")
            indices = np.random.choice(original_size, size=args.target_muons, replace=False)
            data_subset = data[indices]
            
            subsampled_size_gb = (data_subset.nbytes / 1e9)
            print(f"  ├─ Subsampled shape: {data_subset.shape}")
            print(f"  ├─ Subsampled size: {subsampled_size_gb:.2f} GB")
            total_size_subsampled_gb += subsampled_size_gb
            
            # Guardar con nuevo nombre
            output_file = os.path.join(
                args.path_merged_output,
                f"MERGED_{namefile}_Muons_{args.target_muons}_{args.dimension}.npy"
            )
            np.save(output_file, data_subset)
            print(f"  └─ Saved: {Path(output_file).name}")
            files_processed += 1
            
        else:
            print(f"  └─ File has {original_size:,} muons (< {args.target_muons:,}), skipping")
            files_skipped += 1
            
        # Liberar memoria
        del data
        if original_size > args.target_muons:
            del data_subset
        
    except Exception as e:
        print(f"  └─ [ERROR] Failed to process: {e}")
        files_skipped += 1

# ===========================================================================
# RESUMEN FINAL
# ===========================================================================
print("\n" + "="*70)
print("[FINAL SUMMARY]")
print("="*70)
print(f"[INFO] Configuration:")
print(f"       Source muons per file      = {args.source_muons:,}")
print(f"       Target muons per file      = {args.target_muons:,}")
print(f"       Dimension                  = {args.dimension}")
print(f"[INFO] Results:")
print(f"       Files processed            = {files_processed}")
print(f"       Files skipped              = {files_skipped}")
print(f"       Original total size        = {total_size_original_gb:.2f} GB")
print(f"       Subsampled total size      = {total_size_subsampled_gb:.2f} GB")
if files_processed > 0:
    print(f"       Compression ratio          = {(total_size_subsampled_gb / total_size_original_gb * 100):.1f}%")
print(f"[INFO] Output location:")
print(f"       {args.path_merged_output}")
print("="*70)
