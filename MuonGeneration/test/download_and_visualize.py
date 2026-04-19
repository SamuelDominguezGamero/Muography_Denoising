#!/usr/bin/env python3
"""
download_and_visualize.py

Descarga archivos MERGED*.npy del cluster a local usando scp,
luego visualiza los mapas 2D del POCA para verificar sentido físico.

Flujo:
1. Descarga archivos con scp
2. Carga diccionarios con allow_pickle=True
3. Visualiza 3 canales: log_counts, mean_theta_sq_z, var_theta_z
4. Muestra estadísticas básicas
"""

import subprocess
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import getpass
import shutil

# ===========================================================================
# CONFIGURATION
# ===========================================================================
CLUSTER_USER = "dominguezs"
CLUSTER_HOST = "login2.ifca.es"
CLUSTER_PATH = f"/gpfs/users/{CLUSTER_USER}/Muography_Denoising/MuonGeneration/data/merged_poca_data"

LOCAL_BASE = Path.home() / "Work/Muography_Denoising/MuonGeneration/data"
LOCAL_DOWNLOAD = LOCAL_BASE / "visual_testing"
LOCAL_DOWNLOAD.mkdir(parents=True, exist_ok=True)

max_geometries = 5  # Limitar a N geometrías para no descargar todo

# ===========================================================================
# STEP 1: GET SSH PASSWORD
# ===========================================================================
print("="*70)
print("DESCARGADOR DE MAPAS MERGED - POCA VISUALIZATION")
print("="*70)
print()
print(f"[INFO] Remote: {CLUSTER_USER}@{CLUSTER_HOST}:{CLUSTER_PATH}")
print(f"[INFO] Local:  {LOCAL_DOWNLOAD}")
print(f"[INFO] Max geometrías a descargar: {max_geometries}")
print()

# Verificar si sshpass está disponible
has_sshpass = shutil.which("sshpass") is not None
if not has_sshpass:
    print("[WARNING] sshpass no está instalado. Se pedirá contraseña para cada archivo.")
    print("[TIP] Instala con: sudo apt-get install sshpass")
    print()

ssh_password = getpass.getpass("Introduce la contraseña del SSH: ")
if not ssh_password:
    print("[ERROR] Contraseña requerida.")
    sys.exit(1)

print()

# ===========================================================================
# STEP 2: LISTAR ARCHIVOS EN CLUSTER
# ===========================================================================
print("[INFO] Escaneando archivos remotos...")

if has_sshpass:
    list_cmd = [
        "sshpass", "-p", ssh_password,
        "ssh", "-T", "-o", "BatchMode=yes", f"{CLUSTER_USER}@{CLUSTER_HOST}",
        f"ls -1 {CLUSTER_PATH}/MERGED_*.npy 2>/dev/null | head -100"
    ]
else:
    list_cmd = [
        "ssh", "-T", "-o", "BatchMode=yes", f"{CLUSTER_USER}@{CLUSTER_HOST}",
        f"ls -1 {CLUSTER_PATH}/MERGED_*.npy 2>/dev/null | head -100"
    ]

try:
    result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=10)
    remote_files = result.stdout.strip().split('\n')
    remote_files = [f for f in remote_files if f]  # Filtrar líneas vacías
    print(f"[SUCCESS] Encontrados {len(remote_files)} archivos en cluster")
except Exception as e:
    print(f"[ERROR] Fallo al listar archivos remotos: {e}")
    sys.exit(1)

if not remote_files:
    print("[ERROR] No hay archivos MERGED en el cluster.")
    sys.exit(1)

# Limitar a max_geometries
remote_files = remote_files[:max_geometries]
print(f"[INFO] Descargando {len(remote_files)} archivo(s)...\n")

# ===========================================================================
# STEP 3: DESCARGAR ARCHIVOS
# ===========================================================================
downloaded_files = []

for i, remote_file in enumerate(remote_files, 1):
    filename = os.path.basename(remote_file)
    local_file = LOCAL_DOWNLOAD / filename
    
    if local_file.exists():
        print(f"[{i}/{len(remote_files)}] {filename} (YA EXISTE, saltando)")
        downloaded_files.append(str(local_file))
        continue
    
    print(f"[{i}/{len(remote_files)}] Descargando {filename}...", end=" ", flush=True)
    
    if has_sshpass:
        scp_cmd = [
            "sshpass", "-p", ssh_password,
            "scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            f"{CLUSTER_USER}@{CLUSTER_HOST}:{remote_file}",
            str(local_file)
        ]
    else:
        scp_cmd = [
            "scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            f"{CLUSTER_USER}@{CLUSTER_HOST}:{remote_file}",
            str(local_file)
        ]
    
    try:
        result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            downloaded_files.append(str(local_file))
            size_mb = local_file.stat().st_size / (1024**2)
            print(f"✓ ({size_mb:.2f} MB)")
        else:
            print(f"✗ Error: {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        print("✗ Timeout")
    except Exception as e:
        print(f"✗ {e}")

if not downloaded_files:
    print("[ERROR] No se pudo descargar ningún archivo.")
    sys.exit(1)

print(f"\n[SUCCESS] {len(downloaded_files)} archivo(s) descargado(s)")
print()

# ===========================================================================
# STEP 4: CARGAR Y VISUALIZAR
# ===========================================================================
print("="*70)
print("VISUALIZACIÓN DE MAPAS POCA")
print("="*70)
print()

for local_file in downloaded_files:
    filename = os.path.basename(local_file)
    print(f"\n{'='*70}")
    print(f"Archivo: {filename}")
    print(f"{'='*70}")
    
    try:
        # Cargar diccionario
        data = np.load(local_file, allow_pickle=True).item()
        print(f"[INFO] Claves disponibles: {list(data.keys())}")
        
        # Extraer datos
        log_counts = data.get("log_counts")
        mean_theta_sq = data.get("mean_theta_sq_z")
        var_theta = data.get("var_theta_z")
        
        if log_counts is None:
            print("[WARNING] No se encontró 'log_counts' en el archivo")
            continue
        
        # Remover dimensión Z (last dimension es 1)
        log_counts_2d = log_counts[:, :, 0] if log_counts.ndim == 3 else log_counts
        mean_theta_sq_2d = mean_theta_sq[:, :, 0] if mean_theta_sq.ndim == 3 else mean_theta_sq
        var_theta_2d = var_theta[:, :, 0] if var_theta.ndim == 3 else var_theta
        
        # Estadísticas
        print(f"\n[STATS] log_counts:")
        print(f"  Shape: {log_counts_2d.shape}")
        print(f"  Min: {np.min(log_counts_2d):.4f} | Max: {np.max(log_counts_2d):.4f}")
        print(f"  Mean: {np.mean(log_counts_2d):.4f} | Median: {np.median(log_counts_2d):.4f}")
        print(f"  Non-zero cells: {np.sum(log_counts_2d > 0)} / {log_counts_2d.size}")
        
        print(f"\n[STATS] mean_theta_sq_z:")
        print(f"  Min: {np.min(mean_theta_sq_2d[mean_theta_sq_2d > 0]):.6f} | Max: {np.max(mean_theta_sq_2d):.6f}")
        print(f"  Mean (non-zero): {np.mean(mean_theta_sq_2d[mean_theta_sq_2d > 0]):.6f}")
        
        print(f"\n[STATS] var_theta_z:")
        print(f"  Min: {np.min(var_theta_2d[var_theta_2d > 0]):.6f} | Max: {np.max(var_theta_2d):.6f}")
        print(f"  Mean (non-zero): {np.mean(var_theta_2d[var_theta_2d > 0]):.6f}")
        
        # VISUALIZACIÓN
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(filename, fontsize=14, fontweight='bold')
        
        # Canal 1: log_counts (estadística - debe mostrar geometría)
        im0 = axes[0].imshow(log_counts_2d, cmap="viridis", origin="lower", aspect="auto")
        axes[0].set_title("log_counts\n(estadística: geometría se ve aquí)", fontsize=11)
        axes[0].set_xlabel("X (npx)")
        axes[0].set_ylabel("Y (npy)")
        cbar0 = plt.colorbar(im0, ax=axes[0])
        cbar0.set_label("log(N+1)")
        
        # Canal 2: mean_theta_sq_z (ángulo promedio)
        im1 = axes[1].imshow(np.clip(mean_theta_sq_2d, 0, np.percentile(mean_theta_sq_2d[mean_theta_sq_2d > 0], 95)), 
                             cmap="plasma", origin="lower", aspect="auto")
        axes[1].set_title("<θ²>_z\n(ángulo cuadrático medio)", fontsize=11)
        axes[1].set_xlabel("X (npx)")
        axes[1].set_ylabel("Y (npy)")
        cbar1 = plt.colorbar(im1, ax=axes[1])
        cbar1.set_label("Radianes²")
        
        # Canal 3: var_theta (varianza)
        im2 = axes[2].imshow(np.clip(var_theta_2d, 0, np.percentile(var_theta_2d[var_theta_2d > 0], 95)), 
                             cmap="hot", origin="lower", aspect="auto")
        axes[2].set_title("Var(θ)_z\n(varianza del ángulo)", fontsize=11)
        axes[2].set_xlabel("X (npx)")
        axes[2].set_ylabel("Y (npy)")
        cbar2 = plt.colorbar(im2, ax=axes[2])
        cbar2.set_label("Radianes²")
        
        plt.tight_layout()
        
        # Guardar figura
        output_png = Path(local_file).with_suffix(".png")
        plt.savefig(output_png, dpi=100, bbox_inches='tight')
        print(f"\n[SUCCESS] Figura guardada: {output_png.name}")
        
        plt.show()
        
        # ANÁLISIS FÍSICO
        print(f"\n[ANÁLISIS FÍSICO]")
        non_zero_mask = log_counts_2d > 0
        
        if np.sum(non_zero_mask) > 0:
            mean_counts = np.mean(log_counts_2d[non_zero_mask])
            max_counts = np.max(log_counts_2d)
            
            print(f"  ✓ Se detectó actividad en {np.sum(non_zero_mask)} celdas ({100*np.sum(non_zero_mask)/log_counts_2d.size:.1f}%)")
            print(f"  ✓ Distribución de conteos en rango [0, {max_counts:.2f}]")
            print(f"  ✓ log_counts promedio (células activas): {mean_counts:.2f}")
            
            # Verificar si hay estructura (picos)
            if np.max(log_counts_2d) > 2 * np.mean(log_counts_2d[non_zero_mask]):
                print(f"  ✓ Hay VARIACIÓN significativa (picos) → Estructura geométrica visible")
            else:
                print(f"  ⚠ Distribución muy uniforme → Verificar si geometría coincide con grid")
            
            # Verificar varianza de ángulos
            if np.max(var_theta_2d) > 1e-4:
                print(f"  ✓ Varianza de ángulos no-nula → Dispersión angular detectada")
            else:
                print(f"  ⚠ Varianza muy baja → Verificar si hay scattering suficiente")
        else:
            print(f"  ✗ NO HAY ACTIVIDAD - El archivo puede estar vacío o corrupto")
        
    except Exception as e:
        print(f"[ERROR] Fallo al procesar {filename}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print("[SUCCESS] Visualización completada")
print(f"Archivos guardados en: {LOCAL_DOWNLOAD}")
print("="*70)
