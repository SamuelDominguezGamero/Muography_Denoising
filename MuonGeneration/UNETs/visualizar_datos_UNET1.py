#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualizar_datos_UNET1.py

Lee archivos .npy generados para UNET1 y los visualiza como mapas de calor.

Soporta dos formatos:
    1. merge_results_1.py: 5 canales separados (counts_all_2d, counts_scat_2d, etc.)
    2. training_channels: 4 canales UNET1 (log_counts, weighted_mean_z, std_z, top20_theta_sq)

USAGE:
    python3 visualizar_datos_UNET1.py <npy_file> [<npy_file2> ...]
    
    O sin argumentos para procesar todos los MERGED_*.npy en merged_poca_data/:
    python3 visualizar_datos_UNET1.py

"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

# Control de visualización
save = True  # Si True, guarda PNGs; si False, solo muestra en pantalla
environment = "local"  # "local" o "cluster" (detecta automáticamente si es cluster)

# Nombres descriptivos de canales (para ambos formatos)
CHANNEL_NAMES_4 = [
    "Log Counts (event density)",
    "Weighted Mean Z",
    "Std(Z) (thickness proxy)",
    "Top-20 θ² (scattering power)"
]

CHANNEL_NAMES_5 = [
    "Counts All (MTR — body shape)",
    "Counts Scattered",
    "Mean θ² (scattering density)",
    "Top-3 θ² (radiation length proxy)",
    "Std(Z) (thickness proxy)"
]

# Nombres cortos
CHANNEL_SHORT_4 = ["log_counts", "weighted_z", "std_z", "top20_theta2"]
CHANNEL_SHORT_5 = ["counts_all", "counts_scat", "mean_theta2", "top3_theta2", "std_z"]

# Colormaps para cada canal
COLORMAPS_4 = ["viridis", "plasma", "inferno", "magma"]
COLORMAPS_5 = ["viridis", "plasma", "inferno", "magma", "coolwarm"]

# Directorio de salida
OUTPUT_DIR = "./visualizaciones_UNET1"

# Detectar si es cluster
if "SLURM_JOB_ID" in os.environ or "SGE_TASK_ID" in os.environ:
    environment = "cluster"

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_npy_file(filepath):
    """
    Carga un archivo .npy y extrae los canales.
    
    Soporta dos formatos:
    1. merge_results_1.py: 5 canales separados
    2. training_channels: 4 canales UNET1
    
    Retorna:
        channels : ndarray of shape (H, W, N_channels)
        n_channels : int, número de canales
        success : bool
    """
    try:
        data = np.load(filepath, allow_pickle=True).item()
        
        # Intenta primero cargar formato con training_channels (4 canales UNET1)
        if "training_channels" in data:
            channels = data["training_channels"]  # Shape: (H, W, 4)
            if channels.ndim == 3 and channels.shape[2] == 4:
                print(f"  ✓ Loaded 4-channel format (training_channels)")
                return channels.astype(np.float32), 4, True
        
        # Intenta cargar formato de merge_results_1.py (5 canales)
        expected_keys = [
            "counts_all_2d",
            "counts_scat_2d",
            "mean_theta_sq_2d",
            "top3_theta_sq_2d",
            "std_z_2d"
        ]
        
        if all(key in data for key in expected_keys):
            channels = np.stack([
                data["counts_all_2d"][:, :, 0],
                data["counts_scat_2d"][:, :, 0],
                data["mean_theta_sq_2d"][:, :, 0],
                data["top3_theta_sq_2d"][:, :, 0],
                data["std_z_2d"][:, :, 0]
            ], axis=2)  # Shape: (H, W, 5)
            print(f"  ✓ Loaded 5-channel format (merge_results_1.py)")
            return channels.astype(np.float32), 5, True
        
        # Intenta cargar formato alternativo con channel_0, channel_1, etc.
        channel_keys = [k for k in data.keys() if k.startswith("channel_")]
        if channel_keys:
            channel_keys_sorted = sorted(channel_keys, key=lambda x: int(x.split('_')[1]))
            channels = np.stack([
                data[k][:, :, 0] if data[k].ndim == 3 else data[k]
                for k in channel_keys_sorted
            ], axis=2)
            n_ch = len(channel_keys_sorted)
            print(f"  ✓ Loaded {n_ch}-channel format (individual channel keys)")
            return channels.astype(np.float32), n_ch, True
        
        print(f"  ❌ Unknown file format")
        print(f"     Available keys: {list(data.keys())[:10]}...")
        return None, 0, False
        
    except Exception as e:
        print(f"  ❌ Error loading {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, False


def get_channel_config(n_channels):
    """Retorna los nombres, colormaps y short names según el número de canales."""
    if n_channels == 4:
        return CHANNEL_NAMES_4, COLORMAPS_4, CHANNEL_SHORT_4
    elif n_channels == 5:
        return CHANNEL_NAMES_5, COLORMAPS_5, CHANNEL_SHORT_5
    else:
        # Nombres genéricos
        names = [f"Channel {i}" for i in range(n_channels)]
        shorts = [f"ch{i}" for i in range(n_channels)]
        cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdYlBu', 'hsv']
        cmaps = cmaps * ((n_channels // len(cmaps)) + 1)
        return names, cmaps[:n_channels], shorts


def visualize_channels_combined(channels, output_basename):
    """
    Visualiza todos los canales en una sola figura (grid layout).
    
    Parámetros:
        channels : ndarray of shape (H, W, N)
        output_basename : str, nombre base para guardar figura
    """
    n_channels = channels.shape[2]
    names, cmaps, shorts = get_channel_config(n_channels)
    
    # Determinar layout: 1 fila para 4 canales, 2 filas para 5 canales
    if n_channels <= 4:
        nrows, ncols = 1, n_channels
        figsize = (5*n_channels, 5)
    else:
        nrows = 2
        ncols = (n_channels + 1) // 2  # Redondear hacia arriba
        figsize = (5*ncols, 5*nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_channels == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Título principal
    fig.suptitle(f"{output_basename}", fontsize=14, fontweight='bold')
    
    for ch_idx in range(n_channels):
        ax = axes[ch_idx]
        channel_data = channels[:, :, ch_idx]
        
        # Crear mapa de calor
        im = ax.imshow(
            channel_data,
            cmap=cmaps[ch_idx],
            origin='lower',
            aspect='auto'
        )
        
        ax.set_title(names[ch_idx], fontweight='bold', fontsize=10)
        ax.set_xlabel('X Voxels')
        ax.set_ylabel('Y Voxels')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Value', fontsize=8)
        
        # Estadísticas en el gráfico
        vmin, vmax, vmean = channel_data.min(), channel_data.max(), channel_data.mean()
        stats_text = f"min={vmin:.3f}\nmax={vmax:.3f}\nmean={vmean:.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace')
    
    # Ocultar ejes sobrantes si n_channels no llena toda la grid
    for ch_idx in range(n_channels, len(axes)):
        axes[ch_idx].axis('off')
    
    plt.tight_layout()
    
    if save:
        output_file = os.path.join(OUTPUT_DIR, f"{output_basename}_{n_channels}channels.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved: {output_file}")
        plt.close(fig)
    else:
        if environment == "local":
            plt.show()
        else:
            print(f"  [INFO] No visualization in cluster mode")
            plt.close(fig)


def visualize_channels_individual(channels, output_basename):
    """
    Visualiza cada canal por separado en su propia figura.
    Solo guarda si save=True y está configurado.
    
    Parámetros:
        channels : ndarray of shape (H, W, N)
        output_basename : str, nombre base para guardar figuras
    """
    if not save:
        return  # No hacer nada si save=False
    
    n_channels = channels.shape[2]
    names, cmaps, shorts = get_channel_config(n_channels)
    
    for ch_idx in range(n_channels):
        fig, ax = plt.subplots(figsize=(10, 8))
        channel_data = channels[:, :, ch_idx]
        
        im = ax.imshow(
            channel_data,
            cmap=cmaps[ch_idx],
            origin='lower',
            aspect='auto'
        )
        
        ax.set_title(f"{names[ch_idx]}\n{output_basename}",
                     fontweight='bold', fontsize=12)
        ax.set_xlabel('X Voxels', fontsize=10)
        ax.set_ylabel('Y Voxels', fontsize=10)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Value', fontsize=10)
        
        output_file = os.path.join(OUTPUT_DIR,
                                   f"{output_basename}_ch{ch_idx}_{shorts[ch_idx]}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"    → Saved: {Path(output_file).name}")
        
        plt.close(fig)


def process_file(filepath):
    """Procesa un archivo .npy: lo carga y lo visualiza."""
    basename = Path(filepath).stem
    print(f"\n[PROCESSING] {basename}")
    
    channels, n_channels, success = load_npy_file(filepath)
    if not success or channels is None:
        return
    
    print(f"  ✓ Loaded shape: {channels.shape}")
    
    # Obtener nombres de canales
    names, cmaps, shorts = get_channel_config(n_channels)
    
    # Mostrar estadísticas
    print(f"  ✓ Channel statistics:")
    for ch_idx in range(n_channels):
        ch_data = channels[:, :, ch_idx]
        print(f"    [{ch_idx}] {shorts[ch_idx]:15s}: "
              f"min={ch_data.min():10.4f} max={ch_data.max():10.4f} "
              f"mean={ch_data.mean():10.4f} nonzero={np.count_nonzero(ch_data)}")
    
    # Visualizar — primero figura combinada
    visualize_channels_combined(channels, basename)
    
    # Luego figuras individuales (solo si save=True)
    if save:
        visualize_channels_individual(channels, basename)


# ============================================================================
# MAIN
# ============================================================================

def main():
    global OUTPUT_DIR, save, environment
    
    parser = argparse.ArgumentParser(
        description="Visualiza archivos .npy con canales UNET1 como mapas de calor."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Archivos .npy a procesar. Si no se especifican, busca MERGED_*.npy en merged_poca_data/"
    )
    parser.add_argument(
        "-o", "--output",
        default=OUTPUT_DIR,
        help=f"Directorio de salida (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Guardar archivos PNG (default: True)"
    )
    parser.add_argument(
        "--no-save",
        action="store_false",
        dest="save",
        help="No guardar archivos PNG, solo mostrar en pantalla"
    )
    parser.add_argument(
        "--env",
        choices=["local", "cluster"],
        default=environment,
        help="Entorno: 'local' muestra figuras con plt.show(), 'cluster' no las muestra"
    )
    
    args = parser.parse_args()
    
    # Actualizar variables globales
    OUTPUT_DIR = args.output
    save = args.save
    environment = args.env
    
    # Crear directorio de salida solo si save=True
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"[INFO] Output directory: {OUTPUT_DIR}")
    
    print(f"[INFO] Configuration:")
    print(f"       save = {save}")
    print(f"       environment = {environment}")
    
    # Determinar archivos a procesar
    files_to_process = []
    
    if args.files:
        # Usar archivos especificados
        files_to_process = args.files
    else:
        # Buscar MERGED_*.npy en merged_poca_data/
        pattern = "./merged_poca_data/MERGED_*.npy"
        files_to_process = sorted(glob.glob(pattern))
        
        if not files_to_process:
            print(f"[ERROR] No files found matching: {pattern}")
            print("[INFO] Usage:")
            print("  python3 visualizar_datos_UNET1.py <file1.npy> [<file2.npy> ...]")
            print("  python3 visualizar_datos_UNET1.py  # busca en merged_poca_data/")
            print("\nOptions:")
            print("  --save              : Guardar PNGs (default)")
            print("  --no-save           : Solo mostrar en pantalla, sin guardar")
            print("  --env local/cluster : Entorno (default: auto-detectado)")
            sys.exit(1)
    
    print(f"[INFO] Found {len(files_to_process)} file(s) to process\n")
    
    # Procesar cada archivo
    for filepath in files_to_process:
        if os.path.isfile(filepath):
            process_file(filepath)
        else:
            print(f"[WARNING] File not found: {filepath}")
    
    print(f"\n{'='*70}")
    print(f"✅ VISUALIZACIÓN COMPLETADA")
    if save:
        print(f"   Archivos guardados en: {OUTPUT_DIR}")
    else:
        print(f"   Modo: mostrar en pantalla (sin guardar)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
