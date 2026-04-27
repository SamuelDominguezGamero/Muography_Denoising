#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para visualizar los 4 canales POCA como heatmaps.
Genera mapas de calor para cada canal de cada archivo MERGED.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATTERN = os.path.join(SCRIPT_DIR, "MERGED_*_2D.npy")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "heatmaps")

# Nombres de canales POCA
CHANNEL_NAMES = [
    "Canal 0: log(1+N) - Event Counts",
    "Canal 1: Weighted Mean(Z, θ²) - Z Scattering",
    "Canal 2: Std(Z) - Thickness Proxy",
    "Canal 3: Mean(Top-20 θ²) - Scattering Power"
]

CHANNEL_SHORT = ["counts", "weighted_z", "std_z", "scattering_power"]
COLORMAPS = ["viridis", "plasma", "inferno", "magma"]

# ============================================================================
# CREAR DIRECTORIO DE OUTPUT
# ============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[INFO] Output directory: {OUTPUT_DIR}")

# ============================================================================
# ENCONTRAR ARCHIVOS
# ============================================================================
files = glob.glob(INPUT_PATTERN)
if not files:
    print(f"[ERROR] No MERGED files found matching: {INPUT_PATTERN}")
    exit(1)

print(f"[INFO] Found {len(files)} MERGED files")

# ============================================================================
# PROCESAR CADA ARCHIVO
# ============================================================================
for filepath in sorted(files):
    basename = Path(filepath).stem
    print(f"\n[PROCESSING] {basename}")
    
    # Cargar datos
    try:
        data_dict = np.load(filepath, allow_pickle=True).item()
        channels = data_dict["training_channels"]  # Shape: (H, W, 4)
    except Exception as e:
        print(f"  [ERROR] Failed to load: {e}")
        continue
    
    if channels.shape[2] != 4:
        print(f"  [ERROR] Expected 4 channels, got {channels.shape[2]}")
        continue
    
    H, W = channels.shape[:2]
    print(f"  [INFO] Shape: {H}x{W}x4")
    
    # Crear figura con 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"POCA 4-Channel Visualization\n{basename}", fontsize=14, fontweight='bold')
    
    # Procesar cada canal
    for ch_idx in range(4):
        ax = axes[ch_idx // 2, ch_idx % 2]
        channel_data = channels[:, :, ch_idx]
        
        # Crear heatmap
        im = ax.imshow(channel_data, cmap=COLORMAPS[ch_idx], origin='lower', aspect='auto')
        ax.set_title(CHANNEL_NAMES[ch_idx], fontweight='bold')
        ax.set_xlabel('X Voxels')
        ax.set_ylabel('Y Voxels')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        
        # Estadísticas
        vmin, vmax = np.min(channel_data), np.max(channel_data)
        vmean = np.mean(channel_data)
        stats_text = f"Min: {vmin:.3f}\nMax: {vmax:.3f}\nMean: {vmean:.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9, family='monospace')
        
        print(f"  [CHANNEL {ch_idx}] {CHANNEL_SHORT[ch_idx]}: min={vmin:.4f}, max={vmax:.4f}, mean={vmean:.4f}")
    
    plt.tight_layout()
    
    # Guardar figura
    output_file = os.path.join(OUTPUT_DIR, f"{basename}_heatmap.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {output_file}")
    
    # También guardar cada canal por separado
    for ch_idx in range(4):
        fig_single, ax = plt.subplots(figsize=(8, 7))
        channel_data = channels[:, :, ch_idx]
        
        im = ax.imshow(channel_data, cmap=COLORMAPS[ch_idx], origin='lower', aspect='auto')
        ax.set_title(f"{CHANNEL_NAMES[ch_idx]}\n{basename}", fontweight='bold')
        ax.set_xlabel('X Voxels')
        ax.set_ylabel('Y Voxels')
        plt.colorbar(im, ax=ax, label='Value')
        
        channel_file = os.path.join(OUTPUT_DIR, f"{basename}_ch{ch_idx}_{CHANNEL_SHORT[ch_idx]}.png")
        plt.savefig(channel_file, dpi=150, bbox_inches='tight')
        plt.close(fig_single)
        print(f"    → Saved: {Path(channel_file).name}")
    
    plt.close(fig)

print(f"\n{'='*70}")
print(f"✅ VISUALIZACIÓN COMPLETADA")
print(f"   Archivos guardados en: {OUTPUT_DIR}")
print(f"{'='*70}")
