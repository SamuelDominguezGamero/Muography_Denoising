#!/usr/bin/env python3
"""
Visualizar instancias aleatorias del dataset H5
Uso: python3 checkear_instancias_fromh5.py [--split train] [--num 3] [--save]
"""

import argparse
import random
from pathlib import Path
import numpy as np

try:
    import h5py
except ImportError:
    print("[ERROR] h5py required. Install: pip install h5py")
    exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] matplotlib required. Install: pip install matplotlib")
    exit(1)


# ===========================================================================
# CONFIG
# ===========================================================================

BASE = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration")
DATA = BASE / "data"
DATASETS = DATA / "datasets"

H5_FILE = DATASETS / "dataset0.h5"


# ===========================================================================
# INSPECTION
# ===========================================================================

def inspect_h5(h5_path):
    """Print H5 structure and statistics"""
    print("\n" + "=" * 80)
    print(f"H5 FILE: {h5_path.name}")
    print("=" * 80)
    
    with h5py.File(h5_path, 'r') as f:
        for split in ['train', 'val', 'test']:
            if split not in f:
                continue
            
            grp = f[split]
            n_poca = grp['poca'].shape[0]
            n_gt = grp['gt'].shape[0]
            
            poca_data = grp['poca']
            gt_data = grp['gt']
            
            print(f"\n[{split.upper()}]")
            print(f"  Samples: {n_poca}")
            print(f"  POCA shape: {poca_data.shape} | dtype: {poca_data.dtype}")
            print(f"  GT shape:   {gt_data.shape} | dtype: {gt_data.dtype}")
            print(f"  POCA range: [{poca_data[0].min()}, {poca_data[0].max()}]")
            print(f"  GT range:   [{gt_data[0].min()}, {gt_data[0].max()}]")
    
    print("\n" + "=" * 80)


def visualize_sample(h5_path, split='train', sample_idx=None, save_path=None):
    """Visualize a random or specific sample"""
    
    with h5py.File(h5_path, 'r') as f:
        if split not in f:
            print(f"[ERROR] Split '{split}' not found in H5")
            return
        
        grp = f[split]
        n_samples = grp['poca'].shape[0]
        
        if sample_idx is None:
            sample_idx = random.randint(0, n_samples - 1)
        elif sample_idx >= n_samples:
            print(f"[ERROR] Sample {sample_idx} out of range [0, {n_samples-1}]")
            return
        
        # Load sample
        poca = grp['poca'][sample_idx]
        gt = grp['gt'][sample_idx]
    
    # Create figure: 2 rows (POCA, GT) × 3 cols (XY, XZ, YZ)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Dataset Sample #{sample_idx} from {split.upper()} split\n"
        f"POCA (top) vs Ground Truth (bottom)",
        fontsize=14, fontweight='bold'
    )
    
    titles = ['XY Projection\n(view from Z)', 'XZ Projection\n(view from Y)', 'YZ Projection\n(view from X)']
    
    # POCA row (top)
    for i in range(3):
        im = axes[0, i].imshow(poca[:, :, i], cmap='hot', origin='lower')
        axes[0, i].set_title(f'POCA {titles[i]}', fontweight='bold', color='darkred', fontsize=11)
        axes[0, i].set_xlabel('Horizontal [bin]')
        axes[0, i].set_ylabel('Vertical [bin]')
        plt.colorbar(im, ax=axes[0, i], label='Density (0-255)')
    
    # GT row (bottom)
    for i in range(3):
        im = axes[1, i].imshow(gt[:, :, i], cmap='gray', origin='lower')
        axes[1, i].set_title(f'GT {titles[i]}', fontweight='bold', color='darkgreen', fontsize=11)
        axes[1, i].set_xlabel('Horizontal [bin]')
        axes[1, i].set_ylabel('Vertical [bin]')
        plt.colorbar(im, ax=axes[1, i], label='Occupancy (0-1)')
    
    plt.tight_layout()
    
    # Print statistics
    print(f"\n[SAMPLE {sample_idx}] Statistics:")
    print(f"  POCA shape: {poca.shape}")
    print(f"    XY channel: min={poca[:,:,0].min()}, max={poca[:,:,0].max()}, mean={poca[:,:,0].mean():.1f}")
    print(f"    XZ channel: min={poca[:,:,1].min()}, max={poca[:,:,1].max()}, mean={poca[:,:,1].mean():.1f}")
    print(f"    YZ channel: min={poca[:,:,2].min()}, max={poca[:,:,2].max()}, mean={poca[:,:,2].mean():.1f}")
    
    print(f"  GT shape: {gt.shape}")
    print(f"    XY channel: occupancy={100*(gt[:,:,0]>0).sum()/gt[:,:,0].size:.2f}%")
    print(f"    XZ channel: occupancy={100*(gt[:,:,1]>0).sum()/gt[:,:,1].size:.2f}%")
    print(f"    YZ channel: occupancy={100*(gt[:,:,2]>0).sum()/gt[:,:,2].size:.2f}%")
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  ✓ Saved: {save_path}")
    else:
        plt.show()


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Inspect H5 training dataset")
    parser.add_argument("--split", type=str, default='train', 
                       choices=['train', 'val', 'test'],
                       help="Dataset split to inspect (default: train)")
    parser.add_argument("--num", type=int, default=3, 
                       help="Number of samples to visualize (default: 3)")
    parser.add_argument("--sample", type=int, default=None,
                       help="Specific sample index (default: random)")
    parser.add_argument("--save", type=str, default=None,
                       help="Save visualizations to directory")
    
    args = parser.parse_args()
    
    if not H5_FILE.exists():
        print(f"[ERROR] H5 file not found: {H5_FILE}")
        exit(1)
    
    # Inspect structure
    inspect_h5(H5_FILE)
    
    # Visualize samples
    print(f"\nVisualizing {args.num} sample(s) from '{args.split}' split...")
    
    for i in range(args.num):
        sample_idx = args.sample if args.sample is not None else None
        save_path = None
        
        if args.save:
            save_dir = Path(args.save)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"sample_{i:02d}_{args.split}.png"
        
        visualize_sample(H5_FILE, args.split, sample_idx, save_path)
        print()


if __name__ == "__main__":
    main()
