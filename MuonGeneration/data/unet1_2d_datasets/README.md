# UNET1_2D Datasets Directory

## Purpose
HDF5 datasets for UNET1_2D training. Contains 4-channel POCA reconstructions + ground truth.

## Main Files
- **128x128x4.h5**: Primary training dataset
  - Input: 4-channel POCA (log_counts, mean_theta_sq_z, var_poca_z, std_theta)
  - Label: Ground truth density (128, 128, 1)
  - Splits: train (70%), validation (15%), test (15%)
  - Size: varies based on simulation count

- **128x128x4_augmented.h5**: Augmented version (auto-generated)
  - 16x expanded via rotations + flips
  - Same samples with transformations tracked
  - Used for more robust training

## Creation Process
```
create_h5_dataset1.py reads:
├─ MERGED_*_2D.npy (from merge_results_1.py - 4 channels)
└─ *_ground_truth_density2D.npy (ground truth masks)

Creates:
└─ 128x128x4.h5
```

## File Attributes (in HDF5)
- `n_muons_min`, `n_muons_max`, `n_muons_avg`: Noise level statistics
- `has_mixed_muon_counts`: True if variable-noise dataset
- `train_ratio`, `val_ratio`, `test_ratio`: Split proportions

## Per-Sample Metadata
Each training sample includes:
- `n_muons_total`: Exact muon count (affects noise)
- `material`: lead, iron, or uranium
- `word`: geometry type
- Geometry parameters: Lpx, Lpy, npx, npy, etc.
