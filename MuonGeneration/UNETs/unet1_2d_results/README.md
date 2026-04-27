# UNET1_2D Results Directory

## Purpose
Output location for training visualizations and analysis results from UNET1_2D training.

## Contents
- **prediction_comparisons/**: Input | Prediction | Ground Truth visualizations
- **channel_visualizations/**: Per-channel normalized data before/after
- **augmentation_samples/**: Data augmentation examples (16x: 4 rotations × 4 flips)
- **training_curves/**: Loss and metric plots over epochs
- **sample_metadata/**: CSV/JSON files with model predictions on test set

## File Naming
- `predictions_{model_name}_{timestamp}.png`
- `normalized_channels_{model_name}.png`
- `augmentation_demo_{model_name}.png`

## Organization
Results are automatically timestamped and organized by model configuration.
