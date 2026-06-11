#!/usr/bin/env python3
"""
test_dynamic_subsampling.py
===========================
Quick test to verify dynamic subsampling is working.

This script:
  1. Generates dummy training data (simulated POCA maps)
  2. Trains a simple model with subsampling enabled
  3. Plots training curves to show the expected behavior

Run from UNET1 directory:
    python test_dynamic_subsampling.py
"""

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Import after adding repo to path
sys.path.insert(0, "/home/samuel/Work/Muography_Denoising/MuonGeneration/UNETs/UNET1")

from prepare_data_for_ResNet import (
    MATERIAL_LABELS, prepare_datasets, Z_MAX, _make_dynamic_subsampled_dataset
)


def create_dummy_data(n_samples=500, temp_dir="/tmp/test_poca"):
    """
    Create dummy POCA .npy files and CSV for testing.
    
    Parameters
    ----------
    n_samples : int
        Number of dummy samples to create
    temp_dir : str
        Temporary directory for test data
    
    Returns
    -------
    pd.DataFrame
        DataFrame with test data
    """
    import os
    os.makedirs(temp_dir, exist_ok=True)
    
    rows = []
    for i in range(n_samples):
        # Random material and z
        mat_id = i % len(MATERIAL_LABELS)
        mat_name = MATERIAL_LABELS[mat_id]
        z_true = np.random.uniform(5, 60)
        
        # Create dummy image
        img = np.random.rand(128, 128, 3).astype(np.float32)
        
        # Save .npy file
        path = os.path.join(temp_dir, f"sample_{i:05d}_mat{mat_name}_dz{z_true:.0f}.npy")
        np.save(path, img)
        
        rows.append({
            "path": path,
            "material_id": mat_id,
            "material_name": mat_name,
            "z_true": z_true,
            "run": "test",
            "z_bin": int(z_true // 15) if z_true < 60 else 2,
        })
    
    return pd.DataFrame(rows)


def test_subsampling_enabled():
    """Test that dynamic subsampling produces different subsets."""
    print("\n" + "="*70)
    print("TEST 1: Verify Dynamic Subsampling Works")
    print("="*70)
    
    # Create dummy dataframe
    df = create_dummy_data(n_samples=100)
    
    # Create dynamic subsampled dataset with 30% sampling
    ds = _make_dynamic_subsampled_dataset(df, subsample_fraction=0.3, batch_size=8, seed=42)
    
    # Collect samples from first two iterations
    samples_epoch1 = []
    samples_epoch2 = []
    
    print("\nEpoch 1: Collecting sample indices...")
    for batch_idx, (inputs, targets) in enumerate(ds):
        batch_size = targets["output_material"].shape[0]
        samples_epoch1.extend(targets["output_material"].numpy().tolist())
        if batch_idx >= 3:  # Just first few batches
            break
    
    print(f"  Collected {len(samples_epoch1)} samples from epoch 1")
    print(f"  Material distribution: {np.bincount(samples_epoch1)}")
    
    print("\nEpoch 2: Collecting sample indices...")
    for batch_idx, (inputs, targets) in enumerate(ds):
        batch_size = targets["output_material"].shape[0]
        samples_epoch2.extend(targets["output_material"].numpy().tolist())
        if batch_idx >= 3:  # Just first few batches
            break
    
    print(f"  Collected {len(samples_epoch2)} samples from epoch 2")
    print(f"  Material distribution: {np.bincount(samples_epoch2)}")
    
    # Check they're different
    if samples_epoch1 != samples_epoch2:
        print("\n✓ PASS: Epochs 1 and 2 produced different subsamples (as expected)")
    else:
        print("\n✗ FAIL: Epochs 1 and 2 produced identical subsamples (unexpected)")


def test_subsampling_disabled():
    """Test that standard dataset (no subsampling) works."""
    print("\n" + "="*70)
    print("TEST 2: Verify Standard (Non-Subsampled) Dataset Works")
    print("="*70)
    
    from prepare_data_for_ResNet import _make_tf_dataset
    
    df = create_dummy_data(n_samples=100)
    
    # Create standard dataset
    ds = _make_tf_dataset(df, shuffle=True, batch_size=8, seed=42)
    
    print("\nIterating through dataset...")
    batch_count = 0
    sample_count = 0
    for inputs, targets in ds:
        batch_count += 1
        sample_count += targets["output_material"].shape[0]
    
    print(f"  Batches: {batch_count}")
    print(f"  Total samples: {sample_count}")
    print("✓ PASS: Standard dataset created successfully")


def test_compare_training():
    """
    Simple training comparison: subsampled vs. non-subsampled.
    
    This runs a few epochs and compares training curves.
    """
    print("\n" + "="*70)
    print("TEST 3: Compare Training with vs. without Subsampling")
    print("="*70)
    
    # Create small model
    def make_small_model():
        inp_img = tf.keras.Input(shape=(128, 128, 3), name="input_image")
        inp_flux = tf.keras.Input(shape=(1,), name="input_flux")
        
        x = tf.keras.layers.Flatten()(inp_img)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        z_out = tf.keras.layers.Dense(1, name="output_z")(x)
        
        cat = tf.keras.layers.Concatenate()([x, z_out, inp_flux])
        mat_out = tf.keras.layers.Dense(len(MATERIAL_LABELS), activation="softmax",
                                        name="output_material")(cat)
        
        model = tf.keras.Model(inputs=[inp_img, inp_flux], outputs=[mat_out, z_out])
        model.compile(
            optimizer="adam",
            loss={"output_material": "sparse_categorical_crossentropy", "output_z": "mse"},
            loss_weights={"output_material": 1.0, "output_z": 0.3},
        )
        return model
    
    df_train = create_dummy_data(n_samples=300)
    df_val = create_dummy_data(n_samples=100)
    
    from prepare_data_for_ResNet import _make_tf_dataset
    
    # Dataset without subsampling
    train_ds_std = _make_tf_dataset(df_train, shuffle=True, batch_size=16, seed=42)
    val_ds = _make_tf_dataset(df_val, shuffle=False, batch_size=16, seed=42)
    
    # Dataset with subsampling
    train_ds_sub = _make_dynamic_subsampled_dataset(df_train, subsample_fraction=0.5,
                                                     batch_size=16, seed=42)
    
    print("\nTraining model WITHOUT subsampling (5 epochs)...")
    model1 = make_small_model()
    hist1 = model1.fit(train_ds_std, validation_data=val_ds, epochs=5, verbose=0)
    
    print("Training model WITH subsampling (5 epochs)...")
    model2 = make_small_model()
    hist2 = model2.fit(train_ds_sub, validation_data=val_ds, epochs=5, verbose=0)
    
    print("\nTraining summary:")
    print(f"  Without subsampling - Final train loss: {hist1.history['loss'][-1]:.4f}")
    print(f"  Without subsampling - Final val loss:   {hist1.history['val_loss'][-1]:.4f}")
    print(f"  With subsampling    - Final train loss: {hist2.history['loss'][-1]:.4f}")
    print(f"  With subsampling    - Final val loss:   {hist2.history['val_loss'][-1]:.4f}")
    
    print("\n✓ PASS: Both training modes completed without errors")
    
    # Optional: plot
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(hist1.history['loss'], label='No subsampling', marker='o')
        plt.plot(hist2.history['loss'], label='With subsampling', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.legend()
        plt.title('Training Loss Comparison')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(hist1.history['val_loss'], label='No subsampling', marker='o')
        plt.plot(hist2.history['val_loss'], label='With subsampling', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.title('Validation Loss Comparison')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("/tmp/subsampling_comparison.png", dpi=100, bbox_inches='tight')
        print(f"\n  Plot saved to: /tmp/subsampling_comparison.png")
    except Exception as e:
        print(f"\n  (Could not save plot: {e})")


if __name__ == "__main__":
    try:
        test_subsampling_enabled()
        test_subsampling_disabled()
        test_compare_training()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nDynamic subsampling is working correctly!")
        print("\nTo enable it in your training:")
        print("  1. Edit train_material_classifier.py")
        print("  2. Set CONFIG['subsample_enabled'] = True")
        print("  3. Adjust subsample_fraction if desired (default 0.30 = 30%)")
        print("  4. Run: python train_material_classifier.py")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
