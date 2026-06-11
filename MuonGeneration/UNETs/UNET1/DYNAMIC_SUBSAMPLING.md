# Dynamic Subsampling — Combat Overfitting on Simulated Data

## Problem

Simulated datasets often have **high redundancy**:
- Many variations of the same physical setup
- Systematic patterns repeated across samples
- Limited diversity compared to real-world data

Training on **all data every epoch** risks overfitting to these patterns, making the model generalize poorly to experimental data.

---

## Solution: Dynamic Subsampling per Epoch

Instead of using a fixed subset (e.g., always the same 30%), dynamically select a **different random 30% each epoch**.

**Key insight:** The model never sees the exact same subsample twice, but across 200 epochs with 30% subsampling, it sees ~99.7% of the data in different combinations. This:

✓ Destroys overfitting to specific instances  
✓ Maintains full statistical richness of the dataset  
✓ Makes the model robust to data redundancy  

### Mathematical guarantee

Probability that any single sample is **NOT** seen in $N$ epochs with subsampling fraction $p$:

$$P(\text{never seen}) = (1 - p)^N$$

For $N=200, p=0.3$: $(0.7)^{200} \approx 10^{-35}$ (effectively zero).

So with 200 epochs and 30% subsampling, essentially every sample gets trained on.

---

## How to Enable

### In `train_material_classifier.py`

Change the CONFIG dict:

```python
CONFIG = {
    # ... other settings ...
    
    # ── Dynamic subsampling ───────────────────────────────────────────────────
    "subsample_enabled":  True,     # ← Set to True
    "subsample_fraction": 0.30,     # ← Fraction per epoch (0.3 = 30%)
}
```

Then run training as usual:

```bash
python train_material_classifier.py
```

### Console Output

When enabled, you'll see:

```
  [DYNAMIC SUBSAMPLING ENABLED]
    Each epoch: use 30% of training data (different subset each epoch)
    This combats overfitting on simulated data with high redundancy.
```

This confirms that the training dataset is **regenerated each epoch** with a fresh random subsample.

---

## Expected Behavior

### With `subsample_enabled=False` (default)

```
Epoch 001/200:  Train loss: 0.8543  |  Val loss: 0.7821  |  Test accuracy: 0.845
Epoch 002/200:  Train loss: 0.8134  |  Val loss: 0.7650  |  Test accuracy: 0.851
Epoch 003/200:  Train loss: 0.7823  |  Val loss: 0.7521  |  Test accuracy: 0.856
...
```

Train loss decreases smoothly (model memorizing patterns).

### With `subsample_enabled=True` (subsampling)

```
Epoch 001/200:  Train loss: 0.8901  |  Val loss: 0.7821  |  Test accuracy: 0.839
Epoch 002/200:  Train loss: 0.8234  |  Val loss: 0.7650  |  Test accuracy: 0.848
Epoch 003/200:  Train loss: 0.8567  |  Val loss: 0.7521  |  Test accuracy: 0.855
...
```

Train loss is **noisier** (oscillates because each epoch uses different data), but:
- Val/test curves are smoother and more stable
- No sharp overfitting after epoch ~50
- Better generalization to unseen data

---

## Tuning `subsample_fraction`

| Fraction | Use Case |
|----------|----------|
| **0.10** (10%) | Very redundant data; aggressive regularization |
| **0.20** (20%) | Highly simulated, want maximum robustness |
| **0.30** (30%) | Default; good balance (covers 99.7% data in 200 epochs) |
| **0.50** (50%) | Moderate redundancy; faster training per epoch |
| **1.00** (100%) | Same as disabling; trains on all data every epoch |

### Strategy

1. Start with `subsample_fraction=0.30`
2. Monitor validation loss for stability
3. If validation still overfits → lower to 0.20
4. If validation is too noisy → increase to 0.50

---

## Technical Details

### How It Works

In `prepare_data_for_ResNet.py`:

```python
def _make_dynamic_subsampled_dataset(df, subsample_fraction, batch_size, seed):
    """
    Returns a tf.data.Dataset that regenerates a random subsample each time
    it's iterated.
    
    Each epoch: 
      1. Randomly select (fraction * total_samples) indices
      2. Load those samples
      3. Shuffle within the subset
      4. Yield batches
      5. Next epoch: repeat steps 1-4 with DIFFERENT random indices
    """
```

### Memory Usage

- **Without subsampling:** ~Full dataset in memory simultaneously
- **With subsampling:** Only the selected fraction loaded per batch
- **Net effect:** Slightly **lower** memory usage

### Training Speed

- **Epoch 1:** Same speed (still loading samples)
- **Epochs 2+:** Potentially **faster** (fewer samples per epoch)
  - 30% subsampling ≈ 3x faster per epoch
  - But need more epochs to converge → net trade-off

---

## Validation Protocol

To verify subsampling is working, temporarily add a callback that logs dataset sizes:

```python
# In your training code
class DatasetInfoCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_ds):
        self.train_ds = train_ds
    
    def on_epoch_begin(self, epoch, logs=None):
        # Count batches (approximate total samples / batch_size)
        n_batches = len(self.train_ds)
        print(f"  Epoch {epoch+1}: {n_batches} batches (≈ {n_batches * 32} samples)")

# Add to callbacks in train_material_classifier.py
```

With 30% subsampling on ~2500 training samples:
- Expected: ~240 samples per epoch
- With batch_size=32: ~8 batches per epoch
- Without subsampling: ~78 batches per epoch (2500/32)

---

## Common Pitfalls

❌ **Don't:** Combine subsampling with very aggressive data augmentation
- Subsampling already adds variability; over-augmenting could hurt

✓ **Do:** Keep augmentation moderate when using subsampling

❌ **Don't:** Use the same random_seed across runs
- You want different subsets each run; let seed vary or use datetime-based seed

✓ **Do:** Set `random_seed` deterministically for reproducibility, but subsampling will still be different per epoch (internal randomness in generator)

---

## Disabling (Reverting to Standard Training)

Simply set:

```python
CONFIG = {
    "subsample_enabled": False,  # ← Back to normal
    # ...
}
```

This reverts to standard behavior: train on full dataset every epoch.

---

## Further Reading

- **Curriculum Learning:** Similar idea, but gradually include harder samples
- **Active Learning:** Select most informative samples (more sophisticated)
- **Data Dropout:** Similar to subsampling; random exclusion at batch level

For your use case (redundant simulation data), simple random subsampling is effective and easy to implement.
