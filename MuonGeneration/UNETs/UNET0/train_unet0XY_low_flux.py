#!/usr/bin/env python3
"""
UNET2D fine-tuning for POCA projection denoising — XY channel, 10 % flux.

Starts from best_model_0_XY.keras (full-flux pretrained) and fine-tunes with
Binomial-subsampled POCA inputs: for each pixel with stored count v (uint8),
  v_sparse ~ Binomial(v, FLUX_FRACTION=0.10)
so the network learns to denoise heavily under-sampled projections.
Ground truth stays unchanged (full-flux GT).

Input : dataset0.h5  (channel 0: XY projection, sliced on-the-fly)
        Pretrained  : data/models_XY/checkpoints/best_model_0_XY.keras
Output: data/models_XY_10_perc/  (suffix _0_XY_10_perc on all files)

See UNET0XY.txt for full design rationale and architecture documentation.
"""

import os
import sys
import gc
import time
import json
from pathlib import Path

import numpy as np

# ─── Silence TensorFlow noise BEFORE importing it ────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'  # Prevents freeze on first batch

gc.collect()

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras import mixed_precision
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
    keras.backend.clear_session()
    mixed_precision.set_global_policy('mixed_bfloat16')
    tf.config.optimizer.set_jit(False)
except ImportError:
    print("[ERROR] TensorFlow required. Install: pip install tensorflow")
    sys.exit(1)

try:
    import h5py
except ImportError:
    print("[ERROR] h5py required. Install: pip install h5py")
    sys.exit(1)


# ===========================================================================
# CONFIG
# ===========================================================================

BASE     = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration")
DATA     = BASE / "data"
DATASETS = DATA / "datasets"
MODELS   = DATA / "models_XY_10_perc"

H5_FILE              = DATASETS / "dataset0.h5"
OUTPUT_DIR           = MODELS
PRETRAINED_MODEL_PATH = DATA / "models_XY" / "checkpoints" / "best_model_0_XY.keras"

# Hyperparameters
BATCH_SIZE    = 32
LEARNING_RATE = 1e-4      # lower LR for fine-tuning
EPOCHS        = 100
SIZE_IMAGES   = 128
N_FILTERS     = 64
FILTER_SIZE   = 3
N_LEVELS      = 4

FLUX_FRACTION = 0.05      # Binomial subsampling: keep ~10 % of muon counts per pixel

USE_GPU            = True
FORCE_RESTART      = True   # If True, start from PRETRAINED_MODEL_PATH (ignore 10%-flux checkpoints)
USE_AUGMENTATION   = True   # On-the-fly rotations ×4 (train split only)
EVAL_ONLY          = False  # If True, skip training and load best 10%-flux model → evaluate + visualize
VISUALIZE_RESULTS  = True   # If True, save input/pred/GT comparison PNGs after training
N_VIZ              = 8      # Samples per split to visualize


# ===========================================================================
# ARCHITECTURE UNET2D  (standard — single channel, no groups needed)
# ===========================================================================

def conv_block(input_tensor, n_filters, name, repeat_conv=2):
    """Two Conv2D layers with BN + ReLU. Standard — no channel isolation needed."""
    x = input_tensor
    for i in range(1, repeat_conv + 1):
        x = layers.Conv2D(
            n_filters,
            (FILTER_SIZE, FILTER_SIZE),
            padding='same',
            use_bias=False,
            name=f"{name}_conv{i}"
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.Activation('relu', name=f"{name}_relu{i}")(x)
    return x


def unet_2d(input_shape=(SIZE_IMAGES, SIZE_IMAGES, 1), n_levels=N_LEVELS, n_filters=N_FILTERS):
    """UNET2D — single channel input/output with Conv2DTranspose upsampling."""
    inputs = layers.Input(shape=input_shape)
    encoder_outputs = []

    # ENCODER
    x = inputs
    for i in range(n_levels):
        filters = n_filters * (2 ** i)
        x = conv_block(x, filters, name=f"Enc{i+1}")
        encoder_outputs.append(x)
        x = layers.MaxPooling2D((2, 2), name=f"Pool{i+1}")(x)

    # BOTTLENECK
    x = conv_block(x, n_filters * (2 ** n_levels), name="Bottleneck")

    # DECODER  — Conv2DTranspose: learned upsampling
    for i in range(n_levels, 0, -1):
        filters = n_filters * (2 ** (i - 1))
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name=f"Up{i}")(x)
        x = layers.concatenate([x, encoder_outputs[i - 1]], name=f"Concat{i}")
        x = conv_block(x, filters, name=f"Dec{i}")

    # OUTPUT
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', dtype='float32', name="Output")(x)

    return Model(inputs, outputs)


# ===========================================================================
# LOSS & METRICS
# ===========================================================================

def charbonnier_loss(y_true, y_pred, eps=1e-3):
    """Charbonnier loss: smooth L1, preserves edges better than MSE."""
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + eps * eps))


def psnr_metric(y_true, y_pred):
    """Peak Signal-to-Noise Ratio (dB). Higher = better."""
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def ssim_metric(y_true, y_pred):
    """Structural Similarity Index. Range [-1, 1], higher = better."""
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


# ===========================================================================
# DATA STREAMING GENERATOR
# ===========================================================================

def _apply_low_flux(poca_uint8, flux_fraction=FLUX_FRACTION, rng=None):
    """
    Binomial subsampling of a POCA count image.

    For each pixel with stored count v (uint8, 0-255), draw:
        v_sparse ~ Binomial(v, flux_fraction)
    This is the statistically correct simulation of reduced muon flux
    (independent Bernoulli trials per muon, Poisson-process compatible).

    poca_uint8 : ndarray, dtype uint8 or int32, shape (..., H, W, 1)
    Returns    : float32 array in [0, 1], same shape, normalised by /255.
    """
    counts = poca_uint8.astype(np.int32)
    if rng is not None:
        sparse = rng.binomial(counts, flux_fraction)
    else:
        sparse = np.random.binomial(counts, flux_fraction)
    return sparse.astype(np.float32) / 255.0


class H5DataGenerator:
    """
    Streams XY channel (channel 0) from dataset0.h5 with 10 % flux subsampling.

    For each sample, the POCA image is Binomially subsampled at FLUX_FRACTION
    before being yielded.  Ground truth is unchanged (full-flux GT).

    Augmentation (train only, USE_AUGMENTATION=True):
      Random rotation from {0°, 90°, 180°, 270°} applied identically to x and y.
    """

    def __init__(self, h5_path, split, augment=False):
        self.h5_path = h5_path
        self.split   = split
        self.augment = augment
        with h5py.File(self.h5_path, 'r') as f:
            self.indices = list(range(f[f'{self.split}/poca'].shape[0]))

    def __call__(self):
        with h5py.File(self.h5_path, 'r') as f:
            poca_ds = f[f'{self.split}/poca']
            gt_ds   = f[f'{self.split}/gt']

            if self.split == 'train':
                random_indices = self.indices.copy()
                np.random.shuffle(random_indices)
            else:
                random_indices = self.indices

            for idx in random_indices:
                # Load raw uint8 counts, apply Binomial 10 % subsampling → [0, 1]
                x_raw = poca_ds[idx, :, :, 0:1]                # uint8, shape (128, 128, 1)
                x     = _apply_low_flux(x_raw)                  # float32, shape (128, 128, 1)
                y     = gt_ds[idx, :, :, 0:1].astype(np.float32)

                if self.augment:
                    k = np.random.randint(0, 4)   # 0=no-op, 1=90°, 2=180°, 3=270°
                    if k > 0:
                        x = np.rot90(x, k, axes=(0, 1))
                        y = np.rot90(y, k, axes=(0, 1))

                yield x, y


def get_dataset(h5_path, split, batch_size, shuffle=False, augment=False):
    """Build a tf.data pipeline with prefetching."""
    gen = H5DataGenerator(h5_path, split, augment=augment)

    output_signature = (
        tf.TensorSpec(shape=(SIZE_IMAGES, SIZE_IMAGES, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(SIZE_IMAGES, SIZE_IMAGES, 1), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=256, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=shuffle)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# ===========================================================================
# UTILITIES
# ===========================================================================

def _count_samples(h5_path):
    """Read split sizes from H5 (metadata only — cheap)."""
    with h5py.File(h5_path, 'r') as f:
        return {s: f[f'{s}/poca'].shape[0] for s in ('train', 'val', 'test')}


def _visualize_split(model, h5_path, split, n_samples, output_dir):
    """Save a PNG grid: Input (10 % POCA) | UNET Output | Ground Truth."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN]  matplotlib not available — skipping visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        n = min(n_samples, f[f'{split}/poca'].shape[0])
        pocas_raw = np.array([f[f'{split}/poca'][i, :, :, 0] for i in range(n)])  # uint8
        gts       = np.array([f[f'{split}/gt'][i,   :, :, 0] for i in range(n)], dtype=np.float32)

    # Apply 10 % subsampling (different draw per sample, consistent with training)
    pocas = np.random.binomial(pocas_raw.astype(np.int32), FLUX_FRACTION).astype(np.float32) / 255.0

    preds = model.predict(pocas[..., np.newaxis], batch_size=8, verbose=0)[..., 0]

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    col_titles = [f'Input (POCA {int(FLUX_FRACTION*100)} %)', 'UNET Output', 'Ground Truth']
    for row, (src, pred, gt) in enumerate(zip(pocas, preds, gts)):
        vmax_row = max(float(src.max()), float(pred.max()), float(gt.max()), 1e-6)
        for col, (img, title) in enumerate(zip([src, pred, gt], col_titles)):
            ax = axes[row, col]
            ax.imshow(img, cmap='viridis', vmin=0, vmax=vmax_row)
            ax.axis('off')
            if row == 0:
                ax.set_title(title, fontsize=10)

    plt.suptitle(f"UNET0XY_10perc — {split} samples", fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = output_dir / f"viz_{split}.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[VIZ]   {split:5s} → {out_path.name}")


# ===========================================================================
# MAIN
# ===========================================================================

_CUSTOM_OBJECTS = {
    'charbonnier_loss': charbonnier_loss,
    'psnr_metric':      psnr_metric,
    'ssim_metric':      ssim_metric,
}


def main():
    print("=" * 80)
    print(f" UNET2D-XY FINE-TUNING — 10 % flux (Binomial subsampling, Charbonnier loss)")
    print(f" Pretrained base : {PRETRAINED_MODEL_PATH.name}")
    print("=" * 80)

    start_time = time.time()

    # ── Hardware ────────────────────────────────────────────────────────────
    if USE_GPU:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"[WARN]  set_memory_growth failed: {e}")
            gpu_name = gpus[0].name.split('/')[-1]
            print(f"[HW]    GPU detected     : {len(gpus)} × {gpu_name}")
            print(f"[HW]    Mixed precision  : mixed_bfloat16")
            print(f"[HW]    XLA JIT          : disabled")
        else:
            print("[HW]    No GPU detected — running on CPU")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("[HW]    GPU disabled by config — running on CPU")

    tf.random.set_seed(42)
    np.random.seed(42)

    if not H5_FILE.exists():
        sys.exit(f"[ERROR] H5 file not found: {H5_FILE}")

    # ── Data ────────────────────────────────────────────────────────────────
    sizes       = _count_samples(H5_FILE)
    steps_train = sizes['train'] // BATCH_SIZE
    steps_val   = sizes['val']   // BATCH_SIZE
    steps_test  = sizes['test']  // BATCH_SIZE
    print(f"[DATA]  Source           : {H5_FILE.name}  (channel 0: XY projection)")
    print(f"[DATA]  Splits           : train={sizes['train']}  val={sizes['val']}  test={sizes['test']}")
    print(f"[DATA]  Batch size       : {BATCH_SIZE}  ({steps_train} steps/epoch)")
    print(f"[DATA]  Flux fraction    : {FLUX_FRACTION:.0%}  (Binomial per-pixel subsampling)")
    print(f"[DATA]  Augmentation     : {'rotations ×4 (train only)' if USE_AUGMENTATION else 'none'}")

    train_ds = get_dataset(H5_FILE, 'train', BATCH_SIZE, shuffle=True,  augment=USE_AUGMENTATION)
    val_ds   = get_dataset(H5_FILE, 'val',   BATCH_SIZE, shuffle=False, augment=False)
    test_ds  = get_dataset(H5_FILE, 'test',  BATCH_SIZE, shuffle=False, augment=False)

    # ──────────────────────────────────────────────────────────────────────────
    # EVAL_ONLY MODE: Load best 10 %-flux model and skip training
    # ──────────────────────────────────────────────────────────────────────────
    if EVAL_ONLY:
        print("\n" + "#" * 80)
        print("###  EVAL_ONLY MODE: Loading best 10 %-flux model and skipping training")
        print("#" * 80)

        CHECKPOINT_DIR  = OUTPUT_DIR / "checkpoints"
        best_model_path = CHECKPOINT_DIR / "best_model_0_XY_10_perc.keras"

        if not best_model_path.exists():
            sys.exit(f"[ERROR] Best model not found: {best_model_path}")

        best_model = keras.models.load_model(str(best_model_path), custom_objects=_CUSTOM_OBJECTS)
        print(f"[EVAL]  Best model loaded : {best_model_path.name}")

        print("\n" + "=" * 80)
        print(" EVALUATION ON TEST SET  (10 % flux inputs)")
        print("=" * 80)
        results = best_model.evaluate(test_ds, steps=steps_test, verbose=0, return_dict=True)
        print(f"[EVAL]  Test Charbonnier : {results['loss']:.6f}")
        print(f"[EVAL]  Test PSNR        : {results['psnr_metric']:.3f} dB")
        print(f"[EVAL]  Test SSIM        : {results['ssim_metric']:.4f}")

        if VISUALIZE_RESULTS:
            viz_dir = OUTPUT_DIR / "visualizations_XY_10_perc"
            print(f"\n[VIZ]   Saving visualizations → {viz_dir}/")
            for split in ('train', 'val', 'test'):
                _visualize_split(best_model, H5_FILE, split, N_VIZ, viz_dir)

        print("=" * 80 + "\n")
        return

    # ── Model — load pretrained or resume 10 %-flux checkpoint ──────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path     = CHECKPOINT_DIR / "training_meta_0_XY_10_perc.json"
    initial_epoch = 0

    print("\n" + "#" * 80)
    if FORCE_RESTART:
        print("###  TRAINING MODE : FROM PRETRAINED FULL-FLUX MODEL (FORCE_RESTART=True)")
        print(f"###  Base model     : {PRETRAINED_MODEL_PATH}")
        print("###  Any previous 10 %-flux checkpoint will be IGNORED and OVERWRITTEN.")
        if meta_path.exists():
            meta_path.unlink()
        if not PRETRAINED_MODEL_PATH.exists():
            sys.exit(f"[ERROR] Pretrained model not found: {PRETRAINED_MODEL_PATH}")
        model = keras.models.load_model(str(PRETRAINED_MODEL_PATH), custom_objects=_CUSTOM_OBJECTS)
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss=charbonnier_loss,
            metrics=[psnr_metric, ssim_metric],
            jit_compile=False,
        )
    elif meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        initial_epoch = meta.get('last_epoch', 0)
        last_ckpt = CHECKPOINT_DIR / "last_checkpoint_0_XY_10_perc.keras"
        if last_ckpt.exists():
            model = keras.models.load_model(str(last_ckpt), custom_objects=_CUSTOM_OBJECTS)
            print(f"###  TRAINING MODE : RESUMING from epoch {initial_epoch + 1}")
            print(f"###  Loaded checkpoint    : {last_ckpt.name}")
            print(f"###  Previous best val_loss : {meta.get('best_val_loss', 'n/a')}")
        else:
            print("###  TRAINING MODE : FROM PRETRAINED FULL-FLUX MODEL")
            print("###  (meta found but no 10 %-flux checkpoint — loading pretrained base)")
            initial_epoch = 0
            if not PRETRAINED_MODEL_PATH.exists():
                sys.exit(f"[ERROR] Pretrained model not found: {PRETRAINED_MODEL_PATH}")
            model = keras.models.load_model(str(PRETRAINED_MODEL_PATH), custom_objects=_CUSTOM_OBJECTS)
            optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            model.compile(
                optimizer=optimizer,
                loss=charbonnier_loss,
                metrics=[psnr_metric, ssim_metric],
                jit_compile=False,
            )
    else:
        print("###  TRAINING MODE : FROM PRETRAINED FULL-FLUX MODEL (no previous 10 %-flux run)")
        if not PRETRAINED_MODEL_PATH.exists():
            sys.exit(f"[ERROR] Pretrained model not found: {PRETRAINED_MODEL_PATH}")
        model = keras.models.load_model(str(PRETRAINED_MODEL_PATH), custom_objects=_CUSTOM_OBJECTS)
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss=charbonnier_loss,
            metrics=[psnr_metric, ssim_metric],
            jit_compile=False,
        )
    print("#" * 80)

    n_params = model.count_params()
    print(f"[MODEL] Architecture     : UNET2D standard (single-channel XY)")
    print(f"[MODEL] Levels / filters : {N_LEVELS} levels, base={N_FILTERS}, kernel={FILTER_SIZE}×{FILTER_SIZE}")
    print(f"[MODEL] Parameters       : {n_params:,}")
    print(f"[MODEL] Loss / metrics   : Charbonnier / PSNR, SSIM")
    print(f"[MODEL] Optimizer        : Adam(lr={LEARNING_RATE})  [fine-tuning]")

    summary_file = OUTPUT_DIR / "model_summary_0_XY_10_perc.txt"
    with open(summary_file, "w") as f:
        model.summary(line_length=110, print_fn=lambda s: f.write(s + "\n"))
    print(f"[MODEL] Full summary     : {summary_file}")

    # ── Callbacks ───────────────────────────────────────────────────────────
    full_checkpoint = FullTrainingCheckpoint(CHECKPOINT_DIR, monitor='val_loss')
    early_stopping  = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=False, verbose=1
    )
    csv_logger  = keras.callbacks.CSVLogger(
        OUTPUT_DIR / "training_log_0_XY_10_perc.csv", append=True
    )
    progress_cb = EpochProgressCallback(steps_per_epoch=steps_train, log_every=20)
    monitor_cb  = EpochMonitoringCallback(H5_FILE, CHECKPOINT_DIR / "epoch_monitoring")

    print("\n" + "-" * 80)
    print(f" Fine-tuning: epochs {initial_epoch + 1} → {EPOCHS}")
    print("-" * 80)

    model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_train,
        initial_epoch=initial_epoch,
        validation_data=val_ds,
        validation_steps=steps_val,
        callbacks=[full_checkpoint, early_stopping, csv_logger, progress_cb, monitor_cb],
        verbose=0,
    )

    elapsed = time.time() - start_time
    print("-" * 80)
    print(f"[TIME]  Fine-tuning finished : {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # ── Final test evaluation with best weights ─────────────────────────────
    best_model_path = CHECKPOINT_DIR / "best_model_0_XY_10_perc.keras"
    print("\n" + "=" * 80)
    print(" FINAL EVALUATION ON TEST SET (best val_loss checkpoint, 10 % flux inputs)")
    print("=" * 80)
    best_model = None
    try:
        best_model = keras.models.load_model(str(best_model_path), custom_objects=_CUSTOM_OBJECTS)
        print(f"[EVAL]  Best model       : {best_model_path.name}")
        results = best_model.evaluate(test_ds, steps=steps_test, verbose=0, return_dict=True)
        print(f"[EVAL]  Test Charbonnier : {results['loss']:.6f}")
        print(f"[EVAL]  Test PSNR        : {results['psnr_metric']:.3f} dB")
        print(f"[EVAL]  Test SSIM        : {results['ssim_metric']:.4f}")
    except Exception as e:
        print(f"[ERROR] Could not evaluate best model on test: {e}")

    final_model_path = OUTPUT_DIR / "model_epoch_final_0_XY_10_perc.keras"
    model.save(final_model_path)
    print(f"[SAVE]  Final-epoch model : {final_model_path.name}")

    # ── Optional visualization ───────────────────────────────────────────────
    if VISUALIZE_RESULTS:
        eval_model = best_model if best_model is not None else model
        viz_dir = OUTPUT_DIR / "visualizations_XY_10_perc"
        print(f"\n[VIZ]   Saving visualizations → {viz_dir}/")
        for split in ('train', 'val', 'test'):
            _visualize_split(eval_model, H5_FILE, split, N_VIZ, viz_dir)

    print("=" * 80 + "\n")
    return


# ===========================================================================
# CALLBACKS
# ===========================================================================

class EpochProgressCallback(keras.callbacks.Callback):
    """Prints intra-epoch progress every `log_every` batches."""

    def __init__(self, steps_per_epoch, log_every=20):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.log_every       = log_every
        self._epoch          = 0
        self._t0_epoch       = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch    = epoch + 1
        self._t0_epoch = time.time()
        print(f"\n[TRAIN] Epoch {self._epoch}/{self.params['epochs']}")

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        step = batch + 1
        if step % self.log_every == 0 or step == self.steps_per_epoch:
            elapsed   = time.time() - self._t0_epoch
            remaining = (elapsed / step) * (self.steps_per_epoch - step)
            pct  = 100 * step / self.steps_per_epoch
            loss = logs.get('loss', float('nan'))
            print(
                f"  step {step:4d}/{self.steps_per_epoch}  "
                f"[{'#' * int(pct // 5):<20}] {pct:5.1f}%  "
                f"loss={loss:.6f}  "
                f"elapsed={elapsed:6.1f}s  eta={remaining:5.1f}s"
            )

    def on_epoch_end(self, epoch, logs=None):
        logs      = logs or {}
        elapsed   = time.time() - self._t0_epoch
        val_loss  = logs.get('val_loss',        float('nan'))
        val_psnr  = logs.get('val_psnr_metric', float('nan'))
        val_ssim  = logs.get('val_ssim_metric', float('nan'))
        print(
            f"  → Epoch {self._epoch} done in {elapsed:.1f}s  "
            f"val_loss={val_loss:.6f}  PSNR={val_psnr:.2f}dB  SSIM={val_ssim:.4f}"
        )


class FullTrainingCheckpoint(keras.callbacks.Callback):
    """Saves best_model_0_XY_10_perc.keras (val_loss) and last_checkpoint_0_XY_10_perc.keras each epoch."""

    def __init__(self, checkpoint_dir, monitor='val_loss'):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor   = monitor
        self.best      = np.inf
        self.meta_path = self.checkpoint_dir / "training_meta_0_XY_10_perc.json"

        if self.meta_path.exists():
            with open(self.meta_path) as f:
                meta = json.load(f)
            self.best = meta.get('best_val_loss', np.inf)
            print(f"[CKPT]  Previous best    : val_loss={self.best:.6f}")

    def on_epoch_end(self, epoch, logs=None):
        logs    = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        last_path = self.checkpoint_dir / "last_checkpoint_0_XY_10_perc.keras"
        self.model.save(last_path)

        meta = {
            'last_epoch':    epoch + 1,
            'best_val_loss': float(self.best),
            'last_val_loss': float(current),
            'last_val_psnr': float(logs.get('val_psnr_metric', 0.0)),
            'last_val_ssim': float(logs.get('val_ssim_metric', 0.0)),
        }

        if current < self.best:
            self.best = current
            best_path = self.checkpoint_dir / "best_model_0_XY_10_perc.keras"
            self.model.save(best_path)
            meta['best_val_loss'] = float(self.best)
            print(f"[CKPT]  ✓ New best        : val_loss={current:.6f} → saved {best_path.name}")

        with open(self.meta_path, 'w') as f:
            json.dump(meta, f, indent=2)


class EpochMonitoringCallback(keras.callbacks.Callback):
    """
    Saves a publication-quality monitoring figure at the end of each epoch.

    Layout: 3 rows (Train / Val / Test) × 3 cols (Input 10%-POCA / UNET Output / GT).
      - Train sample : random index each epoch.
      - Val / Test   : index 0 fixed across all epochs → visual convergence tracking.
      - Fixed-seed Binomial subsampling for val/test (reproducible reference images).

    Output: <checkpoint_dir>/epoch_monitoring/UNET0XY_10perc_epoch{NNN}_monitoring.png
    """

    def __init__(self, h5_path, monitoring_dir):
        super().__init__()
        self.h5_path        = h5_path
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)

        # Fixed-seed RNG for reproducible val/test monitoring images
        _rng = np.random.default_rng(seed=0)

        with h5py.File(self.h5_path, 'r') as f:
            val_raw        = f['val/poca'][0,  :, :, 0:1]
            self._val_x    = _apply_low_flux(val_raw, rng=_rng)
            self._val_y    = f['val/gt'][0,    :, :, 0:1].astype(np.float32)

            test_raw       = f['test/poca'][0, :, :, 0:1]
            self._test_x   = _apply_low_flux(test_raw, rng=_rng)
            self._test_y   = f['test/gt'][0,   :, :, 0:1].astype(np.float32)

            self._n_train  = f['train/poca'].shape[0]

    def on_epoch_end(self, epoch, logs=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epoch_num = epoch + 1
        logs = logs or {}

        # Random train sample with fresh Binomial draw
        idx = np.random.randint(0, self._n_train)
        with h5py.File(self.h5_path, 'r') as f:
            train_raw = f['train/poca'][idx, :, :, 0:1]
            train_x   = _apply_low_flux(train_raw)
            train_y   = f['train/gt'][idx, :, :, 0:1].astype(np.float32)

        # Single forward pass for all 3 samples
        batch = np.stack([train_x, self._val_x, self._test_x], axis=0)  # (3, 128, 128, 1)
        preds = self.model.predict(batch, verbose=0)                     # (3, 128, 128, 1)

        rows = [
            ('Train  (random)',   train_x[..., 0], preds[0, ..., 0], train_y[..., 0]),
            ('Val    (fixed #0)', self._val_x[..., 0], preds[1, ..., 0], self._val_y[..., 0]),
            ('Test   (fixed #0)', self._test_x[..., 0], preds[2, ..., 0], self._test_y[..., 0]),
        ]

        val_loss = logs.get('val_loss',        float('nan'))
        val_psnr = logs.get('val_psnr_metric', float('nan'))
        val_ssim = logs.get('val_ssim_metric', float('nan'))

        fig, axes = plt.subplots(3, 3, figsize=(13, 11))
        fig.patch.set_facecolor('white')
        fig.suptitle(
            f"UNET0XY\_10perc  ·  Epoch {epoch_num:03d} / {self.params['epochs']}"
            f"    |    val\_loss = {val_loss:.5f}    PSNR = {val_psnr:.2f} dB    SSIM = {val_ssim:.4f}",
            fontsize=12, fontweight='bold', y=0.995,
        )

        col_titles = [f'Input POCA ({int(FLUX_FRACTION*100)} %)', 'UNET Output  (denoised)', 'Ground Truth']
        for row_idx, (row_label, src, pred, gt) in enumerate(rows):
            vmax_row = max(float(src.max()), float(pred.max()), float(gt.max()), 1e-6)
            for col_idx, (img, col_title) in enumerate(zip([src, pred, gt], col_titles)):
                ax = axes[row_idx, col_idx]
                im = ax.imshow(img, cmap='viridis', vmin=0, vmax=vmax_row,
                               aspect='equal', interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(col_title, fontsize=11, fontweight='bold', pad=8)
                if col_idx == 0:
                    ax.set_ylabel(row_label, fontsize=10, fontweight='bold', labelpad=8)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=7)
                cbar.set_label('Normalised density', fontsize=7, labelpad=4)

        plt.tight_layout(rect=[0, 0, 1, 0.965])
        out_path = self.monitoring_dir / f"UNET0XY_10perc_epoch{epoch_num:03d}_monitoring.png"
        plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[MON]   Epoch {epoch_num:03d} → {out_path.name}")


if __name__ == "__main__":
    main()