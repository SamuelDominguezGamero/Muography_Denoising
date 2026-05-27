#!/usr/bin/env python3
"""
UNET2D fine-tuning for POCA projection denoising — XY channel, 1 % flux.
Single-channel input: Binomial-subsampled sparse POCA counts.

Strategy — two-stage curriculum learning:

  Stage 1 (train_unet0XY_low_flux.py):
    Full-flux pretrained model → fine-tuned at ~5 % flux.

  Stage 2 (this script):
    Low-flux checkpoint → fine-tuned at 1 % flux.
    Direct weight loading + fine-tuning; the domain shift from 10 % to 1 %
    is bridged gradually by a low LR.

Preprocessing pipeline per sample:
    uint8 POCA [0-255]
        → Binomial(v, 0.01)    # statistically correct reduced-flux simulation
        → / 255.0              # normalise to [0, 1]
        → (128, 128, 1) float32 UNET input

Input : dataset0.h5  (channel 0: XY projection)
        Pretrained  : data/models_XY_10_perc/checkpoints/best_model_0_XY_10_perc.keras
Output: data/models_XY_1_perc/  (suffix _0_XY_1_perc)
"""

import os
import sys
import gc
import time
import json
from pathlib import Path

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

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

H5_FILE               = DATASETS / "dataset0.h5"
OUTPUT_DIR            = DATA / "models_XY_1_perc"
PRETRAINED_MODEL_PATH = DATA / "models_XY_10_perc" / "checkpoints" / "best_model_0_XY_10_perc.keras"

# Hyperparameters
BATCH_SIZE    = 16
LEARNING_RATE = 1e-3
EPOCHS        = 100
SIZE_IMAGES   = 128
N_FILTERS     = 128
FILTER_SIZE   = 3
N_LEVELS      = 4

FLUX_FRACTION  = 0.01   # Binomial subsampling: keep ~1 % of muon counts per pixel

USE_GPU            = True
FORCE_RESTART      = False   # True: start from PRETRAINED_MODEL_PATH (ignore 1%-flux checkpoints)
USE_AUGMENTATION   = True   # On-the-fly rotations x4 (train split only)
EVAL_ONLY          = False  # True: load best 1%-flux model, evaluate + visualize, skip training
VISUALIZE_RESULTS  = True
N_VIZ              = 8

# Dataset preview (mutually exclusive with training — set True to explore only)
PREVISUALIZE_DATASET_ULTRALOW_FLUX = False   # True: interactive explorer, then sys.exit()
PREVIEW_SPLIT = 'train'   # starting split: 'train' | 'val' | 'test'
PREVIEW_SEED  = 0         # base seed for Binomial draws in preview (seed + idx for each sample)


# ===========================================================================
# ARCHITECTURE — UNET2D  (2-channel input, 1-channel output)
# ===========================================================================

def conv_block(input_tensor, n_filters, name, repeat_conv=2):
    x = input_tensor
    for i in range(1, repeat_conv + 1):
        x = layers.Conv2D(
            n_filters, (FILTER_SIZE, FILTER_SIZE),
            padding='same', use_bias=False,
            name=f"{name}_conv{i}"
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.Activation('relu', name=f"{name}_relu{i}")(x)
    return x


def unet_2d(input_shape=(SIZE_IMAGES, SIZE_IMAGES, 1),
            n_levels=N_LEVELS, n_filters=N_FILTERS):
    """UNET2D — single-channel input, single-channel sigmoid output."""
    inputs = layers.Input(shape=input_shape)
    encoder_outputs = []

    x = inputs
    for i in range(n_levels):
        filters = n_filters * (2 ** i)
        x = conv_block(x, filters, name=f"Enc{i+1}")
        encoder_outputs.append(x)
        x = layers.MaxPooling2D((2, 2), name=f"Pool{i+1}")(x)

    x = conv_block(x, n_filters * (2 ** n_levels), name="Bottleneck")

    for i in range(n_levels, 0, -1):
        filters = n_filters * (2 ** (i - 1))
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name=f"Up{i}")(x)
        x = layers.concatenate([x, encoder_outputs[i - 1]], name=f"Concat{i}")
        x = conv_block(x, filters, name=f"Dec{i}")

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', dtype='float32', name="Output")(x)
    return Model(inputs, outputs)


# ===========================================================================
# LOSS & METRICS
# ===========================================================================

def charbonnier_loss(y_true, y_pred, eps=1e-3):
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + eps * eps))


def psnr_metric(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def ssim_metric(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


_CUSTOM_OBJECTS = {
    'charbonnier_loss': charbonnier_loss,
    'psnr_metric':      psnr_metric,
    'ssim_metric':      ssim_metric,
}


# ===========================================================================
# PREPROCESSING
# ===========================================================================

def _apply_low_flux(poca_uint8, flux_fraction=FLUX_FRACTION, rng=None):
    """
    Binomial subsampling: v_sparse ~ Binomial(v, flux_fraction).
    Returns normalised float32 in [0, 1] (counts / 255). Shape unchanged.
    """
    counts = poca_uint8.astype(np.int32)
    if rng is not None:
        sparse = rng.binomial(counts, flux_fraction)
    else:
        sparse = np.random.binomial(counts, flux_fraction)
    return sparse.astype(np.float32) / 255.0


# ===========================================================================
# DATA STREAMING GENERATOR
# ===========================================================================

class H5DataGenerator:
    """
    Streams single-channel XY inputs from dataset0.h5.

    Per sample:
      raw uint8 → Binomial(v, FLUX_FRACTION) → / 255  →  float32 (128,128,1)

    GT is unchanged (full-flux float32, single channel).
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
                idx_list = self.indices.copy()
                np.random.shuffle(idx_list)
            else:
                idx_list = self.indices

            for idx in idx_list:
                x_raw = poca_ds[idx, :, :, 0:1]              # uint8  (128,128,1)
                x     = _apply_low_flux(x_raw)               # float32 (128,128,1) in [0,1]
                y     = gt_ds[idx, :, :, 0:1].astype(np.float32)

                if self.augment:
                    k = np.random.randint(0, 4)
                    if k > 0:
                        x = np.rot90(x, k, axes=(0, 1))
                        y = np.rot90(y, k, axes=(0, 1))

                yield x, y


def get_dataset(h5_path, split, batch_size, shuffle=False, augment=False):
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
    with h5py.File(h5_path, 'r') as f:
        return {s: f[f'{s}/poca'].shape[0] for s in ('train', 'val', 'test')}


def _visualize_split(model, h5_path, split, n_samples, output_dir):
    """
    3-column grid per sample: Input (1% POCA) | UNET Output | Ground Truth
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN]  matplotlib not available — skipping visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        n         = min(n_samples, f[f'{split}/poca'].shape[0])
        pocas_raw = np.array([f[f'{split}/poca'][i, :, :, 0] for i in range(n)])
        gts       = np.array([f[f'{split}/gt'][i,   :, :, 0] for i in range(n)], dtype=np.float32)

    inputs = np.stack([
        _apply_low_flux(pocas_raw[i, :, :, np.newaxis])[..., 0]
        for i in range(n)
    ])

    preds = model.predict(inputs[..., np.newaxis], batch_size=8, verbose=0)[..., 0]

    col_titles = [f'Input (POCA {int(FLUX_FRACTION*100)} %)', 'UNET Output', 'Ground Truth']

    fig, axes = plt.subplots(n, 3, figsize=(10, 3.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for row, (src, pred, gt) in enumerate(zip(inputs, preds, gts)):
        vmax_row = max(float(src.max()), float(pred.max()), float(gt.max()), 1e-6)
        for col, (img, title) in enumerate(zip([src, pred, gt], col_titles)):
            ax = axes[row, col]
            ax.imshow(img, cmap='viridis', vmin=0, vmax=vmax_row)
            ax.axis('off')
            if row == 0:
                ax.set_title(title, fontsize=9)

    plt.suptitle(f"UNET0XY_1perc — {split} samples", fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = output_dir / f"viz_{split}.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[VIZ]   {split:5s} → {out_path.name}")


# ===========================================================================
# DATASET PREVIEW
# ===========================================================================

def _interactive_preview():
    """
    Interactive dataset explorer — opens a matplotlib window and lets you scroll
    through every sample in a split to compare flux levels side by side.

    Columns per sample:
      Full POCA (100 %) | Ch0 Sparse (1 %) | Ch1 Smoothed (Gauss) | Ground Truth

    Keyboard controls:
      →  /  n        next sample
      ←  /  p        previous sample
      t  /  v  /  e  switch to train / val / test split
      r              re-roll Binomial draw for the current sample
      q  /  Escape   close window and exit

    Each sample index uses a deterministic seed (PREVIEW_SEED + idx) so the
    default draw is always reproducible; pressing 'r' increments an offset to
    show alternative realisations of the same sample.

    NOTE: requires an interactive display (X11 / Wayland / macOS).  On a
    headless server the window cannot open; run locally or use port-forwarding.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    if not H5_FILE.exists():
        sys.exit(f"[ERROR] H5 file not found: {H5_FILE}")

    with h5py.File(H5_FILE, 'r') as _f:
        sizes = {s: _f[f'{s}/poca'].shape[0] for s in ('train', 'val', 'test')}

    print("[PREVIEW] Dataset preview mode")
    print(f"[PREVIEW] Flux fraction  : {FLUX_FRACTION:.0%}  (Binomial subsampling)")
    print(f"[PREVIEW] Splits         : " + "  ".join(f"{s}={n}" for s, n in sizes.items()))
    print("[PREVIEW] Controls       : ← → navigate   t/v/e split   r re-roll   q quit")

    state = {
        'split':       PREVIEW_SPLIT,
        'idx':         0,
        'seed_offset': 0,
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.patch.set_facecolor('#1a1a2e')

    def _load_and_draw():
        split       = state['split']
        idx         = state['idx']
        seed_offset = state['seed_offset']
        rng = np.random.default_rng(PREVIEW_SEED + idx * 1000 + seed_offset)

        with h5py.File(H5_FILE, 'r') as f:
            poca_uint8 = f[f'{split}/poca'][idx, :, :, 0:1]              # uint8 (128,128,1)
            full_poca  = poca_uint8[..., 0].astype(np.float32) / 255.0  # full-flux normalised
            gt         = f[f'{split}/gt'][idx, :, :, 0].astype(np.float32)

        ch0 = _apply_low_flux(poca_uint8, rng=rng)[..., 0]  # float32 in [0,1]

        vmax_in = max(float(full_poca.max()), float(ch0.max()), 1e-6)
        vmax_gt = max(float(gt.max()), 1e-6)

        panels = [
            (full_poca, 'Full POCA (100 %)',                        vmax_in),
            (ch0,       f'Sparse ({int(FLUX_FRACTION*100)} % flux)', vmax_in),
            (gt,        'Ground Truth',                              vmax_gt),
        ]

        for ax, (img, title, vmax) in zip(axes, panels):
            ax.clear()
            ax.imshow(img, cmap='viridis', vmin=0, vmax=vmax,
                      aspect='equal', interpolation='nearest')
            ax.set_title(title, fontsize=10, fontweight='bold', color='white', pad=6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#555555')

        roll_note = f"   (re-roll #{seed_offset})" if seed_offset > 0 else ""
        fig.suptitle(
            f"split: {split}   sample: {idx + 1} / {sizes[split]}{roll_note}\n"
            f"← →  navigate    t v e  switch split    r  re-roll    q  quit",
            fontsize=10, color='white', y=1.0,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        fig.canvas.draw_idle()

    def _on_key(event):
        key   = event.key
        split = state['split']
        n     = sizes[split]
        if key in ('right', 'n'):
            state['idx'] = (state['idx'] + 1) % n
            state['seed_offset'] = 0
        elif key in ('left', 'p'):
            state['idx'] = (state['idx'] - 1) % n
            state['seed_offset'] = 0
        elif key == 't':
            state['split'] = 'train'; state['idx'] = 0; state['seed_offset'] = 0
        elif key == 'v':
            state['split'] = 'val';   state['idx'] = 0; state['seed_offset'] = 0
        elif key == 'e':
            state['split'] = 'test';  state['idx'] = 0; state['seed_offset'] = 0
        elif key == 'r':
            state['seed_offset'] += 1
        elif key in ('q', 'escape'):
            plt.close('all')
            return
        else:
            return
        _load_and_draw()

    fig.canvas.mpl_connect('key_press_event', _on_key)
    _load_and_draw()

    try:
        plt.show(block=True)
    except Exception as e:
        print(f"[PREVIEW] Interactive display failed ({e})")
        print("[PREVIEW] Saving static grid to preview_dataset.png instead ...")
        _save_static_preview()


def _save_static_preview(n=8):
    """Fallback for headless environments: save a static N-sample grid PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_path = OUTPUT_DIR / "preview_dataset_ultralow_flux.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(H5_FILE, 'r') as f:
        n = min(n, f[f'{PREVIEW_SPLIT}/poca'].shape[0])
        pocas_raw = np.array([f[f'{PREVIEW_SPLIT}/poca'][i, :, :, 0] for i in range(n)])
        gts       = np.array([f[f'{PREVIEW_SPLIT}/gt'][i,   :, :, 0] for i in range(n)], dtype=np.float32)

    fig, axes = plt.subplots(n, 3, figsize=(10, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        'Full POCA (100 %)',
        f'Sparse ({int(FLUX_FRACTION*100)} % flux)',
        'Ground Truth',
    ]

    for i in range(n):
        rng    = np.random.default_rng(PREVIEW_SEED + i)
        raw    = pocas_raw[i, :, :, np.newaxis]
        sparse = _apply_low_flux(raw, rng=rng)[..., 0]
        full   = pocas_raw[i].astype(np.float32) / 255.0
        imgs   = [full, sparse, gts[i]]
        vmax_in = max(float(full.max()), float(sparse.max()), 1e-6)
        vmax_gt = max(float(gts[i].max()), 1e-6)
        vmaxes  = [vmax_in, vmax_in, vmax_gt]
        for j, (img, title, vmax) in enumerate(zip(imgs, col_titles, vmaxes)):
            ax = axes[i, j]
            ax.imshow(img, cmap='viridis', vmin=0, vmax=vmax, aspect='equal')
            ax.axis('off')
            if i == 0:
                ax.set_title(title, fontsize=9)

    plt.suptitle(f"Dataset preview — {PREVIEW_SPLIT} split  ({n} samples, {FLUX_FRACTION:.0%} flux)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[PREVIEW] Saved → {out_path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    # ── Dataset preview mode (mutually exclusive with training) ─────────────
    if PREVISUALIZE_DATASET_ULTRALOW_FLUX:
        _interactive_preview()
        sys.exit(0)

    print("=" * 80)
    print(f" UNET2D-XY FINE-TUNING — {FLUX_FRACTION:.0%} flux  (Binomial subsampling, single channel)")
    print(f" Curriculum base : {PRETRAINED_MODEL_PATH.name}")
    print("=" * 80)

    start_time = time.time()

    if USE_GPU:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"[WARN]  set_memory_growth failed: {e}")
            gpu_name = gpus[0].name.split('/')[-1]
            print(f"[HW]    GPU detected     : {len(gpus)} x {gpu_name}")
            print(f"[HW]    Mixed precision  : mixed_bfloat16  |  XLA JIT: disabled")
        else:
            print("[HW]    No GPU detected — running on CPU")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("[HW]    GPU disabled by config")

    tf.random.set_seed(42)
    np.random.seed(42)

    if not H5_FILE.exists():
        sys.exit(f"[ERROR] H5 file not found: {H5_FILE}")

    sizes       = _count_samples(H5_FILE)
    steps_train = sizes['train'] // BATCH_SIZE
    steps_val   = sizes['val']   // BATCH_SIZE
    steps_test  = sizes['test']  // BATCH_SIZE
    print(f"[DATA]  Source           : {H5_FILE.name}  (channel 0: XY)")
    print(f"[DATA]  Splits           : train={sizes['train']}  val={sizes['val']}  test={sizes['test']}")
    print(f"[DATA]  Batch size       : {BATCH_SIZE}  ({steps_train} steps/epoch)")
    print(f"[DATA]  Flux fraction    : {FLUX_FRACTION:.0%}  (Binomial per-pixel subsampling)")
    print(f"[DATA]  Augmentation     : {'rotations x4 (train only)' if USE_AUGMENTATION else 'none'}")

    train_ds = get_dataset(H5_FILE, 'train', BATCH_SIZE, shuffle=True,  augment=USE_AUGMENTATION)
    val_ds   = get_dataset(H5_FILE, 'val',   BATCH_SIZE, shuffle=False, augment=False)
    test_ds  = get_dataset(H5_FILE, 'test',  BATCH_SIZE, shuffle=False, augment=False)

    # ── EVAL_ONLY ────────────────────────────────────────────────────────────
    if EVAL_ONLY:
        print("\n" + "#" * 80)
        print(f"###  EVAL_ONLY: Loading best {FLUX_FRACTION:.0%}-flux model, skipping training")
        print("#" * 80)
        CHECKPOINT_DIR  = OUTPUT_DIR / "checkpoints"
        best_model_path = CHECKPOINT_DIR / "best_model_0_XY_1_perc.keras"
        if not best_model_path.exists():
            sys.exit(f"[ERROR] Best model not found: {best_model_path}")
        best_model = keras.models.load_model(str(best_model_path), custom_objects=_CUSTOM_OBJECTS)
        print(f"[EVAL]  Best model loaded : {best_model_path.name}")
        results = best_model.evaluate(test_ds, steps=steps_test, verbose=0, return_dict=True)
        print(f"[EVAL]  Test Charbonnier : {results['loss']:.6f}")
        print(f"[EVAL]  Test PSNR        : {results['psnr_metric']:.3f} dB")
        print(f"[EVAL]  Test SSIM        : {results['ssim_metric']:.4f}")
        if VISUALIZE_RESULTS:
            viz_dir = OUTPUT_DIR / "visualizations_XY_1_perc"
            print(f"\n[VIZ]   Saving visualizations → {viz_dir}/")
            for split in ('train', 'val', 'test'):
                _visualize_split(best_model, H5_FILE, split, N_VIZ, viz_dir)
        print("=" * 80 + "\n")
        return

    # ── Model — weight-transfer from 1-ch pretrained OR resume 1%-flux ckpt ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path     = CHECKPOINT_DIR / "training_meta_0_XY_1_perc.json"
    initial_epoch = 0

    def _load_pretrained():
        if not PRETRAINED_MODEL_PATH.exists():
            sys.exit(f"[ERROR] Pretrained model not found: {PRETRAINED_MODEL_PATH}")
        m = keras.models.load_model(str(PRETRAINED_MODEL_PATH), custom_objects=_CUSTOM_OBJECTS)
        m.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=charbonnier_loss, metrics=[psnr_metric, ssim_metric], jit_compile=False)
        return m

    print("\n" + "#" * 80)
    if FORCE_RESTART:
        print("###  TRAINING MODE : FROM CURRICULUM BASE  (FORCE_RESTART=True)")
        print(f"###  Base model     : {PRETRAINED_MODEL_PATH}")
        print(f"###  Any previous {FLUX_FRACTION:.0%}-flux checkpoint will be IGNORED and OVERWRITTEN.")
        if meta_path.exists():
            meta_path.unlink()
        model = _load_pretrained()
    elif meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        initial_epoch = meta.get('last_epoch', 0)
        last_ckpt = CHECKPOINT_DIR / "last_checkpoint_0_XY_1_perc.keras"
        if last_ckpt.exists():
            model = keras.models.load_model(str(last_ckpt), custom_objects=_CUSTOM_OBJECTS)
            print(f"###  TRAINING MODE : RESUMING from epoch {initial_epoch + 1}")
            print(f"###  Loaded checkpoint    : {last_ckpt.name}")
            print(f"###  Previous best val_loss : {meta.get('best_val_loss', 'n/a')}")
        else:
            print("###  TRAINING MODE : FROM CURRICULUM BASE")
            print("###  (meta found but no 1%-flux checkpoint — loading pretrained base)")
            initial_epoch = 0
            model = _load_pretrained()
    else:
        print(f"###  TRAINING MODE : FROM CURRICULUM BASE (no previous {FLUX_FRACTION:.0%}-flux run)")
        model = _load_pretrained()
    print("#" * 80)

    n_params = model.count_params()
    print(f"[MODEL] Architecture     : UNET2D (single-channel XY)")
    print(f"[MODEL] Levels / filters : {N_LEVELS} levels, base={N_FILTERS}, kernel={FILTER_SIZE}x{FILTER_SIZE}")
    print(f"[MODEL] Parameters       : {n_params:,}")
    print(f"[MODEL] Loss / metrics   : Charbonnier / PSNR, SSIM")
    print(f"[MODEL] Optimizer        : Adam(lr={LEARNING_RATE})  [curriculum fine-tuning]")

    summary_file = OUTPUT_DIR / "model_summary_0_XY_1_perc.txt"
    with open(summary_file, "w") as f:
        model.summary(line_length=110, print_fn=lambda s: f.write(s + "\n"))
    print(f"[MODEL] Summary          : {summary_file}")

    # ── Callbacks ────────────────────────────────────────────────────────────
    full_checkpoint = FullTrainingCheckpoint(CHECKPOINT_DIR, monitor='val_loss')
    early_stopping  = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=False, verbose=1
    )
    csv_logger  = keras.callbacks.CSVLogger(
        OUTPUT_DIR / "training_log_0_XY_1_perc.csv", append=True
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

    # ── Final test evaluation ────────────────────────────────────────────────
    best_model_path = CHECKPOINT_DIR / "best_model_0_XY_1_perc.keras"
    print("\n" + "=" * 80)
    print(f" FINAL EVALUATION ON TEST SET  (best val_loss, {FLUX_FRACTION:.0%} flux, 2-ch input)")
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
        print(f"[ERROR] Could not evaluate best model: {e}")

    final_model_path = OUTPUT_DIR / "model_epoch_final_0_XY_1_perc.keras"
    model.save(final_model_path)
    print(f"[SAVE]  Final-epoch model : {final_model_path.name}")

    if VISUALIZE_RESULTS:
        eval_model = best_model if best_model is not None else model
        viz_dir = OUTPUT_DIR / "visualizations_XY_1_perc"
        print(f"\n[VIZ]   Saving visualizations → {viz_dir}/")
        for split in ('train', 'val', 'test'):
            _visualize_split(eval_model, H5_FILE, split, N_VIZ, viz_dir)

    print("=" * 80 + "\n")


# ===========================================================================
# CALLBACKS
# ===========================================================================

class EpochProgressCallback(keras.callbacks.Callback):
    """Intra-epoch progress bar printed every `log_every` batches."""

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
                f"loss={loss:.6f}  elapsed={elapsed:6.1f}s  eta={remaining:5.1f}s"
            )

    def on_epoch_end(self, epoch, logs=None):
        logs     = logs or {}
        elapsed  = time.time() - self._t0_epoch
        val_loss = logs.get('val_loss',        float('nan'))
        val_psnr = logs.get('val_psnr_metric', float('nan'))
        val_ssim = logs.get('val_ssim_metric', float('nan'))
        print(
            f"  → Epoch {self._epoch} done in {elapsed:.1f}s  "
            f"val_loss={val_loss:.6f}  PSNR={val_psnr:.2f}dB  SSIM={val_ssim:.4f}"
        )


class FullTrainingCheckpoint(keras.callbacks.Callback):
    """Saves best_model_0_XY_1_perc.keras (by val_loss) + last_checkpoint each epoch."""

    def __init__(self, checkpoint_dir, monitor='val_loss'):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor   = monitor
        self.best      = np.inf
        self.meta_path = self.checkpoint_dir / "training_meta_0_XY_1_perc.json"

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

        last_path = self.checkpoint_dir / "last_checkpoint_0_XY_1_perc.keras"
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
            best_path = self.checkpoint_dir / "best_model_0_XY_1_perc.keras"
            self.model.save(best_path)
            meta['best_val_loss'] = float(self.best)
            print(f"[CKPT]  * New best        : val_loss={current:.6f} → saved {best_path.name}")

        with open(self.meta_path, 'w') as f:
            json.dump(meta, f, indent=2)


class EpochMonitoringCallback(keras.callbacks.Callback):
    """
    Saves a monitoring figure at the end of each epoch.

    Layout: 3 rows (Train / Val / Test) x 4 cols:
      Ch0 Sparse (1%) | Ch1 Smoothed (Gauss) | UNET Output | Ground Truth

    Val/Test use a fixed-seed draw (reproducible across epochs).
    Train uses a fresh random index + new Binomial draw each epoch.

    Output: <monitoring_dir>/UNET0XY_1perc_epoch{NNN:03d}_monitoring.png
    """

    def __init__(self, h5_path, monitoring_dir):
        super().__init__()
        self.h5_path        = h5_path
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)

        _rng = np.random.default_rng(seed=0)

        with h5py.File(self.h5_path, 'r') as f:
            val_raw        = f['val/poca'][0,  :, :, 0:1]
            self._val_x    = _apply_low_flux(val_raw,  rng=_rng)
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

        idx = np.random.randint(0, self._n_train)
        with h5py.File(self.h5_path, 'r') as f:
            train_raw = f['train/poca'][idx, :, :, 0:1]
            train_x   = _apply_low_flux(train_raw)
            train_y   = f['train/gt'][idx, :, :, 0:1].astype(np.float32)

        # Single forward pass for all 3 samples  (3, 128, 128, 1)
        batch = np.stack([train_x, self._val_x, self._test_x], axis=0)
        preds = self.model.predict(batch, verbose=0)            # (3, 128, 128, 1)

        rows = [
            ('Train  (random)',   train_x,        preds[0], train_y),
            ('Val    (fixed #0)', self._val_x,    preds[1], self._val_y),
            ('Test   (fixed #0)', self._test_x,   preds[2], self._test_y),
        ]

        val_loss = logs.get('val_loss',        float('nan'))
        val_psnr = logs.get('val_psnr_metric', float('nan'))
        val_ssim = logs.get('val_ssim_metric', float('nan'))

        col_titles = [
            f'Input POCA ({int(FLUX_FRACTION*100)} %)',
            'UNET Output',
            'Ground Truth',
        ]

        fig, axes = plt.subplots(3, 3, figsize=(13, 11))
        fig.patch.set_facecolor('white')
        fig.suptitle(
            f"UNET0XY_1perc  |  Epoch {epoch_num:03d}/{self.params['epochs']}"
            f"    val_loss={val_loss:.5f}    PSNR={val_psnr:.2f}dB    SSIM={val_ssim:.4f}",
            fontsize=11, fontweight='bold', y=0.998,
        )

        for row_idx, (row_label, inp, pred, gt) in enumerate(rows):
            src  = inp[..., 0]
            pred = pred[..., 0]
            gt   = gt[..., 0]
            imgs = [src, pred, gt]
            vmax_row = max(float(x.max()) for x in imgs) or 1e-6
            for col_idx, (img, col_title) in enumerate(zip(imgs, col_titles)):
                ax = axes[row_idx, col_idx]
                im = ax.imshow(img, cmap='viridis', vmin=0, vmax=vmax_row,
                               aspect='equal', interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(col_title, fontsize=10, fontweight='bold', pad=6)
                if col_idx == 0:
                    ax.set_ylabel(row_label, fontsize=9, fontweight='bold', labelpad=6)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=6)

        plt.tight_layout(rect=[0, 0, 1, 0.962])
        out_path = self.monitoring_dir / f"UNET0XY_1perc_epoch{epoch_num:03d}_monitoring.png"
        plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[MON]   Epoch {epoch_num:03d} → {out_path.name}")


if __name__ == "__main__":
    main()
