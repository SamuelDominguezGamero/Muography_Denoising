#!/usr/bin/env python3
"""
UNET2D training for POCA denoising (YZ channel).

── Usage ───────────────────────────────────────────────────────────────────────
  Edit the CONFIG section, then:
      python3 train_unet0YZ_definitive.py

── Key CONFIG variables ────────────────────────────────────────────────────────
  RUN_SELECTION   "run0", "run1", "run2", ["run0","run2"], or "all"
  FLUX_FRACTION   0.0 = full flux;  0.1 = 10% (Binomial downsampling)
  EPOCHS / BATCH_SIZE / N_FILTERS / N_LEVELS  — model & training knobs
  FORCE_RESTART   True  = start from scratch;  False = resume from checkpoint
  EVAL_ONLY       True  = skip training, evaluate best model only
  RUN_NAME        ""    = auto-name from hyperparams (recommended)
                  "my_exp" = use this name for the output folder

── Output folder (one per experiment, auto-named) ──────────────────────────────
  data/models_YZ/<run_id>/
  ├── run_config.json          ← all hyperparameters, saved at start
  ├── training_log_2_YZ.csv   ← loss / PSNR / SSIM per epoch
  ├── model_summary_2_YZ.txt  ← architecture
  ├── data_indices/
  │   └── train.csv  val.csv  test.csv  ← reproducible split (x_path, y_path)
  └── checkpoints/
      ├── best_model_2_YZ.keras
      ├── last_checkpoint_2_YZ.keras
      └── epoch_monitoring/
          └── UNET2YZ_epochNNN_monitoring.png

  Auto-name format:  <runs>__loss_<tag>__flux<pct>__bs<batch>__f<filters>__l<levels>__seed<seed>
  Example:           run0__loss_charb__flux100__bs32__f64__l4__seed42

  Same CONFIG → same folder (safe to resume).
  Change any hyperparameter → new folder → experiments never overwrite each other.

── Data sources ────────────────────────────────────────────────────────────────
  run0: /data/simulation_data/run0_definitive_words  (14k+ files)
  run1: /data/simulation_data/run1_definitive_forms
  run2: /data/simulation_data/run2_definitive_blocks
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


# ===========================================================================
# CONFIG
# ===========================================================================

BASE     = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration")
DATA     = BASE / "data"
SIM_DATA = DATA / "simulation_data"
MODELS   = DATA / "models_YZ"

# ── Projection (fixed for this script — do not change) ──────────────────────
_PROJECTION  = "YZ"   # projection name, used in labels and run_config.json
_CH_IDX      = 2      # channel index in the (128, 128, 3) NPY arrays
_MODEL_LABEL = "2_YZ" # suffix used in checkpoint file names

# ── Data folder registry ────────────────────────────────────────────────────
# Select which runs to use: "run0", "run1", "run2", a list like ["run0", "run2"], or "all"
RUN_SELECTION = "run0"

RUN_FOLDERS = {
    "run0": SIM_DATA / "run0_definitive_words",
    "run1": SIM_DATA / "run1_definitive_forms",
    "run2": SIM_DATA / "run2_definitive_blocks",
}

# Ground truth folders — paired by filename with simulation data
GT_FOLDERS = {
    "run0": DATA / "ground_truth_data" / "run0_letters",
    "run1": DATA / "ground_truth_data" / "run1_geometries_variety",
    "run2": DATA / "ground_truth_data" / "run2_blocks",
}

# ── Loss function ─────────────────────────────────────────────────────────
# Controls which loss is used for training and appears in the folder name.
#   "charbonnier" — smooth L1 (sqrt(e² + ε²)), edge-preserving, robust to outliers
#   "mae"         — Mean Absolute Error (L1), simple, robust
#   "mse"         — Mean Squared Error (L2), penalises large errors more, tends to blur
LOSS_FN = "charbonnier"

# ── Experiment naming ──────────────────────────────────────────────────────
# Leave empty to auto-generate: run0__loss_charb__flux100__bs32__f64__l4__seed42
# Set a string to use a fixed folder name: "my_experiment_v2"
RUN_NAME = ""

# Hyperparameters
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
EPOCHS        = 100
SIZE_IMAGES   = 128
N_FILTERS     = 64       # pure power of 2 — optimal for Tensor Cores
FILTER_SIZE   = 3
N_LEVELS      = 4

USE_GPU            = True
FORCE_RESTART      = True    # If True, ignore previous checkpoints and train from scratch
USE_AUGMENTATION   = True   # On-the-fly rotations ×4 (train split only)
EVAL_ONLY          = False  # If True, skip training and load best model → evaluate + visualize only
VISUALIZE_RESULTS  = True   # If True, save input/pred/GT comparison PNGs after training
N_VIZ              = 8      # Samples per split to visualize

# ── Train/val/test split fractions ─────────────────────────────────────────
VAL_FRACTION      = 0.15
TEST_FRACTION     = 0.15
RANDOM_SEED       = 42

# ── Dynamic subsampling (regularization on-the-fly) ────────────────────────────
# If > 0, use only this fraction of training data per epoch (different random
# fraction each epoch). Useful to combat overfitting and reduce epoch duration.
# Set to 0.0 to disable. Examples: 0.5 = 50%, 0.3 = 30%, 0.0 = full dataset.
SUBSAMPLE_FRACTION = 0.4

# ── Low-flux simulation (Binomial downsampling of POCA counts) ──────────────────
# If > 0, apply Binomial(counts, FLUX_FRACTION) to X before feeding the network.
# Physically: simulates acquiring data with a reduced muon flux (shorter exposure).
# Pixels with few counts become empty; spatial distribution is statistically preserved.
# Set to 0.0 to disable (full flux, no downsampling).
# Examples: 0.1 = 10% flux, 0.3 = 30% flux, 1.0 = full flux (same as 0.0).
FLUX_FRACTION = 0.0


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

    # DECODER  — Conv2DTranspose: learned upsampling (valid without groups)
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


def mae_loss(y_true, y_pred):
    """Mean Absolute Error (L1)."""
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def mse_loss(y_true, y_pred):
    """Mean Squared Error (L2)."""
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(tf.square(y_true - y_pred))


_LOSS_REGISTRY = {
    "charbonnier": charbonnier_loss,
    "mae":         mae_loss,
    "mse":         mse_loss,
}

_LOSS_TAG = {
    "charbonnier": "charb",
    "mae":         "mae",
    "mse":         "mse",
}


def _get_loss_fn():
    """Return the loss function selected by LOSS_FN config variable."""
    fn = _LOSS_REGISTRY.get(LOSS_FN)
    if fn is None:
        raise ValueError(f"Unknown LOSS_FN='{LOSS_FN}'. Valid: {list(_LOSS_REGISTRY)}")
    return fn


def psnr_metric(y_true, y_pred):
    """Peak Signal-to-Noise Ratio (dB). Higher = better."""
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def ssim_metric(y_true, y_pred):
    """SSIM with 5×5 filter (lighter than tf.image.ssim's 11×11). Range [0, 1], higher = better."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=5, filter_sigma=1.0)


# ===========================================================================
# DATA LOADING — NPY from multiple folders
# ===========================================================================

def _apply_low_flux(poca_raw, flux_fraction):
    """Binomial downsampling of raw POCA counts to simulate low muon flux.

    Each pixel count v is treated as v independent muon hits; each is retained
    with probability flux_fraction. Result: v_sparse ~ Binomial(v, flux_fraction).
    Returns float32 in [0, 1] (per-image max normalisation).
    """
    sparse = np.random.binomial(poca_raw.astype(np.int32), flux_fraction).astype(np.float32)
    x_max = float(sparse.max())
    return sparse / max(x_max, 1.0)

def _resolve_runs(run_sel):
    """Convert run_sel (str/list) to list of run names."""
    if isinstance(run_sel, str):
        return list(RUN_FOLDERS.keys()) if run_sel.lower() == "all" else [run_sel]
    return list(run_sel)


def _x_to_y_path(x_path, gt_folder):
    """Derive Y ground truth path from X simulation path by filename convention.

    Two naming conventions exist across runs:

    run0  X: tensor_2D_POCA_<payload>[_Muons_100000_2D].npy
    run0  Y: tensor_2D__<payload>.npy          ← double underscore, no POCA_

    run1/2 X: tensor_2D_POCA_POCA_merged_<payload>_Muons_100000_2D.npy
    run1/2 Y: tensor_2D_<payload>.npy          ← single underscore, no POCA_POCA_merged_
    """
    stem = x_path.stem.replace("_Muons_100000_2D", "")

    if stem.startswith("tensor_2D_POCA_POCA_merged_"):
        # run1 / run2
        y_stem = "tensor_2D_" + stem[len("tensor_2D_POCA_POCA_merged_"):]
    else:
        # run0
        y_stem = stem.replace("tensor_2D_POCA_", "tensor_2D__", 1)

    return gt_folder / f"{y_stem}.npy"


def _scan_npy_pairs(run_names):
    """Scan run folders for (X, Y) paired files. Returns sorted list of (x_path, y_path)."""
    pairs = []
    for run in run_names:
        x_folder = RUN_FOLDERS[run]
        y_folder = GT_FOLDERS[run]
        if not x_folder.is_dir():
            print(f"[WARN]  X folder not found: {x_folder}")
            continue
        if not y_folder.is_dir():
            print(f"[WARN]  Y folder not found: {y_folder}")
            continue
        found, missing = 0, 0
        for x_path in sorted(x_folder.glob("*.npy")):
            y_path = _x_to_y_path(x_path, y_folder)
            if y_path.exists():
                pairs.append((x_path, y_path))
                found += 1
            else:
                missing += 1
        print(f"[DATA]  {run:5s}  →  {found:6d} pairs  ({missing} unmatched X files)")
    if not pairs:
        raise RuntimeError(f"No paired (X, Y) files found for runs: {run_names}")
    return pairs


def _train_val_test_split(pairs, val_frac=VAL_FRACTION, test_frac=TEST_FRACTION, seed=RANDOM_SEED):
    """Deterministic train/val/test split of (x, y) pairs. Returns dict with 'train'/'val'/'test' keys."""
    files = sorted(pairs)
    n = len(files)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_test, n_val = max(1, int(n * test_frac)), max(1, int(n * val_frac))
    n_train = n - n_test - n_val
    return {
        'train': [files[i] for i in sorted(indices[:n_train])],
        'val':   [files[i] for i in sorted(indices[n_train:n_train+n_val])],
        'test':  [files[i] for i in sorted(indices[n_train+n_val:])],
    }


class NPYDataGenerator:
    """Streams YZ channel from NPY files with optional per-epoch subsampling."""

    def __init__(self, file_list, split_name, augment=False, subsample_frac=0.0, flux_fraction=0.0):
        self.file_list      = list(file_list)
        self.split_name     = split_name
        self.augment        = augment
        self.subsample_frac = subsample_frac
        self.flux_fraction  = flux_fraction
        self.n_files        = len(self.file_list)

    def __call__(self):
        """Generator: yields (x, y) tuples. Each epoch may use different subsample."""
        indices = np.arange(self.n_files)

        # Per-epoch random subsampling (train only)
        if self.split_name == 'train' and self.subsample_frac > 0:
            n_subsample = max(1, int(self.n_files * self.subsample_frac))
            indices = np.random.choice(indices, size=n_subsample, replace=False)

        if self.split_name == 'train':
            np.random.shuffle(indices)

        for idx in indices:
            try:
                x_path, y_path = self.file_list[idx]
                x_data = np.load(x_path, allow_pickle=False).astype(np.float32)
                y_data = np.load(y_path, allow_pickle=False).astype(np.float32)
                if self.flux_fraction > 0:
                    x = _apply_low_flux(x_data[:, :, _CH_IDX], self.flux_fraction)[..., np.newaxis]
                else:
                    x_max = float(x_data[:, :, _CH_IDX].max())
                    x = x_data[:, :, _CH_IDX:_CH_IDX+1] / max(x_max, 1.0)  # per-image normalization → [0, 1]
                y = y_data[:, :, _CH_IDX:_CH_IDX+1]    # GT binary mask, already in [0, 1]

                if self.augment:
                    k = np.random.randint(0, 4)  # Random 90° rotation
                    if k > 0:
                        x = np.rot90(x, k, axes=(0, 1))
                        y = np.rot90(y, k, axes=(0, 1))

                yield x, y
            except Exception as e:
                print(f"[WARN]  Failed to load pair: {e}")


def get_dataset(file_list, split_name, batch_size, shuffle=False, augment=False, subsample_frac=0.0, flux_fraction=0.0):
    """Build tf.data pipeline from NPY files."""
    gen = NPYDataGenerator(file_list, split_name, augment=augment, subsample_frac=subsample_frac, flux_fraction=flux_fraction)
    output_sig = (
        tf.TensorSpec(shape=(SIZE_IMAGES, SIZE_IMAGES, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(SIZE_IMAGES, SIZE_IMAGES, 1), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_sig)
    ds = ds.repeat()
    if shuffle:
        ds = ds.shuffle(buffer_size=256, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=shuffle).prefetch(tf.data.AUTOTUNE)
    return ds


# ===========================================================================
# UTILITIES
# ===========================================================================

def _count_files(splits_dict):
    """Count files in each split."""
    return {k: len(v) for k, v in splits_dict.items()}


def _save_split_csvs(splits, output_dir):
    """Save train/val/test splits to CSV files for reproducibility and external use."""
    import csv
    csv_dir = Path(output_dir) / "data_indices"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for split_name, pairs in splits.items():
        csv_path = csv_dir / f"{split_name}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x_path', 'y_path'])
            for x_path, y_path in pairs:
                writer.writerow([str(x_path), str(y_path)])
    print(f"[DATA]  Split CSVs saved  : {csv_dir}/  (train/val/test.csv)")


def _check_config_match(output_dir: Path) -> bool:
    """Validate that the current script config matches the run_config.json in output_dir.

    Called before resuming an existing experiment to catch accidental
    hyperparameter mismatches early — before overwriting a good checkpoint.

    Returns True if everything matches (safe to resume), False otherwise.
    """
    cfg_path = output_dir / "run_config.json"
    if not cfg_path.exists():
        return True  # no config to compare — treat as first run

    with open(cfg_path) as f:
        saved = json.load(f)

    current_run = RUN_SELECTION if isinstance(RUN_SELECTION, str) else list(RUN_SELECTION)
    checks = {
        "projection":         _PROJECTION,
        "channel_index":      _CH_IDX,
        "run_selection":      current_run,
        "flux_fraction":      FLUX_FRACTION,
        "subsample_fraction": SUBSAMPLE_FRACTION,
        "batch_size":         BATCH_SIZE,
        "loss_fn":            LOSS_FN,
        "n_filters":          N_FILTERS,
        "n_levels":           N_LEVELS,
        "filter_size":        FILTER_SIZE,
        "val_fraction":       VAL_FRACTION,
        "test_fraction":      TEST_FRACTION,
        "random_seed":        RANDOM_SEED,
    }

    mismatches = []
    for key, current_val in checks.items():
        saved_val = saved.get(key)
        if saved_val is None:
            continue  # key not present in older config — skip (backwards compat)
        if saved_val != current_val:
            mismatches.append((key, saved_val, current_val))

    if mismatches:
        print("[WARN]  Hyperparameter mismatches with existing run_config.json:")
        for key, old, new in mismatches:
            print(f"  {key:25s}  saved={old!r}  current={new!r}")
        print("[WARN]  Set FORCE_RESTART=True to start a new experiment, "
              "or revert the config variables to the saved values to resume safely.\n")
        return False

    return True


def _build_output_dir() -> Path:
    """Return the experiment output directory.

    Uses RUN_NAME if set; otherwise auto-generates a name that encodes the key
    hyperparameters so each distinct configuration gets its own folder.
    Example: run0__loss_charb__flux100__bs32__f64__l4__seed42
    """
    if RUN_NAME:
        return MODELS / RUN_NAME
    run_names = _resolve_runs(RUN_SELECTION)
    run_tag  = "_".join(run_names)
    flux_pct = int(round(FLUX_FRACTION * 100)) if FLUX_FRACTION > 0 else 100
    loss_tag = _LOSS_TAG.get(LOSS_FN, LOSS_FN)
    name = f"{run_tag}__loss_{loss_tag}__flux{flux_pct}__bs{BATCH_SIZE}__f{N_FILTERS}__l{N_LEVELS}__seed{RANDOM_SEED}"
    return MODELS / name


def _save_run_config(output_dir: Path, splits: dict) -> None:
    """Dump all hyperparameters + dataset sizes to run_config.json."""
    import datetime
    config = {
        "run_id":             output_dir.name,
        "timestamp_start":    datetime.datetime.now().isoformat(timespec='seconds'),
        "projection":         _PROJECTION,
        "channel_index":      _CH_IDX,
        "run_selection":      RUN_SELECTION if isinstance(RUN_SELECTION, str) else list(RUN_SELECTION),
        "flux_fraction":      FLUX_FRACTION,
        "subsample_fraction": SUBSAMPLE_FRACTION,
        "batch_size":         BATCH_SIZE,
        "learning_rate":      LEARNING_RATE,
        "epochs":             EPOCHS,
        "loss_fn":            LOSS_FN,
        "n_filters":          N_FILTERS,
        "n_levels":           N_LEVELS,
        "filter_size":        FILTER_SIZE,
        "use_augmentation":   USE_AUGMENTATION,
        "val_fraction":       VAL_FRACTION,
        "test_fraction":      TEST_FRACTION,
        "random_seed":        RANDOM_SEED,
        "n_train":            len(splits['train']),
        "n_val":              len(splits['val']),
        "n_test":             len(splits['test']),
        "output_dir":         str(output_dir),
    }
    out_path = output_dir / "run_config.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[CFG]   run_config.json  : {out_path}")


def _visualize_split(model, pairs, split, n_samples, output_dir):
    """Save a PNG grid with columns: Input (POCA) | UNET Output | Ground Truth."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN]  matplotlib not available — skipping visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    n = min(n_samples, len(pairs))
    pocas = []
    gts   = []

    for x_path, y_path in pairs[:n]:
        try:
            x_data = np.load(x_path, allow_pickle=False).astype(np.float32)
            y_data = np.load(y_path, allow_pickle=False).astype(np.float32)
            if FLUX_FRACTION > 0:
                pocas.append(_apply_low_flux(x_data[:, :, _CH_IDX], FLUX_FRACTION))
            else:
                x_max = float(x_data[:, :, _CH_IDX].max())
                pocas.append(x_data[:, :, _CH_IDX] / max(x_max, 1.0))
            gts.append(y_data[:, :, _CH_IDX])
        except Exception as e:
            print(f"[WARN]  Failed to load {x_path.name}: {e}")

    pocas = np.array(pocas, dtype=np.float32)
    gts   = np.array(gts,   dtype=np.float32)

    if len(pocas) == 0:
        print(f"[WARN]  No samples could be loaded for {split}")
        return

    preds = model.predict(pocas[..., np.newaxis], batch_size=8, verbose=0)[..., 0]

    fig, axes = plt.subplots(len(pocas), 3, figsize=(9, 3 * len(pocas)))
    if len(pocas) == 1:
        axes = axes[np.newaxis, :]
    col_titles = ['Input (POCA)', 'UNET Output', 'Ground Truth']
    for row, (src, pred, gt) in enumerate(zip(pocas, preds, gts)):
        vmax_row = max(float(src.max()), float(pred.max()), float(gt.max()), 1e-6)
        for col, (img, title) in enumerate(zip([src, pred, gt], col_titles)):
            ax = axes[row, col]
            ax.imshow(img, cmap='viridis', vmin=0, vmax=vmax_row)
            ax.axis('off')
            if row == 0:
                ax.set_title(title, fontsize=10)

    plt.suptitle(f"UNET2YZ — {split} samples", fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = output_dir / f"viz_{split}.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[VIZ]   {split:5s} → {out_path.name}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    OUTPUT_DIR = _build_output_dir()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f" UNET2D-YZ TRAINING — POCA Denoising, YZ channel ({LOSS_FN} loss)")
    print(f" Output : {OUTPUT_DIR}")
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
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.total,memory.free,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    total, free, used = result.stdout.strip().split(", ")
                    print(f"[HW]    GPU detected     : {len(gpus)} × {gpu_name}")
                    print(f"[HW]    VRAM total       : {int(total):,} MiB")
                    print(f"[HW]    VRAM free        : {int(free):,} MiB  (used by other processes: {int(used):,} MiB)")
                    if int(free) < 2000:
                        print(f"[WARN]  Less than 2 GB free — consider closing browser/desktop apps")
            except Exception:
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

    # ── Data ────────────────────────────────────────────────────────────────
    print(f"[DATA]  RUN_SELECTION  : {RUN_SELECTION}")
    run_names = _resolve_runs(RUN_SELECTION)
    print(f"[DATA]  Resolved runs  : {run_names}")

    all_pairs = _scan_npy_pairs(run_names)
    print(f"[DATA]  Total pairs    : {len(all_pairs)}")

    splits = _train_val_test_split(all_pairs, val_frac=VAL_FRACTION, test_frac=TEST_FRACTION, seed=RANDOM_SEED)
    sizes  = _count_files(splits)
    _save_split_csvs(splits, OUTPUT_DIR)
    _save_run_config(OUTPUT_DIR, splits)

    steps_train = sizes['train'] // BATCH_SIZE
    steps_val   = sizes['val']   // BATCH_SIZE
    steps_test  = sizes['test']  // BATCH_SIZE

    print(f"[DATA]  Source           : NPY files from {', '.join(run_names)}")
    print(f"[DATA]  Splits           : train={sizes['train']}  val={sizes['val']}  test={sizes['test']}")
    print(f"[DATA]  Batch size       : {BATCH_SIZE}  ({steps_train} steps/epoch)")
    print(f"[DATA]  Augmentation     : {'rotations ×4 (train only)' if USE_AUGMENTATION else 'none'}")
    if FLUX_FRACTION > 0:
        print(f"[DATA]  Low-flux mode    : Binomial(counts, {FLUX_FRACTION})  →  {FLUX_FRACTION*100:.0f}% flux")

    train_ds = get_dataset(splits['train'], 'train', BATCH_SIZE, shuffle=True,  augment=USE_AUGMENTATION, subsample_frac=SUBSAMPLE_FRACTION, flux_fraction=FLUX_FRACTION)
    val_ds   = get_dataset(splits['val'],   'val',   BATCH_SIZE, shuffle=False, augment=False, flux_fraction=FLUX_FRACTION)
    test_ds  = get_dataset(splits['test'],  'test',  BATCH_SIZE, shuffle=False, augment=False, flux_fraction=FLUX_FRACTION)

    # ──────────────────────────────────────────────────────────────────────────
    # EVAL_ONLY MODE: Load best model and skip training
    # ──────────────────────────────────────────────────────────────────────────
    if EVAL_ONLY:
        print("\n" + "#" * 80)
        print("###  EVAL_ONLY MODE: Loading best model and skipping training")
        print("#" * 80)

        CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
        best_model_path = CHECKPOINT_DIR / f"best_model_{_MODEL_LABEL}.keras"

        if not best_model_path.exists():
            sys.exit(f"[ERROR] Best model not found: {best_model_path}")

        best_model = keras.models.load_model(
            str(best_model_path),
            custom_objects={
                'charbonnier_loss': charbonnier_loss,
                'mae_loss':         mae_loss,
                'mse_loss':         mse_loss,
                'psnr_metric':      psnr_metric,
                'ssim_metric':      ssim_metric,
            },
        )
        print(f"[EVAL]  Best model loaded : {best_model_path.name}")

        print("\n" + "=" * 80)
        print(" EVALUATION ON TEST SET")
        print("=" * 80)
        results = best_model.evaluate(test_ds, steps=steps_test, verbose=0, return_dict=True)
        print(f"[EVAL]  Test {LOSS_FN:<12} : {results['loss']:.6f}")
        print(f"[EVAL]  Test PSNR        : {results['psnr_metric']:.3f} dB")
        print(f"[EVAL]  Test SSIM        : {results['ssim_metric']:.4f}")

        if VISUALIZE_RESULTS:
            viz_dir = OUTPUT_DIR / "visualizations_YZ"
            print(f"\n[VIZ]   Saving visualizations → {viz_dir}/")
            for split_name, file_list in splits.items():
                _visualize_split(best_model, file_list, split_name, N_VIZ, viz_dir)

        print("=" * 80 + "\n")
        return

    # ── Model ───────────────────────────────────────────────────────────────
    model     = unet_2d()
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=_get_loss_fn(),
        metrics=[psnr_metric, ssim_metric],
        jit_compile=False,
    )

    n_params = model.count_params()
    print(f"[MODEL] Architecture     : UNET2D standard (single-channel YZ)")
    print(f"[MODEL] Levels / filters : {N_LEVELS} levels, base={N_FILTERS}, kernel={FILTER_SIZE}×{FILTER_SIZE}")
    print(f"[MODEL] Parameters       : {n_params:,}")
    print(f"[MODEL] Loss / metrics   : {LOSS_FN} / PSNR, SSIM(5×5)")
    print(f"[MODEL] Optimizer        : Adam(lr={LEARNING_RATE})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = OUTPUT_DIR / f"model_summary_{_MODEL_LABEL}.txt"
    with open(summary_file, "w") as f:
        model.summary(line_length=110, print_fn=lambda s: f.write(s + "\n"))
    print(f"[MODEL] Full summary     : {summary_file}")

    # ── Checkpoint / resume ─────────────────────────────────────────────────
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path     = CHECKPOINT_DIR / f"training_meta_{_MODEL_LABEL}.json"
    initial_epoch = 0

    print("\n" + "#" * 80)
    if FORCE_RESTART:
        print("###  TRAINING MODE : FROM SCRATCH (FORCE_RESTART=True)")
        print("###  Any previous checkpoint will be IGNORED and OVERWRITTEN.")
        if meta_path.exists():
            meta_path.unlink()
    elif meta_path.exists():
        config_ok = _check_config_match(OUTPUT_DIR)

        with open(meta_path) as f:
            meta = json.load(f)
        initial_epoch = meta.get('last_epoch', 0)
        last_ckpt = CHECKPOINT_DIR / f"last_checkpoint_{_MODEL_LABEL}.keras"
        if last_ckpt.exists() and config_ok:
            model = keras.models.load_model(
                str(last_ckpt),
                custom_objects={
                    'charbonnier_loss': charbonnier_loss,
                    'mae_loss':         mae_loss,
                    'mse_loss':         mse_loss,
                    'psnr_metric':      psnr_metric,
                    'ssim_metric':      ssim_metric,
                },
            )
            print(f"###  TRAINING MODE : RESUMING from epoch {initial_epoch + 1}")
            print(f"###  Loaded checkpoint : {last_ckpt.name}")
            print(f"###  Previous best val_loss : {meta.get('best_val_loss', 'n/a')}")
        else:
            print("###  TRAINING MODE : FROM SCRATCH")
            if not config_ok:
                print("###  (hyperparameter mismatch — not safe to resume)")
            else:
                print("###  (meta found but no checkpoint file — starting fresh)")
            initial_epoch = 0
    else:
        print("###  TRAINING MODE : FROM SCRATCH (no previous run detected)")
    print("#" * 80)

    # ── Callbacks ───────────────────────────────────────────────────────────
    callbacks = [
        FullTrainingCheckpoint(CHECKPOINT_DIR, monitor='val_loss'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=False, verbose=1),
        keras.callbacks.CSVLogger(OUTPUT_DIR / f"training_log_{_MODEL_LABEL}.csv", append=True),
        EpochProgressCallback(steps_per_epoch=steps_train, log_every=20),
        EpochMonitoringCallback(splits, CHECKPOINT_DIR / "epoch_monitoring"),
    ]

    print("\n" + "-" * 80)
    print(f" Training: epochs {initial_epoch + 1} → {EPOCHS}")
    if SUBSAMPLE_FRACTION > 0:
        print(f" Subsampling: {SUBSAMPLE_FRACTION*100:.0f}% per epoch (dynamic)")
    print("-" * 80)

    model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=int(steps_train * max(SUBSAMPLE_FRACTION, 1.0)),
        initial_epoch=initial_epoch,
        validation_data=val_ds,
        validation_steps=steps_val,
        callbacks=callbacks,
        verbose=0,
    )

    elapsed = time.time() - start_time
    print("-" * 80)
    print(f"[TIME]  Training finished : {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # ── Final test evaluation with best weights ─────────────────────────────
    best_model_path = CHECKPOINT_DIR / f"best_model_{_MODEL_LABEL}.keras"
    print("\n" + "=" * 80)
    print(" FINAL EVALUATION ON TEST SET (best val_loss checkpoint)")
    print("=" * 80)
    best_model = None
    try:
        best_model = keras.models.load_model(
            str(best_model_path),
            custom_objects={
                'charbonnier_loss': charbonnier_loss,
                'mae_loss':         mae_loss,
                'mse_loss':         mse_loss,
                'psnr_metric':      psnr_metric,
                'ssim_metric':      ssim_metric,
            },
        )
        print(f"[EVAL]  Best model       : {best_model_path.name}")
        results = best_model.evaluate(test_ds, steps=steps_test, verbose=0, return_dict=True)
        print(f"[EVAL]  Test {LOSS_FN:<12} : {results['loss']:.6f}")
        print(f"[EVAL]  Test PSNR        : {results['psnr_metric']:.3f} dB")
        print(f"[EVAL]  Test SSIM        : {results['ssim_metric']:.4f}")
    except Exception as e:
        print(f"[ERROR] Could not evaluate best model on test: {e}")

    final_model_path = OUTPUT_DIR / f"model_epoch_final_{_MODEL_LABEL}.keras"
    model.save(final_model_path)
    print(f"[SAVE]  Final-epoch model : {final_model_path.name}")

    # ── Optional visualization ───────────────────────────────────────────────
    if VISUALIZE_RESULTS:
        eval_model = best_model if best_model is not None else model
        viz_dir = OUTPUT_DIR / "visualizations_YZ"
        print(f"\n[VIZ]   Saving visualizations → {viz_dir}/")
        for split_name, file_list in splits.items():
            _visualize_split(eval_model, file_list, split_name, N_VIZ, viz_dir)

    print("=" * 80 + "\n")
    return


# ===========================================================================
# CALLBACKS
# ===========================================================================

class EpochProgressCallback(keras.callbacks.Callback):
    """Print training progress per epoch."""
    def __init__(self, steps_per_epoch, log_every=20):
        super().__init__()
        self.steps = steps_per_epoch
        self.log_every = log_every
        self._t0 = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()
        print(f"\n[TRAIN] Epoch {epoch + 1}/{self.params['epochs']}")

    def on_train_batch_end(self, batch, logs=None):
        step = batch + 1
        if step % self.log_every == 0 or step == self.steps:
            pct = 100 * step / self.steps
            loss = logs.get('loss', float('nan'))
            elapsed = time.time() - self._t0
            print(f"  step {step:4d}/{self.steps}  [{pct:5.1f}%]  loss={loss:.6f}  {elapsed:5.1f}s")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.time() - self._t0
        val_loss = logs.get('val_loss', float('nan'))
        val_psnr = logs.get('val_psnr_metric', float('nan'))
        val_ssim = logs.get('val_ssim_metric', float('nan'))
        print(f"  val_loss={val_loss:.6f}  PSNR={val_psnr:.2f}dB  SSIM={val_ssim:.4f}  ({elapsed:.1f}s)")


class EpochMonitoringCallback(keras.callbacks.Callback):
    """Save epoch monitoring figure: 3 rows (Train/Val/Test) × 3 cols (Input/Output/GT)."""

    def __init__(self, file_splits, monitoring_dir):
        super().__init__()
        self.file_splits = file_splits
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        self.n_train = len(file_splits['train'])

        # Pre-cache val/test fixed samples (index 0)
        self._val_x, self._val_y = self._load_sample(file_splits['val'][0])
        self._test_x, self._test_y = self._load_sample(file_splits['test'][0])

    def _load_sample(self, pair):
        """Load a (x_path, y_path) pair, extract YZ channel, return (x, y)."""
        x_path, y_path = pair
        x_data = np.load(x_path, allow_pickle=False).astype(np.float32)
        y_data = np.load(y_path, allow_pickle=False).astype(np.float32)
        if FLUX_FRACTION > 0:
            x = _apply_low_flux(x_data[:, :, _CH_IDX], FLUX_FRACTION)[..., np.newaxis]
        else:
            x_max = float(x_data[:, :, _CH_IDX].max())
            x = x_data[:, :, _CH_IDX:_CH_IDX+1] / max(x_max, 1.0)
        y = y_data[:, :, _CH_IDX:_CH_IDX+1]    # GT binary mask [0, 1]
        return x, y

    def on_epoch_end(self, epoch, logs=None):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN]  matplotlib not available")
            return

        epoch_num = epoch + 1
        logs = logs or {}

        # Random train sample
        idx_train = np.random.randint(0, self.n_train)
        train_x, train_y = self._load_sample(self.file_splits['train'][idx_train])

        # Single forward pass for all 3 samples
        batch = np.stack([train_x, self._val_x, self._test_x], axis=0)
        preds = self.model.predict(batch, verbose=0)

        rows = [
            ('Train  (random)',   train_x[..., 0], preds[0, ..., 0], train_y[..., 0]),
            ('Val    (fixed #0)', self._val_x[..., 0], preds[1, ..., 0], self._val_y[..., 0]),
            ('Test   (fixed #0)', self._test_x[..., 0], preds[2, ..., 0], self._test_y[..., 0]),
        ]

        val_loss = logs.get('val_loss', float('nan'))
        val_psnr = logs.get('val_psnr_metric', float('nan'))
        val_ssim = logs.get('val_ssim_metric', float('nan'))

        fig, axes = plt.subplots(3, 3, figsize=(13, 11))
        fig.patch.set_facecolor('white')
        fig.suptitle(
            f"UNET2YZ  ·  Epoch {epoch_num:03d} / {self.params['epochs']}"
            f"    |    val_loss = {val_loss:.5f}    PSNR = {val_psnr:.2f} dB    SSIM = {val_ssim:.4f}",
            fontsize=12, fontweight='bold', y=0.995,
        )

        col_titles = ['Input  (POCA)', 'UNET Output  (denoised)', 'Ground Truth']
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
        out_path = self.monitoring_dir / f"UNET2YZ_epoch{epoch_num:03d}_monitoring.png"
        plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[MON]   Epoch {epoch_num:03d} → {out_path.name}")


class FullTrainingCheckpoint(keras.callbacks.Callback):
    """Save best and last model + metadata JSON."""

    def __init__(self, ckpt_dir, monitor='val_loss'):
        super().__init__()
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.best = np.inf
        self.meta_path = self.ckpt_dir / f"training_meta_{_MODEL_LABEL}.json"

        if self.meta_path.exists():
            with open(self.meta_path) as f:
                self.best = json.load(f).get('best_val_loss', np.inf)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        self.model.save(str(self.ckpt_dir / f"last_checkpoint_{_MODEL_LABEL}.keras"))

        meta = {
            'last_epoch': epoch + 1,
            'best_val_loss': float(self.best),
            'last_val_loss': float(current),
            'last_val_psnr': float(logs.get('val_psnr_metric', 0.0)),
            'last_val_ssim': float(logs.get('val_ssim_metric', 0.0)),
        }

        if current < self.best:
            self.best = current
            self.model.save(str(self.ckpt_dir / f"best_model_{_MODEL_LABEL}.keras"))
            meta['best_val_loss'] = float(self.best)
            print(f"[CKPT]  New best: val_loss={current:.6f}")

        with open(self.meta_path, 'w') as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
