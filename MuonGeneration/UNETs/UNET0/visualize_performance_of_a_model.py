#!/usr/bin/env python3
"""
Visualise UNET denoising performance on the muography dataset.

Loads a trained model, takes one or more samples from each split
(train / val / test) and shows a grid:

    rows  : one per (split × sample)
    cols  : Input (POCA) | Prediction | Ground Truth

Usage examples
--------------
  # Full-flux XY model, 1 random sample per split:
  python visualize_performance_of_a_model.py --model XY

  # Ultra-low-flux model, simulate 1 % flux, per-image norm + gamma stretch:
  python visualize_performance_of_a_model.py --model XY_ultralow --flux 0.01 --norm perimage --gamma 0.4

  # XZ model, specific sample indices (one per split: train val test):
  python visualize_performance_of_a_model.py --model XZ --idx 0 42 7

  # Low-flux model, 2 samples per split, per-column norm:
  python visualize_performance_of_a_model.py --model XY_low --flux 0.05 --n-samples 2 --norm percol

Available models
----------------
  XY          : XY projection, full-flux
  XY_low      : XY projection, low-flux fine-tune  (~5 %)
  XY_ultralow : XY projection, ultra-low-flux fine-tune (~1 %), 2-channel input
  XZ          : XZ projection, full-flux
  YZ          : YZ projection, full-flux
"""

import argparse
import os
import sys
import gc
from pathlib import Path

import numpy as np

# ── silence TF before importing ───────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
gc.collect()

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import mixed_precision
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
    keras.backend.clear_session()
    mixed_precision.set_global_policy('mixed_bfloat16')
    tf.config.optimizer.set_jit(False)
except ImportError:
    print("[ERROR] TensorFlow required.  pip install tensorflow")
    sys.exit(1)

try:
    import h5py
except ImportError:
    print("[ERROR] h5py required.  pip install h5py")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('TkAgg')          # interactive window; falls back to Qt if needed
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("[ERROR] matplotlib required.  pip install matplotlib")
    sys.exit(1)

try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    print("[WARN]  scipy not found — 2-channel smoothed input will duplicate ch0")


# ===========================================================================
# PATHS & MODEL REGISTRY
# ===========================================================================

_BASE     = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration")
_DATA     = _BASE / "data"
_H5_FILE  = _DATA / "datasets" / "dataset0.h5"

# channel  : H5 dataset channel index (0=XY, 1=XZ, 2=YZ)
# input_ch : number of channels the model's first layer expects
# flux_def : default flux fraction (what the model was trained with)
MODELS = {
    "XY": {
        "path":     _DATA / "models_XY"         / "checkpoints" / "best_model_0_XY.keras",
        "channel":  0,
        "input_ch": 1,
        "flux_def": 1.0,
        "label":    "XY  full-flux",
    },
    "XY_low": {
        "path":     _DATA / "models_XY_10_perc" / "checkpoints" / "best_model_0_XY_10_perc.keras",
        "channel":  0,
        "input_ch": 1,
        "flux_def": 0.05,
        "label":    "XY  low-flux (5 %)",
    },
    "XY_ultralow": {
        "path":     _DATA / "models_XY_1_perc"  / "checkpoints" / "best_model_0_XY_1_perc.keras",
        "channel":  0,
        "input_ch": 2,
        "flux_def": 0.01,
        "label":    "XY  ultra-low-flux (1 %)",
    },
    "XZ": {
        "path":     _DATA / "models_XZ"         / "checkpoints" / "best_model_0_XZ.keras",
        "channel":  1,
        "input_ch": 1,
        "flux_def": 1.0,
        "label":    "XZ  full-flux",
    },
    "YZ": {
        "path":     _DATA / "models_YZ"         / "checkpoints" / "best_model_0_YZ.keras",
        "channel":  2,
        "input_ch": 1,
        "flux_def": 1.0,
        "label":    "YZ  full-flux",
    },
}

SPLITS = ("train", "val", "test")

_CUSTOM_OBJECTS = {
    'charbonnier_loss': lambda y_true, y_pred: tf.reduce_mean(
        tf.sqrt(tf.square(tf.cast(y_true, y_pred.dtype) - y_pred) + 1e-6)),
    'psnr_metric': lambda y_true, y_pred: tf.image.psnr(
        tf.cast(y_true, y_pred.dtype), y_pred, max_val=1.0),
    'ssim_metric': lambda y_true, y_pred: tf.image.ssim(
        tf.cast(y_true, y_pred.dtype), y_pred, max_val=1.0),
}


# ===========================================================================
# PREPROCESSING
# ===========================================================================

def _apply_flux(poca_uint8_hw, flux_fraction, rng=None):
    """Binomial subsampling of uint8 pixel counts. Returns float32 raw counts."""
    if flux_fraction >= 1.0:
        return poca_uint8_hw.astype(np.float32)
    counts = poca_uint8_hw.astype(np.int32)
    if rng is not None:
        sparse = rng.binomial(counts, flux_fraction)
    else:
        sparse = np.random.binomial(counts, flux_fraction)
    return sparse.astype(np.float32)


def preprocess(poca_uint8_hw, cfg, flux_fraction, sigma=1.0, rng=None):
    """
    Build network input tensor and a display image from a raw (H,W) uint8 patch.

    Returns
    -------
    net_input   : (1, H, W, C)  float32 — ready for model.predict()
    display_img : (H, W)        float32 in [0, 1] — the first channel only
    """
    raw_float = _apply_flux(poca_uint8_hw, flux_fraction, rng=rng)  # float32 counts

    if cfg["input_ch"] == 2:
        # dual-channel: ch0 = sparse/255,  ch1 = gaussian(sparse)/255
        ch0 = (raw_float / 255.0)[..., np.newaxis]          # (H, W, 1)
        if sigma > 0 and _SCIPY_AVAILABLE:
            smoothed = _scipy_gaussian(raw_float, sigma=sigma).astype(np.float32)
            ch1 = (smoothed / 255.0)[..., np.newaxis]       # (H, W, 1)
        else:
            ch1 = ch0.copy()
        net_input = np.concatenate([ch0, ch1], axis=-1)     # (H, W, 2)
        display_img = ch0[..., 0]                           # (H, W)
    else:
        net_input   = (raw_float / 255.0)[..., np.newaxis]  # (H, W, 1)
        display_img = net_input[..., 0]                     # (H, W)

    return net_input[np.newaxis], display_img               # (1,H,W,C), (H,W)


# ===========================================================================
# NORMALISATION FOR DISPLAY
# ===========================================================================

def _stretch_one(img, gamma=1.0, pct_lo=1.0, pct_hi=99.5):
    """Percentile-clip → [0,1] → optional gamma stretch. Returns (img_out, lo, hi)."""
    lo = float(np.percentile(img, pct_lo))
    hi = float(np.percentile(img, pct_hi))
    if hi - lo < 1e-8:
        hi = lo + 1e-8
    img_n = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    if gamma != 1.0:
        img_n = np.power(img_n, gamma)
    return img_n, lo, hi


def normalise_grid(images, mode, gamma, pct_lo=1.0, pct_hi=99.5):
    """
    Normalise a list-of-rows-of-images for display.

    images : list[ list[np.ndarray (H,W)] ]   shape (n_rows, n_cols)
    mode   : 'perimage' | 'percol' | 'global'

    Returns (display_images, norm_info) of the same shape.
    norm_info[r][c] = (lo, hi) raw pixel values that map to [0, 1].
    """
    n_rows = len(images)
    n_cols = len(images[0])
    display   = [[None] * n_cols for _ in range(n_rows)]
    norm_info = [[None] * n_cols for _ in range(n_rows)]

    if mode == 'perimage':
        for r in range(n_rows):
            for c in range(n_cols):
                d, lo, hi = _stretch_one(images[r][c], gamma, pct_lo, pct_hi)
                display[r][c]   = d
                norm_info[r][c] = (lo, hi)

    elif mode == 'percol':
        for c in range(n_cols):
            all_vals = np.concatenate([images[r][c].ravel() for r in range(n_rows)])
            lo = float(np.percentile(all_vals, pct_lo))
            hi = float(np.percentile(all_vals, pct_hi))
            if hi - lo < 1e-8:
                hi = lo + 1e-8
            for r in range(n_rows):
                img_n = np.clip((images[r][c] - lo) / (hi - lo), 0.0, 1.0)
                if gamma != 1.0:
                    img_n = np.power(img_n, gamma)
                display[r][c]   = img_n
                norm_info[r][c] = (lo, hi)

    else:  # global
        all_vals = np.concatenate([images[r][c].ravel() for r in range(n_rows) for c in range(n_cols)])
        lo = float(np.percentile(all_vals, pct_lo))
        hi = float(np.percentile(all_vals, pct_hi))
        if hi - lo < 1e-8:
            hi = lo + 1e-8
        for r in range(n_rows):
            for c in range(n_cols):
                img_n = np.clip((images[r][c] - lo) / (hi - lo), 0.0, 1.0)
                if gamma != 1.0:
                    img_n = np.power(img_n, gamma)
                display[r][c]   = img_n
                norm_info[r][c] = (lo, hi)

    return display, norm_info


# ===========================================================================
# ARGUMENT PARSING
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model", required=True, choices=list(MODELS.keys()),
        help="Which trained model to load.",
    )
    p.add_argument(
        "--flux", type=float, default=None,
        metavar="FRACTION",
        help=(
            "Flux fraction to simulate on the input, e.g. 0.05 = 5 %%.  "
            "Applies Binomial subsampling (each pixel count drawn from "
            "Binomial(count, flux)).  Default: the model's training flux "
            "(1.0 for XY/XZ/YZ, 0.05 for XY_low, 0.01 for XY_ultralow)."
        ),
    )
    p.add_argument(
        "--h5", default=str(_H5_FILE),
        metavar="PATH",
        help="Path to the dataset HDF5 file (default: %(default)s).",
    )
    p.add_argument(
        "--idx", type=int, nargs="+", default=None,
        metavar="N",
        help=(
            "Sample index (or indices) for sample 0 of each split.  "
            "Give 1 value (used for all 3 splits) or 3 values "
            "(train val test).  Further samples beyond n-samples=1 "
            "are chosen randomly (controlled by --seed)."
        ),
    )
    p.add_argument(
        "--n-samples", type=int, default=1,
        metavar="N",
        help="Number of samples per split to show (default: 1).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for index selection and Binomial draws (default: 42).",
    )
    p.add_argument(
        "--norm", choices=["perimage", "percol", "global"], default="perimage",
        help=(
            "Normalisation / stretch mode:  "
            "perimage = each subplot stretched independently (best for mixed flux);  "
            "percol   = input/pred/GT columns each share their own range;  "
            "global   = all subplots share one range.  "
            "(default: perimage)"
        ),
    )
    p.add_argument(
        "--gamma", type=float, default=1.0,
        metavar="G",
        help=(
            "Gamma for power-law stretch applied AFTER percentile clipping.  "
            "Values < 1 brighten dim images (try 0.3–0.5 for ultra-low flux).  "
            "default: 1.0 (linear)."
        ),
    )
    p.add_argument(
        "--sigma", type=float, default=1.0,
        metavar="S",
        help="Gaussian sigma (pixels) for the smoothed channel of XY_ultralow (default: 1.0).",
    )
    p.add_argument(
        "--cmap", default="viridis",
        help="Matplotlib colormap (default: viridis).  Try 'inferno', 'hot', 'gray'.",
    )
    p.add_argument(
        "--pct-lo", type=float, default=1.0,
        metavar="PCT",
        help="Lower percentile for clipping before stretch (default: 1.0).",
    )
    p.add_argument(
        "--pct-hi", type=float, default=99.5,
        metavar="PCT",
        help="Upper percentile for clipping before stretch (default: 99.5).",
    )
    return p.parse_args()


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    args = parse_args()

    cfg   = MODELS[args.model]
    flux  = args.flux if args.flux is not None else cfg["flux_def"]
    h5_path = Path(args.h5)
    rng   = np.random.default_rng(args.seed)

    # ── validate ──────────────────────────────────────────────────────────────
    if not (0.0 < flux <= 1.0):
        print(f"[ERROR] --flux must be in (0, 1].  Got {flux}")
        sys.exit(1)
    if not h5_path.exists():
        print(f"[ERROR] H5 file not found: {h5_path}")
        sys.exit(1)
    if not cfg["path"].exists():
        print(f"[ERROR] Model file not found: {cfg['path']}")
        sys.exit(1)

    # ── load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model : {cfg['path'].name}")
    model = keras.models.load_model(str(cfg["path"]), custom_objects=_CUSTOM_OBJECTS)
    print(f"  Input  shape : {model.input_shape}")
    print(f"  Output shape : {model.output_shape}")

    # Override input_ch with the actual model input (saved model may differ from config)
    actual_input_ch = model.input_shape[-1]
    if actual_input_ch != cfg["input_ch"]:
        print(f"  [INFO] input_ch override: {cfg['input_ch']} → {actual_input_ch} (read from model)")
        cfg = dict(cfg)
        cfg["input_ch"] = actual_input_ch

    # ── collect indices ───────────────────────────────────────────────────────
    with h5py.File(h5_path, 'r') as f:
        split_sizes = {s: int(f[f'{s}/poca'].shape[0]) for s in SPLITS}

    n = args.n_samples
    indices = {}   # split → list of n sample indices

    if args.idx is not None:
        if len(args.idx) == 1:
            seed_idxs = [args.idx[0]] * 3
        elif len(args.idx) == 3:
            seed_idxs = args.idx
        else:
            print("[ERROR] --idx takes 1 or 3 values.")
            sys.exit(1)
        for split, seed_idx in zip(SPLITS, seed_idxs):
            base = seed_idx % split_sizes[split]
            extra = [int(rng.integers(0, split_sizes[split])) for _ in range(n - 1)]
            indices[split] = [base] + extra
    else:
        for split in SPLITS:
            indices[split] = [int(rng.integers(0, split_sizes[split])) for _ in range(n)]

    ch = cfg["channel"]

    # ── load raw data, preprocess, predict ────────────────────────────────────
    # row_data : list of (split_label, sample_idx, disp_input, pred_2d, gt_2d)
    row_data = []
    print(f"\nProcessing samples (flux={flux*100:.1f} %):")

    with h5py.File(h5_path, 'r') as f:
        for split in SPLITS:
            for sample_idx in indices[split]:
                poca_uint8 = f[f'{split}/poca'][sample_idx, :, :, ch]   # (H,W) uint8
                gt_raw     = f[f'{split}/gt'][sample_idx,   :, :, ch]   # (H,W)

                gt_arr = gt_raw.astype(np.float32)
                if gt_arr.max() > 1.5:          # stored as uint8 0–255 → normalise
                    gt_arr /= 255.0

                net_input, disp_input = preprocess(poca_uint8, cfg, flux,
                                                   sigma=args.sigma, rng=rng)
                pred = model.predict(net_input, verbose=0)[0, :, :, 0]  # (H,W)

                row_data.append((split, sample_idx, disp_input, pred, gt_arr))
                print(f"  {split:5s} #{sample_idx:4d} | "
                      f"input max={disp_input.max():.4f}  "
                      f"pred max={pred.max():.4f}  "
                      f"gt max={gt_arr.max():.4f}")

    # ── normalise for display ─────────────────────────────────────────────────
    raw_images = [[r[2], r[3], r[4]] for r in row_data]
    display, norm_info = normalise_grid(
        raw_images, args.norm, args.gamma,
        pct_lo=args.pct_lo, pct_hi=args.pct_hi,
    )

    # ── build figure ──────────────────────────────────────────────────────────
    n_rows     = len(row_data)
    col_titles = ["Input (POCA)", "Prediction", "Ground Truth"]
    col_colors = ["#d0e8ff", "#ffe8cc", "#d0ffd0"]   # tint for col headers

    # row height: a bit taller when fewer rows; cap at 3.4
    row_h = min(3.4, max(2.4, 9.0 / n_rows))
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(11, row_h * n_rows + 1.2),
        squeeze=False,
    )
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes.ravel():
        ax.set_facecolor("#1a1a2e")

    # ── column headers (top row only) ─────────────────────────────────────────
    for c, (title, bg) in enumerate(zip(col_titles, col_colors)):
        axes[0, c].annotate(
            title,
            xy=(0.5, 1.0), xycoords="axes fraction",
            fontsize=11, fontweight="bold",
            ha="center", va="bottom",
            xytext=(0, 22), textcoords="offset points",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=bg, alpha=0.25, edgecolor="none"),
        )

    # ── separator line between split groups ──────────────────────────────────
    # Drawn as a thin horizontal line above the first row of each new split
    split_boundaries = []
    prev_split = None
    for row_i, (split, *_) in enumerate(row_data):
        if split != prev_split and row_i > 0:
            split_boundaries.append(row_i)
        prev_split = split

    # ── draw images ───────────────────────────────────────────────────────────
    for row_i, (split, sample_idx, disp_input, pred, gt) in enumerate(row_data):
        raw_row = [disp_input, pred, gt]

        for col_i in range(3):
            ax = axes[row_i, col_i]
            img_d = display[row_i][col_i]
            lo, hi = norm_info[row_i][col_i]
            raw    = raw_row[col_i]

            im = ax.imshow(img_d, cmap=args.cmap, vmin=0, vmax=1,
                           aspect="equal", interpolation="nearest")
            ax.axis("off")

            # per-image stats subtitle
            stats = (
                f"max={raw.max():.3f}  μ={raw.mean():.4f}\n"
                f"range [{lo:.3f}, {hi:.3f}]"
            )
            ax.set_title(stats, fontsize=6.5, color="#cccccc", pad=3)

            # thin colorbar
            cb = plt.colorbar(im, ax=ax, fraction=0.044, pad=0.03)
            cb.ax.tick_params(labelsize=6, colors="#aaaaaa")
            cb.outline.set_edgecolor("#555555")

        # row label on the left
        axes[row_i, 0].text(
            -0.06, 0.5,
            f"{split}\n#{sample_idx}",
            transform=axes[row_i, 0].transAxes,
            ha="right", va="center", rotation=0,
            fontsize=9, color="white", fontweight="bold",
        )

    # ── draw split separators ─────────────────────────────────────────────────
    # Use a thin line across the figure between groups
    plt.tight_layout(rect=[0.07, 0.0, 1.0, 0.97])

    fig_height = fig.get_figheight()
    for boundary_row in split_boundaries:
        # compute y position in figure coords after layout
        ax_above = axes[boundary_row - 1, 0]
        ax_below = axes[boundary_row,     0]
        y_above  = ax_above.get_position().y0
        y_below  = ax_below.get_position().y1
        y_mid    = (y_above + y_below) / 2.0
        line = plt.Line2D([0.07, 1.0], [y_mid, y_mid],
                          transform=fig.transFigure,
                          color="#555577", linewidth=0.8, linestyle="--")
        fig.add_artist(line)

    # ── figure title ──────────────────────────────────────────────────────────
    flux_str   = f"{flux * 100:.1f} %" if flux < 1.0 else "full"
    norm_label = {"perimage": "per-image norm", "percol": "per-col norm",
                  "global": "global norm"}[args.norm]
    gamma_str  = f"  γ={args.gamma}" if args.gamma != 1.0 else ""

    fig.suptitle(
        f"Model: {cfg['label']}   |   flux: {flux_str}   |   "
        f"{norm_label}{gamma_str}",
        fontsize=12, fontweight="bold", color="white", y=0.995,
    )

    plt.show()


if __name__ == "__main__":
    main()
