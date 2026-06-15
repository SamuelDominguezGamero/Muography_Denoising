#!/usr/bin/env python3
"""
Test a trained UNET denoiser on a single NPY simulation file.

Usage:
  python test_denoiser_model.py  <model.keras>  <simulation.npy>  [options]

  # Basic usage:
  python test_denoiser_model.py models_XY/checkpoints/best_model_0_XY.keras \\
      /path/to/tensor_2D_POCA_....npy

  # With downsample to simulate low flux (e.g. 10% of muons):
  python test_denoiser_model.py model.keras simulation.npy --downsample 0.1

  # Save as PDF:
  python test_denoiser_model.py model.keras simulation.npy --save_pdf

  # Both:
  python test_denoiser_model.py model.keras simulation.npy --downsample 0.05 --save_pdf
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# ---------------------------------------------------------------------------
# Ground-truth resolution (mirrors train script naming convention)
# ---------------------------------------------------------------------------

BASE     = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration")
DATA     = BASE / "data"

GT_FOLDERS = {
    "run0": DATA / "ground_truth_data" / "run0_letters",
    "run1": DATA / "ground_truth_data" / "run1_geometries_variety",
    "run2": DATA / "ground_truth_data" / "run2_blocks",
}


def _find_gt(x_path: Path) -> Path | None:
    """Try to find the matching GT file for a simulation file."""
    stem = x_path.stem.replace("_Muons_100000_2D", "")
    y_stem = stem.replace("tensor_2D_POCA_", "tensor_2D__", 1)
    for gt_folder in GT_FOLDERS.values():
        candidate = gt_folder / f"{y_stem}.npy"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------

def _apply_downsample(poca_raw: np.ndarray, fraction: float) -> np.ndarray:
    """Binomial subsampling of raw integer POCA counts.

    Statistically correct: each count v in a pixel is modelled as v independent
    muon hits, each retained with probability `fraction`.
    Returns float32 array in [0, 1] (per-image max normalization).
    """
    counts = poca_raw.astype(np.int32)
    sparse = np.random.binomial(counts, fraction).astype(np.float32)
    x_max = float(sparse.max())
    return sparse / max(x_max, 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a trained UNET denoiser on a single simulation NPY file."
    )
    parser.add_argument("--model",      type=str, required=True, help="Path to .keras model file")
    parser.add_argument("--input",      type=str, required=True, dest="simulation", help="Path to simulation .npy file (POCA data)")
    parser.add_argument(
        "--downsample", type=float, default=None, metavar="FRAC",
        help="Binomial-subsample the input to simulate low muon flux (e.g. 0.1 = 10%%). "
             "Must be in (0, 1]. Default: no downsampling."
    )
    parser.add_argument(
        "--save_pdf", action="store_true",
        help="Save the figure as PDF in /home/samuel/Work/Muography_Denoising/"
    )
    args = parser.parse_args()

    # ── Validate inputs ──────────────────────────────────────────────────────
    model_path = Path(args.model)
    sim_path   = Path(args.simulation)

    if not model_path.exists():
        sys.exit(f"[ERROR] Model not found: {model_path}")
    if not sim_path.exists():
        sys.exit(f"[ERROR] Simulation file not found: {sim_path}")
    if args.downsample is not None and not (0 < args.downsample <= 1):
        sys.exit("[ERROR] --downsample must be in (0, 1]")

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"Loading model : {model_path.name}")
    import tensorflow as tf
    from tensorflow import keras

    # Define custom objects used during training
    def charbonnier_loss(y_true, y_pred, eps=1e-3):
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + eps * eps))

    def psnr_metric(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.image.psnr(y_true, y_pred, max_val=1.0)

    def ssim_metric(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=5, filter_sigma=1.0)

    model = keras.models.load_model(
        str(model_path),
        custom_objects={
            "charbonnier_loss": charbonnier_loss,
            "psnr_metric":      psnr_metric,
            "ssim_metric":      ssim_metric,
        },
    )
    print(f"  Parameters   : {model.count_params():,}")

    # ── Load simulation file ─────────────────────────────────────────────────
    print(f"Loading input  : {sim_path.name}")
    raw = np.load(sim_path, allow_pickle=False).astype(np.float32)
    poca_channel = raw[:, :, 0]  # XY channel

    # ── Prepare input X ──────────────────────────────────────────────────────
    if args.downsample is not None:
        frac = args.downsample
        print(f"Downsampling   : Binomial(counts, {frac:.3f})  →  {frac*100:.1f}% flux")
        x_norm = _apply_downsample(poca_channel, frac)
        input_label = f"Input POCA  ({frac*100:.0f}% flux)"
    else:
        x_max = float(poca_channel.max())
        x_norm = poca_channel / max(x_max, 1.0)
        input_label = "Input POCA  (full flux)"

    # ── Run model ────────────────────────────────────────────────────────────
    x_tensor = x_norm[np.newaxis, :, :, np.newaxis]  # (1, 128, 128, 1)
    pred = model.predict(x_tensor, verbose=0)[0, :, :, 0]  # (128, 128)

    # ── Load ground truth (optional) ─────────────────────────────────────────
    gt_path = _find_gt(sim_path)
    if gt_path is not None:
        print(f"Ground truth   : {gt_path.name}")
        gt = np.load(gt_path, allow_pickle=False).astype(np.float32)[:, :, 0]
    else:
        print("[WARN]  No matching ground truth found — GT panel will be blank")
        gt = None

    # ── Compute metrics ──────────────────────────────────────────────────────
    if gt is not None:
        x_t  = tf.constant(x_norm[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
        p_t  = tf.constant(pred[np.newaxis, :, :, np.newaxis],   dtype=tf.float32)
        gt_t = tf.constant(gt[np.newaxis, :, :, np.newaxis],     dtype=tf.float32)

        psnr_in  = float(tf.image.psnr(gt_t, x_t,  max_val=1.0).numpy()[0])
        psnr_out = float(tf.image.psnr(gt_t, p_t,  max_val=1.0).numpy()[0])
        ssim_in  = float(tf.image.ssim(gt_t, x_t,  max_val=1.0, filter_size=5, filter_sigma=1.0).numpy()[0])
        ssim_out = float(tf.image.ssim(gt_t, p_t,  max_val=1.0, filter_size=5, filter_sigma=1.0).numpy()[0])

        print(f"\n  PSNR  input → GT : {psnr_in:.2f} dB")
        print(f"  PSNR  pred  → GT : {psnr_out:.2f} dB  (Δ = {psnr_out - psnr_in:+.2f} dB)")
        print(f"  SSIM  input → GT : {ssim_in:.4f}")
        print(f"  SSIM  pred  → GT : {ssim_out:.4f}  (Δ = {ssim_out - ssim_in:+.4f})")

    # ── Plot ─────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg' if args.save_pdf else 'TkAgg')
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("[ERROR] matplotlib required: pip install matplotlib")

    n_panels = 3 if gt is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    fig.patch.set_facecolor('white')

    panels = [(x_norm, input_label), (pred, "UNET Output  (denoised)")]
    if gt is not None:
        panels.append((gt, "Ground Truth"))

    # Common vmax across all panels for honest comparison
    vmax = max(float(x_norm.max()), float(pred.max()), float(gt.max()) if gt is not None else 0, 1e-6)

    for ax, (img, title) in zip(axes, panels):
        im = ax.imshow(img, cmap='viridis', vmin=0, vmax=vmax,
                       aspect='equal', interpolation='nearest')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Normalised density', fontsize=8)

    # Title with metrics
    title_parts = [f"Model: {model_path.stem}", f"File: {sim_path.stem[:60]}"]
    if gt is not None:
        title_parts.append(
            f"PSNR: {psnr_in:.1f}→{psnr_out:.1f} dB  |  SSIM: {ssim_in:.3f}→{ssim_out:.3f}"
        )
    if args.downsample is not None:
        title_parts.append(f"Downsample: {args.downsample*100:.0f}% flux (Binomial)")

    fig.suptitle("\n".join(title_parts), fontsize=10, y=1.01)
    plt.tight_layout()

    # ── Save / show ──────────────────────────────────────────────────────────
    if args.save_pdf:
        pdf_dir = Path("/home/samuel/Work/Muography_Denoising")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_ds{args.downsample}" if args.downsample is not None else ""
        pdf_name = f"denoiser_test_{model_path.stem}_{sim_path.stem[:40]}{suffix}.pdf"
        pdf_path = pdf_dir / pdf_name
        plt.savefig(pdf_path, dpi=180, bbox_inches='tight', facecolor='white', format='pdf')
        print(f"\n[SAVED]  {pdf_path}")

    # Always try to show interactively too (will silently skip if no display)
    try:
        matplotlib.use('TkAgg')
        plt.show()
    except Exception:
        if not args.save_pdf:
            # Fallback: save PNG next to the simulation file
            fallback = sim_path.parent / f"denoiser_test_{sim_path.stem}.png"
            plt.savefig(fallback, dpi=150, bbox_inches='tight')
            print(f"[INFO]  No display available — saved PNG: {fallback}")

    plt.close(fig)


if __name__ == "__main__":
    main()
