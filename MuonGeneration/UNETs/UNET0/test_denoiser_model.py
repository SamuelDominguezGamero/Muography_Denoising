#!/usr/bin/env python3
"""
Test a trained UNET denoiser — single file or full test split.

── Single-file mode ───────────────────────────────────────────────────────────────────
  python test_denoiser_model.py --model <model.keras> --input <sim.npy>
  python test_denoiser_model.py --model m.keras --input s.npy --downsample 0.1
  python test_denoiser_model.py --model m.keras --input s.npy --save_pdf

── Full-test mode ────────────────────────────────────────────────────────────────────
  Reads run_config.json (flux, channel, loss) and data_indices/test.csv
  automatically from the experiment folder, then saves an N×3 PDF grid
  (Input | Prediction | Ground Truth) with per-row PSNR/SSIM.

  python test_denoiser_model.py \\
      --model_folder /path/to/models_XY/run0__loss_charb__flux100__bs32__f64__l4__seed42 \\
      --full_test 15

  # Override flux (test a full-flux model on low-flux inputs):
  python test_denoiser_model.py --model_folder /path/... --full_test 15 --downsample 0.1

  Saves to: /home/samuel/Work/Muography_Denoising/full_test_<run_id>_Nsamples.pdf
"""

import argparse
import json
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
    """Try to find the matching GT file for a simulation file.

    Handles two naming conventions:
      run0  X: tensor_2D_POCA_<payload>[_Muons_100000_2D].npy
             Y: tensor_2D__<payload>.npy          ← double underscore
      run1/2 X: tensor_2D_POCA_POCA_merged_<payload>_Muons_100000_2D.npy
             Y: tensor_2D_<payload>.npy            ← single underscore
    """
    stem = x_path.stem.replace("_Muons_100000_2D", "")
    if stem.startswith("tensor_2D_POCA_POCA_merged_"):
        y_stem = "tensor_2D_" + stem[len("tensor_2D_POCA_POCA_merged_"):]
    else:
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
# Full-test mode
# ---------------------------------------------------------------------------

def _run_full_test(model_folder: Path, n_samples: int, downsample_override):
    """Evaluate the best model from an experiment folder on N random test samples.

    Reads run_config.json for flux_fraction and channel index, loads the best
    model checkpoint, samples N pairs from data_indices/test.csv, computes
    PSNR/SSIM for each, and saves an N×3 PDF grid.
    """
    import csv as _csv
    import tensorflow as tf
    from tensorflow import keras

    # ── Load config ───────────────────────────────────────────────────────────
    config_path = model_folder / "run_config.json"
    if not config_path.exists():
        sys.exit(f"[ERROR] run_config.json not found in {model_folder}")
    with open(config_path) as f:
        cfg = json.load(f)

    flux_fraction = cfg.get("flux_fraction", 0.0)
    channel_idx   = cfg.get("channel_index", 0)
    projection    = cfg.get("projection", "XY")
    loss_fn_name  = cfg.get("loss_fn", "charbonnier")
    run_id        = model_folder.name

    if downsample_override is not None:
        print(f"[INFO]  --downsample overrides config flux_fraction "
              f"{flux_fraction} → {downsample_override}")
        flux_fraction = downsample_override

    print(f"[CFG]   run_id     : {run_id}")
    print(f"[CFG]   projection : {projection}  (channel {channel_idx})")
    print(f"[CFG]   loss       : {loss_fn_name}")
    print(f"[CFG]   flux       : {'disabled (full flux)' if flux_fraction == 0 else f'{flux_fraction*100:.0f}%'}")

    # ── Find best model ───────────────────────────────────────────────────────
    ckpt_dir = model_folder / "checkpoints"
    candidates = sorted(ckpt_dir.glob("best_model_*.keras"))
    if not candidates:
        sys.exit(f"[ERROR] No best_model_*.keras found in {ckpt_dir}")
    model_path = candidates[0]
    print(f"[MODEL] {model_path.name}")

    # ── Load test CSV ─────────────────────────────────────────────────────────
    test_csv = model_folder / "data_indices" / "test.csv"
    if not test_csv.exists():
        sys.exit(f"[ERROR] test.csv not found: {test_csv}")

    all_pairs = []
    with open(test_csv, newline='') as f:
        for row in _csv.DictReader(f):
            xp, yp = Path(row["x_path"]), Path(row["y_path"])
            if xp.exists() and yp.exists():
                all_pairs.append((xp, yp))

    if not all_pairs:
        sys.exit("[ERROR] No valid pairs found in test.csv (files may have moved).")

    n = min(n_samples, len(all_pairs))
    print(f"[DATA]  Test split : {len(all_pairs)} pairs  →  sampling {n}")

    rng = np.random.default_rng()
    selected = [all_pairs[i] for i in sorted(rng.choice(len(all_pairs), size=n, replace=False))]

    # ── Load model ────────────────────────────────────────────────────────────
    def charbonnier_loss(y_true, y_pred, eps=1e-3):
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + eps * eps))
    def mae_loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(tf.abs(y_true - y_pred))
    def mse_loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(tf.square(y_true - y_pred))
    def psnr_metric(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.image.psnr(y_true, y_pred, max_val=1.0)
    def ssim_metric(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=5, filter_sigma=1.0)

    print("Loading model ...")
    model = keras.models.load_model(
        str(model_path),
        custom_objects={
            "charbonnier_loss": charbonnier_loss,
            "mae_loss":         mae_loss,
            "mse_loss":         mse_loss,
            "psnr_metric":      psnr_metric,
            "ssim_metric":      ssim_metric,
        },
    )
    print(f"  Parameters : {model.count_params():,}\n")

    # ── Inference ─────────────────────────────────────────────────────────────
    inputs_list, preds_list, gts_list, metrics_rows = [], [], [], []
    print(f"Running inference on {n} test samples ...")
    for i, (x_path, y_path) in enumerate(selected):
        x_data = np.load(x_path, allow_pickle=False).astype(np.float32)
        y_data = np.load(y_path, allow_pickle=False).astype(np.float32)

        if flux_fraction > 0:
            x_norm = _apply_downsample(x_data[:, :, channel_idx], flux_fraction)
        else:
            x_ch   = x_data[:, :, channel_idx]
            x_norm = x_ch / max(float(x_ch.max()), 1.0)
        y_ch = y_data[:, :, channel_idx]

        pred = model.predict(x_norm[np.newaxis, :, :, np.newaxis], verbose=0)[0, :, :, 0]

        gt_t = tf.constant(y_ch  [np.newaxis, :, :, np.newaxis], dtype=tf.float32)
        x_t  = tf.constant(x_norm[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
        p_t  = tf.constant(pred  [np.newaxis, :, :, np.newaxis], dtype=tf.float32)
        psnr_in  = float(tf.image.psnr(gt_t, x_t, max_val=1.0).numpy()[0])
        psnr_out = float(tf.image.psnr(gt_t, p_t, max_val=1.0).numpy()[0])
        ssim_in  = float(tf.image.ssim(gt_t, x_t, max_val=1.0, filter_size=5, filter_sigma=1.0).numpy()[0])
        ssim_out = float(tf.image.ssim(gt_t, p_t, max_val=1.0, filter_size=5, filter_sigma=1.0).numpy()[0])

        inputs_list.append(x_norm)
        preds_list.append(pred)
        gts_list.append(y_ch)
        metrics_rows.append((psnr_in, psnr_out, ssim_in, ssim_out))
        print(f"  [{i+1:3d}/{n}]  PSNR {psnr_in:6.2f}→{psnr_out:6.2f} dB  |  "
              f"SSIM {ssim_in:.4f}→{ssim_out:.4f}  |  {x_path.stem[:55]}")

    # ── Summary ───────────────────────────────────────────────────────────────
    avg_psnr_in  = float(np.mean([m[0] for m in metrics_rows]))
    avg_psnr_out = float(np.mean([m[1] for m in metrics_rows]))
    avg_ssim_in  = float(np.mean([m[2] for m in metrics_rows]))
    avg_ssim_out = float(np.mean([m[3] for m in metrics_rows]))
    print(f"\n  Average  PSNR : {avg_psnr_in:.2f} → {avg_psnr_out:.2f} dB  (Δ = {avg_psnr_out - avg_psnr_in:+.2f})")
    print(f"  Average  SSIM : {avg_ssim_in:.4f} → {avg_ssim_out:.4f}  (Δ = {avg_ssim_out - avg_ssim_in:+.4f})")

    # ── Figure ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    col_titles = [f"Input  (POCA · {projection})", "UNET Output  (denoised)", "Ground Truth"]
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.patch.set_facecolor('white')
    flux_str = f"  ·  flux {flux_fraction*100:.0f}%" if flux_fraction > 0 else "  ·  full flux"
    fig.suptitle(
        f"Full test  ·  {run_id}{flux_str}\n"
        f"Avg PSNR: {avg_psnr_in:.2f} → {avg_psnr_out:.2f} dB   |   "
        f"Avg SSIM: {avg_ssim_in:.4f} → {avg_ssim_out:.4f}   ({n} samples)",
        fontsize=11, fontweight='bold', y=1.002,
    )

    for row_idx, (src, pred, gt, (pi, po, si, so)) in enumerate(
        zip(inputs_list, preds_list, gts_list, metrics_rows)
    ):
        vmax = max(float(src.max()), float(pred.max()), float(gt.max()), 1e-6)
        row_label = f"PSNR {pi:.1f}→{po:.1f}  SSIM {si:.3f}→{so:.3f}"
        for col_idx, (img, col_title) in enumerate(zip([src, pred, gt], col_titles)):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(img, cmap='viridis', vmin=0, vmax=vmax,
                           aspect='equal', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_title, fontsize=11, fontweight='bold', pad=6)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=8, labelpad=6)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    pdf_dir  = Path("/home/samuel/Work/Muography_Denoising")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    flux_tag = f"_flux{int(flux_fraction*100)}" if flux_fraction > 0 else ""
    pdf_name = f"full_test_{run_id[:60]}{flux_tag}_{n}samples.pdf"
    pdf_path = pdf_dir / pdf_name
    plt.savefig(pdf_path, dpi=150, bbox_inches='tight', facecolor='white', format='pdf')
    print(f"\n[SAVED]  {pdf_path}")

    try:
        matplotlib.use('TkAgg')
        plt.show()
    except Exception:
        pass
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a trained UNET denoiser — single file or full test split."
    )
    # ── Single-file mode ──────────────────────────────────────────────────────
    parser.add_argument("--model",  type=str, default=None,
                        help="(single-file) Path to .keras model file")
    parser.add_argument("--input",  type=str, default=None, dest="simulation",
                        help="(single-file) Path to simulation .npy file (POCA data)")
    # ── Full-test mode ────────────────────────────────────────────────────────
    parser.add_argument("--model_folder", type=str, default=None,
                        help="(full-test) Experiment folder with run_config.json + checkpoints/")
    parser.add_argument("--full_test", type=int, default=None, metavar="N",
                        help="(full-test) Evaluate N random test samples from data_indices/test.csv "
                             "and save an N×3 PDF grid (Input | Prediction | Ground Truth)")
    # ── Shared ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--downsample", type=float, default=None, metavar="FRAC",
        help="Binomial-subsample input to simulate low muon flux (e.g. 0.1 = 10%%). "
             "In full-test mode, overrides the flux_fraction from run_config.json."
    )
    parser.add_argument("--save_pdf", action="store_true",
                        help="(single-file) Save the output figure as PDF.")
    args = parser.parse_args()

    # ── Dispatch: full-test mode ──────────────────────────────────────────────
    if args.model_folder is not None:
        if args.full_test is None:
            sys.exit("[ERROR] --model_folder requires --full_test N")
        _run_full_test(Path(args.model_folder), args.full_test, args.downsample)
        return

    # ── Single-file mode: validate ────────────────────────────────────────────
    if args.model is None or args.simulation is None:
        parser.error("Single-file mode requires both --model and --input")

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

    def mae_loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    def mse_loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(tf.square(y_true - y_pred))

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
            "mae_loss":         mae_loss,
            "mse_loss":         mse_loss,
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
