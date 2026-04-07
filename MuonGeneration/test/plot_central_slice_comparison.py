"""
Create a PNG comparison between ground truth and merged POCA on the full XY plane
at the central z slice (z=0 by default).

Designed for headless execution on clusters (no GUI required).
"""

import argparse
from pathlib import Path

import matplotlib
# Use a non-interactive backend so the script works in cluster environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_npy(path: Path):
    """Load .npy handling both ndarray and pickled-dict content."""
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        return obj.item()
    return obj


def _compute_poca_map(poca_obj, metric: str) -> np.ndarray:
    """Return a 3D POCA map with shape (npy, npx, npz)."""
    if isinstance(poca_obj, dict):
        required = {"n_events", "sum_theta", "sum_theta_sq"}
        missing = required - set(poca_obj.keys())
        if missing:
            raise ValueError(f"POCA dict missing keys: {sorted(missing)}")

        n_events = np.asarray(poca_obj["n_events"], dtype=float)
        sum_theta = np.asarray(poca_obj["sum_theta"], dtype=float)
        sum_theta_sq = np.asarray(poca_obj["sum_theta_sq"], dtype=float)

        if n_events.ndim != 3:
            raise ValueError(f"POCA arrays must be 3D. Got shape {n_events.shape}")

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_theta = np.divide(
                sum_theta,
                n_events,
                out=np.zeros_like(sum_theta),
                where=n_events > 0,
            )
            variance = np.divide(
                sum_theta_sq,
                n_events,
                out=np.zeros_like(sum_theta_sq),
                where=n_events > 0,
            ) - mean_theta**2
            variance = np.maximum(variance, 0.0)
            std_theta = np.sqrt(variance)

        if metric == "std":
            return std_theta
        if metric == "mean":
            return mean_theta
        if metric == "counts":
            return n_events
        raise ValueError(f"Unknown POCA metric: {metric}")

    poca_arr = np.asarray(poca_obj)
    if poca_arr.ndim != 3:
        raise ValueError(
            "POCA input must be either a dict with n_events/sum_theta/sum_theta_sq "
            "or a direct 3D array."
        )
    return poca_arr


def _z_to_index(z_value: float, lz: float, nz: int) -> int:
    """Convert physical z value (cm) to nearest voxel index using voxel centers."""
    dz = lz / nz
    z_min = -lz / 2.0
    idx_float = (z_value - z_min) / dz - 0.5
    idx = int(np.floor(idx_float + 0.5))
    return int(np.clip(idx, 0, nz - 1))


def main():
    parser = argparse.ArgumentParser(
        description="Save a PNG comparing ground truth vs merged POCA on the full XY plane."
    )
    parser.add_argument(
        "--poca_merged",
        required=True,
        help="Path to merged POCA .npy (dict from merge_results or direct 3D array).",
    )
    parser.add_argument(
        "--ground_truth_tensor",
        required=True,
        help="Path to ground-truth density tensor .npy (3D array).",
    )
    parser.add_argument(
        "--poca_metric",
        default="std",
        choices=["std", "mean", "counts"],
        help="POCA map to visualize if input is dict.",
    )
    parser.add_argument("--Lx", type=float, default=128.0, help="Physical size in X [cm].")
    parser.add_argument("--Ly", type=float, default=128.0, help="Physical size in Y [cm].")
    parser.add_argument("--Lz", type=float, default=128.0, help="Physical size in Z [cm].")
    parser.add_argument(
        "--z",
        type=float,
        default=0.0,
        help="Physical z plane [cm] to display (default: 0).",
    )
    parser.add_argument(
        "--output_dir",
        default="png_comparisons",
        help="Directory where the PNG comparison will be saved.",
    )
    parser.add_argument(
        "--output_name",
        default=None,
        help="Optional PNG file name. If omitted, a name is generated automatically.",
    )
    args = parser.parse_args()

    gt_path = Path(args.ground_truth_tensor)
    poca_path = Path(args.poca_merged)

    gt = np.asarray(_load_npy(gt_path), dtype=float)
    if gt.ndim != 3:
        raise ValueError(f"Ground truth must be 3D. Got shape {gt.shape}")

    poca_map = _compute_poca_map(_load_npy(poca_path), args.poca_metric)

    if gt.shape != poca_map.shape:
        raise ValueError(
            f"Shape mismatch ground truth vs POCA: {gt.shape} != {poca_map.shape}. "
            "Use arrays with matching (npy, npx, npz)."
        )

    ny, nx, nz = gt.shape
    iz = _z_to_index(args.z, args.Lz, nz)
    z_center = -args.Lz / 2.0 + (iz + 0.5) * (args.Lz / nz)

    gt_slice = gt[:, :, iz]
    poca_slice = poca_map[:, :, iz]

    extent = [-args.Lx / 2.0, args.Lx / 2.0, -args.Ly / 2.0, args.Ly / 2.0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = axes[0].imshow(gt_slice, origin="lower", extent=extent, cmap="viridis")
    axes[0].set_title("Ground Truth (density)")
    axes[0].set_xlabel("X [cm]")
    axes[0].set_ylabel("Y [cm]")
    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar0.set_label("Density")

    im1 = axes[1].imshow(poca_slice, origin="lower", extent=extent, cmap="magma")
    axes[1].set_title(f"POCA ({args.poca_metric})")
    axes[1].set_xlabel("X [cm]")
    axes[1].set_ylabel("Y [cm]")
    cbar1 = fig.colorbar(im1, ax=axes[1])
    if args.poca_metric == "std":
        cbar1.set_label("Std(theta) [rad]")
    elif args.poca_metric == "mean":
        cbar1.set_label("Mean(theta) [rad]")
    else:
        cbar1.set_label("Counts")

    fig.suptitle(
        f"Ground Truth vs POCA ({args.poca_metric}) on XY at z={z_center:.2f} cm"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_name:
        output_name = args.output_name
    else:
        output_name = f"comparison_{poca_path.stem}_vs_{gt_path.stem}_z{args.z:.2f}.png"
        output_name = output_name.replace(" ", "_")

    out_path = output_dir / output_name
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[INFO] Comparison PNG saved to: {out_path}")


if __name__ == "__main__":
    main()
