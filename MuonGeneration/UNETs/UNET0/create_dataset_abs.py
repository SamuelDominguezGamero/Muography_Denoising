#!/usr/bin/env python3
"""
create_dataset_abs.py — Re-extract POCA projections as ABSOLUTE muon counts
and build dataset0_abs.h5.

Problem with dataset0.h5
------------------------
Each POCA projection was normalised PER IMAGE before saving:

    tensor_uint8 = np.uint8(histogram / histogram.max() * 255)

This destroys the absolute flux information: a scene with 5000 muons and one
with 50 muons produce identical-looking uint8 images — the relative
signal-to-noise difference is gone.

What this script does
---------------------
1. Re-reads every .root file from the simulation TAR WITHOUT normalising.
2. Saves float32 histograms (raw muon counts) to an intermediate directory
   so the extraction can be interrupted and resumed.
3. Pairs with the existing binary GT (.npy) files.
4. Shuffles (fixed seed) and splits train/val/test.
5. Writes dataset0_abs.h5 with:
     poca : float32 (N, 128, 128, 3) — absolute counts, NO per-image norm
     gt   : float32 (N, 128, 128, 3) — binary mask {0.0, 1.0}
6. Stores global train-split statistics (max, percentiles) in H5 root attrs
   to facilitate on-the-fly normalisation during training.

Usage
-----
  python create_dataset_abs.py                              # use defaults
  python create_dataset_abs.py --source /path/to/run0.tar
  python create_dataset_abs.py --out /path/to/dataset0_abs.h5
  python create_dataset_abs.py --skip-extract               # if step 1 already done
"""

import argparse
import random
import shutil
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    raise SystemExit("[ERROR] h5py required.  pip install h5py")

try:
    import ROOT
    ROOT.gErrorIgnoreLevel = ROOT.kError
except ImportError:
    raise SystemExit(
        "[ERROR] ROOT (PyROOT) is required to read .root files.\n"
        "        Source your ROOT environment:  source /path/to/root/bin/thisroot.sh"
    )


# ===========================================================================
# DEFAULT PATHS  (override via CLI)
# ===========================================================================

_BASE        = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration")
_DATA        = _BASE / "data"

SOURCE_TAR   = _DATA / "simulation_data" / "run0.tar"
ABS_NPY_DIR  = _DATA / "simulation_data" / "run0_POCA_projections_abs"   # float32 .npy
GT_2D_DIR    = _DATA / "ground_truth_data" / "2Dimensions"
OUT_H5       = _DATA / "datasets" / "dataset0_abs.h5"

WORLD_SIZE   = 128.0     # cm — histogram edges: [-64, +64]
N_BINS       = 128
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
SEED         = 0         # fixed → reproducible split
BUFFER_SIZE  = 64        # samples to buffer in RAM before writing to H5


# ===========================================================================
# STEP 1 — EXTRACT ABSOLUTE POCA PROJECTIONS FROM .root FILES
# ===========================================================================

def _histogram_float32(h, transpose=True):
    """Return float32 histogram (transposed so row=y, col=x)."""
    arr = h.astype(np.float32)
    return arr.T if transpose else arr


def _poca_from_root(root_path):
    """
    Build (128, 128, 3) float32 tensor of absolute muon-count projections
    from a single .root file. Returns None on error or empty file.
    """
    try:
        df  = ROOT.RDataFrame("events", str(root_path))
        df  = df.Filter("abs(theta) > 1e-8")
        res = df.AsNumpy(columns=["poca_x", "poca_y", "poca_z"])
        x, y, z = res["poca_x"], res["poca_y"], res["poca_z"]
    except Exception as e:
        print(f"\n  [WARN] RDataFrame failed on {root_path.name}: {e}")
        return None

    if len(x) == 0:
        return None

    lo    = -WORLD_SIZE / 2.0
    hi    = +WORLD_SIZE / 2.0
    edges = np.linspace(lo, hi, N_BINS + 1)

    xy, _, _ = np.histogram2d(x, y, bins=edges)
    xz, _, _ = np.histogram2d(x, z, bins=edges)
    yz, _, _ = np.histogram2d(y, z, bins=edges)

    # Stack: shape (128, 128, 3), float32 absolute counts — NO normalisation
    return np.stack(
        [_histogram_float32(xy), _histogram_float32(xz), _histogram_float32(yz)],
        axis=2
    )


def extract_poca_absolute(source_path, out_npy_dir, skip_existing=True):
    """
    Iterate over all .root files in source_path (TAR file or directory).
    For each file, compute float32 absolute-count projections and save as .npy.

    Skips files whose .npy already exists (allows resuming after interruption).

    Returns: list of (metadata_str, npy_path)
    """
    source_path  = Path(source_path)
    out_npy_dir  = Path(out_npy_dir)
    out_npy_dir.mkdir(parents=True, exist_ok=True)

    results   = []
    skipped   = 0
    errors    = 0

    if source_path.is_file() and source_path.suffix == ".tar":
        print(f"  Source TAR : {source_path}  ({source_path.stat().st_size/1e9:.1f} GB)")
        tmp_dir = Path(tempfile.mkdtemp(prefix="poca_abs_", dir=source_path.parent))
        try:
            with tarfile.open(source_path, "r") as tar:
                members = [m for m in tar.getmembers() if m.name.endswith(".root")]
                total   = len(members)
                print(f"  Found {total} .root files in TAR")

                for i, member in enumerate(members, 1):
                    metadata = Path(member.name).stem.replace("POCA_merged__", "")
                    npy_path = out_npy_dir / f"poca_abs_{metadata}.npy"

                    if skip_existing and npy_path.exists():
                        results.append((metadata, npy_path))
                        skipped += 1
                        continue

                    tar.extract(member, path=tmp_dir)
                    root_file = tmp_dir / member.name
                    try:
                        tensor = _poca_from_root(root_file)
                        if tensor is not None:
                            np.save(npy_path, tensor)
                            results.append((metadata, npy_path))
                        else:
                            errors += 1
                    finally:
                        # Delete extracted .root immediately to save disk space
                        if root_file.exists():
                            root_file.unlink()
                        for parent in root_file.parents:
                            if parent == tmp_dir:
                                break
                            try:
                                parent.rmdir()
                            except OSError:
                                break

                    if i % max(1, total // 20) == 0 or i == total:
                        pct = 100 * i / total
                        print(f"  [{i:5d}/{total}]  {pct:5.1f} %  "
                              f"saved={len(results)-skipped}  "
                              f"skipped={skipped}  errors={errors}", end="\r")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        print()

    elif source_path.is_dir():
        root_files = sorted(source_path.glob("*.root"))
        total      = len(root_files)
        print(f"  Found {total} .root files in {source_path}")

        for i, rf in enumerate(root_files, 1):
            metadata = rf.stem.replace("POCA_merged__", "")
            npy_path = out_npy_dir / f"poca_abs_{metadata}.npy"

            if skip_existing and npy_path.exists():
                results.append((metadata, npy_path))
                skipped += 1
                continue

            tensor = _poca_from_root(rf)
            if tensor is not None:
                np.save(npy_path, tensor)
                results.append((metadata, npy_path))
            else:
                errors += 1

            if i % max(1, total // 20) == 0 or i == total:
                print(f"  [{i:5d}/{total}]  saved={len(results)-skipped}  "
                      f"skipped={skipped}  errors={errors}", end="\r")
        print()

    else:
        raise ValueError(f"--source must be a .tar file or a directory: {source_path}")

    print(f"  Total saved : {len(results)}  "
          f"(skipped {skipped} existing, {errors} errors)")
    return results


# ===========================================================================
# STEP 2 — GT LOOKUP FROM EXISTING .npy FILES
# ===========================================================================

def load_gt_lookup(gt_2d_dir):
    """
    Return dict: metadata_str → Path for all tensorGT_2D_*.npy files.
    GT is a binary max-projection mask (uint8 {0,1}) — loaded as float32.
    """
    gt_dir = Path(gt_2d_dir)
    lookup = {}
    for f in gt_dir.glob("tensorGT_2D_*.npy"):
        # Strip prefix and leading underscores from core ID
        core = f.stem.replace("tensorGT_2D_", "").lstrip("_")
        lookup[core] = f
    print(f"  Found {len(lookup)} GT files")
    return lookup


# ===========================================================================
# STEP 3 — MATCH & SPLIT
# ===========================================================================

def match_and_split(poca_items, gt_lookup):
    """
    poca_items : list of (metadata_str, poca_npy_path)
    Returns    : dict  split_name → list of (poca_path, gt_path)
    """
    pairs     = []
    unmatched = 0

    for metadata, poca_path in poca_items:
        if metadata in gt_lookup:
            pairs.append((poca_path, gt_lookup[metadata]))
        else:
            unmatched += 1
            # Uncomment to debug mismatches:
            # print(f"  [no GT] {metadata}")

    print(f"  Matched: {len(pairs)}   |   no GT found: {unmatched}")
    if not pairs:
        raise RuntimeError(
            "No matching pairs found.  "
            "Check that POCA metadata keys match GT filenames."
        )

    rng = random.Random(SEED)
    rng.shuffle(pairs)

    n       = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train : n_train + n_val],
        "test":  pairs[n_train + n_val :],
    }
    for name, sp in splits.items():
        print(f"  {name:5s}: {len(sp):5d}  ({100*len(sp)/n:.1f} %)")

    return splits


# ===========================================================================
# STEP 4 — WRITE H5
# ===========================================================================

def _compute_train_stats(train_pairs, sample_every=10):
    """
    Compute global statistics over the train split (every N-th file to save time).
    Returns dict with max, p95, p99, p99_5.
    """
    print(f"  Sampling every {sample_every}-th train file for statistics …")
    sample_paths = [p for i, (p, _) in enumerate(train_pairs) if i % sample_every == 0]
    chunks = []
    for p in sample_paths:
        arr = np.load(p, allow_pickle=False)
        # Only non-zero pixels for meaningful percentiles
        vals = arr[arr > 0]
        if vals.size > 0:
            chunks.append(vals)

    if not chunks:
        return {"max": 255.0, "p95": 200.0, "p99": 240.0, "p99_5": 248.0}

    all_vals = np.concatenate(chunks)
    return {
        "max":   float(all_vals.max()),
        "p95":   float(np.percentile(all_vals, 95)),
        "p99":   float(np.percentile(all_vals, 99)),
        "p99_5": float(np.percentile(all_vals, 99.5)),
    }


def write_h5(splits, out_path):
    """
    Write dataset0_abs.h5.
      poca: float32 (N, 128, 128, 3)  — absolute muon counts
      gt  : float32 (N, 128, 128, 3)  — binary mask {0.0, 1.0}

    Root attrs include global train statistics for on-the-fly normalisation.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── global statistics (train split only) ─────────────────────────────────
    print("\n  Computing global train statistics …")
    stats = _compute_train_stats(splits["train"])
    print(f"  Train POCA (non-zero pixels):  "
          f"max={stats['max']:.1f}  "
          f"p95={stats['p95']:.2f}  "
          f"p99={stats['p99']:.2f}  "
          f"p99.5={stats['p99_5']:.2f}")

    # ── write ─────────────────────────────────────────────────────────────────
    print(f"\n  Writing → {out_path}")

    with h5py.File(out_path, "w") as f:
        # root-level metadata
        f.attrs["created"]           = datetime.now().isoformat()
        f.attrs["source_script"]     = "create_dataset_abs.py"
        f.attrs["poca_dtype"]        = "float32  absolute muon counts per pixel — NO per-image normalisation"
        f.attrs["gt_dtype"]          = "float32  binary mask {0.0, 1.0}"
        f.attrs["world_size_cm"]     = WORLD_SIZE
        f.attrs["n_bins"]            = N_BINS
        f.attrs["train_ratio"]       = TRAIN_RATIO
        f.attrs["val_ratio"]         = VAL_RATIO
        f.attrs["split_seed"]        = SEED
        f.attrs["poca_train_max"]    = stats["max"]
        f.attrs["poca_train_p95"]    = stats["p95"]
        f.attrs["poca_train_p99"]    = stats["p99"]
        f.attrs["poca_train_p99_5"]  = stats["p99_5"]

        for split_name, pairs in splits.items():
            n = len(pairs)
            if n == 0:
                continue

            grp     = f.create_group(split_name)
            ds_poca = grp.create_dataset(
                "poca", shape=(n, N_BINS, N_BINS, 3),
                dtype=np.float32, chunks=(1, N_BINS, N_BINS, 3),
            )
            ds_gt   = grp.create_dataset(
                "gt",   shape=(n, N_BINS, N_BINS, 3),
                dtype=np.float32, chunks=(1, N_BINS, N_BINS, 3),
            )

            poca_buf, gt_buf, buf_start = [], [], 0

            for idx, (poca_path, gt_path) in enumerate(pairs):
                poca_arr = np.load(poca_path, allow_pickle=False).astype(np.float32)
                gt_arr   = np.load(gt_path,   allow_pickle=False).astype(np.float32)
                # GT stored as uint8 {0,1} → float32 {0.0,1.0} after astype

                poca_buf.append(poca_arr)
                gt_buf.append(gt_arr)

                if len(poca_buf) == BUFFER_SIZE or idx + 1 == n:
                    end = idx + 1
                    ds_poca[buf_start:end] = np.array(poca_buf, dtype=np.float32)
                    ds_gt  [buf_start:end] = np.array(gt_buf,   dtype=np.float32)
                    poca_buf, gt_buf, buf_start = [], [], end

                if (idx + 1) % max(1, n // 10) == 0 or idx + 1 == n:
                    print(f"    {split_name:5s}  [{idx+1:5d}/{n}]", end="\r")
            print()

    size_mb = out_path.stat().st_size / 1e6
    print(f"\n  ✓  Saved {out_path.name}  ({size_mb:.0f} MB)")

    # Quick verification
    print("\n  Verification:")
    with h5py.File(out_path, "r") as f:
        for split in ("train", "val", "test"):
            if split in f:
                p = f[f"{split}/poca"]
                g = f[f"{split}/gt"]
                sample_max = float(f[f"{split}/poca"][0, :, :, 0].max())
                print(f"    {split:5s}: poca={p.shape} dtype={p.dtype}  "
                      f"gt={g.shape} dtype={g.dtype}  "
                      f"sample[0] max={sample_max:.1f}")
        print(f"    poca_train_max stored in attrs: {f.attrs['poca_train_max']:.1f}")


# ===========================================================================
# ARGUMENT PARSING
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source", default=str(SOURCE_TAR),
        help=f"TAR file or directory of .root files (default: {SOURCE_TAR})",
    )
    p.add_argument(
        "--abs-npy-dir", default=str(ABS_NPY_DIR),
        help=f"Directory for intermediate float32 .npy files (default: {ABS_NPY_DIR})",
    )
    p.add_argument(
        "--gt-dir", default=str(GT_2D_DIR),
        help=f"Directory with tensorGT_2D_*.npy files (default: {GT_2D_DIR})",
    )
    p.add_argument(
        "--out", default=str(OUT_H5),
        help=f"Output H5 path (default: {OUT_H5})",
    )
    p.add_argument(
        "--skip-extract", action="store_true",
        help="Skip re-extraction from TAR if float32 .npy files already exist.",
    )
    p.add_argument(
        "--force-extract", action="store_true",
        help="Force re-extraction even if .npy already exists (overwrite).",
    )
    return p.parse_args()


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    args = parse_args()

    t0 = time.time()
    print()
    print("=" * 70)
    print("  CREATE ABSOLUTE-FLUX DATASET  →  dataset0_abs.h5")
    print("=" * 70)
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Source  : {args.source}")
    print(f"  GT dir  : {args.gt_dir}")
    print(f"  NPY dir : {args.abs_npy_dir}")
    print(f"  Output  : {args.out}")
    print()

    # ── Step 1: extract ────────────────────────────────────────────────────
    if args.skip_extract:
        print("[1/4] Skip-extract flag set — loading existing .npy list …")
        abs_npy_dir = Path(args.abs_npy_dir)
        poca_items = []
        for f in sorted(abs_npy_dir.glob("poca_abs_*.npy")):
            metadata = f.stem.replace("poca_abs_", "")
            poca_items.append((metadata, f))
        print(f"  Found {len(poca_items)} existing float32 .npy files")
    else:
        print("[1/4] Extracting absolute POCA projections from source …")
        skip_existing = not args.force_extract
        poca_items = extract_poca_absolute(
            args.source, args.abs_npy_dir, skip_existing=skip_existing
        )

    if not poca_items:
        raise SystemExit("[ERROR] No POCA items to process. Aborting.")

    # ── Step 2: GT lookup ──────────────────────────────────────────────────
    print("\n[2/4] Loading GT lookup …")
    gt_lookup = load_gt_lookup(args.gt_dir)

    # ── Step 3: match & split ──────────────────────────────────────────────
    print("\n[3/4] Matching POCA ↔ GT and splitting …")
    splits = match_and_split(poca_items, gt_lookup)

    # ── Step 4: write H5 ──────────────────────────────────────────────────
    print("\n[4/4] Writing H5 dataset …")
    write_h5(splits, args.out)

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print(f"  Done in {elapsed / 60:.1f} min")
    print(f"  Output : {args.out}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
