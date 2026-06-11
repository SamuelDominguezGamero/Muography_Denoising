"""
prepare_data_for_ResNet.py
==========================
Data preparation module for the material classifier ResNet.

Responsibilities
----------------
  • Scan run folders, parse labels from filenames
  • Stratify by (material × z_thickness_bin) for balanced train/val/test
  • Save indices to CSV files under data_indices/  (editable, reproducible)
  • Build tf.data pipelines from CSV indices

Indices persistence
-------------------
  data_indices/
    ├─ metadata.json       (timestamp, seed, runs, z_bins)
    ├─ train_index.csv     (path, material_id, material_name, z_true, run, z_bin)
    ├─ val_index.csv
    └─ test_index.csv

Public API
----------
    from prepare_data_for_ResNet import prepare_datasets

    train_ds, val_ds, test_ds = prepare_datasets("all")
    train_ds, val_ds, test_ds = prepare_datasets("run0")
    train_ds, val_ds, test_ds = prepare_datasets(["run0", "run2"],
                                                 config={"batch_size": 64})
    # To regenerate indices (ignore existing):
    train_ds, val_ds, test_ds = prepare_datasets("all", force_rebuild=True)

Each dataset element: (inputs, targets)
    inputs  = {"input_image": float32 [128, 128, 3],
               "input_flux":  float32 [1]}
    targets = {"output_material": int32   [],
               "output_z":        float32 [1]}

Filename conventions (two formats, both parsed automatically)
-------------------------------------------------------------
  run0 (words) :  ...mat<name>_word<W>_stroke<N>_depthZ<Z>.npy
  run1 / run2  :  ...mat<name>_dz<Z>.npy
"""

import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit


# ---------------------------------------------------------------------------
# Run folder registry & persistence directory
# ---------------------------------------------------------------------------

_SIM_DATA = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/simulation_data"
_UNET_ROOT = "/home/samuel/Work/Muography_Denoising/MuonGeneration/UNETs/UNET1"

RUN_FOLDERS = {
    "run0": os.path.join(_SIM_DATA, "run0_definitive_words"),
    "run1": os.path.join(_SIM_DATA, "run1_definitive_forms"),
    "run2": os.path.join(_SIM_DATA, "run2_definitive_blocks"),
}

# Directory where train/val/test indices are stored
DATA_INDICES_DIR = os.path.join(_UNET_ROOT, "data_indices")


# ---------------------------------------------------------------------------
# Material definitions
# ---------------------------------------------------------------------------

# Physically similar materials are merged into groups to avoid ambiguity.
#   aluminium / silicon  →  Z≈13-14, almost identical scattering
#   iron / steel         →  Z≈26,    near-identical interaction cross-section
_MATERIAL_MAP = {
    "aluminium": "aluminium-silicon",
    "silicon":   "aluminium-silicon",
    "iron":      "iron-steel",
    "steel":     "iron-steel",
    "lead":      "lead",
    "uranium":   "uranium",
    "water":     "water",
}

# Sorted so class indices are always the same regardless of insertion order.
MATERIAL_LABELS = sorted(set(_MATERIAL_MAP.values()))


# ---------------------------------------------------------------------------
# Normalisation constants & Z stratification bins
# ---------------------------------------------------------------------------

Z_MAX    = 60.0      # maximum Z thickness (cm);  z_norm = z_true / Z_MAX ∈ [0, 1]
MAX_FLUX = 25_000.0  # flux upper bound for log1p normalisation

# Z bins for stratification: ensures train/val/test have same proportions
# of thin, medium, and thick objects for each material
Z_BINS = [0, 15, 30, 60]  # [5-15), [15-30), [30-60] cm (labels: "thin", "med", "thick")


# ---------------------------------------------------------------------------
# Default config  (overridable via the config= argument of prepare_datasets)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "val_fraction":  0.10,
    "test_fraction": 0.10,
    "random_seed":   42,
    "batch_size":    32,
    # Dynamic subsampling (optional): if True, each epoch trains on a different
    # random subsample to combat overfitting on simulated data with redundancy.
    "subsample_enabled":  False,   # Set to True to enable dynamic subsampling
    "subsample_fraction": 0.30,    # Fraction of training data to use per epoch (0.3 = 30%)
}


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_RE_MAT     = re.compile(r'mat([a-z]+)')
_RE_DEPTH_Z = re.compile(r'depthZ(\d+(?:\.\d+)?)')             # run0: ...depthZ20...
_RE_DZ      = re.compile(r'_dz(\d+(?:\.\d+)?)(?:\.npy|_|$)')  # run1/2: ..._dz20...


def _parse_filename(path):
    """
    Extract (material_id, z_true) from a .npy filename.

    Returns None for files that should be skipped:
      - filenames containing "EMPTY"   (background samples, no object)
      - unknown material names
      - missing Z-thickness field
    """
    name = os.path.basename(path)

    if "EMPTY" in name:
        return None

    mat_match = _RE_MAT.search(name)
    if mat_match is None or mat_match.group(1) not in _MATERIAL_MAP:
        return None

    z_match = _RE_DEPTH_Z.search(name) or _RE_DZ.search(name)
    if z_match is None:
        return None

    group       = _MATERIAL_MAP[mat_match.group(1)]
    material_id = MATERIAL_LABELS.index(group)
    z_true      = float(z_match.group(1))

    return material_id, z_true


# ---------------------------------------------------------------------------
# Dataset index
# ---------------------------------------------------------------------------

def _build_index(run_names):
    """
    input: list of run names (e.g. ["run0", "run2"])
    Scan the requested run folders and return a DataFrame (one row per valid file).

    Columns: path, material_id, material_name, z_true, run
    """
    rows = []
    for run in run_names:
        folder = RUN_FOLDERS[run]
        if not os.path.isdir(folder):
            print(60 * "!"); print(60 * "!"); print(60 * "!")
            print(f"  [WARNING] Folder not found, skipping: {folder}")
            print(60 * "!"); print(60 * "!"); print(60 * "!")
            continue

        files     = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
        n_valid   = 0
        n_skipped = 0

        for fname in files:
            result = _parse_filename(fname)
            if result is None:
                n_skipped += 1
                continue
            material_id, z_true = result
            rows.append({
                "path":          os.path.join(folder, fname),
                "material_id":   material_id,
                "material_name": MATERIAL_LABELS[material_id],
                "z_true":        z_true,
                "run":           run,
            })
            n_valid += 1

        print(f"  {run} ({os.path.basename(folder)})  →  {n_valid} valid, {n_skipped} skipped")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Z-thickness binning & stratification
# ---------------------------------------------------------------------------

def _assign_z_bin(z_true):
    """
    Assign a z_bin label for stratification.
    E.g.: z=10cm → "thin", z=22cm → "med", z=50cm → "thick"
    """
    for i, (lo, hi) in enumerate(zip(Z_BINS[:-1], Z_BINS[1:])):
        if lo <= z_true < hi:
            return i
    return len(Z_BINS) - 2  # fallback to last bin if z_true >= 60


def _stratified_split(df, val_fraction, test_fraction, seed):
    """
    2D stratified split on (material × z_bin).

    Every (material, z_thickness_range) combination is proportionally
    represented in train/val/test, ensuring balanced thickness distribution
    across all partitions.
    """
    # Create z_bin column for stratification
    df = df.copy()
    df["z_bin"] = df["z_true"].apply(_assign_z_bin)

    # Encode (material_id, z_bin) as a unique integer for stratification
    n_bins = len(Z_BINS) - 1
    strat = df["material_id"].values * n_bins + df["z_bin"].values

    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction + test_fraction, random_state=seed
    )
    train_idx, valtest_idx = next(sss1.split(df, strat))

    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_fraction / (val_fraction + test_fraction),
        random_state=seed,
    )
    val_sub, test_sub = next(sss2.split(valtest_idx, strat[valtest_idx]))

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[valtest_idx[val_sub]].reset_index(drop=True),
        df.iloc[valtest_idx[test_sub]].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------

def _indices_exist_and_match(run_names, seed):
    """
    Check if train/val/test CSVs exist and have matching metadata.
    """
    meta_path = os.path.join(DATA_INDICES_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        return False

    try:
        with open(meta_path) as f:
            meta = json.load(f)
        # Check if seed and runs match
        if meta.get("random_seed") != seed or set(meta.get("runs", [])) != set(run_names):
            return False
        # Check if all CSV files exist
        for split in ["train", "val", "test"]:
            if not os.path.exists(os.path.join(DATA_INDICES_DIR, f"{split}_index.csv")):
                return False
        return True
    except Exception:
        return False


def _load_indices_from_csv():
    """
    Load train/val/test DataFrames from existing CSV files.
    """
    train_df = pd.read_csv(os.path.join(DATA_INDICES_DIR, "train_index.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_INDICES_DIR, "val_index.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_INDICES_DIR, "test_index.csv"))
    return train_df, val_df, test_df


def _save_indices_to_csv(train_df, val_df, test_df, run_names, seed):
    """
    Save train/val/test DataFrames to CSV files and metadata.json.
    """
    os.makedirs(DATA_INDICES_DIR, exist_ok=True)

    # Save splits
    train_df.to_csv(os.path.join(DATA_INDICES_DIR, "train_index.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_INDICES_DIR, "val_index.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_INDICES_DIR, "test_index.csv"), index=False)

    # Save metadata
    meta = {
        "timestamp":   datetime.now().isoformat(),
        "runs":        sorted(run_names),
        "random_seed": seed,
        "z_bins":      Z_BINS,
        "z_max":       Z_MAX,
    }
    with open(os.path.join(DATA_INDICES_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Indices saved to: {DATA_INDICES_DIR}/")
    print(f"  (You can manually edit the CSVs to customize splits)")
    print()


# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------

def _load_npy(path_bytes, material_id, z_true):
    """
    Load and normalise one sample.  Runs inside tf.numpy_function.

    Normalisation
    -------------
    image   : log1p(x) / (log1p(max(x)) + ε)   per-sample → [0, 1]
              POCA maps are right-skewed; log1p prevents dense voxels from
              dominating the gradient.
    flux    : log1p(channel-0 sum) / log1p(MAX_FLUX)  clipped to [0, 1]
              Keeps absolute muon statistics even after per-sample scaling.
    z_target: z_true / Z_MAX  → [0, 1]
              Puts MSE and SCCE losses on comparable scales.
    """
    img = np.load(path_bytes.decode()).astype(np.float32)  # [128, 128, 3]

    img_log  = np.log1p(img)
    img_norm = img_log / (img_log.max() + 1e-8)

    flux_raw  = float(img[:, :, 0].sum())
    flux_norm = float(np.clip(np.log1p(flux_raw) / np.log1p(MAX_FLUX), 0.0, 1.0))

    return (
        img_norm.astype(np.float32),
        np.array([flux_norm],              dtype=np.float32),
        np.int32(material_id),
        np.array([float(z_true) / Z_MAX],  dtype=np.float32),
    )


def _make_dynamic_subsampled_dataset(df, subsample_fraction, batch_size, seed):
    """
    Build a tf.data.Dataset with DYNAMIC subsampling per epoch.

    Key idea: Each time the dataset is iterated (i.e., each epoch), a fresh
    random subsample is selected. The model never sees the exact same subset
    twice, which combats overfitting on redundant simulated data while still
    exploiting the full statistical richness across all N epochs.

    Parameters
    ----------
    df : pd.DataFrame
        Full training data with columns: path, material_id, z_true, ...
    subsample_fraction : float
        Fraction of data to use per epoch, e.g. 0.3 = use 30% per epoch.
    batch_size : int
        Batch size for the returned dataset.
    seed : int
        Random seed for reproducibility (used to initialize np.random).

    Returns
    -------
    tf.data.Dataset
        When iterated, yields batches. Each epoch uses a different random
        subsample of the data (at indices level, before loading .npy files).
    """
    n_total = len(df)
    n_subsample = max(1, int(n_total * subsample_fraction))

    def generator():
        # This generator is called fresh at the start of each epoch,
        # so each iteration through the dataset gets a different subsample.
        rng = np.random.RandomState(seed)
        subset_indices = rng.choice(n_total, size=n_subsample, replace=False)
        subset_df = df.iloc[subset_indices]

        # Shuffle the subset within the epoch
        subset_df = subset_df.sample(frac=1.0, random_state=rng)

        for _, row in subset_df.iterrows():
            path = row["path"]
            mat_id = row["material_id"]
            z_true = row["z_true"]

            img, flux, mat_out, z_out = _load_npy(path.encode(), mat_id, z_true)

            inputs = {"input_image": img, "input_flux": flux}
            targets = {"output_material": mat_out, "output_z": z_out}
            yield inputs, targets

    # Determine output signature for the generator
    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            {
                "input_image": tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),
                "input_flux":  tf.TensorSpec(shape=(1,), dtype=tf.float32),
            },
            {
                "output_material": tf.TensorSpec(shape=(), dtype=tf.int32),
                "output_z":        tf.TensorSpec(shape=(1,), dtype=tf.float32),
            },
        ),
    )

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _make_tf_dataset(df, shuffle, batch_size, seed):
    """Build a tf.data.Dataset from a split DataFrame."""
    ds = tf.data.Dataset.from_tensor_slices((
        df["path"].values,
        df["material_id"].values.astype(np.int32),
        df["z_true"].values.astype(np.float32),
    ))

    def _map_fn(path, mat_id, z):
        img, flux, mat_out, z_out = tf.numpy_function(
            func=_load_npy,
            inp=[path, mat_id, z],
            Tout=[tf.float32, tf.float32, tf.int32, tf.float32],
        )
        img.set_shape([128, 128, 3])
        flux.set_shape([1])
        mat_out.set_shape([])
        z_out.set_shape([1])

        inputs  = {"input_image": img,         "input_flux": flux}
        targets = {"output_material": mat_out, "output_z":   z_out}
        return inputs, targets

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=2000, seed=seed, reshuffle_each_iteration=True)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_datasets(runs="all", config=None, force_rebuild=False):
    """
    Build train / val / test tf.data.Datasets from the requested run(s).

    Stratification is 2D: by (material × z_thickness_bin), ensuring balanced
    thickness distribution across all three partitions for each material.

    Indices are saved to CSV files under data_indices/ for reproducibility
    and manual review/editing.

    Parameters
    ----------
    runs : str | list[str]
        "run0"           → words geometry     (run0_definitive_words)
        "run1"           → forms geometry     (run1_definitive_forms)
        "run2"           → blocks geometry    (run2_definitive_blocks)
        "all"            → all three runs
        ["run0", "run2"] → any combination

    config : dict, optional
        Override any DEFAULT_CONFIG key:
            val_fraction, test_fraction, random_seed, batch_size

    force_rebuild : bool, default False
        If True, ignore existing CSV indices and regenerate them.

    Returns
    -------
    train_ds, val_ds, test_ds : tf.data.Dataset
        Each element: (inputs, targets)
            inputs  = {"input_image": Tensor[128,128,3], "input_flux": Tensor[1]}
            targets = {"output_material": Tensor[],      "output_z":   Tensor[1]}

    Examples
    --------
        train_ds, val_ds, test_ds = prepare_datasets("all")
        train_ds, val_ds, test_ds = prepare_datasets(["run0", "run2"],
                                                      config={"batch_size": 64})
        # Force regenerate:
        train_ds, val_ds, test_ds = prepare_datasets("all", force_rebuild=True)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    if runs == "all":
        run_names = list(RUN_FOLDERS.keys())
    elif isinstance(runs, str):
        run_names = [runs]
    else:
        run_names = list(runs)

    unknown = set(run_names) - set(RUN_FOLDERS)
    if unknown:
        raise ValueError(f"Unknown run name(s): {unknown}.  Valid: {list(RUN_FOLDERS)}")

    seed = cfg["random_seed"]

    # --- Try loading from existing CSVs ---
    if not force_rebuild and _indices_exist_and_match(run_names, seed):
        print("=== Loading dataset indices from CSV ===")
        train_df, val_df, test_df = _load_indices_from_csv()
        print(f"  Loaded train/val/test splits from: {DATA_INDICES_DIR}/")
        _print_split_summary(train_df, val_df, test_df)
    else:
        # --- Generate fresh indices ---
        print("=== Building dataset index ===")
        df = _build_index(run_names)

        if df.empty:
            raise RuntimeError(
                "No valid samples found. "
                "Check that the data folders exist and contain .npy files."
            )

        print(f"\n  Total    : {len(df)} samples  |  "
              f"Z range: {df['z_true'].min():.0f}–{df['z_true'].max():.0f} cm")
        _print_class_table(df)

        # --- 2D Stratified split (material × z_bin) ---
        print("\n=== Stratifying by (material × z_thickness) ===")
        train_df, val_df, test_df = _stratified_split(
            df, cfg["val_fraction"], cfg["test_fraction"], seed
        )
        _print_split_summary(train_df, val_df, test_df)

        # --- Save to CSV ---
        _save_indices_to_csv(train_df, val_df, test_df, run_names, seed)

    # --- Build tf.data datasets ---
    bs = cfg["batch_size"]
    
    # Use dynamic subsampling for training if enabled
    if cfg.get("subsample_enabled", False):
        frac = cfg.get("subsample_fraction", 0.30)
        print(f"\n  [DYNAMIC SUBSAMPLING ENABLED]")
        print(f"    Each epoch: use {frac*100:.0f}% of training data (different subset each epoch)")
        print(f"    This combats overfitting on simulated data with high redundancy.")
        print()
        train_ds = _make_dynamic_subsampled_dataset(
            train_df, subsample_fraction=frac, batch_size=bs, seed=seed
        )
    else:
        train_ds = _make_tf_dataset(train_df, shuffle=True, batch_size=bs, seed=seed)
    
    val_ds  = _make_tf_dataset(val_df,   shuffle=False, batch_size=bs, seed=seed)
    test_ds = _make_tf_dataset(test_df,  shuffle=False, batch_size=bs, seed=seed)

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_class_table(df):
    dist = df.groupby(["material_name", "run"]).size().unstack(fill_value=0)
    print("\n  Samples per material × run:")
    print(dist.to_string())
    print()


def _print_split_summary(train_df, val_df, test_df):
    print("=== Dataset splits ===")
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = split["material_name"].value_counts().sort_index().to_dict()
        print(f"  {name:5s}: {len(split):5d} samples  |  {counts}")
    
    # Show z_bin distribution to verify stratification
    print("\n=== Z-thickness distribution (stratification check) ===")
    z_labels = ["thin  [0-15)", "med   [15-30)", "thick [30-60]"]
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if "z_bin" in split.columns:
            z_dist = split["z_bin"].value_counts().sort_index()
            counts_str = " | ".join(f"{z_labels[i]}: {z_dist.get(i, 0)}" for i in range(len(z_labels)))
            print(f"  {name:5s}: {counts_str}")
    print()
