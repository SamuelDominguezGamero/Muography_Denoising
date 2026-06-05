"""
train_material_classifier.py
============================
Multi-input, multi-task ResNet-18 classifier for Muon Scattering Tomography (MST).

Problem
-------
Classify the material of an object (6 classes) from 2D POCA projection tensors
[128, 128, 3] (XY, XZ, YZ planes) while simultaneously estimating the object's
Z-thickness via an auxiliary regression head.

The estimated Z-thickness is reinjected into the classification fusion block to
explicitly break the physical degeneracy: a thick low-Z object and a thin high-Z
object can produce visually identical POCA maps. Providing the geometric thickness
as an additional input to the classifier allows it to disambiguate these cases.

Architecture
------------
  input_image [128,128,3] ──► ResNet-18 backbone ──► GAP ──► feat (128-d)
                                                               │
                                                         Dense(1) ──► output_z   (MSE head)
                                                               │         │
  input_flux  [1] ──────────────────────────────────────────►Concat ◄───┘
                                                               │
                                                         Dense(64, relu)
                                                         Dropout(0.4)
                                                         Dense(4, softmax) ──► output_material  (SCCE head)

Datasets used
-------------
  run0_POCA_projections_abs_downsampled/   (letters/words geometry;  Binomial-thinned p=0.14)
  run1_POCA_projections_NotLetters/        (cylinders, spheres, triangles, ...)
  run2_POCA_projections_blocks/            (simple rectangular blocks)

Filename parsing (two conventions handled automatically)
--------------------------------------------------------
  run0:    ...mat<name>_word<W>_stroke<N>_depthZ<Z>.npy
  run1/2:  ...mat<name>_dz<Z>.npy

Usage
-----
  # Train from scratch (or resume if a checkpoint already exists):
      python train_material_classifier.py

  # Evaluate a saved model without training:
      Set  EVAL_ONLY = True  in CONFIG, then run:
      python train_material_classifier.py

  # TensorBoard (if use_tensorboard=True):
      tensorboard --logdir UNETs/UNET1/logs/tensorboard

Output files
------------
  checkpoints/best_model.keras   ← best model by val_loss
  logs/training_log.csv          ← one row per epoch, all metrics
  logs/tensorboard/              ← TensorBoard event files (optional)

"""

import csv
import os
import re
import sys
import time

# Suppress TensorFlow info/warning/error messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel("ERROR")   # suppress Python-level TF warnings too

# To avoid GPU compatibility issues
tf.config.optimizer.set_jit(False)
from sklearn.model_selection import StratifiedShuffleSplit

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

BASE_DATA = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/simulation_data"
BASE_OUT  = "/home/samuel/Work/Muography_Denoising/MuonGeneration/UNETs/UNET1"

CONFIG = {
    # ── Data folders ──────────────────────────────────────────────────────────
    # run0 uses the Binomial-thinned copy (p=0.14) so all three runs have
    # comparable flux (~15k–20k counts in the XY channel).
    "data_folders": [
        os.path.join(BASE_DATA, "run0_POCA_projections_abs_downsampled"),
        os.path.join(BASE_DATA, "run1_POCA_projections_NotLetters"),
        os.path.join(BASE_DATA, "run2_POCA_projections_blocks"),
    ],

    # ── Output paths ─────────────────────────────────────────────────────────
    "checkpoint_path": os.path.join(BASE_OUT, "checkpoints", "best_model.keras"),
    "csv_log_path":    os.path.join(BASE_OUT, "logs", "training_log.csv"),
    "tb_log_dir":      os.path.join(BASE_OUT, "logs", "tensorboard"),

    # ── Class definitions ─────────────────────────────────────────────────────
    # 4 grouped classes: physically indistinguishable pairs are merged.
    #   aluminium-silicon : Z≈13–14  — very similar scattering signatures
    #   iron-steel        : Z≈26    — near-identical muon interaction cross-section
    #   lead              : Z=82
    #   uranium           : Z=92
    "material_labels": ["aluminium-silicon", "iron-steel", "lead", "uranium"],

    "material_to_group": {
        "aluminium": "aluminium-silicon",
        "silicon":   "aluminium-silicon",
        "iron":      "iron-steel",
        "steel":     "iron-steel",
        "lead":      "lead",
        "uranium":   "uranium",
    },

    # ── Normalisation constants ───────────────────────────────────────────────
    # Z_MAX (cm): upper bound for the regression target → normalises z to [0,1].
    # Observed range in dataset: 5–60 cm.
    "z_max": 60.0,

    # MAX_FLUX: upper bound for the flux scalar log1p-normalisation.
    # After Binomial thinning of run0, maximum observed flux is ~19 000.
    # 25 000 provides a safe margin.
    "max_flux": 25_000.0,

    # ── Dataset split ─────────────────────────────────────────────────────────
    "val_fraction":  0.10,   # fraction of total data held out for validation
    "test_fraction": 0.10,   # fraction of total data held out for test
    "random_seed":   42,

    # ── Training hyper-parameters ─────────────────────────────────────────────
    "batch_size":    32,
    "epochs":        200,    # EarlyStopping will terminate training earlier in practice
    "learning_rate": 1e-3,

    # Relative weight of each loss head in the combined loss.
    # SCCE is O(0.1–2); MSE on z_norm is O(0.01–0.1) -> 0.3 keeps them comparable.
    "loss_weight_material": 1.0,
    "loss_weight_z":        0.3,

    # ── Regularisation ────────────────────────────────────────────────────────
    "dropout_rate": 0.05,    # applied in the classification Dense head
    "l2_lambda":    1e-4,   # L2 weight decay on Conv2D and Dense kernel weights

    # ── Callback settings ─────────────────────────────────────────────────────
    "early_stopping_patience": 15,  
    "reduce_lr_patience":       7,  
    "reduce_lr_factor":         0.5,
    "min_lr":                   1e-6,

    # ── Execution flags ───────────────────────────────────────────────────────
    # eval_only:     skip training, load best checkpoint, evaluate on test set.
    # force_restart: ignore any existing checkpoint and train from scratch.
    #                The old checkpoint file is NOT deleted automatically;
    #                it will be overwritten once a new best val_loss is found.
    "eval_only":       False,
    "force_restart":   True,
    "use_tensorboard": True,
}


# =============================================================================
# 2. FILENAME PARSING
#    Two naming conventions co-exist across the three runs.
#    A single function handles both using regex with fallback.
# =============================================================================

# run0 convention:  ...depthZ<value>.npy
_RE_DEPTH_Z = re.compile(r'depthZ(\d+(?:\.\d+)?)')

# run1/run2 convention:  ..._dz<value>.npy  or  ..._dz<value>_...
_RE_DZ = re.compile(r'_dz(\d+(?:\.\d+)?)(?:\.npy|_|$)')

# Material name embedded in the filename as mat<name>
_RE_MATERIAL = re.compile(r'mat([a-z]+)')


def parse_filename(filepath):
    """
    Extract the material class id and Z-thickness from a .npy filename.

    Returns
    -------
    (material_id: int, z_true: float)  if the file is usable, or
    None                               if the file should be skipped.

    Files are skipped when:
      - The filename contains "EMPTY" (no material object present).
      - The material name is not in the known class list.
      - The Z-thickness cannot be extracted.
    """
    fname = os.path.basename(filepath)

    # Skip empty-volume files (no object, not a useful training sample)
    if "EMPTY" in fname:
        return None

    # ── Material ──────────────────────────────────────────────────────────────
    m = _RE_MATERIAL.search(fname)
    if m is None:
        return None
    material_name = m.group(1)
    if material_name not in CONFIG["material_to_group"]:
        return None
    group_name  = CONFIG["material_to_group"][material_name]
    material_id = CONFIG["material_labels"].index(group_name)

    # ── Z thickness ───────────────────────────────────────────────────────────
    # Try run0 convention first; fall back to run1/2 convention.
    z_match = _RE_DEPTH_Z.search(fname) or _RE_DZ.search(fname)
    if z_match is None:
        return None
    z_true = float(z_match.group(1))

    return material_id, z_true


# =============================================================================
# 3. DATASET INDEXING
#    Scan all configured data folders and build a single pandas DataFrame
#    with one row per valid .npy file.
# =============================================================================

def build_file_index(folders):
    """
    Scan each folder, parse every .npy filename, and return a clean DataFrame.

    Columns
    -------
    path          : absolute path to the .npy file
    material_id   : integer class label (0–5)
    material_name : human-readable material name
    z_true        : Z-thickness in cm (from the filename)
    folder_id     : basename of the source folder (used as a stratification key)
    """
    rows = []
    for folder in folders:
        folder_id = os.path.basename(folder)
        if not os.path.isdir(folder):
            print(f"  [WARNING] Folder not found, skipping: {folder}")
            continue

        npy_files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
        n_skipped = 0
        for fname in npy_files:
            fpath  = os.path.join(folder, fname)
            result = parse_filename(fpath)
            if result is None:
                n_skipped += 1
                continue
            material_id, z_true = result
            rows.append({
                "path":          fpath,
                "material_id":   material_id,
                "material_name": CONFIG["material_labels"][material_id],
                "z_true":        z_true,
                "folder_id":     folder_id,
            })
        print(f"  {folder_id:50s}  {len(npy_files)-n_skipped:5d} valid, {n_skipped:3d} skipped")

    df = pd.DataFrame(rows)
    print(f"\n  Total valid files  : {len(df)}")
    print("\n  Class distribution (valid files per material × folder):")
    dist = df.groupby(["material_name", "folder_id"]).size().unstack(fill_value=0)
    print(dist.to_string())
    z_vals = sorted(df["z_true"].unique())
    print(f"\n  Z-thickness range  : {df['z_true'].min():.1f} – {df['z_true'].max():.1f} cm")
    print(f"  Distinct Z values  : {[f'{v:.0f}cm' for v in z_vals]}")
    print()
    return df


# =============================================================================
# 4. STRATIFIED SPLIT
#    80 % train / 10 % val / 10 % test.
#    Stratification key: (material_id × folder_id) encoded as a single integer.
#    This guarantees every (material, geometry-family) combination is
#    proportionally represented in all three splits.
# =============================================================================

def stratified_split(df, config):
    """
    Split the file index DataFrame into train / val / test subsets.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    """
    folder_ids  = sorted(df["folder_id"].unique())
    folder_map  = {f: i for i, f in enumerate(folder_ids)}
    n_folders   = len(folder_ids)

    # Encode (material_id, folder_id) as a unique integer for stratification
    strat_label = (
        df["material_id"].values * n_folders
        + df["folder_id"].map(folder_map).values
    )

    seed      = config["random_seed"]
    val_frac  = config["val_fraction"]
    test_frac = config["test_fraction"]

    # Step 1: separate train from the combined (val + test) pool
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac + test_frac, random_state=seed
    )
    train_idx, valtest_idx = next(sss1.split(df, strat_label))

    # Step 2: split the (val + test) pool into val and test
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_frac / (val_frac + test_frac),
        random_state=seed,
    )
    val_sub_idx, test_sub_idx = next(
        sss2.split(valtest_idx, strat_label[valtest_idx])
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[valtest_idx[val_sub_idx]].reset_index(drop=True)
    test_df  = df.iloc[valtest_idx[test_sub_idx]].reset_index(drop=True)

    print("=== Dataset splits ===")
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        mat_counts = split["material_name"].value_counts().sort_index().to_dict()
        print(f"  {name:5s} : {len(split):5d} samples  |  {mat_counts}")
    print()
    return train_df, val_df, test_df


# =============================================================================
# 5. DATA PIPELINE  (tf.data)
#    Each .npy file is loaded lazily on demand using tf.numpy_function.
#    No full dataset is held in RAM at once.
# =============================================================================

def _load_sample(path_bytes, material_id, z_true):
    """
    Load and preprocess one sample.  Runs inside tf.numpy_function.

    Normalisation applied
    ---------------------
    Image:
        log1p(x) / (log1p(max(x)) + eps)  →  [0, 1] per-sample.
        Rationale: POCA heatmaps are heavily right-skewed (most voxels ≈ 0,
        a few 'hot' voxels with hundreds of counts).  log1p compresses the
        tail so that low-count voxels — which encode the object's shape and
        boundaries — contribute meaningfully to the gradient rather than
        being dominated by the dense core.

    Flux scalar:
        log1p(sum_ch0) / log1p(MAX_FLUX)  →  clipped to [0, 1].
        The XY channel (ch0) is used; all three channels sum to the same
        total (same POCA points projected onto different planes).
        Kept separate so the network retains knowledge of absolute muon
        statistics even after per-sample image normalisation.

    Z target:
        z_true / Z_MAX  →  [0, 1].
        Required so that MSE (O(0.01–0.1)) and SCCE (O(0.1–2)) are on
        comparable scales, making loss_weights numerically meaningful.

    Returns
    -------
    img_norm   : float32 [128, 128, 3]
    flux_arr   : float32 [1]
    material_id: int32   []
    z_arr      : float32 [1]
    """
    path = path_bytes.decode("utf-8")
    img  = np.load(path).astype(np.float32)   # [128, 128, 3]

    # Raw flux from channel 0 before any normalisation
    flux_raw = float(img[:, :, 0].sum())

    # Image: log1p then scale per-sample to [0, 1]
    img_log = np.log1p(img)
    eps     = 1e-8
    img_max = float(img_log.max())
    img_norm = img_log / (img_max + eps)

    # Flux: log1p-normalise and clip to [0, 1] as safety guard
    flux_norm = np.log1p(flux_raw) / np.log1p(CONFIG["max_flux"])
    flux_norm = float(np.clip(flux_norm, 0.0, 1.0))
    flux_arr  = np.array([flux_norm], dtype=np.float32)

    # Z target: normalise to [0, 1]
    z_arr = np.array([float(z_true) / CONFIG["z_max"]], dtype=np.float32)

    return (
        img_norm.astype(np.float32),
        flux_arr,
        np.int32(material_id),
        z_arr,
    )


def build_dataset(df, training, config):
    """
    Build a tf.data.Dataset from a split DataFrame.

    Each element is a tuple (inputs, targets):
        inputs  = {"input_image": Tensor[128,128,3],  "input_flux": Tensor[1]}
        targets = {"output_material": Tensor[],       "output_z":   Tensor[1]}

    Parameters
    ----------
    df       : pd.DataFrame  (one of train_df / val_df / test_df)
    training : bool          (True enables shuffling)
    config   : dict
    """
    paths        = df["path"].values
    material_ids = df["material_id"].values.astype(np.int32)
    z_trues      = df["z_true"].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, material_ids, z_trues))

    def _map_fn(path, mat_id, z_true):
        # tf.numpy_function bridges the tf.data graph with NumPy/Python code.
        # Output shapes are unknown after this call and must be set explicitly.
        img_norm, flux_arr, mat_id_out, z_arr = tf.numpy_function(
            func=_load_sample,
            inp=[path, mat_id, z_true],
            Tout=[tf.float32, tf.float32, tf.int32, tf.float32],
        )
        # Declare static shapes so downstream layers know what to expect
        img_norm.set_shape([128, 128, 3])
        flux_arr.set_shape([1])
        mat_id_out.set_shape([])
        z_arr.set_shape([1])

        inputs  = {"input_image": img_norm, "input_flux": flux_arr}
        targets = {"output_material": mat_id_out, "output_z": z_arr}
        return inputs, targets

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        # Reshuffle every epoch so the model sees a different order each time
        ds = ds.shuffle(
            buffer_size=2000,
            seed=config["random_seed"],
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(config["batch_size"])
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# =============================================================================
# 6. MODEL DEFINITION
# =============================================================================

def _resnet_block(x, filters, stride=1, l2_lambda=1e-4):
    """
    Basic ResNet residual block  (He et al., 2015 — "Deep Residual Learning").

    Layout
    ------
      input
        │
        ├── Conv(3×3, stride) ── BN ── ReLU ── Conv(3×3, 1) ── BN
        │                                                         │
        └── (identity or 1×1 projection if shape changes)        Add ── ReLU
                                                                  │
                                                               output

    The skip connection lets gradients flow directly to earlier layers,
    mitigating vanishing gradients in deeper networks.
    BatchNorm after each Conv speeds up convergence and acts as a mild
    regulariser (reduces the need for large dropout in the backbone).

    Parameters
    ----------
    x         : Keras tensor  (input feature map)
    filters   : int           (number of output channels)
    stride    : int           (spatial downsampling factor; 1 = no downsampling)
    l2_lambda : float         (L2 weight decay applied to Conv kernels)
    """
    reg      = tf.keras.regularizers.l2(l2_lambda)
    shortcut = x

    # First convolution (may downsample if stride > 1)
    x = tf.keras.layers.Conv2D(
        filters, 3, strides=stride, padding="same",
        use_bias=False, kernel_regularizer=reg,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Second convolution (always stride=1; no activation before the Add)
    x = tf.keras.layers.Conv2D(
        filters, 3, strides=1, padding="same",
        use_bias=False, kernel_regularizer=reg,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Project the shortcut when spatial dimensions or channel count change
    if stride != 1 or int(shortcut.shape[-1]) != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, 1, strides=stride, padding="same",
            use_bias=False, kernel_regularizer=reg,
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x


def build_model(config):
    """
    Build the multi-input, multi-task ResNet-18 model using the Keras Functional API.

    Inputs
    ------
    input_image : shape [128, 128, 3]  log1p-normalised POCA projections
    input_flux  : shape [1]            log1p-normalised total POCA count

    Outputs
    -------
    output_material : shape [6]   softmax class probabilities  (main task)
    output_z        : shape [1]   normalised Z-thickness       (auxiliary task)

    Spatial resolution through the backbone
    ----------------------------------------
    input          128×128
    stem Conv      → 64×64
    stem MaxPool   → 32×32
    Stage 1 (×2)   → 32×32   (32 filters)
    Stage 2 (×2)   → 16×16   (64 filters, first block stride=2)
    Stage 3 (×2)   →  8×8    (128 filters, first block stride=2)
    GAP            → 128-d vector

    Why z_pred is reinjected into the fusion block
    -----------------------------------------------
    The main classification head receives:
        feat_visual (128-d) + z_pred (1-d) + flux (1-d)  →  130-d
    Providing z_pred explicitly lets the classifier reason:
        "given this scattering pattern AND this estimated thickness,
         which material best explains the observation?"
    This directly breaks the physical degeneracy between material Z and
    object thickness that makes the problem hard from vision features alone.
    """
    reg     = tf.keras.regularizers.l2(config["l2_lambda"])
    n_class = len(config["material_labels"])

    # ── Inputs ───────────────────────────────────────────────────────────────
    input_image = tf.keras.Input(shape=(128, 128, 3), name="input_image")
    input_flux  = tf.keras.Input(shape=(1,),          name="input_flux")

    # ── ResNet-18 backbone ────────────────────────────────────────────────────
    # Stem: large 7×7 kernel captures broad spatial context cheaply.
    x = tf.keras.layers.Conv2D(
        32, 7, strides=2, padding="same",
        use_bias=False, kernel_regularizer=reg, name="stem_conv",
    )(input_image)                                          # → 64×64×32
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)
    x = tf.keras.layers.MaxPooling2D(
        3, strides=2, padding="same", name="stem_pool"
    )(x)                                                    # → 32×32×32

    # Stage 1: two blocks, 32 filters, spatial size unchanged
    x = _resnet_block(x, 32, stride=1, l2_lambda=config["l2_lambda"])
    x = _resnet_block(x, 32, stride=1, l2_lambda=config["l2_lambda"])   # 32×32

    # Stage 2: two blocks, 64 filters, halve spatial size in the first block
    x = _resnet_block(x, 64, stride=2, l2_lambda=config["l2_lambda"])   # 16×16
    x = _resnet_block(x, 64, stride=1, l2_lambda=config["l2_lambda"])   # 16×16

    # Stage 3: two blocks, 128 filters, halve spatial size in the first block
    x = _resnet_block(x, 128, stride=2, l2_lambda=config["l2_lambda"])  #  8×8
    x = _resnet_block(x, 128, stride=1, l2_lambda=config["l2_lambda"])  #  8×8

    # Aggregate the spatial feature map into a fixed-size vector
    feat = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)         # 128-d

    # ── Auxiliary Z-thickness regression head ──────────────────────────────────
    # This Dense(1) serves a dual purpose:
    #   1. As a supervised output (loss: MSE on z_norm).
    #   2. Its value is fed into the fusion block below.
    z_pred = tf.keras.layers.Dense(
        1, activation="linear",
        kernel_regularizer=reg,
        name="output_z",
    )(feat)

    # ── Late fusion block ─────────────────────────────────────────────────────
    # Concatenate: visual features (128) + Z estimate (1) + flux scalar (1)
    fused = tf.keras.layers.Concatenate(name="fusion")([feat, z_pred, input_flux])

    # Two-layer classification head with dropout regularisation
    x = tf.keras.layers.Dense(
        64, activation="relu",
        kernel_regularizer=reg,
        name="fc1",
    )(fused)
    x = tf.keras.layers.Dropout(config["dropout_rate"], name="dropout")(x)

    output_material = tf.keras.layers.Dense(
        n_class, activation="softmax", name="output_material"
    )(x)

    # ── Assemble model ────────────────────────────────────────────────────────
    model = tf.keras.Model(
        inputs=[input_image, input_flux],
        outputs=[output_material, z_pred],
        name="MaterialClassifier_ResNet18",
    )
    return model


# =============================================================================
# 7. TEST EVALUATION HELPER + CUSTOM CALLBACK — EPOCH MONITOR
# =============================================================================

def _eval_test_metrics(model, test_dataset, z_max, loss_weight_mat, loss_weight_z):
    """
    Evaluate the test dataset manually, bypassing model.metrics_names.

    Motivation: model.metrics_names varies across TF versions and can omit
    per-output metrics (accuracy, mae) entirely in TF 2.16+ / Keras 3.
    A manual numpy pass is version-independent and always correct.

    Returns a dict with keys:
        accuracy, mat_loss, z_mae_cm, z_mse, total_loss
    """
    mat_probs_list, z_pred_list = [], []
    mat_true_list,  z_true_list = [], []

    for inputs, targets in test_dataset:
        # Run forward pass in inference mode (BN uses running stats, dropout off)
        out = model(inputs, training=False)
        # model.outputs order: [output_material, output_z]
        mat_probs_list.append(out[0].numpy())
        z_pred_list.append(out[1].numpy())
        mat_true_list.append(targets["output_material"].numpy())
        z_true_list.append(targets["output_z"].numpy())

    mat_probs = np.concatenate(mat_probs_list, axis=0)          # [N, 6]
    z_preds   = np.concatenate(z_pred_list,   axis=0).flatten() # [N]
    mat_true  = np.concatenate(mat_true_list, axis=0).astype(np.int32)  # [N]
    z_true    = np.concatenate(z_true_list,   axis=0).flatten() # [N]

    # Material accuracy
    mat_acc = float(np.mean(np.argmax(mat_probs, axis=1) == mat_true))

    # Material loss (SparseCategoricalCrossentropy)
    # Clip predictions to avoid log(0)
    mat_probs_clipped = np.clip(mat_probs, 1e-7, 1.0)
    log_probs = np.log(mat_probs_clipped)
    mat_loss  = float(-np.mean(log_probs[np.arange(len(mat_true)), mat_true]))

    # Z metrics
    z_mae_norm = float(np.mean(np.abs(z_preds - z_true)))
    z_mse      = float(np.mean((z_preds - z_true) ** 2))

    total_loss = loss_weight_mat * mat_loss + loss_weight_z * z_mse

    return {
        "accuracy":   mat_acc,
        "mat_loss":   mat_loss,
        "z_mae_cm":   z_mae_norm * z_max,
        "z_mse":      z_mse,
        "total_loss": total_loss,
    }

class EpochMonitor(tf.keras.callbacks.Callback):
    """
    Per-epoch metrics table printed to stdout + written to a CSV file.

    Metrics shown for train, val, and test splits
    ---------------------------------------------
    Material accuracy     : fraction of correctly classified samples
    Material loss (SCCE)  : SparseCategoricalCrossentropy
    Z MAE (cm)            : mean absolute error converted back to centimetres
    Z loss (MSE)          : MeanSquaredError on the normalised [0,1] Z target
    Total loss            : weighted combination of both losses

    The test set is evaluated via a manual numpy forward pass each epoch
    (not model.evaluate) to avoid metric name issues across TF versions.
    It is NOT used for EarlyStopping or ModelCheckpoint — no data leakage.

    Parameters
    ----------
    test_dataset     : tf.data.Dataset
    z_max            : float   (Z_MAX in cm)
    csv_path         : str     (path to the CSV log file)
    loss_weight_mat  : float   (weight for material loss in total loss)
    loss_weight_z    : float   (weight for z loss in total loss)
    """

    def __init__(self, test_dataset, z_max, csv_path, loss_weight_mat, loss_weight_z):
        super().__init__()
        self.test_dataset    = test_dataset
        self.z_max           = z_max
        self.csv_path        = csv_path
        self.loss_weight_mat = loss_weight_mat
        self.loss_weight_z   = loss_weight_z
        self._csv_file       = None
        self._csv_writer     = None
        self._epoch_start    = None
        self._train_start    = None

    def on_train_begin(self, logs=None):
        self._train_start = time.time()
        print("  Columns logged each epoch: mat_acc | mat_loss | Z_MAE(cm) | Z_MSE | total_loss")
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self._csv_file   = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "epoch", "lr",
            "train_mat_acc",  "val_mat_acc",  "test_mat_acc",
            "train_mat_loss", "val_mat_loss", "test_mat_loss",
            "train_z_mae_cm", "val_z_mae_cm", "test_z_mae_cm",
            "train_z_mse",    "val_z_mse",    "test_z_mse",
            "train_loss",     "val_loss",     "test_loss",
        ])
        self._csv_file.flush()

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Manual test evaluation — version-independent, bypasses metrics_names
        td = _eval_test_metrics(
            self.model, self.test_dataset, self.z_max,
            self.loss_weight_mat, self.loss_weight_z,
        )

        # Convenience helper to read a float from the Keras logs dict
        def g(key):
            return float(logs.get(key, 0.0))

        # ── Material classification metrics ───────────────────────────────────
        train_acc   = g("output_material_accuracy")
        val_acc     = g("val_output_material_accuracy")
        test_acc    = td["accuracy"]

        train_mloss = g("output_material_loss")
        val_mloss   = g("val_output_material_loss")
        test_mloss  = td["mat_loss"]

        # ── Z-thickness regression metrics (MAE converted to cm) ──────────────
        train_z_mae = g("output_z_mae") * self.z_max
        val_z_mae   = g("val_output_z_mae") * self.z_max
        test_z_mae  = td["z_mae_cm"]

        train_z_mse = g("output_z_loss")
        val_z_mse   = g("val_output_z_loss")
        test_z_mse  = td["z_mse"]

        # ── Total weighted loss ───────────────────────────────────────────────
        train_loss = g("loss")
        val_loss   = g("val_loss")
        test_loss  = td["total_loss"]

        # Current learning rate (get_value works for both scalar and schedule LRs)
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        # ── Formatted table ───────────────────────────────────────────────────
        n_epochs = self.params.get("epochs", CONFIG["epochs"])
        bar = "─" * 74
        print(f"\n{bar}")
        print(f"  Epoch {epoch+1:03d}/{n_epochs:03d}"
              f"                     TRAIN        VAL          TEST")
        print(bar)
        print(f"  Material accuracy  :       {train_acc:7.4f}      {val_acc:7.4f}      {test_acc:7.4f}")
        print(f"  Material loss      :       {train_mloss:7.4f}      {val_mloss:7.4f}      {test_mloss:7.4f}")
        print(f"  Z MAE (cm)         :       {train_z_mae:7.2f}      {val_z_mae:7.2f}      {test_z_mae:7.2f}")
        print(f"  Z loss (MSE norm)  :       {train_z_mse:7.4f}      {val_z_mse:7.4f}      {test_z_mse:7.4f}")
        elapsed = time.time() - self._epoch_start if self._epoch_start else 0.0
        print(f"  Total loss         :       {train_loss:7.4f}      {val_loss:7.4f}      {test_loss:7.4f}")
        print(f"  Learning rate      :       {lr:.2e}")
        print(f"  Epoch time         :       {elapsed:.1f}s")
        print(bar)
        sys.stdout.flush()

        # ── CSV row ───────────────────────────────────────────────────────────
        self._csv_writer.writerow([
            epoch + 1,         f"{lr:.2e}",
            f"{train_acc:.6f}",   f"{val_acc:.6f}",   f"{test_acc:.6f}",
            f"{train_mloss:.6f}", f"{val_mloss:.6f}", f"{test_mloss:.6f}",
            f"{train_z_mae:.4f}", f"{val_z_mae:.4f}", f"{test_z_mae:.4f}",
            f"{train_z_mse:.6f}", f"{val_z_mse:.6f}", f"{test_z_mse:.6f}",
            f"{train_loss:.6f}",  f"{val_loss:.6f}",  f"{test_loss:.6f}",
        ])
        self._csv_file.flush()

    def on_train_end(self, logs=None):
        if self._csv_file:
            self._csv_file.close()
        if self._train_start:
            total = time.time() - self._train_start
            h, rem = divmod(int(total), 3600)
            m, s   = divmod(rem, 60)
            print(f"\n  Total training time : {h:02d}h {m:02d}m {s:02d}s")


# =============================================================================
# 8. COMPILATION AND CALLBACKS
# =============================================================================

def compile_model(model, config):
    """
    Attach the optimiser, losses, and metrics to the model.

    Loss assignment
    ---------------
    output_material : SparseCategoricalCrossentropy   (integer class labels)
    output_z        : MeanSquaredError on z_norm ∈ [0,1]

    Metrics tracked per head
    ------------------------
    output_material : SparseCategoricalAccuracy  (reported as 'accuracy' in logs)
    output_z        : MeanAbsoluteError          (reported as 'mae' in logs)
                      → multiply by z_max to convert to cm
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss={
            "output_material": tf.keras.losses.SparseCategoricalCrossentropy(),
            "output_z":        tf.keras.losses.MeanSquaredError(),
        },
        loss_weights={
            "output_material": config["loss_weight_material"],
            "output_z":        config["loss_weight_z"],
        },
        metrics={
            "output_material": [
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ],
            "output_z": [
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
            ],
        },
        # Disable XLA JIT: Keras 3 defaults to jit_compile="auto" which
        # enables XLA on GPU.  On RTX 5060 (Compute Capability 12.0 /
        # Blackwell) this triggers CUDNN_STATUS_EXECUTION_FAILED at runtime.
        jit_compile=False,
    )


def get_callbacks(test_dataset, config):
    """
    Return the list of callbacks used during training.

    Callbacks included
    ------------------
    EpochMonitor      : custom — prints formatted table + writes CSV each epoch
    ModelCheckpoint   : saves best model by val_output_material_accuracy
    EarlyStopping     : stops when val_output_material_accuracy stops improving (patience=15)
    ReduceLROnPlateau : halves LR when val_output_material_accuracy plateaus (patience=7)
    TensorBoard       : optional; controlled by config["use_tensorboard"]
    """
    os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)

    callbacks = [
        EpochMonitor(
            test_dataset=test_dataset,
            z_max=config["z_max"],
            csv_path=config["csv_log_path"],
            loss_weight_mat=config["loss_weight_material"],
            loss_weight_z=config["loss_weight_z"],
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config["checkpoint_path"],
            monitor="val_output_material_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_output_material_accuracy",
            mode="max",
            patience=config["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_output_material_accuracy",
            mode="max",
            factor=config["reduce_lr_factor"],
            patience=config["reduce_lr_patience"],
            min_lr=config["min_lr"],
            verbose=1,
        ),
    ]

    if config["use_tensorboard"]:
        os.makedirs(config["tb_log_dir"], exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=config["tb_log_dir"],
                histogram_freq=0,    # set to 1 for weight histograms (slower)
                update_freq="epoch",
            )
        )

    return callbacks


# =============================================================================
# 9. MAIN
# =============================================================================

def _print_config_summary(config):
    """Print a compact, human-readable summary of the active configuration."""
    print("=== Training configuration ===")
    print(f"  Batch size         : {config['batch_size']}")
    print(f"  Max epochs         : {config['epochs']}  (EarlyStopping patience={config['early_stopping_patience']})")
    print(f"  Initial LR         : {config['learning_rate']}")
    print(f"  Loss weights       : material={config['loss_weight_material']},  z={config['loss_weight_z']}")
    print(f"  Dropout            : {config['dropout_rate']}  |  L2 lambda: {config['l2_lambda']}")
    print(f"  ReduceLR           : factor={config['reduce_lr_factor']},  patience={config['reduce_lr_patience']},  min_lr={config['min_lr']}")
    print(f"  Checkpoint         : {config['checkpoint_path']}")
    print(f"  CSV log            : {config['csv_log_path']}")
    print()


def _print_final_metrics(model, test_ds, config):
    """Helper: evaluate on test set and print a readable metric summary."""
    print("\n=== Final evaluation on test set ===")
    td = _eval_test_metrics(
        model, test_ds, config["z_max"],
        config["loss_weight_material"], config["loss_weight_z"],
    )
    print(f"  Material accuracy  : {td['accuracy']:.4f}  ({td['accuracy']*100:.1f} %)")
    print(f"  Material loss      : {td['mat_loss']:.4f}")
    print(f"  Z MAE              : {td['z_mae_cm']:.2f} cm")
    print(f"  Z loss (MSE norm)  : {td['z_mse']:.4f}")
    print(f"  Total loss         : {td['total_loss']:.4f}")


def main():
    print("=" * 74)
    print("  Material Classifier — Muon Scattering Tomography")
    print("=" * 74)
    print(f"  TensorFlow version : {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  GPUs available     : {len(gpus)}  (memory growth enabled)")
        for i, gpu in enumerate(gpus):
            try:
                details = tf.config.experimental.get_device_details(gpu)
                name = details.get("device_name", gpu.name)
            except Exception:
                name = gpu.name
            print(f"    GPU {i}            : {name}")
    else:
        print("  [WARNING] No GPU detected — training will run on CPU (slow)")
    print()

    _print_config_summary(CONFIG)

    # ── Step 1: Build file index ──────────────────────────────────────────────
    print("=== Building dataset index ===")
    df = build_file_index(CONFIG["data_folders"])

    # ── Step 2: Stratified split ──────────────────────────────────────────────
    train_df, val_df, test_df = stratified_split(df, CONFIG)

    # ── Step 3: Build tf.data pipelines ──────────────────────────────────────
    print("=== Building tf.data pipelines ===")
    train_ds = build_dataset(train_df, training=True,  config=CONFIG)
    val_ds   = build_dataset(val_df,   training=False, config=CONFIG)
    test_ds  = build_dataset(test_df,  training=False, config=CONFIG)
    bs = CONFIG["batch_size"]
    print(f"  Train : {len(train_df):5d} samples  →  {(len(train_df) + bs - 1)//bs} batches/epoch")
    print(f"  Val   : {len(val_df):5d} samples  →  {(len(val_df)   + bs - 1)//bs} batches/epoch")
    print(f"  Test  : {len(test_df):5d} samples  →  {(len(test_df)  + bs - 1)//bs} batches/epoch")
    print()

    # ── Step 4: Build model ───────────────────────────────────────────────────
    print("=== Building model ===")
    model = build_model(CONFIG)
    compile_model(model, CONFIG)
    n_total  = model.count_params()
    n_train  = sum(int(tf.size(w)) for w in model.trainable_weights)
    n_fixed  = n_total - n_train
    print(f"  Total parameters   : {n_total:,}")
    print(f"  Trainable          : {n_train:,}  (backbone + heads)")
    print(f"  Non-trainable (BN) : {n_fixed:,}  (batch norm running stats)")
    print()

    # ── Step 5: Eval-only mode ────────────────────────────────────────────────
    if CONFIG["eval_only"]:
        ckpt = CONFIG["checkpoint_path"]
        if not os.path.exists(ckpt):
            print(f"[ERROR] eval_only=True but checkpoint not found at: {ckpt}")
            sys.exit(1)
        print(f"=== Loading checkpoint: {ckpt} ===")
        model = tf.keras.models.load_model(ckpt)
        compile_model(model, CONFIG)
        _print_final_metrics(model, test_ds, CONFIG)
        return

    # ── Step 6: Resume from checkpoint if one exists ─────────────────────────
    ckpt = CONFIG["checkpoint_path"]
    if CONFIG["force_restart"]:
        print("=== force_restart=True — starting training from scratch ===\n")
    elif os.path.exists(ckpt):
        print(f"=== Checkpoint found — resuming from: {ckpt} ===\n")
        model = tf.keras.models.load_model(ckpt)
        compile_model(model, CONFIG)

    # ── Step 7: Train ─────────────────────────────────────────────────────────
    print("=== Starting training ===\n")
    callbacks = get_callbacks(test_ds, CONFIG)

    # verbose=0 suppresses the default Keras progress bar.
    # EpochMonitor handles all per-epoch output.
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["epochs"],
        callbacks=callbacks,
        verbose=0,
    )

    # ── Step 8: Final evaluation ──────────────────────────────────────────────
    _print_final_metrics(model, test_ds, CONFIG)
    print(f"\n  Best model saved to : {CONFIG['checkpoint_path']}")
    print(f"  Training log at     : {CONFIG['csv_log_path']}")


if __name__ == "__main__":
    main()
