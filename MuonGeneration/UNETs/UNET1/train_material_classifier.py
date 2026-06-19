"""
train_material_classifier.py
============================
Multi-input, multi-task ResNet-18 for Muon Scattering Tomography.

Data workflow
-------------
  1. prepare_datasets() reads train/val/test from data_indices/*.csv
     (or generates & saves them if not found)
  2. CSVs contain: path, material_id, material_name, z_true, run, z_bin
  3. tf.data pipeline loads .npy files from paths in the CSVs

You can manually edit data_indices/*.csv to customize train/val/test splits.

Architecture
------------
  input_image [128,128,3]  ──>  ResNet-18 backbone  ──>  GAP  ──>  feat (128-d)
                                                                       │
                                                                 Dense(1)  ──>  output_z    (MSE)
                                                                       │          │
  input_flux  [1]  ────────────────────────────────────────>  Concat  <──────────┘
                                                                       │
                                                                 Dense(64, relu)
                                                                 Dropout
                                                                 Dense(n_class, softmax)  ──>  output_material  (SCCE)

Why z_pred is fed back into the classifier
------------------------------------------
  A thick low-Z object and a thin high-Z object can produce identical POCA maps.
  Providing the estimated thickness to the classification head lets the model
  reason "given this pattern AND this thickness, which material fits?",
  directly breaking the physical degeneracy.

Usage
-----
  python train_material_classifier.py          # train (or resume from checkpoint)
  # Set eval_only=True in CONFIG to skip training and only evaluate.
  tensorboard --logdir logs/tensorboard        # optional live monitoring
"""

import csv
import datetime
import json
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.config.optimizer.set_jit(False)

from prepare_data_for_ResNet import prepare_datasets, MATERIAL_LABELS, Z_MAX, DATA_INDICES_DIR
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, cohen_kappa_score,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE       = "/home/samuel/Work/Muography_Denoising/MuonGeneration"

# Root directory where one sub-folder per experiment will be created.
# Each folder is auto-named from key hyperparameters (or from run_name if set).
_MODELS_DIR = os.path.join(_BASE, "data", "models_ResNet")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # ── Which data to use ─────────────────────────────────────────────────────
    # "run0", "run1", "run2", a list like ["run0","run2"], or "all"
    "runs": ["run0", "run1"],

    # ── Experiment naming ─────────────────────────────────────────────────────
    # Leave empty → auto-generate name from key hyperparameters (recommended).
    # Example auto-name: run0__bs32__dr0.05__seed42
    # Set a fixed string to use that as the folder name instead.
    "run_name": "",

    # ── Data split & loading ──────────────────────────────────────────────────
    "val_fraction":  0.10,
    "test_fraction": 0.10,
    "random_seed":   42,
    "batch_size":    32,

    # ── Dynamic subsampling (combat overfitting on simulated data) ─────────────
    # Each epoch trains on a different random subsample of the training data,
    # destroying overfitting to specific instances while exploiting full statistics.
    "subsample_enabled":  False,   # Set to True to enable
    "subsample_fraction": 0.30,    # Fraction used per epoch: 0.3 = 30%

    # ── Training ──────────────────────────────────────────────────────────────
    "epochs":        200,
    "learning_rate": 1e-3,

    # Material loss (SCCE) is O(0.1–2); Z loss (MSE) is O(0.01–0.1).
    # weight_z = 0.3 keeps both heads contributing comparably.
    "loss_weight_material": 1.0,
    "loss_weight_z":        0.3,

    # ── Regularisation ────────────────────────────────────────────────────────
    "dropout_rate": 0.05,
    "l2_lambda":    1e-4,

    # ── Callbacks ─────────────────────────────────────────────────────────────
    "early_stopping_patience": 15,
    "reduce_lr_patience":       7,
    "reduce_lr_factor":         0.5,
    "min_lr":                   1e-6,

    # ── Flags ─────────────────────────────────────────────────────────────────
    # eval_only    : skip training, load best_model.keras, evaluate on test set.
    # force_restart: ignore existing checkpoint and always train from scratch.
    # use_tensorboard: write TensorBoard logs (tensorboard --logdir <output>/tensorboard).
    "eval_only":       False,
    "force_restart":   True,
    "use_tensorboard": True,
}


# ---------------------------------------------------------------------------
# Experiment folder helpers
# ---------------------------------------------------------------------------

def _build_output_dir():
    """Return the experiment output directory.

    Uses CONFIG["run_name"] if set; otherwise auto-generates a name that encodes
    the key hyperparameters so each distinct configuration gets its own folder.

    Auto-name format:  <runs>__bs<batch>__dr<dropout>__seed<seed>
    With subsampling:  <runs>__sub<pct>__bs<batch>__dr<dropout>__seed<seed>

    Examples:
        run0__bs32__dr0.05__seed42
        all__sub30__bs64__dr0.10__seed0
    """
    if CONFIG["run_name"]:
        return os.path.join(_MODELS_DIR, CONFIG["run_name"])

    runs = CONFIG["runs"]
    if runs == "all":
        run_tag = "all"
    elif isinstance(runs, list):
        run_tag = "_".join(runs)
    else:
        run_tag = runs

    sub_tag = ""
    if CONFIG["subsample_enabled"]:
        sub_tag = f"__sub{int(CONFIG['subsample_fraction'] * 100)}"

    name = (
        f"{run_tag}"
        f"{sub_tag}"
        f"__bs{CONFIG['batch_size']}"
        f"__dr{CONFIG['dropout_rate']}"
        f"__seed{CONFIG['random_seed']}"
    )
    return os.path.join(_MODELS_DIR, name)


def _save_run_config(output_dir, n_train, n_val, n_test):
    """Write all hyperparameters and dataset sizes to run_config.json.

    Called once at the start of training so every experiment folder is fully
    self-documenting.  Fields marked None are filled in by _update_run_config()
    at the end of training.
    """
    config = {
        # ── Identity ──────────────────────────────────────────────────────────
        "run_id":           os.path.basename(output_dir),
        "timestamp_start":  datetime.datetime.now().isoformat(timespec="seconds"),
        "timestamp_end":    None,  # filled after training
        "output_dir":       output_dir,

        # ── Data ──────────────────────────────────────────────────────────────
        "runs":             CONFIG["runs"] if isinstance(CONFIG["runs"], str) else list(CONFIG["runs"]),
        "n_train":          n_train,
        "n_val":            n_val,
        "n_test":           n_test,
        "val_fraction":     CONFIG["val_fraction"],
        "test_fraction":    CONFIG["test_fraction"],
        "random_seed":      CONFIG["random_seed"],

        # ── Dynamic subsampling ───────────────────────────────────────────────
        "subsample_enabled":  CONFIG["subsample_enabled"],
        "subsample_fraction": CONFIG["subsample_fraction"] if CONFIG["subsample_enabled"] else None,

        # ── Model ─────────────────────────────────────────────────────────────
        "n_classes":        len(MATERIAL_LABELS),
        "material_labels":  MATERIAL_LABELS,
        "z_max_cm":         Z_MAX,

        # ── Training hyperparameters ──────────────────────────────────────────
        "batch_size":             CONFIG["batch_size"],
        "learning_rate":          CONFIG["learning_rate"],
        "epochs_max":             CONFIG["epochs"],
        "loss_weight_material":   CONFIG["loss_weight_material"],
        "loss_weight_z":          CONFIG["loss_weight_z"],
        "dropout_rate":           CONFIG["dropout_rate"],
        "l2_lambda":              CONFIG["l2_lambda"],
        "early_stopping_patience": CONFIG["early_stopping_patience"],
        "reduce_lr_patience":     CONFIG["reduce_lr_patience"],
        "reduce_lr_factor":       CONFIG["reduce_lr_factor"],
        "min_lr":                 CONFIG["min_lr"],

        # ── Results (filled after training by _update_run_config) ────────────
        "epochs_trained":           None,
        "training_time_s":          None,
        "test_accuracy":            None,
        "test_f1_macro":            None,
        "test_precision_macro":     None,
        "test_recall_macro":        None,
        "test_auc_ovr":             None,
        "test_kappa":               None,
        "test_top2_accuracy":       None,
        "test_mat_loss":            None,
        "test_z_mae_cm":            None,
        "test_z_mse":               None,
        "test_total_loss":          None,
        "test_f1_per_class":        None,
        "test_precision_per_class": None,
        "test_recall_per_class":    None,
        "test_confusion_matrix":    None,
    }

    out_path = os.path.join(output_dir, "run_config.json")
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[CFG]   run_config.json  : {out_path}")


def _update_run_config(output_dir, metrics, elapsed_s, epochs_trained):
    """Update run_config.json with final test metrics and training time.

    Call once after training ends and the best model has been evaluated.
    """
    path = os.path.join(output_dir, "run_config.json")
    with open(path) as f:
        cfg = json.load(f)

    # auc_ovr can be NaN if a class is absent from the test split
    auc_val = metrics["auc_ovr"]
    auc_rounded = round(auc_val, 6) if np.isfinite(auc_val) else None

    cfg.update({
        "timestamp_end":            datetime.datetime.now().isoformat(timespec="seconds"),
        "epochs_trained":           epochs_trained,
        "training_time_s":          round(elapsed_s, 1),
        "test_accuracy":            round(metrics["accuracy"],        6),
        "test_f1_macro":            round(metrics["f1_macro"],        6),
        "test_precision_macro":     round(metrics["precision_macro"], 6),
        "test_recall_macro":        round(metrics["recall_macro"],    6),
        "test_auc_ovr":             auc_rounded,
        "test_kappa":               round(metrics["kappa"],           6),
        "test_top2_accuracy":       round(metrics["top2_accuracy"],   6),
        "test_mat_loss":            round(metrics["mat_loss"],        6),
        "test_z_mae_cm":            round(metrics["z_mae_cm"],        4),
        "test_z_mse":               round(metrics["z_mse"],          6),
        "test_total_loss":          round(metrics["total_loss"],      6),
        # Per-class dicts keyed by material name for easy reading
        "test_f1_per_class":        {MATERIAL_LABELS[i]: round(v, 6) for i, v in enumerate(metrics["f1_per_class"])},
        "test_precision_per_class": {MATERIAL_LABELS[i]: round(v, 6) for i, v in enumerate(metrics["precision_per_class"])},
        "test_recall_per_class":    {MATERIAL_LABELS[i]: round(v, 6) for i, v in enumerate(metrics["recall_per_class"])},
        "test_confusion_matrix":    metrics["confusion_matrix"],
    })

    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[CFG]   run_config.json updated with final metrics")


def _check_config_match(output_dir):
    """Validate that the current CONFIG matches the run_config.json in output_dir.

    Called before resuming an existing experiment so we catch accidental
    hyperparameter mismatches early — before overwriting a good checkpoint.

    Prints a warning for each mismatch.  Fields that do NOT affect reproducibility
    (eval_only, force_restart, use_tensorboard) are intentionally ignored.

    Returns True if everything matches (safe to resume), False otherwise.
    """
    cfg_path = os.path.join(output_dir, "run_config.json")
    if not os.path.exists(cfg_path):
        return True  # no config to compare — treat as first run

    with open(cfg_path) as f:
        saved = json.load(f)

    # Map of: config key in run_config.json → current CONFIG key
    checks = {
        "runs":                   CONFIG["runs"] if isinstance(CONFIG["runs"], str) else list(CONFIG["runs"]),
        "val_fraction":           CONFIG["val_fraction"],
        "test_fraction":          CONFIG["test_fraction"],
        "random_seed":            CONFIG["random_seed"],
        "batch_size":             CONFIG["batch_size"],
        "subsample_enabled":      CONFIG["subsample_enabled"],
        "dropout_rate":           CONFIG["dropout_rate"],
        "l2_lambda":              CONFIG["l2_lambda"],
        "loss_weight_material":   CONFIG["loss_weight_material"],
        "loss_weight_z":          CONFIG["loss_weight_z"],
    }

    mismatches = []
    for key, current_val in checks.items():
        saved_val = saved.get(key)
        if saved_val != current_val:
            mismatches.append((key, saved_val, current_val))

    if mismatches:
        print("[WARN]  Hyperparameter mismatches with existing run_config.json:")
        for key, old, new in mismatches:
            print(f"  {key:30s}  saved={old!r}  current={new!r}")
        print("[WARN]  Set force_restart=True to start a new experiment, "
              "or revert CONFIG to the saved values to resume safely.\n")
        return False

    return True

def _resnet_block(x, filters, stride=1, l2_lambda=1e-4):
    """
    Basic ResNet residual block.

      input ──► Conv(3×3) ──► BN ──► ReLU ──► Conv(3×3) ──► BN ──► Add ──► ReLU
        └──────────────── (identity or 1×1 projection) ──────────────────┘
    """
    reg      = tf.keras.regularizers.l2(l2_lambda)
    shortcut = x

    x = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding="same",
                               use_bias=False, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, 3, strides=1, padding="same",
                               use_bias=False, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if stride != 1 or int(shortcut.shape[-1]) != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding="same",
                                          use_bias=False, kernel_regularizer=reg)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x


def build_model(config):
    """
    Build the ResNet-18 multi-task model.

    Spatial resolution through the backbone:
        input 128×128  →  stem 64×64  →  pool 32×32
        stage1 32×32 (32 filters)
        stage2 16×16 (64 filters)
        stage3  8×8  (128 filters)
        GAP  →  128-d vector
    """
    reg     = tf.keras.regularizers.l2(config["l2_lambda"])
    n_class = len(MATERIAL_LABELS)

    input_image = tf.keras.Input(shape=(128, 128, 3), name="input_image")
    input_flux  = tf.keras.Input(shape=(1,),          name="input_flux")

    # Backbone
    x = tf.keras.layers.Conv2D(32, 7, strides=2, padding="same",
                               use_bias=False, kernel_regularizer=reg,
                               name="stem_conv")(input_image)       # 64×64
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same",
                                     name="stem_pool")(x)           # 32×32

    x = _resnet_block(x, 32,  stride=1, l2_lambda=config["l2_lambda"])
    x = _resnet_block(x, 32,  stride=1, l2_lambda=config["l2_lambda"])  # 32×32

    x = _resnet_block(x, 64,  stride=2, l2_lambda=config["l2_lambda"])  # 16×16
    x = _resnet_block(x, 64,  stride=1, l2_lambda=config["l2_lambda"])

    x = _resnet_block(x, 128, stride=2, l2_lambda=config["l2_lambda"])  #  8×8
    x = _resnet_block(x, 128, stride=1, l2_lambda=config["l2_lambda"])

    feat = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)     # 128-d

    # Auxiliary Z-regression head (also injected into the classifier)
    z_pred = tf.keras.layers.Dense(1, activation="linear",
                                   kernel_regularizer=reg,
                                   name="output_z")(feat)

    # Classification head: visual features + z estimate + flux
    fused = tf.keras.layers.Concatenate(name="fusion")([feat, z_pred, input_flux])
    x = tf.keras.layers.Dense(64, activation="relu",
                              kernel_regularizer=reg, name="fc1")(fused)
    x = tf.keras.layers.Dropout(config["dropout_rate"], name="dropout")(x)
    output_material = tf.keras.layers.Dense(n_class, activation="softmax",
                                            name="output_material")(x)

    return tf.keras.Model(
        inputs=[input_image, input_flux],
        outputs=[output_material, z_pred],
        name="MaterialClassifier_ResNet18",
    )


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _eval_test_metrics(model, test_dataset, loss_weight_mat, loss_weight_z):
    """
    Manual forward pass over the test set.

    Bypasses model.metrics_names (unreliable across TF versions).
    Returns a dict with base metrics (accuracy, mat_loss, z_mae_cm, z_mse,
    total_loss) plus extended classification metrics: f1_macro,
    precision_macro, recall_macro, auc_ovr (One-vs-Rest), kappa (Cohen's),
    top2_accuracy, and per-class versions of precision / recall / F1,
    plus the full confusion matrix.
    """
    mat_probs_list, z_pred_list = [], []
    mat_true_list,  z_true_list = [], []

    for inputs, targets in test_dataset:
        out = model(inputs, training=False)
        mat_probs_list.append(out[0].numpy())
        z_pred_list.append(out[1].numpy())
        mat_true_list.append(targets["output_material"].numpy())
        z_true_list.append(targets["output_z"].numpy())

    mat_probs = np.concatenate(mat_probs_list, axis=0)
    z_preds   = np.concatenate(z_pred_list,   axis=0).flatten()
    mat_true  = np.concatenate(mat_true_list, axis=0).astype(np.int32)
    z_true    = np.concatenate(z_true_list,   axis=0).flatten()

    mat_pred = np.argmax(mat_probs, axis=1)

    # ── Base metrics ──────────────────────────────────────────────────────────
    mat_acc  = float(np.mean(mat_pred == mat_true))
    mat_loss = float(-np.mean(np.log(np.clip(mat_probs, 1e-7, 1.0))
                              [np.arange(len(mat_true)), mat_true]))
    z_mae    = float(np.mean(np.abs(z_preds - z_true)))
    z_mse    = float(np.mean((z_preds - z_true) ** 2))

    # ── Extended classification metrics (sklearn) ─────────────────────────────
    f1_per_class        = f1_score(mat_true, mat_pred, average=None,    zero_division=0).tolist()
    precision_per_class = precision_score(mat_true, mat_pred, average=None, zero_division=0).tolist()
    recall_per_class    = recall_score(mat_true, mat_pred, average=None, zero_division=0).tolist()

    f1_macro        = float(f1_score(mat_true, mat_pred,        average="macro", zero_division=0))
    precision_macro = float(precision_score(mat_true, mat_pred, average="macro", zero_division=0))
    recall_macro    = float(recall_score(mat_true, mat_pred,    average="macro", zero_division=0))
    kappa           = float(cohen_kappa_score(mat_true, mat_pred))

    # AUC-OvR: uses probability scores directly; may be undefined if a class
    # has no positive samples in this split (e.g., a very small test set).
    try:
        auc_ovr = float(roc_auc_score(mat_true, mat_probs, multi_class="ovr", average="macro"))
    except ValueError:
        auc_ovr = float("nan")

    # Top-2 accuracy: true class ranks among the top-2 predictions.
    # Useful to quantify near-misses in a 5-class problem.
    top2_indices  = np.argsort(mat_probs, axis=1)[:, -2:]
    top2_accuracy = float(np.mean([mat_true[i] in top2_indices[i] for i in range(len(mat_true))]))

    return {
        # ── Regression & loss ─────────────────────────────────────────────────
        "accuracy":            mat_acc,
        "mat_loss":            mat_loss,
        "z_mae_cm":            z_mae * Z_MAX,
        "z_mse":               z_mse,
        "total_loss":          loss_weight_mat * mat_loss + loss_weight_z * z_mse,
        # ── Macro-averaged classification metrics ─────────────────────────────
        "f1_macro":            f1_macro,
        "precision_macro":     precision_macro,
        "recall_macro":        recall_macro,
        "auc_ovr":             auc_ovr,
        "kappa":               kappa,
        "top2_accuracy":       top2_accuracy,
        # ── Per-class and confusion matrix ────────────────────────────────────
        "f1_per_class":        f1_per_class,
        "precision_per_class": precision_per_class,
        "recall_per_class":    recall_per_class,
        "confusion_matrix":    confusion_matrix(mat_true, mat_pred).tolist(),
    }


# ---------------------------------------------------------------------------
# Custom callback
# ---------------------------------------------------------------------------

class EpochMonitor(tf.keras.callbacks.Callback):
    """
    Prints a formatted metrics table each epoch and writes a CSV log.

    Evaluates train (from Keras logs), val (from Keras logs), and test
    (manual forward pass) for each of:
        material accuracy, material loss, Z MAE (cm), Z MSE, total loss.

    The test set is never used for EarlyStopping or ModelCheckpoint.
    """

    def __init__(self, test_dataset, csv_path, loss_weight_mat, loss_weight_z):
        super().__init__()
        self.test_dataset    = test_dataset
        self.csv_path        = csv_path
        self.loss_weight_mat = loss_weight_mat
        self.loss_weight_z   = loss_weight_z
        self._csv_file       = None
        self._csv_writer     = None
        self._epoch_start    = None
        self._train_start    = None

    def on_train_begin(self, logs=None):
        self._train_start = time.time()
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
            "test_f1_macro",  "test_precision_macro", "test_recall_macro",
            "test_auc_ovr",   "test_kappa",
        ])
        self._csv_file.flush()

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        td   = _eval_test_metrics(
            self.model, self.test_dataset,
            self.loss_weight_mat, self.loss_weight_z,
        )

        def g(key):
            return float(logs.get(key, 0.0))

        train_acc    = g("output_material_accuracy")
        val_acc      = g("val_output_material_accuracy")
        train_mloss  = g("output_material_loss")
        val_mloss    = g("val_output_material_loss")
        train_z_mae  = g("output_z_mae")  * Z_MAX
        val_z_mae    = g("val_output_z_mae") * Z_MAX
        train_z_mse  = g("output_z_loss")
        val_z_mse    = g("val_output_z_loss")
        train_loss   = g("loss")
        val_loss     = g("val_loss")
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        n_epochs = self.params.get("epochs", CONFIG["epochs"])
        bar      = "─" * 74
        elapsed  = time.time() - self._epoch_start if self._epoch_start else 0.0

        print(f"\n{bar}")
        print(f"  Epoch {epoch+1:03d}/{n_epochs:03d}"
              f"                     TRAIN        VAL          TEST")
        print(bar)
        print(f"  Material accuracy  :       {train_acc:7.4f}      {val_acc:7.4f}      {td['accuracy']:7.4f}")
        print(f"  F1-macro (test)    :       {'':7}      {'':7}      {td['f1_macro']:7.4f}")
        print(f"  Material loss      :       {train_mloss:7.4f}      {val_mloss:7.4f}      {td['mat_loss']:7.4f}")
        print(f"  Z MAE (cm)         :       {train_z_mae:7.2f}      {val_z_mae:7.2f}      {td['z_mae_cm']:7.2f}")
        print(f"  Z loss (MSE norm)  :       {train_z_mse:7.4f}      {val_z_mse:7.4f}      {td['z_mse']:7.4f}")
        print(f"  Total loss         :       {train_loss:7.4f}      {val_loss:7.4f}      {td['total_loss']:7.4f}")
        print(f"  Learning rate      :       {lr:.2e}")
        print(f"  Epoch time         :       {elapsed:.1f}s")
        print(bar)
        sys.stdout.flush()

        self._csv_writer.writerow([
            epoch + 1, f"{lr:.2e}",
            f"{train_acc:.6f}",   f"{val_acc:.6f}",          f"{td['accuracy']:.6f}",
            f"{train_mloss:.6f}", f"{val_mloss:.6f}",         f"{td['mat_loss']:.6f}",
            f"{train_z_mae:.4f}", f"{val_z_mae:.4f}",         f"{td['z_mae_cm']:.4f}",
            f"{train_z_mse:.6f}", f"{val_z_mse:.6f}",         f"{td['z_mse']:.6f}",
            f"{train_loss:.6f}",  f"{val_loss:.6f}",          f"{td['total_loss']:.6f}",
            f"{td['f1_macro']:.6f}",       f"{td['precision_macro']:.6f}",
            f"{td['recall_macro']:.6f}",   f"{td['auc_ovr']:.6f}",
            f"{td['kappa']:.6f}",
        ])
        self._csv_file.flush()

    def on_train_end(self, logs=None):
        if self._csv_file:
            self._csv_file.close()
        if self._train_start:
            h, rem = divmod(int(time.time() - self._train_start), 3600)
            m, s   = divmod(rem, 60)
            print(f"\n  Total training time : {h:02d}h {m:02d}m {s:02d}s")


class LastCheckpoint(tf.keras.callbacks.Callback):
    """Save the last-epoch weights and a training_meta.json after every epoch.

    This enables true resume from the exact epoch where training stopped,
    as opposed to best_model.keras which only tracks the best val score.

    Files written to <output_dir>/checkpoints/:
        last_checkpoint.keras    — overwritten every epoch
        training_meta.json       — {last_epoch, best_val_acc, last_val_acc}
    """

    def __init__(self, ckpt_dir):
        super().__init__()
        self.ckpt_dir          = ckpt_dir
        self.last_ckpt_path    = os.path.join(ckpt_dir, "last_checkpoint.keras")
        self.meta_path         = os.path.join(ckpt_dir, "training_meta.json")
        self.best_val_acc      = 0.0

        # Load previous best from meta if it exists (used for display only)
        if os.path.exists(self.meta_path):
            with open(self.meta_path) as f:
                self.best_val_acc = json.load(f).get("best_val_acc", 0.0)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Overwrite last checkpoint every epoch
        self.model.save(self.last_ckpt_path)

        val_acc = float(logs.get("val_output_material_accuracy", 0.0))
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

        meta = {
            "last_epoch":    epoch + 1,
            "best_val_acc":  round(self.best_val_acc, 6),
            "last_val_acc":  round(val_acc, 6),
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Compile & callbacks
# ---------------------------------------------------------------------------

def compile_model(model, config):
    """Attach optimiser, losses, and metrics."""
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
            "output_material": [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            "output_z":        [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        },
        # Disable XLA JIT — causes CUDNN_STATUS_EXECUTION_FAILED on RTX 5060
        # (Blackwell / Compute Capability 12.0) with Keras 3 auto JIT.
        jit_compile=False,
    )


def get_callbacks(test_dataset, output_dir, config):
    """Return training callbacks: EpochMonitor, Checkpoint, EarlyStopping, ReduceLR.

    All output paths are derived from output_dir so every experiment folder is
    self-contained.
    """
    ckpt_dir        = os.path.join(output_dir, "checkpoints")
    best_model_path = os.path.join(ckpt_dir,   "best_model.keras")
    csv_log_path    = os.path.join(output_dir,  "training_log.csv")

    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        EpochMonitor(
            test_dataset=test_dataset,
            csv_path=csv_log_path,
            loss_weight_mat=config["loss_weight_material"],
            loss_weight_z=config["loss_weight_z"],
        ),
        LastCheckpoint(ckpt_dir),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_output_material_accuracy",
            mode="max",
            save_best_only=True,
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
        tb_log_dir = os.path.join(output_dir, "tensorboard")
        os.makedirs(tb_log_dir, exist_ok=True)
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=tb_log_dir, histogram_freq=0, update_freq="epoch"
        ))

    return callbacks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Output directory ───────────────────────────────────────────────────────
    output_dir      = _build_output_dir()
    ckpt_dir        = os.path.join(output_dir, "checkpoints")
    best_model_path = os.path.join(ckpt_dir, "best_model.keras")
    summary_path    = os.path.join(output_dir, "model_summary.txt")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir,   exist_ok=True)

    print("=" * 74)
    print("  Material Classifier — Muon Scattering Tomography")
    print(f"  Output : {output_dir}")
    print("=" * 74)
    print(f"  TensorFlow : {tf.__version__}")

    # ── GPU setup ─────────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  GPUs       : {len(gpus)}  (memory growth enabled)")
        for i, gpu in enumerate(gpus):
            try:
                name = tf.config.experimental.get_device_details(gpu).get("device_name", gpu.name)
            except Exception:
                name = gpu.name
            print(f"    GPU {i}    : {name}")
    else:
        print("  [WARNING] No GPU — training on CPU (slow)")
    print()

    # ── Datasets ──────────────────────────────────────────────────────────────
    data_config = {k: CONFIG[k] for k in
                   ("val_fraction", "test_fraction", "random_seed", "batch_size",
                    "subsample_enabled", "subsample_fraction")}
    train_ds, val_ds, test_ds = prepare_datasets(CONFIG["runs"], config=data_config)

    # Read split sizes from the CSVs that prepare_datasets() saves.
    n_train = len(pd.read_csv(os.path.join(DATA_INDICES_DIR, "train_index.csv")))
    n_val   = len(pd.read_csv(os.path.join(DATA_INDICES_DIR, "val_index.csv")))
    n_test  = len(pd.read_csv(os.path.join(DATA_INDICES_DIR, "test_index.csv")))

    print(f"  Train: {n_train} samples  |  "
          f"Val: {n_val}  |  "
          f"Test: {n_test}  |  "
          f"Batch size: {CONFIG['batch_size']}")
    if CONFIG["subsample_enabled"]:
        print(f"  Subsampling: {int(CONFIG['subsample_fraction']*100)}% per epoch (dynamic)")
    print()

    # ── Model ─────────────────────────────────────────────────────────────────
    print("=== Building model ===")
    model = build_model(CONFIG)
    compile_model(model, CONFIG)

    n_total     = model.count_params()
    n_trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    print(f"  Parameters : {n_total:,} total  |  {n_trainable:,} trainable  |  {n_total - n_trainable:,} BN stats")
    print(f"  Classes    : {MATERIAL_LABELS}")

    with open(summary_path, "w") as f:
        model.summary(line_length=100, print_fn=lambda s: f.write(s + "\n"))
    print(f"  Summary    : {summary_path}")
    print()

    # ── Save run config at start ───────────────────────────────────────────────
    _save_run_config(output_dir, n_train, n_val, n_test)
    print()

    # ── Config summary ─────────────────────────────────────────────────────────
    print("=== Training configuration ===")
    print(f"  Runs           : {CONFIG['runs']}")
    print(f"  Epochs         : {CONFIG['epochs']}  (patience={CONFIG['early_stopping_patience']})")
    print(f"  LR             : {CONFIG['learning_rate']}  →  min {CONFIG['min_lr']}")
    print(f"  Loss weights   : material={CONFIG['loss_weight_material']},  z={CONFIG['loss_weight_z']}")
    print(f"  Dropout / L2   : {CONFIG['dropout_rate']} / {CONFIG['l2_lambda']}")
    print(f"  Checkpoint     : {best_model_path}")
    print()

    # ── Eval-only mode ─────────────────────────────────────────────────────────
    if CONFIG["eval_only"]:
        if not os.path.exists(best_model_path):
            print(f"[ERROR] eval_only=True but no checkpoint at:\n  {best_model_path}")
            sys.exit(1)
        print(f"=== Loading checkpoint ===")
        print(f"  {best_model_path}")
        model = tf.keras.models.load_model(best_model_path)
        compile_model(model, CONFIG)
        metrics = _eval_test_metrics(model, test_ds,
                                     CONFIG["loss_weight_material"], CONFIG["loss_weight_z"])
        _print_final_metrics(metrics)
        return

    # ── Resume or fresh start ─────────────────────────────────────────────────
    meta_path      = os.path.join(ckpt_dir, "training_meta.json")
    last_ckpt_path = os.path.join(ckpt_dir, "last_checkpoint.keras")
    initial_epoch  = 0

    print("\n" + "#" * 74)
    if CONFIG["force_restart"]:
        print("###  TRAINING MODE : FROM SCRATCH (force_restart=True)")
        print("###  Any previous checkpoint will be IGNORED and OVERWRITTEN.")
        # Remove meta so the next run starts clean
        if os.path.exists(meta_path):
            os.remove(meta_path)

    elif os.path.exists(meta_path):
        # Validate CONFIG against the saved run_config.json before loading weights
        config_ok = _check_config_match(output_dir)

        with open(meta_path) as f:
            meta = json.load(f)
        initial_epoch = meta.get("last_epoch", 0)

        if os.path.exists(last_ckpt_path) and config_ok:
            model = tf.keras.models.load_model(last_ckpt_path)
            compile_model(model, CONFIG)
            print(f"###  TRAINING MODE : RESUMING from epoch {initial_epoch + 1}")
            print(f"###  Checkpoint    : {last_ckpt_path}")
            print(f"###  Best val acc  : {meta.get('best_val_acc', 'n/a')}")
        else:
            print("###  TRAINING MODE : FROM SCRATCH")
            if not config_ok:
                print("###  (hyperparameter mismatch — not safe to resume)")
            else:
                print("###  (meta found but no checkpoint file — starting fresh)")
            initial_epoch = 0

    else:
        print("###  TRAINING MODE : FROM SCRATCH (no previous run detected)")
    print("#" * 74)

    # ── Train ──────────────────────────────────────────────────────────────────
    print("\n=== Starting training ===\n")
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["epochs"],
        initial_epoch=initial_epoch,
        callbacks=get_callbacks(test_ds, output_dir, CONFIG),
        verbose=0,   # EpochMonitor handles all output
    )
    elapsed = time.time() - t0
    epochs_trained = len(history.history.get("loss", []))

    h, rem = divmod(int(elapsed), 3600)
    m, s   = divmod(rem, 60)
    print(f"\n  Training time : {h:02d}h {m:02d}m {s:02d}s  ({epochs_trained} epochs)")

    # ── Load best model and evaluate ───────────────────────────────────────────
    if os.path.exists(best_model_path):
        print(f"\n=== Final evaluation on test set (best_model.keras) ===")
        best_model = tf.keras.models.load_model(best_model_path)
        compile_model(best_model, CONFIG)
        metrics = _eval_test_metrics(best_model, test_ds,
                                     CONFIG["loss_weight_material"], CONFIG["loss_weight_z"])
    else:
        # EarlyStopping with restore_best_weights left best weights in memory
        print(f"\n=== Final evaluation on test set (restored best weights) ===")
        metrics = _eval_test_metrics(model, test_ds,
                                     CONFIG["loss_weight_material"], CONFIG["loss_weight_z"])

    # ── Update config.json with results ───────────────────────────────────────
    _update_run_config(output_dir, metrics, elapsed, epochs_trained)

    _print_final_metrics(metrics)
    print()
    print(f"[SAVE]  Output folder    : {output_dir}")
    print(f"[SAVE]  Best model       : {best_model_path}")
    print(f"[SAVE]  run_config.json  : {os.path.join(output_dir, 'run_config.json')}")
    print(f"[SAVE]  Training log     : {os.path.join(output_dir, 'training_log.csv')}")
    print(f"[SAVE]  Model summary    : {summary_path}")


def _print_final_metrics(metrics):
    """Print a full classification report: summary + per-class table + confusion matrix."""
    bar = "─" * 74

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{bar}")
    print(f"  {'Metric':<28} {'Value':>10}")
    print(bar)
    print(f"  {'Accuracy':<28} {metrics['accuracy']:>10.4f}  ({metrics['accuracy']*100:.1f} %)")
    print(f"  {'F1-score (macro)':<28} {metrics['f1_macro']:>10.4f}")
    print(f"  {'Precision (macro)':<28} {metrics['precision_macro']:>10.4f}")
    print(f"  {'Recall (macro)':<28} {metrics['recall_macro']:>10.4f}")
    auc_str = f"{metrics['auc_ovr']:>10.4f}" if np.isfinite(metrics['auc_ovr']) else "       n/a"
    print(f"  {'AUC-OvR (macro)':<28} {auc_str}")
    print(f"  {'Cohen\'s Kappa':<28} {metrics['kappa']:>10.4f}")
    print(f"  {'Top-2 accuracy':<28} {metrics['top2_accuracy']:>10.4f}")
    print(f"  {'Material loss (SCCE)':<28} {metrics['mat_loss']:>10.4f}")
    print(f"  {'Z MAE':<28} {metrics['z_mae_cm']:>10.2f} cm")
    print(f"  {'Z loss (MSE norm)':<28} {metrics['z_mse']:>10.4f}")
    print(f"  {'Total loss':<28} {metrics['total_loss']:>10.4f}")
    print(bar)

    # ── Per-class metrics ─────────────────────────────────────────────────────
    print(f"\n  {'Class':<24} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'─'*24} {'─'*10} {'─'*8} {'─'*8}")
    for i, label in enumerate(MATERIAL_LABELS):
        print(f"  {label:<24} {metrics['precision_per_class'][i]:>10.4f}"
              f" {metrics['recall_per_class'][i]:>8.4f}"
              f" {metrics['f1_per_class'][i]:>8.4f}")
    print()

    # ── Confusion matrix (rows = true label, cols = predicted) ────────────────
    # Use short abbreviations to keep columns aligned
    abbr = ["al-si", "fe-stl", "lead", "uranium", "water"]
    if len(abbr) != len(MATERIAL_LABELS):           # fallback if labels change
        abbr = [lbl[:6] for lbl in MATERIAL_LABELS]
    col_w = 8
    print(f"  Confusion matrix (rows = true, cols = predicted):")
    header = " " * 26 + "".join(f"{a:>{col_w}}" for a in abbr)
    print(f"  {header}")
    for label, row in zip(MATERIAL_LABELS, metrics["confusion_matrix"]):
        row_str = "".join(f"{v:>{col_w}}" for v in row)
        print(f"  {label:<24}  {row_str}")
    print()


if __name__ == "__main__":
    main()
