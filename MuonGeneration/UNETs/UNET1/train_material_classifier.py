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
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.config.optimizer.set_jit(False)

from prepare_data_for_ResNet import prepare_datasets, MATERIAL_LABELS, Z_MAX


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BASE = "/home/samuel/Work/Muography_Denoising/MuonGeneration/UNETs/UNET1"

CONFIG = {
    # ── Which data to use ─────────────────────────────────────────────────────
    # "run0", "run1", "run2", a list like ["run0","run2"], or "all"
    "runs": "run0",

    # ── Data split & loading ──────────────────────────────────────────────────
    "val_fraction":  0.10,
    "test_fraction": 0.10,
    "random_seed":   42,
    "batch_size":    32,

    # ── Dynamic subsampling (combat overfitting on simulated data) ──────────────
    # Set subsample_enabled=True to use only a fraction of training data per epoch,
    # with a DIFFERENT random fraction selected each epoch. This prevents overfitting
    # to redundant simulated data while exploiting full statistical richness.
    "subsample_enabled":  False,   # Set to True to enable
    "subsample_fraction": 0.30,    # Fraction per epoch: 0.3 = 30%

    # ── Output files ──────────────────────────────────────────────────────────
    "checkpoint_path": os.path.join(_BASE, "checkpoints", "best_model.keras"),
    "csv_log_path":    os.path.join(_BASE, "logs",        "training_log.csv"),
    "tb_log_dir":      os.path.join(_BASE, "logs",        "tensorboard"),

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
    # eval_only    : skip training, load checkpoint, evaluate on test set.
    # force_restart: ignore existing checkpoint and train from scratch.
    "eval_only":       False,
    "force_restart":   True,
    "use_tensorboard": True,
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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
    Returns dict with: accuracy, mat_loss, z_mae_cm, z_mse, total_loss.
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

    mat_acc  = float(np.mean(np.argmax(mat_probs, axis=1) == mat_true))
    mat_loss = float(-np.mean(np.log(np.clip(mat_probs, 1e-7, 1.0))
                              [np.arange(len(mat_true)), mat_true]))
    z_mae    = float(np.mean(np.abs(z_preds - z_true)))
    z_mse    = float(np.mean((z_preds - z_true) ** 2))

    return {
        "accuracy":   mat_acc,
        "mat_loss":   mat_loss,
        "z_mae_cm":   z_mae * Z_MAX,
        "z_mse":      z_mse,
        "total_loss": loss_weight_mat * mat_loss + loss_weight_z * z_mse,
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
        ])
        self._csv_file.flush()

    def on_train_end(self, logs=None):
        if self._csv_file:
            self._csv_file.close()
        if self._train_start:
            h, rem = divmod(int(time.time() - self._train_start), 3600)
            m, s   = divmod(rem, 60)
            print(f"\n  Total training time : {h:02d}h {m:02d}m {s:02d}s")


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


def get_callbacks(test_dataset, config):
    """Return training callbacks: EpochMonitor, Checkpoint, EarlyStopping, ReduceLR."""
    os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)

    callbacks = [
        EpochMonitor(
            test_dataset=test_dataset,
            csv_path=config["csv_log_path"],
            loss_weight_mat=config["loss_weight_material"],
            loss_weight_z=config["loss_weight_z"],
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config["checkpoint_path"],
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
        os.makedirs(config["tb_log_dir"], exist_ok=True)
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=config["tb_log_dir"], histogram_freq=0, update_freq="epoch"
        ))

    return callbacks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 74)
    print("  Material Classifier — Muon Scattering Tomography")
    print("=" * 74)
    print(f"  TensorFlow : {tf.__version__}")

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

    bs = CONFIG["batch_size"]
    for name, ds_len in [("Train", len(train_ds)), ("Val", len(val_ds)), ("Test", len(test_ds))]:
        print(f"  {name:5s}: {ds_len} batches/epoch")
    print()

    # ── Model ─────────────────────────────────────────────────────────────────
    print("=== Building model ===")
    model = build_model(CONFIG)
    compile_model(model, CONFIG)
    n_total = model.count_params()
    n_train = sum(int(tf.size(w)) for w in model.trainable_weights)
    print(f"  Parameters : {n_total:,} total  |  {n_train:,} trainable  |  {n_total - n_train:,} BN stats")
    print(f"  Classes    : {MATERIAL_LABELS}")
    print()

    # ── Config summary ─────────────────────────────────────────────────────────
    print("=== Training configuration ===")
    print(f"  Runs           : {CONFIG['runs']}")
    print(f"  Epochs         : {CONFIG['epochs']}  (patience={CONFIG['early_stopping_patience']})")
    print(f"  LR             : {CONFIG['learning_rate']}  →  min {CONFIG['min_lr']}")
    print(f"  Loss weights   : material={CONFIG['loss_weight_material']},  z={CONFIG['loss_weight_z']}")
    print(f"  Dropout / L2   : {CONFIG['dropout_rate']} / {CONFIG['l2_lambda']}")
    print(f"  Checkpoint     : {CONFIG['checkpoint_path']}")
    print()

    # ── Eval-only mode ─────────────────────────────────────────────────────────
    if CONFIG["eval_only"]:
        ckpt = CONFIG["checkpoint_path"]
        if not os.path.exists(ckpt):
            print(f"[ERROR] eval_only=True but no checkpoint at: {ckpt}")
            sys.exit(1)
        print(f"=== Loading checkpoint: {ckpt} ===")
        model = tf.keras.models.load_model(ckpt)
        compile_model(model, CONFIG)
        _print_final_metrics(model, test_ds)
        return

    # ── Resume or fresh start ─────────────────────────────────────────────────
    ckpt = CONFIG["checkpoint_path"]
    if CONFIG["force_restart"]:
        print("=== force_restart=True — training from scratch ===\n")
    elif os.path.exists(ckpt):
        print(f"=== Resuming from checkpoint: {ckpt} ===\n")
        model = tf.keras.models.load_model(ckpt)
        compile_model(model, CONFIG)

    # ── Train ──────────────────────────────────────────────────────────────────
    print("=== Starting training ===\n")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["epochs"],
        callbacks=get_callbacks(test_ds, CONFIG),
        verbose=0,   # EpochMonitor handles all output
    )

    # ── Final evaluation ───────────────────────────────────────────────────────
    _print_final_metrics(model, test_ds)
    print(f"\n  Best model : {CONFIG['checkpoint_path']}")
    print(f"  CSV log    : {CONFIG['csv_log_path']}")


def _print_final_metrics(model, test_ds):
    print("\n=== Final evaluation on test set ===")
    td = _eval_test_metrics(model, test_ds,
                            CONFIG["loss_weight_material"], CONFIG["loss_weight_z"])
    print(f"  Material accuracy  : {td['accuracy']:.4f}  ({td['accuracy']*100:.1f} %)")
    print(f"  Material loss      : {td['mat_loss']:.4f}")
    print(f"  Z MAE              : {td['z_mae_cm']:.2f} cm")
    print(f"  Z loss (MSE norm)  : {td['z_mse']:.4f}")
    print(f"  Total loss         : {td['total_loss']:.4f}")


if __name__ == "__main__":
    main()
