# Data Indices — CSV-based train/val/test Splits

## Overview

When you run `train_material_classifier.py`, the system will:

1. **First time:** Scan your data folders (`run0`, `run1`, `run2`), parse labels from filenames, and generate stratified splits
2. **Save:** Write indices to CSV files under `data_indices/` for reproducibility
3. **Subsequent runs:** Read from existing CSVs (fast, reproducible)
4. **You can edit:** Modify the CSV files manually if you want to customize splits

---

## Directory Structure

```
UNET1/
  data_indices/
    ├─ metadata.json       ← timestamp, seed, runs, z_bins used
    ├─ train_index.csv     ← training set: one sample per row
    ├─ val_index.csv       ← validation set
    └─ test_index.csv      ← test set
```

---

## CSV Format

Each CSV has these columns:

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `path` | str | `/data/run0_definitive_words/file.npy` | Absolute path to .npy file |
| `material_id` | int | 0 | Class index (alphabetically sorted: aluminium-silicon, iron-steel, lead, uranium, water) |
| `material_name` | str | `lead` | Human-readable material name |
| `z_true` | float | 20.5 | Z-thickness in cm (from filename) |
| `run` | str | `run0` | Which data run: run0, run1, or run2 |
| `z_bin` | int | 1 | Thickness bin: 0=thin [0-15), 1=med [15-30), 2=thick [30-60] |

---

## Stratification (2D)

Splits are stratified on **both material AND thickness**:

- Every material (aluminium-silicon, iron-steel, lead, uranium, water) is proportionally split
- Within each material, samples are balanced across thickness ranges:
  - **thin**: 0–15 cm
  - **med**: 15–30 cm  
  - **thick**: 30–60 cm

This ensures train/val/test have identical distributions of both material and thickness.

**Example:**
```
Material: lead, thin [0-15)    →  Train: 48,  Val: 6,  Test: 6
Material: lead, med [15-30)    →  Train: 40,  Val: 5,  Test: 5
Material: lead, thick [30-60]  →  Train: 32,  Val: 4,  Test: 4
```

---

## Workflow

### First Run
```bash
cd MuonGeneration/UNETs/UNET1
python train_material_classifier.py
```

**Output:**
```
=== Building dataset index ===
  run0 (run0_definitive_words)  →  1234 valid, 5 skipped
  run1 (run1_definitive_forms)  →  987 valid, 2 skipped
  run2 (run2_definitive_blocks) →  876 valid, 3 skipped

  Total    : 3097 samples  |  Z range: 5–60 cm

=== Stratifying by (material × z_thickness) ===
=== Dataset splits ===
  Train: 2478 samples  |  {...}
  Val:    310 samples   |  {...}
  Test:   309 samples   |  {...}

  Indices saved to: /path/to/data_indices/
  (You can manually edit the CSVs to customize splits)
```

### Subsequent Runs
```bash
python train_material_classifier.py
```

**Output:**
```
=== Loading dataset indices from CSV ===
  Loaded train/val/test splits from: /path/to/data_indices/
=== Dataset splits ===
  Train: 2478 samples  |  {...}
  Val:    310 samples   |  {...}
  Test:   309 samples   |  {...}
```

(Much faster — no scanning/parsing files)

### Regenerate (Force Rebuild)

```bash
python train_material_classifier.py
```

Edit `train_material_classifier.py` and set `force_rebuild=True` in the call:

```python
data_config = {k: CONFIG[k] for k in
               ("val_fraction", "test_fraction", "random_seed", "batch_size")}
train_ds, val_ds, test_ds = prepare_datasets(
    CONFIG["runs"], 
    config=data_config,
    force_rebuild=True  # ← Add this
)
```

---

## Manual Editing

You can safely edit the CSV files to customize splits:

1. **Add/remove rows:** Add or delete lines from any CSV
2. **Move samples between splits:** Cut from `train_index.csv`, paste into `val_index.csv`
3. **No parsing changes needed:** Just keep the columns in the same order

**After editing:**
- Next run will use your modified splits (matching seed + runs → CSVs already exist)
- If you want fresh indices, delete the CSVs or use `force_rebuild=True`

---

## Metadata

`metadata.json` tracks:
```json
{
  "timestamp": "2026-06-10T15:30:45.123456",
  "runs": ["run0", "run1", "run2"],
  "random_seed": 42,
  "z_bins": [0, 15, 30, 60],
  "z_max": 60.0
}
```

This helps you remember when indices were generated and under what settings.

---

## Tips

- **Reproducibility:** Same `random_seed` + same `runs` = same splits (CSVs won't change)
- **Experiment tracking:** Commit your modified CSVs to Git to version-control your train/val/test splits
- **Sanity checks:** Open the CSVs in a spreadsheet editor (LibreOffice Calc, Excel) to visually inspect distributions
- **Reset:** Delete `data_indices/` folder to start fresh
