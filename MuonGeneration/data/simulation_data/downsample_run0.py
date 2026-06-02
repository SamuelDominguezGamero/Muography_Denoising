"""
downsample_run0.py
------------------
Applies Binomial thinning (Poisson thinning) to every .npy file in
run0_POCA_projections_abs and writes the result to
run0_POCA_projections_abs_downsampled.

Physical basis: if each voxel count X ~ Poisson(lambda), then
  Y = Binomial(X, p) ~ Poisson(p * lambda)
This is statistically equivalent to having run the simulation with a
fraction p of the original muon flux. It is NOT a simple rescaling —
the Poisson noise structure is correctly preserved.

Chosen p = 0.14, which maps the mean flux of run0 (~121 700 counts)
down to ~17 000 counts, matching the flux distribution of run1 and run2.

Usage:
    python downsample_run0.py
"""

import os
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(BASE, "run0_POCA_projections_abs")
DST  = os.path.join(BASE, "run0_POCA_projections_abs_downsampled")

# ── Thinning parameter ───────────────────────────────────────────────────────
# Derived from: mean(run1 flux, run2 flux) / mean(run0 flux) ≈ 0.14
P = 0.14

RANDOM_SEED = 42

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DST, exist_ok=True)

    files = sorted(f for f in os.listdir(SRC) if f.endswith(".npy"))
    n_total = len(files)
    print(f"Source folder : {SRC}")
    print(f"Output folder : {DST}")
    print(f"Files to process : {n_total}")
    print(f"Thinning factor p : {P}")
    print()

    rng = np.random.default_rng(RANDOM_SEED)

    for i, fname in enumerate(files):
        src_path = os.path.join(SRC, fname)
        dst_path = os.path.join(DST, fname)

        # Load original array — shape [128, 128, 3], likely float32
        arr = np.load(src_path)

        # Convert to int32 for binomial sampling (counts must be integers)
        arr_int = np.round(arr).astype(np.int32)

        # Apply Binomial thinning independently to each element
        arr_thinned = rng.binomial(arr_int, P).astype(np.float32)

        # Save with identical filename so the rest of the pipeline is unchanged
        np.save(dst_path, arr_thinned)

        # Progress report every 200 files
        if (i + 1) % 200 == 0 or (i + 1) == n_total:
            flux_before = float(arr[:, :, 0].sum())
            flux_after  = float(arr_thinned[:, :, 0].sum())
            print(f"  [{i+1:4d}/{n_total}]  {fname[:60]}")
            print(f"           flux before={flux_before:.0f}  after={flux_after:.0f}  ratio={flux_after/flux_before:.3f}")

    print(f"\nDone. {n_total} files written to {DST}")


if __name__ == "__main__":
    main()
