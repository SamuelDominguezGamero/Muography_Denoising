#!/bin/bash
# ---------------------------------------------------------------------------
# pipeline_post_Geant4.sh — Post-Geant4 pipeline for lead cube 50x100x20cm
# ---------------------------------------------------------------------------

set -e  # stop on first error

WORKDIR=~/Work/Muography_Denoising
SCRIPTS=$WORKDIR/MuonGeneration/dataAnalysis

RAW=$WORKDIR/raw_g4_lead_50x100x20.root
CONFIG=$WORKDIR/00_lead_cube_50x100x20.json
CORRELATED=$WORKDIR/correlated_lead_50x100x20.root
POCA=$WORKDIR/poca_lead_50x100x20.root

echo "=================================================="
echo "[INFO] Starting pipeline..."
echo "[INFO] RAW input  : $RAW"
echo "[INFO] Config     : $CONFIG"
echo "[INFO] Correlated : $CORRELATED"
echo "[INFO] POCA output: $POCA"
echo "=================================================="

# STEP 1 — makeHLTuple: correlate hits into tracks
echo ""
echo "[STEP 1] makeHLTuple — correlating tracks..."
python3 $SCRIPTS/makeHLTuple.py \
    -i $RAW \
    -c $CONFIG \
    -o $CORRELATED
echo "[STEP 1] Done."

# STEP 2 — POCA: compute Point Of Closest Approach
echo ""
echo "[STEP 2] POCA — computing scattering points..."
python3 $SCRIPTS/POCA1.py \
    --input  $CORRELATED \
    --output $POCA \
    --Lpx 128 --Lpy 128 --Lpz 128 \
    --npx 128 --npy 128 --npz 128
echo "[STEP 2] Done."

# STEP 3 — Visualize projections
echo ""
echo "[STEP 3] Visualizing POCA projections..."
python3 $SCRIPTS/visualizar_proyecciones.py \
    --input $POCA
echo "[STEP 3] Done."

echo ""
echo "=================================================="
echo "[CORRECT] Pipeline finished."
echo "=================================================="
