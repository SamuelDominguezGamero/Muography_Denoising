#!/bin/bash
#SBATCH --job-name=poca_lead_cube
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ---------------------------------------------------------------------------
# Ajusta estas líneas a tu cluster
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------



WORKDIR=/gpfs/users/dominguezs/Muography_Denoising
SCRIPTS=$WORKDIR/MuonGeneration/dataAnalysis

RAW=$WORKDIR/raw_g4_lead_50x100x20.root
CONFIG=$WORKDIR/00_lead_cube_50x100x20.json
CORRELATED=$WORKDIR/correlated_lead_50x100x20.root
POCA=$WORKDIR/poca_lead_50x100x20.root

echo "=================================================="
echo "[INFO] Starting pipeline on cluster..."
echo "[INFO] Node    : $(hostname)"
echo "[INFO] Date    : $(date)"
echo "[INFO] RAW     : $RAW"
echo "=================================================="

# STEP 1 — makeHLTuple
echo "[STEP 1] makeHLTuple..."
cd $SCRIPTS
python3 makeHLTuple.py \
    -i $RAW \
    -c $CONFIG \
    -o $CORRELATED
echo "[STEP 1] Done."

# STEP 2 — POCA
echo "[STEP 2] POCA..."
python3 POCA1.py \
    --input  $CORRELATED \
    --output $POCA \
    --Lpx 128 --Lpy 128 --Lpz 128 \
    --npx 128 --npy 128 --npz 128
echo "[STEP 2] Done."

# STEP 3 — Visualización
echo "[STEP 3] NOT Visualizing BC WE ARE IN CLUSTER"
#python3 visualizar_proyecciones.py \
#    --input $POCA
#echo "[STEP 3] Done."

echo "=================================================="
echo "[CORRECT] Pipeline finished at $(date)"
echo "=================================================="
