#!/bin/bash
#SBATCH --job-name=muon_seed358885362
#SBATCH --output=/gpfs/projects/cms/dominguezs/data/logs/log__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2_seed358885362.out
#SBATCH --error=/gpfs/projects/cms/dominguezs/data/logs/log__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2_seed358885362.err
#SBATCH --partition=wncompute_ifca
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source /gpfs/users/dominguezs/Muography_Denoising/setup.sh

echo "[INFO] Job started: _Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2 | seed=358885362"

# --- 1st: Geant4 Monte Carlo simulation ---
echo "[INFO] Running Geant4 simulation..."
cd /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration-build/
/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration-build/Generator \
    --input  /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/geometric_configurations_json/_Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2.json \
    --output /gpfs/projects/cms/dominguezs/data/0_raw_geant4/Out__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2_seed358885362.root \
    --number 100000 \
    --seed   358885362
if [ $? -ne 0 ]; then echo "[ERROR] Geant4 failed. Aborting."; exit 1; fi
echo "[CORRECT] Geant4 done."

# --- 2nd: Track correlation ---
echo "[INFO] Running makeHLTuple..."
python3 -u /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/dataAnalysis/makeHLTuple.py \
    --input  /gpfs/projects/cms/dominguezs/data/0_raw_geant4/Out__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2_seed358885362.root \
    --conf   /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/geometric_configurations_json/_Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2.json \
    --output /gpfs/projects/cms/dominguezs/data/1_post_makeHLT/Pre__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2_seed358885362.root
if [ $? -ne 0 ]; then echo "[ERROR] makeHLTuple failed. Aborting."; exit 1; fi
echo "[CORRECT] makeHLTuple done."

# eliminate intermediate files to save space
rm /gpfs/projects/cms/dominguezs/data/0_raw_geant4/Out__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2_seed358885362.root
echo "[INFO] Raw file removed to save space: /gpfs/projects/cms/dominguezs/data/0_raw_geant4/Out__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2_seed358885362.root"

# --- 3rd: POCA reconstruction ---
echo "[INFO] Running POCA1..."
python3 -u /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/dataAnalysis/POCA1.py \
    --input  /gpfs/projects/cms/dominguezs/data/1_post_makeHLT/Pre__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2_seed358885362.root \
    --output /gpfs/projects/cms/dominguezs/data/2_poca/POCA__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2_seed358885362.root \
    --Lpx 128 --Lpy 128 --Lpz 128 \
    --npx 128 --npy 128 --npz 128 \
    --dimension 2D
if [ $? -ne 0 ]; then echo "[ERROR] POCA failed. Aborting."; exit 1; fi
echo "[CORRECT] POCA done."

# ==================================================
# AGGRESSIVE CLEANUP: Free disk space immediately
# ==================================================
echo "[INFO] Aggressive cleanup: removing intermediate files..."

# Remove raw file if it still exists (should have been removed earlier)
if [ -f "/gpfs/projects/cms/dominguezs/data/0_raw_geant4/Out__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2_seed358885362.root" ]; then
    rm -f "/gpfs/projects/cms/dominguezs/data/0_raw_geant4/Out__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2_seed358885362.root" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "[✓] Raw file removed (was still there): /gpfs/projects/cms/dominguezs/data/0_raw_geant4/Out__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2_seed358885362.root"
    else
        echo "[✗] Failed to remove raw file: /gpfs/projects/cms/dominguezs/data/0_raw_geant4/Out__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2_seed358885362.root"
    fi
fi

# Remove preprocessed file
if [ -f "/gpfs/projects/cms/dominguezs/data/1_post_makeHLT/Pre__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2_seed358885362.root" ]; then
    rm -f "/gpfs/projects/cms/dominguezs/data/1_post_makeHLT/Pre__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2_seed358885362.root" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "[✓] Preprocessed file removed: /gpfs/projects/cms/dominguezs/data/1_post_makeHLT/Pre__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2_seed358885362.root"
    else
        echo "[✗] Failed to remove preprocessed file: /gpfs/projects/cms/dominguezs/data/1_post_makeHLT/Pre__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2_seed358885362.root"
    fi
fi

# Verify disk freed
echo "[INFO] Cleanup finished. POCA output ready: /gpfs/projects/cms/dominguezs/data/2_poca/POCA__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2_seed358885362.root"
echo "[CORRECT] Job finished: _Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2 | seed=358885362"
