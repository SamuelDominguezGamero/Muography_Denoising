#!/bin/bash
#SBATCH --job-name=muon_seed1
#SBATCH --output=/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs/log_2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1_seed1.out
#SBATCH --error=/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs/log_2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1_seed1.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00

source /gpfs/users/dominguezs/Muography_Denoising/setup.sh

echo "[INFO] Job started: 2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1 | seed=1"

# --- 1st: Geant4 Monte Carlo simulation ---
echo "[INFO] Running Geant4 simulation..."
cd /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration-build/
/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration-build/Generator \
    --input  /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/geometric_configurations_json/2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1.json \
    --output /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_raw/Out_2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1_seed1.root \
    --number 10000 \
    --seed   1
if [ $? -ne 0 ]; then echo "[ERROR] Geant4 failed. Aborting."; exit 1; fi
echo "[CORRECT] Geant4 done."

# --- 2nd: Track correlation ---
echo "[INFO] Running makeHLTuple..."
python3 -u /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/dataAnalysis/makeHLTuple.py \
    --input  /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_raw/Out_2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1_seed1.root \
    --conf   /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/geometric_configurations_json/2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1.json \
    --output /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_preprocessed/Pre_2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1_seed1.root
if [ $? -ne 0 ]; then echo "[ERROR] makeHLTuple failed. Aborting."; exit 1; fi
echo "[CORRECT] makeHLTuple done."

# --- 3rd: POCA reconstruction ---
echo "[INFO] Running POCA..."
python3 -u /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/dataAnalysis/POCA.py \
    --input  /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_preprocessed/Pre_2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1_seed1.root \
    --output /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/post_POCA_data/POCA_2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1_seed1.npy \
    --Lpx 256 --Lpy 256 --Lpz 256 \
    --npx 128 --npy 128 --npz 128
if [ $? -ne 0 ]; then echo "[ERROR] POCA failed. Aborting."; exit 1; fi
echo "[CORRECT] POCA done."

echo "[CORRECT] Job finished: 2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1 | seed=1"
