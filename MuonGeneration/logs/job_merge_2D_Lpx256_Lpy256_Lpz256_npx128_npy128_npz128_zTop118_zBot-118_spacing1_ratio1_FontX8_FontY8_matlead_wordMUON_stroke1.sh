#!/bin/bash
#SBATCH --job-name=merge_MUON
#SBATCH --output=/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs/log_merge_2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1.out
#SBATCH --error=/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs/log_merge_2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

source /gpfs/users/dominguezs/Muography_Denoising/setup.sh

echo "[INFO] Starting merge for: 2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1"

python3 -u /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/test/merge_results.py \
    --namefile         2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1 \
    --n_jobs           10 \
    --npx              128 \
    --npy              128 \
    --npz              128 \
    --path_poca_output /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/post_POCA_data \
    --output           /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/merged_poca_data/MERGED_2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1.npy

echo "[CORRECT] Merge finished for: 2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1"
