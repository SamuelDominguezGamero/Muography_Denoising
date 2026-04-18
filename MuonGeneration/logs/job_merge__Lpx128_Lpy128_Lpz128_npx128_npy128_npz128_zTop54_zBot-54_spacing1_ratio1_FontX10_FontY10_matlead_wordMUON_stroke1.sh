#!/bin/bash
#SBATCH --job-name=merge__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing1_ratio1_FontX10_FontY10_matlead_wordMUON_stroke1
#SBATCH --output=/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs/log_merge__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing1_ratio1_FontX10_FontY10_matlead_wordMUON_stroke1.out
#SBATCH --error=/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs/log_merge__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing1_ratio1_FontX10_FontY10_matlead_wordMUON_stroke1.err
#SBATCH --partition=wncompute_ifca

source /gpfs/users/dominguezs/Muography_Denoising/setup.sh

echo "[INFO] Starting merge for: _Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing1_ratio1_FontX10_FontY10_matlead_wordMUON_stroke1"

python3 -u /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/test/merge_results.py \
    --namefile         _Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing1_ratio1_FontX10_FontY10_matlead_wordMUON_stroke1 \
    --n_jobs           10 \
    --npx              128 \
    --npy              128 \
    --npz              128 \
    --path_poca_output /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/post_POCA_data \
    --output           /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/merged_poca_data/MERGED__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing1_ratio1_FontX10_FontY10_matlead_wordMUON_stroke1_2D.npy
if [ $? -ne 0 ]; then echo "[ERROR] Merge failed. Aborting."; exit 1; fi

echo "[CORRECT] Merge finished for: _Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing1_ratio1_FontX10_FontY10_matlead_wordMUON_stroke1"

echo "[INFO] Removing splitted POCA files for: _Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing1_ratio1_FontX10_FontY10_matlead_wordMUON_stroke1"
# Usamos el prefijo específico para no borrar lo de otros jobs
rm /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/post_POCA_data/POCA__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing1_ratio1_FontX10_FontY10_matlead_wordMUON_stroke1_seed*.npy
echo "[CORRECT] Split POCA files removed for: _Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing1_ratio1_FontX10_FontY10_matlead_wordMUON_stroke1"

echo "[INFO] Cleaning up seed logs..."
sleep 10
rm /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs/log__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing1_ratio1_FontX10_FontY10_matlead_wordMUON_stroke1_seed*.out
rm /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs/log__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing1_ratio1_FontX10_FontY10_matlead_wordMUON_stroke1_seed*.err
rm /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs/job__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing1_ratio1_FontX10_FontY10_matlead_wordMUON_stroke1_seed*.sh
echo "[CORRECT] Cleanup finished."
