#!/bin/bash
#SBATCH --job-name=merge__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX12_FontY12_mataluminium_wordMUON_stroke2
#SBATCH --output=/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs/log_merge__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX12_FontY12_mataluminium_wordMUON_stroke2.out
#SBATCH --error=/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs/log_merge__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX12_FontY12_mataluminium_wordMUON_stroke2.err
#SBATCH --partition=wncompute_ifca
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

source /gpfs/users/dominguezs/Muography_Denoising/setup.sh

echo "[INFO] Starting merge for: _Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX12_FontY12_mataluminium_wordMUON_stroke2"

python3 -u /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/test/merge_results_1.py \
    --namefile         _Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX12_FontY12_mataluminium_wordMUON_stroke2 \
    --n_jobs           30 \
    --npx              128 \
    --npy              128 \
    --npz              128 \
    --dimension        2D \
    --path_poca_output /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/post_POCA_data \
    --output           /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/merged_poca_data/UNET1/MERGED__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX12_FontY12_mataluminium_wordMUON_stroke2_Muons_900000_2D.npy
if [ $? -ne 0 ]; then echo "[ERROR] Merge failed. Aborting."; exit 1; fi

echo "[CORRECT] Merge finished for: _Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX12_FontY12_mataluminium_wordMUON_stroke2"

echo "[INFO] Removing splitted POCA files for: _Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX12_FontY12_mataluminium_wordMUON_stroke2"
rm /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/post_POCA_data/POCA__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX12_FontY12_mataluminium_wordMUON_stroke2_seed*.npy
echo "[CORRECT] Split POCA files removed for: _Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX12_FontY12_mataluminium_wordMUON_stroke2"

echo "[INFO] Cleaning up intermediate logs..."
sleep 5
rm /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs/log__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX12_FontY12_mataluminium_wordMUON_stroke2_seed*.out
rm /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs/log__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX12_FontY12_mataluminium_wordMUON_stroke2_seed*.err
rm /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs/job__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX12_FontY12_mataluminium_wordMUON_stroke2_seed*.sh

echo "[CORRECT] Cleanup finished."
