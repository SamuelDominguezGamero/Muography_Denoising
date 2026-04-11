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
if [ $? -ne 0 ]; then echo "[ERROR] Merge failed. Aborting."; exit 1; fi

if [ ! -f /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/ground_truth_data/2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1_ground_truth_density.npy ]; then
    echo "[ERROR] Ground truth tensor not found: /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/ground_truth_data/2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1_ground_truth_density.npy"
    exit 1
fi

echo "[INFO] Generating PNG comparison for: 2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1"
python3 -u /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/test/plot_central_slice_comparison.py \
    --poca_merged         /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/merged_poca_data/MERGED_2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1.npy \
    --ground_truth_tensor /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/ground_truth_data/2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1_ground_truth_density.npy \
    --Lx 256 --Ly 256 --Lz 256 \
    --z 0 \
    --output_dir /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/png_comparisons \
    --output_name 2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1.png
if [ $? -ne 0 ]; then echo "[ERROR] PNG comparison generation failed. Aborting."; exit 1; fi

echo "[CORRECT] Merge finished for: 2D_Lpx256_Lpy256_Lpz256_npx128_npy128_npz128_zTop118_zBot-118_spacing1_ratio1_FontX8_FontY8_matlead_wordMUON_stroke1"
