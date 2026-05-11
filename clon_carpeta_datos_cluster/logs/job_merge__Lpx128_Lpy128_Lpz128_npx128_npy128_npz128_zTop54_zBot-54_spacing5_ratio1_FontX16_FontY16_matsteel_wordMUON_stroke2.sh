#!/bin/bash
#SBATCH --job-name=merge__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2
#SBATCH --output=/gpfs/projects/cms/dominguezs/data/logs/log_merge__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2.out
#SBATCH --error=/gpfs/projects/cms/dominguezs/data/logs/log_merge__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2.err
#SBATCH --partition=wncompute_ifca
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source /gpfs/users/dominguezs/Muography_Denoising/setup.sh

echo "[INFO] Starting merge pipeline for: _Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2"

# --- 1st: Merge all Pre_*.root files ---
echo "[INFO] Running merge_hits.py..."
python3 -u /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/test/merge_hits.py \
    --namefile          _Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2 \
    --n_jobs            300 \
    --path_hits_input   /gpfs/projects/cms/dominguezs/data/1_post_makeHLT \
    --path_hits_output  /gpfs/projects/cms/dominguezs/data/2_merged_post_makeHLT \
    --output            /gpfs/projects/cms/dominguezs/data/2_merged_post_makeHLT/Pre_merged__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2.root
if [ $? -ne 0 ]; then echo "[ERROR] merge_hits.py failed. Aborting."; exit 1; fi
echo "[CORRECT] merge_hits.py done."

# --- 2nd: Run POCA1.py on merged hits ---
echo "[INFO] Running POCA1.py on merged hits..."
python3 -u /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/dataAnalysis/POCA1.py \
    --input     /gpfs/projects/cms/dominguezs/data/2_merged_post_makeHLT/Pre_merged__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2.root \
    --output    /gpfs/projects/cms/dominguezs/data/3_merged_poca/POCA_merged__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2.root \
    --Lpx 128 --Lpy 128 --Lpz 128 \
    --npx 128 --npy 128 --npz 128 \
    --dimension 2D
if [ $? -ne 0 ]; then echo "[ERROR] POCA1.py failed. Aborting."; exit 1; fi
echo "[CORRECT] POCA1.py done."

# --- 3rd: Cleanup intermediate files ---
echo "[INFO] Cleaning up intermediate files..."

# Remove individual Pre_*.root files
rm /gpfs/projects/cms/dominguezs/data/1_post_makeHLT/Pre__Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2_seed*.root
echo "[CORRECT] Individual Pre_*.root files removed."

echo "[INFO] Cleanup finished."
echo "[CORRECT] Merge pipeline finished for: _Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matsteel_wordMUON_stroke2"
