"""
This file automates the full simulation pipeline:
  1. Creates all geometry files (JSON) for the neural network training dataset.
  2. Submits SLURM jobs for each geometry x seed combination. Each job:
       - Runs the Geant4 Monte Carlo simulation
       - Correlates muon tracks (makeHLTuple.py)
       - Runs the POCA reconstruction (POCA.py)

Author: Samuel Dominguez
"""

import subprocess
import os
import sys
import glob


# ===========================================================================
# CONTROL FLAGS
# ===========================================================================
create_geometries = True   # set to False to skip geometry creation
simulate          = False  # set to True to submit SLURM jobs (cluster only)
environment       = "local"  # "local" or "cluster"
dimension         = "2D"     # "2D" or "3D", used as prefix in filenames


# ===========================================================================
# SECURITY CHECKS
# ===========================================================================
if environment == "local":
    simulate = False  # simulation can only run in the cluster

if not create_geometries:
    sys.exit("[INFO] create_geometries=False. Set it to True to create geometries.")


# ===========================================================================
# PATHS
# ===========================================================================
SCRIPT_DIR             = os.path.dirname(os.path.abspath(__file__))
CREATE_GEOMETRY_SCRIPT = os.path.join(SCRIPT_DIR, "create_geometry.py")

if environment == "cluster":
    PATH_geometry_files = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/geometric_configurations_json"
    PATH_density_files  = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/ground_truth_data"
    PATH_output_raw     = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_raw"
    PATH_preprocessed   = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_preprocessed"
    PATH_poca_output    = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/post_POCA_data"
    PATH_logs           = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs"
    PATH_data_analysis  = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/dataAnalysis"
    PATH_generator      = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration-build/Generator"
    PATH_setup          = "/gpfs/users/dominguezs/Muography_Denoising/setup.sh"

elif environment == "local":
    PATH_geometry_files = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/geometric_configurations_json"
    PATH_density_files  = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/ground_truth_data"
    PATH_output_raw     = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/data_raw"
    PATH_preprocessed   = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/data_preprocessed"
    PATH_poca_output    = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/post_POCA_data"
    PATH_logs           = "/home/samuel/Work/Muography_Denoising/MuonGeneration/logs"
    PATH_data_analysis  = "/home/samuel/Work/Muography_Denoising/MuonGeneration/dataAnalysis"
    PATH_generator      = None  # not available locally
    PATH_setup          = None  # not available locally

else:
    sys.exit("[ERROR] environment must be 'local' or 'cluster'.")

print(f"[INFO] Environment: {environment}")
print(f"[INFO] Script directory: {SCRIPT_DIR}")


# ===========================================================================
# GEOMETRY PARAMETERS
# Fixed POCA voxelization parameters (should remain constant across dataset)
# ===========================================================================
Lpx = 128   # physical size of the reconstruction volume in X [cm]
Lpy = 128   # physical size of the reconstruction volume in Y [cm]
Lpz = 128   # physical size of the reconstruction volume in Z [cm]
npx = 128   # number of POCA voxels in X
npy = 128   # number of POCA voxels in Y
npz = 128   # number of POCA voxels in Z
zPosDetector_top =  118  # Z position of the top detector [cm]
zPosDetector_bot = -118  # Z position of the bottom detector [cm]


# ===========================================================================
# GEOMETRY VARIATIONS (dataset sweep)
# ===========================================================================
spacings       = [1]        # spacing between letters [G4 voxels]
ratios         = [1]        # G4 voxel size / POCA voxel size
FontsSizeX     = [8]        # font size in X [G4 voxels] — available: 8, 10, 12, 14, 16
materials      = ["lead"]   # available: "lead"
words_geometry = ["MUON"]   # words to embed in the geometry
strokes        = [1]        # stroke width [G4 voxels]

total_geometries = (
    len(spacings) * len(ratios) * len(FontsSizeX) *
    len(materials) * len(words_geometry) * len(strokes)
)
print(f"[INFO] Total geometries to generate: {total_geometries}")


# ===========================================================================
# SIMULATION PARAMETERS
# ===========================================================================
total_muons_per_geometry = 100_000_000
n_muons_per_job          = 10_000
n_jobs_per_geometry      = total_muons_per_geometry // n_muons_per_job
print(f"[INFO] Muons per geometry: {total_muons_per_geometry:,}")
print(f"[INFO] Muons per job:      {n_muons_per_job:,}")
print(f"[INFO] Jobs per geometry:  {n_jobs_per_geometry}")
print(f"[INFO] Total jobs:         {total_geometries * n_jobs_per_geometry:,}")


# ===========================================================================
# STEP 1: GEOMETRY CREATION
# ===========================================================================
print("\n" + "="*60)
print("STEP 1: GEOMETRY CREATION")
print("="*60)

i = 0
for spacing in spacings:
    for ratio in ratios:
        for x in FontsSizeX:
            for material in materials:
                for word in words_geometry:
                    for stroke in strokes:
                        i += 1

                        namefile = (
                            f"{dimension}"
                            f"_Lpx{Lpx}_Lpy{Lpy}_Lpz{Lpz}"
                            f"_npx{npx}_npy{npy}_npz{npz}"
                            f"_zTop{zPosDetector_top}_zBot{zPosDetector_bot}"
                            f"_spacing{spacing}_ratio{ratio}"
                            f"_FontX{x}_FontY{x}"
                            f"_mat{material}_word{word}_stroke{stroke}"
                        )

                        output_json    = os.path.join(PATH_geometry_files, namefile + ".json")
                        output_density = os.path.join(PATH_density_files,  namefile + "_ground_truth_density.npy")

                        command = [
                            "python3", CREATE_GEOMETRY_SCRIPT,
                            "--Lpx", str(Lpx),
                            "--Lpy", str(Lpy),
                            "--Lpz", str(Lpz),
                            "--npx", str(npx),
                            "--npy", str(npy),
                            "--npz", str(npz),
                            "--zPosDetector_top", str(zPosDetector_top),
                            "--zPosDetector_bot", str(zPosDetector_bot),
                            "--spacing",          str(spacing),
                            "--ratio",            str(ratio),
                            "--FontSizeX",        str(x),
                            "--FontSizeY",        str(x),
                            "--material",         material,
                            "--word_geometry",    word,
                            "--StrokeWidth",      str(stroke),
                            "--output_json",               output_json,
                            "--output_ground_truth_density", output_density,
                        ]

                        result = subprocess.run(command, capture_output=True, text=True)

                        if result.returncode != 0:
                            print(f"[ERROR] Geometry {i}/{total_geometries} failed:\n{result.stderr}")
                        else:
                            print(f"[CORRECT] Geometry {i}/{total_geometries} created: {namefile}")


print("\n[CORRECT] ALL GEOMETRIES CREATED SUCCESSFULLY")
print("="*60 + "\n")


# ===========================================================================
# STEP 2: SLURM JOB SUBMISSION
# One job per geometry x seed. Each job runs: Geant4 + makeHLTuple + POCA
# ===========================================================================
print("="*60)
print("STEP 2: SLURM JOB SUBMISSION")
print("="*60)

if not simulate:
    sys.exit("[INFO] simulate=False. Set it to True to submit SLURM jobs.")

os.makedirs(PATH_logs, exist_ok=True)

jobs_submitted = 0
jobs_failed    = 0

# Re-iterate over the same geometry sweep so we have all parameters available
# without needing to parse the JSON files (which would risk breaking the C++ pipeline)
for spacing in spacings:
    for ratio in ratios:
        for x in FontsSizeX:
            for material in materials:
                for word in words_geometry:
                    for stroke in strokes:

                        # Reconstruct the geometry name (must match Step 1 exactly)
                        namefile = (
                            f"{dimension}"
                            f"_Lpx{Lpx}_Lpy{Lpy}_Lpz{Lpz}"
                            f"_npx{npx}_npy{npy}_npz{npz}"
                            f"_zTop{zPosDetector_top}_zBot{zPosDetector_bot}"
                            f"_spacing{spacing}_ratio{ratio}"
                            f"_FontX{x}_FontY{x}"
                            f"_mat{material}_word{word}_stroke{stroke}"
                        )
                        geometry_file = os.path.join(PATH_geometry_files, namefile + ".json")

                        # Safety check: make sure the geometry file exists before submitting
                        if not os.path.exists(geometry_file):
                            print(f"[ERROR] Geometry file not found, skipping: {geometry_file}")
                            continue

                        print(f"\n[INFO] Submitting {n_jobs_per_geometry} jobs for: {namefile}")

                        for job in range(n_jobs_per_geometry):
                            seed = job

                            # Output file paths for this specific job
                            out_raw   = os.path.join(PATH_output_raw,  f"Out_{namefile}_seed{seed}.root")
                            out_pre   = os.path.join(PATH_preprocessed, f"Pre_{namefile}_seed{seed}.root")
                            out_poca  = os.path.join(PATH_poca_output,  f"POCA_{namefile}_seed{seed}.npy")
                            out_log   = os.path.join(PATH_logs, f"log_{namefile}_seed{seed}.out")
                            out_err   = os.path.join(PATH_logs, f"log_{namefile}_seed{seed}.err")
                            out_sh    = os.path.join(PATH_logs, f"job_{namefile}_seed{seed}.sh")

                            job_script = f"""#!/bin/bash
#SBATCH --job-name=muon_seed{seed}
#SBATCH --output={out_log}
#SBATCH --error={out_err}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00

source {PATH_setup}

echo "[INFO] ======================================================"
echo "[INFO] Job started: {namefile} | seed={seed}"
echo "[INFO] ======================================================"

# --- 1st: Geant4 Monte Carlo simulation ---
echo "[INFO] Running Geant4 simulation..."
{PATH_generator} \\
    --input  {geometry_file} \\
    --output {out_raw} \\
    --number {n_muons_per_job} \\
    --seed   {seed}

if [ $? -ne 0 ]; then echo "[ERROR] Geant4 failed. Aborting."; exit 1; fi
echo "[CORRECT] Geant4 done."

# --- 2nd: Track correlation ---
echo "[INFO] Running makeHLTuple..."
python3 -u {PATH_data_analysis}/makeHLTuple.py \\
    --input  {out_raw} \\
    --output {out_pre}

if [ $? -ne 0 ]; then echo "[ERROR] makeHLTuple failed. Aborting."; exit 1; fi
echo "[CORRECT] makeHLTuple done."

# --- 3rd: POCA reconstruction ---
echo "[INFO] Running POCA..."
python3 -u {PATH_data_analysis}/POCA.py \\
    --input  {out_pre} \\
    --output {out_poca} \\
    --Lpx {Lpx} --Lpy {Lpy} --Lpz {Lpz} \\
    --npx {npx} --npy {npy} --npz {npz}

if [ $? -ne 0 ]; then echo "[ERROR] POCA failed. Aborting."; exit 1; fi
echo "[CORRECT] POCA done."

echo "[CORRECT] ======================================================"
echo "[CORRECT] Job finished: {namefile} | seed={seed}"
echo "[CORRECT] ======================================================"
"""

                            # Write the job script to disk and submit it
                            with open(out_sh, "w") as f:
                                f.write(job_script)

                            result = subprocess.run(
                                ["sbatch", out_sh],
                                capture_output=True, text=True
                            )

                            if result.returncode != 0:
                                print(f"[ERROR] sbatch failed: seed={seed} | {result.stderr.strip()}")
                                jobs_failed += 1
                            else:
                                print(f"[SUBMITTED] seed={seed:04d} --> {result.stdout.strip()}")
                                jobs_submitted += 1


print("\n" + "="*60)
print(f"[INFO] Jobs submitted: {jobs_submitted}")
print(f"[INFO] Jobs failed:    {jobs_failed}")
print("="*60)