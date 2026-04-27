"""
This file automates the full simulation pipeline:
  1. Creates all geometry files (JSON) for the neural network training dataset.
  2. Submits SLURM jobs for each geometry x seed combination. Each job:
       - Runs the Geant4 Monte Carlo simulation
       - Correlates muon tracks (makeHLTuple.py)
       - Runs the POCA reconstruction (POCA.py)
  3. Submits a merge job per geometry with --dependency=afterok,
     so it only runs when ALL jobs for that geometry finish successfully.

"""

import subprocess
import os
import sys
import time
import glob
import shutil
import numpy as np
from pathlib import Path
from numpy.random import Generator, PCG64, SeedSequence

# ===========================================================================
# CONTROL FLAGS
# ===========================================================================
create_geometries = True
simulate          = True    # set to True to submit SLURM jobs (cluster only)
environment       = "cluster"  # "local" or "cluster"
dimension         = "2D"     # 2D or 3D, first we should stick to 2D for faster iterations
max_geometries    = 900      # the first geometries to be tested on
max_geometries_simulated = 900  # the first geometries to be simulated (if simulate=True)

force_resimulate  = False    # set to True to re-process geometries even if merged results exist

# ===========================================================================
# SLURM JOB THROTTLING
# ===========================================================================
# Maximum number of jobs allowed in the queue at the same time for this user.
# Run `sacctmgr show user <username> withassoc` or ask your sysadmin.
# A safe default is to leave ~10% headroom below your real limit.
MAX_JOBS_IN_QUEUE = 1500      # adjust to your cluster's limit
THROTTLE_SLEEP    = 30       # seconds to wait when queue is full before retrying

def get_current_job_count(username="dominguezs"):
    """Returns the number of jobs currently in the SLURM queue for the user."""
    result = subprocess.run(
        ["squeue", "-u", username, "-h", "--format=%i"],
        capture_output=True, text=True
    )
    lines = [l for l in result.stdout.strip().split('\n') if l]
    return len(lines)

def wait_for_slot(username="dominguezs", max_jobs=MAX_JOBS_IN_QUEUE, sleep=THROTTLE_SLEEP):
    """Blocks until there is room in the SLURM queue to submit at least one more job."""
    while True:
        count = get_current_job_count(username)
        if count < max_jobs:
            return count   # return current count so caller can log it
        print(f"[THROTTLE] Queue full ({count}/{max_jobs} jobs). Waiting {sleep}s before retrying...")
        time.sleep(sleep)

# ===========================================================================
# SECURITY CHECKS
# ===========================================================================
if environment == "local":
    simulate = False

if not create_geometries:
    print("[INFO] create_geometries=False. Set it to True to create geometries.")


# ===========================================================================
# PATHS
# ===========================================================================
SCRIPT_DIR             = os.path.dirname(os.path.abspath(__file__))
CREATE_GEOMETRY_SCRIPT = os.path.join(SCRIPT_DIR, "create_geometry.py")
MERGE_SCRIPT           = os.path.join(SCRIPT_DIR, "merge_results.py")
PLOT_SCRIPT            = os.path.join(SCRIPT_DIR, "plot_central_slice_comparison.py")

if environment == "cluster":
    PATH_geometry_files = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/geometric_configurations_json"
    PATH_density_files3D  = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/ground_truth_data/3Dimensions"
    PATH_density_files2D  = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/ground_truth_data/2Dimensions"
    PATH_output_raw     = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_raw"
    PATH_preprocessed   = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_preprocessed"
    PATH_poca_output    = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/post_POCA_data"
    PATH_merged_output  = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/merged_poca_data"
    PATH_png_comparisons = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/png_comparisons"
    PATH_logs           = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs"
    PATH_data_analysis  = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/dataAnalysis"
    PATH_generator      = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration-build/Generator"
    PATH_setup          = "/gpfs/users/dominguezs/Muography_Denoising/setup.sh"
    SLURM_USER          = "dominguezs"

elif environment == "local":
    PATH_geometry_files = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/geometric_configurations_json"
    PATH_density_files3D  = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/ground_truth_data/3Dimensions"
    PATH_density_files2D  = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/ground_truth_data/2Dimensions"
    PATH_output_raw     = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/data_raw"
    PATH_preprocessed   = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/data_preprocessed"
    PATH_poca_output    = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/post_POCA_data"
    PATH_merged_output  = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/merged_poca_data"
    PATH_png_comparisons = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/png_comparisons"
    PATH_logs           = "/home/samuel/Work/Muography_Denoising/MuonGeneration/logs"
    PATH_data_analysis  = "/home/samuel/Work/Muography_Denoising/MuonGeneration/dataAnalysis"
    PATH_generator      = None
    PATH_setup          = None
    SLURM_USER          = None

else:
    sys.exit("[ERROR] environment must be 'local' or 'cluster'.")

print(f"[INFO] Environment: {environment}")
print(f"[INFO] Script directory: {SCRIPT_DIR}")


# ===========================================================================
# GEOMETRY PARAMETERS (fixed)
# ===========================================================================
Lpx = 128
Lpy = 128
Lpz = 128
npx = 128
npy = 128
npz = 128
zPosDetector_top =  54
zPosDetector_bot = -54


# ===========================================================================
# GEOMETRY VARIATIONS
# ===========================================================================
# IMPORTANT: Not all combinations are valid. Check bitmaps_letters.py BITMAP_DATA:
#   - Sizes 8, 10, 12, 14, 16: Only stroke 1, 2 available
#   - Stroke 3 is NOT available for any size yet
#   - Only letters available: M, U, O, N

spacings       = [5]
ratios         = [1]
# Strategy: Use multiple sizes with stroke variations that are actually available
fontsizes      = [8, 10, 12, 14, 16] # readd 8, 10, 12, 14, 16
strokes        = [2] # 1, 2, 3 (3 not suitable for 14,16)

FontSizes      = [# NOW UNUSED, SHOULD BE REMOVED
    {"size": 8,  "strokes": [1, 2]},
    {"size": 10, "strokes": [1, 2]},
    {"size": 12, "strokes": [1, 2]},
    {"size": 14, "strokes": [1, 2]},
    {"size": 16, "strokes": [1, 2]},
]

materials      = ["steel", "uranium", "aluminium", "iron", "lead"]
words_geometry = [
    "MNUO", "MNOU", "MUNO", "MUON", "MONU", "MOUN",
    "NMUO", "NMOU", "NUMO", "NUOM", "NOMU", "NOUM",
    "UMNO", "UMON", "UNMO", "UNOM", "UOMN", "UONM",
    "OMNU", "OMUN", "ONMU", "ONUM", "OUMN", "OUNM"
]

# Count total valid geometries
total_geometries = 0
for spacing in spacings:
    for ratio in ratios:
        for fontsize in fontsizes:
            for material in materials:
                for word in words_geometry:
                    for stroke in strokes:
                        if stroke == 3 and (fontsize==14 or fontsize==16):  # stroke 3 not available for any size yet
                            continue
                        else:
                            total_geometries += 1

print(f"[INFO] ----- Total geometries to generate: {total_geometries}")
time.sleep(5)
print(60 * "-")
time.sleep(1)
print(60 * "-")
time.sleep(1)

# ===========================================================================
# SIMULATION PARAMETERS
# ===========================================================================
total_muons_per_geometry = 2_000_000
n_muons_per_job          = 75_000
n_jobs_per_geometry      = total_muons_per_geometry // n_muons_per_job
print(f"[INFO] Muons per geometry: {total_muons_per_geometry:,}")
time.sleep(1)
print(60 * "-")
print(f"[INFO] Muons per job:      {n_muons_per_job:,}")
time.sleep(1)
print(60 * "-")
print(f"[INFO] Jobs per geometry:  {n_jobs_per_geometry}")
time.sleep(1)
print(60 * "-")
print(f"[INFO] Total jobs:         {total_geometries * n_jobs_per_geometry:,}")


# ===========================================================================
# STEP 1: GEOMETRY CREATION
# ===========================================================================
print("\n" + "="*60)
print("STEP 1: GEOMETRY CREATION")
print("="*60)

i = 0
done_creating = False
for spacing in spacings:
    if not create_geometries or done_creating:
        break
    for ratio in ratios:
        if done_creating:
            break
        for fontsize in fontsizes:
            if done_creating:
                break
            x = fontsize
            for material in materials:
                if done_creating:
                    break
                for word in words_geometry:
                    if done_creating:
                        break
                    for stroke in strokes:
                        # FIX: stroke 3 not valid for any size yet (condition was inverted before)
                        if stroke == 3:
                            continue
                        i += 1 # geometries counting
                        if i > max_geometries:
                            print(f"[INFO] Reached max_geometries={max_geometries}. Stopping geometry creation.")
                            done_creating = True
                            break
                        namefile = (
                            f"_Lpx{Lpx}_Lpy{Lpy}_Lpz{Lpz}"
                            f"_npx{npx}_npy{npy}_npz{npz}"
                            f"_zTop{zPosDetector_top}_zBot{zPosDetector_bot}"
                            f"_spacing{spacing}_ratio{ratio}"
                            f"_FontX{x}_FontY{x}"
                            f"_mat{material}_word{word}_stroke{stroke}"
                        )

                        if os.path.exists(os.path.join(PATH_geometry_files, namefile + ".json")):
                            print(f"[INFO | EXISTING] ----- Geometry {i}/{total_geometries} ALREADY EXISTS, skipping: {namefile}")
                            continue

                        output_json    = os.path.join(PATH_geometry_files, namefile + ".json")
                        output_density3D = os.path.join(PATH_density_files3D,  namefile + "_ground_truth_density3D.npy")
                        output_density2D = os.path.join(PATH_density_files2D,  namefile + "_ground_truth_density2D.npy")

                        command = [
                            "python3", CREATE_GEOMETRY_SCRIPT,
                            "--Lpx", str(Lpx),
                            "--Lpy", str(Lpy),
                            "--Lpz", str(Lpz),
                            "--npx", str(npx),
                            "--npy", str(npy),
                            "--npz", str(npz),
                            "--zPosDetector_top",             str(zPosDetector_top),
                            "--zPosDetector_bot",             str(zPosDetector_bot),
                            "--spacing",                      str(spacing),
                            "--ratio",                        str(ratio),
                            "--FontSizeX",                    str(x),
                            "--FontSizeY",                    str(x),
                            "--material",                     material,
                            "--word_geometry",                word,
                            "--StrokeWidth",                  str(stroke),
                            "--output_json",                  output_json,
                            "--dimensions",                   str(dimension),
                            "--output2D_density",             output_density2D,
                            "--output3D_density",             output_density3D
                        ]
                        if environment == "local":
                            result = subprocess.run(command, capture_output=True, text=True)
                        elif environment == "cluster":
                            # Throttle: wait if queue is full before submitting geometry job
                            current_count = wait_for_slot(SLURM_USER, MAX_JOBS_IN_QUEUE, THROTTLE_SLEEP)

                            inner_command = " ".join(command)
                            full_wrap = f"source {PATH_setup} && {inner_command}"

                            sbatch_args = [
                                "sbatch",
                                f"--job-name=geom_{i}",
                                "--time=01:00:00",
                                "--mem=8G",
                                "--cpus-per-task=2",
                                f"--output={PATH_logs}/log_geom_{i}.out",
                                f"--chdir={PATH_logs}",
                                f"--wrap={full_wrap}",
                                "--partition=wncompute_ifca"
                            ]

                            result = subprocess.run(sbatch_args, capture_output=True, text=True)

                            print(f"[INFO] Geometry {i}/{total_geometries} creation started: {namefile} (queue: {current_count+1}/{MAX_JOBS_IN_QUEUE})")
                            time.sleep(0.1)
                        if result.returncode != 0:
                            print(f"[ERROR] Geometry {i}/{total_geometries} failed:\n{result.stderr}")
                            print(80 * '-')
                        else:
                            print(f"[CORRECT: JOB SUBMISSION] Geometry {i}/{total_geometries}: {namefile}")
                            print(80 * '-')
                            time.sleep(0.1)


print("="*60)
print("\n[CORRECT] ALL GEOMETRIES CREATED SUCCESSFULLY")
print("="*60 + "\n")

# ===========================================================================
# WAIT FOR GEOMETRY JOBS TO FINISH (if cluster environment)
# ===========================================================================
if create_geometries and environment == "cluster":
    print("[INFO] Waiting for geometry creation jobs to complete...")
    import subprocess as sp
    max_wait = 1800  # 30 minutes max
    elapsed = 0
    while elapsed < max_wait:
        result = sp.run(["squeue", "-u", SLURM_USER, "-h"], capture_output=True, text=True)
        job_count = len([l for l in result.stdout.strip().split('\n') if l and 'geom_' in l])
        if job_count == 0:
            print("[INFO] All geometry jobs finished.")
            break
        print(f"[INFO] Waiting for {job_count} geometry job(s)... ({elapsed}s)")
        time.sleep(5)
        elapsed += 5
    if elapsed >= max_wait:
        print("[WARNING] Timeout waiting for geometry jobs. Proceeding anyway.")
    print()


# ===========================================================================
# STEP 2: SIMULATION == SLURM JOB SUBMISSION + MERGE WITH DEPENDENCY
# ===========================================================================
print("="*60)
print("="*60)
print("STEP 2: SLURM JOB SUBMISSION")
print("="*60)
print("="*60)
time.sleep(10)

if not simulate:
    sys.exit("[INFO] simulate=False. Set it to True to submit SLURM jobs.")

os.makedirs(PATH_logs,          exist_ok=True)
os.makedirs(PATH_output_raw,    exist_ok=True)
os.makedirs(PATH_preprocessed,  exist_ok=True)
os.makedirs(PATH_poca_output,   exist_ok=True)
os.makedirs(PATH_merged_output, exist_ok=True)

jobs_submitted = 0
jobs_failed    = 0
merges_submitted = 0
geometries_skipped = 0


# FIX: glob needs a wildcard pattern to find files inside the directory
all_json_files = []
for file in glob.glob(os.path.join(PATH_geometry_files, "*.json")):
    all_json_files.append(file)
print(f"[INFO] ----- Total number of geometry json files available for simulation: {len(all_json_files)}")

# FIX: was appending to all_json_files instead of all_simulation_DataFiles,
#      and also missing the wildcard pattern
all_simulation_DataFiles = []
for file in glob.glob(os.path.join(PATH_merged_output, "*.npy")):
    all_simulation_DataFiles.append(file)
print(f"[INFO] ----- Total number of simulation data files available: {len(all_simulation_DataFiles)}")



# SIMULATION (SKIPPING ALREADY SIMULATED GEOMETRIES, FOR THE CHOSEN MUON FLUX)
i = 0
for file in all_json_files:
    i += 1
    if i > max_geometries_simulated:
        print(f"[INFO] Reached max_geometries_simulated={max_geometries_simulated}. Stopping simulation.")
        break

    # FIX: extract only the basename without extension, not the full path
    namefile = os.path.splitext(os.path.basename(file))[0]
    geometry_file = file # full path with extension

    if not os.path.exists(geometry_file):
        print(f"[WARNING] ----- Geometry file disappeared: {geometry_file}")
        continue

    # Check if merged result already exists WITH THE EXACT NUMBER OF MUONS
    # Try both new format (_Muons_) and legacy format (for backwards compatibility)
    if dimension == "2D":
        merged_output_new = os.path.join(PATH_merged_output, f"MERGED_{namefile}_Muons_{total_muons_per_geometry}_2D.npy")
        merged_output_legacy = os.path.join(PATH_merged_output, f"MERGED_{namefile}_2D.npy")
    elif dimension == "3D":
        merged_output_new = os.path.join(PATH_merged_output, f"MERGED_{namefile}_Muons_{total_muons_per_geometry}_3D.npy")
        merged_output_legacy = os.path.join(PATH_merged_output, f"MERGED_{namefile}_3D.npy")

    # Check both formats
    merged_exists_new = os.path.exists(merged_output_new)
    merged_exists_legacy = os.path.exists(merged_output_legacy)

    # If legacy format exists, rename it to new format
    if merged_exists_legacy and not merged_exists_new:
        try:
            import shutil
            shutil.move(merged_output_legacy, merged_output_new)
            print(f"[INFO] Renamed legacy format: {Path(merged_output_legacy).name} -> {Path(merged_output_new).name}")
            merged_exists_new = True
        except Exception as e:
            print(f"[WARNING] Failed to rename legacy file: {e}")

    # Skip if already processed
    if merged_exists_new and not force_resimulate:
        print(f"[SKIP] Already processed: {namefile} with {total_muons_per_geometry:,} muons")
        geometries_skipped += 1
        continue
    elif merged_exists_new and force_resimulate:
        print(f"[RESIMULATE] Force flag enabled, re-processing: {namefile}")
        # Clean old files before re-simulating
        old_poca = glob.glob(os.path.join(PATH_poca_output, f"POCA_{namefile}_seed*.npy"))
        for f in old_poca:
            os.remove(f)
        os.remove(merged_output_new)  # Remove old merged result
        print(f"[INFO] Cleaned old files for: {namefile}")

    print(f"\n[INFO] Submitting {n_jobs_per_geometry} jobs for: {namefile}")

    # Collect job IDs for this geometry to use in the merge dependency
    job_ids = []
    base_seed = int(time.time())
    ss = SeedSequence(base_seed)

    # Generate all child seeds at once (more efficient than spawning in loop)
    child_seeds = ss.spawn(n_jobs_per_geometry)

    for job in range(n_jobs_per_geometry):
        # Convert SeedSequence to integer for use in simulation
        rng = Generator(PCG64(child_seeds[job]))
        seed = rng.integers(0, 2**31 - 1)

        out_raw  = os.path.join(PATH_output_raw,   f"Out_{namefile}_seed{seed}.root")
        out_pre  = os.path.join(PATH_preprocessed, f"Pre_{namefile}_seed{seed}.root")
        out_poca = os.path.join(PATH_poca_output,  f"POCA_{namefile}_seed{seed}.npy")
        out_log  = os.path.join(PATH_logs, f"log_{namefile}_seed{seed}.out")
        out_err  = os.path.join(PATH_logs, f"log_{namefile}_seed{seed}.err")
        out_sh   = os.path.join(PATH_logs, f"job_{namefile}_seed{seed}.sh")

        job_script = f"""#!/bin/bash
#SBATCH --job-name=muon_seed{seed}
#SBATCH --output={out_log}
#SBATCH --error={out_err}
#SBATCH --partition=wncompute_ifca
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source {PATH_setup}

echo "[INFO] Job started: {namefile} | seed={seed}"

# --- 1st: Geant4 Monte Carlo simulation ---
echo "[INFO] Running Geant4 simulation..."
cd /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration-build/
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
    --conf   {geometry_file} \\
    --output {out_pre}
if [ $? -ne 0 ]; then echo "[ERROR] makeHLTuple failed. Aborting."; exit 1; fi
echo "[CORRECT] makeHLTuple done."

# eliminate intermediate files to save space
rm {out_raw}
echo "[INFO] Raw file removed to save space: {out_raw}"

# --- 3rd: POCA reconstruction ---
echo "[INFO] Running POCA..."
python3 -u {PATH_data_analysis}/POCA.py \\
    --input  {out_pre} \\
    --output {out_poca} \\
    --Lpx {Lpx} --Lpy {Lpy} --Lpz {Lpz} \\
    --npx {npx} --npy {npy} --npz {npz}
if [ $? -ne 0 ]; then echo "[ERROR] POCA failed. Aborting."; exit 1; fi
echo "[CORRECT] POCA done."

# ==================================================
# AGGRESSIVE CLEANUP: Free disk space immediately
# ==================================================
echo "[INFO] Aggressive cleanup: removing intermediate files..."

# Remove raw file if it still exists (should have been removed earlier)
if [ -f "{out_raw}" ]; then
    rm -f "{out_raw}" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "[✓] Raw file removed (was still there): {out_raw}"
    else
        echo "[✗] Failed to remove raw file: {out_raw}"
    fi
fi

# Remove preprocessed file
if [ -f "{out_pre}" ]; then
    rm -f "{out_pre}" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "[✓] Preprocessed file removed: {out_pre}"
    else
        echo "[✗] Failed to remove preprocessed file: {out_pre}"
    fi
fi

# Verify disk freed
echo "[INFO] Cleanup finished. POCA output ready: {out_poca}"
echo "[CORRECT] Job finished: {namefile} | seed={seed}"
"""
        with open(out_sh, "w") as f:
            f.write(job_script)

        # Throttle: wait if queue is full before submitting each simulation job
        current_count = wait_for_slot(SLURM_USER, MAX_JOBS_IN_QUEUE, THROTTLE_SLEEP)

        result = subprocess.run(
            ["sbatch", "--begin=now", out_sh],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"[ERROR] sbatch failed: seed={seed} | {result.stderr.strip()}")
            jobs_failed += 1
        else:
            # Extract job ID from "Submitted batch job 12345"
            job_id = result.stdout.strip().split()[-1]
            job_ids.append(job_id)
            print(f"[SUBMITTED] seed={seed:04d} --> job_id={job_id} (queue: {current_count+1}/{MAX_JOBS_IN_QUEUE})")
            jobs_submitted += 1

    # --------------------------------------------------
    # Submit merge job with dependency on ALL jobs finishing
    # --dependency=afterok:id1:id2:...:idN means the merge
    # job only runs if ALL listed jobs finish successfully.
    # If any job fails, the merge is cancelled automatically.
    # SLURM handles the waiting automatically, no explicit sleep needed.
    # --------------------------------------------------

    if not job_ids:
        print(f"[WARNING] No jobs submitted for {namefile}, skipping merge.")
        continue
    elif len(job_ids) < n_jobs_per_geometry:
        print(f"[WARNING] Only {len(job_ids)}/{n_jobs_per_geometry} jobs submitted for {namefile}.")
        print(f"[WARNING] Merge job will be submitted with dependency on available jobs.")

    print(f"[INFO] Submitting merge job for: {namefile} with dependency on {len(job_ids)} jobs.")
    dependency_str = "afterok:" + ":".join(job_ids)

    if dimension == "2D":
        out_merged = os.path.join(PATH_merged_output, f"MERGED_{namefile}_Muons_{total_muons_per_geometry}_2D.npy")
        png_name = f"{namefile}_2D.png"
    elif dimension == "3D":
        out_merged = os.path.join(PATH_merged_output, f"MERGED_{namefile}_Muons_{total_muons_per_geometry}_3D.npy")
        png_name = f"{namefile}_3D.png"

    merge_log  = os.path.join(PATH_logs, f"log_merge_{namefile}.out")
    merge_err  = os.path.join(PATH_logs, f"log_merge_{namefile}.err")
    merge_sh   = os.path.join(PATH_logs, f"job_merge_{namefile}.sh")

    merge_script = f"""#!/bin/bash
#SBATCH --job-name=merge_{namefile}
#SBATCH --output={merge_log}
#SBATCH --error={merge_err}
#SBATCH --partition=wncompute_ifca
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

source {PATH_setup}

echo "[INFO] Starting merge for: {namefile}"

python3 -u {MERGE_SCRIPT} \\
    --namefile         {namefile} \\
    --n_jobs           {n_jobs_per_geometry} \\
    --npx              {npx} \\
    --npy              {npy} \\
    --npz              {npz} \\
    --path_poca_output {PATH_poca_output} \\
    --output           {out_merged}
if [ $? -ne 0 ]; then echo "[ERROR] Merge failed. Aborting."; exit 1; fi

echo "[CORRECT] Merge finished for: {namefile}"

echo "[INFO] Removing splitted POCA files for: {namefile}"
# Usamos el prefijo específico para no borrar lo de otros jobs
rm {PATH_poca_output}/POCA_{namefile}_seed*.npy
echo "[CORRECT] Split POCA files removed for: {namefile}"

echo "[INFO] Cleaning up seed logs..."
sleep 10
rm {PATH_logs}/log_{namefile}_seed*.out
rm {PATH_logs}/log_{namefile}_seed*.err
rm {PATH_logs}/job_{namefile}_seed*.sh
echo "[CORRECT] Cleanup finished."
"""
    with open(merge_sh, "w") as f:
        f.write(merge_script)

    # Throttle before submitting the merge job too
    current_count = wait_for_slot(SLURM_USER, MAX_JOBS_IN_QUEUE, THROTTLE_SLEEP)

    result = subprocess.run(
        ["sbatch", "--begin=now", f"--dependency={dependency_str}", merge_sh],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"[ERROR] Merge job submission failed: {result.stderr.strip()}")
    else:
        merge_id = result.stdout.strip().split()[-1]
        print(f"[SUBMITTED] Merge job --> job_id={merge_id} (depends on {len(job_ids)} jobs)")
        merges_submitted += 1


print("\n" + "="*60)
print("[FINAL SUMMARY]")
print("="*60)
print(f"[INFO] Geometries skipped (already processed): {geometries_skipped}")
print(f"[INFO] Simulation jobs submitted:             {jobs_submitted}")
print(f"[INFO] Simulation jobs failed:                {jobs_failed}")
print(f"[INFO] Merge jobs submitted:                  {merges_submitted}")
<<<<<<< HEAD
print("="*60)
=======
print("="*60)z
>>>>>>> a85a374f229d4a7b79e3a609368f3606c7cd35ad
