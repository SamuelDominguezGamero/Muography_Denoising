"""
SIMULATION PIPELINE
- look for geometry files (json)
- simulate with Geant4 (SLURM jobs, throttled to avoid overloading the queue)
- post-process with makeHLTuple and POCA (in the same SLURM job, after simulation)
- merge results with merge_results_1.py (SLURM job dependent on all simulation jobs finishing successfully)
"""

import subprocess
import os
import sys
import time
import glob
import shutil
import json
import numpy as np
from pathlib import Path
from numpy.random import Generator, PCG64, SeedSequence

# ===========================================================================
# CONTROL FLAGS
# ===========================================================================
simulate          = True    # set to True to submit SLURM jobs (cluster only)
environment       = "cluster"  # "local" or "cluster"
dimension         = "2D"     # 2D or 3D, first we should stick to 2D for faster iterations
max_geometries_simulated = 1  # the first geometries to be simulated (if simulate=True)

force_resimulate  = False   # set to True to re-process geometries even if merged results exist

# ===========================================================================
# SLURM JOB THROTTLING
# ===========================================================================
# Maximum number of jobs allowed in the queue at the same time for this user.
# Run `sacctmgr show user <username> withassoc` or ask your sysadmin.
# A safe default is to leave ~10% headroom below your real limit.
MAX_JOBS_IN_QUEUE = 1000      # adjust to your cluster's limit
THROTTLE_SLEEP    = 15      # seconds to wait when queue is full before retrying

def get_current_job_count(username="dominguezs"):
    """Returns the number of jobs currently in the SLURM queue for the user."""
    result = subprocess.run(
        ["squeue", "-u", username, "-h", "--format=%i"],
        capture_output=True, text=True
    )
    lines = [l for l in result.stdout.strip().split('\n') if l]
    return len(lines)

def wait_for_slot(username="dominguezs", max_jobs=MAX_JOBS_IN_QUEUE, sleep=THROTTLE_SLEEP, target_jobs_to_submit=1):
    """
    Blocks until there is enough room in the queue to submit target_jobs_to_submit jobs.
    Returns tuple (current_count, wait_iterations).
    """
    wait_iterations = 0
    while True:
        count = get_current_job_count(username)
        slots_available = max_jobs - count
        
        if slots_available >= target_jobs_to_submit:
            return count, wait_iterations
        
        wait_iterations += 1
        if wait_iterations == 1:
            print(f"[THROTTLE] Need {target_jobs_to_submit} slots, but only {slots_available} available ({count}/{max_jobs} jobs in queue).")
            print(f"[THROTTLE] Waiting for jobs to complete... ({sleep}s between checks)")
        elif wait_iterations % 4 == 0:  # Print every 4 iterations (~60s)
            print(f"[THROTTLE] Still waiting... {slots_available}/{target_jobs_to_submit} slots available ({count}/{max_jobs})")
        
        time.sleep(sleep)


# ===========================================================================
# SECURITY CHECKS
# ===========================================================================
if environment == "local":
    simulate = False

if not simulate:
    sys.exit("[INFO] ----- simulate=False. Set it to True to submit SLURM jobs.")


# ===========================================================================
# PATHS
# ===========================================================================
SCRIPT_DIR             = os.path.dirname(os.path.abspath(__file__))
CREATE_GEOMETRY_SCRIPT = os.path.join(SCRIPT_DIR, "create_geometry.py")
MERGE_SCRIPT           = os.path.join(SCRIPT_DIR, "merge_results_1.py")
PLOT_SCRIPT            = os.path.join(SCRIPT_DIR, "plot_central_slice_comparison.py")

if environment == "cluster":
    # PATH_geometry_files = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/geometric_configurations_json"
    PATH_geometry_files = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/empty_configurations"
    PATH_density_files3D  = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/ground_truth_data/3Dimensions"
    PATH_density_files2D  = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/ground_truth_data/2Dimensions/UNET1"
    PATH_output_raw     = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_raw"
    PATH_preprocessed   = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_preprocessed"
    PATH_poca_output    = "/gpfs/projects/cms/dominguezs/data/POCA_POINTS"
    # PATH_poca_output    = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/post_POCA_data"
    PATH_merged_output  = "/gpfs/projects/cms/dominguezs/data/merged_poca_data/UNET1"
#    PATH_merged_output  = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/merged_poca_data/UNET1"
    PATH_png_comparisons = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/png_comparisons"
    PATH_logs           = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs"
    PATH_data_analysis  = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/dataAnalysis"
    PATH_generator      = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration-build/Generator"
    PATH_setup          = "/gpfs/users/dominguezs/Muography_Denoising/setup.sh"
    SLURM_USER          = "dominguezs"

elif environment == "local":
    PATH_geometry_files = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/geometric_configurations_json"
    PATH_density_files3D  = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/ground_truth_data/3Dimensions"
    PATH_density_files2D  = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/ground_truth_data/2Dimensions/UNET1"
    PATH_output_raw     = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/data_raw"
    PATH_preprocessed   = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/data_preprocessed"
    PATH_poca_output    = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/post_POCA_data"
    PATH_merged_output  = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/merged_poca_data/UNET1"
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
# SIMULATION PARAMETERS
# ===========================================================================
total_muons_per_geometry = 15_000_000 
n_muons_per_job          = 100_000
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

# ===========================================================================
# STEP 2: SIMULATION == SLURM JOB SUBMISSION + MERGE WITH DEPENDENCY
# ===========================================================================
# WORKFLOW:
#   1. For each geometry: wait for queue space, then submit all 60 simulation jobs
#   2. Each job runs: Geant4 → makeHLTuple → POCA → cleanup
#   3. When all 60 jobs finish, submit ONE merge job (depends on all 60 via --dependency)
#   4. Merge job combines all POCA results into single .npy file
#   5. Merge job cleans up intermediate POCA files when done
# THROTTLING: Queue is limited to MAX_JOBS_IN_QUEUE to avoid overloading cluster
#   - We wait ONCE before submitting all 60 jobs for a geometry
#   - SLURM handles job queueing automatically (no manual waits in loop)
# ===========================================================================
print("="*60)
print("="*60)
print("STEP 2: SLURM JOB SUBMISSION")
print("="*60)
print("="*60)
time.sleep(5)


os.makedirs(PATH_logs,          exist_ok=True)
os.makedirs(PATH_output_raw,    exist_ok=True)
os.makedirs(PATH_preprocessed,  exist_ok=True)
os.makedirs(PATH_poca_output,   exist_ok=True)
os.makedirs(PATH_merged_output, exist_ok=True)

jobs_submitted = 0
jobs_failed    = 0
merges_submitted = 0
geometries_skipped = 0
geometries_failed_during_creation = 0


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




# ===== MAIN LOOP: Process each geometry =====
i = 0
for file in all_json_files:
    i += 1
    if i > max_geometries_simulated:
        print(f"[INFO] Reached max_geometries_simulated={max_geometries_simulated}. Stopping simulation.")
        break

    # Extract geometry name and full path
    namefile = os.path.splitext(os.path.basename(file))[0]
    geometry_file = file # full path with extension




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

    print(f"\n[INFO] [{i:4d}/{max_geometries_simulated}] Submitting {n_jobs_per_geometry} jobs for: {namefile}")
    
    # === THROTTLING CHECKPOINT ===
    # Wait for enough slots BEFORE submitting all jobs for this geometry.
    # This is called ONCE per geometry (not per job) to avoid queue saturation.
    print(f"[INFO] Checking available slots in queue...")
    current_queue, wait_iters = wait_for_slot(SLURM_USER, MAX_JOBS_IN_QUEUE, THROTTLE_SLEEP, 
                                               target_jobs_to_submit=min(n_jobs_per_geometry, 50))
    if wait_iters > 0:
        print(f"[INFO] ✓ Queue has space now. Current jobs in queue: {current_queue}/{MAX_JOBS_IN_QUEUE}")
    else:
        print(f"[INFO] Queue status: {current_queue}/{MAX_JOBS_IN_QUEUE} jobs")

    # Collect job IDs for this geometry to use in the merge dependency
    job_ids = []
    base_seed = int(time.time())
    ss = SeedSequence(base_seed)

    # Generate all child seeds at once (more efficient than spawning in loop)
    child_seeds = ss.spawn(n_jobs_per_geometry)

    # === SUBMIT ALL JOBS FOR THIS GEOMETRY ===
    # Number of jobs depends on parameters: n_jobs_per_geometry = total_muons_per_geometry / n_muons_per_job
    for job in range(n_jobs_per_geometry):
        # Convert SeedSequence to integer for use in simulation
        rng = Generator(PCG64(child_seeds[job]))
        seed = rng.integers(0, 2**31 - 1)

        out_raw  = os.path.join(PATH_output_raw,   f"Out_{namefile}_seed{seed}.root")
        out_pre  = os.path.join(PATH_preprocessed, f"Pre_{namefile}_seed{seed}.root")
        out_poca = os.path.join(PATH_poca_output,  f"POCA_{namefile}_seed{seed}.root")
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
echo "[INFO] Running POCA1..."
python3 -u {PATH_data_analysis}/POCA1.py \\
    --input  {out_pre} \\
    --output {out_poca} \\
    --Lpx {Lpx} --Lpy {Lpy} --Lpz {Lpz} \\
    --npx {npx} --npy {npy} --npz {npz} \\
    --dimension {dimension}
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

        # Submit job to queue (no wait here; we already checked space above)
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
            print(f"[SUBMITTED] seed={seed:04d} --> job_id={job_id}")
            jobs_submitted += 1

    # === SUBMIT MERGE JOB (depends on all 60 simulation jobs) ===
    # --dependency=afterok:id1:id2:...:idN ensures merge only starts
    # when ALL simulation jobs finish successfully. If any fails, merge is skipped.
    # SLURM queues the merge automatically, we don't need to wait.

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
    --dimension        {dimension} \\
    --path_poca_output {PATH_poca_output} \\
    --output           {out_merged}
if [ $? -ne 0 ]; then echo "[ERROR] Merge failed. Aborting."; exit 1; fi

echo "[CORRECT] Merge finished for: {namefile}"


### REMOVING RAW POCA POINTS

# echo "[INFO] Removing splitted POCA files for: {namefile}"
# Usamos el prefijo específico para no borrar lo de otros jobs
# rm {PATH_poca_output}/POCA_{namefile}_seed*.root
# echo "[CORRECT] Split POCA files removed for: {namefile}"

echo "[INFO] Cleaning up seed logs..."
sleep 10
rm {PATH_logs}/log_{namefile}_seed*.out
rm {PATH_logs}/log_{namefile}_seed*.err
rm {PATH_logs}/job_{namefile}_seed*.sh
echo "[CORRECT] Cleanup finished."
"""
    with open(merge_sh, "w") as f:
        f.write(merge_script)

    # Submit merge job (SLURM will queue it even if at limit; it will wait for dependencies to complete)
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


print("\n" + "="*70)
print("[FINAL SUMMARY]")
print("="*70)
print(f"[INFO] Configuration:")
print(f"       max_geometries_simulated     = {max_geometries_simulated}")
print(f"       MAX_JOBS_IN_QUEUE           = {MAX_JOBS_IN_QUEUE}")
print(f"       Jobs per geometry           = {n_jobs_per_geometry} simulation + 1 merge")
print(f"[INFO] Results:")
print(f"       Geometries processed        = {i - 1}")
print(f"       Geometries skipped          = {geometries_skipped}")
print(f"       Simulation jobs submitted   = {jobs_submitted}")
print(f"       Simulation jobs failed      = {jobs_failed}")
print(f"       Merge jobs submitted        = {merges_submitted}")
print(f"[INFO] Next step:")
print(f"       Monitor the queue with: squeue -u dominguezs")
print(f"       Check job details with:  scontrol show job <job_id>")
print(f"       Check logs at: {PATH_logs}")
print("="*70)
