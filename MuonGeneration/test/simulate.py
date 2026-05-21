3;1~"""
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
simulate          = True       # set to True to submit SLURM jobs (cluster only)
environment       = "cluster"  # "local" or "cluster"
dimension         = "2D"       # 2D or 3D, first we should stick to 2D for faster iterations
force_resimulate  = True      # set to True to re-process geometries even if merged results exist
max_geometries_simulated = 1 
simulate_just_one_geometry = True
namefile_to_simulate = "EMPTY.json"
print_skips = False

# ===== FILTER BY DEPTH_Z (NEW GEOMETRIES) =====
filter_by_depthZ       = False  # Set to True to filter geometries by depthZ value
depthZ_list_to_simulate = [1, 2, 5, 10, 20]  # Only simulate geometries with these depthZ values (e.g., [2, 5, 10])

# ===========================================================================
# SLURM JOB THROTTLING
# ===========================================================================
MAX_JOBS_IN_QUEUE = 1200      # adjust to your cluster's limit
THROTTLE_SLEEP    = 10        # seconds to wait when queue is full before retrying

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
MERGE_HITS_SCRIPT      = os.path.join(SCRIPT_DIR, "merge_hits.py")
MERGE_POCA_SCRIPT      = os.path.join(SCRIPT_DIR, "merge_poca.py")

if environment == "cluster":
    PATH_geometry_files      = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/geometric_configurations_json"
    PATH_output_raw          = "/gpfs/projects/cms/dominguezs/data/0_raw_geant4"
    PATH_preprocessed        = "/gpfs/projects/cms/dominguezs/data/1_post_makeHLT"
    PATH_merged_post_makeHLT = "/gpfs/projects/cms/dominguezs/data/2_merged_post_makeHLT"
    PATH_poca_output         = "/gpfs/projects/cms/dominguezs/data/3_merged_poca"
    PATH_logs                = "/gpfs/projects/cms/dominguezs/data/logs"
    PATH_data_analysis       = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/dataAnalysis"
    PATH_generator           = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration-build/Generator"
    PATH_setup               = "/gpfs/users/dominguezs/Muography_Denoising/setup.sh"
    SLURM_USER               = "dominguezs"

elif environment == "local":
    PATH_geometry_files      = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/geometric_configurations_json"
    PATH_output_raw          = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/data_raw"
    PATH_preprocessed        = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/data_preprocessed"
    PATH_merged_post_makeHLT = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/data_merged_hits"
    PATH_poca_output         = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/post_POCA_data"
    PATH_logs                = "/home/samuel/Work/Muography_Denoising/MuonGeneration/logs"
    PATH_data_analysis       = "/home/samuel/Work/Muography_Denoising/MuonGeneration/dataAnalysis"
    PATH_generator           = None
    PATH_setup               = None
    SLURM_USER               = None

else:
    sys.exit("[ERROR] environment must be 'local' or 'cluster'.")

print(f"[INFO] Environment: {environment}")
print(f"[INFO] Script directory: {SCRIPT_DIR}")
if simulate_just_one_geometry:
    print(f"[INFO] Looking for geometry containing: {namefile_to_simulate}")
if filter_by_depthZ:
    print(f"[INFO] Filtering by depthZ: {depthZ_list_to_simulate}")
print(f"[INFO] Geometry files path: {PATH_geometry_files}")
print(f"[INFO] Output paths:")
print(f"       Raw:                     {PATH_output_raw}")
print(f"       Preprocessed (hits):     {PATH_preprocessed}")
print(f"       Merged hits:             {PATH_merged_post_makeHLT}")
print(f"       POCA output:             {PATH_poca_output}")
print(f"[INFO] Logs path: {PATH_logs}")
print(f"[INFO] SLURM user: {SLURM_USER}")
print(f"[INFO] SLURM throttling: max {MAX_JOBS_IN_QUEUE} jobs in queue, waiting {THROTTLE_SLEEP}s when full")
print(60 * "=")
time.sleep(1)
print(60 * "=")
time.sleep(1)
print(60 * "=")
print("\n")
print("\n")


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
total_muons_per_geometry = 750_000
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


os.makedirs(PATH_logs,                  exist_ok=True)
os.makedirs(PATH_output_raw,            exist_ok=True)
os.makedirs(PATH_preprocessed,          exist_ok=True)
os.makedirs(PATH_merged_post_makeHLT,   exist_ok=True)
os.makedirs(PATH_poca_output,           exist_ok=True)

jobs_submitted = 0
jobs_failed    = 0
merges_submitted = 0
geometries_skipped = 0
geometries_skipped_by_depthZ = 0
geometries_failed_during_creation = 0


# FIX: glob needs a wildcard pattern to find files inside the directory
all_json_files = []
for file in glob.glob(os.path.join(PATH_geometry_files, "*.json")):
    all_json_files.append(file)
print(f"[INFO] ----- Total number of geometry json files available for simulation: {len(all_json_files)}")





# ===== MAIN LOOP: Process each geometry =====
i = 0
for file in all_json_files:
    i += 1
    if not simulate_just_one_geometry:
        print(f"[INFO] --- Several geometries are going to be simulated.")
        if i > max_geometries_simulated:
            print(f"[INFO] Reached max_geometries_simulated={max_geometries_simulated}. Stopping simulation.")
            break
    else:
        if namefile_to_simulate not in file:
            if print_skips:
                print(f"[SKIP] Looking for specific geometry: {namefile_to_simulate}. Skipping: {file}")
            continue
        else:
            print(60 * "=")
            print(f"[INFO] Found geometry to simulate: {file}")
            print(f"[INFO] Starting simulation for: {file}")
            print(60 * "=")

    # Extract geometry name and full path
    namefile = os.path.splitext(os.path.basename(file))[0]
    geometry_file = file # full path with extension

    # === FILTER BY DEPTHZ (if enabled) ===
    if filter_by_depthZ:
        matches_depthZ = False
        for depthZ in depthZ_list_to_simulate:
            if f"_depthZ{depthZ}" in namefile:
                matches_depthZ = True
                break
        if not matches_depthZ:
            if print_skips:
                print(f"[SKIP] depthZ filter: {namefile} not in {depthZ_list_to_simulate}")
            geometries_skipped_by_depthZ += 1
            continue
        else:
            print(f"[MATCH] depthZ filter matched: {namefile}")

    # Check if merged result already exists WITH THE EXACT NUMBER OF MUONS
    # Try both new format (_Muons_) and legacy format (for backwards compatibility)
    if dimension == "2D":
        merged_poca_output = os.path.join(PATH_poca_output, f"POCA_merged_{namefile}_Muons_{total_muons_per_geometry}_2D.root")
    elif dimension == "3D":
        merged_poca_output = os.path.join(PATH_poca_output, f"POCA_merged_{namefile}_Muons_{total_muons_per_geometry}_3D.root")

    merged_exists = os.path.exists(merged_poca_output)


    # Skip if already processed
    if merged_exists and not force_resimulate:
        print(f"[SKIP] Already processed: {namefile} with {total_muons_per_geometry:,} muons")
        geometries_skipped += 1
        continue
    elif merged_exists and force_resimulate:
        print(f"[RESIMULATE] Force flag enabled, re-processing: {namefile}")
        # Clean old files before re-simulating
        old_hits = glob.glob(os.path.join(PATH_merged_post_makeHLT, f"Pre_merged_{namefile}*.root"))
        for f in old_hits:
            os.remove(f)
        old_poca = glob.glob(os.path.join(PATH_poca_output, f"POCA_merged_{namefile}*.root"))
        for f in old_poca:
            os.remove(f)
        os.remove(merged_poca_output)  # Remove old merged result
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

# # Remove preprocessed file
# if [ -f "{out_pre}" ]; then
#     rm -f "{out_pre}" 2>/dev/null
#     if [ $? -eq 0 ]; then
#         echo "[✓] Preprocessed file removed: {out_pre}"
#     else
#         echo "[✗] Failed to remove preprocessed file: {out_pre}"
#     fi
# fi

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
        out_merged_hits = os.path.join(PATH_merged_post_makeHLT, f"Pre_merged_{namefile}.root")
        out_merged_poca = os.path.join(PATH_poca_output, f"POCA_merged_{namefile}.root")
    elif dimension == "3D":
        out_merged_hits = os.path.join(PATH_merged_post_makeHLT, f"Pre_merged_{namefile}.root")
        out_merged_poca = os.path.join(PATH_poca_output, f"POCA_merged_{namefile}.root")

    merge_log  = os.path.join(PATH_logs, f"log_merge_{namefile}.out")
    merge_err  = os.path.join(PATH_logs, f"log_merge_{namefile}.err")
    merge_sh   = os.path.join(PATH_logs, f"job_merge_{namefile}.sh")

    merge_script = f"""#!/bin/bash
#SBATCH --job-name=merge_{namefile}
#SBATCH --output={merge_log}
#SBATCH --error={merge_err}
#SBATCH --partition=wncompute_ifca
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source {PATH_setup}

echo "[INFO] Starting merge pipeline for: {namefile}"

# --- 1st: Merge all Pre_*.root files ---
echo "[INFO] Running merge_hits.py..."
python3 -u {MERGE_HITS_SCRIPT} \\
    --namefile          {namefile} \\
    --n_jobs            {n_jobs_per_geometry} \\
    --path_hits_input   {PATH_preprocessed} \\
    --path_hits_output  {PATH_merged_post_makeHLT} \\
    --output            {out_merged_hits}
if [ $? -ne 0 ]; then echo "[ERROR] merge_hits.py failed. Aborting."; exit 1; fi
echo "[CORRECT] merge_hits.py done."

# --- 2nd: Run POCA1.py on merged hits ---
echo "[INFO] Running POCA1.py on merged hits..."
python3 -u {PATH_data_analysis}/POCA1.py \\
    --input     {out_merged_hits} \\
    --output    {out_merged_poca} \\
    --Lpx {Lpx} --Lpy {Lpy} --Lpz {Lpz} \\
    --npx {npx} --npy {npy} --npz {npz} \\
    --dimension {dimension}
if [ $? -ne 0 ]; then echo "[ERROR] POCA1.py failed. Aborting."; exit 1; fi
echo "[CORRECT] POCA1.py done."

# --- 3rd: Cleanup intermediate files ---
echo "[INFO] Cleaning up intermediate files..."

# Remove individual Pre_*.root files
rm {PATH_preprocessed}/Pre_{namefile}_seed*.root
echo "[CORRECT] Individual Pre_*.root files removed."

echo "[INFO] Cleanup finished."
echo "[CORRECT] Merge pipeline finished for: {namefile}"
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
if simulate_just_one_geometry:
    print(f"       (Just one geometry simulated: {namefile_to_simulate})")
if filter_by_depthZ:
    print(f"       Geometries skipped (depthZ) = {geometries_skipped_by_depthZ}")
print(f"       Geometries skipped (total)  = {geometries_skipped}")
print(f"       Simulation jobs submitted   = {jobs_submitted}")
print(f"       Simulation jobs failed      = {jobs_failed}")
print(f"       Merge jobs submitted        = {merges_submitted}")
print(f"[INFO] Next step:")
print(f"       Monitor the queue with: squeue -u dominguezs")
print(f"       Check job details with:  scontrol show job <job_id>")
print(f"       Check logs at: {PATH_logs}")
print("="*70)


print(f"[INFO] ----- saved at {PATH_poca_output}")
