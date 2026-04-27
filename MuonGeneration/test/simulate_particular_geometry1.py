"""
Simulate ONE PARTICULAR GEOMETRY - replica of loop_configuration_files1.py logic
FOR UNET1_2D ---> Predict 2D density maps from 2D XY muon data
"""

import subprocess
import os
import sys
import time
import glob
import argparse
from numpy.random import Generator, PCG64, SeedSequence

# ===========================================================================
# PARSE ARGUMENTS
# ===========================================================================
parser = argparse.ArgumentParser(description="Simulate a particular geometry")
parser.add_argument("--Lpx", type=int, default=128)
parser.add_argument("--Lpy", type=int, default=128)
parser.add_argument("--Lpz", type=int, default=128)
parser.add_argument("--npx", type=int, default=128)
parser.add_argument("--npy", type=int, default=128)
parser.add_argument("--npz", type=int, default=128)
parser.add_argument("--spacing", type=int, default=5)
parser.add_argument("--ratio", type=int, default=1)
parser.add_argument("--fontsize", type=int, default=12)
parser.add_argument("--material", type=str, required=True)
parser.add_argument("--word", type=str, required=True)
parser.add_argument("--stroke", type=int, default=2)
parser.add_argument("--total-muons", type=int, default=900000)
parser.add_argument("--muons-per-job", type=int, default=30000)
parser.add_argument("--dimension", type=str, default="2D", choices=["2D", "3D"])
parser.add_argument("--environment", type=str, default="cluster", choices=["cluster", "local"])

args = parser.parse_args()

# ===========================================================================
# CONFIGURATION
# ===========================================================================
Lpx, Lpy, Lpz = args.Lpx, args.Lpy, args.Lpz
npx, npy, npz = args.npx, args.npy, args.npz
spacing, ratio = args.spacing, args.ratio
fontsize, material, word, stroke = args.fontsize, args.material, args.word, args.stroke
dimension = args.dimension
environment = args.environment

total_muons_per_geometry = args.total_muons
n_muons_per_job = args.muons_per_job
n_jobs_per_geometry = total_muons_per_geometry // n_muons_per_job

MAX_JOBS_IN_QUEUE = 500
THROTTLE_SLEEP = 30

# ===========================================================================
# PATHS
# ===========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CREATE_GEOMETRY_SCRIPT = os.path.join(SCRIPT_DIR, "create_geometry.py")
MERGE_SCRIPT = os.path.join(SCRIPT_DIR, "merge_results_1.py")

if environment == "cluster":
    PATH_geometry_files = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/geometric_configurations_json"
    PATH_density_files2D = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/ground_truth_data/2Dimensions/UNET1"
    PATH_output_raw = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_raw"
    PATH_preprocessed = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_preprocessed"
    PATH_poca_output = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/post_POCA_data"
    PATH_merged_output = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/merged_poca_data/UNET1"
    PATH_logs = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs"
    PATH_data_analysis = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/dataAnalysis"
    PATH_generator = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration-build/Generator"
    PATH_setup = "/gpfs/users/dominguezs/Muography_Denoising/setup.sh"
    SLURM_USER = "dominguezs"
else:
    PATH_geometry_files = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/geometric_configurations_json"
    PATH_density_files2D = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/ground_truth_data/2Dimensions/UNET1"
    PATH_output_raw = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/data_raw"
    PATH_preprocessed = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/data_preprocessed"
    PATH_poca_output = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/post_POCA_data"
    PATH_merged_output = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/merged_poca_data/UNET1"
    PATH_logs = "/home/samuel/Work/Muography_Denoising/MuonGeneration/logs"
    PATH_data_analysis = "/home/samuel/Work/Muography_Denoising/MuonGeneration/dataAnalysis"
    PATH_generator = None
    PATH_setup = None
    SLURM_USER = None

# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================
def get_current_job_count(username):
    if environment != "cluster":
        return 0
    result = subprocess.run(["squeue", "-u", username, "-h", "--format=%i"],
                          capture_output=True, text=True)
    lines = [l for l in result.stdout.strip().split('\n') if l]
    return len(lines)

def wait_for_slot(username, max_jobs=MAX_JOBS_IN_QUEUE, sleep=THROTTLE_SLEEP):
    if environment != "cluster":
        return 0
    while True:
        count = get_current_job_count(username)
        if count < max_jobs:
            return count
        print(f"[THROTTLE] Queue full ({count}/{max_jobs}). Waiting {sleep}s...")
        time.sleep(sleep)

# ===========================================================================
# BUILD GEOMETRY NAME
# ===========================================================================
namefile = (
    f"_Lpx{Lpx}_Lpy{Lpy}_Lpz{Lpz}"
    f"_npx{npx}_npy{npy}_npz{npz}"
    f"_zTop54_zBot-54"
    f"_spacing{spacing}_ratio{ratio}"
    f"_FontX{fontsize}_FontY{fontsize}"
    f"_mat{material}_word{word}_stroke{stroke}"
)

geometry_file = os.path.join(PATH_geometry_files, namefile + ".json")

print("\n" + "="*70)
print("SIMULATE PARTICULAR GEOMETRY")
print("="*70)
print(f"[INFO] Geometry: {namefile}")
print(f"[INFO] Total muons: {total_muons_per_geometry:,}")
print(f"[INFO] Jobs: {n_jobs_per_geometry}")
print()

# ===========================================================================
# CHECK IF ALREADY SIMULATED
# ===========================================================================
if dimension == "2D":
    out_merged = os.path.join(PATH_merged_output, f"MERGED_{namefile}_Muons_{total_muons_per_geometry}_2D.npy")
else:
    out_merged = os.path.join(PATH_merged_output, f"MERGED_{namefile}_Muons_{total_muons_per_geometry}_3D.npy")

if os.path.exists(out_merged):
    print(f"[SKIP] Already simulated: {namefile}")
    sys.exit(0)

# Create directories
for path in [PATH_logs, PATH_output_raw, PATH_preprocessed, PATH_poca_output, PATH_merged_output]:
    os.makedirs(path, exist_ok=True)

# ===========================================================================
# CHECK IF GEOMETRY EXISTS
# ===========================================================================
if not os.path.exists(geometry_file):
    print(f"[ERROR] Geometry NOT FOUND: {geometry_file}")
    sys.exit(1)

print(f"[CORRECT] Geometry exists!")

# ===========================================================================
# SUBMIT SIMULATION JOBS
# ===========================================================================
print("\n" + "="*70)
print("SUBMITTING SIMULATION JOBS")
print("="*70)
print(f"[INFO] Submitting {n_jobs_per_geometry} jobs for: {namefile}\n")

job_ids = []
base_seed = int(time.time())
ss = SeedSequence(base_seed)
child_seeds = ss.spawn(n_jobs_per_geometry)

for job in range(n_jobs_per_geometry):
    rng = Generator(PCG64(child_seeds[job]))
    seed = rng.integers(0, 2**31 - 1)

    out_raw = os.path.join(PATH_output_raw, f"Out_{namefile}_seed{seed}.root")
    out_pre = os.path.join(PATH_preprocessed, f"Pre_{namefile}_seed{seed}.root")
    out_poca = os.path.join(PATH_poca_output, f"POCA_{namefile}_seed{seed}.npy")
    out_log = os.path.join(PATH_logs, f"log_{namefile}_seed{seed}.out")
    out_err = os.path.join(PATH_logs, f"log_{namefile}_seed{seed}.err")
    out_sh = os.path.join(PATH_logs, f"job_{namefile}_seed{seed}.sh")

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

rm {out_raw}
echo "[INFO] Raw file removed: {out_raw}"

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

if [ -f "{out_pre}" ]; then
    rm -f "{out_pre}"
    echo "[✓] Preprocessed file removed: {out_pre}"
fi

echo "[CORRECT] Job finished: {namefile} | seed={seed}"
"""
    with open(out_sh, "w") as f:
        f.write(job_script)

    current_count = wait_for_slot(SLURM_USER, MAX_JOBS_IN_QUEUE, THROTTLE_SLEEP)

    result = subprocess.run(["sbatch", "--begin=now", out_sh], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] sbatch failed: seed={seed} | {result.stderr.strip()}")
    else:
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)
        print(f"[SUBMITTED] seed={seed:10d} --> job_id={job_id} (queue: {current_count+1}/{MAX_JOBS_IN_QUEUE})")

if not job_ids:
    print("[ERROR] No jobs were submitted!")
    sys.exit(1)

print(f"\n[INFO] Submitted {len(job_ids)} simulation jobs")

# ===========================================================================
# SUBMIT MERGE JOB WITH DEPENDENCY
# ===========================================================================
print("\n" + "="*70)
print("SUBMITTING MERGE JOB")
print("="*70)

dependency_str = "afterok:" + ":".join(job_ids)

merge_log = os.path.join(PATH_logs, f"log_merge_{namefile}.out")
merge_err = os.path.join(PATH_logs, f"log_merge_{namefile}.err")
merge_sh = os.path.join(PATH_logs, f"job_merge_{namefile}.sh")

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

echo "[INFO] Removing splitted POCA files for: {namefile}"
rm {PATH_poca_output}/POCA_{namefile}_seed*.npy
echo "[CORRECT] Split POCA files removed for: {namefile}"

echo "[INFO] Cleaning up intermediate logs..."
sleep 5
rm {PATH_logs}/log_{namefile}_seed*.out
rm {PATH_logs}/log_{namefile}_seed*.err
rm {PATH_logs}/job_{namefile}_seed*.sh

echo "[CORRECT] Cleanup finished."
"""

with open(merge_sh, "w") as f:
    f.write(merge_script)

current_count = wait_for_slot(SLURM_USER, MAX_JOBS_IN_QUEUE, THROTTLE_SLEEP)

result = subprocess.run(
    ["sbatch", "--begin=now", f"--dependency={dependency_str}", merge_sh],
    capture_output=True, text=True
)

if result.returncode != 0:
    print(f"[ERROR] Merge submission failed: {result.stderr.strip()}")
else:
    merge_id = result.stdout.strip().split()[-1]
    print(f"[SUBMITTED] Merge job --> job_id={merge_id}")
    print(f"[INFO] Will execute after {len(job_ids)} simulation jobs complete")

# ===========================================================================
# SUMMARY
# ===========================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"[INFO] Geometry: {namefile}")
print(f"[INFO] Simulation jobs: {len(job_ids)}")
print(f"[INFO] Merge job: 1")
print(f"[INFO] Output: {out_merged}")
print(f"[INFO] Logs: {PATH_logs}")
print("="*70 + "\n")
