#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate a particular geometry configuration.
Accepts geometry parameters, finds the corresponding JSON config,
and runs full simulation pipeline with SLURM job management.
"""

import subprocess
import os
import sys
import time
import glob
import argparse
from pathlib import Path
from numpy.random import Generator, PCG64, SeedSequence

# ===========================================================================
# ENVIRONMENT & PATHS
# ===========================================================================
ENVIRONMENT = "cluster"  # "local" or "cluster"
DIMENSION = "2D"  # Currently only 2D supported

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CREATE_GEOMETRY_SCRIPT = os.path.join(SCRIPT_DIR, "create_geometry.py")
MERGE_SCRIPT = os.path.join(SCRIPT_DIR, "merge_results_1.py")

if ENVIRONMENT == "cluster":
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
    MAX_JOBS_IN_QUEUE = 500
    THROTTLE_SLEEP = 30

elif ENVIRONMENT == "local":
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
def get_current_job_count(username=SLURM_USER):
    """Returns the number of jobs currently in the SLURM queue for the user."""
    if ENVIRONMENT != "cluster":
        return 0
    result = subprocess.run(
        ["squeue", "-u", username, "-h", "--format=%i"],
        capture_output=True, text=True
    )
    lines = [l for l in result.stdout.strip().split('\n') if l]
    return len(lines)

def wait_for_slot(username=SLURM_USER, max_jobs=MAX_JOBS_IN_QUEUE, sleep=THROTTLE_SLEEP):
    """Blocks until there is room in the SLURM queue."""
    if ENVIRONMENT != "cluster":
        return 0
    while True:
        count = get_current_job_count(username)
        if count < max_jobs:
            return count
        print(f"[THROTTLE] Queue full ({count}/{max_jobs} jobs). Waiting {sleep}s...")
        time.sleep(sleep)

# ===========================================================================
# ARGUMENT PARSING
# ===========================================================================
parser = argparse.ArgumentParser(
    description="Simulate a particular geometry configuration.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python3 simulate_particular_geometry.py --Lpx 128 --material uranium --word MUON --stroke 2 --total-muons 900000 --muons-per-job 30000
  python3 simulate_particular_geometry.py --material steel --word OUMN --fontsize 10 --total-muons 500000 --muons-per-job 50000
    """
)

# Geometry parameters
parser.add_argument("--Lpx", type=int, default=128, help="Physical length X [cm]")
parser.add_argument("--Lpy", type=int, default=128, help="Physical length Y [cm]")
parser.add_argument("--Lpz", type=int, default=128, help="Physical length Z [cm]")
parser.add_argument("--npx", type=int, default=128, help="Voxels in X")
parser.add_argument("--npy", type=int, default=128, help="Voxels in Y")
parser.add_argument("--npz", type=int, default=128, help="Voxels in Z")
parser.add_argument("--spacing", type=int, default=5, help="Spacing parameter")
parser.add_argument("--ratio", type=int, default=1, help="Ratio parameter")
parser.add_argument("--fontsize", type=int, default=12, help="Font size (8, 10, 12, 14, 16)")
parser.add_argument("--material", type=str, required=True, help="Material (steel, uranium, aluminium, iron, lead)")
parser.add_argument("--word", type=str, required=True, help="Word geometry (MUON, MUNO, etc.)")
parser.add_argument("--stroke", type=int, default=2, help="Stroke width (1, 2)")

# Simulation parameters
parser.add_argument("--total-muons", type=int, default=900000, help="Total muons per geometry")
parser.add_argument("--muons-per-job", type=int, default=30000, help="Muons per SLURM job")
parser.add_argument("--no-create", action="store_true", help="Don't ask to create if geometry doesn't exist")
parser.add_argument("--force-resimulate", action="store_true", help="Re-simulate even if merged results exist")

args = parser.parse_args()

# ===========================================================================
# VALIDATE FONTSIZE & STROKE COMBINATION
# ===========================================================================
VALID_COMBINATIONS = {
    8: [1, 2, 3],      # stroke 3 fits in small fonts
    10: [1, 2, 3],
    12: [1, 2, 3],
    14: [1, 2],        # stroke 3 doesn't fit in large fonts
    16: [1, 2]
}

if args.fontsize not in VALID_COMBINATIONS:
    print(f"[ERROR] fontsize {args.fontsize} not supported. Valid sizes: {list(VALID_COMBINATIONS.keys())}")
    sys.exit(1)

if args.stroke not in VALID_COMBINATIONS[args.fontsize]:
    valid_strokes = VALID_COMBINATIONS[args.fontsize]
    print(f"[ERROR] stroke {args.stroke} not valid for fontsize {args.fontsize}")
    print(f"[INFO] Valid strokes for fontsize {args.fontsize}: {valid_strokes}")
    print(f"\nValid combinations:")
    for size, strokes in VALID_COMBINATIONS.items():
        print(f"  fontsize {size}: strokes {strokes}")
    sys.exit(1)

# ===========================================================================
# BUILD GEOMETRY NAME
# ===========================================================================
namefile = (
    f"_Lpx{args.Lpx}_Lpy{args.Lpy}_Lpz{args.Lpz}"
    f"_npx{args.npx}_npy{args.npy}_npz{args.npz}"
    f"_zTop54_zBot-54"
    f"_spacing{args.spacing}_ratio{args.ratio}"
    f"_FontX{args.fontsize}_FontY{args.fontsize}"
    f"_mat{args.material}_word{args.word}_stroke{args.stroke}"
)

geometry_json = os.path.join(PATH_geometry_files, namefile + ".json")

print("\n" + "="*70)
print("SIMULATE PARTICULAR GEOMETRY")
print("="*70)
print(f"\n[INFO] Geometry: {namefile}")
print(f"[INFO] Looking for: {geometry_json}")

# ===========================================================================
# CHECK IF GEOMETRY EXISTS
# ===========================================================================
if not os.path.exists(geometry_json):
    print(f"\n[ERROR] Geometry file NOT FOUND: {geometry_json}")
    
    if args.no_create:
        print("[ERROR] --no-create flag set. Exiting.")
        sys.exit(1)
    
    # Ask user if they want to create it
    response = input("\nWould you like to CREATE this geometry? (yes/no): ").strip().lower()
    
    if response != "yes":
        print("[INFO] Exiting without creating.")
        sys.exit(0)
    
    # Create the geometry
    print(f"\n[INFO] Creating geometry: {namefile}")
    
    output_density2D = os.path.join(PATH_density_files2D, namefile + "_ground_truth_density2D.npy")
    output_density3D = os.path.join(PATH_density_files2D, namefile + "_ground_truth_density3D.npy")
    
    command = [
        "python3", CREATE_GEOMETRY_SCRIPT,
        "--Lpx", str(args.Lpx),
        "--Lpy", str(args.Lpy),
        "--Lpz", str(args.Lpz),
        "--npx", str(args.npx),
        "--npy", str(args.npy),
        "--npz", str(args.npz),
        "--zPosDetector_top", "54",
        "--zPosDetector_bot", "-54",
        "--spacing", str(args.spacing),
        "--ratio", str(args.ratio),
        "--FontSizeX", str(args.fontsize),
        "--FontSizeY", str(args.fontsize),
        "--material", args.material,
        "--word_geometry", args.word,
        "--StrokeWidth", str(args.stroke),
        "--output_json", geometry_json,
        "--dimensions", DIMENSION,
        "--output2D_density", output_density2D,
        "--output3D_density", output_density3D,
    ]
    
    if ENVIRONMENT == "local":
        # Local execution
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] Geometry creation failed:\n{result.stderr}")
            sys.exit(1)
        print(f"[CORRECT] Geometry created successfully!")
    else:
        # Cluster: submit as SLURM job
        print(f"[INFO] Submitting geometry creation job to SLURM...")
        current_count = wait_for_slot(SLURM_USER, MAX_JOBS_IN_QUEUE, THROTTLE_SLEEP)
        
        inner_command = " ".join(command)
        full_wrap = f"source {PATH_setup} && {inner_command}"
        
        geom_log = os.path.join(PATH_logs, f"log_geom_{namefile}.out")
        
        sbatch_args = [
            "sbatch",
            f"--job-name=geom",
            "--time=01:00:00",
            "--mem=8G",
            "--cpus-per-task=2",
            f"--output={geom_log}",
            f"--wrap={full_wrap}",
            "--partition=wncompute_ifca"
        ]
        
        result = subprocess.run(sbatch_args, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] Geometry job submission failed:\n{result.stderr}")
            sys.exit(1)
        
        geom_job_id = result.stdout.strip().split()[-1]
        print(f"[CORRECT] Geometry job submitted with ID: {geom_job_id}")
        print(f"[INFO] Waiting for geometry creation to complete...")
        print(f"[INFO] Log: {geom_log}")
        print(f"[INFO] (You can check status with: sacct -j {geom_job_id})")
        
        # Note: We don't wait for it to complete - user can check with sacct
        time.sleep(2)
else:
    print(f"[CORRECT] Geometry exists!")

# ===========================================================================
# SIMULATION PARAMETERS
# ===========================================================================
total_muons_per_geometry = args.total_muons
n_muons_per_job = args.muons_per_job
n_jobs_per_geometry = total_muons_per_geometry // n_muons_per_job

print(f"\n[INFO] Simulation Parameters:")
print(f"  Total muons per geometry: {total_muons_per_geometry:,}")
print(f"  Muons per job: {n_muons_per_job:,}")
print(f"  Number of jobs: {n_jobs_per_geometry}")
print()

# ===========================================================================
# CHECK IF ALREADY SIMULATED
# ===========================================================================
if DIMENSION == "2D":
    merged_output = os.path.join(PATH_merged_output, f"MERGED_{namefile}_Muons_{total_muons_per_geometry}_2D.npy")
elif DIMENSION == "3D":
    merged_output = os.path.join(PATH_merged_output, f"MERGED_{namefile}_Muons_{total_muons_per_geometry}_3D.npy")

if os.path.exists(merged_output) and not args.force_resimulate:
    print(f"[SKIP] Already simulated with {total_muons_per_geometry:,} muons")
    print(f"[INFO] Output: {merged_output}")
    sys.exit(0)
elif os.path.exists(merged_output) and args.force_resimulate:
    print(f"[RESIMULATE] Force flag enabled. Cleaning old files...")
    # Clean old files
    old_poca = glob.glob(os.path.join(PATH_poca_output, f"POCA_{namefile}_seed*.npy"))
    for f in old_poca:
        os.remove(f)
    os.remove(merged_output)
    print(f"[CORRECT] Cleaned old files.")

# ===========================================================================
# CREATE OUTPUT DIRECTORIES
# ===========================================================================
os.makedirs(PATH_logs, exist_ok=True)
os.makedirs(PATH_output_raw, exist_ok=True)
os.makedirs(PATH_preprocessed, exist_ok=True)
os.makedirs(PATH_poca_output, exist_ok=True)
os.makedirs(PATH_merged_output, exist_ok=True)

# ===========================================================================
# SUBMIT SLURM JOBS
# ===========================================================================
print("="*70)
print("SUBMITTING SLURM JOBS")
print("="*70)
print(f"\n[INFO] Submitting {n_jobs_per_geometry} simulation jobs for: {namefile}\n")

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
#SBATCH --job-name=sim_{job:02d}
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
    --input  {geometry_json} \\
    --output {out_raw} \\
    --number {n_muons_per_job} \\
    --seed   {seed}
if [ $? -ne 0 ]; then echo "[ERROR] Geant4 failed. Aborting."; exit 1; fi
echo "[CORRECT] Geant4 done."

# --- 2nd: Track correlation ---
echo "[INFO] Running makeHLTuple..."
python3 -u {PATH_data_analysis}/makeHLTuple.py \\
    --input  {out_raw} \\
    --conf   {geometry_json} \\
    --output {out_pre}
if [ $? -ne 0 ]; then echo "[ERROR] makeHLTuple failed. Aborting."; exit 1; fi
echo "[CORRECT] makeHLTuple done."

# Clean raw file
rm {out_raw}
echo "[INFO] Raw file removed: {out_raw}"

# --- 3rd: POCA reconstruction ---
echo "[INFO] Running POCA1..."
python3 -u {PATH_data_analysis}/POCA1.py \\
    --input  {out_pre} \\
    --output {out_poca} \\
    --Lpx {args.Lpx} --Lpy {args.Lpy} --Lpz {args.Lpz} \\
    --npx {args.npx} --npy {args.npy} --npz {args.npz} \\
    --dimension {DIMENSION}
if [ $? -ne 0 ]; then echo "[ERROR] POCA failed. Aborting."; exit 1; fi
echo "[CORRECT] POCA done."

# Cleanup preprocessed file
if [ -f "{out_pre}" ]; then
    rm -f "{out_pre}"
    echo "[✓] Preprocessed file removed: {out_pre}"
fi

echo "[CORRECT] Job finished: {namefile} | seed={seed}"
"""
    with open(out_sh, "w") as f:
        f.write(job_script)

    # Throttle
    current_count = wait_for_slot(SLURM_USER, MAX_JOBS_IN_QUEUE, THROTTLE_SLEEP)

    result = subprocess.run(
        ["sbatch", "--begin=now", out_sh],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"[ERROR] sbatch failed: seed={seed} | {result.stderr.strip()}")
    else:
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)
        print(f"[SUBMITTED] job {job+1:02d}/{n_jobs_per_geometry} --> seed={seed} --> job_id={job_id} (queue: {current_count+1}/{MAX_JOBS_IN_QUEUE})")

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

if DIMENSION == "2D":
    png_name = f"{namefile}_2D.png"
elif DIMENSION == "3D":
    png_name = f"{namefile}_3D.png"

merge_log = os.path.join(PATH_logs, f"log_merge_{namefile}.out")
merge_err = os.path.join(PATH_logs, f"log_merge_{namefile}.err")
merge_sh = os.path.join(PATH_logs, f"job_merge_{namefile}.sh")

merge_script = f"""#!/bin/bash
#SBATCH --job-name=merge
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
    --npx              {args.npx} \\
    --npy              {args.npy} \\
    --npz              {args.npz} \\
    --dimension        {DIMENSION} \\
    --path_poca_output {PATH_poca_output} \\
    --output           {merged_output}
if [ $? -ne 0 ]; then echo "[ERROR] Merge failed."; exit 1; fi

echo "[CORRECT] Merge finished for: {namefile}"

echo "[INFO] Removing split POCA files..."
rm {PATH_poca_output}/POCA_{namefile}_seed*.npy

echo "[INFO] Cleaning up logs..."
sleep 5
rm {PATH_logs}/log_{namefile}_seed*.out
rm {PATH_logs}/log_{namefile}_seed*.err
rm {PATH_logs}/job_{namefile}_seed*.sh

echo "[CORRECT] Cleanup finished."
"""

with open(merge_sh, "w") as f:
    f.write(merge_script)

# Throttle before merge
current_count = wait_for_slot(SLURM_USER, MAX_JOBS_IN_QUEUE, THROTTLE_SLEEP)

result = subprocess.run(
    ["sbatch", "--begin=now", f"--dependency={dependency_str}", merge_sh],
    capture_output=True, text=True
)

if result.returncode != 0:
    print(f"[ERROR] Merge submission failed: {result.stderr.strip()}")
else:
    merge_id = result.stdout.strip().split()[-1]
    print(f"\n[SUBMITTED] Merge job --> job_id={merge_id}")
    print(f"[INFO] Will execute after {len(job_ids)} simulation jobs complete")

# ===========================================================================
# SUMMARY
# ===========================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"[INFO] Geometry: {namefile}")
print(f"[INFO] Simulation jobs submitted: {len(job_ids)}")
print(f"[INFO] Merge job submitted: 1")
print(f"[INFO] Output will be: {merged_output}")
print(f"[INFO] Logs available in: {PATH_logs}")
print("="*70 + "\n")
