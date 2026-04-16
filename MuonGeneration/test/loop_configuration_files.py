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

# ===========================================================================
# CONTROL FLAGS
# ===========================================================================
create_geometries = True
simulate          = True    # set to True to submit SLURM jobs (cluster only)
environment       = "cluster"  # "local" or "cluster"
dimension         = "2D"     # 2D or 3D, first we should stick to 2D for faster iterations
max_geometries    = 1        # the first geometries to be tested on
force_resimulate  = False    # set to True to re-process geometries even if merged results exist
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
#   - Sizes 8, 10, 12: Only stroke 1, 2 available
#   - Sizes 14, 16: Stroke 1, 2, 3 available
#   - Only letters available: M, U, O, N

spacings       = [1, 2]
ratios         = [1]
# Strategy: Use multiple sizes with stroke variations that are actually available
FontSizes      = [
    {"size": 8,  "strokes": [1, 2]},
    {"size": 10, "strokes": [1, 2]},
    {"size": 12, "strokes": [1, 2]},
    {"size": 14, "strokes": [1, 2, 3]},  # stroke 3 available
    {"size": 16, "strokes": [1, 2, 3]},  # stroke 3 available
]
materials      = ["lead", "uranium", "iron"]
words_geometry = ["MUON"]

# Count total valid geometries
total_geometries = 0
for spacing in spacings:
    for ratio in ratios:
        for fontsize_dict in FontSizes:
            for material in materials:
                for word in words_geometry:
                    for stroke in fontsize_dict["strokes"]:
                        total_geometries += 1

print(f"[INFO] ----- Total geometries to generate: {total_geometries}")


# ===========================================================================
# SIMULATION PARAMETERS
# ===========================================================================
total_muons_per_geometry = 100_000
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
done_creating = False
for spacing in spacings:
    if not create_geometries or done_creating:
        break
    for ratio in ratios:
        if done_creating:
            break
        for fontsize_dict in FontSizes:
            if done_creating:
                break
            x = fontsize_dict["size"]
            for material in materials:
                if done_creating:
                    break
                for word in words_geometry:
                    if done_creating:
                        break
                    for stroke in fontsize_dict["strokes"]:
                        i += 1
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
                            print(f"[INFO] Geometry {i}/{total_geometries} already exists, skipping: {namefile}")
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
                            inner_command = " ".join(command)
                            
                            full_wrap = f"source {PATH_setup} && {inner_command}"
                            
                            sbatch_args = [
                                "sbatch",
                                f"--job-name=geom_{i}",
                                "--mem=4G",
                                "--time=00:15:00",
                                f"--output={PATH_logs}/log_geom_{i}.out",
                                f"--workdir={PATH_logs}",
                                f"--wrap={full_wrap}"
                            ]
                            
                            result = subprocess.run(sbatch_args, capture_output=True, text=True)
                            
                            print(f"[INFO] Geometry {i}/{total_geometries} creation started: {namefile}")
                        
                        if result.returncode != 0:
                            print(f"[ERROR] Geometry {i}/{total_geometries} failed:\n{result.stderr}")
                        else:
                            print(f"[CORRECT] Geometry {i}/{total_geometries} created: {namefile}")

# if create_geometries:
#     command_GitAdd = f"git add {PATH_geometry_files}/*.json {PATH_density_files}/*_ground_truth_density.npy"
#     subprocess.run(command_GitAdd, shell=True)
#     command_GitCommit = f'git commit -m "Add geometry JSON files and ground truth density tensors"'
#     subprocess.run(command_GitCommit, shell=True)
#     command_GitPush = "git push"
#     subprocess.run(command_GitPush, shell=True)
print("\n[CORRECT] ALL GEOMETRIES CREATED SUCCESSFULLY")
print("="*60 + "\n")


# ===========================================================================
# STEP 2: SLURM JOB SUBMISSION + MERGE WITH DEPENDENCY
# ===========================================================================
print("="*60)
print("STEP 2: SLURM JOB SUBMISSION")
print("="*60)

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

for spacing in spacings:
    for ratio in ratios:
        for fontsize_dict in FontSizes:
            x = fontsize_dict["size"]
            for material in materials:
                for word in words_geometry:
                    for stroke in fontsize_dict["strokes"]:

                        namefile = (
                            f"_Lpx{Lpx}_Lpy{Lpy}_Lpz{Lpz}"
                            f"_npx{npx}_npy{npy}_npz{npz}"
                            f"_zTop{zPosDetector_top}_zBot{zPosDetector_bot}"
                            f"_spacing{spacing}_ratio{ratio}"
                            f"_FontX{x}_FontY{x}"
                            f"_mat{material}_word{word}_stroke{stroke}"
                        )
                        
                        # Check if merged result already exists (skip unless force_resimulate=True)
                        if dimension == "2D":
                            merged_output = os.path.join(PATH_merged_output, f"MERGED_{namefile}_2D.npy")
                        elif dimension == "3D":
                            merged_output = os.path.join(PATH_merged_output, f"MERGED_{namefile}_3D.npy")
                        
                        if os.path.exists(merged_output) and not force_resimulate:
                            print(f"[SKIP] Already processed: {namefile}")
                            continue
                        elif os.path.exists(merged_output) and force_resimulate:
                            print(f"[RESIMULATE] Force flag enabled, re-processing: {namefile}")

                        geometry_file = os.path.join(PATH_geometry_files, namefile + ".json")

                        if not os.path.exists(geometry_file):
                            print(f"[ERROR] Geometry file not found, skipping: {geometry_file}")
                            continue

                        print(f"\n[INFO] Submitting {n_jobs_per_geometry} jobs for: {namefile}")

                        
                        # Collect job IDs for this geometry to use in the merge dependency
                        job_ids = []

                        for job in range(n_jobs_per_geometry):
                            seed = job+1

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
                                #SBATCH --workdir={PATH_logs}
                                #SBATCH --ntasks=1
                                #SBATCH --cpus-per-task=1
                                #SBATCH --mem=4G
                                #SBATCH --time=01:00:00

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

                                # eliminate intermediate files to save space
                                rm {out_pre}
                                echo "[INFO] Preprocessed file removed to save space: {out_pre}"

                                echo "[CORRECT] Job finished: {namefile} | seed={seed}"
                                """
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
                                # Extract job ID from "Submitted batch job 12345"
                                job_id = result.stdout.strip().split()[-1]
                                job_ids.append(job_id)
                                print(f"[SUBMITTED] seed={seed:04d} --> job_id={job_id}")
                                jobs_submitted += 1

                        # ------------------------------------------------------
                        # Submit merge job with dependency on ALL jobs finishing
                        # --dependency=afterok:id1:id2:...:idN means the merge
                        # job only runs if ALL listed jobs finish successfully.
                        # If any job fails, the merge is cancelled automatically.
                        # SLURM handles the waiting automatically, no explicit sleep needed.
                        # ------------------------------------------------------
                        
                        if not job_ids:
                            print(f"[WARNING] No jobs submitted for {namefile}, skipping merge.")
                            continue
                        elif len(job_ids) < n_jobs_per_geometry:
                            print(f"[WARNING] Only {len(job_ids)}/{n_jobs_per_geometry} jobs submitted for {namefile}.")
                            print(f"[WARNING] Merge job will be submitted with dependency on available jobs.")
                        
                        print(f"[INFO] Submitting merge job for: {namefile} with dependency on {len(job_ids)} jobs.")
                        dependency_str = "afterok:" + ":".join(job_ids)
                        
                        if dimension == "2D":
                            out_merged = os.path.join(PATH_merged_output, f"MERGED_{namefile}_2D.npy")
                            png_name = f"{namefile}_2D.png"
                        elif dimension == "3D":
                            out_merged = os.path.join(PATH_merged_output, f"MERGED_{namefile}_3D.npy")
                            png_name = f"{namefile}_3D.png"

                        merge_log  = os.path.join(PATH_logs, f"log_merge_{namefile}.out")
                        merge_err  = os.path.join(PATH_logs, f"log_merge_{namefile}.err")
                        merge_sh   = os.path.join(PATH_logs, f"job_merge_{namefile}.sh")

                        merge_script = f"""#!/bin/bash
                            #SBATCH --job-name=merge_{namefile}
                            #SBATCH --output={merge_log}
                            #SBATCH --error={merge_err}
                            #SBATCH --workdir={PATH_logs}
                            #SBATCH --ntasks=1
                            #SBATCH --cpus-per-task=1
                            #SBATCH --mem=8G
                            #SBATCH --time=01:00:00

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

                        result = subprocess.run(
                            ["sbatch", f"--dependency={dependency_str}", merge_sh],
                            capture_output=True, text=True
                        )

                        if result.returncode != 0:
                            print(f"[ERROR] Merge job submission failed: {result.stderr.strip()}")
                        else:
                            merge_id = result.stdout.strip().split()[-1]
                            print(f"[SUBMITTED] Merge job --> job_id={merge_id} (depends on {len(job_ids)} jobs)")
                            merges_submitted += 1


print("\n" + "="*60)
print(f"[INFO] Simulation jobs submitted: {jobs_submitted}")
print(f"[INFO] Simulation jobs failed:    {jobs_failed}")
print(f"[INFO] Merge jobs submitted:      {merges_submitted}")
print("="*60)
