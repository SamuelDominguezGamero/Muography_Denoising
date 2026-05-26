"""
This file automates the full simulation pipeline:
  1. Creates all geometry files (JSON) for the neural network training dataset.
  2. Submits SLURM jobs for each geometry x seed combination. Each job:
       - Runs the Geant4 Monte Carlo simulation
       - Correlates muon tracks (makeHLTuple.py)
       - Runs the POCA reconstruction (POCA.py)
  3. Submits a merge job per geometry with --dependency=afterok,
     so it only runs when ALL jobs for that geometry finish successfully.

FOR UNET1_2D ---> Predict 2D density maps from 2D XY muon data

USAGE: Modify GEOMETRY VARIATIONS section to customize:
  - spacings, ratios: Grid parameters
  - fontsizes: 8, 10, 12, 14, 16 (14+ may not fit)
  - strokes: 1, 2, 3 (stroke 3 available for all sizes)
  - depth_z_cm_list: Word thickness in cm (efficient single-slab method)
      Example: [2.0, 5.0, 10.0] creates 3 variants with different thicknesses
  - materials: lead, iron, uranium, etc.
  - words_geometry: MUON, MUNO, etc.
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
create_geometries = True
environment       = "local"  # "local" or "cluster"
dimension         = "2D"     # 2D or 3D, first we should stick to 2D for faster iterations
max_geometries    = np.inf     # the first geometries to be tested on


force_resimulate  = False   # set to True to re-process geometries even if merged results exist

# ===========================================================================
# SLURM JOB THROTTLING
# ===========================================================================
# Maximum number of jobs allowed in the queue at the same time for this user.
# Run `sacctmgr show user <username> withassoc` or ask your sysadmin.
# A safe default is to leave ~10% headroom below your real limit.
MAX_JOBS_IN_QUEUE = 2000      # adjust to your cluster's limit
THROTTLE_SLEEP    = 10     # seconds to wait when queue is full before retrying

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
# ERROR TRACKING SYSTEM
# ===========================================================================
# Track which geometries failed during creation, so we can skip them in STEP 2
# Data structure: { "geometry_name": {"error": "error message", "timestamp": "..."} }
ERROR_LOG_FILE = os.path.join(os.path.dirname(__file__), ".geometry_errors.json")

def load_error_log():
    """Load the error log from disk."""
    if os.path.exists(ERROR_LOG_FILE):
        try:
            with open(ERROR_LOG_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_error_log(error_dict):
    """Save the error log to disk."""
    with open(ERROR_LOG_FILE, "w") as f:
        json.dump(error_dict, f, indent=2)

def record_geometry_error(namefile, error_message):
    """Record that a geometry failed during creation."""
    error_log = load_error_log()
    error_log[namefile] = {
        "error": error_message.strip(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_error_log(error_log)
    print(f"[ERROR RECORDED] {namefile} marked as FAILED (logged in {ERROR_LOG_FILE})")

def is_geometry_failed(namefile):
    """Check if a geometry is marked as failed."""
    error_log = load_error_log()
    return namefile in error_log

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
MERGE_SCRIPT           = os.path.join(SCRIPT_DIR, "merge_results_1.py")
PLOT_SCRIPT            = os.path.join(SCRIPT_DIR, "plot_central_slice_comparison.py")

if environment == "cluster":
    PATH_geometry_files = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/geometric_configurations_json"
    PATH_density_files3D  = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/ground_truth_data/3Dimensions"
    PATH_density_files2D  = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/ground_truth_data/2Dimensions/UNET1"
    PATH_output_raw     = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_raw"
    PATH_preprocessed   = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_preprocessed"
    PATH_poca_output    = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/post_POCA_data"
    PATH_merged_output  = "/gpfs/projects/cms/dominguezs/data/merged_poca_data/UNET1"
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
    PATH_logs           = "/home/samuel/Work/Muography_Denoising/MuonGeneration/logs"
    PATH_data_analysis  = "/home/samuel/Work/Muography_Denoising/MuonGeneration/dataAnalysis"
    PATH_generator      = None
    PATH_setup          = None
    SLURM_USER          = None

else:
    sys.exit("[ERROR] environment must be 'local' or 'cluster'.")

# Create output directories if they don't exist
for path in [PATH_geometry_files, PATH_density_files3D, PATH_density_files2D, 
             PATH_output_raw, PATH_preprocessed, PATH_poca_output, PATH_merged_output, 
             PATH_logs, PATH_data_analysis]:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"[INFO] Created directory: {path}")

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
# IMPORTANT: All combinations are now valid. Check bitmaps_letters.py BITMAP_DATA:
#   - Sizes 8, 10, 12, 14, 16: Stroke 1, 2, 3 all available
#   - Only letters available: M, U, O, N

spacings       = [3] 
ratios         = [1, 2]
# Strategy: Use multiple sizes with stroke variations that are actually available
# NOTE: Font 14+ with stroke 3 will show warnings when they don't fit - this is natural and expected
fontsizes      = [10, 16] # All sizes tested; warnings shown if too large
strokes        = [2]
# DEPTH: Variable Z thickness for word geometry in centimeters (efficient method)
# Single slab with configurable thickness - no more inefficient layer repetition!
depth_z_cm_list = [5.0, 15.0, 30.0, 50.0]  # Example: test thin, normal, and thick geometries

# XY/Z offsets in cm. Default [0.0] keeps full backwards compatibility:
# files WITHOUT _xoff/_yoff/_zoff in the filename are treated as offset=(0,0,0) by convention.
# Add non-zero values here to generate offset variants (suffix only appended when any offset != 0).
x_offsets_cm   = [0.0, 10, -10]   # X offset of word center from world center (cm)
y_offsets_cm   = [0.0, 10, -10]   # Y offset of word center from world center (cm)
z_offsets_cm   = [0.0, 10, -10]   # Z position of slab center (cm)

materials      = ["lead", "iron", "uranium", "aluminium", "silicon", "steel"]



words_possible = [
    # 1 letra
    "M", "U", "O", "N",
    # 2 letras
    "MM", "MU", "MO", "MN", "UM", "UU", "UO", "UN", "OM", "OU", "OO", "ON", "NM", "NU", "NO", "NN",
    # 3 letras
    "MMM", "MMU", "MMO", "MMN", "MUM", "MUU", "MUO", "MUN", "MOM", "MOU", "MOO", "MON", "MNM", "MNU", "MNO", "MNN",
    "UMM", "UMU", "UMO", "UMN", "UUM", "UUU", "UUO", "UUN", "UOM", "UOU", "UOO", "UON", "UNM", "UNU", "UNO", "UNN",
    "OMM", "OMU", "OMO", "OMN", "OUM", "OUU", "OUO", "OUN", "OOM", "OOU", "OOO", "OON", "ONM", "ONU", "ONO", "ONN",
    "NMM", "NMU", "NMO", "NMN", "NUM", "NUU", "NUO", "NUN", "NOM", "NOU", "NOO", "NON", "NNM", "NNU", "NNO", "NNN",
    # 4 letras
    "MMMM", "MMMU", "MMMO", "MMMN", "MMUM", "MMUU", "MMUO", "MMUN", "MMOM", "MMOU", "MMOO", "MMON", "MMNM", "MMNU", "MMNO", "MMNN",
    "MUMM", "MUMU", "MUMO", "MUMN", "MUUM", "MUUU", "MUUO", "MUUN", "MUOM", "MUOU", "MUOO", "MUON", "MUNM", "MUNU", "MUNO", "MUNN",
    "MOMM", "MOMU", "MOMO", "MOMN", "MOUM", "MOUU", "MOUO", "MOUN", "MOOM", "MOOU", "MOOO", "MOON", "MONM", "MONU", "MONO", "MONN",
    "MNMM", "MNMU", "MNMO", "MNMN", "MNUM", "MNUU", "MNUO", "MNUN", "MNOM", "MNOU", "MNOO", "MNON", "MNNM", "MNNU", "MNNO", "MNNN",
    "UMMM", "UMMU", "UMMO", "UMMN", "UMUM", "UMUU", "UMUO", "UMUN", "UMOM", "UMOU", "UMOO", "UMON", "UMNM", "UMNU", "UMNO", "UMNN",
    "UUMM", "UUMU", "UUMO", "UUMN", "UUUM", "UUUU", "UUUO", "UUUN", "UUOM", "UUOU", "UUOO", "UUON", "UUNM", "UUNU", "UUNO", "UUNN",
    "UOMM", "UOMU", "UOMO", "UOMN", "UOUM", "UOUU", "UOUO", "UOUN", "UOOM", "UOOU", "UOOO", "UOON", "UONM", "UONU", "UONO", "UONN",
    "UNMM", "UNMU", "UNMO", "UNMN", "UNUM", "UNUU", "UNUO", "UNUN", "UNOM", "UNOU", "UNOO", "UNON", "UNNM", "UNNU", "UNNO", "UNNN",
    "OMMM", "OMMU", "OMMO", "OMMN", "OMUM", "OMUU", "OMUO", "OMUN", "OMOM", "OMOU", "OMOO", "OMON", "OMNM", "OMNU", "OMNO", "OMNN",
    "OUMM", "OUMU", "OUMO", "OUMN", "OUUM", "OUUU", "OUUO", "OUUN", "OUOM", "OUOU", "OUOO", "OUON", "OUNM", "OUNU", "OUNO", "OUNN",
    "OOMM", "OOMU", "OOMO", "OOMN", "OOUM", "OOUU", "OOUO", "OOUN", "OOOM", "OOOU", "OOOO", "OOON", "OONM", "OONU", "OONO", "OONN",
    "ONMM", "ONMU", "ONMO", "ONMN", "ONUM", "ONUU", "ONUO", "ONUN", "ONOM", "ONOU", "ONOO", "ONON", "ONNM", "ONNU", "ONNO", "ONNN",
    "NMMM", "NMMU", "NMMO", "NMMN", "NMUM", "NMUU", "NMUO", "NMUN", "NMOM", "NMOU", "NMOO", "NMON", "NMNM", "NMNU", "NMNO", "NMNN",
    "NUMM", "NUMU", "NUMO", "NUMN", "NUUM", "NUUU", "NUUO", "NUUN", "NUOM", "NUOU", "NUOO", "NUON", "NUNM", "NUNU", "NUNO", "NUNN",
    "NOMM", "NOMU", "NOMO", "NOMN", "NOUM", "NOUU", "NOUO", "NOUN", "NOOM", "NOOU", "NOOO", "NOON", "NONM", "NONU", "NONO", "NONN",
    "NNMM", "NNMU", "NNMO", "NNMN", "NNUM", "NNUU", "NNUO", "NNUN", "NNOM", "NNOU", "NNOO", "NNON", "NNNM", "NNNU", "NNNO", "NNNN"
]

words_geometry = ["M", "U", "O", "N", "MMM", "NNUO", "NOMM", "MUON", "MONU", "OUNN"]


# Count total valid geometries
total_geometries = 0
for material in materials:
    for spacing in spacings:
        for ratio in ratios:
            for fontsize in fontsizes:
                for word in words_geometry:
                    for stroke in strokes:
                        for depth_z_cm in depth_z_cm_list:
                            for x_offset_cm in x_offsets_cm:
                                for y_offset_cm in y_offsets_cm:
                                    for z_offset_cm in z_offsets_cm:
                                        total_geometries += 1

print(f"[INFO] ----- Total geometries to generate: {total_geometries}")
time.sleep(1)
print(60 * "-")
time.sleep(1)
print(60 * "-")
time.sleep(1)


# ===========================================================================
# STEP 1: GEOMETRY CREATION
# ===========================================================================
print("\n" + "="*60)
print("STEP 1: GEOMETRY CREATION")
print("="*60)

i = 0
done_creating = False # control variable to break all loops if max_geometries has been created
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
                        if done_creating:
                            break
                        for depth_z_cm in depth_z_cm_list:
                            if done_creating:
                                break
                            for x_offset_cm in x_offsets_cm:
                                if done_creating:
                                    break
                                for y_offset_cm in y_offsets_cm:
                                    if done_creating:
                                        break
                                    for z_offset_cm in z_offsets_cm:
                                        i += 1 # geometries counting
                                        if i > max_geometries:
                                            print(f"[INFO] Reached max_geometries={max_geometries}. Stopping geometry creation.")
                                            done_creating = True
                                            break

                                        # Offset suffix: only added when any offset is non-zero.
                                        # Files without this suffix have offset=(0,0,0) by convention.
                                        if x_offset_cm != 0.0 or y_offset_cm != 0.0 or z_offset_cm != 0.0:
                                            offset_str = f"_xoff{x_offset_cm:g}_yoff{y_offset_cm:g}_zoff{z_offset_cm:g}"
                                        else:
                                            offset_str = ""

                                        namefile = (
                                            f"_Lpx{Lpx}_Lpy{Lpy}_Lpz{Lpz}"
                                            f"_npx{npx}_npy{npy}_npz{npz}"
                                            f"_zTop{zPosDetector_top}_zBot{zPosDetector_bot}"
                                            f"_spacing{spacing}_ratio{ratio}"
                                            f"_FontX{x}_FontY{x}"
                                            f"_mat{material}_word{word}_stroke{stroke}"
                                            f"_depthZ{int(depth_z_cm)}"
                                            f"{offset_str}"
                                        )

                                        if os.path.exists(os.path.join(PATH_geometry_files, namefile + ".json")):
                                            print(60*'*')
                                            print(f"[INFO | EXISTING] ----- Geometry {i}/{total_geometries} ALREADY EXISTS, skipping: {namefile}")
                                            print(60*'*')
                                            continue

                                        output_json      = os.path.join(PATH_geometry_files, namefile + ".json")

                                        command = [
                                            "python3", CREATE_GEOMETRY_SCRIPT,
                                            "--Lpx", str(Lpx),
                                            "--Lpy", str(Lpy),
                                            "--Lpz", str(Lpz),
                                            "--npx", str(npx),
                                            "--npy", str(npy),
                                            "--npz", str(npz),
                                            "--zPosDetector_top",  str(zPosDetector_top),
                                            "--zPosDetector_bot",  str(zPosDetector_bot),
                                            "--spacing",           str(spacing),
                                            "--ratio",             str(ratio),
                                            "--depth_z_cm",        str(depth_z_cm),
                                            "--x_offset_cm",       str(x_offset_cm),
                                            "--y_offset_cm",       str(y_offset_cm),
                                            "--z_offset_cm",       str(z_offset_cm),
                                            "--FontSizeX",         str(x),
                                            "--FontSizeY",         str(x),
                                            "--material",          material,
                                            "--word_geometry",     word,
                                            "--StrokeWidth",       str(stroke),
                                            "--output_json",       output_json,
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

                                        # Check for errors
                                        if result.returncode != 0:
                                            error_msg = result.stderr if result.stderr else result.stdout

                                            # Check if error is due to geometry not fitting
                                            if "cabe" in error_msg.lower() or "fit" in error_msg.lower():
                                                print(f"\n{'!'*80}")
                                                print(f"{'!'*80}")
                                                print(f"[NO CABE] Geometry {i}/{total_geometries} - TOO LARGE: {namefile}")
                                                print(f"{'!'*80}")
                                                print(f"{'!'*80}\n")
                                            else:
                                                print(f"[ERROR] Geometry {i}/{total_geometries} FAILED: {namefile}")
                                                print(f"        Error: {error_msg[:150]}")

                                            record_geometry_error(namefile, error_msg)
                                            print(80 * '-')
                                        else:
                                            if environment == "local":
                                                print(f"[CORRECT] Geometry {i}/{total_geometries} created: {namefile}")
                                            else:
                                                print(f"[INFO] Geometry {i}/{total_geometries} job submitted: {namefile}")
                                            print(80 * '-')
                                            time.sleep(0.1)



print("="*60)
print("\n[CORRECT] ALL GEOMETRIES CREATED SUCCESSFULLY")
print("="*60 + "\n")
