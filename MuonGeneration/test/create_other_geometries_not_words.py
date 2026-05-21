"""
Generates geometry JSON files for generic shapes (rectangle, L_shape, T_shape,
sphere, triangle, cylinder) for UNET training.

Edit the VARIATIONS section and run. All combinations are generated automatically.
Shapes that don't fit inside the world volume are skipped silently.
"""

import subprocess, os, sys, time, json
from itertools import product

# ===========================================================================
# CONTROL FLAGS
# ===========================================================================
environment    = "local"   # "local" | "cluster"
dimension      = "2D"      # "2D"   | "3D"
create         = True
force_recreate = False
max_geometries = float("inf")

# ===========================================================================
# PATHS
# ===========================================================================
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
CREATE_SCRIPT = os.path.join(SCRIPT_DIR, "create_other_geometries_not_words.py")
ERROR_LOG     = os.path.join(SCRIPT_DIR, ".shape_errors.json")

PATHS = {
    "local": {
        "json":  "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/geometric_configurations_json",
        "den2D": "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/ground_truth_data/2Dimensions/UNET1",
        "den3D": "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/ground_truth_data/3Dimensions",
        "logs":  "/home/samuel/Work/Muography_Denoising/MuonGeneration/logs",
        "setup": None, "user": None,
    },
    "cluster": {
        "json":  "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/geometric_configurations_json",
        "den2D": "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/ground_truth_data/2Dimensions/UNET1",
        "den3D": "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/ground_truth_data/3Dimensions",
        "logs":  "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs",
        "setup": "/gpfs/users/dominguezs/Muography_Denoising/setup.sh",
        "user":  "dominguezs",
    },
}

if environment not in PATHS:
    sys.exit(f"[ERROR] environment must be one of {list(PATHS.keys())}")

P = PATHS[environment]
for key in ("json", "den2D", "den3D", "logs"):
    os.makedirs(P[key], exist_ok=True)

# ===========================================================================
# WORLD / DETECTOR (fixed)
# ===========================================================================
Lpx, Lpy, Lpz    = 128, 128, 128
npx, npy, npz    = 128, 128, 128
zTop, zBot       =  54, -54







# ===========================================================================
# VARIATIONS  <--- EDIT HERE ################################################
# ===========================================================================
# size_x = width or radius | size_y = height (ignored for sphere)
# depth_z = Z thickness   | cx, cy = XY center of shape

shapes        = ["rectangle", "sphere", "cylinder"]
sizes_x       = [10, 20, 40]
sizes_y       = [10, 20, 40]
depth_z_list  = [2, 5, 10, 20]
center_x_list = [0, 3, 5]
center_y_list = [0, 3, 5]
materials     = ["lead", "iron", "uranium", "steel"]
ratios        = [1]




# ===========================================================================
# HELPERS
# ===========================================================================
def fits_in_world(shape, sx, sy, dz, cx, cy):
    hx, hy, hz = Lpx/2, Lpy/2, Lpz/2
    problems = []
    if dz/2 > hz:
        problems.append(f"dz/2={dz/2} > hz={hz}")
    if shape in ("rectangle", "L_shape", "T_shape", "triangle"):
        if not (-hx <= cx - sx/2 and cx + sx/2 <= hx): problems.append("X out of bounds")
        if not (-hy <= cy - sy/2 and cy + sy/2 <= hy): problems.append("Y out of bounds")
    elif shape == "sphere":
        if cx + sx > hx or cx - sx < -hx: problems.append("sphere X out of bounds")
        if cy + sx > hy or cy - sx < -hy: problems.append("sphere Y out of bounds")
        if sx > hz:                        problems.append("sphere radius > hz")
    elif shape == "cylinder":
        if cx + sx > hx or cx - sx < -hx:   problems.append("cylinder X out of bounds")
        if cy + sy/2 > hy or cy - sy/2 < -hy: problems.append("cylinder Y out of bounds")
    return (not problems), "; ".join(problems)


def make_namefile(shape, sx, sy, dz, cx, cy, mat, ratio):
    return (f"shape_{shape}"
            f"_Lpx{Lpx}_Lpy{Lpy}_Lpz{Lpz}_npx{npx}_npy{npy}_npz{npz}"
            f"_zTop{zTop}_zBot{zBot}"
            f"_ratio{ratio}_sx{sx}_sy{sy}_cx{cx}_cy{cy}_mat{mat}_dz{dz}")


def log_error(name, msg):
    log = json.load(open(ERROR_LOG)) if os.path.exists(ERROR_LOG) else {}
    log[name] = {"error": msg.strip(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(log, open(ERROR_LOG, "w"), indent=2)


def submit(command, i):
    if environment == "local":
        return subprocess.run(command, capture_output=True, text=True)
    # cluster: throttle queue, then sbatch
    MAX_JOBS, SLEEP = 2000, 10
    while True:
        n = len([l for l in subprocess.run(
            ["squeue", "-u", P["user"], "-h", "--format=%i"],
            capture_output=True, text=True).stdout.strip().split('\n') if l])
        if n < MAX_JOBS: break
        print(f"[THROTTLE] {n}/{MAX_JOBS} jobs. Waiting {SLEEP}s..."); time.sleep(SLEEP)
    wrap = f"source {P['setup']} && {' '.join(command)}"
    return subprocess.run([
        "sbatch", f"--job-name=shape_{i}", "--time=01:00:00", "--mem=8G",
        "--cpus-per-task=2", f"--output={P['logs']}/log_shape_{i}.out",
        f"--chdir={P['logs']}", f"--wrap={wrap}", "--partition=wncompute_ifca"
    ], capture_output=True, text=True)


# ===========================================================================
# MAIN LOOP
# ===========================================================================
all_combos = list(product(shapes, sizes_x, sizes_y, depth_z_list,
                          center_x_list, center_y_list, materials, ratios))

valid   = [(s,sx,sy,dz,cx,cy,mat,r) for s,sx,sy,dz,cx,cy,mat,r in all_combos
           if fits_in_world(s,sx,sy,dz,cx,cy)[0]]
invalid = len(all_combos) - len(valid)

print(f"[INFO] Valid geometries: {len(valid)} | Skipped (no fit): {invalid}")

if not create:
    sys.exit("[INFO] create=False. Set it to True to generate files.")

for i, (shape, sx, sy, dz, cx, cy, mat, ratio) in enumerate(valid, start=1):
    if i > max_geometries:
        print(f"[INFO] Reached max_geometries={max_geometries}. Stopping."); break

    name     = make_namefile(shape, sx, sy, dz, cx, cy, mat, ratio)
    out_json = os.path.join(P["json"],  name + ".json")
    out_2D   = os.path.join(P["den2D"], name + "_ground_truth_density2D.npy")
    out_3D   = os.path.join(P["den3D"], name + "_ground_truth_density3D.npy")

    if not force_recreate and os.path.exists(out_json):
        print(f"[SKIP | EXISTS] {i}/{len(valid)} {name}"); continue

    command = [
        "python3", CREATE_SCRIPT,
        "--shape", shape,
        "--Lpx", str(Lpx), "--Lpy", str(Lpy), "--Lpz", str(Lpz),
        "--npx", str(npx), "--npy", str(npy), "--npz", str(npz),
        "--zPosDetector_top", str(zTop), "--zPosDetector_bot", str(zBot),
        "--ratio", str(ratio), "--size_x", str(sx), "--size_y", str(sy),
        "--depth_z_cm", str(dz), "--center_x", str(cx), "--center_y", str(cy),
        "--material", mat, "--dimensions", dimension,
        "--output_json", out_json, "--output2D_density", out_2D, "--output3D_density", out_3D,
    ]

    result = submit(command, i)

    if result.returncode != 0:
        err = result.stderr or result.stdout
        print(f"[ERROR] {i}/{len(valid)} FAILED: {name}\n        {err[:200]}")
        log_error(name, err)
    else:
        print(f"[CORRECT] {i}/{len(valid)} {'created' if environment=='local' else 'submitted'}: {name}")