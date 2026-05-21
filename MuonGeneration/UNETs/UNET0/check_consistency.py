#!/usr/bin/env python3
"""
Validación exhaustiva de la pipeline UNET0
Chequea todos los requisitos antes de ejecutar pipeline_data.py
"""

import sys
import json
import subprocess
from pathlib import Path
from collections import defaultdict

# Colores para output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_header(title):
    print(f"\n{BOLD}{'='*80}")
    print(f"{title}")
    print(f"{'='*80}{RESET}\n")

def check_mark(passed):
    return f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"

def print_status(name, passed, message=""):
    status = check_mark(passed)
    msg = f" → {message}" if message else ""
    print(f"  {status} {name}{msg}")

# ============================================================================
# 1. PYTHON ENVIRONMENT
# ============================================================================
def check_python_env():
    print_header("1. PYTHON ENVIRONMENT")
    
    # Python version
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}"
    print(f"  Python: {version}")
    
    # Required packages
    required = {
        "numpy": "NumPy for arrays",
        "h5py": "HDF5 for datasets",
        "pathlib": "Path utilities (built-in)",
        "json": "JSON parsing (built-in)"
    }
    
    optional = {
        "matplotlib": "Visualization",
        "tensorflow": "DNN training",
        "ROOT": "ROOT file reading",
        "uproot": "Alternative ROOT reader"
    }
    
    print("\n  {BOLD}Required Packages:{RESET}".format(BOLD=BOLD, RESET=RESET))
    missing_required = []
    for pkg, desc in required.items():
        try:
            __import__(pkg)
            print_status(pkg, True, desc)
        except ImportError:
            print_status(pkg, False, desc)
            if pkg not in ["pathlib", "json"]:
                missing_required.append(pkg)
    
    print("\n  {BOLD}Optional Packages:{RESET}".format(BOLD=BOLD, RESET=RESET))
    for pkg, desc in optional.items():
        try:
            __import__(pkg)
            print_status(pkg, True, desc)
        except ImportError:
            print_status(pkg, False, f"{desc} (not installed)")
    
    # Check uproot as alternative for ROOT
    try:
        import uproot
        print(f"  {GREEN}✓{RESET} uproot available (can read .root without ROOT)")
    except ImportError:
        print(f"  {YELLOW}⊘{RESET} uproot not installed (install if using TAR with .root files)")
    
    return len(missing_required) == 0

# ============================================================================
# 2. PATHS
# ============================================================================
def check_paths():
    print_header("2. PATHS")
    
    BASE = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration")
    DATA = BASE / "data"
    
    paths_to_check = {
        "BASE (MuonGeneration)": BASE,
        "DATA": DATA,
        "geometric_configurations_json": DATA / "geometric_configurations_json",
        "simulation_data (contains TAR or root_files)": DATA / "simulation_data",
        "ground_truth_data": DATA / "ground_truth_data",
    }
    
    all_exist = True
    for name, path in paths_to_check.items():
        exists = path.exists()
        print_status(name, exists, str(path.relative_to(BASE)) if BASE.exists() else str(path))
        all_exist = all_exist and exists
    
    return all_exist, BASE, DATA

# ============================================================================
# 3. INPUT FILES
# ============================================================================
def check_input_files(BASE, DATA):
    print_header("3. INPUT FILES")
    
    # Check JSON files
    json_dir = DATA / "geometric_configurations_json"
    json_files = []
    if json_dir.exists():
        json_files = list(json_dir.glob("*.json"))
    
    print(f"  JSON files: {len(json_files)}")
    print_status("geometric_configurations_json not empty", len(json_files) > 0, 
                 f"{len(json_files)} .json files found")
    
    # Check for sample JSON structure
    if json_files:
        sample_json = json_files[0]
        try:
            with open(sample_json) as f:
                data = json.load(f)
                has_voxels = "TheVoxels" in data
                print_status("Sample JSON has 'TheVoxels' key", has_voxels, 
                           f"File: {sample_json.name}")
        except Exception as e:
            print_status("Sample JSON is valid", False, str(e))
    
    # Check POCA source (TAR or root_files directory)
    tar_file = DATA / "simulation_data/run0.tar"
    root_dir = DATA / "simulation_data/root_files"
    
    poca_source_exists = False
    poca_type = "None"
    
    if tar_file.exists():
        poca_source_exists = True
        poca_type = "TAR"
        size_gb = tar_file.stat().st_size / (1024**3)
        print_status("POCA source (run0.tar) exists", True, f"{size_gb:.2f} GB")
    elif root_dir.exists():
        root_files = list(root_dir.glob("*.root"))
        poca_source_exists = len(root_files) > 0
        poca_type = "Directory"
        print_status("POCA source (root_files/) exists", poca_source_exists, 
                   f"{len(root_files)} .root files")
    else:
        print_status("POCA source (TAR or root_files/)", False, 
                   "Neither run0.tar nor root_files/ found")
    
    return len(json_files) > 0 and poca_source_exists

# ============================================================================
# 4. IMPORTS & MODULES
# ============================================================================
def check_imports(BASE):
    print_header("4. IMPORTS & MODULES")
    
    # Add UNET0 dir to path
    unet_dir = BASE / "UNETs/UNET0"
    sys.path.insert(0, str(unet_dir))
    
    # Try importing data_utils
    try:
        import data_utils
        print_status("data_utils imports successfully", True)
        
        # Check main functions
        functions = ["extract_poca_from_source", "create_gt_from_json", 
                    "match_poca_with_gt", "create_h5_dataset"]
        all_present = True
        for func in functions:
            has_func = hasattr(data_utils, func)
            print_status(f"  └─ {func}", has_func)
            all_present = all_present and has_func
        
        return all_present
    except Exception as e:
        print_status("data_utils imports successfully", False, str(e))
        return False

# ============================================================================
# 5. OUTPUT SPACE
# ============================================================================
def check_output_space(DATA):
    print_header("5. DISK SPACE")
    
    import shutil
    
    stat = shutil.disk_usage(str(DATA))
    total_gb = stat.total / (1024**3)
    free_gb = stat.free / (1024**3)
    
    # Estimate required space
    # - POCA: ~5-20 GB (many 128x128x3 images)
    # - GT: ~1-5 GB
    # - H5: ~2-10 GB
    estimated_gb = 30  # Conservative estimate
    
    print(f"  Total disk: {total_gb:.1f} GB")
    print(f"  Available: {free_gb:.1f} GB")
    print(f"  Estimated needed: ~{estimated_gb} GB (POCA + GT + H5)")
    
    space_ok = free_gb > estimated_gb
    print_status("Enough disk space", space_ok)
    
    return space_ok

# ============================================================================
# 6. PIPELINE SCRIPTS
# ============================================================================
def check_pipeline_scripts(BASE):
    print_header("6. PIPELINE SCRIPTS")
    
    unet_dir = BASE / "UNETs/UNET0"
    scripts = {
        "data_utils.py": "Core utilities",
        "pipeline_data.py": "Main orchestrator",
        "checkear_instancias_fromh5.py": "H5 visualization",
        "train_unet.py": "UNET training",
        "check_consistency.py": "This script"
    }
    
    all_present = True
    for script, desc in scripts.items():
        path = unet_dir / script
        exists = path.exists()
        print_status(script, exists, desc)
        all_present = all_present and exists
    
    return all_present

# ============================================================================
# MAIN
# ============================================================================
def main():
    print(f"\n{BOLD}VALIDACIÓN EXHAUSTIVA - UNET0 PIPELINE{RESET}")
    print(f"{BOLD}Verificando todos los requisitos antes de ejecutar{RESET}\n")
    
    results = {}
    
    # Run all checks
    results["Python Env"] = check_python_env()
    paths_ok, BASE, DATA = check_paths()
    results["Paths"] = paths_ok
    
    if paths_ok:
        results["Input Files"] = check_input_files(BASE, DATA)
        results["Imports"] = check_imports(BASE)
        results["Disk Space"] = check_output_space(DATA)
        results["Scripts"] = check_pipeline_scripts(BASE)
    else:
        print(f"\n{RED}Cannot proceed: Base paths don't exist{RESET}")
        results["Input Files"] = False
        results["Imports"] = False
        results["Disk Space"] = False
        results["Scripts"] = False
    
    # Summary
    print_header("RESUMEN")
    
    for check, passed in results.items():
        status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
        print(f"  {status}: {check}")
    
    all_passed = all(results.values())
    
    print(f"\n{BOLD}{'='*80}{RESET}")
    if all_passed:
        print(f"{GREEN}{BOLD}✓ TODO OK - Pipeline lista para ejecutar{RESET}")
        print(f"\n  Próximo paso:")
        print(f"    python3 pipeline_data.py")
    else:
        print(f"{RED}{BOLD}✗ ERRORES DETECTADOS - Revisar arriba{RESET}")
        print(f"\n  Soluciones:")
        if not results.get("Python Env"):
            print(f"    pip install numpy h5py")
        if not results.get("Input Files"):
            print(f"    Verificar: {DATA}/geometric_configurations_json/")
            print(f"    Verificar: {DATA}/simulation_data/run0.tar")
    
    print(f"{BOLD}{'='*80}{RESET}\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
