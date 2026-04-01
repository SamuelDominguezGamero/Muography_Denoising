# -*- coding: utf-8 -*-
import os
import sys

# Forzar modo Batch para evitar que ROOT intente abrir ventanas inexistentes
os.environ['ROOT_TERMINAL_UTILS'] = '0'

import numpy as np
import argparse

try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
except ImportError:
    ROOT = None

# Forzar que los prints salgan SIEMPRE
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)
    print(*args, file=sys.stdout, flush=True, **kwargs)

eprint("*Libraries imported successfully ------ [CORRECT]")

def get_poca_info_ROOT(root_input_file):
    if ROOT is None:
        eprint("Error: ROOT is not available.")
        return None, None

    # Limitar hilos para evitar bloqueo en SLURM
    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    ROOT.EnableImplicitMT(n_cpus)
    eprint(f"Initializing POCA with {n_cpus} cores on: {root_input_file}")

    df = ROOT.RDataFrame("events", root_input_file)

    # Definiciones (tu logica es correcta)
    df = df.Define("dx", "x2 - x1").Define("dy", "y2 - y1").Define("dz", "z2 - z1")
    df = df.Define("A", "dx*vx1 + dy*vy1 + dz*vz1")\
           .Define("B", "vx2*vx1 + vy2*vy1 + vz2*vz1")\
           .Define("C", "vx1*vx1 + vy1*vy1 + vz1*vz1")\
           .Define("D", "dx*vx2 + dy*vy2 + dz*vz2")\
           .Define("E", "vx2*vx2 + vy2*vy2 + vz2*vz2")
    
    df = df.Define("denom", "C*E - B*B")
    df = df.Define("t1", "abs(denom) > 1e-9 ? (A*E - B*D) / denom : 0.0")\
           .Define("t2", "abs(denom) > 1e-9 ? (B*A - C*D) / denom : 0.0")

    df = df.Define("poca_x", "((x1 + t1*vx1) + (x2 + t2*vx2)) / 2.0")\
           .Define("poca_y", "((y1 + t1*vy1) + (y2 + t2*vy2)) / 2.0")\
           .Define("poca_z", "((z1 + t1*vz1) + (z2 + t2*vz2)) / 2.0")

    df = df.Filter("poca_x >= -128.0 && poca_x < 128.0 && poca_y >= -128.0 && poca_y < 128.0 && poca_z >= -64.0 && poca_z < 64.0")
    
    df = df.Define("cos_theta", "B / (sqrt(C) * sqrt(E))")\
           .Define("theta", "acos(fmax(-1.0, fmin(1.0, cos_theta)))")

    eprint("Executing RDataFrame graph...")
    # AddProgressBar a veces da problemas en clusters, lo quitamos para asegurar
    res = df.AsNumpy(columns=["poca_x", "poca_y", "poca_z", "theta"])
    
    eprint(f"POCA applied. Events: {len(res['theta'])}")
    return res, res["theta"]

parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--output", default="output.npy")
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

def main():
    X_LIM, Y_LIM, Z_LIM = 128.0, 128.0, 64.0
    NX, NY, NZ = 256, 256, 128
    
    eprint("Geometry loaded. Starting POCA...")
    poca_dict, theta = get_poca_info_ROOT(args.input)
    if poca_dict is None: return

    # ... (Resto de tu logica de acumulacion es perfecta)
    grid_N = np.zeros((NX, NY, NZ))
    grid_sum = np.zeros((NX, NY, NZ))
    grid_sum_sq = np.zeros((NX, NY, NZ))

    ix = np.clip(((poca_dict["poca_x"] + X_LIM)).astype(int), 0, NX-1)
    iy = np.clip(((poca_dict["poca_y"] + Y_LIM)).astype(int), 0, NY-1)
    iz = np.clip(((poca_dict["poca_z"] + Z_LIM)).astype(int), 0, NZ-1)

    np.add.at(grid_N, (ix, iy, iz), 1)
    np.add.at(grid_sum, (ix, iy, iz), theta)
    np.add.at(grid_sum_sq, (ix, iy, iz), theta**2)

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(args.output, {"n_events": grid_N, "sum_theta": grid_sum, "sum_theta_sq": grid_sum_sq})
    eprint(f"Done. Saved to {args.output}")

if __name__ == "__main__":
    main()
