# -*- coding: utf-8 -*-
"""
POCA (Point Of Closest Approach) algorithm for muon scattering tomography.
Takes a ROOT file with muon trajectories and produces a 3D voxelized grid
with scattering angle statistics per voxel.
"""


print("Iniciando POCA (python entered)")
import os
import sys
print("basic libraries imported")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


print("importing ROOT...")
import ROOT
print("ROOT successfully imported")

print("importing numpy...")
import numpy as np
print("numpy successfully imported")

print("importing argparse ...")
import argparse
print("[CORRECT] ----- ALL LIBRARIES successfully imported")



print("ALL IMPORTS DONE")
ROOT.gROOT.SetBatch(True)  # ← imprescindible en clusters, para que no use interfaz gráfica

print("Libraries imported... [CORRECT]")
# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="POCA voxelization of muon scattering data.")
parser.add_argument("--input",  required=True, help="Input ROOT file with muon trajectory data.")
parser.add_argument("--output", default="output.npy", help="Output .npy file containing a dict with keys 'n_events', 'sum_theta', 'sum_theta_sq'.")
parser.add_argument("--plot",   action="store_true", help="If set, plot the resulting voxel grid.")
parser.add_argument("--Lpx", type=float, default=128.0, help="Physical length of the geometry in X [cm].")
parser.add_argument("--Lpy", type=float, default=128.0, help="Physical length of the geometry in Y [cm].")
parser.add_argument("--Lpz", type=float, default=128.0, help="Physical length of the geometry in Z [cm].")
parser.add_argument("--npx", type=int, default=128, help="Number of voxels in X direction.")
parser.add_argument("--npy", type=int, default=128, help="Number of voxels in Y direction.")
parser.add_argument("--npz", type=int, default=128, help="Number of voxels in Z direction.")


# ---------------------------------------------------------------------------
# POCA computation using ROOT RDataFrame
# ---------------------------------------------------------------------------
def get_poca_info_ROOT(root_input_file, X_LIM, Y_LIM, Z_LIM):
    """
    Computes the POCA point and scattering angle for each muon event.

    The POCA is the midpoint between the closest approach points on the
    incoming (1) and outgoing (2) muon trajectories.

    Each trajectory is defined by a point (x,y,z) and a direction (vx,vy,vz).

    Args:
        root_input_file : path to the ROOT file containing the 'events' TTree.
        X_LIM, Y_LIM, Z_LIM : half-lengths of the physical volume [cm].
                               The volume spans [-X_LIM, X_LIM] x [-Y_LIM, Y_LIM] x [-Z_LIM, Z_LIM].

    Returns:
        res   : dict with arrays for poca_x, poca_y, poca_z, theta.
        theta : scattering angle array [rad].
    """
    df = ROOT.RDataFrame("events", root_input_file)

    # Displacement vector between the two trajectory reference points
    df = df.Define("dx", "x2 - x1") \
           .Define("dy", "y2 - y1") \
           .Define("dz", "z2 - z1")

    # Dot products needed for the POCA parametric equations:
    #   A = (p2-p1)·d1,  B = d2·d1,  C = d1·d1,  D = (p2-p1)·d2,  E = d2·d2
    df = df.Define("A", "dx*vx1 + dy*vy1 + dz*vz1") \
           .Define("B", "vx2*vx1 + vy2*vy1 + vz2*vz1") \
           .Define("C", "vx1*vx1 + vy1*vy1 + vz1*vz1") \
           .Define("D", "dx*vx2 + dy*vy2 + dz*vz2") \
           .Define("E", "vx2*vx2 + vy2*vy2 + vz2*vz2")

    # Denominator: zero when trajectories are parallel (no unique POCA)
    df = df.Define("denom", "C*E - B*B")

    # Parametric distances along each trajectory to the closest approach point
    # t1 = (A*E - B*D) / denom
    # t2 = (B*A - C*D) / denom   [equivalent to -(B*A-C*D)/(B*B-C*E)]
    df = df.Define("t1", "abs(denom) > 1e-9 ? (A*E - B*D) / denom : 0.0") \
           .Define("t2", "abs(denom) > 1e-9 ? (B*A - C*D) / denom : 0.0")

    # Closest approach points on each trajectory
    # POCA = midpoint between P1 and P2
    df = df.Define("poca_x", "((x1 + t1*vx1) + (x2 + t2*vx2)) / 2.0") \
           .Define("poca_y", "((y1 + t1*vy1) + (y2 + t2*vy2)) / 2.0") \
           .Define("poca_z", "((z1 + t1*vz1) + (z2 + t2*vz2)) / 2.0")

    # Keep only POCA points that fall inside the physical volume
    df = df.Filter(
        f"poca_x >= {-X_LIM} && poca_x < {X_LIM} &&"
        f"poca_y >= {-Y_LIM} && poca_y < {Y_LIM} &&"
        f"poca_z >= {-Z_LIM} && poca_z < {Z_LIM}"
    )

    # Scattering angle between incoming and outgoing trajectories
    df = df.Define("cos_theta", "B / (sqrt(C) * sqrt(E))") \
           .Define("theta", "acos(fmax(-1.0, fmin(1.0, cos_theta)))")

    print("Executing RDataFrame graph...")
    res = df.AsNumpy(columns=["poca_x", "poca_y", "poca_z", "theta"])
    print(f"POCA applied. Events after volume filter: {len(res['theta'])}")

    return res, res["theta"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parser.parse_args()

    # Half-lengths of the physical volume [cm]
    X_LIM = args.Lpx / 2.0
    Y_LIM = args.Lpy / 2.0
    Z_LIM = args.Lpz / 2.0

    # Voxel grid dimensions
    NX, NY, NZ = args.npx, args.npy, args.npz

    print(f"Volume: [{-X_LIM},{X_LIM}] x [{-Y_LIM},{Y_LIM}] x [{-Z_LIM},{Z_LIM}] cm")
    print(f"Grid:   {NX} x {NY} x {NZ} voxels")
    print(f"Voxel size: {2*X_LIM/NX:.3f} x {2*Y_LIM/NY:.3f} x {2*Z_LIM/NZ:.3f} cm")

    print("\nStarting POCA computation...")
    poca_dict, theta = get_poca_info_ROOT(args.input, X_LIM, Y_LIM, Z_LIM)
    if poca_dict is None:
        return

    # Accumulation grids
    grid_N       = np.zeros((NX, NY, NZ))  # number of muons per voxel
    grid_sum     = np.zeros((NX, NY, NZ))  # sum of scattering angles
    grid_sum_sq  = np.zeros((NX, NY, NZ))  # sum of squared scattering angles (for variance)

    # Convert physical coordinates to voxel indices.
    # Formula: ix = (poca_x - (-X_LIM)) / (2*X_LIM) * NX
    # This correctly handles any combination of physical size and voxel count.
    ix = np.clip(((poca_dict["poca_x"] + X_LIM) / (2*X_LIM) * NX).astype(int), 0, NX-1)
    iy = np.clip(((poca_dict["poca_y"] + Y_LIM) / (2*Y_LIM) * NY).astype(int), 0, NY-1)
    iz = np.clip(((poca_dict["poca_z"] + Z_LIM) / (2*Z_LIM) * NZ).astype(int), 0, NZ-1)

    # Accumulate statistics into the grids
    np.add.at(grid_N,      (ix, iy, iz), 1)
    np.add.at(grid_sum,    (ix, iy, iz), theta)
    np.add.at(grid_sum_sq, (ix, iy, iz), theta**2)

    # Save output
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(args.output, {
        "n_events":     grid_N,
        "sum_theta":    grid_sum,
        "sum_theta_sq": grid_sum_sq
    })
    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()
