# -*- coding: utf-8 -*-
"""
POCA (Point Of Closest Approach) algorithm for muon scattering tomography.
Takes a ROOT file with muon trajectories and produces a 3D voxelized grid
with scattering angle statistics per voxel.

FOR UNET1_2D!!!
"""


print("[INFO] ----- Iniciando POCA (python entered)")
import os
import sys
print("[CORRECT] ----- basic libraries imported")


import ROOT
import numpy as np
import argparse
print("[CORRECT] ----- ALL LIBRARIES successfully imported")


ROOT.gROOT.SetBatch(True)  # ← imprescindible en clusters, para que no use interfaz gráfica

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="POCA voxelization of muon scattering data.")
parser.add_argument("--input",  required=True, help="Input ROOT file with muon trajectory data.")
parser.add_argument("--output", default="output", help="Output .npy file containing a dict with keys 'n_events', 'sum_theta', 'sum_theta_sq'.")
parser.add_argument("--plot",   action="store_true", help="If set, plot the resulting voxel grid.")
parser.add_argument("--Lpx", type=float, default=128.0, help="Physical length of the geometry in X [cm].")
parser.add_argument("--Lpy", type=float, default=128.0, help="Physical length of the geometry in Y [cm].")
parser.add_argument("--Lpz", type=float, default=128.0, help="Physical length of the geometry in Z [cm].")
parser.add_argument("--npx", type=int, default=128, help="Number of voxels in X direction.")
parser.add_argument("--npy", type=int, default=128, help="Number of voxels in Y direction.")
parser.add_argument("--npz", type=int, default=128, help="Number of voxels in Z direction.")

parser.add_argument("--dimension", type=str, default="2D", choices=["2D","3D"], help="Dimensionality of the voxel grid (2D or 3D). If 2, only X and Y dimensions are used for voxelization.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# GEOMETRY INSTANTIATION
# ---------------------------------------------------------------------------
# Half-lengths of the physical volume [cm]
X_LIM = args.Lpx / 2.0
Y_LIM = args.Lpy / 2.0
Z_LIM = args.Lpz / 2.0

# Voxel grid dimensions
NX, NY, NZ = args.npx, args.npy, args.npz

print(f"Volume: [{-X_LIM},{X_LIM}] x [{-Y_LIM},{Y_LIM}] x [{-Z_LIM},{Z_LIM}] cm")
print(f"Grid:   {NX} x {NY} x {NZ} voxels")
print(f"Voxel size: {2*X_LIM/NX:.3f} x {2*Y_LIM/NY:.3f} x {2*Z_LIM/NZ:.3f} cm")



# ---------------------------------------------------------------------------
# POCA computation using ROOT RDataFrame
# ---------------------------------------------------------------------------
def get_poca_info_ROOT(root_input_file, X_LIM, Y_LIM, Z_LIM):
    """
    root_input_file structure:


    Computes the POCA point and scattering angle for each muon event.

    The POCA is the midpoint between the closest approach points on the
    incoming (1) and outgoing (2) muon trajectories.

    Each trajectory is defined by a point (x,y,z) and a direction (vx,vy,vz).

    Args:
        root_input_file : path to the ROOT file containing the 'events' TTree.
        X_LIM, Y_LIM, Z_LIM : half-lengths of the physical volume [cm].
                               The volume spans [-X_LIM, X_LIM] x [-Y_LIM, Y_LIM] x [-Z_LIM, Z_LIM].

    Returns:
        matrix_counts     : 3D numpy array with shape (npy, npx, npz) containing event counts per voxel.
        matrix_sum_theta  : 3D numpy array with shape (npy, npx, npz) containing sum of scattering angles per voxel.
        matrix_sum_theta_sq : 3D numpy array with shape (npy, npx, npz) containing sum of squared scattering angles per voxel.
    """
    df = ROOT.RDataFrame("events", root_input_file)
    num_events_input = df.Count().GetValue()
    print(f"[INFO] ----- Total input events: {num_events_input}")

    

    # tenemos que invertir el sentido de vz; por como está construido makeHLTuple.py
    df = df.Define("vz1_neg", "vz1").Define("vz2_neg", "vz2")
    df = df.Redefine("vz1", "-vz1_neg").Redefine("vz2", "-vz2_neg")



    # Print para observar los datos
    columns_=["x1", "y1", "z1", "vx1", "vy1", "vz1", "x2", "y2", "z2", "vx2", "vy2", "vz2"]
    res = df.AsNumpy(columns=columns_)
    print("\n--- First 20 rows ---")
    print(f"{'x1':>8} {'y1':>8} {'z1':>8} {'vx1':>8} {'vy1':>8} {'vz1':>8} {'x2':>8} {'y2':>8} {'z2':>8} {'vx2':>8} {'vy2':>8} {'vz2':>8}")

    for i in range(20):
        print(f"{res['x1'][i]:>8.2f} {res['y1'][i]:>8.2f} {res['z1'][i]:>8.2f} {res['vx1'][i]:>8.2f} {res['vy1'][i]:>8.2f} {res['vz1'][i]:>8.2f} {res['x2'][i]:>8.2f} {res['y2'][i]:>8.2f} {res['z2'][i]:>8.2f} {res['vx2'][i]:>8.2f} {res['vy2'][i]:>8.2f} {res['vz2'][i]:>8.2f}")

       



    # Displacement vector between the two trajectory reference points
    df = df.Define("dx", "x2 - x1") \
           .Define("dy", "y2 - y1") \
           .Define("dz", "z2 - z1")

    # Dot products needed for the POCA parametric equations:
    #   A = (p2-p1)·d1,  B = d2·d1,  C = d1·d1,  D = (p2-p1)·d2,  E = d2·d2
    df = df.Define("A", "dx*vx1 + dy*vy1   + dz*vz1") \
           .Define("B", "vx2*vx1 + vy2*vy1 + vz2*vz1") \
           .Define("C", "vx1*vx1 + vy1*vy1 + vz1*vz1") \
           .Define("D", "dx*vx2 + dy*vy2   + dz*vz2") \
           .Define("E", "vx2*vx2 + vy2*vy2 + vz2*vz2")

    # Denominator: zero when trajectories are parallel (no unique POCA)
    df = df.Define("denom", "C*E - B*B")
    
    df = df.Define("t1", "(A*E - B*D) / denom") \
           .Define("t2", "(B*A - C*D) / denom")


    # Closest approach points on each trajectory
    # POCA = midpoint between P1 and P2
    df = df.Define("poca_x", "((x1 + t1*vx1) + (x2 + t2*vx2)) / 2.0") \
           .Define("poca_y", "((y1 + t1*vy1) + (y2 + t2*vy2)) / 2.0") \
           .Define("poca_z", "((z1 + t1*vz1) + (z2 + t2*vz2)) / 2.0")


    # Scattering angle between incoming and outgoing trajectories
    df = df.Define("cos_theta", "B / (sqrt(C) * sqrt(E))") \
           .Define("theta", "acos(fmax(-1.0, fmin(1.0, cos_theta)))")
    
    # Filters of Poca points that are located outside the boundaries of the geometry
    df = df.Filter(
    f"fabs(poca_x) <= {X_LIM} && " # fabs en C++ es tomar valor absoluto para floats, abs es usado históricamente para ints
    f"fabs(poca_y) <= {Y_LIM} && "
    f"fabs(poca_z) <= {Z_LIM}"
)

    

    print("[INFO] ------ POCA points outside the geometry have been filtered \n")




#     print("[INFO] ----- Current: Voxelization of POCA points...")
    df = df.Define("voxel_x", f"int(fmin({args.npx}-1, fmax(0, (poca_x + {X_LIM}) / (2*{X_LIM} / {args.npx}))))") \
           .Define("voxel_y", f"int(fmin({args.npy}-1, fmax(0, (poca_y + {Y_LIM}) / (2*{Y_LIM} / {args.npy}))))") \
           .Define("voxel_z", f"int(fmin({args.npz}-1, fmax(0, (poca_z + {Z_LIM}) / (2*{Z_LIM} / {args.npz}))))")

    export_columns = ["theta", "poca_x", "poca_y", "poca_z"]
    df.Snapshot("events", f"{args.output}", export_columns)
    num_events = df.Count().GetValue()
    print("[CORRECT] ----- POCA points and scattering angles saved to ROOT file.")
    print(f"[INFO] ----- TOTAL VALID POCA POINTS: {num_events}")
get_poca_info_ROOT(args.input, X_LIM, Y_LIM, Z_LIM)



