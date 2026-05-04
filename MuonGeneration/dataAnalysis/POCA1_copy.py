# -*- coding: utf-8 -*-
"""
POCA (Point Of Closest Approach) algorithm for muon scattering tomography.
Takes a ROOT file with muon trajectories and produces a 3D voxelized grid
with scattering angle statistics per voxel.

FOR UNET1_2D!!!
"""
import ROOT
import argparse
import numpy as np

ROOT.gROOT.SetBatch(True)  # imprescindible en cluster, para que no use interfaz gráfica

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
parser.add_argument("--channel", type=str, default="theta", choices=["theta"], help="Which scattering variable to accumulate in the voxels. Currently only 'theta' (scattering angle) is implemented.")

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


    df = df.Define("t1", "(A*E-B*D)/(C*E-B*B)") \
           .Define("t2", "-(B*A-C*D)/(B*B-C*E)")

    # Closest approach points on each trajectory
    # POCA = midpoint between P1 and P2
    df = df.Define("poca_x", "((x1 + t1*vx1) + (x2 + t2*vx2)) / 2.0") \
           .Define("poca_y", "((y1 + t1*vy1) + (y2 + t2*vy2)) / 2.0") \
           .Define("poca_z", "((z1 + t1*vz1) + (z2 + t2*vz2)) / 2.0")

    # Keep only POCA points that fall inside the physical volume
    df = df.Filter(
        f"poca_x >= {-X_LIM} && poca_x <= {X_LIM} &&"
        f"poca_y >= {-Y_LIM} && poca_y <= {Y_LIM} &&"
        f"poca_z >= {-Z_LIM} && poca_z <= {Z_LIM}"
    )

    # Scattering angle between incoming and outgoing trajectories
    df = df.Define("cos_theta", "B / (sqrt(C) * sqrt(E))") \
           .Define("theta", "acos(fmax(-1.0, fmin(1.0, cos_theta)))")
    

    poca_and_theta = df.AsNumpy(columns=["theta", "poca_x", "poca_y", "poca_z"])
    return poca_and_theta


poca_and_theta = get_poca_info_ROOT(args.input, X_LIM, Y_LIM, Z_LIM)

poca_x = poca_and_theta["poca_x"]
poca_y = poca_and_theta["poca_y"]
poca_z = poca_and_theta["poca_z"]
theta    = poca_and_theta["theta"]



## Channels (2d)
# 2d counts


# max theta² por celda xy (integrado en z)


# distribución de theta









np.save(args.output, poca_and_theta)






