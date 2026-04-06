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
parser.add_argument("--output", default="output.npy", help="Output .npy file containing a dict with keys 'n_events', 'sum_theta', 'sum_theta_sq'.")
parser.add_argument("--plot",   action="store_true", help="If set, plot the resulting voxel grid.")
parser.add_argument("--Lpx", type=float, default=128.0, help="Physical length of the geometry in X [cm].")
parser.add_argument("--Lpy", type=float, default=128.0, help="Physical length of the geometry in Y [cm].")
parser.add_argument("--Lpz", type=float, default=128.0, help="Physical length of the geometry in Z [cm].")
parser.add_argument("--npx", type=int, default=128, help="Number of voxels in X direction.")
parser.add_argument("--npy", type=int, default=128, help="Number of voxels in Y direction.")
parser.add_argument("--npz", type=int, default=128, help="Number of voxels in Z direction.")
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
        matrix_std_theta : 3D numpy array with shape (npy, npx, npz) containing std of scattering angle per voxel.
                          Indexed as [iy, ix, iz] following standard numpy image convention.
        matrix_counts     : 3D numpy array with shape (npy, npx, npz) containing event counts per voxel.
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
    

    print("[INFO] ----- Current: Voxelization of POCA points...")
    df = df.Define("voxel_x", f"int(fmin({args.npx}-1, fmax(0, (poca_x + {X_LIM}) / (2*{X_LIM} / {args.npx}))))") \
           .Define("voxel_y", f"int(fmin({args.npy}-1, fmax(0, (poca_y + {Y_LIM}) / (2*{Y_LIM} / {args.npy}))))") \
           .Define("voxel_z", f"int(fmin({args.npz}-1, fmax(0, (poca_z + {Z_LIM}) / (2*{Z_LIM} / {args.npz}))))")


    res = df.AsNumpy(columns=["theta", "voxel_x", "voxel_y", "voxel_z"])
    
    n_events_filtered = len(res['theta'])
    print(f"POCA aplicado. Eventos tras filtro: {n_events_filtered}")

    # 2. Inicializar matrices con Numpy
    matrix_counts = np.zeros((args.npy, args.npx, args.npz))
    matrix_sum_theta = np.zeros((args.npy, args.npx, args.npz))
    matrix_sum_theta_sq = np.zeros((args.npy, args.npx, args.npz))

    # 3. Llenar matrices usando los índices de los vóxeles
    # Obtenemos los arrays para evitar accesos repetidos al diccionario
    v_x = res['voxel_x'].astype(int)
    v_y = res['voxel_y'].astype(int)
    v_z = res['voxel_z'].astype(int)
    theta = res['theta']

    # Bucle eficiente en memoria (puedes usar np.add.at para ser aún más rápido)
    np.add.at(matrix_counts, (v_y, v_x, v_z), 1)
    np.add.at(matrix_sum_theta, (v_y, v_x, v_z), theta)
    np.add.at(matrix_sum_theta_sq, (v_y, v_x, v_z), theta**2)

    # 4. Calcular std_theta (sustituye lo que hacía df_voxels)
    # Fórmula: std = sqrt( (sum_sq / N) - (sum/N)^2 )
    with np.errstate(divide='ignore', invalid='ignore'):
        mean = matrix_sum_theta / matrix_counts
        variance = (matrix_sum_theta_sq / matrix_counts) - (mean**2)
        # Limpiar posibles negativos ínfimos por precisión flotante
        variance = np.maximum(0, variance)
        matrix_std_theta = np.sqrt(variance)
        # Poner 0 donde no hay eventos
        matrix_std_theta[matrix_counts == 0] = 0.0

    return matrix_std_theta, matrix_counts, matrix_sum_theta, matrix_sum_theta_sq


# important comment: we are counting from the bottom-left corner of the volume
# the spatial grid is indexed as follows (standard numpy image convention):
# matrix[iy, ix, iz] where:
#   - iy: row index (Y coordinate, 0 at bottom with origin='lower')
#   - ix: column index (X coordinate, 0 at left)
#   - iz: depth index (Z coordinate)

matrix_std_theta, matrix_counts, matrix_sum_theta, matrix_sum_theta_sq = get_poca_info_ROOT(args.input, X_LIM, Y_LIM, Z_LIM)


if args.plot:
    import matplotlib.pyplot as plt
    plt.imshow(matrix_std_theta[:,:,0], origin='lower', extent=[-X_LIM, X_LIM, -Y_LIM, Y_LIM])
    plt.colorbar(label='Std of scattering angle (rad)')
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.title('POCA Scattering Angle Dispersion')
    plt.show()


# Save results in the format expected by merge_results.py
np.save(args.output, {
    "n_events":     matrix_counts,
    "sum_theta":    matrix_sum_theta,
    "sum_theta_sq": matrix_sum_theta_sq
})

print(f"[CORRECT] POCA results saved to: {args.output}")
