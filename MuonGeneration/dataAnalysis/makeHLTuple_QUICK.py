import argparse
import ROOT
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", default="output_quick.root")
args = parser.parse_args()

print("[INFO] Loading data...")
df = ROOT.RDataFrame("hits", args.input)
df = df.Filter("abs(genID) == 13")
res = df.AsNumpy(columns=["eventNumber", "det", "layer", "x", "y", "z", "vx", "vy", "vz", "energy"])

print("[INFO] Data loaded, processing...")
ev_nums = res["eventNumber"]
det     = res["det"]
layer   = res["layer"]
x, y, z = res["x"], res["y"], res["z"]
vx, vy, vz = res["vx"], res["vy"], res["vz"]
energy  = res["energy"]

# Agrupar índices por evento
from collections import defaultdict
event_indices = defaultdict(list)
for i, ev in enumerate(ev_nums):
    event_indices[ev].append(i)

print(f"[INFO] Unique events with muon hits: {len(event_indices)}")

# Ajuste lineal (mismo que makeFit)
def fit(xs, ys, zs):
    n = len(xs)
    sx  = xs.sum(); sy  = ys.sum(); sz  = zs.sum()
    sxz = (xs*zs).sum(); syz = (ys*zs).sum(); szz = (zs*zs).sum()
    dxdz = (n*sxz - sx*sz) / (n*szz - sz*sz)
    dydz = (n*syz - sy*sz) / (n*szz - sz*sz)
    x0 = sx/n - dxdz * sz/n
    y0 = sy/n - dydz * sz/n
    z0 = sz/n
    return x0 + dxdz*z0, y0 + dydz*z0, z0, dxdz, dydz

# Salida
out_file = ROOT.TFile(args.output, "RECREATE")
tree = ROOT.TTree("events", "events")

branches = {}
for name, fmt in [("nevent","I"),("x1","F"),("y1","F"),("z1","F"),
                  ("vx1","F"),("vy1","F"),("vz1","F"),("energy1","F"),
                  ("x2","F"),("y2","F"),("z2","F"),
                  ("vx2","F"),("vy2","F"),("vz2","F"),("energy2","F")]:
    typ = 'i' if fmt == 'I' else 'f'
    branches[name] = np.array([0], dtype=np.int32 if typ=='i' else np.float32)
    tree.Branch(name, branches[name], f"{name}/{fmt}")

valid = 0
SLOTS = {(0,0), (0,1), (1,0), (1,1)}

for ev, idxs in event_indices.items():
    idx = np.array(idxs)
    slots = set(zip(det[idx], layer[idx]))
    if slots != SLOTS:
        continue

    for d in [0, 1]:
        mask = det[idx] == d
        if mask.sum() != 2:
            break
    else:
        idx0 = idx[det[idx] == 0]
        idx1 = idx[det[idx] == 1]

        rx1, ry1, rz1, rdxdz1, rdydz1 = fit(x[idx0], y[idx0], z[idx0])
        rx2, ry2, rz2, rdxdz2, rdydz2 = fit(x[idx1], y[idx1], z[idx1])

        branches["nevent"][0] = ev
        branches["x1"][0] = rx1;  branches["y1"][0] = ry1;  branches["z1"][0] = rz1
        branches["vx1"][0] = rdxdz1; branches["vy1"][0] = rdydz1; branches["vz1"][0] = -1.0
        branches["energy1"][0] = energy[idx0[0]]
        branches["x2"][0] = rx2;  branches["y2"][0] = ry2;  branches["z2"][0] = rz2
        branches["vx2"][0] = rdxdz2; branches["vy2"][0] = rdydz2; branches["vz2"][0] = -1.0
        branches["energy2"][0] = energy[idx1[0]]
        tree.Fill()
        valid += 1

print(f"[INFO] Valid events: {valid}")
out_file.Write()
out_file.Close()
print(f"[INFO] Output saved to {args.output}")