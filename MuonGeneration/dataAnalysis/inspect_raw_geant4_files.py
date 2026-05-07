import argparse
import ROOT
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
args = parser.parse_args()

df = ROOT.RDataFrame("hits", args.input)

n_hits = df.Count().GetValue()
print(f"[INFO] Total hits: {n_hits}")

# Eventos únicos
n_events = df.AsNumpy(columns=["eventNumber"])
n_events = len(np.unique(n_events["eventNumber"]))
print(f"[INFO] Total unique events: {n_events}")


# Hits por evento (media)
print(f"[INFO] Avg hits per event: {n_hits / n_events:.2f}")

# Distribución de genID
res = df.AsNumpy(columns=["genID", "det", "layer"])

genIDs, counts = np.unique(res["genID"], return_counts=True)
print("\n[INFO] genID distribution:")
for gid, c in sorted(zip(counts, genIDs), reverse=True)[:10]:
    print(f"  genID {int(c):>8} : {int(gid)} hits")

dets, counts_d = np.unique(res["det"], return_counts=True)
print("\n[INFO] det distribution:")
for d, c in zip(dets, counts_d):
    print(f"  det {int(d)}: {int(c)} hits")

layers, counts_l = np.unique(res["layer"], return_counts=True)
print("\n[INFO] layer distribution:")
for l, c in zip(layers, counts_l):
    print(f"  layer {int(l)}: {int(c)} hits")