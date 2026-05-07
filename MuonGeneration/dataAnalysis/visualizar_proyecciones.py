import argparse
import ROOT
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
args = parser.parse_args()

df = ROOT.RDataFrame("events", args.input)
res = df.AsNumpy(columns=["poca_x", "poca_y", "poca_z"])

x = res["poca_x"]
y = res["poca_y"]
z = res["poca_z"]


print(f"[INFO] POCA points loaded: {len(x)}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist2d(x, y, bins=128)
axes[0].set_xlabel("poca_x"); axes[0].set_ylabel("poca_y"); axes[0].set_title("XY")

axes[1].hist2d(x, z, bins=128)
axes[1].set_xlabel("poca_x"); axes[1].set_ylabel("poca_z"); axes[1].set_title("XZ")

axes[2].hist2d(y, z, bins=128)
axes[2].set_xlabel("poca_y"); axes[2].set_ylabel("poca_z"); axes[2].set_title("YZ")

plt.tight_layout()
plt.show()
