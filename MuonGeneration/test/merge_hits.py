# -*- coding: utf-8 -*-
"""
merge_hits.py — Merges all Pre_*.root files into a single ROOT file.

Called by simulate.py with:
    --namefile          {namefile}
    --n_jobs            {n_jobs_per_geometry}
    --path_hits_input   {PATH_preprocessed}
    --path_hits_output  {PATH_merged_post_makeHLT}
    --output            {out_merged_hits}

INPUT:  .root files at {path_hits_input}/Pre_{namefile}_seed*.root
        each containing a TTree "events" with branches:
        nevent, x1, y1, z1, vx1, vy1, vz1, energy1, x2, y2, z2, vx2, vy2, vz2, energy2

OUTPUT: Single .root file at {path_hits_output}/Pre_merged_{namefile}.root
        with a TTree "events" containing all events from all input files
"""

print("[INFO] Starting merge_hits.py...")
import os
import sys
import argparse
import glob

import ROOT
ROOT.gROOT.SetBatch(True)
print("[CORRECT] ROOT imported.")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Merge Pre_*.root files into a single ROOT file.")
parser.add_argument("--namefile",          required=True,  help="Geometry name (used to glob Pre_*.root files).")
parser.add_argument("--n_jobs",            required=True,  type=int, help="Expected number of Pre_*.root files.")
parser.add_argument("--path_hits_input",   required=True,  help="Directory where Pre_*.root files are stored.")
parser.add_argument("--path_hits_output",  required=True,  help="Directory where merged Pre_merged_*.root will be saved.")
parser.add_argument("--output",            required=True,  help="Full path for the output merged .root file.")
args = parser.parse_args()

print(f"[INFO] Geometry: {args.namefile}")
print(f"[INFO] Input directory:  {args.path_hits_input}")
print(f"[INFO] Output directory: {args.path_hits_output}")
print(f"[INFO] Output file:      {args.output}")

# ---------------------------------------------------------------------------
# Find and validate input .root files
# ---------------------------------------------------------------------------
pattern = os.path.join(args.path_hits_input, f"Pre_{args.namefile}_seed*.root")
input_files = sorted(glob.glob(pattern))

if len(input_files) == 0:
    print(f"[ERROR] No Pre_*.root files found matching: {pattern}")
    sys.exit(1)

if len(input_files) != args.n_jobs:
    print(f"[WARNING] Expected {args.n_jobs} Pre_*.root files, found {len(input_files)}. Proceeding anyway.")
else:
    print(f"[CORRECT] Found {len(input_files)} Pre_*.root files (matches n_jobs={args.n_jobs}).")

# ---------------------------------------------------------------------------
# Create output directory if it doesn't exist
# ---------------------------------------------------------------------------
os.makedirs(args.path_hits_output, exist_ok=True)

# ---------------------------------------------------------------------------
# Open output ROOT file and create output tree
# ---------------------------------------------------------------------------
from array import array

output_file = ROOT.TFile(args.output, "RECREATE")
output_tree = ROOT.TTree("events", "events")

# Create branches with leaf structure (same as makeHLTuple.py)
nevent = array('i', [0])
x1 = array('f', [0])
y1 = array('f', [0])
z1 = array('f', [0])
vx1 = array('f', [0])
vy1 = array('f', [0])
vz1 = array('f', [0])
energy1 = array('f', [0])
x2 = array('f', [0])
y2 = array('f', [0])
z2 = array('f', [0])
vx2 = array('f', [0])
vy2 = array('f', [0])
vz2 = array('f', [0])
energy2 = array('f', [0])

output_tree.Branch('nevent', nevent, 'nevent/I')
output_tree.Branch('x1', x1, 'x1/F')
output_tree.Branch('y1', y1, 'y1/F')
output_tree.Branch('z1', z1, 'z1/F')
output_tree.Branch('vx1', vx1, 'vx1/F')
output_tree.Branch('vy1', vy1, 'vy1/F')
output_tree.Branch('vz1', vz1, 'vz1/F')
output_tree.Branch('energy1', energy1, 'energy1/F')
output_tree.Branch('x2', x2, 'x2/F')
output_tree.Branch('y2', y2, 'y2/F')
output_tree.Branch('z2', z2, 'z2/F')
output_tree.Branch('vx2', vx2, 'vx2/F')
output_tree.Branch('vy2', vy2, 'vy2/F')
output_tree.Branch('vz2', vz2, 'vz2/F')
output_tree.Branch('energy2', energy2, 'energy2/F')

# ---------------------------------------------------------------------------
# Loop over all input files and copy events
# ---------------------------------------------------------------------------
total_events = 0
for i, input_file in enumerate(input_files):
    print(f"[INFO] Reading file {i+1}/{len(input_files)}: {os.path.basename(input_file)}")
    
    # Open input file
    infile = ROOT.TFile(input_file, "READ")
    if infile.IsZombie():
        print(f"[ERROR] Could not open input file: {input_file}")
        sys.exit(1)
    
    # Get input tree
    intree = infile.Get("events")
    if not intree:
        print(f"[ERROR] TTree 'events' not found in {input_file}")
        sys.exit(1)
    
    n_entries = intree.GetEntries()
    print(f"  Events in this file: {n_entries}")
    
    # Loop over events in input tree and copy to output tree
    for entry in range(n_entries):
        intree.GetEntry(entry)
        
        nevent[0] = intree.nevent
        x1[0] = intree.x1
        y1[0] = intree.y1
        z1[0] = intree.z1
        vx1[0] = intree.vx1
        vy1[0] = intree.vy1
        vz1[0] = intree.vz1
        energy1[0] = intree.energy1
        x2[0] = intree.x2
        y2[0] = intree.y2
        z2[0] = intree.z2
        vx2[0] = intree.vx2
        vy2[0] = intree.vy2
        vz2[0] = intree.vz2
        energy2[0] = intree.energy2
        
        output_tree.Fill()
        total_events += 1
    
    infile.Close()

# ---------------------------------------------------------------------------
# Write output file
# ---------------------------------------------------------------------------
output_file.Write()
output_file.Close()

print(f"[CORRECT] Merge completed.")
print(f"[INFO] Total events in merged file: {total_events}")
print(f"[CORRECT] Output saved to: {args.output}")