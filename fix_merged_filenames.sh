#!/bin/bash

# Script to rename all MERGED_*.npy files to include _Muons_1000000 in the filename
# This fixes the filename format for datasets created before the muon count metadata was added
# Usage: bash fix_merged_filenames.sh /path/to/merged_poca_data

MERGED_DIR="${1:-.}"

if [ ! -d "$MERGED_DIR" ]; then
    echo "[ERROR] Directory not found: $MERGED_DIR"
    exit 1
fi

echo "[INFO] Scanning directory: $MERGED_DIR"
echo "[INFO] This will rename files from:"
echo "       MERGED_..._stroke{N}_2D.npy"
echo "       to:"
echo "       MERGED_..._stroke{N}_Muons_1000000_2D.npy"
echo ""

# Count files to rename
count=$(ls -1 "$MERGED_DIR"/MERGED_*_2D.npy "$MERGED_DIR"/MERGED_*_3D.npy 2>/dev/null | wc -l)
echo "[INFO] Found $count files to rename"
echo ""

# Proceed with renaming
cd "$MERGED_DIR" || exit 1

renamed=0
failed=0

# Process 2D files
for file in MERGED_*_2D.npy; do
    if [ -f "$file" ]; then
        # Check if already has _Muons_ in the name
        if [[ "$file" =~ _Muons_ ]]; then
            echo "[SKIP] Already renamed: $file"
            continue
        fi
        
        # Extract the part before _2D and insert _Muons_1000000
        # Pattern: MERGED_..._stroke{N}_2D.npy -> MERGED_..._stroke{N}_Muons_1000000_2D.npy
        newfile="${file%_2D.npy}_Muons_1000000_2D.npy"
        
        if mv "$file" "$newfile"; then
            echo "[OK] $file -> $newfile"
            ((renamed++))
        else
            echo "[ERROR] Failed to rename: $file"
            ((failed++))
        fi
    fi
done

# Process 3D files
for file in MERGED_*_3D.npy; do
    if [ -f "$file" ]; then
        # Check if already has _Muons_ in the name
        if [[ "$file" =~ _Muons_ ]]; then
            echo "[SKIP] Already renamed: $file"
            continue
        fi
        
        # Extract the part before _3D and insert _Muons_1000000
        newfile="${file%_3D.npy}_Muons_1000000_3D.npy"
        
        if mv "$file" "$newfile"; then
            echo "[OK] $file -> $newfile"
            ((renamed++))
        else
            echo "[ERROR] Failed to rename: $file"
            ((failed++))
        fi
    fi
done

echo ""
echo "============================================================"
echo "[SUMMARY]"
echo "  Successfully renamed: $renamed files"
echo "  Failed: $failed files"
echo "============================================================"

if [ $failed -eq 0 ]; then
    echo "[SUCCESS] All files renamed successfully!"
    exit 0
else
    echo "[WARNING] Some files failed to rename"
    exit 1
fi
