#!/usr/bin/env python3
"""
Validation script to check if all geometry configurations have valid bitmap templates.
Runs BEFORE trying to create 480 geometries.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bitmaps_letters import BITMAP_DATA

# ===========================================================================
# GEOMETRY VARIATIONS (from loop_configuration_files.py)
# ===========================================================================
spacings       = [1, 2]
ratios         = [1]
FontSizes      = [
    {"size": 8,  "strokes": [1, 2]},
    {"size": 10, "strokes": [1, 2]},
    {"size": 12, "strokes": [1, 2]},
    {"size": 14, "strokes": [1, 2, 3]},
    {"size": 16, "strokes": [1, 2, 3]},
]
materials      = ["lead", "uranium"]
words_geometry = ["MUON", "MOUN", "NUMO", "UNOM", "ONUM", "UMON", "MUNU", "NOMU", "UMNU", "MONU"]

# ===========================================================================
# VALIDATION
# ===========================================================================
print("=" * 70)
print("BITMAP CONFIGURATION VALIDATION")
print("=" * 70)

valid_count = 0
invalid_configs = []

for spacing in spacings:
    for ratio in ratios:
        for fontsize_dict in FontSizes:
            size = fontsize_dict["size"]
            for stroke in fontsize_dict["strokes"]:
                # Check if this size+stroke combination exists
                size_key = (size, size)
                if size_key not in BITMAP_DATA:
                    invalid_configs.append(f"Size {size}x{size} not in BITMAP_DATA")
                    continue
                if stroke not in BITMAP_DATA[size_key]:
                    invalid_configs.append(f"Size {size}x{size} stroke {stroke} not in BITMAP_DATA")
                    continue
                
                # Check all letters and words
                for material in materials:
                    for word in words_geometry:
                        # Check if all letters in the word are available
                        available_letters = set(BITMAP_DATA[size_key][stroke].keys())
                        word_letters = set(word.upper())
                        
                        if not word_letters.issubset(available_letters):
                            missing = word_letters - available_letters
                            invalid_configs.append(
                                f"Size {size}x{size} stroke {stroke}: "
                                f"Word '{word}' missing letters {missing}"
                            )
                        else:
                            valid_count += 1

# ===========================================================================
# REPORT
# ===========================================================================
total_configs = len(spacings) * len(ratios) * sum(len(d["strokes"]) for d in FontSizes) \
                * len(materials) * len(words_geometry)

print(f"\n✓ Valid configurations:   {valid_count:,}")
print(f"✗ Invalid configurations: {len(invalid_configs):,}")
print(f"  Total checked:          {total_configs:,}")

if invalid_configs:
    print("\n" + "=" * 70)
    print("INVALID CONFIGURATIONS FOUND:")
    print("=" * 70)
    for i, config in enumerate(invalid_configs[:20], 1):  # Show first 20
        print(f"  {i:3}. {config}")
    if len(invalid_configs) > 20:
        print(f"  ... and {len(invalid_configs) - 20} more")
    print("\n[WARNING] Some configurations will fail during geometry creation!")
    sys.exit(1)
else:
    print("\n" + "=" * 70)
    print("✓ ALL CONFIGURATIONS ARE VALID! Ready to create geometries.")
    print("=" * 70)
    sys.exit(0)
