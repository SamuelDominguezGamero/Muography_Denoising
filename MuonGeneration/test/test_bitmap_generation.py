#!/usr/bin/env python3
"""
Test script to validate bitmap generation with various configurations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bitmaps_letters import get_word, BITMAP_DATA

# Test 1: Check available bitmaps
print("=" * 60)
print("TEST 1: Available bitmap configurations")
print("=" * 60)
print(f"Available sizes: {list(BITMAP_DATA.keys())}")
for size in BITMAP_DATA.keys():
    strokes = list(BITMAP_DATA[size].keys())
    print(f"  Size {size}: strokes {strokes}")

# Test 2: Generate a simple word with uniform parameters
print("\n" + "=" * 60)
print("TEST 2: Simple word with uniform parameters")
print("=" * 60)
try:
    word_matrix, shape = get_word("MUON", res_x_list=8, res_y_list=8, stroke_list=1)
    if word_matrix is not None:
        print(f"✓ Success: Created word 'MUON' with shape {shape}")
    else:
        print("✗ Failed: get_word returned None")
except Exception as e:
    print(f"✗ Exception: {e}")

# Test 3: Generate a word with varying parameters
print("\n" + "=" * 60)
print("TEST 3: Word with varying parameters per letter")
print("=" * 60)
try:
    word_matrix, shape = get_word(
        "MUON", 
        res_x_list=[8, 8, 10, 12],
        res_y_list=[8, 8, 10, 12],
        stroke_list=[1, 1, 2, 3]
    )
    if word_matrix is not None:
        print(f"✓ Success: Created word 'MUON' with varying params, shape {shape}")
    else:
        print("✗ Failed: get_word returned None")
except Exception as e:
    print(f"✗ Exception: {e}")

# Test 4: Try an invalid configuration (should fail gracefully)
print("\n" + "=" * 60)
print("TEST 4: Invalid configuration (missing bitmap)")
print("=" * 60)
try:
    word_matrix, shape = get_word("MUON", res_x_list=999, res_y_list=999, stroke_list=99)
    if word_matrix is None:
        print("✓ Correctly returned None for invalid configuration")
    else:
        print("✗ Unexpectedly succeeded with invalid parameters")
except Exception as e:
    print(f"✓ Caught exception as expected: {type(e).__name__}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
