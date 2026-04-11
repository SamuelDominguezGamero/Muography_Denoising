#!/usr/bin/env python3
"""
QUICK START GUIDE: Bitmap Generation & Geometry Creation Pipeline
"""

print("""
╔════════════════════════════════════════════════════════════════════╗
║         BITMAP GENERATION & GEOMETRY CREATION PIPELINE             ║
║                          QUICK START GUIDE                         ║
╚════════════════════════════════════════════════════════════════════╝

1. UNDERSTAND AVAILABLE BITMAPS
   ────────────────────────────
   File: bitmaps_letters.py
   
   Available sizes & strokes:
   • (8×8):   strokes [1, 2]
   • (10×10): strokes [1, 2]
   • (12×12): strokes [1, 2]
   • (14×14): strokes [1, 2, 3]
   • (16×16): strokes [1, 2, 3]
   
   Letters: M, U, O, N

2. VALIDATE CONFIGURATIONS (OPTIONAL BUT RECOMMENDED)
   ───────────────────────────────────────────────────
   Run before attempting to create 480 geometries:
   
   $ python3 validate_bitmap_configs.py
   
   Output:
   ✓ Valid configurations:   480
   ✓ ALL CONFIGURATIONS ARE VALID! Ready to create geometries.

3. CREATE GEOMETRY FILES
   ──────────────────────
   
   Option A: Create ALL geometries (480 configurations)
   
   $ python3 loop_configuration_files.py
   
   Set these flags in loop_configuration_files.py:
   - create_geometries = True
   - simulate = False (doesn't submit SLURM jobs)
   - environment = "local" or "cluster"
   
   This will:
   ✓ Create 480 JSON geometry files
   ✓ Generate ground truth density arrays
   ✓ Report errors with detailed messages
   
   Example output:
   [INFO] ----- Total geometries to generate: 480
   [CORRECT] Geometry 1/480 created: _Lpx128_Lpy128...
   [ERROR] Geometry 2/480 failed: ... (if any fail)
   
   Option B: Create a SINGLE geometry manually
   
   $ python3 create_geometry.py \\
       --Lpx 128 --Lpy 128 --Lpz 128 \\
       --npx 128 --npy 128 --npz 128 \\
       --zPosDetector_top 54 --zPosDetector_bot -54 \\
       --spacing 1 --ratio 1 \\
       --FontSizeX 8 --FontSizeY 8 \\
       --material lead \\
       --word_geometry MUON \\
       --StrokeWidth 1 \\
       --output_json geometry_test.json \\
       --output_ground_truth_density ground_truth_test.npy

4. TEST BITMAP GENERATION
   ──────────────────────
   
   $ python3 test_bitmap_generation.py
   
   Tests:
   • Test 1: Shows available bitmap configurations
   • Test 2: Creates uniform parameter word ("MUON")
   • Test 3: Creates varying parameter word
   • Test 4: Validates error handling

5. ERROR REPORTING IMPROVEMENTS
   ─────────────────────────────
   
   When a bitmap is missing, you now get detailed info:
   
   [ERROR] Letter bitmap not found:
           Requested: Size=12x12, Letter='N', Stroke=3
           Available sizes: [(8,8), (10,10), (12,12), (14,14), (16,16)]
           Available strokes for (12,12): [1, 2]
   
   This tells you exactly which combination is missing.

6. OUTPUT FILES
   ────────────
   
   After running the geometry creation:
   
   data/geometric_configurations_json/
   ├── _Lpx128...stroke1.json
   ├── _Lpx128...stroke2.json
   └── ... (480 files)
   
   data/ground_truth_data/
   ├── _Lpx128...stroke1_ground_truth_density.npy
   ├── _Lpx128...stroke2_ground_truth_density.npy
   └── ... (480 files)

7. PARAMETERS
   ──────────
   
   Geometry Parameters (fixed):
   • Lpx, Lpy, Lpz: World dimensions = 128 cm
   • npx, npy, npz: Number of voxels = 128
   • zPosDetector_top: +54 cm
   • zPosDetector_bot: -54 cm
   
   Varied Parameters:
   • spacing: [1, 2] voxels between letters
   • ratio: [1] aspect ratio factor
   • FontSizes: [8, 10, 12, 14, 16]
   • stroke: [1, 2] or [1, 2, 3] depending on font size
   • material: ["lead", "uranium"]
   • word: ["MUON", "MOUN", "NUMO", ...] (10 words)

8. TROUBLESHOOTING
   ────────────────
   
   Q: "Bitmap template missing for size X×X, stroke Y"
      → That stroke doesn't exist for that size. 
      → Check bitmaps_letters.py BITMAP_DATA or validate_bitmap_configs.py
   
   Q: Geometry creation hangs
      → Some geometries take time due to voxel computation.
      → Run a single geometry first to estimate time.
   
   Q: Memory issues with 480 geometries
      → Create them in batches by modifying loop_configuration_files.py
      → Or run create_geometry.py for individual configs.

═══════════════════════════════════════════════════════════════════════
For more details, see: loop_configuration_files.py, create_geometry.py
═══════════════════════════════════════════════════════════════════════
""")
