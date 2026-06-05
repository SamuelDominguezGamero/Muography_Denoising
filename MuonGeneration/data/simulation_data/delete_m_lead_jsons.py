"""
Delete half of the M+lead JSON geometries to reduce run0 imbalance.
"""

import random
from pathlib import Path

# Path to run0 JSON folder
json_folder = Path("/home/samuel/Work/Muography_Denoising/MuonGeneration/data/geometric_configurations_json")

# Find all M+lead JSONs
m_lead_files = [f for f in json_folder.glob("*.json") if "wordM_" in f.name and "matlead" in f.name]

print(f"[INFO] Found {len(m_lead_files)} M+lead JSON files")

if len(m_lead_files) == 0:
    print("[ERROR] No M+lead files found!")
    exit(1)

# Select half randomly
num_to_delete = len(m_lead_files) // 2
to_delete = random.sample(m_lead_files, num_to_delete)

print(f"[INFO] Will delete {num_to_delete} files randomly selected")
print(f"[INFO] This will leave {len(m_lead_files) - num_to_delete} M+lead files\n")

# Delete
deleted = 0
for json_file in to_delete:
    try:
        json_file.unlink()
        deleted += 1
        if deleted % 100 == 0:
            print(f"  [DELETED {deleted}/{num_to_delete}] {json_file.name}")
    except Exception as e:
        print(f"  [ERROR] Could not delete {json_file.name}: {e}")

print(f"\n[OK] Deleted {deleted} files successfully")
print(f"[INFO] Remaining M+lead files: {len(m_lead_files) - deleted}")
