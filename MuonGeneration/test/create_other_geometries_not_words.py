import json
import argparse


parser = argparse.ArgumentParser(description="Creation of geometry from POCA resolution ---> key parameter = ratio.")
parser.add_argument("--Lpx", type=float, default=128, help="Dimension X (cm).")
parser.add_argument("--Lpy", type=float, default=128, help="Dimension Y (cm).")
parser.add_argument("--Lpz", type=float, default=128, help="Dimension Z (cm).")
parser.add_argument("--npx", type=int, default=128, help="Number of voxels (X) Geant4.")
parser.add_argument("--npy", type=int, default=128, help="Number of voxels (Y) Geant4.")
parser.add_argument("--npz", type=int, default=128, help="Number of voxels (Z) Geant4.")
parser.add_argument("--ratio", type=int, default=2, help="SizeVoxelGeant4 / SizeVoxelPOCA: (natural >= 1). Keep in mind that the number of voxels in POCA should be greater than or equal to those in Geant4. The resolution of POCA is the resolution of the image that will be given to the neural network.")

parser.add_argument("--zPosDetector_top", type=float, default=118.0,
    help="Z position of the TOP detector (cm), above the geometry.")
parser.add_argument("--zPosDetector_bot", type=float, default=-118.0,
    help="Z position of the BOTTOM detector (cm), below the geometry.")

parser.add_argument("--depth_z_cm", type=float, default=2.0, help="Word thickness in centimeters. Efficient single slab method (variable zSizeVoxel).")
parser.add_argument("--material", type=str, default="lead", help="Material for the word geometry. Default is 'lead'.")
args = parser.parse_args()


# ====== POCA ====== 
Lpx = args.Lpx # Length of the world (X) POCA
Lpy = args.Lpy # Length of the world (Y) POCA
Lpz = args.Lpz # Length of the world (Z) POCA
npx = args.npx # Number of voxels (X) POCA
npy = args.npy # number of voxels (Y) POCA
npz = args.npz # number of voxels (Z) POCA
size_voxel_poca_x = Lpx/npx 
size_voxel_poca_y = Lpy/npy 
size_voxel_poca_z = Lpz/npz 


# ====== GEANT4 ====== 
Lx = Lpx 
Ly = Lpy
Lz = Lpz
size_voxel_G4_x = args.ratio * size_voxel_poca_x
size_voxel_G4_y = args.ratio * size_voxel_poca_y
size_voxel_G4_z = args.ratio * size_voxel_poca_z



# ====== GEOMETRIES ====== 
def rectangle(height, width, depth, center):
    voxels_high = height / size_voxel_G4_y
    voxels_wide = width / size_voxel_G4_x
    if (height %% size_voxel_G4_x != 0) or (width %% size_voxel_G4_x):








# Geometric figures --> to train the UNET with other geometries that are not letters, so it can generalize better










global_dictionary = {
    "theWorld": {
    "xSizeWorld": Lx,
    "ySizeWorld": Ly,
    "zSizeWorld": Lz,
    "sizeBoxCRY": Lx,
    "zOffsetCRY": Lz / 2.0,
    },
    "Detectors": [], # fill with detector_dictionaries
    "VoxelConfig": {}, # usually should remain empty
    "TheVoxels": [], # fill with voxel_dictionaries
}



global_dictionary["Detectors"] = [ # should be replaced with something more modular, pending
        {
            "xPosDetector": 0,
            "yPosDetector": 0,
            "zPosDetector": args.zPosDetector_top,
            "xDirDetector": 0,
            "yDirDetector": 0,
            "zDirDetector": 0,
            "xSizeDetector": Lx,
            "ySizeDetector": Ly,
            "zSizeDetector": 20,
            "Layers": [
                {
                    "xPosLayer": 0,
                    "yPosLayer": 0,
                    "zPosLayer": 0,
                    "xDirLayer": 0,
                    "yDirLayer": 0,
                    "zDirLayer": 0,
                    "xSizeLayer": Lx,
                    "ySizeLayer": Ly,
                    "zSizeLayer": 1
                },
                {
                    "xPosLayer": 0,
                    "yPosLayer": 0,
                    "zPosLayer": -10,
                    "xDirLayer": 0,
                    "yDirLayer": 0,
                    "zDirLayer": 0,
                    "xSizeLayer": Lx,
                    "ySizeLayer": Ly,
                    "zSizeLayer": 1
                }
            ]
        },
        {
            "xPosDetector": 0,
            "yPosDetector": 0,
            "zPosDetector": args.zPosDetector_bot,
            "xDirDetector": 0,
            "yDirDetector": 0,
            "zDirDetector": 0,
            "xSizeDetector": Lx,
            "ySizeDetector": Ly,
            "zSizeDetector": 20,
            "Layers": [
                {
                    "xPosLayer": 0,
                    "yPosLayer": 0,
                    "zPosLayer": 0,
                    "xDirLayer": 0,
                    "yDirLayer": 0,
                    "zDirLayer": 0,
                    "xSizeLayer": Lx,
                    "ySizeLayer": Ly,
                    "zSizeLayer": 1
                },
                {
                    "xPosLayer": 0,
                    "yPosLayer": 0,
                    "zPosLayer": 10,
                    "xDirLayer": 0,
                    "yDirLayer": 0,
                    "zDirLayer": 0,
                    "xSizeLayer": Lx,
                    "ySizeLayer": Ly,
                    "zSizeLayer": 1
                }
            ]
        }
]