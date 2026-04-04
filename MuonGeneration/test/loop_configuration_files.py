"""
This file is created automate the simulation process: 
- call the function create_geometry.py systematically, generating all the geometry files needed for the training of the neural network.
- running the Geant4 simulation for each of the geometry files generated
- process all the outputs in its corresponding folders
- divide the complete process in jobs, so that SLURM can be used to run the simulations in parallel in a cluster
"""


import subprocess # ---> this is the key module that allows us to call bash commands from python scripts
import os # for file management and paths
import sys

# control variables:
simulate = False # security flag to prevent accidental execution of simulations while testing the script
create_geometries = True # security flag to prevent accidental creation of geometries when not needed
environment = "local" # "cluster" or "local", to set the paths correctly

if create_geometries == False:
    sys.exit("[INFO] ----- Create geometries flag is set to False. Exiting without creating geometries. Go to the beggining of this script and set create_geometries = True to create the geometries")


##### Defining the different geometric configurations (all of them are flags for the function create_geometry.py) #####

# Defined structure for POCA voxelization, should be keept fixed:
Lpx = 128
Lpy = 128
Lpz = 128
npx = 128
npy = 128
npz = 128
zPosDetector_top = 118
zPosDetector_bot = -118


# variations:
spacings = [1, 2, 3]
ratios = [1, 2] # this will define the size (cm) of Geant4 voxels
FontsSizeX = [8, 10, 12, 14] # avaliable: 8, 10, 12, 14, 16
    # we can give a list of lists (4 elements), or we can give a list of integers, then every letter will have same font size

materials = ["lead"] # avaliable: "lead"
words_geometry = ['MUON', 'MUNO', 'NUMO', 'NUOM'] # this will be the name of the geometry file, and it is just for identification purposes, it does not affect the geometry itself 
strokes = [1, 2]


# path configuration:
if environment == "cluster":
    PATH_geometry_files = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/geometric_configurations_json"
    PATH_density_files = "/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/ground_truth_data"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # directory of this script, where create_geometry.py is located
    CREATE_GEOMETRY_SCRIPT = os.path.join(SCRIPT_DIR, "create_geometry.py")


elif environment == "local":
    PATH_geometry_files = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/geometric_configurations_json"
    PATH_density_files = "/home/samuel/Work/Muography_Denoising/MuonGeneration/data/ground_truth_data"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # directory of this script, where create_geometry.py is located
    CREATE_GEOMETRY_SCRIPT = os.path.join(SCRIPT_DIR, "create_geometry.py")



else:
    print("Error: environment variable must be set to 'cluster' or 'local'")
    sys.exit(1)





# for loop generating the geometry files
i = 0
total_iterations = len(spacings) * len(ratios) * len(FontsSizeX) * len(materials) * len(words_geometry) * len(strokes) 
for spacing in spacings:
    for ratio in ratios:
        for x in FontsSizeX: # (implicitely the same for y font)
            for material in materials: 
                for word in words_geometry:
                    for stroke in strokes:
                        i += 1
                        # create the geometry file with the corresponding parameters, using the function create_geometry.py
                        namefile = f"Lpx{Lpx}_Lpy{Lpy}_Lpz{Lpz}_npx{npx}_npy{npy}_npz{npz}_zPosDetector_top{zPosDetector_top}_zPosDetector_bot{zPosDetector_bot}_spacing{spacing}_ratio{ratio}_FontSizeX{x}_FontSizeY{x}_material{material}_word{word}_stroke{stroke}.json"
                        
                        command = [
                            "python3", CREATE_GEOMETRY_SCRIPT,
                            "--Lpx", str(Lpx),
                            "--Lpy", str(Lpy),
                            "--Lpz", str(Lpz),
                            "--npx", str(npx),
                            "--npy", str(npy),
                            "--npz", str(npz),
                            "--zPosDetector_top", str(zPosDetector_top),
                            "--zPosDetector_bot", str(zPosDetector_bot),
                            "--spacing", str(spacing),
                            "--ratio", str(ratio),
                            "--FontSizeX", str(x),
                            "--FontSizeY", str(x), # implicitely the same as FontSizeX
                            "--material", material,
                            "--word_geometry", word,
                            "--StrokeWidth", str(stroke),
                            "--output_json", PATH_geometry_files + "/" + namefile,
                            "--output_ground_truth_density", PATH_density_files + "/" + namefile.replace(".json", "_ground_truth_density.npy")
                        ]
                        
                        result = subprocess.run(command, capture_output=True, text=True)
                        
                        if result.returncode != 0:
                            print(f"Error en ejecución: {result.stderr}")
                        else:
                            print(f"Geometría creada exitosamente ----- [CORRECT] ----- [{i}/{total_iterations}]")
                

# for each of geometry generated, we call several jobs
# each of them with differente SEED!!!!!!!!!!

# if simulate == False:
#     sys.exit("[INFO] ----- Simulation flag is set to False. Exiting without running simulations. Go to the beggining of this script and set simulate = True to run the simulations")
# else:
    # total_muons_per_geometry = 10000000
    # n_muons_per_job = 10000
    # n_jobs_per_geometry = total_muons_per_geometry // n_muons_per_job



    # for geometry_file in carpeta_geometry_files:
    #     if simulate == False:
    #         sys.exit("[INFO] ----- Simulation flag is set to False. Exiting without running simulations. Go to the beggining of this script and set simulate = True to run the simulations")

    #     for job in range(n_jobs_per_geometry):
    #         seed = job 
    #         # create a shell script for each job, with the command to run the simulation with the corresponding geometry file and seed
            
    #         result = subprocess.run(command, capture_output=True, text=True)
            
    #         if result.returncode != 0:
    #             print(f"Error en ejecución: {result.stderr}")
    #         else:
    #             print(f"Simulación exitosa para {geometry_file} con seed={seed}")



















# # AI GENERATED CODE, JUST FOR IDEAS: 

# import subprocess

# listax = [1.0, 2.0]
# listay = [0.5, 1.5]
# listaz = [10, 20]

# for x in listax:
#     for y in listay:
#         for z in listaz:
#             # Construcción del comando
#             command = [
#                 "python", "create_geometry.py",
#                 "--x", str(x),
#                 "--y", str(y),
#                 "--z", str(z)
#             ]
            
#             # Ejecución
#             result = subprocess.run(command, capture_output=True, text=True)
            
#             # Verificación de errores
#             if result.returncode != 0:
#                 print(f"Error en ejecución: {result.stderr}")
#             else:
#                 print(f"Salida exitosa para x={x}, y={y}, z={z}")
