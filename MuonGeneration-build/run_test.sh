#!/bin/bash

#### PATH SETTINGS ####
path_data_Analysis="/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/dataAnalysis"
path_output_raw="/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_raw"
path_output_preprocessed="/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_preprocessed"
path_conf_geom="/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/geometric_configurations_json"
path_generator="/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration-build/Generator"
path_setup="/gpfs/users/dominguezs/Muography_Denoising/setup.sh"
source "$path_setup"
path_poca_output="/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/post_POCA_data"


##### 0th: Generating the geometric configuration file #####
# COMPLETE... we have a test configuration file for the moment. In the real project, we'll need to work on this part.


##### 1st: Running the simulation #####
"$path_generator" \
    --input "$path_conf_geom/geometry.json" \
    --output "$path_output_raw/output_generacion_geometrias.root" \
    --number 10000 \
    --seed 1

echo '[[[[[[[[[[[[[[[[[[SUCCESSFUL MONTECARLO SIMULATION]]]]]]]]]]]]]]]]]]]]]]]]'

##### 2nd: Preprocessing the data #####
# the file that preprocess data is called makeHLTuple.py, located at /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/dataAnalysis/makeHLTuple.py

# the system can easily run out of memory due to the large number of events, so we split the preprocessing in several jobs for the case of large datasets ---> check slurm documentation

# python3 -u "$path_data_Analysis/makeHLTuple.py" \
#    --input "$path_output_raw/output_prueba_10kmuons.root" \
#    --conf "$path_conf_geom/confMUON_Custom.json" \
#    --output "$path_output_preprocessed/preprocessed_prueba_10kmuons.root"

# echo '[[[[[[[[[[[[[[[[[[MUON HITS SUCCESSFULLY CORRELATED]]]]]]]]]]]]]]]]]]]]]]]]'

##### 3rd: POCA reconstruction #####
# echo 'Starting POCA reconstruction...'
# python3 -u "$path_data_Analysis/POCA.py" \
# 	--input "$path_output_preprocessed/preprocessed_prueba_10kmuons.root" \
# 	--output "$path_poca_output/final_poca_result_10kmuons.npy"

# echo '[[[[[[[[[[[[[[[[[[POCA IMPLEMENTED SUCCESSFULLY]]]]]]]]]]]]]]]]]]]]]]]'
