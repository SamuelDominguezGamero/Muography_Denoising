** Data workflow**: 
1. Generate the configuration file (json), that will be stored in `.../data/geometric_configurations_json`.
2. Run the simulation: When the "Generator" is called (in the "build" directory), an output file is generated, where all the hits are stored, but there is no labelling nor correlations in those hits, so they will be stored in .../data/data_raw
3. ...data/data_preprocessed will store the processed muon tracks, for each of the simulation
4. POCA algorithm will be applied to preprocessed data, and stored in `.../data/post_POCA_data`. 
   - This will be the data used for **U-Net training**
5. Ground truth, this is, the real geometry (previously stored in `...data/geometric_configurations_json`, used for GEANT4 simulation, will be transformed into a VOXELIZED GEOMETRY (tensor), acting as the ground truth for ML training.
   - We will find those files in `.../data/ground_truth_data`


