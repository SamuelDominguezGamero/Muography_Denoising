** Data workflow**: 
1. Generate the configuration file (json), that will be stored in `.../data/geometric_configurations_json`.
2. Run the simulation: When the "Generator" is called (in the "build" directory), an output file is generated, where all the hits are stored, but there is no labelling nor correlations in those hits, so they will be stored in .../data/data_raw
3. ...data/data_preprocessed will store the processed muon tracks, for each of the simulation
4. POCA algorithm will be applied to preprocessed data, and stored in `.../data/post_POCA_data`.
   Here, data will be stored splitted in several .npy files
5. Splitted data, example: "..._seed1.npy", "..._seed2.npy", ..., will be merged into a single `.npy` file that will be stored in `.../data/merged_POCA_data`
   - This will be the data used for **U-Net training**. Each `.npy` file will be a training instance.
   - Data is saved in an specific format, by using `np.save(path, {dictionary})`. To load this data use `np.load(path, pickle = True).item()`.
6. Ground truth, this is, the real geometry (with its corresponding json file located at `...data/geometric_configurations_json`, used for GEANT4 simulation, will be transformed into a VOXELIZED GEOMETRY (tensor), acting as the ground truth for ML training.
   - We will find those files in `.../data/ground_truth_data`
7. The file 
