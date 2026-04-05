#!/bin/bash


if [ $HOSTNAME == "Minkowski" ]; then
    echo "Setting up environment in Minkowski"
    export G4INSTALLDIR=/home/pablo/Documentos/software/geant4-v11.1.2-install/
    export G4WORKDIR=/home/pablo/Documentos/softwareProjects/Muography_Denoising/
    export JSONCPPDIR=/home/pablo/Documentos/software/jsoncpp/
    source $G4INSTALLDIR/bin/geant4.sh
    export PYTHONPATH=$G4WORKDIR/MuonGeneration/dataAnalysis/
    source /home/pablo/Documentos/software/root_v6.28.04-install/bin/thisroot.sh
fi

if [ $HOSTNAME == "Leibniz" ]; then
    echo "Setting up environment in Leibniz"
    export G4INSTALLDIR=/home/pablo/Documentos/software/geant4-v11.1.2-install
    export G4WORKDIR=/home/pablo/Documentos/softwareProjects/Muography_Denoising/
    export JSONCPPDIR=/home/pablo/Documentos/software/jsoncpp/
    source $G4INSTALLDIR/bin/geant4.sh
    export PYTHONPATH=$G4WORKDIR/MuonGeneration/dataAnalysis/
    source /home/pablo/Documentos/software/root_v6.36.00-install/bin/thisroot.sh
fi


if [ $HOSTNAME == "login2.ifca.es" ] && [ $USER == "parbol" ]; then
    echo "Setting up environment in login2"
    export G4INSTALLDIR=/gpfs/users/parbol/geant4-v11.1.2-install
    export G4WORKDIR=/gpfs/users/parbol/Muography_Denoising/
    export JSONCPPDIR=/gpfs/users/parbol/jsoncpp/
    source $G4INSTALLDIR/bin/geant4.sh
    export PYTHONPATH=$G4WORKDIR/MuonGeneration/dataAnalysis/
    source /gpfs/users/parbol/root_v6.28.04-install/bin/thisroot.sh
fi


if [ $HOSTNAME == "login2.ifca.es" ] && [ $USER == "dominguezs" ]; then
    # instrucciones de Pablo
    
    echo "                                 ,,,,,,,,,,   "
    echo "Setting up environment in login2_dominguezs..."
    echo "                                 ''''''''''   "
    export G4INSTALLDIR=/gpfs/users/dominguezs/geant4-v11.1.2-install
    export G4WORKDIR=/gpfs/users/dominguezs/Muography_Denoising/
    export JSONCPPDIR=/gpfs/users/dominguezs/jsoncpp/
    source $G4INSTALLDIR/bin/geant4.sh
    export PYTHONPATH=$G4WORKDIR/MuonGeneration/dataAnalysis/
    source /gpfs/users/parbol/root_v6.28.04-install/bin/thisroot.sh	
    echo "Pablo's configuration for root successfully set up [CORRECT]"
    alias micro='/gpfs/users/dominguezs/software/micro'
    echo 'Micro software (text editor) successfully set up ----- [CORRECT]' 
    #source /home/pablo/Documentos/software/root_v6.36.00-install/bin/thisroot.sh
	# custom variables:
    path_data_Analysis="/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/dataAnalysis"
    path_output_raw="/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_raw"
    path_output_preprocessed="/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_preprocessed"
    path_conf_geom="/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/geometric_configurations_json"
    path_generator="/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration-build/Generator"
    path_poca_output="/gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/post_POCA_data"
	echo "Custom path variables successfully set up ----- [CORRECT]"    
fi


