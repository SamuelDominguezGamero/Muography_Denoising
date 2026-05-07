# prueba_04mayo.root es el archivo que contiene los hits (crudos) expulsados por Geant4
echo "[INFO] ----- Starting complete pipeline..."

echo  "------------------------------------------------"

python3 ./MuonGeneration/dataAnalysis/makeHLTuple_QUICK.py --input prueba_04mayo.root --output dataset_correlacionado_fixed_QUICK.root

echo  "------------------------------------------------"

python3 ./MuonGeneration/dataAnalysis/POCA1.py --input dataset_correlacionado_fixed_QUICK.root --output poca_points_QUICK.root

echo  "------------------------------------------------"

python3 ./MuonGeneration/dataAnalysis/visualizar_proyecciones_QUICK.py --input poca_points_QUICK.root
