# prueba_04mayo.root es el archivo que contiene los hits (crudos) expulsados por Geant4
echo "[INFO] ----- Starting complete pipeline..."

echo  "------------------------------------------------"

python3 ./makeHLTuple.py -i ./prueba_muones_dispersados_raw.root -c ./00_lead_cube_100x100x20.json
-o ./prueba_cubo_grande_07mayo.root

echo  "------------------------------------------------"

# python3 ./MuonGeneration/dataAnalysis/POCA1.py --input prueba_cubo_grande_07mayo.root --output pocapoints_cubo_grande_07mayo.root

echo  "------------------------------------------------"

# python3 ./MuonGeneration/dataAnalysis/visualizar_proyecciones.py --input pocapoints_cubo_grande_07mayo.root
