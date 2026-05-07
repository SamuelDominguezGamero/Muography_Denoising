# prueba_04mayo.root es el archivo que contiene los hits (crudos) expulsados por Geant4
echo "[INFO] ----- Starting complete pipeline..."

echo  "------------------------------------------------"

python3 ./MuonGeneration/dataAnalysis/makeHLTuple.py -i prueba_04mayo.root -c /home/samuel/Work/Muography_Denoising/_Lpx128_Lpy128_Lpz128_npx128_npy128_npz128_zTop54_zBot-54_spacing5_ratio1_FontX16_FontY16_matiron_wordNMUO_stroke2.json -o dataset_correlacionado_fixed.root

echo  "------------------------------------------------"

python3 ./MuonGeneration/dataAnalysis/POCA1.py --input dataset_correlacionado_fixed.root --output poca_points.root

echo  "------------------------------------------------"

python3 ./MuonGeneration/dataAnalysis/visualizar_proyecciones.py --input poca_points.root
