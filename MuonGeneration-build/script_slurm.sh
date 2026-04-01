#!/bin/bash
#SBATCH --job-name=POCA_Reconstruction
#SBATCH --partition=wncompute_ifca      # La particion que te indico tu profesor
#SBATCH --output=poca_res_%j.out        # Donde se guarda el output (standard out)
#SBATCH --error=poca_res_%j.err         # Donde se guardan los errores (standard error)
#SBATCH --ntasks=1                      # Un solo proceso logico
#SBATCH --cpus-per-task=8               # 8 CPUs para que ROOT vuele con MT
#SBATCH --mem=16G                       # 16 Gigas de RAM para no saturar el nodo
#SBATCH --time=01:00:00                 # Tiempo maximo de 1 hora

# 1. Cargar el entorno de trabajo
# Es fundamental para que el nodo de computo sepa donde esta ROOT y Python
source "/gpfs/users/dominguezs/Muography_Denoising/setup.sh"

# 2. Informacion de diagnostico en el .out
echo "Job iniciado en el nodo: $SLURM_NODELIST"
echo "Fecha y hora: $(date)"

# 3. Ejecutar tu script principal
# Usamos bash para lanzar tu run_test.sh que ya tiene toda la logica
bash /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration-build/run_test.sh

echo "Job finalizado con exito: $(date)"
