#!/bin/bash
#SBATCH --job-name=test_simple
#SBATCH --output=/tmp/test_simple.out
#SBATCH --error=/tmp/test_simple.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:01:00

echo "Job started at $(date)"
sleep 10
echo "Job ended at $(date)"
