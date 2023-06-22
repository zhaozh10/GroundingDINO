#!/bin/bash
#SBATCH -p bme_cpu
#SBATCH --job-name=rgrg_data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH -t 5-00:00:00
source activate nlp

conda list
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python create_dataset.py




