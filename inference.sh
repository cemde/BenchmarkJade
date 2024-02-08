#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=00-00:30:00
#SBATCH --job-name=inference
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:8
#SBATCH --partition=big
#SBATCH --output ./logs/slurm/slurm-%j-%x.out # STDOUT

# define variables
PYTHON="/jmain02/home/J2AD009/ttl04/cxe09-ttl04/anaconda3/envs/advcal/bin/python"
HOME_DIR="/jmain02/home/J2AD009/ttl04/cxe09-ttl04/BenchmarkJade"
DATASET="imagenet"

module purge

$PYTHON ${HOME_DIR}/inference.py --dataset ${DATASET}
