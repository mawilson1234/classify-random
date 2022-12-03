#!/bin/bash

#SBATCH --job-name=classify-random
#SBATCH --output=joblogs/test_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL

module load miniconda

source activate cls-ran

python classify-random.py -m \
	ntrials=1000 \
	ngroups=2,3,4 \
	ntrain=10000 \
	ntest=500 \
	ndims=10,100,768
