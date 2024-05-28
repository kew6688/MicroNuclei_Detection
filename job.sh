#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=4000M  
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y3229wan@cs.toronto.edu
#SBATCH --job-name=train
#SBATCH --output=train.o
#SBATCH --error=train.e
cd ..
source venv/bin/activate
python MN_Classification/main.py 