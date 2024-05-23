#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=4000M  
#SBATCH --time=02:00:00
#SBATCH --job-name=d
#SBATCH --output=d.o
#SBATCH --error=d.e
cd ..
source venv/bin/activate
python MN_Classification/train.py 