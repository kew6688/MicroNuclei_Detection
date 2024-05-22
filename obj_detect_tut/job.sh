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
python obj_detect_tut/detect.py 
# python test.py -m DG -d PDBbind -f nomsa -e af2 -lr 0.0001 -bs 20 -do 0.4 -ne 20sq
