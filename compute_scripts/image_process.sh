#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G 
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=markus.wang@mail.utoronto.ca
#SBATCH --job-name=test
#SBATCH --output=test.o
#SBATCH --error=test.e

# ---------------------------------------------------------------------------------------------
echo "Create a virtual env to run"
# ---------------------------------------------------------------------------------------------

# module load python/3.12
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index --upgrade pip


# ---------------------------------------------------------------------------------------------
echo "Install SAM2"
# ---------------------------------------------------------------------------------------------

# git clone https://github.com/facebookresearch/sam2.git
# cd /home/y3229wan/projects/def-sushant/y3229wan/mn-project/sam
# pip install --no-index -e .
# cd checkpoints && \
# ./download_ckpts.sh && \
# cd ..


# ---------------------------------------------------------------------------------------------
echo "Install mn detection packge"
# ---------------------------------------------------------------------------------------------

# git clone https://github.com/kew6688/MicroNuclei_Detection.git 
# cd /home/y3229wan/projects/def-sushant/y3229wan/mn-project/MN
# pip install --no-index -e .


# ---------------------------------------------------------------------------------------------
echo "Prepare data"
# ---------------------------------------------------------------------------------------------

# cd $SLURM_TMPDIR
# mkdir work
# cd work
# tar -xf /home/y3229wan/scratch/MCF10A.tar
# Now do my computations here on the local disk using the contents of the extracted archive...


# ---------------------------------------------------------------------------------------------
echo "Start main process"
# ---------------------------------------------------------------------------------------------

python image_process.py \
    --src $1 \
    --dst $2
    --mode $3

echo $1
       
# ---------------------------------------------------------------------------------------------
echo "Save output"
# ---------------------------------------------------------------------------------------------

# The computations are done, so clean up the data set...
# cd $SLURM_TMPDIR
# tar -cf ~/scratch/rep3_10000_processed.tar work/

