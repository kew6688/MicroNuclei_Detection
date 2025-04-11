#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=8G 
#SBATCH --time=00:50:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=markus.wang@mail.utoronto.ca
#SBATCH --job-name=test
#SBATCH --output=test.o
#SBATCH --error=test.e

# ---------------------------------------------------------------------------------------------
echo "Create a virtual env to run"
# ---------------------------------------------------------------------------------------------

module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

# pip install -r requirements.txt

# ---------------------------------------------------------------------------------------------
echo "Install SAM2"
# ---------------------------------------------------------------------------------------------

# install SAM2
cd $SLURM_TMPDIR
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .

# download checkpoints for sam2 under folder /sam2
cd checkpoints && \
./download_ckpts.sh && \
cd ..

# ---------------------------------------------------------------------------------------------
echo "Install mn detection packge"
# ---------------------------------------------------------------------------------------------

# Install package
cd $SLURM_TMPDIR
git clone https://github.com/kew6688/MicroNuclei_Detection.git 
cd MicroNuclei_Detection

# Install requirements and build package
pip install -r requirements_cc.txt 
pip install --no-index -e .

# download checkpoints for our model under /MicroNuclei_Detection/checkpoints
cd checkpoints && \
./download_ckpts.sh && \
cd ..

# Download checkpoints
git clone https://huggingface.co/kew1046/MaskRCNN-resnet50FPN

# ---------------------------------------------------------------------------------------------
echo "Prepare data"
# ---------------------------------------------------------------------------------------------

# mkdir work
# cd work
# tar -xf /home/y3229wan/scratch/MCF10A.tar
# Now do my computations here on the local disk using the contents of the extracted archive...

# ---------------------------------------------------------------------------------------------
echo "Start main process"
# ---------------------------------------------------------------------------------------------

# Run the main python script. 
# The arguments should be 
#       the folder for the input images (png, tif)
#       the final json file name
#       the process mode (ALL for both nuc and mn, NUC for only nuclei, MN for only micronuclei)
# Example:
#       >>> python image_process.sh /home/scratch/test test.json ALL

cd $SLURM_TMPDIR
python MicroNuclei_Detection/compute_scripts/image_process.py \
    --src $1 \
    --dst $2 \
    --mode $3

       
# ---------------------------------------------------------------------------------------------
echo "Save output"
# ---------------------------------------------------------------------------------------------

cp $2 ~/scratch

# The computations are done, so clean up the data set...
# cd $SLURM_TMPDIR
# tar -cf ~/scratch/rep3_10000_processed.tar work/

