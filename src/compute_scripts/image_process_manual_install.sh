#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=8G 
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=markus.wang@mail.utoronto.ca
#SBATCH --job-name=test
#SBATCH --output=test.o
#SBATCH --error=test.e

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

python image_process.py \
    --src $1 \
    --dst $2 \
    --mode $3

echo $1
       
# ---------------------------------------------------------------------------------------------
echo "Save output"
# ---------------------------------------------------------------------------------------------

# The computations are done, so clean up the data set...
# cd $SLURM_TMPDIR
# tar -cf ~/scratch/rep3_10000_processed.tar work/

