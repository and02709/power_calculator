#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH -p msismall
#SBATCH --mail-type=FAIL  
#SBATCH --mail-user=and02709@umn.edu 
#SBATCH -o PWR_Ridge.out
#SBATCH -e PWR_Ridge.err
#SBATCH --job-name PWR_Ridge

# Arguments
WRKDIR=$1
FILEDIR=$2
PHENO=$3

module load python3

# Call R script to process this chunk
python3 $FILEDIR/ridge_model_generation.py $WRKDIR $FILEDIR $PHENO
