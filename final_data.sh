#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH -p msismall
#SBATCH --mail-type=FAIL  
#SBATCH --mail-user=and02709@umn.edu 
#SBATCH -o final_data.out
#SBATCH -e final_data.err
#SBATCH --job-name final_data

WRKDIR=$1
KFOLDS=$2
PHENO=$3
FILEDIR=$4

module load R/4.4.0-openblas-rocky8
cd $WRKDIR/pwr_data

# Step 1: generate index file
Rscript $FILEDIR/final_data.R $WRKDIR $FILEDIR