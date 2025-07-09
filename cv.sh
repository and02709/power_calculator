#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH -p msismall
#SBATCH --mail-type=FAIL  
#SBATCH --mail-user=and02709@umn.edu 
#SBATCH -o cv_%a.out
#SBATCH -e cv_%a.err
#SBATCH --job-name cv

# Arguments
WRKDIR=$1
FILEDIR=$2 
NUMFILES=$3
KFOLDS=$4
INDEX=${SLURM_ARRAY_TASK_ID}

module load R/4.4.0-openblas-rocky8
Rscript $FILEDIR/cv.R $WRKDIR $FILEDIR $NUMFILES $KFOLDS $INDEX
