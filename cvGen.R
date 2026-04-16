#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH -p msismall
#SBATCH --mail-type=FAIL  
#SBATCH --mail-user=and02709@umn.edu 
#SBATCH -o cvGen_%a.out
#SBATCH -e cvGen_%a.err
#SBATCH --job-name cvGen

# Arguments
WRKDIR=$1
FILEDIR=$2 
NUMFILES=$3
KFOLDS=$4
INDEX=${SLURM_ARRAY_TASK_ID}

module load python3
python3 $FILEDIR/cvGen.py $WRKDIR $FILEDIR $NUMFILES $KFOLDS $INDEX
