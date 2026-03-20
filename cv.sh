#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
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
EPSILON=$5
INDEX=${SLURM_ARRAY_TASK_ID}
 
# Optional overrides via environment variables
N_COMPONENTS=${N_COMPONENTS:-500}
N_ESTIMATORS=${N_ESTIMATORS:-500}
 
module load python3/3.10.4-anaconda2023.03
 
python3 $FILEDIR/cv.py $WRKDIR $FILEDIR $NUMFILES $KFOLDS $EPSILON $INDEX \
    --n_components $N_COMPONENTS \
    --n_estimators $N_ESTIMATORS