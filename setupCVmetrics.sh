#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH -p msismall
#SBATCH --mail-type=FAIL  
#SBATCH --mail-user=and02709@umn.edu 
#SBATCH -o setupCVmetrics.out
#SBATCH -e setupCVmetrics.err
#SBATCH --job-name setupCVmetrics

WRKDIR=$1
FILEDIR=$2


module load python3
cd $WRKDIR/pwr_data

# Step 1: generate index file
python3 $FILEDIR/setupCVmetrics.py $WRKDIR $FILEDIR