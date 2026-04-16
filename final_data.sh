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
FILEDIR=$2

module load python3
cd $WRKDIR/pwr_data

# Step 1: generate index file
python3 $FILEDIR/final_data.py $WRKDIR $FILEDIR