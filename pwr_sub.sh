#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH -p msismall
#SBATCH --mail-type=FAIL  
#SBATCH --mail-user=and02709@umn.edu 
#SBATCH -o PWR_Sub_%a.out
#SBATCH -e PWR_Sub_%a.err
#SBATCH --job-name PWR_Sub

# Arguments
WRKDIR=$1
CHUNK_SIZE=$2
NINDEX=$3

FILEDIR=/users/0/and02709/power_calculator

# SLURM_ARRAY_TASK_ID is automatically set (1, 2, ..., NJOBS)
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Calculate START and END index based on chunk size
START=$(( (TASK_ID - 1) * CHUNK_SIZE + 1 ))
END=$(( TASK_ID * CHUNK_SIZE ))

# Make sure END doesn't exceed total number of lines
if [ "$END" -gt "$NINDEX" ]; then
  END=$NINDEX
fi

echo "Task $TASK_ID: processing rows $START to $END"

# Call R script to process this chunk
Rscript $FILEDIR/pwr_process_chunk.R $WRKDIR $START $END $FILEDIR
