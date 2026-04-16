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

set -euo pipefail

echo "=================== PWR_Sub.sh START ==================="
date
echo "[INFO] SLURM_JOB_ID           = ${SLURM_JOB_ID:-NA}"
echo "[INFO] SLURM_ARRAY_JOB_ID     = ${SLURM_ARRAY_JOB_ID:-NA}"
echo "[INFO] SLURM_ARRAY_TASK_ID    = ${SLURM_ARRAY_TASK_ID:-NA}"
echo "[INFO] SLURM_JOB_NODELIST     = ${SLURM_JOB_NODELIST:-NA}"
echo "--------------------------------------------------------"

# Arguments
WRKDIR=$1
CHUNK_SIZE=$2
NINDEX=$3
FILEDIR=$4

echo "[ARGS] WRKDIR     = ${WRKDIR}"
echo "[ARGS] CHUNK_SIZE = ${CHUNK_SIZE}"
echo "[ARGS] NINDEX     = ${NINDEX}"
echo "[ARGS] FILEDIR    = ${FILEDIR}"
echo "--------------------------------------------------------"

# Make sure FILEDIR and pwr_process_chunk.R exist
echo "[DEBUG] Listing FILEDIR:"
ls -ld "${FILEDIR}" || { echo "[FATAL] FILEDIR does not exist: ${FILEDIR}"; exit 1; }

echo "[DEBUG] Checking pwr_process_chunk.R:"
if [ ! -f "${FILEDIR}/pwr_process_chunk.R" ]; then
    echo "[FATAL] ${FILEDIR}/pwr_process_chunk.R not found!"
    exit 1
fi

ls -l "${FILEDIR}/pwr_process_chunk.R"
echo "[DEBUG] First 10 lines of pwr_process_chunk.R:"
head -n 10 "${FILEDIR}/pwr_process_chunk.R"
echo "--------------------------------------------------------"

# Show current working directory
echo "[DEBUG] pwd = $(pwd)"
echo "[DEBUG] Listing current directory:"
ls
echo "--------------------------------------------------------"

# SLURM_ARRAY_TASK_ID is automatically set (1, 2, ..., NJOBS)
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Calculate START and END index based on chunk size
START=$(( (TASK_ID - 1) * CHUNK_SIZE + 1 ))
END=$(( TASK_ID * CHUNK_SIZE ))

# Make sure END doesn't exceed total number of lines
if [ "${END}" -gt "${NINDEX}" ]; then
  END=${NINDEX}
fi

echo "[CHUNK] Task ${TASK_ID}: processing rows ${START} to ${END}"
echo "--------------------------------------------------------"

# (Optional) load R here if not already loaded in parent script
# module load R/4.4.0-openblas-rocky8

# Call R script to process this chunk
echo "[RUN] Rscript ${FILEDIR}/pwr_process_chunk.R ${WRKDIR} ${START} ${END} ${FILEDIR}"
Rscript "${FILEDIR}/pwr_process_chunk.R" "${WRKDIR}" "${START}" "${END}" "${FILEDIR}"
RETVAL=$?

echo "--------------------------------------------------------"
echo "[INFO] Rscript exit status = ${RETVAL}"
date
echo "=================== PWR_Sub.sh END ====================="

exit "${RETVAL}"
