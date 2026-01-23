#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH -p msismall
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=and02709@umn.edu
#SBATCH -o pwr_index_file.out
#SBATCH -e pwr_index_file.err
#SBATCH --job-name pwr_index_file

set -euo pipefail

WRKDIR=$1
FILEDIR=$2

# ---- Load Python instead of R ----
module load python3  # adjust if your cluster uses a different module

cd "${WRKDIR}/pwr_data"

echo "[INFO] Running pwr_index_file.py"
echo "[INFO] WRKDIR  = ${WRKDIR}"
echo "[INFO] FILEDIR = ${FILEDIR}"
echo "[INFO] pwd     = $(pwd)"

# ---- Run Python script ----
python3 "${FILEDIR}/pwr_index_file.py"
