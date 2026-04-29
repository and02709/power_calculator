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

# ── Arguments ─────────────────────────────────────────────────────────────────
WRKDIR=$1     # Root working directory
FILEDIR=$2    # Pipeline scripts directory
CONDAENV=${3:-}   # Conda environment name (optional; safe default avoids set -u error)

# ── Environment ───────────────────────────────────────────────────────────────
module purge || true
if [[ "$CONDAENV" == "FC_stability" ]]; then
  source /projects/standard/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh
fi
if [[ -n "$CONDAENV" ]]; then
  conda activate "$CONDAENV"
fi

cd "${WRKDIR}/pwr_data"

# ── Diagnostics ───────────────────────────────────────────────────────────────
echo "[INFO] Running pwr_index_file.py"
echo "[INFO] WRKDIR   = ${WRKDIR}"
echo "[INFO] FILEDIR  = ${FILEDIR}"
echo "[INFO] CONDAENV = ${CONDAENV:-<not set>}"
echo "[INFO] pwd      = $(pwd)"

# ── Run ───────────────────────────────────────────────────────────────────────
python3 "${FILEDIR}/pwr_index_file.py"
