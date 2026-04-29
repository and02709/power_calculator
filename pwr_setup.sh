#!/bin/bash -l
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH -p msismall
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=and02709@umn.edu
#SBATCH -o pwr_setup.out
#SBATCH -e pwr_setup.err
#SBATCH --job-name pwr_setup

set -euo pipefail

# ── Arguments ─────────────────────────────────────────────────────────────────
WRKDIR=$1     # Root working directory
FILEDIR=$2    # Pipeline scripts directory
CONDAENV=${3:-}  # Conda environment name (optional; avoids unbound variable under set -u)

# ── Environment ───────────────────────────────────────────────────────────────
module purge || true
if [[ "$CONDAENV" == "FC_stability" ]]; then
  source /projects/standard/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh
fi
if [[ -n "$CONDAENV" ]]; then
  conda activate "$CONDAENV"
fi

# Change into pwr_data/ so pwr_setup.py can write pwr_index_file.txt
cd "${WRKDIR}/pwr_data"

# ── Diagnostics ───────────────────────────────────────────────────────────────
echo "[INFO] Running pwr_setup.py"
echo "[INFO] WRKDIR   = ${WRKDIR}"
echo "[INFO] FILEDIR  = ${FILEDIR}"
echo "[INFO] CONDAENV = ${CONDAENV:-<not set>}"
echo "[INFO] pwd      = $(pwd)"

# ── Run ───────────────────────────────────────────────────────────────────────
python3 "${FILEDIR}/pwr_setup.py"
