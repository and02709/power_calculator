#!/bin/bash -l
# cvGen.sh — SLURM array task script for Step 5 of the power calculator pipeline.
#
# Generates cross-validation fold indices for one sample size per array task.
# Each task calls cvGen.py, which reads the corresponding full_<size>_cov.npy
# produced in Step 3 and writes KFOLDS train/test index files for that size.
#
# Array indexing: one task per sample size (1..NUMFILES). SLURM_ARRAY_TASK_ID
# is passed directly to cvGen.py as INDEX, which maps it to the corresponding
# sample size by sorting the available full_*_cov.npy files and selecting the
# INDEX-th entry (1-based).
#
# Unlike the simulation array scripts (Steps 2), there is no CHUNK_SIZE or
# START/END logic here — each task owns exactly one sample size end-to-end.
#
# Submitted synchronously (--wait) from PWR.sh so Step 6 (setupCVmetrics)
# only begins once fold indices exist for all sample sizes.
#
# Note: set -euo pipefail is absent here. cvGen.py handles its own error
# reporting; add set -euo pipefail if stricter shell-level error propagation
# is needed.
#
# Usage (via PWR.sh submit()):
#   sbatch --array=1-$NUMFILES cvGen.sh <WRKDIR> <FILEDIR> <NUMFILES> <KFOLDS>
#
# Arguments:
#   WRKDIR   - Root working directory; pwr_data/ holds full_*_cov.npy inputs
#              and fold index outputs.
#   FILEDIR  - Directory containing cvGen.py and all other pipeline scripts.
#   NUMFILES - Total number of sample sizes (= array upper bound).
#   KFOLDS   - Number of CV folds to generate per sample size.

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --mem=96GB               # Elevated relative to other steps — cvGen.py
#SBATCH --time=12:00:00          # loads the full full_<size>_cov.npy into memory
#SBATCH -p msismall              # for the largest sample sizes (up to 2000 × n_edge)
#SBATCH --mail-type=FAIL         # Email only on job failure
#SBATCH --mail-user=and02709@umn.edu
#SBATCH -o cvGen_%a.out          # %a = array task ID (one log file per sample size)
#SBATCH -e cvGen_%a.err
#SBATCH --job-name cvGen

# ── Arguments ─────────────────────────────────────────────────────────────────
WRKDIR="$1"      # Root working directory
FILEDIR="$2"     # Pipeline scripts directory
NUMFILES="$3"    # Total number of sample sizes; passed through to cvGen.py
KFOLDS="$4"      # Number of CV folds per sample size
CONDAENV="$5"   # Conda environment to activate for Python dependencies

# SLURM_ARRAY_TASK_ID is the 1-based index of this task within the array.
# Passed to cvGen.py as INDEX, which uses it to select the target sample size.
INDEX=${SLURM_ARRAY_TASK_ID}

# ── Environment ───────────────────────────────────────────────────────────────
# Uses the cluster's system Python3 module rather than the FC_stability conda
# environment. Ensure all required packages (numpy, scikit-learn, etc.) are
# available in the loaded module, or switch to a conda activate approach
# if dependency conflicts arise.
# Note: the if condition is a workaround for using this on MSI where we have to source the conda environment path
# without this, activate would would fail. Need to find better solution for production.
if [[ "$CONDAENV" == "FC_stability" ]]; then
  source /projects/standard/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh
fi
conda activate "$CONDAENV"

# ── Run ───────────────────────────────────────────────────────────────────────
python3 $FILEDIR/cvGen.py $WRKDIR $FILEDIR $NUMFILES $KFOLDS $INDEX
