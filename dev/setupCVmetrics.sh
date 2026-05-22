#!/bin/bash -l
# setupCVmetrics.sh — SLURM batch script for Step 6 of the power calculator pipeline.
#
# Runs setupCVmetrics.py to prepare the CV metrics framework consumed by
# Step 7 (cv.sh / cv.py). Specifically, setupCVmetrics.py reads the
# full_<size>_fold_<k>_split.npz files produced by Step 5 (cvGen.py) and
# generates any additional index structures or metric scaffolding that
# cv.py requires before fitting models.
#
# Runs as a single (non-array) job submitted synchronously (--wait) from
# PWR.sh, so the calling shell blocks until this step completes before the
# NUMFFILES guard in PWR.sh executes.
#
# Note: set -euo pipefail is absent here, consistent with cvGen.sh.
# Add it if stricter shell-level error propagation is needed; setupCVmetrics.py
# handles its own error reporting.
#
# Usage (via PWR.sh submit()):
#   sbatch setupCVmetrics.sh <WRKDIR> <FILEDIR>
#
# Arguments:
#   WRKDIR   - Root working directory; pwr_data/ holds the split .npz inputs
#              and any metric scaffolding outputs.
#   FILEDIR  - Directory containing setupCVmetrics.py and all other pipeline scripts.

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --mem=96GB               # Matches cvGen.sh — may need to load split .npz
#SBATCH --time=12:00:00          # files across all sizes/folds simultaneously
#SBATCH -p msismall
#SBATCH --mail-type=FAIL         # Email only on job failure
#SBATCH --mail-user=and02709@umn.edu
#SBATCH -o setupCVmetrics.out    # Static filename — SLURM directives are parsed
#SBATCH -e setupCVmetrics.err    # before the shell executes; variables cannot
                                 # be used in -o/-e paths
#SBATCH --job-name setupCVmetrics

# ── Arguments ─────────────────────────────────────────────────────────────────
WRKDIR="$1"    # Root working directory
FILEDIR="$2"   # Pipeline scripts directory
CONDAENV="$3"  # Conda environment to activate

# ── Environment ───────────────────────────────────────────────────────────────
# Uses the cluster's system Python3 module, consistent with cvGen.sh.
# Switch to conda activate FC_stability if additional dependencies are needed.
# Note: the if condition is a workaround for using this on MSI where we have to source the conda environment path
# without this, activate would would fail. Need to find better solution for production.
if [[ "$CONDAENV" == "FC_stability" ]]; then
  source /projects/standard/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh
fi
conda activate "$CONDAENV"

# Change into pwr_data/ so setupCVmetrics.py can resolve split .npz files
# using relative paths if needed. setupCVmetrics.py also receives WRKDIR
# as an explicit argument for absolute path construction.
cd $WRKDIR/pwr_data

# ── Run ───────────────────────────────────────────────────────────────────────
python3 $FILEDIR/setupCVmetrics.py $WRKDIR $FILEDIR
