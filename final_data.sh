#!/bin/bash -l
# final_data.sh — SLURM batch script for Step 8 of the power calculator pipeline.
#
# Runs final_data.py to aggregate all per-fold R² values produced by Step 7
# (cv.py) into the final power curve outputs. This is the terminal compute
# step of the pipeline — its outputs are what the power calculator reports.
#
# Reads all data_<size>_fold_<k>_cvr2.npy files in pwr_data/ and reduces
# them across folds per sample size to produce summary statistics (mean R²,
# SD, etc.) that form the power curve.
#
# Runs as a single (non-array) job submitted synchronously (--wait) from
# PWR.sh, so the calling shell blocks until aggregation is complete. No
# file-existence guard follows in PWR.sh because a non-zero exit from
# final_data.py propagates through submit() and aborts the pipeline before
# the [DONE] echo is reached.
#
# Note: set -euo pipefail is absent here, consistent with cvGen.sh and
# setupCVmetrics.sh. final_data.py handles its own error reporting.
# Add set -euo pipefail if stricter shell-level error propagation is needed.
#
# Usage (via PWR.sh submit()):
#   sbatch final_data.sh <WRKDIR> <FILEDIR>
#
# Arguments:
#   WRKDIR   - Root working directory; pwr_data/ holds cvr2.npy inputs and
#              final power curve outputs.
#   FILEDIR  - Directory containing final_data.py and all other pipeline scripts.

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20        # final_data.py can parallelise aggregation
#SBATCH --mem=96GB                # All cvr2.npy files loaded simultaneously;
#SBATCH --time=12:00:00           # scales with NUMFILES × KFOLDS × result size
#SBATCH -p msismall
#SBATCH --mail-type=FAIL          # Email only on job failure
#SBATCH --mail-user=and02709@umn.edu
#SBATCH -o final_data.out         # Static filename — SLURM directives are parsed
#SBATCH -e final_data.err         # before the shell executes; variables cannot
                                  # be used in -o/-e paths
#SBATCH --job-name final_data

# ── Arguments ─────────────────────────────────────────────────────────────────
WRKDIR="$1"    # Root working directory
FILEDIR="$2"   # Pipeline scripts directory
CONDAENV="$3"  # Conda environment to activate

# ── Environment ───────────────────────────────────────────────────────────────
# Uses the cluster's system Python3 module, consistent with cvGen.sh and
# setupCVmetrics.sh. Switch to conda activate FC_stability if final_data.py
# requires dependencies beyond the system Python3 environment.
# Note: the if condition is a workaround for using this on MSI where we have to source the conda environment path
# without this, activate would would fail. Need to find better solution for production.
if [[ "$CONDAENV" == "FC_stability" ]]; then
  source /projects/standard/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh
fi
conda activate "$CONDAENV"

# Change into pwr_data/ so final_data.py can resolve cvr2.npy files using
# relative paths if needed. final_data.py also receives WRKDIR as an explicit
# argument for absolute path construction.
cd $WRKDIR/pwr_data

# ── Run ───────────────────────────────────────────────────────────────────────
# Step label comment retained to mark this as the single active compute step
# in this script (no pre/post processing steps here unlike some other wrappers).
# Step 1: generate index file
python3 $FILEDIR/final_data.py $WRKDIR $FILEDIR
