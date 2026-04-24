#!/bin/bash -l
# pwr_setup.sh — SLURM batch script for Step 1 of the power calculator pipeline.
#
# Runs pwr_setup.py to initialise the simulation workspace. Specifically,
# pwr_setup.py generates pwr_index_file.txt in $WRKDIR/pwr_data/, which
# defines the full index space of sample-size/dataset-size combinations
# that subsequent array jobs (Steps 2–7) iterate over.
#
# This script is submitted synchronously (--wait) from PWR.sh, so the
# calling shell blocks until this job completes before the index file
# guard in PWR.sh runs.
#
# Usage (via PWR.sh submit()):
#   sbatch pwr_setup.sh <WRKDIR> <FILEDIR>
#
# Arguments:
#   WRKDIR  - Root working directory; pwr_data/ subdirectory must exist.
#   FILEDIR - Directory containing pwr_setup.py and all other pipeline scripts.

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH -p msismall
#SBATCH --mail-type=FAIL               # Email only on job failure
#SBATCH --mail-user=and02709@umn.edu
#SBATCH -o pwr_setup.out               # Static filename — not parameterised
#SBATCH -e pwr_setup.err               # because this runs before PWR.sh can
                                       # inject dynamic output paths via submit()
#SBATCH --job-name pwr_setup

# ── Shell options ─────────────────────────────────────────────────────────────
# -e  exit immediately on any non-zero return code
# -u  treat unset variables as errors
# -o pipefail  propagate failures through pipelines (e.g. cmd | wc -l)
set -euo pipefail

# ── Arguments ─────────────────────────────────────────────────────────────────
WRKDIR=$1    # Root working directory
FILEDIR=$2   # Pipeline scripts directory

# ── Environment ───────────────────────────────────────────────────────────────
# Load the Python module available on this cluster. Adjust the module name
# if your cluster uses a versioned name (e.g. "python3/3.11.3").
module load python3

# Change into pwr_data/ so pwr_setup.py can write pwr_index_file.txt using
# relative paths without needing WRKDIR passed as an argument.
cd "${WRKDIR}/pwr_data"

# ── Diagnostics ───────────────────────────────────────────────────────────────
echo "[INFO] Running pwr_setup.py"
echo "[INFO] WRKDIR  = ${WRKDIR}"
echo "[INFO] FILEDIR = ${FILEDIR}"
echo "[INFO] pwd     = $(pwd)"

# ── Run ───────────────────────────────────────────────────────────────────────
python3 "${FILEDIR}/pwr_setup.py"
