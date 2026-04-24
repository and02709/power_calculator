#!/bin/bash -l
# combine_data.sh — SLURM batch script for Step 3 of the power calculator pipeline.
#
# Runs combine_data.py to aggregate the per-chunk simulation outputs produced
# by Step 2 into one full_<size>_cov.npy (and corresponding _cor.npy, _z.npy)
# file per sample size. Also applies epsilon noise to the covariance matrices
# at this stage via the EPSILON argument.
#
# Memory is elevated to 64GB because combine_data.py loads and concatenates
# all per-chunk .npy arrays for each sample size into memory simultaneously
# before writing the aggregated output.
#
# Runs as a single (non-array) job submitted synchronously (--wait) from
# PWR.sh, so the calling shell blocks until aggregation is complete before
# the file-existence guards in PWR.sh execute.
#
# Usage (via PWR.sh submit()):
#   sbatch combine_data.sh <WRKDIR> <FILEDIR> <EPSILON>
#
# Arguments:
#   WRKDIR   - Root working directory; pwr_data/ holds chunk inputs and outputs.
#   FILEDIR  - Directory containing combine_data.py and all other pipeline scripts.
#   EPSILON  - Noise magnitude applied to covariance matrices (0 = no noise).

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB          # Elevated: all chunk outputs loaded per sample size at once
#SBATCH --time=1:00:00
#SBATCH -p msismall
#SBATCH -o combine_data.out # Static filename — SLURM directives are parsed before
#SBATCH -e combine_data.err # the shell executes, so variables cannot be used here
#SBATCH --job-name=combine_data

set -euo pipefail   # Exit on error (-e), unset variable (-u), or pipeline failure (-o pipefail)

# ── Arguments ─────────────────────────────────────────────────────────────────
WRKDIR="$1"    # Root working directory
FILEDIR="$2"   # Pipeline scripts directory
EPSILON="$3"   # Epsilon noise magnitude; pass 0 to disable noise addition

# ── Environment ───────────────────────────────────────────────────────────────
# Change into pwr_data/ so combine_data.py can resolve relative chunk output
# paths without requiring an explicit directory argument beyond WRKDIR.
cd "$WRKDIR/pwr_data"

# Activate the FC_stability conda environment which contains all required
# Python dependencies (numpy, pandas, etc.). No `module purge` is needed
# here since this runs as a fresh job with a clean environment.
source /projects/standard/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh
conda activate FC_stability

# ── Run ───────────────────────────────────────────────────────────────────────
python "$FILEDIR/combine_data.py" "$WRKDIR" "$FILEDIR" "$EPSILON"
