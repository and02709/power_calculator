#!/bin/bash -l
# apply_epsilon.sh — SLURM batch script for the "noise" step of the pipeline.
#
# Runs apply_epsilon.py to regenerate full_<size>_yt.npy for a new EPSILON
# from combined data that already exists (typically symlinked from a base run
# via PWR.sh --reuse-from). This is the cheap substitute for re-running
# Steps 1-3 when only epsilon changes.
#
# Resources are modest compared with combine_data.sh: nothing here stacks
# per-subject files. The clean phenotype is a single vector per sample size,
# so memory is only elevated enough to cover the fallback path, where
# full_<size>_cor.npy must be loaded to recompute y = cor @ ridge_vec.
#
# Runs as a single (non-array) job submitted synchronously (--wait) from
# PWR.sh, so the CV step only begins once every yt file has been written.
#
# Usage (via PWR.sh submit()):
#   sbatch apply_epsilon.sh <WRKDIR> <FILEDIR> <EPSILON> <CONDAENV> [SEED]
#
# Arguments:
#   WRKDIR   - Root working directory; pwr_data/ holds the combined matrices.
#   FILEDIR  - Directory containing apply_epsilon.py and combine_data.py.
#   EPSILON  - Noise scale factor applied to the phenotype (0 = no noise).
#   CONDAENV - Conda environment to activate for Python execution.
#   SEED     - Optional RNG seed; omit or pass an empty string for a
#              nondeterministic draw.

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB          # Headroom for the cor-matrix fallback path
#SBATCH --time=1:00:00
#SBATCH -p msismall
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=and02709@umn.edu
#SBATCH -o apply_epsilon.out # Static filename — SLURM directives are parsed
#SBATCH -e apply_epsilon.err # before the shell runs, so no variables here
#SBATCH --job-name=apply_epsilon

set -euo pipefail   # Exit on error (-e), unset variable (-u), or pipeline failure (-o pipefail)

# ── Arguments ─────────────────────────────────────────────────────────────────
WRKDIR="$1"        # Root working directory
FILEDIR="$2"       # Pipeline scripts directory
EPSILON="$3"       # Epsilon noise magnitude; pass 0 to disable noise addition
CONDAENV="$4"      # Conda environment to activate
SEED="${5:-}"      # Optional RNG seed (empty = nondeterministic)

# ── Environment ───────────────────────────────────────────────────────────────
# Change into pwr_data/ to match the working directory convention used by the
# other step scripts.
cd "$WRKDIR/pwr_data"

# Note: the if condition is a workaround for using this on MSI where the conda
# environment path has to be sourced before activate will work.
if [[ "$CONDAENV" == "FC_stability" ]]; then
  source /projects/standard/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh
fi
conda activate "$CONDAENV"

# ── Run ───────────────────────────────────────────────────────────────────────
SEED_FLAG=()
if [[ -n "$SEED" ]]; then
  SEED_FLAG=(--seed "$SEED")
fi

python3 "$FILEDIR/apply_epsilon.py" "$WRKDIR" "$FILEDIR" "$EPSILON" "${SEED_FLAG[@]}"
