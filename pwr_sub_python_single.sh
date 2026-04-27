#!/bin/bash -l
# pwr_sub_python_single.sh — SLURM array task script for Step 2 (single-template mode).
#
# Each array task processes a contiguous chunk of rows from pwr_index_file.txt,
# calling pwr_process_chunk_single_z.py to simulate pconn matrices and phenotype
# vectors for the sample sizes in that chunk.
#
# In single-template mode, each simulation row draws one fresh random pconn
# template independently rather than averaging across NUMTEMP templates.
# PCONNREF is passed as both the reference pconn (for matrix dimensions) and
# the sole template source (--pconn1), making the simulation self-contained
# with respect to a single subject's connectivity structure.
#
# This script is the single-template counterpart to pwr_sub_python.sh.
# The only structural difference is the absence of a NUMTEMP argument and
# the addition of --use_one_target and --pconn1 flags.
#
# Usage (via PWR.sh submit()):
#   sbatch --array=1-$NJOBS pwr_sub_python_single.sh \
#     <WRKDIR> <CHUNK_SIZE> <NINDEX> <FILEDIR> <PCONNDIR> <PCONNREF> <NREP> <NTIME>

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=10:00:00
#SBATCH -p msismall
#SBATCH -o pwr_sub_%A_%a.out   # %A = job ID, %a = array task ID
#SBATCH -e pwr_sub_%A_%a.err
#SBATCH --job-name=pwr_sub_py

set -euo pipefail   # Exit on error (-e), unset variable (-u), or pipeline failure (-o pipefail)

# ── Arguments ─────────────────────────────────────────────────────────────────
WRKDIR="${1}"       # Root working directory; pwr_data/ subdirectory holds outputs
CHUNK_SIZE="${2}"   # Number of index rows assigned to each array task
NINDEX="${3}"       # Total rows in pwr_index_file.txt (used to clamp the last chunk)
FILEDIR="${4}"      # Directory containing all pipeline scripts
PCONNDIR="${5}"     # Directory of subject pconn files used as simulation templates
PCONNREF="${6}"     # Single reference pconn — defines matrix dimensions and is passed
                    # as --pconn1 (the one template used in single-template mode)
NREP="${7}"         # Number of simulation repetitions per index row
NTIME="${8}"        # Number of timepoints (passed to Python but overridden — see below)
CONDENV="${9}"      # Conda environment to activate for Python execution

TASK_ID="${SLURM_ARRAY_TASK_ID}"   # 1-based index assigned by SLURM for this task

# ── Simulation parameters ─────────────────────────────────────────────────────
# N_TIME overrides the NTIME argument passed from PWR.sh. The hardcoded value
# of 2000 reflects the expected timeseries length for this dataset and takes
# precedence over whatever was passed on the command line.
N_TIME=2000

# ── Chunk boundary computation ────────────────────────────────────────────────
# Convert 1-based TASK_ID into the START/END row range within pwr_index_file.txt.
# The last task's END is clamped to NINDEX so it doesn't request rows that
# don't exist (when NINDEX is not a perfect multiple of CHUNK_SIZE).
#
# Example: CHUNK_SIZE=100, NINDEX=250, TASK_ID=3
#   START = (3-1)*100 + 1 = 201
#   END   = min(3*100, 250) = 250
START=$(( (TASK_ID - 1) * CHUNK_SIZE + 1 ))
END=$(( TASK_ID * CHUNK_SIZE ))
if [[ "$END" -gt "$NINDEX" ]]; then
  END="$NINDEX"
fi

# ── Environment ───────────────────────────────────────────────────────────────
# Purge any inherited modules to avoid version conflicts, then activate the
# FC_stability conda environment which contains all required Python dependencies
# (numpy, nibabel, etc.).
module purge || true   # || true prevents -e from aborting if no modules are loaded
# Note: the if condition is a workaround for using this on MSI where we have to source the conda environment path
# without this, activate would would fail. Need to find better solution for production.
if [[ "$CONDAENV" == "FC_stability" ]]; then
  source /projects/standard/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh
fi
conda activate "$CONDAENV"

# Resolve the active Python binary explicitly rather than relying on PATH,
# ensuring the correct interpreter is used after conda activation.
PYTHON_BIN=$(which python)
echo "[INFO] Using Python: $PYTHON_BIN"
echo "[INFO] Processing Rows: $START to $END"

# ── Run ───────────────────────────────────────────────────────────────────────
# "placeholder" occupies the NUMTEMP positional argument expected by the
# script's argument parser. Single-template mode ignores this value entirely
# but the parser still requires something in that position.
"$PYTHON_BIN" "$FILEDIR/pwr_process_chunk_single_z.py" \
    "$WRKDIR" \
    "$START" \
    "$END" \
    "$FILEDIR" \
    "$PCONNDIR" \
    "$PCONNREF" \
    "$NREP" \
    "$NTIME" \
    "placeholder" \       # Satisfies NUMTEMP positional arg; ignored at runtime
    --n_time "$N_TIME" \  # Overrides NTIME with the hardcoded 2000
    --use_one_target \    # Activates single-template mode in pwr_process_chunk_single_z.py
    --pconn1 "$PCONNREF"  # Supplies PCONNREF as the sole simulation template

echo "[INFO] Task $TASK_ID finished successfully."
