#!/bin/bash -l
# pwr_sub_python.sh — SLURM array task script for Step 2 (multi-template mode).
#
# Each array task processes a contiguous chunk of rows from pwr_index_file.txt,
# calling pwr_process_chunk_z.py to simulate pconn matrices and phenotype
# vectors for the sample sizes in that chunk.
#
# In multi-template mode, each simulation averages NUMTEMP randomly drawn
# pconn templates before decomposing the result via EigSplit. Unlike
# single-template mode, the template is drawn once per chunk (outside the
# per-row loop in pwr_process_chunk_z.py) rather than once per row.
#
# This script is the multi-template counterpart to pwr_sub_python_single.sh.
# Structural differences from that script:
#   - Adds NUMTEMP as positional argument $7 (shifts NREP and NTIME to $8/$9)
#   - SINGLEPCONN is a local placeholder variable rather than a bare literal,
#     though it serves the same parser-appeasement purpose
#   - Calls pwr_process_chunk_z.py instead of pwr_process_chunk_single_z.py
#
# Usage (via PWR.sh submit()):
#   sbatch --array=1-$NJOBS pwr_sub_python.sh \
#     <WRKDIR> <CHUNK_SIZE> <NINDEX> <FILEDIR> <PCONNDIR> <PCONNREF> \
#     <NUMTEMP> <NREP> <NTIME>

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
PCONNREF="${6}"     # Reference pconn — passed as --pconn1 (accepted for compatibility
                    # by the Python parser but the template is drawn from PCONNDIR)
NUMTEMP="${7}"      # Number of pconn templates to average per simulation row
NREP="${8}"         # Number of simulation repetitions per index row
NTIME="${9}"        # Positional timepoint count (overridden by N_TIME below)

TASK_ID="${SLURM_ARRAY_TASK_ID}"   # 1-based index assigned by SLURM for this task

# ── Simulation parameters ─────────────────────────────────────────────────────
# N_TIME overrides the NTIME argument passed from PWR.sh. The hardcoded value
# of 2000 reflects the expected timeseries length for this dataset.
N_TIME=2000

# Placeholder passed to the SINGLEPCONN positional argument in pwr_process_chunk_z.py.
# The Python parser requires something in that position but the value is never read.
# Named here (rather than inlined as a bare literal) to make the intent explicit.
SINGLEPCONN="placeholder"

# ── Chunk boundary computation ────────────────────────────────────────────────
# Convert 1-based TASK_ID into the START/END row range within pwr_index_file.txt.
# The last task's END is clamped to NINDEX so it doesn't request rows beyond
# the file (when NINDEX is not a perfect multiple of CHUNK_SIZE).
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
# FC_stability conda environment which contains all required Python dependencies.
module purge || true   # || true prevents -e from aborting if no modules are loaded
source /projects/standard/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh
conda activate FC_stability

# Resolve the active Python binary explicitly after conda activation.
PYTHON_BIN=$(which python)
echo "[INFO] Using Python: $PYTHON_BIN"
echo "[INFO] Processing Rows: $START to $END"

# ── Run ───────────────────────────────────────────────────────────────────────
# Argument order matches pwr_process_chunk_z.py's positional parser exactly:
#   WRKDIR START END FILEDIR PCONNDIR PCONNREF NUMTEMP NREP NTIME SINGLEPCONN
# --n_time overrides the positional NTIME with the hardcoded 2000.
# --use_one_target and --pconn1 are passed for parser compatibility;
# their effect depends on the Python script's internal logic.
"$PYTHON_BIN" "$FILEDIR/pwr_process_chunk_z.py" \
    "$WRKDIR" \
    "$START" \
    "$END" \
    "$FILEDIR" \
    "$PCONNDIR" \
    "$PCONNREF" \
    "$NUMTEMP" \         # Number of templates to average (absent in single-template counterpart)
    "$NREP" \
    "$NTIME" \
    "$SINGLEPCONN" \     # Placeholder for parser compatibility; never read by pwr_process_chunk_z.py
    --n_time "$N_TIME" \ # Overrides positional NTIME with hardcoded 2000
    --use_one_target \
    --pconn1 "$PCONNREF"

echo "[INFO] Task $TASK_ID finished successfully."
