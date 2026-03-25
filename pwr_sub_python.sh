#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=10:00:00
#SBATCH -p msismall
#SBATCH -o pwr_sub_%A_%a.out
#SBATCH -e pwr_sub_%A_%a.err
#SBATCH --job-name=pwr_sub_py

set -euo pipefail

# 1. Inputs from command line arguments
WRKDIR="${1}"
CHUNK_SIZE="${2}"
NINDEX="${3}"
FILEDIR="${4}"
PCONNDIR="${5}"
PCONNREF="${6}"
NUMTEMP="${7}"
NREP="${8}"
TASK_ID="${SLURM_ARRAY_TASK_ID}"

# 2. Define Simulation Parameters (Fixes "SEED: unbound variable")
SEED=1
N_TIME=2000
USE_ONE_TARGET=1
SINGLEPCONN="placeholder"

# 3. Compute START/END indices
START=$(( (TASK_ID - 1) * CHUNK_SIZE + 1 ))
END=$(( TASK_ID * CHUNK_SIZE ))
if [[ "$END" -gt "$NINDEX" ]]; then
  END="$NINDEX"
fi

# 4. Environment Activation
module purge || true
source /projects/standard/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh
conda activate FC_stability

# 5. Dynamically find Python
PYTHON_BIN=$(which python)
echo "[INFO] Using Python: $PYTHON_BIN"
echo "[INFO] Processing Rows: $START to $END with SEED: $SEED"

# 6. Execute Python Script
"$PYTHON_BIN" "$FILEDIR/pwr_process_chunk_z.py" \
    "$WRKDIR" \
    "$START" \
    "$END" \
    "$FILEDIR" \
    "$PCONNDIR" \
    "$PCONNREF" \
    "$NUMTEMP" \
    "$NREP" \
    "$SINGLEPCONN" \
    --seed "$SEED" \
    --n_time "$N_TIME" \
    --use_one_target \
    --pconn1 "$PCONNREF"

echo "[INFO] Task $TASK_ID finished successfully."
