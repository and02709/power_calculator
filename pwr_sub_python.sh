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
trap 'echo; echo "[FATAL] line=$LINENO cmd=$BASH_COMMAND" >&2' ERR

WRKDIR="$1"
CHUNK_SIZE="$2"
NINDEX="$3"
FILEDIR="$4"

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "$TASK_ID" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is empty. Are you running as an array job?" >&2
  exit 2
fi

echo "[INFO] Host: $(hostname)"
echo "[INFO] Task ID: $TASK_ID"
echo "[INFO] WRKDIR: $WRKDIR"
echo "[INFO] CHUNK_SIZE: $CHUNK_SIZE"
echo "[INFO] NINDEX: $NINDEX"
echo "[INFO] FILEDIR: $FILEDIR"

mkdir -p "$WRKDIR/pwr_data"
cd "$WRKDIR/pwr_data"
echo "[INFO] PWD: $(pwd)"

# ----------------------------
# Conda env activation
# ----------------------------
module purge || true
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pwr_py

echo "[INFO] which python: $(which python)"
python -V

python - <<'PY'
import sys
print("[INFO] sys.executable:", sys.executable)
import numpy, pandas
print("[INFO] numpy:", numpy.__version__)
print("[INFO] pandas:", pandas.__version__)
import nibabel
print("[INFO] nibabel:", nibabel.__version__)
PY

# ----------------------------
# Compute START/END (1-based)
# ----------------------------
START=$(( (TASK_ID - 1) * CHUNK_SIZE + 1 ))
END=$(( TASK_ID * CHUNK_SIZE ))
if [[ "$END" -gt "$NINDEX" ]]; then
  END="$NINDEX"
fi
echo "[INFO] Processing rows $START to $END"

# ----------------------------
# REQUIRED simulation inputs
# ----------------------------
PCONN1="/projects/standard/feczk001/shared/projects/ABCD/gordon_sets/data/group2_10minonly_FD0p1/sub-NDARINV00J52GPG_ses-baselineYear1Arm1_task-rest_bold_roi-Gordon2014FreeSurferSubcortical_timeseries.ptseries.nii_5_minutes_of_data_at_FD_0.2.pconn.nii"

N_TIME=2000
SEED=1
USE_ONE_TARGET=1

if [[ ! -f "$PCONN1" ]]; then
  echo "[FATAL] PCONN1 not found: $PCONN1" >&2
  exit 3
fi

CMD=( python "$FILEDIR/pwr_process_chunk_z.py"
  "$WRKDIR" "$START" "$END" "$FILEDIR"
  --pconn1 "$PCONN1"
  --n_time "$N_TIME"
  --seed "$SEED"
)

if [[ "$USE_ONE_TARGET" -eq 1 ]]; then
  CMD+=( --use_one_target )
fi

echo "[INFO] Running:"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "[INFO] Done chunk $TASK_ID"
