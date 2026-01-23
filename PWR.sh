#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24GB
#SBATCH --time=12:00:00
#SBATCH -p msismall
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=and02709@umn.edu
#SBATCH --job-name=PWR

set -euo pipefail

WRKDIR=$1
KFOLDS=$2
EPSILON=$3
FILEDIR=/scratch.global/and02709/second_python

OUTDIR="$WRKDIR/OUT"
ERRDIR="$WRKDIR/ERR"
PWRDATA="$WRKDIR/pwr_data"

mkdir -p "$OUTDIR" "$ERRDIR" "$PWRDATA"

echo "[INFO] WRKDIR=$WRKDIR"
echo "[INFO] KFOLDS=$KFOLDS"
echo "[INFO] EPSILON=$EPSILON"
echo "[INFO] FILEDIR=$FILEDIR"
echo "[INFO] PWRDATA=$PWRDATA"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-<none>}"
echo "[INFO] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<none>}"

module load R/4.4.0-openblas-rocky8
cd "$PWRDATA"

# EPSILON check (requires bc)
if ! [[ "$EPSILON" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[FATAL] EPSILON must be numeric" >&2
  exit 1
fi
if (( $(echo "$EPSILON <= 0" | bc -l) )); then
  echo "[FATAL] EPSILON must be > 0" >&2
  exit 1
fi

manifest="$OUTDIR/job_manifest.tsv"
echo -e "step\tjobid\tstdout\tstderr" > "$manifest"

# submit STEP TIME MEM CPUS [sbatch args ...] -- script args ...
submit() {
  local step="$1"; shift
  local time="$1"; shift
  local mem="$1"; shift
  local cpus="$1"; shift

  local outpat="$OUTDIR/${step}_%A_%a.out"
  local errpat="$ERRDIR/${step}_%A_%a.err"

  # Collect extra sbatch args up to '--'
  local sbatch_extra=()
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
      shift
      break
    fi
    sbatch_extra+=("$1")
    shift
  done

  if [[ $# -lt 1 ]]; then
    echo "[FATAL] submit($step): missing script" >&2
    exit 1
  fi

  local script="$1"; shift

  echo "[SUBMIT] $step"
  echo "  script: $script $*"
  echo "  out:    $outpat"
  echo "  err:    $errpat"
  echo "  extra:  ${sbatch_extra[*]:-<none>}"

  # Print the full sbatch line (for debugging)
  echo "[CMD] sbatch --parsable --chdir=$PWRDATA --time=$time --mem=$mem --cpus-per-task=$cpus -N 1 --output=$outpat --error=$errpat ${sbatch_extra[*]:-} $script $*"

  local jid
  jid=$(sbatch --parsable \
    --chdir="$PWRDATA" \
    --time="$time" --mem="$mem" --cpus-per-task="$cpus" \
    -N 1 \
    --output="$outpat" --error="$errpat" \
    "${sbatch_extra[@]}" \
    "$script" "$@")

  echo "[JOB] $step -> $jid"

  local so se
  so=$(scontrol show job "${jid%%_*}" 2>/dev/null | awk -F= '/StdOut=/{print $2}' | awk '{print $1}' || true)
  se=$(scontrol show job "${jid%%_*}" 2>/dev/null | awk -F= '/StdErr=/{print $2}' | awk '{print $1}' || true)
  echo "[SCONTROL] StdOut=$so"
  echo "[SCONTROL] StdErr=$se"

  echo -e "${step}\t${jid}\t${so}\t${se}" >> "$manifest"
}

# Step 1
submit "pwr_index_file" "1:00:00" "4GB" "1" -- --wait \
  "$FILEDIR/pwr_index_file.sh" "$WRKDIR" "$FILEDIR"

if [ ! -s "$PWRDATA/pwr_index_file.txt" ]; then
  echo "[FATAL] pwr_index_file.txt missing/empty" >&2
  ls -lh "$PWRDATA" | head -n 80
  exit 1
fi

NINDEX=$(wc -l < "$PWRDATA/pwr_index_file.txt" | tr -d ' ')
CHUNK_SIZE=100
NJOBS=$(( (NINDEX + CHUNK_SIZE - 1) / CHUNK_SIZE ))
echo "[INFO] NINDEX=$NINDEX CHUNK_SIZE=$CHUNK_SIZE NJOBS=$NJOBS"

# Step 5
submit "pwr_setup" "1:00:00" "16GB" "2" -- --wait \
  "$FILEDIR/pwr_setup.sh" "$WRKDIR" "$FILEDIR"

# Step 6 (array)
submit "pwr_sub_python" "10:00:00" "16GB" "2" -- --array=1-"$NJOBS" --wait \
  "$FILEDIR/pwr_sub_python.sh" "$WRKDIR" "$CHUNK_SIZE" "$NINDEX" "$FILEDIR"

# Step 7
submit "combine_data" "1:00:00" "64GB" "4" -- --wait \
  "$FILEDIR/combine_data.sh" "$WRKDIR" "$FILEDIR"

N_FULL_COV=$(ls "$PWRDATA"/full_*_cov.npy 2>/dev/null | wc -l | tr -d ' ')
echo "[INFO] full_*_cov.npy count=$N_FULL_COV"
if [ "$N_FULL_COV" -le 0 ]; then
  echo "[FATAL] combine_data produced no full_*_cov.npy" >&2
  ls -lh "$PWRDATA" | head -n 80
  exit 1
fi

# Step 8
submit "ridge" "8:00:00" "64GB" "2" -- --wait \
  "$FILEDIR/ridge.sh" "$WRKDIR" "$FILEDIR"

NUMFILES=$(
  ls "$PWRDATA"/full_*_cov.npy 2>/dev/null \
  | sed -E 's/.*\/full_([0-9]+)_cov\.npy/\1/' \
  | sort -u | wc -l | tr -d ' '
)
echo "[INFO] NUMFILES=$NUMFILES"
if [ "$NUMFILES" -le 0 ]; then
  echo "[FATAL] NUMFILES=0" >&2
  exit 1
fi

# Step 10
submit "cvGen" "1:00:00" "16GB" "2" -- --array=1-"$NUMFILES" --wait \
  "$FILEDIR/cvGen.sh" "$WRKDIR" "$FILEDIR" "$NUMFILES" "$KFOLDS"

# Step 11
submit "setupCVmetrics" "1:00:00" "16GB" "2" -- --wait \
  "$FILEDIR/setupCVmetrics.sh" "$WRKDIR" "$FILEDIR"

# IMPORTANT: cvGen writes .npz splits (not .npy)
NUMFFILES=$(ls "$PWRDATA"/full_*_fold_*_split.npz 2>/dev/null | wc -l | tr -d ' ')
echo "[INFO] NUMFFILES=$NUMFFILES"
if [ "$NUMFFILES" -le 0 ]; then
  echo "[FATAL] NUMFFILES=0 (no full_*_fold_*_split.npz found)" >&2
  ls -lh "$PWRDATA" | head -n 120
  exit 1
fi

# Step 13 (array)
submit "cv" "2:00:00" "32GB" "2" -- --array=1-"$NUMFFILES" --wait \
  "$FILEDIR/cv.sh" "$WRKDIR" "$FILEDIR" "$NUMFILES" "$KFOLDS" "$EPSILON"

# Step 14
submit "final_data" "12:00:00" "96GB" "8" -- --wait \
  "$FILEDIR/final_data.sh" "$WRKDIR" "$FILEDIR"

echo "[DONE] manifest=$manifest"

