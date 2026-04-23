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

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
  cat <<EOF
Usage: sbatch PWR.sh [OPTIONS]

Required:
  --pconnref     PCONNREF      Pconn reference
  --singletemp   0|1           Single-temperature mode (0=multi, 1=single)
  --numtemp      INT           Number of temperatures (>= 1)
  --kfolds       INT           Number of CV folds
  --epsilon      FLOAT         Epsilon value (>= 0)

Optional with defaults:
  --wrkdir       WRKDIR        Working directory (default: PWD)
  --pconndir     PCONNDIR      Pconn directory (default: PWD)
  --filedir      FILEDIR       Script/file directory (default: PWD)
  --nrep         INT           Number of repetitions (default: 10)
  --ntime        INT           Number of timepoints (default: 1000)

Optional model selection:
  --model        MODEL_FILE    Model to use (default: random_forest)
                               Options: random_forest, ridge, lasso,
                                        elastic_net, svr, neural_net, gradient_boosting
  --n-components INT           PCA components (default: 500)
  --n-estimators INT           RF estimators (default: 500)

Optional model hyperparameters:
  --ridge-alpha   FLOAT        Ridge alpha (default: 1.0)
  --lasso-alpha   FLOAT        Lasso alpha (default: 0.01)
  --en-alpha      FLOAT        ElasticNet alpha (default: 0.01)
  --en-l1-ratio   FLOAT        ElasticNet L1 ratio (default: 0.5)
  --svr-c         FLOAT        SVR C parameter (default: 1.0)
  --nn-hidden     LAYERS       NN hidden layers e.g. 256,128 (default: 256,128)
  --nn-lr         FLOAT        NN learning rate (default: 0.001)
  --gb-estimators INT          GB n_estimators (default: 300)
  --gb-lr         FLOAT        GB learning rate (default: 0.05)

  -h, --help                   Show this help message

Example:
  sbatch PWR.sh --wrkdir /path/to/work --pconndir /path/to/pconn \\
                --pconnref myref --singletemp 0 --numtemp 5 \\
                --filedir /path/to/scripts --kfolds 5 --nrep 10 \\
                --ntime 500 --epsilon 0.1 --model ridge --ridge-alpha 0.5
EOF
  exit 1
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
WRKDIR="$(pwd)"
PCONNDIR="$(pwd)"
PCONNREF=""
SINGLETEMP=""
NUMTEMP=""
FILEDIR="$(pwd)"
KFOLDS=""
NREP=10
NTIME=1000
EPSILON=""

MODEL_FILE="${MODEL_FILE:-random_forest}"
N_COMPONENTS="${N_COMPONENTS:-500}"
N_ESTIMATORS="${N_ESTIMATORS:-500}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
LASSO_ALPHA="${LASSO_ALPHA:-0.01}"
EN_ALPHA="${EN_ALPHA:-0.01}"
EN_L1_RATIO="${EN_L1_RATIO:-0.5}"
SVR_C="${SVR_C:-1.0}"
NN_HIDDEN_LAYERS="${NN_HIDDEN_LAYERS:-256,128}"
NN_LR="${NN_LR:-0.001}"
GB_N_ESTIMATORS="${GB_N_ESTIMATORS:-300}"
GB_LR="${GB_LR:-0.05}"

# ---------------------------------------------------------------------------
# Parse named arguments
# ---------------------------------------------------------------------------
if [[ $# -eq 0 ]]; then
  usage
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wrkdir)       WRKDIR="$2";           shift 2 ;;
    --pconndir)     PCONNDIR="$2";         shift 2 ;;
    --pconnref)     PCONNREF="$2";         shift 2 ;;
    --singletemp)   SINGLETEMP="$2";       shift 2 ;;
    --numtemp)      NUMTEMP="$2";          shift 2 ;;
    --filedir)      FILEDIR="$2";          shift 2 ;;
    --kfolds)       KFOLDS="$2";           shift 2 ;;
    --nrep)         NREP="$2";             shift 2 ;;
    --ntime)        NTIME="$2";            shift 2 ;;
    --epsilon)      EPSILON="$2";          shift 2 ;;
    --model)        MODEL_FILE="$2";       shift 2 ;;
    --n-components) N_COMPONENTS="$2";     shift 2 ;;
    --n-estimators) N_ESTIMATORS="$2";     shift 2 ;;
    --ridge-alpha)  RIDGE_ALPHA="$2";      shift 2 ;;
    --lasso-alpha)  LASSO_ALPHA="$2";      shift 2 ;;
    --en-alpha)     EN_ALPHA="$2";         shift 2 ;;
    --en-l1-ratio)  EN_L1_RATIO="$2";      shift 2 ;;
    --svr-c)        SVR_C="$2";            shift 2 ;;
    --nn-hidden)    NN_HIDDEN_LAYERS="$2"; shift 2 ;;
    --nn-lr)        NN_LR="$2";            shift 2 ;;
    --gb-estimators) GB_N_ESTIMATORS="$2"; shift 2 ;;
    --gb-lr)        GB_LR="$2";            shift 2 ;;
    -h|--help)      usage ;;
    *)
      echo "[FATAL] Unknown argument: $1" >&2
      usage
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Validate required arguments
# ---------------------------------------------------------------------------
missing=()
[[ -z "$PCONNREF"   ]] && missing+=(--pconnref)
[[ -z "$SINGLETEMP" ]] && missing+=(--singletemp)
[[ -z "$NUMTEMP"    ]] && missing+=(--numtemp)
[[ -z "$KFOLDS"     ]] && missing+=(--kfolds)
[[ -z "$EPSILON"    ]] && missing+=(--epsilon)

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "[FATAL] Missing required arguments: ${missing[*]}" >&2
  usage
fi

if [[ "$SINGLETEMP" != "0" && "$SINGLETEMP" != "1" ]]; then
  echo "[FATAL] --singletemp must be 0 or 1" >&2
  exit 1
fi

if ! [[ "$NUMTEMP" =~ ^[0-9]+$ ]] || (( NUMTEMP < 1 )); then
  echo "[FATAL] --numtemp must be an integer >= 1 (got: '$NUMTEMP')" >&2
  exit 1
fi

if ! [[ "$EPSILON" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[FATAL] --epsilon must be a non-negative number (got: '$EPSILON')" >&2
  exit 1
fi
if (( $(echo "$EPSILON < 0" | bc -l) )); then
  echo "[FATAL] --epsilon must be >= 0" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------------
OUTDIR="$WRKDIR/OUT"
ERRDIR="$WRKDIR/ERR"
PWRDATA="$WRKDIR/pwr_data"

# ---------------------------------------------------------------------------
# Info dump
# ---------------------------------------------------------------------------
echo "=== PWR Pipeline ==="
echo "[INFO] WRKDIR=$WRKDIR"
echo "[INFO] PCONNDIR=$PCONNDIR"
echo "[INFO] PCONNREF=$PCONNREF"
echo "[INFO] SINGLETEMP=$SINGLETEMP"
echo "[INFO] NUMTEMP=$NUMTEMP"
echo "[INFO] FILEDIR=$FILEDIR"
echo "[INFO] KFOLDS=$KFOLDS"
echo "[INFO] NREP=$NREP"
echo "[INFO] NTIME=$NTIME"
echo "[INFO] EPSILON=$EPSILON"
echo "[INFO] MODEL_FILE=$MODEL_FILE"
echo "[INFO] N_COMPONENTS=$N_COMPONENTS"
echo "[INFO] N_ESTIMATORS=$N_ESTIMATORS"
echo "[INFO] RIDGE_ALPHA=$RIDGE_ALPHA"
echo "[INFO] LASSO_ALPHA=$LASSO_ALPHA"
echo "[INFO] EN_ALPHA=$EN_ALPHA"
echo "[INFO] EN_L1_RATIO=$EN_L1_RATIO"
echo "[INFO] SVR_C=$SVR_C"
echo "[INFO] NN_HIDDEN_LAYERS=$NN_HIDDEN_LAYERS"
echo "[INFO] NN_LR=$NN_LR"
echo "[INFO] GB_N_ESTIMATORS=$GB_N_ESTIMATORS"
echo "[INFO] GB_LR=$GB_LR"
echo "[INFO] PWRDATA=$PWRDATA"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-<none>}"
echo "[INFO] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<none>}"
echo "===================="

mkdir -p "$OUTDIR" "$ERRDIR" "$PWRDATA"
cd "$PWRDATA"

manifest="$OUTDIR/job_manifest.tsv"
echo -e "step\tjobid\tstdout\tstderr" > "$manifest"

# ---------------------------------------------------------------------------
# submit STEP TIME MEM CPUS [sbatch args ...] -- script args ...
# ---------------------------------------------------------------------------
submit() {
  local step="$1"; shift
  local time="$1"; shift
  local mem="$1"; shift
  local cpus="$1"; shift

  local outpat="$OUTDIR/${step}_%A_%a.out"
  local errpat="$ERRDIR/${step}_%A_%a.err"

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

# ---------------------------------------------------------------------------
# Step 1 — Setup
# ---------------------------------------------------------------------------
submit "pwr_setup" "1:00:00" "16GB" "2" -- --wait \
  "$FILEDIR/pwr_setup.sh" "$WRKDIR" "$FILEDIR"

if [ ! -s "$PWRDATA/pwr_index_file.txt" ]; then
  echo "[FATAL] pwr_index_file.txt missing/empty" >&2
  ls -lh "$PWRDATA" | head -n 80
  exit 1
fi

NINDEX=$(wc -l < "$PWRDATA/pwr_index_file.txt" | tr -d ' ')
CHUNK_SIZE=100
NJOBS=$(( (NINDEX + CHUNK_SIZE - 1) / CHUNK_SIZE ))
echo "[INFO] NINDEX=$NINDEX CHUNK_SIZE=$CHUNK_SIZE NJOBS=$NJOBS"

# ---------------------------------------------------------------------------
# Step 2 — Python array jobs
# ---------------------------------------------------------------------------
if [[ "$SINGLETEMP" == "1" ]]; then
  echo "Running in single-temp mode"
  submit "pwr_sub_python_single" "10:00:00" "16GB" "2" -- --array=1-"$NJOBS" --wait \
    "$FILEDIR/pwr_sub_python_single.sh" \
    "$WRKDIR" "$CHUNK_SIZE" "$NINDEX" "$FILEDIR" "$PCONNDIR" "$PCONNREF" "$NREP"
else
  echo "Running in multi-temp mode"
  submit "pwr_sub_python" "10:00:00" "16GB" "2" -- --array=1-"$NJOBS" --wait \
    "$FILEDIR/pwr_sub_python.sh" \
    "$WRKDIR" "$CHUNK_SIZE" "$NINDEX" "$FILEDIR" "$PCONNDIR" "$PCONNREF" "$NUMTEMP" "$NREP"
fi

# ---------------------------------------------------------------------------
# Step 3 — Combine data
# ---------------------------------------------------------------------------
submit "combine_data" "1:00:00" "64GB" "4" -- --wait \
  "$FILEDIR/combine_data.sh" "$WRKDIR" "$FILEDIR" "$EPSILON"

N_FULL_COV=$(ls "$PWRDATA"/full_*_cov.npy 2>/dev/null | wc -l | tr -d ' ')
echo "[INFO] full_*_cov.npy count=$N_FULL_COV"
if [ "$N_FULL_COV" -le 0 ]; then
  echo "[FATAL] combine_data produced no full_*_cov.npy" >&2
  ls -lh "$PWRDATA" | head -n 80
  exit 1
fi

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

# ---------------------------------------------------------------------------
# Step 5 — CV generation
# ---------------------------------------------------------------------------
submit "cvGen" "1:00:00" "16GB" "2" -- --array=1-"$NUMFILES" --wait \
  "$FILEDIR/cvGen.sh" "$WRKDIR" "$FILEDIR" "$NUMFILES" "$KFOLDS"

# ---------------------------------------------------------------------------
# Step 6 — Setup CV metrics
# ---------------------------------------------------------------------------
submit "setupCVmetrics" "1:00:00" "16GB" "2" -- --wait \
  "$FILEDIR/setupCVmetrics.sh" "$WRKDIR" "$FILEDIR"

NUMFFILES=$(ls "$PWRDATA"/full_*_fold_*_split.npz 2>/dev/null | wc -l | tr -d ' ')
echo "[INFO] NUMFFILES=$NUMFFILES"
if [ "$NUMFFILES" -le 0 ]; then
  echo "[FATAL] NUMFFILES=0 (no full_*_fold_*_split.npz found)" >&2
  ls -lh "$PWRDATA" | head -n 120
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 7 — Cross-validation (model array jobs)
# ---------------------------------------------------------------------------
submit "cv" "2:00:00" "32GB" "2" -- \
  --array=1-"$NUMFFILES" --wait \
  --export=ALL,MODEL_FILE="$MODEL_FILE",N_COMPONENTS="$N_COMPONENTS",N_ESTIMATORS="$N_ESTIMATORS",RIDGE_ALPHA="$RIDGE_ALPHA",LASSO_ALPHA="$LASSO_ALPHA",EN_ALPHA="$EN_ALPHA",EN_L1_RATIO="$EN_L1_RATIO",SVR_C="$SVR_C",NN_HIDDEN_LAYERS="$NN_HIDDEN_LAYERS",NN_LR="$NN_LR",GB_N_ESTIMATORS="$GB_N_ESTIMATORS",GB_LR="$GB_LR" \
  -- \
  "$FILEDIR/cv.sh" "$WRKDIR" "$FILEDIR" "$NUMFILES" "$KFOLDS" "$EPSILON"

# ---------------------------------------------------------------------------
# Step 8 — Final data
# ---------------------------------------------------------------------------
submit "final_data" "12:00:00" "96GB" "8" -- --wait \
  "$FILEDIR/final_data.sh" "$WRKDIR" "$FILEDIR"

echo "[DONE] manifest=$manifest"
