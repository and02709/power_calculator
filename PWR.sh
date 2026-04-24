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

# Read in command line arguments for the following parameters.
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
  --pca                        Enable PCA preprocessing (default: off)

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
# These represent default arguments if now arguments are provided by the command line.
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

# These arguments represent hyperparameters for machine learning methods for the cross validation module.
MODEL_FILE="${MODEL_FILE:-random_forest}"
USE_PCA="${USE_PCA:-false}"
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
# We use this to read in command line arguments using --argument usage
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
    --pca)          USE_PCA="true";        shift 1 ;;
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
# We use this to prevent execution of the script if these five arguments are missing from the command line.
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
# We create a new directory to contain all output logs in the working directory.
OUTDIR="$WRKDIR/OUT"
# We create a new directory to contain all error logs in the working directory.
ERRDIR="$WRKDIR/ERR"
# We create a directory to hold all the relevent information for our power calculation.
PWRDATA="$WRKDIR/pwr_data"

# ---------------------------------------------------------------------------
# Info dump
# ---------------------------------------------------------------------------
# We print back the arguments passed to our script for debugging purposes.
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
echo "[INFO] USE_PCA=$USE_PCA"
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
# submit() — Wrapper around sbatch for standardized job submission.
#
# Submits a SLURM job with consistent output/error path naming, optional
# extra sbatch flags, and logs job metadata (job ID, stdout/stderr paths)
# to a manifest file for tracking.
#
# Usage:
#   submit <step> <time> <mem> <cpus> [extra_sbatch_args...] [--] <script> [script_args...]
#
# Arguments:
#   step    - Logical pipeline step name (e.g. "pwr_setup", "combine_data").
#             Used to name output/error files and label manifest entries.
#   time    - SLURM walltime limit (e.g. "02:00:00").
#   mem     - Memory request (e.g. "16G").
#   cpus    - Number of CPUs per task (e.g. 4).
#   [extra] - Optional additional sbatch flags (e.g. --array=1-100 --dependency=...).
#             Consumed until a bare "--" separator or the script path is reached.
#   --      - Optional explicit separator between extra sbatch flags and the script.
#   script  - Path to the script to submit.
#   [args]  - Arguments forwarded to the script.
#
# Outputs:
#   - Stdout log: $OUTDIR/<step>_<jobID>_<arrayID>.out
#   - Stderr log: $ERRDIR/<step>_<jobID>_<arrayID>.err
#   - Appends a tab-separated line to $manifest:
#       <step>  <jobID>  <stdout_path>  <stderr_path>
#
# Globals read:
#   OUTDIR   - Directory for .out log files
#   ERRDIR   - Directory for .err log files
#   PWRDATA  - Working directory passed to --chdir
#   manifest - Path to the job manifest/tracking file
#
# Exits with code 1 if no script argument is provided.

submit() {
  local step="$1"; shift       # Pipeline step label (used in filenames + manifest)
  local time="$1"; shift       # SLURM walltime (--time)
  local mem="$1"; shift        # Memory request (--mem)
  local cpus="$1"; shift       # CPUs per task (--cpus-per-task)

  # Output/error filename patterns; %A = job ID, %a = array task ID
  local outpat="$OUTDIR/${step}_%A_%a.out"
  local errpat="$ERRDIR/${step}_%A_%a.err"

  # Collect any extra sbatch flags that appear before the script path.
  # A bare "--" can be used to explicitly end the extra-flags section.
  local sbatch_extra=()
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
      shift   # Discard the "--" separator and stop collecting extra flags
      break
    fi
    sbatch_extra+=("$1")
    shift
  done

  # Remaining positional args must start with the script path
  if [[ $# -lt 1 ]]; then
    echo "[FATAL] submit($step): missing script" >&2
    exit 1
  fi
  local script="$1"; shift   # Script to submit; remaining "$@" are its arguments

  # Log what is about to be submitted for easier debugging
  echo "[SUBMIT] $step"
  echo "  script: $script $*"
  echo "  out:    $outpat"
  echo "  err:    $errpat"
  echo "  extra:  ${sbatch_extra[*]:-<none>}"
  echo "[CMD] sbatch --parsable --chdir=$PWRDATA --time=$time --mem=$mem --cpus-per-task=$cpus -N 1 --output=$outpat --error=$errpat ${sbatch_extra[*]:-} $script $*"

  # Submit the job; --parsable returns only the job ID (or "jobid;cluster")
  local jid
  jid=$(sbatch --parsable \
    --chdir="$PWRDATA" \
    --time="$time" --mem="$mem" --cpus-per-task="$cpus" \
    -N 1 \
    --output="$outpat" --error="$errpat" \
    "${sbatch_extra[@]}" \
    "$script" "$@")

  echo "[JOB] $step -> $jid"

  # Query SLURM for the resolved stdout/stderr paths on the scheduler side.
  # "${jid%%_*}" strips any array suffix (e.g. "12345_1" -> "12345") so
  # scontrol can look up the parent job record.
  local so se
  so=$(scontrol show job "${jid%%_*}" 2>/dev/null | awk -F= '/StdOut=/{print $2}' | awk '{print $1}' || true)
  se=$(scontrol show job "${jid%%_*}" 2>/dev/null | awk -F= '/StdErr=/{print $2}' | awk '{print $1}' || true)

  echo "[SCONTROL] StdOut=$so"
  echo "[SCONTROL] StdErr=$se"

  # Append a manifest record: step, job ID, resolved stdout, resolved stderr
  echo -e "${step}\t${jid}\t${so}\t${se}" >> "$manifest"
}
# ---------------------------------------------------------------------------
# Step 1 — Setup
# ---------------------------------------------------------------------------
# Run pwr_setup.sh synchronously (--wait) before any downstream steps.
# This step generates pwr_index_file.txt, which defines the simulation
# index space (one row per sample-size/dataset-size combination). All
# subsequent array jobs depend on this file existing and being non-empty.
submit "pwr_setup" "1:00:00" "16GB" "2" -- --wait \
  "$FILEDIR/pwr_setup.sh" "$WRKDIR" "$FILEDIR"

# Guard: confirm pwr_index_file.txt was actually produced and is non-empty.
# -s tests that the file exists AND has size > 0. If it's missing or empty,
# something went wrong in pwr_setup and there is nothing to array over —
# abort early rather than silently submitting zero-work array jobs.
if [ ! -s "$PWRDATA/pwr_index_file.txt" ]; then
  echo "[FATAL] pwr_index_file.txt missing/empty" >&2
  ls -lh "$PWRDATA" | head -n 80   # Dump directory listing to aid diagnosis
  exit 1
fi

# ── Compute array dimensions ─────────────────────────────────────────────────
# NINDEX    = total number of simulation rows in the index file
# CHUNK_SIZE = number of index rows assigned to each array task
# NJOBS     = number of array tasks needed to cover all rows
#
# The ceiling-division formula (N + C - 1) / C ensures the last chunk is
# still submitted even when NINDEX is not a perfect multiple of CHUNK_SIZE.
# Example: 250 rows / 100 per chunk → 3 jobs (jobs 1-100, 101-200, 201-250).
NINDEX=$(wc -l < "$PWRDATA/pwr_index_file.txt" | tr -d ' ')
CHUNK_SIZE=100
NJOBS=$(( (NINDEX + CHUNK_SIZE - 1) / CHUNK_SIZE ))

echo "[INFO] NINDEX=$NINDEX CHUNK_SIZE=$CHUNK_SIZE NJOBS=$NJOBS"

# ---------------------------------------------------------------------------
# Step 2 — Python array jobs
# ---------------------------------------------------------------------------
# Distributes the simulation workload across NJOBS SLURM array tasks.
# Each task processes CHUNK_SIZE rows from pwr_index_file.txt, generating
# simulated pconn matrices and phenotype vectors for a range of sample sizes.
#
# Two modes are supported, selected by the SINGLETEMP flag:
#
#   SINGLETEMP=1  (single-template mode)
#     Each simulation draws one fresh random pconn template per index row.
#     Uses pwr_sub_python_single.sh → pwr_process_chunk_single_z.py.
#     NUMTEMP is not passed — template count is fixed at 1 internally.
#
#   SINGLETEMP=0  (multi-template mode)
#     Each simulation averages NUMTEMP randomly drawn pconn templates to
#     form a single representative connectivity matrix before simulating.
#     Uses pwr_sub_python.sh → pwr_process_chunk_z.py.
#
# Both modes run synchronously (--wait) so Step 3 only begins once all
# array tasks have completed or failed.
#
# Array indexing: tasks are 1-based (1..NJOBS). Each task derives its
# chunk boundaries from its SLURM_ARRAY_TASK_ID at runtime:
#   START = (TASK_ID - 1) * CHUNK_SIZE + 1
#   END   = min(TASK_ID * CHUNK_SIZE, NINDEX)

if [[ "$SINGLETEMP" == "1" ]]; then
  echo "Running in single-temp mode"
  submit "pwr_sub_python_single" "10:00:00" "16GB" "2" -- --array=1-"$NJOBS" --wait \
    "$FILEDIR/pwr_sub_python_single.sh" \
    "$WRKDIR" "$CHUNK_SIZE" "$NINDEX" "$FILEDIR" "$PCONNDIR" "$PCONNREF" "$NREP" "$NTIME"
    # Args:  working dir  rows/task   total rows  script dir  pconn pool  reference pconn  simulations/row  timepoints
else
  echo "Running in multi-temp mode"
  submit "pwr_sub_python" "10:00:00" "16GB" "2" -- --array=1-"$NJOBS" --wait \
    "$FILEDIR/pwr_sub_python.sh" \
    "$WRKDIR" "$CHUNK_SIZE" "$NINDEX" "$FILEDIR" "$PCONNDIR" "$PCONNREF" "$NUMTEMP" "$NREP" "$NTIME"
    # Args:  working dir  rows/task   total rows  script dir  pconn pool  reference pconn  templates/sim  simulations/row  timepoints
fi

# ---------------------------------------------------------------------------
# Step 3 — Combine data
# ---------------------------------------------------------------------------
# Aggregates the per-chunk .npy arrays produced by Step 2 into one
# full_<size>_cov.npy file per sample size. Also applies epsilon noise
# (controlled by EPSILON) to the covariance matrices at this stage.
# Runs synchronously (--wait) before the guards below execute.
#
# Memory is elevated to 64GB here because combine_data.py loads and
# concatenates all chunk outputs for each sample size into memory at once.
submit "combine_data" "1:00:00" "64GB" "4" -- --wait \
  "$FILEDIR/combine_data.sh" "$WRKDIR" "$FILEDIR" "$EPSILON"

# ── Guard 1: at least one output file was created ────────────────────────────
# Verifies that combine_data.sh produced at least one full_*_cov.npy.
# A count of zero means either all chunk outputs were missing/malformed
# or combine_data itself crashed before writing anything.
N_FULL_COV=$(ls "$PWRDATA"/full_*_cov.npy 2>/dev/null | wc -l | tr -d ' ')
echo "[INFO] full_*_cov.npy count=$N_FULL_COV"
if [ "$N_FULL_COV" -le 0 ]; then
  echo "[FATAL] combine_data produced no full_*_cov.npy" >&2
  ls -lh "$PWRDATA" | head -n 80   # Dump directory listing to aid diagnosis
  exit 1
fi

# ── Guard 2: count distinct sample sizes represented in the outputs ───────────
# Extracts the numeric size token from each full_<size>_cov.npy filename
# and counts how many unique sizes are present. This catches the case where
# combine_data ran and wrote files, but only produced output for a subset
# of the expected sample sizes (e.g. some sizes crashed mid-aggregation).
#
# sed strips everything except the size token:
#   full_100_cov.npy  →  100
#   full_2500_cov.npy →  2500
# sort -u deduplicates (guards against any accidental filename duplicates).
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
# Generates train/test fold indices for each sample size in parallel.
# One array task is launched per sample size (1..NUMFILES), where each
# task creates the CV split files needed by Step 6 (cv.py).
#
# Each task receives its target sample size implicitly via
# SLURM_ARRAY_TASK_ID, which cvGen.sh maps to the corresponding
# full_<size>_cov.npy produced in Step 3.
#
# KFOLDS controls how many folds are generated per sample size.
# Output files are expected to follow the pattern:
#   $PWRDATA/cv_<size>_fold<k>.npz   (train/test index arrays)
#
# Runs synchronously (--wait) so Step 6 only begins once fold
# indices exist for all sample sizes.
#
# Note: labeled Step 5 rather than Step 4 because ridge model
# generation (Step 4) runs between combine_data and CV generation.
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
  --export=ALL,MODEL_FILE="$MODEL_FILE",USE_PCA="$USE_PCA",N_COMPONENTS="$N_COMPONENTS",N_ESTIMATORS="$N_ESTIMATORS",RIDGE_ALPHA="$RIDGE_ALPHA",LASSO_ALPHA="$LASSO_ALPHA",EN_ALPHA="$EN_ALPHA",EN_L1_RATIO="$EN_L1_RATIO",SVR_C="$SVR_C",NN_HIDDEN_LAYERS="$NN_HIDDEN_LAYERS",NN_LR="$NN_LR",GB_N_ESTIMATORS="$GB_N_ESTIMATORS",GB_LR="$GB_LR" \
  -- \
  "$FILEDIR/cv.sh" "$WRKDIR" "$FILEDIR" "$NUMFILES" "$KFOLDS" "$EPSILON"

# ---------------------------------------------------------------------------
# Step 8 — Final data
# ---------------------------------------------------------------------------
submit "final_data" "12:00:00" "96GB" "8" -- --wait \
  "$FILEDIR/final_data.sh" "$WRKDIR" "$FILEDIR"

echo "[DONE] manifest=$manifest"
