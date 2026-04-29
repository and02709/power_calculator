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
  --epsilon      FLOAT         Epsilon value (>= 0)
  --condaenv     ENV_NAME      Conda environment to use

Optional with defaults:
  --wrkdir       WRKDIR        Working directory (default: PWD)
  --pconndir     PCONNDIR      Pconn directory (default: PWD)
  --filedir      FILEDIR       Script/file directory (default: PWD)
  --nrep         INT           Number of repetitions (default: 10)
  --ntime        INT           Number of timepoints (default: 1000)

Optional CV topology (sklearn RepeatedKFold):
  --k-outer      INT           Outer CV folds (default: 10)
  --n-outer      INT           Outer CV repeats (default: 2; total = k*n)
  --random-state INT           RNG seed for fold generation (default: 123456)
  --n-jobs       INT           Parallel jobs for cross_validate (default: 1)

Optional model selection:
  --model        MODEL_FILE    Model to use (default: ridge)
                               Options: ridge, ridge_nested, lasso, elastic_net,
                                        svr, neural_network, random_forest,
                                        gradient_boosting
  --pca                        Enable PCA preprocessing (default: off)
  --n-components INT           PCA components (default: 500)

Optional model hyperparameters (passed through; ignored if not relevant):
  --ridge-alphas  STR          Comma-sep alpha candidates (default: '1,10,...,1e5')
  --lasso-n-alphas INT         LassoCV alpha grid size (default: 100)
  --en-l1-ratios  STR          Comma-sep l1_ratio candidates (default: '0.1,...,1.0')
  --svr-c-vals    STR          Comma-sep SVR C candidates (default: '0.1,1,10,100')
  --nn-hidden     LAYERS       NN hidden layers e.g. 256,128 (default: 256,128)
  --nn-lr         FLOAT        NN learning rate (default: 0.001)
  --gb-estimators INT          GB n_estimators (default: 300)
  --gb-lr         FLOAT        GB learning rate (default: 0.05)
  --rf-tune                    Enable nested GridSearchCV for RF (default: off)
  --gb-tune                    Enable nested GridSearchCV for GB (default: off)

  -h, --help                   Show this help message

Example:
  sbatch PWR.sh --wrkdir /path/to/work --pconndir /path/to/pconn \\
                --pconnref myref --singletemp 0 --numtemp 5 \\
                --filedir /path/to/scripts --nrep 10 \\
                --ntime 500 --epsilon 0.1 --model ridge \\
                --k-outer 10 --n-outer 2 --ridge-alphas '1,10,100,1000'
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
NREP=10
NTIME=1000
EPSILON=""
CONDAENV=""

# ── CV topology (sklearn RepeatedKFold) ───────────────────────────────────────
K_OUTER="${K_OUTER:-10}"
N_OUTER="${N_OUTER:-2}"
RANDOM_STATE="${RANDOM_STATE:-123456}"
N_JOBS="${N_JOBS:-1}"

# ── Model selection ───────────────────────────────────────────────────────────
MODEL_FILE="${MODEL_FILE:-ridge}"
USE_PCA="${USE_PCA:-false}"
N_COMPONENTS="${N_COMPONENTS:-500}"

# ── Model hyperparameters ─────────────────────────────────────────────────────
RIDGE_ALPHAS="${RIDGE_ALPHAS:-1,10,100,1000,10000,100000}"
RIDGE_CV_FOLDS="${RIDGE_CV_FOLDS:-5}"
RIDGE_K_INNER="${RIDGE_K_INNER:-5}"
LASSO_N_ALPHAS="${LASSO_N_ALPHAS:-100}"
LASSO_CV_FOLDS="${LASSO_CV_FOLDS:-5}"
LASSO_MAX_ITER="${LASSO_MAX_ITER:-5000}"
EN_L1_RATIOS="${EN_L1_RATIOS:-0.1,0.5,0.7,0.9,0.95,1.0}"
EN_N_ALPHAS="${EN_N_ALPHAS:-100}"
EN_CV_FOLDS="${EN_CV_FOLDS:-5}"
RF_N_ESTIMATORS="${RF_N_ESTIMATORS:-500}"
RF_MAX_FEATURES="${RF_MAX_FEATURES:-1.0}"
RF_TUNE="${RF_TUNE:-false}"
RF_K_INNER="${RF_K_INNER:-3}"
SVR_C_VALS="${SVR_C_VALS:-0.1,1,10,100}"
SVR_KERNEL="${SVR_KERNEL:-rbf}"
SVR_EPSILON="${SVR_EPSILON:-0.1}"
SVR_K_INNER="${SVR_K_INNER:-5}"
NN_HIDDEN_LAYERS="${NN_HIDDEN_LAYERS:-256,128}"
NN_LR="${NN_LR:-0.001}"
NN_MAX_ITER="${NN_MAX_ITER:-500}"
NN_ALPHA="${NN_ALPHA:-0.0001}"
GB_N_ESTIMATORS="${GB_N_ESTIMATORS:-300}"
GB_LR="${GB_LR:-0.05}"
GB_MAX_DEPTH="${GB_MAX_DEPTH:-4}"
GB_TUNE="${GB_TUNE:-false}"
GB_K_INNER="${GB_K_INNER:-3}"

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
    --nrep)           NREP="$2";             shift 2 ;;
    --ntime)          NTIME="$2";            shift 2 ;;
    --epsilon)        EPSILON="$2";          shift 2 ;;
    --condaenv)       CONDAENV="$2";         shift 2 ;;
    # CV topology
    --k-outer)        K_OUTER="$2";          shift 2 ;;
    --n-outer)        N_OUTER="$2";          shift 2 ;;
    --random-state)   RANDOM_STATE="$2";     shift 2 ;;
    --n-jobs)         N_JOBS="$2";           shift 2 ;;
    # Model
    --model)          MODEL_FILE="$2";       shift 2 ;;
    --pca)            USE_PCA="true";        shift 1 ;;
    --n-components)   N_COMPONENTS="$2";     shift 2 ;;
    # Hyperparameters
    --ridge-alphas)   RIDGE_ALPHAS="$2";     shift 2 ;;
    --ridge-cv-folds) RIDGE_CV_FOLDS="$2";   shift 2 ;;
    --ridge-k-inner)  RIDGE_K_INNER="$2";    shift 2 ;;
    --lasso-n-alphas) LASSO_N_ALPHAS="$2";   shift 2 ;;
    --lasso-cv-folds) LASSO_CV_FOLDS="$2";   shift 2 ;;
    --lasso-max-iter) LASSO_MAX_ITER="$2";   shift 2 ;;
    --en-l1-ratios)   EN_L1_RATIOS="$2";     shift 2 ;;
    --en-n-alphas)    EN_N_ALPHAS="$2";      shift 2 ;;
    --en-cv-folds)    EN_CV_FOLDS="$2";      shift 2 ;;
    --rf-n-estimators) RF_N_ESTIMATORS="$2"; shift 2 ;;
    --rf-max-features) RF_MAX_FEATURES="$2"; shift 2 ;;
    --rf-tune)        RF_TUNE="true";        shift 1 ;;
    --rf-k-inner)     RF_K_INNER="$2";       shift 2 ;;
    --svr-c-vals)     SVR_C_VALS="$2";       shift 2 ;;
    --svr-kernel)     SVR_KERNEL="$2";       shift 2 ;;
    --svr-epsilon)    SVR_EPSILON="$2";      shift 2 ;;
    --svr-k-inner)    SVR_K_INNER="$2";      shift 2 ;;
    --nn-hidden)      NN_HIDDEN_LAYERS="$2"; shift 2 ;;
    --nn-lr)          NN_LR="$2";            shift 2 ;;
    --nn-max-iter)    NN_MAX_ITER="$2";      shift 2 ;;
    --nn-alpha)       NN_ALPHA="$2";         shift 2 ;;
    --gb-estimators)  GB_N_ESTIMATORS="$2";  shift 2 ;;
    --gb-lr)          GB_LR="$2";            shift 2 ;;
    --gb-max-depth)   GB_MAX_DEPTH="$2";     shift 2 ;;
    --gb-tune)        GB_TUNE="true";        shift 1 ;;
    --gb-k-inner)     GB_K_INNER="$2";       shift 2 ;;
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
[[ -z "$EPSILON"    ]] && missing+=(--epsilon)
#not adding condaenv check here as we have a specific error message for it below


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
if [[ -z "$CONDAENV" ]]; then
  echo "[FATAL] --condaenv must be specified. Either you've forgotten to add this or your conda environment doesnt exist." >&2
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
echo "[INFO] CONDAENV=$CONDAENV"
echo "[INFO] WRKDIR=$WRKDIR"
echo "[INFO] PCONNDIR=$PCONNDIR"
echo "[INFO] PCONNREF=$PCONNREF"
echo "[INFO] SINGLETEMP=$SINGLETEMP"
echo "[INFO] NUMTEMP=$NUMTEMP"
echo "[INFO] FILEDIR=$FILEDIR"
echo "[INFO] NREP=$NREP"
echo "[INFO] NTIME=$NTIME"
echo "[INFO] EPSILON=$EPSILON"
echo "[INFO] K_OUTER=$K_OUTER  N_OUTER=$N_OUTER  RANDOM_STATE=$RANDOM_STATE  N_JOBS=$N_JOBS"
echo "[INFO] MODEL_FILE=$MODEL_FILE"
echo "[INFO] USE_PCA=$USE_PCA"
echo "[INFO] N_COMPONENTS=$N_COMPONENTS"
echo "[INFO] RIDGE_ALPHAS=$RIDGE_ALPHAS"
echo "[INFO] RIDGE_CV_FOLDS=$RIDGE_CV_FOLDS  RIDGE_K_INNER=$RIDGE_K_INNER"
echo "[INFO] LASSO_N_ALPHAS=$LASSO_N_ALPHAS  LASSO_CV_FOLDS=$LASSO_CV_FOLDS  LASSO_MAX_ITER=$LASSO_MAX_ITER"
echo "[INFO] EN_L1_RATIOS=$EN_L1_RATIOS  EN_N_ALPHAS=$EN_N_ALPHAS  EN_CV_FOLDS=$EN_CV_FOLDS"
echo "[INFO] RF_N_ESTIMATORS=$RF_N_ESTIMATORS  RF_MAX_FEATURES=$RF_MAX_FEATURES  RF_TUNE=$RF_TUNE  RF_K_INNER=$RF_K_INNER"
echo "[INFO] SVR_C_VALS=$SVR_C_VALS  SVR_KERNEL=$SVR_KERNEL  SVR_EPSILON=$SVR_EPSILON  SVR_K_INNER=$SVR_K_INNER"
echo "[INFO] NN_HIDDEN_LAYERS=$NN_HIDDEN_LAYERS  NN_LR=$NN_LR  NN_MAX_ITER=$NN_MAX_ITER  NN_ALPHA=$NN_ALPHA"
echo "[INFO] GB_N_ESTIMATORS=$GB_N_ESTIMATORS  GB_LR=$GB_LR  GB_MAX_DEPTH=$GB_MAX_DEPTH  GB_TUNE=$GB_TUNE  GB_K_INNER=$GB_K_INNER"
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
  "$FILEDIR/pwr_setup.sh" "$WRKDIR" "$FILEDIR" "$CONDAENV"

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
#     Each simulated individual is based on the same pconn provided by --pconnref.
#     Uses pwr_sub_python_single.sh → pwr_process_chunk_single_z.py.
#     NUMTEMP is not passed — template count is fixed at 1 internally.
#
#   SINGLETEMP=0  (multi-template mode)
#     Each simulation averages NUMTEMP randomly drawn pconn templates from --pconndir
#     to form form a single representative connectivity matrix before simulating.
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
    "$WRKDIR" "$CHUNK_SIZE" "$NINDEX" "$FILEDIR" "$PCONNDIR" "$PCONNREF" "$NREP" "$NTIME" "$CONDAENV"
    # Args:  working dir  rows/task   total rows  script dir  pconn pool  reference pconn  simulations/row  timepoints
else
  echo "Running in multi-temp mode"
  submit "pwr_sub_python" "10:00:00" "16GB" "2" -- --array=1-"$NJOBS" --wait \
    "$FILEDIR/pwr_sub_python.sh" \
    "$WRKDIR" "$CHUNK_SIZE" "$NINDEX" "$FILEDIR" "$PCONNDIR" "$PCONNREF" "$NUMTEMP" "$NREP" "$NTIME" "$CONDAENV"
    # Args:  working dir  rows/task   total rows  script dir  pconn pool  reference pconn  templates/sim  simulations/row  timepoints
fi

# ---------------------------------------------------------------------------
# Step 3 — Combine data
# ---------------------------------------------------------------------------
# Aggregates the per-chunk .npy arrays produced by Step 2 into one
# full_<size>_cov.npy file per sample size. Also applies epsilon noise
# (controlled by EPSILON) to the calculated phenotypes formed by the dot product
# between full imaging data and the phenotype mapping given by haufe.csv.
# Runs synchronously (--wait) before the guards below execute.
#
# Memory is elevated to 64GB here because combine_data.py loads and
# concatenates all chunk outputs for each sample size into memory at once.
submit "combine_data" "1:00:00" "64GB" "4" -- --wait \
  "$FILEDIR/combine_data.sh" "$WRKDIR" "$FILEDIR" "$EPSILON" "$CONDAENV"

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
# Step 5 — Cross-validation (sklearn cross_validate; one task per sample size)
# ---------------------------------------------------------------------------
# Fits and evaluates the chosen predictive model across all outer CV folds
# for each sample size in parallel.  One array task per sample size
# (1..NUMFILES); all folds run inside a single cv.py call via sklearn's
# cross_validate(RepeatedKFold(K_OUTER, N_OUTER)).
#
# Replaces the legacy Steps 5+6+7 (cvGen → setupCVmetrics → cv array).
# No pre-split .npz files are written; splits are generated in-memory from
# FCs_<size>.npy and y_<size>.npy produced by Step 3 (combine_data).
#
# Outputs per (size, model):
#   cv_results_size<N>_<model>.csv   — per-fold train+test scores
#   cv_summary_size<N>_<model>.csv   — mean ± SD across all folds
#   cv_stamp_size<N>_<model>.txt     — provenance stamp
#
# Runs synchronously (--wait) so Step 6 (final_data) begins only after
# all sample-size CV jobs are complete.
submit "cv" "24:00:00" "128GB" "20" -- \
  --array=1-"$NUMFILES" --wait \
  --export=ALL,MODEL_FILE="$MODEL_FILE",USE_PCA="$USE_PCA",N_COMPONENTS="$N_COMPONENTS",K_OUTER="$K_OUTER",N_OUTER="$N_OUTER",RANDOM_STATE="$RANDOM_STATE",N_JOBS="$N_JOBS",RIDGE_ALPHAS="$RIDGE_ALPHAS",RIDGE_CV_FOLDS="$RIDGE_CV_FOLDS",RIDGE_K_INNER="$RIDGE_K_INNER",LASSO_N_ALPHAS="$LASSO_N_ALPHAS",LASSO_CV_FOLDS="$LASSO_CV_FOLDS",LASSO_MAX_ITER="$LASSO_MAX_ITER",EN_L1_RATIOS="$EN_L1_RATIOS",EN_N_ALPHAS="$EN_N_ALPHAS",EN_CV_FOLDS="$EN_CV_FOLDS",RF_N_ESTIMATORS="$RF_N_ESTIMATORS",RF_MAX_FEATURES="$RF_MAX_FEATURES",RF_TUNE="$RF_TUNE",RF_K_INNER="$RF_K_INNER",SVR_C_VALS="$SVR_C_VALS",SVR_KERNEL="$SVR_KERNEL",SVR_EPSILON="$SVR_EPSILON",SVR_K_INNER="$SVR_K_INNER",NN_HIDDEN_LAYERS="$NN_HIDDEN_LAYERS",NN_LR="$NN_LR",NN_MAX_ITER="$NN_MAX_ITER",NN_ALPHA="$NN_ALPHA",GB_N_ESTIMATORS="$GB_N_ESTIMATORS",GB_LR="$GB_LR",GB_MAX_DEPTH="$GB_MAX_DEPTH",GB_TUNE="$GB_TUNE",GB_K_INNER="$GB_K_INNER" \
  -- \
  "$FILEDIR/cv.sh" "$WRKDIR" "$FILEDIR" "$NUMFILES" "$CONDAENV"

# ---------------------------------------------------------------------------
# Step 8 — Final data
# ---------------------------------------------------------------------------
# Collects all per-fold CV results produced in Step 5 and aggregates them
# into the final power curve outputs (e.g. mean/SD of prediction accuracy
# across folds, per sample size). This is the terminal compute step of the
# pipeline — its outputs are what the power calculator actually reports.
#
# Resource profile is the heaviest in the pipeline (12h / 96GB / 8 CPUs):
#   - 12h walltime: aggregation across all sample sizes and folds can be
#     slow if results are large or post-processing is extensive.
#   - 96GB memory:  all per-fold result arrays are loaded simultaneously
#     before reduction, which scales with NUMFILES × (K_OUTER × N_OUTER) × result size.
#   - 8 CPUs:       final_data.py can parallelise aggregation across sample
#     sizes using multiprocessing.
#
# Runs as a single (non-array) job synchronously (--wait). No guard follows
# because the manifest echo below serves as the implicit success signal —
# if final_data.sh crashes, the submit() call will propagate a non-zero
# exit and the pipeline will abort before reaching it.
submit "final_data" "12:00:00" "96GB" "8" -- --wait \
  "$FILEDIR/final_data.sh" "$WRKDIR" "$FILEDIR" "$CONDAENV"

# Pipeline complete — print the manifest path so the caller knows where
# to find the full record of submitted job IDs and their log file paths.
echo "[DONE] manifest=$manifest"
