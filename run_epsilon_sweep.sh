#!/bin/bash
# run_epsilon_sweep.sh — launch a power calculation across a range of epsilons.
#
# Run this from a login node (it calls sbatch; it is not itself an sbatch job).
#
# What it does
# ------------
# Epsilon only affects the phenotype noise applied in Step 3, so the simulated
# brain data is identical for every value in the sweep. This launcher exploits
# that:
#
#   1. One base run does Steps 1-3 (setup, simulate, combine) and stops. It
#      produces full_<size>_{cov,cor,y}.npy — the expensive part, done once.
#   2. One run per epsilon starts at the noise step, symlinks the base run's
#      matrices via --reuse-from, rewrites full_<size>_yt.npy for its own
#      epsilon, then runs CV and aggregation.
#
# Every epsilon is therefore evaluated on the same simulated subjects, so the
# spread between power curves reflects phenotype SNR and nothing else.
#
# The per-epsilon runs are submitted with --dependency=afterok on the base run,
# so they queue immediately and start as soon as the base run finishes.
#
# Usage:
#   ./run_epsilon_sweep.sh --sweepdir DIR --pconnref FILE --condaenv ENV [options]
#
# Required:
#   --sweepdir   DIR    Parent directory; base/ and eps_<value>/ are created here
#   --pconnref   FILE   Reference .pconn.nii (not needed with --base-ready)
#   --condaenv   ENV    Conda environment name
#
# Optional:
#   --pconndir   DIR    Pconn pool directory            (default: PWD)
#   --filedir    DIR    Pipeline scripts directory      (default: this script's dir)
#   --eps-min    FLOAT  First epsilon                   (default: 1)
#   --eps-max    FLOAT  Last epsilon                    (default: 10)
#   --eps-step   FLOAT  Epsilon increment               (default: 1)
#   --singletemp 0|1    Template mode                   (default: 0)
#   --numtemp    INT    Templates per simulation        (default: 1)
#   --nrep       INT    Simulations per subject         (default: 10)
#   --ntime      INT    Timepoints per simulation       (default: 1000)
#   --model      NAME   Model for the CV step           (default: ridge)
#   --noise-seed INT    Base RNG seed; run N uses SEED+N (default: 123456)
#   --base-ready DIR    Skip the base run and reuse this finished run instead
#   --extra      "STR"  Extra flags forwarded verbatim to every per-epsilon run
#                       (e.g. --extra "--pca --n-components 300")
#   --dry-run           Print the sbatch commands without submitting
#
# Example:
#   ./run_epsilon_sweep.sh \
#     --sweepdir /scratch.global/$USER/pwr_sweep \
#     --pconndir /projects/standard/feczk001/shared/projects/ABCD/gordon_sets/data/group2_10minonly_FD0p1 \
#     --pconnref /projects/.../sub-NDARINV00J52GPG_..._pconn.nii \
#     --filedir  /scratch.global/$USER/power_calculator \
#     --condaenv FC_stability \
#     --eps-min 1 --eps-max 10 --eps-step 1

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SWEEPDIR=""
PCONNREF=""
PCONNDIR="$(pwd)"
FILEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDAENV=""
EPS_MIN=1
EPS_MAX=10
EPS_STEP=1
SINGLETEMP=0
NUMTEMP=1
NREP=10
NTIME=1000
MODEL="ridge"
NOISE_SEED=123456
BASE_READY=""
EXTRA=""
DRY_RUN=false

usage() { sed -n '2,60p' "${BASH_SOURCE[0]}"; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweepdir)   SWEEPDIR="$2";   shift 2 ;;
    --pconnref)   PCONNREF="$2";   shift 2 ;;
    --pconndir)   PCONNDIR="$2";   shift 2 ;;
    --filedir)    FILEDIR="$2";    shift 2 ;;
    --condaenv)   CONDAENV="$2";   shift 2 ;;
    --eps-min)    EPS_MIN="$2";    shift 2 ;;
    --eps-max)    EPS_MAX="$2";    shift 2 ;;
    --eps-step)   EPS_STEP="$2";   shift 2 ;;
    --singletemp) SINGLETEMP="$2"; shift 2 ;;
    --numtemp)    NUMTEMP="$2";    shift 2 ;;
    --nrep)       NREP="$2";       shift 2 ;;
    --ntime)      NTIME="$2";      shift 2 ;;
    --model)      MODEL="$2";      shift 2 ;;
    --noise-seed) NOISE_SEED="$2"; shift 2 ;;
    --base-ready) BASE_READY="$2"; shift 2 ;;
    --extra)      EXTRA="$2";      shift 2 ;;
    --dry-run)    DRY_RUN=true;    shift 1 ;;
    -h|--help)    usage ;;
    *) echo "[FATAL] Unknown argument: $1" >&2; usage ;;
  esac
done

[[ -z "$SWEEPDIR" ]] && { echo "[FATAL] --sweepdir is required" >&2; exit 1; }
[[ -z "$CONDAENV" ]] && { echo "[FATAL] --condaenv is required" >&2; exit 1; }
if [[ -z "$BASE_READY" && -z "$PCONNREF" ]]; then
  echo "[FATAL] --pconnref is required unless --base-ready is given" >&2
  exit 1
fi

# ── Base run ──────────────────────────────────────────────────────────────────
# Steps 1-3 only. --epsilon 0 is passed because combine_data.py requires a
# value; the yt file it writes is discarded and rewritten per epsilon.
BASEDIR="${BASE_READY:-$SWEEPDIR/base}"
BASE_JOB=""

if [[ -z "$BASE_READY" ]]; then
  mkdir -p "$BASEDIR"
  base_cmd=(sbatch --parsable
            -o "$BASEDIR/PWR_base_%j.out" -e "$BASEDIR/PWR_base_%j.err"
            "$FILEDIR/PWR.sh"
            --wrkdir "$BASEDIR" --pconndir "$PCONNDIR" --pconnref "$PCONNREF"
            --filedir "$FILEDIR" --condaenv "$CONDAENV"
            --singletemp "$SINGLETEMP" --numtemp "$NUMTEMP"
            --nrep "$NREP" --ntime "$NTIME"
            --epsilon 0 --stop-step combine)

  echo "[BASE] ${base_cmd[*]}"
  if [[ "$DRY_RUN" == "false" ]]; then
    BASE_JOB=$("${base_cmd[@]}")
    echo "[BASE] job=$BASE_JOB  wrkdir=$BASEDIR"
  else
    # Placeholder so the dry run still shows the --dependency flag that the
    # real per-epsilon submissions would carry.
    BASE_JOB="<BASE_JOB_ID>"
  fi
else
  echo "[BASE] reusing finished run at $BASEDIR (no base job submitted)"
  if [[ ! -d "$BASEDIR/pwr_data" ]]; then
    echo "[FATAL] --base-ready '$BASEDIR' has no pwr_data/ directory" >&2
    exit 1
  fi
fi

# ── Per-epsilon runs ──────────────────────────────────────────────────────────
# seq handles fractional steps, so --eps-step 0.5 works the same way. Values
# are used verbatim in the directory name (eps_1, eps_2.5, ...), which is what
# collect_sweep.py parses back out.
n_submitted=0
run_n=0

for EPS in $(seq "$EPS_MIN" "$EPS_STEP" "$EPS_MAX"); do
  run_n=$(( run_n + 1 ))
  WRK="$SWEEPDIR/eps_${EPS}"
  mkdir -p "$WRK"

  # Distinct seed per epsilon so the noise draws are independent across the
  # sweep while the whole thing stays reproducible from --noise-seed.
  SEED=$(( NOISE_SEED + run_n ))

  eps_cmd=(sbatch --parsable
           -o "$WRK/PWR_eps${EPS}_%j.out" -e "$WRK/PWR_eps${EPS}_%j.err"
           --job-name "PWR_eps${EPS}")

  # Queue behind the base run when there is one to wait for.
  if [[ -n "$BASE_JOB" ]]; then
    eps_cmd+=(--dependency=afterok:"$BASE_JOB")
  fi

  eps_cmd+=("$FILEDIR/PWR.sh"
            --wrkdir "$WRK" --filedir "$FILEDIR" --condaenv "$CONDAENV"
            --epsilon "$EPS" --reuse-from "$BASEDIR"
            --noise-seed "$SEED" --model "$MODEL")

  # Word splitting on $EXTRA is intentional: it carries multiple flags.
  if [[ -n "$EXTRA" ]]; then
    # shellcheck disable=SC2206
    eps_cmd+=($EXTRA)
  fi

  echo "[EPS $EPS] ${eps_cmd[*]}"
  if [[ "$DRY_RUN" == "false" ]]; then
    jid=$("${eps_cmd[@]}")
    echo "[EPS $EPS] job=$jid  wrkdir=$WRK"
    n_submitted=$(( n_submitted + 1 ))
  fi
done

echo "[DONE] base=${BASE_JOB:-<reused>}  epsilon runs submitted=$n_submitted"
echo "[NEXT] once everything finishes:"
echo "  python3 $FILEDIR/collect_sweep.py $SWEEPDIR"
