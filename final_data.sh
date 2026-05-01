#!/bin/bash -l
# final_data.sh — SLURM batch script for the terminal step of the power calculator pipeline.
#
# Runs final_data.py to aggregate all per-fold CV scores produced by cv.py
# into the final power curve outputs.
#
# Reads all cv_results_size<N>_<model>.csv files in pwr_data/ and reduces
# them across folds per sample size to produce:
#   metrics_data.{pkl,csv}    — per-fold R² table
#   metrics_summary.{pkl,csv} — power curve (mean R² ± SD per size)
#   mean_metric_by_size.png   — power curve plot
#
# Runs as a single (non-array) job submitted synchronously (--wait) from
# PWR.sh, so the calling shell blocks until aggregation is complete.
#
# Usage (via PWR.sh submit()):
#   sbatch final_data.sh <WRKDIR> <FILEDIR> <CONDAENV>

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH -p msismall
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=and02709@umn.edu
#SBATCH -o final_data.out
#SBATCH -e final_data.err
#SBATCH --job-name final_data

# ── Arguments ─────────────────────────────────────────────────────────────────
WRKDIR="$1"    # Root working directory
FILEDIR="$2"   # Pipeline scripts directory
CONDAENV="$3"  # Conda environment to activate

# ── Model (injected via --export from PWR.sh) ─────────────────────────────────
MODEL_FILE=${MODEL_FILE:-ridge}
OUT_FORMAT=${OUT_FORMAT:-csv}

# ── Environment ───────────────────────────────────────────────────────────────
module purge || true
if [[ "$CONDAENV" == "FC_stability" ]]; then
  source /projects/standard/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh
fi
conda activate "$CONDAENV"

cd "$WRKDIR/pwr_data"

# ── Diagnostics ───────────────────────────────────────────────────────────────
echo "[INFO] Running final_data.py"
echo "[INFO] WRKDIR=$WRKDIR"
echo "[INFO] FILEDIR=$FILEDIR"
echo "[INFO] MODEL_FILE=$MODEL_FILE"
echo "[INFO] OUT_FORMAT=$OUT_FORMAT"
echo "[INFO] CONDAENV=$CONDAENV"

# ── Run ───────────────────────────────────────────────────────────────────────
python3 "$FILEDIR/final_data.py" \
    "$WRKDIR" \
    "$FILEDIR" \
    --model      "$MODEL_FILE" \
    --out_format "$OUT_FORMAT"
