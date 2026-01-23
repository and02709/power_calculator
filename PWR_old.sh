#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24GB
#SBATCH --time=12:00:00
#SBATCH -p msismall
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=and02709@umn.edu
#SBATCH -o PWR.out
#SBATCH -e PWR.err
#SBATCH --job-name=PWR

set -euo pipefail

WRKDIR=$1
KFOLDS=$2
EPSILON=$3
FILEDIR=/users/0/and02709/power_calculator/new_work

module load R/4.4.0-openblas-rocky8

cd "$WRKDIR"
mkdir -p "$WRKDIR/pwr_data"
cd "$WRKDIR/pwr_data"

# Step 0: check the EPSILON index
# (kept as-is; uses bc)
if ! [[ "$EPSILON" =~ ^[0-9]+([.][0-9]+)?$ ]] || (( $(echo "$EPSILON <= 0" | bc -l) )); then
  echo "Error: EPSILON must be a number greater than 0." >&2
  exit 1
fi

# Step 1: generate index file
sbatch --chdir="$WRKDIR/pwr_data" --time=1:00:00 --mem=4GB --cpus-per-task=1 --wait -N1 \
  "$FILEDIR/pwr_index_file.sh" "$WRKDIR" "$FILEDIR"

# Step 2: get number of rows
if [[ ! -s pwr_index_file.txt ]]; then
  echo "Error: pwr_index_file.txt not found (or empty) in $WRKDIR/pwr_data" >&2
  ls -lah >&2
  exit 2
fi
NINDEX=$(wc -l < pwr_index_file.txt)

# Step 3: define chunk size
CHUNK_SIZE=100

# Step 4: compute number of jobs needed
NJOBS=$(( (NINDEX + CHUNK_SIZE - 1) / CHUNK_SIZE ))
if [[ "$NJOBS" -lt 1 ]]; then
  echo "Error: NJOBS computed as $NJOBS (NINDEX=$NINDEX). Check pwr_index_file.txt" >&2
  exit 2
fi

# Step 5: setup data
sbatch --chdir="$WRKDIR/pwr_data" --time=1:00:00 --mem=16GB --cpus-per-task=2 --wait -N1 \
  "$FILEDIR/pwr_setup.sh" "$WRKDIR" "$FILEDIR"

# Step 6: submit array job (this calls pwr_sub_python.sh, which activates conda + runs python)
sbatch --time=10:00:00 --mem=16GB --cpus-per-task=2 --array=1-$NJOBS --wait -N1 \
  $FILEDIR/pwr_sub_python.sh $WRKDIR $CHUNK_SIZE $NINDEX $FILEDIR


# Step 7: combine output
sbatch --chdir="$WRKDIR/pwr_data" --time=1:00:00 --mem=64GB --cpus-per-task=4 --wait -N1 \
  "$FILEDIR/combine_data.sh" "$WRKDIR" "$FILEDIR"

# Step 8: Develop Ridge Model
sbatch --chdir="$WRKDIR/pwr_data" --time=8:00:00 --mem=64GB --cpus-per-task=2 --wait -N1 \
  "$FILEDIR/ridge.sh" "$WRKDIR" "$FILEDIR"

# Step 9: count number of files
NUMFILES=$(ls full_[0-9]*.rds 2>/dev/null | wc -l)
if [[ "$NUMFILES" -lt 1 ]]; then
  echo "Error: NUMFILES=0. No full_[0-9]*.rds found after ridge step." >&2
  ls -lah "$WRKDIR/pwr_data" >&2
  exit 3
fi

# Step 10: Partition datasets
sbatch --chdir="$WRKDIR/pwr_data" --time=1:00:00 --mem=16GB --cpus-per-task=2 --array=1-"$NUMFILES" --wait -N1 \
  "$FILEDIR/cvGen.sh" "$WRKDIR" "$FILEDIR" "$NUMFILES" "$KFOLDS"

# Step 11: Setup metrics files
sbatch --chdir="$WRKDIR/pwr_data" --time=1:00:00 --mem=16GB --cpus-per-task=2 --wait -N1 \
  "$FILEDIR/setupCVmetrics.sh" "$WRKDIR" "$FILEDIR"

# Step 12: count number folds and files
NUMFFILES=$(ls "$WRKDIR"/pwr_data/full_*_fold_*_split.rds 2>/dev/null | wc -l)
if [[ "$NUMFFILES" -lt 1 ]]; then
  echo "Error: NUMFFILES=0. No full_*_fold_*_split.rds found after cvGen/setupCVmetrics." >&2
  ls -lah "$WRKDIR/pwr_data" >&2
  exit 4
fi

# Step 13: Perform CV metrics
sbatch --chdir="$WRKDIR/pwr_data" --time=2:00:00 --mem=32GB --cpus-per-task=2 --array=1-"$NUMFFILES" --wait -N1 \
  "$FILEDIR/cv.sh" "$WRKDIR" "$FILEDIR" "$NUMFILES" "$KFOLDS" "$EPSILON"

# Step 14: Final Compilation
sbatch --chdir="$WRKDIR/pwr_data" --time=12:00:00 --mem=96GB --cpus-per-task=8 --wait -N1 \
  "$FILEDIR/final_data.sh" "$WRKDIR" "$FILEDIR"

# Step 15: Move logs (safe)
cd "$WRKDIR"
mkdir -p "$WRKDIR/OUT" "$WRKDIR/ERR"

shopt -s nullglob
out_files=("$WRKDIR/pwr_data"/*.out)
err_files=("$WRKDIR/pwr_data"/*.err)

if (( ${#out_files[@]} )); then mv "${out_files[@]}" "$WRKDIR/OUT/"; fi
if (( ${#err_files[@]} )); then mv "${err_files[@]}" "$WRKDIR/ERR/"; fi
