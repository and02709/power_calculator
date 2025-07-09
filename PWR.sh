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

WRKDIR=$1
KFOLDS=$2
PHENO=$3
EPSILON=$4
FILEDIR=/users/0/and02709/power_calculator

module load R/4.4.0-openblas-rocky8
cd $WRKDIR
mkdir -p $WRKDIR/pwr_data
cd $WRKDIR/pwr_data

# Step 0: check the EPSILON index
# Check if EPSILON is a number and within the valid range [0, 1)
if ! [[ "$EPSILON" =~ ^[0-9]+([.][0-9]+)?$ ]] || (( $(echo "$EPSILON < 0" | bc -l) )) || (( $(echo "$EPSILON >= 1" | bc -l) )); then
    echo "Error: EPSILON must be a number greater than or equal to 0 and less than 1." >&2
    exit 1
fi

# Step 1: generate index file
sbatch --time=1:00:00 --mem=4GB --cpus-per-task=1 --wait -N1 $FILEDIR/pwr_index_file.sh $WRKDIR $FILEDIR $PHENO $FILEDIR

# Step 2: get number of rows
NINDEX=$(wc -l < pwr_index_file.txt)

# Step 3: define chunk size
CHUNK_SIZE=50

# Step 4: compute number of jobs needed
NJOBS=$(( (NINDEX + CHUNK_SIZE - 1) / CHUNK_SIZE ))

# Step 5: setup data
sbatch --time=1:00:00 --mem=16GB --cpus-per-task=2 --wait -N1 $FILEDIR/pwr_setup.sh $WRKDIR $FILEDIR $PHENO $FILEDIR

# Step 6: submit array job
sbatch --time=10:00:00 --mem=16GB --cpus-per-task=2 --array=1-$NJOBS --wait -N1 $FILEDIR/pwr_sub.sh $WRKDIR $CHUNK_SIZE $NINDEX

# Step 7: combine output
sbatch --time=1:00:00 --mem=64GB --cpus-per-task=4 --wait -N1 $FILEDIR/combine_data.sh $WRKDIR $FILEDIR $PHENO $FILEDIR

# Step 8: Develop Ridge Model
sbatch --time=8:00:00 --mem=64GB --cpus-per-task=2 --wait -N1 $FILEDIR/ridge.sh $WRKDIR $FILEDIR $PHENO

# Step 8: count number of files
NUMFILES=$(ls full_[0-9]*.rds 2>/dev/null | wc -l)

# Step 9: Partition datasets
sbatch --time=1:00:00 --mem=16GB --cpus-per-task=2 --array=1-$NUMFILES --wait -N1 $FILEDIR/cvGen.sh $WRKDIR $FILEDIR $NUMFILES $KFOLDS

# Step 10: Setup metrics files
sbatch --time=1:00:00 --mem=16GB --cpus-per-task=2 --wait -N1 $FILEDIR/setupCVmetrics.sh $WRKDIR $FILEDIR $PHENO $FILEDIR

# Step 11: count number folds and files
NUMFFILES=$(ls "$WRKDIR"/pwr_data/full_*_fold_*_split.rds 2>/dev/null | wc -l)

# Step 12: Perform CV metrics
sbatch --time=2:00:00 --mem=32GB --cpus-per-task=2 --array=1-$NUMFFILES --wait -N1 $FILEDIR/cv.sh $WRKDIR $FILEDIR $NUMFILES $KFOLDS $EPSILON

# Step 13: Final Compilation
sbatch --time=12:00:00 --mem=96GB --cpus-per-task=8 --wait -N1 $FILEDIR/final_data.sh $WRKDIR $FILEDIR $PHENO $FILEDIR 

# Step 14: Remove extra files
cd $WRKDIR
mkdir -p $WRKDIR/OUT
mkdir -p $WRKDIR/ERR

# Move .out files
mv "$WRKDIR/pwr_data"/*.out "$WRKDIR/OUT/"

# Move .err files
mv "$WRKDIR/pwr_data"/*.err "$WRKDIR/ERR/"

rm -r $WRKDIR/pwr_data
