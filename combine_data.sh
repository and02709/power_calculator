#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=1:00:00
#SBATCH -p msismall
#SBATCH -o combine_data.out
#SBATCH -e combine_data.err
#SBATCH --job-name=combine_data

set -euo pipefail

WRKDIR="$1"
FILEDIR="$2"

cd "$WRKDIR/pwr_data"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pwr_py

python "$FILEDIR/combine_data.py" "$WRKDIR" "$FILEDIR"
# If you want RDS outputs too (requires pyreadr in env):
# python "$FILEDIR/python_refactor/combine_data.py" "$WRKDIR" "$FILEDIR" --write-rds
