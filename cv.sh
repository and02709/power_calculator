#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH -p msismall
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=and02709@umn.edu
#SBATCH -o cv_%a.out
#SBATCH -e cv_%a.err
#SBATCH --job-name cv

# ---------------------------------------------------------------------------
# Arguments (positional — unchanged from original)
# ---------------------------------------------------------------------------
WRKDIR=$1
FILEDIR=$2
NUMFILES=$3
KFOLDS=$4
EPSILON=$5
INDEX=${SLURM_ARRAY_TASK_ID}

# ---------------------------------------------------------------------------
# Model selection
#   Pass MODEL_FILE as an environment variable, e.g.:
#     MODEL_FILE=ridge sbatch --array=1-50 cv.sh ...
#   Default: random_forest  (preserves original behaviour)
# ---------------------------------------------------------------------------
MODEL_FILE=${MODEL_FILE:-random_forest}

# ---------------------------------------------------------------------------
# Model-specific hyperparameter overrides (ignored if not relevant to model)
# ---------------------------------------------------------------------------
N_COMPONENTS=${N_COMPONENTS:-500}
N_ESTIMATORS=${N_ESTIMATORS:-500}   # random_forest
RIDGE_ALPHA=${RIDGE_ALPHA:-1.0}     # ridge
LASSO_ALPHA=${LASSO_ALPHA:-0.01}    # lasso
EN_ALPHA=${EN_ALPHA:-0.01}          # elastic_net
EN_L1_RATIO=${EN_L1_RATIO:-0.5}    # elastic_net
SVR_C=${SVR_C:-1.0}                 # svr
NN_HIDDEN_LAYERS=${NN_HIDDEN_LAYERS:-256,128}  # neural_network
NN_LR=${NN_LR:-0.001}              # neural_network
GB_N_ESTIMATORS=${GB_N_ESTIMATORS:-300}  # gradient_boosting
GB_LR=${GB_LR:-0.05}               # gradient_boosting

module load python3/3.10.4-anaconda2023.03

python3 $FILEDIR/cv.py \
    $WRKDIR $FILEDIR $NUMFILES $KFOLDS $EPSILON $INDEX \
    --model_file   $MODEL_FILE \
    --n_components $N_COMPONENTS \
    --n_estimators $N_ESTIMATORS \
    --ridge_alpha  $RIDGE_ALPHA \
    --lasso_alpha  $LASSO_ALPHA \
    --en_alpha     $EN_ALPHA \
    --en_l1_ratio  $EN_L1_RATIO \
    --svr_C        $SVR_C \
    --nn_hidden_layers "$NN_HIDDEN_LAYERS" \
    --nn_lr        $NN_LR \
    --gb_n_estimators $GB_N_ESTIMATORS \
    --gb_lr        $GB_LR
