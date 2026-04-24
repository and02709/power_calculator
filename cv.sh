#!/bin/bash -l
# cv.sh — SLURM array task script for Step 7 of the power calculator pipeline.
#
# Fits and evaluates a predictive model on one sample-size/fold combination
# per array task. Each task calls cv.py, which loads the corresponding
# full_<size>_fold_<k>_split.npz produced by Step 5 (cvGen.py), fits the
# selected model on the training set, and evaluates it on the test set.
#
# Array indexing: one task per split file (1..NUMFFILES, where
# NUMFFILES = NUMFILES × KFOLDS). SLURM_ARRAY_TASK_ID is passed to cv.py
# as INDEX, which maps it to a specific (size, fold) pair via cv.pkl.
#
# Model selection is controlled entirely via environment variables passed
# through --export in PWR.sh rather than positional arguments, so this
# script stays model-agnostic. Hyperparameters irrelevant to the selected
# model are forwarded but silently ignored by cv.py.
#
# Resource profile (128GB / 24h / 20 CPUs) is the highest in the pipeline:
#   - 128GB: in-memory model fitting on the full training fold for the
#            largest sample sizes, plus PCA if enabled.
#   - 24h:   neural network and gradient boosting models can be slow at
#            large sample sizes with many estimators.
#   - 20 CPUs: passed to sklearn models that support n_jobs=-1 parallelism.
#
# Usage (via PWR.sh submit()):
#   sbatch --array=1-$NUMFFILES \
#     --export=ALL,MODEL_FILE=...,USE_PCA=...,<hyperparams> \
#     cv.sh <WRKDIR> <FILEDIR> <NUMFILES> <KFOLDS> <EPSILON>

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20        # Available to sklearn models via n_jobs=-1
#SBATCH --mem=128GB               # Highest in pipeline — large training folds + PCA
#SBATCH --time=24:00:00           # NN/GB models can be slow at large sample sizes
#SBATCH -p msismall
#SBATCH --mail-type=FAIL          # Email only on job failure
#SBATCH --mail-user=and02709@umn.edu
#SBATCH -o cv_%a.out              # %a = array task ID (one log file per size/fold)
#SBATCH -e cv_%a.err
#SBATCH --job-name cv

# ── Arguments ─────────────────────────────────────────────────────────────────
WRKDIR=$1      # Root working directory
FILEDIR=$2     # Pipeline scripts directory
NUMFILES=$3    # Number of sample sizes (used by cv.py to resolve INDEX → size)
KFOLDS=$4      # Number of CV folds per sample size
EPSILON=$5     # Noise scale factor (passed through to cv.py for reference)

# 1-based array task ID; cv.py maps this to a (size, fold) pair via cv.pkl
INDEX=${SLURM_ARRAY_TASK_ID}

# ── Model selection ───────────────────────────────────────────────────────────
# MODEL_FILE selects the model module from FILEDIR/models/<MODEL_FILE>.py.
# Injected via --export in PWR.sh; defaults to random_forest if not set,
# preserving the original pipeline behaviour before multi-model support was added.
MODEL_FILE=${MODEL_FILE:-random_forest}

# USE_PCA controls whether PCA dimensionality reduction is applied to the
# feature matrix before model fitting. Converted to a flag (--pca) below
# rather than passed as a string argument so cv.py uses action="store_true".
USE_PCA=${USE_PCA:-false}

# ── Hyperparameter defaults ───────────────────────────────────────────────────
# All hyperparameters default here so cv.sh is self-contained and runnable
# without --export. Values injected via PWR.sh --export override these defaults.
# Parameters irrelevant to the selected MODEL_FILE are forwarded to cv.py
# but silently ignored there — no conditional logic is needed in this script.
N_COMPONENTS=${N_COMPONENTS:-500}        # PCA: number of components (if USE_PCA=true)
N_ESTIMATORS=${N_ESTIMATORS:-500}        # random_forest: number of trees
RIDGE_ALPHA=${RIDGE_ALPHA:-1.0}          # ridge: regularisation strength
LASSO_ALPHA=${LASSO_ALPHA:-0.01}         # lasso: regularisation strength
EN_ALPHA=${EN_ALPHA:-0.01}               # elastic_net: regularisation strength
EN_L1_RATIO=${EN_L1_RATIO:-0.5}         # elastic_net: L1/L2 mixing (0=Ridge, 1=Lasso)
SVR_C=${SVR_C:-1.0}                      # svr: regularisation parameter
NN_HIDDEN_LAYERS=${NN_HIDDEN_LAYERS:-256,128}  # neural_network: layer sizes (comma-separated)
NN_LR=${NN_LR:-0.001}                   # neural_network: learning rate
GB_N_ESTIMATORS=${GB_N_ESTIMATORS:-300} # gradient_boosting: number of boosting stages
GB_LR=${GB_LR:-0.05}                    # gradient_boosting: learning rate (shrinkage)

# ── PCA flag construction ─────────────────────────────────────────────────────
# cv.py uses --pca as a store_true flag rather than --pca true/false, so
# the flag must be either present or absent in the python3 call. An empty
# PCA_FLAG is safe to include in the argument list — the shell expands it
# to nothing rather than passing a blank string argument.
PCA_FLAG=""
if [[ "$USE_PCA" == "true" ]]; then
  PCA_FLAG="--pca"
fi

# ── Environment ───────────────────────────────────────────────────────────────
# Purge inherited modules before activating conda to avoid version conflicts.
module purge || true   # || true prevents abort under set -e if no modules loaded
source /projects/standard/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh
conda activate FC_stability

# ── Run ───────────────────────────────────────────────────────────────────────
# NN_HIDDEN_LAYERS is quoted to prevent word-splitting on the comma separator
# (e.g. "256,128" must arrive as one argument, not two). All other
# hyperparameters are unquoted scalars and do not require quoting.
python3 $FILEDIR/cv.py \
    $WRKDIR $FILEDIR $NUMFILES $KFOLDS $EPSILON $INDEX \
    --model_file      $MODEL_FILE \
    --n_components    $N_COMPONENTS \
    --n_estimators    $N_ESTIMATORS \
    --ridge_alpha     $RIDGE_ALPHA \
    --lasso_alpha     $LASSO_ALPHA \
    --en_alpha        $EN_ALPHA \
    --en_l1_ratio     $EN_L1_RATIO \
    --svr_C           $SVR_C \
    --nn_hidden_layers "$NN_HIDDEN_LAYERS" \   # Quoted: comma in value must not split
    --nn_lr           $NN_LR \
    --gb_n_estimators $GB_N_ESTIMATORS \
    --gb_lr           $GB_LR \
    $PCA_FLAG                                  # Expands to --pca or nothing
