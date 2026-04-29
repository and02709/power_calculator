#!/bin/bash -l
# cv.sh — SLURM array task script for Step 6 of the power calculator pipeline.
#          (sklearn CV edition — replaces the legacy Step 5 cvGen + Step 7 cv split)
#
# Fits and evaluates a predictive model across all outer CV folds for ONE
# sample size per array task.  Calls cv.py (v2), which loads the full
# FCs_<size>.npy and y_<size>.npy directly and delegates all fold generation,
# preprocessing, fitting, and scoring to sklearn's cross_validate().
#
# Changes from the legacy cv.sh
# ------------------------------
# Legacy:  array size = NUMFILES × KFOLDS  (one task per size/fold pair)
# New:     array size = NUMFILES            (one task per sample size;
#                                            all folds run inside cv.py)
#
# The KFOLDS positional argument is replaced by --k_outer and --n_outer flags
# that map directly to RepeatedKFold in cv.py.  EPSILON is no longer passed
# to cv.py (it is applied upstream in combine_data.py and is irrelevant here).
#
# Model selection works identically to the legacy version: MODEL_FILE and all
# hyperparameter env vars are injected via --export in PWR.sh.  Flags
# irrelevant to the selected model are forwarded to cv.py and silently ignored.
#
# Resource profile notes
# ----------------------
# 128GB / 24h / 20 CPUs:
#   - All outer folds for one size run sequentially (N_JOBS=1) or in parallel
#     (N_JOBS=-1) within a single task.  With N_JOBS=1, peak memory is one
#     fold's training set; with N_JOBS=-1, k_outer simultaneous fold fits
#     may require proportionally more RAM.
#   - Neural network and gradient boosting models at large sizes can be slow;
#     24h is retained as a conservative ceiling.
#   - If using N_JOBS > 1 for the outer loop AND the model uses n_jobs=-1
#     internally (RF, SVR with GridSearchCV), set N_JOBS=1 here to avoid
#     CPU over-subscription.

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20        # Available to sklearn via n_jobs=-1 inside cv.py
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH -p msismall
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=and02709@umn.edu
#SBATCH -o cv_%a.out              # %a = array task ID (one log file per sample size)
#SBATCH -e cv_%a.err
#SBATCH --job-name cv

# ── Arguments ─────────────────────────────────────────────────────────────────
WRKDIR="$1"      # Root working directory
FILEDIR="$2"     # Pipeline scripts directory (contains cv.py, models/)
NUMFILES="$3"    # Number of sample sizes (determines array upper bound)
CONDAENV="$4"    # Conda environment name

# 1-based array task ID; cv.py maps this to the INDEX-th sample size
INDEX=${SLURM_ARRAY_TASK_ID}

# ── CV topology ───────────────────────────────────────────────────────────────
# These mirror the RepeatedKFold parameters inside cv.py.
K_OUTER=${K_OUTER:-10}       # k in RepeatedKFold — folds per repeat
N_OUTER=${N_OUTER:-2}        # n repeats of k-fold CV
RANDOM_STATE=${RANDOM_STATE:-123456}

# ── Runtime ───────────────────────────────────────────────────────────────────
# N_JOBS controls cross_validate()'s outer loop parallelism.
# Keep at 1 if the plugin's inner GridSearchCV already uses n_jobs=-1
# (e.g. ridge_nested, svr) to avoid CPU over-subscription.
N_JOBS=${N_JOBS:-1}
SAVE_ESTIMATORS=${SAVE_ESTIMATORS:-false}

# ── Model selection ───────────────────────────────────────────────────────────
MODEL_FILE=${MODEL_FILE:-ridge}

# ── Common flags ──────────────────────────────────────────────────────────────
USE_PCA=${USE_PCA:-false}
N_COMPONENTS=${N_COMPONENTS:-500}

PCA_FLAG=""
if [[ "$USE_PCA" == "true" ]]; then
  PCA_FLAG="--pca"
fi

SAVE_EST_FLAG=""
if [[ "$SAVE_ESTIMATORS" == "true" ]]; then
  SAVE_EST_FLAG="--save_estimators"
fi

# ── Model-specific hyperparameter defaults ────────────────────────────────────
# All values default here so cv.sh is self-contained.  Values injected via
# --export in PWR.sh override these defaults.  Flags irrelevant to the selected
# model are forwarded and silently ignored by cv.py / the plugin parser.

# ridge / ridge_nested
RIDGE_ALPHAS=${RIDGE_ALPHAS:-"1,10,100,1000,10000,100000"}
RIDGE_CV_FOLDS=${RIDGE_CV_FOLDS:-5}      # for RidgeCV inner selection
RIDGE_K_INNER=${RIDGE_K_INNER:-5}        # for GridSearchCV inner folds (ridge_nested)

# lasso
LASSO_N_ALPHAS=${LASSO_N_ALPHAS:-100}
LASSO_CV_FOLDS=${LASSO_CV_FOLDS:-5}
LASSO_MAX_ITER=${LASSO_MAX_ITER:-5000}

# elastic_net
EN_L1_RATIOS=${EN_L1_RATIOS:-"0.1,0.5,0.7,0.9,0.95,1.0"}
EN_N_ALPHAS=${EN_N_ALPHAS:-100}
EN_CV_FOLDS=${EN_CV_FOLDS:-5}
EN_MAX_ITER=${EN_MAX_ITER:-5000}

# random_forest
RF_N_ESTIMATORS=${RF_N_ESTIMATORS:-500}
RF_MAX_FEATURES=${RF_MAX_FEATURES:-"1.0"}
RF_TUNE=${RF_TUNE:-false}
RF_K_INNER=${RF_K_INNER:-3}

RF_TUNE_FLAG=""
if [[ "$RF_TUNE" == "true" ]]; then
  RF_TUNE_FLAG="--rf_tune"
fi

# svr
SVR_C_VALS=${SVR_C_VALS:-"0.1,1,10,100"}
SVR_KERNEL=${SVR_KERNEL:-rbf}
SVR_EPSILON=${SVR_EPSILON:-0.1}
SVR_K_INNER=${SVR_K_INNER:-5}

# neural_network
NN_HIDDEN_LAYERS=${NN_HIDDEN_LAYERS:-"256,128"}
NN_ACTIVATION=${NN_ACTIVATION:-relu}
NN_LR=${NN_LR:-0.001}
NN_MAX_ITER=${NN_MAX_ITER:-500}
NN_ALPHA=${NN_ALPHA:-0.0001}

# gradient_boosting
GB_N_ESTIMATORS=${GB_N_ESTIMATORS:-300}
GB_LR=${GB_LR:-0.05}
GB_MAX_DEPTH=${GB_MAX_DEPTH:-4}
GB_SUBSAMPLE=${GB_SUBSAMPLE:-0.8}
GB_TUNE=${GB_TUNE:-false}
GB_K_INNER=${GB_K_INNER:-3}

GB_TUNE_FLAG=""
if [[ "$GB_TUNE" == "true" ]]; then
  GB_TUNE_FLAG="--gb_tune"
fi

# ── Environment ───────────────────────────────────────────────────────────────
module purge || true
if [[ "$CONDAENV" == "FC_stability" ]]; then
  source /projects/standard/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh
fi
conda activate "$CONDAENV"

# ── Run ───────────────────────────────────────────────────────────────────────
# Note: NN_HIDDEN_LAYERS and string alpha lists are quoted to prevent
# word-splitting on comma separators (e.g. "256,128" must arrive as one arg).
python3 "$FILEDIR/cv.py" \
    "$WRKDIR" "$FILEDIR" "$NUMFILES" "$INDEX" \
    --model_file      "$MODEL_FILE"            \
    --k_outer         "$K_OUTER"               \
    --n_outer         "$N_OUTER"               \
    --random_state    "$RANDOM_STATE"          \
    --n_jobs          "$N_JOBS"                \
    --n_components    "$N_COMPONENTS"          \
    --ridge_alphas    "$RIDGE_ALPHAS"          \
    --ridge_cv_folds  "$RIDGE_CV_FOLDS"        \
    --ridge_k_inner   "$RIDGE_K_INNER"         \
    --lasso_n_alphas  "$LASSO_N_ALPHAS"        \
    --lasso_cv_folds  "$LASSO_CV_FOLDS"        \
    --lasso_max_iter  "$LASSO_MAX_ITER"        \
    --en_l1_ratios    "$EN_L1_RATIOS"          \
    --en_n_alphas     "$EN_N_ALPHAS"           \
    --en_cv_folds     "$EN_CV_FOLDS"           \
    --en_max_iter     "$EN_MAX_ITER"           \
    --rf_n_estimators "$RF_N_ESTIMATORS"       \
    --rf_max_features "$RF_MAX_FEATURES"       \
    --rf_k_inner      "$RF_K_INNER"            \
    --svr_C_vals      "$SVR_C_VALS"            \
    --svr_kernel      "$SVR_KERNEL"            \
    --svr_epsilon     "$SVR_EPSILON"           \
    --svr_k_inner     "$SVR_K_INNER"           \
    --nn_hidden_layers "$NN_HIDDEN_LAYERS"     \
    --nn_activation   "$NN_ACTIVATION"         \
    --nn_lr           "$NN_LR"                 \
    --nn_max_iter     "$NN_MAX_ITER"           \
    --nn_alpha        "$NN_ALPHA"              \
    --gb_n_estimators "$GB_N_ESTIMATORS"       \
    --gb_lr           "$GB_LR"                 \
    --gb_max_depth    "$GB_MAX_DEPTH"          \
    --gb_subsample    "$GB_SUBSAMPLE"          \
    --gb_k_inner      "$GB_K_INNER"            \
    $PCA_FLAG                                  \
    $RF_TUNE_FLAG                              \
    $GB_TUNE_FLAG                              \
    $SAVE_EST_FLAG
