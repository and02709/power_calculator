<p align="center">
  <i>A pipeline for simulating brain imaging data and associated phenotypes for power calculations.</i>
  <br/>
</p>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white)

A brain imaging power calculator developed by the Masonic Institute of the Developing Brain at the University of Minnesota. The pipeline is built using [Python](https://www.python.org/) and [bash](https://www.gnu.org/software/bash/), and is designed to run on HPC clusters with a SLURM scheduler.

---

## Overview

This pipeline estimates statistical power for brain imaging studies by:

1. Generating a grid of sample sizes to evaluate
2. Simulating functional connectivity covariance matrices from real `.pconn.nii` reference data
3. Combining simulated data across sample sizes
4. Running k-fold cross-validation with a pluggable ML model to predict phenotypes from connectivity
5. Aggregating CV metrics and producing power curves

The primary entry point is `PWR.sh`, which orchestrates all steps as a chain of dependent SLURM jobs.

---

## Requirements

- HPC cluster with SLURM scheduler
- Python 3 with the following packages:
  - `numpy`, `pandas`, `nibabel`, `scikit-learn`, `matplotlib`
- Conda environment `FC_stability` (used internally by worker scripts)
  - Activate via the miniconda3 loader at `/projects/standard/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh`

---

## Quick Start

Clone the repository and submit the pipeline:

```bash
git clone https://github.com/your-org/power_calculator.git
cd power_calculator

sbatch PWR.sh \
  --pconnref  /path/to/reference.pconn.nii \
  --singletemp 0 \
  --numtemp    5 \
  --kfolds     5 \
  --epsilon    1.0
```

`--wrkdir`, `--pconndir`, and `--filedir` all default to `$PWD`. `--nrep` defaults to `10` and `--ntime` to `1000`.

---

## Usage

```
sbatch PWR.sh [OPTIONS]
```

### Required Arguments

| Flag            | Description                                      |
|-----------------|--------------------------------------------------|
| `--pconnref`    | Path to the reference `.pconn.nii` file          |
| `--singletemp`  | `0` = multi-temperature mode, `1` = single-temperature mode |
| `--numtemp`     | Number of temperatures (integer >= 1)            |
| `--kfolds`      | Number of cross-validation folds                 |
| `--epsilon`     | Regularization/convergence epsilon (float >= 0)  |

### Optional Arguments (with defaults)

| Flag          | Default  | Description                                      |
|---------------|----------|--------------------------------------------------|
| `--wrkdir`    | `$PWD`   | Working directory for outputs                    |
| `--pconndir`  | `$PWD`   | Directory containing `.pconn.nii` subject files  |
| `--filedir`   | `$PWD`   | Directory containing pipeline scripts            |
| `--nrep`      | `10`     | Number of simulation repetitions                 |
| `--ntime`     | `1000`   | Number of timepoints per simulation              |

### Model Selection

| Flag              | Default          | Description                                           |
|-------------------|------------------|-------------------------------------------------------|
| `--model`         | `random_forest`  | ML model to use for CV (see available models below)   |
| `--n-components`  | `500`            | Number of PCA components                             |
| `--n-estimators`  | `500`            | Number of trees (random forest / gradient boosting)   |

### Model Hyperparameters

| Flag               | Default   | Model           |
|--------------------|-----------|-----------------|
| `--ridge-alpha`    | `1.0`     | Ridge           |
| `--lasso-alpha`    | `0.01`    | Lasso           |
| `--en-alpha`       | `0.01`    | ElasticNet      |
| `--en-l1-ratio`    | `0.5`     | ElasticNet      |
| `--svr-c`          | `1.0`     | SVR             |
| `--nn-hidden`      | `256,128` | Neural Network  |
| `--nn-lr`          | `0.001`   | Neural Network  |
| `--gb-estimators`  | `300`     | Gradient Boosting |
| `--gb-lr`          | `0.05`    | Gradient Boosting |

### Example Call

```bash
sbatch /scratch.global/and02709/power_calculator/PWR.sh \
  --wrkdir     /scratch.global/and02709/p2 \
  --pconndir   /projects/standard/feczk001/shared/projects/ABCD/gordon_sets/data/group2_10minonly_FD0p1 \
  --pconnref   /projects/standard/feczk001/shared/projects/ABCD/gordon_sets/data/group2_10minonly_FD0p1/sub-NDARINV00J52GPG_ses-baselineYear1Arm1_task-rest_bold_roi-Gordon2014FreeSurferSubcortical_timeseries.ptseries.nii_5_minutes_of_data_at_FD_0.2.pconn.nii \
  --singletemp 0 \
  --numtemp    1 \
  --filedir    /scratch.global/and02709/power_calculator \
  --kfolds     5 \
  --nrep       20 \
  --ntime      2000 \
  --epsilon    1
```

---

## Pipeline Steps

The orchestrator (`PWR.sh`) submits the following SLURM jobs in sequence, each waiting for the previous to complete:

| Step | Script               | Description                                                                 |
|------|----------------------|-----------------------------------------------------------------------------|
| 1    | `pwr_setup.sh`       | Generates `pwr_index_file.txt`: a grid of (sample size, replicate index) combinations to evaluate |
| 2    | `pwr_sub_python.sh` / `pwr_sub_python_single.sh` | Array job — simulates FC covariance matrices from the reference pconn for each chunk of indices |
| 3    | `combine_data.sh`    | Combines per-index covariance `.npy` files into full sample-size matrices   |
| 4    | `cvGen.sh`           | Generates k-fold train/test splits (`.npz`) for each sample size            |
| 5    | `setupCVmetrics.sh`  | Initializes metric collection structures for CV output                      |
| 6    | `cv.sh`              | Array job — runs cross-validation for each fold/sample-size combination     |
| 7    | `final_data.sh`      | Aggregates CV metrics, computes power curves, and generates summary plots   |

A `job_manifest.tsv` is written to `$WRKDIR/OUT/` recording the job ID, stdout, and stderr path for every submitted step.

### Single vs. Multi-Temperature Mode

- **`--singletemp 1`**: Uses `pwr_sub_python_single.sh` → `pwr_process_chunk_single_z.py`. Simulates from a single reference pconn with `--use_one_target`.
- **`--singletemp 0`**: Uses `pwr_sub_python.sh` → `pwr_process_chunk_z.py`. Simulates across multiple temperatures controlled by `--numtemp`.

---

## Available ML Models

Models live in the `models/` directory. Each is a self-contained plugin implementing `CVModel` from `models/base.py`.

| Model name          | Description                        |
|---------------------|------------------------------------|
| `random_forest`     | PCA + Random Forest (default)      |
| `ridge`             | PCA + Ridge Regression             |
| `lasso`             | PCA + Lasso Regression             |
| `elastic_net`       | PCA + ElasticNet                   |
| `svr`               | PCA + Support Vector Regression    |
| `neural_network`    | PCA + MLP Regressor                |
| `gradient_boosting` | PCA + Gradient Boosting            |

### Adding a Custom Model

1. Copy `models/TEMPLATE.py` to `models/<your_model_name>.py`
2. Implement `cli_args()`, `__init__()`, `fit()`, and `predict()`
3. Decorate the class with `@register("<your_model_name>")`
4. Pass `--model <your_model_name>` to `PWR.sh`

No changes to `cv.py`, `cv.sh`, or `PWR.sh` are required.

---

## Output

All outputs are written under `$WRKDIR/`:

```
$WRKDIR/
├── OUT/                        # SLURM stdout logs and job_manifest.tsv
├── ERR/                        # SLURM stderr logs
└── pwr_data/
    ├── pwr_index_file.txt      # Sample size / replicate index grid
    ├── dat_size_*_index_*_cov.npy     # Per-replicate covariance matrices
    ├── dat_size_*_index_*_meta.json   # Metadata (pconn sources, params)
    ├── full_*_cov.npy          # Combined covariance matrices per sample size
    ├── full_*_fold_*_split.npz # CV train/test splits
    ├── full_*_fold_*_*.npy     # Per-fold CV metrics
    └── metrics_summary.pkl     # Final aggregated power curve data
```

---

## Repository Structure

```
power_calculator/
├── PWR.sh                        # Main orchestrator — submit this
├── pwr_setup.sh / .py            # Step 1: index grid generation
├── pwr_sub_python.sh             # Step 2: multi-temp worker dispatcher
├── pwr_sub_python_single.sh      # Step 2: single-temp worker dispatcher
├── pwr_process_chunk_z.py        # Core simulation (multi-temp)
├── pwr_process_chunk_single_z.py # Core simulation (single-temp)
├── combine_data.sh / .py         # Step 3: covariance aggregation
├── cvGen.sh / .py                # Step 4: CV split generation
├── setupCVmetrics.sh / .py       # Step 5: metric initialization
├── cv.sh / .py                   # Step 6: cross-validation runner
├── final_data.sh / .py           # Step 7: aggregation and plotting
├── ridge_model_generation.py     # Standalone ridge model utility
├── models/
│   ├── TEMPLATE.py               # Template for adding new models
│   ├── base.py                   # CVModel base class and registry
│   ├── random_forest.py
│   ├── ridge.py
│   ├── lasso.py
│   ├── elastic_net.py
│   ├── svr.py
│   ├── neural_network.py
│   └── gradient_boosting.py
└── haufe.csv                     # Reference ridge weights (if used)
```

---

## License

[MIT licensed](LICENSE).
