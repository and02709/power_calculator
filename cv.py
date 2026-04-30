#!/usr/bin/env python3
"""
cv.py — Step 6 of the power calculator pipeline (sklearn CV edition).

Replaces the manual split/fit/predict loop with sklearn's cross_validate(),
which handles fold generation, preprocessing safety, fitting, and scoring in
a single call.  The bespoke split-file approach (cvGen.py Step 5) is no
longer required; splits are generated in-memory from the full dataset.

Processing flow
---------------
For the sample size selected by INDEX:
  1. Load full X = full_<size>_cor.npy and y = full_<size>_y.npy from pwr_data/.
  2. Build the estimator via plugin.build_estimator(args).
     The plugin returns a Pipeline or GridSearchCV — always unfitted.
  3. Build outer_cv = RepeatedKFold(n_splits=k_outer, n_repeats=n_outer).
  4. Call cross_validate(estimator, X, y, cv=outer_cv, scoring=SCORING, …).
     sklearn handles all fold generation, scaler fitting, model fitting,
     and metric computation internally.
  5. Write per-fold scores, summary statistics, and a provenance stamp.

Cross-validation topology
--------------------------
Outer CV (model evaluation):
    RepeatedKFold(n_splits=--k_outer, n_repeats=--n_outer)
    Total outer folds = k_outer × n_outer

Inner CV (hyperparameter tuning, if the plugin uses GridSearchCV):
    Configured by the plugin's build_estimator(); typically KFold(k_inner).
    cross_validate() triggers the inner CV automatically on each outer fold.

Scoring
-------
SCORING dict maps metric names to sklearn scorer strings or make_scorer()
objects.  All three metrics are collected for both train and test folds:
    RMSE  — neg_root_mean_squared_error (sign-flipped; cv.py restores sign)
    MAE   — neg_mean_absolute_error    (sign-flipped; cv.py restores sign)
    R2    — coefficient of determination

To add Pearson r or Spearman ρ, uncomment the block below SCORING and add
the scorer objects to the dict.

Outputs (written to WRKDIR/pwr_data/)
--------------------------------------
  cv_results_size<N>_<model>.csv   — per-fold train+test scores (one row per fold)
  cv_summary_size<N>_<model>.csv   — mean ± SD across all folds
  cv_stamp_size<N>_<model>.txt     — provenance: timestamp, args, host
  cv_estimators_size<N>_<model>/   — (--save_estimators) joblib'd fitted estimators

SLURM array indexing
--------------------
INDEX (= SLURM_ARRAY_TASK_ID) selects one sample size from the sorted list of
sizes present in pwr_data/.  The array size is therefore NUMFILES (number of
sample sizes), not NUMFILES × KFOLDS as in the legacy design.

Exit codes
----------
  0   Success, or non-fatal skip (out-of-range INDEX, existing output)
  1   Fatal configuration error (bad model name, missing pwr_data/)
  2   Fatal data error (NaN/Inf in feature matrix or target)
"""

import argparse
import datetime as dt
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import RepeatedKFold, cross_validate

# ── Plugin registry bootstrap ─────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import models  # noqa: E402 — triggers __init__.py, auto-imports all plugins
from models.base import get_model_class, list_models  # noqa: E402

VERSION = "cv.py v2-sklearn 2026-04-29"

# ---------------------------------------------------------------------------
# Scoring dictionary
# ---------------------------------------------------------------------------
# All metrics are collected for both the training fold and the test fold by
# cross_validate(return_train_score=True).  sklearn prefixes negated metrics
# with "neg_"; the sign is flipped back to positive in _restore_sign() below.
#
# To add Pearson r or Spearman ρ:
#
#   from scipy.stats import pearsonr, spearmanr
#
#   def _pearson(y_true, y_pred):
#       r, _ = pearsonr(y_true, y_pred)
#       return float(r)
#
#   def _spearman(y_true, y_pred):
#       rho, _ = spearmanr(y_true, y_pred)
#       return float(rho)
#
#   score_pearson  = metrics.make_scorer(_pearson,  greater_is_better=True)
#   score_spearman = metrics.make_scorer(_spearman, greater_is_better=True)
#
#   SCORING["r"]   = score_pearson
#   SCORING["Rho"] = score_spearman

SCORING = {
    "RMSE": "neg_root_mean_squared_error",
    "MAE":  "neg_mean_absolute_error",
    "R2":   "r2",
}

# Metric names whose values are negated by sklearn and need sign restoration
_NEG_METRICS = {"RMSE", "MAE"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_data_sizes(pwr_dir: Path):
    """
    Scan *pwr_dir* and return a sorted list of integer sample sizes for which
    both ``full_<size>_cor.npy`` and ``full_<size>_y.npy`` are present.

    Only sizes where both files exist are returned; a partial pair (e.g. FCs
    written but y missing) is excluded to prevent loading errors downstream.

    Parameters
    ----------
    pwr_dir : Path
        Directory to scan (typically WRKDIR/pwr_data/).

    Returns
    -------
    list of int
        Sorted list of valid sample sizes.  Empty if none found.
    """
    fc_pat = re.compile(r"^full_(\d+)_cor\.npy$")
    y_pat  = re.compile(r"^full_(\d+)_y\.npy$")
    fc_sizes: set = set()
    y_sizes:  set = set()

    for p in pwr_dir.iterdir():
        m = fc_pat.match(p.name)
        if m:
            fc_sizes.add(int(m.group(1)))
            continue
        m = y_pat.match(p.name)
        if m:
            y_sizes.add(int(m.group(1)))

    return sorted(fc_sizes & y_sizes)


def _restore_sign(cv_res: pd.DataFrame) -> pd.DataFrame:
    """
    Flip the sign of negated metric columns so reported values are positive.

    sklearn's cross_validate() negates loss metrics (RMSE, MAE) so that
    higher is always better internally.  This restores them to their natural
    (positive) scale for human-readable output and CSV files.

    Affected columns match the pattern ``(test|train)_<metric>`` where
    ``<metric>`` is in ``_NEG_METRICS`` (RMSE, MAE).

    Parameters
    ----------
    cv_res : pd.DataFrame
        DataFrame of raw cross_validate() scores.

    Returns
    -------
    pd.DataFrame
        Copy with negated columns restored to positive values.
    """
    df = cv_res.copy()
    for col in df.columns:
        for m in _NEG_METRICS:
            if col.endswith(f"_{m}"):
                df[col] = df[col].abs()
    return df


def _print_per_fold_hyperparams(estimators) -> None:
    """
    Print the chosen hyperparameter(s) for each outer fold.

    Works for both GridSearchCV estimators (``best_params_``) and Pipeline
    estimators whose last step is a built-in CV model (e.g. ``RidgeCV``,
    ``LassoCV`` expose ``alpha_``; ``ElasticNetCV`` also exposes ``l1_ratio_``).

    Parameters
    ----------
    estimators : list
        ``scores['estimator']`` from cross_validate().
    """
    for i, est in enumerate(estimators):
        prefix = f"[INFO] fold {i + 1:>2}"
        if hasattr(est, "best_params_"):
            # GridSearchCV: print the winning parameter set
            print(f"{prefix} best_params={est.best_params_}  "
                  f"best_score={est.best_score_:.4f}")
        elif hasattr(est, "named_steps"):
            # Pipeline: inspect the final step for CV-selected attributes
            last = list(est.named_steps.values())[-1]
            parts = []
            if hasattr(last, "alpha_"):
                parts.append(f"alpha={last.alpha_:.4g}")
            if hasattr(last, "l1_ratio_"):
                parts.append(f"l1_ratio={last.l1_ratio_:.4g}")
            if parts:
                print(f"{prefix} chosen {', '.join(parts)}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_base_parser(add_model_args: bool = False) -> argparse.ArgumentParser:
    """
    Build the base argument parser, optionally enabling --help.

    Called in a two-pass strategy:
      Pass 1 (add_model_args=False): parse --model_file only; suppress --help
          and silently ignore unknown args (plugin flags not yet registered).
      Pass 2 (add_model_args=True):  full parser with --help enabled; plugin
          flags are registered via model_cls.cli_args(ap) before parsing.

    Parameters
    ----------
    add_model_args : bool
        If True, enable --help and allow the caller to register plugin flags.

    Returns
    -------
    argparse.ArgumentParser
        Base parser (without model-specific flags; those are added by the caller).
    """
    ap = argparse.ArgumentParser(
        description=(
            "Sklearn-based CV runner for the power calculator pipeline. "
            "Select the model with --model_file <stem>."
        ),
        add_help=add_model_args,
    )

    # Positional args kept for shell-script compatibility with legacy cv.sh
    ap.add_argument("WRKDIR",   help="Root working directory.")
    ap.add_argument("FILEDIR",  help="Pipeline scripts directory (accepted for compatibility; unused).")
    ap.add_argument("NUMFILES", type=int, help="Total number of sample sizes (informational).")
    ap.add_argument("INDEX",    type=int, help="1-based task index (= SLURM_ARRAY_TASK_ID).")

    ap.add_argument(
        "--model_file",
        type=str,
        default="ridge",
        help=(
            f"Model plugin to use — file stem inside models/. "
            f"Available: {list_models()}. "
            "Default: ridge."
        ),
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: WRKDIR/pwr_data/).",
    )

    # ── CV topology ──────────────────────────────────────────────────────────
    ap.add_argument(
        "--k_outer",
        type=int,
        default=10,
        help=(
            "Number of outer CV folds per repeat (k in RepeatedKFold). "
            "(default: 10)"
        ),
    )
    ap.add_argument(
        "--n_outer",
        type=int,
        default=2,
        help=(
            "Number of outer CV repeats (n in RepeatedKFold). "
            "Total outer folds = k_outer × n_outer.  (default: 2)"
        ),
    )
    ap.add_argument(
        "--random_state",
        type=int,
        default=123456,
        help=(
            "Random seed for RepeatedKFold fold generation. "
            "Set to the same value across runs for reproducibility.  (default: 123456)"
        ),
    )

    # ── Runtime ──────────────────────────────────────────────────────────────
    ap.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help=(
            "Parallel jobs for cross_validate's outer loop (default: 1). "
            "Set to -1 to use all available cores.  "
            "Caution: if the plugin's inner GridSearchCV also uses n_jobs=-1, "
            "nested parallelism can over-subscribe CPUs on shared nodes."
        ),
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing cv_results_*.csv output. "
            "Default: skip the task if the file already exists."
        ),
    )
    ap.add_argument(
        "--save_estimators",
        action="store_true",
        help=(
            "Persist each fold's fitted estimator to disk as a joblib file. "
            "Enables post-hoc Haufe-transform, residual analysis, and prediction "
            "on new data without re-fitting.  Requires the joblib package.  (default: off)"
        ),
    )
    ap.add_argument(
        "--debug_dump",
        action="store_true",
        help=(
            "On fatal data errors, write a DEBUG_size<N>.npz with shape/quality "
            "diagnostics for post-mortem inspection.  (default: off)"
        ),
    )

    return ap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Two-pass argument parsing, data loading, cross-validation, and result writing.

    Two-pass parse strategy
    -----------------------
    Pass 1: parse --model_file only (ignore unknown args) so the correct
            plugin class can be loaded before its CLI flags are registered.
    Pass 2: rebuild the full parser with model-specific flags added via
            model_cls.cli_args(ap), then parse the complete argv.
    Unknown args after Pass 2 are logged and ignored (they come from other
    models' flags forwarded via --export in PWR.sh).

    Returns
    -------
    int
        Exit code: 0 success, 1 config error, 2 data error.
    """

    # ── Pass 1: identify model plugin ────────────────────────────────────────
    pass1    = build_base_parser(add_model_args=False)
    known, _ = pass1.parse_known_args()
    model_name = known.model_file

    try:
        model_cls = get_model_class(model_name)
    except ValueError as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        return 1

    # ── Pass 2: full parse with plugin flags ──────────────────────────────────
    ap = build_base_parser(add_model_args=True)
    model_cls.cli_args(ap)
    args, unknown = ap.parse_known_args()
    if unknown:
        # Unknown args come from other models' flags forwarded via --export;
        # log them for traceability but do not fail.
        print(f"[INFO] Ignoring unrecognised args (other model flags): {unknown}")

    # ── Startup banner ────────────────────────────────────────────────────────
    print(f"[INFO] {VERSION}")
    print(f"[INFO] model={args.model_file}")
    print(f"[INFO] k_outer={args.k_outer}  n_outer={args.n_outer}  "
          f"total_folds={args.k_outer * args.n_outer}  random_state={args.random_state}")
    print(f"[INFO] script_path={os.path.abspath(__file__)}")
    print(f"[INFO] cwd={os.getcwd()}")

    pwr_dir = Path(args.outdir) if args.outdir else Path(args.WRKDIR) / "pwr_data"
    print(f"[INFO] pwr_dir={pwr_dir}")

    if not pwr_dir.exists():
        print(f"[FATAL] Missing directory: {pwr_dir}", file=sys.stderr)
        return 1

    # ── Discover available sample sizes ───────────────────────────────────────
    sizes = find_data_sizes(pwr_dir)
    print(f"[INFO] Found sizes: {sizes}")

    if not sizes:
        print("[FATAL] No full_<size>_cor.npy / full_<size>_y.npy pairs found in pwr_dir.",
              file=sys.stderr)
        return 1

    if args.INDEX < 1 or args.INDEX > len(sizes):
        print(f"[WARN] INDEX={args.INDEX} out of range 1..{len(sizes)}; no-op exit 0")
        return 0

    size = sizes[args.INDEX - 1]
    print(f"[INFO] Selected size={size}")

    # ── Skip guard ────────────────────────────────────────────────────────────
    out_csv = pwr_dir / f"cv_results_size{size}_{model_name}.csv"
    if out_csv.exists() and not args.overwrite:
        print(f"[WARN] {out_csv.name} exists and --overwrite not set; skipping.")
        return 0

    # ── Load data ─────────────────────────────────────────────────────────────
    fc_path = pwr_dir / f"full_{size}_cor.npy"
    y_path  = pwr_dir / f"full_{size}_y.npy"

    print(f"[INFO] Loading {fc_path.name} ...")
    X = np.load(str(fc_path))
    print(f"[INFO] Loading {y_path.name} ...")
    y = np.ravel(np.load(str(y_path)))   # ensure 1-D

    print(f"[INFO] X shape={X.shape}  dtype={X.dtype}")
    print(f"[INFO] y shape={y.shape}  dtype={y.dtype}")

    # ── Data quality checks ───────────────────────────────────────────────────
    fc_finite = float(np.isfinite(X).mean())
    print(f"[INFO] X finite_frac={fc_finite:.6f}")
    if fc_finite < 0.999:
        msg = f"[FATAL] X (FCs) contains NaN/Inf (finite_frac={fc_finite:.4f})"
        print(msg, file=sys.stderr)
        if args.debug_dump:
            dbg = pwr_dir / f"DEBUG_size{size}.npz"
            np.savez(str(dbg), note=np.array(msg), X_shape=np.array(X.shape),
                     X_finite_frac=np.array(fc_finite))
            print(f"[WARN] wrote debug dump: {dbg.name}", file=sys.stderr)
        return 2

    if not np.all(np.isfinite(y)):
        msg = "[FATAL] y contains NaN/Inf"
        print(msg, file=sys.stderr)
        return 2

    # ── Build estimator ───────────────────────────────────────────────────────
    # build_estimator() returns an unfitted Pipeline or GridSearchCV.
    # cross_validate() will call .fit(X_train, y_train) on each outer fold.
    estimator = model_cls.build_estimator(args)
    print(f"[INFO] Estimator: {estimator!r}")

    # ── Outer CV ──────────────────────────────────────────────────────────────
    outer_cv = RepeatedKFold(
        n_splits=args.k_outer,
        n_repeats=args.n_outer,
        random_state=args.random_state,
    )

    print(
        f"[INFO] Running {args.n_outer}×{args.k_outer}-fold CV "
        f"({args.k_outer * args.n_outer} outer folds) "
        f"with n_jobs={args.n_jobs} ..."
    )

    scores = cross_validate(
        estimator,
        X,
        y,
        cv=outer_cv,
        scoring=SCORING,
        return_train_score=True,
        return_estimator=True,   # needed for per-fold hyperparameter logging
        verbose=2,
        n_jobs=args.n_jobs,
    )

    # ── Process and display results ───────────────────────────────────────────
    # Separate estimator list from numeric columns before building DataFrame
    estimators = scores.pop("estimator")
    cv_res = _restore_sign(pd.DataFrame(scores))

    summary = pd.DataFrame({"mean": cv_res.mean(), "sd": cv_res.std()})

    print("\n[RESULTS] Per-fold scores:")
    print(cv_res.to_string(index=True))
    print("\n[RESULTS] Mean ± SD across all folds:")
    print(summary.to_string())

    _print_per_fold_hyperparams(estimators)

    # ── Write outputs ─────────────────────────────────────────────────────────
    cv_res.to_csv(str(out_csv), index_label="fold")
    print(f"\n[OK] wrote {out_csv.name}")

    summary_csv = pwr_dir / f"cv_summary_size{size}_{model_name}.csv"
    summary.to_csv(str(summary_csv))
    print(f"[OK] wrote {summary_csv.name}")

    # Provenance stamp
    stamp = pwr_dir / f"cv_stamp_size{size}_{model_name}.txt"
    stamp.write_text("\n".join([
        f"timestamp={dt.datetime.now().isoformat(timespec='seconds')}",
        f"version={VERSION}",
        f"model={model_name}",
        f"size={size}",
        f"k_outer={args.k_outer}",
        f"n_outer={args.n_outer}",
        f"random_state={args.random_state}",
        f"n_jobs={args.n_jobs}",
        f"host={os.uname().nodename if hasattr(os, 'uname') else 'NA'}",
        f"cwd={os.getcwd()}",
        f"script={os.path.abspath(__file__)}",
    ]) + "\n")
    print(f"[OK] wrote {stamp.name}")

    # Optional: persist fitted estimators for Haufe-transform, residual analysis,
    # or prediction on held-out data without re-fitting.
    if args.save_estimators:
        try:
            import joblib
            est_dir = pwr_dir / f"cv_estimators_size{size}_{model_name}"
            est_dir.mkdir(exist_ok=True)
            for i, est in enumerate(estimators):
                joblib.dump(est, est_dir / f"fold_{i + 1:03d}.joblib")
            print(f"[OK] wrote {len(estimators)} estimators to {est_dir.name}/")
        except ImportError:
            print("[WARN] joblib not available; --save_estimators skipped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
