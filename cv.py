#!/usr/bin/env python3
"""
cv.py — Step 7 of the power calculator pipeline. Modular cross-validation runner.

Fits and evaluates one predictive model on one (size, fold) combination,
selected by a 1-based INDEX (= SLURM_ARRAY_TASK_ID). Called once per array
task by cv.sh; the full array covers all NUMFILES × KFOLDS combinations.

The model is selected at runtime via --model_file <stem>, where <stem> is
the filename (without .py) of a plugin inside the models/ directory. This
makes cv.py model-agnostic: adding a new model requires no changes here.

Built-in models:
    random_forest      Random Forest         (original behaviour)
    ridge              Ridge Regression
    lasso              Lasso Regression
    elastic_net        ElasticNet
    svr                Support Vector Regression
    neural_network     MLP Regressor
    gradient_boosting  Gradient Boosting

Adding a new model:
    1. Create  models/<name>.py
    2. Subclass CVModel from models.base
    3. Decorate with @register("<name>")
    4. Pass --model_file <name> to cv.py

Outputs per (size, fold) task (written to WRKDIR/pwr_data/):
    data_<size>_fold_<fold>_cvr2.npy   — scalar test R² (the power metric)
    y_hat_size<size>_fold<fold>.npy    — predicted phenotype on the test set
    cv_stamp_size<size>_fold<fold>.txt — provenance: timestamp, model, args,
                                         host, model-specific diagnostics

Exit codes:
    0   Success, or non-fatal out-of-range INDEX, or skip (file exists, no --overwrite)
    1   Fatal configuration error (bad model name, missing pwr_data/)
    2   Fatal data error (missing split key, NaN/Inf in feature matrix)
    5   Non-finite R² — refused to write to avoid corrupting downstream results
"""

import argparse
import datetime as dt
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ── Plugin registry bootstrap ─────────────────────────────────────────────────
# Ensure the directory containing cv.py is on sys.path so `import models`
# works regardless of the caller's cwd (e.g. when invoked from pwr_data/).
# This must happen before argparse so plugins can register their CLI flags.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import models  # noqa: E402  — triggers __init__.py which auto-imports all plugins
from models.base import get_model_class, list_models  # noqa: E402

# Matches output filenames from cvGen.py: full_<size>_fold_<k>_split.npz
RE_SPLIT = re.compile(r"^full_(\d+)_fold_(\d+)_split\.npz$")

VERSION = "cv.py v2026-04-16 modular"


# ── Helpers ───────────────────────────────────────────────────────────────────

def list_splits(pwr_dir: Path) -> List[Tuple[int, int, Path]]:
    """
    Scan pwr_dir for split .npz files and return a sorted list of
    (size, fold, path) tuples.

    Sorting by (size, fold) ensures that INDEX → (size, fold) mapping is
    deterministic and consistent across all array tasks in the same job,
    regardless of filesystem directory entry ordering.

    Args:
        pwr_dir: Directory to scan (typically WRKDIR/pwr_data/).

    Returns:
        List of (dataset_size, fold_number, Path) tuples, sorted ascending
        by (size, fold). Empty list if no matching files are found.
    """
    out: List[Tuple[int, int, Path]] = []
    for p in pwr_dir.iterdir():
        if not p.is_file():
            continue
        m = RE_SPLIT.match(p.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), p))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def finite_frac(x: np.ndarray) -> float:
    """
    Return the fraction of finite (non-NaN, non-Inf) elements in x.

    Used as a data quality check before fitting: values below ~0.999
    indicate NaN/Inf contamination in the feature matrix, which would
    silently produce unreliable model fits.

    Args:
        x: Array of any shape.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 for empty arrays.
    """
    x = np.asarray(x)
    return float(np.isfinite(x).mean()) if x.size else 0.0


def minmax_finite(x: np.ndarray):
    """
    Return (min, max) of the finite elements of x.

    Used in diagnostic log lines to show the range of yt_train/yt_test
    without crashing when the array contains NaN or Inf values.

    Args:
        x: Array of any shape.

    Returns:
        Tuple (float, float) of (min, max) over finite elements.
        Returns (nan, nan) if no finite elements exist.
    """
    x = np.asarray(x)
    f = np.isfinite(x)
    if f.sum() == 0:
        return (np.nan, np.nan)
    xf = x[f]
    return (float(np.min(xf)), float(np.max(xf)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the coefficient of determination (R²) between true and predicted values.

    R² = 1 - SS_res / SS_tot, where SS_res = Σ(y - ŷ)² and SS_tot = Σ(y - ȳ)².
    This is the primary power metric written to data_<size>_fold_<fold>_cvr2.npy.

    Returns nan (rather than raising) when SS_tot == 0, which occurs when all
    true values are identical (zero-variance target). A non-finite result
    triggers exit code 5 in main() to prevent writing NaN to the output file.

    Args:
        y_true: 1-D array of ground-truth phenotype values.
        y_pred: 1-D array of model predictions, same length as y_true.

    Returns:
        Scalar float R². May be negative (worse than predicting the mean),
        zero, positive up to 1.0, or nan.
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def write_stamp(
    pwr_dir: Path,
    size: int,
    fold: int,
    split_path: Path,
    args: argparse.Namespace,
    extra: dict,
) -> Path:
    """
    Write a provenance stamp file for this (size, fold) task.

    Records runtime context (timestamp, host, cwd, version, model choice,
    key flags) and model-specific diagnostics from model.extra_info().
    Useful for post-hoc debugging of unexpected R² values without needing
    to re-run the job.

    Args:
        pwr_dir:    Output directory.
        size:       Dataset sample size for this task.
        fold:       CV fold number for this task.
        split_path: Path to the .npz split file used.
        args:       Parsed argument namespace (for model_file, INDEX, flags).
        extra:      Dict of model-specific key/value pairs from model.extra_info().

    Returns:
        Path to the written stamp file
        (cv_stamp_size<size>_fold<fold>.txt).
    """
    stamp = pwr_dir / f"cv_stamp_size{size}_fold{fold}.txt"
    now   = dt.datetime.now().isoformat(timespec="seconds")
    lines = [
        f"timestamp={now}",
        f"version={VERSION}",
        f"model_file={args.model_file}",
        f"script_path={os.path.abspath(__file__)}",
        f"cwd={os.getcwd()}",
        f"host={os.uname().nodename if hasattr(os, 'uname') else 'NA'}",
        f"pwr_dir={pwr_dir}",
        f"split_file={split_path.name}",
        f"INDEX={args.INDEX}",
        f"use={args.use}",
        f"overwrite={args.overwrite}",
        f"debug_dump={args.debug_dump}",
    ]
    # Prefix model-specific entries with "model_" to namespace them from
    # the core fields above and make them easy to grep in bulk stamp files.
    for k, v in extra.items():
        lines.append(f"model_{k}={v}")
    stamp.write_text("\n".join(lines) + "\n")
    return stamp


def debug_dump(
    pwr_dir: Path,
    size: int,
    fold: int,
    split_path: Path,
    X,
    note: str,
) -> None:
    """
    Write a minimal diagnostic .npz when a fatal data error is detected.

    Called just before returning a non-zero exit code so that the feature
    matrix shape and finite fraction are preserved for post-mortem inspection
    without needing to reload the full split file. Only written when
    --debug_dump is set to avoid cluttering pwr_data/ in normal runs.

    Args:
        pwr_dir:    Output directory.
        size:       Dataset sample size for this task.
        fold:       CV fold number for this task.
        split_path: Path to the .npz split file that triggered the error.
        X:          Feature matrix at time of failure, or None if unavailable.
        note:       The [FATAL] error message string, stored verbatim.
    """
    dbg     = pwr_dir / f"DEBUG_size{size}_fold{fold}.npz"
    payload = {
        "note":       np.array(note),
        "version":    np.array(VERSION),
        "split_file": np.array(split_path.name),
    }
    if X is not None:
        payload["X_shape"]       = np.array(X.shape)
        payload["X_finite_frac"] = np.array(finite_frac(X))
    np.savez(str(dbg), **payload)
    print(f"[WARN] wrote debug dump: {dbg.name}", file=sys.stderr)


# ── Argument parser ───────────────────────────────────────────────────────────

def build_base_parser(add_model_args: bool = False) -> argparse.ArgumentParser:
    """
    Build the base argument parser, optionally with help enabled.

    Called twice in a two-pass strategy:
        Pass 1 (add_model_args=False): parse --model_file only, suppress
            help and ignore unknown args so plugin-specific flags don't
            cause errors before the plugin is loaded.
        Pass 2 (add_model_args=True):  full parser with help enabled;
            the chosen model's flags are added via model_cls.cli_args(ap)
            before this parser is used to parse the full argv.

    Args:
        add_model_args: If True, enable --help and allow the caller to
                        add model-specific flags after construction.

    Returns:
        Configured ArgumentParser (without model-specific flags; those
        are added by the caller via model_cls.cli_args(ap) in Pass 2).
    """
    ap = argparse.ArgumentParser(
        description=(
            "Modular CV R² runner. "
            "Select the model with --model_file <stem>."
        ),
        add_help=add_model_args,   # Suppressed in Pass 1 to avoid premature --help exit
    )

    # Positional arguments — identical to original cv.py for shell script compatibility
    ap.add_argument("WRKDIR")
    ap.add_argument("FILEDIR")              # Accepted for compatibility; not used at runtime
    ap.add_argument("NUMFILES", type=int)   # Informational; actual split list derived by scanning
    ap.add_argument("KFOLDS",   type=int)   # Informational; actual fold count from split filenames
    ap.add_argument("EPSILON",  type=float) # Passed through for stamp provenance; noise applied upstream
    ap.add_argument("INDEX",    type=int)   # 1-based index into sorted split file list

    ap.add_argument(
        "--model_file",
        type=str,
        default="random_forest",
        help=(
            "Model plugin to use — file stem inside models/ directory. "
            f"Available: {list_models()}. "
            "Default: random_forest (original behaviour)."
        ),
    )
    ap.add_argument(
        "--outdir", default=None,
        help="Output directory (default: WRKDIR/pwr_data)",
    )
    ap.add_argument(
        "--use",
        choices=["cov_train", "cor_train"],
        default="cor_train",
        help=(
            "Training matrix key to load from split .npz. "
            "cor_train (default) uses Fisher-Z correlation edges; "
            "cov_train uses raw covariance edges."
        ),
    )
    ap.add_argument(
        "--use_test",
        choices=["cov_test", "cor_test"],
        default="cor_test",
        help="Test matrix key in split .npz (default: cor_test). Should match --use.",
    )
    ap.add_argument(
        "--overwrite",  action="store_true",
        help="Overwrite existing cvr2.npy output. Default: skip if file exists.",
    )
    ap.add_argument(
        "--debug_dump", action="store_true",
        help="Write DEBUG_size<N>_fold<K>.npz on fatal data errors for post-mortem inspection.",
    )

    return ap


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    """
    Two-pass argument parsing, data loading, model fitting, and result writing.

    Two-pass parse strategy:
        Pass 1: parse --model_file only (ignore unknown args) so the correct
                plugin class can be loaded before its CLI flags are registered.
        Pass 2: rebuild the full parser with model-specific flags added via
                model_cls.cli_args(ap), then parse the complete argv.
        Unknown args after Pass 2 are logged and ignored (they come from
        other models' flags forwarded via --export in PWR.sh).

    Processing flow:
        1. Resolve pwr_dir and scan for split .npz files via list_splits().
        2. Map INDEX (1-based) to the corresponding (size, fold, path) entry.
        3. Load X_train, X_test, yt_train, yt_test from the split .npz.
        4. Validate data quality: check required keys and finite fraction.
        5. Instantiate model_cls(args), call fit() and predict().
        6. Compute R² via r2_score().
        7. Write stamp, y_hat, and cvr2.npy.

    Returns:
        0  Success, or non-fatal skip (out-of-range INDEX, existing file).
        1  Fatal configuration error (bad model name, missing pwr_data/).
        2  Fatal data error (missing split key, NaN/Inf in feature matrix).
        5  Non-finite R² — refused to write to avoid corrupting final_data.py.
    """

    # ── Pass 1: identify model so we can register its CLI flags ──────────────
    pass1      = build_base_parser(add_model_args=False)
    known, _   = pass1.parse_known_args()
    model_name = known.model_file

    try:
        model_cls = get_model_class(model_name)
    except ValueError as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        return 1

    # ── Pass 2: full parse with model-specific flags added ───────────────────
    ap = build_base_parser(add_model_args=True)
    model_cls.cli_args(ap)                     # Plugin registers its own flags here
    args, unknown = ap.parse_known_args()
    if unknown:
        # Unknown args are other models' hyperparameters forwarded via --export;
        # logging them aids debugging without failing the job.
        print(f"[INFO] Ignoring unrecognised args (other model flags): {unknown}")

    # ── Startup banner ────────────────────────────────────────────────────────
    print(f"[INFO] {VERSION}")
    print(f"[INFO] model_file={args.model_file}")
    print(f"[INFO] script_path={os.path.abspath(__file__)}")
    print(f"[INFO] cwd={os.getcwd()}")

    pwr_dir = Path(args.outdir) if args.outdir else (Path(args.WRKDIR) / "pwr_data")
    print(f"[INFO] pwr_dir={pwr_dir}")

    if not pwr_dir.exists():
        print(f"[FATAL] Missing directory: {pwr_dir}", file=sys.stderr)
        return 1

    # ── Discover split files ──────────────────────────────────────────────────
    # list_splits() sorts by (size, fold) so the INDEX → (size, fold) mapping
    # is identical across all array tasks in the same job submission.
    splits = list_splits(pwr_dir)
    print(f"[INFO] Found {len(splits)} split .npz files")

    if not splits:
        print("[FATAL] No split files found.", file=sys.stderr)
        return 1

    # Out-of-range INDEX exits 0 (non-fatal) so the array job does not fail;
    # the Step 8 guard catches missing cvr2.npy files after all tasks complete.
    if args.INDEX < 1 or args.INDEX > len(splits):
        print(f"[WARN] INDEX={args.INDEX} out of range 1..{len(splits)}; no-op exit 0")
        return 0

    size, fold, split_path = splits[args.INDEX - 1]   # Convert 1-based INDEX to 0-based
    print(f"[INFO] Using split: size={size} fold={fold} file={split_path.name}")

    # ── Load split .npz ───────────────────────────────────────────────────────
    z = np.load(str(split_path), allow_pickle=True)
    print(f"[INFO] split keys={list(z.keys())}")

    train_key = args.use       # "cor_train" (default) or "cov_train"
    test_key  = args.use_test  # "cor_test"  (default) or "cov_test"

    # Validate required keys before loading arrays to produce a clear error
    # message rather than a cryptic KeyError inside np.asarray().
    for key in (train_key, test_key, "yt_train", "yt_test"):
        if key not in z:
            msg = f"[FATAL] split missing '{key}'"
            print(msg, file=sys.stderr)
            if args.debug_dump:
                debug_dump(pwr_dir, size, fold, split_path, None, msg)
            return 2

    X_train  = np.asarray(z[train_key],  dtype=np.float64)
    X_test   = np.asarray(z[test_key],   dtype=np.float64)
    yt_train = np.asarray(z["yt_train"], dtype=np.float64)
    yt_test  = np.asarray(z["yt_test"],  dtype=np.float64)

    # Diagnostic log: shape and data quality for both train and test sets
    print(f"[INFO] {train_key} shape={X_train.shape} finite_frac={finite_frac(X_train):.6f}")
    print(f"[INFO] {test_key}  shape={X_test.shape}  finite_frac={finite_frac(X_test):.6f}")
    y_min, y_max = minmax_finite(yt_train)
    print(f"[INFO] yt_train shape={yt_train.shape} range=[{y_min:.4g}, {y_max:.4g}]")
    y_min, y_max = minmax_finite(yt_test)
    print(f"[INFO] yt_test  shape={yt_test.shape}  range=[{y_min:.4g}, {y_max:.4g}]")

    # Reject matrices with significant NaN/Inf contamination before fitting.
    # Threshold < 0.999 catches any non-trivial contamination while allowing
    # for floating-point edge cases in very large matrices.
    for label, arr in ((train_key, X_train), (test_key, X_test)):
        if finite_frac(arr) < 0.999:
            msg = f"[FATAL] {label} contains NaN/Inf"
            print(msg, file=sys.stderr)
            if args.debug_dump:
                debug_dump(pwr_dir, size, fold, split_path, arr, msg)
            return 2

    # ── Model fitting and evaluation ──────────────────────────────────────────
    model = model_cls(args)           # Plugin instantiated with full parsed args
    model.fit(X_train, yt_train)      # Train on the fold's training set
    y_hat = model.predict(X_test)     # Predict on the held-out test set
    r2    = r2_score(yt_test, y_hat)  # Primary power metric

    print(f"[INFO] y_hat  range=[{float(y_hat.min()):.4g}, {float(y_hat.max()):.4g}]")
    print(f"[INFO] Test R² (cvr2) = {r2:.6f}")

    # ── Write provenance stamp ────────────────────────────────────────────────
    extra      = model.extra_info()   # Model-specific diagnostics (e.g. n_components used)
    stamp_path = write_stamp(pwr_dir, size, fold, split_path, args, extra)
    print(f"[INFO] wrote stamp: {stamp_path.name}")

    # Save raw predictions for optional downstream use (e.g. residual analysis)
    np.save(str(pwr_dir / f"y_hat_size{size}_fold{fold}.npy"), y_hat)
    print(f"[INFO] wrote y_hat_size{size}_fold{fold}.npy")

    # ── Write cvr2.npy ────────────────────────────────────────────────────────
    out_metric = pwr_dir / f"data_{size}_fold_{fold}_cvr2.npy"

    # Skip without error if output already exists and --overwrite not set.
    # This allows safe re-submission of failed array tasks without duplicating
    # work for tasks that already succeeded.
    if out_metric.exists() and not args.overwrite:
        print(f"[WARN] {out_metric.name} exists and --overwrite not set; NOT overwriting.")
        return 0

    # Refuse to write non-finite R² to avoid silently corrupting final_data.py's
    # aggregation. Exit code 5 distinguishes this from data errors (2) so it
    # can be grep'd separately in SLURM logs.
    if not np.isfinite(r2):
        msg = f"[FATAL] refusing to write non-finite R² ({r2})"
        print(msg, file=sys.stderr)
        if args.debug_dump:
            debug_dump(pwr_dir, size, fold, split_path, X_test, msg)
        return 5

    np.save(str(out_metric), np.array(r2, dtype=float))
    print(f"[OK] wrote {out_metric.name}  value={r2:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
