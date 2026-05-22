#!/usr/bin/env python3
"""
scripts/cv_reference.py — Standalone CV smoke-test and reference implementation.

Run locally against real or synthetic data to validate a model plugin before
submitting to the cluster.  This script mirrors the logic inside cv.py but
is self-contained and does not require a SLURM array setup.

Usage
-----
Against real data:
    python3 scripts/cv_reference.py \
        --model_file ridge \
        --data_dir   path/to/pwr_data \
        --size       2000 \
        --k_outer    10 \
        --n_outer    2

Against synthetic data (no files needed):
    python3 scripts/cv_reference.py --model_file ridge --synthetic

Nested CV example:
    python3 scripts/cv_reference.py --model_file ridge_nested --synthetic

Plugin development workflow
---------------------------
1. Create models/<name>.py and implement cli_args() / build_estimator().
2. Run this script with --model_file <name> --synthetic to verify correctness.
3. Check that per-fold metrics and hyperparameter logging look sensible.
4. Submit to the cluster via cv.sh / PWR.sh.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_validate

# ── Bootstrap: ensure models/ package is importable from scripts/ ─────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import models  # noqa: E402
from models.base import get_model_class, list_models  # noqa: E402

# Mirror the scoring dict from cv.py (kept in sync manually)
SCORING = {
    "RMSE": "neg_root_mean_squared_error",
    "MAE":  "neg_mean_absolute_error",
    "R2":   "r2",
}
_NEG_METRICS = {"RMSE", "MAE"}


def _restore_sign(cv_res: pd.DataFrame) -> pd.DataFrame:
    df = cv_res.copy()
    for col in df.columns:
        for m in _NEG_METRICS:
            if col.endswith(f"_{m}"):
                df[col] = df[col].abs()
    return df


def _print_hyperparams(estimators) -> None:
    for i, est in enumerate(estimators):
        prefix = f"  fold {i + 1:>2}"
        if hasattr(est, "best_params_"):
            print(f"{prefix}: best_params={est.best_params_}  "
                  f"best_score={est.best_score_:.4f}")
        elif hasattr(est, "named_steps"):
            last = list(est.named_steps.values())[-1]
            parts = []
            if hasattr(last, "alpha_"):
                parts.append(f"alpha={last.alpha_:.4g}")
            if hasattr(last, "l1_ratio_"):
                parts.append(f"l1_ratio={last.l1_ratio_:.4g}")
            if parts:
                print(f"{prefix}: chosen {', '.join(parts)}")


def main() -> int:
    # ── Pass 1: identify model ────────────────────────────────────────────────
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model_file", default="ridge")
    known, _ = pre.parse_known_args()

    try:
        model_cls = get_model_class(known.model_file)
    except ValueError as e:
        print(f"[FATAL] {e}")
        return 1

    # ── Pass 2: full parser with plugin flags ─────────────────────────────────
    ap = argparse.ArgumentParser(
        description="Standalone CV reference / smoke-test for power_calculator plugins.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model_file", default="ridge",
                    help=f"Model plugin (available: {list_models()}).")
    ap.add_argument("--data_dir",   default="data",
                    help="Directory containing FCs_<size>.npy and y_<size>.npy.")
    ap.add_argument("--size",       type=int, default=2000,
                    help="Sample size to load.")
    ap.add_argument("--synthetic",  action="store_true",
                    help="Generate synthetic data instead of loading from --data_dir.")
    ap.add_argument("--n_features", type=int, default=500,
                    help="Number of features for synthetic data (default: 500).")
    ap.add_argument("--k_outer",    type=int, default=10,
                    help="Outer CV folds.")
    ap.add_argument("--n_outer",    type=int, default=2,
                    help="Outer CV repeats.")
    ap.add_argument("--random_state", type=int, default=123456)
    ap.add_argument("--n_jobs",     type=int, default=1)
    model_cls.cli_args(ap)
    args = ap.parse_args()

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.synthetic:
        rng = np.random.default_rng(args.random_state)
        X   = rng.standard_normal((args.size, args.n_features))
        w   = rng.standard_normal(args.n_features)
        y   = X @ w + rng.standard_normal(args.size) * 0.5
        print(f"[INFO] Synthetic data: X={X.shape}  y={y.shape}")
    else:
        data_dir = Path(args.data_dir)
        X = np.load(str(data_dir / f"FCs_{args.size}.npy"))
        y = np.ravel(np.load(str(data_dir / f"y_{args.size}.npy")))
        print(f"[INFO] Loaded data: X={X.shape}  y={y.shape}")

    # ── Build estimator and run CV ────────────────────────────────────────────
    estimator = model_cls.build_estimator(args)
    print(f"[INFO] Estimator: {estimator!r}\n")

    outer_cv = RepeatedKFold(
        n_splits=args.k_outer,
        n_repeats=args.n_outer,
        random_state=args.random_state,
    )

    print(f"Running {args.n_outer}x{args.k_outer}-fold CV "
          f"({args.k_outer * args.n_outer} total outer folds) ...")

    scores = cross_validate(
        estimator, X, y,
        cv=outer_cv,
        scoring=SCORING,
        return_train_score=True,
        return_estimator=True,
        verbose=2,
        n_jobs=args.n_jobs,
    )

    estimators = scores.pop("estimator")
    cv_res  = _restore_sign(pd.DataFrame(scores))
    summary = pd.DataFrame({"mean": cv_res.mean(), "sd": cv_res.std()})

    print("\nPer-fold results:")
    print(cv_res.to_string(index=True))
    print("\nMean +/- SD across folds:")
    print(summary.to_string())

    print("\nPer-fold hyperparameters:")
    _print_hyperparams(estimators)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
