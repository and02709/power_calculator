#!/usr/bin/env python3
"""
cv.py — Modular cross-validation runner.

Selects the ML model at runtime via --model_file <stem>, where <stem>
is the filename (without .py) of a plugin inside the models/ directory.

Built-in models
---------------
  random_forest    PCA + Random Forest  (original behaviour)
  ridge            PCA + Ridge Regression
  lasso            PCA + Lasso Regression
  elastic_net      PCA + ElasticNet
  svr              PCA + Support Vector Regression
  neural_network   PCA + MLP Regressor
  gradient_boosting PCA + Gradient Boosting

Adding a new model
------------------
  1. Create  models/<name>.py
  2. Subclass CVModel from models.base
  3. Decorate with @register("<name>")
  4. Pass --model_file <name>  to cv.py

All other CLI arguments (WRKDIR, FILEDIR, INDEX, etc.) remain identical
to the original cv.py so existing shell scripts (cv.sh, PWR.sh) need
only one new flag: --model_file.
"""

import argparse
import datetime as dt
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Load the plugin registry.  Must happen before argparse so that each
# plugin can add its own CLI flags via a two-pass parse strategy.
# ---------------------------------------------------------------------------
# Ensure the directory containing cv.py is on sys.path so `import models`
# works regardless of the caller's cwd.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import models  # noqa: E402  (triggers auto-import of all plugins)
from models.base import get_model_class, list_models  # noqa: E402

RE_SPLIT = re.compile(r"^full_(\d+)_fold_(\d+)_split\.npz$")
VERSION = "cv.py v2026-04-16 modular"


# ---------------------------------------------------------------------------
# Helpers (unchanged from original)
# ---------------------------------------------------------------------------

def list_splits(pwr_dir: Path) -> List[Tuple[int, int, Path]]:
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
    x = np.asarray(x)
    return float(np.isfinite(x).mean()) if x.size else 0.0


def minmax_finite(x: np.ndarray):
    x = np.asarray(x)
    f = np.isfinite(x)
    if f.sum() == 0:
        return (np.nan, np.nan)
    xf = x[f]
    return (float(np.min(xf)), float(np.max(xf)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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
    stamp = pwr_dir / f"cv_stamp_size{size}_fold{fold}.txt"
    now = dt.datetime.now().isoformat(timespec="seconds")
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
    dbg = pwr_dir / f"DEBUG_size{size}_fold{fold}.npz"
    payload = {
        "note": np.array(note),
        "version": np.array(VERSION),
        "split_file": np.array(split_path.name),
    }
    if X is not None:
        payload["X_shape"] = np.array(X.shape)
        payload["X_finite_frac"] = np.array(finite_frac(X))
    np.savez(str(dbg), **payload)
    print(f"[WARN] wrote debug dump: {dbg.name}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Two-pass argparse:
#   Pass 1 — parse --model_file only (ignore unknown args)
#   Pass 2 — build a full parser that includes the chosen model's flags
# ---------------------------------------------------------------------------

def build_base_parser(add_model_args: bool = False) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Modular CV R² runner. "
            "Select the model with --model_file <stem>."
        ),
        # Don't exit on unknown args in pass-1
        add_help=add_model_args,
    )

    # ------------------------------------------------------------------
    # Positional args — identical to original cv.py
    # ------------------------------------------------------------------
    ap.add_argument("WRKDIR")
    ap.add_argument("FILEDIR")             # unused, kept for compatibility
    ap.add_argument("NUMFILES", type=int)  # informational
    ap.add_argument("KFOLDS",   type=int)  # informational
    ap.add_argument("EPSILON",  type=float)
    ap.add_argument("INDEX",    type=int)  # 1-based index into sorted split files

    # ------------------------------------------------------------------
    # Core optional flags
    # ------------------------------------------------------------------
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
        help="Training matrix key in split .npz (default: cor_train)",
    )
    ap.add_argument(
        "--use_test",
        choices=["cov_test", "cor_test"],
        default="cor_test",
        help="Test matrix key in split .npz (default: cor_test)",
    )
    ap.add_argument("--overwrite",  action="store_true")
    ap.add_argument("--debug_dump", action="store_true")

    return ap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # --- Pass 1: identify the model so we can add its flags ---------------
    pass1 = build_base_parser(add_model_args=False)
    known, _ = pass1.parse_known_args()
    model_name = known.model_file

    try:
        model_cls = get_model_class(model_name)
    except ValueError as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        return 1

    # --- Pass 2: full parser with model-specific flags --------------------
    ap = build_base_parser(add_model_args=True)
    model_cls.cli_args(ap)
    args = ap.parse_args()

    # --- Startup banner ---------------------------------------------------
    print(f"[INFO] {VERSION}")
    print(f"[INFO] model_file={args.model_file}")
    print(f"[INFO] script_path={os.path.abspath(__file__)}")
    print(f"[INFO] cwd={os.getcwd()}")

    pwr_dir = Path(args.outdir) if args.outdir else (Path(args.WRKDIR) / "pwr_data")
    print(f"[INFO] pwr_dir={pwr_dir}")

    if not pwr_dir.exists():
        print(f"[FATAL] Missing directory: {pwr_dir}", file=sys.stderr)
        return 1

    splits = list_splits(pwr_dir)
    print(f"[INFO] Found {len(splits)} split .npz files")

    if not splits:
        print("[FATAL] No split files found.", file=sys.stderr)
        return 1

    if args.INDEX < 1 or args.INDEX > len(splits):
        print(f"[WARN] INDEX={args.INDEX} out of range 1..{len(splits)}; no-op exit 0")
        return 0

    size, fold, split_path = splits[args.INDEX - 1]
    print(f"[INFO] Using split: size={size} fold={fold} file={split_path.name}")

    # ------------------------------------------------------------------
    # Load split .npz
    # ------------------------------------------------------------------
    z = np.load(str(split_path), allow_pickle=True)
    print(f"[INFO] split keys={list(z.keys())}")

    train_key = args.use
    test_key  = args.use_test

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

    print(f"[INFO] {train_key} shape={X_train.shape} finite_frac={finite_frac(X_train):.6f}")
    print(f"[INFO] {test_key}  shape={X_test.shape}  finite_frac={finite_frac(X_test):.6f}")
    y_min, y_max = minmax_finite(yt_train)
    print(f"[INFO] yt_train shape={yt_train.shape} range=[{y_min:.4g}, {y_max:.4g}]")
    y_min, y_max = minmax_finite(yt_test)
    print(f"[INFO] yt_test  shape={yt_test.shape}  range=[{y_min:.4g}, {y_max:.4g}]")

    for label, arr in ((train_key, X_train), (test_key, X_test)):
        if finite_frac(arr) < 0.999:
            msg = f"[FATAL] {label} contains NaN/Inf"
            print(msg, file=sys.stderr)
            if args.debug_dump:
                debug_dump(pwr_dir, size, fold, split_path, arr, msg)
            return 2

    # ------------------------------------------------------------------
    # Instantiate model, fit, predict
    # ------------------------------------------------------------------
    model = model_cls(args)
    model.fit(X_train, yt_train)
    y_hat = model.predict(X_test)
    r2    = r2_score(yt_test, y_hat)

    print(f"[INFO] y_hat  range=[{float(y_hat.min()):.4g}, {float(y_hat.max()):.4g}]")
    print(f"[INFO] Test R² (cvr2) = {r2:.6f}")

    # ------------------------------------------------------------------
    # Write stamp (includes model-specific diagnostics)
    # ------------------------------------------------------------------
    extra = model.extra_info()
    stamp_path = write_stamp(pwr_dir, size, fold, split_path, args, extra)
    print(f"[INFO] wrote stamp: {stamp_path.name}")

    # Save y_hat
    np.save(str(pwr_dir / f"y_hat_size{size}_fold{fold}.npy"), y_hat)
    print(f"[INFO] wrote y_hat_size{size}_fold{fold}.npy")

    # ------------------------------------------------------------------
    # Save cvr2.npy
    # ------------------------------------------------------------------
    out_metric = pwr_dir / f"data_{size}_fold_{fold}_cvr2.npy"

    if out_metric.exists() and not args.overwrite:
        print(f"[WARN] {out_metric.name} exists and --overwrite not set; NOT overwriting.")
        return 0

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
