#!/usr/bin/env python3
"""
cv.py — PCA + Random Forest CV R²

For each split .npz:
  1. Load cor_train / cor_test  (X_train, X_test)
  2. Compute y_train = X_train @ ridge_vec
             y_test  = X_test  @ ridge_vec
  3. StandardScaler + PCA on X_train; transform X_test with same scaler/PCA
  4. Fit RandomForestRegressor on (X_train_pca, y_train)
  5. Predict y_hat = rf.predict(X_test_pca)
  6. Compute R² between y_test and y_hat -> save as data_<size>_fold_<fold>_cvr2.npy

All other behaviour (stamp file, overwrite guard, debug dump, CLI signature)
is identical to the original cv.py so cv.sh / PWR.sh need no changes.
"""
import argparse
import datetime as dt
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

RE_SPLIT = re.compile(r"^full_(\d+)_fold_(\d+)_split\.npz$")
VERSION = "cv.py v2026-01-01 PCA_RF_R2"


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


def write_stamp(pwr_dir: Path, size: int, fold: int, split_path: Path,
                args: argparse.Namespace) -> Path:
    stamp = pwr_dir / f"cv_stamp_size{size}_fold{fold}.txt"
    now = dt.datetime.now().isoformat(timespec="seconds")
    body = (
        f"timestamp={now}\n"
        f"version={VERSION}\n"
        f"script_path={os.path.abspath(__file__)}\n"
        f"cwd={os.getcwd()}\n"
        f"host={os.uname().nodename if hasattr(os, 'uname') else 'NA'}\n"
        f"pwr_dir={pwr_dir}\n"
        f"split_file={split_path.name}\n"
        f"INDEX={args.INDEX}\n"
        f"use={args.use}\n"
        f"ridge_arg={args.ridge}\n"
        f"n_components={args.n_components}\n"
        f"n_estimators={args.n_estimators}\n"
        f"overwrite={args.overwrite}\n"
        f"debug_dump={args.debug_dump}\n"
    )
    stamp.write_text(body)
    return stamp


def debug_dump(pwr_dir: Path, size: int, fold: int, split_path: Path,
               X: Optional[np.ndarray], ridge: Optional[np.ndarray],
               note: str) -> None:
    dbg = pwr_dir / f"DEBUG_size{size}_fold{fold}.npz"
    payload = {
        "note": np.array(note),
        "version": np.array(VERSION),
        "split_file": np.array(split_path.name),
    }
    if X is not None:
        payload["X_shape"] = np.array(X.shape)
        payload["X_finite_frac"] = np.array(finite_frac(X))
    if ridge is not None:
        payload["ridge_shape"] = np.array(ridge.shape)
        payload["ridge_finite_frac"] = np.array(finite_frac(ridge))
    np.savez(str(dbg), **payload)
    print(f"[WARN] wrote debug dump: {dbg.name}", file=sys.stderr)


def load_ridge(pwr_dir: Path, ridge_arg: Optional[str]) -> Tuple[Path, np.ndarray]:
    if ridge_arg:
        rp = Path(ridge_arg)
        if not rp.exists():
            raise FileNotFoundError(f"--ridge not found: {rp}")
        r = np.load(str(rp), allow_pickle=True)
        return rp, np.asarray(r).reshape(-1).astype(float, copy=False)

    rp = pwr_dir / "ridge.npy"
    if rp.exists():
        r = np.load(str(rp), allow_pickle=True)
        return rp, np.asarray(r).reshape(-1).astype(float, copy=False)

    cand = sorted(pwr_dir.glob("ridge*.npy"))
    if not cand:
        raise FileNotFoundError("No ridge.npy / ridge*.npy found in pwr_data")
    rp = cand[0]
    r = np.load(str(rp), allow_pickle=True)
    return rp, np.asarray(r).reshape(-1).astype(float, copy=False)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="CV R² via PCA + Random Forest (NPZ splits -> scalar cvr2.npy files)"
    )
    # Positional args — identical to original cv.py so cv.sh needs no changes
    ap.add_argument("WRKDIR")
    ap.add_argument("FILEDIR")             # unused, kept for compatibility
    ap.add_argument("NUMFILES", type=int)  # informational
    ap.add_argument("KFOLDS",   type=int)  # informational
    ap.add_argument("EPSILON",  type=float)  # kept for compatibility, not used in RF metric
    ap.add_argument("INDEX",    type=int)  # 1-based index into sorted split files

    # Optional flags
    ap.add_argument("--outdir",  default=None,
                    help="default: WRKDIR/pwr_data")
    ap.add_argument("--ridge",   default=None,
                    help="default: outdir/ridge.npy or first ridge*.npy")
    ap.add_argument("--use",     choices=["cov_train", "cor_train"], default="cor_train",
                    help="Training matrix key in split .npz (default: cor_train)")
    ap.add_argument("--use_test", choices=["cov_test", "cor_test"], default="cor_test",
                    help="Test matrix key in split .npz (default: cor_test)")
    ap.add_argument("--n_components", type=int, default=500,
                    help="Number of PCA components (default: 500)")
    ap.add_argument("--n_estimators", type=int, default=500,
                    help="Number of RF trees (default: 500)")
    ap.add_argument("--n_jobs",  type=int, default=-1,
                    help="Parallel jobs for RF, -1 = all cores (default: -1)")
    ap.add_argument("--overwrite",   action="store_true",
                    help="Overwrite existing cvr2.npy (default: do not overwrite)")
    ap.add_argument("--debug_dump",  action="store_true",
                    help="Write DEBUG npz on failure")
    args = ap.parse_args()

    print(f"[INFO] {VERSION}")
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

    stamp_path = write_stamp(pwr_dir, size, fold, split_path, args)
    print(f"[INFO] wrote stamp: {stamp_path.name}")

    # ------------------------------------------------------------------
    # Load ridge.npy
    # ------------------------------------------------------------------
    try:
        ridge_path, ridge_vec = load_ridge(pwr_dir, args.ridge)
    except Exception as e:
        msg = f"[FATAL] ridge load failed: {e}"
        print(msg, file=sys.stderr)
        if args.debug_dump:
            debug_dump(pwr_dir, size, fold, split_path, None, None, msg)
        return 3

    print(f"[INFO] ridge file={ridge_path.name} shape={ridge_vec.shape} "
          f"finite_frac={finite_frac(ridge_vec):.6f}")

    # ------------------------------------------------------------------
    # Load split .npz — X_train and X_test
    # ------------------------------------------------------------------
    z = np.load(str(split_path), allow_pickle=True)
    print(f"[INFO] split keys={list(z.keys())}")

    train_key = args.use
    test_key  = args.use_test

    for key in (train_key, test_key):
        if key not in z:
            msg = f"[FATAL] split missing '{key}'"
            print(msg, file=sys.stderr)
            if args.debug_dump:
                debug_dump(pwr_dir, size, fold, split_path, None, ridge_vec, msg)
            return 2

    X_train = np.asarray(z[train_key], dtype=np.float64)
    X_test  = np.asarray(z[test_key],  dtype=np.float64)

    print(f"[INFO] {train_key} shape={X_train.shape} finite_frac={finite_frac(X_train):.6f}")
    print(f"[INFO] {test_key}  shape={X_test.shape}  finite_frac={finite_frac(X_test):.6f}")

    if finite_frac(X_train) < 0.999:
        msg = f"[FATAL] {train_key} contains NaN/Inf"
        print(msg, file=sys.stderr)
        if args.debug_dump:
            debug_dump(pwr_dir, size, fold, split_path, X_train, ridge_vec, msg)
        return 2

    if finite_frac(X_test) < 0.999:
        msg = f"[FATAL] {test_key} contains NaN/Inf"
        print(msg, file=sys.stderr)
        if args.debug_dump:
            debug_dump(pwr_dir, size, fold, split_path, X_test, ridge_vec, msg)
        return 2

    if ridge_vec.shape[0] != X_train.shape[1]:
        msg = (f"[FATAL] ridge length {ridge_vec.shape[0]} "
               f"!= n_edges {X_train.shape[1]}")
        print(msg, file=sys.stderr)
        if args.debug_dump:
            debug_dump(pwr_dir, size, fold, split_path, X_train, ridge_vec, msg)
        return 3

    # ------------------------------------------------------------------
    # 1. Compute y via dot product of ridge_vec with each observation row
    # ------------------------------------------------------------------
    y_train = X_train @ ridge_vec   # (n_train,)
    y_test  = X_test  @ ridge_vec   # (n_test,)

    y_min, y_max = minmax_finite(y_train)
    print(f"[INFO] y_train shape={y_train.shape} range=[{y_min:.4g}, {y_max:.4g}]")
    y_min, y_max = minmax_finite(y_test)
    print(f"[INFO] y_test  shape={y_test.shape}  range=[{y_min:.4g}, {y_max:.4g}]")

    # ------------------------------------------------------------------
    # 2. StandardScaler + PCA on X_train; apply same transform to X_test
    # ------------------------------------------------------------------
    n_components = min(args.n_components, X_train.shape[0], X_train.shape[1])
    print(f"[INFO] Running StandardScaler + PCA (n_components={n_components}) ...")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca  = pca.transform(X_test_sc)

    var_explained = float(np.cumsum(pca.explained_variance_ratio_)[-1]) * 100
    print(f"[INFO] PCA variance explained: {var_explained:.1f}%")

    # ------------------------------------------------------------------
    # 3. Fit Random Forest on PCA-reduced training data
    # ------------------------------------------------------------------
    print(f"[INFO] Fitting RandomForestRegressor "
          f"(n_estimators={args.n_estimators}, n_jobs={args.n_jobs}) ...")

    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        random_state=42,
        oob_score=True,
    )
    rf.fit(X_train_pca, y_train)
    print(f"[INFO] OOB R²: {rf.oob_score_:.4f}")

    # ------------------------------------------------------------------
    # 4. Predict on test PCs and compute R²
    # ------------------------------------------------------------------
    y_hat  = rf.predict(X_test_pca)
    r2     = r2_score(y_test, y_hat)

    print(f"[INFO] y_hat  range=[{float(y_hat.min()):.4g}, {float(y_hat.max()):.4g}]")
    print(f"[INFO] Test R² (cvr2) = {r2:.6f}")

    # ------------------------------------------------------------------
    # 5. Save cvr2.npy  (same filename convention as original)
    # ------------------------------------------------------------------
    out_metric = pwr_dir / f"data_{size}_fold_{fold}_cvr2.npy"

    if out_metric.exists() and not args.overwrite:
        print(f"[WARN] {out_metric.name} exists and --overwrite not set; NOT overwriting.")
        return 0

    if not np.isfinite(r2):
        msg = f"[FATAL] refusing to write non-finite R² ({r2})"
        print(msg, file=sys.stderr)
        if args.debug_dump:
            debug_dump(pwr_dir, size, fold, split_path, X_test, ridge_vec, msg)
        return 5

    np.save(str(out_metric), np.array(r2, dtype=float))
    print(f"[OK] wrote {out_metric.name}  value={r2:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
