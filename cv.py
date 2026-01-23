#!/usr/bin/env python3
"""
cv.py (diagnostic-heavy) — Python 3.8/3.9 compatible

Key features:
- Prints script path + VERSION
- Writes a per-task stamp file into pwr_data
- Prints EPSILON, ridge stats, keep counts, and cov_test stats
- Computes metric 3 ways (eps-threshold, top-frac, and all edges)
- Refuses to overwrite existing cvr2 unless --overwrite
- Optionally writes a DEBUG npz snapshot if anything is non-finite
"""
import argparse
import datetime as dt
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

RE_SPLIT = re.compile(r"^full_(\d+)_fold_(\d+)_split\.npz$")
VERSION = "cv.py v2025-12-22e DIAG_TRIPLE_METRIC_STAMP_PY39"


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


def quantile_finite(x: np.ndarray, q: float) -> float:
    x = np.asarray(x)
    f = np.isfinite(x)
    if f.sum() == 0:
        return float("nan")
    return float(np.quantile(x[f], q))


def write_stamp(pwr_dir: Path, size: int, fold: int, split_path: Path, args: argparse.Namespace) -> Path:
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
        f"EPSILON={args.EPSILON}\n"
        f"fallback_top_frac={args.fallback_top_frac}\n"
        f"use={args.use}\n"
        f"ridge_arg={args.ridge}\n"
        f"overwrite={args.overwrite}\n"
        f"debug_dump={args.debug_dump}\n"
    )
    stamp.write_text(body)
    return stamp


def debug_dump(
    pwr_dir: Path,
    size: int,
    fold: int,
    split_path: Path,
    X: Optional[np.ndarray],
    ridge: Optional[np.ndarray],
    keep: Optional[np.ndarray],
    note: str
) -> None:
    dbg = pwr_dir / f"DEBUG_size{size}_fold{fold}.npz"
    payload = {
        "note": np.array(note),
        "version": np.array(VERSION),
        "script_path": np.array(os.path.abspath(__file__)),
        "split_file": np.array(split_path.name),
    }
    if X is not None:
        n_edge = X.shape[1] if X.ndim == 2 else 0
        payload["X_shape"] = np.array(X.shape)
        payload["X_head"] = X[:5, :min(50, n_edge)] if X.ndim == 2 else X.reshape(-1)[:200]
        payload["X_finite_frac"] = np.array(finite_frac(X))
    if ridge is not None:
        payload["ridge_shape"] = np.array(ridge.shape)
        payload["ridge_head"] = ridge[:min(200, ridge.size)]
        payload["ridge_finite_frac"] = np.array(finite_frac(ridge))
    if keep is not None:
        payload["keep_sum"] = np.array(int(np.sum(keep)))
        payload["keep_head"] = keep[:min(500, keep.size)]
    np.savez(str(dbg), **payload)
    print(f"[WARN] wrote debug dump: {dbg.name}", file=sys.stderr)


def load_ridge(pwr_dir: Path, ridge_arg: Optional[str]) -> Tuple[Path, np.ndarray]:
    if ridge_arg:
        rp = Path(ridge_arg)
        if not rp.exists():
            raise FileNotFoundError(f"--ridge not found: {rp}")
        r = np.load(str(rp), allow_pickle=True)
        r = np.asarray(r).reshape(-1).astype(float, copy=False)
        return rp, r

    rp = pwr_dir / "ridge.npy"
    if rp.exists():
        r = np.load(str(rp), allow_pickle=True)
        r = np.asarray(r).reshape(-1).astype(float, copy=False)
        return rp, r

    cand = sorted(pwr_dir.glob("ridge*.npy"))
    if not cand:
        raise FileNotFoundError("No ridge.npy / ridge*.npy found in pwr_data")
    rp = cand[0]
    r = np.load(str(rp), allow_pickle=True)
    r = np.asarray(r).reshape(-1).astype(float, copy=False)
    return rp, r


def compute_metric(X: np.ndarray, keep: np.ndarray) -> float:
    """Mean over subjects of mean(|X|) across kept edges."""
    if keep is None or int(np.sum(keep)) == 0:
        return float("nan")
    sub = X[:, keep]
    per_sub = np.mean(np.abs(sub), axis=1)
    return float(np.mean(per_sub))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="CV metrics with heavy diagnostics (NPZ splits -> scalar cvr2.npy files)"
    )
    ap.add_argument("WRKDIR")
    ap.add_argument("FILEDIR")            # unused
    ap.add_argument("NUMFILES", type=int) # informational
    ap.add_argument("KFOLDS", type=int)   # informational
    ap.add_argument("EPSILON", type=float)
    ap.add_argument("INDEX", type=int)    # 1-based index into sorted split files
    ap.add_argument("--outdir", default=None, help="default: WRKDIR/pwr_data")
    ap.add_argument("--ridge", default=None, help="default: outdir/ridge.npy or first ridge*.npy")
    ap.add_argument("--fallback_top_frac", type=float, default=0.01,
                    help="top fraction of |ridge| for alternate metric (default 0.01=top1%)")
    ap.add_argument("--use", choices=["cov_test", "cor_test"], default="cov_test",
                    help="use cov_test or cor_test from split (default: cov_test)")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing data_*_cvr2.npy (default: do not overwrite)")
    ap.add_argument("--debug_dump", action="store_true",
                    help="write DEBUG_size*_fold*.npz when something looks wrong")
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

    z = np.load(str(split_path), allow_pickle=True)
    keys = list(z.keys())
    print(f"[INFO] split keys={keys}")

    if args.use not in z:
        msg = f"[FATAL] split missing {args.use}"
        print(msg, file=sys.stderr)
        if args.debug_dump:
            debug_dump(pwr_dir, size, fold, split_path, None, None, None, msg)
        return 2

    X = np.asarray(z[args.use])
    if X.ndim != 2:
        msg = f"[FATAL] {args.use} not 2D: shape={X.shape}"
        print(msg, file=sys.stderr)
        if args.debug_dump:
            debug_dump(pwr_dir, size, fold, split_path, X, None, None, msg)
        return 2

    X_ff = finite_frac(X)
    X_min, X_max = minmax_finite(X)
    print(f"[INFO] {args.use} shape={X.shape} dtype={X.dtype} finite_frac={X_ff:.6f} min={X_min:.6g} max={X_max:.6g}")

    if X_ff < 0.999:
        msg = f"[FATAL] {args.use} contains NaN/Inf (finite_frac={X_ff:.6f})"
        print(msg, file=sys.stderr)
        if args.debug_dump:
            debug_dump(pwr_dir, size, fold, split_path, X, None, None, msg)
        return 2

    try:
        ridge_path, ridge = load_ridge(pwr_dir, args.ridge)
    except Exception as e:
        msg = f"[FATAL] ridge load failed: {e}"
        print(msg, file=sys.stderr)
        if args.debug_dump:
            debug_dump(pwr_dir, size, fold, split_path, X, None, None, msg)
        return 3

    absridge = np.abs(ridge)
    r_ff = finite_frac(ridge)
    rmin, rmax = minmax_finite(ridge)
    armin, armax = minmax_finite(absridge)
    q50 = quantile_finite(absridge, 0.50)
    q90 = quantile_finite(absridge, 0.90)
    q95 = quantile_finite(absridge, 0.95)
    q99 = quantile_finite(absridge, 0.99)

    print(f"[INFO] ridge file={ridge_path.name} shape={ridge.shape} finite_frac={r_ff:.6f}")
    print(f"[INFO] ridge min/max: {rmin:.6g} {rmax:.6g} |ridge| min/max: {armin:.6g} {armax:.6g}")
    print(f"[INFO] |ridge| quantiles: q50={q50:.6g} q90={q90:.6g} q95={q95:.6g} q99={q99:.6g}")

    if ridge.shape[0] != X.shape[1]:
        msg = f"[FATAL] ridge length {ridge.shape[0]} != n_edge {X.shape[1]}"
        print(msg, file=sys.stderr)
        if args.debug_dump:
            debug_dump(pwr_dir, size, fold, split_path, X, ridge, None, msg)
        return 3

    eps = float(args.EPSILON)
    keep_eps = np.isfinite(absridge) & (absridge >= eps)
    k_eps = int(np.sum(keep_eps))

    top_frac = float(args.fallback_top_frac)
    if not (0.0 < top_frac <= 1.0):
        msg = f"[FATAL] fallback_top_frac must be in (0,1], got {top_frac}"
        print(msg, file=sys.stderr)
        return 3

    thr_top = np.quantile(absridge[np.isfinite(absridge)], 1.0 - top_frac)
    keep_top = np.isfinite(absridge) & (absridge >= thr_top)
    k_top = int(np.sum(keep_top))

    keep_all = np.isfinite(absridge)
    k_all = int(np.sum(keep_all))

    print(f"[INFO] EPSILON={eps:.6g} keep_eps={k_eps}/{X.shape[1]}")
    print(f"[INFO] top_frac={top_frac} thr_top={thr_top:.6g} keep_top={k_top}/{X.shape[1]}")
    print(f"[INFO] keep_all={k_all}/{X.shape[1]}")

    m_eps = compute_metric(X, keep_eps)
    m_top = compute_metric(X, keep_top)
    m_all = compute_metric(X, keep_all)

    print(f"[INFO] metric_eps={m_eps} (finite={np.isfinite(m_eps)})")
    print(f"[INFO] metric_top={m_top} (finite={np.isfinite(m_top)})")
    print(f"[INFO] metric_all={m_all} (finite={np.isfinite(m_all)})")

    if np.isfinite(m_eps):
        chosen, chosen_name = m_eps, "eps"
    elif np.isfinite(m_top):
        chosen, chosen_name = m_top, "top"
    elif np.isfinite(m_all):
        chosen, chosen_name = m_all, "all"
    else:
        msg = "[FATAL] all three metrics are non-finite (unexpected given finite splits)"
        print(msg, file=sys.stderr)
        if args.debug_dump:
            debug_dump(pwr_dir, size, fold, split_path, X, ridge, keep_all, msg)
        return 4

    print(f"[INFO] chosen_metric={chosen} chosen_name={chosen_name}")

    out_metric = pwr_dir / f"data_{size}_fold_{fold}_cvr2.npy"
    if out_metric.exists() and not args.overwrite:
        print(f"[WARN] {out_metric.name} exists and --overwrite not set; NOT overwriting.")
        print("[WARN] This often means you are looking at stale NaN outputs from a prior run.")
        return 0

    if not np.isfinite(chosen):
        msg = f"[FATAL] refusing to write NaN chosen metric (chosen_name={chosen_name})"
        print(msg, file=sys.stderr)
        if args.debug_dump:
            debug_dump(pwr_dir, size, fold, split_path, X, ridge, keep_eps, msg)
        return 5

    np.save(str(out_metric), np.array(chosen, dtype=float))
    print(f"[OK] wrote {out_metric.name} value={chosen} (from {chosen_name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

