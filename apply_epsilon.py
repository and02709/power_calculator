#!/usr/bin/env python3
"""
apply_epsilon.py — regenerate the noisy phenotype for a new epsilon.

Why this exists
---------------
EPSILON only ever enters the pipeline at Step 3 (combine_data.py), where it
scales the noise added to the phenotype:

    y  = cor_mat @ ridge_vec                        # clean signal
    yt = y + N(0, (epsilon * sqrt(var(y)))^2)       # what cv.py actually fits

Nothing upstream of that depends on epsilon: the simulated per-subject
matrices, and the full_<size>_{cov,cor}.npy matrices stacked from them, are
identical for every epsilon value. Re-running Steps 1-3 for each epsilon in a
sweep therefore repeats the most expensive part of the pipeline for no reason.

This script consumes the already-combined outputs of a previous run and writes
only the piece that epsilon changes:

    reads   full_<size>_y.npy    (or recomputes it from full_<size>_cor.npy)
    writes  full_<size>_yt.npy

cv.py reads full_<size>_cor.npy and full_<size>_yt.npy, so once this has run
the CV step proceeds exactly as it would after combine_data.py.

A second benefit: every epsilon in the sweep is then evaluated on the *same*
simulated subjects, so differences between power curves are attributable to
the phenotype SNR rather than to a fresh draw of simulated brains.

Inputs (in WRKDIR/pwr_data/, typically symlinked from a base run by PWR.sh
--reuse-from):
    full_<size>_cor.npy   required — used to infer the size ladder, and to
                          recompute y if full_<size>_y.npy is absent
    full_<size>_y.npy     optional — the clean phenotype; recomputed from
                          cor @ ridge_vec when missing
    ridge*.npy / haufe.csv  only needed for that recomputation

Outputs:
    full_<size>_yt.npy    one per sample size
    epsilon_stamp.txt     provenance: epsilon, seed, sizes, timestamp

Usage:
    python3 apply_epsilon.py <WRKDIR> <FILEDIR> <EPSILON> [--seed INT] [--outdir DIR]
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

RE_COR = re.compile(r"^full_(\d+)_cor\.npy$")


def discover_sizes(outdir: Path):
    """
    Return the sorted list of sample sizes with a full_<size>_cor.npy present.

    The cor matrices are the feature matrices cv.py fits on, so they define
    which sizes this run can produce results for.
    """
    sizes = []
    for f in sorted(outdir.glob("full_*_cor.npy")):
        m = RE_COR.match(f.name)
        if m:
            sizes.append(int(m.group(1)))
    return sorted(set(sizes))


def load_clean_y(outdir: Path, filedir: Path, size: int) -> np.ndarray:
    """
    Load the clean phenotype for one sample size.

    Prefers the cached full_<size>_y.npy written by combine_data.py. Falls
    back to recomputing y = cor_mat @ ridge_vec, which requires loading the
    (large) correlation matrix and the ridge weight vector, so the cached
    path is much cheaper when it exists.

    Args:
        outdir:  pwr_data/ directory holding the combined matrices.
        filedir: Pipeline scripts directory (source of haufe.csv fallback).
        size:    Sample size to load.

    Returns:
        1-D array of clean phenotype values, one per simulated subject.
    """
    y_path = outdir / f"full_{size}_y.npy"
    if y_path.exists():
        return np.ravel(np.load(y_path))

    # Fallback: recompute from the correlation matrix. load_ridge() is reused
    # from combine_data.py so the ridge source resolution order (ridge.npy in
    # pwr_data, then ridge*.npy, then FILEDIR/haufe.csv) stays identical.
    print(f"[INFO] size={size}: full_{size}_y.npy absent — recomputing y from cor @ ridge_vec")
    sys.path.insert(0, str(filedir))
    from combine_data import load_ridge  # noqa: E402  (deliberately late import)

    cor_mat   = np.load(outdir / f"full_{size}_cor.npy")
    ridge_vec = load_ridge(outdir, filedir)
    if ridge_vec.shape[0] != cor_mat.shape[1]:
        raise ValueError(
            f"size={size}: ridge length {ridge_vec.shape[0]} != n_edge {cor_mat.shape[1]}"
        )
    y = cor_mat @ ridge_vec
    np.save(y_path, y)   # Cache it so later epsilons in the sweep skip this path
    print(f"[OK] size={size}: wrote {y_path.name} {y.shape}")
    return y


def main():
    ap = argparse.ArgumentParser(
        description="Regenerate full_<size>_yt.npy for a new epsilon from existing combined data."
    )
    ap.add_argument("WRKDIR",  help="Working directory (contains pwr_data/)")
    ap.add_argument("FILEDIR", help="Pipeline scripts directory (haufe.csv fallback source)")
    ap.add_argument("EPSILON", type=float,
                    help="Noise scale factor: error ~ N(0, (epsilon * sqrt(var(y)))^2)")
    ap.add_argument("--outdir", default=None,
                    help="Override the pwr_data directory (default: WRKDIR/pwr_data)")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed. Streams are derived per sample size as [seed, size], "
                         "so results are reproducible and independent across sizes. "
                         "Omit for a nondeterministic draw.")
    args = ap.parse_args()

    outdir  = Path(args.outdir) if args.outdir else Path(args.WRKDIR) / "pwr_data"
    filedir = Path(args.FILEDIR)
    epsilon = args.EPSILON

    if epsilon < 0:
        raise ValueError(f"EPSILON must be >= 0 (got {epsilon})")
    if not outdir.exists():
        raise FileNotFoundError(f"pwr_data not found: {outdir}")

    sizes = discover_sizes(outdir)
    if not sizes:
        raise RuntimeError(
            f"[FATAL] No full_<size>_cor.npy files found in {outdir}. "
            f"Run Steps 1-3 first, or pass --reuse-from to PWR.sh."
        )

    print(f"[INFO] outdir  = {outdir}")
    print(f"[INFO] epsilon = {epsilon}")
    print(f"[INFO] seed    = {args.seed if args.seed is not None else '<random>'}")
    print(f"[INFO] sizes   = {sizes}")

    for size in sizes:
        y = load_clean_y(outdir, filedir, size)

        # Noise is scaled to the signal's own SD, so epsilon is an SNR control
        # independent of y's absolute scale: epsilon=0 leaves yt == y, epsilon=1
        # matches signal SD, epsilon>1 pushes SNR below 1. Matches combine_data.py.
        var_y     = np.var(y)
        noise_std = epsilon * np.sqrt(var_y)

        # Seeding with [seed, size] gives every sample size its own independent
        # stream while keeping the whole sweep reproducible from one seed.
        rng   = np.random.default_rng([args.seed, size] if args.seed is not None else None)
        error = rng.normal(loc=0.0, scale=noise_std, size=y.shape)
        yt    = y + error

        out_yt = outdir / f"full_{size}_yt.npy"
        np.save(out_yt, yt)
        print(f"[OK] size={size}: wrote {out_yt.name} {yt.shape}  "
              f"noise_std={noise_std:.4g}  range=[{yt.min():.4g}, {yt.max():.4g}]")

    # Provenance stamp — without it, a pwr_data directory of yt files gives no
    # indication of which epsilon produced them.
    stamp = outdir / "epsilon_stamp.txt"
    with open(stamp, "w") as fh:
        fh.write(f"timestamp={datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"epsilon={epsilon}\n")
        fh.write(f"seed={args.seed}\n")
        fh.write(f"sizes={sizes}\n")
    print(f"[INFO] wrote {stamp.name}")
    print("[DONE] apply_epsilon complete.")


if __name__ == "__main__":
    main()
