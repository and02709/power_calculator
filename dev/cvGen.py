#!/usr/bin/env python3
"""
cvGen.py — Step 5 of the power calculator pipeline.

Generates cross-validation train/test split files for one sample size,
selected by a 1-based INDEX (= SLURM_ARRAY_TASK_ID). Called once per
array task by cvGen.sh; the full array covers all NUMFILES sample sizes
in parallel.

For the selected sample size, reads the stacked matrices produced by
Step 3 (combine_data.py) and writes one .npz split file per fold:

    full_<size>_fold_<k>_split.npz   k = 1..KFOLDS

Each .npz contains:
    size       — scalar: the dataset sample size
    fold       — scalar: 1-based fold number
    train_idx  — 1-D int array: row indices for the training set
    test_idx   — 1-D int array: row indices for the test set
    cov_train  — 2-D float array: covariance rows for training subjects
    cor_train  — 2-D float array: correlation rows for training subjects
    cov_test   — 2-D float array: covariance rows for test subjects
    cor_test   — 2-D float array: correlation rows for test subjects
    yt_train   — 1-D float array: noisy phenotype for training subjects
    yt_test    — 1-D float array: noisy phenotype for test subjects

Fold assignment uses systematic (modular) sampling: subject i goes to
fold (i % KFOLDS) + 1. This distributes subjects evenly without shuffling,
preserving the original row order within each fold.

Error handling: exceptions during split generation are caught and logged
as [WARN] rather than propagating, so the array job exits 0 and the
pipeline continues. The PWR.sh guard (NUMFFILES check) catches missing
outputs after all tasks complete.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List

import numpy as np


# Compiled regexes for discovering available sample sizes from filenames.
# Group 1 = integer sample size in both cases.
RE_FULL_COV = re.compile(r"^full_(\d+)_cov\.npy$")
RE_FULL_COR = re.compile(r"^full_(\d+)_cor\.npy$")


def find_full_sizes(pwr_data_dir: Path) -> List[int]:
    """
    Scan pwr_data_dir and return sorted sample sizes that have both
    full_<size>_cov.npy and full_<size>_cor.npy present.

    Intersecting cov and cor sizes guards against partial Step 3 outputs
    where one file was written but the other was not. Only fully paired
    sizes are eligible for fold generation.

    Args:
        pwr_data_dir: Directory to scan (typically WRKDIR/pwr_data/).

    Returns:
        Sorted list of integer sample sizes with both cov and cor files.
        Returns an empty list if no paired files are found.
    """
    cov_sizes = set()
    cor_sizes = set()

    for p in pwr_data_dir.iterdir():
        if not p.is_file():
            continue
        m = RE_FULL_COV.match(p.name)
        if m:
            cov_sizes.add(int(m.group(1)))
            continue
        m = RE_FULL_COR.match(p.name)
        if m:
            cor_sizes.add(int(m.group(1)))

    # Intersect: only sizes where both files exist are returned
    return sorted(list(cov_sizes & cor_sizes))


def split_indices(n_rows: int, kfolds: int) -> List[np.ndarray]:
    """
    Partition row indices into KFOLDS non-overlapping test sets using
    systematic (modular) sampling.

    Assigns row i to fold (i % kfolds), so fold k receives indices
    [k, k+kfolds, k+2*kfolds, ...]. This distributes subjects evenly
    across folds without shuffling, and is deterministic given n_rows
    and kfolds — no random seed is needed.

    Example: n_rows=10, kfolds=3
        fold 0 (test): [0, 3, 6, 9]
        fold 1 (test): [1, 4, 7]
        fold 2 (test): [2, 5, 8]

    Args:
        n_rows:  Total number of subjects (rows in the stacked matrix).
        kfolds:  Number of folds. Must be >= 2.

    Returns:
        List of kfolds 1-D arrays, each containing the test row indices
        for one fold. Union of all arrays equals np.arange(n_rows).

    Raises:
        ValueError: If kfolds < 2.
    """
    if kfolds < 2:
        raise ValueError("KFOLDS must be >= 2")
    idx = np.arange(n_rows)
    # idx[f::kfolds] selects every kfolds-th element starting at offset f,
    # which is equivalent to assigning rows where (row_index % kfolds) == f
    return [idx[f::kfolds] for f in range(kfolds)]


def main() -> int:
    """
    Parse arguments, select the target sample size by INDEX, and write
    KFOLDS split .npz files for that size.

    Returns 0 on success or on non-fatal errors (missing files, shape
    mismatches). Returns 1 only for fatal configuration errors (missing
    pwr_data directory) that indicate a pipeline setup problem.

    Arguments:
        WRKDIR   Root working directory; pwr_data/ subdirectory is used
                 as the default input/output location.
        FILEDIR  Pipeline scripts directory. Accepted for argument-order
                 consistency with other pipeline scripts; not used here.
        NUMFILES Total number of sample sizes. Accepted by the parser for
                 consistency; the actual size list is derived by scanning
                 pwr_data/ via find_full_sizes() rather than trusting this
                 count, so a mismatch is non-fatal.
        KFOLDS   Number of CV folds to generate per sample size.
        INDEX    1-based task index (= SLURM_ARRAY_TASK_ID). Selects the
                 INDEX-th size from the sorted list returned by
                 find_full_sizes(). Out-of-range values exit 0 silently.

    Optional arguments:
        --outdir PATH   Override input/output directory (default: WRKDIR/pwr_data/).
        --debug         Accepted for future use; has no effect currently.
    """
    ap = argparse.ArgumentParser(
        description="cvGen: for one dataset size (selected by INDEX), write full_<size>_fold_<fold>_split.npz"
    )
    ap.add_argument("WRKDIR")
    ap.add_argument("FILEDIR")           # Accepted for argument-order consistency; not used
    ap.add_argument("NUMFILES", type=int)  # Accepted for consistency; size list derived by scanning
    ap.add_argument("KFOLDS",   type=int)
    ap.add_argument("INDEX",    type=int)  # 1-based SLURM_ARRAY_TASK_ID
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--debug",  action="store_true")  # Reserved for future diagnostic use
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else (Path(args.WRKDIR) / "pwr_data")
    if not outdir.exists():
        print(f"[FATAL] pwr_data not found: {outdir}", file=sys.stderr)
        return 1   # Fatal: pipeline cannot proceed without pwr_data/

    sizes = find_full_sizes(outdir)
    print(f"[INFO] outdir={outdir}")
    print(f"[INFO] found_sizes={sizes}")
    print(f"[INFO] NUMFILES_arg={args.NUMFILES} KFOLDS={args.KFOLDS} INDEX={args.INDEX}")

    # Out-of-range INDEX is a no-op rather than an error: it can occur if
    # NUMFILES was set larger than the number of sizes actually present in
    # pwr_data/ (e.g. some Step 3 outputs are missing). Exiting 0 lets the
    # rest of the array complete; the PWR.sh NUMFFILES guard catches the gap.
    if args.INDEX < 1 or args.INDEX > len(sizes):
        print(f"[WARN] INDEX={args.INDEX} out of range 1..{len(sizes)}; no-op exit 0")
        return 0

    # Convert 1-based INDEX to the corresponding sample size
    size     = sizes[args.INDEX - 1]
    cov_path = outdir / f"full_{size}_cov.npy"
    cor_path = outdir / f"full_{size}_cor.npy"
    yt_path  = outdir / f"full_{size}_yt.npy"   # Noisy phenotype from combine_data.py

    try:
        cov = np.load(str(cov_path))
        cor = np.load(str(cor_path))
        yt  = np.load(str(yt_path))

        # Shape validation: all three inputs must be consistent
        if cov.ndim != 2 or cor.ndim != 2:
            raise ValueError(f"Expected 2D arrays; got cov{cov.shape} cor{cor.shape}")
        if cov.shape != cor.shape:
            raise ValueError(f"Shape mismatch: cov{cov.shape} vs cor{cor.shape}")
        if yt.ndim != 1 or yt.shape[0] != cov.shape[0]:
            raise ValueError(f"Expected yt shape ({cov.shape[0]},); got {yt.shape}")

        n_rows = cov.shape[0]
        kfolds = int(args.KFOLDS)

        # Clamp KFOLDS to n_rows if the requested fold count exceeds the
        # number of subjects — otherwise split_indices would produce empty folds.
        if kfolds > n_rows:
            print(f"[WARN] KFOLDS={kfolds} > n_rows={n_rows}; clamping to {n_rows}")
            kfolds = n_rows

        folds = split_indices(n_rows, kfolds)

        for f, test_idx in enumerate(folds, start=1):
            # Derive train_idx as the complement of test_idx using a boolean mask.
            # np.where returns the indices where the mask is True.
            train_mask          = np.ones(n_rows, dtype=bool)
            train_mask[test_idx] = False
            train_idx           = np.where(train_mask)[0]

            out_path = outdir / f"full_{size}_fold_{f}_split.npz"
            # Store both the index arrays and the pre-sliced data matrices so
            # cv.py can load a single .npz and have everything it needs without
            # re-loading the full stacked matrices.
            np.savez_compressed(
                str(out_path),
                size      = np.array([size], dtype=int),
                fold      = np.array([f],    dtype=int),
                train_idx = train_idx.astype(int),
                test_idx  = test_idx.astype(int),
                cov_train = cov[train_idx, :],
                cor_train = cor[train_idx, :],
                cov_test  = cov[test_idx,  :],
                cor_test  = cor[test_idx,  :],
                yt_train  = yt[train_idx],
                yt_test   = yt[test_idx],
            )

        print(f"[OK] INDEX={args.INDEX} size={size}: wrote {kfolds} split files")
        return 0

    except Exception as e:
        # Non-fatal: log clearly and exit 0 so the array job does not fail.
        # The PWR.sh NUMFFILES guard (Step 6) will catch missing split files
        # after all tasks complete, providing a single consolidated failure
        # point rather than failing individual array tasks.
        print(f"[WARN] cvGen failed for INDEX={args.INDEX} size={size}", file=sys.stderr)
        print(f"[WARN] cov_path={cov_path}",  file=sys.stderr)
        print(f"[WARN] cor_path={cor_path}",  file=sys.stderr)
        print(f"[WARN] yt_path={yt_path}",    file=sys.stderr)
        print(f"[WARN] error={e}",            file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
