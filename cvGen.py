#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path
from typing import List

import numpy as np


RE_FULL_COV = re.compile(r"^full_(\d+)_cov\.npy$")
RE_FULL_COR = re.compile(r"^full_(\d+)_cor\.npy$")


def find_full_sizes(pwr_data_dir: Path) -> List[int]:
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

    return sorted(list(cov_sizes & cor_sizes))


def split_indices(n_rows: int, kfolds: int) -> List[np.ndarray]:
    if kfolds < 2:
        raise ValueError("KFOLDS must be >= 2")
    idx = np.arange(n_rows)
    return [idx[f::kfolds] for f in range(kfolds)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="cvGen: for one dataset size (selected by INDEX), write full_<size>_fold_<fold>_split.npz"
    )
    ap.add_argument("WRKDIR")
    ap.add_argument("FILEDIR")  # unused
    ap.add_argument("NUMFILES", type=int)
    ap.add_argument("KFOLDS", type=int)
    ap.add_argument("INDEX", type=int)  # 1-based SLURM_ARRAY_TASK_ID
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else (Path(args.WRKDIR) / "pwr_data")
    if not outdir.exists():
        print(f"[FATAL] pwr_data not found: {outdir}", file=sys.stderr)
        return 1

    sizes = find_full_sizes(outdir)
    print(f"[INFO] outdir={outdir}")
    print(f"[INFO] found_sizes={sizes}")
    print(f"[INFO] NUMFILES_arg={args.NUMFILES} KFOLDS={args.KFOLDS} INDEX={args.INDEX}")

    if args.INDEX < 1 or args.INDEX > len(sizes):
        print(f"[WARN] INDEX={args.INDEX} out of range 1..{len(sizes)}; no-op exit 0")
        return 0

    size = sizes[args.INDEX - 1]
    cov_path = outdir / f"full_{size}_cov.npy"
    cor_path = outdir / f"full_{size}_cor.npy"

    try:
        cov = np.load(str(cov_path))
        cor = np.load(str(cor_path))

        if cov.ndim != 2 or cor.ndim != 2:
            raise ValueError(f"Expected 2D arrays; got cov{cov.shape} cor{cor.shape}")
        if cov.shape != cor.shape:
            raise ValueError(f"Shape mismatch: cov{cov.shape} vs cor{cor.shape}")

        n_rows = cov.shape[0]
        kfolds = int(args.KFOLDS)
        if kfolds > n_rows:
            print(f"[WARN] KFOLDS={kfolds} > n_rows={n_rows}; clamping to {n_rows}")
            kfolds = n_rows

        folds = split_indices(n_rows, kfolds)

        for f, test_idx in enumerate(folds, start=1):
            train_mask = np.ones(n_rows, dtype=bool)
            train_mask[test_idx] = False
            train_idx = np.where(train_mask)[0]

            out_path = outdir / f"full_{size}_fold_{f}_split.npz"
            np.savez_compressed(
                str(out_path),
                size=np.array([size], dtype=int),
                fold=np.array([f], dtype=int),
                train_idx=train_idx.astype(int),
                test_idx=test_idx.astype(int),
                cov_train=cov[train_idx, :],
                cor_train=cor[train_idx, :],
                cov_test=cov[test_idx, :],
                cor_test=cor[test_idx, :],
            )

        print(f"[OK] INDEX={args.INDEX} size={size}: wrote {kfolds} split files")
        return 0

    except Exception as e:
        # keep pipeline alive, but log clearly
        print(f"[WARN] cvGen failed for INDEX={args.INDEX} size={size}", file=sys.stderr)
        print(f"[WARN] cov_path={cov_path}", file=sys.stderr)
        print(f"[WARN] cor_path={cor_path}", file=sys.stderr)
        print(f"[WARN] error={e}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
