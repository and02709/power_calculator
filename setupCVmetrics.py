#!/usr/bin/env python3
import argparse
import os
import re
import warnings
import pickle
import pandas as pd


# Option A: explicit CV split files
# Examples:
#   full_100_fold_3_split.npy
#   dat_size_100_fold_3_split.npy
SPLIT_RE = re.compile(r"^(?P<stem>.+)_fold_(?P<fold>\d+)_split\.npy$")

DATA_FROM_FULL = re.compile(r"(?:^|_)full_(\d+)(?:_|$)")
DATA_FROM_DATSIZE = re.compile(r"(?:^|_)dat_size_(\d+)(?:_|$)")


def infer_data_from_stem(stem: str):
    m = DATA_FROM_FULL.search(stem)
    if m:
        return int(m.group(1))
    m = DATA_FROM_DATSIZE.search(stem)
    if m:
        return int(m.group(1))
    return None


# Option B: index-based simulation outputs (fallback)
# Examples:
#   dat_size_100_index_42_cov.npy
#   dat_size_100_index_42_cor.npy
COV_RE = re.compile(r"^dat_size_(\d+)_index_(\d+)_cov\.npy$")
COR_RE = re.compile(r"^dat_size_(\d+)_index_(\d+)_cor\.npy$")


def main():
    parser = argparse.ArgumentParser(description="Setup CV metrics (Python, npy)")
    parser.add_argument("WRKDIR", type=str, help="Working directory (contains pwr_data/)")
    parser.add_argument("FILEDIR", type=str, help="Unused (kept for compatibility)")
    parser.add_argument("--outdir", default=None, help="Override output directory (default: WRKDIR/pwr_data)")
    parser.add_argument("--debug", action="store_true", help="Verbose logging to SLURM .out")
    args = parser.parse_args()

    warnings.warn("Running setupCVmetrics")

    pwr_data_dir = args.outdir if args.outdir else os.path.join(args.WRKDIR, "pwr_data")
    if not os.path.isdir(pwr_data_dir):
        raise FileNotFoundError(f"pwr_data directory not found: {pwr_data_dir}")

    files = sorted(os.listdir(pwr_data_dir))

    if args.debug:
        print(f"[DEBUG] pwr_data_dir = {pwr_data_dir}")
        print(f"[DEBUG] n_files_in_dir = {len(files)}")
        print("[DEBUG] first 40 files:")
        for f in files[:40]:
            print("  ", f)

    # -----------------------
    # Pass 1: fold-split mode
    # -----------------------
    rows = []
    for fname in files:
        m = SPLIT_RE.match(fname)
        if not m:
            continue
        stem = m.group("stem")
        fold = int(m.group("fold"))
        data = infer_data_from_stem(stem)
        if data is None:
            if args.debug:
                print(f"[DEBUG] matched fold/split but couldn't infer data from: {fname}")
            continue
        rows.append((data, fold))

    if len(rows) > 0:
        cv = pd.DataFrame(rows, columns=["data", "fold"]).astype(int)
        cv["metric"] = 0
        out_path = os.path.join(pwr_data_dir, "cv.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(cv, f)
        print(f"[OK] Found fold/split files. Saved CV table with {len(cv)} rows")
        print(f"[OK] Output: {out_path}")
        if args.debug:
            print("[DEBUG] head(cv):")
            print(cv.head())
        return

    # -----------------------------------------
    # Pass 2: fallback to index-based cov/cor
    # -----------------------------------------
    cov_keys = set()
    cor_keys = set()

    for fname in files:
        mc = COV_RE.match(fname)
        if mc:
            size = int(mc.group(1))
            idx = int(mc.group(2))
            cov_keys.add((size, idx))
            continue
        mr = COR_RE.match(fname)
        if mr:
            size = int(mr.group(1))
            idx = int(mr.group(2))
            cor_keys.add((size, idx))

    paired = sorted(cov_keys & cor_keys)

    if len(paired) == 0:
        raise RuntimeError(
            f"[FATAL] No fold/split files found AND no paired dat_size_*_index_*_(cov|cor).npy files found in {pwr_data_dir}.\n"
            f"[HINT] This script expects either:\n"
            f"  - *fold split* files: *_fold_<n>_split.npy\n"
            f"  - OR *simulation outputs*: dat_size_<size>_index_<idx>_cov.npy + dat_size_<size>_index_<idx>_cor.npy\n"
        )

    # Here "fold" is not truly fold; it's the index. But it preserves downstream schema.
    cv = pd.DataFrame(paired, columns=["data", "fold"]).astype(int)
    cv["metric"] = 0

    out_path = os.path.join(pwr_data_dir, "cv.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(cv, f)

    print(f"[OK] No fold/split files found; fell back to dat_size/index pairs.")
    print(f"[OK] Saved CV table with {len(cv)} rows (fold=index)")
    print(f"[OK] Output: {out_path}")

    if args.debug:
        print("[DEBUG] head(cv):")
        print(cv.head())


if __name__ == "__main__":
    main()
