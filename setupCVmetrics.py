#!/usr/bin/env python3
"""
setupCVmetrics.py — Step 6 of the power calculator pipeline.

Scans pwr_data/ and writes cv.pkl: a DataFrame that serves as the job
manifest for Step 7 (cv.py). Each row in cv.pkl identifies one model-
fitting task by its (data, fold) pair, which cv.py uses to locate the
corresponding full_<size>_fold_<k>_split.npz and run the model.

Two discovery modes are attempted in order:

    Pass 1 — fold/split mode (normal pipeline flow):
        Scans for *_fold_<k>_split.npz files produced by Step 5 (cvGen.py).
        Extracts (data, fold) from each filename via SPLIT_RE and
        infer_data_from_stem(). Writes cv.pkl and exits immediately if
        any split files are found.

    Pass 2 — index fallback (pre-cvGen outputs):
        Used when Step 5 has not run or its outputs are missing. Scans for
        dat_size_<size>_index_<k>_cov.npy + _cor.npy pairs produced by
        Step 2 (pwr_process_chunk_*.py). The "fold" column is populated
        with the replicate index rather than a true fold number. Downstream
        cv.py behaviour differs in this mode since split .npz files do not
        exist — cv.py must handle raw cov/cor files directly.

Output:
    pwr_data/cv.pkl — pickled pandas DataFrame with columns:
        data   (int): sample size
        fold   (int): fold number (Pass 1) or replicate index (Pass 2)
        metric (int): initialised to 0; updated by cv.py as folds complete

Raises:
    FileNotFoundError: If pwr_data/ does not exist.
    RuntimeError:      If neither fold/split files nor paired cov/cor files
                       are found (indicates upstream pipeline failure).
"""

import argparse
import os
import re
import warnings
import pickle
import pandas as pd


# ── Filename patterns ─────────────────────────────────────────────────────────

# Pass 1: matches any *_fold_<k>_split.npz file, regardless of prefix.
# Named groups: stem (everything before _fold_), fold (integer fold number).
# Handles both full_<size>_fold_<k>_split.npz and dat_size_<size>_fold_<k>_split.npz.
SPLIT_RE = re.compile(r"^(?P<stem>.+)_fold_(?P<fold>\d+)_split\.npz$")

# Regexes used by infer_data_from_stem to extract the sample size integer
# from the stem captured by SPLIT_RE.
DATA_FROM_FULL    = re.compile(r"(?:^|_)full_(\d+)(?:_|$)")      # matches "full_<N>"
DATA_FROM_DATSIZE = re.compile(r"(?:^|_)dat_size_(\d+)(?:_|$)")  # matches "dat_size_<N>"

# Pass 2 fallback: matches Step 2 per-chunk simulation output files.
COV_RE = re.compile(r"^dat_size_(\d+)_index_(\d+)_cov\.npy$")
COR_RE = re.compile(r"^dat_size_(\d+)_index_(\d+)_cor\.npy$")


def infer_data_from_stem(stem: str):
    """
    Extract the integer sample size from the stem portion of a split filename.

    Tries two patterns in order:
        1. DATA_FROM_FULL    — e.g. "full_100"    → 100
        2. DATA_FROM_DATSIZE — e.g. "dat_size_100" → 100

    Both patterns use non-capturing word-boundary anchors ((?:^|_) and (?:_|$))
    to avoid matching partial numbers embedded in longer tokens.

    Args:
        stem: The portion of the filename before "_fold_<k>_split.npz",
              as captured by SPLIT_RE's "stem" group.

    Returns:
        Integer sample size, or None if neither pattern matches.
        A None return causes the file to be skipped with a [DEBUG] warning.
    """
    m = DATA_FROM_FULL.search(stem)
    if m:
        return int(m.group(1))
    m = DATA_FROM_DATSIZE.search(stem)
    if m:
        return int(m.group(1))
    return None


def main():
    """
    Discover CV tasks from pwr_data/ and write cv.pkl.

    Attempts Pass 1 (fold/split files) first; falls back to Pass 2
    (raw cov/cor pairs) only if no split files are found.

    Arguments:
        WRKDIR   Root working directory; pwr_data/ is the default scan target.
        FILEDIR  Accepted for argument-order consistency with other pipeline
                 scripts; not used at runtime.

    Optional arguments:
        --outdir PATH   Override the scan/output directory (default: WRKDIR/pwr_data/).
                        Useful for testing without affecting the live pwr_data/.
        --debug         Print verbose file-listing and DataFrame diagnostics to
                        the SLURM .out log. Activates: directory contents (first 40
                        files), unmatched fold/split stems, and cv.pkl head().
    """
    parser = argparse.ArgumentParser(description="Setup CV metrics (Python, npy)")
    parser.add_argument("WRKDIR",   type=str, help="Working directory (contains pwr_data/)")
    parser.add_argument("FILEDIR",  type=str, help="Unused (kept for compatibility)")
    parser.add_argument("--outdir", default=None,
                        help="Override output directory (default: WRKDIR/pwr_data)")
    parser.add_argument("--debug",  action="store_true",
                        help="Verbose logging to SLURM .out")
    args = parser.parse_args()

    warnings.warn("Running setupCVmetrics")   # Mirrors the R original convention

    pwr_data_dir = args.outdir if args.outdir else os.path.join(args.WRKDIR, "pwr_data")
    if not os.path.isdir(pwr_data_dir):
        raise FileNotFoundError(f"pwr_data directory not found: {pwr_data_dir}")

    files = sorted(os.listdir(pwr_data_dir))   # Sorted for deterministic processing order

    if args.debug:
        print(f"[DEBUG] pwr_data_dir = {pwr_data_dir}")
        print(f"[DEBUG] n_files_in_dir = {len(files)}")
        print("[DEBUG] first 40 files:")
        for f in files[:40]:
            print("  ", f)

    # ── Pass 1: fold/split mode ───────────────────────────────────────────────
    # Preferred path: uses full_<size>_fold_<k>_split.npz files from Step 5.
    # Extracts (data, fold) from each matched filename and exits immediately
    # after writing cv.pkl — Pass 2 is only reached if this loop finds nothing.
    rows = []
    for fname in files:
        m = SPLIT_RE.match(fname)
        if not m:
            continue
        stem = m.group("stem")
        fold = int(m.group("fold"))
        data = infer_data_from_stem(stem)
        if data is None:
            # Filename matched the split pattern but sample size couldn't be
            # extracted — likely a non-standard naming convention. Skip and warn.
            if args.debug:
                print(f"[DEBUG] matched fold/split but couldn't infer data from: {fname}")
            continue
        rows.append((data, fold))

    if len(rows) > 0:
        cv = pd.DataFrame(rows, columns=["data", "fold"]).astype(int)
        cv["metric"] = 0   # Initialised to 0; cv.py updates this as folds complete
        out_path = os.path.join(pwr_data_dir, "cv.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(cv, f)
        print(f"[OK] Found fold/split files. Saved CV table with {len(cv)} rows")
        print(f"[OK] Output: {out_path}")
        if args.debug:
            print("[DEBUG] head(cv):")
            print(cv.head())
        return   # Early exit: Pass 2 is not needed

    # ── Pass 2: index-based fallback ─────────────────────────────────────────
    # Reached only when no fold/split files exist (e.g. Step 5 was skipped or
    # failed entirely). Scans for raw per-chunk simulation outputs from Step 2.
    #
    # Important schema difference: "fold" here is the replicate index, not a
    # true CV fold number. cv.py must account for this when loading data, since
    # the corresponding full_<size>_fold_<k>_split.npz files do not exist in
    # this mode — cv.py must read cov/cor directly.
    cov_keys = set()
    cor_keys = set()

    for fname in files:
        mc = COV_RE.match(fname)
        if mc:
            cov_keys.add((int(mc.group(1)), int(mc.group(2))))
            continue
        mr = COR_RE.match(fname)
        if mr:
            cor_keys.add((int(mr.group(1)), int(mr.group(2))))

    # Intersect cov and cor keys: only include rows where both files exist,
    # consistent with the same pattern used in combine_data.py and cvGen.py.
    paired = sorted(cov_keys & cor_keys)

    if len(paired) == 0:
        raise RuntimeError(
            f"[FATAL] No fold/split files found AND no paired dat_size_*_index_*_(cov|cor).npy files found in {pwr_data_dir}.\n"
            f"[HINT] This script expects either:\n"
            f"  - *fold split* files: *_fold_<n>_split.npz\n"
            f"  - OR *simulation outputs*: dat_size_<size>_index_<idx>_cov.npy + dat_size_<size>_index_<idx>_cor.npy\n"
        )

    cv = pd.DataFrame(paired, columns=["data", "fold"]).astype(int)
    cv["metric"] = 0   # Same schema as Pass 1 for downstream compatibility

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
