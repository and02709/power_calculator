#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


RE_COV = re.compile(r"^dat_size_(\d+)_index_(\d+)_cov\.npy$")
RE_COR = re.compile(r"^dat_size_(\d+)_index_(\d+)_cor\.npy$")


def parse_cov_name(p: Path):
    m = RE_COV.match(p.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_cor_name(p: Path):
    m = RE_COR.match(p.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def main():
    ap = argparse.ArgumentParser(
        description="Combine dat_size_*_index_*_{cov,cor}.npy into full_<size> outputs (R combine_data.R equivalent)."
    )
    ap.add_argument("WRKDIR", help="Working directory (contains pwr_data/)")
    ap.add_argument("FILEDIR", help="Compatibility argument (unused).")
    ap.add_argument("--outdir", default=None, help="Override output directory (default: WRKDIR/pwr_data)")
    ap.add_argument("--write-rds", action="store_true",
                    help="Also write full_<size>.rds (requires pyreadr).")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else (Path(args.WRKDIR) / "pwr_data")
    if not outdir.exists():
        raise FileNotFoundError(f"pwr_data not found: {outdir}")

    cov_files = sorted(outdir.glob("dat_size_*_index_*_cov.npy"))
    cor_files = sorted(outdir.glob("dat_size_*_index_*_cor.npy"))

    if len(cov_files) == 0:
        raise RuntimeError(f"[FATAL] No cov files found in {outdir} ending with _cov.npy")
    if len(cor_files) == 0:
        raise RuntimeError(f"[FATAL] No cor files found in {outdir} ending with _cor.npy")

    # Build maps keyed by (size, index)
    cov_map = {}
    for f in cov_files:
        key = parse_cov_name(f)
        if key is not None:
            cov_map[key] = f

    cor_map = {}
    for f in cor_files:
        key = parse_cor_name(f)
        if key is not None:
            cor_map[key] = f

    # Intersect keys so cov/cor stay aligned
    keys = sorted(set(cov_map.keys()) & set(cor_map.keys()))
    if len(keys) == 0:
        raise RuntimeError(
            f"[FATAL] Found cov files ({len(cov_map)}) and cor files ({len(cor_map)}), "
            f"but no matching (size,index) pairs."
        )

    # Group keys by size, like R: num_sets <- unique(df$size)
    by_size = defaultdict(list)
    for (size, idx) in keys:
        by_size[size].append(idx)

    sizes_sorted = sorted(by_size.keys())
    print(f"[INFO] Found {len(keys)} paired rows across {len(sizes_sorted)} dataset sizes.")

    # Optional RDS writer
    if args.write_rds:
        try:
            import pyreadr  # type: ignore
            import pandas as pd  # type: ignore
        except Exception as e:
            raise RuntimeError("--write-rds requested, but pyreadr/pandas not available in this environment.") from e
    else:
        pyreadr = None
        pd = None

    for size in sizes_sorted:
        idxs = sorted(by_size[size])  # arrange(index)

        # Load first to infer n_edge
        first = np.load(cov_map[(size, idxs[0])])
        if first.ndim != 1:
            raise ValueError(f"Expected 1D vectors; got {first.shape} for {cov_map[(size, idxs[0])].name}")
        n_edge = first.shape[0]

        cov_mat = np.zeros((len(idxs), n_edge), dtype=float)
        cor_mat = np.zeros((len(idxs), n_edge), dtype=float)

        for r, idx in enumerate(idxs):
            cov_vec = np.load(cov_map[(size, idx)])
            cor_vec = np.load(cor_map[(size, idx)])

            if cov_vec.shape != (n_edge,) or cor_vec.shape != (n_edge,):
                raise ValueError(
                    f"Shape mismatch for size={size}, index={idx}: cov{cov_vec.shape} cor{cor_vec.shape} "
                    f"expected ({n_edge},)"
                )

            cov_mat[r, :] = cov_vec
            cor_mat[r, :] = cor_vec

        # Save per-size outputs (Python-native)
        out_cov = outdir / f"full_{size}_cov.npy"
        out_cor = outdir / f"full_{size}_cor.npy"
        np.save(out_cov, cov_mat)
        np.save(out_cor, cor_mat)

        print(f"[OK] size={size}: wrote {out_cov.name} {cov_mat.shape}")
        print(f"[OK] size={size}: wrote {out_cor.name} {cor_mat.shape}")

        # Optional: write exactly like R did: full_<size>.rds containing cov matrix as a data.frame
        if args.write_rds and pyreadr is not None and pd is not None:
            # Convert to data frame with columns V1..Vn like R would when rbind vectors
            cov_df = pd.DataFrame(cov_mat)
            rds_path = outdir / f"full_{size}.rds"
            pyreadr.write_rds(str(rds_path), cov_df)
            print(f"[OK] size={size}: wrote {rds_path.name} (cov only, R-style)")

    print("[DONE] combine_data complete.")


if __name__ == "__main__":
    main()

