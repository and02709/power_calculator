#!/usr/bin/env python3
import argparse
import csv
import json
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
 
 
def upper_triangle_vec(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu].astype(np.float64, copy=False)
 
 
def load_ridge_from_haufe(filedir: Path) -> np.ndarray:
    """Load haufe.csv from FILEDIR and return upper-triangle vector (same as ridge_model_generation.py)."""
    import pandas as pd
    haufe_path = filedir / "haufe.csv"
    if not haufe_path.exists():
        raise FileNotFoundError(f"haufe.csv not found at: {haufe_path}")
    haufe = pd.read_csv(haufe_path, header=None).to_numpy(dtype=np.float64, copy=False)
    if haufe.ndim != 2 or haufe.shape[0] != haufe.shape[1]:
        raise ValueError(f"Expected square matrix in haufe.csv; got {haufe.shape}")
    return upper_triangle_vec(haufe)
 
 
def load_ridge(outdir: Path, filedir: Path) -> np.ndarray:
    """
    Try to load ridge_vec in this order:
      1. outdir/ridge.npy  (pre-computed by ridge_model_generation.py)
      2. any outdir/ridge*.npy
      3. derive directly from FILEDIR/haufe.csv
    """
    rp = outdir / "ridge.npy"
    if rp.exists():
        r = np.load(str(rp), allow_pickle=True)
        print(f"[INFO] Loaded ridge from {rp}")
        return np.asarray(r).reshape(-1).astype(np.float64, copy=False)
 
    cand = sorted(outdir.glob("ridge*.npy"))
    if cand:
        r = np.load(str(cand[0]), allow_pickle=True)
        print(f"[INFO] Loaded ridge from {cand[0]}")
        return np.asarray(r).reshape(-1).astype(np.float64, copy=False)
 
    # Fall back to haufe.csv
    print("[INFO] No ridge.npy found in pwr_data — deriving ridge_vec from FILEDIR/haufe.csv")
    return load_ridge_from_haufe(filedir)
 
 
def write_template_lookup(outdir: Path) -> None:
    """
    Scan all _meta.json files in outdir and write pconn_template_lookup.csv.
 
    One row per simulation. Columns:
        simulation_id       -- e.g. dat_size_100_index_3
        dat_size            -- integer dataset size
        index               -- integer replicate index
        n_templates         -- number of pconn templates averaged
        template_1 ...      -- one column per template (width = max n_templates across all sims)
    """
    meta_files = sorted(outdir.glob("dat_size_*_index_*_meta.json"))
    if not meta_files:
        print("[WARN] No _meta.json files found — skipping pconn_template_lookup.csv")
        return
 
    # Load all records first so we know the max number of templates (for header width)
    records = []
    for mf in meta_files:
        try:
            with open(mf) as jf:
                meta = json.load(jf)
        except Exception as e:
            print(f"[WARN] Could not read {mf.name}: {e}")
            continue
        records.append(meta)
 
    if not records:
        print("[WARN] All _meta.json files failed to load — skipping pconn_template_lookup.csv")
        return
 
    max_templates = max(len(r.get("template_pconns", [])) for r in records)
    template_cols = [f"template_{i + 1}" for i in range(max_templates)]
 
    lookup_path = outdir / "pconn_template_lookup.csv"
 
    with open(lookup_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["simulation_id", "dat_size", "index", "n_templates"] + template_cols)
 
        for meta in records:
            sim_id = f"dat_size_{meta['dat_size']}_index_{meta['index']}"
            tpconns = meta.get("template_pconns", [])
            # Pad with empty strings if this sim used fewer templates than the max
            padded = tpconns + [""] * (max_templates - len(tpconns))
            writer.writerow([sim_id, meta["dat_size"], meta["index"], meta["n_templates"]] + padded)
 
    print(f"[INFO] wrote {lookup_path.name} ({len(records)} rows, {max_templates} template column(s))")
 
 
def main():
    ap = argparse.ArgumentParser(
        description="Combine dat_size_*_index_*_{cov,cor}.npy into full_<size> outputs and compute y = X @ ridge_vec."
    )
    ap.add_argument("WRKDIR", help="Working directory (contains pwr_data/)")
    ap.add_argument("FILEDIR", help="Directory containing haufe.csv (used to derive ridge_vec if ridge.npy absent).")
    ap.add_argument("EPSILON", type=float, help="Noise scale factor: error ~ N(0, (epsilon * sqrt(var(y)))^2).")
    ap.add_argument("--outdir", default=None, help="Override output directory (default: WRKDIR/pwr_data)")
    ap.add_argument("--write-rds", action="store_true",
                    help="Also write full_<size>.rds (requires pyreadr).")
    args = ap.parse_args()
 
    outdir  = Path(args.outdir) if args.outdir else (Path(args.WRKDIR) / "pwr_data")
    filedir = Path(args.FILEDIR)
    epsilon = args.EPSILON

    if not outdir.exists():
        raise FileNotFoundError(f"pwr_data not found: {outdir}")
 
    cov_files = sorted(outdir.glob("dat_size_*_index_*_cov.npy"))
    cor_files = sorted(outdir.glob("dat_size_*_index_*_cor.npy"))
 
    if len(cov_files) == 0:
        raise RuntimeError(f"[FATAL] No cov files found in {outdir} ending with _cov.npy")
    if len(cor_files) == 0:
        raise RuntimeError(f"[FATAL] No cor files found in {outdir} ending with _cor.npy")
 
    # ------------------------------------------------------------------
    # Load ridge_vec — from ridge.npy if available, else from haufe.csv
    # ------------------------------------------------------------------
    try:
        ridge_vec = load_ridge(outdir, filedir)
        print(f"[INFO] ridge_vec shape={ridge_vec.shape}")
    except Exception as e:
        print(f"[FATAL] Could not load ridge_vec: {e}")
        raise
 
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
 
    # Group keys by size
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
            raise RuntimeError("--write-rds requested, but pyreadr/pandas not available.") from e
    else:
        pyreadr = None
        pd = None
 
    for size in sizes_sorted:
        idxs = sorted(by_size[size])
 
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
 
        # Save per-size outputs
        out_cov = outdir / f"full_{size}_cov.npy"
        out_cor = outdir / f"full_{size}_cor.npy"
        np.save(out_cov, cov_mat)
        np.save(out_cor, cor_mat)
 
        print(f"[OK] size={size}: wrote {out_cov.name} {cov_mat.shape}")
        print(f"[OK] size={size}: wrote {out_cor.name} {cor_mat.shape}")
 
        # ------------------------------------------------------------------
        # Compute y = cor_mat @ ridge_vec  (shape: n_obs,)
        # ------------------------------------------------------------------
        if ridge_vec.shape[0] != n_edge:
            print(
                f"[WARN] size={size}: ridge length {ridge_vec.shape[0]} != n_edge {n_edge} "
                f"— skipping y and yt computation"
            )
        else:
            y = cor_mat @ ridge_vec          # (n_obs,)
            out_y = outdir / f"full_{size}_y.npy"
            np.save(out_y, y)
            print(f"[OK] size={size}: wrote {out_y.name} {y.shape}  "
                  f"range=[{y.min():.4g}, {y.max():.4g}]")

            # ------------------------------------------------------------------
            # Compute yt = y + error, where error ~ N(0, (epsilon * sqrt(var(y)))^2)
            # ------------------------------------------------------------------
            var_y = np.var(y)
            noise_std = epsilon * np.sqrt(var_y)
            error = np.random.normal(loc=0.0, scale=noise_std, size=y.shape)
            yt = y + error
            out_yt = outdir / f"full_{size}_yt.npy"
            np.save(out_yt, yt)
            print(f"[OK] size={size}: wrote {out_yt.name} {yt.shape}  "
                  f"noise_std={noise_std:.4g}  range=[{yt.min():.4g}, {yt.max():.4g}]")

        # Optional RDS
        if args.write_rds and pyreadr is not None and pd is not None:
            cov_df = pd.DataFrame(cov_mat)
            rds_path = outdir / f"full_{size}.rds"
            pyreadr.write_rds(str(rds_path), cov_df)
            print(f"[OK] size={size}: wrote {rds_path.name} (cov only, R-style)")
 
    write_template_lookup(outdir)
    print("[DONE] combine_data complete.")
 
 
if __name__ == "__main__":
    main()