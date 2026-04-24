#!/usr/bin/env python3
"""
combine_data.py — Step 3 of the power calculator pipeline.

Aggregates the per-chunk simulation outputs produced by Step 2 into one
stacked matrix per sample size, then computes predicted phenotype vectors
and adds calibrated noise. Called by combine_data.sh.

For each sample size N found in pwr_data/:
    1. Collects all dat_size_<N>_index_<k>_cov.npy and _cor.npy vectors
       produced across all Step 2 array tasks.
    2. Stacks them row-wise into full_<N>_cov.npy and full_<N>_cor.npy,
       each of shape (n_subjects, n_edge).
    3. Computes y = cor_mat @ ridge_vec — the predicted phenotype for each
       subject using the pre-fitted ridge model weights.
    4. Computes yt = y + N(0, (epsilon * sqrt(var(y)))^2) — the noise-
       corrupted phenotype used as the actual regression target in cv.py.

Outputs per sample size (written to WRKDIR/pwr_data/):
    full_<size>_cov.npy   — stacked covariance matrix,  shape (n_obs, n_edge)
    full_<size>_cor.npy   — stacked correlation matrix, shape (n_obs, n_edge)
    full_<size>_y.npy     — clean phenotype vector,     shape (n_obs,)
    full_<size>_yt.npy    — noisy phenotype vector,     shape (n_obs,)
    full_<size>.rds       — optional R-format covariance matrix (--write-rds)

Global outputs:
    pconn_template_lookup.csv — provenance table mapping each simulation row
                                to its source pconn template file(s)

Ridge vector loading priority (load_ridge):
    1. pwr_data/ridge.npy          (pre-computed by ridge_model_generation.py)
    2. pwr_data/ridge*.npy         (any matching file, sorted, first taken)
    3. FILEDIR/haufe.csv           (derived on the fly if no .npy exists)
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


# Compiled regexes for parsing simulation output filenames.
# Group 1 = dataset size (integer), Group 2 = replicate index (integer).
RE_COV = re.compile(r"^dat_size_(\d+)_index_(\d+)_cov\.npy$")
RE_COR = re.compile(r"^dat_size_(\d+)_index_(\d+)_cor\.npy$")


def parse_cov_name(p: Path):
    """
    Extract (size, index) from a covariance output filename.

    Args:
        p: Path whose .name is matched against RE_COV.

    Returns:
        Tuple (int, int) of (dataset_size, replicate_index), or None if
        the filename does not match the expected pattern.
    """
    m = RE_COV.match(p.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_cor_name(p: Path):
    """
    Extract (size, index) from a correlation output filename.

    Args:
        p: Path whose .name is matched against RE_COR.

    Returns:
        Tuple (int, int) of (dataset_size, replicate_index), or None if
        the filename does not match the expected pattern.
    """
    m = RE_COR.match(p.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def upper_triangle_vec(M: np.ndarray) -> np.ndarray:
    """
    Extract the strict upper triangle of a square matrix as a float64 1-D vector.

    Used to convert the square haufe.csv matrix into the same edge-vector
    format as the per-subject _cor.npy arrays, so that y = cor_mat @ ridge_vec
    is a valid inner product.

    Args:
        M: Square 2-D array of shape (n, n).

    Returns:
        1-D float64 array of length n*(n-1)/2.
    """
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu].astype(np.float64, copy=False)


def load_ridge_from_haufe(filedir: Path) -> np.ndarray:
    """
    Derive the ridge weight vector directly from FILEDIR/haufe.csv.

    Used as the final fallback in load_ridge() when no pre-computed ridge.npy
    exists. Reads the square Haufe-transform matrix and vectorises its upper
    triangle to produce a vector of the same length as the per-subject edge
    vectors in the _cor.npy files.

    Args:
        filedir: Directory expected to contain haufe.csv.

    Returns:
        1-D float64 array of length n*(n-1)/2.

    Raises:
        FileNotFoundError: If haufe.csv does not exist in filedir.
        ValueError:        If haufe.csv does not contain a square matrix.
    """
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
    Load the ridge weight vector, trying three sources in priority order.

    Priority:
        1. outdir/ridge.npy        — canonical output of ridge_model_generation.py
        2. outdir/ridge*.npy       — any matching file (sorted, first taken);
                                     handles versioned or renamed ridge files
        3. FILEDIR/haufe.csv       — derived on the fly; slower but avoids
                                     requiring a separate ridge generation step

    The returned vector is always flattened to 1-D float64, regardless of
    the shape it was stored in.

    Args:
        outdir:  pwr_data/ directory where ridge.npy would be written.
        filedir: Pipeline scripts directory containing haufe.csv.

    Returns:
        1-D float64 array of length n_edge (= n*(n-1)/2 for the pconn ROI count).

    Raises:
        FileNotFoundError: If all three sources are unavailable.
    """
    rp = outdir / "ridge.npy"
    if rp.exists():
        r = np.load(str(rp), allow_pickle=True)
        print(f"[INFO] Loaded ridge from {rp}")
        return np.asarray(r).reshape(-1).astype(np.float64, copy=False)

    # Try any ridge*.npy in outdir (e.g. ridge_v2.npy, ridge_alpha0.1.npy)
    cand = sorted(outdir.glob("ridge*.npy"))
    if cand:
        r = np.load(str(cand[0]), allow_pickle=True)
        print(f"[INFO] Loaded ridge from {cand[0]}")
        return np.asarray(r).reshape(-1).astype(np.float64, copy=False)

    # Final fallback: derive from haufe.csv in FILEDIR
    print("[INFO] No ridge.npy found in pwr_data — deriving ridge_vec from FILEDIR/haufe.csv")
    return load_ridge_from_haufe(filedir)


def write_template_lookup(outdir: Path) -> None:
    """
    Scan all _meta.json files in outdir and write pconn_template_lookup.csv.

    Produces a provenance table that maps every simulation row back to the
    pconn template file(s) used to generate it. Useful for diagnosing
    unexpected variation in outputs or auditing template reuse across rows.

    Column layout:
        simulation_id  — e.g. "dat_size_100_index_3"
        dat_size       — integer dataset size
        index          — integer replicate index within that size
        n_templates    — number of pconn templates averaged for this row
        template_1 ... — one column per template path; width equals the
                         maximum n_templates seen across all simulations.
                         Rows with fewer templates are padded with "".

    Args:
        outdir: pwr_data/ directory containing the _meta.json files.

    Side effects:
        Writes pconn_template_lookup.csv to outdir.
        Prints warnings for any _meta.json files that fail to load.
        Skips writing entirely if no _meta.json files are found.
    """
    meta_files = sorted(outdir.glob("dat_size_*_index_*_meta.json"))
    if not meta_files:
        print("[WARN] No _meta.json files found — skipping pconn_template_lookup.csv")
        return

    # Load all records first so we can determine the maximum template count
    # before writing the header, since CSV column count must be fixed up front.
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

    # Width of the template columns = max templates across all sims
    max_templates = max(len(r.get("template_pconns", [])) for r in records)
    template_cols = [f"template_{i + 1}" for i in range(max_templates)]

    lookup_path = outdir / "pconn_template_lookup.csv"

    with open(lookup_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["simulation_id", "dat_size", "index", "n_templates"] + template_cols)

        for meta in records:
            sim_id  = f"dat_size_{meta['dat_size']}_index_{meta['index']}"
            tpconns = meta.get("template_pconns", [])
            # Pad shorter rows with "" so every row has the same column count
            padded  = tpconns + [""] * (max_templates - len(tpconns))
            writer.writerow([sim_id, meta["dat_size"], meta["index"], meta["n_templates"]] + padded)

    print(f"[INFO] wrote {lookup_path.name} ({len(records)} rows, {max_templates} template column(s))")


def main():
    """
    Parse arguments, aggregate per-chunk outputs, compute y/yt, write results.

    Processing flow:
        1. Glob all _cov.npy and _cor.npy files and parse their (size, index) keys.
        2. Intersect cov and cor key sets so only fully paired rows are stacked.
           Unpaired files (one present, other missing) are silently excluded.
        3. Load the ridge weight vector via load_ridge().
        4. For each sample size (sorted ascending):
             a. Stack per-index vectors into a (n_obs, n_edge) matrix.
             b. Save full_<size>_cov.npy and full_<size>_cor.npy.
             c. Compute y  = cor_mat @ ridge_vec and save full_<size>_y.npy.
             d. Compute yt = y + N(0, (epsilon * sqrt(var(y)))^2) and save
                full_<size>_yt.npy. Skip y/yt if ridge length != n_edge.
             e. Optionally write full_<size>.rds (covariance only, R format).
        5. Write pconn_template_lookup.csv via write_template_lookup().

    Arguments:
        WRKDIR   Root working directory containing pwr_data/.
        FILEDIR  Directory containing haufe.csv (ridge fallback source).
        EPSILON  Noise scale factor for yt. 0 = no noise (yt == y).
                 Noise std = epsilon * sqrt(var(y)), so the SNR scales
                 inversely with epsilon regardless of y's absolute scale.

    Optional arguments:
        --outdir PATH    Override the default output directory (WRKDIR/pwr_data/).
                         Useful for testing without modifying the live pwr_data/.
        --write-rds      Also write full_<size>.rds (requires pyreadr + pandas).
                         Covariance matrix only; provided for R-based downstream use.
    """
    ap = argparse.ArgumentParser(
        description="Combine dat_size_*_index_*_{cov,cor}.npy into full_<size> outputs and compute y = X @ ridge_vec."
    )
    ap.add_argument("WRKDIR",   help="Working directory (contains pwr_data/)")
    ap.add_argument("FILEDIR",  help="Directory containing haufe.csv (used to derive ridge_vec if ridge.npy absent).")
    ap.add_argument("EPSILON",  type=float,
                    help="Noise scale factor: error ~ N(0, (epsilon * sqrt(var(y)))^2).")
    ap.add_argument("--outdir", default=None,
                    help="Override output directory (default: WRKDIR/pwr_data)")
    ap.add_argument("--write-rds", action="store_true",
                    help="Also write full_<size>.rds (requires pyreadr).")
    args = ap.parse_args()

    outdir  = Path(args.outdir) if args.outdir else (Path(args.WRKDIR) / "pwr_data")
    filedir = Path(args.FILEDIR)
    epsilon = args.EPSILON

    if not outdir.exists():
        raise FileNotFoundError(f"pwr_data not found: {outdir}")

    # Glob all per-chunk simulation output files
    cov_files = sorted(outdir.glob("dat_size_*_index_*_cov.npy"))
    cor_files = sorted(outdir.glob("dat_size_*_index_*_cor.npy"))

    if len(cov_files) == 0:
        raise RuntimeError(f"[FATAL] No cov files found in {outdir} ending with _cov.npy")
    if len(cor_files) == 0:
        raise RuntimeError(f"[FATAL] No cor files found in {outdir} ending with _cor.npy")

    # ── Load ridge weight vector ──────────────────────────────────────────────
    # Must be loaded before the per-size loop since all sizes share the same
    # ridge_vec. A shape mismatch between ridge_vec and n_edge is handled
    # per-size inside the loop rather than here, since different sizes could
    # in principle have different n_edge (though in practice they do not).
    try:
        ridge_vec = load_ridge(outdir, filedir)
        print(f"[INFO] ridge_vec shape={ridge_vec.shape}")
    except Exception as e:
        print(f"[FATAL] Could not load ridge_vec: {e}")
        raise

    # ── Build (size, index) → filepath maps ──────────────────────────────────
    # Keyed by (dataset_size, replicate_index) tuples parsed from filenames.
    # Files whose names don't match the expected pattern are silently skipped
    # by parse_cov_name / parse_cor_name returning None.
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

    # Intersect keys: only stack rows where both cov and cor files exist.
    # This guards against partial Step 2 failures where one file was written
    # but the other was not (e.g. job killed between the two np.save calls).
    keys = sorted(set(cov_map.keys()) & set(cor_map.keys()))
    if len(keys) == 0:
        raise RuntimeError(
            f"[FATAL] Found cov files ({len(cov_map)}) and cor files ({len(cor_map)}), "
            f"but no matching (size,index) pairs."
        )

    # Group paired keys by sample size for per-size stacking
    by_size = defaultdict(list)
    for (size, idx) in keys:
        by_size[size].append(idx)

    sizes_sorted = sorted(by_size.keys())
    print(f"[INFO] Found {len(keys)} paired rows across {len(sizes_sorted)} dataset sizes.")

    # Conditionally import RDS dependencies once rather than per-size iteration
    if args.write_rds:
        try:
            import pyreadr  # type: ignore
            import pandas as pd  # type: ignore
        except Exception as e:
            raise RuntimeError("--write-rds requested, but pyreadr/pandas not available.") from e
    else:
        pyreadr = None
        pd = None

    # ── Per-size aggregation loop ─────────────────────────────────────────────
    for size in sizes_sorted:
        idxs = sorted(by_size[size])

        # Load the first vector to infer n_edge; all subsequent vectors for
        # this size must match this shape or a ValueError is raised below.
        first  = np.load(cov_map[(size, idxs[0])])
        if first.ndim != 1:
            raise ValueError(f"Expected 1D vectors; got {first.shape} for {cov_map[(size, idxs[0])].name}")
        n_edge = first.shape[0]

        # Pre-allocate stacked matrices; rows = subjects, cols = edges
        cov_mat = np.zeros((len(idxs), n_edge), dtype=float)
        cor_mat = np.zeros((len(idxs), n_edge), dtype=float)

        for r, idx in enumerate(idxs):
            cov_vec = np.load(cov_map[(size, idx)])
            cor_vec = np.load(cor_map[(size, idx)])

            if cov_vec.shape != (n_edge,) or cor_vec.shape != (n_edge,):
                raise ValueError(
                    f"Shape mismatch for size={size}, index={idx}: "
                    f"cov{cov_vec.shape} cor{cor_vec.shape} expected ({n_edge},)"
                )

            cov_mat[r, :] = cov_vec
            cor_mat[r, :] = cor_vec

        # ── Save stacked matrices ─────────────────────────────────────────────
        out_cov = outdir / f"full_{size}_cov.npy"
        out_cor = outdir / f"full_{size}_cor.npy"
        np.save(out_cov, cov_mat)
        np.save(out_cor, cor_mat)
        print(f"[OK] size={size}: wrote {out_cov.name} {cov_mat.shape}")
        print(f"[OK] size={size}: wrote {out_cor.name} {cor_mat.shape}")

        # ── Compute y = cor_mat @ ridge_vec ───────────────────────────────────
        # Projects each subject's correlation edge vector onto the ridge weight
        # vector to produce a scalar predicted phenotype. Shape: (n_obs,).
        # Skipped (with a warning) if ridge_vec length doesn't match n_edge,
        # which would indicate a mismatch between the ridge model and the pconn
        # ROI set used for simulation.
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

            # ── Compute yt = y + noise ────────────────────────────────────────
            # Adds Gaussian noise scaled to epsilon * sqrt(var(y)), so the
            # noise magnitude is proportional to the signal's own standard
            # deviation. This makes EPSILON a true SNR control: EPSILON=0
            # gives a clean signal (yt == y), EPSILON=1 gives noise_std equal
            # to the signal std, EPSILON>1 degrades SNR below 1.
            var_y      = np.var(y)
            noise_std  = epsilon * np.sqrt(var_y)
            error      = np.random.normal(loc=0.0, scale=noise_std, size=y.shape)
            yt         = y + error
            out_yt     = outdir / f"full_{size}_yt.npy"
            np.save(out_yt, yt)
            print(f"[OK] size={size}: wrote {out_yt.name} {yt.shape}  "
                  f"noise_std={noise_std:.4g}  range=[{yt.min():.4g}, {yt.max():.4g}]")

        # ── Optional RDS output ───────────────────────────────────────────────
        # Writes covariance matrix only (not cor, y, or yt) in R's native
        # serialisation format for downstream R-based analysis.
        if args.write_rds and pyreadr is not None and pd is not None:
            cov_df   = pd.DataFrame(cov_mat)
            rds_path = outdir / f"full_{size}.rds"
            pyreadr.write_rds(str(rds_path), cov_df)
            print(f"[OK] size={size}: wrote {rds_path.name} (cov only, R-style)")

    write_template_lookup(outdir)
    print("[DONE] combine_data complete.")


if __name__ == "__main__":
    main()
