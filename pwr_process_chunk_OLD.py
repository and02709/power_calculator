#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import cormat
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib


def read_pconn_matrix(path: str) -> np.ndarray:
    img = nib.load(path)
    data = np.asanyarray(img.get_fdata())
    data = np.squeeze(data)
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError(f"Unexpected pconn shape for {path}: {data.shape}")
    return data.astype(np.float64)


def symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def corr_from_cov(C: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(C), 1e-12, np.inf))
    Dinv = np.diag(1.0 / d)
    R = Dinv @ C @ Dinv
    R = symmetrize(R)
    np.fill_diagonal(R, 1.0)
    return R


def ensure_corr(M: np.ndarray) -> np.ndarray:
    """
    If diagonal is not ~1, treat as covariance and convert to correlation.
    """
    M = symmetrize(M)
    d = np.diag(M)
    if np.median(d) < 0.5 or np.median(d) > 1.5:
        return corr_from_cov(M)
    R = M.copy()
    np.fill_diagonal(R, 1.0)
    return R


def nearest_spd(A: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    A = symmetrize(A)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    return symmetrize((V * w) @ V.T)


def vectorize_uplo(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]

def simulate_subject_timeseries(
    R_target: np.ndarray,
    n_time: int,
    rng: np.random.Generator,
    spd_eps: float = 1e-6,
) -> np.ndarray:
    """
    Simulate time series X ~ N(0, R_target)
    """
    R = ensure_corr(R_target)
    R = nearest_spd(R, eps=spd_eps)

    L = np.linalg.cholesky(R)
    Z = rng.standard_normal((n_time, R.shape[0]))
    X = Z @ L.T
    return X

    def main():
    ap = argparse.ArgumentParser(description="Simple pconn-based simulation chunk.")
    ap.add_argument("WRKDIR", type=str)
    ap.add_argument("START", type=int)
    ap.add_argument("END", type=int)
    ap.add_argument("FILEDIR", type=str)

    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n_sub", type=int, default=10)
    ap.add_argument("--n_time", type=int, default=2000)

    ap.add_argument(
        "--pconn_root",
        type=str,
        default="/projects/standard/feczk001/shared/projects/ABCD/gordon_sets/data/group2_10minonly_FD0p1",
    )

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # -----------------------------
    # Load index
    # -----------------------------
    index_path = Path(args.WRKDIR) / "pwr_data" / "pwr_index_file.txt"
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    index = pd.read_csv(index_path, sep=r"\s+", header=None)
    chunk = index.iloc[args.START - 1 : args.END]
    n_chunks = chunk.shape[0]

    # -----------------------------
    # Find pconn files
    # -----------------------------
    pconn_files = sorted(Path(args.pconn_root).rglob("*.pconn.nii"))
    if not pconn_files:
        raise RuntimeError("No pconn files found")

    print(f"[INFO] Found {len(pconn_files)} pconn files")

    # -----------------------------
    # Use FIRST pconn as target (intentionally simple)
    # -----------------------------
    target_path = pconn_files[0]
    print(f"[INFO] Using target pconn: {target_path}")

    R_target = read_pconn_matrix(str(target_path))
    R_target = ensure_corr(R_target)

    n_node = R_target.shape[0]
    n_edge = n_node * (n_node - 1) // 2

    synth_cov = np.zeros((n_chunks, n_edge))
    synth_cor = np.zeros((n_chunks, n_edge))

    # -----------------------------
    # Simulate per chunk row
    # -----------------------------
    for i in range(n_chunks):
        cov_accum = np.zeros(n_edge)
        cor_accum = np.zeros(n_edge)

        for s in range(args.n_sub):
            X = simulate_subject_timeseries(
                R_target=R_target,
                n_time=args.n_time,
                rng=rng,
            )

            C = np.cov(X, rowvar=False)
            R = np.corrcoef(X, rowvar=False)

            cov_accum += vectorize_uplo(C)
            cor_accum += vectorize_uplo(R)

        synth_cov[i] = cov_accum / args.n_sub
        synth_cor[i] = cor_accum / args.n_sub

        print(f"[INFO] simulated dataset {i+1}/{n_chunks}")

    # -----------------------------
    # Write outputs
    # -----------------------------
    outdir = Path(args.WRKDIR) / "pwr_data"
    outdir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(chunk.itertuples(index=False)):
        sample_count = int(row[1])
        dataset_size = int(row[2])

        stem = outdir / f"dat_size_{dataset_size}_index_{sample_count}"
        np.save(stem.with_suffix("_cov.npy"), synth_cov[i])
        np.save(stem.with_suffix("_cor.npy"), synth_cor[i])

    print("[OK] Done")


if __name__ == "__main__":
    main()