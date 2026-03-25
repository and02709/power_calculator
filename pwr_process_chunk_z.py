#!/usr/bin/env python3
import os
import argparse
import json
import random
from typing import Optional
 
import numpy as np
import pandas as pd
import nibabel as nib
 
 
def fisher_z_from_corr(R: np.ndarray, clip: float = 1e-7) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    R = np.clip(R, -1.0 + clip, 1.0 - clip)
    Z = np.arctanh(R)
    np.fill_diagonal(Z, 0.0)
    return Z
 
 
def vectorize_uplo(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    return mat[iu]
 
 
def import_pconn(
    file: str,
    force_diag_one: bool = True,
    project_to_psd: bool = True,
    psd_eps: Optional[float] = None,
) -> np.ndarray:
    img = nib.load(file)
    mat = np.squeeze(np.asanyarray(img.get_fdata())).astype(np.float64, copy=False)
 
    mat_sym = 0.5 * (mat + mat.T)
 
    if force_diag_one:
        np.fill_diagonal(mat_sym, 1.0)
 
    if project_to_psd:
        if psd_eps is None:
            psd_eps = float(np.finfo(np.float32).eps)
        w, v = np.linalg.eigh(mat_sym)
        w_clip = np.maximum(w, psd_eps)
        mat_psd = (v * w_clip) @ v.T
        mat_sym = 0.5 * (mat_psd + mat_psd.T)
        if force_diag_one:
            np.fill_diagonal(mat_sym, 1.0)
 
    return mat_sym
 
 
class EigSplit:
    def __init__(self, symmetric_matrix: np.ndarray):
        eps = np.finfo(np.float32).eps
        w, v = np.linalg.eigh(symmetric_matrix)
        self.eigenvectors = v
        w_pos = np.maximum(w, eps)
        self.sqrt_eigenvalues_pos = np.sqrt(w_pos)
        w_neg = np.maximum(-w, eps)
        self.sqrt_eigenvalues_neg = np.sqrt(w_neg)
 
 
def get_averaged_pconn(pconn_dir: str, num_to_select: int):
    all_files = [
        os.path.join(pconn_dir, f)
        for f in os.listdir(pconn_dir)
        if f.endswith('.pconn.nii')
    ]
    if not all_files:
        raise FileNotFoundError(f"No .pconn.nii files found in {pconn_dir}")
 
    num_to_select = min(num_to_select, len(all_files))
    selected_files = random.sample(all_files, num_to_select)
 
    print(f"[INFO] Averaging {len(selected_files)} pconns from {pconn_dir}")
    mats = [import_pconn(f, project_to_psd=False) for f in selected_files]
    return np.mean(mats, axis=0), selected_files
 
 
def build_x_aug_from_rng(n_sub: int, one_target: bool, rng: np.random.Generator) -> np.ndarray:
    need_cols = 1 if one_target else 2
    x_sub = rng.standard_normal((n_sub, need_cols))
    if one_target:
        x_aug = np.column_stack([x_sub[:, 0], -x_sub[:, 0]])
    else:
        x_aug = np.column_stack([x_sub[:, 0], -x_sub[:, 0], x_sub[:, 1], -x_sub[:, 1]])
    return x_aug + abs(float(np.min(x_aug)))
 
 
def simulate_iteration(eig1, eig2, x_aug, n_time, rng, clip):
    n_sub = x_aug.shape[0]
    n_node = eig1.eigenvectors.shape[0]
    n_edge = (n_node * (n_node - 1)) // 2
    s_cov, s_cor, s_z = np.zeros(n_edge), np.zeros(n_edge), np.zeros(n_edge)
 
    for i in range(n_sub):
        print(i)
        trand = rng.standard_normal((n_time, n_node))
        t = (
            np.sqrt(x_aug[i, 0]) * (trand * eig1.sqrt_eigenvalues_pos).dot(eig1.eigenvectors.T)
            + np.sqrt(x_aug[i, 1]) * (trand * eig1.sqrt_eigenvalues_neg).dot(eig1.eigenvectors.T)
        )
 
        if eig2 is not None:
            trand2 = rng.standard_normal((n_time, n_node))
            t += (
                np.sqrt(x_aug[i, 2]) * (trand2 * eig2.sqrt_eigenvalues_pos).dot(eig2.eigenvectors.T)
                + np.sqrt(x_aug[i, 3]) * (trand2 * eig2.sqrt_eigenvalues_neg).dot(eig2.eigenvectors.T)
            )
 
        C = np.cov(t, rowvar=False)
        R = np.corrcoef(t, rowvar=False)
        Z = fisher_z_from_corr(R, clip=clip)
        s_cov += vectorize_uplo(C)
        s_cor += vectorize_uplo(R)
        s_z += vectorize_uplo(Z)
 
    return s_cov / n_sub, s_cor / n_sub, s_z / n_sub
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("WRKDIR")
    ap.add_argument("START", type=int)
    ap.add_argument("END", type=int)
    ap.add_argument("FILEDIR")
    ap.add_argument("PCONNDIR")
    ap.add_argument("PCONNREF")
    ap.add_argument("NUMTEMP", type=int)
    ap.add_argument("NREP", type=int)
    ap.add_argument("UNUSED_ARG")
    ap.add_argument("--n_time", type=int, default=1000)
    ap.add_argument("--pconn1")
    ap.add_argument("--use_one_target", action="store_true")
    ap.add_argument("--z_clip", type=float, default=1e-7)
    args = ap.parse_args()
 
    out_dir = os.path.join(args.WRKDIR, "pwr_data")
    index_path = os.path.join(out_dir, "pwr_index_file.txt")
    index_file = pd.read_csv(index_path, sep=r"\s+", header=None, engine="python")
    chunk_rows = index_file.iloc[args.START - 1 : args.END]
 
    avg_mat, selected_pconns = get_averaged_pconn(args.PCONNDIR, args.NUMTEMP)
    w, v = np.linalg.eigh(avg_mat)
    avg_mat_psd = (v * np.maximum(w, 1e-12)) @ v.T
    np.fill_diagonal(avg_mat_psd, 1.0)
 
    eig1 = EigSplit(avg_mat_psd)
    eig2 = None if args.use_one_target else eig1
 
    for i in range(len(chunk_rows)):
        sample_count = int(chunk_rows.iloc[i, 1])
        dataset_size = int(chunk_rows.iloc[i, 2])
        rng = np.random.default_rng()
        x_aug = build_x_aug_from_rng(args.NREP, args.use_one_target, rng)
 
        m_cov, m_cor, m_z = simulate_iteration(eig1, eig2, x_aug, args.n_time, rng, args.z_clip)
 
        stem = os.path.join(out_dir, f"dat_size_{dataset_size}_index_{sample_count}")
        np.save(f"{stem}_cov.npy", m_cov)
        np.save(f"{stem}_cor.npy", m_cor)
        np.save(f"{stem}_z.npy", m_z)
 
        meta = {
            "dat_size": dataset_size,
            "index": sample_count,
            "pconn_dir": args.PCONNDIR,
            "n_templates": len(selected_pconns),
            "template_pconns": sorted(selected_pconns),
        }
        with open(f"{stem}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[INFO] wrote {stem}_meta.json")
 
 
if __name__ == "__main__":
    main()
