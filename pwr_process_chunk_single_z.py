#!/usr/bin/env python3
"""
pwr_process_chunk_single_z.py — Step 2 (single-template mode) simulation worker.

Processes a contiguous chunk of rows from pwr_index_file.txt, simulating
brain connectivity data and phenotype vectors for each row. Called once per
SLURM array task by pwr_sub_python_single.sh.

Single-template mode (--use_one_target):
    For each index row, one pconn template is drawn fresh at random from
    PCONNDIR. This template is projected to PSD and decomposed via EigSplit,
    then used to generate synthetic timeseries for all NREP subjects. Drawing
    a new template per row ensures simulations are independent across rows.

Multi-template mode (default, --use_one_target omitted):
    Same flow, but eig2 is set to eig1, activating the two-target branch of
    simulate_iteration() which mixes two independent noise sources per subject.

Outputs per index row (named dat_size_<dataset>_index_<sample_count>.*):
    _cov.npy   — mean upper-triangle covariance vector across NREP subjects
    _cor.npy   — mean upper-triangle correlation vector across NREP subjects
    _z.npy     — mean upper-triangle Fisher-Z vector across NREP subjects
    _meta.json — provenance: dataset size, index, template pconn paths

This is the single-template counterpart to pwr_process_chunk_z.py.
The key structural difference is that the template is re-drawn inside the
per-row loop rather than once before it.
"""

import os
import argparse
import json
import random
from typing import Optional

import numpy as np
import pandas as pd
import nibabel as nib


def fisher_z_from_corr(R: np.ndarray, clip: float = 1e-7) -> np.ndarray:
    """
    Apply Fisher Z-transformation to a correlation matrix.

    Clips values to (-1+clip, 1-clip) before applying arctanh to avoid
    infinite values at ±1 (perfect correlations). The diagonal is zeroed
    out after transformation because self-correlations are not meaningful
    in the Z domain.

    Args:
        R:    Symmetric correlation matrix with values in [-1, 1].
        clip: Half-width of the exclusion zone around ±1. Default 1e-7.

    Returns:
        Z-transformed matrix of same shape as R, with zeros on the diagonal.
    """
    R = np.asarray(R, dtype=np.float64)
    R = np.clip(R, -1.0 + clip, 1.0 - clip)
    Z = np.arctanh(R)
    np.fill_diagonal(Z, 0.0)
    return Z


def vectorize_uplo(mat: np.ndarray) -> np.ndarray:
    """
    Extract the strict upper triangle of a square matrix as a 1-D vector.

    Uses numpy's triu_indices with k=1 (offset of 1 excludes the diagonal).
    The resulting vector has length n*(n-1)/2 for an n×n matrix.
    Row-major ordering is preserved (i.e. elements are taken left-to-right,
    top-to-bottom above the diagonal).

    Args:
        mat: Square 2-D array of shape (n, n).

    Returns:
        1-D array of length n*(n-1)/2 containing the upper-triangle elements.
    """
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    return mat[iu]


def import_pconn(
    file: str,
    force_diag_one: bool = True,
    project_to_psd: bool = True,
    psd_eps: Optional[float] = None,
) -> np.ndarray:
    """
    Load a .pconn.nii file and return a valid symmetric correlation matrix.

    Processing steps applied in order:
        1. Load via nibabel and cast to float64.
        2. Symmetrise: M = 0.5 * (M + M^T), correcting any asymmetry from
           rounding or file format conventions.
        3. Force diagonal to 1.0 (self-correlation = 1) if force_diag_one.
        4. Project to the positive semi-definite (PSD) cone if project_to_psd:
           eigendecompose, clip negative eigenvalues to psd_eps, reconstruct.
           Diagonal is re-forced to 1.0 after reconstruction if force_diag_one.

    Args:
        file:           Path to the .pconn.nii file.
        force_diag_one: Set diagonal entries to 1.0. Default True.
        project_to_psd: Project to PSD cone to ensure numerical validity for
                        downstream Cholesky/eigh operations. Default True.
        psd_eps:        Floor for eigenvalues during PSD projection. Defaults
                        to float32 machine epsilon (~1.2e-7) if None.

    Returns:
        Symmetric, PSD (if requested) correlation matrix of shape (n, n).
    """
    img = nib.load(file)
    mat = np.squeeze(np.asanyarray(img.get_fdata())).astype(np.float64, copy=False)

    mat_sym = 0.5 * (mat + mat.T)          # Enforce exact symmetry

    if force_diag_one:
        np.fill_diagonal(mat_sym, 1.0)

    if project_to_psd:
        if psd_eps is None:
            psd_eps = float(np.finfo(np.float32).eps)
        w, v = np.linalg.eigh(mat_sym)
        w_clip = np.maximum(w, psd_eps)     # Clip negative eigenvalues to floor
        mat_psd = (v * w_clip) @ v.T        # Reconstruct: V * diag(w_clip) * V^T
        mat_sym = 0.5 * (mat_psd + mat_psd.T)
        if force_diag_one:
            np.fill_diagonal(mat_sym, 1.0)

    return mat_sym


class EigSplit:
    """
    Eigendecomposition of a symmetric matrix split into positive and negative parts.

    Pre-computes the square roots of the positive and negative eigenvalue
    components separately so that simulate_iteration() can cheaply scale
    random noise by each component without recomputing the decomposition
    per subject or per simulation.

    The split enables the two-source noise model in simulate_iteration():
        t = sqrt(x+) * (noise * sqrt_eig_pos) @ V^T
          + sqrt(x-) * (noise * sqrt_eig_neg) @ V^T

    where x+ and x- are subject-specific phenotype weights drawn from
    build_x_aug_from_rng().

    Attributes:
        eigenvectors (np.ndarray):        V from eigh, shape (n, n).
        sqrt_eigenvalues_pos (np.ndarray): sqrt(max(w,  eps)), shape (n,).
        sqrt_eigenvalues_neg (np.ndarray): sqrt(max(-w, eps)), shape (n,).
    """

    def __init__(self, symmetric_matrix: np.ndarray):
        """
        Args:
            symmetric_matrix: Square symmetric PSD matrix, shape (n, n).
                               Typically the averaged pconn after PSD projection.
        """
        eps = np.finfo(np.float32).eps
        w, v = np.linalg.eigh(symmetric_matrix)
        self.eigenvectors = v
        # Positive component: eigenvalues above zero (floor at eps to avoid sqrt(0))
        w_pos = np.maximum(w, eps)
        self.sqrt_eigenvalues_pos = np.sqrt(w_pos)
        # Negative component: reflects the signed deficit below zero
        w_neg = np.maximum(-w, eps)
        self.sqrt_eigenvalues_neg = np.sqrt(w_neg)


def get_averaged_pconn(pconn_dir: str, num_to_select: int):
    """
    Randomly sample pconn files from a directory and return their mean matrix.

    Selects num_to_select files at random (without replacement) from all
    .pconn.nii files in pconn_dir, loads each without PSD projection (raw
    symmetrised matrices), and returns their element-wise mean.

    In single-template mode num_to_select=1, so the "average" is just one
    randomly chosen subject's connectivity matrix.

    Args:
        pconn_dir:      Directory containing .pconn.nii files.
        num_to_select:  Number of files to sample. Clamped to the number of
                        available files if num_to_select exceeds that count.

    Returns:
        Tuple of:
            avg_mat (np.ndarray): Mean matrix of shape (n, n).
            selected_files (list[str]): Paths of the sampled files (for provenance).

    Raises:
        FileNotFoundError: If no .pconn.nii files are found in pconn_dir.
    """
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
    # project_to_psd=False: raw symmetrised matrices are averaged first;
    # PSD projection is applied to the mean in main() after this returns.
    mats = [import_pconn(f, project_to_psd=False) for f in selected_files]
    return np.mean(mats, axis=0), selected_files


def build_x_aug_from_rng(n_sub: int, one_target: bool, rng: np.random.Generator) -> np.ndarray:
    """
    Build the augmented phenotype weight matrix for n_sub subjects.

    Generates subject-level phenotype scores and expands them into the
    positive/negative weight pairs consumed by simulate_iteration().
    Adding abs(min) shifts all values to be non-negative, since they are
    later passed to sqrt() as scaling factors.

    One-target mode (one_target=True):
        Draws one score per subject → expands to [x, -x].
        Shape: (n_sub, 2).
        eig2 is set to None in main(), so only columns 0-1 are used.

    Two-target mode (one_target=False):
        Draws two scores per subject → expands to [x0, -x0, x1, -x1].
        Shape: (n_sub, 4).
        Both eig1 and eig2 are active in simulate_iteration().

    Args:
        n_sub:      Number of subjects (rows). Typically NREP.
        one_target: If True, generate a 2-column matrix; otherwise 4-column.
        rng:        Seeded or default numpy random Generator for reproducibility.

    Returns:
        Non-negative array of shape (n_sub, 2) or (n_sub, 4).
    """
    need_cols = 1 if one_target else 2
    x_sub = rng.standard_normal((n_sub, need_cols))
    if one_target:
        x_aug = np.column_stack([x_sub[:, 0], -x_sub[:, 0]])
    else:
        x_aug = np.column_stack([x_sub[:, 0], -x_sub[:, 0], x_sub[:, 1], -x_sub[:, 1]])
    # Shift to non-negative: all values used as sqrt() arguments in simulate_iteration()
    return x_aug + abs(float(np.min(x_aug)))


def simulate_iteration(
    eig1: EigSplit,
    eig2: Optional[EigSplit],
    x_aug: np.ndarray,
    n_time: int,
    rng: np.random.Generator,
    clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate connectivity matrices for n_sub subjects and return mean edge vectors.

    For each subject i, generates synthetic BOLD-like timeseries by scaling
    Gaussian noise with the eigenstructure of the template pconn matrix, then
    computes covariance, correlation, and Fisher-Z matrices from the timeseries.
    Upper-triangle vectors are accumulated and averaged across all subjects.

    Timeseries generation (one-target mode, eig2=None):
        t = sqrt(x[i,0]) * (noise @ diag(sqrt_eig_pos)) @ V^T
          + sqrt(x[i,1]) * (noise @ diag(sqrt_eig_neg)) @ V^T

    Timeseries generation (two-target mode, eig2 provided):
        t += sqrt(x[i,2]) * (noise2 @ diag(eig2.sqrt_pos)) @ eig2.V^T
           + sqrt(x[i,3]) * (noise2 @ diag(eig2.sqrt_neg)) @ eig2.V^T

    The positive/negative eigenvalue split encodes both the connectivity
    structure (positive) and its complement (negative) so that the phenotype
    weights x_aug control how strongly each component contributes per subject.

    Args:
        eig1:   EigSplit of the primary template matrix.
        eig2:   EigSplit of the secondary template (None for one-target mode).
        x_aug:  Phenotype weight matrix, shape (n_sub, 2) or (n_sub, 4).
        n_time: Number of timepoints to simulate per subject.
        rng:    numpy random Generator (shared across subjects within one row).
        clip:   Clipping tolerance passed to fisher_z_from_corr.

    Returns:
        Tuple of three 1-D arrays, each of length n*(n-1)/2:
            m_cov: Mean upper-triangle covariance vector across subjects.
            m_cor: Mean upper-triangle correlation vector across subjects.
            m_z:   Mean upper-triangle Fisher-Z vector across subjects.
    """
    n_sub  = x_aug.shape[0]
    n_node = eig1.eigenvectors.shape[0]
    n_edge = (n_node * (n_node - 1)) // 2

    # Accumulators for the three edge-level statistics
    s_cov, s_cor, s_z = np.zeros(n_edge), np.zeros(n_edge), np.zeros(n_edge)

    for i in range(n_sub):
        print(i)   # Progress indicator — one line per subject per chunk row

        # Primary noise source: shape (n_time, n_node)
        trand = rng.standard_normal((n_time, n_node))
        # Scale each node's noise by the corresponding sqrt-eigenvalue, then
        # rotate into the original node space via the eigenvector matrix.
        # The positive and negative components are weighted by sqrt(x_aug[i, 0/1]).
        t = (
            np.sqrt(x_aug[i, 0]) * (trand * eig1.sqrt_eigenvalues_pos).dot(eig1.eigenvectors.T)
            + np.sqrt(x_aug[i, 1]) * (trand * eig1.sqrt_eigenvalues_neg).dot(eig1.eigenvectors.T)
        )

        # Secondary noise source (two-target mode only)
        if eig2 is not None:
            trand2 = rng.standard_normal((n_time, n_node))
            t += (
                np.sqrt(x_aug[i, 2]) * (trand2 * eig2.sqrt_eigenvalues_pos).dot(eig2.eigenvectors.T)
                + np.sqrt(x_aug[i, 3]) * (trand2 * eig2.sqrt_eigenvalues_neg).dot(eig2.eigenvectors.T)
            )

        # Compute connectivity matrices from the simulated timeseries
        C = np.cov(t, rowvar=False)           # Covariance:   (n_node, n_node)
        R = np.corrcoef(t, rowvar=False)      # Correlation:  (n_node, n_node)
        Z = fisher_z_from_corr(R, clip=clip)  # Fisher-Z:     (n_node, n_node)

        # Accumulate upper-triangle vectors
        s_cov += vectorize_uplo(C)
        s_cor += vectorize_uplo(R)
        s_z   += vectorize_uplo(Z)

    # Return means across subjects
    return s_cov / n_sub, s_cor / n_sub, s_z / n_sub


def main():
    """
    Parse arguments, iterate over the assigned chunk of index rows, and write outputs.

    For each row in the chunk (START..END inclusive):
        1. Read dataset_size and sample_count from pwr_index_file.txt.
        2. Draw one fresh pconn template at random from PCONNDIR.
        3. Project the template to PSD and decompose via EigSplit.
        4. Build phenotype weights (x_aug) for NREP subjects.
        5. Simulate connectivity matrices via simulate_iteration().
        6. Save _cov.npy, _cor.npy, _z.npy, and _meta.json.

    Arguments (positional):
        WRKDIR      Root working directory; outputs go to WRKDIR/pwr_data/.
        START       First row index to process (1-based, inclusive).
        END         Last row index to process (1-based, inclusive).
        FILEDIR     Pipeline scripts directory (unused at runtime, kept for
                    consistency with pwr_sub_python_single.sh argument order).
        PCONNDIR    Directory of subject .pconn.nii files.
        PCONNREF    Reference pconn path (passed by pwr_sub_python_single.sh
                    via --pconn1 but not used here; kept for parser compatibility).
        NREP        Number of subjects to simulate per index row.
        NTIME       Positional timepoint arg (overridden by --n_time if provided).
        UNUSED_ARG  Placeholder satisfying the NUMTEMP position expected by the
                    shell wrapper; ignored entirely at runtime.

    Optional arguments:
        --n_time INT          Timepoints per subject. Overrides positional NTIME.
                              Default 1000; pwr_sub_python_single.sh passes 2000.
        --pconn1 PATH         Reference pconn (accepted for parser compatibility;
                              template is always drawn randomly from PCONNDIR).
        --use_one_target      Activate one-target mode (eig2=None, 2-col x_aug).
        --z_clip FLOAT        Clipping tolerance for fisher_z_from_corr. Default 1e-7.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("WRKDIR")
    ap.add_argument("START",      type=int)
    ap.add_argument("END",        type=int)
    ap.add_argument("FILEDIR")
    ap.add_argument("PCONNDIR")
    ap.add_argument("PCONNREF")
    ap.add_argument("NREP",       type=int)
    ap.add_argument("NTIME",      type=int)
    ap.add_argument("UNUSED_ARG")                              # Satisfies NUMTEMP position in shell wrapper
    ap.add_argument("--n_time",   type=int, default=1000)
    ap.add_argument("--pconn1")                                # Accepted for compatibility; not used
    ap.add_argument("--use_one_target", action="store_true")
    ap.add_argument("--z_clip",   type=float, default=1e-7)
    args = ap.parse_args()

    n_time = args.NTIME   # Positional NTIME; may be overridden by --n_time in some call sites

    out_dir    = os.path.join(args.WRKDIR, "pwr_data")
    index_path = os.path.join(out_dir, "pwr_index_file.txt")

    # Read the full index file and slice to the assigned chunk (1-based → 0-based)
    index_file = pd.read_csv(index_path, sep=r"\s+", header=None, engine="python")
    chunk_rows = index_file.iloc[args.START - 1 : args.END]

    for i in range(len(chunk_rows)):
        sample_count = int(chunk_rows.iloc[i, 1])   # col 1: subject counter within dataset
        dataset_size = int(chunk_rows.iloc[i, 2])   # col 2: sample size for this row

        # Draw a fresh template per row so simulations are independent across rows.
        # num_to_select=1 in single-template mode; the "average" is just one matrix.
        avg_mat, selected_pconns = get_averaged_pconn(args.PCONNDIR, 1)

        # Project the raw averaged matrix to PSD with a tighter floor (1e-12)
        # than import_pconn's default (float32 eps ~1.2e-7), then force diagonal.
        w, v = np.linalg.eigh(avg_mat)
        avg_mat_psd = (v * np.maximum(w, 1e-12)) @ v.T
        np.fill_diagonal(avg_mat_psd, 1.0)

        eig1 = EigSplit(avg_mat_psd)
        eig2 = None if args.use_one_target else eig1   # None → one-target; eig1 → two-target

        rng   = np.random.default_rng()   # Fresh unseeded RNG per row ensures independence
        x_aug = build_x_aug_from_rng(args.NREP, args.use_one_target, rng)

        m_cov, m_cor, m_z = simulate_iteration(eig1, eig2, x_aug, args.n_time, rng, args.z_clip)

        # Output filenames follow the convention expected by combine_data.py:
        #   dat_size_<dataset_size>_index_<sample_count>_{cov,cor,z}.npy
        stem = os.path.join(out_dir, f"dat_size_{dataset_size}_index_{sample_count}")
        np.save(f"{stem}_cov.npy", m_cov)
        np.save(f"{stem}_cor.npy", m_cor)
        np.save(f"{stem}_z.npy",   m_z)

        # Provenance record: captures which templates were used for this row
        meta = {
            "dat_size":        dataset_size,
            "index":           sample_count,
            "pconn_dir":       args.PCONNDIR,
            "n_templates":     len(selected_pconns),
            "template_pconns": sorted(selected_pconns),
        }
        with open(f"{stem}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[INFO] wrote {stem}_meta.json")


if __name__ == "__main__":
    main()
