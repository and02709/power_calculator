#!/usr/bin/env python3
"""
pwr_process_chunk_single_z.py — Step 2 (single-template mode) simulation worker.

Processes a contiguous chunk of rows from pwr_index_file.txt, simulating
brain connectivity data and phenotype vectors for each row. Called once per
SLURM array task by pwr_sub_python_single.sh.

Single-template mode:
    A single reference pconn file (--pconn1 / PCONNREF) is loaded once at
    startup and projected to PSD. Its eigendecomposition (EigSplit) is computed
    exactly once and reused for every simulated subject across all index rows
    in the chunk. No fresh pconn templates are drawn per row: the entire chunk
    shares one fixed eigenstructure.

    This contrasts with pwr_process_chunk_z.py (multi-template mode), which
    draws NUMTEMP fresh pconn templates per row and recomputes the
    eigendecomposition for each row. Here NUMTEMP is accepted as a positional
    placeholder (for argument-order compatibility with the shell wrapper) but
    is otherwise ignored.

Outputs per index row (named dat_size_<dataset>_index_<sample_count>.*):
    _cov.npy   — mean upper-triangle covariance vector across NREP subjects
    _cor.npy   — mean upper-triangle correlation vector across NREP subjects
    _z.npy     — mean upper-triangle Fisher-Z vector across NREP subjects
    _meta.json — provenance: dataset size, index, reference pconn path

Relationship to pwr_process_chunk_z.py:
    All helper functions (fisher_z_from_corr, vectorize_uplo, import_pconn,
    EigSplit, get_averaged_pconn, build_x_aug_from_rng, simulate_iteration)
    are identical between the two scripts. The sole structural difference is
    that this script performs the eigendecomposition once (outside the row
    loop) using the reference pconn, while pwr_process_chunk_z.py recomputes
    it inside the loop from freshly sampled templates.
"""

import os
import argparse
import json

import numpy as np
import pandas as pd
import nibabel as nib
from typing import Optional


def fisher_z_from_corr(R: np.ndarray, clip: float = 1e-7) -> np.ndarray:
    """
    Apply Fisher Z-transformation to a correlation matrix.

    Clips values to (-1+clip, 1-clip) before applying arctanh to avoid
    infinite values at ±1 (perfect correlations). The diagonal is zeroed
    after transformation because self-correlations are meaningless in the
    Z domain.

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

    Uses triu_indices with k=1 to exclude the diagonal. The resulting vector
    has length n*(n-1)/2 for an n×n matrix. Row-major ordering is preserved
    (elements taken left-to-right, top-to-bottom above the diagonal).

    Args:
        mat: Square 2-D array of shape (n, n).

    Returns:
        1-D array of length n*(n-1)/2.
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
        2. Symmetrise: M = 0.5 * (M + M^T), correcting asymmetry from
           rounding or file format conventions.
        3. Force diagonal to 1.0 (self-correlation = 1) if force_diag_one.
        4. Project to PSD cone if project_to_psd: eigendecompose, clip
           negative eigenvalues to psd_eps, reconstruct. Diagonal is
           re-forced to 1.0 after reconstruction if force_diag_one.

    Args:
        file:           Path to the .pconn.nii file.
        force_diag_one: Set diagonal to 1.0. Default True.
        project_to_psd: Project to PSD cone. Default True.
        psd_eps:        Eigenvalue floor for PSD projection. Defaults to
                        float32 machine epsilon (~1.2e-7) if None.

    Returns:
        Symmetric, optionally PSD-projected correlation matrix of shape (n, n).
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

    Pre-computes square roots of the positive and negative eigenvalue components
    separately so simulate_iteration() can cheaply scale random noise by each
    component without recomputing the decomposition per subject or per row.

    The split enables the two-source noise model in simulate_iteration():
        t = sqrt(x+) * (noise * sqrt_eig_pos) @ V^T
          + sqrt(x-) * (noise * sqrt_eig_neg) @ V^T

    where x+ and x- are subject-specific phenotype weights from
    build_x_aug_from_rng().

    Attributes:
        eigenvectors (np.ndarray):         V from eigh, shape (n, n).
        sqrt_eigenvalues_pos (np.ndarray): sqrt(max(w,  eps)), shape (n,).
        sqrt_eigenvalues_neg (np.ndarray): sqrt(max(-w, eps)), shape (n,).
    """

    def __init__(self, symmetric_matrix: np.ndarray):
        """
        Args:
            symmetric_matrix: Square symmetric PSD matrix, shape (n, n).
                               Typically the PSD-projected reference pconn.
        """
        eps = np.finfo(np.float32).eps
        w, v = np.linalg.eigh(symmetric_matrix)
        self.eigenvectors = v
        # Positive component: eigenvalues above zero (floored at eps to avoid sqrt(0))
        w_pos = np.maximum(w, eps)
        self.sqrt_eigenvalues_pos = np.sqrt(w_pos)
        # Negative component: reflects the signed deficit below zero
        w_neg = np.maximum(-w, eps)
        self.sqrt_eigenvalues_neg = np.sqrt(w_neg)


def build_x_aug_from_rng(n_sub: int, one_target: bool, rng: np.random.Generator) -> np.ndarray:
    """
    Build the augmented phenotype weight matrix for n_sub subjects.

    Generates subject-level phenotype scores and expands them into the
    positive/negative weight pairs consumed by simulate_iteration().
    Adding abs(min) shifts all values non-negative since they are passed
    to sqrt() as scaling factors.

    One-target mode (one_target=True):
        Draws one score per subject → expands to [x, -x].
        Shape: (n_sub, 2). eig2 is None in main(), columns 0-1 only used.

    Two-target mode (one_target=False):
        Draws two scores per subject → expands to [x0, -x0, x1, -x1].
        Shape: (n_sub, 4). Both eig1 and eig2 active in simulate_iteration().

    Args:
        n_sub:      Number of subjects (rows). Typically NREP.
        one_target: If True, generate a 2-column matrix; otherwise 4-column.
        rng:        numpy random Generator for reproducibility.

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

    For each subject, generates synthetic BOLD-like timeseries by scaling
    Gaussian noise with the eigenstructure of the template pconn, then computes
    covariance, correlation, and Fisher-Z matrices from those timeseries.
    Upper-triangle vectors are accumulated and averaged across all subjects.

    Timeseries generation (one-target mode, eig2=None):
        t = sqrt(x[i,0]) * (noise @ diag(sqrt_eig_pos)) @ V^T
          + sqrt(x[i,1]) * (noise @ diag(sqrt_eig_neg)) @ V^T

    Timeseries generation (two-target mode, eig2 provided):
        t += sqrt(x[i,2]) * (noise2 @ diag(eig2.sqrt_pos)) @ eig2.V^T
           + sqrt(x[i,3]) * (noise2 @ diag(eig2.sqrt_neg)) @ eig2.V^T

    Args:
        eig1:   EigSplit of the reference pconn matrix (computed once in main).
        eig2:   EigSplit of the secondary template (None for one-target mode).
                In single-template mode eig2=eig1 when --use_one_target is not
                set; both refer to the same decomposition object, not a copy.
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
        # Scale each node's noise by the sqrt-eigenvalue, then rotate into
        # node space via the eigenvector matrix. Positive and negative
        # components are weighted by sqrt(x_aug[i, 0/1]).
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
        C = np.cov(t, rowvar=False)           # Covariance:  (n_node, n_node)
        R = np.corrcoef(t, rowvar=False)      # Correlation: (n_node, n_node)
        Z = fisher_z_from_corr(R, clip=clip)  # Fisher-Z:    (n_node, n_node)

        # Accumulate upper-triangle vectors
        s_cov += vectorize_uplo(C)
        s_cor += vectorize_uplo(R)
        s_z   += vectorize_uplo(Z)

    # Return means across subjects
    return s_cov / n_sub, s_cor / n_sub, s_z / n_sub


def main():
    """
    Parse arguments, load the reference pconn once, iterate over the assigned
    chunk of index rows, and write outputs.

    Key difference from pwr_process_chunk_z.py:
        The reference pconn (--pconn1 / PCONNREF) is loaded and decomposed
        ONCE before the row loop. The resulting EigSplit object is reused for
        every index row and every simulated subject in the chunk. No fresh
        pconn templates are drawn inside the loop.

    For each row in the chunk (START..END inclusive):
        1. Read dataset_size and sample_count from pwr_index_file.txt.
        2. Build phenotype weights (x_aug) for NREP subjects using a fresh RNG.
        3. Simulate connectivity matrices via simulate_iteration() using the
           single pre-computed EigSplit.
        4. Save _cov.npy, _cor.npy, _z.npy, and _meta.json.

    Arguments (positional):
        WRKDIR      Root working directory; outputs go to WRKDIR/pwr_data/.
        START       First row index to process (1-based, inclusive).
        END         Last row index to process (1-based, inclusive).
        FILEDIR     Pipeline scripts directory (retained for argument-order
                    consistency with the shell wrapper; not used at runtime).
        PCONNDIR    Directory of subject .pconn.nii files (accepted for parser
                    compatibility with pwr_sub_python_single.sh; not used here
                    since the template is fixed to --pconn1).
        PCONNREF    Reference pconn path (accepted for parser compatibility;
                    superseded by --pconn1 when passed by the shell wrapper).
        NREP        Number of subjects to simulate per index row.
        NTIME       Positional timepoint count (present for parser compatibility;
                    overridden by --n_time when passed by pwr_sub_python_single.sh).
        UNUSED_ARG  Placeholder satisfying the NUMTEMP position in the shell
                    wrapper (pwr_sub_python_single.sh passes "placeholder"); ignored.

    Optional arguments:
        --n_time INT          Timepoints per subject. Overrides positional NTIME.
                              Default 1000; pwr_sub_python_single.sh passes 2000.
        --pconn1 PATH         Path to the single reference pconn used for the
                              shared eigendecomposition. When omitted, PCONNREF
                              is used as fallback.
        --use_one_target      Activate one-target mode (eig2=None, 2-col x_aug).
        --z_clip FLOAT        Clipping tolerance for fisher_z_from_corr. Default 1e-7.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("WRKDIR")
    ap.add_argument("START",      type=int)
    ap.add_argument("END",        type=int)
    ap.add_argument("FILEDIR")
    ap.add_argument("PCONNDIR")   # Accepted for compatibility; not used (template is fixed)
    ap.add_argument("PCONNREF")   # Fallback reference pconn if --pconn1 is not supplied
    ap.add_argument("NREP",       type=int)
    ap.add_argument("NTIME",      type=int)
    ap.add_argument("UNUSED_ARG")              # Satisfies NUMTEMP position in shell wrapper
    ap.add_argument("--n_time",   type=int, default=1000)
    ap.add_argument("--pconn1")                # Primary reference pconn path
    ap.add_argument("--use_one_target", action="store_true")
    ap.add_argument("--z_clip",   type=float, default=1e-7)
    args = ap.parse_args()

    n_time = args.n_time   # --n_time overrides positional NTIME

    out_dir    = os.path.join(args.WRKDIR, "pwr_data")
    index_path = os.path.join(out_dir, "pwr_index_file.txt")

    # ── Single eigendecomposition ─────────────────────────────────────────────
    # Resolve which pconn to use as the reference template.
    # --pconn1 takes precedence; fall back to the positional PCONNREF.
    ref_pconn_path = args.pconn1 if args.pconn1 else args.PCONNREF

    print(f"[INFO] Loading reference pconn: {ref_pconn_path}")
    ref_mat = import_pconn(ref_pconn_path, force_diag_one=True, project_to_psd=False)

    # Project to PSD with a tight floor (1e-12), then force diagonal to 1.
    # This mirrors the PSD step applied to the averaged matrix in
    # pwr_process_chunk_z.py, but is done once here rather than per row.
    w, v = np.linalg.eigh(ref_mat)
    ref_mat_psd = (v * np.maximum(w, 1e-12)) @ v.T
    np.fill_diagonal(ref_mat_psd, 1.0)

    # Compute the single EigSplit that will be shared across all rows and subjects.
    eig1 = EigSplit(ref_mat_psd)
    eig2 = None if args.use_one_target else eig1   # None → one-target; eig1 → two-target

    print(f"[INFO] Eigendecomposition complete. Matrix shape: {ref_mat_psd.shape}")

    # ── Row loop ──────────────────────────────────────────────────────────────
    index_file = pd.read_csv(index_path, sep=r"\s+", header=None, engine="python")
    chunk_rows = index_file.iloc[args.START - 1 : args.END]

    for i in range(len(chunk_rows)):
        sample_count = int(chunk_rows.iloc[i, 1])   # col 1: subject counter within dataset
        dataset_size = int(chunk_rows.iloc[i, 2])   # col 2: sample size for this row

        # Fresh unseeded RNG per row ensures independence between rows while
        # the eigenstructure (eig1/eig2) remains fixed across all rows.
        rng   = np.random.default_rng()
        x_aug = build_x_aug_from_rng(args.NREP, args.use_one_target, rng)

        m_cov, m_cor, m_z = simulate_iteration(eig1, eig2, x_aug, n_time, rng, args.z_clip)

        # Output filenames follow the convention expected by combine_data.py:
        #   dat_size_<dataset_size>_index_<sample_count>_{cov,cor,z}.npy
        stem = os.path.join(out_dir, f"dat_size_{dataset_size}_index_{sample_count}")
        np.save(f"{stem}_cov.npy", m_cov)
        np.save(f"{stem}_cor.npy", m_cor)
        np.save(f"{stem}_z.npy",   m_z)

        # Provenance record: captures the single reference pconn used for all rows
        meta = {
            "dat_size":       dataset_size,
            "index":          sample_count,
            "ref_pconn":      ref_pconn_path,
            "n_templates":    1,              # Always 1 in single-template mode
            "template_pconns": [ref_pconn_path],
        }
        with open(f"{stem}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[INFO] wrote {stem}_meta.json")


if __name__ == "__main__":
    main()
