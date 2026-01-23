#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def upper_triangle_vec(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu].astype(np.float64, copy=False)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ridge.npy from haufe.csv (upper triangle vector)."
    )
    parser.add_argument("WRKDIR", type=str, help="Working directory")
    parser.add_argument("FILEDIR", type=str, help="Unused (compatibility)")
    # Accept optional 3rd positional so ridge.sh can pass 'default' without crashing
    parser.add_argument("PHENO", nargs="?", default=None, help="Optional phenotype label (positional)")

    # Also allow flag form if you ever want it
    parser.add_argument("--pheno", default=None, help="Optional phenotype label (flag form)")

    args = parser.parse_args()

    wrkdir = Path(args.WRKDIR)
    out_dir = wrkdir / "pwr_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # phenotype label resolution
    pheno = args.pheno if args.pheno is not None else args.PHENO
    if pheno is not None:
        pheno = str(pheno).strip()
        if pheno == "" or pheno.lower() == "default":
            pheno = None

    haufe_path = wrkdir / "haufe.csv"
    if not haufe_path.exists():
        print(f"[FATAL] haufe.csv not found at: {haufe_path}", file=sys.stderr)
        sys.exit(1)

    # Load haufe.csv (no header)
    haufe = pd.read_csv(haufe_path, header=None).to_numpy(dtype=np.float64, copy=False)

    if haufe.ndim != 2 or haufe.shape[0] != haufe.shape[1]:
        raise ValueError(f"Expected square matrix in haufe.csv; got {haufe.shape}")

    ridge_vec = upper_triangle_vec(haufe)

    ridge_path = out_dir / "ridge.npy"
    np.save(ridge_path, ridge_vec)
    print(f"[OK] wrote {ridge_path} (n={ridge_vec.size})")

    if pheno is not None:
        ridge_ph = out_dir / f"ridge_{pheno}.npy"
        np.save(ridge_ph, ridge_vec)
        print(f"[OK] wrote {ridge_ph} (pheno={pheno})")

    print("[DONE] ridge_model_generation.py complete.")


if __name__ == "__main__":
    main()
