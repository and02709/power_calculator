"""
pwr_setup.py — Step 1 of the power calculator pipeline.

Generates pwr_index_file.txt, which defines the full simulation index space
consumed by all downstream array jobs. Each row in the file represents one
simulation task: a specific subject drawn from a specific sample size.

The index space is constructed by repeating each sample size N by itself N
times, producing one row per subject per sample size. This means larger
sample sizes contribute proportionally more rows, reflecting the fact that
each subject within a sample-size condition is simulated independently.

Example (truncated):
    sample_sizes = [200, 286, ...]

    Row 1:    index=1,    sample_count=1,   dataset=200
    Row 2:    index=2,    sample_count=2,   dataset=200
    ...
    Row 200:  index=200,  sample_count=200, dataset=200
    Row 201:  index=201,  sample_count=1,   dataset=286
    ...

Total rows = sum(sample_sizes) = 16,164 for the default size ladder
(10 log-spaced sizes from 200 to 5000).

Output:
    pwr_index_file.txt  — tab-separated, no header, three columns:
        col 1: index        global 1-based row index (used by array tasks
                            to slice their assigned chunk)
        col 2: sample_count per-dataset subject counter (1..N for each N)
        col 3: dataset      sample size this row belongs to

Must be run from $WRKDIR/pwr_data/ (pwr_setup.sh handles the cd).
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.warn("Running pwr_setup")   # Mirrors the R original: warning("Running pwr_setup")

# ── Sample size ladder ────────────────────────────────────────────────────────
# Log-spaced values from SIZE_MIN to SIZE_MAX, chosen to produce a smooth power
# curve across the range of realistic neuroimaging study sizes.
#
# Default ladder (10 log steps, 200 -> 5000):
#   [200, 286, 409, 585, 836, 1196, 1710, 2445, 3497, 5000]
#   sum(sample_sizes) = 16,164 index rows -> 162 array tasks at CHUNK_SIZE=100.
#
# Override without editing this file by exporting SAMPLE_SIZES as a
# comma-separated list before PWR.sh runs, e.g.
#   export SAMPLE_SIZES="100,200,400,800"
SIZE_MIN = int(os.environ.get("SIZE_MIN", 200))
SIZE_MAX = int(os.environ.get("SIZE_MAX", 5000))
N_SIZES  = int(os.environ.get("N_SIZES", 10))

_sizes_env = os.environ.get("SAMPLE_SIZES", "").strip()
if _sizes_env:
    # Explicit ladder wins over the log-spaced default.
    sample_sizes = np.array(
        sorted({int(float(tok)) for tok in _sizes_env.split(",") if tok.strip()})
    )
else:
    # np.unique also sorts, and collapses any duplicates produced by rounding
    # when the requested ladder is dense relative to its range.
    sample_sizes = np.unique(
        np.round(
            np.logspace(np.log10(SIZE_MIN), np.log10(SIZE_MAX), N_SIZES)
        ).astype(int)
    )

print(f"[INFO] sample_sizes = {sample_sizes.tolist()}")
print(f"[INFO] total index rows = {int(sample_sizes.sum())}")

# ── Build index space ─────────────────────────────────────────────────────────
# np.repeat replicates each size N exactly N times, so the full repeated
# vector has sum(sample_sizes) = 16,164 entries for the default ladder.
# This mirrors the R idiom: rep(sample_sizes, sample_sizes)
repeated_vector = np.repeat(sample_sizes, sample_sizes)

# Per-dataset subject counter: resets to 1 at the start of each new size.
# Concatenates [1..200], [1..286], ..., [1..5000].
# Mirrors the R idiom: unlist(lapply(sample_sizes, seq_len))
count_vector = np.concatenate([np.arange(1, n + 1) for n in sample_sizes])

n_index = len(repeated_vector)          # Total rows = sum(sample_sizes)
index   = np.arange(1, n_index + 1)    # Global 1-based row index

# ── Assemble and write ────────────────────────────────────────────────────────
# Column order matches what pwr_process_chunk_z.py expects when it reads
# the file with pd.read_csv and accesses columns by position (iloc[:,0..2]).
df = pd.DataFrame({
    "index":        index,          # col 0: global row index
    "sample_count": count_vector,   # col 1: subject counter within dataset
    "dataset":      repeated_vector # col 2: sample size for this row
})

# Write tab-separated with no header and no row index, matching the R idiom:
#   write.table(df, "pwr_index_file.txt", sep="\t", row.names=FALSE, col.names=FALSE)
df.to_csv(
    "pwr_index_file.txt",
    sep="\t",
    header=False,
    index=False
)
