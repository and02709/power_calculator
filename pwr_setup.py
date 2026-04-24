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
    sample_sizes = [100, 139, ...]

    Row 1:    index=1,    sample_count=1,   dataset=100
    Row 2:    index=2,    sample_count=2,   dataset=100
    ...
    Row 100:  index=100,  sample_count=100, dataset=100
    Row 101:  index=101,  sample_count=1,   dataset=139
    ...

Total rows = sum(sample_sizes) = 6,806 for the default size ladder.

Output:
    pwr_index_file.txt  — tab-separated, no header, three columns:
        col 1: index        global 1-based row index (used by array tasks
                            to slice their assigned chunk)
        col 2: sample_count per-dataset subject counter (1..N for each N)
        col 3: dataset      sample size this row belongs to

Must be run from $WRKDIR/pwr_data/ (pwr_setup.sh handles the cd).
"""

import warnings
import numpy as np
import pandas as pd

warnings.warn("Running pwr_setup")   # Mirrors the R original: warning("Running pwr_setup")

# ── Sample size ladder ────────────────────────────────────────────────────────
# Log-spaced values from 100 to 2000, chosen to produce a smooth power curve
# across the range of realistic neuroimaging study sizes.
sample_sizes = np.array([100, 139, 194, 271, 378, 528, 736, 1027, 1433, 2000])

# ── Build index space ─────────────────────────────────────────────────────────
# np.repeat replicates each size N exactly N times, so the full repeated
# vector has sum(sample_sizes) = 6,806 entries for the default ladder.
# This mirrors the R idiom: rep(sample_sizes, sample_sizes)
repeated_vector = np.repeat(sample_sizes, sample_sizes)

# Per-dataset subject counter: resets to 1 at the start of each new size.
# Concatenates [1..100], [1..139], ..., [1..2000].
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
