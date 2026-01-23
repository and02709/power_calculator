import warnings
import numpy as np
import pandas as pd

# ---- Match R: warning("Running pwr_setup") ----
warnings.warn("Running pwr_setup")

# ---- Input ----
sample_sizes = np.array([100, 139, 194, 271, 378, 528, 736, 1027, 1433, 2000])

# ---- Replicate R behavior ----
repeated_vector = np.repeat(sample_sizes, sample_sizes)
count_vector = np.concatenate([np.arange(1, n + 1) for n in sample_sizes])

n_index = len(repeated_vector)
index = np.arange(1, n_index + 1)

# ---- Build DataFrame ----
df = pd.DataFrame({
    "index": index,
    "sample_count": count_vector,
    "dataset": repeated_vector
})

# ---- Write file (matches write.table in R) ----
df.to_csv(
    "pwr_index_file.txt",
    sep="\t",
    header=False,
    index=False
)
