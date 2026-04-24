#!/usr/bin/env python3
"""
final_data.py — Step 8 (terminal step) of the power calculator pipeline.

Aggregates all per-fold R² values produced by Step 7 (cv.py) into the
final power curve outputs. This is the last compute step — its outputs
are what the power calculator reports.

Processing flow:
    1. Build pconn_template_lookup.csv — a provenance table mapping each
       simulation row to its source pconn template file(s), derived from
       the _meta.json files written by Step 2.
    2. Collect all data_<size>_fold_<k>_cvr2.npy files from pwr_data/.
    3. Parse (size, fold) from each filename and read the scalar R² value.
    4. Write metrics_data — the full per-fold R² table (one row per file).
    5. Compute per-size mean and SD of R² across folds.
    6. Write metrics_summary — the power curve table (one row per size).
    7. Plot the power curve as mean R² ± SD vs sample size.

Outputs (written to WRKDIR/):
    pconn_template_lookup.csv  — provenance: simulation → pconn template paths
    metrics_data.{pkl,csv}     — per-fold R² table: file_list, size, fold, metrics
    metrics_summary.{pkl,csv}  — power curve: size, mean_metric, sd_metric
    mean_metric_by_size.png    — power curve plot (R² ± SD vs sample size)

Output format is controlled by --out_format (default: pkl for downstream
compatibility with any R-based consumers; use csv for human-readable output).
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_scalar_metric(path: Path, metric_ext: str) -> float:
    """
    Read a single scalar R² value from a metric file.

    Supports three formats to accommodate different pipeline configurations:
        npy  — numpy scalar or 1-element array (default; written by cv.py)
        txt  — plain text file containing a single float string
        pkl  — pickled Python float or numeric scalar

    The .reshape(-1)[0] pattern for npy handles both 0-D arrays (np.save
    of a scalar) and 1-element 1-D arrays without requiring a shape check.

    Args:
        path:       Path to the metric file.
        metric_ext: File extension without the dot ("npy", "txt", or "pkl").

    Returns:
        Scalar float R² value.

    Raises:
        ValueError: If metric_ext is not one of the supported formats.
    """
    if metric_ext == "npy":
        arr = np.load(str(path))
        # reshape(-1)[0] handles both 0-D scalars and 1-element 1-D arrays
        return float(np.asarray(arr).reshape(-1)[0])

    if metric_ext == "txt":
        with open(path, "r") as f:
            return float(f.read().strip())

    if metric_ext == "pkl":
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return float(obj)

    raise ValueError(f"Unsupported metric_ext: {metric_ext}")


def build_pconn_lookup(pwr_dir: Path) -> pd.DataFrame:
    """
    Scan pwr_dir for all _meta.json files and build a wide-format provenance table.

    Each row corresponds to one simulation (one index row from pwr_index_file.txt).
    Template pconn paths are spread across columns pconn_1, pconn_2, ..., pconn_N,
    where N is the maximum number of templates used by any single simulation.
    Rows with fewer templates than the maximum are padded with empty strings so
    the DataFrame has uniform column width.

    This is the final_data.py counterpart to the pconn_template_lookup.csv written
    by combine_data.py. Both produce the same schema; final_data.py writes a copy
    to WRKDIR/ (one level above pwr_data/) for easier access alongside the other
    pipeline outputs.

    Args:
        pwr_dir: pwr_data/ directory containing dat_size_*_index_*_meta.json files.

    Returns:
        Wide-format DataFrame sorted by stem, or an empty DataFrame if no
        _meta.json files are found.
    """
    meta_files = sorted(pwr_dir.glob("dat_size_*_index_*_meta.json"))

    if not meta_files:
        return pd.DataFrame()

    rows = []
    max_pconns = 0

    for mf in meta_files:
        with open(mf, "r") as f:
            meta = json.load(f)
        stem   = mf.name.replace("_meta.json", "")   # e.g. "dat_size_100_index_1"
        pconns = meta.get("template_pconns", [])
        rows.append({"stem": stem, "pconns": pconns})
        max_pconns = max(max_pconns, len(pconns))

    # Build wide format: one column per template slot across all simulations.
    # Simulations with fewer templates than max_pconns are padded with "".
    records = []
    for row in rows:
        record = {"stem": row["stem"]}
        for j, p in enumerate(row["pconns"], start=1):
            record[f"pconn_{j}"] = p
        # Pad missing slots so every row has the same column count
        for j in range(len(row["pconns"]) + 1, max_pconns + 1):
            record[f"pconn_{j}"] = ""
        records.append(record)

    df = pd.DataFrame(records).sort_values("stem").reset_index(drop=True)
    return df


def main():
    """
    Aggregate per-fold R² values and write the final power curve outputs.

    Arguments:
        WRKDIR   Root working directory. pwr_data/ is read for inputs;
                 all summary outputs are written directly to WRKDIR/.
        FILEDIR  Accepted for argument-order consistency with other pipeline
                 scripts; not used at runtime.

    Optional arguments:
        --metric_ext {npy,pkl,txt}   Extension of metric files to read.
                                     Default: npy (written by cv.py).
                                     Use pkl or txt if cv.py was configured
                                     to write a different format.
        --out_format {pkl,csv}       Output format for metrics_data and
                                     metrics_summary. Default: pkl for
                                     downstream compatibility. Use csv for
                                     human-readable output or R import.
    """
    parser = argparse.ArgumentParser(description="final_data (Python-native)")
    parser.add_argument("WRKDIR",  type=str)
    parser.add_argument("FILEDIR", type=str, help="Unused (kept for compatibility)")
    parser.add_argument(
        "--metric_ext", choices=["npy", "pkl", "txt"], default="npy",
        help="Extension of metric files to read (default: npy)",
    )
    parser.add_argument(
        "--out_format", choices=["pkl", "csv"], default="pkl",
        help="Save outputs as pickle (default) or csv",
    )
    args = parser.parse_args()

    WRKDIR = Path(args.WRKDIR)
    warnings.warn("Running final_data")   # Mirrors the R original convention

    pwr_dir = WRKDIR / "pwr_data"
    if not pwr_dir.exists():
        raise SystemExit(f"[FATAL] Missing directory: {pwr_dir}")

    # ── Provenance: pconn template lookup ────────────────────────────────────
    # Written to WRKDIR/ (not pwr_data/) so it sits alongside the other final
    # outputs and is easy to locate without descending into pwr_data/.
    # combine_data.py writes an equivalent file to pwr_data/ during Step 3;
    # this copy is the definitive version in the final output directory.
    df_pconn = build_pconn_lookup(pwr_dir)
    if df_pconn.empty:
        print("[WARN] No _meta.json files found in pwr_data — skipping pconn lookup CSV")
    else:
        pconn_lookup_path = WRKDIR / "pconn_template_lookup.csv"
        df_pconn.to_csv(pconn_lookup_path, index=False)
        print(f"[OK] Saved pconn lookup: {pconn_lookup_path} ({len(df_pconn)} rows)")

    # ── Collect metric files ──────────────────────────────────────────────────
    # Glob for all cvr2 files matching the configured extension.
    # Files whose names don't match data_<size>_fold_<k>_cvr2.<ext> are
    # included in the file list but get NaN for size/fold after regex parsing,
    # and are still included in metrics_data for traceability.
    metric_suffix = f"_cvr2.{args.metric_ext}"
    file_list = sorted([
        p for p in pwr_dir.iterdir()
        if p.is_file() and p.name.endswith(metric_suffix)
    ])
    n_files = len(file_list)

    if n_files == 0:
        raise SystemExit(
            f"[FATAL] No matching metric files found in {pwr_dir} ending with {metric_suffix}"
        )

    # ── Parse (size, fold) from filenames ─────────────────────────────────────
    # Filenames follow the pattern: data_<size>_fold_<fold>_cvr2.<ext>
    # Files that don't match get NaN for size and fold; they appear in
    # metrics_data but are grouped under NaN in the summary (dropna=False
    # in groupby preserves them for visibility rather than silently dropping).
    sizes = []
    folds = []
    for p in file_list:
        m = re.match(r"^data_(\d+)_fold_(\d+)_cvr2\.", p.name)
        if m:
            sizes.append(float(m.group(1)))
            folds.append(float(m.group(2)))
        else:
            sizes.append(np.nan)
            folds.append(np.nan)

    # ── Build per-fold R² table ───────────────────────────────────────────────
    df = pd.DataFrame({
        "file_list": [str(p) for p in file_list],
        "size":      sizes,
        "fold":      folds,
        "metrics":   0.0    # Placeholder; overwritten by read_scalar_metric below
    })

    for i, p in enumerate(file_list):
        df.loc[i, "metrics"] = read_scalar_metric(p, args.metric_ext)

    # ── Write metrics_data ────────────────────────────────────────────────────
    # Full per-fold table: one row per cvr2 file.
    # pkl preserves dtypes exactly for downstream Python consumers;
    # csv is human-readable and importable by R read.csv().
    metrics_data_path = WRKDIR / (
        "metrics_data.pkl" if args.out_format == "pkl" else "metrics_data.csv"
    )
    if args.out_format == "pkl":
        df.to_pickle(metrics_data_path)
    else:
        df.to_csv(metrics_data_path, index=False)

    # ── Compute power curve summary ───────────────────────────────────────────
    # Aggregate per-fold R² values across folds for each sample size.
    # dropna=False preserves any NaN-size rows in the summary rather than
    # silently dropping malformed filenames, making data quality issues visible.
    # std() returns NaN for sizes with only one fold — expected for KFOLDS=1.
    df_summary = (
        df.groupby("size", dropna=False)["metrics"]
          .agg(mean_metric="mean", sd_metric="std")
          .reset_index()
          .sort_values("size")
    )

    # ── Write metrics_summary ─────────────────────────────────────────────────
    # Power curve table: one row per sample size with mean R² and SD.
    # This is the primary deliverable of the entire pipeline.
    metrics_summary_path = WRKDIR / (
        "metrics_summary.pkl" if args.out_format == "pkl" else "metrics_summary.csv"
    )
    if args.out_format == "pkl":
        df_summary.to_pickle(metrics_summary_path)
    else:
        df_summary.to_csv(metrics_summary_path, index=False)

    # ── Plot power curve ──────────────────────────────────────────────────────
    # Mean R² ± SD vs sample size. Error bars show fold-to-fold variability.
    # Integer sizes are formatted without decimal points in tick labels.
    # Saved to WRKDIR/ at 300 dpi for publication-quality output.
    x_labels = [
        str(int(s)) if float(s).is_integer() else str(s)
        for s in df_summary["size"].tolist()
    ]
    x    = np.arange(len(x_labels))
    y    = df_summary["mean_metric"].to_numpy()
    yerr = df_summary["sd_metric"].to_numpy()

    plt.figure(figsize=(6, 4))
    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
    plt.xticks(x, x_labels, rotation=0)
    plt.xlabel("Size")
    plt.ylabel("Mean CV Metric")
    plt.title("Mean CV Metric by Size")
    plt.tight_layout()

    plot_path = WRKDIR / "mean_metric_by_size.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()   # Release figure memory; important in batch contexts with many plots

    print(f"[OK] Read {n_files} metric files ({metric_suffix})")
    print(f"[OK] Saved: {metrics_data_path}")
    print(f"[OK] Saved: {metrics_summary_path}")
    print(f"[OK] Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
