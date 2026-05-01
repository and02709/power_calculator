#!/usr/bin/env python3
"""
final_data.py — Step 6 (terminal step) of the power calculator pipeline.

Aggregates all per-fold CV scores produced by cv.py into the final power
curve outputs.  This is the last compute step — its outputs are what the
power calculator reports.

Processing flow
---------------
1. Build pconn_template_lookup.csv — provenance table mapping each simulation
   row to its source pconn template file(s), derived from the _meta.json
   files written by Step 2.
2. Collect all cv_results_size<N>_<model>.csv files from pwr_data/.
   These are written by cv.py (v2) and contain per-fold train+test scores
   for RMSE, MAE, and R² across all outer CV folds.
3. Parse sample size from each filename, read the CSV, and extract the
   test R² column (test_R2) as the primary metric.
4. Write metrics_data — the full per-fold R² table (one row per fold per size).
5. Compute per-size mean and SD of R² across folds.
6. Write metrics_summary — the power curve table (one row per size).
7. Plot the power curve as mean R² ± SD vs sample size.

Outputs (written to WRKDIR/)
-----------------------------
  pconn_template_lookup.csv  — provenance: simulation → pconn template paths
  metrics_data.{pkl,csv}     — per-fold R² table: file_list, size, fold, metrics
  metrics_summary.{pkl,csv}  — power curve: size, mean_metric, sd_metric
  mean_metric_by_size.png    — power curve plot (R² ± SD vs sample size)
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_pconn_lookup(pwr_dir: Path) -> pd.DataFrame:
    """
    Scan pwr_dir for all _meta.json files and build a wide-format provenance table.

    Each row corresponds to one simulation (one index row from pwr_index_file.txt).
    Template pconn paths are spread across columns pconn_1, pconn_2, ..., pconn_N,
    where N is the maximum number of templates used by any single simulation.

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
        stem   = mf.name.replace("_meta.json", "")
        pconns = meta.get("template_pconns", [])
        rows.append({"stem": stem, "pconns": pconns})
        max_pconns = max(max_pconns, len(pconns))

    records = []
    for row in rows:
        record = {"stem": row["stem"]}
        for j, p in enumerate(row["pconns"], start=1):
            record[f"pconn_{j}"] = p
        for j in range(len(row["pconns"]) + 1, max_pconns + 1):
            record[f"pconn_{j}"] = ""
        records.append(record)

    df = pd.DataFrame(records).sort_values("stem").reset_index(drop=True)
    return df


def collect_cv_results(pwr_dir: Path, model: str) -> pd.DataFrame:
    """
    Collect all cv_results_size<N>_<model>.csv files and build a unified
    per-fold R² table matching the schema expected by the rest of final_data.py.

    Each cv_results CSV has one row per outer fold with columns including
    test_R2, train_R2, test_RMSE, train_RMSE, test_MAE, train_MAE.

    The returned DataFrame has columns:
        file_list : str   — source CSV path
        size      : float — sample size parsed from filename
        fold      : float — 0-based fold index within that size
        metrics   : float — test_R2 value for that fold

    Args:
        pwr_dir : Path to pwr_data/ directory.
        model   : Model name used to glob cv_results_size*_<model>.csv files.

    Returns:
        DataFrame with one row per (size, fold) combination.

    Raises:
        SystemExit if no cv_results files are found.
    """
    # Glob for all cv_results files for this model
    pattern = f"cv_results_size*_{model}.csv"
    cv_files = sorted(pwr_dir.glob(pattern))

    if not cv_files:
        raise SystemExit(
            f"[FATAL] No matching metric files found in {pwr_dir} "
            f"matching pattern '{pattern}'\n"
            f"Check that cv.py completed successfully and --model matches "
            f"the model used during the CV step."
        )

    rows = []
    size_pat = re.compile(rf"^cv_results_size(\d+)_{re.escape(model)}\.csv$")

    for csv_path in cv_files:
        m = size_pat.match(csv_path.name)
        if not m:
            print(f"[WARN] Filename does not match expected pattern, skipping: {csv_path.name}")
            continue

        size = float(m.group(1))
        df_cv = pd.read_csv(str(csv_path))

        if "test_R2" not in df_cv.columns:
            print(f"[WARN] test_R2 column missing in {csv_path.name}, skipping")
            continue

        for fold_idx, r2_val in enumerate(df_cv["test_R2"]):
            rows.append({
                "file_list": str(csv_path),
                "size":      size,
                "fold":      float(fold_idx),
                "metrics":   float(r2_val),
            })

    if not rows:
        raise SystemExit(
            f"[FATAL] cv_results files found but no test_R2 values could be extracted."
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="final_data (sklearn CV edition)")
    parser.add_argument("WRKDIR",  type=str)
    parser.add_argument("FILEDIR", type=str, help="Unused (kept for compatibility)")
    parser.add_argument(
        "--model", type=str, default="ridge",
        help="Model name used during CV step — must match cv_results_size*_<model>.csv "
             "filenames in pwr_data/ (default: ridge)",
    )
    parser.add_argument(
        "--metric", type=str, default="test_R2",
        choices=["test_R2", "test_RMSE", "test_MAE",
                 "train_R2", "train_RMSE", "train_MAE"],
        help="Metric column to use as the primary power curve metric (default: test_R2)",
    )
    parser.add_argument(
        "--out_format", choices=["pkl", "csv"], default="pkl",
        help="Save outputs as pickle (default) or csv",
    )
    args = parser.parse_args()

    WRKDIR = Path(args.WRKDIR)
    warnings.warn("Running final_data")

    pwr_dir = WRKDIR / "pwr_data"
    if not pwr_dir.exists():
        raise SystemExit(f"[FATAL] Missing directory: {pwr_dir}")

    # ── Provenance: pconn template lookup ─────────────────────────────────────
    df_pconn = build_pconn_lookup(pwr_dir)
    if df_pconn.empty:
        print("[WARN] No _meta.json files found — skipping pconn lookup CSV")
    else:
        pconn_lookup_path = WRKDIR / "pconn_template_lookup.csv"
        df_pconn.to_csv(pconn_lookup_path, index=False)
        print(f"[OK] Saved pconn lookup: {pconn_lookup_path} ({len(df_pconn)} rows)")

    # ── Collect cv_results CSVs ───────────────────────────────────────────────
    print(f"[INFO] Collecting cv_results_size*_{args.model}.csv from {pwr_dir}")
    df = collect_cv_results(pwr_dir, args.model)

    # If a non-default metric was requested, re-extract it from the CSVs
    if args.metric != "test_R2":
        print(f"[INFO] Re-extracting metric column: {args.metric}")
        rows = []
        size_pat = re.compile(
            rf"^cv_results_size(\d+)_{re.escape(args.model)}\.csv$"
        )
        for csv_path in sorted(pwr_dir.glob(f"cv_results_size*_{args.model}.csv")):
            m = size_pat.match(csv_path.name)
            if not m:
                continue
            size   = float(m.group(1))
            df_cv  = pd.read_csv(str(csv_path))
            if args.metric not in df_cv.columns:
                print(f"[WARN] {args.metric} missing in {csv_path.name}, skipping")
                continue
            for fold_idx, val in enumerate(df_cv[args.metric]):
                rows.append({
                    "file_list": str(csv_path),
                    "size":      size,
                    "fold":      float(fold_idx),
                    "metrics":   float(val),
                })
        df = pd.DataFrame(rows)

    n_folds = len(df)
    n_sizes = df["size"].nunique()
    print(f"[INFO] Collected {n_folds} fold results across {n_sizes} sample sizes")
    print(f"[INFO] Sizes: {sorted(df['size'].unique().tolist())}")

    # ── Write metrics_data ────────────────────────────────────────────────────
    metrics_data_path = WRKDIR / (
        "metrics_data.pkl" if args.out_format == "pkl" else "metrics_data.csv"
    )
    if args.out_format == "pkl":
        df.to_pickle(metrics_data_path)
    else:
        df.to_csv(metrics_data_path, index=False)
    print(f"[OK] Saved: {metrics_data_path}")

    # ── Compute power curve summary ───────────────────────────────────────────
    df_summary = (
        df.groupby("size", dropna=False)["metrics"]
          .agg(mean_metric="mean", sd_metric="std")
          .reset_index()
          .sort_values("size")
    )

    # ── Write metrics_summary ─────────────────────────────────────────────────
    metrics_summary_path = WRKDIR / (
        "metrics_summary.pkl" if args.out_format == "pkl" else "metrics_summary.csv"
    )
    if args.out_format == "pkl":
        df_summary.to_pickle(metrics_summary_path)
    else:
        df_summary.to_csv(metrics_summary_path, index=False)
    print(f"[OK] Saved: {metrics_summary_path}")

    # ── Plot power curve ──────────────────────────────────────────────────────
    x_labels = [
        str(int(s)) if float(s).is_integer() else str(s)
        for s in df_summary["size"].tolist()
    ]
    x    = np.arange(len(x_labels))
    y    = df_summary["mean_metric"].to_numpy()
    yerr = df_summary["sd_metric"].to_numpy()

    plt.figure(figsize=(6, 4))
    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
    plt.xticks(x, x_labels, rotation=45, ha="right")
    plt.xlabel("Sample Size")
    plt.ylabel(f"Mean {args.metric}")
    plt.title(f"Power Curve — {args.model} ({args.metric})")
    plt.tight_layout()

    plot_path = WRKDIR / "mean_metric_by_size.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[OK] Saved plot: {plot_path}")

    print(f"\n[OK] Done. {n_folds} fold results from {n_sizes} sizes aggregated.")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
