#!/usr/bin/env python3
import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_scalar_metric(path: Path, metric_ext: str) -> float:
    if metric_ext == "npy":
        arr = np.load(str(path))
        # allow scalar or 1-element array
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


def main():
    parser = argparse.ArgumentParser(description="final_data (Python-native)")
    parser.add_argument("WRKDIR", type=str)
    parser.add_argument("FILEDIR", type=str, help="Unused (kept for compatibility)")
    parser.add_argument("--metric_ext", choices=["npy", "pkl", "txt"], default="npy",
                        help="Extension of metric files to read (default: npy)")
    parser.add_argument("--out_format", choices=["pkl", "csv"], default="pkl",
                        help="Save outputs as pickle (default) or csv")
    args = parser.parse_args()

    WRKDIR = Path(args.WRKDIR)
    warnings.warn("Running final_data")

    pwr_dir = WRKDIR / "pwr_data"
    if not pwr_dir.exists():
        raise SystemExit(f"[FATAL] Missing directory: {pwr_dir}")

    metric_suffix = f"_cvr2.{args.metric_ext}"
    file_list = sorted([p for p in pwr_dir.iterdir() if p.is_file() and p.name.endswith(metric_suffix)])
    n_files = len(file_list)

    if n_files == 0:
        raise SystemExit(
            f"[FATAL] No matching metric files found in {pwr_dir} ending with {metric_suffix}"
        )

    # Expect filenames: data_<size>_fold_<fold>_cvr2.<ext>
    sizes = []
    folds = []
    for p in file_list:
        s = p.name
        m = re.match(r"^data_(\d+)_fold_(\d+)_cvr2\.", s)
        if m:
            sizes.append(float(m.group(1)))
            folds.append(float(m.group(2)))
        else:
            sizes.append(np.nan)
            folds.append(np.nan)

    df = pd.DataFrame({
        "file_list": [str(p) for p in file_list],
        "size": sizes,
        "fold": folds,
        "metrics": 0.0
    })

    for i, p in enumerate(file_list):
        df.loc[i, "metrics"] = read_scalar_metric(p, args.metric_ext)

    metrics_data_path = WRKDIR / ("metrics_data.pkl" if args.out_format == "pkl" else "metrics_data.csv")
    if args.out_format == "pkl":
        df.to_pickle(metrics_data_path)
    else:
        df.to_csv(metrics_data_path, index=False)

    df_summary = (
        df.groupby("size", dropna=False)["metrics"]
          .agg(mean_metric="mean", sd_metric="std")
          .reset_index()
          .sort_values("size")
    )

    metrics_summary_path = WRKDIR / ("metrics_summary.pkl" if args.out_format == "pkl" else "metrics_summary.csv")
    if args.out_format == "pkl":
        df_summary.to_pickle(metrics_summary_path)
    else:
        df_summary.to_csv(metrics_summary_path, index=False)

    x_labels = [str(int(s)) if float(s).is_integer() else str(s) for s in df_summary["size"].tolist()]
    x = np.arange(len(x_labels))
    y = df_summary["mean_metric"].to_numpy()
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
    plt.close()

    print(f"[OK] Read {n_files} metric files ({metric_suffix})")
    print(f"[OK] Saved: {metrics_data_path}")
    print(f"[OK] Saved: {metrics_summary_path}")
    print(f"[OK] Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
