#!/usr/bin/env python3
"""
collect_sweep.py — combine the per-epsilon power curves from a sweep.

Each run launched by run_epsilon_sweep.sh writes its own metrics_summary.csv
(or .pkl) into its working directory: one row per sample size, with the mean
and SD of the CV metric across folds. This script stacks those tables into one
long-format CSV and draws every curve on a single axis.

Inputs:
    <SWEEPDIR>/eps_<value>/metrics_summary.{csv,pkl}

Outputs (written to SWEEPDIR):
    power_curves_by_epsilon.csv   long format: epsilon, size, mean_metric, sd_metric
    power_curves_by_epsilon.png   mean metric +/- SD vs sample size, one line per epsilon

Usage:
    python3 collect_sweep.py /scratch.global/$USER/pwr_sweep
    python3 collect_sweep.py /path/to/sweep --no-errorbars --metric-label "Mean CV R^2"
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # Headless backend: this runs on a login node with no display
import matplotlib.pyplot as plt     # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import pandas as pd                 # noqa: E402

RE_EPS_DIR = re.compile(r"^eps_([0-9]*\.?[0-9]+)$")


def load_summary(run_dir: Path) -> pd.DataFrame:
    """
    Read one run's metrics_summary table, whichever format it was written in.

    PWR.sh passes OUT_FORMAT=csv, but final_data.py can also write .pkl, so
    both are accepted and the CSV is preferred when both exist.

    Returns an empty DataFrame when neither file is present, which happens
    for runs that failed or are still queued.
    """
    for name in ("metrics_summary.csv", "metrics_summary.pkl"):
        path = run_dir / name
        if path.exists():
            return pd.read_csv(path) if path.suffix == ".csv" else pd.read_pickle(path)
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(
        description="Combine per-epsilon metrics_summary tables into one power-curve figure."
    )
    ap.add_argument("SWEEPDIR", help="Sweep directory containing eps_<value>/ run directories")
    ap.add_argument("--metric-label", default="Mean CV R\u00b2",
                    help="Y-axis label (default: Mean CV R^2)")
    ap.add_argument("--no-errorbars", action="store_true",
                    help="Plot means only; with ten curves the SD bars can overlap badly")
    ap.add_argument("--linear-x", action="store_true",
                    help="Use a linear x-axis (default: log, matching the log-spaced ladder)")
    args = ap.parse_args()

    sweepdir = Path(args.SWEEPDIR)
    if not sweepdir.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweepdir}")

    # ── Collect ───────────────────────────────────────────────────────────────
    frames = []
    missing = []
    for run_dir in sorted(sweepdir.iterdir()):
        if not run_dir.is_dir():
            continue
        m = RE_EPS_DIR.match(run_dir.name)
        if not m:
            continue   # base/ and anything else that is not an epsilon run
        eps = float(m.group(1))

        df = load_summary(run_dir)
        if df.empty:
            missing.append(run_dir.name)
            continue
        frames.append(df.assign(epsilon=eps))

    if missing:
        print(f"[WARN] no metrics_summary found for: {', '.join(missing)}")
    if not frames:
        raise RuntimeError(f"[FATAL] No completed epsilon runs found under {sweepdir}")

    df = (pd.concat(frames, ignore_index=True)
            .sort_values(["epsilon", "size"])
            .reset_index(drop=True))

    out_csv = sweepdir / "power_curves_by_epsilon.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}  ({df['epsilon'].nunique()} epsilon values, "
          f"{df['size'].nunique()} sample sizes)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for eps, g in df.groupby("epsilon"):
        if args.no_errorbars or "sd_metric" not in g.columns:
            ax.plot(g["size"], g["mean_metric"], marker="o", label=f"\u03b5 = {eps:g}")
        else:
            ax.errorbar(g["size"], g["mean_metric"], yerr=g["sd_metric"],
                        marker="o", capsize=3, label=f"\u03b5 = {eps:g}")

    if not args.linear_x:
        # The size ladder is log-spaced, so a log axis spaces the points evenly.
        ax.set_xscale("log")
        ax.set_xticks(sorted(df["size"].unique()))
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlabel("Sample size (N)")
    ax.set_ylabel(args.metric_label)
    ax.set_title("Power curves by phenotype noise (epsilon)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8, title="epsilon")
    fig.tight_layout()

    out_png = sweepdir / "power_curves_by_epsilon.png"
    fig.savefig(out_png, dpi=150)
    print(f"[OK] wrote {out_png}")


if __name__ == "__main__":
    main()
