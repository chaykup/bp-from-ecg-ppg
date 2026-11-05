"""Exploratory analysis of the feature table.

Usage (from the repo root, after build_table.py):
    python src/eda.py

Reads data/processed/features.parquet and build_log.json, saves figures to figures/eda/ and prints
summary numbers: rejection counts, missing values, IQR outliers and how much of
the BP variance sits between records rather than within them.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # figures go to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from features import FEATURE_COLS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures" / "eda"
LABELS = ["sbp", "dbp"]
IQR_K = 1.5     # Tukey's rule: outside [Q1 - 1.5 IQR, Q3 + 1.5 IQR]
N_EXAMPLE_RECORDS = 6
MIN_WINDOWS = 40    # example records need enough windows (>= 5 min) to show a within-record trend


def save(fig, name: str):
    """Save a figure to FIG_DIR as PNG and close it."""
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"    saved figures/eda/{name}.png")


def plot_label_hists(df: pd.DataFrame):
    """SBP and DBP histograms with the label bounds marked: no spikes at the bounds means the cuts are clean."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, col, bounds in zip(axes, LABELS, [config.SBP_BOUNDS, config.DBP_BOUNDS]):
        sns.histplot(df[col], bins=100, ax=ax)
        for b in bounds:
            ax.axvline(b, color="k", ls="--", lw=1)
        ax.set_xlabel(f"{col.upper()} (mmHg)")
        ax.set_title(f"{col.upper()}: median {df[col].median():.0f}, IQR {df[col].quantile(.25):.0f}-{df[col].quantile(.75):.0f}")
    save(fig, "label_hist")


def plot_feature_hists(df: pd.DataFrame):
    """One histogram per feature, clipped to the 0.5-99.5th percentile so long tails don't hide the shape."""
    n = len(FEATURE_COLS)
    ncols = 4
    fig, axes = plt.subplots(int(np.ceil(n / ncols)), ncols, figsize=(14, 9))
    for ax, col in zip(axes.flat, FEATURE_COLS):
        x = df[col].dropna()
        lo, hi = x.quantile([0.005, 0.995])
        sns.histplot(x[(x >= lo) & (x <= hi)], bins=60, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
    for ax in axes.flat[n:]:
        ax.axis("off")
    save(fig, "feature_hist")


def plot_corr(df: pd.DataFrame):
    """Pearson correlation heatmap of features and labels."""
    corr = df[FEATURE_COLS + LABELS].corr()
    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, square=True,
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Pearson correlation (all windows)")
    save(fig, "corr_heatmap")
    return corr


def within_record_corr(df: pd.DataFrame, x: str, y: str) -> pd.Series:
    """Pearson r between x and y inside each record with at least MIN_WINDOWS windows and some BP spread."""
    def r(g):
        g = g[[x, y]].dropna()
        if len(g) < MIN_WINDOWS or g[y].std() < 5 or g[x].std() == 0:
            return np.nan
        return g[x].corr(g[y])
    return df.groupby("record_id").apply(r, include_groups=False).dropna()


def pick_example_records(df: pd.DataFrame, rng: np.random.Generator) -> list[str]:
    """Long records with a wide SBP range, so within-record trends are visible."""
    stats = df.groupby("record_id").agg(n=("sbp", "size"), sbp_sd=("sbp", "std"))
    pool = stats[(stats["n"] >= MIN_WINDOWS) & (stats["sbp_sd"] >= 8)].index.to_numpy()
    return list(rng.choice(pool, size=min(N_EXAMPLE_RECORDS, len(pool)), replace=False))


def plot_pat_vs_sbp(df: pd.DataFrame, records: list[str], r_within: pd.Series):
    """PAT vs SBP: a few records colored (within-record trends, between-record offsets) next to the
    distribution of within-record correlations against the pooled one."""
    pooled = df[["pat_foot_s", "sbp"]].corr().iloc[0, 1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.4, 1]})
    sub = df[df["record_id"].isin(records)]
    sns.scatterplot(data=sub, x="pat_foot_s", y="sbp", hue="record_id", s=12, ax=axes[0])
    axes[0].set_xlabel("PAT to foot (s)")
    axes[0].set_ylabel("SBP (mmHg)")
    axes[0].set_title(f"{len(records)} example records")
    axes[0].legend(fontsize=7, markerscale=1.5)

    sns.histplot(r_within, bins=40, ax=axes[1])
    axes[1].axvline(pooled, color="r", lw=2, label=f"pooled r = {pooled:.2f}")
    axes[1].axvline(r_within.median(), color="k", ls="--", label=f"median within-record r = {r_within.median():.2f}")
    axes[1].set_xlabel("Pearson r, PAT to foot vs SBP, within one record")
    axes[1].set_title(f"{len(r_within)} records with >= {MIN_WINDOWS} windows")
    axes[1].legend(fontsize=8)
    save(fig, "pat_vs_sbp")
    return pooled


def plot_rejections(log: dict):
    """Bar chart of why windows were rejected during windowing (signals.window_record)."""
    w = {k: v for k, v in log["windows"].items() if k != "kept" and v > 0}
    w = dict(sorted(w.items(), key=lambda kv: kv[1]))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(list(w), list(w.values()))
    for i, v in enumerate(w.values()):
        ax.text(v, i, f" {v:,} ({v / log['windows_total']:.1%})", va="center", fontsize=8)
    ax.set_xlabel("windows rejected")
    ax.set_title(f"{log['windows']['kept']:,} of {log['windows_total']:,} windows kept")
    ax.set_xlim(0, max(w.values()) * 1.35)
    save(fig, "rejections")


def iqr_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Per feature: Tukey fences and how many windows fall outside them."""
    rows = []
    for col in FEATURE_COLS:
        x = df[col].dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        lo, hi = q1 - IQR_K * (q3 - q1), q3 + IQR_K * (q3 - q1)
        out = (x < lo) | (x > hi)
        rows.append({"feature": col, "q1": q1, "q3": q3, "lower": lo, "upper": hi,
                     "n_low": int((x < lo).sum()), "n_high": int((x > hi).sum()),
                     "pct_out": 100 * out.mean(), "min": x.min(), "max": x.max()})
    return pd.DataFrame(rows).set_index("feature")


def between_record_share(df: pd.DataFrame, col: str) -> float:
    """Fraction of a column's variance explained by record means (between-record / total)."""
    rec_mean = df.groupby("record_id")[col].transform("mean")
    return float(((rec_mean - df[col].mean()) ** 2).sum() / ((df[col] - df[col].mean()) ** 2).sum())


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA / "features.parquet")
    with open(DATA / "build_log.json") as f:
        log = json.load(f)
    rng = np.random.default_rng(config.RANDOM_STATE)

    print(f"{len(df):,} windows from {df['record_id'].nunique():,} records "
          f"({log['records'] - df['record_id'].nunique():,} of {log['records']:,} records kept no windows)")
    print("\nWindows per Part:\n", df.groupby("part").agg(windows=("sbp", "size"), records=("record_id", "nunique"),
                                                        sbp_mean=("sbp", "mean"), dbp_mean=("dbp", "mean")).round(1).to_string())
    print("\nWindows per record:\n", df.groupby("record_id").size().describe().round(1).to_string())
    print("\nMissing values:\n", df.isna().sum()[lambda s: s > 0].to_string())

    print("\nFigures:")
    plot_rejections(log)
    plot_label_hists(df)
    plot_feature_hists(df)
    corr = plot_corr(df)
    r_within = within_record_corr(df, "pat_foot_s", "sbp")
    pooled = plot_pat_vs_sbp(df, pick_example_records(df, rng), r_within)

    print("\nCorrelation with labels (Pearson, all windows):")
    print(corr.loc[FEATURE_COLS, LABELS].round(3).sort_values("sbp").to_string())
    high = [(a, b, corr.loc[a, b]) for i, a in enumerate(FEATURE_COLS) for b in FEATURE_COLS[i + 1:] if abs(corr.loc[a, b]) >= 0.7]
    print("\nFeature pairs with |r| >= 0.7:")
    for a, b, v in high:
        print(f"    {a:18s} {b:18s} {v:6.2f}")

    print(f"\nPAT to foot vs SBP: pooled r = {pooled:.3f}; within-record r over {len(r_within)} records: "
          f"median {r_within.median():.3f}, IQR {r_within.quantile(.25):.3f} to {r_within.quantile(.75):.3f}, "
          f"{(r_within < 0).mean():.0%} negative")
    print("Share of variance between records (record means / total):")
    for col in LABELS + ["pat_foot_s", "hr_bpm"]:
        print(f"    {col:12s} {between_record_share(df, col):.1%}")

    out = iqr_outliers(df)
    out.round(4).to_csv(FIG_DIR / "iqr_outliers.csv")
    print(f"\nIQR outliers (k = {IQR_K}), also saved to figures/eda/iqr_outliers.csv:")
    print(out[["lower", "upper", "n_low", "n_high", "pct_out", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
