#!/usr/bin/env python3
"""
14_spatial.py — P1: Spatial characterization across localities

P1 — How do CPUE and species diversity vary across localities in relation to
     platform distance and MPA status? (between-estimator, n=7)

Primary: n=7 (excluding Areia Branca).
AB is plotted in figures with a distinct symbol for visual reference only.

Outputs (data/processed/):
  spatial_p1_local_means.csv    Per-locality summary (n=7 + AB for reference)
  spatial_p1_spearman.csv       Spearman correlations (n=7)

Outputs (outputs/figures/):
  spatial_p1_scatter.png        CPUE and diversity vs platform dist / MPA (n=7)
  spatial_p1_heatmap.png        Metric × locality heatmap
"""

import logging
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, ".")
import config_00 as cfg

warnings.filterwarnings("ignore")
cfg.setup_dirs()
logging.basicConfig(
    format=cfg.LOG_FORMAT, datefmt=cfg.LOG_DATEFMT, level=logging.INFO,
    handlers=[
        logging.FileHandler(cfg.LOGS / "14_spatial.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("14_spatial")

FIG = cfg.OUTPUTS / "figures"
DAT = cfg.DATA_PROCESSED

AB       = "AREIA BRANCA"
CPUE_COL = "cpue_per_fisherman"

MPA_CLR  = {True: "#d62728", False: "#1f77b4"}
AB_MARKER = "D"   # diamond for AB in reference plots


def _sig(p):
    if np.isnan(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""


# ── data ──────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DAT / "productivity_local_year.csv")
    df[CPUE_COL] = df["production_ton"] / df["estimated_fishermen"]
    df["year_c"] = df["year"] - df["year"].median()
    return df


# ── P1 — spatial characterization ─────────────────────────────────────────────

def p1_spatial(df):
    # Full set (n=8) for reference; primary analysis on n=7
    df_all = df.copy()
    df_7   = df[df["local"] != AB].copy()

    def _locality_means(data):
        return data.groupby("local", sort=True).agg(
            cpue_median    = (CPUE_COL, "median"),
            cpue_mean      = (CPUE_COL, "mean"),
            cpue_cv        = (CPUE_COL, lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan),
            richness_mean  = ("richness",  "mean"),
            shannon_mean   = ("shannon_h", "mean"),
            pielou_mean    = ("pielou_j",  "mean"),
            platform_dist  = ("platform_dist_km_mean",  "first"),
            mpa_dist       = ("mpa_dist_km_mean",        "first"),
            inside_mpa     = ("inside_any_mpa_any",      "first"),
            combined_class = ("combined_exposure_class", "first"),
            n_years        = ("year", "count"),
        ).reset_index()

    means_all = _locality_means(df_all)
    means_7   = _locality_means(df_7)

    # Save full set (AB flagged)
    means_all["is_ab"] = means_all["local"] == AB
    means_all.to_csv(DAT / "spatial_p1_local_means.csv", index=False)
    log.info("Saved spatial_p1_local_means.csv  (n=%d)", len(means_all))

    # Spearman correlations on n=7
    spear_rows = []
    for metric, metric_label in [
        ("cpue_median",  "CPUE (t/fisherman)"),
        ("richness_mean","Richness"),
        ("shannon_mean", "Shannon H′"),
        ("pielou_mean",  "Pielou J′"),
    ]:
        for pred, pred_label in [
            ("platform_dist", "Platform dist. (km)"),
            ("mpa_dist",      "MPA dist. (km)"),
        ]:
            valid = means_7.dropna(subset=[metric, pred])
            if len(valid) < 3:
                continue
            r, p = spearmanr(valid[pred], valid[metric])
            spear_rows.append(dict(
                metric=metric, metric_label=metric_label,
                predictor=pred, predictor_label=pred_label,
                n=len(valid), rho=round(r, 3), p=round(p, 4),
                sig=_sig(p),
            ))
            log.info("P1  %-20s ~ %-20s  ρ=%+.3f  p=%.4f %s  (n=%d)",
                     metric, pred, r, p, _sig(p), len(valid))

    # MPA status: Spearman with inside_mpa (binary)
    for metric, metric_label in [
        ("cpue_median",  "CPUE (t/fisherman)"),
        ("shannon_mean", "Shannon H′"),
    ]:
        valid = means_7.dropna(subset=[metric])
        r, p = spearmanr(valid["inside_mpa"].astype(int), valid[metric])
        spear_rows.append(dict(
            metric=metric, metric_label=metric_label,
            predictor="inside_mpa", predictor_label="Inside MPA (binary)",
            n=len(valid), rho=round(r, 3), p=round(p, 4), sig=_sig(p),
        ))
        log.info("P1  %-20s ~ inside_mpa          ρ=%+.3f  p=%.4f %s  (n=%d)",
                 metric, r, p, _sig(p), len(valid))

    spear_df = pd.DataFrame(spear_rows)
    spear_df.to_csv(DAT / "spatial_p1_spearman.csv", index=False)
    log.info("Saved spatial_p1_spearman.csv  (%d rows)", len(spear_df))

    return means_all, means_7, spear_df


def fig_scatter(means_all, means_7, spear_df):
    """4×2 scatter: CPUE and Shannon vs platform dist / MPA dist / inside_mpa (n=7).
    AB shown as gray diamond for visual reference."""
    pairs = [
        ("cpue_median",  "platform_dist", "Median CPUE (t/fish.)",  "Platform dist. (km)"),
        ("cpue_median",  "mpa_dist",       "Median CPUE (t/fish.)",  "MPA dist. (km)"),
        ("shannon_mean", "platform_dist",  "Mean Shannon H′",         "Platform dist. (km)"),
        ("shannon_mean", "mpa_dist",       "Mean Shannon H′",         "MPA dist. (km)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("P1 — CPUE and diversity across localities (n=7, excl. Areia Branca)",
                 fontsize=12)
    ab_row = means_all[means_all["local"] == AB]

    for ax, (y_col, x_col, ylabel, xlabel) in zip(axes.flat, pairs):
        # n=7 points
        for _, row in means_7.iterrows():
            color = MPA_CLR[row["inside_mpa"]]
            ax.scatter(row[x_col], row[y_col], c=color, s=90, zorder=4,
                       edgecolors="k", linewidths=0.4)
            ax.annotate(row["local"][:9], (row[x_col], row[y_col]),
                        fontsize=6.5, xytext=(3, 3), textcoords="offset points", color=color)

        # AB reference (gray diamond)
        if not ab_row.empty and pd.notna(ab_row.iloc[0][x_col]) and pd.notna(ab_row.iloc[0][y_col]):
            ab_x = float(ab_row.iloc[0][x_col])
            ab_y = float(ab_row.iloc[0][y_col])
            ax.scatter(ab_x, ab_y, marker=AB_MARKER,
                       c="none", edgecolors="gray", s=90, zorder=3, linewidths=1.2)
            ax.annotate("AB (ref)", (ab_x, ab_y),
                        fontsize=6, xytext=(3, -9), textcoords="offset points", color="gray",
                        style="italic")

        # Spearman annotation
        sp_row = spear_df.query(f"metric=='{y_col}' and predictor=='{x_col}'")
        if not sp_row.empty:
            r, p, sig = sp_row.iloc[0][["rho", "p", "sig"]]
            ax.set_title(f"ρ = {r:+.2f}  p = {p:.3f} {sig}", fontsize=9)

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)

    # Legend
    legend_els = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MPA_CLR[True],  markersize=8, label="Inside MPA"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MPA_CLR[False], markersize=8, label="Outside MPA"),
        Line2D([0], [0], marker=AB_MARKER, color="w", markeredgecolor="gray", markersize=8, label="Areia Branca (ref)"),
    ]
    axes[0, 1].legend(handles=legend_els, fontsize=8, frameon=False)

    plt.tight_layout()
    out = FIG / "spatial_p1_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def fig_heatmap(means_all):
    """Locality × metric heatmap (z-scored); AB row separated."""
    metrics = {
        "cpue_median":   "CPUE (t/fish.)",
        "richness_mean": "Richness",
        "shannon_mean":  "Shannon H′",
        "pielou_mean":   "Pielou J′",
        "platform_dist": "Platform dist.",
        "mpa_dist":      "MPA dist.",
    }

    # Separate AB from n=7
    ab_row  = means_all[means_all["local"] == AB]
    core    = means_all[means_all["local"] != AB].sort_values("platform_dist")
    ordered = pd.concat([core, ab_row], ignore_index=True)

    mat = ordered[list(metrics.keys())].values.astype(float)
    # Z-score per column
    means_col = np.nanmean(mat, axis=0)
    stds_col  = np.nanstd(mat, axis=0)
    z = (mat - means_col) / np.where(stds_col == 0, 1, stds_col)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(list(metrics.values()), rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(ordered)))
    labels = [row["local"] + (" (AB)" if row["local"] == AB else "")
              for _, row in ordered.iterrows()]
    ax.set_yticklabels(labels, fontsize=8)

    # AB separator line
    n_core = len(core)
    ax.axhline(n_core - 0.5, color="k", lw=1.5, ls="--")

    # Annotate raw values
    for r in range(len(ordered)):
        for c, col in enumerate(metrics.keys()):
            val = ordered.iloc[r][col]
            if pd.notna(val):
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5, color="k")

    plt.colorbar(im, ax=ax, label="Z-score", shrink=0.7)
    ax.set_title("P1 — Locality profile (z-scored); AB shown for reference", fontsize=11)
    plt.tight_layout()
    out = FIG / "spatial_p1_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def print_summary(means_7, spear_df):
    HR = "─" * 68
    print("\n" + "=" * 68)
    print("P1 — SPATIAL CHARACTERIZATION (n=7 primary)")
    print("=" * 68)
    print(f"{'Local':<22} {'CPUE med':>9} {'Richness':>9} {'Shannon':>8} {'PlatDist':>9} {'InsideMPA':>10}")
    print(HR)
    for _, r in means_7.sort_values("platform_dist").iterrows():
        print(f"  {r['local']:<20} {r['cpue_median']:>9.4f} {r['richness_mean']:>9.1f} "
              f"{r['shannon_mean']:>8.3f} {r['platform_dist']:>9.1f} {'Yes' if r['inside_mpa'] else 'No':>10}")
    print()
    print("Spearman correlations (n=7):")
    print(f"  {'Metric':<22} {'Predictor':<22} {'ρ':>7}  {'p':>7}  {'sig':>4}")
    print(HR)
    for _, r in spear_df.iterrows():
        print(f"  {r['metric_label']:<22} {r['predictor_label']:<22} "
              f"{r['rho']:>+7.3f}  {r['p']:>7.4f}  {r['sig']:>4}")
    print("=" * 68)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_data()

    log.info("── P1: Spatial characterization (n=7) ────────────────")
    means_all, means_7, spear_df = p1_spatial(df)

    fig_scatter(means_all, means_7, spear_df)
    fig_heatmap(means_all)
    print_summary(means_7, spear_df)

    log.info("14_spatial.py complete.")


if __name__ == "__main__":
    main()
