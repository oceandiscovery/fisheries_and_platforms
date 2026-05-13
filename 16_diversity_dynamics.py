#!/usr/bin/env python3
"""
16_diversity_dynamics.py — P3a + P5: Diversity temporal dynamics

P3a — Temporal trends in species diversity within localities (MK + Sen, n=7)
P5  — Diversity trajectories by MPA context (LMM year×MPA, n=7 primary)
      Sensitivity: n=8 (including AB) to quantify AB's influence.

Outputs (data/processed/):
  diversity_p3a_mannkendall.csv    MK + Sen per locality × metric (n=7)
  diversity_p5_lmm.csv             LMM year_c × inside_mpa per metric (n=7)
  diversity_p5_lmm_n8.csv          Same, n=8 sensitivity
  diversity_p5_trajectories.csv    GAM trajectories inside vs outside (n=7)
  diversity_p5_trajectories_n8.csv Same, n=8 sensitivity

Outputs (outputs/figures/):
  diversity_p3a_trends.png          MK τ barplots per metric (n=7)
  diversity_p3a_timeseries.png      Diversity time series by exposure (n=7)
  diversity_p5_trajectories.png     Inside vs outside MPA trajectories (n=7)
  diversity_p5_sensitivity.png      n=7 vs n=8 side-by-side (Pielou focus)
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
from scipy.stats import kendalltau
from pygam import LinearGAM, s, f
import statsmodels.formula.api as smf

sys.path.insert(0, ".")
import config_00 as cfg

warnings.filterwarnings("ignore")
cfg.setup_dirs()
logging.basicConfig(
    format=cfg.LOG_FORMAT, datefmt=cfg.LOG_DATEFMT, level=logging.INFO,
    handlers=[
        logging.FileHandler(cfg.LOGS / "16_diversity_dynamics.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("16_diversity_dynamics")

FIG = cfg.OUTPUTS / "figures"
DAT = cfg.DATA_PROCESSED

AB = "AREIA BRANCA"

CMAP = {
    "0-20 km × 0-10 km":   "#d62728",
    "20-50 km × inside":    "#2ca02c",
    "20-50 km × 10-25 km":  "#ff7f0e",
    "20-50 km × 25-50 km":  "#1f77b4",
}
MPA_CLR = {"inside": "#d62728", "outside": "#1f77b4"}

DIV_COLS = {
    "richness":  "Species richness (S)",
    "shannon_h": "Shannon H′",
    "pielou_j":  "Pielou J′",
}


def _sig(p):
    if np.isnan(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""


def _sens_slope(y, x):
    slopes = [(y[j] - y[i]) / (x[j] - x[i])
              for i in range(len(y)) for j in range(i + 1, len(y)) if x[j] != x[i]]
    return float(np.median(slopes)) if slopes else np.nan


# ── data ──────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DAT / "productivity_local_year.csv")
    df["year_c"]     = df["year"] - df["year"].median()
    df["inside_mpa"] = df["inside_any_mpa_any"].astype(int)

    df_7 = df[df["local"] != AB].copy()
    locals_7 = sorted(df_7["local"].unique())

    log.info("n=8: %d obs | n=7: %d obs", len(df), len(df_7))
    for col in DIV_COLS:
        log.info("  %s  median=%.3f (n=7)", col, df_7[col].median())
    return df, df_7, locals_7


# ── P3a — temporal trends in diversity ────────────────────────────────────────

def p3a_diversity_trends(df_7, locals_7):
    mk_rows = []
    for loc in locals_7:
        sub = df_7[df_7["local"] == loc].sort_values("year")
        for col, label in DIV_COLS.items():
            vals = sub[col].dropna()
            if len(vals) < 5:
                continue
            yr = sub.loc[vals.index, "year"].values
            tau, p = kendalltau(yr, vals.values)
            slope = _sens_slope(vals.values, yr)
            mk_rows.append(dict(
                local=loc, metric=col, label=label,
                tau=round(tau, 3), p=round(p, 4),
                sens_slope=round(slope, 6), n=len(vals),
                combined_class=sub["combined_exposure_class"].iloc[0],
                inside_mpa=bool(sub["inside_any_mpa_any"].iloc[0]),
                sig=_sig(p),
            ))

    mk_df = pd.DataFrame(mk_rows)
    mk_df.to_csv(DAT / "diversity_p3a_mannkendall.csv", index=False)

    log.info("P3a — Mann-Kendall diversity (n=7):")
    for col in DIV_COLS:
        sub = mk_df[mk_df["metric"] == col]
        log.info("  %-12s  positive=%d/%d  p<0.05=%d/%d  median_τ=%+.3f",
                 col, (sub["tau"] > 0).sum(), len(sub),
                 (sub["p"] < 0.05).sum(), len(sub), sub["tau"].median())

    return mk_df


def fig_p3a_trends(mk_df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("P3a — Temporal trends in species diversity (n=7, excl. Areia Branca)",
                 fontsize=12)

    for ax, (col, label) in zip(axes, DIV_COLS.items()):
        sub = mk_df[mk_df["metric"] == col].sort_values("tau")
        bar_colors = [CMAP.get(cc, "gray") for cc in sub["combined_class"]]
        ax.barh(sub["local"], sub["tau"], color=bar_colors, edgecolor="k", linewidth=0.5)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        for _, row in sub.iterrows():
            if row["sig"]:
                xpos = row["tau"] + (0.01 if row["tau"] >= 0 else -0.01)
                ha   = "left" if row["tau"] >= 0 else "right"
                ax.text(xpos, row["local"],
                        f"{row['sig']} ({row['sens_slope']:+.4f})",
                        va="center", ha=ha, fontsize=7.5)
        ax.set_xlabel("Kendall's τ", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8)

    legend_els = [Line2D([0], [0], color=c, lw=2, label=k) for k, c in CMAP.items()]
    axes[2].legend(handles=legend_els, fontsize=7, loc="lower right", frameon=False)

    plt.tight_layout()
    out = FIG / "diversity_p3a_trends.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def fig_p3a_timeseries(df_7):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("P3a — Diversity time series by exposure context (n=7)", fontsize=12)

    for ax, (col, label) in zip(axes, DIV_COLS.items()):
        for loc in sorted(df_7["local"].unique()):
            sub = df_7[df_7["local"] == loc].sort_values("year")
            cc  = sub["combined_exposure_class"].iloc[0]
            ax.plot(sub["year"], sub[col], color=CMAP.get(cc, "gray"),
                    alpha=0.6, lw=1.3, marker="o", markersize=3)
        ax.set_xlabel("Year", fontsize=9); ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10); ax.tick_params(labelsize=8)

    legend_els = [Line2D([0], [0], color=c, lw=2, label=k) for k, c in CMAP.items()]
    axes[2].legend(handles=legend_els, fontsize=7, loc="lower right", frameon=False)

    plt.tight_layout()
    out = FIG / "diversity_p3a_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


# ── P5 — diversity trajectories by exposure ───────────────────────────────────

def _fit_group_gam(sub_df, metric, n_splines=6):
    locals_sub = sorted(sub_df["local"].unique())
    enc = {loc: i for i, loc in enumerate(locals_sub)}
    sub = sub_df.copy()
    sub["_enc"] = sub["local"].map(enc)
    mask = sub[metric].notna() & sub["year_c"].notna()
    X = sub.loc[mask, ["year_c", "_enc"]].values
    y = sub.loc[mask, metric].values
    gam = LinearGAM(s(0, n_splines=n_splines) + f(1)).gridsearch(X, y, progress=False)
    return gam, len(locals_sub)


def _run_p5(data, label=""):
    """Run LMM + GAM trajectories for P5 on a given dataset."""
    yr_median = data["year"].median()
    lmm_rows  = []

    for col, col_label in DIV_COLS.items():
        df_sub = data.dropna(subset=[col]).copy()
        for method in ("lbfgs", "bfgs", "nm"):
            try:
                res = smf.mixedlm(f"{col} ~ year_c * inside_mpa",
                                  data=df_sub, groups=df_sub["local"]
                                  ).fit(reml=True, method=method)
                for term in res.params.index:
                    p_val = float(res.pvalues.get(term, np.nan))
                    ci = res.conf_int()
                    lo = float(ci.loc[term, 0]) if term in ci.index else np.nan
                    hi = float(ci.loc[term, 1]) if term in ci.index else np.nan
                    lmm_rows.append(dict(
                        metric=col, term=term,
                        coef=round(float(res.params[term]), 4),
                        p=round(p_val, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                        sig=_sig(p_val),
                    ))
                int_p = float(res.pvalues.get("year_c:inside_mpa", np.nan))
                log.info("P5 LMM %s %-14s %-10s: year×MPA  p=%.4f %s",
                         label, col, col_label, int_p, _sig(int_p))
                break
            except Exception as e:
                log.warning("P5 LMM %s %s method=%s: %s", label, col, method, e)

    lmm_df = pd.DataFrame(lmm_rows)

    # GAM trajectories by MPA status
    year_c_grid = np.linspace(data["year_c"].min(), data["year_c"].max(), 100)
    traj_rows   = []

    for col, col_label in DIV_COLS.items():
        for mpa_val, mpa_label in [(1, "inside"), (0, "outside")]:
            sub = data[data["inside_mpa"] == mpa_val].dropna(subset=[col])
            if sub["local"].nunique() < 2:
                continue
            try:
                gam, n_locs = _fit_group_gam(sub, col)
                dev = gam.statistics_["pseudo_r2"]["explained_deviance"]
                log.info("P5 GAM %s %s [%s]  dev_exp=%.3f", label, col_label, mpa_label, dev)
                for yc in year_c_grid:
                    preds = [gam.predict(np.array([[yc, i]]))[0] for i in range(n_locs)]
                    cis   = [gam.confidence_intervals(np.array([[yc, i]]), width=0.95)[0]
                             for i in range(n_locs)]
                    traj_rows.append(dict(
                        metric=col, group=mpa_label,
                        year_c=yc, year=yc + yr_median,
                        pred=float(np.mean(preds)),
                        ci_lo=float(np.mean([c[0] for c in cis])),
                        ci_hi=float(np.mean([c[1] for c in cis])),
                    ))
            except Exception as e:
                log.warning("P5 GAM %s %s %s: %s", label, col, mpa_label, e)

    return lmm_df, pd.DataFrame(traj_rows)


def p5_trajectories(df, df_7):
    lmm7,  traj7  = _run_p5(df_7, "n=7")
    lmm8,  traj8  = _run_p5(df,   "n=8")

    if not lmm7.empty:
        lmm7.to_csv(DAT / "diversity_p5_lmm.csv", index=False)
    if not lmm8.empty:
        lmm8.to_csv(DAT / "diversity_p5_lmm_n8.csv", index=False)
    traj7.to_csv(DAT / "diversity_p5_trajectories.csv", index=False)
    traj8.to_csv(DAT / "diversity_p5_trajectories_n8.csv", index=False)
    log.info("Saved P5 trajectories (n=7 + n=8 sensitivity)")

    return lmm7, lmm8, traj7, traj8


def fig_p5_trajectories(df_7, lmm7, traj7):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("P5 — Diversity trajectories: inside vs outside MPA (n=7, excl. Areia Branca)",
                 fontsize=12)

    for ax, (col, col_label) in zip(axes, DIV_COLS.items()):
        for mpa_label, color in MPA_CLR.items():
            sub_t = traj7[(traj7["metric"] == col) & (traj7["group"] == mpa_label)]
            if sub_t.empty:
                continue
            ax.plot(sub_t["year"], sub_t["pred"], color=color, lw=2.2, label=mpa_label)
            ax.fill_between(sub_t["year"], sub_t["ci_lo"], sub_t["ci_hi"], color=color, alpha=0.15)

        # Raw scatter
        for mpa_val, mpa_label in [(True, "inside"), (False, "outside")]:
            sub_raw = df_7[df_7["inside_any_mpa_any"] == mpa_val].groupby("year")[col].mean()
            ax.scatter(sub_raw.index, sub_raw.values,
                       color=MPA_CLR[mpa_label], s=18, alpha=0.35, zorder=3)

        ax.set_xlabel("Year", fontsize=9); ax.set_ylabel(col_label, fontsize=9)

        if not lmm7.empty:
            int_row = lmm7[(lmm7["metric"] == col) & (lmm7["term"] == "year_c:inside_mpa")]
            if not int_row.empty:
                r = int_row.iloc[0]
                ax.set_title(f"{col_label}\nyear×MPA: β={r['coef']:+.4f}  p={r['p']:.3f} {r['sig']}",
                             fontsize=9)
            else:
                ax.set_title(col_label, fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    out = FIG / "diversity_p5_trajectories.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def fig_p5_sensitivity(df, df_7, traj7, traj8, lmm7, lmm8):
    """Side-by-side n=7 vs n=8 for Pielou J' (most affected by AB)."""
    col = "pielou_j"
    col_label = DIV_COLS[col]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"P5 sensitivity — {col_label}: n=7 vs n=8 (AB influence)", fontsize=12)

    for ax, traj, lmm, data, title in [
        (axes[0], traj7, lmm7, df_7, "n=7 (excl. Areia Branca)"),
        (axes[1], traj8, lmm8, df,   "n=8 (incl. Areia Branca — sensitivity)"),
    ]:
        for mpa_label, color in MPA_CLR.items():
            sub_t = traj[(traj["metric"] == col) & (traj["group"] == mpa_label)]
            if sub_t.empty:
                continue
            ax.plot(sub_t["year"], sub_t["pred"], color=color, lw=2.2, label=mpa_label)
            ax.fill_between(sub_t["year"], sub_t["ci_lo"], sub_t["ci_hi"], color=color, alpha=0.15)
        for mpa_val, mpa_label in [(True, "inside"), (False, "outside")]:
            sub_raw = data[data["inside_any_mpa_any"] == mpa_val].groupby("year")[col].mean()
            ax.scatter(sub_raw.index, sub_raw.values,
                       color=MPA_CLR[mpa_label], s=18, alpha=0.35, zorder=3)

        if not lmm.empty:
            int_row = lmm[(lmm["metric"] == col) & (lmm["term"] == "year_c:inside_mpa")]
            if not int_row.empty:
                r = int_row.iloc[0]
                ax.set_title(f"{title}\nyear×MPA: β={r['coef']:+.4f}  p={r['p']:.4f} {r['sig']}",
                             fontsize=9)
            else:
                ax.set_title(title, fontsize=9)
        ax.set_xlabel("Year", fontsize=9); ax.set_ylabel(col_label, fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    out = FIG / "diversity_p5_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def print_summary(mk_df, lmm7, lmm8):
    HR = "─" * 68
    print("\n" + "=" * 68)
    print("P3a + P5 — DIVERSITY TEMPORAL DYNAMICS")
    print("=" * 68)
    print()
    print("P3a — Mann-Kendall diversity trends (n=7):")
    print(f"  {'Metric':<14} {'Loc':<22} {'τ':>7}  {'p':>7}  {'sig':>4}")
    print(HR)
    for _, r in mk_df.sort_values(["metric", "tau"]).iterrows():
        print(f"  {r['metric']:<14} {r['local']:<22} {r['tau']:>+7.3f}  {r['p']:>7.4f}  {r['sig']:>4}")
    print()
    print("P5 — LMM year×MPA interaction:")
    for label, lmm in [("n=7 (primary)", lmm7), ("n=8 (sensitivity)", lmm8)]:
        print(f"\n  {label}:")
        if lmm.empty:
            print("    (model failed)"); continue
        int_rows = lmm[lmm["term"] == "year_c:inside_mpa"]
        if int_rows.empty:
            print("    (interaction term not found)"); continue
        for _, r in int_rows.iterrows():
            print(f"    {r['metric']:<14}  β={r['coef']:+.4f}  p={r['p']:.4f} {r['sig']}")
    print("=" * 68)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df, df_7, locals_7 = load_data()

    log.info("── P3a: Diversity trends within localities (n=7) ─────")
    mk_df = p3a_diversity_trends(df_7, locals_7)
    fig_p3a_trends(mk_df)
    fig_p3a_timeseries(df_7)

    log.info("── P5: Diversity trajectories by MPA (n=7 + n=8 sensitivity)")
    lmm7, lmm8, traj7, traj8 = p5_trajectories(df, df_7)
    fig_p5_trajectories(df_7, lmm7, traj7)
    fig_p5_sensitivity(df, df_7, traj7, traj8, lmm7, lmm8)

    print_summary(mk_df, lmm7, lmm8)
    log.info("16_diversity_dynamics.py complete.")


if __name__ == "__main__":
    main()
