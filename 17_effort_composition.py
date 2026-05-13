#!/usr/bin/env python3
"""
17_effort_composition.py — P3b + P6: Effort composition dynamics

P3b — Temporal trends in gear and boat composition within localities
      (Mann-Kendall + Sen's slope, n=7)
P6  — Within-local CPUE ~ gear type + boat type composition, controlling for
      non-linear temporal trend (LinearGAM with s(active_dm) + s(motorised_dm) + s(year_c))
      Year-controlled GAM is the PRIMARY analysis (not sensitivity).

Outputs (data/processed/):
  effort_p3b_mannkendall.csv     MK + Sen per locality × share (n=7)
  effort_p6_spearman.csv         Within-local Spearman (n=7, no year control)
  effort_p6_lmm.csv              LMM with year + active_dm + motorised_dm (n=7)
  effort_p6_gam_partial.csv      GAM partial effects (s(active) + s(motorised) + s(year))
  effort_p6_year_smooth.csv      Year smooth from P6 GAM

Outputs (outputs/figures/):
  effort_p3b_trends.png          MK τ per share variable (n=7)
  effort_p3b_timeseries.png      Share time series by exposure context (n=7)
  effort_p6_cpue_composition.png CPUE ~ gear/boat: scatter + GAM partial + year smooth
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
from scipy.stats import kendalltau, spearmanr
from pygam import LinearGAM, s
import statsmodels.formula.api as smf

sys.path.insert(0, ".")
import config_00 as cfg

warnings.filterwarnings("ignore")
cfg.setup_dirs()
logging.basicConfig(
    format=cfg.LOG_FORMAT, datefmt=cfg.LOG_DATEFMT, level=logging.INFO,
    handlers=[
        logging.FileHandler(cfg.LOGS / "17_effort_composition.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("17_effort_composition")

FIG = cfg.OUTPUTS / "figures"
DAT = cfg.DATA_PROCESSED

AB       = "AREIA BRANCA"
CPUE_COL = "cpue_per_fisherman"

CMAP = {
    "0-20 km × 0-10 km":   "#d62728",
    "20-50 km × inside":    "#2ca02c",
    "20-50 km × 10-25 km":  "#ff7f0e",
    "20-50 km × 25-50 km":  "#1f77b4",
}

SHARE_COLS = {
    "active":    "Share active gear",
    "passive":   "Share passive gear",
    "motorised": "Share motorised vessels",
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
    prod = pd.read_csv(DAT / "productivity_local_year.csv")
    prod[CPUE_COL] = prod["production_ton"] / prod["estimated_fishermen"]
    prod["log_cpue"] = np.log(prod[CPUE_COL].clip(lower=1e-6))
    prod["year_c"] = prod["year"] - prod["year"].median()

    # Gear shares: active/passive by local×year
    gs_long = pd.read_csv(DAT / "gear_share_local_year.csv")
    gear = (gs_long.groupby(["local", "year", "gear_group"])["gear_share"]
                   .sum().unstack(fill_value=0).reset_index())
    gear.columns.name = None
    for col in ["active", "passive", "mixed"]:
        if col not in gear.columns:
            gear[col] = 0.0

    # Boat shares: motorised by local×year
    bs_long = pd.read_csv(DAT / "boat_share_local_year.csv")
    boat = (bs_long.groupby(["local", "year", "propulsion"])["boat_share"]
                   .sum().unstack(fill_value=0).reset_index())
    boat.columns.name = None
    for col in ["motorised"]:
        if col not in boat.columns:
            boat[col] = 0.0

    df = (prod.merge(gear[["local", "year", "active", "passive"]], on=["local", "year"], how="left")
              .merge(boat[["local", "year", "motorised"]], on=["local", "year"], how="left"))

    df_7 = df[df["local"] != AB].copy()
    locals_7 = sorted(df_7["local"].unique())

    log.info("n=7 data: %d obs | locals: %s", len(df_7), locals_7)
    return df_7, locals_7


# ── P3b — temporal trends in effort composition ───────────────────────────────

def p3b_effort_trends(df_7, locals_7):
    mk_rows = []
    for loc in locals_7:
        sub = df_7[df_7["local"] == loc].sort_values("year")
        cc  = sub["combined_exposure_class"].iloc[0]
        for col, label in SHARE_COLS.items():
            vals = sub[col].dropna()
            if vals.nunique() < 3 or len(vals) < 5:
                continue
            yr = sub.loc[vals.index, "year"].values
            tau, p = kendalltau(yr, vals.values)
            slope = _sens_slope(vals.values, yr)
            mk_rows.append(dict(
                local=loc, share=col, label=label,
                tau=round(tau, 3), p=round(p, 4),
                sens_slope=round(slope, 6), n=len(vals),
                combined_class=cc,
                inside_mpa=bool(sub["inside_any_mpa_any"].iloc[0]),
                sig=_sig(p),
            ))

    mk_df = pd.DataFrame(mk_rows)
    mk_df.to_csv(DAT / "effort_p3b_mannkendall.csv", index=False)

    log.info("P3b — Mann-Kendall effort composition (n=7):")
    for col in SHARE_COLS:
        sub = mk_df[mk_df["share"] == col]
        log.info("  %-12s  positive=%d/%d  p<0.05=%d/%d  median_τ=%+.3f",
                 col, (sub["tau"] > 0).sum(), len(sub),
                 (sub["p"] < 0.05).sum(), len(sub), sub["tau"].median())

    return mk_df


def fig_p3b_trends(mk_df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("P3b — Temporal trends in effort composition (n=7, excl. Areia Branca)",
                 fontsize=12)

    for ax, (col, label) in zip(axes, SHARE_COLS.items()):
        sub = mk_df[mk_df["share"] == col].sort_values("tau")
        if sub.empty:
            ax.set_visible(False); continue
        bar_colors = [CMAP.get(cc, "gray") for cc in sub["combined_class"]]
        ax.barh(sub["local"], sub["tau"], color=bar_colors, edgecolor="k", linewidth=0.5)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        for _, row in sub.iterrows():
            if row["sig"]:
                xpos = row["tau"] + (0.012 if row["tau"] >= 0 else -0.012)
                ha   = "left" if row["tau"] >= 0 else "right"
                ax.text(xpos, row["local"],
                        f"{row['sig']} ({row['sens_slope']:+.4f})",
                        va="center", ha=ha, fontsize=7.5)
        ax.set_xlabel("Kendall's τ", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8)

    legend_els = [Line2D([0], [0], color=c, lw=2, label=k) for k, c in CMAP.items()]
    axes[-1].legend(handles=legend_els, fontsize=7, loc="lower right", frameon=False)

    plt.tight_layout()
    out = FIG / "effort_p3b_trends.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def fig_p3b_timeseries(df_7):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("P3b — Effort share time series by exposure context (n=7)", fontsize=12)

    for ax, (col, label) in zip(axes, SHARE_COLS.items()):
        for loc in sorted(df_7["local"].unique()):
            sub = df_7[df_7["local"] == loc].sort_values("year")
            cc  = sub["combined_exposure_class"].iloc[0]
            ax.plot(sub["year"], sub[col], color=CMAP.get(cc, "gray"),
                    alpha=0.65, lw=1.3, marker="o", markersize=3)
        ax.set_xlabel("Year", fontsize=9); ax.set_ylabel("Share", fontsize=9)
        ax.set_title(label, fontsize=10); ax.tick_params(labelsize=8)

    legend_els = [Line2D([0], [0], color=c, lw=2, label=k) for k, c in CMAP.items()]
    axes[-1].legend(handles=legend_els, fontsize=7, frameon=False)

    plt.tight_layout()
    out = FIG / "effort_p3b_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


# ── P6 — CPUE ~ gear + boat + year (primary: year-controlled) ─────────────────

def p6_cpue_composition(df_7):
    df = df_7.copy()

    # Within-local demeaning
    for col in ["log_cpue", "active", "passive", "motorised"]:
        if col in df.columns:
            df[f"{col}_dm"] = df[col] - df.groupby("local")[col].transform("mean")

    # ── 1. Within-local Spearman (no year control, for comparison) ────────────
    spear_rows = []
    for share, label in [("active", "Active gear"), ("motorised", "Motorised boats")]:
        dm_col = f"{share}_dm"
        mask   = df[dm_col].notna() & df["log_cpue_dm"].notna()
        r, p   = spearmanr(df.loc[mask, dm_col], df.loc[mask, "log_cpue_dm"])
        spear_rows.append(dict(predictor=share, label=label,
                               rho=round(r, 3), p=round(p, 4), sig=_sig(p)))
        log.info("P6 Spearman (no yr control)  %-18s  ρ=%+.3f  p=%.4f %s",
                 label, r, p, _sig(p))
    spear_df = pd.DataFrame(spear_rows)
    spear_df.to_csv(DAT / "effort_p6_spearman.csv", index=False)

    # ── 2. LMM (linear, with year + gear + boat) ──────────────────────────────
    lmm_df = pd.DataFrame()
    mask_lmm = df[["active_dm", "motorised_dm", "log_cpue_dm"]].notna().all(axis=1)
    sub_lmm  = df[mask_lmm].copy()
    for method in ("lbfgs", "bfgs", "nm"):
        try:
            res = smf.mixedlm("log_cpue ~ year_c + active_dm + motorised_dm",
                               data=sub_lmm, groups=sub_lmm["local"]
                               ).fit(reml=True, method=method)
            rows = []
            for term in res.params.index:
                p_val = float(res.pvalues.get(term, np.nan))
                ci = res.conf_int()
                lo = float(ci.loc[term, 0]) if term in ci.index else np.nan
                hi = float(ci.loc[term, 1]) if term in ci.index else np.nan
                rows.append(dict(term=term,
                                 coef=round(float(res.params[term]), 4),
                                 se=round(float(res.bse[term]), 4),
                                 p=round(p_val, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                                 sig=_sig(p_val)))
            lmm_df = pd.DataFrame(rows)
            log.info("P6 LMM  log_cpue ~ year_c + active_dm + motorised_dm (method=%s):", method)
            for _, r in lmm_df[~lmm_df["term"].str.contains("Group Var")].iterrows():
                log.info("  %-30s  β=%+.4f  p=%.4f %s", r["term"], r["coef"], r["p"], r["sig"])
            break
        except Exception as e:
            log.warning("P6 LMM method=%s failed: %s", method, e)
    if not lmm_df.empty:
        lmm_df.to_csv(DAT / "effort_p6_lmm.csv", index=False)

    # ── 3. LinearGAM: s(active_dm) + s(motorised_dm) + s(year_c) [PRIMARY] ───
    feat_cols = ["active_dm", "motorised_dm", "year_c"]
    mask_gam  = df[feat_cols + ["log_cpue_dm"]].notna().all(axis=1)
    X_gam = df.loc[mask_gam, feat_cols].values
    y_gam = df.loc[mask_gam, "log_cpue_dm"].values

    gam = LinearGAM(s(0, n_splines=6) + s(1, n_splines=6) + s(2, n_splines=5))
    gam.gridsearch(X_gam, y_gam, progress=False)
    dev = gam.statistics_["pseudo_r2"]["explained_deviance"]
    log.info("P6 GAM  s(active_dm)+s(motorised_dm)+s(year_c)  dev_exp=%.3f", dev)

    partial_rows = []
    for i, col_name in enumerate(["active_dm", "motorised_dm"]):
        XX = gam.generate_X_grid(term=i, n=200)
        pd_eff, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        for j in range(len(XX)):
            partial_rows.append(dict(term=col_name, x=float(XX[j, i]),
                                     effect=float(pd_eff[j]),
                                     ci_lo=float(confi[j, 0]),
                                     ci_hi=float(confi[j, 1])))
    partial_df = pd.DataFrame(partial_rows)
    partial_df.to_csv(DAT / "effort_p6_gam_partial.csv", index=False)

    # Year smooth (term 2)
    XX_yr   = gam.generate_X_grid(term=2, n=200)
    pd_yr, ci_yr = gam.partial_dependence(term=2, X=XX_yr, width=0.95)
    yr_smooth = pd.DataFrame({
        "year_c":  XX_yr[:, 2],
        "year":    XX_yr[:, 2] + df["year"].median(),
        "effect":  pd_yr,
        "ci_lo":   ci_yr[:, 0],
        "ci_hi":   ci_yr[:, 1],
    })
    yr_smooth.to_csv(DAT / "effort_p6_year_smooth.csv", index=False)
    log.info("Saved P6 GAM partial effects and year smooth")

    return spear_df, lmm_df, partial_df, yr_smooth, dev, df


def fig_p6(df_7_merged, spear_df, partial_df, yr_smooth, dev):
    """3×2 figure: top row = scatter, bottom row = GAM partials + year smooth."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "P6 — Within-local CPUE ~ gear and boat composition\n"
        "(s(active) + s(motorised) + s(year_c); n=7, excl. Areia Branca)",
        fontsize=12)

    pairs = [
        ("active_dm",    "active",    "Share active gear − mean"),
        ("motorised_dm", "motorised", "Share motorised boats − mean"),
    ]

    for col_i, (dm_col, base_col, xlabel) in enumerate(pairs):
        ax_sc  = axes[0, col_i]
        ax_gam = axes[1, col_i]

        # Scatter (demeaned, colored by exposure class)
        for loc in sorted(df_7_merged["local"].unique()):
            sub = df_7_merged[df_7_merged["local"] == loc].dropna(
                subset=[dm_col, "log_cpue_dm"])
            cc  = sub["combined_exposure_class"].iloc[0]
            ax_sc.scatter(sub[dm_col], sub["log_cpue_dm"],
                          color=CMAP.get(cc, "gray"), s=18, alpha=0.5, zorder=2)
        ax_sc.axhline(0, color="k", lw=0.7, ls="--")
        ax_sc.axvline(0, color="k", lw=0.7, ls="--")
        rr = spear_df.loc[spear_df["predictor"] == base_col].iloc[0]
        ax_sc.set_xlabel(xlabel, fontsize=9); ax_sc.set_ylabel("log(CPUE) − mean", fontsize=9)
        ax_sc.set_title(f"Spearman ρ={rr['rho']:+.3f}  p={rr['p']:.4f} {rr['sig']}", fontsize=9)
        ax_sc.tick_params(labelsize=8)

        # GAM partial effect
        sub_p = partial_df[partial_df["term"] == dm_col]
        ax_gam.plot(sub_p["x"], sub_p["effect"], "b-", lw=2)
        ax_gam.fill_between(sub_p["x"], sub_p["ci_lo"], sub_p["ci_hi"], alpha=0.18, color="blue")
        ax_gam.axhline(0, color="k", lw=0.7, ls="--")
        ax_gam.axvline(0, color="k", lw=0.7, ls="--")
        ax_gam.set_xlabel(xlabel, fontsize=9)
        ax_gam.set_ylabel("Partial effect on log(CPUE)", fontsize=9)
        ax_gam.set_title(f"GAM partial | s(year_c)  (dev.exp.={dev:.3f})", fontsize=9)
        ax_gam.tick_params(labelsize=8)

    # Year smooth (bottom right)
    axes[0, 2].set_visible(False)
    ax_yr = axes[1, 2]
    ax_yr.plot(yr_smooth["year"], yr_smooth["effect"], "g-", lw=2)
    ax_yr.fill_between(yr_smooth["year"], yr_smooth["ci_lo"], yr_smooth["ci_hi"],
                       alpha=0.18, color="green")
    ax_yr.axhline(0, color="k", lw=0.7, ls="--")
    ax_yr.set_xlabel("Year", fontsize=9)
    ax_yr.set_ylabel("Partial effect of year on log(CPUE)", fontsize=9)
    ax_yr.set_title("s(year_c) — residual temporal trend after gear/boat control", fontsize=9)
    ax_yr.tick_params(labelsize=8)

    # Legend (top-right panel area)
    legend_els = [Line2D([0], [0], color=c, lw=2, label=k) for k, c in CMAP.items()]
    axes[0, 2].set_visible(True)
    axes[0, 2].axis("off")
    axes[0, 2].legend(handles=legend_els, fontsize=8, loc="center", frameon=False)

    plt.tight_layout()
    out = FIG / "effort_p6_cpue_composition.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def print_summary(mk_df, lmm_df, spear_df):
    HR = "─" * 68
    print("\n" + "=" * 68)
    print("P3b + P6 — EFFORT COMPOSITION DYNAMICS")
    print("=" * 68)
    print()
    print("P3b — Mann-Kendall effort trends (n=7):")
    for col in SHARE_COLS:
        sub = mk_df[mk_df["share"] == col]
        print(f"  {col:<12}:  positive={( sub['tau']>0).sum()}/{len(sub)}  "
              f"p<0.05={(sub['p']<0.05).sum()}/{len(sub)}  "
              f"median_τ={sub['tau'].median():+.3f}")
    print()
    print("P6 — LMM (n=7):")
    print(HR)
    for _, r in lmm_df[~lmm_df["term"].str.contains("Group Var")].iterrows():
        print(f"  {r['term']:<30}  β={r['coef']:+.4f}  "
              f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  p={r['p']:.4f} {r['sig']}")
    print()
    print("P6 — Spearman within-local (no year control):")
    for _, r in spear_df.iterrows():
        print(f"  {r['label']:<22}  ρ={r['rho']:+.3f}  p={r['p']:.4f} {r['sig']}")
    print("=" * 68)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df_7, locals_7 = load_data()

    log.info("── P3b: Effort composition trends (n=7) ──────────────")
    mk_df = p3b_effort_trends(df_7, locals_7)
    fig_p3b_trends(mk_df)
    fig_p3b_timeseries(df_7)

    log.info("── P6: CPUE ~ gear/boat + year control (n=7) ─────────")
    spear_df, lmm_df, partial_df, yr_smooth, dev, df_merged = p6_cpue_composition(df_7)
    fig_p6(df_merged, spear_df, partial_df, yr_smooth, dev)

    print_summary(mk_df, lmm_df, spear_df)
    log.info("17_effort_composition.py complete.")


if __name__ == "__main__":
    main()
