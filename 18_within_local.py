#!/usr/bin/env python3
"""
18_within_local.py — P7 + P8: Within-local CPUE associations

P7 — Is there a density-dependent relationship between effort level and CPUE?
     LinearGAM s(log_fishermen_dm) on within-local demeaned data (n=7)

P8 — Is CPUE associated with species diversity, controlling for non-linear
     temporal trend?  LinearGAM s(diversity_dm) + s(year_c) (n=7)
     Year-controlled GAM is the PRIMARY analysis for P8.

Outputs (data/processed/):
  within_p7_density_dep.csv       GAM partial: s(log_fishermen) — density-dep
  within_p8_spearman.csv          Within-local Spearman (diversity ~ CPUE, no year ctrl)
  within_p8_gam_partial.csv       GAM partial: s(diversity_dm) | year controlled
  within_p8_year_smooth.csv       Year smooth from P8 GAM

Outputs (outputs/figures/):
  within_p7_density_dep.png       Scatter + GAM: log_fish_dm vs log_cpue_dm
  within_p8_cpue_diversity.png    3×3: scatter + year-uncontrolled + year-controlled GAM
"""

import logging
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pygam import LinearGAM, s
import statsmodels.formula.api as smf

sys.path.insert(0, ".")
import config_00 as cfg

warnings.filterwarnings("ignore")
cfg.setup_dirs()
logging.basicConfig(
    format=cfg.LOG_FORMAT, datefmt=cfg.LOG_DATEFMT, level=logging.INFO,
    handlers=[
        logging.FileHandler(cfg.LOGS / "18_within_local.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("18_within_local")

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

DIV_COLS = {
    "richness":  "Species richness (S)",
    "shannon_h": "Shannon H′",
    "pielou_j":  "Pielou J′",
}


def _sig(p):
    if np.isnan(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""


# ── data ──────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DAT / "productivity_local_year.csv")
    df[CPUE_COL]   = df["production_ton"] / df["estimated_fishermen"]
    df["log_cpue"] = np.log(df[CPUE_COL].clip(lower=1e-6))
    df["year_c"]   = df["year"] - df["year"].median()
    df["log_fish"] = np.log(df["estimated_fishermen"].clip(lower=0.1))

    df_7 = df[df["local"] != AB].copy()

    # Within-local demeaning
    for col in ["log_cpue", "log_fish", "year_c"] + list(DIV_COLS.keys()):
        if col in df_7.columns:
            df_7[f"{col}_dm"] = df_7[col] - df_7.groupby("local")[col].transform("mean")

    log.info("n=7: %d obs | locals: %d", len(df_7), df_7["local"].nunique())
    return df_7


# ── P7 — density-dependence: CPUE ~ effort level ──────────────────────────────

def p7_density_dep(df_7):
    # LMM (linear reference)
    beta, p_lin = np.nan, np.nan
    for method in ("lbfgs", "bfgs", "nm"):
        try:
            res = smf.mixedlm("log_cpue ~ log_fish_dm", data=df_7,
                               groups=df_7["local"]).fit(reml=True, method=method)
            beta  = float(res.params["log_fish_dm"])
            p_lin = float(res.pvalues["log_fish_dm"])
            log.info("P7 LMM  log_cpue ~ log_fish_dm (method=%s):  β=%+.3f  p=%.3f %s",
                     method, beta, p_lin, _sig(p_lin))
            break
        except Exception as e:
            log.warning("P7 LMM method=%s failed: %s", method, e)

    # LinearGAM (non-linear)
    mask = df_7[["log_fish_dm", "log_cpue_dm"]].notna().all(axis=1)
    X_dm = df_7.loc[mask, "log_fish_dm"].values.reshape(-1, 1)
    y_dm = df_7.loc[mask, "log_cpue_dm"].values

    gam = LinearGAM(s(0, n_splines=6)).gridsearch(X_dm, y_dm, progress=False)
    dev = gam.statistics_["pseudo_r2"]["explained_deviance"]
    log.info("P7 GAM  s(log_fish_dm)  dev_exp=%.3f", dev)

    XX = gam.generate_X_grid(term=0, n=200)
    pd_eff, confi = gam.partial_dependence(term=0, X=XX, width=0.95)
    eff_df = pd.DataFrame({
        "log_fish_dm":  XX[:, 0],
        "cpue_effect":  pd_eff,
        "ci_lo":        confi[:, 0],
        "ci_hi":        confi[:, 1],
    })
    eff_df.to_csv(DAT / "within_p7_density_dep.csv", index=False)
    log.info("Saved within_p7_density_dep.csv")

    return eff_df, beta, p_lin, dev, df_7[mask].copy()


def fig_p7(df_mask, eff_df, beta, p_lin, dev):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("P7 — Within-local CPUE ~ fishing effort (density-dependence; n=7)",
                 fontsize=12)

    # Scatter
    for loc in sorted(df_mask["local"].unique()):
        sub = df_mask[df_mask["local"] == loc]
        cc  = sub["combined_exposure_class"].iloc[0]
        ax1.scatter(sub["log_fish_dm"], sub["log_cpue_dm"],
                    color=CMAP.get(cc, "gray"), s=18, alpha=0.5, zorder=2)
    ax1.axhline(0, color="k", lw=0.7, ls="--")
    ax1.axvline(0, color="k", lw=0.7, ls="--")
    ax1.set_xlabel("log(fishermen) − local mean", fontsize=9)
    ax1.set_ylabel("log(CPUE) − local mean", fontsize=9)
    ax1.set_title(f"Within-local scatter  LMM β={beta:+.3f}  p={p_lin:.3f} {_sig(p_lin)}",
                  fontsize=9)
    ax1.tick_params(labelsize=8)

    # GAM partial
    ax2.plot(eff_df["log_fish_dm"], eff_df["cpue_effect"], "b-", lw=2)
    ax2.fill_between(eff_df["log_fish_dm"], eff_df["ci_lo"], eff_df["ci_hi"],
                     alpha=0.18, color="blue")
    ax2.axhline(0, color="k", lw=0.7, ls="--")
    ax2.axvline(0, color="k", lw=0.7, ls="--")
    ax2.set_xlabel("log(fishermen) − local mean", fontsize=9)
    ax2.set_ylabel("Partial effect on log(CPUE)", fontsize=9)
    ax2.set_title(f"GAM s(log_fish_dm)  dev.exp.={dev:.3f}", fontsize=9)
    ax2.tick_params(labelsize=8)

    plt.tight_layout()
    out = FIG / "within_p7_density_dep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


# ── P8 — CPUE ~ species diversity + year control ──────────────────────────────

def p8_cpue_diversity(df_7):
    # Within-local Spearman (reference, no year control)
    spear_rows = []
    for col, label in DIV_COLS.items():
        dm_col = f"{col}_dm"
        mask   = df_7[[dm_col, "log_cpue_dm"]].notna().all(axis=1)
        r, p   = spearmanr(df_7.loc[mask, dm_col], df_7.loc[mask, "log_cpue_dm"])
        spear_rows.append(dict(metric=col, label=label,
                               rho=round(r, 3), p=round(p, 4), sig=_sig(p)))
        log.info("P8 Spearman (no yr ctrl)  %-20s  ρ=%+.3f  p=%.4f %s", label, r, p, _sig(p))
    spear_df = pd.DataFrame(spear_rows)
    spear_df.to_csv(DAT / "within_p8_spearman.csv", index=False)

    # Year-controlled GAM: s(diversity_dm) + s(year_c)  [PRIMARY]
    partial_rows = []
    yr_smooth_rows = []
    gam_stats = []

    for col, label in DIV_COLS.items():
        dm_col = f"{col}_dm"
        mask   = df_7[[dm_col, "year_c", "log_cpue_dm"]].notna().all(axis=1)
        X = df_7.loc[mask, [dm_col, "year_c"]].values
        y = df_7.loc[mask, "log_cpue_dm"].values

        try:
            gam = LinearGAM(s(0, n_splines=6) + s(1, n_splines=5)).gridsearch(
                X, y, progress=False)
            dev = gam.statistics_["pseudo_r2"]["explained_deviance"]
            log.info("P8 GAM  s(%s_dm)+s(year_c)  dev_exp=%.3f", col, dev)
            gam_stats.append((col, dev))

            # Diversity partial (term 0)
            XX = gam.generate_X_grid(term=0, n=200)
            pd_eff, confi = gam.partial_dependence(term=0, X=XX, width=0.95)
            for j in range(len(XX)):
                partial_rows.append(dict(metric=col, x=float(XX[j, 0]),
                                         effect=float(pd_eff[j]),
                                         ci_lo=float(confi[j, 0]),
                                         ci_hi=float(confi[j, 1])))

            # Year smooth (term 1) — same for all diversity metrics, save once
            if col == "richness":
                XX_yr = gam.generate_X_grid(term=1, n=200)
                pd_yr, ci_yr = gam.partial_dependence(term=1, X=XX_yr, width=0.95)
                for j in range(len(XX_yr)):
                    yr_smooth_rows.append(dict(
                        year_c  = float(XX_yr[j, 1]),
                        year    = float(XX_yr[j, 1]) + df_7["year"].median(),
                        effect  = float(pd_yr[j]),
                        ci_lo   = float(ci_yr[j, 0]),
                        ci_hi   = float(ci_yr[j, 1]),
                    ))
        except Exception as e:
            log.warning("P8 GAM %s failed: %s", col, e)

    partial_df  = pd.DataFrame(partial_rows)
    yr_smooth   = pd.DataFrame(yr_smooth_rows)
    partial_df.to_csv(DAT / "within_p8_gam_partial.csv", index=False)
    yr_smooth.to_csv(DAT / "within_p8_year_smooth.csv", index=False)
    log.info("Saved P8 GAM partial effects and year smooth")

    return spear_df, partial_df, yr_smooth, gam_stats


def fig_p8(df_7, spear_df, partial_df, yr_smooth, gam_stats):
    """3×3 figure: col per diversity metric; row 0=scatter, row 1=year-ctrl GAM, row 2=year smooth."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle(
        "P8 — Within-local CPUE ~ species diversity\n"
        "(s(diversity_dm) + s(year_c); n=7, excl. Areia Branca)",
        fontsize=12)

    dev_map = {col: dev for col, dev in gam_stats}

    for col_i, (col, label) in enumerate(DIV_COLS.items()):
        dm_col   = f"{col}_dm"
        ax_sc    = axes[0, col_i]
        ax_gam   = axes[1, col_i]
        ax_yr    = axes[2, col_i]

        # Row 0: scatter
        for loc in sorted(df_7["local"].unique()):
            sub = df_7[df_7["local"] == loc].dropna(subset=[dm_col, "log_cpue_dm"])
            cc  = sub["combined_exposure_class"].iloc[0]
            ax_sc.scatter(sub[dm_col], sub["log_cpue_dm"],
                          color=CMAP.get(cc, "gray"), s=16, alpha=0.5, zorder=2)
        ax_sc.axhline(0, color="k", lw=0.7, ls="--")
        ax_sc.axvline(0, color="k", lw=0.7, ls="--")
        rr = spear_df.loc[spear_df["metric"] == col].iloc[0]
        ax_sc.set_xlabel(f"{label} − mean", fontsize=8)
        ax_sc.set_ylabel("log(CPUE) − mean", fontsize=8)
        ax_sc.set_title(f"Spearman (no yr ctrl)\nρ={rr['rho']:+.3f}  p={rr['p']:.4f} {rr['sig']}",
                        fontsize=8)
        ax_sc.tick_params(labelsize=7)

        # Row 1: year-controlled GAM partial
        sub_p = partial_df[partial_df["metric"] == col]
        if not sub_p.empty:
            dev_val = dev_map.get(col, np.nan)
            ax_gam.plot(sub_p["x"], sub_p["effect"], "b-", lw=2)
            ax_gam.fill_between(sub_p["x"], sub_p["ci_lo"], sub_p["ci_hi"],
                                alpha=0.18, color="blue")
            ax_gam.axhline(0, color="k", lw=0.7, ls="--")
            ax_gam.axvline(0, color="k", lw=0.7, ls="--")
            ax_gam.set_xlabel(f"{label} − mean", fontsize=8)
            ax_gam.set_ylabel("Partial effect | s(year)", fontsize=8)
            ax_gam.set_title(f"GAM partial (year-controlled)\ndev.exp.={dev_val:.3f}", fontsize=8)
            ax_gam.tick_params(labelsize=7)

        # Row 2: year smooth (same data, shown per column for clarity)
        if not yr_smooth.empty:
            ax_yr.plot(yr_smooth["year"], yr_smooth["effect"], "g-", lw=2)
            ax_yr.fill_between(yr_smooth["year"], yr_smooth["ci_lo"], yr_smooth["ci_hi"],
                               alpha=0.18, color="green")
            ax_yr.axhline(0, color="k", lw=0.7, ls="--")
            ax_yr.set_xlabel("Year", fontsize=8)
            ax_yr.set_ylabel("Partial effect of year", fontsize=8)
            ax_yr.set_title("s(year_c) — residual temporal trend", fontsize=8)
            ax_yr.tick_params(labelsize=7)

    plt.tight_layout()
    out = FIG / "within_p8_cpue_diversity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def print_summary(eff_df, beta, p_lin, spear_df, gam_stats):
    HR = "─" * 68
    print("\n" + "=" * 68)
    print("P7 + P8 — WITHIN-LOCAL CPUE ASSOCIATIONS")
    print("=" * 68)
    print()
    print("P7 — Density-dependence  CPUE ~ log(fishermen_dm)  (n=7):")
    print(f"  LMM β = {beta:+.3f}  p = {p_lin:.4f} {_sig(p_lin)}")
    print(f"  Note: β < 0 → more fishermen → lower CPUE (density-dependent)")
    print(f"        β > 0 → positive effect (scale/technology)")
    print()
    print("P8 — CPUE ~ diversity (year-controlled GAM)  (n=7):")
    print(f"  {'Metric':<14}  {'Spearman ρ':>12}  {'sig':>4}  {'GAM dev.exp.':>14}")
    print(HR)
    dev_map = {col: dev for col, dev in gam_stats}
    for _, r in spear_df.iterrows():
        dev_val = dev_map.get(r["metric"], np.nan)
        print(f"  {r['label']:<14}  {r['rho']:>+12.3f}  {r['sig']:>4}  {dev_val:>14.3f}")
    print("=" * 68)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df_7 = load_data()

    log.info("── P7: Density-dependence (n=7) ──────────────────────")
    eff_df, beta, p_lin, dev7, df_mask = p7_density_dep(df_7)
    fig_p7(df_mask, eff_df, beta, p_lin, dev7)

    log.info("── P8: CPUE ~ diversity + year control (n=7) ─────────")
    spear_df, partial_df, yr_smooth, gam_stats = p8_cpue_diversity(df_7)
    fig_p8(df_7, spear_df, partial_df, yr_smooth, gam_stats)

    print_summary(eff_df, beta, p_lin, spear_df, gam_stats)
    log.info("18_within_local.py complete.")


if __name__ == "__main__":
    main()
