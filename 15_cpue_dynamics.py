#!/usr/bin/env python3
"""
15_cpue_dynamics.py — P2 + P4: CPUE temporal dynamics

P2 — Temporal trends in CPUE within localities (Mann-Kendall + GAM, n=7)
P4 — Do CPUE trajectories differ by MPA / exposure context? (LMM year×MPA, n=7 primary)
     Sensitivity: n=8 (including AB) reported to quantify AB's influence.

Primary: n=7 (excluding Areia Branca).
Sensitivity: n=8 labelled explicitly in outputs.

Outputs (data/processed/):
  cpue_p2_mannkendall.csv       MK + Sen's slope per locality (n=7)
  cpue_p2_gam_trend.csv         Regional GAM trend (n=7)
  cpue_p4_lmm.csv               LMM year_c × inside_mpa (n=7)
  cpue_p4_lmm_n8.csv            Same model, n=8 sensitivity
  cpue_p4_trajectories.csv      GAM trajectories by MPA status (n=7)
  cpue_p4_trajectories_n8.csv   Same, n=8 sensitivity

Outputs (outputs/figures/):
  cpue_p2_trends.png            MK τ barplot + time series by locality (n=7)
  cpue_p4_trajectories.png      CPUE trajectories inside vs outside MPA (n=7)
  cpue_p4_sensitivity.png       n=7 vs n=8 side-by-side LMM interaction
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
from pygam import GammaGAM, s, f
import statsmodels.formula.api as smf

sys.path.insert(0, ".")
import config_00 as cfg

warnings.filterwarnings("ignore")
cfg.setup_dirs()
logging.basicConfig(
    format=cfg.LOG_FORMAT, datefmt=cfg.LOG_DATEFMT, level=logging.INFO,
    handlers=[
        logging.FileHandler(cfg.LOGS / "15_cpue_dynamics.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("15_cpue_dynamics")

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
MPA_CLR = {"inside": "#d62728", "outside": "#1f77b4"}


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
    df[CPUE_COL]     = df["production_ton"] / df["estimated_fishermen"]
    df["log_cpue"]   = np.log(df[CPUE_COL].clip(lower=1e-6))
    df["year_c"]     = df["year"] - df["year"].median()
    df["inside_mpa"] = df["inside_any_mpa_any"].astype(int)
    locals_all = sorted(df["local"].unique())
    df["local_enc"] = df["local"].map({loc: i for i, loc in enumerate(locals_all)})

    df_7 = df[df["local"] != AB].copy()
    locals_7 = sorted(df_7["local"].unique())
    df_7["local_enc7"] = df_7["local"].map({loc: i for i, loc in enumerate(locals_7)})

    log.info("All data:  %d obs | %d locals", len(df),   df["local"].nunique())
    log.info("n=7 data:  %d obs | %d locals", len(df_7), df_7["local"].nunique())
    log.info("%s  median=%.4f  range=%.4f–%.4f",
             CPUE_COL, df_7[CPUE_COL].median(), df_7[CPUE_COL].min(), df_7[CPUE_COL].max())
    return df, df_7, locals_7


# ── P2 — temporal trends ──────────────────────────────────────────────────────

def p2_cpue_trends(df_7, locals_7):
    mk_rows = []
    for loc in locals_7:
        sub = df_7[df_7["local"] == loc].sort_values("year")
        vals = sub[CPUE_COL].dropna()
        if len(vals) < 5:
            continue
        yr = sub.loc[vals.index, "year"].values
        tau, p = kendalltau(yr, vals.values)
        slope  = _sens_slope(vals.values, yr)
        mk_rows.append(dict(
            local=loc, tau=round(tau, 3), p=round(p, 4),
            sens_slope=round(slope, 6), n=len(vals),
            combined_class=sub["combined_exposure_class"].iloc[0],
            inside_mpa=bool(sub["inside_any_mpa_any"].iloc[0]),
            sig=_sig(p),
        ))

    mk_df = pd.DataFrame(mk_rows)
    mk_df.to_csv(DAT / "cpue_p2_mannkendall.csv", index=False)

    log.info("P2 — Mann-Kendall CPUE (n=7):")
    for _, r in mk_df.iterrows():
        log.info("  %-22s  τ=%+.3f  p=%.4f %-3s  slope=%+.5f t/fish/yr",
                 r["local"], r["tau"], r["p"], r["sig"], r["sens_slope"])
    log.info("  Positive trends: %d/%d  |  Significant (p<0.05): %d/%d",
             (mk_df["tau"] > 0).sum(), len(mk_df),
             (mk_df["p"] < 0.05).sum(), len(mk_df))

    # Regional GAM: s(year_c) + f(local_enc7)
    X   = df_7[["year_c", "local_enc7"]].values
    y   = df_7[CPUE_COL].values
    gam = GammaGAM(s(0, n_splines=8) + f(1)).gridsearch(X, y, progress=False)
    dev = gam.statistics_["pseudo_r2"]["explained_deviance"]
    aic = gam.statistics_["AIC"]
    log.info("P2 — Regional GAM (n=7)  dev_exp=%.3f  AIC=%.1f", dev, aic)

    yr_median = df_7["year"].median()
    year_c_grid = np.linspace(df_7["year_c"].min(), df_7["year_c"].max(), 100)
    n_loc = len(locals_7)
    trend_rows = []
    for yc in year_c_grid:
        preds = [gam.predict(np.array([[yc, i]]))[0] for i in range(n_loc)]
        trend_rows.append(dict(year_c=yc, year=yc + yr_median,
                               cpue_trend=float(np.mean(preds))))
    trend_df = pd.DataFrame(trend_rows)
    trend_df.to_csv(DAT / "cpue_p2_gam_trend.csv", index=False)
    log.info("Saved cpue_p2_gam_trend.csv")

    return mk_df, trend_df, dev


def fig_p2(df_7, mk_df, trend_df, dev):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("P2 — CPUE temporal trends within localities (n=7, excl. Areia Branca)",
                 fontsize=12)

    # Left: trajectories + GAM
    for loc in sorted(df_7["local"].unique()):
        sub = df_7[df_7["local"] == loc].sort_values("year")
        cc  = sub["combined_exposure_class"].iloc[0]
        ax1.plot(sub["year"], sub[CPUE_COL], color=CMAP.get(cc, "gray"),
                 alpha=0.5, lw=1.3)
        ax1.scatter(sub["year"], sub[CPUE_COL], color=CMAP.get(cc, "gray"), s=14, alpha=0.5)
    ax1.plot(trend_df["year"], trend_df["cpue_trend"], "k-", lw=2.5,
             label=f"Regional GAM (dev.exp.={dev:.2f})", zorder=5)
    ax1.set_xlabel("Year", fontsize=9); ax1.set_ylabel("CPUE (t/fisherman)", fontsize=9)
    ax1.set_title("Trajectories + regional GAM trend", fontsize=10)
    legend_els  = [Line2D([0], [0], color=c, lw=2, label=k) for k, c in CMAP.items()]
    legend_els += [Line2D([0], [0], color="k", lw=2.5, label="Regional GAM")]
    ax1.legend(handles=legend_els, fontsize=7, loc="upper left", frameon=False)

    # Right: MK τ barplot
    mk_s = mk_df.sort_values("tau")
    bar_colors = [CMAP.get(cc, "gray") for cc in mk_s["combined_class"]]
    ax2.barh(mk_s["local"], mk_s["tau"], color=bar_colors, edgecolor="k", linewidth=0.5)
    ax2.axvline(0, color="k", lw=0.8, ls="--")
    for _, row in mk_s.iterrows():
        if row["sig"]:
            xpos = row["tau"] + (0.01 if row["tau"] >= 0 else -0.01)
            ha   = "left" if row["tau"] >= 0 else "right"
            ax2.text(xpos, row["local"], f"{row['sig']} ({row['sens_slope']:+.3f})",
                     va="center", ha=ha, fontsize=7.5)
    ax2.set_xlabel("Kendall's τ", fontsize=9)
    ax2.set_title("Mann-Kendall trend  (τ) + Sen's slope (t/fish/yr)", fontsize=10)
    ax2.tick_params(labelsize=8)

    plt.tight_layout()
    out = FIG / "cpue_p2_trends.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


# ── P4 — trajectories by exposure ─────────────────────────────────────────────

def _fit_lmm(data, label=""):
    """LMM log_cpue ~ year_c * inside_mpa with (1|local)."""
    rows = []
    for method in ("lbfgs", "bfgs", "nm"):
        try:
            res = smf.mixedlm("log_cpue ~ year_c * inside_mpa",
                               data=data, groups=data["local"]).fit(reml=True, method=method)
            for term in res.params.index:
                p_val = float(res.pvalues.get(term, np.nan))
                ci = res.conf_int()
                lo = float(ci.loc[term, 0]) if term in ci.index else np.nan
                hi = float(ci.loc[term, 1]) if term in ci.index else np.nan
                rows.append(dict(
                    term=term,
                    coef=round(float(res.params[term]), 4),
                    se=round(float(res.bse[term]), 4),
                    p=round(p_val, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                    sig=_sig(p_val),
                ))
            log.info("P4 LMM %-10s (method=%s, n_loc=%d):",
                     label, method, data["local"].nunique())
            for r in rows:
                if "year_c" in r["term"]:
                    log.info("  %-35s  β=%+.4f  p=%.4f %-3s",
                             r["term"], r["coef"], r["p"], r["sig"])
            return pd.DataFrame(rows)
        except Exception as e:
            log.warning("P4 LMM %s method=%s failed: %s", label, method, e)
    return pd.DataFrame()


def _fit_group_gam(sub_df, n_splines=6):
    """GammaGAM(s(year_c) + f(local)) for a group subset."""
    locals_sub = sorted(sub_df["local"].unique())
    enc = {loc: i for i, loc in enumerate(locals_sub)}
    sub = sub_df.copy()
    sub["_enc"] = sub["local"].map(enc)
    X = sub[["year_c", "_enc"]].values
    y = sub[CPUE_COL].values
    gam = GammaGAM(s(0, n_splines=n_splines) + f(1)).gridsearch(X, y, progress=False)
    return gam, len(locals_sub)


def _gam_trajectories(data, yr_median):
    """Compute GAM trajectories by MPA status for a given dataset."""
    year_c_grid = np.linspace(data["year_c"].min(), data["year_c"].max(), 100)
    rows = []
    for mpa_val, label in [(1, "inside"), (0, "outside")]:
        sub = data[data["inside_mpa"] == mpa_val]
        if sub["local"].nunique() < 2:
            log.warning("P4 GAM [mpa=%s]: only %d locality, skipping", label, sub["local"].nunique())
            continue
        gam, n_locs = _fit_group_gam(sub)
        dev = gam.statistics_["pseudo_r2"]["explained_deviance"]
        log.info("P4 GAM [mpa=%s, n_loc=%d]  dev_exp=%.3f", label, n_locs, dev)
        for yc in year_c_grid:
            preds = [gam.predict(np.array([[yc, i]]))[0] for i in range(n_locs)]
            cis   = [gam.confidence_intervals(np.array([[yc, i]]), width=0.95)[0]
                     for i in range(n_locs)]
            rows.append(dict(
                year_c=yc, year=yc + yr_median, group=label,
                cpue_pred=float(np.mean(preds)),
                ci_lo=float(np.mean([c[0] for c in cis])),
                ci_hi=float(np.mean([c[1] for c in cis])),
            ))
    return pd.DataFrame(rows)


def p4_trajectories(df, df_7):
    yr7  = df_7["year"].median()
    yr8  = df["year"].median()

    # Primary: n=7
    lmm7  = _fit_lmm(df_7,  "n=7")
    traj7 = _gam_trajectories(df_7, yr7)

    # Sensitivity: n=8
    lmm8  = _fit_lmm(df,    "n=8")
    traj8 = _gam_trajectories(df, yr8)

    if not lmm7.empty:
        lmm7.to_csv(DAT / "cpue_p4_lmm.csv", index=False)
    if not lmm8.empty:
        lmm8.to_csv(DAT / "cpue_p4_lmm_n8.csv", index=False)
    traj7.to_csv(DAT / "cpue_p4_trajectories.csv", index=False)
    traj8.to_csv(DAT / "cpue_p4_trajectories_n8.csv", index=False)
    log.info("Saved P4 trajectories (n=7 + n=8 sensitivity)")

    return lmm7, lmm8, traj7, traj8


def fig_p4_trajectories(df_7, traj7, lmm7):
    """P4 primary figure: GAM trajectories inside vs outside MPA (n=7)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("P4 — CPUE trajectories: inside vs outside MPA (n=7, excl. Areia Branca)",
                 fontsize=12)

    for label, color in MPA_CLR.items():
        sub = traj7[traj7["group"] == label]
        if sub.empty:
            continue
        ax.plot(sub["year"], sub["cpue_pred"], color=color, lw=2.2, label=label)
        ax.fill_between(sub["year"], sub["ci_lo"], sub["ci_hi"], color=color, alpha=0.15)

    # Raw annual means as scatter
    for mpa_val, label in [(True, "inside"), (False, "outside")]:
        sub_raw = df_7[df_7["inside_any_mpa_any"] == mpa_val]
        yr_mean = sub_raw.groupby("year")[CPUE_COL].mean()
        ax.scatter(yr_mean.index, yr_mean.values, color=MPA_CLR[label],
                   s=22, alpha=0.4, zorder=4)

    # Interaction term annotation
    if not lmm7.empty:
        int_row = lmm7[lmm7["term"] == "year_c:inside_mpa"]
        if not int_row.empty:
            r = int_row.iloc[0]
            ax.annotate(f"Interaction year×MPA:\nβ = {r['coef']:+.4f}  p = {r['p']:.4f} {r['sig']}",
                        xy=(0.03, 0.93), xycoords="axes fraction",
                        fontsize=9, color="#333333",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    ax.set_xlabel("Year", fontsize=10); ax.set_ylabel("CPUE (t/fisherman)", fontsize=10)
    ax.legend(fontsize=10, frameon=False)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    out = FIG / "cpue_p4_trajectories.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def fig_p4_sensitivity(df, df_7, traj7, traj8, lmm7, lmm8):
    """Side-by-side n=7 vs n=8: trajectories + LMM interaction comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("P4 sensitivity — n=7 (excl. AB) vs n=8 (incl. AB)", fontsize=12)

    datasets = [
        (df_7, traj7, lmm7, "n=7 (excl. Areia Branca)", axes[0]),
        (df,   traj8, lmm8, "n=8 (incl. Areia Branca — sensitivity)", axes[1]),
    ]

    for data, traj, lmm, title, ax in datasets:
        for label, color in MPA_CLR.items():
            sub = traj[traj["group"] == label]
            if sub.empty:
                continue
            ax.plot(sub["year"], sub["cpue_pred"], color=color, lw=2.2, label=label)
            ax.fill_between(sub["year"], sub["ci_lo"], sub["ci_hi"], color=color, alpha=0.15)
        for mpa_val, label in [(True, "inside"), (False, "outside")]:
            yr_mean = data[data["inside_any_mpa_any"] == mpa_val].groupby("year")[CPUE_COL].mean()
            ax.scatter(yr_mean.index, yr_mean.values, color=MPA_CLR[label],
                       s=18, alpha=0.35, zorder=4)
        if not lmm.empty:
            int_row = lmm[lmm["term"] == "year_c:inside_mpa"]
            if not int_row.empty:
                r = int_row.iloc[0]
                ax.set_title(f"{title}\nInteraction β={r['coef']:+.4f}  p={r['p']:.4f} {r['sig']}",
                             fontsize=9)
            else:
                ax.set_title(title, fontsize=9)
        ax.set_xlabel("Year", fontsize=9); ax.set_ylabel("CPUE (t/fisherman)", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    out = FIG / "cpue_p4_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def print_summary(mk_df, lmm7, lmm8):
    HR = "─" * 68
    print("\n" + "=" * 68)
    print("P2 + P4 — CPUE TEMPORAL DYNAMICS")
    print("=" * 68)
    print()
    print("P2 — Mann-Kendall (n=7):")
    print(f"  {'Local':<22} {'τ':>7}  {'p':>7}  {'sig':>4}  {'slope':>10}")
    print(HR)
    for _, r in mk_df.sort_values("tau").iterrows():
        print(f"  {r['local']:<22} {r['tau']:>+7.3f}  {r['p']:>7.4f}  {r['sig']:>4}  "
              f"{r['sens_slope']:>+10.5f}")
    print()
    print(f"  Positive: {(mk_df['tau'] > 0).sum()}/{len(mk_df)}  "
          f"Significant p<0.05: {(mk_df['p'] < 0.05).sum()}/{len(mk_df)}")

    print()
    print("P4 — LMM year_c × inside_mpa:")
    for label, lmm in [("n=7 (primary)", lmm7), ("n=8 (sensitivity)", lmm8)]:
        print(f"\n  {label}:")
        if lmm.empty:
            print("    (model failed)")
            continue
        for _, r in lmm[~lmm["term"].str.contains("Group Var")].iterrows():
            print(f"    {r['term']:<35}  β={r['coef']:+.4f}  p={r['p']:.4f} {r['sig']}")
    print("=" * 68)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df, df_7, locals_7 = load_data()

    log.info("── P2: CPUE temporal trends (n=7) ────────────────────")
    mk_df, trend_df, dev = p2_cpue_trends(df_7, locals_7)
    fig_p2(df_7, mk_df, trend_df, dev)

    log.info("── P4: Trajectories by exposure (n=7 primary + n=8 sensitivity)")
    lmm7, lmm8, traj7, traj8 = p4_trajectories(df, df_7)
    fig_p4_trajectories(df_7, traj7, lmm7)
    fig_p4_sensitivity(df, df_7, traj7, traj8, lmm7, lmm8)

    print_summary(mk_df, lmm7, lmm8)
    log.info("15_cpue_dynamics.py complete.")


if __name__ == "__main__":
    main()
