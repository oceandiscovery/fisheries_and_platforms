#!/usr/bin/env python3
"""
13_ab_case.py — Areia Branca as FAD case study (Question PA)

PA — Did offshore platforms act as Fish Aggregating Devices (FAD) for large
pelagics in Areia Branca?

Approach: descriptive-mechanistic comparison of Areia Branca (AB) against
Porto do Mangue (same exposure class, no transition) and the system mean
of the remaining 7 localities.  No formal statistical model — inference is
indirect via chronology, spatial specificity, and species ecology.

Sub-questions:
  PA.1 — Transition chronology vs platform expansion (2008–2009)
  PA.2 — Gear reorganization: abandonment of covo/lagosta, adoption of linha
  PA.3 — AB vs Porto do Mangue: what differs?
  PA.4 — Diversity pattern consistent with pelagic longline bycatch?

Inputs:
  data/processed/productivity_local_year.csv
  data/processed/species_share_local_year.csv
  data/processed/gear_share_local_year.csv
  data/processed/boat_share_local_year.csv

Outputs (data/processed/):
  ab_case_metrics.csv        Key annual metrics for AB, PdM, system mean
  ab_case_gear_shares.csv    Gear share time series (AB + PdM)
  ab_case_species_top.csv    Top species shares over time for AB

Outputs (outputs/figures/):
  ab_case_transitions.png    Multi-panel timeline: CPUE, Albacora, Linha, H', richness
  ab_case_species.png        AB species composition stacked + PdM comparison
  ab_case_pa3_comparison.png AB vs PdM vs system: key metric summary
"""

import logging
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import config_00 as cfg

warnings.filterwarnings("ignore")
cfg.setup_dirs()
logging.basicConfig(
    format=cfg.LOG_FORMAT, datefmt=cfg.LOG_DATEFMT, level=logging.INFO,
    handlers=[
        logging.FileHandler(cfg.LOGS / "13_ab_case.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("13_ab_case")

FIG = cfg.OUTPUTS / "figures"
DAT = cfg.DATA_PROCESSED

AB   = "AREIA BRANCA"
PDM  = "PORTO DO MANGUE"
CPUE_COL = "cpue_per_fisherman"

CLR_AB  = "#d62728"   # red — AB
CLR_PDM = "#ff7f0e"   # orange — Porto do Mangue
CLR_SYS = "#666666"   # gray — system mean (other 7)


# ── data ──────────────────────────────────────────────────────────────────────

def load_productivity():
    df = pd.read_csv(DAT / "productivity_local_year.csv")
    df[CPUE_COL] = df["production_ton"] / df["estimated_fishermen"]
    return df


def load_species():
    return pd.read_csv(DAT / "species_share_local_year.csv")


def load_gear():
    gs = pd.read_csv(DAT / "gear_share_local_year.csv")
    # Aggregate to active/passive share by local×year
    agg = (gs.groupby(["local", "year", "gear_group"])["gear_share"]
             .sum().unstack(fill_value=0).reset_index())
    agg.columns.name = None
    for col in ["active", "passive", "mixed"]:
        if col not in agg.columns:
            agg[col] = 0.0
    # Linha (longline) = gear_cod "LIN" (pure linha / palangre de fundo)
    # All codes starting with "L" that include "linha" also counted for broader signal
    gs["is_linha"] = gs["gear_cod"].str.upper().str.match(r"^LIN$")
    linha = (gs.groupby(["local", "year"])
               .apply(lambda x: x.loc[x["is_linha"], "gear_share"].sum())
               .reset_index(name="linha_share"))
    # Covo lagosta = CVL (covo lagosta) and CPL (covo peixe e lagosta)
    gs["is_covo"] = gs["gear_cod"].str.upper().isin(["CVL", "CPL"])
    covo = (gs.groupby(["local", "year"])
              .apply(lambda x: x.loc[x["is_covo"], "gear_share"].sum())
              .reset_index(name="covo_share"))
    result = agg.merge(linha, on=["local", "year"], how="left") \
                .merge(covo, on=["local", "year"], how="left")
    result["linha_share"] = result["linha_share"].fillna(0)
    result["covo_share"]  = result["covo_share"].fillna(0)
    return result


def build_metrics(prod, gear, species):
    # System mean: mean over localities excluding AB (n=7 others)
    others = prod[prod["local"] != AB].copy()
    sys_mean = (others.groupby("year").agg(
        cpue_sys      = (CPUE_COL, "mean"),
        shannon_sys   = ("shannon_h", "mean"),
        richness_sys  = ("richness", "mean"),
        pielou_sys    = ("pielou_j", "mean"),
    ).reset_index())

    # AB and PdM productivity
    ab  = prod[prod["local"] == AB].sort_values("year")
    pdm = prod[prod["local"] == PDM].sort_values("year")

    # Albacora share (species_share_local_year)
    alb = species[species["species"] == "Albacora"][["local", "year", "species_share"]] \
              .rename(columns={"species_share": "albacora_share"})
    ab  = ab.merge(alb[alb["local"] == AB],  on=["local", "year"], how="left")
    pdm = pdm.merge(alb[alb["local"] == PDM], on=["local", "year"], how="left")
    ab["albacora_share"]  = ab["albacora_share"].fillna(0)
    pdm["albacora_share"] = pdm["albacora_share"].fillna(0)

    # Lagosta share
    lag = species[species["species"] == "Lagosta"][["local", "year", "species_share"]] \
              .rename(columns={"species_share": "lagosta_share"})
    ab  = ab.merge(lag[lag["local"] == AB],  on=["local", "year"], how="left")
    pdm = pdm.merge(lag[lag["local"] == PDM], on=["local", "year"], how="left")
    ab["lagosta_share"]  = ab["lagosta_share"].fillna(0)
    pdm["lagosta_share"] = pdm["lagosta_share"].fillna(0)

    # Gear shares
    ab  = ab.merge(gear[gear["local"] == AB][["year", "active", "passive", "linha_share", "covo_share"]],
                   on="year", how="left")
    pdm = pdm.merge(gear[gear["local"] == PDM][["year", "active", "passive", "linha_share", "covo_share"]],
                    on="year", how="left")

    log.info("AB years: %d  |  PdM years: %d", len(ab), len(pdm))
    log.info("AB Albacora share: min=%.3f  max=%.3f  mean=%.3f",
             ab["albacora_share"].min(), ab["albacora_share"].max(), ab["albacora_share"].mean())
    log.info("AB Linha share: min=%.3f  max=%.3f  mean=%.3f",
             ab["linha_share"].min(), ab["linha_share"].max(), ab["linha_share"].mean())

    return ab, pdm, sys_mean


def save_outputs(ab, pdm, sys_mean, gear, species):
    # Key metrics CSV
    metrics = ab[["year", CPUE_COL, "shannon_h", "richness", "pielou_j",
                  "albacora_share", "lagosta_share", "linha_share", "covo_share",
                  "fleet_monitored", "estimated_fishermen"]].copy()
    metrics["local"] = AB
    pdm_out = pdm[["year", CPUE_COL, "shannon_h", "richness", "pielou_j",
                   "albacora_share", "lagosta_share", "linha_share", "covo_share",
                   "fleet_monitored", "estimated_fishermen"]].copy()
    pdm_out["local"] = PDM
    metrics = pd.concat([metrics, pdm_out], ignore_index=True)
    metrics.to_csv(DAT / "ab_case_metrics.csv", index=False)
    log.info("Saved ab_case_metrics.csv")

    # Top species shares for AB
    ab_sp = species[species["local"] == AB].copy()
    top10 = (ab_sp.groupby("species")["sp_production_ton"].sum()
                  .sort_values(ascending=False).head(10).index.tolist())
    ab_top = ab_sp[ab_sp["species"].isin(top10)][["year", "species", "species_share"]]
    ab_top.to_csv(DAT / "ab_case_species_top.csv", index=False)
    log.info("Saved ab_case_species_top.csv")


# ── figures ───────────────────────────────────────────────────────────────────

def fig_transitions(ab, pdm, sys_mean):
    """5-panel timeline: CPUE, Albacora share, Linha share, Shannon H', richness."""
    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.32)
    ax_cpue  = fig.add_subplot(gs[0, :])   # full width
    ax_alb   = fig.add_subplot(gs[1, 0])
    ax_linha = fig.add_subplot(gs[1, 1])
    ax_h     = fig.add_subplot(gs[2, 0])
    ax_rich  = fig.add_subplot(gs[2, 1])

    years_all = sorted(set(ab["year"]) | set(pdm["year"]) | set(sys_mean["year"]))

    def _plot_trio(ax, ab_col, pdm_col, sys_col, ylabel, title, ymin=None, ymax=None):
        # System mean band — only when sys_col is provided
        if sys_col is not None:
            sys_vals = sys_mean.set_index("year").reindex(years_all)
            ax.fill_between(sys_vals.index,
                            sys_vals[sys_col] * 0.7, sys_vals[sys_col] * 1.3,
                            color=CLR_SYS, alpha=0.10, zorder=1)
            ax.plot(sys_mean["year"], sys_mean[sys_col], color=CLR_SYS, lw=1.5,
                    ls="--", label="System mean (n=7)", zorder=2)
        # Porto do Mangue
        ax.plot(pdm.dropna(subset=[pdm_col])["year"],
                pdm.dropna(subset=[pdm_col])[pdm_col],
                color=CLR_PDM, lw=2, marker="s", ms=4, label="Porto do Mangue", zorder=3)
        # AB
        ax.plot(ab.dropna(subset=[ab_col])["year"],
                ab.dropna(subset=[ab_col])[ab_col],
                color=CLR_AB, lw=2.5, marker="o", ms=5, label="Areia Branca", zorder=4)
        # Transition marker
        ax.axvspan(2007.5, 2009.5, alpha=0.10, color="#f0c040", zorder=0)
        ax.axvline(2008.5, color="#c09000", lw=1, ls=":", alpha=0.8)
        ax.set_xlabel("Year", fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10); ax.tick_params(labelsize=8)
        if ymin is not None: ax.set_ylim(bottom=ymin)
        if ymax is not None: ax.set_ylim(top=ymax)

    _plot_trio(ax_cpue,  CPUE_COL, CPUE_COL, "cpue_sys",
               "CPUE (t/fisherman)", "PA.1 — CPUE: Areia Branca vs comparators", ymin=0)

    # Albacora and Linha: sys_col=None — system mean added manually below
    _plot_trio(ax_alb,   "albacora_share", "albacora_share", None,
               "Albacora share (prop.)", "PA.1 — Albacora (yellowfin tuna) share", ymin=0, ymax=1)

    _plot_trio(ax_linha, "linha_share", "linha_share", None,
               "Linha (longline) share", "PA.2 — Longline gear share", ymin=0, ymax=1)

    _plot_trio(ax_h,     "shannon_h",  "shannon_h",  "shannon_sys",
               "Shannon H′", "PA.4 — Species diversity H′")

    _plot_trio(ax_rich,  "richness",   "richness",   "richness_sys",
               "Species richness (S)", "PA.4 — Species richness")

    # Add system mean lines for Albacora and Linha panels
    from pandas import read_csv
    sp_long = read_csv(DAT / "species_share_local_year.csv")
    others_sp = sp_long[sp_long["local"] != AB]
    sys_alb = (others_sp[others_sp["species"] == "Albacora"]
               .groupby("year")["species_share"].mean().reset_index())
    if not sys_alb.empty:
        ax_alb.plot(sys_alb["year"], sys_alb["species_share"], color=CLR_SYS, lw=1.5,
                    ls="--", zorder=2, label="System mean (n=7)")

    gs_long = read_csv(DAT / "gear_share_local_year.csv")
    gs_long["is_linha"] = gs_long["gear_cod"].str.upper().str.match(r"^LIN$")
    others_gs = gs_long[gs_long["local"] != AB]
    sys_linha = (others_gs.groupby("year")
                          .apply(lambda x: x.loc[x["is_linha"], "gear_share"].sum())
                          .reset_index(name="linha_share"))
    if not sys_linha.empty:
        ax_linha.plot(sys_linha["year"], sys_linha["linha_share"], color=CLR_SYS, lw=1.5,
                      ls="--", zorder=2, label="System mean (n=7)")

    # Shared legend on CPUE panel
    handles, labels = ax_cpue.get_legend_handles_labels()
    ax_cpue.legend(handles[:3], labels[:3], fontsize=9, frameon=False, loc="upper left")

    # Annotation: transition arrow
    ax_cpue.annotate("Platform-driven\ntransition\n(2008–2009)",
                     xy=(2009, ab.set_index("year").reindex([2009])[CPUE_COL].values[0] if 2009 in ab["year"].values else 0.3),
                     xytext=(2012, ax_cpue.get_ylim()[1] * 0.75),
                     arrowprops=dict(arrowstyle="->", color="#c09000", lw=1.4),
                     fontsize=8, color="#c09000")

    fig.suptitle("PA — Areia Branca: chronology of the Albacora/FAD transition",
                 fontsize=13, y=1.01)
    out = FIG / "ab_case_transitions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def fig_species_composition(species):
    """Stacked area chart of AB species shares over time + PdM comparison."""
    ab_sp  = species[species["local"] == AB].copy()
    pdm_sp = species[species["local"] == PDM].copy()

    # Top 8 species in AB by cumulative production
    top8_ab = (ab_sp.groupby("species")["sp_production_ton"].sum()
                    .sort_values(ascending=False).head(8).index.tolist())

    years_ab  = sorted(ab_sp["year"].unique())
    years_pdm = sorted(pdm_sp["year"].unique())

    def _wide(sp_df, loca, top_sp, years):
        wide = (sp_df[sp_df["species"].isin(top_sp)]
                .pivot_table(index="year", columns="species",
                             values="species_share", aggfunc="first")
                .reindex(years).fillna(0))
        return wide

    ab_wide  = _wide(ab_sp, AB, top8_ab, years_ab)
    pdm_wide = _wide(pdm_sp, PDM, top8_ab, years_pdm)

    cmap_sp = plt.cm.tab10(np.linspace(0, 1, len(top8_ab)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
    fig.suptitle("PA — Species composition over time: Areia Branca vs Porto do Mangue",
                 fontsize=12)

    for ax, wide, title, years in [
        (ax1, ab_wide,  "Areia Branca", years_ab),
        (ax2, pdm_wide, "Porto do Mangue", years_pdm),
    ]:
        bottom = np.zeros(len(years))
        for i, sp in enumerate(top8_ab):
            vals = wide[sp].values if sp in wide.columns else np.zeros(len(years))
            ax.bar(years, vals, bottom=bottom, color=cmap_sp[i], label=sp,
                   width=0.7, edgecolor="none", alpha=0.85)
            bottom += vals
        ax.axvspan(2007.5, 2009.5, alpha=0.12, color="#f0c040", zorder=0, label="Transition 2008–09")
        ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Species share (proportion)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        if ax is ax1:
            ax.legend(loc="upper left", fontsize=7, ncol=2, frameon=False)

    ax2.set_xlabel("Year", fontsize=9)
    plt.tight_layout()
    out = FIG / "ab_case_species.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def fig_pa3_comparison(ab, pdm, sys_mean):
    """PA.3 — Summary comparison: AB vs PdM vs system on key structural metrics."""
    # Period comparison: early (2001–2008) vs recent (2016–2024)
    periods = {"early": (2001, 2008), "recent": (2016, 2024)}

    def _period_means(df, col):
        out = {}
        for p, (y0, y1) in periods.items():
            sub = df[(df["year"] >= y0) & (df["year"] <= y1)]
            out[p] = sub[col].mean() if len(sub) > 0 else np.nan
        return out

    metrics = {
        "CPUE (t/fisherman)": CPUE_COL,
        "Albacora share":     "albacora_share",
        "Linha share":        "linha_share",
        "Shannon H′":         "shannon_h",
        "Species richness":   "richness",
        "Pielou J′":          "pielou_j",
    }

    # Compute system mean values (others)
    others = pd.read_csv(DAT / "productivity_local_year.csv")
    others[CPUE_COL] = others["production_ton"] / others["estimated_fishermen"]
    sp_long = pd.read_csv(DAT / "species_share_local_year.csv")
    gs_long = pd.read_csv(DAT / "gear_share_local_year.csv")
    gs_long["is_linha"] = gs_long["gear_cod"].str.upper().str.match(r"^LIN$")
    sys_linha = (gs_long[gs_long["local"] != AB]
                 .groupby("year")
                 .apply(lambda x: x.loc[x["is_linha"], "gear_share"].sum())
                 .reset_index(name="linha_share"))
    sys_alb = (sp_long[(sp_long["local"] != AB) & (sp_long["species"] == "Albacora")]
               .groupby("year")["species_share"].mean().reset_index(name="albacora_share"))
    sys_df = (others[others["local"] != AB]
              .groupby("year").agg(**{c: (c, "mean") for c in [CPUE_COL, "shannon_h", "richness", "pielou_j"]})
              .reset_index()
              .merge(sys_alb, on="year", how="left")
              .merge(sys_linha, on="year", how="left"))
    sys_df["albacora_share"] = sys_df["albacora_share"].fillna(0)
    sys_df["linha_share"]    = sys_df["linha_share"].fillna(0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("PA.3 — Areia Branca vs Porto do Mangue vs system mean\n"
                 "(early: 2001–2008  /  recent: 2016–2024)", fontsize=12)

    sources = [
        (ab,      "AB",  CLR_AB),
        (pdm,     "PdM", CLR_PDM),
        (sys_df,  "Sys", CLR_SYS),
    ]
    period_labels = list(periods.keys())

    for ax_i, (label, col) in enumerate(metrics.items()):
        ax = axes.flat[ax_i]
        x = np.arange(len(period_labels))
        width = 0.25
        for j, (df_src, name, color) in enumerate(sources):
            vals = [_period_means(df_src, col)[p] for p in period_labels]
            bars = ax.bar(x + j * width, vals, width, color=color, label=name,
                          edgecolor="k", linewidth=0.5, alpha=0.85)
        ax.set_xticks(x + width)
        ax.set_xticklabels(period_labels, fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8)
        if ax_i == 0:
            ax.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    out = FIG / "ab_case_pa3_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def print_summary(ab, pdm, sys_mean):
    HR = "─" * 70
    print("\n" + "=" * 70)
    print("PA — AREIA BRANCA FAD CASE STUDY")
    print("=" * 70)
    print(f"  AB years: {sorted(ab['year'].tolist())[:3]} … "
          f"{sorted(ab['year'].tolist())[-3:]}")
    print()

    print(HR)
    print("Albacora (yellowfin tuna) share:")
    pre  = ab[ab["year"] < 2009]["albacora_share"].mean()
    post = ab[ab["year"] >= 2009]["albacora_share"].mean()
    rec  = ab[ab["year"] >= 2016]["albacora_share"].mean()
    print(f"  AB  pre-2009 : {pre:.3f}   post-2009: {post:.3f}   2016+: {rec:.3f}")
    pdm_pre  = pdm[pdm["year"] < 2009]["albacora_share"].mean()
    pdm_post = pdm[pdm["year"] >= 2009]["albacora_share"].mean()
    print(f"  PdM pre-2009 : {pdm_pre:.3f}   post-2009: {pdm_post:.3f}")

    print()
    print("Linha (longline) share:")
    l_pre  = ab[ab["year"] < 2009]["linha_share"].mean()
    l_post = ab[ab["year"] >= 2009]["linha_share"].mean()
    print(f"  AB  pre-2009 : {l_pre:.3f}   post-2009: {l_post:.3f}")
    pl_pre  = pdm[pdm["year"] < 2009]["linha_share"].mean()
    pl_post = pdm[pdm["year"] >= 2009]["linha_share"].mean()
    print(f"  PdM pre-2009 : {pl_pre:.3f}   post-2009: {pl_post:.3f}")

    print()
    print("CPUE per fisherman (t/fisherman):")
    for name, df in [("AB", ab), ("PdM", pdm), ("System (n=7)", sys_mean)]:
        col = CPUE_COL if name != "System (n=7)" else "cpue_sys"
        vals = df[col].dropna()
        print(f"  {name:<16}  median={vals.median():.4f}  "
              f"range={vals.min():.4f}–{vals.max():.4f}")

    print()
    print("Shannon H′:")
    for name, df in [("AB", ab), ("PdM", pdm)]:
        h_pre  = df[df["year"] < 2009]["shannon_h"].mean()
        h_post = df[df["year"] >= 2009]["shannon_h"].mean()
        print(f"  {name:<16}  pre-2009={h_pre:.3f}   post-2009={h_post:.3f}")

    print()
    print(HR)
    print("PA.3 — Structural differences AB vs PdM:")
    print(f"  Platform dist:  AB={ab['platform_dist_km_mean'].iloc[0]:.1f} km  "
          f"PdM={pdm['platform_dist_km_mean'].iloc[0]:.1f} km")
    print(f"  Fleet (mean):   AB={ab['fleet_monitored'].mean():.1f}  "
          f"PdM={pdm['fleet_monitored'].mean():.1f}")
    print(f"  Fishermen (mean): AB={ab['estimated_fishermen'].mean():.1f}  "
          f"PdM={pdm['estimated_fishermen'].mean():.1f}")
    print("=" * 70)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cfg.setup_dirs()

    prod    = load_productivity()
    species = load_species()
    gear    = load_gear()

    ab, pdm, sys_mean = build_metrics(prod, gear, species)
    save_outputs(ab, pdm, sys_mean, gear, species)

    log.info("── PA transitions timeline ────────────────────────────")
    fig_transitions(ab, pdm, sys_mean)

    log.info("── PA species composition ────────────────────────────")
    fig_species_composition(species)

    log.info("── PA.3 comparison summary ───────────────────────────")
    fig_pa3_comparison(ab, pdm, sys_mean)

    print_summary(ab, pdm, sys_mean)
    log.info("13_ab_case.py complete.")


if __name__ == "__main__":
    main()
