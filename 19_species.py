#!/usr/bin/env python3
"""
19_species.py — P9: Species composition and temporal trends

P9a — Which species contribute most to between-locality dissimilarity
      (SIMPER by platform class, MPA status, combined exposure)?
      Primary: n=7 (excluding Areia Branca).
      AB reported separately to document its distinctiveness.

P9b — Species with significant temporal trends within localities
      (Mann-Kendall per species × locality; n=7 + AB reported separately).

Key note: In the n=8 SIMPER, Albacora dominates the inside-vs-outside MPA
contrast (13.7% contribution) entirely due to AB.  This script separates:
  - SIMPER inside_MPA (AB excluded) → reflects PdM + Macau vs outside
  - AB uniqueness: SIMPER AB vs rest-of-system

Outputs (data/processed/):
  species_p9_simper_n7.csv         SIMPER contrasts (n=7)
  species_p9_simper_ab.csv         AB vs rest of system
  species_p9_species_mk.csv        MK per species × locality (n=7)
  species_p9_species_mk_ab.csv     MK per species for AB only

Outputs (outputs/figures/):
  species_p9_simper.png            SIMPER bar chart: platform + MPA (n=7)
  species_p9_simper_ab.png         AB vs system contrast
  species_p9_mk_heatmap.png        Heatmap: MK τ per locality × species (n=7)
  species_p9_trajectories.png      Top species trajectories by exposure (n=7)
  species_p9_ab_trajectories.png   Top AB species trajectories vs system mean
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

sys.path.insert(0, ".")
import config_00 as cfg

warnings.filterwarnings("ignore")
cfg.setup_dirs()
logging.basicConfig(
    format=cfg.LOG_FORMAT, datefmt=cfg.LOG_DATEFMT, level=logging.INFO,
    handlers=[
        logging.FileHandler(cfg.LOGS / "19_species.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("19_species")

FIG = cfg.OUTPUTS / "figures"
DAT = cfg.DATA_PROCESSED

AB    = "AREIA BRANCA"
TOP_N = 12

CMAP = {
    "0-20 km × 0-10 km":   "#d62728",
    "20-50 km × inside":    "#2ca02c",
    "20-50 km × 10-25 km":  "#ff7f0e",
    "20-50 km × 25-50 km":  "#1f77b4",
}
CLR_AB  = "#d62728"
CLR_SYS = "#666666"


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
    exp_cols = ["local", "combined_exposure_class", "inside_any_mpa_any",
                "platform_exposure_class", "mpa_exposure_class"]
    exposure = prod[exp_cols].drop_duplicates("local")

    sp_long = pd.read_csv(DAT / "species_catch_matrix_local_year.csv")

    # Build wide relative-abundance matrix
    sp_wide = (sp_long.pivot_table(index=["local", "year"], columns="species",
                                   values="rel_abundance", aggfunc="first")
                      .fillna(0).reset_index())
    sp_wide = sp_wide.merge(exposure, on="local", how="left")
    sp_cols = [c for c in sp_wide.columns
               if c not in ("local", "year") + tuple(exp_cols[1:])]

    sp_all  = sp_wide.copy()
    sp_7    = sp_wide[sp_wide["local"] != AB].copy()
    locals_7 = sorted(sp_7["local"].unique())

    log.info("Species matrix (n=8): %d obs  |  %d species  |  %d localities",
             len(sp_all), len(sp_cols), sp_all["local"].nunique())
    log.info("Species matrix (n=7): %d obs  |  %d localities", len(sp_7), len(locals_7))
    return sp_all, sp_7, sp_cols, exposure, locals_7


# ── SIMPER ─────────────────────────────────────────────────────────────────────

def _simper(mat_a, mat_b, sp_names):
    """Contribution of each species to mean Bray-Curtis dissimilarity (SIMPER)."""
    n_a, n_b = len(mat_a), len(mat_b)
    contribs = np.zeros(len(sp_names))
    n_pairs  = 0
    for i in range(n_a):
        for j in range(n_b):
            total = (mat_a[i] + mat_b[j]).sum()
            if total > 0:
                contribs += np.abs(mat_a[i] - mat_b[j]) / total
                n_pairs  += 1
    if n_pairs > 0:
        contribs /= n_pairs
    total_bc = contribs.sum()
    pct = contribs / total_bc * 100 if total_bc > 0 else contribs
    idx = np.argsort(-contribs)
    cum = 0
    rows = []
    for rank, k in enumerate(idx):
        cum += pct[k]
        rows.append(dict(
            species=sp_names[k],
            mean_a=round(float(mat_a[:, k].mean()), 5),
            mean_b=round(float(mat_b[:, k].mean()), 5),
            contribution_pct=round(float(pct[k]), 3),
            cumulative_pct=round(cum, 3),
            rank=rank + 1,
        ))
    return pd.DataFrame(rows)


def p9_simper(sp_7, sp_all, sp_cols):
    rows_7  = []
    rows_ab = []

    # A — n=7: platform class contrast (0-20 km vs 20-50 km)
    mat_a = sp_7.loc[sp_7["platform_exposure_class"] == "0-20 km", sp_cols].values
    mat_b = sp_7.loc[sp_7["platform_exposure_class"] == "20-50 km", sp_cols].values
    if len(mat_a) >= 2 and len(mat_b) >= 2:
        sim = _simper(mat_a, mat_b, sp_cols)
        sim["contrast_type"] = "platform"; sim["group_a"] = "0-20 km"; sim["group_b"] = "20-50 km"
        rows_7.append(sim)
        log.info("SIMPER platform 0-20 km vs 20-50 km (n=7): top5 = %s",
                 ", ".join(sim.head(5)["species"].tolist()))

    # B — n=7: MPA inside vs outside (AB excluded — PdM + Macau vs others)
    mat_in  = sp_7.loc[sp_7["inside_any_mpa_any"] == True,  sp_cols].values
    mat_out = sp_7.loc[sp_7["inside_any_mpa_any"] == False, sp_cols].values
    if len(mat_in) >= 2 and len(mat_out) >= 2:
        sim_mpa = _simper(mat_in, mat_out, sp_cols)
        sim_mpa["contrast_type"] = "mpa_status"; sim_mpa["group_a"] = "inside"; sim_mpa["group_b"] = "outside"
        rows_7.append(sim_mpa)
        log.info("SIMPER MPA inside vs outside (n=7, AB excl.): top5 = %s",
                 ", ".join(sim_mpa.head(5)["species"].tolist()))

    # C — n=7: all pairwise combined exposure classes
    classes = sorted(sp_7["combined_exposure_class"].dropna().unique())
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            ca, cb = classes[i], classes[j]
            mat_a = sp_7.loc[sp_7["combined_exposure_class"] == ca, sp_cols].values
            mat_b = sp_7.loc[sp_7["combined_exposure_class"] == cb, sp_cols].values
            if len(mat_a) < 2 or len(mat_b) < 2:
                continue
            sim_c = _simper(mat_a, mat_b, sp_cols)
            sim_c["contrast_type"] = "combined"; sim_c["group_a"] = ca; sim_c["group_b"] = cb
            rows_7.append(sim_c)

    simper_7 = pd.concat(rows_7, ignore_index=True)
    simper_7.to_csv(DAT / "species_p9_simper_n7.csv", index=False)
    log.info("Saved species_p9_simper_n7.csv (%d rows)", len(simper_7))

    # D — AB uniqueness: AB vs rest of system
    mat_ab  = sp_all.loc[sp_all["local"] == AB, sp_cols].values
    mat_sys = sp_all.loc[sp_all["local"] != AB, sp_cols].values
    if len(mat_ab) >= 2 and len(mat_sys) >= 2:
        sim_ab = _simper(mat_ab, mat_sys, sp_cols)
        sim_ab["contrast_type"] = "ab_vs_system"; sim_ab["group_a"] = "AB"; sim_ab["group_b"] = "system"
        rows_ab.append(sim_ab)
        log.info("SIMPER AB vs system: top5 = %s",
                 ", ".join(sim_ab.head(5)["species"].tolist()))

    simper_ab = pd.concat(rows_ab, ignore_index=True) if rows_ab else pd.DataFrame()
    if not simper_ab.empty:
        simper_ab.to_csv(DAT / "species_p9_simper_ab.csv", index=False)
        log.info("Saved species_p9_simper_ab.csv")

    return simper_7, simper_ab


def fig_simper_main(simper_7):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("P9a — SIMPER: species contributing to between-group dissimilarity (n=7)",
                 fontsize=12)

    for ax, contrast, title in [
        (ax1, "platform",   "Platform: 0-20 km vs 20-50 km"),
        (ax2, "mpa_status", "MPA: inside (PdM+Macau) vs outside  [AB excluded]"),
    ]:
        sub = simper_7[(simper_7["contrast_type"] == contrast) &
                       (simper_7["rank"] <= TOP_N)].sort_values("rank", ascending=False)
        if sub.empty:
            ax.set_visible(False); continue
        ax.barh(sub["species"], sub["contribution_pct"],
                color="#5899c8", edgecolor="k", linewidth=0.4)
        for _, row in sub.iterrows():
            ax.text(row["contribution_pct"] + 0.1, row["species"],
                    f"{row['contribution_pct']:.1f}%", va="center", ha="left", fontsize=7)
        ax.set_xlabel("Contribution to dissimilarity (%)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    out = FIG / "species_p9_simper.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def fig_simper_ab(simper_ab):
    if simper_ab.empty:
        return
    sub = simper_ab[simper_ab["rank"] <= TOP_N].sort_values("rank", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(sub["species"], sub["contribution_pct"],
            color=CLR_AB, edgecolor="k", linewidth=0.4, alpha=0.8)
    for _, row in sub.iterrows():
        ax.text(row["contribution_pct"] + 0.1, row["species"],
                f"{row['contribution_pct']:.1f}%  (AB={row['mean_a']:.3f}  sys={row['mean_b']:.3f})",
                va="center", ha="left", fontsize=7)
    ax.set_xlabel("Contribution to dissimilarity (%)", fontsize=9)
    ax.set_title(f"PA/P9a — Areia Branca vs rest of system\n(top {TOP_N} species)",
                 fontsize=11)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    out = FIG / "species_p9_simper_ab.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


# ── P9b — temporal trends in species dominance ────────────────────────────────

def p9_species_mk(sp_7, sp_all, sp_cols, locals_7):
    # Top species in n=7 by mean relative abundance
    sp_means_7 = sp_7[sp_cols].mean().sort_values(ascending=False)
    top_sp_7   = sp_means_7.head(TOP_N).index.tolist()

    # n=7 MK
    mk_rows = []
    for loc in locals_7:
        sub = sp_7[sp_7["local"] == loc].sort_values("year")
        cc  = sub["combined_exposure_class"].iloc[0]
        for sp in top_sp_7:
            vals = sub[sp].values if sp in sub.columns else None
            if vals is None or np.sum(vals > 0) < 4:
                continue
            tau, p = kendalltau(sub["year"].values, vals)
            slope  = _sens_slope(vals, sub["year"].values)
            mk_rows.append(dict(
                local=loc, species=sp,
                tau=round(tau, 3), p=round(p, 4),
                sens_slope=round(slope, 7), n=len(vals),
                combined_class=cc,
                mean_rel_abund=round(float(vals.mean()), 5),
                sig=_sig(p),
            ))
    mk_df = pd.DataFrame(mk_rows)
    mk_df.to_csv(DAT / "species_p9_species_mk.csv", index=False)
    log.info("Species MK (n=7): %d local×species combinations", len(mk_df))

    # AB separately
    ab_mk_rows = []
    ab_sub = sp_all[sp_all["local"] == AB].sort_values("year")
    sp_means_ab = ab_sub[sp_cols].mean().sort_values(ascending=False)
    top_sp_ab   = sp_means_ab.head(TOP_N).index.tolist()
    for sp in top_sp_ab:
        vals = ab_sub[sp].values if sp in ab_sub.columns else None
        if vals is None or np.sum(vals > 0) < 4:
            continue
        tau, p = kendalltau(ab_sub["year"].values, vals)
        slope  = _sens_slope(vals, ab_sub["year"].values)
        ab_mk_rows.append(dict(
            local=AB, species=sp,
            tau=round(tau, 3), p=round(p, 4),
            sens_slope=round(slope, 7), n=len(vals),
            mean_rel_abund=round(float(vals.mean()), 5),
            sig=_sig(p),
        ))
    ab_mk_df = pd.DataFrame(ab_mk_rows)
    ab_mk_df.to_csv(DAT / "species_p9_species_mk_ab.csv", index=False)
    log.info("Species MK (AB only): %d species", len(ab_mk_df))

    return mk_df, ab_mk_df, top_sp_7, top_sp_ab


def fig_mk_heatmap(mk_df):
    pivot = mk_df.pivot_table(index="local", columns="species", values="tau")
    order_sp  = pivot.abs().mean().sort_values(ascending=False).index.tolist()
    order_loc = (mk_df.drop_duplicates("local")
                      .sort_values(["combined_class", "local"])["local"].tolist())
    pivot = pivot.reindex(index=order_loc, columns=order_sp)

    fig, ax = plt.subplots(figsize=(max(8, len(order_sp) * 0.65), 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(order_sp)))
    ax.set_xticklabels(order_sp, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(order_loc)))
    ax.set_yticklabels(order_loc, fontsize=8)

    # Significance annotations
    p_pivot = mk_df.pivot_table(index="local", columns="species", values="p")
    p_pivot = p_pivot.reindex(index=order_loc, columns=order_sp)
    for r in range(len(order_loc)):
        for c in range(len(order_sp)):
            pv = p_pivot.iloc[r, c] if not pd.isna(p_pivot.iloc[r, c]) else 1
            sym = "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            if sym:
                ax.text(c, r, sym, ha="center", va="center", fontsize=6, color="k")

    plt.colorbar(im, ax=ax, label="Kendall τ", shrink=0.7)
    ax.set_title(f"P9b — Temporal trends in species dominance (n=7; top {TOP_N} species)",
                 fontsize=10)
    plt.tight_layout()
    out = FIG / "species_p9_mk_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def fig_trajectories_n7(sp_7, top_sp_7):
    top6  = top_sp_7[:6]
    classes = sorted(sp_7["combined_exposure_class"].dropna().unique())

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle("P9b — Top species trajectories by exposure context (n=7)", fontsize=12)

    for ax, sp in zip(axes.flat, top6):
        if sp not in sp_7.columns:
            ax.set_visible(False); continue
        for cc in classes:
            sub = (sp_7[sp_7["combined_exposure_class"] == cc]
                   .groupby("year")[sp].mean())
            ax.plot(sub.index, sub.values,
                    color=CMAP.get(cc, "gray"), lw=1.5, marker="o", markersize=3, alpha=0.8)
        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel("Mean rel. abundance", fontsize=8)
        ax.set_title(sp, fontsize=9)
        ax.tick_params(labelsize=7)

    legend_els = [Line2D([0], [0], color=c, lw=2, label=k) for k, c in CMAP.items()]
    axes[0, 2].legend(handles=legend_els, fontsize=7, loc="upper right", frameon=False)

    plt.tight_layout()
    out = FIG / "species_p9_trajectories.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def fig_ab_trajectories(sp_all, top_sp_ab):
    top4 = top_sp_ab[:4]
    ab_sub  = sp_all[sp_all["local"] == AB].sort_values("year")
    sys_sub = sp_all[sp_all["local"] != AB]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("PA/P9b — AB top species vs system mean (shaded = system ±1 SD)", fontsize=12)

    for ax, sp in zip(axes, top4):
        if sp not in ab_sub.columns:
            ax.set_visible(False); continue

        # System mean + SD
        sys_yr = sys_sub.groupby("year")[sp].agg(["mean", "std"]).reset_index()
        ax.fill_between(sys_yr["year"],
                        sys_yr["mean"] - sys_yr["std"],
                        sys_yr["mean"] + sys_yr["std"],
                        color=CLR_SYS, alpha=0.15)
        ax.plot(sys_yr["year"], sys_yr["mean"], color=CLR_SYS, lw=1.5,
                ls="--", label="System mean (n=7)")

        # AB
        ax.plot(ab_sub["year"], ab_sub[sp], color=CLR_AB, lw=2, marker="o", ms=4,
                label="Areia Branca")

        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel("Relative abundance", fontsize=8)
        ax.set_title(sp, fontsize=9)
        ax.tick_params(labelsize=7)
        if sp == top4[0]:
            ax.legend(fontsize=7, frameon=False)

    plt.tight_layout()
    out = FIG / "species_p9_ab_trajectories.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def print_summary(simper_7, simper_ab, mk_df, ab_mk_df):
    HR = "─" * 68
    print("\n" + "=" * 68)
    print("P9 — SPECIES COMPOSITION AND TEMPORAL TRENDS")
    print("=" * 68)
    print()
    print("P9a — SIMPER top 5 (n=7):")
    for contrast, label in [("platform", "Platform 0-20 vs 20-50 km"),
                             ("mpa_status", "MPA inside vs outside (AB excl.)")]:
        sub = simper_7[simper_7["contrast_type"] == contrast].head(5)
        print(f"\n  {label}:")
        for _, r in sub.iterrows():
            print(f"    {r['rank']:2d}. {r['species']:<18}  {r['contribution_pct']:>5.1f}%  "
                  f"(cumul. {r['cumulative_pct']:>5.1f}%)")

    if not simper_ab.empty:
        print(f"\n  AB vs system:")
        for _, r in simper_ab.head(5).iterrows():
            print(f"    {r['rank']:2d}. {r['species']:<18}  {r['contribution_pct']:>5.1f}%  "
                  f"AB={r['mean_a']:.4f}  sys={r['mean_b']:.4f}")

    print()
    print(f"P9b — MK summary (n=7):  significant p<0.05: "
          f"{(mk_df['p'] < 0.05).sum()} / {len(mk_df)} local×species combinations")
    print(f"  Most consistently increasing:")
    top_inc = mk_df.groupby("species")["tau"].mean().sort_values(ascending=False).head(5)
    for sp, tau in top_inc.items():
        print(f"    {sp:<20}  mean τ={tau:+.3f}")
    print()
    print(f"  AB top trends:")
    for _, r in ab_mk_df.sort_values("tau", ascending=False).head(5).iterrows():
        print(f"    {r['species']:<20}  τ={r['tau']:+.3f}  {r['sig']}")
    print("=" * 68)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    sp_all, sp_7, sp_cols, exposure, locals_7 = load_data()

    log.info("── P9a: SIMPER (n=7 primary + AB separate) ───────────")
    simper_7, simper_ab = p9_simper(sp_7, sp_all, sp_cols)
    fig_simper_main(simper_7)
    fig_simper_ab(simper_ab)

    log.info("── P9b: Temporal trends in species dominance ──────────")
    mk_df, ab_mk_df, top_sp_7, top_sp_ab = p9_species_mk(sp_7, sp_all, sp_cols, locals_7)
    fig_mk_heatmap(mk_df)
    fig_trajectories_n7(sp_7, top_sp_7)
    fig_ab_trajectories(sp_all, top_sp_ab)

    print_summary(simper_7, simper_ab, mk_df, ab_mk_df)
    log.info("19_species.py complete.")


if __name__ == "__main__":
    main()
