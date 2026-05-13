"""
12_cpue_std.py
==============
CPUE standardization using fleet size and fishermen count as covariates.

The raw index cpue_ton_per_trip = production_ton / assisted_trips already
controls for the number of trips.  Standardization goes further: it removes
the influence of changes in fleet size (fleet_monitored) and total fishermen
(estimated_fishermen) on the CPUE signal, to produce an effort-corrected
abundance indicator.

Why a GAMM is properly identified here
---------------------------------------
fleet_monitored and estimated_fishermen BOTH vary within each locality
across years (within-local CV ≈ 0.22–0.38).  This makes them genuine
time-varying covariates.  A local random intercept is therefore correctly
identified alongside these predictors — the collinearity problem that
affected platform/MPA exposure classes (which were LOCAL-LEVEL constants)
does not apply here.

CPUE formulations compared
---------------------------
  cpue_per_trip       = production_ton / assisted_trips      (current)
  cpue_per_vessel     = production_ton / fleet_monitored
  cpue_per_fisherman  = production_ton / estimated_fishermen

Models
------
  Model G — GammaGAM (pygam)
      CPUE_per_trip ~ s(year_c) + l(log_fleet_c) + l(log_fishermen_c) + f(local)
      Gamma distribution with log link.
      local as fixed factor (7 dummies); fleet and fishermen as linear-log terms.
      All predictors vary within local → f(local) is properly identified.

  Model M — LMM log-normal (statsmodels MixedLM)
      log(CPUE_per_trip) ~ year_c + log_fleet_c + log_fishermen_c + (1 | local)
      Feasible because all predictors vary within local.
      Random intercept for local absorbs residual between-local CPUE differences.

Standardized CPUE index
-----------------------
  For each year: predicted CPUE with fleet and fishermen fixed at their
  grand geometric means (= log-centred covariates set to 0), averaged
  over all 8 locals.  Reported as absolute values and as an index
  normalised to the period mean = 1.

Inputs:  data/processed/productivity_local_year.csv

Outputs (data/processed/):
  cpue_std_data.csv        Dataset with all CPUE formulations
  cpue_std_gam_summary.csv GammaGAM model statistics
  cpue_std_gam_partial.csv Partial effects: s(year), l(log_fleet), l(log_fishermen)
  cpue_std_index.csv       Standardized index by year (GAM + LMM)
  cpue_std_lmm.csv         LMM fixed-effects table

Outputs (outputs/figures/):
  cpue_std_raw_comparison.png     Three raw CPUE metrics over time by local
  cpue_std_covariates_time.png    Fleet and fishermen over time by local
  cpue_std_partial_effects.png    GAM partial effects
  cpue_std_index.png              Standardized vs raw CPUE index (population-level)
  cpue_std_by_local.png           Per-local raw vs standardized CPUE

Run:
  python 12_cpue_std.py
"""

import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import GammaGAM, f, l, s
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_00 as cfg  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format=cfg.LOG_FORMAT,
    datefmt=cfg.LOG_DATEFMT,
    handlers=[
        logging.FileHandler(cfg.LOGS / "12_cpue_std.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("12_cpue_std")
warnings.filterwarnings("ignore")

FIG_DIR = cfg.OUTPUTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─── Data preparation ─────────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    raw = pd.read_csv(cfg.DATA_PROCESSED / "productivity_local_year.csv")
    df = raw[[
        "local", "year",
        "production_ton", "assisted_trips",
        "fleet_monitored", "estimated_fishermen",
        "cpue_ton_per_trip",
        "platform_exposure_class", "mpa_exposure_class",
    ]].dropna(subset=["cpue_ton_per_trip", "fleet_monitored", "estimated_fishermen"])

    # Alternative CPUE formulations
    df["cpue_per_vessel"]    = df["production_ton"] / df["fleet_monitored"]
    df["cpue_per_fisherman"] = df["production_ton"] / df["estimated_fishermen"]

    # Log-centred effort covariates (centred at grand geometric mean → model intercept
    # represents CPUE at mean fleet size and mean fishermen count)
    df["log_fleet"]       = np.log(df["fleet_monitored"])
    df["log_fishermen"]   = np.log(df["estimated_fishermen"])
    df["log_fleet_c"]     = df["log_fleet"]     - df["log_fleet"].mean()
    df["log_fishermen_c"] = df["log_fishermen"] - df["log_fishermen"].mean()

    # Centred year
    df["year_c"] = df["year"] - df["year"].median()
    df["log_cpue"] = np.log(df["cpue_ton_per_trip"])

    df = df.sort_values(["local", "year"]).reset_index(drop=True)

    log.info("Dataset: %d obs  |  %d locals  |  %d–%d",
             len(df), df["local"].nunique(), df["year"].min(), df["year"].max())
    log.info("cpue_per_trip:      median=%.4f  range=%.4f–%.4f",
             df["cpue_ton_per_trip"].median(),
             df["cpue_ton_per_trip"].min(), df["cpue_ton_per_trip"].max())
    log.info("cpue_per_vessel:    median=%.4f  range=%.4f–%.4f",
             df["cpue_per_vessel"].median(),
             df["cpue_per_vessel"].min(), df["cpue_per_vessel"].max())
    log.info("cpue_per_fisherman: median=%.4f  range=%.4f–%.4f",
             df["cpue_per_fisherman"].median(),
             df["cpue_per_fisherman"].min(), df["cpue_per_fisherman"].max())

    # Within-local correlation of effort covariates with log(CPUE)
    from scipy.stats import spearmanr
    for v in ["log_fleet_c", "log_fishermen_c"]:
        tmp = df.copy()
        tmp["v_dm"] = tmp[v] - tmp.groupby("local")[v].transform("mean")
        tmp["c_dm"] = tmp["log_cpue"] - tmp.groupby("local")["log_cpue"].transform("mean")
        r, p = spearmanr(tmp["v_dm"], tmp["c_dm"])
        log.info("Within-local Spearman  %s vs log(CPUE): ρ=%.3f  p=%.4f", v, r, p)

    return df


# ─── Model G: GammaGAM ────────────────────────────────────────────────────────

def fit_gamma_gam(df: pd.DataFrame):
    """
    GammaGAM: CPUE ~ s(year_c) + l(log_fleet_c) + l(log_fishermen_c) + f(local)
    Feature matrix columns:
      0: year_c           → s(0, n_splines=8)
      1: log_fleet_c      → l(1)
      2: log_fishermen_c  → l(2)
      3: local_enc        → f(3)
    """
    local_cats = sorted(df["local"].unique())
    local_enc  = {c: i for i, c in enumerate(local_cats)}

    X = np.column_stack([
        df["year_c"].values,
        df["log_fleet_c"].values,
        df["log_fishermen_c"].values,
        df["local"].map(local_enc).values,
    ])
    y = df["cpue_ton_per_trip"].values

    gam = GammaGAM(
        s(0, n_splines=8) +
        l(1) +
        l(2) +
        f(3),
        fit_intercept=True,
    )
    gam.gridsearch(X, y, progress=False)

    st = gam.statistics_
    log.info(
        "GammaGAM  deviance_explained=%.3f  AIC=%.1f  GCV=%.4f  edof=%.2f",
        st["pseudo_r2"]["explained_deviance"], st["AIC"], st["GCV"], st["edof"],
    )
    return gam, X, local_cats, local_enc


def gam_partial_effects(gam, X: np.ndarray) -> pd.DataFrame:
    """Partial effects for s(year_c), l(log_fleet_c), l(log_fishermen_c)."""
    term_info = [
        (0, "year_c",         "smooth"),
        (1, "log_fleet_c",    "linear"),
        (2, "log_fishermen_c","linear"),
    ]
    records = []
    for term_idx, name, kind in term_info:
        XX   = gam.generate_X_grid(term=term_idx, n=200)
        pdep, confint = gam.partial_dependence(term=term_idx, X=XX, width=0.95)
        for x, pd_, lo, hi in zip(XX[:, term_idx], pdep, confint[:, 0], confint[:, 1]):
            records.append(dict(
                term=name, kind=kind,
                x_value=round(float(x), 6),
                partial_effect=round(float(pd_), 6),
                ci_lower=round(float(lo), 6),
                ci_upper=round(float(hi), 6),
            ))
    return pd.DataFrame(records)


def build_std_index_gam(gam, df: pd.DataFrame,
                        local_enc: dict) -> pd.DataFrame:
    """
    Standardized CPUE index from GAM:
    For each observed year, predict CPUE with log_fleet_c=0, log_fishermen_c=0
    (fleet and fishermen at their grand geometric means), averaged over all locals.
    """
    year_median = df["year"].median()
    years       = sorted(df["year"].unique())
    n_locals    = len(local_enc)

    rows = []
    for yr in years:
        yc = yr - year_median
        preds = []
        for li in range(n_locals):
            x_row = np.array([[yc, 0.0, 0.0, li]])
            preds.append(float(gam.predict(x_row)[0]))
        # Also get prediction without local (average over f(local) effect is
        # approximated by predicting each local and averaging)
        rows.append({
            "year":       yr,
            "cpue_std_gam": round(np.mean(preds), 6),
            "cpue_std_gam_min": round(np.min(preds), 6),
            "cpue_std_gam_max": round(np.max(preds), 6),
        })

    idx = pd.DataFrame(rows)
    # Also compute raw annual mean CPUE (observed, averaged over locals with
    # equal weight so it matches the GAM averaging)
    raw = (
        df.groupby("year")["cpue_ton_per_trip"]
          .mean().reset_index()
          .rename(columns={"cpue_ton_per_trip": "cpue_raw_mean"})
    )
    idx = idx.merge(raw, on="year", how="left")

    # Normalise both to their period means (index mean = 1)
    idx["cpue_std_gam_idx"] = idx["cpue_std_gam"]   / idx["cpue_std_gam"].mean()
    idx["cpue_raw_idx"]     = idx["cpue_raw_mean"]   / idx["cpue_raw_mean"].mean()
    return idx


# ─── Model M: LMM log-normal ─────────────────────────────────────────────────

def fit_lmm(df: pd.DataFrame) -> tuple:
    """
    LMM: log(CPUE) ~ year_c + log_fleet_c + log_fishermen_c + (1 | local)
    All predictors vary within local → random intercept is properly identified.
    """
    formula = "log_cpue ~ year_c + log_fleet_c + log_fishermen_c"
    model   = smf.mixedlm(formula, data=df, groups=df["local"])
    result  = model.fit(reml=True)

    re_var    = float(result.cov_re.iloc[0, 0])
    resid_var = float(result.scale)
    icc       = re_var / (re_var + resid_var) if (re_var + resid_var) > 0 else np.nan

    log.info("LMM  RE_var=%.4f  resid_var=%.4f  ICC=%.3f", re_var, resid_var, icc)
    log.info(result.summary())

    def _sig(p):
        if np.isnan(p): return "n/a"
        return "***" if p < 0.001 else ("**" if p < 0.01 else
               ("*" if p < 0.05 else ("." if p < 0.10 else "ns")))

    fe = result.fe_params
    ci = result.conf_int()
    pv = result.pvalues
    se = result.bse

    table = pd.DataFrame({
        "term":     fe.index,
        "coef":     fe.values.round(5),
        "se":       se[fe.index].values.round(5),
        "ci_lower": ci.loc[fe.index, 0].values.round(5),
        "ci_upper": ci.loc[fe.index, 1].values.round(5),
        "pvalue":   pv[fe.index].values.round(4),
    })
    table["sig"] = table["pvalue"].apply(_sig)
    table.attrs.update({
        "RE_var":    round(re_var, 6),
        "resid_var": round(resid_var, 6),
        "ICC":       round(icc, 4),
        "AIC":       round(float(result.aic), 2) if not np.isnan(result.aic) else np.nan,
        "loglik":    round(float(result.llf), 3),
    })
    return result, table


def build_std_index_lmm(result, df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardized CPUE from LMM: population-level prediction with fleet and
    fishermen at their means (log_fleet_c = log_fishermen_c = 0).
    Returns one value per year (no random effect, i.e., 'average local').
    """
    fe = result.fe_params
    year_median = df["year"].median()
    years = sorted(df["year"].unique())

    b0           = float(fe.get("Intercept", 0))
    b_year       = float(fe.get("year_c", 0))
    b_fleet      = float(fe.get("log_fleet_c", 0))
    b_fishermen  = float(fe.get("log_fishermen_c", 0))

    rows = []
    for yr in years:
        yc = yr - year_median
        log_cpue_std = b0 + b_year * yc  # fleet & fishermen at means → 0
        rows.append({"year": yr, "cpue_std_lmm": round(np.exp(log_cpue_std), 6)})

    idx = pd.DataFrame(rows)
    idx["cpue_std_lmm_idx"] = idx["cpue_std_lmm"] / idx["cpue_std_lmm"].mean()
    return idx


# ─── Per-local standardized CPUE ─────────────────────────────────────────────

def build_local_std_gam(gam, df: pd.DataFrame,
                        local_enc: dict) -> pd.DataFrame:
    """
    For each local × year: predicted CPUE with fleet and fishermen at their
    grand means (log-centred to 0), keeping the local fixed effect.
    Compared against the observed CPUE to show the correction.
    """
    year_median = df["year"].median()
    rows = []
    for _, row in df.iterrows():
        yc = row["year_c"]
        li = local_enc[row["local"]]
        x_row = np.array([[yc, 0.0, 0.0, li]])
        pred = float(gam.predict(x_row)[0])
        rows.append({
            "local":        row["local"],
            "year":         row["year"],
            "cpue_raw":     row["cpue_ton_per_trip"],
            "cpue_std_gam": round(pred, 6),
            "correction":   round(pred / row["cpue_ton_per_trip"], 4),
        })
    return pd.DataFrame(rows)


# ─── Figures ──────────────────────────────────────────────────────────────────

_PALETTE  = list(plt.cm.Set1.colors[:8])
_LOC_COLS: dict = {}


def _lc(local: str, all_locals: list) -> tuple:
    if local not in _LOC_COLS:
        _LOC_COLS[local] = _PALETTE[all_locals.index(local) % len(_PALETTE)]
    return _LOC_COLS[local]


def plot_raw_comparison(df: pd.DataFrame) -> None:
    """Three raw CPUE formulations over time, one panel per metric, lines by local."""
    all_locals = sorted(df["local"].unique())
    metrics = [
        ("cpue_ton_per_trip",  "CPUE per trip (ton/trip)"),
        ("cpue_per_vessel",    "CPUE per vessel (ton/vessel)"),
        ("cpue_per_fisherman", "CPUE per fisherman (ton/fisherman)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    fig.suptitle("Three raw CPUE formulations by local  (2001–2024)", fontsize=10)

    for ax, (col, ylabel) in zip(axes, metrics):
        for loc in all_locals:
            sub = df[df["local"] == loc].sort_values("year")
            ax.plot(sub["year"], sub[col], marker="o", ms=3, lw=1.2,
                    color=_lc(loc, all_locals), label=loc, alpha=0.85)
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    out = FIG_DIR / "cpue_std_raw_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def plot_covariates_time(df: pd.DataFrame) -> None:
    """Fleet and fishermen over time by local — shows what we are standardising for."""
    all_locals = sorted(df["local"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Effort covariates over time by local", fontsize=10)

    for ax, (col, ylabel) in zip(axes, [
        ("fleet_monitored",    "Fleet monitored (vessels)"),
        ("estimated_fishermen","Estimated fishermen"),
    ]):
        for loc in all_locals:
            sub = df[df["local"] == loc].sort_values("year")
            ax.plot(sub["year"], sub[col], marker="o", ms=3, lw=1.2,
                    color=_lc(loc, all_locals), label=loc, alpha=0.85)
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    out = FIG_DIR / "cpue_std_covariates_time.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def plot_partial_effects(partial: pd.DataFrame, gam_stats: dict) -> None:
    """GAM partial effects: s(year_c), l(log_fleet_c), l(log_fishermen_c)."""
    terms = [
        ("year_c",          "Year (centred)"),
        ("log_fleet_c",     "log(fleet_monitored) [centred]"),
        ("log_fishermen_c", "log(estimated_fishermen) [centred]"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"GammaGAM partial effects on CPUE (log scale)\n"
        f"deviance explained={gam_stats['deviance_explained']:.3f}  "
        f"AIC={gam_stats['AIC']:.1f}  GCV={gam_stats['GCV']:.4f}  "
        f"edof={gam_stats['edof']:.1f}",
        fontsize=10,
    )
    colors = ["#1565C0", "#E53935", "#2E7D32"]
    for ax, (term, xlabel), color in zip(axes, terms, colors):
        sub = partial[partial["term"] == term]
        x, y = sub["x_value"].values, sub["partial_effect"].values
        lo, hi = sub["ci_lower"].values, sub["ci_upper"].values
        ax.fill_between(x, lo, hi, alpha=0.18, color=color)
        ax.plot(x, y, color=color, lw=2)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Partial effect (log CPUE)", fontsize=9)
        ax.tick_params(labelsize=8)
        # Mark x=0 (covariate at its geometric mean)
        if term != "year_c":
            ax.axvline(0, color="orange", lw=0.8, ls=":", alpha=0.7,
                       label="grand mean")
            ax.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    out = FIG_DIR / "cpue_std_partial_effects.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def plot_std_index(idx: pd.DataFrame) -> None:
    """
    Standardized (effort-corrected) vs raw CPUE index, both normalised to mean=1.
    Also shows the ratio std/raw to highlight the correction magnitude.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Standardized vs raw CPUE index  (mean = 1)", fontsize=10)

    ax = axes[0]
    ax.plot(idx["year"], idx["cpue_raw_idx"],     color="#78909C", lw=2,
            label="Raw CPUE (mean-normalised)", marker="o", ms=3)
    ax.plot(idx["year"], idx["cpue_std_gam_idx"], color="#1565C0", lw=2,
            label="Standardised — GAM", marker="s", ms=3)
    if "cpue_std_lmm_idx" in idx.columns:
        ax.plot(idx["year"], idx["cpue_std_lmm_idx"], color="#E53935", lw=1.5,
                ls="--", label="Standardised — LMM", marker="^", ms=3)
    ax.axhline(1, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("CPUE index (mean = 1)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)

    # Ratio: standardised / raw
    ax2 = axes[1]
    ratio_gam = idx["cpue_std_gam_idx"] / idx["cpue_raw_idx"]
    ax2.bar(idx["year"], ratio_gam - 1, color=np.where(ratio_gam >= 1, "#1565C0", "#E53935"),
            alpha=0.6, label="(std_GAM / raw) − 1")
    ax2.axhline(0, color="grey", lw=0.8, ls="--")
    ax2.set_xlabel("Year", fontsize=9)
    ax2.set_ylabel("Relative correction (std/raw − 1)", fontsize=9)
    ax2.set_title("Correction magnitude: how much does standardisation shift the index?",
                  fontsize=9)
    ax2.tick_params(labelsize=8)

    plt.tight_layout()
    out = FIG_DIR / "cpue_std_index.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def plot_by_local(local_std: pd.DataFrame) -> None:
    """Per-local: raw vs standardised CPUE over time (8 sub-panels)."""
    all_locals = sorted(local_std["local"].unique())
    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6), sharex=True)
    fig.suptitle("Per-local: raw vs standardised CPUE  (effort at grand mean)", fontsize=10)

    for ax, loc in zip(axes.flat, all_locals):
        sub = local_std[local_std["local"] == loc].sort_values("year")
        color = _lc(loc, all_locals)
        ax.plot(sub["year"], sub["cpue_raw"],     color="grey",  lw=1.2,
                marker="o", ms=2.5, label="raw")
        ax.plot(sub["year"], sub["cpue_std_gam"], color=color,   lw=1.5,
                marker="s", ms=2.5, label="std GAM")
        ax.set_title(loc, fontsize=8, color=color)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Year", fontsize=7)
        ax.set_ylabel("CPUE (ton/trip)", fontsize=7)

    # Hide unused axes
    for ax in axes.flat[len(all_locals):]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([0], [0], color="grey",  lw=1.5, label="raw"),
        plt.Line2D([0], [0], color="#333",  lw=1.5, ls="--", label="std GAM"),
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    plt.tight_layout()
    out = FIG_DIR / "cpue_std_by_local.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


# ─── Three-way comparison index ──────────────────────────────────────────────

def build_comparison_index(df: pd.DataFrame, std_index: pd.DataFrame) -> pd.DataFrame:
    """
    Annual index (mean = 1) for three CPUE measures:
      - cpue_trip_raw : raw cpue_ton_per_trip averaged over available locals
      - cpue_trip_std : effort-corrected cpue_ton_per_trip (GAM, fleet/fishermen at means)
      - cpue_fisherman: raw cpue_per_fisherman averaged over available locals
    Equal-weight average across locals present in each year.
    """
    # Per-year means with equal weight per locality
    trip_raw = (df.groupby("year")["cpue_ton_per_trip"]
                  .mean().reset_index()
                  .rename(columns={"cpue_ton_per_trip": "cpue_trip_raw"}))
    fish_raw = (df.groupby("year")["cpue_per_fisherman"]
                  .mean().reset_index()
                  .rename(columns={"cpue_per_fisherman": "cpue_fisherman"}))

    comp = trip_raw.merge(fish_raw, on="year")
    comp = comp.merge(std_index[["year", "cpue_std_gam"]].rename(
        columns={"cpue_std_gam": "cpue_trip_std"}), on="year", how="left")

    # Normalise each to its period mean = 1
    for col in ["cpue_trip_raw", "cpue_trip_std", "cpue_fisherman"]:
        comp[f"{col}_idx"] = comp[col] / comp[col].mean()

    return comp


def plot_three_way_comparison(comp: pd.DataFrame) -> None:
    """
    Four-panel figure comparing the three CPUE measures.
    Panel A: time series (normalised, mean=1)
    Panel B: corrected vs raw trip CPUE scatter
    Panel C: corrected trip CPUE vs per-fisherman scatter
    Panel D: Mann-Kendall tau + Sen slope for each measure
    """
    from scipy.stats import pearsonr, kendalltau
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)
    ax_ts  = fig.add_subplot(gs[0, :])       # top row: full width
    ax_s1  = fig.add_subplot(gs[1, 0])
    ax_s2  = fig.add_subplot(gs[1, 1])
    ax_bar = fig.add_subplot(gs[1, 2])

    colors = {
        "cpue_trip_raw": "#78909C",
        "cpue_trip_std": "#1565C0",
        "cpue_fisherman": "#2E7D32",
    }
    labels = {
        "cpue_trip_raw": "t/viaje — sin corrección",
        "cpue_trip_std": "t/viaje — corregida (GAM)",
        "cpue_fisherman": "t/pescador",
    }

    # ── Panel A: time series ──────────────────────────────────────────────────
    for col, color in colors.items():
        idx_col = f"{col}_idx"
        ax_ts.plot(comp["year"], comp[idx_col], color=color, lw=2,
                   marker="o", ms=3.5, label=labels[col])
    ax_ts.axhline(1, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax_ts.set_xlabel("Año", fontsize=10)
    ax_ts.set_ylabel("Índice CPUE (media = 1)", fontsize=10)
    ax_ts.set_title("Comparación de formulaciones de CPUE (media del período = 1)",
                    fontsize=11)
    ax_ts.legend(fontsize=9, frameon=False)
    ax_ts.tick_params(labelsize=9)

    # ── Panel B: corrected vs raw (trip) ──────────────────────────────────────
    mask = comp["cpue_trip_std"].notna() & comp["cpue_trip_raw"].notna()
    x_b, y_b = comp.loc[mask, "cpue_trip_raw_idx"], comp.loc[mask, "cpue_trip_std_idx"]
    r_b, _ = pearsonr(x_b, y_b)
    sc = ax_s1.scatter(x_b, y_b, c=comp.loc[mask, "year"],
                       cmap="RdYlBu_r", s=45, zorder=3, edgecolors="k", linewidths=0.3)
    mn, mx = min(x_b.min(), y_b.min()), max(x_b.max(), y_b.max())
    ax_s1.plot([mn, mx], [mn, mx], "k--", lw=0.8, alpha=0.4)
    ax_s1.set_xlabel("t/viaje — sin corrección (índice)", fontsize=9)
    ax_s1.set_ylabel("t/viaje — corregida (índice)", fontsize=9)
    ax_s1.set_title(f"Corrección vs sin corrección\nr = {r_b:.3f}", fontsize=9)
    plt.colorbar(sc, ax=ax_s1, label="Año", shrink=0.8)

    # ── Panel C: corrected trip vs per-fisherman ──────────────────────────────
    x_c, y_c = comp["cpue_trip_std_idx"], comp["cpue_fisherman_idx"]
    r_c, _ = pearsonr(x_c.dropna(), y_c.dropna())
    sc2 = ax_s2.scatter(x_c, y_c, c=comp["year"],
                        cmap="RdYlBu_r", s=45, zorder=3, edgecolors="k", linewidths=0.3)
    mn2, mx2 = min(x_c.min(), y_c.min()), max(x_c.max(), y_c.max())
    ax_s2.plot([mn2, mx2], [mn2, mx2], "k--", lw=0.8, alpha=0.4)
    ax_s2.set_xlabel("t/viaje — corregida (índice)", fontsize=9)
    ax_s2.set_ylabel("t/pescador (índice)", fontsize=9)
    ax_s2.set_title(f"Corregida vs t/pescador\nr = {r_c:.3f}", fontsize=9)
    plt.colorbar(sc2, ax=ax_s2, label="Año", shrink=0.8)

    # ── Panel D: Mann-Kendall tau and significance ────────────────────────────
    mk_data = []
    for col, label in labels.items():
        vals = comp[col].dropna()
        yrs  = comp.loc[vals.index, "year"]
        tau, p = kendalltau(yrs.values, vals.values)
        sig = "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else "ns"
        mk_data.append({"label": label, "tau": tau, "p": p, "sig": sig,
                        "color": colors[col]})

    y_pos = range(len(mk_data))
    bars = ax_bar.barh(
        [d["label"] for d in mk_data],
        [d["tau"] for d in mk_data],
        color=[d["color"] for d in mk_data],
        edgecolor="k", linewidth=0.5,
    )
    ax_bar.axvline(0, color="k", lw=0.8, ls="--")
    for i, d in enumerate(mk_data):
        xpos = d["tau"] + (0.02 if d["tau"] >= 0 else -0.02)
        ha   = "left" if d["tau"] >= 0 else "right"
        ax_bar.text(xpos, i, f"{d['sig']}  p={d['p']:.3f}",
                    va="center", ha=ha, fontsize=8)
    ax_bar.set_xlabel("Kendall τ", fontsize=9)
    ax_bar.set_title("Tendencia temporal (Mann-Kendall)", fontsize=9)
    ax_bar.tick_params(labelsize=8)

    out = FIG_DIR / "cpue_three_way_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out.name)


def print_comparison_stats(comp: pd.DataFrame) -> None:
    from scipy.stats import pearsonr, spearmanr, kendalltau

    HR = "─" * 68
    print(f"\n{HR}")
    print("THREE-WAY CPUE COMPARISON")
    print(HR)
    print(f"{'Measure':<35} {'Median':>8}  {'CV':>6}  {'MK τ':>7}  {'p':>6}")
    print(HR)

    raw_cols = ["cpue_trip_raw", "cpue_trip_std", "cpue_fisherman"]
    names    = ["t/viaje sin corrección", "t/viaje corregida (GAM)", "t/pescador"]
    for col, name in zip(raw_cols, names):
        vals = comp[col].dropna()
        cv   = vals.std() / vals.mean()
        yrs  = comp.loc[vals.index, "year"]
        tau, p = kendalltau(yrs.values, vals.values)
        sig  = "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else "ns"
        print(f"  {name:<33} {vals.median():>8.4f}  {cv:>6.3f}  {tau:>+7.3f}  {p:>6.3f} {sig}")

    print(f"\n{HR}")
    print("Pearson r between indices (normalised):")
    idx_cols = [f"{c}_idx" for c in raw_cols]
    idx_names = ["raw", "std", "fish"]
    for i in range(len(idx_cols)):
        for j in range(i+1, len(idx_cols)):
            x = comp[idx_cols[i]].dropna()
            y = comp[idx_cols[j]].dropna()
            idx = x.index.intersection(y.index)
            r, p = pearsonr(x[idx], y[idx])
            rs, ps = spearmanr(x[idx], y[idx])
            print(f"  {idx_names[i]} vs {idx_names[j]:6s}  r={r:+.3f}  ρ={rs:+.3f}  p(r)={p:.4f}")
    print(HR)


# ─── Console summary ──────────────────────────────────────────────────────────

def print_summary(df, gam_stats, partial, lmm_table, std_index):
    HR = "─" * 70
    print("\n" + "=" * 70)
    print("CPUE STANDARDIZATION SUMMARY")
    print("=" * 70)
    print(f"  n={len(df)} obs  |  8 locals  |  {df['year'].min()}–{df['year'].max()}")
    print()

    print("─── Raw CPUE formulations (median across all obs) ─────────────────")
    for col, label in [
        ("cpue_ton_per_trip",  "per trip     (production / trips)"),
        ("cpue_per_vessel",    "per vessel   (production / fleet_monitored)"),
        ("cpue_per_fisherman", "per fisherman(production / est_fishermen)"),
    ]:
        print(f"  {label:<45} median={df[col].median():.4f}  "
              f"range={df[col].min():.4f}–{df[col].max():.4f}")
    print()

    print("─── Model G: GammaGAM ─────────────────────────────────────────────")
    print(f"  deviance explained : {gam_stats['deviance_explained']:.3f}")
    print(f"  AIC                : {gam_stats['AIC']:.1f}")
    print(f"  GCV                : {gam_stats['GCV']:.4f}")
    print(f"  edof               : {gam_stats['edof']:.2f}")
    print()
    print("  Partial effects (linear terms, at x=0 = grand geometric mean):")
    for term in ["log_fleet_c", "log_fishermen_c"]:
        sub = partial[partial["term"] == term]
        mid = sub.iloc[len(sub)//2]
        lo, hi = mid["ci_lower"], mid["ci_upper"]
        sig = "*" if (lo > 0 or hi < 0) else ""
        print(f"    {term:<22} effect at mean={mid['partial_effect']:+.4f}  "
              f"[{lo:+.3f}, {hi:+.3f}] {sig}")
    print()

    print(HR)
    print("─── Model M: LMM log-normal  (random intercept for local) ─────────")
    print(f"  ICC (local)  : {lmm_table.attrs['ICC']:.3f}")
    print(f"  RE variance  : {lmm_table.attrs['RE_var']:.4f}")
    print(f"  Residual var : {lmm_table.attrs['resid_var']:.4f}")
    print()
    print("  Fixed effects:")
    for _, r in lmm_table.iterrows():
        print(f"    {r['term']:<25} β={r['coef']:+.5f}  "
              f"[{r['ci_lower']:+.5f}, {r['ci_upper']:+.5f}]  {r['sig']}")
    print()

    print(HR)
    print("─── Standardized index: correction magnitude ───────────────────────")
    idx = std_index.copy()
    idx["correction"] = idx["cpue_std_gam_idx"] / idx["cpue_raw_idx"] - 1
    print(f"  Mean |correction| : {idx['correction'].abs().mean():.4f}")
    print(f"  Max  |correction| : {idx['correction'].abs().max():.4f}  "
          f"(year {idx.loc[idx['correction'].abs().idxmax(),'year']:.0f})")
    print(f"  Corr(std, raw)    : "
          f"{float(np.corrcoef(idx['cpue_std_gam_idx'], idx['cpue_raw_idx'])[0,1]):.4f}")
    print()
    print("  (A correction near 0 means fleet/fishermen changes had little")
    print("   influence on the raw CPUE index in that year.)")
    print()
    print("Figures saved to:", FIG_DIR)
    print("=" * 70)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg.setup_dirs()

    df = build_dataset()
    df.to_csv(cfg.DATA_PROCESSED / "cpue_std_data.csv", index=False)

    # Model G: GammaGAM
    log.info("Fitting GammaGAM …")
    gam, X, local_cats, local_enc = fit_gamma_gam(df)

    gam_stats = {
        "deviance_explained": round(float(gam.statistics_["pseudo_r2"]["explained_deviance"]), 4),
        "AIC":  round(float(gam.statistics_["AIC"]), 2),
        "GCV":  round(float(gam.statistics_["GCV"]), 6),
        "edof": round(float(gam.statistics_["edof"]), 2),
        "n_obs": len(df),
    }
    pd.DataFrame([gam_stats]).to_csv(cfg.DATA_PROCESSED / "cpue_std_gam_summary.csv", index=False)

    partial = gam_partial_effects(gam, X)
    partial.to_csv(cfg.DATA_PROCESSED / "cpue_std_gam_partial.csv", index=False)

    std_index_gam = build_std_index_gam(gam, df, local_enc)
    local_std     = build_local_std_gam(gam, df, local_enc)

    # Model M: LMM
    log.info("Fitting LMM (log-normal, random intercept) …")
    lmm_result, lmm_table = fit_lmm(df)
    lmm_table.to_csv(cfg.DATA_PROCESSED / "cpue_std_lmm.csv", index=False)

    std_index_lmm = build_std_index_lmm(lmm_result, df)

    # Merge standardized indices
    std_index = std_index_gam.merge(std_index_lmm, on="year", how="left")
    std_index.to_csv(cfg.DATA_PROCESSED / "cpue_std_index.csv", index=False)
    log.info("Saved cpue_std_index.csv  (%d years)", len(std_index))

    # Three-way comparison
    comp = build_comparison_index(df, std_index)
    comp.to_csv(cfg.DATA_PROCESSED / "cpue_three_way_index.csv", index=False)
    log.info("Saved cpue_three_way_index.csv  (%d years)", len(comp))

    # Figures
    plot_raw_comparison(df)
    plot_covariates_time(df)
    plot_partial_effects(partial, gam_stats)
    plot_std_index(std_index)
    plot_by_local(local_std)
    plot_three_way_comparison(comp)

    print_summary(df, gam_stats, partial, lmm_table, std_index)
    print_comparison_stats(comp)
    log.info("12_cpue_std.py complete.")


if __name__ == "__main__":
    main()
