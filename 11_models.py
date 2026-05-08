"""
11_models.py
============
Statistical models and tests for Fish catches × Oil platforms Brazil.

Scientific questions:
  Q1. Does CPUE differ significantly between platform proximity classes?
  Q2. Does CPUE differ significantly between MPA exposure classes?
  Q3. Is there a temporal trend in CPUE/diversity, and does it differ by class?
  Q4. Does species composition differ between exposure classes?
  Q5. Do continuous distance metrics correlate with productivity/diversity?

Methods:
  A. Mann-Kendall trend test (Sen's slope) — aggregated exposure-class series
  B. Mann-Kendall per local — individual trend slopes compared by class
  C. Mann-Whitney U / Kruskal-Wallis + Holm correction — group differences
  D. Spearman ρ — continuous distance vs local-mean productivity/diversity
  E. PERMANOVA (Bray-Curtis, 999 permutations) — species composition

Inputs (data/processed/ or data/interim/):
  timeseries_platform.csv, timeseries_mpa.csv, timeseries_combined.csv
  productivity_local_year.csv, species_catch_wide_local_year.csv
  local_exposure.csv (interim)

Outputs (data/processed/):
  models_mann_kendall.csv        Trend tests on aggregated class time series
  models_mann_kendall_local.csv  Trend test per local
  models_kruskal.csv             KW overall + pairwise Mann-Whitney results
  models_spearman.csv            Spearman ρ between distances and metrics
  models_permanova.csv           PERMANOVA results for species composition
  models_summary.txt             Human-readable narrative of all results

Run:
  python 11_models.py
"""

import logging
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_00 as cfg  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format=cfg.LOG_FORMAT, datefmt=cfg.LOG_DATEFMT,
    handlers=[
        logging.FileHandler(cfg.LOGS / "11_models.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("11_models")
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _check(path: Path) -> None:
    if not path.exists():
        log.error("Missing input: %s", path)
        sys.exit(1)


def _save(df: pd.DataFrame, name: str) -> None:
    out = cfg.DATA_PROCESSED / name
    df.to_csv(out, index=False)
    log.info("Saved %s  (%d rows × %d cols)", out.name, len(df), df.shape[1])


def _load(name: str) -> pd.DataFrame:
    for base in [cfg.DATA_PROCESSED, cfg.DATA_INTERIM]:
        p = base / name
        if p.exists():
            return pd.read_csv(p)
    log.error("File not found: %s", name)
    sys.exit(1)


def _sig(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return "ns"


# ─── A. Mann-Kendall trend test ───────────────────────────────────────────────

def mann_kendall(values: np.ndarray) -> dict:
    """
    Mann-Kendall trend test with Sen's slope estimator.
    Uses scipy.stats.kendalltau on (time_index, values).
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 4:
        return dict(n=n, tau=np.nan, pvalue=np.nan,
                    sens_slope=np.nan, slope_pct_yr=np.nan,
                    trend="insufficient data")

    time_idx = np.arange(n)
    tau, p = stats.kendalltau(time_idx, arr)

    # Sen's slope: median of all pairwise slopes
    slopes = [
        (arr[j] - arr[i]) / (j - i)
        for i in range(n) for j in range(i + 1, n)
    ]
    slope = float(np.median(slopes))

    median_val = float(np.median(arr))
    slope_pct = (slope / median_val * 100) if median_val != 0 else np.nan

    if p < 0.05:
        trend = "increasing" if tau > 0 else "decreasing"
    elif p < 0.10:
        trend = "increasing (marginal)" if tau > 0 else "decreasing (marginal)"
    else:
        trend = "no significant trend"

    return dict(
        n=n, tau=round(float(tau), 4), pvalue=round(float(p), 4),
        sens_slope=round(slope, 6), slope_pct_yr=round(slope_pct, 3),
        trend=trend,
    )


def mk_on_timeseries(
    ts: pd.DataFrame, group_col: str, value_cols: list
) -> pd.DataFrame:
    """Mann-Kendall on each group's annual time series."""
    records = []
    for group, sub in ts.groupby(group_col):
        sub_sorted = sub.sort_values("year")
        for col in value_cols:
            if col not in sub_sorted.columns:
                continue
            mk = mann_kendall(sub_sorted[col].values)
            records.append({group_col: group, "variable": col, **mk})
    return pd.DataFrame(records)


def mk_per_local(master: pd.DataFrame, value_cols: list) -> pd.DataFrame:
    """Mann-Kendall for each local × variable."""
    records = []
    for local, sub in master.groupby("local"):
        sub_sorted = sub.sort_values("year")
        meta = {
            "local": local,
            "platform_exposure_class": sub_sorted["platform_exposure_class"].iloc[0],
            "mpa_exposure_class": sub_sorted["mpa_exposure_class"].iloc[0],
            "platform_dist_km": round(float(sub_sorted["platform_dist_km_mean"].iloc[0]), 2),
            "mpa_dist_km": round(float(sub_sorted["mpa_dist_km_mean"].iloc[0]), 2),
        }
        for col in value_cols:
            if col not in sub_sorted.columns:
                continue
            mk = mann_kendall(sub_sorted[col].values)
            records.append({**meta, "variable": col, **mk})
    return pd.DataFrame(records)


# ─── B. Kruskal-Wallis + pairwise Mann-Whitney ────────────────────────────────

def kruskal_pairwise(
    df: pd.DataFrame, group_col: str, value_col: str
) -> tuple:
    """
    Overall Kruskal-Wallis + Holm-corrected pairwise Mann-Whitney U.
    Returns (overall_row_dict, pairwise_df).
    Note: observations are local-year rows; groups differ only in which locals
    they contain, so effective n is the number of locals per class.
    """
    groups = {
        g: df.loc[df[group_col] == g, value_col].dropna().values
        for g in sorted(df[group_col].dropna().unique())
    }
    groups = {g: v for g, v in groups.items() if len(v) > 0}
    n_grp = len(groups)

    if n_grp >= 2:
        kw_stat, kw_p = kruskal(*groups.values())
    else:
        kw_stat, kw_p = np.nan, np.nan

    overall = {
        "group_col": group_col, "variable": value_col,
        "n_groups": n_grp,
        "kw_statistic": round(float(kw_stat), 4) if not np.isnan(kw_stat) else np.nan,
        "kw_pvalue": round(float(kw_p), 4) if not np.isnan(kw_p) else np.nan,
        "sig": _sig(kw_p),
    }

    pairs = list(combinations(groups.keys(), 2))
    raw_p, u_stats = [], []
    for a, b in pairs:
        u, p = mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        u_stats.append(float(u))
        raw_p.append(float(p))

    adj_p = multipletests(raw_p, method="holm")[1] if raw_p else []

    rows = []
    for i, (a, b) in enumerate(pairs):
        rows.append({
            "group_col": group_col, "variable": value_col,
            "group_a": a, "group_b": b,
            "n_obs_a": len(groups[a]), "n_obs_b": len(groups[b]),
            "median_a": round(float(np.median(groups[a])), 4),
            "median_b": round(float(np.median(groups[b])), 4),
            "U_statistic": u_stats[i],
            "pvalue_raw": round(raw_p[i], 4),
            "pvalue_holm": round(float(adj_p[i]), 4),
            "sig_holm": _sig(float(adj_p[i])),
        })

    return overall, pd.DataFrame(rows)


# ─── C. Spearman correlations ─────────────────────────────────────────────────

def spearman_distance_metrics(
    local_summary: pd.DataFrame,
    dist_cols: list,
    metric_cols: list,
) -> pd.DataFrame:
    """
    Spearman ρ between each distance variable and each productivity/diversity metric.
    One row per local (local-level means collapsed over years).
    """
    records = []
    for dist_col in dist_cols:
        for metric_col in metric_cols:
            sub = local_summary[[dist_col, metric_col]].dropna()
            n = len(sub)
            if n < 4:
                records.append(dict(distance_var=dist_col, metric_var=metric_col,
                                    n=n, rho=np.nan, pvalue=np.nan, sig="n/a"))
                continue
            rho, p = spearmanr(sub[dist_col].values, sub[metric_col].values)
            records.append(dict(
                distance_var=dist_col, metric_var=metric_col, n=n,
                rho=round(float(rho), 4), pvalue=round(float(p), 4),
                sig=_sig(float(p)),
            ))
    return pd.DataFrame(records)


# ─── D. PERMANOVA ─────────────────────────────────────────────────────────────

def _pseudo_f(D: np.ndarray, groups: np.ndarray) -> float:
    """PERMANOVA pseudo-F statistic from a pre-computed distance matrix."""
    unique = np.unique(groups)
    n = len(groups)
    SS_T = np.sum(D ** 2) / n
    SS_W = 0.0
    for u in unique:
        idx = np.where(groups == u)[0]
        if len(idx) > 1:
            SS_W += np.sum(D[np.ix_(idx, idx)] ** 2) / len(idx)
    SS_B = SS_T - SS_W
    df_B = len(unique) - 1
    df_W = n - len(unique)
    if df_W <= 0 or SS_W == 0:
        return np.nan
    return (SS_B / df_B) / (SS_W / df_W)


def permanova(
    comm: np.ndarray,
    groups: np.ndarray,
    n_perm: int = 999,
    seed: int = 42,
) -> dict:
    """
    One-way PERMANOVA using Bray-Curtis dissimilarity on row-normalised
    relative abundances. Excludes all-zero rows.
    """
    mask = comm.sum(axis=1) > 0
    comm, groups = comm[mask], groups[mask]
    if len(np.unique(groups)) < 2 or len(groups) < 4:
        return dict(n_obs=len(groups), n_groups=len(np.unique(groups)),
                    pseudo_F=np.nan, pvalue=np.nan, n_permutations=n_perm)

    row_sums = comm.sum(axis=1, keepdims=True)
    comm_rel = comm / np.where(row_sums > 0, row_sums, 1.0)
    D = squareform(pdist(comm_rel, metric="braycurtis"))

    F_obs = _pseudo_f(D, groups)
    rng = np.random.default_rng(seed)
    F_null = np.array([_pseudo_f(D, rng.permutation(groups)) for _ in range(n_perm)])
    valid = F_null[~np.isnan(F_null)]
    p_val = (float(np.sum(valid >= F_obs)) + 1) / (len(valid) + 1) if (
        not np.isnan(F_obs) and len(valid) > 0
    ) else np.nan

    unique, counts = np.unique(groups, return_counts=True)
    return {
        "n_obs": int(len(groups)),
        "n_groups": int(len(unique)),
        "groups": " | ".join(unique.astype(str)),
        "group_sizes": ", ".join(f"{u}:{c}" for u, c in zip(unique, counts)),
        "pseudo_F": round(float(F_obs), 4) if not np.isnan(F_obs) else np.nan,
        "pvalue": round(float(p_val), 4) if not np.isnan(p_val) else np.nan,
        "sig": _sig(float(p_val)) if not np.isnan(p_val) else "n/a",
        "n_permutations": n_perm,
    }


def run_permanova_suite(wide: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    """
    Run PERMANOVA for all combinations of grouping variable × temporal filter.
    """
    sp_cols = [c for c in wide.columns if c not in ("local", "year")]

    wide = wide.merge(
        exposure[["local", "platform_exposure_class", "mpa_exposure_class",
                  "combined_exposure_class"]],
        on="local", how="left",
    )

    def _period(y):
        for p, (a, b) in cfg.PERIOD_BREAKS.items():
            if a <= int(y) <= b:
                return p
        return None

    wide["period"] = wide["year"].apply(_period)

    tests = [
        ("platform_exposure_class", None,       "all years"),
        ("mpa_exposure_class",      None,       "all years"),
        ("combined_exposure_class", None,       "all years"),
        ("platform_exposure_class", "early",    "early"),
        ("platform_exposure_class", "middle",   "middle"),
        ("platform_exposure_class", "recent",   "recent"),
        ("mpa_exposure_class",      "early",    "early"),
        ("mpa_exposure_class",      "middle",   "middle"),
        ("mpa_exposure_class",      "recent",   "recent"),
        ("period",                  None,       "all groups"),
    ]

    records = []
    for group_col, period_filter, period_label in tests:
        sub = wide[wide["period"] == period_filter].copy() if period_filter else wide.copy()
        sub = sub.dropna(subset=[group_col])

        groups = sub[group_col].astype(str).values
        comm   = sub[sp_cols].fillna(0).values.astype(float)

        result = permanova(comm, groups)
        records.append({
            "grouping_variable": group_col,
            "period_filter": period_label,
            **result,
        })
        log.info("PERMANOVA [%s | %s]: F=%.3f  p=%.3f  %s",
                 group_col, period_label,
                 result["pseudo_F"] if not np.isnan(result["pseudo_F"]) else float("nan"),
                 result["pvalue"]   if not np.isnan(result["pvalue"])   else float("nan"),
                 result["sig"])

    return pd.DataFrame(records)


# ─── E. Summary text ──────────────────────────────────────────────────────────

def write_summary(
    mk_agg: pd.DataFrame,
    mk_local: pd.DataFrame,
    kw_overall: pd.DataFrame,
    kw_pairs: pd.DataFrame,
    sp_df: pd.DataFrame,
    perm_df: pd.DataFrame,
) -> str:
    lines = []
    HR = "─" * 70

    lines += [
        "=" * 70,
        "FISH × PLATFORMS BRAZIL — STATISTICAL MODELS SUMMARY",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "Note: n=8 locals (2 platform classes; 4 MPA classes). Statistical",
        "      power is low; interpret p-values with caution.",
        "=" * 70,
    ]

    # A. MK aggregated
    lines += ["", "A. MANN-KENDALL TRENDS (aggregated by exposure class)", HR]
    _grp_cols = ["platform_exposure_class", "mpa_exposure_class", "combined_exposure_class"]
    for _, r in mk_agg.iterrows():
        grp_col = next(
            (c for c in _grp_cols if c in r.index and not pd.isna(r.get(c))), None
        )
        grp_val = str(r[grp_col]) if grp_col else "?"
        lines.append(
            f"  {grp_val:<30} {r['variable']:<22} "
            f"τ={r['tau']:+.3f}  p={r['pvalue']:.3f} {_sig(r['pvalue'])}  → {r['trend']}"
        )

    # B. MK per local
    lines += ["", "B. MANN-KENDALL PER LOCAL (CPUE)", HR]
    sub = mk_local[mk_local["variable"] == "cpue_ton_per_trip"]
    for _, r in sub.sort_values("platform_exposure_class").iterrows():
        lines.append(
            f"  {r['local']:<20} [{r['platform_exposure_class']}, {r['mpa_exposure_class']}]  "
            f"τ={r['tau']:+.3f}  p={r['pvalue']:.3f} {_sig(r['pvalue'])}  "
            f"slope={r['sens_slope']:+.4f} t/trip/yr  → {r['trend']}"
        )

    # C. Kruskal-Wallis
    lines += ["", "C. KRUSKAL-WALLIS + MANN-WHITNEY (group differences)", HR]
    for _, r in kw_overall.iterrows():
        lines.append(
            f"  KW [{r['group_col']} | {r['variable']}]: "
            f"H={r['kw_statistic']:.3f}  p={r['kw_pvalue']:.3f} {r['sig']}"
        )
    lines.append("")
    if not kw_pairs.empty:
        sig_pairs = kw_pairs[kw_pairs["sig_holm"].isin(["*", "**", "***", "."])]
        if not sig_pairs.empty:
            lines.append("  Significant pairwise contrasts (Holm-corrected):")
            for _, r in sig_pairs.iterrows():
                lines.append(
                    f"    {r['variable']}: {r['group_a']} vs {r['group_b']}  "
                    f"median {r['median_a']:.3f} vs {r['median_b']:.3f}  "
                    f"p_holm={r['pvalue_holm']:.3f} {r['sig_holm']}"
                )
        else:
            lines.append("  No pairwise contrasts significant after Holm correction.")

    # D. Spearman
    lines += ["", "D. SPEARMAN ρ (distance vs productivity/diversity)", HR]
    for _, r in sp_df.iterrows():
        lines.append(
            f"  {r['distance_var']:<25} vs {r['metric_var']:<22} "
            f"ρ={r['rho']:+.3f}  p={r['pvalue']:.3f} {r['sig']}  (n={r['n']})"
        )

    # E. PERMANOVA
    lines += ["", "E. PERMANOVA (Bray-Curtis species composition)", HR]
    for _, r in perm_df.iterrows():
        lines.append(
            f"  [{r['grouping_variable']:<30} | {r['period_filter']:<12}] "
            f"n={r['n_obs']}  F={r['pseudo_F']:.3f}  p={r['pvalue']:.3f} {r['sig']}"
        )

    lines.append("")
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg.setup_dirs()

    for f in [
        "timeseries_platform.csv",
        "timeseries_mpa.csv",
        "timeseries_combined.csv",
        "productivity_local_year.csv",
        "species_catch_wide_local_year.csv",
    ]:
        _check(cfg.DATA_PROCESSED / f)
    _check(cfg.DATA_INTERIM / "local_exposure.csv")

    ts_plat  = _load("timeseries_platform.csv")
    ts_mpa   = _load("timeseries_mpa.csv")
    ts_comb  = _load("timeseries_combined.csv")
    master   = _load("productivity_local_year.csv")
    wide     = _load("species_catch_wide_local_year.csv")
    exposure = _load("local_exposure.csv")

    # ── A. Mann-Kendall on aggregated class time series ───────────────────────
    log.info("A. Mann-Kendall on aggregated time series")

    mk_plat = mk_on_timeseries(
        ts_plat, "platform_exposure_class",
        ["cpue", "production_ton", "assisted_trips"],
    )
    mk_mpa = mk_on_timeseries(
        ts_mpa, "mpa_exposure_class",
        ["cpue", "production_ton", "assisted_trips"],
    )
    mk_comb = mk_on_timeseries(
        ts_comb, "combined_exposure_class",
        ["cpue", "production_ton"],
    )
    mk_agg = pd.concat([mk_plat, mk_mpa, mk_comb], ignore_index=True)
    _save(mk_agg, "models_mann_kendall.csv")

    # ── B. Mann-Kendall per local ──────────────────────────────────────────────
    log.info("B. Mann-Kendall per local")

    mk_loc = mk_per_local(
        master,
        ["cpue_ton_per_trip", "production_ton", "richness", "shannon_h", "pielou_j"],
    )
    _save(mk_loc, "models_mann_kendall_local.csv")

    # ── C. Kruskal-Wallis / Mann-Whitney ──────────────────────────────────────
    log.info("C. Kruskal-Wallis + pairwise Mann-Whitney")

    metrics = ["cpue_ton_per_trip", "production_ton", "richness", "shannon_h"]
    kw_overall_rows, kw_pair_rows = [], []

    for group_col in ["platform_exposure_class", "mpa_exposure_class"]:
        for metric in metrics:
            overall, pairs = kruskal_pairwise(master, group_col, metric)
            kw_overall_rows.append(overall)
            kw_pair_rows.append(pairs)

    kw_overall = pd.DataFrame(kw_overall_rows)
    kw_pairs   = pd.concat(kw_pair_rows, ignore_index=True)
    _save(kw_overall, "models_kruskal.csv")

    # Append pairwise to the same file as a second sheet would be ideal, but
    # keep it a single CSV and append pairwise as a separate file.
    _save(kw_pairs, "models_mann_whitney_pairwise.csv")

    # ── D. Spearman correlations ───────────────────────────────────────────────
    log.info("D. Spearman correlations (local-level means)")

    local_means = master.groupby("local").agg(
        platform_dist_km_mean=("platform_dist_km_mean", "first"),
        mpa_dist_km_mean=("mpa_dist_km_mean", "first"),
        cpue_mean=("cpue_ton_per_trip", "median"),
        production_median=("production_ton", "median"),
        richness_mean=("richness", "mean"),
        shannon_mean=("shannon_h", "mean"),
        pielou_mean=("pielou_j", "mean"),
    ).reset_index()

    sp_df = spearman_distance_metrics(
        local_means,
        dist_cols=["platform_dist_km_mean", "mpa_dist_km_mean"],
        metric_cols=["cpue_mean", "production_median", "richness_mean",
                     "shannon_mean", "pielou_mean"],
    )
    _save(sp_df, "models_spearman.csv")

    # ── E. PERMANOVA ──────────────────────────────────────────────────────────
    log.info("E. PERMANOVA (species composition)")

    perm_df = run_permanova_suite(wide.copy(), exposure)
    _save(perm_df, "models_permanova.csv")

    # ── Summary text ──────────────────────────────────────────────────────────
    summary = write_summary(mk_agg, mk_loc, kw_overall, kw_pairs, sp_df, perm_df)
    out_txt = cfg.OUTPUTS / "models_summary.txt"
    out_txt.write_text(summary, encoding="utf-8")
    log.info("Saved %s", out_txt.name)
    print(summary)

    log.info("11_models.py complete.")


if __name__ == "__main__":
    main()
