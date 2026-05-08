"""
06_species_composition.py
=========================
Builds species catch matrices, relative-abundance matrices, and
compositional-change summaries.

The main question answered here:
  Does species composition differ across platform exposure, MPA exposure,
  and combined exposure contexts? Which species drive those differences?

Outputs (data/processed/):
  - species_catch_matrix_local_year.csv     long: local × year × species × production
  - species_rel_matrix_local_year.csv       same, normalised to relative abundance
  - species_catch_wide_local_year.csv       wide pivot (local×year rows, species cols)
  - species_rel_wide_local_year.csv         wide pivot, relative abundance
  - species_catch_wide_platform.csv         species × platform_exposure_class pivot
  - species_catch_wide_mpa.csv              species × mpa_exposure_class pivot
  - species_catch_wide_combined.csv         species × combined_exposure_class pivot
  - species_SIMPER_platform.csv             species ranked by contribution to
                                             platform-exposure class differences
  - species_SIMPER_mpa.csv
  - species_SIMPER_combined.csv
  - species_turnover_period.csv             Bray-Curtis-based turnover between periods

Run:
  python 06_species_composition.py
"""

import logging
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.distance import braycurtis

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_00 as cfg  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format=cfg.LOG_FORMAT, datefmt=cfg.LOG_DATEFMT,
    handlers=[
        logging.FileHandler(cfg.LOGS / "06_species_composition.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("06_species_composition")


def _check(path: Path) -> None:
    if not path.exists():
        log.error("Missing input: %s", path)
        sys.exit(1)


def _save(df: pd.DataFrame, name: str) -> None:
    out = cfg.DATA_PROCESSED / name
    df.to_csv(out, index=False)
    log.info("Saved %s  (%d rows × %d cols)", out.name, len(df), df.shape[1])


# ─── Species matrices ─────────────────────────────────────────────────────────

def build_long_matrix(land: pd.DataFrame) -> pd.DataFrame:
    """
    Returns tidy long-format species catch table with relative abundance.
    """
    df = land[["local", "year", "species", "sp_production_ton"]].copy()
    total = df.groupby(["local", "year"])["sp_production_ton"].transform("sum")
    df["rel_abundance"] = np.where(total > 0, df["sp_production_ton"] / total, np.nan)
    return df


def build_wide_matrix(long_df: pd.DataFrame, index: list[str],
                      value_col: str, fill: float = 0.0) -> pd.DataFrame:
    """
    Pivot long → wide: rows = index groups, cols = species.
    """
    wide = long_df.pivot_table(
        index=index, columns="species",
        values=value_col, aggfunc="sum", fill_value=fill,
    ).reset_index()
    wide.columns.name = None
    return wide


def build_exposure_matrix(land: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Aggregate species catches by an exposure class column.
    Returns wide pivot indexed by group_col.
    """
    grp = land.groupby([group_col, "species"])["sp_production_ton"].sum().reset_index()
    total = grp.groupby(group_col)["sp_production_ton"].transform("sum")
    grp["rel_abundance"] = np.where(total > 0, grp["sp_production_ton"] / total, np.nan)
    wide = grp.pivot_table(
        index=group_col, columns="species",
        values="rel_abundance", fill_value=0,
    ).reset_index()
    wide.columns.name = None
    return wide


# ─── SIMPER-style species contribution analysis ───────────────────────────────

def simper_contribution(
    land: pd.DataFrame,
    group_col: str,
    value_col: str = "sp_production_ton",
) -> pd.DataFrame:
    """
    For each pair of exposure classes, compute each species' average
    contribution to the Bray-Curtis dissimilarity between them.

    Returns a long DataFrame:
      group_a, group_b, species, mean_abundance_a, mean_abundance_b,
      mean_contribution, cumulative_contribution, rank
    """
    # Build relative abundance matrix: rows = groups, cols = species
    grp = land.groupby([group_col, "species"])[value_col].sum().reset_index()
    total = grp.groupby(group_col)[value_col].transform("sum")
    grp["rel"] = np.where(total > 0, grp[value_col] / total, 0.0)

    wide = grp.pivot_table(
        index=group_col, columns="species", values="rel", fill_value=0,
    )
    groups   = wide.index.tolist()
    species  = wide.columns.tolist()
    records  = []

    for g_a, g_b in combinations(groups, 2):
        a = wide.loc[g_a].values
        b = wide.loc[g_b].values
        denom = (a + b).sum()
        if denom == 0:
            continue
        contribs = np.abs(a - b) / denom  # proportional contribution to B-C
        total_bc = braycurtis(a, b)

        df_pair = pd.DataFrame({
            "group_a":           g_a,
            "group_b":           g_b,
            "species":           species,
            "mean_abundance_a":  a,
            "mean_abundance_b":  b,
            "contribution":      contribs,
        })
        df_pair = df_pair.sort_values("contribution", ascending=False).reset_index(drop=True)
        df_pair["cumulative_contribution"] = df_pair["contribution"].cumsum()
        df_pair["rank"]                    = range(1, len(df_pair) + 1)
        df_pair["bray_curtis_total"]        = total_bc
        records.append(df_pair)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


# ─── Bray-Curtis turnover between periods ────────────────────────────────────

def period_turnover(land: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Bray-Curtis dissimilarity between temporal periods
    at the level of each local and overall.
    """
    def assign_period(y):
        for period, (a, b) in cfg.PERIOD_BREAKS.items():
            if pd.notna(y) and a <= int(y) <= b:
                return period
        return pd.NA

    land = land.copy()
    land["period"] = land["year"].apply(assign_period)
    land = land.dropna(subset=["period"])

    records = []
    for local, sub in [("ALL", land)] + list(land.groupby("local")):
        grp = sub.groupby(["period", "species"])["sp_production_ton"].sum().reset_index()
        total_p = grp.groupby("period")["sp_production_ton"].transform("sum")
        grp["rel"] = np.where(total_p > 0, grp["sp_production_ton"] / total_p, 0.0)
        wide = grp.pivot_table(
            index="period", columns="species", values="rel", fill_value=0,
        )
        periods = wide.index.tolist()
        for p_a, p_b in combinations(periods, 2):
            a = wide.loc[p_a].values if p_a in wide.index else None
            b = wide.loc[p_b].values if p_b in wide.index else None
            if a is None or b is None:
                continue
            bc = braycurtis(a, b)
            records.append({
                "local":    local,
                "period_a": p_a,
                "period_b": p_b,
                "bray_curtis": bc,
            })

    return pd.DataFrame(records)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg.setup_dirs()

    for f in ["landings_clean.csv", "local_exposure.csv"]:
        _check(cfg.DATA_INTERIM / f)

    land     = pd.read_csv(cfg.DATA_INTERIM / "landings_clean.csv")
    exposure = pd.read_csv(cfg.DATA_INTERIM / "local_exposure.csv")

    # Join exposure classes to landings
    land = land.merge(
        exposure[["local", "platform_exposure_class", "mpa_exposure_class",
                  "combined_exposure_class", "municipality"]],
        on="local", how="left",
    )

    # ── Long and wide matrices ────────────────────────────────────────────────
    long_mat = build_long_matrix(land)
    _save(long_mat, "species_catch_matrix_local_year.csv")

    wide_abs = build_wide_matrix(long_mat, ["local", "year"], "sp_production_ton")
    wide_rel = build_wide_matrix(long_mat, ["local", "year"], "rel_abundance")
    _save(wide_abs, "species_catch_wide_local_year.csv")
    _save(wide_rel, "species_rel_wide_local_year.csv")

    # Exposure-level wide matrices
    for group_col, suffix in [
        ("platform_exposure_class", "platform"),
        ("mpa_exposure_class",      "mpa"),
        ("combined_exposure_class", "combined"),
    ]:
        wide_exp = build_exposure_matrix(land, group_col)
        _save(wide_exp, f"species_catch_wide_{suffix}.csv")

    # ── SIMPER-style analysis ─────────────────────────────────────────────────
    for group_col, suffix in [
        ("platform_exposure_class", "platform"),
        ("mpa_exposure_class",      "mpa"),
        ("combined_exposure_class", "combined"),
    ]:
        log.info("SIMPER analysis for %s …", group_col)
        simper_df = simper_contribution(land, group_col)
        if not simper_df.empty:
            _save(simper_df, f"species_SIMPER_{suffix}.csv")

    # ── Turnover between periods ──────────────────────────────────────────────
    log.info("Computing Bray-Curtis turnover between periods …")
    turnover = period_turnover(land)
    _save(turnover, "species_turnover_period.csv")

    log.info("06_species_composition.py complete.")


if __name__ == "__main__":
    main()
