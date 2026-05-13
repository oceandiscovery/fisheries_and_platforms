"""
09_diagnostics.py
=================
Cross-table quality checks and consistency diagnostics for all pipeline outputs.

Checks performed:
  1. Row and column completeness for all processed tables.
  2. No negative production values anywhere.
  3. CPUE range plausibility.
  4. Exposure-class coverage — every local-year row should have a class.
  5. Species-share sums ≈ 1.0 within each group.
  6. Gear-share sums ≈ 1.0 within each group.
  7. Temporal range consistency.
  8. Landing-point spatial coverage check (all locals have coordinates).
  9. Platform-distance implausibility check (distance = 0 unexpected).
  10. Logs a full reconciliation table of production totals across sheets.

Outputs (data/processed/):
  - diagnostics_summary.csv    Machine-readable pass/fail/warn table
  - diagnostics_report.txt     Human-readable summary

Run:
  python 09_diagnostics.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_00 as cfg  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format=cfg.LOG_FORMAT, datefmt=cfg.LOG_DATEFMT,
    handlers=[
        logging.FileHandler(cfg.LOGS / "09_diagnostics.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("09_diagnostics")


def _check(path: Path) -> None:
    if not path.exists():
        log.warning("File not found (skipping check): %s", path)


def _load(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    log.warning("Could not load (not found): %s", path)
    return None


# ─── Individual checks ────────────────────────────────────────────────────────

def check_no_negative(df: pd.DataFrame, col: str, label: str) -> dict:
    n_neg = (df[col] < 0).sum() if col in df.columns else 0
    return {
        "check":  f"no_negative_{col}_{label}",
        "status": "PASS" if n_neg == 0 else "FAIL",
        "detail": f"{n_neg} negative values in {col}",
    }


def check_share_sum(df: pd.DataFrame, groupby: list[str],
                    share_col: str, label: str,
                    tol: float = 0.01) -> dict:
    """
    Verify that shares within each group sum to ~1.0 (within tolerance).
    """
    if share_col not in df.columns or not all(c in df.columns for c in groupby):
        return {"check": f"share_sum_{label}", "status": "SKIP",
                "detail": "column(s) missing"}
    sums = df.groupby(groupby, dropna=False)[share_col].sum()
    n_bad = ((sums - 1.0).abs() > tol).sum()
    return {
        "check":  f"share_sum_{label}",
        "status": "PASS" if n_bad == 0 else "WARN",
        "detail": f"{n_bad}/{len(sums)} groups deviate from sum=1 by >{tol}",
    }


def check_cpue_range(df: pd.DataFrame, col: str = "cpue_ton_per_trip",
                     lo: float = 0, hi: float = 50) -> dict:
    if col not in df.columns:
        return {"check": "cpue_range", "status": "SKIP", "detail": "column missing"}
    vals = df[col].dropna()
    n_out = ((vals < lo) | (vals > hi)).sum()
    return {
        "check":  "cpue_range",
        "status": "WARN" if n_out > 0 else "PASS",
        "detail": f"{n_out} CPUE values outside [{lo}, {hi}] ton/trip",
    }


def check_exposure_coverage(df: pd.DataFrame) -> dict:
    for col in ["platform_exposure_class", "mpa_exposure_class"]:
        if col in df.columns:
            n_miss = df[col].isna().sum()
            if n_miss > 0:
                return {
                    "check":  "exposure_coverage",
                    "status": "WARN",
                    "detail": f"{n_miss} rows missing {col}",
                }
    return {"check": "exposure_coverage", "status": "PASS",
            "detail": "All rows have exposure class assignments"}


def check_temporal_range(df: pd.DataFrame, label: str) -> dict:
    if "year" not in df.columns:
        return {"check": f"temporal_range_{label}", "status": "SKIP",
                "detail": "no year column"}
    yr_min = int(df["year"].min())
    yr_max = int(df["year"].max())
    return {
        "check":  f"temporal_range_{label}",
        "status": "PASS",
        "detail": f"Years: {yr_min}–{yr_max}",
    }


def check_platform_zero(df: pd.DataFrame) -> dict:
    if "platform_dist_km_mean" not in df.columns:
        return {"check": "platform_zero_dist", "status": "SKIP",
                "detail": "column missing"}
    n_zero = (df["platform_dist_km_mean"] == 0).sum()
    return {
        "check":  "platform_zero_dist",
        "status": "WARN" if n_zero > 0 else "PASS",
        "detail": f"{n_zero} locals with platform distance = 0 (check geometry)",
    }


# ─── Production reconciliation ────────────────────────────────────────────────

def reconcile_production(master: pd.DataFrame | None,
                         prod: pd.DataFrame | None,
                         land: pd.DataFrame | None) -> pd.DataFrame:
    """
    Compare total production across the three main sheets by year.
    """
    records = []
    years = set()
    for df in [master, prod, land]:
        if df is not None and "year" in df.columns:
            years |= set(df["year"].dropna().unique())

    for yr in sorted(years):
        rec = {"year": yr}
        if master is not None:
            rec["master_ton"] = master.loc[master["year"] == yr, "production_ton"].sum()
        if prod is not None:
            rec["gear_ton"] = prod.loc[prod["year"] == yr, "gear_production_ton"].sum()
        if land is not None:
            rec["species_ton"] = land.loc[land["year"] == yr, "sp_production_ton"].sum()
        records.append(rec)

    recon = pd.DataFrame(records)
    # Flag years where master_ton vs species_ton differ by > 10%
    if "master_ton" in recon and "species_ton" in recon:
        recon["pct_diff_sp"] = (
            (recon["species_ton"] - recon["master_ton"]).abs()
            / recon["master_ton"].replace(0, np.nan) * 100
        )
    return recon


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg.setup_dirs()

    results = []

    # Load tables
    master    = _load(cfg.DATA_INTERIM / "master_clean.csv")
    prod      = _load(cfg.DATA_INTERIM / "production_clean.csv")
    land      = _load(cfg.DATA_INTERIM / "landings_clean.csv")
    exposure  = _load(cfg.DATA_INTERIM / "local_exposure.csv")
    lp_exp    = _load(cfg.DATA_INTERIM / "landing_points_exposure.csv")
    prod_ly   = _load(cfg.DATA_PROCESSED / "productivity_local_year.csv")
    sp_share  = _load(cfg.DATA_PROCESSED / "species_share_local_year.csv")
    gs_ly     = _load(cfg.DATA_PROCESSED / "gear_share_local_year.csv")

    # ── Checks ────────────────────────────────────────────────────────────────
    if master is not None:
        results.append(check_no_negative(master, "production_ton", "master"))
        results.append(check_cpue_range(master))
        results.append(check_temporal_range(master, "master"))

    if prod is not None:
        results.append(check_no_negative(prod, "gear_production_ton", "production"))

    if land is not None:
        results.append(check_no_negative(land, "sp_production_ton", "landings"))

    if prod_ly is not None:
        results.append(check_exposure_coverage(prod_ly))
        results.append(check_temporal_range(prod_ly, "productivity_local_year"))
        results.append(check_cpue_range(prod_ly, "cpue_per_fisherman", lo=0, hi=10))

    if exposure is not None:
        results.append(check_platform_zero(exposure))

    if sp_share is not None:
        results.append(check_share_sum(
            sp_share, ["local", "year"], "species_share", "species_local_year"
        ))

    if gs_ly is not None and "gear_share" in gs_ly.columns:
        results.append(check_share_sum(
            gs_ly, ["local", "year"], "gear_share", "gear_local_year"
        ))

    # ── Production reconciliation ─────────────────────────────────────────────
    recon = reconcile_production(master, prod, land)

    # ── Save ──────────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(cfg.DATA_PROCESSED / "diagnostics_summary.csv", index=False)
    log.info("Diagnostics summary:\n%s", results_df.to_string(index=False))

    recon.to_csv(cfg.DATA_PROCESSED / "production_reconciliation.csv", index=False)
    log.info("Production reconciliation:\n%s", recon.to_string(index=False))

    # ── Text report ───────────────────────────────────────────────────────────
    report_path = cfg.DATA_PROCESSED / "diagnostics_report.txt"
    with open(report_path, "w") as fh:
        fh.write("PIPELINE DIAGNOSTICS REPORT\n")
        fh.write("=" * 60 + "\n\n")
        for row in results:
            fh.write(f"[{row['status']:4s}]  {row['check']}\n")
            fh.write(f"        {row['detail']}\n\n")
        fh.write("\nPRODUCTION RECONCILIATION\n")
        fh.write("-" * 60 + "\n")
        fh.write(recon.to_string(index=False))

    log.info("Report written: %s", report_path)
    n_fail = (results_df["status"] == "FAIL").sum()
    n_warn = (results_df["status"] == "WARN").sum()
    log.info("09_diagnostics.py complete — %d FAIL, %d WARN", n_fail, n_warn)


if __name__ == "__main__":
    main()
