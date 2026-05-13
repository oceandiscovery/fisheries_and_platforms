#!/usr/bin/env python3
"""
run_pipeline.py
===============
Master runner for the Fish catches × Oil platforms Brazil pipeline.
Executes all numbered scripts in order, stopping on first failure.

Two stages:
  pipeline  (01–11) — cleaning, crosswalks, spatial exposure, productivity,
                       effort, species, temporal dynamics, diagnostics, figures
  analysis  (12–19) — CPUE standardisation, AB case study, and research
                       questions P1–P9 (all n=7 primary, AB excluded)

Usage:
    python run_pipeline.py                    # full pipeline + analysis
    python run_pipeline.py --stage pipeline   # data pipeline only (01–11)
    python run_pipeline.py --stage analysis   # analysis scripts only (12–19)
    python run_pipeline.py --from 14          # re-run from script 14 onward
    python run_pipeline.py --only 15          # run only script 15
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Ensure the conda env's PROJ database is used, not the stale base-env one.
_proj_data = Path(sys.executable).parent.parent / "share" / "proj"
if _proj_data.exists():
    os.environ["PROJ_DATA"] = str(_proj_data)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_00 as cfg  # noqa: E402

cfg.setup_dirs()

logging.basicConfig(
    level=logging.INFO, format=cfg.LOG_FORMAT, datefmt=cfg.LOG_DATEFMT,
    handlers=[
        logging.FileHandler(cfg.LOGS / "run_pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("run_pipeline")

# ── Stage 1: data pipeline ────────────────────────────────────────────────────
PIPELINE_SCRIPTS = [
    "01_cleaning.py",
    "02_crosswalks.py",
    "03_spatial_exposure.py",
    "04_productivity_diversity.py",
    "05_effort_structure.py",
    "06_species_composition.py",
    "07_species_shares.py",
    "08_temporal_dynamics.py",
    "09_diagnostics.py",
    "10_figures.py",
    "11_models.py",
]

# ── Stage 2: analysis (research questions) ────────────────────────────────────
# All analysis scripts default to n=7 (excl. Areia Branca); n=8 reported as
# sensitivity where relevant (P4, P5, P9).
ANALYSIS_SCRIPTS = [
    "12_cpue_std.py",           # CPUE measure comparison (t/trip vs t/fish vs standardised)
    "13_ab_case.py",            # PA  — Areia Branca FAD case study
    "14_spatial.py",            # P1  — Spatial gradients CPUE + diversity (n=7)
    "15_cpue_dynamics.py",      # P2 + P4  — CPUE trends + MPA trajectories (n=7 + n=8 sens.)
    "16_diversity_dynamics.py", # P3a + P5 — Diversity trends + MPA trajectories (n=7 + n=8 sens.)
    "17_effort_composition.py", # P3b + P6 — Gear/boat trends + CPUE ~ composition (n=7)
    "18_within_local.py",       # P7 + P8  — Density-dep + CPUE ~ diversity (n=7)
    "19_species.py",            # P9a + P9b — SIMPER + species MK (n=7 + AB separate)
]

SCRIPTS = PIPELINE_SCRIPTS + ANALYSIS_SCRIPTS


def script_number(name: str) -> int:
    return int(name.split("_")[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Start from this script number (e.g. 14)")
    parser.add_argument("--only", dest="only_step", type=int, default=None,
                        help="Run only this script number")
    parser.add_argument("--stage", dest="stage", default="all",
                        choices=["all", "pipeline", "analysis"],
                        help="all (default) | pipeline (01–11) | analysis (12–19)")
    args = parser.parse_args()

    if args.stage == "pipeline":
        pool = PIPELINE_SCRIPTS
    elif args.stage == "analysis":
        pool = ANALYSIS_SCRIPTS
    else:
        pool = SCRIPTS

    if args.only_step is not None:
        scripts = [s for s in pool if script_number(s) == args.only_step]
    else:
        scripts = [s for s in pool if script_number(s) >= args.from_step]

    if not scripts:
        log.error("No matching scripts found for the given arguments.")
        sys.exit(1)

    log.info("Pipeline: %d script(s) to run  [stage=%s]", len(scripts), args.stage)
    stage_banner_shown = set()
    for script in scripts:
        path = Path(__file__).parent / script
        if not path.exists():
            log.error("Script not found: %s", script)
            sys.exit(1)

        # Stage banner
        num = script_number(script)
        stage_key = "pipeline" if num <= 11 else "analysis"
        if stage_key not in stage_banner_shown:
            if stage_key == "pipeline":
                log.info("━" * 60)
                log.info("STAGE 1 — DATA PIPELINE  (01–11)")
            else:
                log.info("━" * 60)
                log.info("STAGE 2 — ANALYSIS  (12–19)  [n=7 primary, excl. Areia Branca]")
            stage_banner_shown.add(stage_key)

        log.info("=" * 60)
        log.info("Running: %s", script)
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=False,
        )
        if result.returncode != 0:
            log.error("FAILED: %s (exit code %d)", script, result.returncode)
            sys.exit(result.returncode)
        log.info("DONE:    %s", script)

    log.info("=" * 60)
    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()