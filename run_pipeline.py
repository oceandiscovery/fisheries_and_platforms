#!/usr/bin/env python3
"""
run_pipeline.py
===============
Master runner for the Fish catches × Oil platforms Brazil pipeline.
Executes all numbered scripts in order, stopping on first failure.

Usage:
    python run_pipeline.py               # full pipeline
    python run_pipeline.py --from 04    # re-run from script 04 onward
    python run_pipeline.py --only 10    # run only script 10
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

SCRIPTS = [
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


def script_number(name: str) -> int:
    return int(name.split("_")[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Start from this script number (e.g. 4)")
    parser.add_argument("--only", dest="only_step", type=int, default=None,
                        help="Run only this script number")
    args = parser.parse_args()

    scripts = SCRIPTS
    if args.only_step is not None:
        scripts = [s for s in scripts if script_number(s) == args.only_step]
    else:
        scripts = [s for s in scripts if script_number(s) >= args.from_step]

    if not scripts:
        log.error("No matching scripts found for the given arguments.")
        sys.exit(1)

    log.info("Pipeline: %d script(s) to run", len(scripts))
    for script in scripts:
        path = Path(__file__).parent / script
        if not path.exists():
            log.error("Script not found: %s", script)
            sys.exit(1)

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