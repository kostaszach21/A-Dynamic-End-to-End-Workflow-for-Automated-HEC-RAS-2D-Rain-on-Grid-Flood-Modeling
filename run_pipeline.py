"""Run the whole workflow: rainfall to HEC-DSS, HEC-RAS, maps.

Fill in config.ini, then run this. Nothing else needs editing.

    python run_pipeline.py
    python run_pipeline.py --skip-compute     prepare inputs, do not run HEC-RAS
    python run_pipeline.py --from-step 3      results already exist, just map
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from config import TEMPLATE_FILE, ConfigError, settings

HERE = Path(__file__).parent.resolve()
CONVERT = HERE / "netcdf_to_dss.py"
VERIFY = HERE / "verify_rainfall_grid.py"
COMPUTE = HERE / "hecras_run.py"
MAPS = HERE / "flood_maps.py"

# Values that must be present before anything can run.
REQUIRED = [
    ("project", "folder"),
    ("project", "unsteady_file"),
    ("project", "plan_file"),
    ("project", "results_folder"),
    ("rainfall", "netcdf"),
    ("rainfall", "dss"),
    ("grid", "target_epsg"),
    ("output", "maps_folder"),
]


# Packages each step needs, and what fails without them.
NEEDED = [
    ("numpy", "everything"),
    ("scipy", "resampling"),
    ("pyproj", "reprojection"),
    ("hecdss", "writing the DSS file"),
    ("tifffile", "reading the result rasters"),
    ("matplotlib", "drawing the maps"),
    ("ras_commander", "running HEC-RAS"),
]

OPTIONAL = [("netCDF4", "reads every NetCDF format; without it only NetCDF-4")]

# The converter needs one of these. netCDF4 is preferred; h5py is the
# fallback and reads NetCDF-4 only. Neither present is a hard failure.
READERS = [("netCDF4", "every NetCDF format"), ("h5py", "NetCDF-4 only")]


def check_environment() -> bool:
    """Report the interpreter and any missing package. True if usable."""
    import importlib.util

    print("\n  Python  %s" % sys.executable)

    missing = [(m, why) for m, why in NEEDED
               if importlib.util.find_spec(m) is None]
    absent = [(m, why) for m, why in OPTIONAL
              if importlib.util.find_spec(m) is None]

    if all(importlib.util.find_spec(m) is None for m, _ in READERS):
        missing.append(("netCDF4 or h5py", "reading the NetCDF input"))

    for name, why in absent:
        print("  [--] %-14s missing, %s" % (name, why))

    if not missing:
        print("  [OK] all required packages present")
        return True

    print("\n[ERROR] This interpreter is missing:")
    for name, why in missing:
        print("    %-16s needed for %s" % (name, why))
    print("\n  Either activate the environment that has them, or install:")
    print("    %s -m pip install -r requirements.txt" % sys.executable)
    return False


def run_step(number: int, name: str, script: Path, args: Sequence[str],
             optional: bool = False) -> bool:
    """Run one child script, streaming its output."""
    print()
    print("=" * 78)
    print("  STEP %d  %s" % (number, name))
    print("=" * 78)
    print()

    if not script.exists():
        print("[ERROR] Missing script: %s" % script)
        if optional:
            return False
        sys.exit(1)

    try:
        subprocess.run([sys.executable, str(script), *args], check=True)
    except subprocess.CalledProcessError as exc:
        if optional:
            print("\n[WARNING] Step %d failed (exit %d). Continuing."
                  % (number, exc.returncode))
            return False
        print("\n[ERROR] Step %d failed with exit code %d. Stopping."
              % (number, exc.returncode))
        sys.exit(exc.returncode)

    print("\n[OK] Step %d done." % number)
    return True


def newest_raster(folder: Path) -> float | None:
    """Modification time of the newest GeoTIFF, or None."""
    if not folder.is_dir():
        return None
    rasters = list(folder.glob("*.tif"))
    return max(r.stat().st_mtime for r in rasters) if rasters else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the rain-on-grid workflow end to end.",
        allow_abbrev=False)
    parser.add_argument("--from-step", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--skip-compute", action="store_true",
                        help="Prepare the HEC-RAS inputs but do not run it.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the rainfall grid verification figure.")
    parser.add_argument("--allow-stale-results", action="store_true",
                        help="Map results even if they predate this run.")
    parser.add_argument("--label", default="",
                        help="Suffix added to the map file names.")
    args = parser.parse_args(argv)

    print()
    print("#" * 78)
    print("#  RAIN-ON-GRID WORKFLOW   %s"
          % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("#" * 78)

    # --- environment ------------------------------------------------------
    if not check_environment():
        return 1

    # --- configuration ----------------------------------------------------
    if not settings.exists:
        print("\n[ERROR] No config.ini in %s" % settings.file.parent)
        if TEMPLATE_FILE.exists():
            print("\n  Copy the template, then fill it in:")
            print("      copy %s %s      (Windows)"
                  % (TEMPLATE_FILE.name, settings.file.name))
            print("      cp   %s %s      (macOS, Linux)"
                  % (TEMPLATE_FILE.name, settings.file.name))
        else:
            print("\n  %s is missing from this folder too."
                  % TEMPLATE_FILE.name)
        return 1

    blanks = settings.missing(*REQUIRED)
    if blanks:
        print("\n[ERROR] Fill these in %s before running:" % settings.file)
        for item in blanks:
            print("    %s" % item)
        return 1

    try:
        nc_file = settings.require("rainfall", "netcdf")
        dss_file = settings.require("rainfall", "dss")
        prj_folder = settings.require("project", "folder")
        u_file = settings.require("project", "unsteady_file")
        p_file = settings.require("project", "plan_file")
        results_folder = settings.require("project", "results_folder")
        maps_folder = settings.require("output", "maps_folder")
        target_epsg = settings.integer("grid", "target_epsg")
        cell_size = settings.number("grid", "cell_size_m") or 1000.0
        resample = settings.text("grid", "resample") or "linear"
        convention = settings.text("rainfall", "interval_convention") or "auto"
        accumulation = settings.text("rainfall", "accumulation")
        drop_first = settings.flag("rainfall", "drop_first_step")
        dem = settings.path("terrain", "dem")
        ras_exe = settings.text("project", "ras_executable")
    except ConfigError as exc:
        print("\n[ERROR] %s" % exc)
        return 1

    json_file = Path(str(dss_file) + ".window.json")

    print("\n  Settings from %s" % settings.file.name)
    print(settings.report())

    # --- pre-flight -------------------------------------------------------
    print("\n  Checking inputs")
    ok = True
    for label, path, kind in (("rainfall NetCDF", nc_file, "file"),
                              ("project folder", prj_folder, "dir"),
                              ("unsteady file", u_file, "file"),
                              ("plan file", p_file, "file")):
        good = path.is_dir() if kind == "dir" else path.is_file()
        print("    [%s] %-16s %s" % ("OK" if good else "--", label, path))
        ok &= good
    if prj_folder.is_dir() and not list(prj_folder.glob("*.prj")):
        print("    [--] no .prj file inside the project folder")
        ok = False
    if not ok:
        print("\n[ERROR] Fix the paths above and run again.")
        return 1

    rasters_before = newest_raster(results_folder)

    # --- step 1: rainfall to DSS -----------------------------------------
    if args.from_step <= 1:
        step = ["--nc-file", str(nc_file),
                "--dss-file", str(dss_file),
                "--json-file", str(json_file),
                "--target-epsg", str(target_epsg),
                "--cell-size", str(cell_size),
                "--resample", resample,
                "--interval-convention", convention]
        if accumulation:
            # Left blank, the converter refuses and says what to declare.
            step += ["--accumulation", accumulation]
        if drop_first:
            step.append("--drop-first-step")
        run_step(1, "Rainfall to HEC-DSS", CONVERT, step)

    # --- step 2: grid verification figure --------------------------------
    if args.from_step <= 2 and not args.no_verify:
        maps_folder.mkdir(parents=True, exist_ok=True)
        step = ["--json_file", str(json_file),
                "--nc_file", str(nc_file),
                "--out", str(maps_folder / "rainfall_grid_verification.png")]
        if dem:
            step += ["--dem_file", str(dem)]
        run_step(2, "Verify rainfall grid alignment", VERIFY, step, optional=True)

    # --- step 3: update inputs and compute -------------------------------
    if args.from_step <= 3:
        step = ["--prj_folder", str(prj_folder),
                "--u_file", str(u_file),
                "--p_file", str(p_file),
                "--dss_file", str(dss_file),
                "--window_json", str(json_file),
                "--require-json"]
        if ras_exe:
            step += ["--ras_version", ras_exe]
        if args.skip_compute:
            step.append("--skip-compute")
        run_step(3, "Update HEC-RAS inputs and compute", COMPUTE, step)

        if args.skip_compute:
            print("\n  Stopped before the maps (--skip-compute).")
            return 0

    # --- freshness --------------------------------------------------------
    if args.from_step <= 3:
        rasters_after = newest_raster(results_folder)
        if rasters_after is None:
            print("\n[ERROR] No GeoTIFFs in %s. Check that RAS Mapper is set "
                  "to store maps to disk." % results_folder)
            return 1
        if rasters_before is not None and rasters_after <= rasters_before:
            stamp = datetime.fromtimestamp(rasters_after)
            print("\n[ERROR] The result rasters have not changed (newest: %s)."
                  % stamp.strftime("%Y-%m-%d %H:%M:%S"))
            print("        The run reported success but the stored maps were "
                  "not rewritten,")
            print("        so these would be a previous run's results.")
            if not args.allow_stale_results:
                return 1
            print("[WARNING] --allow-stale-results: continuing anyway.")

    # --- step 4: maps -----------------------------------------------------
    if args.from_step <= 4:
        step = ["--plan_folder", str(results_folder),
                "--output_dir", str(maps_folder)]
        if dem:
            step += ["--dem_file", str(dem)]
        if args.label:
            step += ["--label", args.label]
        run_step(4, "Render maps", MAPS, step)

    print()
    print("#" * 78)
    print("#  DONE")
    print("#  maps       %s" % maps_folder)
    print("#  provenance %s" % json_file)
    print("#" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
