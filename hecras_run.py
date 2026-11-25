"""
hecras_run.py
==========================
HEC-RAS 2D rain-on-grid automation via ras-commander.

This is a corrected rewrite of ``hecras_run.py``. The original script
is left untouched; this file supersedes it.

The simulation window is never re-derived from the bytes of the DSS file.
It is resolved, in order, from:

            1. explicit --sim-start / --sim-end arguments;
            2. the sidecar JSON written by netcdf_to_dss.py (preferred);
            3. the structured HEC-DSS catalogue via the hecdss library,
               filtered on the C-part.

          If none of those are available the script stops. It never guesses,
          and it never reads the DSS as text.

Also carried over from the review of the original pipeline:
  - argparse runs with allow_abbrev=False, so a flag that does not exist is an
    error instead of silently binding to a different option (this is what made
    the orchestrator pass a .prj FILE into --prj_folder, which expects a
    DIRECTORY, and surface as a confusing "Project: NOT FOUND");
  - backups are timestamped, so the pristine originals survive a second run;
  - post-run diagnostics report mass balance instead of swallowing it.

Usage:
    python hecras_run.py
    python hecras_run.py --skip-compute
    python hecras_run.py --window-json path\\to\\out.dss.window.json
"""

import os
import re
import sys
import json
import shutil
import argparse
import time as _time
from pathlib import Path
from datetime import datetime, timedelta

from config import settings

# =============================================================================
# SETTINGS
# =============================================================================

# Defaults come from config.ini. A command-line argument overrides them.
PRJ_FOLDER  = settings.path("project", "folder")
RAS_VERSION = settings.text("project", "ras_executable")
U_FILE      = settings.path("project", "unsteady_file")
P_FILE      = settings.path("project", "plan_file")
DSS_FILE    = settings.path("rainfall", "dss")

DSS_PATHNAME = None     # None = leave the Gridded DSS Pathname line unchanged
PLAN_NUMBER  = None     # None = auto-detect from the .pNN extension
NUM_CORES    = None     # None = HEC-RAS default

# C-part identifying the precipitation records in the DSS catalogue.
PRECIP_CPART = "TP"

MONTHS = {1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
          7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"}
MONTH_NUM = {v: k for k, v in MONTHS.items()}


# =============================================================================
# RESOLVING THE SIMULATION WINDOW
# =============================================================================

def parse_dss_timestamp(text):
    """
    Parse a single HEC-DSS timestamp such as '15JUN2023:0600'.

    '2400' denotes midnight at the END of the named day, i.e. 00:00 of the
    following day. This is the exact inverse of to_dss_time() in
    netcdf_to_dss.py.
    """
    m = re.fullmatch(r"(\d{2})([A-Z]{3})(\d{4}):(\d{2})(\d{2})", text.strip())
    if not m:
        raise ValueError("Not a DSS timestamp: %r" % text)

    day, mon, year = int(m.group(1)), m.group(2), int(m.group(3))
    hour, minute = int(m.group(4)), int(m.group(5))
    if mon not in MONTH_NUM:
        raise ValueError("Unknown month %r in %r" % (mon, text))

    if hour == 24:
        return datetime(year, MONTH_NUM[mon], day, 0, minute) + timedelta(days=1)
    return datetime(year, MONTH_NUM[mon], day, hour, minute)


def window_from_json(json_path, dss_file):
    """
    Read the simulation window from the sidecar JSON written by step 1.

    This is the preferred source: the values were computed directly from the
    NetCDF time axis, so nothing has to be inferred from the DSS afterwards.
    The JSON is checked against the DSS it claims to describe, so a stale file
    left over from a previous run cannot be used by accident.
    """
    if not os.path.exists(json_path):
        return None, "sidecar JSON not found at %s" % json_path

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        return None, "sidecar JSON is unreadable (%s)" % e

    for key in ("simulation_start", "simulation_end", "dss_file"):
        if key not in meta:
            return None, "sidecar JSON lacks '%s'" % key

    claimed = os.path.normcase(os.path.abspath(meta["dss_file"]))
    actual = os.path.normcase(os.path.abspath(dss_file))
    if claimed != actual:
        return None, ("sidecar JSON describes a different DSS\n"
                      "             JSON: %s\n             using: %s"
                      % (meta["dss_file"], dss_file))

    size_now = os.path.getsize(dss_file)
    size_then = meta.get("dss_size_bytes")
    if size_then is not None and size_then != size_now:
        return None, ("the DSS has changed since the JSON was written "
                      "(%d -> %d bytes); re-run step 1" % (size_then, size_now))

    start = datetime.fromisoformat(meta["simulation_start"])
    end = datetime.fromisoformat(meta["simulation_end"])

    print("    Source          : sidecar JSON (exact, from the NetCDF time axis)")
    print("    Generated by    : %s at %s"
          % (meta.get("generated_by", "?"), meta.get("generated_at", "?")))
    print("    Records         : %s" % meta.get("n_records", "?"))
    print("    Timestep        : %s s" % meta.get("timestep_seconds", "?"))
    print("    Convention      : %s" % meta.get("accumulation_convention", "?"))
    print("    Target CRS      : EPSG:%s, cell %s m"
          % (meta.get("target_epsg", "?"), meta.get("cell_size_m", "?")))
    if meta.get("volume_difference_percent") is not None:
        print("    Volume drift    : %+.3f %%" % meta["volume_difference_percent"])

    _validate_continuity(meta)
    return (start, end), None


def _validate_continuity(meta):
    """
    Check that the record count matches the declared window and timestep.

    A gap in the middle of the series would leave the window looking correct
    while HEC-RAS silently treats the missing hours as zero rainfall.
    """
    try:
        start = datetime.fromisoformat(meta["simulation_start"])
        end = datetime.fromisoformat(meta["simulation_end"])
        step = int(meta["timestep_seconds"])
        n = int(meta["n_records"])
    except (KeyError, TypeError, ValueError):
        return

    expected = int((end - start).total_seconds() // step)
    if expected != n:
        print("    [WARNING] %d records span %s..%s at %d s, but a contiguous"
              % (n, start, end, step))
        print("              series would need %d. The DSS may have gaps; "
              "HEC-RAS would" % expected)
        print("              treat the missing steps as zero rainfall.")


def window_from_catalog(dss_file, cpart=PRECIP_CPART):
    """
    Fall back to the structured HEC-DSS catalogue.

    This reads the catalogue through the hecdss library and parses the D and E
    parts of the pathnames, filtered on the C-part. It is a genuine read of the
    file's structure, unlike scanning the raw bytes for anything date-shaped.
    """
    try:
        from hecdss import HecDss
    except ImportError:
        return None, "hecdss is not installed, so the catalogue cannot be read"

    try:
        with HecDss(dss_file) as dss:
            paths = list(dss.get_catalog())
    except Exception as e:
        return None, "could not open the DSS catalogue (%s)" % e

    starts, ends, skipped, degenerate = [], [], 0, 0
    for p in paths:
        if cpart and (p.C or "").upper() != cpart.upper():
            skipped += 1
            continue
        try:
            d = parse_dss_timestamp(p.D)
            e = parse_dss_timestamp(p.E)
        except ValueError:
            skipped += 1
            continue
        starts.append(d)
        ends.append(e)
        if d == e:
            degenerate += 1

    if not starts:
        return None, ("no '/%s/' records with parsable dates in the catalogue "
                      "(%d records skipped)" % (cpart, skipped))

    start, end = min(starts), max(ends)

    print("    Source          : DSS catalogue (structured read, C-part '%s')"
          % cpart)
    print("    Records matched : %d of %d" % (len(starts), len(paths)))

    if degenerate:
        print()
        print("    " + "!" * 62)
        print("    [ZERO-LENGTH INTERVALS] %d of %d records have D-part == "
              "E-part." % (degenerate, len(starts)))
        print("    A PER-CUM record describes an accumulation OVER an "
              "interval, so an")
        print("    interval of zero length leaves the rainfall intensity "
              "undefined and")
        print("    leaves HEC-RAS to infer a duration. This DSS was written "
              "by the old")
        print("    converter; regenerate it with netcdf_to_dss.py.")
        print("    " + "!" * 62)

    return (start, end), None


def resolve_simulation_window(args, dss_file):
    """Resolve the window from the first source that works. Never guesses."""
    print("\n  Resolving the simulation window...")

    if args.sim_start and args.sim_end:
        start = datetime.fromisoformat(args.sim_start)
        end = datetime.fromisoformat(args.sim_end)
        print("    Source          : explicit --sim-start / --sim-end")
        return start, end

    json_path = args.window_json or (dss_file + ".window.json")
    window, why = window_from_json(json_path, dss_file)
    if window:
        return window
    print("    [--] sidecar JSON unusable: %s" % why)

    if args.require_json:
        _die("--require-json was given but the sidecar JSON could not be used.")

    window, why2 = window_from_catalog(dss_file)
    if window:
        return window
    print("    [--] DSS catalogue unusable: %s" % why2)

    _die("Could not establish the simulation window.\n"
         "        Sidecar JSON : %s\n"
         "        Catalogue    : %s\n"
         "        Re-run netcdf_to_dss.py, or pass --sim-start/--sim-end "
         "explicitly.\n"
         "        This script will not scrape the DSS binary for date-shaped "
         "bytes." % (why, why2))


def _die(msg):
    print("\n  [FATAL] %s" % msg)
    sys.exit(1)


# =============================================================================
# FILE UPDATES
# =============================================================================

def _backup(path):
    """Timestamped backup, so the pristine original survives repeated runs."""
    path = Path(path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = path.with_suffix(path.suffix + ".%s.bak" % stamp)
    shutil.copy2(path, dest)
    return dest


def _safe_write(path, lines):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(lines)
        dest = _backup(path)
        print("    [OK] Backup: %s" % dest.name)
        shutil.move(str(tmp), str(path))
        print("    [OK] Saved:  %s" % path.name)
        return True
    except Exception as e:
        print("    [ERROR] Failed to save: %s" % e)
        if tmp.exists():
            tmp.unlink()
        return False


def update_unsteady_flow_file(u_path, new_dss_path, new_dss_pathname=None):
    """Point the Gridded DSS Filename (and optionally Pathname) at the new DSS."""
    u_path = Path(u_path)
    if not u_path.exists():
        print("    [ERROR] File not found: %s" % u_path)
        return False

    print("\n" + "=" * 68)
    print("  STEP 1A: Unsteady Flow file - %s" % u_path.name)
    print("=" * 68)

    with open(u_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    print("    Read %d lines" % len(lines))

    fn_key = "Met BC=Precipitation|Gridded DSS Filename="
    pn_key = "Met BC=Precipitation|Gridded DSS Pathname="
    fb_key = "DSS File="

    fn_done = pn_done = False
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(fn_key):
            print("    [OK] Line %d DSS Filename" % (i + 1))
            print("          Old: %s" % s[len(fn_key):])
            print("          New: %s" % new_dss_path)
            lines[i] = "%s%s\n" % (fn_key, new_dss_path)
            fn_done = True
        elif new_dss_pathname and s.startswith(pn_key):
            print("    [OK] Line %d DSS Pathname" % (i + 1))
            print("          Old: %s" % s[len(pn_key):])
            print("          New: %s" % new_dss_pathname)
            lines[i] = "%s%s\n" % (pn_key, new_dss_pathname)
            pn_done = True

    if not fn_done:
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith(fb_key) and not s.startswith("Met BC="):
                print("    [OK] Line %d DSS File (fallback key)" % (i + 1))
                lines[i] = "%s%s\n" % (fb_key, new_dss_path)
                fn_done = True
                break

    if not fn_done:
        print("    [ERROR] No DSS filename line found (looked for %r or %r)."
              % (fn_key, fb_key))
        return False
    if new_dss_pathname and not pn_done:
        print("    [WARNING] DSS Pathname line not found.")

    return _safe_write(u_path, lines)


def datetime_to_ras(dt, is_end=False):
    """
    HEC-RAS 'Simulation Date' format, DDMMMYYYY,HHMM.

    HEC-RAS writes the end of a simulation that lands on midnight as 2400 of
    the previous day; a start at midnight stays 0000 of its own day.
    """
    if is_end and dt.hour == 0 and dt.minute == 0:
        prev = dt - timedelta(days=1)
        return "%02d%s%04d,2400" % (prev.day, MONTHS[prev.month], prev.year)
    return "%02d%s%04d,%02d%02d" % (dt.day, MONTHS[dt.month], dt.year,
                                    dt.hour, dt.minute)


def update_plan_file(p_path, start_dt, end_dt):
    """Write the resolved window into the plan file's Simulation Date line."""
    p_path = Path(p_path)
    if not p_path.exists():
        print("    [ERROR] File not found: %s" % p_path)
        return False

    print("\n" + "=" * 68)
    print("  STEP 1B: Plan file - %s" % p_path.name)
    print("=" * 68)

    new_value = "%s,%s" % (datetime_to_ras(start_dt, False),
                           datetime_to_ras(end_dt, True))
    print("    Start: %s  ->  %s" % (start_dt, datetime_to_ras(start_dt, False)))
    print("    End:   %s  ->  %s" % (end_dt, datetime_to_ras(end_dt, True)))
    print("    New Simulation Date = %s" % new_value)

    with open(p_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    print("    Read %d lines" % len(lines))

    key = "Simulation Date="
    for i, line in enumerate(lines):
        if line.strip().startswith(key):
            print("    [OK] Line %d" % (i + 1))
            print("          Old: %s" % line.strip()[len(key):])
            print("          New: %s" % new_value)
            lines[i] = "%s%s\n" % (key, new_value)
            return _safe_write(p_path, lines)

    print("    [ERROR] '%s' not found in the plan file." % key)
    return False


# =============================================================================
# EXECUTION
# =============================================================================

def run_hec_ras(prj_folder, ras_version, plan_number, num_cores=None):
    try:
        from ras_commander import init_ras_project, RasCmdr, RasPrj
    except ImportError:
        print("    [ERROR] ras-commander is not installed. Run: "
              "pip install ras-commander")
        return False

    try:
        from ras_commander.callbacks import ConsoleCallback
        has_cb = True
    except ImportError:
        has_cb = False

    prj_folder = str(Path(prj_folder).resolve())
    if not os.path.isdir(prj_folder):
        print("    [ERROR] Project folder not found: %s" % prj_folder)
        return False

    print("\n" + "=" * 68)
    print("  STEP 2: Execute HEC-RAS (ras-commander)")
    print("=" * 68)
    print("    Project folder : %s" % prj_folder)
    print("    RAS version    : %s" % ras_version)
    print("    Plan number    : %s" % plan_number)
    print("    CPU cores      : %s" % (num_cores or "(default)"))

    try:
        print("\n    [RC] Initializing project...")
        project = RasPrj()
        init_ras_project(prj_folder, ras_version, ras_object=project)
        print("    [RC] Project initialized")

        try:
            for _, row in project.plan_df.iterrows():
                pn = row.get("plan_number", "?")
                mark = "  <-- TARGET" if str(pn) == str(plan_number) else ""
                print("         Plan %s: %s%s" % (pn, row.get("plan_title", "?"), mark))
        except Exception:
            print("    [RC] Could not list plans (non-critical)")

        kwargs = {"plan_number": plan_number, "ras_object": project,
                  "clear_geompre": True}
        if num_cores is not None:
            kwargs["num_cores"] = num_cores
            try:
                from ras_commander import RasPlan
                RasPlan.set_num_cores(plan_number, num_cores, ras_object=project)
                print("    [RC] CPU cores set to %d" % num_cores)
            except Exception as e:
                print("    [WARNING] Could not set CPU cores: %s" % e)
        if has_cb:
            kwargs["stream_callback"] = ConsoleCallback(verbose=True)

        print("\n    [RC] Computing plan %s - this may take several minutes...\n"
              % plan_number)
        t0 = _time.time()
        ok = RasCmdr.compute_plan(**kwargs)
        elapsed = _time.time() - t0

        print()
        if ok:
            print("    [RC] SUCCESS in %s" % _fmt(elapsed))
            post_run_diagnostics(plan_number, project)
        else:
            print("    [RC] FAILED after %s" % _fmt(elapsed))
        return bool(ok)

    except Exception as e:
        print("\n    [ERROR] ras-commander failure: %s" % e)
        import traceback
        traceback.print_exc()
        return False


def post_run_diagnostics(plan_number, project):
    """
    Report runtime, compute messages and mass balance.

    The original swallowed every exception here with `except Exception: pass`,
    which hid exactly the volume-accounting and compute-log problems worth
    knowing about. Failures are now named.
    """
    try:
        from ras_commander import HdfResultsPlan
    except ImportError:
        print("\n    [--] HdfResultsPlan unavailable; skipping diagnostics.")
        return

    print("\n    " + "-" * 56)
    print("    POST-RUN DIAGNOSTICS (HDF5)")
    print("    " + "-" * 56)

    try:
        df = HdfResultsPlan.get_runtime_data(plan_number, ras_object=project)
        if df is not None and not df.empty:
            sp = df.get("Complete Process Speed (hr/hr)")
            if sp is not None:
                v = sp.iloc[0] if hasattr(sp, "iloc") else sp
                print("      Simulation speed   : %.0fx realtime" % float(v))
    except Exception as e:
        print("      [--] runtime data unavailable: %s" % e)

    try:
        msgs = HdfResultsPlan.get_compute_messages(plan_number, ras_object=project)
        if msgs:
            blob = str(msgs).upper()
            hits = [k for k in ("ERROR", "FAILED", "UNSTABLE", "WARNING")
                    if k in blob]
            if hits:
                print("      Compute log        : %s PRESENT - inspect the log"
                      % ", ".join(hits))
            else:
                print("      Compute log        : clean")
    except Exception as e:
        print("      [--] compute messages unavailable: %s" % e)

    try:
        vol = HdfResultsPlan.get_volume_accounting(plan_number, ras_object=project)
        if vol is not None and not vol.empty:
            print("      Volume accounting  :")
            for col in vol.columns:
                try:
                    print("        %-38s %s" % (col, vol.iloc[0][col]))
                except Exception:
                    pass
        else:
            print("      [--] volume accounting is empty")
    except Exception as e:
        print("      [--] volume accounting unavailable: %s" % e)


def _fmt(sec):
    if sec < 60:
        return "%.1f s" % sec
    if sec < 3600:
        return "%d min %d s" % (sec // 60, sec % 60)
    return "%d h %d min" % (sec // 3600, (sec % 3600) // 60)


def detect_plan_number(p_file):
    ext = Path(p_file).suffix
    return ext[2:] if ext.startswith(".p") and len(ext) > 2 else None


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="HEC-RAS 2D automation (corrected)",
        allow_abbrev=False)   # a wrong flag is an error, not a silent rebind
    ap.add_argument("--prj_folder", default=PRJ_FOLDER,
                    help="HEC-RAS project FOLDER (the directory holding .prj)")
    ap.add_argument("--ras_version", default=RAS_VERSION)
    ap.add_argument("--u_file", default=U_FILE)
    ap.add_argument("--p_file", default=P_FILE)
    ap.add_argument("--dss_file", default=DSS_FILE)
    ap.add_argument("--window_json", default=None,
                    help="Sidecar JSON from netcdf_to_dss.py "
                         "(default: <dss_file>.window.json)")
    ap.add_argument("--sim-start", default=None, help="ISO datetime override")
    ap.add_argument("--sim-end", default=None, help="ISO datetime override")
    ap.add_argument("--require-json", action="store_true",
                    help="Fail rather than fall back to the DSS catalogue.")
    ap.add_argument("--plan_number", default=PLAN_NUMBER)
    ap.add_argument("--num_cores", type=int, default=NUM_CORES)
    ap.add_argument("--skip-compute", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve and report the window; change nothing.")
    args = ap.parse_args()

    for value, flag in ((args.prj_folder, "--prj_folder"),
                        (args.u_file, "--u_file"),
                        (args.p_file, "--p_file"),
                        (args.dss_file, "--dss_file")):
        if value is None:
            print("\n  [ERROR] %s was not given and is blank in config.ini."
                  % flag)
            return 1

    plan_number = args.plan_number or detect_plan_number(args.p_file)
    if plan_number is None:
        _die("Could not detect the plan number from %r. Use --plan_number."
             % args.p_file)

    print()
    print("#" * 68)
    print("#   HEC-RAS 2D Automation (Rain-on-Grid) - corrected pipeline, v2")
    print("#   Engine: ras-commander")
    print("#" * 68)
    print("\n  Settings:")
    print("    Project folder : %s" % args.prj_folder)
    print("    RAS version    : %s" % args.ras_version)
    print("    U-file         : %s" % args.u_file)
    print("    P-file         : %s" % args.p_file)
    print("    DSS            : %s" % args.dss_file)
    print("    Plan #         : %s" % plan_number)
    print("    Cores          : %s" % (args.num_cores or "(default)"))
    print("    Compute        : %s" % ("SKIP" if args.skip_compute else "YES"))
    if args.dry_run:
        print("    MODE           : DRY RUN")

    # --- Pre-flight -------------------------------------------------------
    print("\n  Checking files...")
    ok = True
    if os.path.isdir(args.prj_folder):
        print("    [OK] Project folder: %s" % args.prj_folder)
        prjs = list(Path(args.prj_folder).glob("*.prj"))
        print("         .prj files found: %s"
              % (", ".join(p.name for p in prjs) if prjs else "NONE"))
        if not prjs:
            ok = False
    else:
        print("    [ERROR] Project folder: NOT A DIRECTORY - %s" % args.prj_folder)
        if os.path.isfile(args.prj_folder):
            print("            That path is a FILE. --prj_folder wants the "
                  "directory that")
            print("            contains the .prj, e.g. %s"
                  % os.path.dirname(args.prj_folder))
        ok = False

    for label, path in (("U-file", args.u_file), ("P-file", args.p_file),
                        ("DSS", args.dss_file)):
        if os.path.exists(path):
            print("    [OK] %-6s: %s (%.1f KB)"
                  % (label, path, os.path.getsize(path) / 1024))
        else:
            print("    [ERROR] %-6s: NOT FOUND - %s" % (label, path))
            if label == "DSS":
                print("            Run netcdf_to_dss.py first.")
            ok = False

    if not ok:
        print("\n  [ERROR] Missing inputs. Fix the paths and run again.")
        sys.exit(1)

    # --- Simulation window -------------------------------------------------
    sim_start, sim_end = resolve_simulation_window(args, args.dss_file)
    print("\n    SIMULATION WINDOW: %s  ->  %s" % (sim_start, sim_end))
    print("    Duration        : %s" % (sim_end - sim_start))

    if sim_end <= sim_start:
        _die("The resolved window ends at or before it starts (%s -> %s)."
             % (sim_start, sim_end))

    if args.dry_run:
        print("\n  [DRY RUN] Window resolved. No files changed.")
        return

    if not update_unsteady_flow_file(args.u_file, args.dss_file, DSS_PATHNAME):
        _die("Failed to update the Unsteady Flow file.")
    if not update_plan_file(args.p_file, sim_start, sim_end):
        _die("Failed to update the Plan file.")

    print("\n  [OK] STEP 1 complete - input files updated")

    if args.skip_compute:
        print("\n  [INFO] --skip-compute: HEC-RAS not executed.")
        return

    success = run_hec_ras(args.prj_folder, args.ras_version, plan_number,
                          args.num_cores)

    print()
    print("#" * 68)
    if success:
        print("#   COMPLETED SUCCESSFULLY - check the results in RAS Mapper.")
    else:
        print("#   EXECUTION FAILED. Check:")
        print("#   - does the DSS grid cover the 2D flow area?")
        print("#   - does the rainfall window cover the simulation window?")
        print("#   - are the boundary conditions correct?")
        print("#   - see the HEC-RAS computation log")
    print("#" * 68)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
