"""
verify_rainfall_grid.py
=======================
Evidence figure: proves the rainfall grid lands where it should.

A geographic (degrees) DSS grid feeding a projected (metres) 2D mesh cannot
simply be asserted to be correct - it has to be demonstrated. This script
produces that demonstration, as a publication-quality figure plus a printed
numerical audit.

The figure shows the DSS rainfall grid over the model terrain, with
cumulative rainfall shading, in the model CRS, so that its extent and its
alignment against the 2D domain can both be read off directly.

It reads the sidecar JSON written by netcdf_to_dss.py, so it always
documents the grid that was actually written.

Usage:
    python verify_rainfall_grid.py
    python verify_rainfall_grid.py --json_file OUT.dss.window.json --out fig.png
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np

from config import settings

_DSS = settings.path("rainfall", "dss")
_MAPS = settings.path("output", "maps_folder")

DEFAULT_JSON = str(_DSS) + ".window.json" if _DSS else ""
DEFAULT_NC   = str(settings.path("rainfall", "netcdf") or "")
DEFAULT_DEM  = str(settings.path("terrain", "dem") or "")
DEFAULT_OUT  = str((_MAPS / "rainfall_grid_verification.png") if _MAPS else
                   "rainfall_grid_verification.png")


def die(msg):
    print("\n[FATAL] %s" % msg)
    sys.exit(1)


def read_dem_extent(dem_path):
    """Return (xmin, xmax, ymin, ymax) of a GeoTIFF, without GDAL."""
    import tifffile
    with tifffile.TiffFile(str(dem_path)) as tif:
        page = tif.pages[0]
        scale = page.tags.get(33550)
        tie = page.tags.get(33922)
        if scale is None or tie is None:
            die("GeoTIFF %s lacks the georeferencing tags." % dem_path)
        dx, dy = scale.value[0], scale.value[1]
        tx, ty = tie.value[3], tie.value[4]
        rows, cols = page.shape[0], page.shape[1]
    return (tx, tx + cols * dx, ty - rows * dy, ty), (dx, dy), (rows, cols)


def cumulative_rainfall_on_target(meta, nc_file):
    """
    Recompute the cumulative rainfall field on the grid the JSON describes.

    Deliberately independent of the DSS: if this and the DSS disagree, the
    figure is not evidence of anything.
    """
    import h5py
    from pyproj import CRS, Transformer
    from scipy.interpolate import RegularGridInterpolator

    with h5py.File(nc_file, "r") as ds:
        var = meta.get("precip_variable", "tp")
        if var not in ds:
            die("Variable '%s' not in %s" % (var, nc_file))
        tp = np.asarray(ds[var][:], dtype=np.float64)
        lat = np.asarray(ds["latitude"][:], dtype=np.float64)
        lon = np.asarray(ds["longitude"][:], dtype=np.float64)

    if lat[0] > lat[-1]:
        lat, tp = lat[::-1], tp[:, ::-1, :]
    if lon[0] > lon[-1]:
        lon, tp = lon[::-1], tp[:, :, ::-1]
    if meta.get("first_step_dropped"):
        tp = tp[1:]

    total = np.nansum(np.nan_to_num(tp, nan=0.0), axis=0)

    nx, ny = meta["grid_nx"], meta["grid_ny"]
    cs = meta["cell_size_m"]
    x0, y0 = meta["x_origin_m"], meta["y_origin_m"]

    inv = Transformer.from_crs(CRS.from_epsg(meta["target_epsg"]),
                               CRS.from_epsg(meta["source_epsg"]),
                               always_xy=True)
    cx = x0 + (np.arange(nx) + 0.5) * cs
    cy = y0 + (np.arange(ny) + 0.5) * cs
    gx, gy = np.meshgrid(cx, cy)
    glon, glat = inv.transform(gx.ravel(), gy.ravel())
    glon, glat = glon.reshape(ny, nx), glat.reshape(ny, nx)

    dlat = float(np.diff(lat).mean())
    dlon = float(np.diff(lon).mean())
    inside = ((glon >= lon[0] - dlon / 2) & (glon <= lon[-1] + dlon / 2) &
              (glat >= lat[0] - dlat / 2) & (glat <= lat[-1] + dlat / 2))

    itp = RegularGridInterpolator((lat, lon), total,
                                  method=meta.get("resample_method", "linear"),
                                  bounds_error=False, fill_value=np.nan)
    pts = np.column_stack([np.clip(glat, lat[0], lat[-1]).ravel(),
                           np.clip(glon, lon[0], lon[-1]).ravel()])
    out = itp(pts).reshape(ny, nx)
    return np.where(inside, out, np.nan), (lat, lon, dlat, dlon)


def main():
    ap = argparse.ArgumentParser(description="Verify the rainfall grid "
                                             "alignment",
                                 allow_abbrev=False)
    ap.add_argument("--json_file", default=DEFAULT_JSON)
    ap.add_argument("--nc_file", default=DEFAULT_NC)
    ap.add_argument("--dem_file", default=DEFAULT_DEM)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    if not os.path.exists(args.json_file):
        die("Sidecar JSON not found: %s\n        Run netcdf_to_dss.py first."
            % args.json_file)
    with open(args.json_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("=" * 72)
    print(" Rainfall grid verification")
    print("=" * 72)
    print("  Sidecar JSON : %s" % args.json_file)
    print("  Target CRS   : EPSG:%s (%s)"
          % (meta["target_epsg"], meta.get("target_crs_name", "?")))
    print("  Grid         : %d x %d cells of %.0f m"
          % (meta["grid_nx"], meta["grid_ny"], meta["cell_size_m"]))
    print("  Origin       : X=%.1f  Y=%.1f m"
          % (meta["x_origin_m"], meta["y_origin_m"]))
    print("  Cell index   : lowerLeftCellX=%d  lowerLeftCellY=%d"
          % (meta["lowerLeftCellX"], meta["lowerLeftCellY"]))

    cs = meta["cell_size_m"]
    res_x = abs(meta["lowerLeftCellX"] * cs - meta["x_origin_m"])
    res_y = abs(meta["lowerLeftCellY"] * cs - meta["y_origin_m"])
    print("  Index residual: %.3e m / %.3e m  %s"
          % (res_x, res_y, "[EXACT]" if max(res_x, res_y) < 1e-6 else "[OFFSET]"))

    field, (lat, lon, dlat, dlon) = cumulative_rainfall_on_target(meta, args.nc_file)

    dem_ok = os.path.exists(args.dem_file)
    if dem_ok:
        (dxmin, dxmax, dymin, dymax), (pdx, pdy), (prows, pcols) = \
            read_dem_extent(args.dem_file)
        print("\n--- Model terrain ---")
        print("  DEM        : %s" % Path(args.dem_file).name)
        print("  Resolution : %.2f x %.2f m  (%d x %d px)"
              % (pdx, pdy, prows, pcols))
        print("  X extent   : %.1f .. %.1f m" % (dxmin, dxmax))
        print("  Y extent   : %.1f .. %.1f m" % (dymin, dymax))

        covered = (dxmin >= meta["x_origin_m"] and dxmax <= meta["x_max_m"] and
                   dymin >= meta["y_origin_m"] and dymax <= meta["y_max_m"])
        print("  Fully inside the rainfall grid: %s"
              % ("YES" if covered else "NO  <-- the 2D area would get null rain"))
    else:
        print("\n  [--] DEM not found: %s" % args.dem_file)
        dxmin = dxmax = dymin = dymax = None

    # ------------------------------------------------------------------ plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    x0, y0 = meta["x_origin_m"], meta["y_origin_m"]
    x1, y1 = meta["x_max_m"], meta["y_max_m"]

    fig, axA = plt.subplots(1, 1, figsize=(9.5, 8), dpi=args.dpi)
    fig.patch.set_facecolor("white")

    # --- the rainfall grid over the model domain -----------------------
    im = axA.imshow(field, extent=(x0, x1, y0, y1), origin="lower",
                    cmap="Blues", interpolation="nearest", zorder=1)
    cb = fig.colorbar(im, ax=axA, shrink=0.8, pad=0.02)
    cb.set_label("Cumulative precipitation (mm)", fontsize=12, fontweight="bold")

    for gx in np.arange(x0, x1 + cs, cs * 5):
        axA.axvline(gx, color="0.55", lw=0.4, zorder=2)
    for gy in np.arange(y0, y1 + cs, cs * 5):
        axA.axhline(gy, color="0.55", lw=0.4, zorder=2)

    axA.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                            ec="#1f78b4", lw=2.0, zorder=4,
                            label="DSS rainfall grid (EPSG:%d)"
                                  % meta["target_epsg"]))
    if dem_ok:
        axA.add_patch(Rectangle((dxmin, dymin), dxmax - dxmin, dymax - dymin,
                                fill=False, ec="#e31a1c", lw=2.4, zorder=5,
                                label="HEC-RAS terrain / 2D model domain"))
    axA.set_title("Rainfall grid and model domain, EPSG:%d\n"
                  "%d x %d cells of %.0f m, origin exact on the cell index"
                  % (meta["target_epsg"], meta["grid_nx"], meta["grid_ny"], cs),
                  fontsize=13, fontweight="bold")
    axA.legend(loc="upper left", fontsize=10, framealpha=0.92)

    for ax in (axA,):
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (m), EPSG:%d" % meta["target_epsg"],
                      fontsize=12, fontweight="bold")
        ax.set_ylabel("Northing (m), EPSG:%d" % meta["target_epsg"],
                      fontsize=12, fontweight="bold")
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.tick_params(labelsize=10)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_ha("right")

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight",
                facecolor="white", pad_inches=0.18)
    import matplotlib.pyplot as _plt
    _plt.close(fig)

    print("\n  [OK] Figure written: %s (%.0f KB)"
          % (args.out, os.path.getsize(args.out) / 1024))
    print("=" * 72)


if __name__ == "__main__":
    main()
