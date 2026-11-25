"""Render publication-quality flood maps from HEC-RAS result rasters.

Reads the GeoTIFFs a HEC-RAS plan writes as stored maps and produces figures of
maximum water depth, maximum flow velocity, the depth-velocity hazard product,
flood arrival time and the inundation boundary.

Design of the class intervals
-----------------------------
Every layer uses **fixed** class boundaries, identical for every scenario, so
that maps of different model runs can be compared directly. Auto-scaling each
map to its own data would make two very different results look alike.

The boundaries were chosen from the distribution of the results themselves,
pooled across scenarios, rather than from round numbers. The measured medians
are low - about 0.12 m depth, 0.15 m/s velocity and 0.017 m2/s hazard - so the
classes are concentrated where the data are, while still resolving the
dangerous tail. Where a recognised safety threshold exists it is used as a
boundary.

Downsampling
------------
The result rasters are far larger than any figure, so they are reduced before
plotting. The reduction takes the **maximum** of each block rather than a
sample of it, which guarantees that a narrow flow path one or two cells wide
survives into the figure instead of being dropped by nearest-neighbour
sampling. The terrain backdrop is reduced by averaging, which suits a smooth
hillshade.

Usage
-----
::

    python flood_maps.py --plan_folder <folder> --output_dir <folder>
    python flood_maps.py --plan_folder <folder> --output_dir <folder> \\
        --label early_peaked --title-suffix "Early-peaked design storm"

Requires: numpy, tifffile, matplotlib
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from config import settings

__all__ = ["LAYERS", "MapSpec", "block_reduce", "main", "render"]

__version__ = "2.0.0"

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=RuntimeWarning)

NODATA = -9999.0

#: Coordinate reference system of the model, for the axis labels.
_EPSG = settings.integer("grid", "target_epsg")
CRS_LABEL = "EPSG:%d" % _EPSG if _EPSG else "model CRS"

#: Patterns identifying each stored map in the plan folder.
FILE_PATTERNS = {
    "depth": "Depth (Max)",
    "velocity": "Velocity (Max)",
    "hazard": "D _ V (Max)",
    "arrival": "Arrival Time",
}


@dataclass
class MapSpec:
    """Everything that defines one rendered map."""

    key: str
    filename: str
    title: str
    label: str
    boundaries: list[float]
    colours: list[str]
    class_names: list[str]
    over_colour: str
    rationale: str
    reduce: str = "max"
    tick_format: str = "%g"
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Class intervals
# ---------------------------------------------------------------------------
#
# Ten classes per layer, not seven. Five or six classes put the bulk of the
# floodplain into one colour: the measured median depth is 0.142 m and p90 is
# 0.770 m, so almost everything the eye sees fell in the first two classes.
# Ten boundaries resolve that bulk while keeping every threshold that carries
# a physical meaning rather than being a round number.
#
# Colour ramps are resampled from the anchor colours below rather than written
# out by hand, so a change in the number of classes cannot silently leave the
# palette the wrong length.


def _ramp(anchors: Sequence[str], count: int) -> list[str]:
    """Resample a short list of anchor colours to exactly `count` steps."""
    from matplotlib.colors import LinearSegmentedColormap, to_hex
    cmap = LinearSegmentedColormap.from_list("ramp", list(anchors), N=256)
    return [to_hex(cmap(i / (count - 1))) for i in range(count)]


CLASS_COUNT = 10

# Depth. p50 0.142, p75 0.278, p90 0.770, p95 1.390 m. 0.25 m is where
# vehicles begin to be affected, 0.50 m where an adult becomes unstable in
# moving water, 1.00 m is ground-floor inundation; the rest subdivide the bulk.
DEPTH_BOUNDS = [0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.75, 1.00, 2.00]
DEPTH_COLOURS = _ramp(["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
                      CLASS_COUNT)
DEPTH_NAMES = ["< 0.05", "0.05 - 0.10", "0.10 - 0.15", "0.15 - 0.25",
               "0.25 - 0.35", "0.35 - 0.50", "0.50 - 0.75", "0.75 - 1.00",
               "1.00 - 2.00", "> 2.00"]

# Velocity. p25 0.093, p50 0.145, p75 0.334, p90 0.690 m/s. 1 m/s is where
# flow starts to be hard to stand in, 2 m/s is erosive for most unprotected
# surfaces.
VELOCITY_BOUNDS = [0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.75, 1.00, 2.00]
VELOCITY_COLOURS = _ramp(["#FFF5EB", "#FDD0A2", "#FD8D3C", "#E6550D",
                          "#7F2704"], CLASS_COUNT)
VELOCITY_NAMES = ["< 0.05", "0.05 - 0.10", "0.10 - 0.15", "0.15 - 0.25",
                  "0.25 - 0.35", "0.35 - 0.50", "0.50 - 0.75", "0.75 - 1.00",
                  "1.00 - 2.00", "> 2.00"]

# Depth-velocity hazard product. p50 0.017, p75 0.063, p90 0.280, p99 1.747
# m2/s - by far the most skewed of the three, so the lower boundaries are
# closely spaced. The upper ones keep the customary stability thresholds.
HAZARD_BOUNDS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.50, 1.00, 2.00]
HAZARD_COLOURS = _ramp(["#E8F5E9", "#C5E1A5", "#FFF59D", "#FFCC80",
                        "#FB8C00", "#E53935", "#8E0000"], CLASS_COUNT)
HAZARD_NAMES = ["< 0.01", "0.01 - 0.02", "0.02 - 0.05", "0.05 - 0.10",
                "0.10 - 0.15", "0.15 - 0.25",
                "0.25 - 0.50  children at risk",
                "0.50 - 1.00  adults at risk",
                "1.00 - 2.00  vehicles unstable", "> 2.00  extreme"]

# Arrival time. Fixed at 0-24 h for every scenario, so the layer where storm
# timing shows most strongly stays directly comparable between runs.
ARRIVAL_BOUNDS = [1, 2, 3, 4, 6, 8, 12, 16, 20]
ARRIVAL_COLOURS = _ramp(["#00224E", "#35486E", "#7C7B78", "#D3C164",
                         "#FDE725"], CLASS_COUNT)
ARRIVAL_NAMES = ["< 1", "1 - 2", "2 - 3", "3 - 4", "4 - 6", "6 - 8",
                 "8 - 12", "12 - 16", "16 - 20", "> 20"]


LAYERS: list[MapSpec] = [
    MapSpec(
        key="depth",
        filename="depth_max",
        title="Maximum Water Depth",
        label="Maximum depth (m)",
        boundaries=DEPTH_BOUNDS,
        colours=DEPTH_COLOURS,
        class_names=DEPTH_NAMES,
        over_colour="#041F44",
        rationale="median 0.12 m; classes resolve the shallow bulk",
        tick_format="%.2f",
    ),
    MapSpec(
        key="velocity",
        filename="velocity_max",
        title="Maximum Flow Velocity",
        label="Maximum velocity (m s$^{-1}$)",
        boundaries=VELOCITY_BOUNDS,
        colours=VELOCITY_COLOURS,
        class_names=VELOCITY_NAMES,
        over_colour="#7F2704",
        rationale="median 0.15 m/s; classes resolve the slow bulk",
        tick_format="%.2f",
    ),
    MapSpec(
        key="hazard",
        filename="hazard_dv",
        title="Flood Hazard, Depth $\\times$ Velocity",
        label="d $\\times$ v (m$^2$ s$^{-1}$)",
        boundaries=HAZARD_BOUNDS,
        colours=HAZARD_COLOURS,
        class_names=HAZARD_NAMES,
        over_colour="#6D0000",
        rationale="median 0.017; upper classes follow stability thresholds",
        tick_format="%.2f",
    ),
    MapSpec(
        key="arrival",
        filename="arrival_time",
        title="Flood Arrival Time",
        label="Time from start of storm (h)",
        boundaries=ARRIVAL_BOUNDS,
        colours=ARRIVAL_COLOURS,
        class_names=ARRIVAL_NAMES,
        over_colour="#FDE725",
        rationale="fixed 0-24 h so scenarios are directly comparable",
        # Water arrives at a place when the FIRST water gets there, so blocks
        # are reduced by their minimum. Taking the maximum would report the
        # slowest cell in each block and push the whole map towards late
        # arrival.
        reduce="min",
        tick_format="%g",
    ),
]


# ---------------------------------------------------------------------------
# Raster handling
# ---------------------------------------------------------------------------

def find_raster(folder: Path, pattern: str) -> Path | None:
    """Locate the stored map whose name contains the pattern."""
    hits = [p for p in folder.iterdir()
            if p.suffix.lower() == ".tif" and pattern.lower() in p.name.lower()]
    if not hits:
        return None
    hits.sort(key=lambda p: len(p.name), reverse=True)
    return hits[0]


def read_geotiff(path: Path) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Read a GeoTIFF and its extent, without GDAL."""
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        data = page.asarray().astype(np.float32)
        scale_tag = page.tags.get(33550)
        tie_tag = page.tags.get(33922)
        if scale_tag is None or tie_tag is None:
            raise ValueError("%s has no georeferencing tags" % path.name)
        dx, dy = scale_tag.value[0], scale_tag.value[1]
        tie_col, tie_row = tie_tag.value[0], tie_tag.value[1]
        tie_x, tie_y = tie_tag.value[3], tie_tag.value[4]

    rows, cols = data.shape
    x_min = tie_x + tie_col * dx
    x_max = x_min + cols * dx
    y_max = tie_y - tie_row * dy
    y_min = y_max - rows * dy
    return np.where(data == NODATA, np.nan, data), (x_min, x_max, y_min, y_max)


def block_reduce(data: np.ndarray, factor: int, how: str = "max") -> np.ndarray:
    """Shrink a raster by an integer factor, block by block.

    ``max`` keeps the largest value in each block, so a flow path only one or
    two cells wide still appears in the output. Nearest-neighbour sampling, the
    default when a large image is drawn into a small figure, would simply miss
    it. ``mean`` suits a terrain backdrop, where smoothness is wanted.
    """
    if factor <= 1:
        return data
    rows = data.shape[0] // factor * factor
    cols = data.shape[1] // factor * factor
    blocks = data[:rows, :cols].reshape(
        rows // factor, factor, cols // factor, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if how == "mean":
            return np.nanmean(blocks, axis=(1, 3))
        if how == "min":
            return np.nanmin(blocks, axis=(1, 3))
        return np.nanmax(blocks, axis=(1, 3))


def valid_bounds(data: np.ndarray, pad: int = 12) -> tuple[int, int, int, int] | None:
    """Row and column limits of the data, so NoData margins can be cropped."""
    mask = np.isfinite(data)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return (max(0, r0 - pad), min(data.shape[0] - 1, r1 + pad),
            max(0, c0 - pad), min(data.shape[1] - 1, c1 + pad))


def crop(data: np.ndarray, extent: tuple[float, float, float, float],
         bounds: tuple[int, int, int, int]):
    """Crop an array and recompute its extent."""
    r0, r1, c0, c1 = bounds
    x_min, x_max, y_min, y_max = extent
    dx = (x_max - x_min) / data.shape[1]
    dy = (y_max - y_min) / data.shape[0]
    return data[r0:r1 + 1, c0:c1 + 1], (
        x_min + c0 * dx, x_min + (c1 + 1) * dx,
        y_max - (r1 + 1) * dy, y_max - r0 * dy,
    )


# ---------------------------------------------------------------------------
# Cartographic furniture
# ---------------------------------------------------------------------------

def add_north_arrow(ax) -> None:
    """A north arrow in the upper-left corner, clear of the data and legend."""
    import matplotlib.patches as patches
    import matplotlib.patheffects as path_effects

    x, y = 0.048, 0.945
    length = 0.105
    width = 0.032
    halo = [path_effects.withStroke(linewidth=3.4, foreground="white")]

    left = patches.Polygon(
        [[x, y], [x - width / 2, y - length], [x, y - length * 0.72]],
        facecolor="black", edgecolor="black", lw=0.9,
        transform=ax.transAxes, zorder=12)
    right = patches.Polygon(
        [[x, y], [x + width / 2, y - length], [x, y - length * 0.72]],
        facecolor="white", edgecolor="black", lw=0.9,
        transform=ax.transAxes, zorder=12)
    ax.add_patch(left)
    ax.add_patch(right)

    text = ax.text(x, y + 0.012, "N", transform=ax.transAxes,
                   ha="center", va="bottom", fontsize=13,
                   fontweight="bold", zorder=13)
    text.set_path_effects(halo)


#: Which corner the scale bar sits in: "lower left" or "lower right".
#: The basin is a crescent along the coast, so the lower-left corner of
#: the map extent falls outside the mesh and is empty white, while the
#: lower-right corner sits on the hillshade and on the steepest drainage
#: lines. The bar reads more clearly off the terrain than on it.
SCALE_BAR_CORNER = "lower left"


def add_scale_bar(ax, extent) -> None:
    """A metric scale bar in the corner named by SCALE_BAR_CORNER."""
    import matplotlib.patches as patches
    import matplotlib.patheffects as path_effects

    x_min, x_max, y_min, y_max = extent
    width = x_max - x_min
    height = y_max - y_min

    target = width * 0.22
    length = 100.0
    for candidate in (50000, 20000, 10000, 5000, 2000, 1000,
                      500, 200, 100, 50):
        if target >= candidate:
            length = float(candidate)
            break

    if SCALE_BAR_CORNER == "lower left":
        x0 = x_min + width * 0.04
    else:
        x0 = x_max - width * 0.04 - length
    y0 = y_min + height * 0.055
    bar_h = height * 0.011

    segments = 4
    seg = length / segments
    for i in range(segments):
        ax.add_patch(patches.Rectangle(
            (x0 + i * seg, y0), seg, bar_h,
            facecolor="black" if i % 2 == 0 else "white",
            edgecolor="black", lw=0.9, zorder=12))

    halo = [path_effects.withStroke(linewidth=3.0, foreground="white")]
    if length >= 1000:
        labels = [("0", 0.0), ("%g" % (length / 2000), length / 2),
                  ("%g km" % (length / 1000), length)]
    else:
        labels = [("0", 0.0), ("%g" % (length / 2), length / 2),
                  ("%g m" % length, length)]
    for text, offset in labels:
        item = ax.text(x0 + offset, y0 + bar_h * 1.6, text,
                       ha="center", va="bottom", fontsize=9.5,
                       fontweight="bold", zorder=13)
        item.set_path_effects(halo)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(spec: MapSpec, data: np.ndarray, terrain: np.ndarray | None,
           extent, out_path: Path, dpi: int, subtitle: str) -> dict:
    """Draw one map and return the share of wet area in each class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    colours = list(spec.colours)
    cmap = mcolors.ListedColormap(colours)
    cmap.set_bad(alpha=0.0)
    cmap.set_over(spec.over_colour)
    norm = mcolors.BoundaryNorm(spec.boundaries, cmap.N, extend="both")

    fig, ax = plt.subplots(figsize=(12.5, 8.4), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    if terrain is not None:
        light = mcolors.LightSource(azdeg=315, altdeg=45)
        terrain_cmap = plt.get_cmap("Greys_r")
        shaded = light.shade(np.ma.masked_invalid(terrain), cmap=terrain_cmap,
                             blend_mode="soft", vert_exag=2.0,
                             vmin=np.nanmin(terrain), vmax=np.nanmax(terrain))
        ax.imshow(shaded, extent=extent, origin="upper", aspect="equal",
                  interpolation="bilinear", alpha=0.55, zorder=1)

    image = ax.imshow(data, extent=extent, origin="upper", cmap=cmap,
                      norm=norm, aspect="equal", interpolation="nearest",
                      alpha=0.92, zorder=3)

    bar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.015, aspect=26,
                       extend="both", spacing="uniform")
    bar.set_label(spec.label, fontsize=12.5, fontweight="bold", labelpad=12)
    bar.set_ticks(spec.boundaries)
    bar.ax.set_yticklabels([spec.tick_format % b for b in spec.boundaries])
    bar.ax.tick_params(labelsize=10.5)
    bar.outline.set_linewidth(0.9)

    ax.set_title(spec.title, fontsize=16, fontweight="bold", pad=16)
    if subtitle:
        ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=11.5, style="italic", color="#444444")

    ax.set_xlabel("Easting (m), %s" % CRS_LABEL, fontsize=11.5,
                  fontweight="bold", labelpad=8)
    ax.set_ylabel("Northing (m), %s" % CRS_LABEL, fontsize=11.5,
                  fontweight="bold", labelpad=8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%d" % int(v)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%d" % int(v)))
    ax.tick_params(labelsize=10)
    for label in ax.get_xticklabels():
        label.set_rotation(0)

    ax.grid(True, ls=":", lw=0.6, color="#B0B0B0", alpha=0.75, zorder=2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#333333")

    add_north_arrow(ax)
    add_scale_bar(ax, extent)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white",
                pad_inches=0.16)
    plt.close(fig)

    wet = np.isfinite(data)
    classes = np.digitize(data[wet], spec.boundaries)
    shares = {}
    for index, name in enumerate(spec.class_names):
        shares[name] = 100.0 * float(np.mean(classes == index)) if wet.any() else 0.0
    return shares


def render_inundation(data: np.ndarray, terrain: np.ndarray | None, extent,
                      out_path: Path, dpi: int, subtitle: str) -> float:
    """Binary inundation extent, drawn over the terrain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    fig, ax = plt.subplots(figsize=(12.5, 8.4), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    if terrain is not None:
        light = mcolors.LightSource(azdeg=315, altdeg=45)
        shaded = light.shade(np.ma.masked_invalid(terrain),
                             cmap=plt.get_cmap("Greys_r"), blend_mode="soft",
                             vert_exag=2.0, vmin=np.nanmin(terrain),
                             vmax=np.nanmax(terrain))
        ax.imshow(shaded, extent=extent, origin="upper", aspect="equal",
                  interpolation="bilinear", alpha=0.6, zorder=1)

    wet = np.isfinite(data) & (data > 0)
    layer = np.where(wet, 1.0, np.nan)
    cmap = mcolors.ListedColormap(["#1B5FA8"])
    cmap.set_bad(alpha=0.0)
    ax.imshow(layer, extent=extent, origin="upper", cmap=cmap,
              aspect="equal", interpolation="nearest", alpha=0.85, zorder=3)

    ax.legend(handles=[patches.Patch(facecolor="#1B5FA8", edgecolor="black",
                                     lw=0.6, label="Inundated area")],
              loc="upper right", fontsize=11, framealpha=0.94)

    ax.set_title("Inundation Extent", fontsize=16, fontweight="bold", pad=16)
    if subtitle:
        ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=11.5, style="italic", color="#444444")
    ax.set_xlabel("Easting (m), %s" % CRS_LABEL, fontsize=11.5,
                  fontweight="bold", labelpad=8)
    ax.set_ylabel("Northing (m), %s" % CRS_LABEL, fontsize=11.5,
                  fontweight="bold", labelpad=8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%d" % int(v)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%d" % int(v)))
    ax.tick_params(labelsize=10)
    ax.grid(True, ls=":", lw=0.6, color="#B0B0B0", alpha=0.75, zorder=2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#333333")

    add_north_arrow(ax)
    add_scale_bar(ax, extent)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white",
                pad_inches=0.16)
    plt.close(fig)
    return float(np.sum(wet))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render publication-quality flood maps.",
        allow_abbrev=False,
    )
    parser.add_argument("--plan_folder", "--plan-folder", dest="plan_folder",
                        required=True, help="Folder holding the stored maps.")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir",
                        required=True, help="Where to write the figures.")
    parser.add_argument("--dem_file", "--dem-file", dest="dem_file",
                        default=str(settings.path("terrain", "dem") or ""),
                        help="Terrain raster used as the backdrop.")
    parser.add_argument("--label", default="",
                        help="Scenario name, added to the file names.")
    parser.add_argument("--title-suffix", default="",
                        help="Subtitle placed under every map title.")
    parser.add_argument("--dpi", type=int,
                        default=settings.integer("output", "dpi") or 400)
    parser.add_argument("--target-width", type=int, default=3400,
                        help="Approximate width in pixels the rasters are "
                             "reduced to before plotting.")
    parser.add_argument("--min-depth", type=float,
                        default=settings.number("output", "min_depth") or 0.01,
                        help="Depth below which a cell is treated as dry (m). "
                             "Suppresses isolated millimetre-deep sheet flow.")
    parser.add_argument("--no-terrain", action="store_true",
                        help="Omit the hillshade backdrop.")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    plan_folder = Path(args.plan_folder)
    out_dir = Path(args.output_dir)
    if not plan_folder.is_dir():
        print("[ERROR] Plan folder not found: %s" % plan_folder)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = ("_" + args.label) if args.label else ""
    subtitle = args.title_suffix

    print("=" * 78)
    print(" Flood map rendering  (%s)" % __version__)
    print("=" * 78)
    print("  Plan folder : %s" % plan_folder)
    print("  Output      : %s" % out_dir)
    print("  Label       : %s" % (args.label or "(none)"))
    print()

    # --- locate and read -------------------------------------------------
    rasters = {}
    for key, pattern in FILE_PATTERNS.items():
        path = find_raster(plan_folder, pattern)
        if path is None:
            print("  [--] %-9s not found (%s)" % (key, pattern))
            continue
        rasters[key] = path
        print("  [OK] %-9s %s" % (key, path.name))

    if "depth" not in rasters:
        print("\n[ERROR] The depth raster is required and was not found.")
        return 1

    print("\n  Reading rasters...")
    depth_full, extent_full = read_geotiff(rasters["depth"])
    bounds = valid_bounds(depth_full)
    if bounds is None:
        print("[ERROR] The depth raster is empty.")
        return 1

    factor = max(1, int(round(depth_full.shape[1] / max(args.target_width, 1))))
    print("  Full resolution      : %d x %d" % depth_full.shape)
    print("  Block reduction      : %dx (taking the MAXIMUM of each block, so "
          "narrow flow paths survive)" % factor)

    data = {}
    depth, extent = crop(depth_full, extent_full, bounds)
    del depth_full
    data["depth"] = block_reduce(depth, factor, "max")
    del depth

    reductions = {spec.key: spec.reduce for spec in LAYERS}
    for key in ("velocity", "hazard", "arrival"):
        if key not in rasters:
            continue
        full, ext = read_geotiff(rasters[key])
        cropped, extent = crop(full, ext, bounds)
        del full
        how = reductions.get(key, "max")
        data[key] = block_reduce(cropped, factor, how)
        if how != "max":
            print("  %-20s reduced by %s" % (key, how.upper()))
        del cropped

    print("  Reduced resolution   : %d x %d" % data["depth"].shape)

    terrain = None
    if not args.no_terrain and Path(args.dem_file).exists():
        dem_full, dem_ext = read_geotiff(Path(args.dem_file))
        dem_cropped, _ = crop(dem_full, dem_ext, bounds)
        del dem_full
        dem_cropped = np.where(dem_cropped < -9000, np.nan, dem_cropped)
        terrain = block_reduce(dem_cropped, factor, "mean")
        del dem_cropped
        print("  Terrain backdrop     : reduced by averaging")
    elif not args.no_terrain:
        print("  [--] Terrain not found: %s" % args.dem_file)

    # --- mask to the wetted area ------------------------------------------
    # A depth threshold removes the isolated cells carrying a millimetre or two
    # of sheet flow, which otherwise speckle the map without conveying
    # anything. The value used is reported so it can be stated alongside the
    # figures.
    raw_wet = int(np.sum(np.isfinite(data["depth"]) & (data["depth"] > 0)))
    dry = ~np.isfinite(data["depth"]) | (data["depth"] <= args.min_depth)
    for key in list(data):
        data[key] = np.where(dry, np.nan, data[key])

    wet_cells = int(np.sum(~dry))
    print("  Wet-cell threshold   : %.3f m" % args.min_depth)
    print("  Wet cells (reduced)  : %s of %s (%.2f %%)"
          % (format(wet_cells, ","), format(data["depth"].size, ","),
             100 * wet_cells / data["depth"].size))
    if args.min_depth > 0 and raw_wet:
        print("  Removed by threshold : %s cells (%.1f %% of anything wet)"
              % (format(raw_wet - wet_cells, ","),
                 100 * (raw_wet - wet_cells) / raw_wet))

    # --- render ------------------------------------------------------------
    print("\n  Rendering at %d dpi..." % args.dpi)
    summary = {}
    for spec in LAYERS:
        if spec.key not in data:
            print("    [SKIP] %-14s data unavailable" % spec.filename)
            continue
        out_path = out_dir / ("%s%s.png" % (spec.filename, suffix))
        shares = render(spec, data[spec.key], terrain, extent, out_path,
                        args.dpi, subtitle)
        summary[spec.title] = (spec, shares)
        print("    [OK]   %-34s %6.0f KB   (%s)"
              % (out_path.name, out_path.stat().st_size / 1024, spec.rationale))

    out_path = out_dir / ("inundation_extent%s.png" % suffix)
    render_inundation(data["depth"], terrain, extent, out_path, args.dpi,
                      subtitle)
    print("    [OK]   %-34s %6.0f KB"
          % (out_path.name, out_path.stat().st_size / 1024))

    # --- class table -------------------------------------------------------
    print()
    print("=" * 78)
    print(" Share of the inundated area in each class")
    print("=" * 78)
    for title, (spec, shares) in summary.items():
        print()
        print("  %s" % title)
        for name, share in shares.items():
            bar = "#" * int(round(share / 2))
            print("    %-32s %6.2f %%  %s" % (name, share, bar))

    print()
    print("  Class boundaries are fixed, so these figures can be compared "
          "directly")
    print("  with those of any other scenario rendered by this script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
