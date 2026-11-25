"""Convert gridded precipitation from NetCDF to HEC-DSS.

Reads a NetCDF file holding a time series of precipitation on a regular
latitude/longitude grid, reprojects it onto a projected coordinate reference
system, and writes it as HEC-DSS period-cumulative gridded records suitable for
a HEC-RAS rain-on-grid simulation.

Reading
-------
The file is read through the ``netCDF4`` library when it is available, falling
back to ``h5py`` otherwise. The netCDF4 backend accepts every NetCDF format
including the classic ones, resolves axis identity from the declared dimensions,
unpacks ``scale_factor``/``add_offset``, masks fill values and navigates groups.
The h5py backend reproduces as much of that as the raw HDF5 layout allows, and
is limited to NetCDF-4 files.

Input requirements
------------------
The conversion refuses ambiguous input rather than guessing. It requires:

* a three-dimensional precipitation variable on a regular, uniformly spaced
  latitude/longitude grid, with a uniform time axis;
* units of depth (millimetres, centimetres, metres or inches), declared on the
  variable and consistent with the magnitude of the data;
* per-interval increments rather than a cumulative run-total;
* metadata sufficient to establish whether a timestamp labels the start or the
  end of its accumulation interval.

Output
------
Alongside the DSS file a sidecar JSON document records the simulation window,
the grid geometry, the conversion settings and the quality-assurance results,
so downstream steps consume exact values instead of re-deriving them.

Example
-------
::

    python netcdf_to_dss.py --dry-run
    python netcdf_to_dss.py --nc-file storm.nc --dss-file storm.dss

    from netcdf_to_dss_v3 import convert, ConversionError
    result = convert("storm.nc", "storm.dss")

Requires
--------
numpy, scipy, pyproj, hecdss, and either netCDF4 or h5py.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from config import settings

__all__ = [
    "ConversionError",
    "ConversionResult",
    "SourceGrid",
    "TargetGrid",
    "TimeAxis",
    "UnitResolution",
    "build_intervals",
    "build_target_grid",
    "convert",
    "main",
    "open_dataset",
    "resolve_units",
    "to_dss_timestamp",
]

__version__ = "3.1.0"

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DSS_NULL = np.float32(-3.4028234663852886e38)
DSS_GRID_TYPE_SPECIFIED = 430
DSS_DATA_TYPE_PER_CUM = 1
DSS_SRS_TYPE_WKT = 0

PRECIP_VARIABLE_NAMES = (
    "tp", "precip", "precipitation", "pr",
    "RAINC", "RAINNC", "rain", "rainfall", "Tot_Precip",
)
LATITUDE_VARIABLE_NAMES = ("latitude", "lat", "XLAT", "y", "RLAT")
LONGITUDE_VARIABLE_NAMES = ("longitude", "lon", "XLONG", "x", "RLON")
TIME_VARIABLE_NAMES = ("time", "Time", "TIME", "times")

MONTH_ABBREVIATIONS = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}

#: Accepted units of precipitation depth, as millimetres per unit.
#:
#: Only depth units are accepted. A rate or flux (for example
#: ``kg m-2 s-1``) is rejected rather than converted, because interpreting one
#: requires assumptions about the averaging period that belong to the person
#: preparing the data, not to this tool.
DEPTH_UNITS_MM = {
    "mm": 1.0,
    "millimeter": 1.0, "millimeters": 1.0,
    "millimetre": 1.0, "millimetres": 1.0,
    "cm": 10.0,
    "centimeter": 10.0, "centimeters": 10.0,
    "centimetre": 10.0, "centimetres": 10.0,
    "m": 1000.0,
    "meter": 1000.0, "meters": 1000.0,
    "metre": 1000.0, "metres": 1000.0,
    "in": 25.4, "inch": 25.4, "inches": 25.4,
}

#: The highest hourly rainfall reliably observed is about 305 mm (Holt,
#: Missouri, 1947). A model cell averages over kilometres, so anything near
#: this is already extraordinary.
MAX_CREDIBLE_MM_PER_HOUR = 300.0

#: A modelled storm whose peak never reaches this anywhere is not a storm.
MIN_CREDIBLE_PEAK_MM_PER_HOUR = 0.5

_DEACCUMULATION_MARKERS = (
    "apcp", "cumulative", "accumulated", "derived from", "difference",
    "de-accumulat",
)

_WRF_SPHERE_RADIUS_M = 6371229.0
_METRES_PER_DEGREE_LATITUDE = 111132.0
_METRES_PER_DEGREE_LONGITUDE_EQUATOR = 111320.0

# Defaults come from config.ini. A command-line argument overrides them.
DEFAULT_NC_FILE = settings.path("rainfall", "netcdf")
DEFAULT_DSS_FILE = settings.path("rainfall", "dss")
DEFAULT_TARGET_EPSG = settings.integer("grid", "target_epsg")
DEFAULT_SOURCE_EPSG = settings.integer("grid", "source_epsg") or 4326
DEFAULT_CELL_SIZE_M = settings.number("grid", "cell_size_m") or 1000.0
DEFAULT_RESAMPLE_METHOD = settings.text("grid", "resample") or "linear"
DEFAULT_INTERVAL_CONVENTION = (
    settings.text("rainfall", "interval_convention") or "auto")
# No fallback: an undeclared accumulation is refused, with evidence.
DEFAULT_ACCUMULATION = settings.text("rainfall", "accumulation")

#: Resampling methods this converter accepts. The first four are solved by
#: ``scipy.interpolate.RegularGridInterpolator`` on the latitude/longitude
#: axes; ``kriging`` is ordinary kriging, solved here, in the projected plane.
RESAMPLE_METHODS = ("nearest", "linear", "cubic", "pchip", "kriging")

#: Ordinary kriging uses a moving neighbourhood of this many source cells per
#: target cell. A global system over all source cells would be exact but
#: needs an n x n factorisation for n cells, which at WRF resolution is a
#: 3,366-square matrix solved once per target cell; the nearest cells carry
#: almost all of the weight, and GSLIB-family implementations restrict the
#: neighbourhood for the same reason.
KRIGING_NEIGHBOURS = 16

#: Added to the diagonal of the kriging system. A regular grid produces many
#: exactly equal distances, so the matrix can be singular to working
#: precision; this is small against any real nugget and keeps the solve
#: stable.
KRIGING_JITTER = 1e-10

#: Pairs sampled when estimating the empirical semivariogram. All pairs of a
#: 3,366-cell grid is 5.7 million, which is more than a three-parameter fit
#: needs.
KRIGING_VARIOGRAM_PAIRS = 200_000
KRIGING_VARIOGRAM_BINS = 24

VOLUME_DRIFT_WARNING_THRESHOLD = 0.02
_CUMULATIVE_MONOTONICITY_THRESHOLD = 0.9999
# A constant series repeats one value exactly; the tolerance only absorbs
# the float noise left by unpacking scale_factor and add_offset.
_FLAT_TOLERANCE_MM = 1e-6


class ConversionError(RuntimeError):
    """Raised when the input cannot be converted correctly.

    Carries an explanation of what could not be established and what the caller
    should do about it.
    """


# ---------------------------------------------------------------------------
# Reading: a uniform view over either backend
# ---------------------------------------------------------------------------

class SourceDataset:
    """Uniform read-only view over a NetCDF file.

    Subclasses hide the differences between the netCDF4 and h5py backends so
    the conversion logic never has to know which one is in use.
    """

    name = "abstract"

    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> "SourceDataset":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def variable_paths(self) -> list[str]:
        """Every variable in the file, as ``group/name`` paths."""
        raise NotImplementedError

    def global_attributes(self) -> dict[str, Any]:
        raise NotImplementedError

    def attributes(self, path: str) -> dict[str, Any]:
        raise NotImplementedError

    def shape(self, path: str) -> tuple[int, ...]:
        raise NotImplementedError

    def read(self, path: str) -> np.ndarray:
        """Read a variable as float64, decoded and with missing data as NaN."""
        raise NotImplementedError

    def dimension_names(self, path: str) -> tuple[str, ...] | None:
        """Axis names in order, or None when the file does not record them."""
        raise NotImplementedError

    # -- shared helpers ---------------------------------------------------

    def locate(self, candidates: Sequence[str], purpose: str) -> str:
        """Find a variable by candidate name, preferring the root group."""
        paths = self.variable_paths()
        by_base: dict[str, list[str]] = {}
        for path in paths:
            by_base.setdefault(path.rsplit("/", 1)[-1], []).append(path)

        for candidate in candidates:
            matches = by_base.get(candidate)
            if matches:
                # Prefer a root-level variable over one nested in a group.
                matches.sort(key=lambda p: p.count("/"))
                if len(matches) > 1:
                    LOGGER.warning(
                        "Several variables named '%s' (%s); using '%s'",
                        candidate, ", ".join(matches), matches[0],
                    )
                return matches[0]

        raise ConversionError(
            f"Could not find the {purpose}. Looked for {list(candidates)}; "
            f"the file contains {paths}."
        )

    def attribute_text(self, paths: Sequence[str]) -> str:
        """All global and per-variable attribute text, lowercased."""
        parts = [str(v) for v in self.global_attributes().values()]
        for path in paths:
            parts.extend(str(v) for v in self.attributes(path).values())
        return " ".join(parts).lower()


def _decode(value: Any) -> Any:
    """Decode an attribute that may be stored as bytes."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return [item.decode("utf-8", "replace") for item in value]
    return value


class NetCDF4Dataset(SourceDataset):
    """Backend built on the netCDF4 library.

    Handles every NetCDF format, applies CF decoding, and reports the declared
    dimension names so axis identity never has to be inferred.
    """

    name = "netCDF4"

    def __init__(self, path: Path) -> None:
        import netCDF4

        self._netCDF4 = netCDF4
        self._ds = netCDF4.Dataset(str(path))
        self._index: dict[str, Any] = {}
        for var_path, variable in self._walk(self._ds, ""):
            self._index[var_path] = variable

    @staticmethod
    def _walk(group: Any, prefix: str) -> Iterator[tuple[str, Any]]:
        for name, variable in group.variables.items():
            yield prefix + name, variable
        for name, subgroup in group.groups.items():
            yield from NetCDF4Dataset._walk(subgroup, prefix + name + "/")

    def close(self) -> None:
        self._ds.close()

    def variable_paths(self) -> list[str]:
        return list(self._index)

    def global_attributes(self) -> dict[str, Any]:
        return {k: _decode(self._ds.getncattr(k)) for k in self._ds.ncattrs()}

    def attributes(self, path: str) -> dict[str, Any]:
        variable = self._index[path]
        return {k: _decode(variable.getncattr(k)) for k in variable.ncattrs()}

    def shape(self, path: str) -> tuple[int, ...]:
        return tuple(self._index[path].shape)

    def read(self, path: str) -> np.ndarray:
        variable = self._index[path]
        variable.set_auto_maskandscale(True)
        values = variable[:]
        if np.ma.isMaskedArray(values):
            values = values.astype(np.float64).filled(np.nan)
        return np.asarray(values, dtype=np.float64)

    def read_raw(self, path: str) -> np.ndarray:
        """Undecoded values, used only to report what packing was applied."""
        variable = self._index[path]
        variable.set_auto_maskandscale(False)
        values = np.asarray(variable[:], dtype=np.float64)
        variable.set_auto_maskandscale(True)
        return values

    def dimension_names(self, path: str) -> tuple[str, ...] | None:
        return tuple(self._index[path].dimensions)

    def decode_time(self, path: str) -> list[datetime] | None:
        """Decode a time coordinate with cftime, honouring the calendar."""
        variable = self._index[path]
        units = getattr(variable, "units", None)
        if units is None:
            return None
        calendar = getattr(variable, "calendar", "standard")
        try:
            values = self._netCDF4.num2date(
                variable[:], units, calendar,
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            )
        except (ValueError, TypeError) as exc:
            LOGGER.debug("cftime could not decode the time axis: %s", exc)
            return None
        return [
            v if isinstance(v, datetime)
            else datetime(v.year, v.month, v.day, v.hour, v.minute, v.second)
            for v in np.atleast_1d(values)
        ]


class H5pyDataset(SourceDataset):
    """Backend built on h5py, for environments without netCDF4.

    Limited to NetCDF-4 files, and reproduces the CF decoding that the netCDF
    library would otherwise perform: packing, fill values and the dimension
    scale machinery.
    """

    name = "h5py"

    def __init__(self, path: Path) -> None:
        import h5py

        self._h5py = h5py
        try:
            self._f = h5py.File(str(path), "r")
        except OSError as exc:
            import importlib.util

            remedy = (
                "Drop --backend h5py and let the netCDF4 backend read it."
                if importlib.util.find_spec("netCDF4") is not None
                else "Install it with: pip install netCDF4"
            )
            raise ConversionError(
                f"h5py cannot open {path}. NetCDF-4 files are HDF5 and open "
                f"fine; the classic NetCDF-3 formats are a different binary "
                f"layout and need the netCDF4 library. {remedy} "
                f"Underlying error: {exc}"
            ) from exc
        self._index = dict(self._walk(self._f, ""))

    @classmethod
    def _walk(cls, group: Any, prefix: str) -> Iterator[tuple[str, Any]]:
        import h5py

        for name, obj in group.items():
            path = prefix + name
            if isinstance(obj, h5py.Group):
                yield from cls._walk(obj, path + "/")
            else:
                yield path, obj

    def close(self) -> None:
        self._f.close()

    def variable_paths(self) -> list[str]:
        return list(self._index)

    def global_attributes(self) -> dict[str, Any]:
        return {k: _decode(v) for k, v in self._f.attrs.items()}

    def attributes(self, path: str) -> dict[str, Any]:
        return {k: _decode(v) for k, v in self._index[path].attrs.items()}

    def shape(self, path: str) -> tuple[int, ...]:
        return tuple(self._index[path].shape)

    def read(self, path: str) -> np.ndarray:
        variable = self._index[path]
        values = np.asarray(variable[:], dtype=np.float64)
        attrs = variable.attrs

        # Missing data first, while the sentinel is still recognisable.
        for key in ("_FillValue", "missing_value"):
            if key in attrs:
                sentinel = float(np.atleast_1d(attrs[key])[0])
                if not math.isnan(sentinel):
                    values = np.where(values == sentinel, np.nan, values)

        # Then CF unpacking, which the netCDF library would apply for us.
        if "scale_factor" in attrs:
            values = values * float(np.atleast_1d(attrs["scale_factor"])[0])
        if "add_offset" in attrs:
            values = values + float(np.atleast_1d(attrs["add_offset"])[0])
        return values

    def dimension_names(self, path: str) -> tuple[str, ...] | None:
        """Recover axis names from the HDF5 dimension scale attributes."""
        variable = self._index[path]
        dimension_list = variable.attrs.get("DIMENSION_LIST")
        if dimension_list is not None:
            names: list[str] = []
            for refs in dimension_list:
                resolved = [self._f[ref].name.lstrip("/") for ref in refs]
                if len(resolved) != 1:
                    return None
                names.append(resolved[0])
            return tuple(names)

        coordinates = variable.attrs.get("_Netcdf4Coordinates")
        if coordinates is None:
            return None
        by_id = {}
        for other_path, other in self._index.items():
            if other.attrs.get("CLASS", b"") == b"DIMENSION_SCALE":
                dim_id = int(np.atleast_1d(other.attrs["_Netcdf4Dimid"])[0])
                by_id[dim_id] = other_path
        try:
            return tuple(by_id[int(c)] for c in np.atleast_1d(coordinates))
        except KeyError:
            return None


def open_dataset(path: str | os.PathLike[str],
                 backend: str = "auto") -> SourceDataset:
    """Open a NetCDF file with the best available backend.

    Parameters
    ----------
    path
        File to open.
    backend
        ``'auto'`` prefers netCDF4 and falls back to h5py; ``'netcdf4'`` and
        ``'h5py'`` force one.

    Returns
    -------
    SourceDataset
    """
    path = Path(path)
    if not path.exists():
        raise ConversionError(f"Input NetCDF not found: {path}")

    want_netcdf4 = backend in ("auto", "netcdf4")
    if want_netcdf4:
        try:
            dataset = NetCDF4Dataset(path)
            LOGGER.info("Backend         : netCDF4 (full CF decoding)")
            return dataset
        except ImportError:
            if backend == "netcdf4":
                raise ConversionError(
                    "The netCDF4 library was requested but is not installed. "
                    "Install it with: pip install netCDF4"
                ) from None
            LOGGER.warning(
                "netCDF4 is not installed; falling back to h5py, which reads "
                "only NetCDF-4 files"
            )

    try:
        dataset = H5pyDataset(path)
    except ImportError:
        raise ConversionError(
            "No NetCDF reader is installed. Install either library:\n"
            "    pip install netCDF4     reads every NetCDF format\n"
            "    pip install h5py        reads NetCDF-4 only"
        ) from None
    LOGGER.info("Backend         : h5py (NetCDF-4 only)")
    return dataset


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UnitResolution:
    """Outcome of establishing what units the precipitation is in."""

    declared: str
    canonical: str
    factor_to_mm: float
    peak_mm: float
    peak_mm_per_hour: float
    verdict: str
    note: str = ""


def normalise_units(raw: Any) -> str:
    """Fold the many spellings of a unit onto a canonical token."""
    text = str(raw).strip().lower()
    text = text.replace("**", "").replace("^", "")
    text = text.replace("\u00b7", " ").replace("*", " ")
    return re.sub(r"\s+", " ", text)


def _credibility_score(mm_per_hour: float) -> float:
    """Distance, in log space, from the centre of the credible band.

    Lower is better. Used to rank candidate units so the most physically
    sensible one is suggested first rather than the alphabetically first.
    """
    if mm_per_hour <= 0:
        return math.inf
    centre = math.sqrt(MIN_CREDIBLE_PEAK_MM_PER_HOUR * MAX_CREDIBLE_MM_PER_HOUR)
    return abs(math.log(mm_per_hour / centre))


def _classify(mm_per_hour: float) -> str:
    if mm_per_hour > MAX_CREDIBLE_MM_PER_HOUR:
        return "implausibly large"
    if mm_per_hour < MIN_CREDIBLE_PEAK_MM_PER_HOUR:
        return "implausibly small"
    return "plausible"


def resolve_units(
    declared: Any,
    peak_raw: float,
    timestep: timedelta,
    assume: str | None = None,
) -> UnitResolution:
    """Establish the conversion from the stored values to millimetres.

    The declared ``units`` attribute is only a string written by whoever
    produced the file, so it is both parsed and cross-checked against the
    magnitude of the data. When the declared unit yields an implausible
    intensity and a different unit yields a plausible one, the conversion is
    refused and the likely correct unit is named.

    Parameters
    ----------
    declared
        The variable's ``units`` attribute, or None if absent.
    peak_raw
        Largest stored value, before conversion.
    timestep
        Length of one accumulation interval, used to express the peak as an
        equivalent hourly intensity.
    assume
        Unit to use instead of the declared one, bypassing both the lookup and
        the plausibility gate.

    Returns
    -------
    UnitResolution

    Raises
    ------
    ConversionError
        If the unit is absent, unrecognised, or contradicted by the data.
    """
    hours = timestep.total_seconds() / 3600.0

    if assume is not None:
        canonical = normalise_units(assume)
        if canonical not in DEPTH_UNITS_MM:
            raise ConversionError(
                f"Assumed units {assume!r} are not a recognised depth unit. "
                f"Accepted: {sorted(set(DEPTH_UNITS_MM))}."
            )
        factor = DEPTH_UNITS_MM[canonical]
        peak_mm = peak_raw * factor
        per_hour = peak_mm / hours if hours else float("nan")
        LOGGER.warning(
            "Units forced to '%s' by the caller; the plausibility check is \n"
            "bypassed", canonical,
        )
        return UnitResolution(str(declared), canonical, factor, peak_mm,
                              per_hour, "assumed", "forced by the caller")

    if declared is None:
        raise ConversionError(
            "The precipitation variable declares no 'units' attribute, so the \n"
            "physical meaning of its values cannot be established. Re-run with \n"
            "--assume-units to declare it explicitly. Accepted units: "
            f"{sorted(set(DEPTH_UNITS_MM))}."
        )

    canonical = normalise_units(declared)
    if canonical not in DEPTH_UNITS_MM:
        raise ConversionError(
            f"Unrecognised precipitation units {str(declared)!r} (normalised "
            f"to {canonical!r}). Only units of depth are accepted: "
            f"{sorted(set(DEPTH_UNITS_MM))}. A rate or flux such as "
            f"'kg m-2 s-1' must be converted to a per-interval depth before "
            f"conversion, so that the averaging period is chosen deliberately."
        )

    factor = DEPTH_UNITS_MM[canonical]
    peak_mm = peak_raw * factor
    per_hour = peak_mm / hours if hours else float("nan")
    verdict = _classify(per_hour)

    LOGGER.info(
        "Units           : '%s' -> '%s', x%g", declared, canonical, factor
    )
    LOGGER.info(
        "Peak intensity  : %.4f mm per step = %.4f mm/h (%s)",
        peak_mm, per_hour, verdict,
    )

    if verdict == "plausible":
        return UnitResolution(str(declared), canonical, factor, peak_mm,
                              per_hour, verdict)

    # The declared unit is not credible. Is another one?
    alternatives = []
    for name, alt_factor in DEPTH_UNITS_MM.items():
        alt_per_hour = peak_raw * alt_factor / hours if hours else float("nan")
        if _classify(alt_per_hour) == "plausible":
            alternatives.append((_credibility_score(alt_per_hour), name,
                                 alt_factor, alt_per_hour))
    alternatives.sort()

    if alternatives:
        # Report one entry per distinct conversion factor, best first.
        seen: set[float] = set()
        lines = []
        for _, name, alt_factor, alt_per_hour in alternatives:
            if alt_factor in seen:
                continue
            seen.add(alt_factor)
            lines.append("'%s' would give %.2f mm/h" % (name, alt_per_hour))
        raise ConversionError(
            f"The declared units {str(declared)!r} imply a peak intensity of "
            f"{per_hour:.4g} mm/h, which is {verdict} for precipitation. "
            f"The data are more consistent with: {'; '.join(lines[:3])}. "
            f"Correct the file, or override with --assume-units."
        )

    if verdict == "implausibly large":
        raise ConversionError(
            f"A peak intensity of {per_hour:.4g} mm/h exceeds anything ever "
            f"observed, and no supported unit makes these values credible. "
            f"The data are most likely still packed (a 'scale_factor' that was "
            f"not applied) or corrupt. Override with --assume-units only if "
            f"you are certain."
        )

    LOGGER.warning(
        "Peak intensity is only %.4g mm/h. No alternative unit fits either, so "
        "this is accepted as a genuinely dry period, but verify that the "
        "period you selected actually contains the storm.", per_hour,
    )
    return UnitResolution(str(declared), canonical, factor, peak_mm, per_hour,
                          verdict, "accepted as a dry period")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeAxis:
    """Decoded time coordinate."""

    times: list[datetime]
    units: str

    def __len__(self) -> int:
        return len(self.times)


@dataclass(frozen=True)
class SourceGrid:
    """Regular latitude/longitude grid, coordinates being ascending centres."""

    latitude: np.ndarray
    longitude: np.ndarray
    latitude_step: float
    longitude_step: float
    latitude_reversed: bool = False
    longitude_reversed: bool = False

    @property
    def shape(self) -> tuple[int, int]:
        return self.latitude.size, self.longitude.size

    @property
    def west(self) -> float:
        return float(self.longitude[0] - self.longitude_step / 2.0)

    @property
    def east(self) -> float:
        return float(self.longitude[-1] + self.longitude_step / 2.0)

    @property
    def south(self) -> float:
        return float(self.latitude[0] - self.latitude_step / 2.0)

    @property
    def north(self) -> float:
        return float(self.latitude[-1] + self.latitude_step / 2.0)

    def cell_dimensions_metres(self) -> tuple[float, float]:
        """Approximate cell size on the ground as ``(north_south, east_west)``.

        Equal steps in degrees do not give a square cell, because a degree of
        longitude contracts by the cosine of the latitude.
        """
        mean_latitude = float(self.latitude.mean())
        north_south = self.latitude_step * _METRES_PER_DEGREE_LATITUDE
        east_west = (
            self.longitude_step
            * _METRES_PER_DEGREE_LONGITUDE_EQUATOR
            * np.cos(np.radians(mean_latitude))
        )
        return float(north_south), float(east_west)

    def cell_areas(self) -> np.ndarray:
        """Exact spherical area of every cell, in square metres."""
        top = np.radians(self.latitude + self.latitude_step / 2.0)
        bottom = np.radians(self.latitude - self.latitude_step / 2.0)
        row = (
            _WRF_SPHERE_RADIUS_M ** 2
            * np.radians(self.longitude_step)
            * (np.sin(top) - np.sin(bottom))
        )
        return np.repeat(row[:, None], self.longitude.size, axis=1)


@dataclass(frozen=True)
class TargetGrid:
    """Projected grid whose origin is an exact integer multiple of the cell."""

    nx: int
    ny: int
    cell_size: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    lower_left_cell_x: int
    lower_left_cell_y: int
    epsg: int
    crs_name: str
    wkt: str
    sample_latitude: np.ndarray = field(repr=False)
    sample_longitude: np.ndarray = field(repr=False)

    @property
    def cell_area(self) -> float:
        return self.cell_size ** 2

    def index_residual(self) -> tuple[float, float]:
        """Distance between the stored cell index and the true origin."""
        return (
            abs(self.lower_left_cell_x * self.cell_size - self.x_min),
            abs(self.lower_left_cell_y * self.cell_size - self.y_min),
        )


@dataclass
class ConversionResult:
    """Outcome of a conversion, and the provenance record written beside it."""

    dss_path: Path
    json_path: Path | None
    backend: str
    simulation_start: datetime
    simulation_end: datetime
    timestep: timedelta
    pathnames: list[str]
    target_grid: TargetGrid
    units: UnitResolution
    source_volume_m3: float
    target_volume_m3: float
    cells_outside_source_fraction: float
    interval_convention: str
    interval_convention_evidence: str
    first_step_dropped: bool
    spinup_artifact_detected: bool
    accumulation_detected: str
    deaccumulated: bool
    negative_increments_clipped: int
    axis_source: str
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def volume_drift(self) -> float:
        if self.source_volume_m3 <= 0:
            return 0.0
        return (self.target_volume_m3 - self.source_volume_m3) / self.source_volume_m3

    @property
    def record_count(self) -> int:
        return len(self.pathnames)


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def to_dss_timestamp(moment: datetime) -> str:
    """Format a datetime in the HEC-DSS pathname convention.

    HEC-DSS has no ``0000``; midnight is written as ``2400`` of the preceding
    day, which denotes the same instant.

    Examples
    --------
    >>> to_dss_timestamp(datetime(2023, 6, 15, 6, 0))
    '15JUN2023:0600'
    >>> to_dss_timestamp(datetime(2023, 6, 16, 0, 0))
    '15JUN2023:2400'
    """
    if moment.hour == 0 and moment.minute == 0:
        previous = moment - timedelta(days=1)
        return (
            f"{previous.day:02d}{MONTH_ABBREVIATIONS[previous.month]}"
            f"{previous.year:04d}:2400"
        )
    return (
        f"{moment.day:02d}{MONTH_ABBREVIATIONS[moment.month]}"
        f"{moment.year:04d}:{moment.hour:02d}{moment.minute:02d}"
    )


def _parse_cf_time(raw: np.ndarray, units: str) -> list[datetime]:
    """Decode a CF time axis without cftime, for the h5py backend."""
    match = re.match(
        r"(\w+)\s+since\s+(\d{4})-(\d{1,2})-(\d{1,2})"
        r"(?:[\sT]+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?",
        units,
    )
    if match is None:
        raise ConversionError(
            f"Cannot interpret time units {units!r}. Expected a CF string such "
            f"as 'hours since 2023-06-15 06:00:00'."
        )
    origin = datetime(
        int(match.group(2)), int(match.group(3)), int(match.group(4)),
        int(match.group(5) or 0), int(match.group(6) or 0),
        int(match.group(7) or 0),
    )
    unit = match.group(1).lower()
    keyword = next(
        (kw for prefix, kw in (
            ("hour", "hours"), ("day", "days"),
            ("minute", "minutes"), ("second", "seconds"),
        ) if unit.startswith(prefix)),
        None,
    )
    if keyword is None:
        raise ConversionError(f"Unsupported time unit {unit!r}.")
    return [origin + timedelta(**{keyword: float(v)}) for v in raw]


def read_time_axis(dataset: SourceDataset, path: str) -> TimeAxis:
    """Decode the time coordinate into datetimes."""
    attributes = dataset.attributes(path)
    units = attributes.get("units")
    if units is None:
        raise ConversionError(
            f"Time variable '{path}' has no 'units' attribute, so the time "
            f"origin cannot be established."
        )
    units = str(units)

    times = None
    if isinstance(dataset, NetCDF4Dataset):
        times = dataset.decode_time(path)
    if times is None:
        times = _parse_cf_time(dataset.read(path), units)

    LOGGER.info("Time variable   : '%s'", path)
    LOGGER.info("Time units      : %s", units)
    LOGGER.info("Time steps      : %d  (%s to %s)",
                len(times), times[0], times[-1])
    return TimeAxis(times=times, units=units)


def read_source_grid(dataset: SourceDataset,
                     lat_path: str, lon_path: str) -> SourceGrid:
    """Read the coordinate axes as a regular, ascending grid."""
    latitude = dataset.read(lat_path)
    longitude = dataset.read(lon_path)

    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ConversionError(
            f"Coordinates are {latitude.ndim}-D and {longitude.ndim}-D rather "
            f"than 1-D, indicating a curvilinear grid. Regrid the file to a "
            f"regular latitude/longitude mesh before converting it."
        )

    latitude_reversed = bool(latitude[0] > latitude[-1])
    longitude_reversed = bool(longitude[0] > longitude[-1])
    if latitude_reversed:
        latitude = latitude[::-1]
    if longitude_reversed:
        longitude = longitude[::-1]

    lat_steps = np.diff(latitude)
    lon_steps = np.diff(longitude)
    for axis_name, steps in (("latitude", lat_steps), ("longitude", lon_steps)):
        spread = float((steps.max() - steps.min()) / abs(steps.mean()))
        if spread > 1e-6:
            raise ConversionError(
                f"The {axis_name} spacing is not uniform (relative spread "
                f"{spread:.3e}). Regrid the file to a regular mesh first."
            )

    LOGGER.info(
        "Latitude  '%s' : %d cells, %.6f to %.6f deg, step %.6f%s",
        lat_path, latitude.size, latitude[0], latitude[-1], lat_steps.mean(),
        " (reversed to ascending)" if latitude_reversed else "",
    )
    LOGGER.info(
        "Longitude '%s' : %d cells, %.6f to %.6f deg, step %.6f%s",
        lon_path, longitude.size, longitude[0], longitude[-1], lon_steps.mean(),
        " (reversed to ascending)" if longitude_reversed else "",
    )

    return SourceGrid(
        latitude=latitude,
        longitude=longitude,
        latitude_step=float(lat_steps.mean()),
        longitude_step=float(lon_steps.mean()),
        latitude_reversed=latitude_reversed,
        longitude_reversed=longitude_reversed,
    )


def resolve_axis_order(
    dataset: SourceDataset,
    precip_path: str,
    time_path: str,
    lat_path: str,
    lon_path: str,
) -> tuple[tuple[int, int, int], str]:
    """Determine which array axis is time, latitude and longitude.

    The declared dimension names are authoritative. Matching axis lengths
    against the coordinate arrays is only a fallback, and is ambiguous whenever
    two axes happen to share a length, so it is refused in that case rather
    than guessed.

    Returns
    -------
    order
        Positions of the time, latitude and longitude axes.
    source
        Where the answer came from, for the provenance record.
    """
    shape = dataset.shape(precip_path)
    declared = dataset.dimension_names(precip_path)

    if declared is not None and len(declared) == 3:
        wanted = {
            time_path.rsplit("/", 1)[-1]: 0,
            lat_path.rsplit("/", 1)[-1]: 1,
            lon_path.rsplit("/", 1)[-1]: 2,
        }
        positions: dict[int, int] = {}
        for axis, dim_name in enumerate(declared):
            role = wanted.get(dim_name.rsplit("/", 1)[-1])
            if role is not None:
                positions[role] = axis
        if len(positions) == 3:
            order = (positions[0], positions[1], positions[2])
            LOGGER.info(
                "Axis order      : %s -> time=axis %d, latitude=axis %d, "
                "longitude=axis %d", declared, *order,
            )
            return order, "declared dimensions %s" % (declared,)
        LOGGER.debug("Declared dimensions %s do not name the coordinate "
                     "variables; falling back to axis lengths", declared)

    lengths = (
        dataset.shape(time_path)[0],
        dataset.shape(lat_path)[0],
        dataset.shape(lon_path)[0],
    )
    if len(set(lengths)) != 3:
        raise ConversionError(
            f"The file does not declare which axis of the precipitation "
            f"variable is which, and two of the axes have the same length "
            f"(time={lengths[0]}, latitude={lengths[1]}, "
            f"longitude={lengths[2]}), so the order cannot be inferred "
            f"unambiguously. Add CF dimension metadata to the file."
        )

    try:
        order = tuple(shape.index(length) for length in lengths)
    except ValueError:
        raise ConversionError(
            f"Precipitation has shape {shape}, which does not match the "
            f"coordinate lengths time={lengths[0]}, latitude={lengths[1]}, "
            f"longitude={lengths[2]}."
        ) from None

    if len(set(order)) != 3:
        raise ConversionError(
            f"Could not match the axes of {shape} onto the coordinates "
            f"{lengths} unambiguously."
        )

    LOGGER.warning(
        "The file declares no usable dimension metadata; axis order %s was "
        "inferred from the distinct axis lengths %s", order, lengths,
    )
    return order, "inferred from distinct axis lengths %s" % (lengths,)


# ---------------------------------------------------------------------------
# Interval semantics
# ---------------------------------------------------------------------------

def detect_interval_convention(
    dataset: SourceDataset, paths: Sequence[str]
) -> tuple[str | None, str]:
    """Determine whether timestamps label the start or the end of an interval.

    A series obtained by differencing a cumulative run-total,
    ``value[i] = total[i] - total[i-1]``, inherits the label of the later
    sample, so its timestamps mark the end of each interval. The distinction
    shifts the whole hyetograph by one timestep, so it is read from the file's
    metadata rather than assumed.
    """
    text = dataset.attribute_text(paths)
    hits = [marker for marker in _DEACCUMULATION_MARKERS if marker in text]
    if hits:
        return "end", "metadata mentions " + ", ".join(repr(h) for h in hits)
    if "cell_methods" in text and "sum" in text:
        return "end", "cell_methods declares a time sum"
    return None, "no decisive metadata"


def build_intervals(
    times: Sequence[datetime], convention: str
) -> tuple[list[tuple[datetime, datetime]], timedelta]:
    """Expand timestamps into accumulation intervals.

    Raises
    ------
    ConversionError
        If fewer than two timestamps are present, or the spacing varies.
    """
    if len(times) < 2:
        raise ConversionError(
            f"At least two time steps are needed to establish the accumulation "
            f"interval; the file has {len(times)}."
        )

    spacings = {
        (times[index + 1] - times[index]).total_seconds()
        for index in range(len(times) - 1)
    }
    if len(spacings) != 1:
        raise ConversionError(
            f"The time axis is not uniform; spacings present (seconds): "
            f"{sorted(spacings)}. A period-cumulative series requires a single "
            f"interval length."
        )

    timestep = timedelta(seconds=spacings.pop())
    if convention == "end":
        intervals = [(moment - timestep, moment) for moment in times]
    elif convention == "begin":
        intervals = [(moment, moment + timestep) for moment in times]
    else:
        raise ConversionError(f"Unknown interval convention {convention!r}.")
    return intervals, timestep


def _accumulation_stats(field_data: np.ndarray) -> tuple[float, float, bool]:
    """Measure the three properties that tell the three storage forms apart.

    Returns the fraction of transitions that do not fall, the fraction of
    values that are exactly zero, and whether the series is flat in time.
    """
    filled = np.nan_to_num(field_data, nan=0.0)
    differences = np.diff(filled, axis=0)
    non_decreasing = float(np.mean(differences >= -_FLAT_TOLERANCE_MM))
    zero_fraction = float(np.mean(filled == 0.0))
    flat = bool(np.all(np.abs(differences) <= _FLAT_TOLERANCE_MM))
    return non_decreasing, zero_fraction, flat


def classify_accumulation(field_data: np.ndarray) -> str:
    """Report how the series stores depth: 'cumulative', 'constant' or 'increments'.

    Period-cumulative records expect the depth that fell during each interval.
    A run-total written into them would restate the entire storm at every
    timestep, so the two must not be confused.

    The three forms are separated exactly, with no tuned threshold:

    * a value that falls anywhere means increments, because a run-total never
      decreases;
    * differences that are everywhere zero mean the series is constant in
      time, because a uniform hyetograph repeats one value rather than merely
      failing to decrease;
    * never falling, yet not flat, means a run-total.

    ``constant`` is genuinely ambiguous and is never resolved here. A uniform
    hyetograph of 4.86 mm per hour and a run-total of 4.86 mm that gained
    nothing over the window are the same numbers; no property of the data
    separates them. The caller is told which it is, and is refused if it does
    not say.
    """
    if field_data.shape[0] < 2:
        return "increments"

    non_decreasing, zero_fraction, flat = _accumulation_stats(field_data)
    LOGGER.info(
        "Increment check : %.2f%% of transitions non-decreasing, %.2f%% of "
        "values exactly zero, %s in time", non_decreasing * 100.0,
        zero_fraction * 100.0, "flat" if flat else "varying",
    )
    if non_decreasing < _CUMULATIVE_MONOTONICITY_THRESHOLD:
        LOGGER.info("Increment check : series holds per-interval increments")
        return "increments"
    if flat:
        LOGGER.info(
            "Increment check : series is constant in time. That is either a "
            "uniform hyetograph or a run-total that gained nothing; the data "
            "cannot tell them apart")
        return "constant"
    LOGGER.info("Increment check : series never decreases and is not flat: "
                "cumulative run-totals")
    return "cumulative"


def accumulation_evidence(field_data: np.ndarray) -> str:
    """One sentence of measured evidence, for an error the user must act on."""
    if field_data.shape[0] < 2:
        return "The series holds a single record, so nothing can be measured."
    non_decreasing, zero_fraction, flat = _accumulation_stats(field_data)
    return (
        "The series is non-decreasing at %.2f%% of transitions, %.2f%% of its "
        "values are exactly zero, and it is %s in time."
        % (non_decreasing * 100.0, zero_fraction * 100.0,
           "flat" if flat else "not flat")
    )


def deaccumulate(
    field_data: np.ndarray, times: Sequence[datetime]
) -> tuple[np.ndarray, list[datetime], int]:
    """Difference a run-total into per-interval increments.

    ``value[i]`` is the depth accumulated up to ``times[i]``, so the increment
    for the interval ``(times[i - 1], times[i])`` is ``value[i] - value[i-1]``.
    The first timestamp has no predecessor and therefore yields no interval. It
    is dropped rather than invented: taking ``value[0] - 0`` would assume the
    accumulation began exactly at ``times[0]``, which is the very assumption
    that produces the de-accumulation artifact this module warns about.

    A bucketed accumulator that resets mid-series produces negative
    differences. Those are clipped to zero and counted, because a reset cannot
    be distinguished from a reset-plus-rainfall after the fact.
    """
    if field_data.shape[0] < 2:
        raise ConversionError(
            "A cumulative series needs at least two time steps to yield one "
            "interval."
        )

    increments = np.diff(np.nan_to_num(field_data, nan=0.0), axis=0)
    clipped = int(np.count_nonzero(increments < -1e-9))
    if clipped:
        LOGGER.warning(
            "De-accumulation produced %s negative increments (%.3f%% of "
            "values), which means the accumulator resets during the series. "
            "They are clipped to zero; rainfall falling in the same step as a "
            "reset is lost.",
            format(clipped, ","), 100.0 * clipped / increments.size,
        )
        increments = np.clip(increments, 0.0, None)

    LOGGER.info(
        "De-accumulation : %d run-totals -> %d increments; the first timestamp "
        "(%s) carries no interval and is dropped",
        field_data.shape[0], increments.shape[0], times[0],
    )
    return increments, list(times[1:]), clipped


def detect_spinup_artifact(
    field_data: np.ndarray, times: Sequence[datetime], will_drop: bool = False
) -> bool:
    """Report whether the first record is a de-accumulation artifact.

    With no earlier sample to subtract from, ``value[0]`` holds everything
    accumulated before the first output time, including spin-up. The signature
    is a wet first step followed immediately by a long dry spell.
    """
    totals = np.array([np.nansum(step) for step in field_data])
    if totals.size < 7 or totals[0] <= 0:
        return False

    tail = totals[1:]
    non_zero_tail = tail[tail > 0]
    if non_zero_tail.size == 0:
        return False

    dry_steps_after = int(np.sum(tail[:6] == 0))
    if dry_steps_after < 4 or totals[0] <= np.median(non_zero_tail):
        return False

    LOGGER.warning(
        "First record (%s) carries %.3f mm domain-mean but %d of the following "
        "6 steps are dry. This is the signature of a de-accumulation artifact: "
        "the first record holds everything accumulated before the first output "
        "time, not one interval's worth. %s",
        times[0], float(np.nanmean(field_data[0])), dry_steps_after,
        "Dropping it, as requested." if will_drop
        else "Use --drop-first-step to exclude it.",
    )
    return True


# ---------------------------------------------------------------------------
# Spatial registration
# ---------------------------------------------------------------------------

def _wkt1(crs: Any) -> str:
    """Render a CRS as WKT1, which is what HEC-DSS expects."""
    from pyproj.enums import WktVersion

    return crs.to_wkt(WktVersion.WKT1_GDAL)


def build_target_grid(
    source: SourceGrid,
    source_epsg: int,
    target_epsg: int,
    cell_size: float,
    boundary_samples: int = 200,
) -> TargetGrid:
    """Define a projected grid covering the source domain.

    HEC-DSS stores a grid origin as an integer cell index, so the origin is
    representable only if it falls on a whole multiple of the cell size::

        x_origin = x_coord_of_grid_cell_zero + lower_left_cell_x * cell_size

    The bounds are therefore snapped outward to whole multiples, making the
    index exact by construction, and the residual is asserted rather than
    assumed.
    """
    from pyproj import CRS, Transformer

    source_crs = CRS.from_epsg(source_epsg)
    target_crs = CRS.from_epsg(target_epsg)
    if not target_crs.is_projected:
        raise ConversionError(
            f"EPSG:{target_epsg} is not a projected CRS. Gridded precipitation "
            f"must be registered in linear map units to align with the mesh."
        )

    forward = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    inverse = Transformer.from_crs(target_crs, source_crs, always_xy=True)

    edge = np.linspace(0.0, 1.0, boundary_samples)
    boundary_lon = np.concatenate([
        source.west + edge * (source.east - source.west),
        np.full(boundary_samples, source.east),
        source.east - edge * (source.east - source.west),
        np.full(boundary_samples, source.west),
    ])
    boundary_lat = np.concatenate([
        np.full(boundary_samples, source.south),
        source.south + edge * (source.north - source.south),
        np.full(boundary_samples, source.north),
        source.north - edge * (source.north - source.south),
    ])
    projected_x, projected_y = forward.transform(boundary_lon, boundary_lat)

    x_min = float(np.floor(projected_x.min() / cell_size) * cell_size)
    x_max = float(np.ceil(projected_x.max() / cell_size) * cell_size)
    y_min = float(np.floor(projected_y.min() / cell_size) * cell_size)
    y_max = float(np.ceil(projected_y.max() / cell_size) * cell_size)

    nx = int(round((x_max - x_min) / cell_size))
    ny = int(round((y_max - y_min) / cell_size))

    centres_x = x_min + (np.arange(nx) + 0.5) * cell_size
    centres_y = y_min + (np.arange(ny) + 0.5) * cell_size
    mesh_x, mesh_y = np.meshgrid(centres_x, centres_y)
    sample_lon, sample_lat = inverse.transform(mesh_x.ravel(), mesh_y.ravel())

    grid = TargetGrid(
        nx=nx, ny=ny, cell_size=float(cell_size),
        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
        lower_left_cell_x=int(round(x_min / cell_size)),
        lower_left_cell_y=int(round(y_min / cell_size)),
        epsg=target_epsg, crs_name=target_crs.name, wkt=_wkt1(target_crs),
        sample_latitude=sample_lat.reshape(ny, nx),
        sample_longitude=sample_lon.reshape(ny, nx),
    )

    residual_x, residual_y = grid.index_residual()
    if residual_x > 1e-6 or residual_y > 1e-6:
        raise ConversionError(
            f"The grid origin does not fall on an exact cell multiple "
            f"(residual {residual_x:.9f} m, {residual_y:.9f} m). Refusing to "
            f"write a misregistered grid."
        )

    LOGGER.info("Target CRS      : EPSG:%d (%s)", target_epsg, target_crs.name)
    LOGGER.info("Cell size       : %.1f m", cell_size)
    LOGGER.info("Grid size       : %d x %d cells", nx, ny)
    LOGGER.info("X extent        : %.1f to %.1f m", x_min, x_max)
    LOGGER.info("Y extent        : %.1f to %.1f m", y_min, y_max)
    LOGGER.info("Cell index      : (%d, %d), residual %.2e m",
                grid.lower_left_cell_x, grid.lower_left_cell_y,
                max(residual_x, residual_y))
    return grid


def _build_sampler(
    source: SourceGrid, target: TargetGrid
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the time-invariant sampling geometry.

    An interpolator over a latitude/longitude grid is defined only between the
    outermost cell centres, while the field extends half a cell further, to the
    cell edges. Sampling without compensation would discard that rim and lose
    the corresponding water volume. Coordinates are therefore clamped to the
    centre hull, which is nearest-neighbour extrapolation across half a cell and
    is exact for a piecewise-constant grid cell. Points beyond the true
    cell-edge boundary remain outside and are masked.
    """
    sample_lat = target.sample_latitude
    sample_lon = target.sample_longitude
    inside = (
        (sample_lon >= source.west) & (sample_lon <= source.east)
        & (sample_lat >= source.south) & (sample_lat <= source.north)
    )
    points = np.column_stack([
        np.clip(sample_lat, source.latitude[0], source.latitude[-1]).ravel(),
        np.clip(sample_lon, source.longitude[0], source.longitude[-1]).ravel(),
    ])
    return points, inside


@dataclass(frozen=True)
class Variogram:
    """Isotropic exponential semivariogram, with distances in metres.

    ``gamma(h) = nugget + sill * (1 - exp(-3h / range))``

    The factor of three makes ``range`` the practical range, the distance at
    which the model reaches 95 % of its sill, which is the convention the
    geostatistical literature and GSLIB use.
    """

    nugget: float
    sill: float
    practical_range: float
    fitted: bool

    def __call__(self, distance: np.ndarray) -> np.ndarray:
        gamma = self.nugget + self.sill * (
            1.0 - np.exp(-3.0 * np.asarray(distance, dtype="float64")
                         / self.practical_range))
        return np.where(np.asarray(distance) == 0.0, 0.0, gamma)


@dataclass(frozen=True)
class KrigingPlan:
    """Neighbour indices and ordinary-kriging weights per target cell.

    Both depend only on geometry and on the variogram, so they are built once
    and reused for every time step: each step is then one weighted sum.
    """

    indices: np.ndarray                       # (targets, neighbours)
    weights: np.ndarray                       # (targets, neighbours)
    variogram: Variogram
    singular: int


def _fit_variogram(
    coordinates: np.ndarray, values: np.ndarray, extent: float,
) -> Variogram:
    """Fit an exponential semivariogram to a reference field.

    The reference field is the event total rather than one time step, because
    a single step can be almost empty and its variogram then describes noise.

    A spatially uniform field - every design storm in this study - has zero
    variance and no variogram to fit. Kriging is still well defined there:
    with equal values at every source cell the weights, which sum to one,
    return that value whatever the model. A nominal model is used so the
    system stays non-singular, and ``fitted`` records that it was not
    estimated from data.
    """
    finite = np.isfinite(values)
    present = values[finite]
    spread = float(np.nanvar(present)) if finite.any() else 0.0
    level = float(np.nanmean(np.abs(present))) if finite.any() else 0.0
    nominal = Variogram(nugget=0.0, sill=1.0,
                        practical_range=max(extent / 3.0, 1.0), fitted=False)
    # A uniform field is uniform to within float rounding, not exactly: the
    # five design storms in this study carry the same depth in every cell, and
    # their variance comes out at 1e-13 of the mean rather than at zero.
    # Testing against zero therefore sent them into a fit whose sill had no
    # scale, which failed on its own bounds and logged a warning on every run.
    if finite.sum() < 8 or spread <= 1e-20 * max(level * level, 1.0):
        LOGGER.info("Variogram        : field is spatially uniform "
                    "(variance %.3g); using a nominal model", spread)
        return nominal

    points = coordinates[finite]
    sample = values[finite]
    count = points.shape[0]
    generator = np.random.default_rng(0)
    pairs = min(KRIGING_VARIOGRAM_PAIRS, count * (count - 1) // 2)
    left = generator.integers(0, count, size=pairs)
    right = generator.integers(0, count, size=pairs)
    keep = left != right
    left, right = left[keep], right[keep]

    separation = np.hypot(points[left, 0] - points[right, 0],
                          points[left, 1] - points[right, 1])
    squared = 0.5 * (sample[left] - sample[right]) ** 2

    limit = separation.max() / 2.0
    if not np.isfinite(limit) or limit <= 0.0:
        return nominal
    edges = np.linspace(0.0, limit, KRIGING_VARIOGRAM_BINS + 1)
    slot = np.digitize(separation, edges) - 1
    centres, gammas, counts = [], [], []
    for index in range(KRIGING_VARIOGRAM_BINS):
        chosen = slot == index
        if chosen.sum() >= 30:
            centres.append(0.5 * (edges[index] + edges[index + 1]))
            gammas.append(float(squared[chosen].mean()))
            counts.append(int(chosen.sum()))
    if len(centres) < 4:
        return nominal

    centres = np.asarray(centres)
    gammas = np.asarray(gammas)
    counts = np.asarray(counts, dtype="float64")

    def model(distance, nugget, sill, practical_range):
        return nugget + sill * (1.0 - np.exp(-3.0 * distance
                                             / practical_range))

    try:
        from scipy.optimize import curve_fit
        lower = [0.0, 1e-12, float(centres.min())]
        upper = [float(gammas.max()) + 1e-9,
                 10.0 * float(gammas.max()) + 1.0,
                 10.0 * float(centres.max())]
        # curve_fit rejects a guess on the wrong side of a bound, which is how
        # a field with a vanishing sill used to raise instead of fitting
        guess = [min(max(value, low), high) for value, low, high
                 in zip([0.0, float(gammas.max()), float(centres.mean())],
                        lower, upper)]
        bounds = (lower, upper)
        # weighted by pair count: the near bins carry the shape that matters
        # for interpolation, and they are also the best populated
        parameters, _ = curve_fit(model, centres, gammas, p0=guess,
                                  bounds=bounds, sigma=1.0 / np.sqrt(counts),
                                  maxfev=20000)
    except Exception as error:                        # pragma: no cover
        LOGGER.warning("Variogram fit failed (%s); using a nominal model",
                       error)
        return nominal

    nugget, sill, practical_range = (float(value) for value in parameters)
    LOGGER.info(
        "Variogram        : exponential, nugget %.4g, sill %.4g, "
        "practical range %.0f m", nugget, sill, practical_range)
    return Variogram(nugget=nugget, sill=sill,
                     practical_range=practical_range, fitted=True)


def _build_kriging(
    source: SourceGrid,
    target: TargetGrid,
    reference: np.ndarray,
    source_epsg: int,
) -> KrigingPlan:
    """Solve the ordinary-kriging system once, for every target cell.

    Kriging is carried out in the projected plane, not in degrees. A degree of
    longitude is shorter than a degree of latitude by the cosine of the
    latitude, so an isotropic model fitted in degrees would be anisotropic on
    the ground - at 40.4 N by a factor of 0.76. Working in metres removes that
    distortion, and it is also why kriging is not restricted, as the
    RegularGridInterpolator methods are, to the axis-aligned source grid.

    For target cell ``t`` with neighbours ``i, j``::

        [ gamma(|x_i - x_j|)   1 ] [ w ]   [ gamma(|x_i - t|) ]
        [        1'            0 ] [ mu ] = [        1        ]

    The Lagrange multiplier ``mu`` enforces unbiasedness by making the weights
    sum to one, which is what distinguishes ordinary kriging from simple
    kriging and from inverse-distance weighting.
    """
    from pyproj import CRS, Transformer
    from scipy.spatial import cKDTree

    forward = Transformer.from_crs(CRS.from_epsg(source_epsg),
                                  CRS.from_epsg(target.epsg), always_xy=True)
    mesh_lon, mesh_lat = np.meshgrid(source.longitude, source.latitude)
    source_x, source_y = forward.transform(mesh_lon.ravel(), mesh_lat.ravel())
    coordinates = np.column_stack([source_x, source_y])

    centres_x = target.x_min + (np.arange(target.nx) + 0.5) * target.cell_size
    centres_y = target.y_min + (np.arange(target.ny) + 0.5) * target.cell_size
    grid_x, grid_y = np.meshgrid(centres_x, centres_y)
    targets = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    extent = float(np.hypot(np.ptp(coordinates[:, 0]),
                            np.ptp(coordinates[:, 1])))
    variogram = _fit_variogram(coordinates, np.asarray(reference,
                                                       dtype="float64").ravel(),
                               extent)

    neighbours = int(min(KRIGING_NEIGHBOURS, coordinates.shape[0]))
    tree = cKDTree(coordinates)
    _, indices = tree.query(targets, k=neighbours, workers=-1)
    indices = np.atleast_2d(indices)

    picked = coordinates[indices]                       # (targets, k, 2)
    separation = np.linalg.norm(picked[:, :, None, :] - picked[:, None, :, :],
                                axis=-1)
    left = np.empty((targets.shape[0], neighbours + 1, neighbours + 1))
    left[:, :neighbours, :neighbours] = variogram(separation)
    left[:, :neighbours, neighbours] = 1.0
    left[:, neighbours, :neighbours] = 1.0
    left[:, neighbours, neighbours] = 0.0
    diagonal = np.arange(neighbours)
    left[:, diagonal, diagonal] += KRIGING_JITTER

    to_target = np.linalg.norm(picked - targets[:, None, :], axis=-1)
    right = np.empty((targets.shape[0], neighbours + 1))
    right[:, :neighbours] = variogram(to_target)
    right[:, neighbours] = 1.0

    singular = 0
    try:
        solution = np.linalg.solve(left, right[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        solution = np.empty_like(right)
        for row in range(targets.shape[0]):
            try:
                solution[row] = np.linalg.solve(left[row], right[row])
            except np.linalg.LinAlgError:
                solution[row], *_ = np.linalg.lstsq(left[row], right[row],
                                                    rcond=None)
                singular += 1
        if singular:
            LOGGER.warning("Kriging system singular at %d of %d target cells; "
                           "solved by least squares", singular,
                           targets.shape[0])

    weights = solution[:, :neighbours]
    LOGGER.info("Kriging          : %d neighbours per cell, weights sum to "
                "%.6f on average", neighbours, float(weights.sum(axis=1).mean()))
    return KrigingPlan(indices=indices, weights=weights, variogram=variogram,
                       singular=singular)


def _resample(
    field_data: np.ndarray,
    source: SourceGrid,
    target: TargetGrid,
    method: str,
    sampler: tuple[np.ndarray, np.ndarray],
    kriging: KrigingPlan | None = None,
) -> np.ndarray:
    """Resample one slice onto the target grid by inverse mapping."""
    points, inside = sampler

    if method == "kriging":
        if kriging is None:
            raise ConversionError(
                "Kriging was requested but no kriging plan was built.")
        values = np.asarray(field_data, dtype="float64").ravel()
        gathered = values[kriging.indices]
        estimate = np.einsum("tk,tk->t", kriging.weights, gathered)
        resampled = estimate.reshape(target.ny, target.nx)
        # a neighbourhood made entirely of missing cells cannot be estimated
        missing = np.isnan(gathered).all(axis=1).reshape(target.ny, target.nx)
        resampled = np.where(missing, np.nan, resampled)
    else:
        from scipy.interpolate import RegularGridInterpolator

        interpolator = RegularGridInterpolator(
            (source.latitude, source.longitude), field_data,
            method=method, bounds_error=False, fill_value=np.nan,
        )
        resampled = interpolator(points).reshape(target.ny, target.nx)

    resampled = np.where(inside, resampled, np.nan)
    return np.where(np.isnan(resampled), np.nan, np.maximum(resampled, 0.0))


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(
    nc_file: str | os.PathLike[str],
    dss_file: str | os.PathLike[str],
    *,
    json_file: str | os.PathLike[str] | None = None,
    backend: str = "auto",
    interval_convention: str = "auto",
    assume_units: str | None = None,
    accumulation: str | None = None,
    drop_first_step: bool = False,
    source_epsg: int = DEFAULT_SOURCE_EPSG,
    target_epsg: int = DEFAULT_TARGET_EPSG,
    cell_size: float = DEFAULT_CELL_SIZE_M,
    resample_method: str = DEFAULT_RESAMPLE_METHOD,
    dry_run: bool = False,
) -> ConversionResult:
    """Convert a NetCDF precipitation field to HEC-DSS gridded records.

    Parameters
    ----------
    nc_file, dss_file
        Input and output paths. An existing DSS file is replaced.
    json_file
        Provenance document. Defaults to ``<dss_file>.window.json``.
    backend
        ``'auto'``, ``'netcdf4'`` or ``'h5py'``.
    interval_convention
        ``'auto'`` to read it from the metadata, or ``'end'`` / ``'begin'``.
    assume_units
        Override the declared units, bypassing the plausibility check.
    drop_first_step
        Exclude the first record, which is not a valid interval total when the
        series was produced by differencing a run-total.
    source_epsg, target_epsg
        CRS of the input coordinates, and of the output grid. The target must
        match the hydraulic model.
    cell_size
        Target cell size in the linear units of ``target_epsg``.
    resample_method
        ``'linear'`` or ``'nearest'``.
    dry_run
        Perform every computation and check, but write nothing.

    Raises
    ------
    ConversionError
        If the input cannot be interpreted unambiguously, or a check fails.
    """
    nc_path = Path(nc_file)
    dss_path = Path(dss_file)
    json_path = Path(json_file) if json_file else Path(str(dss_path) + ".window.json")

    # The DSS writer is only reached at the very end. Check for it now, so a
    # missing package costs a second rather than a full resampling pass.
    if not dry_run:
        import importlib.util

        if importlib.util.find_spec("hecdss") is None:
            raise ConversionError(
                "The 'hecdss' package is required to write DSS files but is "
                "not available to %s. Install it with: pip install hecdss"
                % sys.executable
            )

    LOGGER.info("Input NetCDF    : %s", nc_path)
    LOGGER.info("Output DSS      : %s", dss_path)
    LOGGER.info("Provenance JSON : %s", json_path)

    with open_dataset(nc_path, backend) as dataset:
        backend_name = dataset.name

        precip_path = dataset.locate(PRECIP_VARIABLE_NAMES,
                                     "precipitation variable")
        time_path = dataset.locate(TIME_VARIABLE_NAMES, "time variable")
        lat_path = dataset.locate(LATITUDE_VARIABLE_NAMES, "latitude coordinate")
        lon_path = dataset.locate(LONGITUDE_VARIABLE_NAMES,
                                  "longitude coordinate")

        precip_attrs = dataset.attributes(precip_path)
        declared_units = precip_attrs.get("units")
        LOGGER.info("Precipitation   : '%s', shape %s",
                    precip_path, dataset.shape(precip_path))

        if len(dataset.shape(precip_path)) != 3:
            raise ConversionError(
                f"Precipitation must be three-dimensional (time, latitude, "
                f"longitude); found {len(dataset.shape(precip_path))} "
                f"dimensions."
            )

        # A rotated-pole or curvilinear grid stores two-dimensional
        # coordinates. Catch that here, before axis resolution reports the
        # symptom instead of the cause.
        lat_shape = dataset.shape(lat_path)
        lon_shape = dataset.shape(lon_path)
        if len(lat_shape) != 1 or len(lon_shape) != 1:
            raise ConversionError(
                f"Coordinates '{lat_path}' and '{lon_path}' are "
                f"{len(lat_shape)}-D and {len(lon_shape)}-D, not 1-D. The file "
                f"is on a rotated-pole or otherwise curvilinear grid, as used "
                f"by CORDEX and by native WRF output. Regrid it to a regular "
                f"latitude/longitude mesh first; this tool will not "
                f"approximate a uniform spacing."
            )

        axis_order, axis_source = resolve_axis_order(
            dataset, precip_path, time_path, lat_path, lon_path
        )
        precipitation = np.transpose(dataset.read(precip_path), axis_order)

        time_axis = read_time_axis(dataset, time_path)
        source = read_source_grid(dataset, lat_path, lon_path)
        detected_convention, evidence = detect_interval_convention(
            dataset, [precip_path, time_path]
        )

    if precipitation.shape[0] != len(time_axis):
        raise ConversionError(
            f"The time axis has {len(time_axis)} entries but the field has "
            f"{precipitation.shape[0]} slices."
        )
    if precipitation.shape[1:] != source.shape:
        raise ConversionError(
            f"Field shape {precipitation.shape[1:]} does not match the "
            f"coordinate grid {source.shape}."
        )

    if source.latitude_reversed:
        precipitation = precipitation[:, ::-1, :]
    if source.longitude_reversed:
        precipitation = precipitation[:, :, ::-1]

    detected_accumulation = classify_accumulation(precipitation)
    if accumulation in (None, "", "auto"):
        raise ConversionError(
            "How the series stores depth is not declared. Period-cumulative "
            "DSS records expect the depth that fell in each interval, so a "
            "run-total has to be differenced first and an incremental series "
            "must not be.\n"
            "        " + accumulation_evidence(precipitation) + "\n"
            "        Set accumulation in the [rainfall] section of config.ini "
            "to 'increments' or 'cumulative', or pass --accumulation. There "
            "is deliberately no automatic choice: a series that is constant "
            "in time is a uniform hyetograph and a stalled run-total at the "
            "same time, and only you know which."
        )
    if accumulation not in ("increments", "cumulative"):
        raise ConversionError(
            f"Unknown accumulation mode {accumulation!r}. Use increments or "
            f"cumulative."
        )

    # The declaration decides; the classifier only checks it. A run-total
    # written as increments restates the whole storm at every step, so that
    # contradiction is refused rather than warned about.
    treat_as_cumulative = accumulation == "cumulative"
    if not treat_as_cumulative and detected_accumulation == "cumulative":
        raise ConversionError(
            "Declared as increments, but the series holds cumulative "
            "run-totals.\n"
            "        " + accumulation_evidence(precipitation) + "\n"
            "        Writing it unchanged would restate the whole storm at "
            "every timestep. Set accumulation = cumulative to difference it, "
            "or difference it yourself before converting."
        )
    if treat_as_cumulative and detected_accumulation == "increments":
        LOGGER.warning(
            "Declared as cumulative, but the series falls at some cells, "
            "which a run-total never does. Differencing it anyway; any "
            "negative increments will be clipped and counted.")
    if treat_as_cumulative and detected_accumulation == "constant":
        LOGGER.warning(
            "Declared as cumulative, but the series is constant in time, so "
            "differencing it yields no rain at all. Check that this is the "
            "window you meant.")

    if interval_convention == "auto":
        if detected_convention is None and treat_as_cumulative:
            # Differencing value[i] - value[i - 1] yields the depth that fell
            # over (times[i - 1], times[i]], and that is the timestamp kept,
            # so the convention follows from the operation and need not be
            # read from metadata.
            convention = "end"
            evidence = "implied by differencing a cumulative run-total"
            LOGGER.info("Interval conv.  : 'end' (%s)", evidence)
        elif detected_convention is None:
            raise ConversionError(
                f"Could not determine whether timestamps mark the start or the "
                f"end of their accumulation interval ({evidence}). Pass "
                f"interval_convention='end' or 'begin' explicitly. 'end' is "
                f"correct for a series produced by differencing a cumulative "
                f"run-total."
            )
        else:
            convention = detected_convention
            LOGGER.info("Interval conv.  : '%s' (%s)", convention, evidence)
    else:
        convention = interval_convention
        LOGGER.info("Interval conv.  : '%s' (declared by caller)", convention)
        if detected_convention and detected_convention != convention:
            LOGGER.warning("Metadata suggests '%s' (%s) but '%s' was declared",
                           detected_convention, evidence, convention)

    times = list(time_axis.times)
    negative_clipped = 0
    if treat_as_cumulative:
        precipitation, times, negative_clipped = deaccumulate(
            precipitation, times)

    spinup_detected = detect_spinup_artifact(
        precipitation, times, will_drop=drop_first_step)
    if drop_first_step:
        if not spinup_detected:
            LOGGER.warning(
                "--drop-first-step was given, but the first record shows no "
                "de-accumulation signature. A valid record is being discarded."
            )
        LOGGER.info("Dropping the first record (%s)", times[0])
        precipitation = precipitation[1:]
        times = times[1:]

    intervals, timestep = build_intervals(times, convention)
    simulation_start = intervals[0][0]
    simulation_end = intervals[-1][1]

    LOGGER.info("Timestep        : %s", timestep)
    LOGGER.info("Records         : %d", len(intervals))
    LOGGER.info("Window          : %s to %s", simulation_start, simulation_end)

    peak_raw = float(np.nanmax(precipitation)) if precipitation.size else 0.0
    units = resolve_units(declared_units, peak_raw, timestep, assume_units)
    if units.factor_to_mm != 1.0:
        LOGGER.info("Converting to millimetres (x%g)", units.factor_to_mm)
        precipitation = precipitation * units.factor_to_mm

    target = build_target_grid(source, source_epsg, target_epsg, cell_size)
    north_south, east_west = source.cell_dimensions_metres()
    LOGGER.info(
        "Source cell     : %.0f m north-south by %.0f m east-west (ratio %.3f)",
        north_south, east_west, east_west / north_south,
    )

    if resample_method not in RESAMPLE_METHODS:
        raise ConversionError(
            f"Unknown resampling method {resample_method!r}. Choose one of: "
            f"{', '.join(RESAMPLE_METHODS)}."
        )

    LOGGER.info("Resampling with '%s' interpolation", resample_method)
    sampler = _build_sampler(source, target)
    kriging = None
    if resample_method == "kriging":
        # the variogram is fitted to the event total, so it describes the
        # spatial structure of the storm rather than that of one time step
        reference = np.nansum(np.where(np.isnan(precipitation), 0.0,
                                       precipitation), axis=0)
        kriging = _build_kriging(source, target, reference, source_epsg)
    source_areas = source.cell_areas()

    resampled: list[np.ndarray] = []
    source_volume = 0.0
    target_volume = 0.0
    for index, slice_data in enumerate(precipitation):
        cleaned = np.where(np.isnan(slice_data), 0.0, slice_data)
        projected = _resample(cleaned, source, target, resample_method,
                              sampler, kriging)
        resampled.append(projected)
        source_volume += float(np.sum(cleaned / 1000.0 * source_areas))
        target_volume += float(
            np.sum(np.nan_to_num(projected, nan=0.0) / 1000.0 * target.cell_area)
        )
        if (index + 1) % 20 == 0 or index + 1 == len(precipitation):
            LOGGER.debug("Resampled %d/%d", index + 1, len(precipitation))

    outside_fraction = float(np.mean(np.isnan(resampled[0])))
    drift = (target_volume - source_volume) / source_volume if source_volume else 0.0

    LOGGER.info("Source volume   : %.1f thousand m3", source_volume / 1e3)
    LOGGER.info("Target volume   : %.1f thousand m3", target_volume / 1e3)
    LOGGER.info("Volume drift    : %+.3f%%", drift * 100.0)
    LOGGER.info("Outside source  : %.2f%% of target cells (written as null)",
                outside_fraction * 100.0)
    if abs(drift) > VOLUME_DRIFT_WARNING_THRESHOLD:
        LOGGER.warning(
            "Volume drift exceeds %.0f%%. Review the cell size and resampling \n"
            "method before using these results.",
            VOLUME_DRIFT_WARNING_THRESHOLD * 100.0,
        )

    pathnames = [
        f"///TP/{to_dss_timestamp(start)}/{to_dss_timestamp(end)}//"
        for start, end in intervals
    ]
    LOGGER.info("First pathname  : %s", pathnames[0])
    LOGGER.info("Last pathname   : %s", pathnames[-1])

    result = ConversionResult(
        dss_path=dss_path,
        json_path=None if dry_run else json_path,
        backend=backend_name,
        simulation_start=simulation_start,
        simulation_end=simulation_end,
        timestep=timestep,
        pathnames=pathnames,
        target_grid=target,
        units=units,
        source_volume_m3=source_volume,
        target_volume_m3=target_volume,
        cells_outside_source_fraction=outside_fraction,
        interval_convention=convention,
        interval_convention_evidence=evidence,
        first_step_dropped=drop_first_step,
        spinup_artifact_detected=spinup_detected,
        accumulation_detected=detected_accumulation,
        deaccumulated=treat_as_cumulative,
        negative_increments_clipped=negative_clipped,
        axis_source=axis_source,
    )

    if dry_run:
        LOGGER.info("Dry run: nothing written")
        return result

    _write_dss(dss_path, pathnames, resampled, target)
    result.metadata = _build_metadata(
        result, nc_path=nc_path, precip_variable=precip_path,
        time_units=time_axis.units, source_epsg=source_epsg,
        resample_method=resample_method,
    )
    _write_provenance(json_path, result.metadata)
    return result


def _write_dss(
    dss_path: Path,
    pathnames: Sequence[str],
    fields: Sequence[np.ndarray],
    target: TargetGrid,
) -> None:
    """Write the resampled series as HEC-DSS period-cumulative grids."""
    try:
        from hecdss import HecDss
        from hecdss.gridded_data import GriddedData
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ConversionError(
            "The 'hecdss' package is required to write DSS files. \n"
            "Install it with: pip install hecdss"
        ) from exc

    dss_path.parent.mkdir(parents=True, exist_ok=True)
    if dss_path.exists():
        LOGGER.info("Replacing existing DSS file")
        dss_path.unlink()
    for suffix in (".dsc", ".dsd"):
        stray = Path(str(dss_path) + suffix)
        if stray.exists():
            stray.unlink()

    with HecDss(str(dss_path)) as dss:
        for index, (pathname, values) in enumerate(zip(pathnames, fields)):
            payload = np.where(np.isnan(values), DSS_NULL, values).astype(np.float32)
            record = GriddedData.create(
                path=pathname,
                type=DSS_GRID_TYPE_SPECIFIED,
                dataType=DSS_DATA_TYPE_PER_CUM,
                lowerLeftCellX=target.lower_left_cell_x,
                lowerLeftCellY=target.lower_left_cell_y,
                numberOfCellsX=target.nx,
                numberOfCellsY=target.ny,
                srsDefinitionType=DSS_SRS_TYPE_WKT,
                timeZoneID="UTC",
                isInterval=1,
                isTimeStamped=1,
                dataUnits="MM",
                dataSource="NetCDF",
                srsName=f"EPSG:{target.epsg}",
                srsDefinition=target.wkt,
                cellSize=target.cell_size,
                xCoordOfGridCellZero=0.0,
                yCoordOfGridCellZero=0.0,
                nullValue=float(DSS_NULL),
                data=payload,
            )
            record.update_grid_info()
            dss.put(record)
            if (index + 1) % 20 == 0 or index + 1 == len(pathnames):
                LOGGER.debug("Wrote %d/%d records", index + 1, len(pathnames))

    LOGGER.info("DSS written     : %s (%.2f MB)",
                dss_path, dss_path.stat().st_size / 1024 ** 2)


def _build_metadata(
    result: ConversionResult,
    *,
    nc_path: Path,
    precip_variable: str,
    time_units: str,
    source_epsg: int,
    resample_method: str,
) -> dict[str, Any]:
    """Assemble the provenance record written alongside the DSS file."""
    target = result.target_grid
    return {
        "generated_by": f"{Path(__file__).name} {__version__}",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "backend": result.backend,
        "source_netcdf": str(nc_path.resolve()),
        "dss_file": str(result.dss_path.resolve()),
        "dss_size_bytes": result.dss_path.stat().st_size,
        "simulation_start": result.simulation_start.isoformat(),
        "simulation_end": result.simulation_end.isoformat(),
        "timestep_seconds": int(result.timestep.total_seconds()),
        "n_records": result.record_count,
        "accumulation_convention": result.interval_convention,
        "accumulation_evidence": result.interval_convention_evidence,
        "axis_order_source": result.axis_source,
        "first_step_dropped": result.first_step_dropped,
        "spinup_artifact_detected": result.spinup_artifact_detected,
        "accumulation_detected": result.accumulation_detected,
        "deaccumulated": result.deaccumulated,
        "negative_increments_clipped": result.negative_increments_clipped,
        "time_units": time_units,
        "precip_variable": precip_variable,
        "precip_units_declared": result.units.declared,
        "precip_units_canonical": result.units.canonical,
        "precip_units_factor_to_mm": result.units.factor_to_mm,
        "peak_mm_per_step": result.units.peak_mm,
        "peak_mm_per_hour": result.units.peak_mm_per_hour,
        "units_plausibility": result.units.verdict,
        "source_epsg": source_epsg,
        "target_epsg": target.epsg,
        "target_crs_name": target.crs_name,
        "cell_size_m": target.cell_size,
        "grid_nx": target.nx,
        "grid_ny": target.ny,
        "lowerLeftCellX": target.lower_left_cell_x,
        "lowerLeftCellY": target.lower_left_cell_y,
        "x_origin_m": target.x_min,
        "y_origin_m": target.y_min,
        "x_max_m": target.x_max,
        "y_max_m": target.y_max,
        "resample_method": resample_method,
        "volume_source_m3": result.source_volume_m3,
        "volume_target_m3": result.target_volume_m3,
        "volume_difference_percent": result.volume_drift * 100.0,
        "cells_outside_source_percent": result.cells_outside_source_fraction * 100.0,
        "dss_data_type": "PER-CUM",
        "pathnames": list(result.pathnames),
    }


def _write_provenance(json_path: Path, metadata: dict[str, Any]) -> None:
    """Write the provenance document beside the DSS file."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    LOGGER.info("Provenance      : %s", json_path)


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert NetCDF gridded precipitation to HEC-DSS.",
        allow_abbrev=False,
    )
    parser.add_argument("--nc-file", "--nc_file", dest="nc_file",
                        default=DEFAULT_NC_FILE, help="Input NetCDF file.")
    parser.add_argument("--dss-file", "--dss_file", dest="dss_file",
                        default=DEFAULT_DSS_FILE, help="Output HEC-DSS file.")
    parser.add_argument("--json-file", "--json_file", dest="json_file",
                        default=None,
                        help="Provenance document "
                             "(default: <dss-file>.window.json).")
    parser.add_argument("--backend", choices=("auto", "netcdf4", "h5py"),
                        default="auto",
                        help="Reader to use. 'auto' prefers netCDF4.")
    parser.add_argument("--interval-convention", "--accum-convention",
                        dest="interval_convention",
                        choices=("auto", "end", "begin"),
                        default=DEFAULT_INTERVAL_CONVENTION,
                        help="Whether a timestamp marks the end or the start "
                             "of its accumulation interval.")
    parser.add_argument("--assume-units", default=None,
                        help="Override the declared units (mm, cm, m or in), "
                             "bypassing the plausibility check.")
    parser.add_argument("--accumulation", default=DEFAULT_ACCUMULATION,
                        help="How the series stores depth: increments or "
                             "cumulative. Required; there is no automatic "
                             "choice. Read from config.ini when set there.")
    parser.add_argument("--drop-first-step", action="store_true",
                        help="Exclude the first record, which is not a valid "
                             "interval total for a de-accumulated series.")
    parser.add_argument("--source-epsg", type=int, default=DEFAULT_SOURCE_EPSG,
                        help="CRS the NetCDF coordinates are interpreted in.")
    parser.add_argument("--target-epsg", type=int, default=DEFAULT_TARGET_EPSG,
                        help="CRS to write the grid in; must match the model.")
    parser.add_argument("--cell-size", type=float, default=DEFAULT_CELL_SIZE_M,
                        help="Target cell size in target CRS units.")
    parser.add_argument("--resample", choices=RESAMPLE_METHODS,
                        default=DEFAULT_RESAMPLE_METHOD,
                        help="Interpolation method.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run every check and computation, write nothing.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Include per-record progress.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Warnings and errors only.")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose, args.quiet)

    for value, flag in ((args.nc_file, "--nc-file"),
                        (args.dss_file, "--dss-file"),
                        (args.target_epsg, "--target-epsg")):
        if value is None:
            LOGGER.error("%s was not given and is blank in config.ini.", flag)
            return 1

    try:
        result = convert(
            args.nc_file, args.dss_file,
            json_file=args.json_file,
            backend=args.backend,
            interval_convention=args.interval_convention,
            assume_units=args.assume_units,
            accumulation=args.accumulation,
            drop_first_step=args.drop_first_step,
            source_epsg=args.source_epsg,
            target_epsg=args.target_epsg,
            cell_size=args.cell_size,
            resample_method=args.resample,
            dry_run=args.dry_run,
        )
    except ConversionError as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        LOGGER.error("Interrupted")
        return 130

    LOGGER.info("Done. %d records covering %s to %s",
                result.record_count, result.simulation_start,
                result.simulation_end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
