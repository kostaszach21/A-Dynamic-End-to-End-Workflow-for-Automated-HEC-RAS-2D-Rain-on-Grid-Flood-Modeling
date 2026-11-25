# HEC-RAS 2D rain-on-grid pipeline

Gridded precipitation in, flood hazard maps out, with no interactive step in
between. The chain reads a NetCDF rainfall field, writes it as a projected
HEC-DSS grid, verifies that the grid is registered where it claims to be, runs
the HEC-RAS 2D unsteady solver, and renders the result as maps.

Nothing in the code is specific to a site or a dataset. The simulation window,
the time step, the grid geometry and origin, the coordinate registration, the
units and packing, and the accumulation convention are all resolved from the
input file at run time, so a different storm or a different catchment needs a
new `config.ini` rather than modified source.

```
  Design or forecast rainfall          config.ini
      gridded NetCDF              every path and every option
           |                               |
           v                               v
  netcdf_to_dss.py  ->  verify_rainfall_grid.py  ->  hecras_run.py  ->  flood_maps.py
           |                    |                        |                  |
       storm.dss          registration             results HDF          five maps
   .window.json              figure              stored-map GeoTIFF      PNG
```

---

## The scripts

### `run_pipeline.py` — the orchestrator

Chains the four stages in order and stops at the first failure. It reads every
path from `config.ini`, checks that the required entries are present before
anything runs, and confirms that the HEC-RAS results are newer than the DSS
file before mapping them, so a crashed solve cannot be mapped as though it had
succeeded.

| option | effect |
|---|---|
| `--from-step {1,2,3,4}` | start partway through, when earlier outputs already exist |
| `--skip-compute` | prepare the HEC-RAS inputs but do not run the solver |
| `--no-verify` | skip the registration figure |
| `--allow-stale-results` | map results that predate this run |
| `--label NAME` | suffix added to the map file names |

### `netcdf_to_dss.py` — rainfall to model input

The substantial part of the pipeline. It locates the precipitation, time,
latitude and longitude variables by name, resolves the axis order, converts
the units to millimetres, determines whether the series holds per-interval
increments or a running total, reprojects the field onto the model's
coordinate reference system, and writes it as HEC-DSS period-cumulative grid
records.

Three things it does rather than assume:

- **The grid origin is snapped to an exact cell index.** HEC-DSS stores the
  origin as an integer multiple of the cell size, not as a coordinate. A
  fractional origin would be written misregistered, so the bounds are snapped
  outward to whole multiples and the residual is asserted, not hoped for.
- **Water volume is audited across the reprojection.** Source volume is
  computed from the exact spherical area of each source cell and compared
  against the volume written; a drift above 2 % raises a warning.
- **A provenance record is written beside the DSS file** (`.window.json`),
  holding the window, the time step, the grid definition, the resampling
  method and the volume audit. Later stages read it rather than re-deriving
  anything.

### `verify_rainfall_grid.py` — independent registration check

Rebuilds the rainfall grid from the provenance file alone and confirms that it
lands where the converter said it would, that the origin residual is zero, and
that the model terrain lies inside the rainfall extent. It produces a figure
showing the rainfall grid, the terrain footprint and their overlap, which is
the artefact to inspect when a run looks misaligned.

The check is independent by construction: it never reads the converter's
in-memory state, only the file it wrote.

### `hecras_run.py` — set up and solve

Backs up the plan and unsteady flow files, rewrites their simulation window
and time step to match the rainfall series, points the plan at the DSS file,
computes through `ras-commander`, and reads the HEC-RAS compute messages back
so that a failed solve is reported as a failure rather than as a silent
absence of results.

It also validates continuity before computing: the declared window, the number
of records and the time step must agree with one another, and the DSS file it
is about to use must be the same file, with the same size, that the provenance
record describes.

### `flood_maps.py` — results to figures

Locates the stored-map rasters HEC-RAS writes during the run and renders five
maps on **fixed class intervals**, so that any two runs can be compared
directly rather than each being stretched to its own range.

| map | quantity |
|---|---|
| depth | maximum water depth |
| velocity | maximum flow velocity |
| hazard | depth × velocity |
| arrival time | when water first exceeds the dry threshold |
| extent | every cell that is ever wet |

Rasters are block-reduced by **maximum** rather than by mean before plotting,
which keeps narrow flow paths visible instead of averaging them away against
their own banks.

---

## Quick start

```bash
git clone https://github.com/kostaszach21/A-Dynamic-End-to-End-Workflow-for-Automated-HEC-RAS-2D-Rain-on-Grid-Flood-Modeling.git
cd A-Dynamic-End-to-End-Workflow-for-Automated-HEC-RAS-2D-Rain-on-Grid-Flood-Modeling
pip install -r requirements.txt
cp config.example.ini config.ini
# edit config.ini
python run_pipeline.py
```

Each stage also runs on its own, with command-line arguments overriding
`config.ini`:

```bash
python netcdf_to_dss.py --nc-file rain.nc --dss-file storm.dss \
    --accumulation increments --target-epsg 2100 --cell-size 1000 --dry-run
```

`--dry-run` performs every check and every computation and writes nothing,
which is the fastest way to find out whether an input will be accepted.

---

## What NetCDF it accepts

**Structure.** A three-dimensional precipitation variable on a regular
latitude/longitude grid, with one time axis. The axis order is detected, not
assumed. NetCDF-3 and NetCDF-4 are both read; `netCDF4` is preferred and
`h5py` is the fallback, which reads NetCDF-4 only.

**Variable names**, matched case-sensitively against these sets:

| axis | accepted names |
|---|---|
| precipitation | `tp`, `precip`, `precipitation`, `pr`, `RAINC`, `RAINNC`, `rain`, `rainfall`, `Tot_Precip` |
| time | `time`, `Time`, `TIME`, `times` |
| latitude | `latitude`, `lat`, `XLAT`, `y`, `RLAT` |
| longitude | `longitude`, `lon`, `XLONG`, `x`, `RLON` |

**Units.** Depth units only: `mm`, `cm`, `m`, `in` and their spelled-out
forms. A rate or flux such as `kg m-2 s-1` is **rejected, not converted**,
because interpreting one requires an assumption about the averaging period
that belongs to the person preparing the data. `--assume-units` overrides an
absent or wrong declaration, and bypasses the plausibility check with it.

**Time.** A CF-style `units` attribute on the time variable
(`hours since 2023-06-15 00:00:00`). Whether a timestamp marks the **end** or
the **start** of its interval is read from the metadata when it is declared
and can be forced with `interval_convention`.

**Accumulation.** Whether each record is that interval's depth
(`increments`) or the running total since the start (`cumulative`) must be
**declared**. There is no automatic choice, because a series that is constant
in time is a uniform hyetograph and a stalled run-total at the same time, and
no property of the data separates them. The declaration is checked against the
series and the conversion is refused if the two contradict each other.

**Coordinates.** `source_epsg` is what the NetCDF coordinates are interpreted
in, 4326 unless stated otherwise. `target_epsg` must be a **projected** CRS
and must match the model terrain; a geographic CRS is refused, because a grid
in degrees cannot align with a mesh in metres.

**What is not supported.** Rotated or curvilinear grids, irregular time steps
within one file, and rate units. Files with a non-monotonic time axis are
reported rather than reordered.

---

## What the user can change

Everything lives in `config.ini`. Command-line arguments override it.

### `[project]`

| key | meaning |
|---|---|
| `folder` | HEC-RAS project folder, the directory holding the `.prj` |
| `unsteady_file` | unsteady flow file to rewrite, e.g. `basin.u01` |
| `plan_file` | plan file to rewrite, e.g. `basin.p01` |
| `results_folder` | where HEC-RAS writes its stored maps |
| `ras_executable` | path to `Ras.exe`, or a version string such as `6.6` |

### `[rainfall]`

| key | values | meaning |
|---|---|---|
| `netcdf` | path | input rainfall file |
| `dss` | path | HEC-DSS file to write |
| `interval_convention` | `auto` · `end` · `begin` | whether a timestamp ends or starts its interval |
| `accumulation` | `increments` · `cumulative` | **required**, no default |
| `drop_first_step` | `yes` · `no` | drop a first record that is not a valid interval total, as produced by differencing a cumulative field |

### `[grid]`

| key | default | meaning |
|---|---|---|
| `target_epsg` | — | CRS to write the grid in; must be projected and match the model |
| `source_epsg` | `4326` | CRS the NetCDF coordinates are in |
| `cell_size_m` | `1000` | rainfall grid cell size, in target CRS units. Use a value finer than the native NetCDF cell |
| `resample` | `linear` | resampling method, below |

#### Resampling methods

`nearest` · `linear` · `cubic` · `pchip` · `kriging`

### `[terrain]` and `[output]`

| key | default | meaning |
|---|---|---|
| `dem` | — | DEM used as the hillshade backdrop on the maps. Optional |
| `maps_folder` | — | where the PNG maps are written |
| `dpi` | `600` | figure resolution; 300–600 suits a journal figure |
| `min_depth` | `0.01` | depth in metres below which a cell is drawn as dry. Suppresses isolated millimetre-deep sheet flow |

---

## Outputs

| file | written by | content |
|---|---|---|
| `storm.dss` | `netcdf_to_dss.py` | the rainfall field as HEC-DSS period-cumulative grid records |
| `storm.dss.window.json` | `netcdf_to_dss.py` | provenance: window, time step, grid definition, resampling method, volume audit |
| registration figure | `verify_rainfall_grid.py` | rainfall grid against terrain footprint |
| results HDF and stored-map GeoTIFF | HEC-RAS | the solved run |
| five PNG maps | `flood_maps.py` | depth, velocity, hazard, arrival time, extent |

Any run can be rebuilt from its provenance record, which is why that file is
written before the solver is ever started.

---

## Requirements

**Python 3.13**, and **HEC-RAS 6.7** installed on the same
machine. The versions below are those the published results were produced
with; they are pinned because a silent change in `pyproj` or `hecdss` moves
the grid.

```
numpy==2.4.6            scipy==1.18.0           pandas==3.0.3
matplotlib==3.11.0      pyproj==3.7.2           tifffile==2026.7.31
netCDF4==1.7.4          h5py==3.16.0
hecdss==0.1.29          ras-commander==0.99.2
```

```bash
pip install -r requirements.txt
```

`netCDF4` and `h5py` are both NetCDF readers and only one is strictly needed:
`netCDF4` handles every NetCDF format, `h5py` reads NetCDF-4 only and is the
fallback. `scipy` supplies the interpolation, the nearest-neighbour search and
the variogram fit; `pyproj` the reprojection; `hecdss` the DSS writing;
`ras-commander` the HEC-RAS control.

`config.ini` is in `.gitignore`. Publish `config.example.ini` instead — the
real one holds machine-specific paths.
