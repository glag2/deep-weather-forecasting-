# How data ingestion works

This document explains how the GRIB files downloaded from the Climate Data Store become
tensors ready for training. It is meant for whoever picks up the project without having
written it: it describes **what** every step does, **why** it is done this way and **how**
to rerun or extend it.

The numbers reported are measured on the real ingestion of January 2024, not estimated.

---

## 1. The problem

ERA5 distributes GRIB, a format designed for meteorological archiving: compact,
self-describing, but terrible for training. Reading a field requires
decoding messages sequentially, there is no efficient random access to "slot
number 1417", and instantaneous and accumulated variables have different and
incompatible time structures.

A model that trains needs the opposite: reading arbitrary windows
of 30 consecutive slots quickly, thousands of times per epoch, in random order.

The pipeline solves this with **two storage levels**, each chosen for the
job it has to do.

---

## 2. The two levels

### Level 1: Zarr, the tensors

`datasets/era5_slots.zarr` contains the numeric data, with dimensions
`slot × latitude × longitude` = `2862 × 261 × 401`, one variable per array.

Zarr is, in essence, **a numpy array on disk**: split into independently
compressed blocks (*chunks*), readable in parallel and without loading the rest. Every chunk
covers 8 slots and the whole spatial grid, so reading a time window never
touches useless spatial data.

Why not something else:

| Format | Why it was discarded |
|---|---|
| CSV | 229 million rows per month, uncompressed text, no random access. Out of the question. |
| GRIB directly | No random access; repeated decoding at every epoch. |
| Single NetCDF file | A single file of GB, hard to write incrementally and concurrently. |
| Parquet for the tensors | Columnar but row oriented: a 3-D tensor becomes a long table and reading amplifies by about 11 times. |

Zarr is also language independent: the same store is read from Python, Julia,
R or JavaScript.

### Level 2: Parquet, the metadata

`datasets/tables/*.parquet` contains everything that **describes** the data and is queried with
Polars: which slots exist, which are usable, which statistics they have, how they are
divided into folds.

These are small tables (25 KB for the catalogue of 2862 slots) on which filters and
groupings are done. It is exactly the use case where Polars excels, while the tensors
stay in Zarr.

**The dividing rule**: numbers on a grid → Zarr; everything you want to run a
query on → Parquet.

---

## 3. The complete flow

```
CDS API                  scripts/download_era5.py
   │
   ├─ instantaneous_YYYY-MM.grib   7 variables, hours 3..22, ~139 MB
   ├─ accumulated_YYYY-MM.grib     2 variables, forecast runs, ~248 MB
   └─ static.grib                  2 invariant fields, 613 KB
   │
   ▼                       scripts/ingest_era5.py
 ingestion
   │
   ├─ era5_slots.zarr      9 variables × 2862 slots
   ├─ era5_static.zarr     2 fields × grid
   └─ tables/*.parquet     slots, slot_stats, folds, variables
```

### 3.1 Preallocating the store

`initialize_store` creates the whole store for **all** the configured period, filled with
NaN, before ingesting any month. Every month is then written into its region with
`region=`, without rewriting the rest.

This makes it possible to ingest the months **in any order and while the download continues**:
the months not yet arrived stay NaN and the catalogue marks them as not usable.

A verified detail that makes the choice cheap: **Zarr does not write to disk the chunks
entirely equal to the fill value**. After preallocating 2862 slots and ingesting
only January 2024, there are 12 chunks per variable instead of 358. Preallocating costs
no space.

### 3.2 Instantaneous variables

Real structure, verified on the files: a flat `time` axis with all the requested hours,
plus `latitude` and `longitude`. Reading is direct: the instants of the slots
(06, 12, 18 UTC) are selected and written.

### 3.3 Accumulated variables, the delicate point

Here the GRIB does not have a flat time axis. ERA5 archives the accumulations as **forecast
runs**: two runs per day, based at 06 and 18 UTC, each with several
hourly `step` values. The structure is `(time, step)` and the real instant of every value is in a
**two-dimensional** coordinate `valid_time = time + step`.

`flatten_accumulated` reduces it to a single ordered axis:

1. `stack` merges `(time, step)` into a single axis;
2. `swap_dims` replaces it with `valid_time`;
3. `sortby` orders chronologically;
4. duplicate instants are removed keeping the first;
5. **`transpose` imposes the order `(valid_time, latitude, longitude)`**.

Point 5 is not cosmetic. `stack` places the new axis in **last** position, so
without transpose the arrays come out as `(latitude, longitude, valid_time)` and every
time indexing actually selects the latitude. This defect showed up
as `IndexError: index 261 is out of bounds` and is covered by a dedicated
test.

Facts verified on the real files: 756 valid instants for a month of 31 days, **zero
duplicates**, all 24 hours present. The two daily runs sit side by side without
overlapping.

### 3.4 The accumulation window

Precipitation and snow do not have an "instantaneous" value: they only make sense over an
interval. Every slot receives the sum of the **8 hours centred on it**:

| Slot | Hours summed |
|---|---|
| 06 UTC | 03–10 |
| 12 UTC | 09–16 |
| 18 UTC | 15–22 |

Three windows of 8 hours cover 20 of the 24 hours of the day. The choice is constrained by a
property that simplifies everything: **no window crosses midnight**, so every
month is self-sufficient and can be ingested without reading the previous or following month.

The aggregation was verified by recomputing it independently from the GRIB for one
slot: **maximum difference 0,0**.

### 3.5 Quality checks

For every slot and variable, `compute_stats` records minimum, maximum, mean, number of
valid values and of NaN in `slot_stats.parquet`.

The catalogue marks `usable = false` if a month was not ingested **or** if it contains
non-finite values. An unusable slot would propagate NaN to all the samples that
contain it, so the filter is upstream of the training dataset.

Physical plausibility check on January 2024, all variables in range:

| Variable | Minimum | Maximum | Mean |
|---|---|---|---|
| `t2m` (K) | 217,1 | 313,2 | 281,0 |
| `msl` (Pa) | 93 876 | 104 809 | 101 379 |
| `u10` (m/s) | −23,2 | 30,0 | 0,3 |
| `tcc` (0–1) | 0,00 | 1,00 | 0,62 |
| `sd` (m) | 0,0 | 10,0 | 0,22 |
| `tp` (m/8h) | 0,0 | 0,070 | 0,0007 |
| `sf` (m/8h) | 0,0 | 0,049 | 0,0002 |

### 3.6 A real anomaly: `sf > tp`

Physically snow, expressed as water equivalent, cannot exceed total
precipitation. In the data it happens at **12,26 %** of the points.

It is not a defect of the ingestion: it is **GRIB quantization noise**, which compresses the
values into scaled integers. `tp` and `sf` are packed independently, so
when it snows purely (`sf ≈ tp`) the rounding can make it exceed `tp`. Measurements:

- maximum violation: **0,0055 mm**;
- maximum `sf/tp` ratio: **1,043**;
- where `tp = 0`, the maximum `sf` is 0,005 mm.

The ingestion **does not correct** the data, in order to stay faithful to the source. The correction
belongs to the construction of the target, where the ratio is clipped to [0, 1] and
defined only above the threshold of 0,1 mm.

### 3.7 The fourteen invariant fields, verified one by one

The surface descriptors were downloaded in a single request of 4,0 MB
(`static.grib`, a single instant) and read with `tmp/diagnostica/verifica_statici.py`. All
fourteen are present on the 261x401 grid. Measured ranges:

| Field | Meaning | Minimum | Maximum | Mean |
|---|---|---|---|---|
| `lsm` | land fraction | 0 | 1 | 0,5200 |
| `z` | geopotential (m2 s-2) | -1260,77 | 31544,98 | 2588,78 |
| `slt` | soil type (classes) | 0 | 7 | 1,11 |
| `cvh` | high vegetation fraction | 0 | 1 | 0,1335 |
| `cvl` | low vegetation fraction | 0 | 1 | 0,1921 |
| `tvh` | high vegetation type | 0 | 19 | 4,05 |
| `tvl` | low vegetation type | 0 | 17 | 2,26 |
| `cl` | inland water fraction | 0 | 1 | 0,0165 |
| `dl` | inland water depth (m) | 0,50 | 6218,29 | 1129,69 |
| `sdor` | orography dispersion (m) | 0 | 672,35 | 30,53 |
| `isor` | orography anisotropy | 0 | 0,9869 | 0,2712 |
| `anor` | orography orientation (rad) | -1,5578 | 1,5627 | 0,3823 |
| `slor` | orography slope | 0,0001 | 0,1177 | 0,0051 |
| `sdfor` | filtered orography dispersion (m) | 0 | 526,94 | 21,09 |

Two values look anomalous and must be explained, not corrected.

**The geopotential is negative where there is land below sea level.** The minimum
-1260,77 m2 s-2 corresponds to -128,6 m: it is real terrain, not a sign error. The
domain includes the Caspian depression and other areas below sea level.

**`dl` is not a field usable as a raw channel.** A mean depth of 1130 m
is impossible for European inland waters, and the maximum of 6218 m does not belong to
any lake: ERA5 defines `dl` **over the whole grid**, with fill values where
there are no lakes, and the fraction `cl` has mean 0,0165, that is the field is meaningful over
less than 2% of the points. Used as it stands, `dl` would inject a "deep water"
signal over the ocean and over land. It must therefore be used **only multiplied by
`cl`**, or left out: for this reason it does not enter the default list of static
fields, while the other thirteen are usable directly.

---

## 4. The Parquet tables

| Table | Content | Role |
|---|---|---|
| `slots` | one record per slot: instant, year, month, hour, `usable`, indicative `split` | general catalogue |
| `slot_stats` | statistics per slot and variable | quality check |
| `folds` | **authoritative**: `fold`, `split`, `slot_index`, `is_sample_start` | data split |
| `variables` | variable metadata: unit, family, role | documentation |
| `downloads` | outcome of every download | tracking |

### Why the folds have a table of their own

With rolling window validation, **the same slot changes split from one fold to another**:
what is test in fold 0 becomes train in fold 3. A single `split` column cannot
represent that, so `folds.parquet` is the authoritative source and the `split` column of
`slots` is only indicative. A test checks precisely that at least one slot exists with more than one
split.

`is_sample_start` is true only if **all** the 30 slots of the window (21 of input + 9 of
target) are usable. The training dataset reads only this column.

---

## 5. How to run it

```bash
# 1. Download the GRIB files (long: ~9 min per month)
uv run python scripts/download_era5.py --dry-run        # preview
uv run python scripts/download_era5.py --from-month 2025-01

# 2. See what is ready
uv run python scripts/ingest_era5.py --list

# 3. Ingest
uv run python scripts/ingest_era5.py                    # all available months
uv run python scripts/ingest_era5.py --month 2024-01    # a single one
```

The ingestion is **idempotent**: rerunning it on a month already done rewrites it with the
same result. `--recreate-store` clears the store and forces reingesting everything; it is needed
only if the configuration of the grid or of the period changes.

### Measured costs

| Operation | Time | Space |
|---|---|---|
| Download of one month | ~9 min | 387 MB (GRIB) |
| Ingestion of one month | 17,7 s | ~150 MB (Zarr) |
| Complete period, 32 months | ~4,7 h | ~12,4 GB GRIB + ~4,6 GB Zarr |

The Zarr is smaller than the GRIB because it is a **reduction**: it keeps 3 slots per day
instead of 24 hourly fields, and the accumulations already summed over the window.

---

## 6. How to extend it

**Adding a variable**: declare it in `src/dwf/variables.py` with CDS name and GRIB
name, add it to the list in `configs/default.yaml`, download again and reingest with
`--recreate-store` (the store has one array per variable, decided at creation).

**Lengthening the period**: change `time.start` / `time.end` in the configuration,
download the new months, reingest with `--recreate-store`. Careful: changing the
period also changes the number of folds.

**Changing the accumulation window**: `time.accum_window_hours`. The constraint of not
crossing midnight is verified by a test, which will fail if the window becomes
too wide for the configured slots.

**Changing the region**: `region` in the configuration. It requires downloading everything again.

---

## 7. Defects found and solved during development

An honest list, useful to whoever will debug the code in the future:

| Defect | Symptom | Cause |
|---|---|---|
| Axis order | `IndexError: index 261 out of bounds` | `stack` puts the new axis last |
| Time zones | `UserWarning: no explicit representation of timezones` | numpy discarded the time zone silently |
| Month outside the period | Returned the whole month | Wrong comparison of the bounds in `days_for_month` |
| Data destination | The GB ended up in `Data/`, tracked by git | Windows ignores the case difference |

---

## 8. What is not done yet

The ingestion is complete and verified. Still to be built, downstream:

- `features.py`: derived channels (time changes, seasonal encodings, static);
- `dataset.py`: torch dataset that reads `folds.parquet` and crops the windows;
- normalization computed **only on the train** of each fold, so as not to leak
  information from the future.
