# Weather forecasting with deep learning on ERA5 reanalysis

Forecast of the **next 3 days** (morning, midday, evening) over the
Euro-Atlantic area, starting from the **previous 7 days** of ERA5 reanalysis, with a
convolutional network written from scratch.

Forecast variables: **temperature**, **precipitation**, **snow**, each with its own
calibrated **uncertainty**.

![Example of an ERA5 field: 2 m temperature](image/README/1730719737389.png)

## Domain

| | |
|---|---|
| Area | lat 10 N - 75 N, lon 40 W - 60 E |
| Resolution | 0.25 degrees (ERA5 native) |
| Grid | **261 x 401 = 104.661 points** |
| Daily slots | 06, 12, 18 UTC |
| Input | 21 slots (7 days) |
| Output | 9 slots (3 days) |

Dataset: [ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) for environment management
- A Copernicus CDS account (free)
- No GPU needed: training is designed for CPU

## Installation

```bash
uv sync --extra notebooks
```

On slow connections the download of the largest wheels (`torch`, `polars`, `scipy`)
can exceed the default network timeout of `uv`, which is 30 seconds. In that
case:

```bash
# Linux / macOS
UV_HTTP_TIMEOUT=600 uv sync --extra notebooks
```

```powershell
# Windows PowerShell
$env:UV_HTTP_TIMEOUT=600; uv sync --extra notebooks
```

## CDS credentials

The project reads credentials from environment variables, with the same order of
precedence used by `cdsapi`: first the environment, then `~/.cdsapirc`.

**Required variables:**

| Variable | Value |
|---|---|
| `CDSAPI_URL` | `https://cds.climate.copernicus.eu/api` |
| `CDSAPI_KEY` | your own Personal Access Token |

**Procedure:**

1. Register at <https://cds.climate.copernicus.eu> and log in.
2. Open <https://cds.climate.copernicus.eu/how-to-api>: the page shows your own
   Personal Access Token.
3. Accept the dataset *Terms of Use*. This is a separate step and easy to forget:
   without it every API request fails even with a valid token. It is at the
   bottom of the form in the *Download* tab of
   <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels>,
   or it can be accepted via API with `--accept-licences` (see below).
4. Create in the project root a `.env` file with the two variables:

   ```
   CDSAPI_URL=https://cds.climate.copernicus.eu/api
   CDSAPI_KEY=your-own-token
   ```

`.env` is excluded from versioning. It must not be committed, nor pasted into chat, logs or
notebooks: if a key leaves your own storage it must be considered compromised and
regenerated from the CDS profile.

> On Windows, Notepad and PowerShell 5.1 `Set-Content -Encoding utf8` write a
> UTF-8 BOM at the head of the file. The project handles it by reading with `utf-8-sig`, but
> many other tools do not.

### Checking access

Before queueing dozens of requests it is worth checking that everything is in order. The
script distinguishes the three reasons why a download fails, which otherwise
blend into a single HTTP error:

```bash
uv run python scripts/check_cds_access.py
uv run python scripts/check_cds_access.py --accept-licences
```

It also reports **the last ERA5 date actually available**, searching backwards
from today instead of assuming a fixed latency.

## Configuration

Everything is declared in [`configs/default.yaml`](configs/default.yaml) and validated at
runtime: area aligned to the grid, variables of consistent type, targets present among
the downloaded variables, paths confined under the data folder.

### Rolling window validation

The model is not evaluated on a single final block. A contiguous test would fall
entirely in the tail of the period, which is summer: snow would not be measurable and
temperature would be evaluated on a single weather regime.

The split therefore uses **rolling origin validation**: the origin advances in
time and every fold has its own train, validation and test, always in this
chronological order. With the reference configuration there are 6 folds whose test blocks
cover **all twelve months**. The cost is that training must be repeated for each
fold; `split.n_folds` allows limiting them during development, and
`split.mode: chronological` goes back to the single-block split.

## Structure

```
configs/default.yaml       reference configuration
src/dwf/
  variables.py             ERA5 variable registry (CDS name <-> GRIB short name)
  slots.py                 time slots, accumulation windows, leakage-free split
  config.py                validated configuration
  credentials.py           CDS credentials, never exposing their value
  tables.py                Polars/Parquet layer with verified schemas
  data/                    download, ingestion, features, dataset
  models/                  convolutional network and probabilistic heads
scripts/
  check_cds_access.py      CDS access diagnosis
  download_era5.py         download of the configured period, resumable
  benchmark_model.py       model cost on CPU
tests/                     pytest suite
datasets/                  (ignored by git) GRIB, Zarr, tables, artifacts
```

The data root is `datasets/` and not `data/`: on Windows `data` would resolve
to the `Data/` folder already present in the repository, mixing tens of generated GB
with tracked files, while on Linux and in Docker it would stay a separate folder. The
name is configurable with `paths.data_root`.

## Downloading the data

```bash
uv run python scripts/download_era5.py --dry-run   # shows the plan, sends nothing
uv run python scripts/download_era5.py --limit 2   # a single month, to measure
uv run python scripts/download_era5.py             # the whole period
```

The download is **resumable**: files already present and not empty are skipped, and
every request writes to a `.partial` file renamed only once the download is complete,
so an interruption does not leave a truncated GRIB that would look valid. Requests
are sequential because the CDS limits concurrent ones per user.

Data is organized on two levels: **Zarr** for the numeric tensors, on which
training performs random access to spatiotemporal windows, and **Polars/Parquet** as
the registry of clean data (slot catalogue, quality checks, normalization
statistics, metrics, calibration, final forecast).

## How the pipeline works

Every step reads what the previous one wrote. They can be run individually.

| # | Command | What it does |
|---|---|---|
| 1 | `scripts/check_cds_access.py` | Checks token, licences and last available ERA5 date. |
| 2 | `scripts/download_era5.py` | Downloads the GRIB files month by month into `datasets/raw/`. Resumable. |
| 3 | `scripts/ingest_era5.py` | Converts the GRIB files into a single Zarr store `(slot, 261, 401)` and records the slots in Parquet. De-accumulates rain and snow. |
| 4 | `scripts/analyze_data.py` | Exploratory analysis of the store: coverage, distributions, predictability. Writes `docs/DATA_ANALYSIS.md`. |
| 5 | `scripts/screen_features.py` | Measures which channel families help to predict **change**. Writes `docs/FEATURES.md`. |
| 6 | `scripts/screen_input_days.py` | Compares 3, 7, 10, 14 days of history. Writes `docs/INPUT_DAYS.md`. |
| 7 | `scripts/compare_variants.py` | Compares the five architectures under the same protocol. Writes `docs/VARIANTS.md`. |
| 8 | `scripts/train_model.py --fold 0` | Trains one fold. Saves weights and statistics in `models/fold_00/`. |
| 9 | `scripts/evaluate_model.py --fold 0 --split test` | Metrics on the test set, probability calibration, threshold selection, comparison with the persistence baselines. |
| 10 | `scripts/predict_forecast.py --fold 0` | 3-day forecast over the whole grid, in Parquet. |
| 11 | `scripts/report_forecast.py --fold 0` | 8-page PDF report with the maps. |
| 12 | `scripts/refresh_data.py` | Downloads and ingests only what is missing, to update without redoing everything. |

In between, the data passes through these forms:

```
monthly GRIB -> Zarr store (slot x 261 x 401) -> window of 21 slots
   -> 245 channels (state, tendencies, wind, sun, thermodynamics, static)
   -> network -> 45 output channels -> 9 forecast slots x 4 quantities + uncertainty
   -> calibration -> Parquet -> PDF
```

## Performance

Model evaluated on the **test** block of fold 0, never used either for training or
for choosing the thresholds. The reference is the **diurnal persistence** (repeating yesterday at the
same hour), which on this domain is a very strong opponent.

| Quantity | Metric | Model | Diurnal persistence | Naive persistence |
|---|---|---:|---:|---:|
| Temperature | RMSE (degC) | **2.92** | 3.16 | 4.73 |
| Rain yes/no | F1 | **0.635** | 0.609 | 0.624 |
| Rain yes/no | Accuracy | 0.709 | 0.726 | **0.739** |
| Snow yes/no | F1 | 0.493 | 0.561 | **0.590** |
| Snow yes/no | Accuracy | 0.847 | 0.919 | **0.925** |
| Rain and snow | Macro F1 | 0.564 | 0.585 | **0.607** |

241 test windows, 9 lead times each, whole domain. Obtained with
`scripts/evaluate_model.py --fold 0 --split test`, which writes `metrics.parquet`.

**How to read it.** On temperature the model beats diurnal persistence at **all
nine lead times**, and the advantage is not concentrated on the first ones: 2.02 against 2.43 degC
at six hours, 3.40 against 3.68 at three days.

On snow it **loses**, and it is better to say why instead of hiding it. The model forecasts
snow too often: it recovers 82 % of the cases against 60 % for persistence, but only
35 % of its warnings are correct against 55 %. The decision threshold is
chosen on validation, where it gives F1 0.568; on the test set it drops to 0.493. Changing it by
looking at the test set would shift the trade-off, but it would be cheating.

**Snow accuracy must not be read as a result.** Snow appears in 9 % of the
cases, so always answering "no" would give 90.9 %: both persistence baselines, at 92.5 %,
exceed that trivial threshold only slightly. This is the reason why the table also reports
F1, which a constant answer cannot inflate.

On the **quality of the probability**, which is what is needed to decide, the model
wins everywhere, snow included: Brier score 0.181 against 0.274 on rain and
0.066 against 0.081 on snow, with calibration error 0.053 against 0.274.

**Macro F1** is the average of the two binary F1 values (rain and snow). A single F1 over the whole
model would make no sense: temperature is continuous and has no notion of
"positive".

## Notebooks

Generated by `scripts/build_notebooks.py`, so they must not be edited by hand.

| Notebook | What it answers |
|---|---|
| `notebooks/01_training.ipynb` | Trains a fold and shows why validation uses a rolling window. |
| `notebooks/02_inference.ipynb` | Produces a forecast and reads it on the maps. |
| `notebooks/03_collaudo.ipynb` | **Tests a trained model**: consistency of the artifacts, gain per lead time, check of the 2 degree target at 24 hours, tuning of the uncertainty, reliability of the probabilities, error map, and the blurring check. Every section explains how it is read, including the pitfalls. |

## Development

```bash
uv run pytest tests -q
uv run ruff check src tests scripts
uv run python scripts/build_notebooks.py
```

## Documentation

The reports are in `docs/`, and all except the first are **regenerated by a
script**: they must not be edited by hand, because the next run overwrites them.

| File | What it contains | Who produces it |
|---|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | Work plan, with the truthful status of every item | by hand |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | Work status, decisions and open problems | by hand |
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | State of the art read and what was taken from it | by hand |
| [`docs/INGESTION.md`](docs/INGESTION.md) | How the GRIB files become Zarr, and the anomalies found | by hand |
| [`docs/DATA_ANALYSIS.md`](docs/DATA_ANALYSIS.md) | Analysis of the ingested dataset | `scripts/analyze_data.py` |
| [`docs/VARIANTS.md`](docs/VARIANTS.md) | Comparison between network variants | `scripts/compare_variants.py` |
| [`docs/INPUT_DAYS.md`](docs/INPUT_DAYS.md) | How many days of input history | `scripts/screen_input_days.py` |
| [`docs/OCCURRENCE_ANCHOR.md`](docs/OCCURRENCE_ANCHOR.md) | Anchoring of the rain probability | `scripts/screen_occurrence_anchor.py` |
| [`docs/FEATURES.md`](docs/FEATURES.md) | Which channel families help to predict change | `scripts/screen_features.py` |
| [`docs/NEARTIME.md`](docs/NEARTIME.md) | Real ERA5 latency and sources for near real time | by hand |

## What the model is worth, today

The absolute error does not tell whether a model is useful. The comparison that tells it is against
diurnal persistence, that is the hypothesis "tomorrow like yesterday at the same hour", which costs
nothing. On the test split, 2 metre temperature:

| hours ahead | model | yesterday same hour | gain |
|---|---|---|---|
| +12 | 1,87 C | 2,43 C | +23 % |
| +24 | 2,27 C | 2,40 C | **+5,5 %** |
| +48 | 2,99 C | 3,24 C | +7,6 % |
| +72 | 3,32 C | 3,68 C | +9,8 % |

At twenty-four hours the model gains five per cent over doing nothing. What
it has learned is the daily cycle, which was already given to it by the anchoring, plus a
local smoothing; the dynamics are not there. The measured causes are in
[docs/PROGRESS.md](docs/PROGRESS.md) section 14: effective receptive field of 130 km,
no upper-level variable, 2,7 visits per window in the whole training.

## Known limits

- **ERA5 has 5-6 days of latency.** A forecast started from the last available
  data therefore concerns days that have already passed: it is a verifiable hindcast,
  useful for validating, not an operational forecast. For real time a
  different source would be needed, such as <https://data.ecmwf.int>.
- **The model is undertrained, and by how much is known.** With the settings used so far
  training saw 2560 steps, that is 0,94 passes over the data and 2,7 visits per
  window. A model that has looked at every example fewer than three times does not yet have a
  limit of its own: it has a compute time limit.
- **Training is on the whole window, not on crops.** The 96x96 crop made
  training much cheaper, but it taught the network to decide within a horizon
  that does not exist in forecasting. Measured cost of the change: 3000 ms per step on the whole
  domain against 800 ms for four crops, that is 28,7 against 21,7 seconds per million
  forecast points.
- **The declared uncertainty is measurable only from the recent evaluations.** The tables
  produced before the introduction of the `spread_skill_ratio` and `coverage_90` metrics do not
  contain them; it is enough to rerun `scripts/evaluate_model.py`.

## Licence

See [LICENSE](LICENSE). ERA5 data is subject to the Copernicus licence.
