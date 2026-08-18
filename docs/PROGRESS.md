# Deep Weather Forecasting - handover document

This document exists so that a person who has never seen the project can pick it up
again. It does not only describe **what** is here, but **why** it is like this, which
alternatives were discarded and on which measurement, and where the real limits are.

Reading rule adopted throughout the project: **the data are true**. ERA5 is a reanalysis
produced by assimilating observations into a physical model. When faced with a
surprising number, the first hypothesis to check is a mistake by the person doing the
analysis. This document reports the cases where that hypothesis turned out to be right,
because they are the most instructive part of the work.

---

## 1. What it does

**Three-day** forecast on a grid, trained and obtained **locally on CPU**.

- **Input**: 7 days of ERA5 history, that is 21 slots (06, 12, 18 UTC).
- **Output**: 9 future slots (3 days x 3 hours of the day), all produced **in a single
  pass**, not autoregressively.
- **Domain**: Euro-Atlantic, 261 x 401 cells at 0.25 degrees (75 N .. 10 N, 40 W .. 60 E).
- **Predicted quantities**: 2 m temperature in **degrees Celsius**, precipitation
  (probability and amount), snow (fraction of precipitation) and **calibrated
  uncertainty** on each of them.
- **Delivery**: Parquet tables, two notebooks and a **PDF report** with maps.

The project has a declared place of interest, **Vigo di Cadore**, which appears in the
loss weighting, in the data analysis and in the report.

### Status

| | |
|---|---|
| Tests | **678**, all green, including inside the container |
| Lint | `ruff` clean on `src`, `tests`, `scripts` |
| Commits on the branch | 46, none pushed |
| Data downloaded | 21 months (2024-01, 2025-01 .. 2026-08); the remaining part of 2024 is downloading |
| Data ingested | 16 months, 1458 slots, zero missing values |
| Docker | image built and **verified by running the suite inside it** |

---

## 2. How to run it

### Locally

```bash
uv sync --extra notebooks          # reproducible environment from the lockfile
uv run python scripts/check_cds_access.py
uv run python scripts/download_era5.py --dry-run
uv run python scripts/download_era5.py
uv run python scripts/ingest_era5.py
uv run python scripts/analyze_data.py      # -> DATA_ANALYSIS.md
uv run python scripts/compare_variants.py  # -> VARIANTS.md
uv run python scripts/train_model.py --fold 0
uv run python scripts/evaluate_model.py --fold 0 --split test
uv run python scripts/predict_forecast.py --fold 0
uv run python scripts/report_forecast.py --fold 0   # -> PDF
```

The CDS credentials live in `.env`, which is not tracked. They must never be pasted into
a chat or committed: a key that has travelled through an unencrypted channel must be
considered compromised and rotated.

### In a container

```bash
docker compose build
docker compose run --rm dwf python scripts/ingest_era5.py --list
docker compose up jupyter        # notebook at http://localhost:8888
```

Data, configurations, models and notebooks are **volumes**, not image content: they
weigh tens of gigabytes and must survive a rebuild.

**Verified warning**: Debian bookworm ships ecCodes 2.28 while `cfgrib` recommends 2.42.
The suite passes in the container anyway, but GRIB ingestion is better run on the host.

---

## 3. Repository map

```
src/dwf/
  config.py          Configuration validated with pydantic; single source of truth
  variables.py       Registry of the 24 variables: units, transforms, offsets
  slots.py           Slot arithmetic; diurnal_reference_index lives here
  tables.py          Parquet schemas declared and verified
  credentials.py     Reading of the CDS credentials, never printed
  solar.py           Solar geometry (Spencer 1971): 3 channels
  thermo.py          Moist air thermodynamics: 4 channels
  weighting.py       Spatial loss weights (area + local focus)
  persistence.py     Model saving without pickle
  calibration.py     Isotonic calibration of the probabilities
  train.py evaluate.py predict.py report.py
  data/
    download.py ingest.py features.py dataset.py freshness.py refresh.py
  models/
    heads.py blocks.py network.py losses.py
    variants/        conv, attention, fourier, recurrent, hybrid
scripts/             12 command line entry points
tests/               20 files, 673 tests
```

Documents: `INGESTION.md` (verified facts about the GRIB files), `RESEARCH.md`
(technology research and motivated rejections), `DATA_ANALYSIS.md` (exploratory
analysis), `VARIANTS.md` (comparison between architectures).

---

## 4. The data

### 4.1 Domain and period

The domain was not chosen by hand: it was derived from the outputs already present in
the exploratory notebook of the original repository. The period ends at the date
declared by the metadata of the CDS collection, read at runtime and not written from
memory: ERA5 has about six days of latency, so it **does not reach today**. From this
follows a fact that matters for the honesty of the project: what we call a "forecast" is
in fact a *verifiable hindcast*, and this is an advantage, because every forecast has a
truth to be compared against.

### 4.2 Storage structure

Zarr for the tensor, Parquet for the catalogues. The split is not cosmetic: a dense
tensor of 2862 x 261 x 401 values per variable needs chunked access and compression,
things a row-oriented columnar format does not offer. Polars stays for what it is
unbeatable at, that is registries, metrics and joins.

Chunks `(8, 261, 401)`. An attempt to rechunk in space as well was **measured and
discarded**: it made per-window reads worse, and those are the dominant use case.

### 4.3 Anomalies explained, not corrected

**Snow greater than total precipitation in 12.26% of the cells.** It looks like a
physical violation. It is not: the GRIB messages pack `tp` and `sf` as integers with
**independent** quantization steps. The violation never exceeds 1.5 times the
quantization step and its correlation with precipitation intensity is 0.013, that is
none. It is representation noise, not an error in the data. This is why the snow
fraction target is clipped to [0, 1] **in the target** and not in the stored data: you
constrain what you ask of the model, you do not falsify the archive.

**The slots are not equally spaced.** 06Z, 12Z and 18Z are 6, 6 and 12 hours apart.
Three slots make exactly one day. It looks like a detail and is instead the most
important finding of the analysis: see section 6.

---

## 5. The input quantities

245 channels: 189 of state, 27 of tendency, 21 of wind speed, 2 static, 2 of latitude,
4 of time encoding.

**Temperatures are in degrees Celsius.** The conversion is declared on the variable and
applied **only once**, right after reading. This is not cosmetic: converting further
downstream, `normalize` and `denormalize` would stop being each other's inverse. The
already trained checkpoint stayed valid **bit for bit**, because translating the data
and the mean by the same amount does not change the normalized value, and a test
verifies it.

### 5.1 Derived physics, without downloading anything new

**`solar.py`** implements the formulas of Spencer (1971): declination, Earth-Sun
distance factor, equation of time, cosine of the zenith angle, insolation at the top of
the atmosphere, day length with explicit handling of the polar case. Thirty tests
compare it against known astronomical references: obliquity 23.44 degrees, perihelion on
the third day of the year, aphelion on the 185th.

**`thermo.py`** derives from the already downloaded temperature and dew point:
saturation vapour pressure, relative humidity, dew point depression, surface pressure
from mean sea level pressure, specific humidity, mixing ratio, **latent heat content**
and Bolton equivalent potential temperature. Thirty-five tests against tabulated values.

During that check one test was failing. **The code was right, the assertion was
wrong**: the invariant I had written, theta_e >= T, holds only below 1000 hPa, because
above that pressure compression brings the potential temperature below the actual one.
The correct invariant is theta_e >= theta. The test was fixed, not the code.

---

## 6. The finding that changed the project

The predictability analysis showed a correlation that did **not** decay monotonically
with lead time: it rose again at 3, 6 and 9 slots. Applying the rule "the data are
true", the explanation turned out to be a labelling mistake of mine: I had written the
hours column as `lead x 6`, but the slots are not equally spaced, and 3 slots are
**exactly one day**. The peaks were simply **the same hour of the day**.

This exposed a much stronger baseline than the one we were using.

| lead | hours | naive persistence | **diurnal persistence** | gain |
|---:|---:|---:|---:|---:|
| 1 | 8 | 4.535 | **2.401** | 47.1% |
| 3 | 24 | 2.401 | 2.401 | 0% |
| 5 | 40 | 5.201 | **3.173** | 39.0% |
| 9 | 72 | 3.557 | 3.557 | 0% |

Repeating *yesterday at the same hour* costs nothing and reaches a root mean square
error of about 3.0 degrees. The trained model was making **4.45**. The advantage claimed
previously was therefore an artefact of a baseline that was too weak: against the right
baseline, **the model lost**.

### 6.1 The operational consequence

If a free baseline is that strong, asking the network to rebuild it from scratch is a
waste of capacity. The Gaussian head now produces a **residual** that is added to the
most recent observation at the same hour as the target. Only the mean is shifted: the
log-variance describes the uncertainty of the residual and must not be shifted.

The effect is measured, not assumed: with the protocol held absolutely constant, three
reduced passes bring the root mean square error from **6.613** to **3.652** degrees.

The diurnal baseline also entered the evaluation as a model of its own
(`persistence_diurnal`), alongside the naive one and climatology.

---

## 7. The model

### 7.1 Contract

The network is a fully convolutional U-shaped encoder-decoder: it is trained on crops
and applied to the whole grid. It produces **a single tensor** `(B, C, H, W)`; the
mapping of the channels onto (variable, component, lead) is declared in `OutputLayout`.
Indexing those channels by hand would be the quietest possible mistake: swapping mean
and log-variance makes nothing fail, it only produces wrong forecasts.

The final weights are zeroed at initialization: the network starts from a constant
forecast, not from noise.

### 7.2 The comparable variants

A comparison between architectures has value only if **one thing changes**. Here the one
thing is the **elementary block**: skeleton, input channels, output layout, data, loss
and protocol stay identical.

| variant | parameters | ms/pass | idea |
|---|---:|---:|---|
| `conv` | 9.98 M | 225 | residual convolution, baseline |
| `attention` | 13.37 M | 415 | attention over 8x8 windows with relative position bias |
| `fourier` | 8.99 M | 201 | spectral convolution over the low modes, globally receptive |
| `recurrent` | 28.93 M | 1114 | convolutional recurrence with shared weights |
| `hybrid` | 16.42 M | 371 | sum of a local branch and a spectral branch |

**Outcome of the comparison** (details in `VARIANTS.md`). With the protocol held
absolutely constant the five architectures score between 3.642 and 3.666 degrees. But
repeating **the same** variant changing only the seed produces 3.652 / 3.557 / 3.665,
that is a standard deviation of **0.059 degrees**: the whole difference between the
architectures, 0.024 degrees, sits inside less than half a standard deviation. **The
bench cannot distinguish them.** Declaring the first in the ranking the winner would
mean reading the seed, not the model. This is exactly the backbone saturation documented
in arXiv:2407.14129.

The default therefore goes to `conv`, not because it won but because, at statistically
equal accuracy, it costs 99 s per pass against 151, 159, 185 and 428 of the others. On
CPU time is the real constraint.

**The variants that lose are not deleted.** They stay in
`src/dwf/models/variants/`, documented and selectable with `model.variant`, because the
result could flip with more data and because the measurement that discarded them must
stay reproducible.

The only effect that **comes out** of the noise is the diurnal anchoring: 6.613 against
an average of 3.625, about fifty standard deviations. The way the problem was posed
mattered more than any architectural choice.

Two honest clarifications:

- The Fourier variant had reached **228 million parameters**. The cause was mine: I was
  allocating 16 modes per axis while at the bottleneck the grid shrinks to 12 cells, so
  most of the weights were truncated on every pass and were never trained. With 8 modes
  and a narrower spectral branch it costs **less** than the convolution.
- The recurrent variant is **not the temporal ConvLSTM** of the literature. That one
  consumes a sequence `(B, T, C, H, W)`, which would change the input layout, that is
  exactly the variable the comparison holds fixed. Here the other interesting property
  is measured: effective depth at constant parameter count.

Technologies excluded and why, in detail in `RESEARCH.md`: spherical representations (the
domain is not a sphere), graphs (on a regular grid they are a slower convolution),
diffusion (it produces ensembles, while here the uncertainty is already calibrated),
foundation models (they require pressure levels that were not downloaded).

### 7.3 The loss

Weighted sum of the heads, plus two terms added on request, each with its own measured
justification.

**Area weight.** The grid is regular in degrees, not in kilometres: at 70 degrees a cell
covers 34% of an equatorial cell. Without correction the network would spend capacity on
the Arctic.

**Focus on Vigo di Cadore.** A bell around the point, isotropic in kilometres and not in
degrees. The gain is deliberately small: raising it would turn a domain model into a
local model trained on a handful of cells, which would generalize worse everywhere, Vigo
included.

Both weights are **normalized to unit mean**, so changing them does not change the scale
of the loss and the relative weights between the heads stay comparable.

**Spectral term.** Under pure squared error, if the correlation between forecast and
reality is rho, the minimum is obtained by producing a field whose amplitude is rho
times the true one: blurring pays off, because a structure in the wrong place is
punished twice, where it is and where it is missing. This is exactly the defect measured
on the first model, which underestimated the midday range by 4.8 degrees. Comparing the
moduli of the Fourier transform rewards the correct amplitude **without** reintroducing
the position penalty. A test shows it on synthetic data: with squared error alone the
optimum falls at 0.6 for a correlation of 0.6, and adding the term it moves towards full
amplitude. A second test verifies that translating the field leaves the term at zero.

The term is applied **only to the continuous variables**: the literature documents a
degradation of the precipitation occurrence metrics.

---

## 8. Evaluation

### 8.1 Protocol

**Rolling window** validation, 6 folds. A single split into three contiguous blocks
would have concentrated the test in the summer tail of the period: snow would not have
been assessable and temperature would have been measured on a single regime. With the
folds the test blocks cover **all twelve months**, keeping in each fold the order train
-> validation -> test. At the junctions 30 slots are discarded to attenuate
autocorrelation.

Calibration and decision thresholds are estimated **on the validation set** and measured
**on the test set**. Estimating and measuring them on the same block would inflate the
result.

### 8.2 Results on the test set of fold 0

| | dwf | persistence | **diurnal persistence** |
|---|---:|---:|---:|
| t2m RMSE (degC) | 4.45 | 4.73 | **3.16** |
| tp Brier | **0.182** | 0.261 | 0.274 |
| tp skill score | **+0.193** | -0.156 | -0.212 |
| sf Brier | **0.061** | 0.075 | 0.081 |

Honest reading: the model **wins clearly on the probabilities** of rain and snow, and on
temperature it **loses** against the diurnal baseline. This is the reason the anchoring
was introduced, and its effect is already measured in the comparative bench.

### 8.3 Calibration

Isotonic, with a hand-written PAVA. On the test set: calibration error from 0.0702 to
**0.0392**, Brier from 0.1841 to **0.1785**. The thresholds chosen on the validation set
are 0.38 for rain and 0.23 for snow. The latter fixed a real defect: the threshold had
been left at 0.5 and the F1 score for snow was 0.194; choosing it on the data it rose to
**0.553**.

---

## 9. Vigo di Cadore

Cell row 114, column 210 (46.50 N, 12.50 E), land fraction 1.00.

| | |
|---|---|
| model elevation | 1463 m |
| actual elevation of the village | 951 m |
| difference | **512 m** |
| implied thermal bias (6.5 K/km) | about **3.3 K** colder |

It is not an error of the model nor of the data: at 0.25 degrees a cell covers about
28 km and averages the whole Cadore, ridges included. To go from the cell to the village
an explicit elevation correction is needed. The fact is declared in the PDF report, so
that the reader does not mistake a resolution limit for a forecast error.

The measured local diurnal range is **6.7 degrees**, almost twice the domain average: it
is a harsh point for the model.

---

## 10. Security

The weights are saved in `models/` as `.npz` read with `allow_pickle=False`, with the
metadata in a separate JSON. The reason is concrete: the code used
`torch.load(..., weights_only=False)`, that is the maximally insecure setting, which
executes arbitrary code contained in the file.

Security is not **declared** but **demonstrated**: one test builds a payload that under
pickle **really does execute**, and a second test verifies that the project's loader
rejects it **without executing it**. Without the first test, the second would prove
nothing.

Credentials are never printed or logged. `.env`, `datasets/` and `models/` are outside
version control.

---

## 11. Defects found and fixed

Many are mine. They are listed because the method counts as much as the result.

| defect | how it surfaced | outcome |
|---|---|---|
| Axis order in the stacking | `IndexError` at runtime | explicit transposition |
| `log1p` ineffective on rain in metres | inspection of the scale | scale factor 1000 |
| 0.42 s per sample | profiling, not intuition | `valid_time` cached, 32 times faster |
| Wrong index in PAVA | monotonicity test | rewritten with explicit variables |
| Snow threshold left at 0.5 | absurdly low F1 | chosen on the validation set, F1 0.553 |
| Wrong theta_e invariant **in the test** | red test | the test was fixed, not the code |
| Wrong Zarr path | "I cannot find the data" | the data were there, I was wrong |
| Manifest written only at the end of the run | inspection | written after every task |
| `torch.load(weights_only=False)` | security review | replaced and demonstrated safe |
| Hour labels `lead x 6` | non-monotonic correlation | slots are not equally spaced |
| Baseline too weak | consequence of the previous one | diurnal persistence added |
| 228 M parameters in the spectral variant | cost measurement | modes reduced, now lighter than `conv` |
| Docker build failed at the last layer | real build | `README.md` and `LICENSE` missing in the image |

---

## 12. Known limits

1. **The current model is still the unanchored one.** The anchoring, the weighting and
   the spectral term are implemented and tested, and the comparative bench measures
   their effect, but the final full-scale model has to be retrained.
2. **The bench protocol is reduced** (small crops, few passes) and measured on **a
   single fold**. The spread across seeds was measured only for `conv` and is assumed
   similar for the others: plausible, not verified. The per-pass times are also
   contaminated by the load on the machine, so they must be compared only within the
   same run.
3. **The latent heat in the report** uses the dew point of the last observed slot,
   because the forecast does not contain it. It is a weak assumption over 72 hours: that
   field must be read as spatial structure, not as a humidity forecast. The limit is
   printed on the page.
4. **The maps do not have faithful geographic proportions**: cartopy is not among the
   dependencies and at different latitudes the north-south and east-west scales diverge.
5. **ecCodes 2.28 in the container** against the recommended 2.42: ingest on the host.
6. **The CDS key must be rotated** if it has ever travelled through an unencrypted
   channel.

The point about the incompleteness of 2024 is superseded: the ingestion covers 2862
slots out of 2862 expected, from 2024-01-01 to 2026-08-11, with no gaps.

---

## 13. What I would do next

> Superseded: the updated plan is in **[PLAN.md](PLAN.md)**, with the truthful status of
> every item. What follows is the list as it was before the measurements of section 14
> and remains only as a historical trace.

1. Retrain `conv` with anchoring at full scale and re-evaluate it honestly against
   diurnal persistence. The bench is closed: the choice is made and motivated.
2. Measure the **capacity curve** instead of enlarging the network by intuition. The
   bench has already provided the first evidence in this direction: 28.9 million
   parameters (`recurrent`) do not beat 9.9 million (`conv`), and cost four times as
   much.
3. Screening of the candidate features against the **future change**, not against the
   future value: a variable that predicts the value well but not the change adds nothing
   to persistence.
4. Explicit elevation correction for the step from cell to locality.
5. Extend the folds to the two complete years.

---

## 14. The model is not badly designed: it is under-trained

This section is later than the previous ones and, where it contradicts them, it
prevails.

### 14.1 The arithmetic nobody had done

`samples_per_epoch: 512` with `batch_size: 4` are **128 optimization steps per epoch**.
Twenty epochs make **2560 steps** for a network of ten million parameters.

The data coverage is even starker. A 96x96 crop is 9216 points out of the 104,661 of the
domain, 8.8%:

| | grid points |
|---|---|
| available in training (961 windows) | 100,579,221 |
| seen in twenty epochs (10,240 crops) | 94,371,840 |
| ratio | **0.94** |

Over the whole training the model sees the equivalent of **less than one pass** over the
data. The word "epoch" in the logs is misleading: it revisits 961 windows ten times each
looking every time at a tenth of the domain, it does not pass twenty times over the
data.

Consequence for the interpretation of everything above: every comparison between
architectures, variants and features so far was carried out in an under-training regime,
where the winner is the one that starts better, not the one that gets further. It is the
same reason why the rain anchoring seemed useful at reduced scale.

### 14.2 The effective receptive field is tiny

Differentiating one output point of the trained model with respect to the whole input:
**50% of the influence comes from less than 130 km**, 90% from 2189 km. A mid-latitude
system travels 500-1000 km per day, so at three days the useful information starts from
1500-3000 km away. The network can reach that in theory and does not reach it in
practice.

Careful with the crop: training on 96x96 the model **never** sees anything beyond 96
pixels, that is 2664 km. It cannot learn a relation that was never shown to it. Part of
the narrow receptive field may be caused by the crop, not by the convolution. It has to
be separated by enlarging the window seen, not by reducing the data.

### 14.3 What is missing in the data

All the input variables are **surface** ones. The only geopotential present is the static
orography. Mid-latitude weather is driven by the 500 hPa flow, and the models that work
(GraphCast, Pangu) use few time slots but **many vertical levels**. It is the most likely
gap among all those listed.

Choice of the fields, guided by the reference variables of WeatherBench 2 and not by
intuition: **z500** (drives the flow), **t850** (thermal advection, standard for snow),
**t500** (stability together with t850), **q700** (available humidity). Excluded u500 and
v500: a convolutional network derives the geostrophic wind from the z500 gradient, so
they would be largely redundant at full cost.

Verified on real files already downloaded, not only in theory: z500 between 4886 and 5939
geopotential metres (typical European January 4900-5800), t500 between -47 and -2 C, t850
up to +29 C on the Saharan edge of the domain, q700 between 0 and 0.010 kg/kg. Even the
risky case, two variables on the same level in the same GRIB, is read correctly.

Enabling them brings the channels from 245 to 341.

### 14.4 Two checkpoints lost, and the protection that was missing

The destination folder of a training run **does not depend on the architecture**:
`models/fold_00` for all of them. Two runs launched together end up in the same place and
the second overwrites the first as soon as it improves. It happened: the weights of the
full-scale model and those of the rival were lost, without a single message.

The field to separate them, `paths.models_subdir`, existed: the mistake was the
operator's. The defect in the code was another one and more serious: the metadata **did
not record the architecture**, so a checkpoint on disk was indistinguishable from one
produced by another network. Now the architecture is in the metadata, loading verifies
it, and startup refuses to write over a different architecture.

### 14.5 Order of the next interventions, by expected effect

1. **More optimization steps.** It is the constraint that ties everything else.
2. **More years.** 2022 and 2023, currently downloading, bring the windows from 961 to
   about 1750.
3. **Pressure levels.** Downloading, ~2.4 GB.
4. **Wider window seen**, to separate the crop from the architecture.
5. **Architecture.** The global-context rival is five times smaller and three times
   faster on the whole domain: at equal CPU hours it allows more steps, which by point 1
   is the advantage that counts.

### 14.6 How much the model is worth against doing nothing

The average over nine leads hides the number that matters. Root mean square error of
temperature in degrees, test split, 241 windows, comparison with diurnal persistence
("tomorrow like yesterday at the same hour"):

| lead | hours ahead | model | yesterday same hour | gain |
|---|---|---|---|---|
| 0 | +12 | 1.866 | 2.429 | +23.2% |
| 1 | +18 | 2.124 | 2.409 | +11.8% |
| 2 | +24 | 2.270 | 2.404 | **+5.5%** |
| 3 | +36 | 2.785 | 3.208 | +13.2% |
| 4 | +42 | 2.924 | 3.232 | +9.5% |
| 5 | +48 | 2.992 | 3.237 | +7.6% |
| 6 | +60 | 3.278 | 3.711 | +11.7% |
| 7 | +66 | 3.335 | 3.717 | +10.3% |
| 8 | +72 | 3.322 | 3.685 | +9.8% |

The gain is smallest at the leads that are multiples of 24 hours, where diurnal
persistence coincides with simple persistence and is therefore at its strongest. At
twenty-four hours the model beats the do-nothing hypothesis by **5.5%**.

This, and not the absolute value of 2.27 degrees, is the defect: 2.40 degrees are
obtained without any model. What the model has learned is the daily cycle, which was
already given to it by the anchoring, plus a local smoothing. The dynamics, that is the
fact that tomorrow different air arrives from elsewhere, is not there.

The three measurements of this day converge: effective receptive field of 130 km, no
upper-level variable, 2.7 visits per window over the whole training. To learn the
dynamics, spatial reach, information on the flow and compute time are all missing at the
same time. None of the three alone would explain the result.

Threshold declared by the user: below 2 degrees at twenty-four hours. The gain over
persistence has to go from 5.5% to 17%, that is it has to triple.

## 15. Whole window, verified loss, judged uncertainty

This section covers the day of 18 August 2026. Every number comes from a script in
`tmp/diagnostica/`, not from an expectation.

### 15.1 Training on the whole domain

`crop_size: null`, `batch_size: 1`. The 96x96 crop trained the network inside an
artificial horizon: beyond the border there was nothing to look at, so the network could
not learn to use information that at forecast time is given to it anyway.

Cost measured with `tmp/diagnostica/finestra_piena.py`, with a training run going on in
parallel:

| configuration | global core | U-net |
|---|---|---|
| 1 crop of 96 | 189 ms | 2251 ms |
| 4 crops of 96 | 800 ms | 3343 ms |
| 1 crop of 192 | 1098 ms | 3480 ms |
| whole domain 261x401 | **3000 ms** | **10499 ms** |

Per million predicted points: 21,705 ms with four crops, 28,660 with the whole domain.
The whole domain is about **a third less efficient per point**, because the cost of
attention grows with the square of the number of tokens. It is paid because it removes a
defect of the training, not because it is convenient.

### 15.2 The loss: two suspicions disproved, one defect found

Checked by measuring (`tmp/diagnostica/verifica_perdita.py`), not by re-reading it.

| suspicion | outcome | measurement |
|---|---|---|
| the spectral term depends on the crop size | **disproved** | 0.42 / 0.44 / 0.43 with sides 48, 96, 261 |
| the Huber protection on the tails never comes into play | **disproved** | 37.8% of the rainy points beyond beta = 1, normalized maximum 6.1 |
| the limit on the log-variance is harmless | **real defect** | gradient **exactly zero** outside the interval: 0.000000 at log-variance 15 |

The hard clipping made a channel pushed outside the interval mute forever. Replaced with
`soft_clamp`, two mirrored softplus: deviation 0.0009 seven units inside, 0.049 three
units inside, gradient 3.3e-3 at 15 and 1.5e-7 at 25.

On the same occasion the loss weights became **mandatory**: they were read with a default
by name, and renaming `precip_occurrence` would have silently moved that term from 0.5 to
1.0. Now the absence of a name stops the construction, and `spectral: 0.0` is written in
`configs/default.yaml` so that it stays a choice.

### 15.3 The declared uncertainty is now judged

The project promises a forecast that says how confident it is, and no metric was reading
it: the Gaussian head could announce any variance. `evaluate.py` collects `t2m_sigma` and
produces three rows per lead:

- `spread_celsius`: mean declared uncertainty, in degrees;
- `spread_skill_ratio`: spread divided by root mean square error. **1 is the right
  value**, below 1 the model is too confident;
- `coverage_90`: share of observations inside the 90% interval, it must be 0.90.

The rows do not exist for the models that do not declare uncertainty, such as
persistence. The evaluations saved before today do not contain them: the notebook says so
instead of failing, and to obtain them it is enough to re-run `evaluate_model.py`.

### 15.4 Acceptance notebook

`notebooks/03_collaudo.ipynb`, generated by `scripts/build_notebooks.py`. Seven sections:
consistency of the artefacts and learning curve, gain per lead with the labels spelled
out, explicit check of the 2 degrees at 24 hours goal with the required gain, calibration
of the uncertainty, reliability diagram with the trap of accuracy on rare events, error
map with Vigo di Cadore, predicted against observed with the blurring check via spatial
standard deviation.

Run against fold 0. Two things found while it was running: the saved evaluation predates
the uncertainty metrics, and the error map over **eight** windows exhausts the memory if
a training run is in progress, which is why the notebook uses four.

### 15.5 New code, switched off

| where | what | why it is off |
|---|---|---|
| `src/dwf/optim.py` | CMuon with Newton-Schulz orthogonalization, AdamW branch for the 1-D tensors, stem and output | orthogonalizing costs 107 ms against 42, the step goes from 235 to 319 ms: it has to be compared **at equal compute time** |
| `global_network.py` | attention sink (one extra key and value, with a learned value) | never measured at this scale |
| `global_network.py` | compressed HCA context, 33x51 -> 9x13 tokens, 1x1 injection initialized to zero | same; the zero injection guarantees that switching it on does not change the starting point |
| 14 ERA5 invariant fields | downloaded (4.0 MB, 143 s) and verified one by one | only 8 are continuous and usable; `dl` is unusable raw (mean 1130 m, fill values outside the lakes), the codes are not numbers |

Two results of the tests on Newton-Schulz worth remembering: the quintic iteration does
**not** converge to the identity (fixed point between 0.68 and 1.14), and five steps are
**not enough** on degenerate gradients (from 1e4 you get to 37; with ten steps to 1.7).

### 15.6 Open decision for the user

The default of `model.architecture` stays `unet`. The global core has a fifth of the
parameters, is 3.5 times faster on the whole domain and won the comparison on validation
(0.9755 against 1.3031), but it does not yet have numbers on the test block. Changing a
shared default without those numbers is exactly what this project avoids.
