# Development plan

Every entry has a **dedicated branch**, a description of what has to be done and a
verifiable **completion criterion**. A step is not finished because the code exists: it is
finished when the criterion is satisfied and the verification has really been run.

Status legend: `[ ]` to do, `[~]` in progress, `[x]` done, `[!]` blocked.

The steps marked **R** are recurring reviews. They are not at the end: they are
interleaved, because a review done only at the end finds problems when they cost the most
to fix.

---

## Phase 0 — close the work already open

Nothing new starts until this phase is closed: leaving experiments half done is the
fastest way to lose the ability to say which number belongs to which code.

- [~] **0.1 Input window** — `feature/input-window-screening`
  Complete the comparison at 3, 7, 10, 14 days and choose the window.
  *Done when*: `docs/finestra-input.md` reports the four settings with at least two seeds
  each, the gain is compared with the noise between seeds (0.059 degrees) and the choice
  is motivated or declared indistinguishable.
  *Note*: 10 and 14 days had failed because of a checkpoint path defect, fixed in
  `Config.fold_dir` with `paths.models_subdir`.

- [ ] **0.2 Honest evaluation of the full scale model** — `feature/full-scale-evaluation`
  The trained model (fold 0, best at epoch 16) has to be measured on the **test** block,
  never seen, against naive persistence and diurnal persistence.
  *Done when*: there is a table with t2m RMSE, snow Brier and F1 for the model and for
  both references, and the cases where the model loses are reported, not omitted.

- [ ] **0.3 Performance table in the README** — `docs/pipeline-and-metrics`
  Fill the table left empty with the numbers from 0.2.
  *Done when*: no cell contains a placeholder, and every number is reproducible with the
  command indicated in the line above the table.

- [ ] **0.4 Ingestion of the 2024 backfill** — `feature/backfill-2024`
  Bring the downloaded months into the store and verify the continuity of the series.
  *Done when*: `scripts/analyze_data.py` reports no unexpected gaps and the count of the
  slots present matches the one of the ingested months.

- [ ] **0.5 End to end run** — `test/end-to-end`
  Download, ingest, train a few epochs, evaluate, forecast, generate the PDF, regenerate
  the two notebooks, all inside the Docker image.
  *Done when*: the sequence runs without manual intervention and the produced PDF is
  readable.

---

## R1 — Code review and technology research (after Phase 0)

Two distinct activities that are worth doing together, because the second suggests where
to look in the first.

- [ ] **R1.a Review** — `review/r1`
  Reread the modules touched by Phase 0 looking for: duplicated logic (it has already
  happened with the manifest reader), paths built by hand instead of through `Config`,
  numeric values written in more than one place, `except` blocks that hide errors,
  functions longer than one screen.
  *Done when*: there is a list of the defects found with severity and file, the serious
  ones are fixed, the others are noted in `STATE.md`.

- [ ] **R1.b Research on technologies useful to forecasting** — `research/r1`
  Not "read papers", but answer a precise question: *what, applied to this grid with this
  compute power, could reduce the error?*
  Leads already identified and to be evaluated, in order of ratio between expected benefit
  and cost:
  1. **CRPS loss** in place of the Gaussian one: it measures the probabilistic forecast
     without assuming its shape.
  2. **Diffusion on the residuals**: it generates scenarios instead of the mean only, and
     it mitigates the double penalty problem demonstrated in `docs/ricerca.md`.
  3. **Downscaling towards Vigo**: the cell is at 1463 metres against 951 real ones; a
     correction based on the elevation can be worth more than any network change.
  4. **Multi source inputs**: IFS analysis alongside ERA5 (see Phase 1).
  5. **Attention along the time axis** instead of the spatial one: the slots are not
     equally spaced, and the current network ignores it.
  *Done when*: every lead has a line in `docs/ricerca.md` with estimated cost, expected
  benefit and the reason why it was taken or discarded; at least one is tried on the
  bench.

---

## Phase 1 — The data: automatic acquisition, archive without duplicates

The goal is that the program **fetches by itself** what it needs, and nothing more.
Whoever wants only a forecast must not download years of history; whoever wants to train
must be able to choose the months.

- [ ] **1.1 Fill the 13-14 August gap** — `feature/opendata-backfill`
  Verified: the AWS mirror of ECMWF Open Data keeps those days, while the portal has
  already let them scroll away. See `docs/tempo-reale.md`.
  *Done when*: the two days are in the store, marked with the source they come from.

- [ ] **1.2 Explicit provenance** — `feature/data-provenance`
  Every slot has to know where it comes from: definitive ERA5, preliminary ERA5T, IFS
  analysis. Without this field, mixing the sources makes it impossible to understand an
  error.
  *Done when*: the store has a `fonte` coordinate, the tables report it and the dashboard
  shows it.

- [ ] **1.3 Measure the departure of ERA5 against the IFS analysis** — `research/era5-vs-ifs`
  The two sources overlap by two or three days: there the departure is measured instead of
  assumed. It has to be done **before** using them together in production.
  *Done when*: for every variable there is the mean departure, its dispersion and the
  spatial structure, compared with the model error.

- [ ] **1.4 Automatic daily collection** — `feature/daily-collection`
  The open archive is a rolling one: what is not taken today is lost. A recurring
  collection has to be started right away.
  *Done when*: a single command updates the store to the latest available slot and is
  idempotent, that is rerunning it downloads nothing already present.

- [ ] **1.5 Archive without duplicates** — `feature/store-dedup`
  Today the same instant can arrive from more than one download. A unique key per slot is
  needed, plus the precedence rule between sources (definitive ERA5 beats ERA5T, which
  beats the IFS analysis) and a check that **no information is lost** in the
  deduplication: when a duplicate is discarded, what and why is recorded.
  *Done when*: a run with duplicates built by hand demonstrates that the winner is the
  expected one, that the slot count does not change and that the occupied space drops.

- [ ] **1.6 On demand acquisition** — `feature/on-demand-fetch`
  The piece that makes the project usable by someone arriving for the first time.
  - **forecast** mode: given a model and an instant, the program computes by itself the
    input window needed, checks what is missing and downloads **only that**;
  - **training** mode: the user chooses the months, the program computes what is missing
    and downloads only the difference.
  *Done when*: from an empty store, asking for a forecast downloads a volume equal to the
  window and no more, measured in megabytes; and asking for the same forecast again
  downloads nothing.

- [ ] **1.7 Estimate before downloading** — `feature/download-estimate`
  Before starting, say how many files, how many megabytes and how much time. The
  measurement already exists: about 387 MB and nine minutes per month.
  *Done when*: the estimate appears from the command line and on the site, and the
  departure between estimated and real is recorded in order to correct it.

---

## Phase 2 — The site has to work

The current state is not acceptable: the dashboard exists but it does not behave as it
should. Before adding pages, what is there has to be made solid.

- [ ] **2.1 Diagnosis** — `fix/dashboard-reliability`
  Open every section with the real store and note what does not work: errors, long waits,
  wrong numbers, empty charts. No fix before having the list.
  *Done when*: the list exists, with the cause of each problem, not only the symptom.

- [ ] **2.2 Fixes** — same branch
  *Done when*: every entry of the list is closed or declared unsolvable with the reason;
  no section shows an error trace to the user.

- [ ] **2.3 Behavior when something is missing** — same branch
  Empty store, missing model, forecast never run: the site has to **say what to do**,
  neither break nor show a blank page.
  *Done when*: there is a run that starts from an empty data folder and gets readable
  instructions in every section.

- [ ] **2.4 Response times** — same branch
  Measure how long every section takes. Whatever exceeds two seconds has to be cached or
  computed in advance.
  *Done when*: the table of times exists and no section exceeds two seconds on the second
  load.

---

## Phase 3 — Control of the training from the site

- [ ] **3.1 See the input data** — `feature/dashboard-training`
  Choose instant and variable, see the maps of the channels that really enter the network,
  including the built ones (solar, thermodynamic, tendencies), with the values in physical
  units and not normalized.
  *Done when*: the whole input window can be navigated and every channel shows name, unit
  and provenance.

- [ ] **3.2 See the output data** — same branch
  The nine lead times, mean and uncertainty, next to the observed value where it exists.
  *Done when*: forecast, observation and difference are visible together for every lead
  time and variable.

- [ ] **3.3 Change the parameters** — same branch
  Epochs, learning rate, crop size, samples per epoch, variant, loss weights, seed. The
  values have to be **validated with the same schema** as the configuration: the site must
  not be able to build a configuration that the program would reject.
  *Done when*: a value out of range is rejected with the same message it would give from
  the command line.

- [ ] **3.4 Start and follow the training** — same branch
  Start, stop, progress, loss curve that updates, processor and memory usage. The training
  runs in a separate process: the site must not block.
  *Done when*: a short training is started from the site, followed, stopped, and the
  produced model is loadable.

- [ ] **3.5 Comparison between runs** — same branch
  Every run saves configuration and result in its own folder (`paths.models_subdir`
  already exists). The site lists them and compares them.
  *Done when*: two runs with different parameters are comparable in a table without
  touching the filesystem by hand.

---

## R2 — Review and research (after Phase 3)

- [ ] **R2.a Review** — `review/r2`
  Focused on the boundary between site and library: the site must not contain computation
  logic, it has to call it. If a function exists only for the dashboard, it is in the wrong
  place.
- [ ] **R2.b Research** — `research/r2`
  Reassess the leads of R1.b with what has been learned in the meantime, and try at least
  one more.

---

## Phase 4 — Climate change

The project has an hourly series on a wide domain: it is material suitable for showing
trends, provided it is said honestly how solid they are.

- [ ] **4.1 Basic trends** — `feature/dashboard-climate`
  Mean temperature per year and per season, with trend line and its uncertainty, on the
  whole domain, on subregions and on the Vigo cell.
  *Done when*: every trend reports slope, confidence interval and number of years it is
  computed on.

- [ ] **4.2 Map of the trends** — same branch
  The warming is not uniform: the per cell map shows it better than any mean.
  *Done when*: the map exists with a diverging scale centred on zero and the non
  significant areas are distinguished.

- [ ] **4.3 Extremes and characteristic days** — same branch
  Frost days, tropical nights, intense precipitation, snow depth: counts per year.
  *Done when*: every index has its definition written next to the chart.

- [ ] **4.4 Statistical honesty** — same branch
  With few years a climate trend **is not significant**. It has to be said on the page,
  not hidden.
  *Done when*: the page states the period covered and warns when it is too short to
  conclude.

---

## Phase 5 — Longer forecast horizon

- [ ] **5.1 How far it makes sense** — `research/longer-horizon`
  Measure when the model stops beating diurnal persistence and climatology. Extending
  beyond that point produces numbers, not forecasts.
  *Done when*: the error curve per lead time exists with the two thresholds marked.

- [ ] **5.2 Extension** — `feature/longer-horizon`
  Take the output from 3 to 5-7 days if 5.1 justifies it, evaluating the two roads: a wider
  direct output, or repeated application of the model on itself.
  *Done when*: the two roads are compared on the same test block.

- [ ] **5.3 Comparison with IFS** — `feature/ifs-baseline`
  The IFS forecasts at 144 hours are free and they are the state of the art. As a reference
  they are much harsher than persistence.
  *Done when*: the evaluation reports three references: naive, diurnal, IFS.

---

## Phase 6 — Understand the model

Explicit request: it has to be possible to understand how it works and what is behind it.

- [ ] **6.1 Weight of the inputs** — `feature/explainability-inputs`
  Which channels matter, per variable and per lead time, with occlusion or gradient.
  *Done when*: the ranking per predicted variable exists, with the method stated and its
  limits.

- [ ] **6.2 Spatial reach** — same branch
  From how far away comes the information that determines a cell. It is measured by
  perturbing and observing where the output changes.
  *Done when*: the influence map exists for at least one significant cell (Vigo).

- [ ] **6.3 Anatomy of the network** — `feature/explainability-model`
  Diagram of the blocks, tensor shapes, parameters per block, what each head does.
  *Done when*: the path of a tensor from input to output can be followed without reading
  the code.

- [ ] **6.4 Where it gets it wrong** — `feature/explainability-errors`
  The error per region, season, hour of the day, elevation, sea against land.
  *Done when*: the breakdowns exist and at least one error regime is explained.

- [ ] **6.5 Why this forecast** — same branch
  For a specific forecast: how much comes from the diurnal anchoring and how much from the
  network correction.
  *Done when*: the breakdown appears next to the forecast map.

---

## R3 — Review and research (after Phase 6)

- [ ] **R3.a Review** — `review/r3`
- [ ] **R3.b Research** — `research/r3`
  Last chance to introduce a technique before the consolidation.

---

## Phase 7 — The site as the central point

Constraint to respect: **everything must stay executable from Python**. The site is one
more way in, not the only one; if a function exists only there, it is a structural error.

- [ ] **7.1 Structure of the pages** — `feature/dashboard-hub`
  Reorganize into real pages instead of a sequence of sections: Overview, Data, Model,
  Training, Forecast, Evaluation, Climate, System.
  *Done when*: every page has a stated purpose and no information appears in two places
  with two different values.

- [ ] **7.2 Forecast from the site** — same branch
  Choose model and instant, have the program download what is missing (Phase 1.6), get the
  forecast, download the PDF.
  *Done when*: from an empty store and without touching the terminal you get to the PDF.

- [ ] **7.3 Data management from the site** — same branch
  See what is there, choose the months, start the download, follow it.
  *Done when*: the operations of Phase 1 are available from the site with the same
  validation as the command line.

- [ ] **7.4 All the metrics, ordered** — same branch
  The infographics already produced (calibration, reliability, error maps, comparison
  between variants, feature screening) have to be collected where they are needed, not
  piled up.
  *Done when*: every chart produced by the project is reachable from the site and has a
  line saying what it shows and how to read it.

- [ ] **7.5 Parity between the two ways** — same branch
  *Done when*: the table exists that for every operation indicates the Python command and
  the corresponding page, and there are no empty cells.

---

## Phase 8 — Delivery

- [ ] **8.1 Reorganization of the documents** — `docs/reorganization`
  User documents in `docs/`, development documents in `agent/`. It requires having merged
  the branches first: **the explicit authorization of the user is needed** for the merges.
- [ ] **8.2 Final report** — `docs/final-report`
  Very detailed: what it does, how, with which measured results, with which limits.
- [ ] **8.3 Cold run** — `test/cold-start`
  From a clean machine: installation, first startup, first forecast. Timed.
  *Done when*: a reader who does not know the project gets to a forecast following only
  the README.

---

## How this plan is updated

Whoever works marks `[~]` when they start and `[x]` when the criterion is satisfied. A
step that turns out to be wrong is not deleted: it is marked superseded and the reason is
written. New steps are added in the relevant phase with the same format.
