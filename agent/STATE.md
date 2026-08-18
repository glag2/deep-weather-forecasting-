# Project state

Updated: 2026-08-19, morning. Whoever changes something substantial rewrites this file.

## In one sentence

The complete pipeline exists and runs: download, ingestion, channel construction,
training, evaluation, forecast, calibration, PDF, local site. The full scale training is
**finished and evaluated on the test set**: the model now **beats diurnal persistence on
temperature at all nine lead times** (2.917 against 3.160 degrees of RMSE). Two measured
weaknesses remain: **it loses as a classifier on snow** and its **rain probability depends
too little on the lead time**.

## Numbers needed to get oriented

| quantity | value | where it is verified |
|---|---|---|
| Grid | 261 x 401, 0.25 degree step | `configs/default.yaml` |
| Catalogued slots | 2862 | `scripts/analyze_data.py` |
| Slots present in the store | 1458 | same command, `usable` column |
| Input channels (7 days) | 245 | `InputLayout.from_config` |
| Output channels | 45 | `OutputLayout` |
| Model parameters | 9 979 053 | `scripts/benchmark_model.py` |
| Time per epoch at full scale | 266 s (median over 24) | `models/fold_00/history.json` |
| Best epoch / validation | 16 / -0.7567 | same file |
| Windows in the test block | 241 | `models/fold_00/metrics.parquet` |
| Noise between seeds on the bench | 0.059 degrees | `docs/varianti.md` |
| Noise between seeds on the days bench | 0.045 degrees | `INPUT_DAYS.md` |
| Automated tests | about 740 | `pytest -q` |

**The slots are not equally spaced**: 06, 12, 18 UTC means 6, 6 and 12 hours. Three slots
are a day, not eighteen hours. A labelling error on this point has already made the
correlation with the lead time look non monotonic.

## What is running now

| process | status |
|---|---|
| 2024 download | **finished**, 23 months out of 23 |
| Comparison on the input days | **finished**, all four settings |
| Full scale training | **finished**, 24 epochs, best is 16 with -0.7567 |
| Evaluation on the test set | **finished**, 241 windows, fold 0 |

No long process is running.

The model that came out finished at epoch 16 **was destroyed** by a bench run that was
writing into the same folder. The folder is kept as `models/fold_00_contaminato` for
testing purposes and the dashboard now has a check, `coerenza_artefatti`, that recognizes
exactly that situation. The training relaunched from scratch reproduced **exactly** the
same result (same seed): best epoch 16, validation -0.7567. Reproducibility is therefore
verified by accident.

## Results on the test set, fold 0, 241 windows

The two references are `persistence` (repeat the last slot) and `persistence_diurnal`
(repeat the same slot of the previous day). The second one is the serious one.

| quantity | model | diurnal | naive |
|---|---|---|---|
| t2m RMSE (degrees) | **2.917** | 3.160 | 4.733 |
| t2m MAE (degrees) | **2.058** | 2.165 | 3.144 |
| rain Brier | **0.181** | 0.274 | 0.261 |
| rain calibration error | **0.053** | 0.274 | 0.261 |
| rain F1 | **0.635** | 0.609 | 0.624 |
| snow Brier | **0.066** | 0.081 | 0.075 |
| snow F1 | 0.493 | 0.561 | **0.590** |
| snow accuracy | 0.847 | 0.919 | **0.925** |

On temperature it wins at **all nine** lead times: 2.02 against 2.43 at the first, 3.40
against 3.68 at the last. The error grows in steps of three slots, that is at the change
of day, as is expected with the diurnal anchoring.

The accuracy on snow **must not be used**: the base rate is 0.091, so always answering
"no" would give 0.909, more than the model and than both references.

## Recently decided, with the measurement that supports it

| decision | measurement |
|---|---|
| Input window of **7 days** | gains: 3 d -0.033; 7 d +0.065; 10 d -0.050; 14 d +0.001. Seven and fourteen are indistinguishable (margin 0.063 against uncertainty 0.090), so the cheapest one wins: 245 channels instead of 455. |
| The 13-14 August gap **is recoverable** | the AWS mirror of ECMWF Open Data has those days and goes back to at least mid 2023, with all 11 variables in the index. |
| The training is commanded from the site | every run is a separate process with its own folder and configuration, verified by starting a real one from the browser. |
| Every checkpoint carries the **fingerprint of the data** it was trained on | `data_fingerprint` records window lengths, catalogued and usable slots and the boundaries per split; `confronta_impronte` reports the deviation. It is needed because reingesting 2024 **shifts the fold boundaries**: without the fingerprint, models trained on different data would be compared silently. |

## Open problems

1. ~~The model loses on temperature against diurnal persistence.~~ **Solved and verified
   at full scale**: 2.917 against 3.160 degrees on the test set, a win at all nine lead
   times. The diurnal anchoring was the right fix.
1-bis. **The rain probability depends too little on the lead time.** The Brier of the
   model goes from 0.168 at the first lead time to 0.188 at the last, that is it worsens by
   12 % in three days, while naive persistence goes from 0.165 to 0.297, that is by
   80 %. At the **first** lead time the model therefore **loses** against simply
   repeating the last observation. The probability field is rich in space but almost
   static in time: it resembles a climatology conditioned on the initial state. It is the
   behavior predicted by the literature on the double penalty
   (`RESEARCH.md`): blurring minimizes the squared error on a chaotic field. It is
   fixed by changing the objective function, not the data. **This is now the most
   interesting open problem.**
1-ter. **On snow the model loses as a classifier**: F1 0.493 against 0.590. It has
   recall 0.818 against 0.595 but precision 0.352 against 0.585, that is it signals a lot
   and is often wrong. Its *probability* stays better though (Brier 0.066 against
   0.075): the information is there, it is the threshold that is shifted towards caution.
   Before "fixing it", decide whether a cautious alarm or a balanced classifier is wanted:
   they are different objectives.
2. ~~The site does not behave as it should.~~ **Solved.** Diagnosis carried out section by
   section with the times measured: the fold structure returned 12 690 rows, the table of
   the validation metrics was empty without saying so, the error map cost 46 s at every
   interaction and the model page read keys that the checkpoint does not write. All fixed;
   the Climate and Training pages added.
3. **The architectures are not distinguishable** from each other: the spread between the
   five variants is 0.024 degrees against a noise between seeds of 0.059. Declaring a
   winner would mean reading the seed.
4. **Vigo di Cadore is at 1463 metres in the model against 951 real ones.** No network
   change compensates 512 metres of elevation.
5. **The CDS token has to be rotated**: it was pasted in clear text in a conversation.

## Defects found and fixed, not to be reintroduced

Partial list, the most instructive ones. Details in `DECISIONS.md`.

- `paths.artifacts_subdir` did not affect `fold_dir`: every bench run wrote into the same
  folder. It broke the comparison on the days when a second training was running in
  parallel. Fixed with `paths.models_subdir` and two tests.
- `scripts/download_era5.py` had its own copy of the manifest reader, the strict one,
  while in `freshness` the tolerant one already existed. The backfill died on an old
  manifest.
- `persistence.py` rejected `complex64`, so the spectral variants were not saved. The
  right criterion is not "real" but "fixed size binary representation".
- The dashboard counted the **catalogued** slots instead of the ones present: 2862 instead
  of 1458.
- `.streamlit/config.toml` written by PowerShell with the BOM was ignored silently. It
  holds for any configuration file written by PowerShell.
- The dashboard was reachable from the network. Now it is bound to 127.0.0.1.
- `folds.parquet` freezes the window length used at the time of the ingestion. Making it
  longer afterwards caused reading starts validated for a shorter window, whose tails fell
  on slots never ingested: non finite loss from the first epoch and no checkpoint,
  silently. Visible only at 10 and 14 days (14 % and 33 % of the windows), invisible at 3
  and 7. Now `sample_starts` rechecks and a non finite loss raises.
- `model_copy(update=...)` **does not revalidate**: it would have accepted zero epochs and
  a negative learning rate from the web module. The parameters are rebuilt with
  `model_validate`.
- The variant name was not validated: a nonexistent name failed only at the construction
  of the network, after reading the data. Now the schema checks it.
- A process number, by itself, does not identify a process: they get recycled. The state of
  a run uses the pair number plus birth instant.
- The bench verdict on the days compared a difference between **means** with the
  dispersion of a **single** measurement, declaring real half of the noise.

## Where to restart from

`PLAN.md`, Phase 0. The steps are in order and each one says when it is finished.

Recommended order now that the evaluation is frozen:

1. **Ingest the 2024 already downloaded.** It has to be done now, not before: it shifts
   the fold boundaries and invalidates the numbers above. The data fingerprint will make
   the shift visible instead of silent. After the ingestion **all** the metrics have to be
   regenerated and this file rewritten.
2. **Attack the flatness of the rain probability** (problem 1-bis). It is work on the
   objective function. The spectral loss in `weighting.py` already exists but it is not
   active in the default configuration.
3. Fill 13-15 August from the AWS mirror and measure the departure of ERA5 against IFS on
   the overlap, which is now much wider.
4. Consolidate the twelve branches. **It requires a decision from the user**: no merge and
   no push have been performed.
