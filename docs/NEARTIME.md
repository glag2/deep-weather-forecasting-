# Data latency and near real time sources

## 1. The latency of ERA5, measured

The claim "ERA5 has 5-6 days of latency" was not taken from the documentation: it was
**measured against the service**, on 2026-08-17.

**Collection metadata.** The temporal extent declared by
`reanalysis-era5-single-levels` ends at **2026-08-11**.

**Direct probe.** Metadata are a declaration, not a proof, so a real download of a single
cell, one variable, one hour, was attempted:

| requested date | outcome |
|---|---|
| 2026-08-16 | **refused**, HTTP 400, `invalid request` |
| 2026-08-14 | **refused**, HTTP 400, `invalid request` |
| 2026-08-12 | **delivered**, 116 bytes |

**Conclusion.** The last day actually obtainable is 2026-08-12, that is **5 days** before
the current date. The datum of 16 August is **not** available. The latency is real and not
an inherited assumption.

Technical note: for recent dates the CDS serves **ERA5T**, the preliminary version, which
can be revised in the following months. The 5 day latency is already that of the
preliminary product; definitive ERA5 arrives with **2-3 months** of delay.

## 2. The source that covers the hole

**ECMWF Open Data** (`data.ecmwf.int`), a free subset of the operational IFS and AIFS
forecasts.

| | |
|---|---|
| Resolution | **0.25 degrees**, identical to our grid |
| Format | GRIB2 |
| Runs | 4 per day: 00, 06, 12, 18 UTC |
| License | CC-BY-4.0, commercial use allowed with attribution |
| Access | no key: `ecmwf-opendata` client, or AWS, Azure, GCP |
| Step 0 | the **analysis**, that is the estimated state at the moment of the run |

### Coverage of our variables

Verified entry by entry against the IFS `oper`/`fc` parameter table:

| ours | ECMWF Open Data | id |
|---|---|---|
| `t2m` | `2t` 2 metre temperature | 167 |
| `d2m` | `2d` 2 metre dewpoint temperature | 168 |
| `msl` | `msl` Mean sea level pressure | 151 |
| `u10` | `10u` 10 metre U wind component | 165 |
| `v10` | `10v` 10 metre V wind component | 166 |
| `tcc` | `tcc` Total cloud cover | 164 |
| `sd` | `sd` Snow depth water equivalent | 141 |
| `tp` | `tp` Total precipitation | 228 |
| `sf` | `sf` Snowfall water equivalent | 144 |
| `lsm` | `lsm` Land Sea Mask | 172 |
| `z` | `z` Geopotential (step 0) | 129 |

**All eleven are available.** There is no need to give up any channel nor to retrain with
a reduced set.

## 3. The constraint that decides how it is used

> "Data are retained for the most recent 12 forecast runs, corresponding to
> approximately 2-3 days of forecasts."

The archive is **rolling**. It is not a historical archive: it allows taking the last two
or three days, not recovering the past.

A sharp operational consequence follows:

- **Today** the days 13 and 14 August are not recoverable from either of the two sources:
  ERA5 reaches the 11th-12th, the rolling archive starts from the 14th-15th.
- **From today onwards**, downloading every day and archiving locally, the series stays
  continuous and the latency drops practically to zero.

The hole exists **only once**, at the start. It has to be opened now, not when it will be
needed.

## 4. The scientific problem not to be ignored

ERA5 and IFS **are not the same product**.

ERA5 is a reanalysis: it assimilates observations *even later* than the instant described
and uses a frozen version of the model. The IFS analysis is produced in real time, with
the current operational cycle, and sees only the observations that have already arrived.

Gluing one after the other means changing the distribution of the data **exactly in the
most recent slots**, which are the most influential on the forecast, and which the diurnal
anchoring uses as a reference. A model trained only on ERA5 would meet in production an
input slightly different from anything it has seen.

It is not a reason to give up, it is a reason to **measure**. The two sources overlap in a
window of two or three days, and this allows a direct check:

1. download the same instants from both sources in the overlap window;
2. measure, per variable, the mean deviation and its spatial structure;
3. if the deviation is small compared with the error of the model, use them together;
4. if it is not, correct it explicitly or limit the use of the operational source to the
   less sensitive channels.

This check has to be done **before** declaring the pipeline operational, not after.

## 5. The extra opportunity

ECMWF Open Data does not provide only the analysis: it provides the IFS and AIFS
**forecasts** at 0.25 degrees, up to 144 hours for the 06 and 18 UTC runs, with steps of
3 hours.

They are exactly the same task that this project performs, produced by the best
forecasting centre in the world. They therefore become **a true operational reference**,
much more severe than diurnal persistence, and available for free.

The comparison would be merciless and useful for that very reason: it says how far a model
trained on CPU locally is from the state of the art, instead of limiting itself to saying
that it beats the repetition of yesterday.

## 5-bis. The real retention is much longer than the declared one

The ECMWF documentation declares a rolling window of a few days. Taken literally, it would
make it impossible to recover a hole just discovered, and the missing days of 13-14 August
2026 would be lost forever.

Verified instead against the public mirror on AWS
(`ecmwf-forecasts.s3.eu-central-1.amazonaws.com`, listed on 2026-08-18):

| check | outcome |
|---|---|
| 13, 14, 15 August 2026 present | yes |
| depth of the archive | 2023-06 present, 2023-01 absent |
| size of an `oper` step 0 run | 130.9 MB actual |
| fields in the index of the run | 187 |
| variables of the project present in the index | **11 out of 11** |

Two practical consequences.

The first: the hole **can be filled from the same family of data** that will serve for
near real time, instead of staying uncovered while waiting for ERA5 to arrive. The second:
the archive reaches years back, so the overlap window with ERA5 on which to measure the
deviation between the two sources is not two days but **months**, and the measurement of
point 4 becomes much more solid than expected.

Operational details verified in the field, which cost time if discovered downstream:

- the name of the index file **replaces** the extension, it does not add to it: the index
  of `...-oper-fc.grib2` is `...-oper-fc.index`, not `...-oper-fc.grib2.index`;
- the mirror answers `503 SlowDown` easily: a progressive wait between requests is needed,
  otherwise the listing stops halfway with no evident error.

## 6. What to do, in order

1. Open the daily collection from ECMWF Open Data right away, so the hole stays limited to
   the days already lost.
2. Measure the deviation of ERA5 against the IFS analysis in the overlap window.
3. Only afterwards, allow the forecast pipeline to draw on the operational source for the
   most recent slots.
4. Add the IFS forecast as a third comparison term in the evaluation.

## Sources

- Temporal extent and download probes: measured against the CDS on 2026-08-17.
- Parameter catalogue, license, resolution and retention policy:
  <https://www.ecmwf.int/en/forecasts/datasets/open-data>, consulted on 2026-08-17.
- Access client: <https://github.com/ecmwf/ecmwf-opendata>.
