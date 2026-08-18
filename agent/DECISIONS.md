# Decision log

Why the project is built this way. Append only: a superseded decision is annotated as
superseded, not deleted, because knowing what was tried and did not work is worth as much
as knowing what works.

Every entry reports the **proof**, not the opinion.

---

## D1 — The data root is called `datasets/`

Windows is not case sensitive: `data/` would have collided with the `Data/` tracked in
the repository, mixing generated gigabytes and versioned sources. The rule in
`.gitignore` is anchored (`/datasets/`) because without the anchor it also hid
`src/dwf/data/`, which is code.

## D2 — Zarr for the tensor, Polars for the registry

The proposal to keep everything in Polars was evaluated. The tensor is a dense
dimensional array of hundreds of millions of cells: a columnar format would represent it
as a long table with the coordinates repeated for every value, multiplying the footprint.
Zarr preserves the shape and allows reading a crop without touching the rest.
Polars stays where it is good: the slot catalog, the download manifest, the metrics.

## D3 — The slots are not equally spaced

06, 12, 18 UTC are 6, 6 and 12 hours apart. Discovered because the correlation with the
lead time looked non monotonic, with maxima at positions 3, 6 and 9: it was not a
physical phenomenon, they were the multiples of a day. The label `lead x 6 ore` was
wrong.

Direct consequence: the serious reference is not repeating the last slot, it is repeating
**the same slot of the previous day**. Measured at 24 hours: naive persistence 4.535
degrees, diurnal persistence 2.401.

## D4 — The diurnal anchoring is the only effect outside the noise

The model predicts the departure from diurnal persistence instead of the absolute value.
On the bench: without anchoring 6.613 degrees, with anchoring about 3.625. That is about
fifty times the dispersion between seeds. No other architectural choice comes close.

## D5 — The architectures are not distinguishable, and no winner is declared

Five variants (convolutional, attention, spectral, recurrent, hybrid) cover the range
3.642-3.666 degrees, that is 0.024 of spread. Repeating the convolutional variant alone
with three different seeds: 3.652, 3.557, 3.665, that is a dispersion of 0.059.

**The noise is more than twice the signal.** Naming a winner would mean reporting which
seed was lucky. The variants all stay available in the registry; the default is the
convolutional one because it is the cheapest among the equivalent ones (99 seconds per
epoch against 428 for the recurrent one).

Consistent with the literature read: in `docs/ricerca.md`, arXiv:2407.14129 and
arXiv:2501.19374 report that the backbones saturate and that the gain comes from the data
and from the formulation, not from the block.

## D6 — The model loses on temperature, and that has to be said

Fold 0, test block, before the anchoring: t2m RMSE 4.45 for the model, 4.73 for naive
persistence, **3.16 for the diurnal one**.

The model wins clearly instead on the probabilities: rain Brier 0.182 against 0.274,
skill gain +0.193; snow Brier 0.061.

Reporting only the comparison with naive persistence would have given a false picture.

## D7 — Models are saved in `.npz`, not with pickle

A checkpoint is a file that can come from outside. With pickle, loading it means
executing it. The defense is `numpy.savez` read with `allow_pickle=False`.

Demonstrated with a genuinely malicious payload: under pickle it is executed, through the
project reader it is rejected.

The criterion is not "real numbers only" but **fixed size binary representation**. The
narrow formulation excluded complex numbers and prevented saving the spectral variants;
after verifying that NumPy handles them without pickle, they were allowed.

## D8 — The `sf > tp` anomaly is packing, not an error

In 12.26 per cent of the points the snow comes out higher than the total precipitation,
which is physically impossible. Cause: GRIB quantizes **each field separately** on
integers, so the two fields have different quantization steps.

Proofs: the violation never exceeds one and a half times the packing step, and its
correlation with the precipitation intensity is 0.013, that is absent. If it were a
physical error it would grow with the intensity.

The data is not corrected. It is true: if something does not add up, the error is ours.

## D9 — The ERA5 latency was probed, not quoted

Collection metadata: end at 2026-08-11. Real download probes: 2026-08-12 delivered,
08-14 and 08-16 rejected with HTTP 400. The five day latency is real.

For recent dates the CDS serves ERA5T, preliminary; definitive ERA5 arrives after two or
three months.

## D10 — The 13-14 August gap is filled from the AWS mirror

The ECMWF portal keeps only the last twelve runs, two or three days. But **the AWS mirror
of the same source keeps much more**: the archive goes back to at least mid 2023.

Verified on 2026-08-17: the 13 August run is present for all four cycles, step 0 of the
`oper` stream weighs 130.9 MB, and its index contains 187 fields including **all eleven**
variables of the project.

The scientific point remains: the IFS analysis is not the ERA5 reanalysis, and replacing
it in the most recent slots shifts the distribution exactly where the model is most
sensitive. The departure has to be measured in the overlap window before using the two
sources together.

## D11 — Every run isolates its own checkpoints

`paths.artifacts_subdir` lives under the data root and did not affect `fold_dir`: setting
it in the bench runs had no effect at all and all the runs wrote into `models/fold_00`.

The defect was invisible as long as the runs were sequential and with the same number of
channels, because each one read back what it had just written. It came out when a full
scale training was running in parallel and overwrote the checkpoint between the training
of a run and its evaluation: error
`il checkpoint attende 245 canali, la configurazione ne produce 335`.

Fixed by introducing `paths.models_subdir`, which enters `fold_dir`, with a check that
the resolved path stays under the models folder, because the value can come from the
command line.

## D12 — The site listens only on 127.0.0.1

At startup Streamlit also announced a network address: the dashboard was reachable from
other machines and it exposes internal structure and paths. Bound to the loopback and
verified with `netstat`.

The first configuration attempt had no effect because PowerShell had written the file
with the BOM. It is the second time the BOM costs time in this project.
