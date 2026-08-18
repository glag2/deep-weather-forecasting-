# Work plan

Living document. It replaces the "What I would do next" section of `PROGRESS.md`, which
had fallen behind. Every entry has a truthful status: if something has not been measured,
it says so.

Legend: **done** verified with an executable proof · **in progress** started · **to
do** not started · **unproven** done but not validated at full scale.

---

## 1. Defects found and closed

| | defect | proof that it was real |
|---|---|---|
| done | `sample_starts` filtered a frozen column: every input length gave 961 windows | four different configurations, same count |
| done | static fields indexed by slot: `lsm` averaging 0.41745 instead of 0.5200 | comparison with the true values in the store |
| done | the diurnal anchoring applied only to temperature, the other heads raised a silenced exception | Brier identical at the first and at the last lead time |
| done | `diurnal_reference_index` did not check the upper bound | property test over every plausible combination |
| done | one training run deleted the checkpoint of another architecture without saying so | two checkpoints really lost |
| done | the checkpoint metadata did not record the architecture | one weights file was indistinguishable from that of another network |
| done | the validation crop changed at every epoch, so the choice of the best epoch contained noise | two different networks both collapse **exactly** at epoch 9: 2.609 -> 1.303 and 2.299 -> 0.976 |
| done | every batch contained four crops of the **same** window: the gradient described a single day | reading the sampler, then `windows_per_batch` |
| done | the report claimed that the excluded settings were worse when none had been excluded | reading the report template |
| done | the three-epoch bench rewards prior terms by construction | the precipitation anchoring adopted at 0.1907 and refuted at full scale, 0.181 against 0.192 |

## 2. The main defect, still open

**The model gains 5.5% over doing nothing at twenty-four hours.** It is not that the
error is 2.27 degrees: it is that 2.40 degrees are obtained by repeating yesterday at the
same hour. Per lead time detail in `PROGRESS.md` section 14.6.

Three measured causes, which have to be tackled together because none of them explains
the result on its own:

| status | cause | measure |
|---|---|---|
| in progress | too little training | 2560 steps, 2.7 visits per window, 0.94 passes over the data |
| in progress | no upper air variable | every input is a surface one, the only geopotential is the static orography |
| to do | insufficient effective spatial reach | 50% of the influence within 130 km, against the 500-1000 km per day of a synoptic structure |

## 3. In progress right now

| status | item | notes |
|---|---|---|
| done | download of pressure levels 2024-2026 | 96 downloaded, 65 already present, 2.2 GB in 120 GRIB files |
| in progress | download of 2022 and 2023 | 47 requests out of 121 as of 18/08 at 16:26; it takes the windows from 961 to about 1750 |
| done | rework of the dashboard interface | two sub-agents, 838 green tests, branch merged |
| in progress | comparison of the soil descriptors | `tmp/ab_base.yaml` 245 channels closed at 0.6776 (epoch 17 out of 20), `tmp/ab_suolo.yaml` 261 channels started |
| to do | ingestion of the pressure levels | in a separate store (`tmp/quota.yaml`), 245 -> 341 channels, then redo the folds |
| to do | long run on the global network | 12,288 steps against 2560, configuration ready |

## 4. Checks never done, in order of risk

These are the things that could be broken without anyone knowing.

| status | what to check | why it is risky |
|---|---|---|
| to do | full domain prediction with the global network | never executed: `predict` and `evaluate` have never seen it, and attention over 1683 tokens is a new path |
| to do | does the evaluation on test use random crops? | validation did it, and if the evaluation does it too then every published metric contains noise |
| to do | ingestion of the pressure levels over a full month | only reading one file was tried, not writing into the Zarr with 341 channels |
| to do | `num_workers > 0` on Windows | the reader cache is per process: with more processes the saving in reads could vanish |
| to do | checkpoint-configuration consistency in dashboard, predict, report | the check on the architecture exists only in `load_checkpoint` |
| to do | `weight_decay: 1e-5` | a value a thousand times lower than the typical one for AdamW, never justified; with more steps the overfitting grows |
| to do | how much of the time per epoch is reading and how much is computation | every decision about the budget rests on an estimate, not on a measurement |
| to do | Docker end to end | ecCodes 2.28 in the container against the recommended 2.42 |
| partly | do the notebooks still run? | `03_collaudo.ipynb` executed cell by cell against fold 0: it runs, and the error map over eight windows exhausts the memory if a training run is in progress, so it stays at four. The first two have not been re-executed. |

## 5. Improvements to try, in order of expected effect

| status | item | why |
|---|---|---|
| to do | more optimization steps | it is the constraint that ties everything else |
| to do | more years of data | 2022-2023 on the way |
| to do | upper air variables, 245 -> 341 channels | the flow at 500 hPa is what drives the weather at mid latitudes |
| done | whole window instead of the crop | `crop_size: null` and `batch_size: 1`. Measured cost: 3000 ms per step over the whole domain against 800 for four crops of 96, that is 28.7 against 21.7 seconds per million points. The whole domain is less efficient per point and is paid for anyway, because the crop trained inside an artificial horizon |
| unproven | global core network | it wins at nine epochs out of ten and it is 2.4 times faster, but twelve epochs are few |
| to do | cosine learning rate schedule | implemented and switched off, never measured here |
| to do | advected anchoring instead of diurnal | the current reference ignores that the air moves |
| to do | pretrained backbones with timm or torchvision | to be tried **paired**, pretrained against random, otherwise shape and weights stay confounded |
| unproven | CMuon optimizer | implemented in `src/dwf/optim.py` and switched off. Orthogonalizing costs 107 ms against the 42 of AdamW, the step goes from 235 to 319 ms: it has to learn a third more per step just to break even, so the comparison has to be made **at equal time**, not at equal epochs |
| unproven | attention sink and compressed context (HCA) | implemented in `global_network.py` and switched off; zero injection, so switching the branch on does not alter the starting point |
| to do | mHC style residuals | from the work on DeepSeek-V4, `RESEARCH.md` section 7 |
| to do | averaging over several seeds | typical error reduction of 3-8%, cost linear in runs |

## 6. Closing the project

| status | item |
|---|---|
| to do | final full scale run, by the user's wish postponed to the end |
| done | PDF report with date and time in the name | `tmp/relazione_dwf_2026-08-18_1745.pdf.json` |
| done | acceptance notebook for a trained model | `notebooks/03_collaudo.ipynb` |
| to do | decide the default for `model.architecture` | the global core wins in validation and in speed, but it has no numbers on test: **the user's consent is needed**, it is a shared default |
| to do | consolidate the branches and prepare the push, with the user approving |
| to do | remind the user about the rotation of the CDS token |
| to do | documentation index in `docs/` |

## 7. Rules imposed by the user, not to be violated

1. **All available data.** Seven days of input, no reduction of the window or of the
   period for the convenience of a test.
2. **Single source**: ERA5 from the CDS, never change it.
3. **No push** without explicit approval immediately before.
4. **One branch per task**, atomic commits, messages in English.
5. **Anomalies are explained, not corrected**: if a datum does not add up, the error is
   in the interpretation until proven otherwise.
6. **Do not wait**: if a computation is in progress, work on something else.
7. **Declared threshold**: below 2 degrees of error at twenty-four hours.
