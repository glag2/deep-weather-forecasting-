# Conventions and traps

Local rules that cannot be deduced by reading the code, and traps that have already cost
time at least once.

## Language

Code, comments, documents and user-facing messages in **Italian**, without accents in the
sources (write `perche'`, not `perché`): it avoids encoding surprises between Windows,
Linux and Docker. **Commit messages** are in **English**.

## Names

Descriptive, spelled out names, even long ones. No opaque abbreviations and no
single letter variables, except for obvious mathematical conventions. A function called
`rmse_su_validazione` says what it returns; one called `ev` does not.

## Git

- **One branch per task**, with a name that describes the task.
- Never commit directly on `main`.
- Staging **file by file**. Never `git add -A`.
- Message in English, imperative, explaining **why**, not what: the diff already says
  what.
- No push, merge, rebase or destructive reset without explicit authorization from the
  user, asked immediately before the operation.
- Before every commit: `pytest -q` and `ruff check`.

## Verification

No number enters a document without the command that produces it. If something has not
been measured, it is written that it has not been measured. Formulas are called
"verified" only after they have been executed.

The project has changed direction three times because of a measurement that contradicted
an expectation. The rule matters most when the result is convenient.

## Environment traps

**PowerShell 5.1**: it does not know `&&` or `||`, commands are chained with `;`. It has
no here-doc. For files longer than a few lines use the writing tool, not shell echo.

**The BOM.** PowerShell writes UTF-8 **with** BOM. Several readers reject it silently:
it has already happened with the CDS credentials and with the Streamlit configuration,
the same error twice. If a configuration file looks ignored, the first thing to check is
the first three bytes.

**`data` versus `Data`.** Windows is not case sensitive: the data root is called
`datasets/`, not `data/`, otherwise it collides with the tracked folder. And in
`.gitignore` the rule is anchored (`/datasets/`), because without the anchor it also hid
`src/dwf/data/`.

## Domain traps

**The slots are not equally spaced.** 06, 12, 18 UTC: 6, 6, 12 hours. Three slots make a
day. Any computation that multiplies the index by a fixed interval is wrong.

**The serious reference is diurnal persistence**, not the naive one. Repeating the last
slot is a weak reference: beating it means nothing. Repeating the same slot of the
previous day is much harder to beat.

**GRIB packs each field with independent integers.** That is why `sf > tp` is observed in
about 12 per cent of the points: it is not a data error, it is the quantization error of
the two fields, and it is within one and a half packing steps. The data is true: if
something does not add up, the error is ours.

**The accumulated fields have a different axis order** from the instantaneous ones. The
transfer requires an explicit transposition: it is declared in `ACCUMULATED_DIMS`.

## Models on disk

Saved in `.npz` with `allow_pickle=False`. `torch.load` with `weights_only=False` is not
used: a checkpoint is a file that can come from outside, and with pickle loading it means
executing it. There is a test that demonstrates this with a genuinely malicious payload.

The allowed types are those with a **fixed size binary representation**: reals, integers,
booleans, and also **complex** numbers, which the spectral variants need.

## Model folders

Every run other than the main one must isolate itself with `paths.models_subdir`.
Without it, all of them write into `models/fold_00` and overwrite each other: it has
already happened, and the symptom was an incomprehensible error about the number of
channels.

## Tests

Against real behavior, not form. A test checking that a schema is parseable is useless;
one checking that an inconsistent configuration is rejected is useful. In the network
tests the pause between attempts must be zeroed by a fixture: without it, the suite went
from 9 to 338 seconds.
