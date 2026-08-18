# Agent folder

Documents meant for whoever **develops** the project, human or agent. They are not
reading material for someone who only wants to use the forecaster: that lives in the
root (`README.md`) and in `docs/`.

The separation is deliberate. A document that has to explain to a user how to get a
forecast and a document that has to let an agent pick the work back up cold have
different readers, different lifetimes and different criteria of truth: keeping them
together means neither one stays up to date.

## What to read, and in what order

If you are picking the project back up without having seen anything of what happened
before, read in this sequence:

| order | file | answers |
|---|---|---|
| 1 | [`STATE.md`](STATE.md) | where we are now, what is running, what is broken |
| 2 | [`PLAN.md`](PLAN.md) | what has to be done, in what order, with which completion criterion |
| 3 | [`CONVENTIONS.md`](CONVENTIONS.md) | how work is done here, and which traps have already cost time |
| 4 | [`DECISIONS.md`](DECISIONS.md) | why things are the way they are, with the measurements that prove it |

`STATE.md` must be rewritten every time something really changes. `PLAN.md` must be
updated by marking the finished steps and adding the ones that come up. `DECISIONS.md`
is append only: a superseded decision is annotated as superseded, not deleted.

## The rule that counts more than the others

The statements in these documents must be **verifiable**. If a number is written here,
somewhere there is the command that produces it, and it is stated. If something has not
been measured, it must be written that it has not been measured.

The project has already changed direction three times because a measurement contradicted
a reasonable expectation: the diurnal persistence beating the model, the architectures
indistinguishable from the noise between seeds, the ERA5 latency confirmed only after
probing it. A document reporting impressions instead of measurements would have hidden
all three.
