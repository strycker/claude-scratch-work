# Post-2020 Observations (D-07)

D-06 moved the fence from *looking* to *fitting*: development, tuning, and
model-selection may use only rows dated on or before the 2020-12-31 holdout
cutoff, but the platform notebooks under `notebooks/platform/` read the
**full span**, including post-2020 rows, via
`trading_crab_lib.platform.honesty.holdout.load_full_span` (exposed to
notebooks as `loaders.load_full_span_checkpoint`). Refusing to look at
recent data would let a feature that quietly stopped being predictive stay
weighted with no one noticing.

That looking is not free. Once an operator has seen post-2020 behavior it
cannot be un-seen, and any later claim of an untouched holdout is void for
that decision. This log is the residual honesty guarantee: **any post-2020
observation that changes a decision is recorded here, with its date, so the
influence stays traceable** — this is exactly the record a deflated-Sharpe
denominator would need to account for the number of looks taken.

An empty table (beyond the seeded format-example row below) is a valid
state. It means no post-2020 observation has yet changed a decision — not
that no one has looked. `loaders.load_full_span_checkpoint` logs a WARNING
naming the post-cutoff row count and this file every time a notebook reads
across the boundary, whether or not that look changes anything.

## Log

| date | notebook | observation | decision changed | recorded by |
|------|----------|--------------|-------------------|-------------|
| 2026-09-09 | *(format example — not a real entry)* | *(e.g. "feature X's post-2020 distribution has visibly shifted from its pre-2021 fitted window")* | *(e.g. "flagged X for review in Phase 7's feature screening; no immediate change")* | *(operator name/handle)* |

*(No real observations have been recorded yet as of this plan's execution.)*
