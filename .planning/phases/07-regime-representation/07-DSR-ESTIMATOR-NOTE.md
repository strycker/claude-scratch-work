# Deflated Sharpe Ratio — Estimator Note (Task 1, plan 07-06)

**Source read this session:** the PRIMARY source — Bailey, D.H. and López de Prado, M. (2014),
"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and
Non-Normality," *Journal of Portfolio Management* (SSRN 2460551) — was retrieved and read in
full. SSRN itself returned HTTP 403 to this environment, but the same PDF is hosted directly by
co-author David H. Bailey on his own site (`davidhbailey.com/dhbpapers/deflated-sharpe.pdf`,
22 pages, retrieved HTTP 200 this session). `pdftotext`/`poppler-utils` could not be installed
in this sandbox (package fetch 404'd), so the PDF's text streams were decompressed and extracted
with a small stdlib (`zlib`+`re`) script rather than a PDF library. This recovered all prose
text and the paper's one embedded Python code listing verbatim, but the paper's numbered
equations (Eq. 1–6, including Eq. 2, the DSR formula itself) are rendered as embedded
vector/image objects (typical of a MathType-in-Word export) and did **not** extract as literal
text — no equation glyphs were lost to a secondary source, they were never text in this PDF to
begin with. Where the note below states the DSR formula's exact algebraic form, it is stated
from the paper's own prose description of Eq. 2's five named inputs (page 9, extracted verbatim
below) cross-checked against the closed form's well-known, widely-reproduced expansion (e.g.
`mlfinlab`'s `deflated_sharpe_ratio`, and the RESEARCH.md sketch already in this repo, which this
reading confirms rather than merely trusts) — not copied from a secondary paraphrase of the
concept. This is a full-primary-text read with one specific extraction gap (rendered equation
glyphs), named here rather than silently patched over.

---

## 1. Expected maximum Sharpe ratio under N skill-less trials

Paper's own Appendix 1 (Eq. 5/6) and its Python Snippet 1 (`getExpMaxSR`), extracted verbatim
from the PDF:

```python
def getExpMaxSR(mu, sigma, numTrials):
    # Compute the expected maximum Sharpe ratio (Analytically)
    emc = 0.5772156649  # Euler-Mascheroni constant
    maxZ = (1 - emc) * ss.norm.ppf(1 - 1. / numTrials) + emc * ss.norm.ppf(1 - 1. / (numTrials * np.e))
    return mu + sigma * maxZ
```

Under the null (skill-less trials, `mu = 0`), this is exactly:

```
E[max{SR}] = sqrt(sharpe_variance) * [ (1-e) * Phi^-1(1 - 1/N) + e * Phi^-1(1 - 1/(N*e)) ]
```

Every term named:
- **`e`** (0.5772156649) is the **Euler–Mascheroni constant**. It appears here because the
  paper derives the expected maximum of N iid standard-normal draws via Extreme Value Theory
  (Appendix 1), and the EVT asymptotic expansion for the Gumbel-type maximum of Normal variates
  is itself parameterized by `e`. Assumption carried: N is "large enough" for the EVT
  approximation to be accurate — the paper's own Appendix 2 numerically validates this across a
  wide (mu, sigma, N) grid and reports the approximation error is small, but does not claim
  exactness at very small N (N=2 is the smallest case this project's monotonicity test exercises;
  the formula is still well-defined there, just further from the asymptotic regime it was
  validated against).
- **`Phi^-1(1 - 1/N)`** and **`Phi^-1(1 - 1/(N*e))`** are the two standard-normal quantiles
  (`scipy.stats.norm.ppf`) at `1 - 1/N` and `1 - 1/(N*e)`. They are the two order-statistic
  approximation points EVT uses to interpolate the expected maximum; their specific quantile
  levels are the paper's own derived result (Appendix 1, Eq. 5), not a free parameter.
- **`sqrt(sharpe_variance)`** scales the standardized maximum back into Sharpe-ratio units. The
  paper's own framing (page 7): trials are assumed to be "a set of N independent backtests or
  track records associated with **a particular strategy class**," each an SR estimate drawn
  from **one** Normal distribution with a shared mean and variance for that class. `sharpe_variance`
  IS that shared variance — the dispersion of Sharpe outcomes **within one coherent search**, not
  a generic constant. This assumption — that all N trials being deflated against are draws from
  one class — is the crux of §3's estimator decision below.
- At `N <= 1`, the formula is undefined in the paper's own terms (there is no "maximum of a set"
  with fewer than one comparison); this implementation's required behavior (`expected_max_sharpe`
  returns exactly `0.0` for `n_trials <= 1`) encodes "no selection occurred, so no correction
  applies" — consistent with the paper's own framing that N=1 is not a multiple-testing scenario
  at all.

## 2. The DSR expression itself

Page 9 (extracted verbatim): *"...DSR deflates SR by taking into consideration five additional
variables: The non-Normality of the returns [skewness, kurtosis], the length of the returns
series [T], the variance of the SRs tested [`sharpe_variance`], as well as the number of
independent trials involved in the selection of the investment strategy [N]."* And: *"DSR is a
PSR where the rejection threshold is adjusted to reflect the multiplicity of trials"* — i.e. DSR
is the Probabilistic Sharpe Ratio (Bailey & López de Prado 2012) evaluated at the
expected-maximum-under-the-null threshold from §1 above, rather than at an arbitrary
user-chosen threshold. The PSR closed form (Bailey & López de Prado 2012, restated by DSR's own
five-input description and matching the RESEARCH.md sketch already in this repo verbatim):

```
DSR = Phi[ (SR_hat - E[max{SR}]) * sqrt(T - 1) / sqrt(1 - skew*SR_hat + ((kurtosis-1)/4)*SR_hat^2) ]
```

where `SR_hat` is the observed Sharpe ratio of the selected trial, `T` is the number of return
observations backing that estimate (`n_obs` in this implementation), `skew`/`kurtosis` are the
third and fourth standardized moments of that trial's own return distribution (not the
registry's), `E[max{SR}]` is §1's expected-maximum-Sharpe under N skill-less trials, and `Phi` is
the standard-normal CDF (`scipy.stats.norm.cdf`). **The output lies in the unit interval strictly
because the final step is a CDF evaluation** — `Phi` maps any finite real argument to `(0, 1)`;
this is a structural property of the formula, not an empirical observation, and is exactly why a
bare `0 <= dsr <= 1` test proves nothing (every possible implementation, right or wrong, that
ends in a `norm.cdf` call satisfies it).

## 3. The estimator decision — `sharpe_variance`

**Two candidates considered**, per the plan's instruction:

**(a) Variance of this project's own observed population of historical trial Sharpe ratios**
(sample variance of `metrics["sharpe"]` across non-header registry rows). This is the estimator
the paper's own framing most directly supports — `sharpe_variance` IS defined (page 7) as the
dispersion of SR estimates "for a given strategy class," and this project already has a
git-tracked, tamper-evident ledger of exactly such estimates (design §22: "every grid cell
logged in the trial registry regardless of outcome — true DSR denominator"). What it assumes:
the logged trials are draws from one coherent search over one strategy class, so their spread
reflects genuine "how much did results vary as I searched" information. What breaks: **this
registry is not that.** It mixes wave-1 feature-policy comparison runs
(`P7-W1-frozen-10col` vs `P7-W1-impute-13col-REJECTED`), (K,λ) jump-model grid cells if/when
design §22's protocol runs them, and wave-2's joint-lift and #1-alone evaluations — heterogeneous
legs evaluating different questions with different metrics, not repeated draws from one
strategy family. Feeding their pooled variance into `sharpe_variance` would conflate "how
dispersed was my search" with "how different are the unrelated things I happened to log,"
biasing the correction in an uncontrolled, unverified direction rather than a knowably
conservative one.

**(b) An assumed-iid proxy independent of the registry's contents** — a fixed canonical
constant. What it assumes: the analyst can state, a priori, a plausible variance for Sharpe
estimates in this domain without consulting what was actually run. What breaks: it never
reflects what this specific search process explored — the reported DSR would be identical
whether the registry held 2 wildly heterogeneous trials or 200 tightly-clustered ones, which
defeats the honesty framework's stated purpose (design §8.4: the registry as "the multiple-testing
denominator").

**Decision: (a), the registry's own observed population, with (b) as the documented, named
fallback for the case (a) cannot yet be computed.** This is chosen over always using (b) because
(a) is the only one that can ever reflect this project's actual search history, and design §22
explicitly commits to the registry being the true denominator — abandoning that in favor of a
permanent constant would contradict the design document this note exists to serve. The
heterogeneous-strategy-class complication above is real and is **not** resolved by this choice;
it is accepted and named as this estimator's limitation (§5), the same "acknowledge, don't gate"
posture D-15 already uses for the dependence statistics in this same phase.

**Usable Sharpe observation count, verified live this session, counted separately as instructed:**
- **Archived ledger** (`registry/archive/trials-pre-P7W1-reset.jsonl`, 42 rows): **0** usable
  Sharpe observations. Every row's `metrics` dict contains exactly the keys
  `{"n_steps", "terminal_log_wealth"}` — no row anywhere in the archive carries a `sharpe` (or
  any `*sharpe*`-named) metric. Verified by loading all 42 rows and unioning their `metrics.keys()`.
- **Post-reset ledger** (`registry/trials.jsonl`, 1 row): **0** usable Sharpe observations. The
  sole row is the provenance header itself (`config.record_type == "provenance_header"`), whose
  `metrics` is `{"prior_genuine_trials": 38, "discarded_smoke_rows": 4}` — accounting metadata,
  not a performance measurement, and excluded from the trial population by the same
  `record_type` discriminator `total_trial_count()` (Task 2) uses.
- **Total usable Sharpe observations across the whole registry today: 0.** This is not a rare
  edge case this implementation might someday encounter — it is the current, only reachable
  state of the live ledger. The degenerate-case policy below is load-bearing now, not a
  theoretical safety net.

## 4. Degenerate-case policy

**When fewer than two usable Sharpe observations exist** (today: always, per §3): `sharpe_variance`
falls back to a fixed constant, **`1.0`**. This is conservative in the specific, narrow sense
that matters here: `expected_max_sharpe(n_trials, 0.0)` is REQUIRED to return exactly `0.0` for
every `n_trials` (a zero-variance null hypothesis has a degenerate, single-point distribution
whose maximum is trivially its own mean, `mu=0`) — so a `0.0` fallback would silently **disable**
the entire multiple-testing correction for every trial count, which is precisely
`07-RESEARCH.md`'s "closest analog to a security defect" pattern (T-07-05): systematically
UNDER-penalizing search. `1.0` is chosen instead of `0.0` specifically to keep the correction
mechanically active (nonzero, scaling with N as designed) while the registry has not yet
accumulated real, comparable trial data — an assumption, not a measurement, and logged as such
(WARNING, naming the observation count found) every time it fires so it is never silently
mistaken for a computed value. `1.0` is not derived from this project's own history (there is
none to derive it from yet); it is a round, transparently-a-placeholder value in Sharpe-ratio-squared
units, chosen to be neither vanishingly small (which reduces to the broken `0.0` case in effect)
nor implausibly large (which would manufacture an arbitrarily harsh, equally unjustified
correction). Once the registry accumulates >=2 usable `sharpe`-tagged rows from a coherent
search, `registry_sharpe_variance()` uses their real sample variance instead — no code change
required, only data.

**When the trial count is one or less** (`n_trials <= 1`): `expected_max_sharpe` returns exactly
`0.0` regardless of `sharpe_variance`, per §1's closing point. This is conservative because it
applies **zero** extra deflation when no genuine multiple-testing selection has occurred — with
one trial there is nothing to have selected among, so no correction beyond ordinary PSR is
warranted; inventing a nonzero threshold here would penalize a single evaluated configuration for
a multiplicity it never had.

## 5. The honest claim boundary

This implementation computes **"a deflated Sharpe ratio," not "the deflated Sharpe design §22
specifies."** Two independent reasons, both real and both stated rather than hidden: (1) Eq. 2's
literal rendered glyphs in the primary source could not be extracted as text in this session
(§ above) — the formula implemented here is the well-established PSR/DSR closed form confirmed
by the paper's own prose description of its five named inputs and independently corroborated
against this repo's own pre-existing RESEARCH.md sketch, not visually diffed character-by-character
against Eq. 2's image; and (2) the `sharpe_variance` input rests, as of this session, entirely on
the §4 placeholder assumption (`1.0`) because the registry's real usable-observation count is
zero (§3) — so any DSR value this code reports today is a deflated Sharpe computed under a named,
undischarged assumption, not the fully-registry-grounded number design §22 ultimately intends
once real trial history accumulates.

---

## For ADR-0002 (verbatim-liftable)

**Chosen `sharpe_variance` estimator: sample variance of this project's own trial registry's
observed Sharpe-ratio metrics (non-header rows), read live via `registry_sharpe_variance()`,
falling back to a fixed placeholder constant of `1.0` — never `0.0` — whenever fewer than two
such observations exist.** Justification: the registry is design §22's own declared "true DSR
denominator," so its own population is the only source that can ever reflect what this project's
search actually explored, and a fixed proxy divorced from the registry would report the same
number regardless of how much or how little was searched. Named limitation: verified live this
session, the registry currently holds **zero** usable Sharpe-bearing rows (0 of 42 archived, 0
of 1 post-reset), so the estimator runs entirely on its `1.0` placeholder fallback today — a
deliberate, logged, non-silent assumption, not a measured quantity — and the registry additionally
mixes heterogeneous strategy legs rather than repeated draws from one strategy class, so even
once real Sharpe-bearing rows accumulate, the resulting sample variance will be a cruder,
pooled-population estimate than the paper's own "one strategy class" framing strictly assumes.
This is why the result is reported as "a deflated Sharpe ratio," not "the deflated Sharpe design
§22 specifies," until both gaps close.

---

## AMENDMENT 2026-09-21 — the estimator had a live trap; recording convention set

**Decided by Glenn, 2026-09-21**, after plan 07-11 declined to write a `sharpe` key and
flagged why.

### The trap

`_MIN_USABLE_SHARPE_OBSERVATIONS` was **2**. Any two `sharpe`-bearing rows switched
`registry_sharpe_variance` off its conservative 1.0 placeholder and onto a computed value.
Measured on plan 07-11's two rows (Sharpe **0.917073** and **0.914903**):

| | |
|---|---|
| sample variance (ddof=1) | 2.354450e-06 |
| `expected_max_sharpe(42, 1.0)` | **2.208694** |
| `expected_max_sharpe(42, 2.35e-06)` | **0.003389** |

A **99.85% collapse** of the multiple-testing hurdle, after which essentially any strategy
clears DSR. The trap was armed regardless of what 07-11 did: it would have fired for whoever
added the key next.

### The deeper error, which raising the minimum alone would not have fixed

In Bailey–López de Prado, `sharpe_variance` is the dispersion of Sharpe ratios **across
independently-tried configurations** — it estimates how good the best of N trials looks by luck.
Plan 07-11's two rows are **two arms of one ablation**: the baseline is `blend_weight_1 = 1.0`
of the *same* harness, deliberately near-identical to the joint leg. Including them answers "how
different are the two arms of one comparison" — approximately zero by construction — rather than
"how much do different strategies vary". That is a category error, not a tuning problem.

### Two independent defences, both implemented

1. **`_MIN_USABLE_SHARPE_OBSERVATIONS` 2 → 20.** A variance estimate from n=2 is unusable
   whatever the rows are.
2. **`config["independent_trial"] is False` excludes a row entirely.** Arms of one ablation never
   enter the across-trials variance.

Each is pinned by its own test so removing either fails, plus a test asserting 20 genuine trials
*do* produce a computed variance — otherwise the guard could only ever confirm "placeholder",
which is this project's signature defect shape.

### Recording convention, going forward

Trial rows **do** carry `metrics["sharpe"]`, so the registry accumulates what the estimator will
eventually need. A row that is an arm of an ablation, a sensitivity sweep, or any other
non-independent comparison **must** set `config["independent_trial"] = False`. The default is
independent; the flag is an explicit opt-out for rows that are not.

Plan 07-11's two rows were backfilled accordingly. **Today's verdicts are unchanged**:
`sharpe_variance` is still the 1.0 placeholder, `expected_max_sharpe(42, 1.0)` is still 2.208694,
and both legs' DSR (2.28151e-12 baseline, 1.46904e-11 joint) still do not clear the hurdle.

### What remains an assumption

`DEGENERATE_SHARPE_VARIANCE = 1.0` is still a declared placeholder, not a measured quantity, and
governs every DSR this project reports. It stays that way until 20 independent Sharpe-bearing
trials exist. That is a longer road than before this amendment, deliberately — the previous road
was short because it was wrong.
