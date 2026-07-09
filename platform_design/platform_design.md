# Regime-Conditional Investment Platform — Design Document

**Status:** Design phase (math & theory). No implementation commitments beyond what is stated here.
**Owner:** Glenn Strycker. **Origin:** Claude chat discussion, July 2026.
**Purpose of this file:** portable design record for (a) re-evaluating `trading-crab` / `trading-crab-lib` and (b) building the next iteration. Focus is mathematics and architecture, not code.

---

## 1. Objective & Framing

- **Objective function:** maximize long-run wealth (log-wealth / Kelly-flavored). Risk is managed structurally (position sizing, exposure caps, volatility-scaled stops), not by optimizing Sharpe directly.
- **Benchmark to beat:** buy-and-hold SPY, **net of avoided drawdowns** — the platform's reason to exist is sidestepping 2000/2008-class regime damage (bubble deflation, geopolitical shocks), not maximizing tracking-error-adjusted alpha.
- **User model:** investor, not trader. Decisions weekly; positions held weeks-to-years. Model refits monthly; scoring weekly. Nothing automated — platform produces *guidance* (regime probabilities, forecasts, target mix, entry/exit levels); human executes.
- **Edge thesis:** no institutional constraints (no redemptions, career risk, or capacity limits) → can hold long-horizon positions (valuation tilts, regime tilts) that funds structurally cannot. The un-arbitraged residue lives at monthly+ horizons: momentum, valuation, carry, vol-managed exposure (McLean–Pontiff: published anomalies decay ~30–60% but persist).
- **Efficient-markets stance:** short-horizon structure is arbitraged away; long-horizon premia persist for risk-based and limits-to-arbitrage reasons. We do not need to out-model firms — we need reliable exposure to persistent, slow structure, conditioned on regime.
- **Scope:** generalized over assets. Any asset expressible as a daily USD price series qualifies (ETFs, single names, spliced index/futures histories). Features may include non-tradable series (spot oil, macro rates).
- **Taxes/frictions:** intentionally out of scope for the modeling platform (handled manually by the operator). A small turnover penalty is retained in allocation **for statistical stability only** (it suppresses noise-chasing weight churn), not as a cost model.

## 2. Locked Design Decisions

| # | Decision | Choice |
|---|----------|--------|
| D1 | Spine | Regime-switching ("mixture of experts"): regime layer gates per-regime prediction models |
| D2 | Regime count | 5–6 semantic states; occupancy floor ≥ ~8%, cap ≤ ~35% (emergent balance, **not** forced-balance clustering) |
| D3 | State skeleton | Hand-specified semantics, estimated distributions: calm expansion · volatile/late-cycle expansion · inflationary expansion · disinflationary slowdown · crisis/deleveraging · recovery |
| D4 | Labeling (stage 1) | Full-information (two-sided) labeling is **by design**. Preferred labeler: statistical jump model; benchmark: Student-t HMM with Viterbi decode |
| D5 | Prediction (stage 2) | Causal features only (one-sided filters). Separate **nowcast** model (what regime now) and **transition** model (P of moving to regime j) |
| D6 | Transition memory | Regime-age as a feature (cheap semi-Markov); TVTP softmax if joint model used |
| D7 | Emissions/dists | Fat-tailed (Student-t) everywhere; Gaussian only as a diagnostic baseline |
| D8 | Horizons | Decision horizons 1m and 3m; 12m as strategic tilt input; nothing < 1m |
| D9 | Forecast form | Collections of per-asset, per-horizon point models (distributional upgrades optional later); volatility/covariance forecasts are first-class outputs |
| D10 | Frequency | Monthly modeling spine; weekly scoring/monitoring overlay; monthly (or slower) refits |
| D11 | Data start | ~1962, lean feature set for labeling; richer modern features (HY OAS 1997+, breakevens 2003+) allowed in stage-2 models with NULL-handling / derived-feature fallbacks |
| D12 | Vintage discipline | Market-observed features preferred (never revised); agency data via ALFRED vintages + publication-lag alignment; slow valuation anchors (CAPE, Buffett Indicator) live in the strategic layer |
| D13 | Evaluation discipline | Walk-forward only; purged/embargoed CV; trial registry; HOLDOUT: 2021-01-01 onward, locked, evaluated once at design freeze; deflated Sharpe for headline results |
| D14 | Allocation | Covariance-first: regime-conditional covariances + brutally shrunk return forecasts; vol targeting; fractional Kelly; turnover penalty (stability); vol-scaled, regime-conditional stops |

## 3. Architecture (five layers)

```
┌──────────────────────────────────────────────────────────────────┐
│ L0  DATA        daily USD prices (spliced long histories),       │
│                 market-observed features, ALFRED-vintage macro,  │
│                 slow valuation anchors                           │
├──────────────────────────────────────────────────────────────────┤
│ L1  LABELING    two-sided information → ground-truth regimes     │
│  (offline)      jump model (preferred) / t-HMM+Viterbi (bench)   │
│                 outputs: hard labels s_t, soft confidences γ_t(k)│
├──────────────────────────────────────────────────────────────────┤
│ L2  REGIME      causal features only                             │
│  PREDICTION     • Nowcaster:  P(S_t = k | z_{1:t})               │
│  (online)       • Transition: P(S_{t+1}=j | S_t=i, z_t, age)     │
├──────────────────────────────────────────────────────────────────┤
│ L3  ASSET       regime-conditional mixture of experts:           │
│  PREDICTION     per-asset, per-horizon return models;            │
│                 fair-value gap module; GARCH/EWMA vol layer;     │
│                 regime-conditional covariance                    │
├──────────────────────────────────────────────────────────────────┤
│ L4  ALLOCATION  target mix, buy/sell/hold per holding,           │
│  & TACTICS      prioritized buy list, vol-scaled stops,          │
│                 entry/exit levels; weekly report                 │
└──────────────────────────────────────────────────────────────────┘
```

Information flow rule: **L1 may see the future; L2–L4 may not.** The last 6–12 months of L1 labels are embargoed from L2 training (labels there are unstable by construction — the smoother lacks its backward window). Track **label churn** (fraction of trailing labels revised on refresh) as a monitoring metric.

---

## 4. Layer 1 — Regime Labeling (ground truth)

### 4.1 Preferred: statistical jump model (Bemporad–Boyd 2018; Nystrup et al.)

Minimize over state sequence s ∈ {1..K}^T and centroids μ:

    J(s, μ) = Σ_t ‖x_t − μ_{s_t}‖²  +  λ · Σ_t 1[s_t ≠ s_{t−1}]

- k-means **plus a per-jump penalty** — the direct conceptual upgrade of trading-crab's clustering (adds temporal persistence; removes forced balance).
- Solved by alternation: given μ, optimal s is exact **dynamic programming** (Viterbi-like, O(TK²)); given s, μ are cluster means. Multi-restart; warm-start from existing k-means clusters.
- λ is the single interpretable persistence knob. Tune λ (and K) until **acceptance criteria** (§4.4) pass — occupancy and sojourn targets become the tuning objective, not a distortion of the geometry.
- Robustness: use robust distances or standardized features with winsorization; a continuous-state extension yields soft labels for confidence weighting.
- Empirically more stable out-of-sample than HMM labels (less refresh churn at the sample edge) — directly mitigates the embargo problem.
- Kinship: ℓ1 trend filtering (Kim–Koh–Boyd) is the continuous cousin — same penalized-DP flavor — usable for the fair-value module (§6.3), so L1-for-regimes and L1-for-prices can share machinery.

### 4.2 Benchmark: HMM with Student-t emissions

Model: hidden S_t ∈ {1..K}, transition matrix A, emissions b_k(x_t).

- **Forward recursion** (filtered, real-time): α_t(j) = [Σ_i α_{t−1}(i) A_ij] · b_j(x_t); normalize → P(S_t | x_{1:t}).
- **Backward + smoothing**: γ_t(i) ∝ α_t(i) β_t(i) = P(S_t | x_{1:T}) — the ground-truth generator. Filtered vs smoothed is one model read without/with the backward pass.
- **Decoding for labels:** Viterbi (most probable *path*) — respects transition structure, avoids per-point argmax producing near-impossible transitions; keep γ_t(k) alongside as label confidence.
- **Estimation:** Baum–Welch (EM). E-step: forward-backward → occupancies γ, expected transitions ξ. M-step: weighted moments (emissions), normalized ξ (transitions). Hazards: local optima (multi-restart; k-means warm start; A initialized near-diagonal) and degenerate states (covariance regularization / priors; hmmlearn will not warn you).
- **Student-t emissions mandatory** for the benchmark: with Gaussians, extra states get spent absorbing fat tails ('87 + '08 + '20 is a t-distribution, not a regime).
- **TVTP extension** (if the joint model is ever promoted): P(S_t=j | S_{t−1}=i, z_t) ∝ exp(a_ij + β_j′ z_t). EM still applies; the transition M-step is a **weighted multinomial logistic regression** with ξ as fractional targets — i.e., the TVTP-HMM *is* the two-stage design estimated jointly. (Filardo 1994; Diebold–Lee–Weinbach.)
- Declined alternative (read, don't deploy): sticky HDP-HMM (Fox et al.) — learns K nonparametrically; hallucinates states at our effective sample size.

### 4.3 The geometric-duration problem

First-order Markov ⇒ sojourns geometric (memoryless, mode at 1). Real regimes age. Fix hierarchy: (1) ignore (often fine monthly); (2) HSMM with explicit durations (neg-binomial); (3) **chosen:** regime-age as a feature in the transition model — nearly free, no extra machinery.

### 4.4 Acceptance criteria for a regime solution (all must pass)

1. **Occupancy:** every state ≥ ~8% and ≤ ~35% of months.
2. **Sojourns:** median sojourn measured in months-to-years; no state with median < 3 months at monthly granularity.
3. **Stability:** re-estimate on subsamples (drop first decade / last decade / block bootstrap); states persist with matched emission parameters (match via Hungarian algorithm on distribution distances to defeat label switching). A "regime" that evaporates when 2008–09 is dropped is an *episode*, not a regime.
4. **Interpretability:** one-line economic description per state, consistent with the D3 skeleton.
5. **Decision-relevance:** states must differ in the **conditional distribution of asset returns** (means, vols, correlations), not merely in feature space. Validate K against out-of-sample allocation value, not likelihood. (BIC/AIC over-select K on financial data; ICL is the better information criterion if one is used.)
6. **Effective-sample honesty:** post-war US ≈ 15–30 independent regime transitions. Any structure requiring more parameters than that supports must be constrained (semantic skeleton, shrinkage), not estimated freely.

---

## 5. Layer 2 — Regime Prediction (causal)

Two distinct supervised problems; do not merge them.

### 5.1 Nowcaster — P(S_t = k | causal features through t)

- A discriminative replacement for the HMM filter. Any well-calibrated classifier (multinomial logistic → gradient-boosted trees).
- **Include prior predicted state distribution as a feature** (recursive structure). Without it, persistence is discarded and predictions flicker.
- Sample weights = label confidences γ_t(k) from L1 (downweight periods the smoother itself found ambiguous — don't force the classifier to learn boundaries the ground truth never had).
- **Headline metric: accuracy/log-loss in transition windows** (±3 months around ex-post transitions), reported separately from overall accuracy. Persistence lets a trivial classifier score ~90% overall; the value is at the turns.
- Calibration required (reliability curves / isotonic recalibration): downstream layers consume the *probabilities*, not the argmax.

### 5.2 Transition model — P(S_{t+1} = j | S_t = i, z_t, age_t)

- Multinomial model per current state (or single model with state as feature). Features: credit spreads, curve slope, vol level & change, momentum breadth, **regime age** (D6).
- Feature relevance differs from the nowcaster: spreads/vol *lead* transitions; levels *identify* states. Keep feature sets and evaluations separate.
- Output consumed by L3/L4 as a predictive regime distribution over the decision horizon: π_{t+h} = π_t · Π A_t (feature-conditional), or direct h-step classification for h ∈ {1m, 3m}.

### 5.3 Anti-flicker

Filtered/nowcast probabilities oscillate near transitions. Allocation responds through **hysteresis bands** (act when P crosses ~0.7; unwind below ~0.4; hold in between) and/or smoothed allocation response with bounded turnover.

### 5.4 Honesty metrics for L2 (first-class outputs, not afterthoughts)

- **Smoothed-vs-filtered gap:** performance of any regime-driven decision computed with smoothed labels minus the same with real-time (nowcast/filtered) probabilities. This gap *is the measured hindsight content* of the strategy.
- **Detection lag:** periods from each ex-post transition until real-time probability crosses the action threshold (typically 1–3 months). The ratio **median sojourn / detection lag** largely determines whether regime timing can work; report it prominently. If sojourns ≈ 18m and lag ≈ 2m, most of the regime is captured; if sojourns ≈ 5m, the lag eats the trade.
- **Label churn:** fraction of trailing-24-month labels that change on each L1 refresh.

---

## 6. Layer 3 — Asset Prediction (regime-conditional mixture of experts)

### 6.1 Structure

Final forecast for asset a, horizon h:

    ŷ_{a,h}(t) = Σ_k P(regime k over horizon | t) · f_{a,h,k}(z_t)

- **Gating network:** L2 outputs. **Experts:** per-regime prediction models f_k.
- The econometric twin is the Markov-switching VAR (Krolzig): regime-dependent VAR coefficients and covariances over the joint feature/return vector — the formal version of "one big differential equation whose coefficients depend on market conditions." The Markov objection ("history doesn't matter") is resolved by **state augmentation**: any finite-memory process is Markov once the state includes multi-scale smoothed levels and derivatives; the design question was never Markov-or-not but *what belongs in the state*.
- **Sparse-expert problem:** the crisis expert trains on ~8% of history. Mitigations, in order of preference:
  1. **Soft gating:** every expert trains on all data, sample-weighted by γ_t(k).
  2. **Partial pooling:** shrink regime-k coefficients toward the pooled all-history model (hierarchical structure; exactly multilevel-model logic for sparse segments).
  3. **Implicit MoE baseline:** one gradient-boosted model fed features + regime probabilities + regime age; trees learn the interactions natively. Run as the *ceiling estimate of extractable signal*; keep the structured MoE as the interpretable model allocations actually trust.
- Continuous-drift complement (not spine): TVP-VAR via Kalman filter (Primiceri 2005) for one or two key relationships whose changes are drift rather than switch — equity–bond correlation especially. Smooth-transition (STAR/STVAR) models interpolate regimes with a logistic gate; noted, not adopted.
- Deterministic-ODE framing is retired: noise dominates drift at every horizon of interest; the honest formalism is discrete-time regime-switching stochastic dynamics with fat-tailed innovations, and the coupling coefficients themselves drift.

### 6.2 Volatility & covariance layer (the reliable half of prediction)

- Returns are barely predictable; **volatility is highly predictable** (clustering). GARCH(1,1) or EWMA of squared returns per asset; realized-vol features feed L2 as well.
- **Covariance-first doctrine:** most allocation value comes from the covariance forecast, not the mean vector. Regime-conditional covariance matrices (crisis: correlations → 1; inflationary: equity–bond correlation flips sign) are more estimable and more decision-relevant than regime-conditional means.
- Estimation: Ledoit–Wolf shrinkage within regime; DCC-GARCH or regime-mixture covariance for dynamics.
- **Vol targeting** as a default overlay: position size ∝ 1/σ̂ mechanically de-risks into turbulence; empirically improves Sharpe and cuts drawdowns (Moreira–Muniz 2017). Directly serves the beat-SPY-by-avoiding-crashes objective.

### 6.3 Fair-value gap module (Glenn's smoothed-target idea)

- **Construction:** L1-style two-sided smooth of each asset's (log) price ("where price *should* be" — leaky target by design); supervised model predicts the smoothed value from **causal features only**; signal = gap g_t = p_t − p̂_t^{fair}.
- A learned generalization of "distance from the 200-day MA"; ML form of a fair-value regression. ℓ1 trend filtering is a natural smoother choice (shared machinery with the jump model).
- **Caveats (binding):**
  - *Endpoint problem:* the two-sided smooth is unstable at the sample edge — same trailing embargo as regime labels; the target for recent months keeps revising.
  - *KPI is convergence, not fit:* score by **forward realized returns conditional on the gap** (does price converge to fair value, or fair value to price?), never by accuracy on the smoothed series.
  - *Regime-conditional validity:* gap reversion plausibly works in calm regimes and fails in trend/crisis regimes (falling knives). The gap signal therefore enters as an **expert input inside the MoE**, gated by regime — not as a standalone rule.

### 6.4 Meta-labeling (sizing layer)

López de Prado pattern: primary signal proposes direction/tilt; a secondary classifier predicts *whether that specific signal instance will succeed*, using features irrelevant to direction (vol state, regime confidence, signal age/crowding). Bet size ∝ secondary-model confidence. Slots between L3 and L4; optional in v1, planned seam in the architecture.

### 6.5 Prediction targets & labels

- Per-asset, per-horizon models (h ∈ {1m, 3m, 12m}); point forecasts in v1 (D9), with vol layer supplying the distributional information allocation actually needs.
- Prefer forecasting **excess returns over cash** and **relative returns vs SPY** (the stated decision problem is *relative* attractiveness).
- **Overlapping-label discipline:** h-month forward-return labels overlap h−1 months (12m labels ~92% autocorrelated). Purged CV with embargo (López de Prado ch. 7–8); plain TimeSeriesSplit is insufficient.

---

## 7. Layer 4 — Allocation & Tactics

- **Doctrine:** forecast vol/covariance carefully; **shrink return forecasts brutally** toward zero or equilibrium; never feed raw point forecasts to an optimizer (optimizers are error maximizers — Michaud).
- **Weights:** Black–Litterman blending regime/MoE views with equilibrium (market-cap or risk-parity prior), or hierarchical risk parity as the robust non-optimizing alternative. Turnover penalty (L1 on Δw) for stability (see §1 — not a cost model).
- **Sizing:** fractional Kelly (¼–½ Kelly) on the log-wealth objective — full Kelly on estimated edges is a leverage accident; fractional Kelly dominates under parameter uncertainty.
- **Stops (regime- and vol-aware):** fixed-% stops fail — a 10% stop is a weekly event in crisis vol and a 3σ event in calm, and in mean-reverting regimes stops systematically sell local bottoms. Spec: stop distance = m_k · σ̂_{a,t} (multiple of forecast vol) with regime-conditional multiplier m_k — wide/no stops in reversion-friendly regimes, tight in trend/crisis regimes. Stop placement is a **model output**.
- **Crash module (explicit, given the objective):** P(crisis regime within h) from L2, plus regime-conditional left-tail estimates (CVaR under the mixture). This is the direct instrument for "how likely is SPY to crash."
- **Weekly report outputs:** current regime distribution + trajectory; per-holding buy/sell/hold with rationale; prioritized buy list; target mix vs current mix (trades implied); stop levels per holding; crash-probability dashboard.

---

## 8. Evaluation & Honesty Framework

Statistical self-defense for a solo researcher (no skeptical risk committee; the person most capable of being fooled by a beautiful backtest is the person who built it). None of this is regulatory; all of it is standard ML hygiene applied to strategy research.

1. **Walk-forward everything.** At each rebalance t: refit on data ≤ t (L1 labels re-smoothed, L2/L3 re-trained), record decisions, step forward. O(TK²) HMM/jump fits are minutes over 60 years — no computational excuse. The gap between the lazy (full-sample) backtest and the honest walk-forward **is the hindsight measurement**; report it.
2. **Point-in-time data.** ALFRED vintages for agency series; publication-lag alignment (a value enters the feature matrix only after its publication date); market-observed features preferred (never revised). One-sided filters only in causal features — no centered/zero-phase (filtfilt-style) smoothing.
3. **Purged & embargoed CV** for all supervised components with overlapping labels.
4. **Trial registry.** Log every configuration evaluated (features, K, λ, model class, hyperparameters, resulting metrics). This is the multiple-testing denominator: the best of N noise strategies has a computable inflated Sharpe. Treat it as a pre-registration ledger.
5. **Holdout.** All data from **2021-01-01 onward** untouched during development; evaluated **once** at design freeze (= the declared end of iteration; evaluating on it more than once contaminates it, exactly like tuning on a test set).
6. **Deflated Sharpe ratio** (Bailey–López de Prado) for headline performance — corrects for number of trials, non-normality, track length. Calibration for skepticism: Harvey–Liu–Zhu argue t ≈ 3, not 2, for a newly discovered signal.
7. **Brutal baselines.** Buy-and-hold SPY; 60/40; static risk parity; historical-mean return forecast (Goyal–Welch: it beats most published predictors OOS); no-regime versions of every regime-conditional model (the regime layer must pay rent).
8. **Forecast KPIs.** Diebold–Mariano vs naive baseline per asset/horizon; Mincer–Zarnowitz calibration regressions; hit rate and log-loss in transition windows; fair-value module scored by conditional forward returns (§6.3). If/when distributional forecasts arrive: log score / CRPS.
9. **Strategy KPIs (vs SPY benchmark):** terminal log wealth; max drawdown & drawdown duration; CVaR(5%); capture ratios in ex-post crisis windows (2000–02, 2008–09, 2020, 2022 analogues in-sample); turnover.

---

## 9. Data Specification

- **Span:** ~1962 onward (post-1962 gives continuous 10y constant-maturity yields and modern market structure; ~30+ regime episodes candidate material, ~15–30 usable transitions).
- **Asset price histories:** daily USD. ETF histories are short (SPY 1993, GLD 2004, USO 2006, TLT 2002…) → **splice** with index/futures/spot histories (S&P total return, gold spot/futures, WTI spot, constant-maturity Treasury total-return synthetics). Splicing rules documented per asset; tradability applies to the present, not the history.
- **Feature taxonomy:**
  - *Fast layer (regime detection; market-observed, unrevised):* curve slope (10y–3m, 10y–2y), credit spreads (BAA–AAA full-history; HY OAS 1997+ as modern supplement), realized vol (multi-scale), trailing returns/momentum (multi-scale), oil, gold, USD index, breakevens (2003+).
  - *Slow layer (strategic tilt):* CAPE, Buffett Indicator (market cap/GDP), dividend yield vs BAA, real-rate level. Revision noise small relative to decade-scale signal swing.
  - *Agency layer (vintage-corrected only):* unemployment, CPI, GDP growth — via ALFRED, publication-lag aligned; lagging indicators, used sparingly in the fast layer.
- **Missing-feature policy (D11):** modern-only features enter stage-2 models with NULL-tolerant handling (trees) or derived fallbacks (e.g., composite yield blending available maturities across missing windows). Labeling (L1) uses the lean full-history set only.
- **Transforms:** log for level series; one-sided EMAs at multiple half-lives; backward derivatives d1/d2 at multiple scales; standardization fit on training windows only. **No non-stationary raw levels in clustering/labeling features** (rates, spreads, growth rates, and derivatives — yes; log-price levels — no).

---

## 10. Open Questions (deferred, tracked)

1. Distributional forecasts (log score/CRPS) as v2 upgrade to D9 point models — revisit after v1 KPIs.
2. Individual-name cross-sectional module (factor model layer) — later bolt-on; architecture must keep the asset-universe abstraction clean so this slots in.
3. Meta-labeling activation criteria (v1 seam only).
4. TVP-VAR complement for equity–bond correlation drift — pilot after regime spine validates.
5. Jump-model λ and K joint selection protocol details (grid + acceptance criteria pass/fail).
6. Weekly overlay mechanics: which signals are allowed to update weekly (vol, nowcast probabilities, stops) vs monthly (regime refits, target mix) — draft rule: *fast risk down, slow risk up*.
7. Splicing sources and validation per asset history.

---

## 11. trading-crab Re-evaluation Checklist

Findings from repo review (July 2026: `trading-crab`, `trading-crab-lib`) mapped to this design.

| # | Current state | Issue vs design | Action |
|---|---------------|-----------------|--------|
| R1 | `frequency: "Q"` (quarterly spine, ~260 obs from 1950) | D10: monthly spine; quarterly starves the labeler and quadruples detection lag in calendar time | Move ingestion/transforms to monthly; keep quarterly agency series with proper alignment |
| R2 | `balanced_k: 5` (balanced k-means via bipartite matching) | D2/D4: forced balance is statistically artificial; no temporal persistence | Replace with jump model (k-means + jump penalty); keep k-means as warm start; occupancy floor/cap as acceptance criteria instead of hard balance |
| R3 | Empirical constant transition matrix (`build_transition_matrix`) | D5/D6: no feature dependence, no regime age | Supersede with L2 transition model (multinomial w/ features + age); keep empirical matrix as diagnostic |
| R4 | PCA(5) before clustering | Marginal: PCA on mixed-unit features obscures interpretability and the semantic skeleton | Prefer standardized curated features, no PCA (or PCA only as diagnostic); skeleton constraints need named dimensions |
| R5 | `log_sp500`, `log_cpi` **levels** in `initial_features`; d1s in clustering set (levels correctly excluded there — verify) | §9: non-stationary levels must not reach the labeler | Audit final clustering matrix; enforce stationarity check in pipeline |
| R6 | FRED revised series; `shift: true` only for GDP/GNP | D12: revised data leaks; shift handles lag, not revision | ALFRED vintages for agency series; reclassify features into fast/slow/agency taxonomy; prefer market-observed for labeling |
| R7 | `TimeSeriesSplit(5)` CV; forward horizons [1,2,4,8] quarters | §6.5: overlapping labels leak through split boundaries | Add purging + embargo to CV; report transition-window metrics separately |
| R8 | RF/DT/GBM classifiers predicting cluster labels | Sound, but single-model conflates nowcast and transition problems | Split into nowcaster (recursive state feature, γ sample weights) and transition model (spreads/vol/age features) |
| R9 | `tactics.py`: fixed vol/trend thresholds → buy_hold/swing/stand_aside | §6.2/§7: no vol forecasting; thresholds not regime-conditional | Replace with GARCH/EWMA vol layer + vol targeting; regime-conditional stop multipliers |
| R10 | No walk-forward harness; full-sample cluster fit → labels → CV | §8.1: parameter lookahead (labels filtered with future-informed parameters) | Build walk-forward runner as core infrastructure; report smoothed-vs-filtered gap and detection lag |
| R11 | Gaussian k-means/GMM (`gmm.py`) | D7: fat tails masquerade as regimes | Student-t mixtures / robust distances; ICL over BIC if criterion used |
| R12 | No trial logging, no holdout | D13 | Add trial registry (flat file/SQLite: config hash → metrics); carve 2021+ holdout before next iteration begins |
| R13 | 16-ETF universe, histories start 1993–2006 | §9: labeling needs 1962+ | Build spliced long histories per asset class; ETFs remain the tradable instruments |
| R14 | Regime naming heuristics (median-deviation tags) | Compatible — good seed for D3 skeleton | Recast as skeleton *constraints* (sign structure on state means) rather than post-hoc names |
| R15 | Checkpointing, config-driven design, functions-only lib, 24 tests | Strengths — keep | Architecture supports the L0–L4 layering cleanly; extend rather than rewrite |

**Suggested re-build order:** R6/R13 (data layer) → R1 (monthly) → R2/R11/R14 (labeler + acceptance criteria) → R10/R12 (walk-forward + registry — *before* any modeling iteration, so every result is honest from day one) → R8 (L2 split) → R9 (vol layer) → L3 MoE → L4.

---

## 12. Reading List

**Books**
- Kim & Nelson, *State-Space Models with Regime Switching* — the standard reference for MS models + Kalman filtering; matches Glenn's profile.
- López de Prado, *Advances in Financial Machine Learning* — ch. 7–8 (purged CV/embargo), meta-labeling, backtest overfitting, deflated Sharpe.
- Grinold & Kahn, *Active Portfolio Management* — fundamental law (IR ≈ IC·√breadth), portfolio construction doctrine.
- Ang, *Asset Management: A Systematic Approach to Factor Investing* — regimes, factor premia, long-horizon allocation.

**Regime-switching & labeling**
- Hamilton (1989), "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle" — the founding MS paper.
- Filardo (1994), "Business-Cycle Phases and Their Transitional Dynamics" — TVTP.
- Diebold, Lee & Weinbach (1994) — TVTP estimation via EM.
- Krolzig (1997), *Markov-Switching Vector Autoregressions* — the MS-VAR.
- Ang & Bekaert (2002+), regime switches and international asset allocation.
- Bemporad et al. (2018), "Fitting Jump Models" — the jump-model formulation.
- Nystrup, Lindström & Madsen (2020+), jump models for market regimes; OOS label stability.
- Kim, Koh & Boyd (2009), "ℓ1 Trend Filtering."
- Fox, Sudderth, Jordan & Willsky (2011), sticky HDP-HMM — read to decline.
- Guidolin & Timmermann (2007+), regime-dependent asset allocation.

**Predictability, evaluation & overfitting**
- Goyal & Welch (2008), "A Comprehensive Look at the Empirical Performance of Equity Premium Prediction."
- Campbell & Thompson (2008) — what tiny OOS R² is worth.
- McLean & Pontiff (2016), "Does Academic Research Destroy Stock Return Predictability?"
- Harvey, Liu & Zhu (2016), "…and the Cross-Section of Expected Returns" — the t≈3 bar.
- Bailey & López de Prado (2014), "The Deflated Sharpe Ratio."
- Diebold & Mariano (1995), forecast-accuracy comparison; Mincer & Zarnowitz (1969), forecast calibration.

**Volatility, covariance & allocation**
- Moreira & Muniz [Muir] (2017), "Volatility-Managed Portfolios."
- Ledoit & Wolf (2004), covariance shrinkage; Engle (2002), DCC-GARCH.
- Black & Litterman (1992); Michaud (1989), "The Markowitz Optimization Enigma"; López de Prado (2016), hierarchical risk parity.
- Kelly criterion under estimation error: MacLean, Thorp & Ziemba (eds.), *The Kelly Capital Growth Investment Criterion* (fractional Kelly chapters).
- Asness et al., "Sin a Little" (valuation timing in moderation).
- Primiceri (2005), TVP-VAR.

**Context**
- Bridgewater "All Weather" whitepapers — hand-specified growth×inflation quadrants as a deliberate alternative to estimated regimes.

---

## 13. trading-crab vs. This Design — Feature Comparison (v1.1 addendum)

| Dimension | trading-crab (current) | This design | Verdict for superset |
|---|---|---|---|
| Data frequency | Quarterly spine | Monthly spine + weekly scoring | **New** (R1) |
| Data span | 1950 config, but ETF prices 1993–2006+ | 1962+, spliced index/futures/spot histories | **New** (R13) |
| Data vintage | Revised FRED, pub-lag `shift` for GDP/GNP only | ALFRED vintages + fast/slow/agency taxonomy; market-observed preferred | **New** (R6) |
| Feature engineering | Logs, d1–d2 derivatives, cross-ratios, config-driven | Same, plus multi-half-life one-sided EMAs, stationarity enforcement, NULL-tolerant modern features | **Merge** — keep TC transforms, add constraints |
| Regime labeling | Balanced k-means (bipartite), GMM, spectral options; PCA(5) | Jump model (k-means + jump penalty, DP solve); t-HMM/Viterbi benchmark; no PCA | **New core; TC k-means as warm start** (R2, R4, R11) |
| Balance handling | Forced equal occupancy | Occupancy floor/cap as acceptance criteria (emergent) | **New** |
| Temporal persistence | None (i.i.d. clustering) | Jump penalty λ / HMM transitions | **New** — the single biggest modeling upgrade |
| Regime semantics | Post-hoc naming heuristics (median-deviation tags) | Hand-specified semantic skeleton as *constraints* | **Merge** — TC heuristics seed the skeleton (R14) |
| Transition modeling | Empirical constant matrix | Feature-conditional (TVTP-style) + regime age | **New** (R3); keep empirical as diagnostic |
| Regime prediction | RF/DT/GBM on labels, forward horizons [1,2,4,8]Q | Split nowcaster (recursive, γ-weighted) + transition model; calibration; transition-window metrics | **Merge** — TC model zoo survives, problem framing splits (R8) |
| Cross-validation | TimeSeriesSplit(5) | Purged + embargoed CV | **New** (R7) |
| Asset prediction | Historical returns-by-regime tables | Regime-conditional MoE (soft gating, partial pooling), boosted ceiling model, fair-value gap module | **New**; TC tables = the naive baseline the MoE must beat |
| Volatility | `vol_window` threshold in tactics | GARCH/EWMA forecasts, vol targeting, regime-conditional covariance, Ledoit–Wolf | **New** (R9) |
| Allocation | Basic portfolio construction + tactics buy_hold/swing/stand_aside | BL/HRP, fractional Kelly, model-driven vol-scaled stops, crash module, hysteresis | **New** (R9) |
| Evaluation | CV accuracy, classification reports | Walk-forward harness, smoothed-vs-filtered gap, detection lag, DSR, trial registry, 2021+ holdout, baseline gauntlet | **New** (R10, R12) |
| Infrastructure | Functions-only lib, caller-driven config, checkpoints, CLI, 24 tests, weekly email report | (design only — none) | **trading-crab wins outright** (R15) — keep and extend |
| Unique to TC | Spectral clustering, density diagnostics, cluster comparison, multpl scraping, plotting, email reporting | — | Keep as diagnostics/utilities |
| Unique to new | Fair-value gap module, meta-labeling seam, crash-probability dashboard, honesty metrics, holdout discipline | — | Adopt |

**Consolidation verdict: one project — enhance trading-crab, don't restart.** The lib/pipelines split, config discipline, checkpointing, and tests are exactly the chassis this design needs; nothing in the new design conflicts with that architecture, only with the modeling core inside it. Practical move: this document becomes `trading-crab/docs/design.md`; the mathematical core (labeler, L2/L3 models, evaluation harness) is rebuilt per §11's order inside `trading-crab-lib`; legacy clustering modules are retained as warm starts, baselines, and diagnostics rather than deleted.

## 14. Phase Plan (tracer-bullet skeleton, then module upgrades)

Principle: build the thinnest possible end-to-end pipeline first — every layer present, every layer naive — with the honesty framework installed **before** the first model is tuned. Then upgrade modules independently against frozen interfaces. Each phase has exit criteria; a phase isn't done until they pass.

- **Phase 0 — Foundations (no modeling).** Carve the 2021+ holdout physically (separate files/paths the pipeline cannot read by default); trial registry (config-hash → metrics store); walk-forward runner as core infrastructure; monthly data layer with spliced long histories and vintage alignment. *Exit: an end-to-end "data through 2020-12" dataset builds reproducibly; registry logs a dummy trial; walk-forward runner executes a trivial model.*
- **Phase 1 — Tracer bullet.** Lean 1962+ feature set → jump-model labeler (default λ, K=5) → logistic nowcaster → returns-by-regime tables (the TC baseline) + EWMA vol → naive allocation (vol-targeted regime tilt with hysteresis) → weekly report reusing TC's email machinery. All walk-forward. *Exit: honest backtest 1972–2020 runs end to end; smoothed-vs-filtered gap and detection lag reported; beats nothing yet — that's fine.*
- **Phase 2 — Regime quality (L1).** Skeleton constraints, (K, λ) grid against §4.4 acceptance criteria, subsample stability with Hungarian matching, t-HMM benchmark comparison, label-churn monitoring. *Exit: all six acceptance criteria pass; jump vs HMM label stability documented.*
- **Phase 3 — Regime prediction (L2).** Nowcaster upgrade (recursive feature, γ weights, calibration); separate transition model with regime age; hysteresis tuning; detection-lag dashboard; **daily tripwire monitor** (§23.2/§25 — lightweight daily red-flag orchestrator; a minimal 3-signal version may ship with Phase 1). *Exit: transition-window log-loss beats persistence baseline; calibration curves acceptable; sojourn/lag ratio ≥ ~5; tripwire live with family-independent escalation logic.*
- **Phase 4 — Asset prediction (L3).** Regime-conditional covariance + Ledoit–Wolf; GARCH layer; MoE experts (soft gating, partial pooling); boosted ceiling model; fair-value gap module with convergence KPI. *Exit: MoE beats returns-by-regime tables and the historical-mean baseline (Diebold–Mariano) at 1m/3m on walk-forward; covariance forecasts beat sample covariance.*
- **Phase 5 — Allocation & tactics (L4).** BL or HRP weights, fractional Kelly, model-driven stops, crash dashboard, full weekly report. *Exit: walk-forward strategy beats buy-and-hold SPY on terminal log wealth AND max drawdown 1972–2020; survives the baseline gauntlet.*
- **Phase 6 — Freeze & holdout.** DSR computed against the full trial registry; design freeze declared; **single** evaluation on 2021+. *Exit: the one number. If it disappoints, the correct response is a new design cycle with a new future holdout — not iteration on 2021+.*

## 15. Mathematical Extensions Considered (v1.1 addendum)

Adopted into the design: **Hamilton (2018) regression filter** replaces/benchmarks two-sided smoothers in the fair-value module ("Why You Should Never Use the Hodrick-Prescott Filter" — directly relevant to any two-sided smooth used as ground truth); **Ornstein–Uhlenbeck half-life estimation** as the formal model of fair-value gap decay (gap enters allocation only where estimated half-life ≪ regime sojourn); **cointegration/VECM** (Johansen) for long-run cross-asset relationships — error-correction terms as L2/L3 features; **BOCPD** (Adams–MacKay Bayesian online changepoint detection) as an independent regime-shift alarm cross-checking the nowcaster; **random matrix theory** (Marchenko–Pastur eigenvalue cleaning) alongside Ledoit–Wolf; **extreme value theory** (peaks-over-threshold/GPD) for the crash module's tail estimates; **forecast combination** (equal-weight ensembles across expert variants — Timmermann); **conformal prediction** for distribution-free forecast intervals as the cheap path to distributional outputs (Open Question 1).

Considered and declined (with reasons): deep sequence models / transformers for return prediction (monthly N too small; boosted trees are the empirical ceiling here); reinforcement learning for allocation (sample complexity catastrophically exceeds available history); LLM/NLP sentiment features (point-in-time news corpora are expensive and survivorship-riddled; revisit only as an optional modern-era feature); full Bayesian MCMC estimation of the MS model (correct but heavy; priors enter instead as shrinkage/partial pooling — same math, cheaper); particle filters (needed only if the state goes continuous/nonlinear); copulas beyond regime-conditional correlations (the regime mixture already captures the tail-dependence structure copulas would model).

---

## 16. Tactical Layer — Design Backlog (v1.2 addendum)

Extensions from the tactics discussion. None modify the L0–L4 spine; all are modules hanging off L3/L4 or feature-layer expansions. Status: **backlog** unless marked otherwise.

### 16.1 L3b — Factor-projection layer for satellite assets
- Core insight: do not model each satellite asset; map it onto core factors via loadings: r_sat ≈ β′ r_core + ε (miniature Barra/Axioma risk model). Satellite forecast = β′ · (regime-mixture core forecast) + regime-conditional residual tilt.
- Loadings **regime-conditional or rolling** (sector betas are regime-dependent).
- Satellite universe: sector SPDRs (XLC XLY XLP XLE XLF XLV XLI XLB XLRE XLK XLU), caps/breadth (IWM, RSP), duration ladder (SHY/IEF/TLT/EDV), credit (LQD/HYG), international, crypto proxies.
- RSP/SPY ratio is dual-use: satellite asset AND breadth feature.
- Prerequisite share: L3b builds ~80% of what the future single-name module needs.

### 16.2 Sector rotation module
- Model sector returns **relative to SPY** (market component already handled by core; relative returns have lower vol, stabler loadings).
- Inputs: regime probabilities + sector momentum (Moskowitz–Grinblatt industry momentum, 6–12m relative strength).
- Business-cycle rotation map (Stovall/Fidelity: early→discretionary/financials/industrials, mid→tech, late→energy/materials/staples, recession→utilities/healthcare/staples) = hand-labeled prior; encode as skeleton-style sign priors on regime-conditional sector tilts, let data fill magnitudes.

### 16.3 Fast-feature expansion: breadth, dispersion, vol structure (**promote to Phase 3–4**, not backlog)
- Breadth: advance-decline line, % stocks > 200dma, new highs − new lows, McClellan oscillator, RSP/SPY.
- Cross-sectional dispersion (stdev of constituent returns): high dispersion = selection works, "index flat / churn underneath" detector; implied correlation (index vol vs avg single-name vol).
- VIX complex: level largely redundant with realized vol; **VIX futures term structure slope is not** — backwardation ≈ cleanest real-time stress signal (contango ~80% of the time). Add VVIX, skew as refinements.
- All are market-observed, unrevised, fast-moving → ideal per D12; feed L2 nowcaster/transition and crash module.
- **Data-layer task:** breadth history requires a survivorship-clean constituent source (harder than price series; flag in Phase 0/R13 scope).

### 16.4 L4 action generation: no-trade bands
- Tolerance-band rebalancing (Masters): act only when |w − w*| exceeds band; band width ∝ asset vol and ∝ diversification role (higher vol, lower correlation → wider band).
- Trade to **band edge, not target** (half-adjustment; reduces whipsaw). Prioritize actions by expected utility gain per unit turnover.
- Weekly report per holding: HOLD (inside band) / BUY / SELL with size and rationale. Two anti-churn layers total: regime hysteresis (§5.3) + bands.
- Prioritized ticker lists = cross-section sorted by forecast IR (E[excess return]/σ̂) per horizon {1m, 3m, 12m}, meta-label confidence attached when available.

### 16.5 Core-satellite structure & tactical sleeve (swing trades)
- Strategic core: 80–90% of capital, regime-driven (the L0–L4 platform). Tactical sleeve: remainder; the **only** place swing entries live.
- Sleeve trade definition (mechanical, all required): signal (e.g., fair-value gap in reversion-friendly regime; sector momentum breakout) + vol-scaled stop + **time stop** (exit after N weeks regardless — prevents swing trades silently becoming investments) + size via fractional Kelly on meta-label confidence.
- **Separately tracked KPIs**; sleeve must earn its capital within ~1y of live tracking or shrink. Honest purpose is partly behavioral containment.
- Baseline the whole tactical layer must beat: **dual momentum** (Antonacci — absolute + relative momentum, 12m lookback, monthly) — the published strategy closest to this platform's spirit.

### 16.6 Entry/exit mechanics (future deep-dive)
- Tranche entries/exits (scale in on signal + confirmation) — pairs naturally with probabilistic signals.
- Regime-asymmetric exits: trend regimes → stops only, let winners run; reversion regimes → profit targets (gains decay).
- Stop placement already spec'd (§7): m_k · σ̂, regime-conditional multiplier.

### 16.7 Tactical decisions (RESOLVED)
1. **Long-only.** No shorts, no options. Non-leveraged inverse ETFs (e.g., SH) permitted **only** in the tactical sleeve, with mandatory time stop; leveraged inverse excluded (volatility decay: daily rebalancing makes multi-week returns path-dependent and erosive in chop). Crash module's primary de-risk instruments: cash / duration / gold weight shifts.
2. **Sleeve = account-level separation**, not a capital fraction: dedicated swing-trade IRA vs. long-term accounts. Physical enforcement (same philosophy as the holdout carve). Platform treats them as two portfolios sharing one model stack; weekly report splits accordingly. Sleeve longs may ride strong uptrends as long as they outperform SPY (relative-strength hold condition).
3. **Single names:** open-ended, Fidelity-tradable, US-centric (S&P 500 core). **Excluded:** publicly traded limited partnerships / MLPs (K-1 avoidance — note this constrains energy-space names), cryptocurrencies and crypto ETFs.
4. **Sectors: both.** Model relative-to-SPY (statistically cleaner); absolute forecast reconstructed as core SPY forecast + relative forecast; report both.

### 16.8 Additional references (tactical)
- Moskowitz & Grinblatt (1999), "Do Industries Explain Momentum?"
- Antonacci, *Dual Momentum Investing* (2014).
- Masters (2003), "Rebalancing" (tolerance bands), JPM.
- Grinold & Kahn ch. on portfolio construction w/ transaction costs (band logic).
- Menchero et al., Barra risk model handbooks (factor projection reference).
- Whaley (2009), "Understanding the VIX"; Simon & Campasano (2014), VIX futures basis.

---

## 17. AVWAP & Indicator-Translation Module (v1.3 addendum)

Motivation: incorporate Mike Silva (Figuring Out Money) style tactical indicators — AVWAP-centric, Brian Shannon lineage — translated from daily/swing horizons to this platform's 1m/3m horizons, with each indicator admitted as a *candidate feature* through the honesty framework rather than as a rule.

### 17.1 AVWAP formalization
- AVWAP_τ(t) = Σ_{s=τ..t} p_s v_s / Σ_{s=τ..t} v_s — volume-weighted average cost basis of all shares traded since anchor τ. Price vs AVWAP_τ = whether the average post-event entrant is in profit.
- Transmission mechanisms (unusually defensible for TA): disposition effect / breakeven-seeking (underwater holders cluster supply near cost basis); institutional execution benchmarked to VWAP (Almgren–Chriss; Madhavan) → resting institutional interest at these levels; partially self-fulfilling.

### 17.2 Algorithmic anchor selection (removes AVWAP's subjectivity; enables honest backtests)
Rule-generated anchor family — no hindsight anchor-picking:
- **Regime-transition dates** (L1 labels historically; L2 detection in real time) → "average cost basis of everyone positioned since this environment began." Regime-conditional AVWAP: the platform's signature extension; persists for months, matching holding periods.
- 52-week high/low dates; earnings dates (single names); calendar institutional anchors (year/quarter opens); volume-percentile event days (top-decile daily volume, mechanically detected).

### 17.3 Feature-ization (per asset, per active anchor)
- Signed vol-normalized gap (p − AVWAP_τ)/σ̂; AVWAP slope; time above/below; defense count (touch-and-hold events); **confluence density** (count of active AVWAPs within a band of price).
- Kinship: AVWAP gap = volume-weighted cousin of the §6.3 model-smoothed fair-value gap; same module, same KPI — **forward returns conditional on the gap, regime-conditional** (breakeven-seeking behavior likely regime-dependent).
- Sleeve entry archetype (Silva/Shannon style, mechanized): uptrend + price above rising regime-anchor AVWAP + pullback-to-AVWAP defense + relative strength vs SPY → tranche entry; stop below AVWAP (m_k·σ̂ buffer); relative-strength hold condition; time stop.

### 17.4 Discipline for TA-derived features
- Formal record: early positive TA results (Brock–Lakonishok–LeBaron 1992) largely evaporated under data-snooping corrections (Sullivan–Timmermann–White 1999). Therefore: every Silva-derived indicator enters via the trial registry, is evaluated with purged CV and DM tests against the no-TA baseline at 1m/3m, and earns inclusion or is dropped. The platform is the instrument converting "this indicator looks good on the show" into measured incremental value at these horizons.
- **RESOLVED (no Discord access — closed):** no member-only sources will be used. Silva-style indicators are harvested opportunistically from public content and translated through §17.3 as encountered; the framework is indicator-agnostic and nothing blocks on this. The expected-move / 2σ mechanics are reproduced independently from free implied-vol indices (§18).

### 17.5 References (module)
- Shannon, *Maximum Trading Gains with Anchored VWAP*; Shannon, *Technical Analysis Using Multiple Timeframes*.
- Brock, Lakonishok & LeBaron (1992); Sullivan, Timmermann & White (1999) — data-snooping correction.
- Almgren & Chriss (2000), "Optimal Execution of Portfolio Transactions"; Madhavan (2002), "VWAP Strategies."

---

## 18. Options-Implied Information Layer (v1.4 addendum)

Principle: consume the options market as a **forecast market**; trade nothing. Two data tiers with different epistemic status.

### 18.1 Tier 1 — Index level (FREE, full history, backtestable → features)
- CBOE implied-vol index family: VIX (30d SPX), VIX9D/VIX3M/VIX6M (term structure), VVIX (vol-of-vol), SKEW (tail pricing), RVX (small cap), **OVX (oil), GVZ (gold)** — forward-looking vol inputs for the exact core assets.
- **Expected move / 2σ levels reproduced without chain data:** EM ≈ S·(IV/100)·√(T/365); 2σ range = S·(1 ± 2·IV·√T). Per asset class via the matching index.
- **Variance risk premium (highest-value feature):** VRP = IV² − RV² (e.g., VIX² − realized variance). Bollerslev–Tauchen–Zhou (2009): high VRP predicts equity returns at 1–6m horizons with unusually strong OOS R². Interpretation: wedge between risk-neutral (Q) and physical (P) distributions = price of fear; mean-reverts. Feed L2 + L3.
- **Vol-layer blending:** IV is a biased-but-informative realized-vol forecast (bias = VRP); blend IV with GARCH/EWMA in §6.2 — documented improvement. OVX/GVZ give oil/gold vol targeting forward-looking inputs.
- Term-structure slope & SKEW → crash module (strengthens §16.3). CBOE put/call ratios (free) as sentiment candidates via the registry.

### 18.2 Tier 2 — Chain level (no free history → LEVELS, not features)
- Open-interest concentrations ("volume shelves"), GEX/dealer-gamma pinning, max pain: plausible short-term magnet/pin mechanics via delta-hedging flows, but **unbacktestable without chain history** → never admitted as model features.
- Permitted use: utility reads **current** Fidelity chains, marks large-OI strikes as reference levels for sleeve entries/exits — same epistemic status as a hand-drawn support line.
- Backlog: Breeden–Litzenberger (1978) — risk-neutral density = ∂²C/∂K²; full market-implied predictive distribution per expiry. Connects to Open Question 1 (distributional forecasts); blocked on chain history.

## 19. Finance ↔ Mathematics Dictionary (v1.4 addendum)

Purpose: prevent unintentional wheel-reinvention; every finance term mapped to the underlying math object. Reinventions already committed (convergent validity, now named): fair-value gap = **value factor**; regime-conditional coefficients = **conditional betas** (conditional-CAPM literature); MoE = **conditional factor model**.

| Finance term | Mathematical object |
|---|---|
| Beta | OLS slope: Cov(r_i, r_m)/Var(r_m); multi-factor betas = multivariate regression coefficients (= L3b loadings) |
| Alpha | Regression intercept; mean residual return after controlling for exposures; "finding alpha" = E[ε] > 0 with defensible t-stat |
| CAPM / Fama–French / APT | One-regressor linear model / chosen-regressor extension / statistical (latent) factor model |
| Sharpe ratio | Annualized standardized mean excess return; Sharpe·√years = t-statistic (why track length dominates skill inference; why DSR exists) |
| Information ratio | Sharpe of residual (active) returns |
| Information coefficient (IC) | Correlation(forecast, outcome), usually cross-sectional Spearman; Grinold: IR ≈ IC·√breadth |
| R² of factor regression | Systematic variance share; 1−R² = idiosyncratic share (~50–70% for single names) |
| Volatility / realized vol | Annualized return stdev / rolling estimator thereof |
| Implied volatility | σ inverting Black–Scholes from observed option price — a quote convention (unit conversion), not a model endorsement |
| Variance risk premium | E_Q[var] − E_P[var]; wedge between risk-neutral and physical measures |
| Greeks (delta, gamma, vega, theta) | Taylor-expansion terms: ∂/∂S, ∂²/∂S², ∂/∂σ, ∂/∂t |
| Duration / convexity | −(1/P)·∂P/∂y and (1/P)·∂²P/∂y² — same Taylor expansion, in yield |
| Momentum / mean reversion | Positive / negative return autocorrelation at the stated horizon |
| Value | Deviation from a slow-moving anchor (= the fair-value gap, §6.3) |
| Carry | Deterministic drift: expected return if prices are unchanged |
| Kelly criterion | argmax E[log wealth] (= our objective, D-level) |
| VaR / CVaR | Return quantile / expected shortfall beyond it |
| Drawdown | Max peak-to-trough of cumulative log wealth — a path functional, not a moment |
| Turnover | ‖Δw‖₁ per period |

**New model candidates unlocked by the mapping (via registry):**
- **Residual momentum** (Blitz–Huij–Martens): momentum on factor-regression residuals ε (L3b computes these already) — better Sharpe, smaller crashes than raw momentum. Near-zero marginal cost.
- **Betting-against-beta** (Frazzini–Pedersen): low-beta assets persistently over-deliver risk-adjusted returns (leverage-constrained investors overpay for high beta). Long-only implication: low-beta tilt + vol-targeted sizing over high-beta glamour; harmonizes with §6.2.

## 20. Volume Integration (v1.4 addendum)

Theory anchor: Clark (1973) Mixture of Distributions Hypothesis — volume ∝ information-arrival rate; returns are subordinated to an information clock (deep version of dollar bars, López de Prado). Volume features must be detrended: log volume vs own EMA ("volume surprise"), turnover, or normalized dollar volume — never raw volume (secular growth).

Four integration points:
1. **Regime emissions (L1):** aggregate volume-surprise + **Amihud illiquidity** (|r|/dollar volume) — illiquidity is a priced factor and spikes ahead of/during crises; liquidity evaporation is the defining mechanical feature of the crisis state.
2. **Nowcast/transition features (L2):** liquidity deterioration and volume-surprise lead stress transitions.
3. **MoE interaction gate (L3):** Campbell–Grossman–Wang (1993) — high-volume price moves continue; low-volume moves revert. Encode as return × volume-surprise interaction modulating the continuation-vs-reversion coefficient. Refinement to §6.3: fair-value gaps opened on LOW volume revert more reliably than gaps opened on high volume.
4. **AVWAP module (§17):** already volume-weighted by construction.

### References (v1.4)
- Bollerslev, Tauchen & Zhou (2009), "Expected Stock Returns and Variance Risk Premia."
- Breeden & Litzenberger (1978), "Prices of State-Contingent Claims Implicit in Option Prices."
- Blitz, Huij & Martens (2011), "Residual Momentum."
- Frazzini & Pedersen (2014), "Betting Against Beta."
- Clark (1973), "A Subordinated Stochastic Process Model…"; Karpoff (1987), volume-volatility survey.
- Amihud (2002), "Illiquidity and Stock Returns."
- Campbell, Grossman & Wang (1993), "Trading Volume and Serial Correlation in Stock Returns."

---

## 21. No-Trade Band Mathematics (v1.5 — closes §16.4 deep-dive)

- Theory: utility loss from deviation is second-order, L ≈ ½γ(w−w*)′Σ(w−w*); trading cost is first-order → optimal policy is a **no-trade region**, never continuous rebalancing. Cube-root law: δ ∝ (c/γ)^(1/3) (Constantinides 1986; Davis–Norman 1990; Leland 2000) — even tiny costs justify wide bands.
- **Platform reinterpretation (frictions out of scope):** the "cost" is estimation noise; the band is a **confidence interval on w***. Bootstrap/resample allocation inputs (Michaud resampling) → SE(w*). Trading deviations smaller than the target's own noise = trading randomness.
- Spec (weekly action generation):
  1. Per-asset trigger: |w_i − w*_i| > max(floor_i, z·SE(w*_i)), widened ∝ σ_i, narrowed for diversifiers.
  2. Portfolio trigger: (w − w*)′Σ(w − w*) > τ² (catches many-small-deviation drift).
  3. On trigger: trade to **band edge** (half-adjustment), not center; prioritize by utility gain per unit turnover.
- Output per holding: HOLD / BUY x / SELL x with which trigger fired and rationale.

## 22. (K, λ) Selection Protocol for the Jump Model (v1.5 — closes Open Question 5)

- Grid: K ∈ {4,5,6,7} × λ on a log grid. λ normalized by feature dimension (grid transfers across feature-set changes); note λ–K interaction (higher K needs stiffer λ for sojourn targets). Multi-restart per cell; k-means warm start.
- **Stage A — hard acceptance filters (§4.4 operationalized):**
  - Occupancy ∈ [~8%, ~35%] per state; median sojourn ≥ months-scale floor.
  - Skeleton mappability: each state matches one D3 semantic slot via emission signature (Hungarian matching).
  - Stability: ARI/AMI between full-sample labels and leave-one-decade-out labels; **online–offline agreement** (expanding-window refit labels vs final labels — direct predictor of live label churn; the metric where jump models beat HMMs, Nystrup et al.).
- **Stage B — rank survivors by decision-relevance:** regime-separation index on ASSET RETURNS: min pairwise distance (Wasserstein/KS) between regime-conditional return distributions ÷ within-regime dispersion. Winner = most economically distinct states, not best likelihood.
- **Excluded from selection:** downstream strategy P&L (backdoor for overfitting the stack to one historical path). Every grid cell logged in the trial registry regardless of outcome (true DSR denominator).

## 23. Crash-Avoidance & Re-Entry Playbook (v1.5 — the drawdown objective, formalized)

Primary user goal: avoid 2000/2008-class drawdowns (current concern: AI-bubble deflation); re-enter systematically near recoveries; hold crisis-beneficiary ballast.

### 23.1 Honest expectations (set in advance, in writing)
- The system CANNOT sell tops (filtered detection lags 1–3 months; tops are only visible smoothed).
- **Favorable case — slow, grinding bears (2000–02, 2007–09):** internals deteriorate detectably in real time (credit spreads, breadth, vol regime). Realistic outcome: absorb first ~10–15% of drawdown, sidestep the subsequent ~30–40%, re-enter missing first ~10–20% of recovery. Bubble deflations are historically the slow kind — the insurance matches the feared risk.
- **Unfavorable case — fast V (1987, Feb 2020):** monthly detection sells after the bottom, re-buys higher (whipsaw). Vol targeting mechanically softens it. Framing: **insurance whose premium is whipsaw in V-events, payout in prolonged bears.**
- **Baseline to beat:** Faber (2007) 10-month SMA rule (in SPY above, cash below) — the classic crash-avoidance heuristic; strong in 2000/2008, whipsawed in chop. If the regime machinery can't beat it on walk-forward log wealth AND max drawdown, the machinery isn't earning its complexity.

### 23.2 Exit side — fast, rule-triggered, staged
- Composite trigger (OR-logic across independent detectors): L2 P(crisis) crossing hysteresis threshold; VIX term-structure inversion; credit-spread velocity; breadth collapse; drawdown-from-peak backstop.
- **Daily tripwire monitor (new, adopted):** lightweight daily checklist of the fast signals above — flags "run weekly scoring early," does NOT rescore the platform. Closes the speed gap left by the weekly cadence (D10) at near-zero cost.
- Tiered response: Tier 1 (elevated risk) → de-risk to vol target; Tier 2 (crisis confirmed) → minimum equity + crisis allocation.

### 23.3 Crisis-TYPE conditioning (the regime skeleton's core payoff)
- The crisis beneficiary depends on which crisis: **deflationary/disinflationary crash (2000, 2008) → long-duration Treasuries**; **inflationary crash (2022) → gold, cash, short duration; bonds fall WITH stocks**. Gold historically the only both-tails asset (consistent with Glenn's oil-scenario maximin finding).
- Type read in real time during the decline: breakevens, USD, commodity behavior. AI-bubble deflation most plausibly rhymes with 2000 (duration-friendly) — but DETECT, don't assume; a bubble popping into an inflationary backdrop flips the answer.

### 23.4 Re-entry side — slow, staged, pre-committed
- Bottoms are violent; best days cluster near lows; "feels stable" arrives 20–30% off the low → binary wait-for-stability re-entry is the costliest strategy.
- Recovery signatures: **breadth thrust** (Zweig: 10-day advance ratio washout → overwhelming), vol peak-and-decline, credit spreads narrowing from wides, VIX term structure re-steepening to contango.
- Mechanics: 3–4 **pre-committed** tranches, signal-spaced AND time-spaced, vol-targeted sizing ("cost averaging down" converted to protocol). Pre-commitment is the point: maximum opportunity is engineered to feel like maximum danger.
- Asymmetry principle: **fast out, slow in.**

### 23.5 Platform-consistent posture for entering from cash
- 100% cash is a position with its own (realized) risk. Design-consistent entry: staged tranches into a regime-robust core (maximin Robust Core template) at vol-targeted sizing, exit rules pre-committed BEFORE entry. The mechanical exit is what makes holding exposure during a suspected bubble tolerable. (Guidance, not advice; execution decisions are the operator's.)

### References (v1.5)
- Constantinides (1986), "Capital Market Equilibrium with Transaction Costs"; Davis & Norman (1990), "Portfolio Selection with Transaction Costs"; Leland (2000), multi-asset rebalancing with costs.
- Michaud (1998), *Efficient Asset Management* (resampled efficiency → SE(w*)).
- Faber (2007), "A Quantitative Approach to Tactical Asset Allocation."
- Zweig, *Winning on Wall Street* (breadth thrust).

---

## 24. Physics Correspondence & the Three Stickiness Layers (v1.6 addendum)

Clarification (jump model ≠ rebalancing mechanism): the jump model is the **L1 ground-truth labeler**; threshold-gated rebalancing is the **L4 no-trade band** (§21). Both instantiate the same principle — penalize state changes so only persistent evidence moves you — at different layers.

**Exact correspondence (not analogy):** the jump-model objective J(s, μ) = Σ_t ‖x_t − μ_{s_t}‖² + λ·Σ 1[s_t ≠ s_{t−1}] is the energy of a **1D K-state Potts model**: site-dependent external field (data-fit term) + ferromagnetic nearest-neighbor coupling λ. Label decoding = **ground state of the Potts chain** (why DP is exact — 1D transfer-matrix solvability). λ = **domain-wall energy**; regimes = magnetic domains. The HMM is the finite-temperature version: log A_ij = couplings, smoothed posteriors γ = Boltzmann marginals, forward–backward = transfer matrix, Viterbi = zero-temperature ground state.

**Three stickiness mechanisms, three layers, three noise sources:**
1. **L1 labels:** jump penalty λ (domain-wall energy) — suppresses label flicker in ground truth.
2. **L2 actions:** probability hysteresis bands (enter 0.7 / exit 0.4) — true path-dependent hysteresis, state depends on approach direction (magnetization lagging the field).
3. **L4 weights:** no-trade region (§21) — suppresses noise-chasing turnover.
None substitutes for another.

## 25. Multi-Timescale Consensus Framework (v1.6 addendum)

Principle (adopted): all model families (regime nowcast/transition, asset prediction, risk flags) are fit and scored per horizon (D8: 1m/3m/12m); ACTION intensity is a monotone function of cross-horizon and cross-family agreement. Precedent: time-series momentum lookback ensembles (Moskowitz–Ooi–Pedersen); multiple-timeframe analysis (Shannon).

- **Consensus score:** per asset, confidence-weighted sign agreement across the horizon family; response is monotone modulation, not binary gates — position size ↑ with agreement; stop multiplier m_k tightens as red-flag consensus rises; exposure bleeds rather than jumps.
- **Statistical warning (binding):** overlapping windows agree under the null (the 12m window contains the 1m window). Calibrate consensus thresholds against the null correlation structure via block bootstrap — "N horizons agree" is weak evidence relative to N independent detectors.
- **Family-independence rule (tripwire orchestrator):** escalation requires flags from ≥2 structurally independent families — (a) price/trend, (b) vol/options structure, (c) credit, (d) breadth/liquidity. All lookbacks within a family = ONE vote. Turns the tripwire into a voting ensemble of weak, diverse detectors; diversity, not count, suppresses false alarms.
- **Tripwire monitor spec (roadmapped — Phase 3, minimal version optionally Phase 1):** daily checklist, no rescoring; severity = family-weighted flag count; outputs: none / "run weekly scoring early" / "Tier-1 de-risk review" (§23.2). Cheap by construction; feature families reuse §16.3 and §18.1 inputs.

## 26. Placeholder: "Old Fool Indicator" (v1.6 — awaiting definition)

No indicator by this name is locatable in the literature or practitioner sources (searched July 2026). Nearest candidates: odd-lot theory (retail odd-lot activity as contrarian "dumb money" signal), Greater Fool Theory (bubble-dynamics concept, not an indicator), dumb-money/smart-money sentiment composites. **Action:** Glenn to describe the intended measurement; a homegrown version will be derived regardless (stated intent). If contrarian-retail-sentiment family, free ingredients: CBOE equity-only put/call extremes, AAII sentiment spread, fund-flow proxies — composited, z-scored, admitted via the trial registry like any candidate feature.

---

## 27. Stop-Loss Execution Mechanics & Gap Risk (v1.7 addendum — motivated by live incident, 2026-07-08: IAU gapped ~3% through a 1% stop)

### 27.1 Mechanics
- A plain stop-loss is a TRIGGER that converts to a MARKET order; fill = next available price. Opening gaps fill at the gap ("gapped through"); shortfall vs stop price = slippage.
- **Structural exposure:** ETFs on 24h-traded underlyings (gold, oil, international) can only execute stops during US equity hours while risk accrues around the clock → overnight moves surface as opening gaps past any stop. Gaps also cluster on scheduled macro events (FOMC, CPI, NFP).
- **Stop-limit order** = the "conditional stop range": trigger price + limit price; gaps past the limit do NOT fill. Trade-off is fundamental: plain stop guarantees exit, not price; stop-limit guarantees price, not exit — unprotected in exactly the crash scenario. No order type guarantees both (only protective puts do; out of scope per §16.7).

### 27.2 Policy stack (adopted)
1. **Vol-scaled distances (reaffirms §7):** stop distance = m_k · σ̂ (≈2–3× ATR), regime-conditional. A stop inside one daily sigma is a coin-flip exit, not protection.
2. **Size the position, not the stop:** position size = risk budget ÷ stop distance. Gap damage scales with SIZE; wide stop + small position ≻ tight stop + large position at equal planned risk. Never tighten stops to justify an oversized position.
3. **Close-based evaluation + alerts, not resting orders (platform-consistent):** rule = "close beyond level → decide next day." Broker price ALERTS at model-derived levels; operator decides at the close with tripwire context. A resting stop order is an automated trade — contrary to the guidance-only philosophy. Resting orders reserved for cases where same-day execution certainty is explicitly wanted (accepting gap risk).
4. **Event-window awareness:** no fresh tight stops into FOMC/CPI/NFP windows (tripwire calendar already tracks these).
5. **Thesis-consistent stops (deepest rule):** a stop encodes "below this, my thesis is falsified." Trend positions → stops coherent (tight, m_k small in trend/crisis regimes). Reversion/valuation/ballast positions → lower price CONFIRMS the thesis; stops are the wrong tool — use sizing + pre-committed tranche adds (§23.4) instead. Weekly report must label each holding's thesis type (trend vs reversion) and emit the matching risk tool; "would I buy at the stop price?" is the diagnostic — if yes, remove the stop and resize.
