# Phase 4: Asset Prediction & Allocation - Research

**Researched:** 2026-07-22
**Domain:** Regime-conditional asset return/vol analytics, naive vol-targeted allocation with
hysteresis, daily risk tripwire, weekly report/email delivery — Python/pandas, no new
dependencies.
**Confidence:** HIGH (all claims grounded in the actual `platform/` codebase; two genuine data
gaps found and documented below with MEDIUM/CITED confidence fixes)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Holdings YAML schema (L4-03)**
- **D-01:** Weights (fractions) per account. One YAML per account (e.g.
  `config/accounts/<account>.yaml`, gitignored like other `*.local.yaml` secrets-adjacent
  files; a committed `config/accounts/example.yaml` documents the schema). Each file maps
  ticker → weight fraction; weights + cash must sum to ~1.0 (validate with tolerance,
  warn don't fail). No price lookups needed — the report compares target weights vs
  current weights directly. Fidelity CSV parser documented as a placeholder seam only.

**Weekly report delivery (L4-02)**
- **D-02:** Always write `weekly_report` markdown; email is opt-in behind the existing
  incumbent pattern (`--send-email` + SMTP config / TC_* env vars). Reuse the incumbent's
  `email.py` machinery (markdown→HTML, multipart) — do not fork it; the platform report
  writer produces the markdown, the existing sender delivers it. Platform report artifact
  lives under `outputs/reports/platform/` to avoid colliding with the incumbent's
  weekly_report.md.

**Portfolio target volatility (L4-01)**
- **D-03:** 10% annualized target portfolio vol, exposed as a config knob
  (`allocation.target_vol_annual: 0.10`). Defensive by design — the core value is beating
  SPY net of avoided drawdowns, not matching its risk. Position scale ∝
  target_vol / σ̂_portfolio, long-only, capped at 1.0 (no leverage), cash absorbs the
  residual (FZFXX/SPAXX are the cash sleeve).

**Tripwire v1 signal set (L4-04)**
- **D-04:** Confirmed roadmap trio, one per independent family:
  1. Realized-vol spike (vol family) — short-window realized vol vs its trailing baseline
  2. Credit-spread velocity (credit family) — BAA–AAA widening rate
  3. Drawdown-from-peak on SPY (price family)
  OR-logic escalation: none → "run weekly scoring early" (1 trigger) → "Tier-1 de-risk
  review" (2+ triggers). All thresholds config-driven (`tripwire:` section). Daily
  cadence, manual CLI run in v1.

**Carried forward (locked earlier — do not re-decide)**
- Hysteresis bands: act when regime P crosses ~0.7, unwind below ~0.4, hold in between
  (design §5.3; ROADMAP criterion 3). Expose both in config.
- v1 asset universe: Core 5 + satellites + holdings — SPY, GLD, TLT, USO + QQQ, IWM,
  EFA, IEF, SHY, VNQ, VYM, AGG, HYG, LQD, EEM, DBA, MCD, COST, O, TSM, IAU, SLV, GDX,
  FZFXX, SPAXX (cash). Availability limited by what Phase 1 ingested; NULL-tolerant for
  short histories (D11).
- EWMA vol per asset (design §6.2): position size ∝ 1/σ̂; realized-vol features also
  feed the tripwire. GARCH/DCC/Ledoit–Wolf covariance → v2 (L3-V2-01).
- Return forecasts shrunk brutally (design §7): the regime tilt comes from historical
  regime-conditional stats (Phase 3 labels + returns-by-regime tables), never from raw
  point-forecast optimization.
- Nowcaster probabilities (Phase 3 `platform/prediction/nowcaster.py`) drive the regime
  distribution; empirical transition matrix gives the trajectory diagnostic.
- All new code in `src/trading_crab_lib/platform/` (e.g. `platform/assets/`,
  `platform/allocation/`, `platform/report/` — planner's call); incumbent untouched except
  read-only reuse of `email.py` senders.
- Honesty rails apply: dev data ≤2020-12; any evaluated configuration → registry; causal
  features only in anything supervised (returns-by-regime tables are historical
  conditioning, not supervised prediction — no CV needed in v1).
- v2 deferrals bound this phase: L3-V2-01/02/03 (covariance, MoE, fair-value gap),
  L4-V2-01..04 (BL/HRP/Kelly, stops, crash dashboard, tactical sleeve), full tripwire
  orchestrator (L2-V2-03).

### Claude's Discretion
- EWMA half-life/lambda convention for monthly (and daily tripwire) vol — literature
  default (e.g. RiskMetrics-style), exposed in config.
- Regime-tilt construction details: how regime-conditional stats map to tilt weights
  (e.g. within-regime Sharpe rank, clipped ≥ 0, blended by nowcaster probabilities),
  provided it is naive, transparent, and documented in the report.
- Hysteresis state persistence (where the "last acted regime" state lives between runs).
- Report layout/sections, and module split inside `platform/`.
- Tripwire threshold defaults (documented derivation, config-overridable).

### Deferred Ideas (OUT OF SCOPE)
- BL/HRP weights, fractional Kelly, no-trade band math (§21) — v2 (L4-V2-01).
- Model-driven vol-scaled stops + §27 policy stack — v2 (L4-V2-02).
- Crash-probability dashboard + crisis-type conditioning — v2 (L4-V2-03).
- Regime-conditional covariance, Ledoit–Wolf, GARCH/DCC — v2 (L3-V2-01).
- MoE + boosted ceiling model; fair-value gap module — v2 (L3-V2-02/03).
- Full tripwire orchestrator with family-independence voting (§25) — v2 (L2-V2-03).
- Fidelity positions-CSV parser — v2 (L4-V2-05); seam documented in v1.
- Automated scheduled runs + email delivery — v2 (OPS-V2-01).
- Tactical sleeve reporting (§16.5) — v2 (L4-V2-04).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| L3-01 | Returns-by-regime tables for the v1 universe (design §14 Phase 1) | §"Returns-by-Regime Construction" — data sources identified (`monthly_raw` checkpoint already has all needed columns), stats schema specified, D11 NULL-tolerance addressed |
| L3-02 | EWMA volatility forecasts per asset feeding sizing and the tripwire | §"EWMA Vol Conventions" — pandas `.ewm(halflife=...)` verified API, RiskMetrics λ conventions cited, daily vs monthly window split specified |
| L4-01 | Naive vol-targeted regime-tilt allocation with hysteresis bands | §"Vol-Targeted Regime-Tilt" + §"Hysteresis State Machine" — full state-machine semantics derived from design §5.3 + CONTEXT D-05.3, portfolio-vol-without-covariance approach specified |
| L4-02 | Weekly report reusing existing report/email machinery | §"Weekly Report Assembly" — exact `email.py` call sequence verified from source, artifact path convention specified |
| L4-03 | Current holdings input via manual YAML per account | §"Holdings YAML" — schema, validation, and directory convention specified; explicitly does NOT reuse incumbent `load_portfolio()` (silent-normalize behavior conflicts with D-01's warn-don't-fail requirement) |
| L4-04 | Minimal daily tripwire monitor, 3 signals, OR-logic escalation | §"Tripwire" — signal formulas specified; **critical gap found**: SPY and daily credit-spread data are NOT currently ingested by the platform (see Pitfall 1 and Pitfall 2) |
</phase_requirements>

## Summary

Phase 4 is pure composition over Phase 1 (data) and Phase 3 (labels/nowcaster) — no new
libraries are needed; `pandas.DataFrame.ewm()` covers every volatility calculation the phase
requires. The codebase already assembles almost everything L3-01 needs: `transforms_monthly.py`'s
`build_monthly_spine()` orchestrator persists a `monthly_raw` checkpoint that already contains the
5 core spliced research series (`equities_tr`, `long_duration_tr`, `gold`, `oil`, `cash`) **and**
monthly closes for every satellite/holdings ticker in one wide frame, aligned to the same
month-end index the regime labels use. `platform/ingestion/prices_daily.py` similarly persists a
`daily_raw` checkpoint with daily closes for every satellite/holdings/watchlist ticker — the
correct source for EWMA vol and two of the three tripwire signals.

Two concrete gaps were found by tracing the actual data flow (not assumed): **(1)** SPY itself is
never ingested — the "equities" asset class only exists as the monthly, multpl-sourced
`equities_tr` research series, so the tripwire's SPY-drawdown-from-peak signal has no daily price
series to compute against until SPY is added to `universe.satellites` (or a new ticker list) in
`config/platform_settings.yaml`. **(2)** FRED's `BAA`/`AAA` series configured in `fred_monthly` are
monthly-frequency only (confirmed against FRED's own series pages); a genuinely daily
credit-spread-velocity signal needs the daily counterparts `DAAA`/`DBAA`, which are not configured
or fetched anywhere in the codebase yet. Both are small, additive config/ingestion changes — no
existing frozen code is touched — but the planner must schedule them as explicit tasks, not assume
"the data already exists."

Hysteresis, holdings YAML, and weekly-report-email integration all have unambiguous designs once
the exact function signatures in `email.py`, `checkpoints.py`, and design §5.3 are read together:
track P(active regime) specifically (not argmax), switch only on a 0.40-cross-down **combined
with** a competing regime's 0.70-cross-up, and persist that one-row state as a normal parquet
checkpoint (`get_platform_checkpoint_manager().save(...)`) rather than hand-rolling JSON I/O.

**Primary recommendation:** Build four small modules — `platform/assets/{returns,vol}.py`,
`platform/allocation/{hysteresis,tilt}.py`, `platform/tripwire/monitor.py`,
`platform/report/{holdings,weekly}.py` — each following the `gap_lag.py` shape (pure compute
functions + `report_*`/persist function + `__main__` synthetic self-check), add SPY +
DAAA/DBAA to `platform_settings.yaml`, and reuse `email.py` unmodified for delivery.

## Architectural Responsibility Map

This is a batch/CLI Python platform, not a web app — tiers below are the project's actual layers
(ingestion → analytics → decision → delivery/ops), mapped from the generic template categories.

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Returns-by-regime tables (L3-01) | Analytics Layer (`platform/assets/`) | Data Layer (`monthly_raw` checkpoint) | Pure aggregation over already-persisted monthly prices/regime labels; no new ingestion |
| EWMA vol forecasts (L3-02) | Analytics Layer (`platform/assets/`) | Data Layer (`daily_raw`/`monthly_raw`) | `.ewm()` over existing daily/monthly return series; feeds both sizing (L4) and tripwire |
| Vol-targeted regime tilt + hysteresis (L4-01) | Decision Layer (`platform/allocation/`) | Analytics Layer (consumes L3 outputs), Data Layer (state checkpoint) | Combines nowcaster probabilities + returns-by-regime + vol into target weights; owns the persisted hysteresis state artifact |
| Weekly report + email (L4-02) | Delivery Layer (`platform/report/`) | Decision Layer (consumes allocation output), incumbent `email.py` (reused, read-only) | Markdown assembly is new platform code; SMTP delivery is 100% incumbent, unmodified |
| Holdings YAML (L4-03) | Delivery Layer (`platform/report/holdings.py`) | Data Layer (`config/accounts/*.yaml`, gitignored) | Config-file loader + validator, no computation; feeds the trades-implied comparison in the report |
| Daily tripwire (L4-04) | Ops/CLI Layer (`platform/tripwire/monitor.py`, `python3 -m ...tripwire`) | Data Layer (needs SPY + DAAA/DBAA — currently missing, see Pitfalls), Analytics Layer (reuses EWMA vol fn) | Standalone daily-cadence entry point, independent of the weekly report's monthly cadence |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | already pinned ≥2.0 (installed: verify via `python -c "import pandas; print(pandas.__version__)"`) | `.ewm(halflife=...)` EWMA vol, `.rolling()` drawdown/peak tracking, `.pct_change()` returns | Already the project's sole DataFrame engine; `.ewm()` API confirmed via local docstring inspection — supports `halflife` directly, no manual alpha math needed [VERIFIED: local pandas installation docstring] |
| numpy | already pinned ≥1.25 | vectorized min/max/clip for scale-factor capping (`min(1.0, target_vol/sigma_hat)`) | Already a dependency; no new surface needed |
| pyyaml | already pinned ≥6.0 | holdings YAML load/validate | Already used by `config.py`/`email.py`; identical pattern to `load_portfolio()` |
| joblib | already pinned ≥1.3 | N/A this phase (no new models) — noted for completeness since `nowcaster` checkpoint uses it | Existing convention (P27 fix) — no raw pickle |

**No new packages are required.** `pandas.DataFrame.ewm(halflife=...)` (verified locally: `pandas.core.generic.ewm` signature accepts `halflife` as a float, computing `alpha = 1 - exp(-ln(2)/halflife)`) covers every EWMA calculation this phase needs — matches CONTEXT's explicit "zero new pip dependencies expected."

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `pandas.ewm()` | `arch` package (GARCH(1,1)) | Design §6.2 mentions GARCH as an equally valid vol-clustering model, but it's explicitly v2 scope (L3-V2-01 covariance/GARCH layer); a new dependency for a v1 tracer-bullet phase violates the "naive on purpose" doctrine (§14) |
| Hand-rolled JSON state file for hysteresis | `CheckpointManager.save()/.load()` (existing) | Reusing the existing parquet+metadata checkpoint infra is strictly less code and gets freshness/corruption-logging for free (D-01 pattern: never reimplement save/load) |
| New `platform/report/holdings.py` loader | Incumbent `trading_crab_lib.config.load_portfolio()` | Incumbent silently normalizes weights to sum=1 and has no cash concept — conflicts with D-01's "validate with tolerance, warn don't fail" and the explicit cash-sleeve requirement; do not reuse, write a new small loader |

**Installation:** none — no `pip install` step for this phase.

## Package Legitimacy Audit

**Not applicable.** This phase adds zero new pip dependencies (CONTEXT constraint, confirmed
achievable: `pandas.ewm()` covers all EWMA math; `pyyaml`/`joblib`/`pandas`/`numpy` are already
installed project dependencies). No `npm view`/`pip index versions` verification needed because
nothing new is installed.

## Architecture Patterns

### System Architecture Diagram

```
                         ┌─────────────────────────────────────────┐
                         │  Phase 1 checkpoints (platform ns)       │
                         │  monthly_raw  (5 core + satellite tickers,│
                         │                month-end index)          │
                         │  daily_raw    (satellite/holdings tickers,│
                         │                daily index — NO SPY yet) │
                         └───────────────┬───────────────────────────┘
                                         │
        ┌────────────────────────────────┼─────────────────────────────┐
        │                                │                             │
        ▼                                ▼                             ▼
┌───────────────┐              ┌───────────────────┐         ┌──────────────────┐
│ Phase 3        │              │ platform/assets/   │         │ platform/tripwire/│
│ regime_labels  │─────┐        │  returns.py         │         │  monitor.py        │
│ nowcaster.pkl  │     │        │   returns_by_regime()│        │ needs: SPY daily,  │
│ transition mtx │     │        │  vol.py             │         │  DAAA/DBAA daily   │
└───────┬────────┘     │        │   ewma_vol()         │◄────────┤  (MISSING — new    │
        │              │        └─────────┬───────────┘         │  ingestion needed) │
        │ P(regime|t)  │                  │                     └─────────┬──────────┘
        ▼              │                  │ per-asset σ̂, per-regime stats │ escalation enum
┌────────────────────────┐                │                               │ (none/early/Tier1)
│ platform/allocation/     │◄──────────────┘                               ▼
│  hysteresis.py            │                                     python3 -m ...tripwire
│   active_regime state ────┼──► platform checkpoint              (daily manual CLI run)
│   (persisted, load-before- │    "hysteresis_state"
│    save pattern)           │
│  tilt.py                   │
│   vol_targeted_tilt()      │
│   → target weights (≤1.0,  │
│     cash residual)         │
└──────────────┬─────────────┘
               │ target_weights
               ▼
┌────────────────────────────┐        ┌────────────────────────────┐
│ platform/report/holdings.py │        │ incumbent email.py          │
│  load_account_weights()      │        │  (READ-ONLY reuse)           │
│  config/accounts/*.yaml       │───┐    │  build_weekly_email_body()   │
│  (gitignored, weights+cash)   │   │    │  send_weekly_email()         │
└────────────────────────────┘    │    └──────────────┬───────────────┘
                                   │                    │ --send-email (opt-in)
                                   ▼                    ▼
                         ┌───────────────────────────────────────┐
                         │ platform/report/weekly.py               │
                         │  writes outputs/reports/platform/        │
                         │    weekly_report.md  (ALWAYS written)     │
                         │  regime dist + trajectory, per-asset      │
                         │  signals, target-vs-current + trades      │
                         │  implied (BUY/SELL/HOLD Δw per account)   │
                         └───────────────────────────────────────┘
```

### Recommended Project Structure

```
src/trading_crab_lib/platform/
├── assets/                     # L3 — asset prediction (naive)
│   ├── __init__.py
│   ├── returns.py               # monthly returns per asset, returns_by_regime()
│   └── vol.py                   # ewma_vol() daily + monthly, annualization
├── allocation/                  # L4 — allocation
│   ├── __init__.py
│   ├── hysteresis.py            # state machine + persisted active-regime state
│   └── tilt.py                  # vol_targeted_tilt(): regime stats + probs → target weights
├── tripwire/                    # L4-04 — daily monitor
│   ├── __init__.py
│   └── monitor.py                # 3 signals, OR-logic escalation, __main__ CLI
└── report/                      # L4-02/03 — holdings + weekly report
    ├── __init__.py
    ├── holdings.py               # load_account_weights(), schema validation
    └── weekly.py                  # assemble markdown, call incumbent email.py
```
This split is a recommendation (CONTEXT: "Report layout/sections, and module split inside
`platform/` — Claude's Discretion"); the planner may collapse `assets/` and `allocation/` into
fewer files if the total line count stays small — the important constraint is that every new
file lives under `platform/`, per D-01/D-02 conventions already established in Phases 1–3.

### Pattern 1: `gap_lag.py` module shape — the template for every new module

**What:** Pure `compute_*()` functions (no I/O) + one `report_*()`/persist function (writes a
schema-stable parquet artifact + prints a human-readable summary) + a synthetic, no-network
`__main__` self-check block.

**When to use:** Every new file in this phase (`returns.py`, `vol.py`, `hysteresis.py`, `tilt.py`,
`monitor.py`, `weekly.py`) — this is the established pattern across Phases 1–3 (`gap_lag.py`,
`transition_matrix.py`, `diagnostics.py` all follow it).

**Example (adapted for L3-02 EWMA vol):**
```python
# Source: pattern ported from src/trading_crab_lib/platform/honesty/gap_lag.py
from __future__ import annotations
import numpy as np
import pandas as pd

def ewma_vol(returns: pd.Series, *, halflife: float, annualization_factor: float) -> pd.Series:
    """Annualized EWMA volatility of a return series.

    halflife is in the SAME units as returns' index frequency (days for
    daily returns, months for monthly returns) — see RiskMetrics §-style
    convention notes in RESEARCH.md.
    """
    return returns.ewm(halflife=halflife, min_periods=2).std() * np.sqrt(annualization_factor)

if __name__ == "__main__":
    # Synthetic self-check — no network, mirrors gap_lag.py's __main__ footer.
    rng = np.random.default_rng(42)
    synthetic_daily_returns = pd.Series(rng.normal(0, 0.01, 500))
    vol = ewma_vol(synthetic_daily_returns, halflife=11.2, annualization_factor=252)
    print(f"self-check: annualized EWMA vol = {vol.iloc[-1]:.2%}")  # noqa: T201
```

### Pattern 2: Load-before-save state persistence (hysteresis)

**What:** Read the previous state checkpoint BEFORE writing the new one — the exact
`_churn_against_previous()` pattern already proven in `labeling/diagnostics.py` (Pitfall 3 there).

**When to use:** `platform/allocation/hysteresis.py` — the hysteresis state machine needs the
prior "active regime" to decide whether this run acts, unwinds, or holds.

**Example:**
```python
# Source: pattern ported from src/trading_crab_lib/platform/labeling/diagnostics.py
#         (_churn_against_previous / _get_checkpoint_manager)
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager

def load_active_regime(cm=None) -> int | None:
    cm = cm or get_platform_checkpoint_manager()
    try:
        state = cm.load("hysteresis_state")
    except FileNotFoundError:
        return None  # cold start — no prior acted-on regime
    return int(state["active_regime"].iloc[0])

def save_active_regime(active_regime: int, cm=None) -> None:
    import pandas as pd
    cm = cm or get_platform_checkpoint_manager()
    cm.save(pd.DataFrame([{"active_regime": active_regime}]), "hysteresis_state")
```

### Pattern 3: Reusing `email.py` without modification (L4-02)

**What:** `build_weekly_email_body(report_dir, subject_prefix)` reads
`report_dir/weekly_report.md` verbatim as the email body if it exists — no markdown parsing is
required from the platform side; `send_weekly_email(config, subject, body, plot_paths=None)`
handles multipart/alternative or multipart/related delivery.

**When to use:** `platform/report/weekly.py`'s CLI entry point, gated behind `--send-email`
(mirroring the incumbent's flag).

**Example (verified signatures from `src/trading_crab_lib/email.py`):**
```python
# Source: src/trading_crab_lib/email.py (read-only reuse, D-02)
from pathlib import Path
from trading_crab_lib.email import build_weekly_email_body, load_email_config, send_weekly_email

platform_report_dir = Path("outputs/reports/platform")
# weekly.py has already written platform_report_dir / "weekly_report.md"

subject, body = build_weekly_email_body(
    platform_report_dir, subject_prefix="Trading-Crab Platform Weekly Report"
)
if args.send_email:
    cfg = load_email_config()
    if cfg:
        send_weekly_email(cfg, subject, body)  # plot_paths=None in v1 — no plots yet
```

### Pattern 4: Returns-by-regime table construction (L3-01)

**What:** `monthly_raw` (Phase 1 checkpoint) already has every column needed — the 5 core research
series (`equities_tr`, `long_duration_tr`, `gold`, `oil`, `cash`) at month-end plus every
satellite/holdings ticker at its month-end close (via `prices_daily.to_monthly_spine()`, merged in
`build_monthly_spine()`). `regime_labels` (Phase 3 checkpoint) is a `state` column on the same
month-end index. No new ingestion is needed for L3-01 — only alignment and stats computation.

**Example:**
```python
# Source: pattern adapted from incumbent src/trading_crab_lib/asset_returns.py
#         (returns_full_stats) — inspiration only, NOT imported (CONTEXT constraint)
import numpy as np
import pandas as pd

def compute_monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """pct_change on an already-monthly price/level frame — NULL-tolerant
    (D11): assets with shorter history simply produce fewer non-NaN returns,
    never dropped columns or rows."""
    return monthly_prices.pct_change()

def returns_by_regime_stats(returns: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
    """One row per (regime, asset): mean, std, ann. Sharpe, hit_rate, max_dd, n_obs.

    n_obs surfaces D11 short-history assets explicitly rather than hiding
    them — the report should flag any (regime, asset) cell with n_obs < 6
    as low-confidence rather than suppress it.
    """
    common = returns.index.intersection(states.index)
    joined = returns.loc[common].copy()
    joined["regime"] = states.loc[common]

    rows = []
    for regime, group in joined.groupby("regime"):
        asset_data = group.drop(columns=["regime"])
        for ticker in asset_data.columns:
            col = asset_data[ticker].dropna()
            if col.empty:
                continue
            cum = (1 + col).cumprod()
            drawdown = (cum / cum.cummax() - 1).min()
            rows.append({
                "regime": regime, "asset": ticker,
                "mean_monthly_return": col.mean(),
                "std_monthly_return": col.std(),
                "sharpe_annualized": (col.mean() / col.std()) * np.sqrt(12) if col.std() > 0 else np.nan,
                "hit_rate": (col > 0).mean(),
                "max_drawdown": drawdown,
                "n_obs": len(col),
            })
    return pd.DataFrame(rows)
```

### Pattern 5: Naive portfolio vol without a covariance matrix (L4-01)

**What:** Vol-target the CANDIDATE tilt's own historical weighted-return series directly —
`portfolio_returns = (weights * returns).sum(axis=1)`, then EWMA-vol that single series. This
implicitly captures diversification (via the realized covariance structure baked into historical
co-movement) without ever estimating an explicit covariance matrix — the honest v1 answer, and the
natural extension of Pattern 1's `ewma_vol()` (same function, applied to one blended series instead
of N per-asset series).

**Fallback (documented ceiling, not the default):** when a candidate tilt includes an asset with
too few overlapping non-NaN months to compute a stable weighted-return series (D11 short-history
case), fall back to the conservative no-diversification sum `σ̂_port ≈ Σ w_i · σ̂_i` (linear sum
of per-asset EWMA vols) — this systematically OVER-estimates true portfolio vol (ignores
diversification benefit), so it under-levers the tilt rather than over-levers it. Safe-by-construction
given the "defensive by design" D-03 framing.

```python
# ponytail: linear-sum-of-vols fallback deliberately ignores diversification —
# it can only make the tilt MORE conservative than the true portfolio vol, never
# less. Upgrade path: regime-conditional Ledoit-Wolf covariance (L3-V2-01, v2).
def portfolio_vol(weights: pd.Series, asset_returns: pd.DataFrame, *, halflife: float,
                   min_obs: int = 12) -> float:
    common_assets = weights.index.intersection(asset_returns.columns)
    aligned = asset_returns[common_assets].dropna()
    if len(aligned) >= min_obs:
        port_returns = (aligned * weights[common_assets]).sum(axis=1)
        return float(ewma_vol(port_returns, halflife=halflife, annualization_factor=12).iloc[-1])
    # fallback: not enough joint history — conservative linear sum
    per_asset_vol = asset_returns[common_assets].apply(
        lambda s: ewma_vol(s.dropna(), halflife=halflife, annualization_factor=12).iloc[-1]
    )
    return float((weights[common_assets].abs() * per_asset_vol).sum())
```

### Anti-Patterns to Avoid

- **Reusing incumbent `load_portfolio()` for holdings YAML:** silently normalizes to sum=1 and
  drops the cash concept — D-01 explicitly wants warn-don't-fail tolerance validation, not silent
  normalization. Write a new loader.
- **Building the tripwire's SPY signal from `equities_tr`:** that research series is monthly
  (multpl-sourced), not daily, and represents an index-level total-return construct, not SPY's
  actual tradable price. The tripwire needs SPY's real daily closes — add SPY to the universe
  ticker list and use `daily_raw`, don't repurpose the monthly splice output.
- **Trusting the `daily: true` flag in `platform_settings.yaml`'s `fred_monthly.series` block:**
  it is currently inert — `fetch_fred_monthly()` always resamples every series to month-end
  regardless of this flag (confirmed by reading `macro_monthly.py` — no code branches on it
  anywhere). Do not assume T10Y2Y/T10Y3M/VIXCLS are available at daily granularity from the
  existing checkpoint; they are not, today.
- **Tracking argmax(P) for hysteresis instead of P(active regime):** design §5.3's "act at 0.7,
  unwind at 0.4" is a Schmitt-trigger on the CURRENTLY-HELD regime's own probability, not on
  whichever regime currently has the highest probability — conflating the two breaks the no-flip
  invariant test (a competing regime poking briefly above 0.7 must not switch the tilt while the
  active regime's own P is still ≥0.4).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| EWMA volatility | Custom decay-weighted loop | `pd.Series.ewm(halflife=...).std()` | Verified stdlib-adjacent API (pandas, already a dependency); exact RiskMetrics-equivalent math via `halflife` param, no manual alpha derivation needed |
| Markdown→HTML for email | New markdown parser | Incumbent `email.py::_markdown_to_html()` (already exists, stdlib-only, XSS-safe via `html.escape()`) — called internally by `send_weekly_email()`/`_build_html_body_with_plots()` when `plot_paths` is passed | Already built, tested (D42), reused read-only |
| SMTP delivery + TLS/SSL | New sender | Incumbent `email.py::send_weekly_email()` | D-02 explicit constraint: reuse, do not fork |
| Hysteresis state file I/O | Hand-rolled JSON read/write | `CheckpointManager.save()/.load()` (parquet) via `get_platform_checkpoint_manager()` | Gets corrupt-file WARNING logging, freshness checks, and consistent format for free (D-01 checkpoint pattern) |
| Config schema validation | Ad-hoc `if key not in cfg` checks scattered in code | Collect-all-errors-then-raise-once pattern already established in `platform/config.py::validate_platform_config()` and `platform/taxonomy.py::validate_taxonomy()` | Consistent error UX; matches every other platform config validator |

**Key insight:** every "hard problem" this phase touches (EWMA math, state persistence, email
delivery, config validation) already has a working, tested implementation either in `pandas` or
elsewhere in this exact codebase. The actual net-new code is glue: alignment, the hysteresis
switching logic, and the tripwire's OR-logic escalation table — none of which benefit from a
library.

## Common Pitfalls

### Pitfall 1: SPY has no daily price series in the platform namespace (blocks L4-04 criterion 3)

**What goes wrong:** The tripwire's "SPY drawdown-from-peak" signal (D-04, design §23.2) has
nothing to compute against. `universe.satellites`/`holdings`/`watchlist` in
`config/platform_settings.yaml` do NOT include SPY — the "equities" asset class is represented
only by the monthly, multpl-derived `equities_tr` splice research series (`splice.py`), which is
neither daily nor SPY's actual tradable price.

**Why it happens:** D-03 ("model the index, trade the ETF") deliberately keeps the research series
(index-level, long-history) separate from the tradable ETF proxy. Nothing in Phase 1 needed SPY's
actual price — the tradable mapping (`splice.equities.tradable: SPY`) was documentation-only until
now.

**How to avoid:** Add `SPY` to `config/platform_settings.yaml`'s `universe.satellites` (or a new
`universe.tripwire_tickers` list, planner's call) so `prices_daily.fetch_universe_prices()`
ingests it into the existing `daily_raw` checkpoint via the already-built yfinance path — no new
ingestion module needed, only a one-line config addition plus documentation in
`docs/splicing_rules.md` noting SPY is now dual-purpose (tradable proxy AND tripwire input).

**Warning signs:** `daily_raw` checkpoint (once real data exists) has no `SPY` column;
`KeyError: 'SPY'` when the tripwire module tries to compute drawdown-from-peak.

### Pitfall 2: FRED's BAA/AAA series are monthly, not daily — blocks credit-spread-velocity tripwire signal

**What goes wrong:** `config/platform_settings.yaml`'s `fred_monthly.series` configures `BAA` and
`AAA` (Moody's Seasoned Corporate Bond Yields). These are confirmed monthly-frequency series on
FRED [CITED: fred.stlouisfed.org/series/BAA, fred.stlouisfed.org/series/AAA]. A genuinely daily
credit-spread-velocity signal (D-04 signal 2) needs the daily counterparts, `DAAA`/`DBAA`
[CITED: fred.stlouisfed.org/series/DBAA — "available from 1986-01-02 onwards, daily frequency"].
Additionally, `macro_monthly.py::fetch_fred_monthly()` unconditionally resamples every configured
series to month-end regardless of the `daily: true` flag already present (decoratively) on
`T10Y3M`/`T10Y2Y`/`VIXCLS` — that flag is not read anywhere in the code today.

**Why it happens:** Phase 1's monthly data layer never needed daily FRED granularity; the `daily:
true` flag was added as forward-looking documentation (per the config comment) but the
corresponding fetch-path branch was never implemented.

**How to avoid:** Two options for the planner, in order of preference:
1. Add `DAAA`/`DBAA` to config and write a small new daily-FRED fetch helper (mirrors
   `_fetch_fred_monthly()` but resamples nothing — same pattern `prices_daily.py` uses for equity
   prices) — persist as part of (or alongside) `daily_raw`.
2. If daily credit data is deferred, document explicitly in the plan that the tripwire's credit
   signal runs on a stale (month-end) BAA-AAA spread refreshed only once a month — this weakens
   D-04's "daily cadence" framing and should be flagged to the user as a scope reduction, not
   silently shipped.
Recommend option 1 — it's a small, additive, non-breaking config + one new ~30-line function.

**Warning signs:** Credit-spread-velocity signal recomputes the identical value every day within a
month (because the underlying BAA/AAA data hasn't actually refreshed).

### Pitfall 3: Hysteresis semantics — track P(active regime), not argmax(P)

**What goes wrong:** A naive implementation switches the active regime whenever some OTHER
regime's probability crosses 0.70, even while the currently-active regime's own probability is
still comfortably above the 0.40 unwind floor. This flips the tilt on noise and fails the
headline no-flip invariant test (CONTEXT: "a probability path oscillating 0.65↔0.72 must NOT flip
the mix each month").

**Why it happens:** Design §5.3's phrasing ("act when P crosses ~0.7; unwind below ~0.4") reads
ambiguously — it's easy to implement as "switch to whichever regime just crossed 0.7" instead of
"only unwind the CURRENT regime once its own probability collapses below 0.4, and only then
consider switching to a new regime that has itself crossed 0.7."

**How to avoid:** Implement exactly this state machine (confirmed against CONTEXT's own phrasing,
"keep prior tilt until its P<0.40"):
1. Cold start (no persisted state): `active = argmax(P)` if `max(P) >= act_threshold (0.7)`, else
   no position (cash-heavy neutral posture) until some regime first crosses 0.7. **[ASSUMED — not
   explicitly specified in design or CONTEXT; document this cold-start rule in the report and
   flag for confirmation.]**
2. Each run: if `P[active] >= unwind_threshold (0.4)` → hold, `active` unchanged, regardless of
   any other regime's probability.
3. If `P[active] < unwind_threshold` → active regime's confidence has collapsed. If some other
   regime `k` has `P[k] >= act_threshold (0.7)`, switch `active = k`. Otherwise, unwind to the
   neutral/cash-heavy posture (no confident regime) until one crosses 0.7.

**Warning signs:** the tilt changes on months where the persisted `active_regime` probability
never actually dropped below 0.40.

### Pitfall 4: `monthly_raw`/`daily_raw` are not currently holdout-split — full history is visible

**What goes wrong:** `write_monthly_features_split()` (the HON-01 holdout carve mechanism) exists
and is tested (`test_platform_holdout.py`) but is **not called anywhere in production code** —
`build_monthly_spine()` writes `daily_raw`/`monthly_raw`/`monthly_features` directly via
`get_platform_checkpoint_manager().save(...)`, with no holdout split. This means, as of today,
loading these checkpoints returns the FULL history including 2021+.

**Why it happens:** Phase 1/3 built the holdout mechanism as generic infrastructure but never
wired it into the actual pipeline orchestrator (no `main()`/CLI entry point calls
`write_monthly_features_split` yet — confirmed by grep; STATE.md also notes the live 1962+ data
run is still pending).

**How to avoid:** CONTEXT already resolves the policy question for THIS phase: returns-by-regime
tables are "historical conditioning, not supervised training — no CV needed in v1," so reading the
full-history checkpoint is acceptable for L3-01/L3-02/L4-01. However:
- If the planner's tasks tune any threshold (target_vol, hysteresis bands, tripwire thresholds)
  against a computed metric using post-2020 data, that IS an evaluated configuration and must be
  logged via `platform.honesty.registry.append_trial()` — v1 should NOT do this (thresholds are
  literature-derived config defaults, not tuned), but flag this explicitly as a rule for any
  future iteration.
- The planner should NOT wire `write_monthly_features_split` into Phase 4's own new checkpoints
  (returns-by-regime table, vol series) — those are derived analytics artifacts, not raw features,
  and are out of scope for the holdout carve per CONTEXT's framing. Wiring the underlying Phase-1
  raw-checkpoint split is a Phase-1/ops backlog item, not this phase's job.

**Warning signs:** none observable yet in this repo (no real data has been ingested — `data/checkpoints/platform/` does not exist on disk in this environment). This is a documentation/planning-time pitfall, not a runtime bug to chase today.

## Code Examples

### Vol-targeted position scale (L4-01, design §6.2/§7)

```python
# Source: design §6.2 "position size ∝ 1/σ̂" + §7 "shrink return forecasts brutally,
# never feed raw point forecasts to an optimizer" — combined into the naive v1 scale rule.
def vol_target_scale(target_vol_annual: float, portfolio_vol_annual: float) -> float:
    """Long-only, no-leverage scale factor. Cash absorbs 1 - scale."""
    if portfolio_vol_annual <= 0:
        return 0.0  # no signal — degrade to all-cash rather than divide by zero
    return min(1.0, target_vol_annual / portfolio_vol_annual)
```

### Tripwire OR-logic escalation (L4-04, design §23.2)

```python
# Source: design §23.2 "Composite trigger (OR-logic across independent detectors) ...
# Tiered response: Tier 1 (elevated risk) ... Tier 2 (crisis confirmed)" — v1 minimal
# 3-signal subset per D-04 (2-tier escalation: none / early-scoring / Tier-1).
from enum import Enum

class TripwireEscalation(str, Enum):
    NONE = "none"
    RUN_WEEKLY_SCORING_EARLY = "run weekly scoring early"
    TIER1_DERISK_REVIEW = "Tier-1 de-risk review"

def escalate(vol_spike: bool, credit_velocity: bool, spy_drawdown: bool) -> TripwireEscalation:
    n_triggered = sum([vol_spike, credit_velocity, spy_drawdown])
    if n_triggered >= 2:
        return TripwireEscalation.TIER1_DERISK_REVIEW
    if n_triggered == 1:
        return TripwireEscalation.RUN_WEEKLY_SCORING_EARLY
    return TripwireEscalation.NONE
```

### Trades-implied per account (L4-02, report layer — inspiration from incumbent, not imported)

```python
# Source: pattern adapted from incumbent src/trading_crab_lib/reporting.py
#         (generate_recommendation) — inspiration only, do NOT import into platform/ (D-01).
def trades_implied(target_weights: "pd.Series", current_weights: "pd.Series",
                    threshold: float = 0.03) -> "pd.DataFrame":
    """One row per asset: current_pct, target_pct, delta_pct, signal (BUY/SELL/HOLD).
    threshold is the no-trade band — deltas smaller than this are HOLD, avoiding
    noise-chasing turnover (design §21 no-trade bands, simplified for v1 as a flat
    threshold rather than the full band-width optimization, which is L4-V2-01)."""
    ...  # same shape as reporting.py::generate_recommendation, new implementation
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Incumbent `returns_by_regime()` — median only, quarterly cadence | Platform `returns_by_regime_stats()` — mean/std/Sharpe/hit-rate/max-DD, monthly cadence, N-obs surfaced | This phase (net-new, not a replacement — incumbent stays frozen) | Richer stats support the vol-targeting math (need std, not just median); monthly cadence matches D10 |
| Incumbent `load_portfolio()` — silent normalize, no cash | Platform `load_account_weights()` — warn-don't-fail tolerance, explicit cash key, per-account files | This phase | Matches D-01's explicit tolerance-warn requirement and multi-account structure |

**Deprecated/outdated:** N/A — nothing in this phase deprecates existing incumbent code (frozen,
untouched per constraints).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Cold-start hysteresis rule (`active = argmax(P)` if `max(P) >= 0.7`, else no position) is not explicitly specified in design §5.3 or CONTEXT — this is a reasonable inference, not a sourced decision | Pitfall 3 / Pattern 2 | If the user expects a different cold-start behavior (e.g. always start in cash regardless of probability), the first weekly report would show an unexpected initial tilt. Low risk — easily corrected once observed, and the report should document the rule explicitly so it's visible. |
| A2 | Recommended EWMA half-lives — daily tripwire vol: RiskMetrics λ=0.94 ⇒ halflife≈11.2 trading days [CITED via WebSearch, RiskMetrics 1994 convention]; monthly sizing vol: no single sourced default found — recommend a shorter halflife than RiskMetrics' λ=0.97-monthly (≈22.8-month halflife, likely too slow for a monthly-rebalance tilt) — e.g. 6–12 months, config-overridable | §"EWMA Vol Conventions" (via Pattern 1/5) | If the chosen monthly half-life is too long, the vol-targeted tilt reacts sluggishly to genuine regime-driven vol changes; too short, and it whipsaws. This is explicitly flagged as Claude's Discretion in CONTEXT — the planner should expose it as a config default with a one-line documented rationale, not treat it as settled. |
| A3 | Credit-spread-velocity and drawdown thresholds (e.g. "25bp widening over 5 trading days," "-10% drawdown-from-peak") have no literature-sourced numeric anchor in the design doc — CONTEXT explicitly delegates these to Claude's Discretion | Tripwire section | Wrong thresholds either over-trigger (alert fatigue, Glenn ignores the tripwire) or under-trigger (misses the event it exists to catch). Phase 5's honest backtest is the actual validation mechanism — v1 thresholds are necessarily provisional and should be labeled as such in the report/config comments. |
| A4 | The exact module split (`platform/assets/`, `platform/allocation/`, `platform/tripwire/`, `platform/report/`) is a recommendation, not a locked decision — CONTEXT explicitly reserves this as Claude's Discretion | Architecture Patterns | None — purely organizational; any consistent split under `platform/` satisfies the constraints. |

## Open Questions

1. **Should SPY ingestion (Pitfall 1) and DAAA/DBAA ingestion (Pitfall 2) be separate small tasks
   at the start of this phase's plan, or treated as a mid-phase blocking dependency?**
   - What we know: both are small, additive, non-breaking config + fetch-function changes; neither
     touches frozen incumbent code; both are prerequisites for L4-04's daily tripwire criterion.
   - What's unclear: whether the planner sequences them as Wave 0 (before returns-by-regime/vol
     work, since they touch the same `platform_settings.yaml`/ingestion layer) or defers them
     until the tripwire module specifically needs them.
   - Recommendation: Wave 0 — both changes are cheap, low-risk, and unblock the tripwire's daily
     cadence claim; doing them early avoids a late-phase surprise when `daily_raw` turns out not
     to have what's needed.

2. **Does the no-trade-band threshold in `trades_implied()` (used for the report's BUY/SELL/HOLD
   signal, threshold=3% in the incumbent's `generate_recommendation()`) need its own config key,
   or should it reuse `allocation.target_vol_annual`-adjacent config?**
   - What we know: design §21 (no-trade band mathematics) is explicitly deferred to v2
     (L4-V2-01); v1 just needs SOME flat threshold to avoid reporting a "BUY 0.4%" noise trade.
   - What's unclear: the exact default value and whether it belongs under `allocation:` or
     `report:` in `platform_settings.yaml`.
   - Recommendation: reuse the incumbent's 3% default as a documented starting point under
     `report.trade_threshold_pct: 0.03`, config-overridable, explicitly not the full no-trade-band
     math from §21.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| pandas | EWMA vol, returns tables | Project dependency, already installed | ≥2.0 pinned | — |
| FRED_API_KEY | New DAAA/DBAA daily fetch (Pitfall 2) | ✗ in this environment (per STATE.md: "Live 1962+ data run still pending FRED_API_KEY") | — | All new modules must ship with synthetic-frame `__main__` self-checks (established pattern) so they are provable without live data; live wiring is a human-verification item, same as Phase 1 |
| yfinance | New SPY daily fetch (Pitfall 1) | Project dependency (already used by `prices_daily.py`), network access needed for live data | ≥0.2 pinned | Synthetic-frame tests cover the logic; live fetch is best-effort with graceful degradation (existing `prices_daily.py` pattern already handles total-failure → empty frame) |
| SMTP config (`config/email.local.yaml` or `TC_*` env vars) | `--send-email` opt-in path only (L4-02) | Not configured in this environment | — | Default path (`weekly_report.md` always written) works with zero SMTP config — email is explicitly opt-in per D-02 |

**Missing dependencies with no fallback:** none — every gap above has a documented fallback
(synthetic self-checks prove logic; live data wiring is deferred to a human-verification step,
consistent with Phase 1's precedent).

**Missing dependencies with fallback:** FRED_API_KEY, live network access — synthetic no-network
tests are the established mitigation pattern across this entire `platform/` codebase.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest ≥8.0 (already configured, `pyproject.toml`) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` (existing) |
| Quick run command | `pytest tests/unit/test_platform_<module>.py -x` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| L3-01 | `returns_by_regime_stats()` produces correct mean/std/Sharpe/hit-rate/max-DD per (regime, asset), NULL-tolerant for short-history assets | unit | `pytest tests/unit/test_platform_returns.py -x` | ❌ Wave 0 |
| L3-02 | `ewma_vol()` matches hand-computed RiskMetrics-style EWMA on a synthetic series; annualization factor correct for daily vs monthly | unit | `pytest tests/unit/test_platform_vol.py -x` | ❌ Wave 0 |
| L4-01 | Hysteresis no-flip invariant: a P(active) path oscillating 0.65↔0.72 never switches `active_regime`; a path crossing below 0.40 does switch once a competitor crosses 0.70 | unit (headline invariant) | `pytest tests/unit/test_platform_hysteresis.py -x` | ❌ Wave 0 |
| L4-01 | `vol_target_scale()` caps at 1.0 (no leverage), degrades to 0.0 on zero/negative portfolio vol | unit | `pytest tests/unit/test_platform_tilt.py -x` | ❌ Wave 0 |
| L4-02 | `weekly.py` writes markdown to `outputs/reports/platform/weekly_report.md` unconditionally; `build_weekly_email_body()`/`send_weekly_email()` only invoked when `--send-email` passed | unit + smoke | `pytest tests/unit/test_platform_report_weekly.py -x` | ❌ Wave 0 |
| L4-03 | Holdings YAML: weights + cash summing to ~1.0 passes silently; sum deviating beyond tolerance WARNs but does not raise; missing file returns empty/neutral state, not a crash | unit | `pytest tests/unit/test_platform_holdings.py -x` | ❌ Wave 0 |
| L4-04 | `escalate()` OR-logic: 0 triggers → NONE, 1 → early-scoring, 2+ → Tier-1, regardless of WHICH signals fired (family-independence, not signal-identity) | unit (headline invariant) | `pytest tests/unit/test_platform_tripwire.py -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** targeted `pytest tests/unit/test_platform_<module>.py -x`
- **Per wave merge:** `pytest tests/unit/ -k platform -v`
- **Phase gate:** `pytest tests/ -v` full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_platform_returns.py` — covers L3-01
- [ ] `tests/unit/test_platform_vol.py` — covers L3-02
- [ ] `tests/unit/test_platform_hysteresis.py` — covers L4-01 (headline no-flip invariant)
- [ ] `tests/unit/test_platform_tilt.py` — covers L4-01 (vol-target scale math)
- [ ] `tests/unit/test_platform_report_weekly.py` — covers L4-02
- [ ] `tests/unit/test_platform_holdings.py` — covers L4-03
- [ ] `tests/unit/test_platform_tripwire.py` — covers L4-04 (headline OR-logic invariant)
- [ ] Framework install: none — pytest already installed and configured

## Security Domain

### Applicable ASVS Categories

This phase is a local batch/CLI Python tool with no network-facing authentication surface —
most ASVS categories do not apply. The relevant subset:

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No auth surface — local CLI, SMTP credentials already handled by incumbent `email.py`/`TC_*` env vars |
| V3 Session Management | No | No sessions |
| V4 Access Control | No | Single-user local tool |
| V5 Input Validation | Yes | Holdings YAML (`config/accounts/*.yaml`): `yaml.safe_load()` (never `yaml.load()` — already the project convention throughout `config.py`/`email.py`); numeric coercion with `try/except (TypeError, ValueError)` guard (mirrors incumbent `load_portfolio()`); tolerance-checked sum validation (warn, don't crash on malformed input) |
| V6 Cryptography | No | No crypto in this phase; SMTP TLS/SSL already handled by incumbent, unmodified |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed/malicious holdings YAML (e.g. a non-numeric weight, or a `!!python/object` YAML tag) | Tampering | `yaml.safe_load()` only (never `yaml.load()`); numeric coercion wrapped in explicit `try/except`, matching `load_portfolio()`'s existing pattern |
| Holdings files accidentally committed with real account data | Information Disclosure | D-01 already specifies `config/accounts/*.yaml` gitignored (mirroring `*.local.yaml`); committed `example.yaml` must use obviously-fake tickers/weights, never Glenn's real holdings |
| Config-driven thresholds silently misapplied (e.g. tripwire never actually escalates because a threshold is inverted) | — (not STRIDE, a correctness risk not a security one) | Covered by the headline invariant tests above, not a security control — noted here only because a "tripwire that never fires" has real-money consequences even though it isn't a classic security threat |

## Sources

### Primary (HIGH confidence — read directly from this codebase)
- `src/trading_crab_lib/platform/labeling/diagnostics.py` — `label_regimes()`, checkpoint names
  (`regime_labels`, `regime_confidences`, `regime_profiles`), load-before-save pattern
- `src/trading_crab_lib/platform/prediction/nowcaster.py` — `fit_nowcaster`, `predict_proba`
  usage, embargo semantics
- `src/trading_crab_lib/platform/prediction/transition_matrix.py` — `empirical_transition_matrix()`
- `src/trading_crab_lib/platform/checkpoints.py`, `config.py`, `taxonomy.py` — persistence/config
  conventions
- `src/trading_crab_lib/platform/ingestion/prices_daily.py`, `splice.py` — daily/monthly universe
  price ingestion, core research series construction, `universe.satellites`/`holdings` config
- `src/trading_crab_lib/platform/transforms_monthly.py` — `build_monthly_spine()` orchestrator;
  confirmed `daily_raw`/`monthly_raw`/`monthly_features` checkpoint contents and that holdout
  split is NOT wired in
- `src/trading_crab_lib/platform/ingestion/macro_monthly.py` — confirmed `daily: true` config flag
  is currently unused/inert in `fetch_fred_monthly()`
- `src/trading_crab_lib/platform/honesty/{gap_lag,holdout,registry,gating}.py` — module-shape
  pattern, holdout carve mechanics, registry API, causal-gating guard
- `src/trading_crab_lib/email.py` — verified `build_weekly_email_body()`, `send_weekly_email()`,
  `load_email_config()`, `_markdown_to_html()` signatures
- `src/trading_crab_lib/asset_returns.py`, `reporting.py` — incumbent inspiration-only patterns
  (`returns_full_stats`, `blended_regime_portfolio`, `generate_recommendation`)
- `src/trading_crab_lib/config.py::load_portfolio()` — confirmed incumbent silent-normalize
  behavior, why it should NOT be reused
- `config/platform_settings.yaml` — universe/splice/taxonomy/holdout/labeling config, confirmed
  SPY absent from `universe.satellites`
- `platform_design/platform_design.md` §5.3, §6.2, §6.5, §7, §14, §16.7, §23.2, §25, D11 (design
  decisions table) — hysteresis, vol-targeting doctrine, weekly report output list, tracer-bullet
  philosophy, tactical resolution, tripwire spec
- `docs/splicing_rules.md` — confirmed core research series vs tradable-ticker distinction
- `.planning/phases/04-asset-prediction-allocation/04-CONTEXT.md`,
  `.planning/REQUIREMENTS.md`, `.planning/STATE.md` — phase scope, requirements, project history
- Local pandas installation (`pandas.core.generic.ewm` docstring, inspected via Python) — confirms
  `halflife` parameter and its `alpha = 1 - exp(-ln(2)/halflife)` formula

### Secondary (MEDIUM confidence — WebSearch, cross-checked against official source pages)
- RiskMetrics λ=0.94 daily / λ=0.97 monthly EWMA volatility convention, halflife≈11.2 trading days
  at λ=0.94 [CITED: multiple sources converging on the 1994 JPMorgan RiskMetrics technical
  document convention — see ryanoconnellfinance.com, researchgate.net summaries]
- FRED series `BAA`/`AAA` = monthly frequency; `DAAA`/`DBAA` = daily frequency, available from
  1986-01-02 [CITED: fred.stlouisfed.org/series/AAA, fred.stlouisfed.org/series/DBAA — official
  FRED series pages]

### Tertiary (LOW confidence — flagged for validation)
- None — no unverified LOW-confidence claims are treated as fact in this document; all such items
  are listed in the Assumptions Log above with explicit `[ASSUMED]` framing.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — zero new dependencies; `pandas.ewm()` API verified locally against the
  installed version's docstring
- Architecture: HIGH — every integration point (checkpoint names, function signatures, config
  keys) verified by reading the actual source files, not inferred from documentation alone
- Pitfalls: HIGH for the two data-gap pitfalls (SPY, DAAA/DBAA) — verified by tracing actual code
  paths (grep confirms no code branches on the `daily:` flag; grep confirms SPY absent from
  universe config); MEDIUM for the numeric threshold recommendations (tripwire bp/drawdown
  thresholds, EWMA half-lives) — explicitly flagged as Claude's Discretion in CONTEXT and logged
  in the Assumptions table, not presented as settled fact

**Research date:** 2026-07-22
**Valid until:** 30 days (stable codebase/design doc; the FRED series-frequency facts are
effectively permanent, but re-verify if `config/platform_settings.yaml` or `transforms_monthly.py`
change before this phase is planned)
