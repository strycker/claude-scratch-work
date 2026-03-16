# Codebase Concerns

**Analysis Date:** 2026-03-16

## Tech Debt

**Insecure ETF price fetching (SSL verification disabled):**
- Issue: ETF ingestion permanently disables TLS certificate verification and suppresses related warnings to work around `curl_cffi` CA-store limitations.
- Files: `src/market_regime/ingestion/assets.py`
- Impact: Susceptible to man-in-the-middle attacks and silent data tampering when fetching prices; risk applies even on “trusted” networks (corporate proxies, captive portals, etc.).
- Fix approach: Add a `RunConfig`/config flag to control SSL verification (default secure), and make the insecure path explicit (“I understand the risk”). Prefer a secure solution where possible (system trust / `certifi` / documented CA override) and avoid global warning suppression.

**Pipeline scripts depend on `sys.path` mutation (non-packaged execution path):**
- Issue: Multiple entry points insert `src/` into `sys.path` to allow running without installation.
- Files: `run_pipeline.py`, `pipelines/01_ingest.py` (and other `pipelines/*.py`)
- Impact: Environment-dependent imports, risk of accidentally importing wrong modules when run from other working directories; can complicate tooling/packaging.
- Fix approach: Standardize on `pip install -e .` for development (keep `sys.path` fallback if needed, but gate it behind a clear “dev mode” or document it as a supported pattern).

**Broad exception handling hides data quality issues (partial success treated as OK):**
- Issue: In ingestion and model-selection sweeps, failures are caught broadly and logged, but the pipeline continues with partial data.
- Files: `src/market_regime/ingestion/multpl.py`, `src/market_regime/ingestion/fred.py`, `src/market_regime/ingestion/assets.py`, `src/market_regime/gmm.py`, `src/market_regime/spectral.py`, `src/market_regime/density.py`
- Impact: Silent loss of columns/series can change feature distributions and clustering results; downstream outputs can look “valid” while being materially incomplete.
- Fix approach: Track and surface a structured “ingestion completeness” report (expected vs. fetched series/tickers, % coverage, hard-fail thresholds), and optionally fail fast when critical series are missing.

**Checkpoint metadata listing swallows parse errors:**
- Issue: `CheckpointManager.list()` ignores malformed JSON metadata without surfacing which checkpoint is corrupt.
- Files: `src/market_regime/checkpoints.py`
- Impact: Corrupt checkpoint metadata can silently degrade reproducibility/debugging (stale/incorrect checkpoint selection becomes harder to diagnose).
- Fix approach: Log which `*.meta.json` failed to parse (at least DEBUG/WARN) and/or quarantine invalid metadata files.

**Repository contains committed data artifacts (large, mutable, provenance risk):**
- Issue: Several `data/*.pickle` and snapshot files are present in the repo even though most `data/` subdirectories are ignored.
- Files: `data/fred_api_datasets_snapshot_20260216.pickle`, `data/multpl_datasets_snapshot_20260216.pickle`, `data/prepared_quarterly_data_smoothed_20260301.pickle`, `data/standardized_quarterly_data_20260216.pickle`, `data/grok_quarter_classifications_*.pickle`, `data/grok_quarter_classifications_*.xlsx`
- Impact: Large repo size, unclear provenance/licensing for scraped/derived datasets, drift between code and bundled data; increases chance of accidentally training/evaluating on stale artifacts.
- Fix approach: Move these to a dedicated `data/archives/` that is clearly documented and optionally gitignored, or store only small fixtures under `tests/fixtures/` and keep larger artifacts out of version control.

## Known Bugs

**FRED ingestion hard-fails when `FRED_API_KEY` missing (even though config loader only warns):**
- Symptoms: Step 1 fails with `EnvironmentError("FRED_API_KEY is not set")`.
- Files: `src/market_regime/config.py`, `src/market_regime/ingestion/fred.py`, `pipelines/01_ingest.py`
- Trigger: Running ingestion without a configured `FRED_API_KEY`.
- Workaround: Provide the key via environment or `.env` (file exists and is gitignored); otherwise skip FRED ingestion externally.

## Security Considerations

**Untrusted pickle loading (arbitrary code execution risk):**
- Risk: `pickle` and `pd.read_pickle()` can execute arbitrary code when loading attacker-controlled files.
- Files: `src/market_regime/checkpoints.py` (model pickles), `src/market_regime/ingestion/grok.py` (grok pickle), `src/market_regime/cluster_comparison.py` (loads pickled RF models)
- Current mitigation: None (assumes local, trusted files).
- Recommendations: Treat pickles as trusted-only inputs; document this explicitly; prefer safe formats for data interchange (`parquet`, `csv`) and restrict model loading paths (or add signatures/hashes if models are shared).

**TLS verification disabled for external price data:**
- Risk: External data ingestion accepts unverifiable HTTPS responses for yfinance/curl_cffi requests.
- Files: `src/market_regime/ingestion/assets.py`
- Current mitigation: None (warnings suppressed in the insecure path).
- Recommendations: Make insecure network mode opt-in; add prominent logging when running insecurely; prefer secure data sources or verified CA configuration when possible.

## Performance Bottlenecks

**Expensive plotting paths (pairplot/scatter matrix) can be accidentally triggered in notebooks:**
- Problem: Pairplot and scatter-matrix style visualizations are slow and memory-heavy on large datasets.
- Files: `src/market_regime/plotting.py`, `src/market_regime/runtime.py`
- Cause: Pairplot builds \(O(n^2)\) subplot grids; large DataFrames amplify runtime/memory.
- Improvement path: Keep these strictly opt-in (already gated), and consider downsampling/feature caps in plotting helpers to prevent accidental runaway runs.

**Network-bound ingestion without robust retry/backoff:**
- Problem: Scraping and API calls can be slow/flaky; failure handling is mostly “log and continue” without retries.
- Files: `src/market_regime/ingestion/multpl.py`, `src/market_regime/ingestion/fred.py`, `src/market_regime/ingestion/assets.py`
- Cause: External services rate-limit or intermittently fail; only `multpl` has a fixed sleep; other paths rely on best-effort fallbacks.
- Improvement path: Add bounded retries with exponential backoff for idempotent fetches and a consolidated “what failed” report at the end of each ingestion step.

## Fragile Areas

**HTML-scraping dependency on multpl page structure:**
- Files: `src/market_regime/ingestion/multpl.py`
- Why fragile: Relies on `#datatable tr` and specific text formats; small HTML changes can break parsing.
- Safe modification: Keep selectors and parsing isolated; add contract tests using saved HTML fixtures under `tests/fixtures/` (no network).
- Test coverage: Missing explicit scraper contract tests for selector stability and value parsing edge cases.

**Automatic backfill of “canonical” parquet files from checkpoints:**
- Files: `run_pipeline.py` (`_load_parquet`)
- Why fragile: Hidden side effect writes parquet files when the canonical path is missing, which can surprise users and complicate debugging in shared workspaces.
- Safe modification: Make backfill explicit via a flag or log at INFO with full path; avoid writing in read-only contexts.
- Test coverage: Not detected for this specific behavior (side-effect write path).

## Scaling Limits

**Local-filesystem checkpointing only:**
- Current capacity: Single-machine runs with `data/checkpoints/` on local disk.
- Limit: Not suitable for multi-user / multi-machine workflows without explicit artifact syncing; results can diverge due to local cache state.
- Scaling path: Add optional remote artifact store support (e.g., S3/GCS) or provide a first-class “export/import artifacts” command.

## Dependencies at Risk

**`curl_cffi`/yfinance TLS behavior creates operational/security trade-offs:**
- Risk: The “works everywhere” approach is implemented as “disable verification everywhere”.
- Impact: Forces a security compromise for reliability.
- Migration plan: Prefer data providers that support standard TLS verification, or restructure to allow verified requests with user-supplied CA bundles where feasible.

## Missing Critical Features

**No centralized “data quality gate” for ingestion completeness:**
- Problem: Partial ingestion can silently produce plausible outputs.
- Blocks: Confidence in reported regimes/asset signals; hard to compare runs.
- Fix approach: Add a per-step summary artifact (expected vs actual series count, missing series list, coverage %) and enforce minimum completeness thresholds for critical features.

## Test Coverage Gaps

**Ingestion edge cases not fully covered (network failures, partial returns, schema drift):**
- What's not tested: multpl HTML selector drift; FRED series missing/shift logic corner cases; assets ingestion fallback chain behavior (including insecure SSL path) without hitting the network.
- Files: `src/market_regime/ingestion/multpl.py`, `src/market_regime/ingestion/fred.py`, `src/market_regime/ingestion/assets.py`
- Risk: Production runs fail or silently degrade after upstream provider changes; regressions in fallback logic go unnoticed.
- Priority: High

---

*Concerns audit: 2026-03-16*
