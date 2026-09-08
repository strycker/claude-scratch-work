# Offline Dev Snapshots — Platform Checkpoints

`src/trading_crab_lib/platform/snapshots.py` + `scripts/platform_snapshot.py`
(Task 3, quick task `260806-u89`).

## What a snapshot is — and is not

A **snapshot** is a frozen copy of one or more platform checkpoints
(`daily_raw`, `monthly_raw`, `monthly_features` by default), captured at a
stated moment and committed to the repo. It exists so a developer with no
network access can restore a working platform checkpoint set and run the
report/backtest machinery offline.

A snapshot is **not** live data, and is never treated as a substitute for it:

- Loading a snapshot is **opt-in only** — nothing in this library reads one
  unless something explicitly asks it to. There is no automatic fallback
  from a failed live fetch to a snapshot anywhere in the codebase.
- Every read of a snapshot (`load_snapshot()`) logs a WARNING carrying the
  capture timestamp.
- A checkpoint restored from a snapshot (`restore_snapshot()`) carries a
  persistent `"source": "offline-snapshot"` marker in its meta json
  (`is_snapshot_backed()`), so it can never be silently mistaken for a
  live-fetched checkpoint even long after the restore happened.
  `scripts/build_platform_data.py` checks this marker right after
  constructing its checkpoint manager and logs a WARNING for any checkpoint
  that is still snapshot-backed once its build attempt has finished.

## Committed location

`data/snapshots/platform/` — a narrow, deliberate exception to `.gitignore`
(only this subtree; nothing else under `data/` is un-ignored by it). This is
the *only* location the export/restore CLI defaults to, and it is
repo-relative (`trading_crab_lib.ROOT`-based, honors `TC_ROOT_DIR`) so the
files are committable regardless of where `DATA_DIR` has been overridden to.

No snapshot parquet is committed by Task 3 itself — only
`data/snapshots/platform/.gitkeep` exists to reserve the location. The
`daily_raw` checkpoint present in this container at the time Task 3 was
written is a destroyed 0×0 frame (the exact bug Task 1 now prevents);
capturing it as the first snapshot would enshrine that bug into git history.
**The first real snapshot should be captured by running `export` from a
machine that has just completed a successful `build_platform_data.py` run.**

## Commands

```bash
# Export the default checkpoint set (daily_raw, monthly_raw, monthly_features)
# to data/snapshots/platform/
python scripts/platform_snapshot.py export

# Export a narrower/custom set, or to a different location
python scripts/platform_snapshot.py export --names daily_raw monthly_raw \
    --snapshot-dir /tmp/my-snapshot --compression snappy

# Bound the daily_raw frame's size by truncating its lower date (the size lever)
python scripts/platform_snapshot.py export --daily-start 2015-01-01

# Restore a snapshot into the live platform checkpoint namespace
# (data/checkpoints/platform/) — prints a banner naming what was restored
python scripts/platform_snapshot.py restore

# Show the manifest without restoring anything
python scripts/platform_snapshot.py list
```

Programmatic equivalents live in `trading_crab_lib.platform.snapshots`:
`export_snapshot()`, `load_snapshot()`, `restore_snapshot()`,
`is_snapshot_backed()`, `snapshots_enabled()`, `read_manifest()`.

## Manifest fields

`manifest.json` (written alongside the per-checkpoint parquet files) carries:

- `captured_at` — UTC ISO timestamp of the export.
- `git_commit` — best-effort `git rev-parse HEAD` at export time (`null` if
  git or the repo is unavailable; a missing git must never break an export).
- `names` — per-checkpoint `rows`, `columns`, `col_names`, `index_start`,
  `index_end`.
- `source_provenance` — the contents of `splice_provenance.json` (Task 2),
  when present in the checkpoint directory at export time, so a restored
  snapshot also documents which splice-chain candidate resolved for each
  research series (e.g. whether `gold` came from `gold_spot` or fell back
  to `IAU` — see `docs/splicing_rules.md` §3 for the IAU-is-not-spot-gold
  caveat).

## Opt-in loading

`snapshots_enabled()` reads `TC_USE_SNAPSHOTS` and treats
`"1"`/`"true"`/`"yes"` (case-insensitive) as enabled; anything else,
including unset, is disabled. This flag is provided for callers that want to
deliberately opt an entire run into snapshot-backed data — no code in this
plan reads it automatically to fall back from a failed fetch.

## Size budget

Measured on this machine (pyarrow present):

- A 14,267 × 22 float64 `daily_raw` frame: ~2.0 MB snappy / ~1.9 MB zstd.
- `monthly_features`: ~113 KB.
- `monthly_raw`: ~72 KB.
- A full default set (`daily_raw` + `monthly_raw` + `monthly_features`)
  lands near **~2.2 MB total**.

`export_snapshot(max_mb=10.0)` (the CLI's `--max-mb` default) refuses to
write a set larger than that budget and names the file that blew it —
nothing is written if the budget is exceeded, so a runaway export can never
leave a partial snapshot behind.

**~2.2 MB was judged committable** for the default set, so it was made
available — but it was **not made the default export target**. Every
refresh of a snapshot **permanently adds roughly that much to git history**
(git does not deduplicate binary blob deltas the way it does for text), so
snapshots should be refreshed rarely and deliberately, not on every build.
Use `--daily-start` (or the `daily_start=` kwarg) to bound `daily_raw` to a
recent window if you want a smaller, cheaper-to-refresh snapshot instead of
the full 1962+ history.

## Provenance and honesty

Because snapshots are frozen and openly marked as such, they cannot silently
contaminate a live build: `build_monthly_spine()`'s own checkpoint writes go
through `CheckpointManager.save()` (Task 1's merge-on-save), which always
writes real data through its normal path — restoring a snapshot only ever
happens via the explicit `restore_snapshot()` call, never as a side effect
of a build.
