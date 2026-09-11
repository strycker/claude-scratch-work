---
phase: quick-260911-la3
plan: 01
subsystem: infra
tags: [pypi, twine, github-actions, release-engineering, packaging]

requires:
  - phase: quick-260910-vyi
    provides: "dist path fix, PUBLISH/SKIP gate logging, Verify built artifacts version guard"
  - phase: quick-260911-kkj
    provides: "token-presence guard step and its extraction/scenario-testing pattern, reused and extended here"
provides:
  - "trading-crab-lib README.md wired into pyproject.toml readme key; twine check --strict now PASSES clean instead of FAILED with two warnings"
  - "twine check --strict gate step in publish-pypi.yml, run before upload for both matrix legs"
  - "workflow_dispatch target input (testpypi default) with target-aware token guard + TWINE_REPOSITORY_URL on the publish step"
  - "docs/RELEASING.md release checklist covering tag traps, version discipline, burned versions, token validation, .pypirc layout, TestPyPI rehearsal, secrets"
affects: [future-pypi-releases, packaging-ci]

actuals:
  tokens: 5747
  tasks: 4
  commits: 4

tech-stack:
  added: []
  patterns:
    - "extract-verbatim-and-execute: every new/edited run: body is extracted from the committed YAML via yaml.safe_load and driven standalone with env vars, never retyped or asserted from inspection alone"
    - "ternary-the-secret-NAME-not-the-VALUE: target-aware secret selection always resolves a non-empty literal secret name first, then indexes secrets[...] exactly once, avoiding the GH Actions A || B falsy-fallthrough trap"

key-files:
  created:
    - src/trading_crab_lib/README.md
    - docs/RELEASING.md
  modified:
    - src/trading_crab_lib/pyproject.toml
    - .github/workflows/publish-pypi.yml
    - .gitignore

key-decisions:
  - "MANIFEST.in needed no edit -- confirmed empirically that the real sdist already lists README.md via setuptools packages.find, before touching the file"
  - "twine check --strict gate is a separate, distinctly-named step from Verify built artifacts (different failure category: metadata rendering vs tag/version agreement), matching the precedent set by the token-presence guard's own step in the prior quick task"
  - "Publish step's run: body left byte-identical; TestPyPI routing implemented entirely via a new TWINE_REPOSITORY_URL env entry, confirmed against twine's real CLI-to-Settings code path"
  - "Added a .gitignore rule for src/trading_crab_lib/trading_crab_lib/, a recurring build-time duplicate directory this session's own build steps kept producing (Rule 1 cleanup, not part of the plan's file list)"

requirements-completed: [QUICK-260911-la3]

duration: 35min
completed: 2026-09-11
status: complete
---

# Quick Task 260911-la3: Harden the PyPI release procedure Summary

**Fixed trading-crab-lib's blank PyPI page (missing README metadata), added a `twine check --strict` gate before every upload, added a TestPyPI dry-run path reachable only via `workflow_dispatch`, and wrote `docs/RELEASING.md` capturing the full release procedure and every trap discovered.**

## Performance

- **Duration:** ~35 min
- **Tasks:** 4/4 completed
- **Files modified:** 5 (`.github/workflows/publish-pypi.yml`, `src/trading_crab_lib/pyproject.toml`, `src/trading_crab_lib/README.md`, `docs/RELEASING.md`, `.gitignore`)

## Accomplishments

- `trading-crab-lib` goes from `twine check --strict` **FAILED due to warnings** (missing `long_description` / `long_description_content_type`) to a clean **PASSED**, on real freshly-built artifacts, with a library-scoped README whose every named identifier was verified against the real source.
- `.github/workflows/publish-pypi.yml` now runs `twine check --strict` against real dist artifacts for both matrix legs, positioned between `Verify built artifacts` and the actual upload, as its own separately-diagnosed step.
- A `workflow_dispatch` `target` input (default `testpypi`) lets an operator rehearse a real upload against TestPyPI; a tag push always resolves `target=pypi` because that event has no such input to read at all. The token guard and publish step select the target-appropriate secret via a name-then-lookup ternary that cannot silently fall through to the production token.
- `docs/RELEASING.md` documents the tag-convention trap, version-bump discipline, the permanently-burned 0.1.3 version, the 400-vs-403 token-validation technique (and the `--skip-existing` trap that defeats it), a four-section `~/.pypirc` layout, the TestPyPI rehearsal trigger, and all three secrets.

## Task Commits

1. **Task 1: Give trading-crab-lib a real README and prove twine check goes from WARNING to PASSED** — `c25cbc6` (feat)
2. **Task 2: Add a twine check --strict gate and prove it passes on real artifacts for both packages** — `beabc63` (feat)
3. **Task 3: Add a TestPyPI dry-run path reachable only via workflow_dispatch** — `e97c8a5` (feat)
4. **Task 4: Write docs/RELEASING.md — the release checklist and every trap discovered** — `ca837cd` (docs)

## Verification Evidence (captured this session, real builds/runs — not asserted from inspection)

### Task 1 — before/after `twine check --strict` on `trading-crab-lib`

**Before (pre-fix, this session, real fresh build):**
```
Checking /tmp/la3_baseline_dist/trading_crab_lib-0.1.4-py3-none-any.whl: FAILED due to warnings
WARNING `long_description_content_type` missing. defaulting to `text/x-rst`.
WARNING `long_description` missing.
Checking /tmp/la3_baseline_dist/trading_crab_lib-0.1.4.tar.gz: FAILED due to warnings
WARNING `long_description_content_type` missing. defaulting to `text/x-rst`.
WARNING `long_description` missing.
(exit 1)
```

**After (post-fix, this session, real fresh build):**
```
Checking /tmp/la3_lib_dist/trading_crab_lib-0.1.4-py3-none-any.whl: PASSED
Checking /tmp/la3_lib_dist/trading_crab_lib-0.1.4.tar.gz: PASSED
```
Wheel `METADATA` contains `Description-Content-Type: text/markdown` with a non-empty description body (confirmed by direct `grep`/inspection of the unzipped `.dist-info/METADATA`). Sdist tarball listing includes `trading_crab_lib-0.1.4/README.md` at its top level. `MANIFEST.in` was left unchanged — this was verified empirically, not assumed, before writing the summary.

### Task 2 — `twine check --strict` gate driven on real artifacts, both packages

Extracted the new `Check package metadata with twine` step's `run:` body verbatim from the committed YAML (position confirmed strictly between `Verify built artifacts` and `Publish ... to PyPI`, `if:` string-identical to its neighbors, zero `${{ }}` in the body), then built both real packages fresh and ran it against each:
```
Checking /tmp/la3_probe/app_build/dist/trading_crab-0.1.4-py3-none-any.whl: PASSED
Checking /tmp/la3_probe/app_build/dist/trading_crab-0.1.4.tar.gz: PASSED
Checking /tmp/la3_probe/lib_build/dist/trading_crab_lib-0.1.4-py3-none-any.whl: PASSED
Checking /tmp/la3_probe/lib_build/dist/trading_crab_lib-0.1.4.tar.gz: PASSED
```

### Task 3 — gate step `target` resolution, driven under both event shapes

Extracted the gate step's `run:` body verbatim and drove it with real env vars matching each event shape, capturing `$GITHUB_OUTPUT`:

**Tag push (`EVENT_NAME=push`, `GITHUB_REF=refs/tags/v0.1.4`, no `INPUT_TARGET`):**
```
GITHUB_OUTPUT contents: target=pypi
                        publish=true
log:                    TARGET: trading-crab will publish to 'pypi' (event=push)
                        PUBLISH: trading-crab tag=v0.1.4
```

**workflow_dispatch (`EVENT_NAME=workflow_dispatch`, `INPUT_TARGET=testpypi`, `INPUT_PKG=both`):**
```
GITHUB_OUTPUT contents: target=testpypi
                        publish=true
log:                    TARGET: trading-crab-lib will publish to 'testpypi' (event=workflow_dispatch)
                        PUBLISH: trading-crab-lib via workflow_dispatch (input=both)
```

A tag push resolves `target=pypi` with **no dispatch input present at all** — this is the observed proof that production is the only reachable outcome on that path, not a default silently overridden.

**Guard regression (all 4 combinations, real extracted `run:` body):** unset token / `TARGET=pypi` → exit 1, names `PYPI_APP_TOKEN`. Unset token / `TARGET=testpypi` → exit 1, names `TEST_PYPI_API_TOKEN`. Empty-string token / `TARGET=pypi` → exit 1, names `PYPI_APP_TOKEN`, mentions "empty string". Empty-string token / `TARGET=testpypi` → exit 1, names `TEST_PYPI_API_TOKEN`, mentions "empty string". All four passed.

### Task 3 — live `twine.settings.Settings` probe for `TWINE_REPOSITORY_URL` resolution

**Important correction made during this task, recorded honestly:** the first probe attempt called `Settings()` directly with no arguments and observed BOTH the empty-string case and the TestPyPI-literal case resolve to production (`https://upload.pypi.org/legacy/`). Investigating twine 7.0.0's actual source (`twine/settings.py`) showed `Settings.__init__` takes `repository_url: Optional[str] = None` as a plain keyword default — it does **not** read any environment variable itself. Reading `TWINE_REPOSITORY_URL` only happens through argparse's `EnvironmentDefault` action, registered via `Settings.register_argparse_arguments(parser)` and evaluated only when `parser.parse_args(...)` actually runs — which is exactly the sequence `twine upload` performs internally (`twine/commands/upload.py:237-250`: register → `parse_args` → `Settings.from_argparse`). The probe was rewritten to reproduce that real sequence instead of the bare constructor. Re-run with the corrected code path:
```
Case 1 (pypi branch, empty env value): TWINE_REPOSITORY_URL='' -> resolved repository='https://upload.pypi.org/legacy/'
Case 2 (testpypi branch, literal set): TWINE_REPOSITORY_URL='https://test.pypi.org/legacy/' -> resolved repository='https://test.pypi.org/legacy/'
OK: both cases resolved as expected
```
This confirms the mechanism this plan's Task 3 design depends on actually holds for the real, installed twine version, via the same construction path the workflow's `twine upload --skip-existing "$DIST_DIR"/*` invocation uses.

### What could not be executed (stated explicitly, not glossed over)

The GitHub Actions expression engine's own evaluation of the target-aware ternaries
(`steps.gate.outputs.target == 'testpypi' && 'TEST_PYPI_API_TOKEN' || matrix.secret_name`,
and the analogous `TWINE_REPOSITORY_URL` ternary) **cannot be executed in this sandbox** —
there is no live GitHub Actions runner available here. This boundary is verified only
**structurally**: `yaml.safe_load` parses the committed workflow, and Python-level string
assertions confirm (a) the guard's and publish step's `TWINE_PASSWORD` expressions are
byte-identical, (b) the publish step's `TWINE_REPOSITORY_URL` expression contains both the
`target == 'testpypi'` condition and the TestPyPI literal URL, and (c) the `workflow_dispatch`
`target` input has `default: testpypi` and `options: [testpypi, pypi]`. The gate and guard
steps' actual bash logic (everything except the GH Actions `${{ }}` evaluation itself) was
proven by live execution, as detailed above — only the expression engine's own ternary
resolution is unproven, and is reported here as such rather than asserted as passing.

## Regression scope confirmed unchanged

`git diff` across this session's four commits touches exactly:
`.github/workflows/publish-pypi.yml`, `src/trading_crab_lib/pyproject.toml`,
`src/trading_crab_lib/README.md` (new), `docs/RELEASING.md` (new), `.gitignore`.
Confirmed unchanged: tag conventions, the matrix (`build_dir`, `secret_name`, `tag_prefix`
per leg), the gate's pre-existing publish/skip branching, the dist path line
(`DIST_DIR="${{ matrix.build_dir }}/dist"`), `Verify built artifacts`, `--skip-existing`,
both package versions (`0.1.4` in both `pyproject.toml` files and `__init__.py`), and
`.github/workflows/python-package.yml` (zero-line diff, confirmed via
`git diff 4cbc829..HEAD -- .github/workflows/python-package.yml`).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - recurring build hazard] `.gitignore` rule for `src/trading_crab_lib/trading_crab_lib/`**
- **Found during:** Task 1 and Task 2 verification builds (both produced this stray directory).
- **Issue:** Building `trading-crab-lib` from its own subdirectory (`python -m build
  src/trading_crab_lib`) causes setuptools' `packages.find(where=[".."])` to re-discover the
  package one level down, creating a stray duplicate copy of the entire library nested inside
  itself — an untracked directory that this repo has hit three times already per the task
  instructions.
- **Fix:** deleted the stray directory after each build in this session; added an explicit
  `.gitignore` rule with an explanatory comment so it can never be accidentally staged.
- **Files modified:** `.gitignore`.
- **Commit:** `c25cbc6` (bundled with Task 1, since it was discovered during Task 1's verification build).

### Corrections made during execution

**2. [Verification correction, not a plan deviation] Task 3's `twine.settings.Settings` probe rewritten to use the real CLI-to-Settings code path.**
- The plan's context section described a probe that called `Settings(...)` directly; the first
  attempt at reproducing it this session showed that path does not read `TWINE_REPOSITORY_URL`
  at all (it resolved to production in both cases, including the TestPyPI-literal case).
  Investigated twine 7.0.0's source, found the actual mechanism (`register_argparse_arguments` →
  `parser.parse_args([])` → `Settings.from_argparse`), rewrote the probe to reproduce that exact
  sequence, and re-ran it — the corrected probe confirms the mechanism this plan's design depends
  on. Documented in full above so this correction is not silently absorbed into the "PASSED"
  claim.

None of the plan's required file scope, tasks, or success criteria changed as a result of either
item above.

## Known Stubs

None. No hardcoded empty values, placeholder text, or unwired data sources were introduced.

## Threat Flags

None beyond what the plan's own `<threat_model>` already enumerated (T-la3-01 through T-la3-06,
T-la3-SC) — no new network endpoints, auth paths, or trust-boundary changes were introduced
outside that register. All dispositions in that register were honored: the secret-name ternary
pattern (T-la3-02, T-la3-03) was implemented and verified as designed; no real secret value was
ever written to any committed file or scratchpad artifact (T-la3-01, T-la3-06); no new package
installs occurred (T-la3-SC).

## Self-Check: PASSED

- `src/trading_crab_lib/README.md` — FOUND
- `docs/RELEASING.md` — FOUND
- `src/trading_crab_lib/pyproject.toml` (readme key) — FOUND (`readme = "README.md"` present in both `[project]` and `[tool.poetry]`)
- `.github/workflows/publish-pypi.yml` (twine check step + target input) — FOUND
- Commit `c25cbc6` — FOUND in `git log`
- Commit `beabc63` — FOUND in `git log`
- Commit `e97c8a5` — FOUND in `git log`
- Commit `ca837cd` — FOUND in `git log`
- Working tree clean, no stray build artifacts (`dist/`, `*.egg-info/`, `src/trading_crab_lib/trading_crab_lib/`) — CONFIRMED via `git status --porcelain`
