---
phase: quick-260911-nt7
plan: 01
subsystem: infra
tags: [packaging, setuptools, pypi, github-actions, release-engineering]

requires:
  - phase: quick-260910-vyi
    provides: "dist path fix, PUBLISH/SKIP gate logging, Verify built artifacts version guard"
  - phase: quick-260911-kkj
    provides: "token-presence guard step"
  - phase: quick-260911-la3
    provides: "twine check --strict gate, docs/RELEASING.md, lib README wired into pyproject"
provides:
  - "trading-crab-lib wheels that actually contain the package: explicit [tool.setuptools] package-dir mapping + enumerated packages list replacing the silently-empty packages.find glob"
  - "install-and-import smoke-test gate in publish-pypi.yml, run in a throwaway venv with cwd off the checkout, for both matrix legs"
  - "--no-deps on that smoke test so the two parallel matrix legs cannot couple through PyPI dependency resolution"
  - "docs/RELEASING.md sections 9 and 10 (the empty-wheel trap; the parallel-leg independence rule and half-shipped-release recovery)"
  - "both packages bumped 0.1.4 -> 0.1.5; PyPI badges in README.md"
affects: [future-pypi-releases, packaging-ci]

actuals:
  tasks: 4
  commits: 2

tech-stack:
  added: []
  patterns:
    - "prove-the-gate-in-both-directions: the smoke test was verified to exit 0 on the real wheel AND exit 1 on a deliberately constructed metadata-only wheel, rather than asserted from inspection"
    - "cd-off-the-checkout: the import check runs from a temp dir so a local source copy cannot stand in for a missing module"
    - "matrix-leg-independence: no step in one publish matrix leg may depend on state produced by the other; --no-deps enforces it for the smoke test"

key-files:
  created: []
  modified:
    - src/trading_crab_lib/pyproject.toml
    - .github/workflows/publish-pypi.yml
    - pyproject.toml
    - src/trading_crab/__init__.py
    - src/trading_crab_lib/__init__.py
    - docs/RELEASING.md
    - README.md
    - CLAUDE.md
    - .claude/CLAUDE.md
    - .gitignore

key-decisions:
  - "Explicit enumerated packages list, NOT a packages.find discovery glob: src/trading_crab_lib/ is both the project root and the package's own content directory (flat layout), and modern setuptools refuses to let `where` escape the project root — it returned an empty list and the build still exited 0"
  - "Removed the .gitignore rule for src/trading_crab_lib/trading_crab_lib/ added by 260911-la3 — that stray directory was a symptom of this same misconfiguration; keeping the rule would only mask a regression"
  - "Smoke test uses pip install --no-deps: the gate's purpose is to prove a wheel contains its own code, and resolving deps from PyPI made the app leg fail on a lib version its sibling leg published 6 seconds later (observed on both-v0.1.5, which half-shipped)"
  - "Ordering the matrix legs was rejected as a fix — PyPI index propagation is not instantaneous, so ordering would not have made it reliable"

requirements-completed: [QUICK-260911-nt7]

completed: 2026-09-11
status: complete
---

# Quick Task 260911-nt7: Fix the empty trading-crab-lib wheel Summary

**Every published `trading-crab-lib` wheel from 0.1.0 through 0.1.4 contained four metadata
files and zero Python modules. `pip install trading-crab-lib==0.1.4` succeeded and
`import trading_crab_lib` then raised `ModuleNotFoundError`; because `trading-crab` pins the
library, the app's CLI was broken from PyPI too.**

## Root cause

`src/trading_crab_lib/` is simultaneously the project root and the package's own content
directory. The config used `[tool.setuptools.packages.find] where = [".."]` to re-discover
the package one level up. Modern setuptools refuses to let `where` escape the project root,
so it silently returned an **empty package list** and `python -m build` exited 0.

Replaced with an explicit `[tool.setuptools] package-dir = {"trading_crab_lib" = "."}` plus
an enumerated `packages = [...]` list covering all 17 real packages, with an in-file comment
warning against reintroducing a discovery glob.

**Result:** wheel goes from 4,836 bytes / 4 files to 351,595 bytes / 106 modules. Verified by
installing into a clean venv and importing from `/tmp` with no repo on `sys.path` — 16/16
subpackages import with the `[plotting]` extra. The editable dev install
(`pip install -e "src/trading_crab_lib/[all,dev]"`) still works and no longer recreates the
stray `src/trading_crab_lib/trading_crab_lib/` duplicate directory.

## Why no existing gate caught it

An empty wheel has flawless metadata and a flawless filename:

| Gate | What it checks | Why it passed |
|---|---|---|
| `twine check --strict` | metadata rendering | metadata was valid |
| Verify built artifacts | tag/filename version agreement | filename was correct |
| `build-pkg` CI job | that `python -m build` ran | it ran, and exited 0 |

Only installing the wheel and importing it can detect this. Added a smoke-test step that
installs the built wheel into a throwaway venv and imports it, `cd`'d off the checkout so a
local source copy cannot stand in for a missing module. Proven in both directions: exits 0
on the real wheel, exits 1 on a deliberately constructed metadata-only one.

## Follow-on defect in the new gate (commit 2)

The `both-v0.1.5` release shipped only half — `trading-crab-lib` 0.1.5 published, the app
leg's publish step skipped. The app leg's smoke test failed with

```
Could not find a version that satisfies the requirement trading-crab-lib>=0.1.5
(from versions: 0.1.0 ... 0.1.4)
```

Not a flake. The two matrix legs run in parallel with no ordering; the smoke test's plain
`pip install <wheel>` resolved dependencies from PyPI, and the app wheel declares
`trading-crab-lib>=0.1.5` — a version its sibling leg published **six seconds after** the app
leg tried to resolve it (app smoke 18:36:08–18:36:14, lib publish 18:36:20–18:36:23).

Fixed with `pip install --quiet --no-deps`. Verified the gate still does its job (the
metadata-only wheel still exits 1) and that both legs now pass while the app wheel still
declares an unpublished `trading-crab-lib>=0.1.5` — precisely the scenario that failed.

## Commits

| Commit | Subject |
|---|---|
| `f204de3` | fix(packaging): trading-crab-lib shipped zero Python modules; add install-import gate |
| `926ef21` | fix(publish-pypi): smoke-test with --no-deps; parallel matrix legs must not couple |

## Carried forward

Recorded as tech debt in `ROADMAP.md` Tier 0.5 (commit `5535ed6` plus this task's follow-up):

- **R2** — `build-pkg` in `python-package.yml` builds both packages but never installs them,
  so an empty wheel can still merge to `main` and stay invisible until release time.
- **R4** — the enumerated `packages` list is unguarded by any test, and the publish smoke
  test imports only the top-level package. A new subpackage omitted from the list would ship
  missing with every gate green. Directly relevant to Phase 7, which adds modules.

## Deviations from plan

None material. The plan's `files_modified` list was followed; the `.gitignore` change was a
deletion rather than an edit (see key-decisions).
