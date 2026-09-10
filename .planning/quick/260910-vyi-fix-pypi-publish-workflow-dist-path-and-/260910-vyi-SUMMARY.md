---
phase: quick-260910-vyi
plan: 01
subsystem: ci-cd
tags: [github-actions, pypi, publish, versioning]
dependency-graph:
  requires: []
  provides: ["publish-pypi.yml with corrected dist path, loud gate, and version-match guard", "0.1.4 versioned packages"]
  affects: [".github/workflows/publish-pypi.yml", "pyproject.toml", "src/trading_crab_lib/pyproject.toml", "src/trading_crab/__init__.py", "src/trading_crab_lib/__init__.py"]
tech-stack:
  added: []
  patterns: ["hoist matrix/context values into env: so run: bodies contain zero ${{ }} expressions, making them extractable via yaml.safe_load for local verification"]
key-files:
  created: []
  modified:
    - .github/workflows/publish-pypi.yml
    - pyproject.toml
    - src/trading_crab_lib/pyproject.toml
    - src/trading_crab/__init__.py
    - src/trading_crab_lib/__init__.py
    - CLAUDE.md
    - .claude/CLAUDE.md
decisions:
  - "Hardcoded BOTH_PREFIX=\"both-v\" in the guard script rather than adding a new matrix field, since the both-v prefix is a fixed workflow convention documented in the file's header comment, not a per-package value."
  - "Wheel version parsed via 'cut -d- -f2' (second hyphen field) and sdist version via 'strip .tar.gz, take text after last hyphen' per the plan's literal spec — both are unambiguous given PEP 427/PEP 440 filename normalization (dist name hyphens become underscores)."
metrics:
  duration: ~35min
  completed: 2026-09-10
status: complete
actuals:
  tokens: 2129
  tasks: 2
  commits: 2
---

# Quick Task 260910-vyi: Fix PyPI Publish Workflow Dist Path and Version Guards Summary

Fixed a `.github/workflows/publish-pypi.yml` that produced a green CI run publishing nothing:
a doubled dist-directory path broke the twine glob, a skipped matrix leg was indistinguishable
from a successful publish, and `--skip-existing` silently masked a missed version bump. All
three are fixed, both packages are bumped 0.1.2 → 0.1.4 (0.1.3 already burned on PyPI), and the
two new guards were proven to fire — including the two must-fail scenarios — against real local
builds.

## What Was Built

**Task 1 — Version bump + dist-path fix.** Bumped `trading-crab` and `trading-crab-lib` from
0.1.2 to 0.1.4 at all seven live-pin sites (both `pyproject.toml` `version` fields, the app's
`trading-crab-lib>=` dependency pin plus its restating comment, both `__init__.py`
`__version__` strings, and the live-pin prose in `CLAUDE.md`/`.claude/CLAUDE.md`). Left
`CLAUDE.md:1922` (D47 decision-log entry, "since v0.1.2") and everything under `docs/archive/`
and `.planning/` untouched — those are historical records, not live state.

Replaced the publish step's `DIST_DIR="${{ matrix.build_dir == '.' && 'dist' || matrix.build_dir }}/dist"`
with the direct `DIST_DIR="${{ matrix.build_dir }}/dist"`. The old ternary already substituted
the literal `dist` for the app leg (`build_dir == '.'`) and then unconditionally appended
`/dist` again, producing `dist/dist` for the app leg and breaking the twine glob. The direct
form is correct for both legs because `python -m build <srcdir>` always writes to
`<srcdir>/dist`.

**Task 2 — Loud gate + hard publish guard.** The `Decide whether to publish this package` step
now hoists `matrix.name`, `matrix.tag_prefix`, `github.event_name`, and
`github.event.inputs.package` into an `env:` block and emits exactly one `PUBLISH:` or `SKIP:`
line on every branch (workflow_dispatch match, both-v tag, per-package prefix match, no match),
naming the package, tag, expected prefix, and reason. Previously a skipped leg logged nothing
distinguishable from success.

A new `Verify built artifacts` step runs between `Build` and `Publish`, gated by the same
`if: steps.gate.outputs.publish == 'true'` condition. It:
1. Collects `"$BUILD_DIR"/dist/*` with `nullglob`; fails with a named directory if empty.
2. Exits 0 immediately for `workflow_dispatch` runs (no tag to compare).
3. Otherwise derives the tag from `GITHUB_REF`, strips the `both-v` prefix or the leg's own
   `tag_prefix` to get the expected version.
4. Parses each artifact's version from its filename (wheel: 2nd hyphen field; sdist: text after
   the last hyphen with `.tar.gz` stripped) and fails — naming package, filename, artifact
   version, tag, and tag version — on any mismatch.
5. Prints a confirmation line on success.

Every `run:` body touched in this task contains zero `${{ }}` expressions — all matrix/context
values are hoisted into `env:` — which is what makes the shipped bash extractable via
`yaml.safe_load` and runnable standalone, rather than paraphrased.

## Local Verification (evidence, not description)

Both packages were built with `./.venv/bin/python -m build` (`build` installed into the
pre-existing `.venv` via `uv pip install`, no committed dependency change):

```
dist/trading_crab-0.1.4-py3-none-any.whl       (30809 bytes)
dist/trading_crab-0.1.4.tar.gz                 (42360 bytes)
src/trading_crab_lib/dist/trading_crab_lib-0.1.4-py3-none-any.whl  (1893 bytes)
src/trading_crab_lib/dist/trading_crab_lib-0.1.4.tar.gz            (3122 bytes)
```

The corrected `DIST_DIR` glob matched exactly these 2 files per leg, every filename carrying
`0.1.4`, no doubled-path or stale-version artifact present.

**Gate and guard `run:` bodies were extracted verbatim from the committed YAML** via
`yaml.safe_load(open('.github/workflows/publish-pypi.yml'))['jobs']['publish']['steps']`,
written to `gate.sh` / `guard.sh`, and driven with environment variables (never retyped):

- Gate: `v0.1.4` → app leg `PUBLISH:`/`publish=true`, lib leg `SKIP:`/`publish=false`.
  `lib-v0.1.4` → mirror image. `both-v0.1.4` → both legs `PUBLISH:`/`publish=true`.
- Guard: exits 0 for app leg vs. real `dist/` on tag `v0.1.4`; exits 0 for lib leg vs. real
  `src/trading_crab_lib/dist/` on tag `lib-v0.1.4`; exits 0 for `both-v0.1.4` against `dist/`
  (proving the both-prefix strip); exits 0 for a `workflow_dispatch` event (no tag).

**The two must-fail scenarios were observed failing, not merely written:**

- Mismatched tag `v0.1.3` against the real `dist/` (containing 0.1.4 artifacts) — **observed
  exit code non-zero**, with output:
  ```
  ERROR: version mismatch for trading-crab: artifact trading_crab-0.1.4-py3-none-any.whl has
  version 0.1.4 but tag v0.1.3 implies version 0.1.3. Bump the version in this package's
  pyproject.toml or retag.
  ```
  This is exactly the class of failure that let the actual 0.1.3 release publish nothing under
  `--skip-existing`.
- Empty scratch `dist/` directory — **observed exit code non-zero**, with output:
  ```
  ERROR: no artifacts found in /tmp/tmp.XXXXXXXXXX/dist
  ```

Full scenario driver script output is captured in the session scratchpad
(`run_guard_scenarios.sh`, all 8 scenario groups reported `PASS`, final line
`=== ALL SCENARIOS PASSED ===`).

**Cleanup:** `dist/`, `src/trading_crab_lib/dist/`, and a stray
`src/trading_crab_lib/trading_crab_lib/` directory produced as a side effect of building the
lib package from its own subdirectory were all removed after verification. `git status
--porcelain` confirms nothing built is staged or present in the working tree.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking issue] `build` package not installed in `.venv`**
- **Found during:** Task 1 precondition check
- **Issue:** `.venv/bin/python -m build --version` failed with `No module named build`.
- **Fix:** Installed via `/root/.local/bin/uv pip install --python .venv/bin/python build`
  (per the plan's documented precondition remediation and session facts — `uv` and network
  egress through the agent proxy were already confirmed working). No committed dependency file
  changed; `build` is a local verification tool only, matching threat T-vyi-SC's disposition.
- **Files modified:** none (venv-only).
- **Commit:** n/a (not a tracked change).

**2. [Rule 1 - cleanup] Stray `src/trading_crab_lib/trading_crab_lib/` directory from building the lib package**
- **Found during:** Task 1 build step
- **Issue:** Running `python -m build src/trading_crab_lib` produced an untracked nested
  `src/trading_crab_lib/trading_crab_lib/` copy of the package source (a setuptools/build
  artifact of building from within the package's own subdirectory), in addition to the expected
  `dist/`.
- **Fix:** Removed it during the Task 2 cleanup step alongside `dist/` and any `egg-info/`
  directories, so no stray build byproduct was left in the working tree or risked being staged.
- **Files modified:** none tracked (it was never committed).
- **Commit:** cleaned up before the Task 2 commit (0c6fb8d); not itself a commit.

No other deviations. Plan executed as written otherwise.

## Known Stubs

None.

## Threat Flags

None — this task's threat model (T-vyi-01 through T-vyi-04, T-vyi-SC) was fully addressed by
the plan's own design; no new unmitigated surface was introduced.

## Self-Check: PASSED

- `FOUND: .github/workflows/publish-pypi.yml` (DIST_DIR fix + gate + guard present, YAML parses)
- `FOUND: pyproject.toml` (version 0.1.4, pin trading-crab-lib>=0.1.4)
- `FOUND: src/trading_crab_lib/pyproject.toml` (version 0.1.4)
- `FOUND: src/trading_crab/__init__.py` (__version__ = "0.1.4")
- `FOUND: src/trading_crab_lib/__init__.py` (__version__ = "0.1.4")
- `FOUND: CLAUDE.md` line 266 updated to 0.1.4; line 1922 (D47) untouched
- `FOUND: .claude/CLAUDE.md` line 84 updated to 0.1.4
- `FOUND: 9d0153c` — `git log --oneline` confirms Task 1 commit exists
- `FOUND: 0c6fb8d` — `git log --oneline` confirms Task 2 commit exists
- `git status --short` shows only the untracked `.planning/quick/.../` directory (docs, not
  committed by this executor per instructions) — no build artifacts, no staged secrets.

## Must-Haves Verification (from PLAN.md frontmatter)

| Truth | Status | Evidence |
|---|---|---|
| Dist-directory expression yields `./dist` / `src/trading_crab_lib/dist`, never doubled | PROVEN | grep confirms no `dist/dist` pattern; corrected glob matched real files for both legs |
| Both packages build locally, artifacts carry 0.1.4 | PROVEN | 4 real artifact files listed above, all named `*0.1.4*` |
| A skipped leg is distinguishable from a publishing one by one greppable line | PROVEN | `PUBLISH:`/`SKIP:` lines observed for all 3 tag scenarios × 2 legs |
| Job fails before `twine upload` when dist glob matches zero files | PROVEN — observed failing | empty-dist scenario exited non-zero with named error |
| Job fails before `twine upload` when artifact version ≠ tag-implied version | PROVEN — observed failing | v0.1.3-vs-0.1.4 scenario exited non-zero with both versions named |
| Version-mismatch guard observed failing on a deliberately mismatched tag, not merely written | PROVEN | reproduced above verbatim from the actual scenario run |

No must-have is left unproven; no environment limitation prevented any check in this task
(network was only needed for `uv pip install build`, which succeeded through the agent proxy).
