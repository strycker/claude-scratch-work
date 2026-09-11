---
phase: quick-260911-kkj
plan: 01
subsystem: ci-cd
tags: [github-actions, publish-pypi, secrets, workflow]
dependency-graph:
  requires: []
  provides: [pypi-publish-token-presence-guard]
  affects: [.github/workflows/publish-pypi.yml]
tech-stack:
  added: []
  patterns:
    - "zero-template-expression run: bodies with values hoisted into env: (established by 260910-vyi, reused here)"
    - "structural verification of committed CI YAML via yaml.safe_load, never a retyped paraphrase"
key-files:
  created: []
  modified:
    - .github/workflows/publish-pypi.yml
decisions:
  - "New step, not folded into the publish step: failure naming is the whole point, and folding would require editing an out-of-scope dist-path line."
  - "Positioned immediately after the gate step, before Set up Python: the wiring fault is knowable in seconds and should not cost a Python setup + build first."
  - "Emptiness test uses ${TWINE_PASSWORD:-} (default-substituting), not the bare form, so an unset token produces the remedy message instead of a bash nounset diagnostic under set -u."
metrics:
  duration: ~25min
  completed: 2026-09-11
status: complete
actuals:
  tokens: 373
  tasks: 2
  commits: 1
---

# Phase quick-260911-kkj Plan 01: Add a token-presence guard to the PyPI publish workflow Summary

Added a statically-named `Check PyPI token is present` step to
`.github/workflows/publish-pypi.yml` that fails the job before any Python setup, build, or
`twine upload` when a matrix leg's PyPI token secret is absent or empty, naming the exact
secret, both valid locations for it, and the empty-string-on-name-mismatch behavior of the
dynamic `secrets[matrix.secret_name]` lookup — proven to fire on both must-fail scenarios (unset,
empty) on both matrix legs, to pass and leak nothing on a present token, and to leave everything
else in the workflow byte-identical.

## What Was Built

**Task 1 — the guard step itself.**

A new step was inserted into `jobs.publish.steps`, positioned after `Decide whether to publish
this package` (the gate) and before `Set up Python`:

```yaml
- name: Check PyPI token is present
  if: steps.gate.outputs.publish == 'true'
  env:
    PKG: ${{ matrix.name }}
    SECRET_NAME: ${{ matrix.secret_name }}
    TWINE_PASSWORD: ${{ secrets[matrix.secret_name] }}
  run: |
    set -euo pipefail

    if [ -z "${TWINE_PASSWORD:-}" ]; then
      echo "ERROR: PyPI token for $PKG is missing or empty." >&2
      echo "Expected secret name: $SECRET_NAME." >&2
      echo "It must be defined either as a repository secret, or as an environment secret on the 'pypi' environment." >&2
      echo "Note: GitHub's dynamic secrets[...] lookup resolves an unknown or mistyped secret name to an empty string rather than erroring, so an exact-name typo looks identical to an absent secret. Double-check the name is exactly '$SECRET_NAME'." >&2
      exit 1
    fi

    echo "Token present for $PKG (secret: $SECRET_NAME) - value intentionally not shown."
```

It carries the same `if:` condition as every other publish-path step (character-for-character
identical to `Verify built artifacts`'s), and reads `TWINE_PASSWORD` through the exact same
`secrets[matrix.secret_name]` expression the `Publish ... to PyPI` step uses — verified by direct
string comparison of the two parsed `env:` entries, not by eyeballing the diff.

A Python extractor (`extract_token_guard.py`, written to the scratchpad) loads the committed YAML
with `yaml.safe_load`, indexes `w["jobs"]["publish"]["steps"]`, locates the guard by its exact
static name, and asserts: zero template expressions in the `run:` body; `if:` identical to the
version-guard step; position strictly between the gate and publish steps; and the
`TWINE_PASSWORD` env expression byte-identical to the publish step's. Only after all four
assertions pass does it write the `run:` body verbatim to `token_guard.sh` — the script actually
executed in every scenario below, never a retyped paraphrase.

**Task 2 — the full scenario matrix, observed, not asserted.**

`run_token_guard_scenarios.sh` re-runs the extractor against the current committed YAML (so it
never tests a stale extraction) and drives `token_guard.sh` through:

- Must-fail: token entirely unset, for both `PYPI_APP_TOKEN` and `PYPI_LIB_TOKEN` legs.
- Must-fail: token set to the empty string, for both legs.
- Must-pass: token set to a distinctive sentinel (`TC-GUARD-SENTINEL-9f3e7c2a1b-DO-NOT-LEAK`),
  with a grep of the full captured stdout+stderr for that sentinel.
- Runner-fidelity: the skipped-leg-never-runs-the-guard property, asserted structurally from the
  parsed YAML's `if:` string (a skipped step produces no bash to run — simulating a skip would be
  theater, so this is not re-run as bash).
- Scope: `git diff --name-only origin/main -- .github/workflows/` lists only
  `publish-pypi.yml`; the corrected dist-path line, `--skip-existing`, both `0.1.4` versions, and
  `python-package.yml`'s 3.10-3.13 matrix are all unchanged.

Task 2 made no code changes — no defect was found, so there is nothing beyond Task 1's commit.

## Evidence

### Must-fail scenario, token unset, leg = PYPI_APP_TOKEN (actual captured output)

```
exit status: 1
ERROR: PyPI token for trading-crab is missing or empty.
Expected secret name: PYPI_APP_TOKEN.
It must be defined either as a repository secret, or as an environment secret on the 'pypi' environment.
Note: GitHub's dynamic secrets[...] lookup resolves an unknown or mistyped secret name to an empty string rather than erroring, so an exact-name typo looks identical to an absent secret. Double-check the name is exactly 'PYPI_APP_TOKEN'.
```

No `unbound variable` / `nounset` text appears — the `${TWINE_PASSWORD:-}` default-substituting
form was used specifically to avoid that trap under `set -u`.

The same message shape (with `PYPI_LIB_TOKEN` substituted) was independently observed for the
`trading-crab-lib` leg, and for both legs again with the token set to the empty string instead of
unset — all four must-fail combinations exited non-zero and contained the leg's own secret name
plus `repository secret`, `environment secret`, `pypi`, and `empty string`.

### Must-pass scenario with leak check

Token set to the sentinel `TC-GUARD-SENTINEL-9f3e7c2a1b-DO-NOT-LEAK`:

```
exit status: 0
Token present for trading-crab (secret: PYPI_APP_TOKEN) - value intentionally not shown.
```

**A grep of the captured output for the sentinel found nothing.** (`grep -qF -- "$SENTINEL"
"$OUT"` returned non-zero / no match — confirmed explicitly by the scenario harness, which
reports `PASS: sentinel grep found NOTHING in captured output`, and independently re-confirmed
by direct inspection of the printed guard output above, which contains no trace of the sentinel
string.)

### Scope assertions (all observed)

```
changed workflow files vs origin/main: .github/workflows/publish-pypi.yml
PASS: only publish-pypi.yml changed under .github/workflows/
PASS: corrected dist-path line present
PASS: zero occurrences of doubled dist path outside comments
PASS: --skip-existing preserved
PASS: both package versions still 0.1.4
PASS: python-package.yml unchanged vs origin/main
PASS: python-package.yml still lists 3.10..3.13 matrix
```

The plan's Task 2 second `<automated>` verify block (exact literal commands, not a paraphrase)
was also run directly and printed: `preserved steps OK`, `dist path preserved OK`,
`no doubled path OK`, `skip-existing preserved OK`, `versions untouched OK`,
`only publish-pypi.yml changed OK`.

**Nothing was left unproven.** PyYAML 6.0.3, bash, and full git history were all available in
this environment; every scenario in the plan ran to completion with a captured result.

## Deviations from Plan

None — plan executed exactly as written. Both tasks' `<done>` criteria and the plan's
`<success_criteria>` are satisfied by the evidence above.

## Known Stubs

None.

## Threat Flags

None. The one new surface (the token entering a second step's environment) is explicitly
addressed in the plan's own threat register as T-kkj-04 (accepted) and verified per T-kkj-01's
mitigation (sentinel-leak scenario, observed clean).

## Self-Check: PASSED

- `FOUND: .github/workflows/publish-pypi.yml`
- `FOUND: 118bd06` (commit `feat(publish-pypi): add token-presence guard before build steps`)
