---
phase: quick-260910-vyi
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - .github/workflows/publish-pypi.yml
  - pyproject.toml
  - src/trading_crab_lib/pyproject.toml
  - src/trading_crab/__init__.py
  - src/trading_crab_lib/__init__.py
  - CLAUDE.md
  - .claude/CLAUDE.md
autonomous: true
requirements: [QUICK-260910-vyi]
user_setup: []

estimate:
  tokens: 60000
  raw_tokens: 30000
  tasks: 2
  confidence: low        # zero calibration samples available for this repo

must_haves:
  truths:
    - "The dist-directory expression yields `./dist` for the app leg and `src/trading_crab_lib/dist` for the lib leg — the path is never doubled."
    - "Both packages build locally and produce artifacts whose filenames carry 0.1.4."
    - "A reader scanning the job log can tell a skipped matrix leg from a publishing one by a single greppable line."
    - "The job fails before `twine upload` when the dist glob matches zero files."
    - "The job fails before `twine upload` when the built artifact version does not match the version implied by the tag."
    - "The version-mismatch guard has been observed failing on a deliberately mismatched tag, not merely written."
  artifacts:
    - .github/workflows/publish-pypi.yml
    - pyproject.toml
    - src/trading_crab_lib/pyproject.toml
    - src/trading_crab/__init__.py
    - src/trading_crab_lib/__init__.py
  key_links:
    - "dist-directory expression ↔ the default output location of `python -m build <srcdir>` (always `<srcdir>/dist`)"
    - "matrix.tag_prefix (and the both-v prefix) ↔ the tag-version derivation inside the guard"
    - "app pyproject `trading-crab-lib>=` pin ↔ the lib pyproject `version` field"
    - "guard step `run:` body ↔ the locally-extracted script under test (must be byte-identical, extracted from the YAML, never retyped)"
---

<objective>
Fix `.github/workflows/publish-pypi.yml` so a tagged release actually publishes, and bump both
packages from 0.1.2 to 0.1.4 (0.1.3 is already burned on PyPI for both packages).

Purpose: the last release attempt produced a green run that published nothing. Three independent
defects conspired — a doubled dist path, a silent matrix skip, and `--skip-existing` masking a
missed version bump. All three are fixed here, and the guards that catch the latter two are
exercised locally so they are proven to fire.

Output: a corrected publish workflow with a loud gate and two hard guards, both package versions
at 0.1.4, and local evidence that the corrected glob matches real build output.
</objective>

<execution_context>
@/Users/glestryc/personal/github_repos/claude-scratch-work/.claude/gsd-core/workflows/execute-plan.md
@/Users/glestryc/personal/github_repos/claude-scratch-work/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.github/workflows/publish-pypi.yml
@pyproject.toml
@src/trading_crab_lib/pyproject.toml
@CLAUDE.md

**Facts already established this session — do not re-derive:**

- PyPI holds 0.1.0, 0.1.1, 0.1.2 **and** 0.1.3 for BOTH `trading-crab` and `trading-crab-lib`
  (all uploaded manually 2026-04-20, verified via the PyPI JSON API). Next publishable
  version is **0.1.4**.
- Both `pyproject.toml` files currently declare `version = "0.1.2"`.
- Tag `v0.1.3` points at commit 9ebe030, where both files still said 0.1.2 — which is why a
  corrected path alone would still have published nothing under `--skip-existing`.
- Live version-tracking sites (confirmed by grep): `pyproject.toml:7`, `pyproject.toml:21`
  (the `trading-crab-lib>=` pin), `pyproject.toml:112` (a comment restating that pin),
  `src/trading_crab_lib/pyproject.toml:7`, `src/trading_crab/__init__.py:8`,
  `src/trading_crab_lib/__init__.py:16`, `CLAUDE.md:266`, `.claude/CLAUDE.md:84`.
- Historical records that must NOT be rewritten: `CLAUDE.md:1922` (D47 decision log),
  everything under `docs/archive/`, everything under `.planning/`.
- No test anywhere asserts a version string (`grep __version__ tests/` is empty), so no test
  needs updating.
- `dist/`, `build/` and `*.egg-info/` are already in `.gitignore` and none are tracked.
- `PyYAML` is importable from `./.venv/bin/python`; `build` was installed into that venv this
  session via `uv pip install --python .venv/bin/python build` (uv is at `/root/.local/bin/uv`,
  and network egress through the agent proxy works).
- `yaml.safe_load` parses the workflow's `on:` key as the boolean `True` (YAML 1.1 behavior).
  Index steps via `w["jobs"]["publish"]["steps"]`, never via `w["on"]`.
</context>

<tasks>

<task type="tracer">
  <name>Task 1: Bump both packages to 0.1.4 and fix the dist-directory expression — prove build → glob end-to-end</name>
  <files>pyproject.toml, src/trading_crab_lib/pyproject.toml, src/trading_crab/__init__.py, src/trading_crab_lib/__init__.py, CLAUDE.md, .claude/CLAUDE.md, .github/workflows/publish-pypi.yml</files>
  <precondition>`.venv/bin/python -m build --version` succeeds; if not, run `/root/.local/bin/uv pip install --python .venv/bin/python build` first.</precondition>
  <reversibility rating="reversible">Version strings and a single workflow line; revert is a `git checkout` of six files.</reversibility>
  <action>
Wire the one path that failed in CI, end to end: correct version declared → package builds →
the workflow's dist-directory expression finds the built files.

**Version bump (0.1.2 → 0.1.4) at exactly these live sites, and nowhere else:**
- `pyproject.toml` `[project] version`
- `pyproject.toml` `[project.dependencies]` — the `trading-crab-lib>=0.1.2` entry becomes
  `trading-crab-lib>=0.1.4`. The Poetry section uses a path dependency with no version
  constraint; leave it alone.
- `pyproject.toml` line 112 — the explanatory comment that restates the pip/uv pin. Keep it
  consistent with the pin above it.
- `src/trading_crab_lib/pyproject.toml` `[project] version`
- `src/trading_crab/__init__.py` `__version__`
- `src/trading_crab_lib/__init__.py` `__version__`
- `CLAUDE.md` line 266 (the "Two-Package Architecture" prose stating the live pin) — this is a
  live statement of current state, not a record of the past.
- `.claude/CLAUDE.md` line 84 (the Key Dependencies entry restating the live pin).

Do NOT touch `CLAUDE.md` line 1922 (inside the D47 decision-log entry, "since v0.1.2"),
anything under `docs/archive/`, or anything under `.planning/`. Those are records of the past
and rewriting them would falsify project history. Also leave
`src/trading_crab.egg-info/requires.txt` — it is a gitignored build artifact that regenerates.

**Workflow fix — the reported failure.** In the `Publish ${{ matrix.name }} to PyPI` step, the
`DIST_DIR` assignment currently wraps `matrix.build_dir` in a GitHub Actions conditional
expression and then appends a second path segment unconditionally. Because that conditional
already substitutes the literal directory name for the app leg, the result is a doubled path
and `twine` cannot expand the glob. Replace the whole right-hand side with the direct form
`DIST_DIR="${{ matrix.build_dir }}/dist"`. This is uniformly correct for both legs because
`python -m build <srcdir>` always writes to `<srcdir>/dist`: the app leg's build_dir is `.`
so it resolves to the repo-root output directory, and the lib leg resolves to
`src/trading_crab_lib/dist`.

Do not change the tag conventions, the matrix structure, or `--skip-existing` in this task.
Do not switch to OIDC / trusted publishing.

**Build both packages and prove the glob.** Using `./.venv/bin/python`, run `python -m build .`
and `python -m build src/trading_crab_lib`, logging output under the scratchpad directory. Then,
for each leg, expand the exact glob the corrected expression produces and assert it matches at
least one file and that every matched filename contains `0.1.4`. If a build fails for a reason
unrelated to this change (network, toolchain), stop and report that plainly in the summary —
do not claim the check passed.
  </action>
  <verify>
    <automated>cd /home/user/claude-scratch-work && SP=/tmp/claude-0/-home-user-claude-scratch-work/2b704a4a-9590-5ab7-b2d2-213af4621a1f/scratchpad && .venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/publish-pypi.yml')); print('YAML OK')" && grep -q 'DIST_DIR="\${{ matrix.build_dir }}/dist"' .github/workflows/publish-pypi.yml && echo "DIST_DIR form OK" && [ "$(grep -v '^[[:space:]]*#' .github/workflows/publish-pypi.yml | grep -c 'dist/dist')" -eq 0 ] && echo "no doubled path OK" && [ "$(grep -v '^[[:space:]]*#' .github/workflows/publish-pypi.yml | grep -c "build_dir == ")" -eq 0 ] && echo "ternary removed OK" && grep -q '^version = "0.1.4"' pyproject.toml && grep -q '^version = "0.1.4"' src/trading_crab_lib/pyproject.toml && grep -q '__version__ = "0.1.4"' src/trading_crab/__init__.py && grep -q '__version__ = "0.1.4"' src/trading_crab_lib/__init__.py && grep -q 'trading-crab-lib>=0.1.4' pyproject.toml && echo "versions OK" && [ "$(grep -rc '0\.1\.2' pyproject.toml src/trading_crab_lib/pyproject.toml src/trading_crab/__init__.py src/trading_crab_lib/__init__.py | grep -v ':0$' | wc -l)" -eq 0 ] && echo "no stale 0.1.2 in version files OK" && .venv/bin/python -m build . > "$SP/build_app.log" 2>&1 && .venv/bin/python -m build src/trading_crab_lib > "$SP/build_lib.log" 2>&1 && for d in "." "src/trading_crab_lib"; do n=$(ls "$d"/dist/* 2>/dev/null | wc -l); [ "$n" -ge 2 ] || { echo "FAIL: glob $d/dist/* matched $n files"; exit 1; }; ls "$d"/dist/* | grep -q '0\.1\.4' || { echo "FAIL: $d artifacts do not carry 0.1.4"; exit 1; }; ls "$d"/dist/* | grep -v '0\.1\.4' && { echo "FAIL: stale non-0.1.4 artifact in $d/dist"; exit 1; }; echo "glob OK for $d/dist/*: $n files"; done</automated>
  </verify>
  <done>Both pyproject files and both `__init__.py` files declare 0.1.4; the app pins `trading-crab-lib>=0.1.4`; the workflow's dist-directory expression is the direct matrix reference with no conditional and no doubled segment; `python -m build` succeeds for both packages and the corrected glob matches ≥2 files per leg, every one carrying 0.1.4. Historical version references in `CLAUDE.md`'s decision log, `docs/archive/` and `.planning/` are untouched.</done>
</task>

<task type="auto">
  <name>Task 2: Make skips loud and add two hard publish guards — then prove the guards fail when they should</name>
  <files>.github/workflows/publish-pypi.yml</files>
  <precondition>Task 1 has built both packages, so `dist/` and `src/trading_crab_lib/dist/` contain real 0.1.4 artifacts for the guard to run against.</precondition>
  <reversibility rating="reversible">Additive workflow steps plus logging lines; revert is a `git checkout` of one file.</reversibility>
  <action>
Two defects remain: a skipped matrix leg is indistinguishable from a successful publish, and
`--skip-existing` turns a missed version bump into a green run that uploads nothing. Fix both,
and make the fix testable by keeping the new bash free of GitHub Actions template expressions.

**Structural rule for this task:** every `run:` body you add or edit must contain zero `${{ }}`
expressions. Hoist all matrix and context values into the step's `env:` block instead
(`PKG`, `BUILD_DIR`, `TAG_PREFIX`, `EVENT_NAME`, `INPUT_PKG`). This is what makes the shipped
bash extractable and runnable locally — a paraphrase of the logic would not be verification.
`GITHUB_REF` and `GITHUB_OUTPUT` are already real environment variables and need no hoisting.

**(a) Loud gate.** Rewrite the `Decide whether to publish this package` step's `run:` body to
read its inputs from the new `env:` block, preserving the existing decision logic exactly
(workflow_dispatch input matching; the both-v tag publishing every leg; otherwise a tag-prefix
match). Keep the step `id: gate` and keep writing `publish=true` / `publish=false` to
`$GITHUB_OUTPUT`. Add exactly one greppable outcome line on every path, written to stdout:
a line beginning `PUBLISH:` naming the package and the tag when the leg will publish, and a
line beginning `SKIP:` naming the package, the tag, the expected prefix, and the reason when
it will not. Every branch must emit one of the two — a silent branch reintroduces the bug.

**(b) New guard step**, inserted between the build step and the publish step, with a static
name `Verify built artifacts` (static so a local extractor can find it by name), carrying the
same `if: steps.gate.outputs.publish == 'true'` condition as its neighbours. Its body:

1. Enable `set -euo pipefail` and `shopt -s nullglob`, then collect `"$BUILD_DIR"/dist/*` into
   an array. If the array is empty, print an error naming the searched directory and exit
   non-zero. Otherwise print the count and each filename.
2. If `EVENT_NAME` is the manual-dispatch event, print that there is no tag to compare against,
   and exit 0 — dispatch runs legitimately have no tag.
3. Otherwise derive the tag from `GITHUB_REF` by stripping the `refs/tags/` prefix, then derive
   the tag version: when the tag starts with the both-packages prefix strip that; otherwise
   strip `$TAG_PREFIX`. Assign `TAG_PREFIX` to a shell variable before using it in the
   parameter expansion so no template text lands inside the expansion.
4. For each artifact, derive the version from the filename: for a `.whl`, take the second
   hyphen-delimited field (wheel filenames escape hyphens in the distribution name to
   underscores, so this is unambiguous); for a `.tar.gz`, strip the extension and take the
   text after the last hyphen. Any other extension is an error. If a derived version differs
   from the tag version, exit non-zero with a message naming the package, the offending
   filename, the artifact version, the tag, and the tag version, and telling the reader to bump
   the version in that package's pyproject or retag.
5. On success print a single confirmation line naming the matched version.

Never echo `TWINE_PASSWORD` or any secret, and do not enable shell tracing in any step.

**(c) Keep `--skip-existing`** on the upload — it is what makes both-packages tag reruns safe.
The two new guards are what stop it from masking a missed bump. Leave the publish step's
`DIST_DIR` line as corrected in Task 1.

**Local proof that the guards fire.** Write a verification script into the scratchpad that
loads the workflow with `yaml.safe_load`, looks up the two steps by name inside
`w["jobs"]["publish"]["steps"]`, and writes each step's `run:` string verbatim to a `.sh` file —
extracted, never retyped. Then drive those extracted scripts with environment variables:

- Gate, tag `refs/tags/v0.1.4`: the app leg must print a `PUBLISH:` line and write
  `publish=true`; the lib leg must print a `SKIP:` line and write `publish=false`.
- Gate, tag `refs/tags/lib-v0.1.4`: the mirror image of the above.
- Gate, tag `refs/tags/both-v0.1.4`: both legs print `PUBLISH:` and write `publish=true`.
- Guard, app leg, tag `refs/tags/v0.1.4` against the real `dist/`: exits 0.
- Guard, lib leg, tag `refs/tags/lib-v0.1.4` against the real `src/trading_crab_lib/dist/`:
  exits 0.
- Guard, app leg, tag `refs/tags/v0.1.3` against the same real `dist/`: **must exit non-zero**
  and its output must name both versions. This is the case that would have caught the actual
  release failure.
- Guard, tag `refs/tags/both-v0.1.4` against the real `dist/`: exits 0, proving the both-prefix
  branch strips correctly.
- Guard against an empty scratch directory: **must exit non-zero**.

A guard that never fires is not verified — the two must-fail scenarios are the point of this
task, not an afterthought.

**Cleanup.** After the guard scenarios pass, remove `dist/`, `src/trading_crab_lib/dist/`, and
any `*.egg-info/` directories produced by the builds, so nothing built lands in the commit.
They are gitignored, but leaving them invites a stale-artifact mismatch on the next run.
  </action>
  <verify>
    <automated>cd /home/user/claude-scratch-work && SP=/tmp/claude-0/-home-user-claude-scratch-work/2b704a4a-9590-5ab7-b2d2-213af4621a1f/scratchpad && .venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/publish-pypi.yml')); print('YAML OK')" && .venv/bin/python -c "
import yaml,sys
w=yaml.safe_load(open('.github/workflows/publish-pypi.yml'))
steps={s.get('name'):s for s in w['jobs']['publish']['steps']}
for n in ('Decide whether to publish this package','Verify built artifacts'):
    assert n in steps, 'missing step: '+n
    body=steps[n]['run']
    assert '\${{' not in body, 'template expression leaked into run body of '+n
    open(sys.argv[1]+'/'+('gate' if 'Decide' in n else 'guard')+'.sh','w').write(body)
i=[k for k,s in enumerate(w['jobs']['publish']['steps']) if s.get('name')]
names=[s.get('name') for s in w['jobs']['publish']['steps']]
assert names.index('Verify built artifacts') > names.index('Build \${{ matrix.name }}'), 'guard must run after build'
assert names.index('Verify built artifacts') < names.index('Publish \${{ matrix.name }} to PyPI'), 'guard must run before publish'
print('extraction + ordering OK')
" "$SP" && bash "$SP/run_guard_scenarios.sh"</automated>
    <automated>cd /home/user/claude-scratch-work && [ ! -d dist ] && [ ! -d src/trading_crab_lib/dist ] && [ -z "$(find . -maxdepth 3 -name '*.egg-info' -not -path './.venv/*' -print -quit)" ] && echo "build artifacts cleaned OK" && [ -z "$(git status --porcelain --ignored=no | grep -E 'dist/|egg-info')" ] && echo "nothing built is staged OK"</automated>
  </verify>
  <done>The gate step prints exactly one `PUBLISH:` or `SKIP:` line on every branch and both are observed for both legs across the v / lib-v / both-v tag scenarios. `Verify built artifacts` sits between build and publish, contains no template expressions, exits 0 for matching versions on both legs and for the both-v prefix, and exits non-zero — with a message naming both versions — for a mismatched tag and for an empty dist directory. `--skip-existing` and the tag conventions are unchanged. No build artifacts remain in the working tree.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| git tag → GitHub Actions runner | An arbitrary tag string drives which package publishes and which version is expected; it is attacker-influenceable by anyone who can push a tag. |
| runner → PyPI | The runner holds long-lived API tokens and can publish under the project's name. |
| PyPI → downstream installers | Anything uploaded is immediately installable by users; PyPI uploads are irreversible. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-vyi-01 | Tampering | `twine upload` in publish-pypi.yml | high | mitigate | The `Verify built artifacts` guard fails the job before the publish step runs whenever the built artifact version disagrees with the version implied by the tag, so a stale-version tag can no longer reach PyPI as a silent no-op nor as an unintended release. |
| T-vyi-02 | Repudiation | matrix leg gating in publish-pypi.yml | medium | mitigate | Every gate branch emits one greppable `PUBLISH:` / `SKIP:` line naming the package, tag and reason, so the job log becomes an audit record of what was and was not published. Today a skip is indistinguishable from a success. |
| T-vyi-03 | Information Disclosure | job logs of the new gate and guard steps | medium | mitigate | New `run:` bodies echo only filenames, tag strings and version strings. `TWINE_PASSWORD` stays scoped to the publish step's own `env:` block and is never referenced in the guard. No step enables shell tracing. |
| T-vyi-04 | Elevation of Privilege | workflow permissions and PyPI tokens | low | accept | `permissions: contents: read` and the `pypi` environment-scoped `PYPI_LIB_TOKEN` / `PYPI_APP_TOKEN` secrets are unchanged. Migrating to OIDC trusted publishing would remove the long-lived tokens entirely but is explicitly out of scope for this task; tracked, not done here. |
| T-vyi-SC | Tampering | `build` installed into ./.venv for local verification | medium | mitigate | `build` is the PyPA-official build frontend; it is installed only into the pre-existing local `.venv` for verification and is added to no committed dependency file, so no new supply-chain surface is committed. No other package is installed by this plan. |
</threat_model>

<verification>
- The workflow YAML parses under `yaml.safe_load`.
- Both packages build locally with `./.venv/bin/python -m build`, and the glob the corrected
  dist-directory expression produces matches those real files for both legs.
- Every built artifact filename carries 0.1.4, and no non-0.1.4 artifact is left in either
  dist directory.
- The gate and guard bash bodies are extracted from the committed YAML — not retyped — and run
  against real dist directories.
- The version guard is exercised in both directions: it passes on a matching tag and fails on
  a deliberately mismatched one, and the empty-dist guard is exercised too.
- No build artifacts remain in the working tree at the end.
- No secret or token value appears in any committed file.
</verification>

<success_criteria>
- `.github/workflows/publish-pypi.yml`: dist-directory expression is the direct matrix
  reference; gate logs a `PUBLISH:`/`SKIP:` line on every branch; a `Verify built artifacts`
  step sits between build and publish and enforces both the non-empty-glob and
  version-matches-tag conditions; tag conventions, matrix structure and `--skip-existing`
  unchanged.
- `pyproject.toml`, `src/trading_crab_lib/pyproject.toml`, `src/trading_crab/__init__.py`,
  `src/trading_crab_lib/__init__.py` all declare 0.1.4; the app pins
  `trading-crab-lib>=0.1.4`; live prose in `CLAUDE.md` and `.claude/CLAUDE.md` matches.
- Historical version references (`CLAUDE.md` D47 log, `docs/archive/`, `.planning/`) untouched.
- Guard proven to fail on mismatch and on empty dist, with the failure output captured in the
  summary.
- Committed on branch `claude/exciting-keller-2xy11u`. No new branch, no tag, no push, no PR.
</success_criteria>

<output>
Create `.planning/quick/260910-vyi-fix-pypi-publish-workflow-dist-path-and-/260910-vyi-SUMMARY.md` when done.

The summary must record the actual observed guard failure output for the mismatched-tag
scenario (the evidence that the guard fires), and the actual built artifact filenames. If any
build or guard scenario could not be run for an environment reason, say so explicitly rather
than reporting it as passed.

Reminder for the executor: do NOT create a pull request, do NOT create or push a git tag, do
NOT push. Stay on `claude/exciting-keller-2xy11u`.
</output>
