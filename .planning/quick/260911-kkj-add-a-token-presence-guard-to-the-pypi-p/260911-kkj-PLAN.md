---
phase: quick-260911-kkj
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - .github/workflows/publish-pypi.yml
autonomous: true
requirements: [QUICK-260911-kkj]
user_setup: []

estimate:
  tokens: 22000
  raw_tokens: 22000
  tasks: 2
  confidence: low        # gsd-tools estimate-calibration: sample_count=0, factor=1.0, applied=false

must_haves:
  truths:
    - "A matrix leg whose PyPI token secret is absent or empty fails the job before any twine invocation, with a message naming the exact secret that must be created."
    - "The failure message explains that a dynamic secrets[...] lookup resolves to an empty string on a name mismatch rather than erroring, so the reader knows an exact-name typo is a candidate cause."
    - "The failure message names both places the secret may live: a repository secret, or an environment secret on the pypi environment."
    - "A leg whose token is present succeeds the guard and the token value appears nowhere in the guard's output."
    - "The guard is exercised in both directions against the bash extracted verbatim from the committed YAML — not against a retyped paraphrase."
    - "A matrix leg the gate decided to skip does not run the guard and therefore cannot fail on a token it would never use."
    - "The guard reads the token through the same secrets[matrix.secret_name] expression the publish step uses, so a pass proves the publish step will receive a non-empty value."
    - "Nothing else in the workflow changes: tag conventions, matrix, gate logic, dist path, version guard, and --skip-existing are byte-identical to their pre-task state."
  artifacts:
    - .github/workflows/publish-pypi.yml
  key_links:
    - "guard step env TWINE_PASSWORD ↔ publish step env TWINE_PASSWORD (must be the identical expression, or the guard proves nothing about the value the upload gets)"
    - "guard step if: condition ↔ steps.gate.outputs.publish (same condition as every other publish-path step)"
    - "guard step run: body ↔ the locally-extracted script under test (extracted via yaml.safe_load, never retyped)"
    - "matrix.secret_name ↔ the secret name printed in the error message (the remedy must name the concrete secret for that leg)"
---

<objective>
Add a token-presence guard to `.github/workflows/publish-pypi.yml` so a missing or empty PyPI API
token fails the job immediately with an actionable message, instead of reaching `twine upload` and
surfacing as an opaque authentication error that does not name the real cause.

Purpose: `TWINE_PASSWORD` is supplied by a dynamic `secrets[matrix.secret_name]` lookup. GitHub
resolves an unknown key in the secrets context to the empty string rather than failing, so a
secret that is absent, scoped to the wrong place, or named with a single character off produces
a silent empty password and a downstream 403 that blames the credential rather than the wiring.
Neither `PYPI_APP_TOKEN` nor `PYPI_LIB_TOKEN` has ever been exercised by a real run, so this is
the most likely next failure mode on the release path.

Output: one new gated workflow step that fails fast with a remedy-naming message, proven to fire
on an absent token, to pass on a present one, and to leak nothing in either case.
</objective>

<execution_context>
@/Users/glestryc/personal/github_repos/claude-scratch-work/.claude/gsd-core/workflows/execute-plan.md
@/Users/glestryc/personal/github_repos/claude-scratch-work/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.github/workflows/publish-pypi.yml
@.planning/quick/260910-vyi-fix-pypi-publish-workflow-dist-path-and-/260910-vyi-PLAN.md
@CLAUDE.md

**Facts already established this session — do not re-derive:**

- The immediately preceding quick task (260910-vyi) is merged to main at commit `4cbc829`. It
  fixed the doubled dist path, added the loud `PUBLISH:`/`SKIP:` gate, and added the
  `Verify built artifacts` version guard. All of that is in scope to preserve, not to revisit.
- Both packages are at version **0.1.4** and 0.1.4 is unpublished. No version changes here.
- Current branch is `claude/exciting-keller-2xy11u`, freshly reset onto `origin/main`. Stay on it.
- `PyYAML 6.0.3` is importable from `./.venv/bin/python` — confirmed by running it this session.
- `yaml.safe_load` parses this workflow's `on:` key as the boolean `True` (YAML 1.1 behavior).
  Index steps via `w["jobs"]["publish"]["steps"]`, never via `w["on"]`.
- The convention established by 260910-vyi and enforced here: any `run:` body this task adds or
  edits carries zero GitHub Actions template expressions; every matrix and context value is
  hoisted into that step's own `env:` block. `Verify built artifacts` is the pattern to copy.
  This is what makes the shipped bash extractable by `yaml.safe_load` and runnable standalone.
- `GITHUB_REF` and `GITHUB_OUTPUT` are real environment variables on the runner and need no
  hoisting; `secrets` and `matrix` are template contexts and do.
- Scratchpad for all temporary scripts and captured output:
  `/tmp/claude-0/-home-user-claude-scratch-work/2b704a4a-9590-5ab7-b2d2-213af4621a1f/scratchpad`
- This repo has a documented history of sign-offs against evidence that could not detect a wrong
  value (`.planning/UAT-AUDIT-2026-09-09.md`). Extracted-and-executed bash is the standard here;
  reading the diff and declaring it correct is not.
</context>

<tasks>

<task type="tracer">
  <name>Task 1: Add the token-presence guard step and prove the shipped bash runs end-to-end</name>
  <files>.github/workflows/publish-pypi.yml</files>
  <reversibility rating="reversible">One additive workflow step; revert is a `git checkout` of a single file.</reversibility>
  <action>
Wire the one path that matters, end to end: a gated workflow step whose bash is extractable from
the committed YAML and executes correctly outside GitHub Actions.

**Placement decision — a separate step, not an addition to the publish step's `run:` body.**
Implement this as its own step. Three reasons, all of which should survive review:

1. *Failure naming is the entire point of the task.* Folding the check into
   `Publish ... to PyPI` leaves the GitHub UI reporting that the publish step failed — which is
   indistinguishable at a glance from the opaque twine auth error being replaced. A step named
   for the token makes the job summary itself the diagnosis.
2. *It protects an out-of-scope line.* The publish step's `run:` body still contains a template
   expression for the dist directory, which 260910-vyi fixed and verified. Editing that body to
   satisfy the zero-expression convention would mean rewriting the dist-path line this task is
   explicitly forbidden to touch. A separate step leaves that body byte-identical.
3. *The extra exposure is bounded and is the price of testability.* The cost of a separate step
   is that the secret is injected into one more step's environment. That surface is narrow: the
   secret is already available to every step of this job via the `pypi` environment, GitHub only
   materializes it in steps that reference it, both steps carry the same publish gate so it is
   never injected on a skipped leg, and the guard never dereferences the value beyond an
   emptiness test.

**Position in the step list — immediately after the gate step, before `Set up Python`.**
The task is to fail *fast*. A wiring fault that is knowable in seconds should not cost a Python
setup, a dependency install, and two package builds first. The `if:` condition depends only on
`steps.gate.outputs.publish`, which is available from that point onward, so nothing blocks the
earlier position.

**The step.** Give it a static `name` — no template expression in the name, so a local extractor
can find it by exact string the way it finds `Verify built artifacts`. Carry
`if: steps.gate.outputs.publish == 'true'`, character-for-character the same condition as the
other publish-path steps. Its `env:` block hoists exactly three values: the token, read through
the *same* `secrets[matrix.secret_name]` expression the publish step uses (a different expression
would make a passing guard meaningless), the matrix secret name, and the matrix package name.

**The `run:` body.** Zero template expressions. Enable `set -euo pipefail`. Test emptiness with a
default-substituting parameter expansion on the token variable — the bare form would trip the
`nounset` option and abort with an unbound-variable message instead of the actionable one, which
would satisfy "exits non-zero" while failing the actual requirement. On the empty branch, write
an error to stderr and exit non-zero. On the non-empty branch, print one confirmation line naming
the package and the secret name, stating that the value is deliberately not shown, and exit 0.

**Message content.** The error must be a remedy, not a diagnosis. It must contain, at minimum:
the concrete secret name for this leg taken from the hoisted variable; the phrase
`repository secret`; the phrase `environment secret`; the environment name `pypi`; and an
explanation containing the phrase `empty string` covering the fact that the dynamic secrets
lookup resolves an unknown or mistyped key to an empty value rather than erroring, so an
exact-name mismatch looks identical to an absent secret.

**Secret hygiene — non-negotiable.** The body must never echo the token, never print its length,
never print a prefix or suffix, never pipe it to another command, and must not enable shell
tracing. GitHub's log masking is a backstop for accidents, not permission to print. The only
operation performed on the value is the emptiness test.

**Everything else stays.** Do not touch the tag conventions, the matrix, the gate step, the dist
path, the `Verify built artifacts` version guard, `--skip-existing`, the `permissions` block, or
the header comment block. Do not switch to OIDC or trusted publishing. Do not touch
`.github/workflows/python-package.yml` — its Python matrix stays at 3.10 through 3.13, because
both packages still declare a 3.10 floor. Do not change any package version.

**Then prove the shipped bash actually runs.** Write a Python extractor into the scratchpad as
`extract_token_guard.py`. It must load the workflow with `yaml.safe_load`, index
`w["jobs"]["publish"]["steps"]`, locate the guard by its exact static name, and then assert
structure before writing anything: that the body carries no template expressions; that the step's
`if:` string equals the one on `Verify built artifacts`; that the guard's index in the step list
is greater than the gate step's and less than the publish step's; and that the guard's
`env` entry for the token is the identical string to the publish step's `env` entry for the token.
It then writes that `run:` string verbatim to `token_guard.sh`. Extract, never retype — a harness
that runs different text than the workflow ships proves nothing about the workflow.

Finish the task by driving the extracted script once with a non-empty token value to confirm it
exits 0. The adversarial scenarios are Task 2's job.

Never write a real token or any real secret into any file, scratchpad included. Use an obviously
fake value.
  </action>
  <verify>
    <automated>cd /home/user/claude-scratch-work && SP=/tmp/claude-0/-home-user-claude-scratch-work/2b704a4a-9590-5ab7-b2d2-213af4621a1f/scratchpad && ./.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/publish-pypi.yml')); print('YAML OK')" && ./.venv/bin/python "$SP/extract_token_guard.py" "$SP" && test -s "$SP/token_guard.sh" && echo "extraction OK" && TWINE_PASSWORD=not-a-real-token SECRET_NAME=PYPI_APP_TOKEN PKG=trading-crab bash "$SP/token_guard.sh" && echo "happy path exit 0 OK"</automated>
  </verify>
  <done>`.github/workflows/publish-pypi.yml` parses under `yaml.safe_load` and contains one new statically-named step, positioned after the gate step and before the publish step, gated on the identical `steps.gate.outputs.publish == 'true'` condition, whose `env:` binds the token via the same `secrets[matrix.secret_name]` expression as the publish step, and whose `run:` body contains no template expressions. The extractor confirms all of those structural properties from the committed YAML, and the extracted body exits 0 when given a non-empty token.</done>
</task>

<task type="auto">
  <name>Task 2: Prove the guard fires on an absent token, leaks nothing on a present one, and changed nothing else</name>
  <files>.github/workflows/publish-pypi.yml</files>
  <reversibility rating="reversible">Verification-only unless a scenario exposes a defect in Task 1's step, in which case the fix is confined to the same single step.</reversibility>
  <action>
A guard that has never been observed failing is not a guard. Drive the extracted script through
the full scenario matrix and capture real output.

Write `run_token_guard_scenarios.sh` into the scratchpad. It must re-run `extract_token_guard.py`
first so it always tests the current committed YAML rather than a stale extraction, then drive
`token_guard.sh` through every scenario below, capturing stdout and stderr together to a per-
scenario file. Each scenario asserts an exit status and asserts on the captured text; a scenario
that cannot be asserted on is not a scenario.

**Must-fail scenarios — these are the point of the task, not an afterthought:**

- Token variable *entirely unset* in the child environment. Must exit non-zero. Its output must
  contain the leg's concrete secret name, the phrases `repository secret`, `environment secret`,
  `pypi`, and `empty string`. It must NOT contain any unbound-variable or `nounset` text — that
  outcome would mean the emptiness test is using the bare parameter form and the operator would
  get a shell diagnostic instead of a remedy.
- Token variable set to the empty string. Must exit non-zero, with the same message assertions.
- Both of the above repeated for the other matrix leg's secret name, confirming the message names
  whichever secret that leg actually depends on rather than a hardcoded one.

**Must-pass scenario, with the leak assertion:**

- Token set to a distinctive fake sentinel value chosen so that it could not plausibly occur in
  any normal guard output. Must exit 0. Then grep the captured output for that sentinel; the grep
  must find nothing. A found sentinel is an immediate failure of the task, not a warning. Feed
  the sentinel to the script through an environment variable defined inside the harness so the
  literal exists in exactly one place.

**Runner-fidelity scenario:**

- Token set to the empty string *and* the publish gate not applicable — confirm by inspection of
  the extracted YAML, not by running bash, that the step's `if:` condition is the same string as
  the other publish-path steps, so a skipped leg never reaches this script at all. Assert this in
  the extractor's structural checks rather than by simulating a skip; a skipped step produces no
  bash to run, and pretending otherwise would be theater.

**Scope assertions — prove nothing else moved.** In the same harness or alongside it:

- `git diff --name-only` against the merge-base of this branch lists `publish-pypi.yml` as the
  only changed workflow file, and no file under `.github/workflows/` other than that one appears
  in the diff at all.
- The dist-path line from 260910-vyi is still present in its corrected direct-matrix-reference
  form, and the workflow still contains zero occurrences of the doubled-dist-segment defect
  string outside comment lines — the exact literal is in the verify command; do not restate it
  in prose, because a restatement inside this file would be counted by any grep aimed here.
- The `Verify built artifacts` step and the gate step still exist by name, and `--skip-existing`
  is still on the upload.
- Both package versions still read 0.1.4 in both `pyproject.toml` files.
- `.github/workflows/python-package.yml` is unchanged and still lists the 3.10 through 3.13
  matrix.

When counting occurrences in the workflow file for any of these assertions, filter comment lines
out first — the header comment block discusses the secrets by name and would otherwise be counted
as if it were live configuration.

**Honesty requirement.** If any scenario cannot be executed for an environment reason — PyYAML
missing, bash unavailable, git history shallow — say so plainly in the summary and mark that item
unproven. Do not describe an unrun check as passing. Record the actual captured failure output
for at least one must-fail scenario in the summary; that text is the evidence the guard fires.

Leave no real token anywhere. Delete nothing from the repo; the scratchpad artifacts stay in the
scratchpad and are never committed.
  </action>
  <verify>
    <automated>cd /home/user/claude-scratch-work && SP=/tmp/claude-0/-home-user-claude-scratch-work/2b704a4a-9590-5ab7-b2d2-213af4621a1f/scratchpad && bash "$SP/run_token_guard_scenarios.sh"</automated>
    <automated>cd /home/user/claude-scratch-work && W=.github/workflows/publish-pypi.yml && ./.venv/bin/python -c "import yaml; w=yaml.safe_load(open('$W')); n=[s.get('name') for s in w['jobs']['publish']['steps']]; assert 'Verify built artifacts' in n, 'version guard step lost'; assert any(x and x.startswith('Decide whether to publish') for x in n), 'gate step lost'; print('preserved steps OK')" && grep -q 'DIST_DIR="\${{ matrix.build_dir }}/dist"' "$W" && echo "dist path preserved OK" && [ "$(grep -v '^[[:space:]]*#' "$W" | grep -c 'dist/dist')" -eq 0 ] && echo "no doubled path OK" && grep -q -- '--skip-existing' "$W" && echo "skip-existing preserved OK" && grep -q '^version = "0.1.4"' pyproject.toml && grep -q '^version = "0.1.4"' src/trading_crab_lib/pyproject.toml && echo "versions untouched OK" && [ -z "$(git diff --name-only origin/main -- .github/workflows/ | grep -v '^\.github/workflows/publish-pypi\.yml$')" ] && echo "only publish-pypi.yml changed OK"</automated>
  </verify>
  <done>Both must-fail scenarios (token unset; token empty) are observed exiting non-zero for both matrix legs, each printing the leg's own secret name plus the repository-secret, environment-secret, `pypi`, and `empty string` remedy text, and neither printing an unbound-variable diagnostic. The must-pass scenario exits 0 and a grep of its captured output for the sentinel finds nothing. The gate step, `Verify built artifacts` step, corrected dist path, `--skip-existing`, both 0.1.4 versions, and `python-package.yml` are all confirmed unchanged, and `publish-pypi.yml` is the only workflow file in the diff. Any scenario that could not run is reported as unproven rather than as passing.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| GitHub secrets store → workflow step environment | A long-lived PyPI API token crosses into a shell environment that also writes to a publicly readable job log. |
| workflow step → job log | Anything a step writes to stdout or stderr is retained and readable by anyone with repository read access. |
| runner → PyPI | The runner holds tokens that can publish irreversibly under the project's name. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-kkj-01 | Information Disclosure | new token-presence guard step, job log | high | mitigate | The guard performs exactly one operation on the token — a default-substituting emptiness test — and never echoes it, its length, or any substring. No step enables shell tracing. Task 2 asserts a distinctive sentinel token value is absent from the guard's full captured output, so the property is observed rather than asserted by inspection. |
| T-kkj-02 | Information Disclosure | scratchpad harness files | medium | mitigate | All scenarios use obviously fake token values; no real secret is written to any file, and harness artifacts live only in the scratchpad and are never committed. The plan forbids a real token anywhere. |
| T-kkj-03 | Spoofing | dynamic `secrets[matrix.secret_name]` lookup | high | mitigate | This is the defect being fixed: an unknown key silently yields an empty string, so a wrong or absent secret presents as a credential rejection rather than a wiring fault. The guard converts that into a named, pre-upload failure identifying the exact secret and both places it may be defined. |
| T-kkj-04 | Elevation of Privilege | secret injected into one additional step's environment | medium | accept | A separate step widens the token's exposure by one step. Accepted because the secret is already available to the whole job via the `pypi` environment, GitHub only materializes it into steps that reference it, the same publish gate prevents injection on a skipped leg, and the guard never dereferences the value. The alternative — folding the check into the publish step — would require rewriting an out-of-scope line and would leave the failure attributed to the upload. |
| T-kkj-05 | Tampering | publish path regressions from 260910-vyi | medium | mitigate | Task 2 asserts the gate step, `Verify built artifacts` step, corrected dist path, `--skip-existing`, and both 0.1.4 versions are intact, and that `publish-pypi.yml` is the only changed workflow file. |
| T-kkj-SC | Tampering | package-manager installs | low | accept | This plan installs no packages. PyYAML is already present in the pre-existing `./.venv` and was confirmed importable this session. No `## Package Legitimacy Audit` is required because no install task exists. |
</threat_model>

<verification>
- `.github/workflows/publish-pypi.yml` parses under `yaml.safe_load`, with steps indexed via
  `w["jobs"]["publish"]["steps"]` (the `on:` key parses as boolean `True`).
- The guard's `run:` body is extracted verbatim from the committed YAML and executed standalone.
  It is never retyped into the harness.
- Structural properties asserted from the parsed YAML: static step name, no template expressions
  in the body, `if:` identical to the other publish-path steps, position after the gate and
  before the publish step, and a token `env:` expression identical to the publish step's.
- Behavioral scenarios, all observed: token unset → non-zero with the remedy message; token empty
  → non-zero with the remedy message; token set to a sentinel → exit 0 with the sentinel absent
  from all captured output.
- Message content asserted positively: the leg's own secret name, `repository secret`,
  `environment secret`, `pypi`, `empty string`. Asserted negatively: no unbound-variable text.
- Regression scope asserted: gate step, version guard step, dist path, `--skip-existing`, both
  0.1.4 versions, and `python-package.yml` all unchanged; `publish-pypi.yml` is the only workflow
  file in the diff.
- Any check that cannot run for an environment reason is reported as unproven, never as passing.
- No real token or secret value is written to any file, committed or scratchpad.
</verification>

<success_criteria>
- One new statically-named, gated step in `.github/workflows/publish-pypi.yml` fails the job
  before `twine upload` when the leg's PyPI token is absent or empty, with a message naming the
  concrete secret, both valid locations for it, and the empty-string-on-mismatch behavior.
- The guard's `run:` body carries zero GitHub Actions template expressions; every matrix and
  secret value is hoisted into the step's `env:` block.
- The guard reads the token through the same `secrets[matrix.secret_name]` expression as the
  publish step, verified by string comparison of the two `env:` entries.
- Both must-fail scenarios and the must-pass-with-no-leak scenario are observed against bash
  extracted from the committed YAML, with real captured output recorded in the summary.
- Tag conventions, matrix, gate logic, dist path, version guard, `--skip-existing`, package
  versions, and `python-package.yml` are all unchanged.
- Committed on branch `claude/exciting-keller-2xy11u`. No new branch, no tag, no push, no PR.
</success_criteria>

<output>
Create `.planning/quick/260911-kkj-add-a-token-presence-guard-to-the-pypi-p/260911-kkj-SUMMARY.md` when done.

The summary must include the actual captured output of at least one must-fail scenario — the
evidence the guard fires — and must state explicitly that the sentinel grep on the must-pass
scenario found nothing. If any scenario could not be run for an environment reason, say so
plainly and mark it unproven rather than reporting it as passed.

Reminder for the executor: do NOT create a pull request, do NOT create or push a git tag, do NOT
push. Stay on `claude/exciting-keller-2xy11u`. Never write a real token into any file.
</output>
