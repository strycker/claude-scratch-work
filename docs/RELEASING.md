# Releasing to PyPI

The single source of truth for releasing `trading-crab` (the app) and
`trading-crab-lib` (the library) to PyPI. This document may be superseded
by future edits to `.github/workflows/publish-pypi.yml` — **if this
document and the committed workflow ever disagree, the workflow is the
ultimate source of truth.** Re-check it before trusting a section here.

## 1. Tag conventions and the `v*` trap

The workflow reacts to three tag prefixes:

| Tag prefix | Publishes |
|---|---|
| `v0.1.4` | `trading-crab` (the app) only |
| `lib-v0.1.4` | `trading-crab-lib` (the library) only |
| `both-v0.1.4` | both packages, in one run |

**The trap that has already been hit once on this project:** pushing a `v*`
tag **silently skips the lib leg of the matrix**. A skipped matrix leg
reports **GREEN** in the GitHub Actions UI — it looks exactly like an
intentional, successful publish of that package at a glance, because a
skipped job is not a failed job.

**How to check which legs actually ran:** open the workflow run, expand
each matrix leg's `Decide whether to publish this package` step, and read
its own log line. Every leg prints exactly one of:

- `PUBLISH: <package> tag=<tag>` — this leg actually built and uploaded.
- `SKIP: <package> tag=<tag> prefix=<prefix> reason=...` — this leg did
  nothing this run.

If you intended to publish both packages and only see `PUBLISH:` for one
of them, you used the wrong tag prefix (probably `v*` when you meant
`both-v*`).

## 2. Version-bumping discipline

Both `pyproject.toml` files carry independent version numbers and **must
both be bumped before tagging**, in lockstep with whichever tag prefix you
are about to push:

- Root `pyproject.toml` — `trading-crab` (the app).
- `src/trading_crab_lib/pyproject.toml` — `trading-crab-lib` (the library).

The workflow's `Verify built artifacts` step now enforces tag-vs-artifact
version agreement: it extracts the version encoded in every built wheel and
sdist filename and compares it against the version implied by the tag you
pushed. **If they disagree, the job fails before any upload happens** — but
only for the leg whose tag prefix matched. Bump the version file *before*
tagging, not after; a tag is immutable once pushed to a public remote in
any meaningful sense (see below).

## 3. Burned versions are permanent

A version number once uploaded to PyPI (or TestPyPI) can **never be
re-uploaded**, even after deleting the release from the PyPI web UI. PyPI
retains the version number as burned forever, for that project.

This project has already burned **two** real versions this way: **0.1.3 and
0.1.4 are permanently unusable on PyPI for either package.** 0.1.3 was burned
by a failed publish attempt; 0.1.4 was published successfully but shipped a
`trading-crab-lib` wheel containing zero Python modules (see §9), which cannot
be corrected in place. The current version is 0.1.5. Treat every version bump
as one-shot: verify the build, the metadata, AND the installed artifact
(see §8, §9) before you tag, not after.

## 4. Validating a token without publishing

You do not need to perform a real, irreversible upload to find out whether
a PyPI API token is valid. The technique:

1. Pick an artifact whose version has **already been published** to the
   target index (e.g. a wheel for the current, already-live version).
2. Run `twine upload` against it **without** `--skip-existing`.
3. Read the response:
   - **`400 File already exists`** — the token authenticated
     successfully. The rejection is PyPI's normal "you can't re-upload an
     existing file" behavior, not an auth failure. **Good token.**
   - **`403 Forbidden`** — the token itself was rejected before PyPI ever
     got to check the filename. **Bad token** (wrong scope, revoked,
     mistyped, or pointed at the wrong project).

**The trap that defeats this technique:** the workflow's real publish step
always passes `--skip-existing`. `--skip-existing` makes twine call an
**unauthenticated** pre-check against the index first, and if the file
already exists, twine reports success and `continue`s **without ever
authenticating**. Leaving `--skip-existing` on while doing this manual
check means you get a "success" response regardless of whether the token
is any good at all — you have validated nothing. Turn `--skip-existing`
off for this specific manual check, and only for this check; never make it
a permanent change to the workflow's real publish step (skip-existing
there exists so re-running a leg that partially succeeded doesn't fail on
files that already uploaded).

## 5. Local `~/.pypirc` layout

Because this project publishes two independent packages to two independent
registries (production PyPI and TestPyPI), a local `~/.pypirc` needs
**four distinct sections**, one per (package × registry) combination:

```ini
[distutils]
index-servers =
    trading-crab-app
    trading-crab-lib
    trading-crab-app-test
    trading-crab-lib-test

[trading-crab-app]
repository = https://upload.pypi.org/legacy/
username = __token__
password = <PYPI_APP_TOKEN value>

[trading-crab-lib]
repository = https://upload.pypi.org/legacy/
username = __token__
password = <PYPI_LIB_TOKEN value>

[trading-crab-app-test]
repository = https://test.pypi.org/legacy/
username = __token__
password = <TEST_PYPI_API_TOKEN value>

[trading-crab-lib-test]
repository = https://test.pypi.org/legacy/
username = __token__
password = <TEST_PYPI_API_TOKEN value>
```

**Why every section needs an explicit `repository = ` line:** twine's own
config loader only auto-fills the registry URL for the two *literal*
section names `pypi` and `testpypi`. This project needs two sections per
registry (one per package, since each package has its own scoped token),
so there is no way to use the literal name `pypi` twice. Any custom
section name — which this four-way split requires — must specify its
`repository = ` URL explicitly, or `twine upload -r <section>` fails with a
missing-configuration error.

Replace every `<... value>` placeholder above with your actual token. Never
commit a real token to this file or any other file in this repository.

## 6. The TestPyPI rehearsal path

To rehearse a publish without touching production PyPI:

1. Go to the workflow's **Actions → Publish to PyPI → Run workflow** page.
2. Select the package(s) to rehearse via the existing `package` input.
3. Set the `target` input to **`testpypi`** — this is already the
   default, so simply leaving it alone gives you a rehearsal.
4. Run the workflow. It builds, runs `twine check --strict`, and uploads
   to `https://test.pypi.org/legacy/` using the `TEST_PYPI_API_TOKEN`
   secret — production PyPI is never touched.

**A tag push can never reach this path, by design.** The `target` input
only exists on `workflow_dispatch`; a tag push event has no such input to
read at all, so the gate step hardcodes `target=pypi` unconditionally for
every tag-triggered run. There is no tag syntax, flag, or input that
redirects a tag push to TestPyPI.

## 7. Where the secrets live, and the empty-string trap

Three repository secrets are required, exactly as named in the workflow's
header comment:

| Secret | Scope |
|---|---|
| `PYPI_APP_TOKEN` | Production PyPI token for `trading-crab` |
| `PYPI_LIB_TOKEN` | Production PyPI token for `trading-crab-lib` |
| `TEST_PYPI_API_TOKEN` | TestPyPI token, shared by both packages, used only for `workflow_dispatch` rehearsals |

Each may live in either of two valid locations:

- A **repository secret** (Settings → Secrets and variables → Actions →
  Repository secrets), or
- An **environment secret** scoped to the `pypi` environment (Settings →
  Environments → `pypi` → Environment secrets) — the job declares
  `environment: pypi`, so either location resolves.

**The empty-string trap:** GitHub Actions' dynamic `secrets[...]` lookup
(used by both the token-presence guard and the publish step to select the
target-appropriate secret name) resolves an **unknown or mistyped** secret
name to an **empty string**, not an error. A typo in a secret's name is
therefore indistinguishable, from the workflow's perspective, from that
secret never having been created at all. This is exactly why the
token-presence guard step exists: it turns that silent empty-string case
into a named, fast, loud failure (`ERROR: PyPI token for <package>
(target: <target>) is missing or empty. Expected secret name: <name>.`)
before any build or upload work happens, rather than letting the failure
surface later as a confusing authentication error from PyPI itself.

## 8. The `twine check --strict` gate

Every release now runs `twine check --strict` against the real built
`dist/` artifacts, gated identically to the build/upload steps, positioned
after `Verify built artifacts` and before the actual upload. This is a
different failure category than the tag/version checks — it validates
package *metadata* (README rendering, classifiers, URLs), not
tag-vs-version agreement.

This is exactly the check that would have caught, before any real
publish, the defect this project actually shipped once: `trading-crab-lib`
uploaded cleanly to PyPI with no `readme` key wired into its package
metadata, producing a **blank PyPI project page** with no rendered
description. That defect is now fixed (the library ships a real README,
wired via `readme = "README.md"` in its `pyproject.toml`), and this gate
exists so the same class of defect fails CI instead of shipping silently
again.

## 9. A green pipeline can still ship an empty package

**Releases 0.1.0 through 0.1.4 of `trading-crab-lib` shipped ZERO Python
modules.** The wheel was 4,836 bytes: four metadata files and nothing else.
`pip install trading-crab-lib==0.1.4` succeeded, then `import trading_crab_lib`
raised `ModuleNotFoundError`.

Every gate in the pipeline passed, because none of them look at what is
actually inside the artifact:

| Gate | What it inspects | Catches an empty wheel? |
|---|---|---|
| `python -m build` exit code | that the build ran | no |
| `twine check --strict` | metadata (name, version, description) | no |
| version-vs-tag guard | the *filename* | no |
| `build-pkg` job in `python-package.yml` | that the build ran | no |

An empty wheel has flawless metadata and a flawless filename. The only
evidence that an artifact is usable is **installing it and importing it**.
That is what the `Smoke-test the built wheel` step does, and it deliberately
`cd`s off the checkout first — importing from inside the repo would succeed
even if the wheel were empty, which is precisely the failure being tested for.

**Root cause, for future reference.** `src/trading_crab_lib/` is both the
project root (where its `pyproject.toml` lives) and the package's own content
directory. The old config tried to paper over that with
`[tool.setuptools.packages.find] where = [".."]`, re-discovering the package
one level up. Modern setuptools refuses to let `where` escape the project root,
so it silently returned an empty package list and the build exited 0. The fix
is an explicit `package-dir = {"trading_crab_lib" = "."}` plus an enumerated
`packages` list. **Do not reintroduce a discovery glob there.**

### Inspecting a built wheel by hand

```bash
python -m build --wheel --outdir /tmp/w src/trading_crab_lib
unzip -l /tmp/w/*.whl | tail -3          # file count — 4 means it is empty
unzip -l /tmp/w/*.whl | grep '\.py$' | head

# the real check: install it somewhere with no repo on sys.path
python -m venv /tmp/v && /tmp/v/bin/pip install /tmp/w/*.whl
cd /tmp && /tmp/v/bin/python -c "import trading_crab_lib; print(trading_crab_lib.__file__)"
```
