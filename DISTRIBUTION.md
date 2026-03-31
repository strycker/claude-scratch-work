# Distribution Guide — Trading-Crab

How to package, distribute, and deploy this project in various formats.
This answers questions about PyPI, Docker, Maven, npm, and other distribution options.

---

## TL;DR

| Format | Status | Use case |
|--------|--------|---------|
| **PyPI** (`pip install`) | ✅ Current | Python developers, library consumers |
| **Docker** | 📋 Planned (Phase K) | Reproducible weekly runs, server deploy |
| **Maven** | ❌ N/A | Java ecosystem only |
| **npm** | ❌ N/A | JavaScript/Node.js ecosystem only |
| **GitHub Releases** | ✅ Current | Source archives + version tags |

---

## PyPI Distribution (Current)

This monorepo publishes **two independent Python packages** to PyPI:

| Package | pip name | Trigger |
|---------|----------|---------|
| `src/trading_crab_lib/` | `trading-crab-lib` | Push `lib-v*` tag |
| `src/trading_crab/` | `trading-crab` | Push `v*` tag (excl. `lib-v*`) |

### Release Workflow

```bash
# 1. Bump versions in both pyproject.toml files
#    src/trading_crab_lib/pyproject.toml  → version = "0.1.3"
#    pyproject.toml (root)               → version = "0.1.3"

# 2. Commit the version bump
git commit -am "chore: bump version to 0.1.3"
git push origin main

# 3. Tag and push — triggers the appropriate CI publish workflow
git tag lib-v0.1.3       # publishes trading-crab-lib
git push origin lib-v0.1.3

git tag v0.1.3           # publishes trading-crab
git push origin v0.1.3
```

The GitHub Actions workflows (`publish-lib.yml`, `publish-app.yml`) build and upload
to PyPI automatically. Requires `PYPI_LIB_TOKEN` and `PYPI_APP_TOKEN` secrets in the
repo's GitHub Actions settings.

### Installing from PyPI

```bash
# Library only (no CLI, no pipeline scripts)
pip install trading-crab-lib
pip install "trading-crab-lib[ingestion]"    # + FRED/multpl/yfinance
pip install "trading-crab-lib[all]"           # everything

# Full application (CLI + library)
pip install trading-crab
tradingcrab --help
```

### Two-Package Rationale

`trading-crab-lib` is a dependency of `trading-crab` but can be used independently.
This separation matters for consumers who want:
- Notebook-only usage of the transforms/clustering code (no pipeline overhead)
- Library integration into another application (e.g., a Streamlit dashboard or
  a backtesting framework that imports `trading_crab_lib.clustering` directly)
- Lighter installs: `trading-crab-lib[plotting]` without the pipeline's argparse CLI

---

## Docker Distribution (Phase K — Planned)

Docker is the preferred distribution format for the **weekly automated report** use case:
run the full pipeline on a schedule without managing a Python environment on the host.

### Planned Dockerfile (multi-stage)

```dockerfile
# ---- base: core scientific deps ----
FROM python:3.11-slim AS base
WORKDIR /app
COPY src/ src/
COPY pyproject.toml .
RUN pip install --no-cache-dir -e "src/trading_crab_lib/[all]" -e ".[dev]"

# ---- pipeline: full runtime ----
FROM base AS pipeline
COPY config/ config/
COPY scripts/ scripts/
COPY pipelines/ pipelines/
COPY run_pipeline.py .

# Mount data/ and outputs/ as volumes at runtime:
#   docker run -v ./data:/app/data -v ./outputs:/app/outputs ...
VOLUME ["/app/data", "/app/outputs"]

ENV FRED_API_KEY=""
ENTRYPOINT ["tradingcrab"]
CMD ["--refresh", "--recompute", "--plots", "--weekly-report"]
```

### Planned docker-compose.yml (weekly cron service)

```yaml
services:
  weekly-report:
    build: .
    target: pipeline
    environment:
      - FRED_API_KEY=${FRED_API_KEY}
      - TC_SMTP_HOST=${TC_SMTP_HOST}
      - TC_SMTP_USER=${TC_SMTP_USER}
      - TC_SMTP_PASSWORD=${TC_SMTP_PASSWORD}
      - TC_EMAIL_FROM=${TC_EMAIL_FROM}
      - TC_EMAIL_TO=${TC_EMAIL_TO}
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
      - ./config/email.local.yaml:/app/config/email.local.yaml:ro
    restart: "no"     # cron-compatible: run once then exit
```

Invoke with a host-side cron job:
```cron
0 8 * * 5  cd /path/to/trading-crab && docker compose run --rm weekly-report --send-email
```

### Why Docker for the weekly report

- **Reproducibility**: exact Python version + dep versions locked in the image
- **No environment drift**: host system Python upgrades can't break the pipeline
- **Easy secrets**: env vars set in the shell or a `.env` file, never committed
- **Portable**: run the same image locally, on a VPS, or on a cloud scheduler (AWS ECS, GCP Cloud Run)

See `NEXT_STEPS.md` Phase K (K2, K3) for the implementation backlog.

---

## Maven (Java) — Not Applicable

Maven is the standard build/distribution tool for JVM languages (Java, Kotlin, Scala).
It has no direct equivalent for Python.

**Python analogs:**

| Maven concept | Python equivalent |
|--------------|------------------|
| `pom.xml` | `pyproject.toml` |
| Maven Central | PyPI |
| `mvn package` | `python -m build` |
| JAR artifact | wheel (`.whl`) + sdist (`.tar.gz`) |
| `mvn install` (local) | `pip install -e .` |
| BOM (bill of materials) | `requirements.txt` / lock file |

If you need to consume this library from a JVM project, you would either:
1. Call it as a subprocess (recommended for isolation): `subprocess.run(["tradingcrab", ...])`
2. Use Jython (unmaintained, not recommended)
3. Expose a REST API from the Python service and call it from Java

---

## npm (Node.js/JavaScript) — Not Applicable

npm is the package manager for JavaScript and TypeScript. Python packages cannot
be published to the npm registry.

**Python analogs:**

| npm concept | Python equivalent |
|------------|------------------|
| `package.json` | `pyproject.toml` |
| `npm publish` | `twine upload dist/*` |
| `node_modules/` | `.venv/lib/python*/site-packages/` |
| `package-lock.json` | `uv.lock` / `pip-compile` output |
| `npx some-tool` | `uvx some-tool` (uv) or `pipx run some-tool` |

If you need to call this pipeline from a Node.js application:
1. **Recommended**: wrap `tradingcrab` as a subprocess via Node's `child_process`
2. **Alternative**: expose a REST API (FastAPI) and call it over HTTP
3. **WASM**: not practical for this use case (numpy/sklearn have no WASM port)

---

## GitHub Releases

Every version tag triggers a GitHub Release automatically. Source archives (`.tar.gz`,
`.zip`) are attached to each release and downloadable without git.

The release tag naming convention:
- `lib-v0.1.3` → release of `trading-crab-lib` 0.1.3
- `v0.1.3` → release of `trading-crab` 0.1.3

Releases are created from the tag on GitHub: Settings → Releases → Draft a new release,
or with the `gh` CLI:
```bash
gh release create v0.1.3 --title "trading-crab 0.1.3" --notes "Release notes here"
```

---

## Summary

For Python users: **PyPI is the right answer** — `pip install trading-crab-lib[all]`.

For automated/scheduled weekly runs: **Docker is the right answer** (Phase K).

For cross-language integration: **REST API or subprocess** is the right answer.
Maven and npm are Java and JavaScript tools and have no place in this Python project.
