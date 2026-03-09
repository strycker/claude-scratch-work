# Makefile — shortcuts for common Trading-Crab tasks.
# Run `make help` to see all available targets.

.PHONY: help setup setup-dev install install-dev test lint run run-full run-steps clean-checkpoints clean-all

# ── default ────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "Trading-Crab — available make targets"
	@echo "--------------------------------------"
	@echo "  make setup          Set up .venv + install runtime deps (interactive)"
	@echo "  make setup-dev      Set up .venv + install dev deps (tests + notebooks)"
	@echo "  make install        pip install -r requirements.txt into active env"
	@echo "  make install-dev    pip install -r requirements-dev.txt into active env"
	@echo ""
	@echo "  make test           Run the full test suite"
	@echo "  make test-fast      Run tests, stop at first failure"
	@echo ""
	@echo "  make run            Steps 3-7 from cached data (fast, no re-scraping)"
	@echo "  make run-full       Full pipeline — re-scrape + recompute + plots"
	@echo "  make run-cluster    Re-cluster only (step 3) with plots"
	@echo "  make dashboard      Print current dashboard (step 7 only)"
	@echo "  make notebooks      Launch JupyterLab"
	@echo ""
	@echo "  make clean-outputs  Remove generated plots and reports"
	@echo "  make clean-models   Remove saved models"
	@echo "  make clean-all      Remove all generated files (keep raw checkpoints)"
	@echo ""

# ── setup ──────────────────────────────────────────────────────────────────────

setup:
	bash scripts/setup.sh

setup-dev:
	bash scripts/setup.sh --dev

# Install into whatever environment is currently active (no venv management)
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

# ── testing ────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -x -q

# ── pipeline ───────────────────────────────────────────────────────────────────

# Steps 3-7 from cached checkpoints — fast day-to-day run
run:
	python run_pipeline.py --steps 3,4,5,6,7 --plots --market-code grok

# Re-scrape everything from scratch and recompute
run-full:
	python run_pipeline.py --refresh --recompute --plots --market-code grok --save-market-code

# Re-cluster only (useful after editing settings.yaml)
run-cluster:
	python run_pipeline.py --steps 3,4 --plots --recompute --market-code grok

# Just print the dashboard
dashboard:
	python pipelines/07_dashboard.py

# Launch notebooks
notebooks:
	jupyter lab notebooks/

# ── cleanup ────────────────────────────────────────────────────────────────────

clean-outputs:
	rm -f outputs/plots/*.png outputs/plots/*.pdf
	rm -f outputs/reports/*.csv

clean-models:
	rm -f outputs/models/*.pkl outputs/models/*.joblib

clean-all: clean-outputs clean-models
	rm -f data/processed/*.parquet
	rm -f data/regimes/*.parquet data/regimes/*.yaml
	@echo "Kept data/raw/ and data/checkpoints/ — run 'make run' to regenerate"
