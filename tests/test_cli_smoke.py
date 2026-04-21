"""Smoke tests for trading_crab.cli entry points.

Covers: run_pipeline(), setup(), publish_notebooks().
"""

from __future__ import annotations

from unittest.mock import patch


def test_run_pipeline_delegates_to_pipeline_main():
    from trading_crab.cli import run_pipeline
    # main is imported inside run_pipeline(), so patch it at its source module.
    with patch("trading_crab.pipeline.main") as mock_main:
        run_pipeline()
    mock_main.assert_called_once_with()


def test_setup_prints_output(capsys):
    from trading_crab.cli import setup
    setup()
    out = capsys.readouterr().out
    assert out  # some output is produced
    assert "FRED_API_KEY" in out or "setup" in out.lower()


def test_publish_notebooks_prints_output(capsys):
    from trading_crab.cli import publish_notebooks
    publish_notebooks()
    out = capsys.readouterr().out
    assert out  # some output is produced


def test_cli_module_imports_cleanly():
    import trading_crab.cli  # noqa: F401  # pylint: disable=unused-import


def test_run_pipeline_is_callable():
    from trading_crab.cli import run_pipeline
    assert callable(run_pipeline)


def test_setup_is_callable():
    from trading_crab.cli import setup
    assert callable(setup)


def test_publish_notebooks_is_callable():
    from trading_crab.cli import publish_notebooks
    assert callable(publish_notebooks)
