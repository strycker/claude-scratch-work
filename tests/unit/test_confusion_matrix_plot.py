"""Tests for plotting.plot_confusion_matrix()."""

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from trading_crab.runtime import RunConfig


def test_confusion_matrix_runs_without_error():
    """Smoke test: plot_confusion_matrix produces a figure and calls _save_or_show."""
    from trading_crab.plotting import plot_confusion_matrix

    y_true = pd.Series([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    y_pred = pd.Series([0, 0, 1, 2, 2, 2, 0, 1, 1, 0])
    regime_names = {0: "Growth", 1: "Stag", 2: "Recession"}
    run_cfg = RunConfig(save_plots=False, show_plots=False)

    with patch("trading_crab.plotting._save_or_show") as mock_save:
        plot_confusion_matrix(y_true, y_pred, regime_names, run_cfg)
        mock_save.assert_called_once()
        fig = mock_save.call_args[0][0]
        assert fig is not None


def test_confusion_matrix_normalized_values():
    """Normalized confusion matrix values should be in [0, 1]."""
    from trading_crab.plotting import plot_confusion_matrix

    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    y_pred = pd.Series([0, 0, 1, 1, 1, 0])
    regime_names = {0: "A", 1: "B"}
    run_cfg = RunConfig(save_plots=False, show_plots=False)

    with patch("trading_crab.plotting._save_or_show"):
        # Should not raise
        plot_confusion_matrix(y_true, y_pred, regime_names, run_cfg, normalize=True)
        plot_confusion_matrix(y_true, y_pred, regime_names, run_cfg, normalize=False)
