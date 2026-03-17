from __future__ import annotations

from pathlib import Path

import yaml


CFG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def _load_cfg() -> dict:
    with CFG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_fred_series_includes_additional_series() -> None:
    cfg = _load_cfg()
    series = cfg["fred"]["series"]
    for key, name in [
        ("VIXCLS", "fred_vix"),
        ("UNRATE", "fred_unrate"),
        ("M2SL", "fred_m2sl"),
        ("M2NS", "fred_m2ns"),
        ("GS2", "fred_gs2"),
        ("T10Y2Y", "fred_t10y2y"),
        ("T10Y3M", "fred_t10y3m"),
    ]:
        assert key in series
        assert series[key]["name"] == name
        assert series[key].get("shift", False) is False
