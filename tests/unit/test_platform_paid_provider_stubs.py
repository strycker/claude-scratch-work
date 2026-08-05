"""Unit tests for the paid-provider adapter seams (DATA-06).

Norgate and EODHD remain v2 placeholders: they must import cleanly (no
network, no provider SDK) and raise NotImplementedError with a message
pointing to docs/paid_provider_seams.md when called.

Tiingo is NO LONGER a stub — it has a live adapter and leads the daily-price
fallback chain, so it is deliberately absent from STUB_MODULES. Its behavior
is covered by tests/unit/test_platform_tiingo_ingest.py.
"""

from __future__ import annotations

import pytest

from trading_crab_lib.platform.ingestion import eodhd, norgate, tiingo

STUB_MODULES = [norgate, eodhd]


class TestPaidProviderStubsImportCleanly:
    def test_norgate_imports_cleanly(self):
        assert hasattr(norgate, "fetch_prices")

    def test_tiingo_is_a_live_adapter_not_a_stub(self):
        """Guards against Tiingo being silently reverted to a stub."""
        assert hasattr(tiingo, "fetch_prices")
        assert hasattr(tiingo, "fetch_daily_prices")
        assert tiingo not in STUB_MODULES

    def test_eodhd_imports_cleanly(self):
        assert hasattr(eodhd, "fetch_prices")


class TestPaidProviderStubsRaiseNotImplemented:
    @pytest.mark.parametrize("module", STUB_MODULES)
    def test_fetch_prices_raises_not_implemented(self, module):
        with pytest.raises(NotImplementedError) as exc_info:
            module.fetch_prices({})
        msg = str(exc_info.value)
        assert msg
        assert "docs/paid_provider_seams.md" in msg
