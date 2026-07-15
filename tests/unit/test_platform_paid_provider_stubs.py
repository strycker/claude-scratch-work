"""Unit tests for the paid-provider adapter seam stubs (DATA-06).

Norgate, Tiingo, and EODHD are v2 placeholders: they must import cleanly
(no network, no provider SDK) and raise NotImplementedError with a message
pointing to docs/paid_provider_seams.md when called.
"""

from __future__ import annotations

import pytest

from trading_crab_lib.platform.ingestion import eodhd, norgate, tiingo

STUB_MODULES = [norgate, tiingo, eodhd]


class TestPaidProviderStubsImportCleanly:
    def test_norgate_imports_cleanly(self):
        assert hasattr(norgate, "fetch_prices")

    def test_tiingo_imports_cleanly(self):
        assert hasattr(tiingo, "fetch_prices")

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
