# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""CI-fast forecasting tests for BrainClient.

These launch `brain forecast serve` (statistical baselines registered — no model
load, so CI-fast) and drive it over the real JSONL protocol, exercising the
forecast/backtest/capabilities request path and the Panel/Forecast helpers.
"""

import pytest

from brain_py import BrainClient, Panel
from brain_py.client import _find_brain_binary


def _have_brain() -> bool:
    try:
        _find_brain_binary(None)
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _have_brain(),
    reason="brain binary not found; run `cargo build --release`",
)


def _client() -> BrainClient:
    return BrainClient(forecast=True, device="cpu")


def test_capabilities_lists_the_baselines():
    with _client() as b:
        caps = b.capabilities()
        assert "naive" in caps
        # baselines are univariate, distribution-native, unbounded context
        assert caps["naive"]["native_representation"] == "distribution"
        assert caps["naive"]["multivariate"] is False


def test_forecast_returns_quantiles_and_point():
    with _client() as b:
        panel = Panel.from_series([10.0, 11.0, 12.0, 13.0, 14.0], freq="1d")
        fc = b.forecast(panel, horizon=3, model="naive")
        assert fc.model == "naive"
        assert fc.horizon == 3
        # naive forecasts the last value, flat
        med = fc.median()
        assert len(med) == 3
        assert all(abs(v - 14.0) < 1e-3 for v in med), med
        # an 80% interval exists and brackets the median
        lo, hi = fc.interval(0.8)
        assert all(lo[i] <= med[i] <= hi[i] for i in range(3))


def test_forecast_interval_widens_with_horizon():
    with _client() as b:
        panel = Panel.from_series([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], freq="1d")
        fc = b.forecast(panel, horizon=4, model="naive")
        lo, hi = fc.interval(0.8)
        width0 = hi[0] - lo[0]
        width3 = hi[3] - lo[3]
        assert width3 > width0, (width0, width3)


def test_samples_are_returned_and_flagged_derived():
    with _client() as b:
        panel = Panel.from_series([1.0, 2.0, 3.0, 4.0, 5.0], freq="1d")
        fc = b.forecast(panel, horizon=2, model="naive", num_samples=64)
        paths = fc.samples()
        assert paths is not None and len(paths) == 64
        assert all(len(p) == 2 for p in paths)
        # samples were derived from the Gaussian distribution -> flagged
        assert "samples" in fc.derived


def test_unknown_model_raises():
    with _client() as b:
        panel = Panel.from_series([1.0, 2.0, 3.0], freq="1d")
        with pytest.raises(RuntimeError):
            b.forecast(panel, horizon=2, model="does_not_exist")


def test_backtest_compares_models_against_naive():
    with _client() as b:
        # a mild upward line
        series = [100.0 + i * 0.1 for i in range(120)]
        panel = Panel.from_series(series, freq="1d")
        report = b.backtest(panel, horizon=5, models=["naive", "drift"],
                            origins=10, stride=2, metrics=["mase"])
        assert "naive" in report and "drift" in report
        assert "mase" in report["naive"]
