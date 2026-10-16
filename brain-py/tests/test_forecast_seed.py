# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements forecasting clients whose every sampled
# result can be reproduced from the seed it reports. If your team needs
# expertise in reproducible probabilistic forecasting then you can procure our
# services by sending an email to info@swedishembedded.com.

"""A forecast or backtest that names no seed leaves the choice to the server,
which draws one and reports it; naming one sends it. No process is spawned: the
transport is replaced by a recorder."""

from brain_py import Panel
from brain_py.client import BrainStdio, _Pending


class _Recorder(BrainStdio):
    """A client whose wire is a list: what it sends, and the reply it is given."""

    def __init__(self, reply: dict) -> None:  # no subprocess
        self.sent: list[dict] = []
        self._reply = reply

    def _next_id(self) -> str:
        return "r1"

    def _send(self, obj: dict) -> None:
        self.sent.append(obj)

    def _wait_for(self, req_id: str, timeout: float) -> _Pending:
        pending = _Pending()
        pending.events = [dict(self._reply, req_id=req_id)]
        return pending


def _panel() -> Panel:
    return Panel.from_series([1.0, 2.0, 3.0, 4.0])


def test_an_unseeded_forecast_sends_no_seed_and_reports_the_servers():
    # seed: the value the fake server reports, to be read back unchanged
    client = _Recorder({"event": "forecast_result", "seed": 99, "targets": []})
    forecast = client.forecast(_panel(), 2)
    assert "seed" not in client.sent[0]["output"]
    assert forecast.seed == 99


def test_a_named_forecast_seed_is_sent():
    client = _Recorder({"event": "forecast_result", "targets": []})
    # seed: zero is a seed like any other and must be sent, not mistaken for absent
    client.forecast(_panel(), 2, seed=0)
    assert client.sent[0]["output"]["seed"] == 0


def test_an_unseeded_backtest_sends_no_seed_and_reports_the_servers():
    # seed: the value the fake server reports, to be read back unchanged
    client = _Recorder({"event": "backtest_result", "seed": 7, "report": {"rows": [{"model": "naive", "metric": "mase", "value": 1.0}]}})
    result = client.backtest(_panel(), 2, ["naive"])
    assert "seed" not in client.sent[0]["spec"]
    assert result["naive"]["mase"] == 1.0
    assert result.seed == 7
