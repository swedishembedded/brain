# Forecasting: probabilistic time-series prediction

brain runs foundation models for time-series forecasting: feed in a numeric
context (a sensor reading history, demand series, price series) and get a
probabilistic forecast out, without training a model of your own first.
Useful anywhere you have a series and want a plausible continuation with
uncertainty attached — demand planning, financial series, sensor streams.

## Capabilities

Three models — a general-purpose forecaster, a financial-specialized one,
and an OHLCV bar forecaster — sit behind one shared CLI verb (`brain
forecast`) and one shared D-Bus action, so the workflow (import weights,
backtest against baselines, serve, forecast) is the same regardless of which
you pick. Each has its own model id, weights, and fine-tuning story.

None of the three need labeled training data or a training run of your own —
point one at a numeric context and a forecast horizon and it returns a
distribution of plausible futures, not just a single point estimate. Before
trusting a forecaster on real data, the shared `compare` workflow backtests
it against statistical baselines so you know it's actually adding value over
a naive random walk.

See [the forecasting page](../models/forecast.md) for the shared surface,
and follow its links to the individual models — a general series, a
financial series, or OHLCV price bars — for the one that fits your data.
