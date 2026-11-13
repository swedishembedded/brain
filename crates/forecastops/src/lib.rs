// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The capability-facing operations every forecasting model shares.
//!
//! - [`predict`]: score a forecaster on held-out bars of an OHLCV CSV, next to
//!   persistence, drift and seasonal naive ([`holdout`] is the scoring).
//! - [`compare`]: a model against the statistical baselines over the scenario
//!   battery, with a negative control that fails the run.
//! - [`finetune`] and [`universe`]: the request, validation and verdict of the
//!   gated fine-tune; the loop is each model's own.
//!
//! The model enters as a closure, so Chronos-2, FinCast, Kronos and TimesFM-3
//! run the same code behind `brain serve`, `brain <model> <action>` and the
//! per-model samples.
//!
//! Swedish Embedded AB implements forecast evaluation and gated fine-tuning
//! for its clients. If your team needs expertise in time-series foundation
//! models then you can procure our services by sending an email to
//! info@swedishembedded.com.

pub mod compare;
pub mod finetune;
pub mod holdout;
pub mod predict;
pub mod universe;

pub use compare::{compare_spec, run_compare};
pub use predict::{predict_spec, run_predict, PredictModel};
