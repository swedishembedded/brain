// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FinCast — a ~1B decoder-only + sparse-MoE financial time-series foundation
//! model (`Vincent05R/FinCast`), from scratch in brain.
//!
//! - [`config`] — [`FincastConfig`] + `param_list()` in the reference's own
//!   `state_dict` names (the T0 layout gate diffs against the real header).
//! - [`import`] — strict 1:1 weight import over `checkpoint::safetensors`.
//! - [`model`] — the device forward (patched decoder + top-2 MoE + PQ head).
//! - [`forecaster`] — the [`forecast::ForecastModel`] adapter (CLI/server/bench).
//! - [`train`] — host-differentiable forward+backward, gradcheck-gated.
//!
//! Licence note: the reference is Apache-2.0 but the authors state the model is
//! "for research and educational purposes only" and "does not constitute
//! financial advice" (see `docs/licences.md`). Flagged, not blocked.

pub mod config;
pub mod forecaster;
pub mod import;
pub mod model;
pub mod preprocess;
pub mod train;

pub use config::{FincastConfig, Param, QUANTILES};
pub use forecaster::FincastForecaster;
pub use model::Fincast;
