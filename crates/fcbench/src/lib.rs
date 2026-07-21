// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Forecasting baselines, scenario generators, and the rolling-origin
//! backtester — the honest-evaluation half of brain's forecasting stack.
//!
//! This crate is deliberately **light** (it depends only on `brain-forecast`),
//! so the server can serve the statistical baselines and run backtests without
//! pulling in the training/model stack. The registered *benchmarks* that make
//! these show up in `brain bench` live in `crates/bench`, which depends on this.
//!
//! - [`baselines`] — [`RandomWalk`](baselines::RandomWalk) and friends, the
//!   controls every foundation model must beat.
//! - `scenarios` (next task) — seeded synthetic regimes with closed-form ground
//!   truth, guaranteed unseen by construction.
//! - `backtest` (next task) — server-side rolling-origin evaluation.

pub mod backtest;
pub mod baselines;
pub mod harness;
pub mod report;
pub mod rng;
pub mod scenarios;
pub mod score;
pub mod util;

pub use baselines::{Arima, Drift, Garch11, RandomWalk, SeasonalNaive};
pub use harness::{Cell, Comparison};
pub use scenarios::{Scenario, Window};
