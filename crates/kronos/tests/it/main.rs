// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it; `bench_cpu`, `csv_forecast_e2e`, `kvcache_parity`, `parity`, `train_gradcheck`, `train_parity` keep binaries of their own.

mod calendar_e2e;
mod kernels;
mod rankic_eval;
mod recompute;
mod t0_param_layout;
