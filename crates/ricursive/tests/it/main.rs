// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod lsa;
mod lsa_tool;
mod parity;
mod real_nvfp4;
mod real_weight_parity;
mod serve;
mod taps;
mod tiers;
