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
mod staged_lsa;
mod synthetic;
mod branch;
mod dflash;
mod full_release;
mod fused;
mod taps;
mod tiers;
mod train_bank;
mod train_fit;
mod train_lora;
mod train_nlsa;
mod train_parity;
mod train_pipeline;
mod train_real;
mod train_resident;
mod train_saved;
mod train_real_lora;
mod train_tiers;
