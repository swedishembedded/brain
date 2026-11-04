// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod cfg_parallel;
mod condition_encoder_parity;
mod depth_decoder_parity;
mod dit_int8_storage;
mod dit_parity;
mod dit_shard_parity;
mod e2e_short_generation;
mod global_llm_int8_tier;
mod global_llm_parity;
mod vocoder_parity;
mod vocoder_real_chunk;
