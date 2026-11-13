// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod asr_roundtrip;
mod bench_inference;
mod cached_parity;
mod mtp;
mod parity;
mod recompute;
mod resident_device;
mod sampled_decode;
mod served_as_downloaded;
mod sft_train;
mod store;
mod talker;
mod talker_policy;
