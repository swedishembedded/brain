// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod bias;
mod blocks;
mod dynamics_embed;
mod gelu_erf;
mod import;
mod import_inspect;
mod kv_cache;
mod maskgit;
mod parity_dynamics;
mod parity_maskgit;
mod parity_tokenizer;
mod patch;
mod stblock;
mod sttransformer;
mod tiny;
mod tokenizer;
mod vq;
mod world_model;
