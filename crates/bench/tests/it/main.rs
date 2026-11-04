// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it; `survival_mixers` keeps a binary of its own.

mod brain_qa;
mod capscale;
mod default_seed;
mod distributed_profile;
mod dyck;
mod eval;
mod mad_compress;
mod mad_fuzzy_recall;
mod mad_memorize;
mod mad_noisy_recall;
mod mad_recall;
mod mad_selective_copy;
mod mod_add;
mod mqar;
mod parity;
mod qwen;
mod scaling;
mod survival_bench;
mod toolcall;
