// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod block_grad;
mod block_overfit;
mod block_parity;
mod dev_grad;
mod dev_parity;
mod device_train;
mod gguf_import_real;
mod int8_matmul;
mod lora_train;
mod model_parity;
mod model_train;
mod model_trait;
mod real_parity;
mod shard_2card;
mod shard_parity;
