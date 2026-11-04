// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it; `dit_parity`, `gguf_direct_real`, `vae_parity` keep binaries of their own.

mod block_grad;
mod dev_grad;
mod dev_lora_train;
mod device_train;
mod finetune_ab;
mod gguf_import_real;
mod host_forward_parity;
mod lora_inference;
mod lora_int8_delivery;
mod lora_runtime_dev;
mod lora_train;
mod overfit;
mod recompute;
