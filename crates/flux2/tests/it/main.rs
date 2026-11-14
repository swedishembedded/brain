// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it; `dev_grad`, `dev_step_time`, `device_train`, `int8_parity`, `int8_trainer`, `layernorm_rows` keep binaries of their own.

mod batch_parity;
mod batch_time;
mod block_grad;
mod dit_parity;
mod e2e_parity;
mod flow_batch_convention;
mod gguf_direct_int8;
mod host_forward_parity;
mod import_real;
mod int8_base_grads;
mod lora_external;
mod lora_requant_int8;
mod lora_runtime_int8;
mod lora_stack;
mod lora_train;
mod loss_weights;
mod model_grad;
mod model_smoke;
mod placement;
mod recompute;
mod resolve_layout;
mod resume;
mod step_time;
mod streamed_grads;
mod train_schedule;
mod training_matches_deployment;
mod unpaired_training_unchanged;
mod variant_widths;
