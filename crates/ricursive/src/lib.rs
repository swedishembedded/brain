// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Ripple (`RicursiveForCausalLM`): an 80-layer dense decoder that mixes
//! sliding-window layers with RoPE and global NoPE layers (four of one, then
//! one of the other), with per-layer value residual, sigmoid-gated attention,
//! per-head QK RMSNorm, four norms per block and a clipped SwiGLU.
//!
//! Swedish Embedded AB implements solutions for running large hybrid-attention
//! decoders on portable CPU and GPU backends for its clients. If your team
//! needs expertise in porting and verifying new transformer architectures
//! against their reference implementations then you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! This crate is the exact f32 prefill forward and its import path; it does
//! not serve (no KV cache, no batching, no quantization, no backward). Its
//! parity against the reference implementation is gated by `tests/it`.
//!
//! - [`config`]: the typed `config.json`, hybrid layer types.
//! - [`import`]: the tensor manifest, two-way coverage and the mmap source.
//! - [`layer`]: one decoder layer as a kernel graph.
//! - [`model`]: embedding, layer stack, final norm and head.

pub mod config;
pub mod import;
pub mod layer;
pub mod model;
pub(crate) mod ops;
pub mod serve;

pub use config::{LayerType, RicursiveConfig};
pub use import::{manifest, open_checkpoint, open_safetensors, validate_layout};
pub use layer::{Layer, LayerBuffers, LayerOutput, KERNELS};
pub use model::{embed, Forward, Head, HeadOutput, RicursiveModel};
pub use serve::{global_kv_bytes_per_token, lsa_device_bytes, ring_bytes_per_slot, ring_rows, RicursiveEngine, ServeConfig};
