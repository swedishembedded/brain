// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod chunked_prefill;
mod decode_step;
mod decode_tapes;
mod engine_tiers;
mod fast_path_parity_real;
mod gguf_vs_hf_real;
mod hf_nvfp4;
mod layer_streamed;
mod lora_flash;
mod lora_freezes_base;
mod lora_real_weight_smoke;
mod model_i8_smoke;
mod model_smoke;
mod optimizer_state;
mod real_decode_parity;
mod recompute;
mod reference_forward_real;
mod sample_generate;
mod sample_stream;
mod serve;
mod serve_conformance;
mod shard_parity;
mod streamed_construction;
mod vl;
