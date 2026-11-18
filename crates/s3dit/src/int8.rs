// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Int8 inference support for the DiT linears.
//!
//! Weights are quantized once (group-wise symmetric int8, packed 4-per-u32);
//! activations are quantized on-device each forward with dynamic scales per
//! 32 elements, and the product is the one `model::ops::Ops::matmul` selects
//! for the device (a packed-dot GEMM where the device has one, the float-read
//! tile decode where it has not). The 6B model in int8 is ~6 GB: it fits a
//! single 24 GB P40, no sharding.
//!
//! The weight quantizer itself is the engine-wide shared implementation
//! (`model::int8` — also used by `qwen3::q8` and `flux2`); this module re-exports
//! it so zimage callers keep their path.

pub use model::int8::quantize_weight;
