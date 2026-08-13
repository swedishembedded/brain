// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Int8 (DP4A) inference support for the DiT linears — the fast P40 path.
//!
//! Weights are quantized once (per-channel symmetric int8, packed 4-per-u32);
//! activations are quantized on-device each forward with a dynamic per-token
//! scale (`max_abs_row` → `quant_pack`), then the DP4A GEMM
//! (`matmul_i8_dyn`, ~4× the fp32 rate on Pascal) dequantizes with `sx·sw`. The
//! 6B model in int8 is ~6 GB — it fits a single 24 GB P40, no sharding.
//!
//! The weight quantizer itself is the engine-wide shared implementation
//! (`model::int8` — also used by `qwen3::q8` and `flux2`); this module re-exports
//! it so zimage callers keep their path.

pub use model::int8::quantize_weight;

/// Threads used by the activation max-abs reduction (`max_abs_part` width).
pub const QP: u32 = 256;
