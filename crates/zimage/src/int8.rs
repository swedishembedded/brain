// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Int8 (DP4A) inference support for the DiT linears — the fast P40 path.
//!
//! Weights are quantized once (per-tensor symmetric int8, packed 4-per-u32);
//! activations are quantized on-device each forward with a dynamic per-tensor
//! scale (`max_abs_part` → `max_abs_final` → `quant_pack`), then the DP4A GEMM
//! (`matmul_i8_dyn`, ~4× the fp32 rate on Pascal) dequantizes with `sx·sw`. The
//! 6B model in int8 is ~6 GB — it fits a single 24 GB P40, no sharding.

/// Threads used by the activation max-abs reduction (`max_abs_part` width).
pub const QP: u32 = 256;

/// Per-CHANNEL symmetric int8 quantization of a `[n, k]` weight (one scale per
/// output row `n`), packed into `[n, k/4]` u32 (4 int8 per u32, little-endian
/// along K). Returns `(packed, scales[n])` with `scales[r] = max|w[r,:]|/127`.
/// Per-channel (vs per-tensor) is what keeps a deep int8 stack accurate — a
/// single outlier row no longer crushes the whole matrix's resolution.
/// `k` must be a multiple of 4.
pub fn quantize_weight(w: &[f32], n: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    assert_eq!(k % 4, 0, "int8 K must be a multiple of 4 (got {k})");
    assert_eq!(w.len(), n * k, "weight len {} != n*k {}", w.len(), n * k);
    let kg = k / 4;
    let mut sw = vec![0f32; n];
    let mut packed = vec![0u32; n * kg];
    for r in 0..n {
        let row = &w[r * k..r * k + k];
        let amax = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let s = amax.max(1e-8) / 127.0;
        sw[r] = s;
        let inv = 1.0 / s;
        for g in 0..kg {
            let mut word = 0u32;
            for b in 0..4 {
                let q = (row[g * 4 + b] * inv).round().clamp(-127.0, 127.0) as i32;
                word |= ((q as u8) as u32) << (8 * b);
            }
            packed[r * kg + g] = word;
        }
    }
    (packed, sw)
}
