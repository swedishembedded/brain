// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared int8 (DP4A) weight quantization — the host half of the engine's
//! int8 inference tier.
//!
//! One implementation for every model that runs the DP4A path (zimage DiT,
//! Qwen encoder/decoder, FLUX.2 DiT): weights are quantized ONCE at build with
//! [`quantize_weight`]; activations are quantized on-device each forward with a
//! dynamic per-token scale (`max_abs_row` → `quant_pack`), then the DP4A GEMM
//! (`matmul_i8_dyn`, ~4× the fp32 rate on Pascal) dequantizes with `sx·sw`.
//! The packed layout here is exactly what `matmul_i8*.wgsl` consume — if it
//! changes, it changes for every model at once, which is the point.

/// Per-CHANNEL symmetric int8 quantization of an `[n, k]` weight (one scale per
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

/// The exact host-side inverse of [`quantize_weight`]: unpack `[n, k/4]` u32
/// words back to `[n, k]` f32 via `sw[r]` (the same per-row scale
/// `quantize_weight` wrote). Used where a real checkpoint quantized a weight
/// (`omni::import::should_quantize`) that the consuming forward pass still
/// wants as plain f32 — e.g. attention/router projections, which meet the
/// same rank-2/`k%4==0` shape test as the MoE experts but have no int8
/// dispatch path of their own (only the experts do, `model::moe::
/// expert_fwd_i8`). `k` must be a multiple of 4 (mirrors `quantize_weight`'s
/// own requirement, since the packing is 4 lanes per u32).
pub fn dequantize_weight(packed: &[u32], sw: &[f32], n: usize, k: usize) -> Vec<f32> {
    assert_eq!(k % 4, 0, "int8 K must be a multiple of 4 (got {k})");
    let kg = k / 4;
    assert_eq!(packed.len(), n * kg, "packed len {} != n*(k/4) {}", packed.len(), n * kg);
    assert_eq!(sw.len(), n, "scale len {} != n {}", sw.len(), n);
    let mut w = vec![0f32; n * k];
    for r in 0..n {
        let s = sw[r];
        for g in 0..kg {
            let word = packed[r * kg + g];
            for b in 0..4 {
                let q = ((word >> (8 * b)) as u8) as i8;
                w[r * k + g * 4 + b] = q as f32 * s;
            }
        }
    }
    w
}

/// The buffers and kernel slots one dynamic activation quantization needs —
/// the same "bundle the ids so the call stays readable" shape as
/// [`crate::block::FlashIds`].
pub struct QuantRows<'a> {
    /// `[max_abs_row, quant_pack]` in the caller's kernel list.
    pub kernels: [usize; 2],
    /// `[.., k]` f32 activation to quantize.
    pub x: &'a gpu_core::DeviceBuffer,
    /// `[rows]` per-token scale (written).
    pub sx: &'a gpu_core::DeviceBuffer,
    /// `[.., k/4]` packed u32 activation (written).
    pub xq: &'a gpu_core::DeviceBuffer,
}

/// The DEVICE half of the same tier: dynamic per-token activation quantization
/// of rows `r0..r1` of `q.x` — `max_abs_row` writes `q.sx[r0..r1]`,
/// `quant_pack` writes the packed rows of `q.xq`. Returns the two steps in
/// order; ONE call feeds every linear that reads that activation.
///
/// This exists so the *offset units* live in one place. Every buffer is bound
/// as a sub-range and the units differ per buffer — `x` and `xq` are offset in
/// ELEMENTS of their own width (`k` vs `k/4`) while `sx` is offset in ROWS.
/// Getting one wrong is silently wrong arithmetic, not a crash; `step_sliced`'s
/// element-vs-byte contract already cost this repo a SIGSEGV.
///
/// **Alignment:** `step_sliced` turns each offset into a real
/// `BufferBinding::offset`, so every one must clear
/// `min_storage_buffer_offset_alignment` (256 B = 64 floats on a P40). `k` is
/// normally a multiple of 64, so `x`/`xq` are safe for any `r0`; `sx` is offset
/// by `r0` itself, so **`r0` must be a multiple of 64** — asserted here rather
/// than left to each caller to remember (a violation is a wgpu validation
/// error, not a wrong number, so it hides until someone changes a text length).
pub fn quant_rows_steps(g: &gpu_core::Gpu, q: QuantRows, r0: u32, r1: u32, k: u32) -> [gpu_core::Step; 2] {
    assert_eq!(k % 4, 0, "int8 K must be a multiple of 4 (got {k})");
    assert!(
        r0.is_multiple_of(64),
        "int8 activation quant: row base {r0} breaks the 64-float storage-binding alignment of the per-token scale buffer"
    );
    let m = r1 - r0;
    let xo = (r0 as u64 * k as u64, m as u64 * k as u64);
    let so = (r0 as u64, m as u64);
    let qo = (r0 as u64 * (k as u64 / 4), m as u64 * (k as u64 / 4));
    [
        g.step_sliced(q.kernels[0], &[q.x, q.sx], &[xo, so], &[m, k], m),
        g.step_sliced(q.kernels[1], &[q.x, q.sx, q.xq], &[xo, so, qo], &[m, k], m * k / 4),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_within_one_step() {
        let (n, k) = (3, 8);
        let w: Vec<f32> = (0..n * k).map(|i| (i as f32 - 10.0) * 0.37).collect();
        let (packed, sw) = quantize_weight(&w, n, k);
        assert_eq!(packed.len(), n * k / 4);
        for r in 0..n {
            for c in 0..k {
                let word = packed[r * (k / 4) + c / 4];
                let q = ((word >> (8 * (c % 4))) & 0xff) as u8 as i8;
                let deq = q as f32 * sw[r];
                assert!((deq - w[r * k + c]).abs() <= sw[r] * 0.5 + 1e-6, "r{r} c{c}");
            }
        }
    }

    /// [`dequantize_weight`] must agree with the round-trip math
    /// [`round_trips_within_one_step`] checks inline, per-element, over
    /// multiple rows (distinct per-row scales) and a `k` that isn't a
    /// multiple of the packing group in a suspicious way (12, not 4 or 8) —
    /// the real caller of this function (`omni::int8_thinker_resident::
    /// load_mat`) needs it for arbitrary real weight shapes, not just the
    /// one `quantize_weight`'s own test happens to use.
    #[test]
    fn dequantize_weight_matches_quantize_weight_within_one_step() {
        let (n, k) = (5, 12);
        let w: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.11 - 6.0) * (1 + i % 3) as f32).collect();
        let (packed, sw) = quantize_weight(&w, n, k);
        let deq = dequantize_weight(&packed, &sw, n, k);
        assert_eq!(deq.len(), n * k);
        for r in 0..n {
            for c in 0..k {
                let d = deq[r * k + c];
                let orig = w[r * k + c];
                assert!((d - orig).abs() <= sw[r] * 0.5 + 1e-6, "r{r} c{c}: deq={d} orig={orig} scale={}", sw[r]);
            }
        }
    }

    /// Mutation-verify: an intentionally wrong sign-extension (`as u8`
    /// dropped, reading the byte as unsigned) must break the round-trip
    /// tolerance above -- confirms the test actually exercises the sign
    /// handling, not just magnitude.
    #[test]
    fn dequantize_weight_sign_handling_is_load_bearing() {
        let (n, k) = (2, 4);
        let w = vec![-100.0f32, 50.0, -1.0, 90.0, 10.0, -80.0, 60.0, -40.0];
        let (packed, sw) = quantize_weight(&w, n, k);
        let correct = dequantize_weight(&packed, &sw, n, k);
        // Deliberately wrong: treat each packed byte as UNSIGNED, dropping
        // the `as i8` reinterpretation `dequantize_weight` relies on.
        let mut wrong = vec![0f32; n * k];
        for r in 0..n {
            let s = sw[r];
            for g in 0..k / 4 {
                let word = packed[r * (k / 4) + g];
                for b in 0..4 {
                    let q_unsigned = (word >> (8 * b)) as u8;
                    wrong[r * k + g * 4 + b] = q_unsigned as f32 * s;
                }
            }
        }
        let mut any_diverged = false;
        for i in 0..w.len() {
            if (correct[i] - wrong[i]).abs() > 1e-3 {
                any_diverged = true;
            }
        }
        assert!(any_diverged, "unsigned-vs-signed byte reinterpretation should diverge for at least one negative-quantized element");
    }
}
