// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device parity for the two M8.6 portable FP8 GEMV kernels
//! (`matmul_gemv_f8e4m3`/`matmul_gemv_f8e5m2`) against the EXISTING host
//! oracle `model::fp8::dequant_block128` - the same function the import path
//! uses to dequantize a real DeepSeek-V3/Qwen3.5-FP8 checkpoint's weight,
//! not a duplicate. Two shapes, per the milestone's own gate: one 128x128-
//! block-ALIGNED (`n=128,k=256`) and one NOT (`n=200,k=192` - `n` is not a
//! multiple of 128, so the last row-block is a PARTIAL 72-row block; `k` is
//! not a multiple of 128 either, so the last column-block is a partial
//! 64-column block) - the padding case is exactly where a blockwise-scale
//! indexing bug hides.
//!
//! Activations stay plain, UNQUANTIZED f32 (this is a decode-to-f32 STORAGE
//! tier, like `BF16`/`F16`, not W4A8), so unlike `matmul_lut4_gemm.rs`/
//! `matmul_q4_gemm.rs` there is no activation-quant noise floor to budget
//! for - the only approximation is the FP8 weight encoding itself, and the
//! oracle uses the IDENTICAL decoded+dequantized weight the kernel computes
//! from, so the tolerance here is tight (float summation order only).

use data::rng::Lcg;
use gpu_core::Gpu;

const KERNELS: &[(&str, &str)] = &[
    ("matmul_gemv_f8e4m3", kernels::MATMUL_GEMV_F8E4M3),
    ("matmul_gemv_f8e5m2", kernels::MATMUL_GEMV_F8E5M2),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn host_matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += x[r * k + i] * w[j * k + i];
            }
            out[r * n + j] = acc;
        }
    }
    out
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&g, &w) in got.iter().zip(want) {
        num += (g as f64 - w as f64).powi(2);
        den += (w as f64).powi(2);
    }
    (num / den.max(1e-12)).sqrt()
}

/// The largest FINITE magnitude `decode` can produce - the format's own
/// dynamic range, derived by sweeping all 256 bytes rather than hand-coding
/// "448.0"/"57344.0" a second time (those numbers are already pinned, by
/// name, in `crates/kernels/src/template.rs`'s and `crate::fp8`'s own known-
/// value tests - this just needs the VALUE, generically, for the encoder
/// below).
fn max_representable(decode: fn(u8) -> f32) -> f32 {
    (0u16..=255).map(|b| decode(b as u8)).filter(|v| v.is_finite()).fold(0f32, |m, v| m.max(v.abs()))
}

/// A TEST-ONLY encoder (no production FP8 encoder exists in this engine -
/// checkpoints arrive pre-quantized, see `model::fp8`'s own module doc):
/// nearest-neighbour search over the format's own 256-value codebook,
/// exactly the technique `model::lut4::quantize_weight_lut4` already uses
/// for NF4/F4E2M1 (M8.5), applied here per-128x128-BLOCK instead of per-32-
/// element-group. Returns `(bytes, scale)` with `scale` shaped
/// `model::fp8::scale_shape(n, k, block)` - `scale[br,bc] = block's own
/// amax / format's own max finite value`, matching `model::fp8`'s own
/// round-trip test's convention (`round_trip_through_a_real_quantize_
/// recipe_stays_within_e4m3_precision`).
fn encode_blockwise(w: &[f32], n: usize, k: usize, block: usize, decode: fn(u8) -> f32) -> (Vec<u8>, Vec<f32>) {
    let max_rep = max_representable(decode);
    let (rb, cb) = model::fp8::scale_shape(n, k, block);
    let mut scale = vec![0f32; rb * cb];
    for br in 0..rb {
        for bc in 0..cb {
            let mut amax = 0f32;
            for r in br * block..((br + 1) * block).min(n) {
                for c in bc * block..((bc + 1) * block).min(k) {
                    amax = amax.max(w[r * k + c].abs());
                }
            }
            scale[br * cb + bc] = (amax / max_rep).max(1e-12);
        }
    }
    let mut bytes = vec![0u8; n * k];
    for r in 0..n {
        let br = r / block;
        for c in 0..k {
            let bc = c / block;
            let s = scale[br * cb + bc];
            let target = w[r * k + c] / s;
            let mut best = 0u8;
            let mut best_err = f32::INFINITY;
            for b in 0u16..=255 {
                let v = decode(b as u8);
                if !v.is_finite() {
                    continue;
                }
                let err = (target - v).abs();
                if err < best_err {
                    best_err = err;
                    best = b as u8;
                }
            }
            bytes[r * k + c] = best;
        }
    }
    (bytes, scale)
}

/// Pack bytes 4/`u32`, matching the kernel's own `(word >> (8u*(k&3u))) &
/// 0xFFu` extraction exactly: byte `c` of row `r` lives in word `c/4` of
/// that row, at lane `c%4`.
fn pack_bytes_4_per_word(bytes: &[u8], n: usize, k: usize) -> Vec<u32> {
    assert_eq!(k % 4, 0, "this packing requires k % 4 == 0 (got {k})");
    let kw = k / 4;
    let mut words = vec![0u32; n * kw];
    for r in 0..n {
        for c in 0..k {
            words[r * kw + c / 4] |= (bytes[r * k + c] as u32) << (8 * (c % 4));
        }
    }
    words
}

fn check_fp8(kernel_name: &str, decode: fn(u8) -> f32, n: usize, k: usize, seed: u64) {
    let g = gpu_core::testgpu::dev(KERNELS);
    let k_idx = idx(&g, kernel_name);

    let m = 3usize;
    let mut rng = Lcg::new(seed);
    let x_h = rng.vec_scaled(m * k, 1.0);
    let w_h = rng.vec_scaled(n * k, 1.0);

    let (bytes, scale) = encode_blockwise(&w_h, n, k, 128, decode);
    let words = pack_bytes_4_per_word(&bytes, n, k);
    let (rb, cb) = model::fp8::scale_shape(n, k, 128);
    assert_eq!(scale.len(), rb * cb);

    let x = g.storage_init("x", &x_h);
    let wqb = g.storage(words.len() as u64);
    g.write(&wqb, &words);
    let scaleb = g.storage_init("scale", &scale);
    let out = g.storage((m * n) as u64);

    let params = [m as u32, k as u32, n as u32, cb as u32];
    let steps = [g.step(k_idx, &[&x, &wqb, &scaleb, &out], &params, n as u32 * 64)];
    g.submit(&[], &steps);
    let got = g.read(&out, m * n);

    // The oracle: the SAME `model::fp8::dequant_block128` a real checkpoint
    // import uses, over the identical decoded bytes and scale the device
    // read.
    let raw: Vec<f32> = bytes.iter().map(|&b| decode(b)).collect();
    let dequant_w = model::fp8::dequant_block128(&raw, &scale, n, k, 128);
    let want = host_matmul(&x_h, &dequant_w, m, k, n);

    let rel = rel_l2(&got, &want);
    eprintln!("{kernel_name} ({m}x{k}->{n}, block-aligned n%128={} k%128={}): rel_l2={rel:.6}", n % 128, k % 128);
    for (i, v) in got.iter().enumerate() {
        assert!(v.is_finite(), "{kernel_name}: got[{i}]={v} is not finite");
    }
    assert!(rel < 1e-4, "{kernel_name} rel_l2 {rel:.6} >= 1e-4 (weight side is exact, only float summation order should differ)");
}

#[test]
fn matmul_gemv_f8e4m3_matches_dequant_block128_oracle_at_a_128_aligned_shape() {
    check_fp8("matmul_gemv_f8e4m3", checkpoint::safetensors::e4m3fn_to_f32, 128, 256, 8101);
}

#[test]
fn matmul_gemv_f8e4m3_matches_dequant_block128_oracle_at_a_non_128_aligned_shape() {
    // n=200 (not a multiple of 128, partial last row-block of 72 rows),
    // k=192 (not a multiple of 128, partial last col-block of 64 columns;
    // still a multiple of 4 for the byte-packing).
    check_fp8("matmul_gemv_f8e4m3", checkpoint::safetensors::e4m3fn_to_f32, 200, 192, 8102);
}

#[test]
fn matmul_gemv_f8e5m2_matches_dequant_block128_oracle_at_a_128_aligned_shape() {
    check_fp8("matmul_gemv_f8e5m2", model::fp8::e5m2_to_f32, 128, 256, 8103);
}

#[test]
fn matmul_gemv_f8e5m2_matches_dequant_block128_oracle_at_a_non_128_aligned_shape() {
    check_fp8("matmul_gemv_f8e5m2", model::fp8::e5m2_to_f32, 200, 192, 8104);
}

/// The two formats must disagree on the same raw byte with the SAME scale -
/// they are different codebooks (E4M3 has no infinities, a much smaller
/// dynamic range than E5M2's), not one decode with two kernel names.
#[test]
fn f8e4m3_and_f8e5m2_kernels_diverge_on_the_same_bytes() {
    let g = gpu_core::testgpu::dev(KERNELS);
    let (k_e4, k_e5) = (idx(&g, "matmul_gemv_f8e4m3"), idx(&g, "matmul_gemv_f8e5m2"));

    let (m, n, k) = (2usize, 4usize, 128usize);
    let mut rng = Lcg::new(8105);
    // Strictly POSITIVE activations - every weight element decodes to the
    // SAME sign (see below), so a mixed-sign activation would sum `+inf` and
    // `-inf` terms together (NaN, undefined), which is not what this test
    // wants to check. `.abs() + 0.1` keeps every element comfortably away
    // from zero too (`0.0 * inf` is ALSO NaN).
    let x_h: Vec<f32> = rng.vec_scaled(m * k, 1.0).iter().map(|v| v.abs() + 0.1).collect();
    // Bytes with the exponent field maxed (0x7C = 0b01111100): E4M3 decodes
    // this to a large but FINITE value (`checkpoint::safetensors::
    // e4m3fn_to_f32(0x7C)`); E5M2 decodes the identical byte to `+inf`
    // (`model::fp8::e5m2_to_f32(0x7C)`) - same bits, very different math.
    let bytes = vec![0x7Cu8; n * k];
    let scale = vec![1.0f32; model::fp8::scale_shape(n, k, 128).0 * model::fp8::scale_shape(n, k, 128).1];
    let words = pack_bytes_4_per_word(&bytes, n, k);

    let x = g.storage_init("x", &x_h);
    let wqb = g.storage(words.len() as u64);
    g.write(&wqb, &words);
    let scaleb = g.storage_init("scale", &scale);
    let out_e4 = g.storage((m * n) as u64);
    let out_e5 = g.storage((m * n) as u64);
    let cb = model::fp8::scale_shape(n, k, 128).1 as u32;
    let params = [m as u32, k as u32, n as u32, cb];
    let steps = [
        g.step(k_e4, &[&x, &wqb, &scaleb, &out_e4], &params, n as u32 * 64),
        g.step(k_e5, &[&x, &wqb, &scaleb, &out_e5], &params, n as u32 * 64),
    ];
    g.submit(&[], &steps);
    let got_e4 = g.read(&out_e4, m * n);
    let got_e5 = g.read(&out_e5, m * n);

    assert!(got_e4.iter().all(|v| v.is_finite()), "E4M3 output must stay finite: {got_e4:?}");
    assert!(got_e5.iter().any(|v| v.is_infinite()), "E5M2 output must go infinite on 0x7C: {got_e5:?}");
}
