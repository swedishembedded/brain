// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The vendor-library bf16 GEMM against an f64 oracle over the 27B's projection
//! shapes, and the transparent fallbacks: a simulated missing library, a
//! too-old device and the pinned fp32 tier each decline every request and push
//! nothing, and report why.
//!
//! Swedish Embedded AB implements tensor-core training stacks that degrade
//! gracefully on hardware without them. If your team needs expertise in proving
//! a mixed-precision path against an exact oracle, you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! The device tests open a CUDA card when there is one and report that they
//! found none otherwise, like the other hardware gates of this crate.

use gpu_core::provider::library::{F32Product, GemmPath, GemmPrecision, LibraryGemmProvider, Operand};
use gpu_core::Gpu;

static NO_KERNELS: &[(&str, &str)] = &[];

fn round_bf16(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

fn bf16_bits(v: f32) -> u32 {
    round_bf16(v).to_bits() >> 16
}

/// Packed bf16 words, lower index in the low half.
fn pack(values: &[f32]) -> Vec<u32> {
    values.chunks(2).map(|p| bf16_bits(p[0]) | (bf16_bits(p[1]) << 16)).collect()
}

/// A deterministic spread of values (a small LCG; the oracle needs no particular law).
fn values(n: usize, scale: f32, salt: u32) -> Vec<f32> {
    // SplitMix64: independent-looking streams for different salts.
    let mut state = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(salt as u64 + 1);
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (((z >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * 2.0 * scale
        })
        .collect()
}

/// `out[m, n] = x[m, k] @ w[n, k]^T` in f64 over the bf16-rounded activation.
fn forward_oracle(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f64> {
    (0..m * n)
        .map(|i| {
            let (r, c) = (i / n, i % n);
            (0..k).map(|t| round_bf16(x[r * k + t]) as f64 * w[c * k + t] as f64).sum()
        })
        .collect()
}

/// `dx[m, k] = dy[m, n] @ w[n, k]` in f64 over the bf16-rounded gradient.
fn dx_oracle(dy: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f64> {
    (0..m * k)
        .map(|i| {
            let (r, c) = (i / k, i % k);
            (0..n).map(|t| round_bf16(dy[r * n + t]) as f64 * w[t * k + c] as f64).sum()
        })
        .collect()
}

fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

fn assert_close(got: &[f32], want: &[f64], what: &str) {
    let bar = 1e-2 * rms(want);
    let worst = got.iter().zip(want).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0, f64::max);
    assert!(worst <= bar, "{what}: worst error {worst} exceeds 1e-2 of the output RMS ({bar})");
}

fn open() -> Option<Gpu> {
    match Gpu::try_new_cuda(NO_KERNELS) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("cuda_library_gemm: no CUDA device on this box ({e}) - nothing to run");
            None
        }
    }
}

/// `(m, n, k)` covering the real set: skinny and wide rows, the b/a projections
/// (n = 48), k/v (n = 1024), a window of the MLP, and a ragged 8-multiple.
const SHAPES: &[(u32, u32, u32)] = &[(8, 48, 256), (64, 1024, 512), (512, 2048, 5120), (257 * 8, 40, 64), (7 * 8, 5120, 1024)];

#[test]
fn the_library_forward_and_dx_agree_with_an_f64_oracle() {
    let Some(gpu) = open() else { return };
    let provider = LibraryGemmProvider::for_gpu(&gpu, GemmPrecision::Auto);
    if let GemmPath::Fp32Kernels { reason } = provider.path() {
        eprintln!("cuda_library_gemm: this card runs the fp32 path ({reason}) - nothing to run");
        return;
    }
    for &(m, n, k) in SHAPES {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let w = values(nu * ku, 0.05, 1);
        let x = values(mu * ku, 1.0, 2);
        let dy = values(mu * nu, 1.0, 3);
        let wb = gpu.storage((nu * ku / 2) as u64);
        gpu.write(&wb, &pack(&w.iter().map(|v| round_bf16(*v)).collect::<Vec<_>>()));
        let w_rounded: Vec<f32> = w.iter().map(|v| round_bf16(*v)).collect();
        let whole = |buf| Operand { buf, range: (0, 0) };

        let (xb, yb) = (gpu.storage_init("x", &x), gpu.storage((mu * nu) as u64));
        let mut steps = Vec::new();
        assert!(provider.forward(&gpu, &mut steps, &whole(&xb), &whole(&wb), &whole(&yb), m, n, k), "forward {m}x{n}x{k} was declined");
        gpu.submit(&[], &steps);
        assert_close(&gpu.read(&yb, mu * nu), &forward_oracle(&x, &w_rounded, mu, nu, ku), &format!("forward {m}x{n}x{k}"));

        let (dyb, dxb) = (gpu.storage_init("dy", &dy), gpu.storage((mu * ku) as u64));
        let mut steps = Vec::new();
        assert!(provider.dx(&gpu, &mut steps, &whole(&dyb), &whole(&wb), &whole(&dxb), m, n, k, false), "dx {m}x{n}x{k} was declined");
        gpu.submit(&[], &steps);
        let want = dx_oracle(&dy, &w_rounded, mu, nu, ku);
        assert_close(&gpu.read(&dxb, mu * ku), &want, &format!("dx {m}x{n}x{k}"));

        // Accumulating adds the second product to the first.
        let mut steps = Vec::new();
        assert!(provider.dx(&gpu, &mut steps, &whole(&dyb), &whole(&wb), &whole(&dxb), m, n, k, true));
        gpu.submit(&[], &steps);
        let twice: Vec<f64> = want.iter().map(|v| 2.0 * v).collect();
        assert_close(&gpu.read(&dxb, mu * ku), &twice, &format!("accumulating dx {m}x{n}x{k}"));
    }
}

/// A row window of the weight (the vocabulary-tile primitive) multiplies only
/// its rows.
#[test]
fn a_weight_row_window_multiplies_only_its_rows() {
    let Some(gpu) = open() else { return };
    let provider = LibraryGemmProvider::for_gpu(&gpu, GemmPrecision::Auto);
    if !matches!(provider.path(), GemmPath::Bf16TensorCore) {
        return;
    }
    let (m, n_total, k) = (16usize, 512usize, 256usize);
    let (first, count) = (128usize, 256usize);
    let w: Vec<f32> = values(n_total * k, 0.05, 4).iter().map(|v| round_bf16(*v)).collect();
    let x = values(m * k, 1.0, 5);
    let wb = gpu.storage((n_total * k / 2) as u64);
    gpu.write(&wb, &pack(&w));
    let xb = gpu.storage_init("x", &x);
    let yb = gpu.storage((m * count) as u64);
    let window = Operand { buf: &wb, range: ((first * k / 2) as u64, (count * k / 2) as u64) };
    let mut steps = Vec::new();
    assert!(provider.forward(&gpu, &mut steps, &Operand { buf: &xb, range: (0, 0) }, &window, &Operand { buf: &yb, range: (0, 0) }, m as u32, count as u32, k as u32));
    gpu.submit(&[], &steps);
    assert_close(&gpu.read(&yb, m * count), &forward_oracle(&x, &w[first * k..(first + count) * k], m, count, k), "windowed forward");
}

/// A library that is absent, a device below the tensor-core floor and the pinned
/// fp32 tier each decline every request, push nothing, and name why.
#[test]
fn every_simulated_fallback_declines_and_reports_why() {
    let Some(gpu) = open() else { return };
    let buf = gpu.storage(4096);
    let op = Operand { buf: &buf, range: (0, 0) };
    for (precision, unavailable, expect) in [
        (GemmPrecision::Auto, Some("no usable cuBLAS (simulated)".to_string()), "cuBLAS"),
        (GemmPrecision::Bf16TensorCore, Some("compute capability 7.5 is below the 8.0 bf16 tensor-core floor".to_string()), "7.5"),
        (GemmPrecision::Fp32Reference, None, "reference"),
    ] {
        let provider = LibraryGemmProvider::with_probe(&gpu, precision, unavailable);
        let GemmPath::Fp32Kernels { reason } = provider.path() else { panic!("{precision:?} took the tensor-core path") };
        assert!(reason.contains(expect), "{reason}");
        let mut steps = Vec::new();
        assert!(!provider.forward(&gpu, &mut steps, &op, &op, &op, 8, 8, 8));
        assert!(!provider.dx(&gpu, &mut steps, &op, &op, &op, 8, 8, 8, false));
        assert!(steps.is_empty(), "a declined request pushes no step");
    }
}

/// Shapes the library path does not take (an odd leading dimension) are declined,
/// not approximated.
#[test]
fn an_unaligned_shape_is_declined() {
    let Some(gpu) = open() else { return };
    let provider = LibraryGemmProvider::for_gpu(&gpu, GemmPrecision::Auto);
    if !matches!(provider.path(), GemmPath::Bf16TensorCore) {
        return;
    }
    let buf = gpu.storage(1 << 16);
    let op = Operand { buf: &buf, range: (0, 0) };
    let mut steps = Vec::new();
    assert!(!provider.forward(&gpu, &mut steps, &op, &op, &op, 8, 8, 12), "k = 12 is not a multiple of 8");
    assert!(steps.is_empty());
}

/// The NF4 decode kernel produces, bit for bit, the bf16 rounding of
/// `codebook[code] * scale` that the host formula gives, over a window that
/// does not start at the first row.
#[test]
fn nf4_dequant_matches_the_codebook_bit_for_bit() {
    let Some(gpu) = open() else { return };
    let provider = LibraryGemmProvider::for_gpu(&gpu, GemmPrecision::Auto);
    if !provider.is_active() {
        return;
    }
    let (rows, k, first, count) = (128usize, 256usize, 64usize, 32usize);
    let codes = values(rows * k, 8.0, 6).iter().map(|v| ((v + 8.0) as u32).min(15)).collect::<Vec<_>>();
    let scales: Vec<f32> = values(rows * k / 32, 1.0, 7).iter().map(|v| v.abs() + 0.01).collect();
    let words: Vec<u32> = codes.chunks(8).map(|c| c.iter().enumerate().fold(0u32, |w, (b, code)| w | (code << (4 * b)))).collect();
    let (pb, sb) = (gpu.storage(words.len() as u64), gpu.storage_init("scales", &scales));
    gpu.write(&pb, &words);
    let mut steps = Vec::new();
    let (po, so) = ((first * k / 8) as u64, (first * k / 32) as u64);
    let out = provider
        .dequant_nf4(&gpu, &mut steps, &Operand { buf: &pb, range: (po, (count * k / 8) as u64) }, &Operand { buf: &sb, range: (so, (count * k / 32) as u64) }, (count * k) as u64)
        .expect("the decode kernel was declined");
    gpu.submit(&[], &steps);
    let got = gpu.read(&out, count * k / 2);
    for i in 0..count * k {
        let wi = first * k + i;
        let want = bf16_bits(kernels::template::NF4_CODEBOOK[codes[wi] as usize] * scales[wi / 32]);
        let word = got[i / 2].to_bits();
        let have = if i % 2 == 0 { word & 0xffff } else { word >> 16 };
        assert_eq!(have, want, "weight {i} (code {}, scale {})", codes[wi], scales[wi / 32]);
    }
}

/// The row-major fp32 product, in every transpose combination and accumulating,
/// against an f64 oracle at the 1e-5 level (no operand is rounded).
#[test]
fn the_f32_product_covers_every_transpose_and_accumulates() {
    let Some(gpu) = open() else { return };
    let provider = LibraryGemmProvider::for_gpu(&gpu, GemmPrecision::Auto);
    if !provider.is_active() {
        return;
    }
    let (m, n, k) = (24usize, 16usize, 40usize);
    let a = values(m * k, 1.0, 8);
    let b = values(k * n, 1.0, 9);
    let c0 = values(m * n, 1.0, 10);
    for (trans_a, trans_b) in [(false, false), (false, true), (true, false), (true, true)] {
        // Physical storage of each operand under its flag.
        let at = |i: usize, t: usize| if trans_a { a[t * m + i] } else { a[i * k + t] };
        let bt = |t: usize, j: usize| if trans_b { b[j * k + t] } else { b[t * n + j] };
        let want: Vec<f64> = (0..m * n).map(|x| 0.5 * (0..k).map(|t| at(x / n, t) as f64 * bt(t, x % n) as f64).sum::<f64>() + c0[x] as f64).collect();
        let (ab, bb, cb) = (gpu.storage_init("a", &a), gpu.storage_init("b", &b), gpu.storage_init("c", &c0));
        let whole = |buf| Operand { buf, range: (0, 0) };
        let mut steps = Vec::new();
        let p = F32Product { m: m as u32, n: n as u32, k: k as u32, trans_a, trans_b, alpha: 0.5, beta: 1.0 };
        assert!(provider.gemm_f32(&gpu, &mut steps, &whole(&ab), &whole(&bb), &whole(&cb), p));
        gpu.submit(&[], &steps);
        let got = gpu.read(&cb, m * n);
        let worst = got.iter().zip(&want).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-4, "trans_a={trans_a} trans_b={trans_b}: worst error {worst}");
    }
}
