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

/// A spread of output positions of an oracle: the oracle is exact f64 over the full reduction, but only
/// for these positions (at most `PICKS`, strided so every row and column of a large product is hit), which
/// keeps a 5120-deep product over a 512x2048 output from costing minutes of unoptimised host arithmetic.
const PICKS: usize = 4096;

struct Sampled {
    index: Vec<usize>,
    want: Vec<f64>,
}

impl Sampled {
    /// `value(i)` at the picked positions of a `total`-element output.
    fn of(total: usize, value: impl Fn(usize) -> f64) -> Sampled {
        let step = (total / PICKS).max(1) | 1;
        let mut index: Vec<usize> = (0..total).step_by(step).collect();
        index.push(total - 1);
        let want = index.iter().map(|&i| value(i)).collect();
        Sampled { index, want }
    }

    fn scaled(&self, factor: f64) -> Sampled {
        Sampled { index: self.index.clone(), want: self.want.iter().map(|v| v * factor).collect() }
    }

    fn got(&self, full: &[f32]) -> Vec<f64> {
        self.index.iter().map(|&i| full[i] as f64).collect()
    }

    fn worst_relative_to_rms(&self, full: &[f32]) -> f64 {
        let worst = self.got(full).iter().zip(&self.want).map(|(g, w)| (g - w).abs()).fold(0.0, f64::max);
        worst / rms(&self.want)
    }
}

/// `out[m, n] = x[m, k] @ w[n, k]^T` in f64 over the bf16-rounded activation.
fn forward_oracle(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Sampled {
    Sampled::of(m * n, |i| {
        let (r, c) = (i / n, i % n);
        (0..k).map(|t| round_bf16(x[r * k + t]) as f64 * w[c * k + t] as f64).sum()
    })
}

/// `dx[m, k] = dy[m, n] @ w[n, k]` in f64 over the bf16-rounded gradient.
fn dx_oracle(dy: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Sampled {
    Sampled::of(m * k, |i| {
        let (r, c) = (i / k, i % k);
        (0..n).map(|t| round_bf16(dy[r * n + t]) as f64 * w[t * k + c] as f64).sum()
    })
}

fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

fn assert_close(got: &[f32], want: &Sampled, what: &str) {
    let rel = want.worst_relative_to_rms(got);
    assert!(rel <= 1e-2, "{what}: worst error is {rel} of the output RMS (bar 1e-2)");
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
        let twice = want.scaled(2.0);
        assert_close(&gpu.read(&dxb, mu * ku), &twice, &format!("accumulating dx {m}x{n}x{k}"));
    }
}

/// `out[m, n] = x[m, k] @ w[n, k]^T` in f64 over unrounded fp32 operands (sampled, see [`Sampled`]).
fn forward_exact(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Sampled {
    Sampled::of(m * n, |i| (0..k).map(|t| x[(i / n) * k + t] as f64 * w[(i % n) * k + t] as f64).sum())
}

/// The default (Auto) tier multiplies fp32-stored weights and activations on tensor cores with no
/// copy of either: its error against the exact product is the operands' 10-bit mantissa
/// (about 2^-11 relative each), far below the bf16 tier's, and the plain fp32 tier stays
/// exact. Forward and dx, over reduced 8B projection shapes (k = 4096 and 12288).
#[test]
fn the_tf32_tier_multiplies_fp32_operands_within_tf32_rounding() {
    let Some(gpu) = open() else { return };
    let provider = LibraryGemmProvider::for_gpu(&gpu, GemmPrecision::Auto);
    if !provider.tf32_enabled() {
        brain_testutil::skip_unavailable(&format!("cuda_library_gemm: this card has no TF32 tier ({:?})", provider.path()));
        return;
    }
    assert!(!LibraryGemmProvider::for_gpu(&gpu, GemmPrecision::Fp32Reference).tf32_enabled(), "the pinned reference tier must never take TF32");
    for &(m, n, k) in &[(8u32, 4096u32, 4096u32), (512, 2048, 4096), (300, 1024, 12288), (64, 1024, 4096)] {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let w = values(nu * ku, 0.05, 11);
        let x = values(mu * ku, 1.0, 12);
        let dy = values(mu * nu, 1.0, 13);
        let whole = |buf| Operand { buf, range: (0, 0) };
        let (wb, xb, yb) = (gpu.storage_init("w", &w), gpu.storage_init("x", &x), gpu.storage((mu * nu) as u64));
        let mut steps = Vec::new();
        assert!(provider.forward_tf32(&gpu, &mut steps, &whole(&xb), &whole(&wb), &whole(&yb), m, n, k), "tf32 forward {m}x{n}x{k} declined");
        gpu.submit(&[], &steps);
        let want = forward_exact(&x, &w, mu, nu, ku);
        let rel = want.worst_relative_to_rms(&gpu.read(&yb, mu * nu));
        assert!(rel <= 4e-3, "tf32 forward {m}x{n}x{k}: worst error is {rel} of the output RMS");
        assert!(rel >= 1e-6, "tf32 forward {m}x{n}x{k}: error {rel} is fp32-exact, so the TF32 path did not run");

        let (dyb, dxb) = (gpu.storage_init("dy", &dy), gpu.storage((mu * ku) as u64));
        let mut steps = Vec::new();
        assert!(provider.dx_tf32(&gpu, &mut steps, &whole(&dyb), &whole(&wb), &whole(&dxb), m, n, k, false), "tf32 dx {m}x{n}x{k} declined");
        gpu.submit(&[], &steps);
        let want = Sampled::of(mu * ku, |i| (0..nu).map(|t| dy[(i / ku) * nu + t] as f64 * w[t * ku + i % ku] as f64).sum());
        let rel = want.worst_relative_to_rms(&gpu.read(&dxb, mu * ku));
        assert!(rel <= 4e-3, "tf32 dx {m}x{n}x{k}: worst error is {rel} of the output RMS");
    }
}

/// Attention's two products over the heads of a fused `[rows, 3 * d]` buffer: `S = alpha Q K^T` and `ctx = P V`, each one
/// strided batch with a head per entry, on tensor cores. Operands are slices of the fused rows (row stride `3 d`, entry
/// stride one head), the output a slice of a `[rows, d]` buffer.
#[test]
fn batched_tf32_products_slice_heads_out_of_a_fused_buffer() {
    use gpu_core::provider::library::BatchedProduct;
    let Some(gpu) = open() else { return };
    let provider = LibraryGemmProvider::for_gpu(&gpu, GemmPrecision::Auto);
    if !provider.tf32_enabled() {
        brain_testutil::skip_unavailable(&format!("cuda_library_gemm: this card has no TF32 tier ({:?})", provider.path()));
        return;
    }
    let (heads, hd, rows, q0, qn) = (3usize, 16usize, 40usize, 8usize, 24usize);
    let d = heads * hd;
    let qkv = values(rows * 3 * d, 1.0, 21);
    let buf = gpu.storage_init("qkv", &qkv);
    let at = |buf, word: usize| Operand { buf, range: (word as u64, 0) };
    let whole = |buf| Operand { buf, range: (0, 0) };
    let alpha = 0.25f32;

    // S[h][i][j] = alpha * sum_t Q[q0 + i][h][t] K[j][h][t]
    let scores = gpu.storage((heads * qn * rows) as u64);
    let nt = BatchedProduct { batch: heads as u32, m: qn as u32, n: rows as u32, k: hd as u32, lda: (3 * d) as u32, ldb: (3 * d) as u32, ldc: rows as u32, stride_a: hd as i64, stride_b: hd as i64, stride_c: (qn * rows) as i64, alpha };
    let mut steps = Vec::new();
    assert!(provider.bmm_nt_tf32(&gpu, &mut steps, &at(&buf, q0 * 3 * d), &at(&buf, d), &whole(&scores), &nt), "batched Q K^T declined");
    gpu.submit(&[], &steps);
    let q = |i: usize, h: usize, t: usize| qkv[(q0 + i) * 3 * d + h * hd + t] as f64;
    let k = |j: usize, h: usize, t: usize| qkv[j * 3 * d + d + h * hd + t] as f64;
    let want = Sampled::of(heads * qn * rows, |x| {
        let (h, i, j) = (x / (qn * rows), (x / rows) % qn, x % rows);
        alpha as f64 * (0..hd).map(|t| q(i, h, t) * k(j, h, t)).sum::<f64>()
    });
    let rel = want.worst_relative_to_rms(&gpu.read(&scores, heads * qn * rows));
    assert!(rel <= 4e-3, "batched Q K^T: worst error {rel} of the output RMS");

    // ctx[q0 + i][h][t] = sum_j S[h][i][j] V[j][h][t], written into the head's columns of a [rows, d] buffer.
    let ctx = gpu.storage((rows * d) as u64);
    let nn = BatchedProduct { batch: heads as u32, m: qn as u32, n: hd as u32, k: rows as u32, lda: rows as u32, ldb: (3 * d) as u32, ldc: d as u32, stride_a: (qn * rows) as i64, stride_b: hd as i64, stride_c: hd as i64, alpha: 1.0 };
    let mut steps = Vec::new();
    assert!(provider.bmm_nn_tf32(&gpu, &mut steps, &whole(&scores), &at(&buf, 2 * d), &at(&ctx, q0 * d), &nn), "batched P V declined");
    gpu.submit(&[], &steps);
    let s = gpu.read(&scores, heads * qn * rows);
    let v = |j: usize, h: usize, t: usize| qkv[j * 3 * d + 2 * d + h * hd + t] as f64;
    let got = gpu.read(&ctx, rows * d);
    let want = Sampled::of(qn * d, |x| {
        let (i, h, t) = (x / d, (x % d) / hd, x % hd);
        (0..rows).map(|j| s[(h * qn + i) * rows + j] as f64 * v(j, h, t)).sum()
    });
    let rel = want.worst_relative_to_rms(&got[q0 * d..(q0 + qn) * d]);
    assert!(rel <= 4e-3, "batched P V: worst error {rel} of the output RMS");
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
        let p = F32Product { m: m as u32, n: n as u32, k: k as u32, trans_a, trans_b, alpha: 0.5, beta: 1.0, tf32: false };
        assert!(provider.gemm_f32(&gpu, &mut steps, &whole(&ab), &whole(&bb), &whole(&cb), p));
        gpu.submit(&[], &steps);
        let got = gpu.read(&cb, m * n);
        let worst = got.iter().zip(&want).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-4, "trans_a={trans_a} trans_b={trans_b}: worst error {worst}");
    }
}

/// The TF32 form of the same product rounds each operand to 10 mantissa bits (a relative error of 2^-11 each) and
/// accumulates in fp32: against the f64 oracle the error stays within a few of those roundings of the result's
/// scale, and it differs from the exact form (the flag is honoured, not ignored).
#[test]
fn the_tf32_f32_product_stays_within_operand_rounding_of_the_oracle() {
    let Some(gpu) = open() else { return };
    let provider = LibraryGemmProvider::for_gpu(&gpu, GemmPrecision::Auto);
    if !provider.is_active() {
        return;
    }
    let (m, n, k) = (96usize, 16usize, 256usize);
    let a = values(m * k, 1.0, 11);
    let b = values(n * k, 1.0, 12);
    let want: Vec<f64> = (0..m * n).map(|x| (0..k).map(|t| a[(x / n) * k + t] as f64 * b[(x % n) * k + t] as f64).sum::<f64>()).collect();
    let rms = (want.iter().map(|w| w * w).sum::<f64>() / want.len() as f64).sqrt();
    let run = |tf32: bool| {
        let (ab, bb, cb) = (gpu.storage_init("a", &a), gpu.storage_init("b", &b), gpu.storage(m as u64 * n as u64));
        let whole = |buf| Operand { buf, range: (0, 0) };
        let mut steps = Vec::new();
        let p = F32Product { m: m as u32, n: n as u32, k: k as u32, trans_a: false, trans_b: true, alpha: 1.0, beta: 0.0, tf32 };
        assert!(provider.gemm_f32(&gpu, &mut steps, &whole(&ab), &whole(&bb), &whole(&cb), p));
        gpu.submit(&[], &steps);
        let got = gpu.read(&cb, m * n);
        got.iter().zip(&want).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0, f64::max) / rms
    };
    let (exact, tf32) = (run(false), run(true));
    assert!(exact < 1e-5, "the exact form drifted: {exact}");
    assert!(tf32 > 10.0 * exact && tf32 < 3e-3, "TF32 error {tf32} (exact {exact}) should sit at operand-rounding scale");
}
