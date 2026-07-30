// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The** host implementations of the elementwise/normalisation math.
//!
//! Some paths legitimately compute on the host: an `m=1` decode matvec, a
//! reference used to check a device result, a backward pass written before its
//! kernel exists. What is *not* legitimate is each crate writing its own — and
//! that is what had happened. Before this module, `rmsnorm` existed **seven**
//! times (one WGSL kernel plus six host copies in `kronos`, `tts`, `chronos2`,
//! `fincast`, `zimage` and `codec`), `rope` three times and `silu` three times.
//!
//! Duplicated math is worse than duplicated plumbing. Every copy is a place the
//! epsilon, the RoPE layout or the reduction order can silently drift from the
//! WGSL kernel that is supposed to be the source of truth, and no test compares
//! copies against each other. So: one implementation, here, and
//! [`tests`](self#tests) checks it against the WGSL kernels themselves through
//! the CPU backend.
//!
//! # This is not a licence to compute on the host
//!
//! Host math does not run on the selected `--device`. A model whose decode path
//! lives here is invisible to the GPU, the Vulkan backend and the NPU no matter
//! what the user asked for, and a benchmark of it reports host numbers under a
//! device label. Anything on a hot path belongs in a WGSL kernel dispatched
//! through `gpu_core`; this module exists so that the host code which *does*
//! remain is written once.

/// `y = x * rsqrt(mean(x²) + eps) * g`, row-wise over `rows × d`.
///
/// Matches `crates/kernels/wgsl/rmsnorm.wgsl` exactly (same reduction, same
/// epsilon placement inside the sqrt).
pub fn rmsnorm_rows(x: &[f32], g: &[f32], rows: usize, d: usize, eps: f32) -> Vec<f32> {
    let (y, _) = rmsnorm_rows_with_inv(x, g, rows, d, eps);
    y
}

/// [`rmsnorm_rows`] plus the per-row `rsqrt` factor, which the backward pass
/// needs and would otherwise recompute.
pub fn rmsnorm_rows_with_inv(
    x: &[f32],
    g: &[f32],
    rows: usize,
    d: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert!(x.len() >= rows * d, "rmsnorm: x is {}, need {}", x.len(), rows * d);
    assert!(g.len() >= d, "rmsnorm: gain is {}, need {d}", g.len());
    let mut y = vec![0.0f32; rows * d];
    let mut inv = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &x[r * d..r * d + d];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let iv = 1.0 / (ms + eps).sqrt();
        inv[r] = iv;
        for i in 0..d {
            y[r * d + i] = row[i] * iv * g[i];
        }
    }
    (y, inv)
}

/// Single-row convenience: `rmsnorm_rows(x, g, 1, x.len(), eps)`.
pub fn rmsnorm(x: &[f32], g: &[f32], eps: f32) -> Vec<f32> {
    rmsnorm_rows(x, g, 1, x.len(), eps)
}

/// `y = (x - mean) * rsqrt(var + eps) * g + b`, row-wise over `rows × c`.
pub fn layernorm_rows(
    x: &[f32],
    g: &[f32],
    b: &[f32],
    rows: usize,
    c: usize,
    eps: f32,
) -> Vec<f32> {
    let (y, _, _) = layernorm_rows_with_stats(x, g, b, rows, c, eps);
    y
}

/// [`layernorm_rows`] plus the per-row `(mean, rsqrt(var+eps))` the backward
/// pass needs.
pub fn layernorm_rows_with_stats(
    x: &[f32],
    g: &[f32],
    b: &[f32],
    rows: usize,
    c: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(x.len() >= rows * c, "layernorm: x is {}, need {}", x.len(), rows * c);
    let mut y = vec![0.0f32; rows * c];
    let mut mean = vec![0.0f32; rows];
    let mut inv = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &x[r * c..r * c + c];
        let m = row.iter().sum::<f32>() / c as f32;
        let var = row.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / c as f32;
        let iv = 1.0 / (var + eps).sqrt();
        mean[r] = m;
        inv[r] = iv;
        for j in 0..c {
            y[r * c + j] = (row[j] - m) * iv * g[j] + b[j];
        }
    }
    (y, mean, inv)
}

/// SiLU / swish: `x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Elementwise [`silu`].
pub fn silu_slice(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| silu(v)).collect()
}

/// NeoX **half-split** rotary embedding, in place.
///
/// `buf` is `rows × heads × head_dim`; row `r` is rotated at absolute position
/// `pos0 + r`. The half-split pairs element `j` with `j + hd/2` (as opposed to
/// the interleaved layout that pairs `2j` with `2j+1`) — matching
/// `crates/kernels/wgsl/rope.wgsl`.
pub fn rope_neox(buf: &mut [f32], rows: usize, heads: usize, hd: usize, pos0: usize, theta: f32) {
    assert!(hd % 2 == 0, "rope: head_dim {hd} must be even");
    assert!(buf.len() >= rows * heads * hd, "rope: buffer too small");
    let half = hd / 2;
    for r in 0..rows {
        let pos = (pos0 + r) as f32;
        for h in 0..heads {
            let base = (r * heads + h) * hd;
            for j in 0..half {
                let angle = pos * theta.powf(-(2.0 * j as f32) / hd as f32);
                let (s, c) = angle.sin_cos();
                let a = buf[base + j];
                let b = buf[base + j + half];
                buf[base + j] = a * c - b * s;
                buf[base + j + half] = b * c + a * s;
            }
        }
    }
}

/// One row's worth of [`rope_neox`] at a single absolute position — the decode
/// step's `m=1` case.
pub fn rope_neox_row(buf: &mut [f32], heads: usize, hd: usize, pos: usize, theta: f32) {
    rope_neox(buf, 1, heads, hd, pos, theta);
}

/// `out[o] = Σ_k w[o*inn + k] * x[k]` — `y = x·Wᵀ` with `W: [out, inn]`
/// row-major, matching `matmul.wgsl`'s weight layout at `m = 1`.
pub fn matvec(w: &[f32], x: &[f32], out: usize, inn: usize) -> Vec<f32> {
    assert!(w.len() >= out * inn, "matvec: w is {}, need {}", w.len(), out * inn);
    assert!(x.len() >= inn, "matvec: x is {}, need {inn}", x.len());
    (0..out).map(|o| w[o * inn..o * inn + inn].iter().zip(x).map(|(a, b)| a * b).sum()).collect()
}

/// [`matvec`] across all cores — the LM-head shape (one hidden row against a
/// `[vocab, d]` table) is 100+ MFLOP per token at real vocabularies, which a
/// single core turns into hundreds of milliseconds PER TOKEN (measured: 277
/// ms/token in the caption decode, dominated by exactly this call). Same
/// contract, same result, row-parallel via the CPU scheduler's primitives.
#[cfg(not(target_arch = "wasm32"))]
pub fn matvec_par(w: &[f32], x: &[f32], out: usize, inn: usize) -> Vec<f32> {
    assert!(w.len() >= out * inn, "matvec_par: w is {}, need {}", w.len(), out * inn);
    assert!(x.len() >= inn, "matvec_par: x is {}, need {inn}", x.len());
    backend_cpu::par::map_f32(out, |o| {
        w[o * inn..o * inn + inn].iter().zip(x).map(|(a, b)| a * b).sum()
    })
}

/// Numerically-stable softmax over a slice, in place.
pub fn softmax(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return;
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
    }

    /// The point of this module: the host result must equal what the WGSL
    /// kernel computes. Run through the CPU backend so the check is
    /// deterministic and needs no GPU.
    #[test]
    fn rmsnorm_matches_the_wgsl_kernel() {
        let (rows, d, eps) = (5usize, 64usize, 1e-6f32);
        let mut r = rng(7);
        let x: Vec<f32> = (0..rows * d).map(|_| r()).collect();
        let g: Vec<f32> = (0..d).map(|_| r() * 0.5 + 1.0).collect();

        let gpu = gpu_core::Gpu::new_cpu(&[("rmsnorm", kernels::RMSNORM)]);
        let xb = gpu.storage_init("x", &x);
        let gb = gpu.storage_init("g", &g);
        let ob = gpu.storage(rows as u64 * d as u64);
        let step = gpu.step(0, &[&xb, &gb, &ob], &[d as u32, rows as u32], rows as u32);
        gpu.submit(&[], &[step]);
        let want = gpu.read(&ob, rows * d);

        let got = rmsnorm_rows(&x, &g, rows, d, eps);
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 1e-5, "element {i}: host {a} vs wgsl {b}");
        }
    }

    #[test]
    fn rmsnorm_inv_is_the_row_rsqrt() {
        let (rows, d, eps) = (3usize, 8usize, 1e-6f32);
        let x: Vec<f32> = (0..rows * d).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let g = vec![1.0f32; d];
        let (_, inv) = rmsnorm_rows_with_inv(&x, &g, rows, d, eps);
        for r in 0..rows {
            let ms = x[r * d..r * d + d].iter().map(|v| v * v).sum::<f32>() / d as f32;
            assert!((inv[r] - 1.0 / (ms + eps).sqrt()).abs() < 1e-6);
        }
    }

    #[test]
    fn single_row_helper_matches_the_rows_form() {
        let d = 16;
        let mut r = rng(3);
        let x: Vec<f32> = (0..d).map(|_| r()).collect();
        let g: Vec<f32> = (0..d).map(|_| r()).collect();
        assert_eq!(rmsnorm(&x, &g, 1e-6), rmsnorm_rows(&x, &g, 1, d, 1e-6));
    }

    #[test]
    fn layernorm_centres_and_scales() {
        let (rows, c, eps) = (4usize, 32usize, 1e-5f32);
        let mut r = rng(11);
        let x: Vec<f32> = (0..rows * c).map(|_| r() * 3.0 + 2.0).collect();
        let g = vec![1.0f32; c];
        let b = vec![0.0f32; c];
        let (y, mean, inv) = layernorm_rows_with_stats(&x, &g, &b, rows, c, eps);
        for r0 in 0..rows {
            let row = &y[r0 * c..r0 * c + c];
            let m = row.iter().sum::<f32>() / c as f32;
            assert!(m.abs() < 1e-4, "row {r0} mean {m} should be ~0");
            assert!(inv[r0] > 0.0);
            let src = &x[r0 * c..r0 * c + c];
            assert!((mean[r0] - src.iter().sum::<f32>() / c as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn silu_is_x_times_sigmoid() {
        for &v in &[-4.0f32, -0.5, 0.0, 0.5, 4.0] {
            let want = v * (1.0 / (1.0 + (-v).exp()));
            assert!((silu(v) - want).abs() < 1e-6, "silu({v})");
        }
        assert_eq!(silu(0.0), 0.0);
        assert_eq!(silu_slice(&[0.0, 0.0]), vec![0.0, 0.0]);
    }

    /// RoPE is a rotation: it must preserve the norm of each (j, j+half) pair,
    /// and position 0 must be the identity.
    #[test]
    fn rope_is_a_norm_preserving_rotation() {
        let (heads, hd, theta) = (2usize, 8usize, 10000.0f32);
        let mut r = rng(5);
        let orig: Vec<f32> = (0..heads * hd).map(|_| r()).collect();

        let mut at0 = orig.clone();
        rope_neox_row(&mut at0, heads, hd, 0, theta);
        for (a, b) in at0.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-6, "position 0 must be the identity");
        }

        let mut at7 = orig.clone();
        rope_neox_row(&mut at7, heads, hd, 7, theta);
        let half = hd / 2;
        for h in 0..heads {
            for j in 0..half {
                let (i0, i1) = (h * hd + j, h * hd + j + half);
                let before = orig[i0] * orig[i0] + orig[i1] * orig[i1];
                let after = at7[i0] * at7[i0] + at7[i1] * at7[i1];
                assert!((before - after).abs() < 1e-4, "rotation must preserve the pair norm");
            }
        }
    }

    /// The property the KV-cache fast paths rely on: `q_i · k_j` depends only on
    /// `i − j`, so keys cached at absolute positions stay valid as the window
    /// slides.
    #[test]
    fn rope_dot_product_depends_only_on_relative_position() {
        let (heads, hd, theta) = (1usize, 16usize, 10000.0f32);
        let mut r = rng(9);
        let q0: Vec<f32> = (0..hd).map(|_| r()).collect();
        let k0: Vec<f32> = (0..hd).map(|_| r()).collect();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();

        let mut q_a = q0.clone();
        let mut k_a = k0.clone();
        rope_neox_row(&mut q_a, heads, hd, 5, theta);
        rope_neox_row(&mut k_a, heads, hd, 2, theta);

        let mut q_b = q0.clone();
        let mut k_b = k0.clone();
        rope_neox_row(&mut q_b, heads, hd, 11, theta);
        rope_neox_row(&mut k_b, heads, hd, 8, theta);

        assert!(
            (dot(&q_a, &k_a) - dot(&q_b, &k_b)).abs() < 1e-4,
            "same offset (3) must give the same dot product"
        );
    }

    #[test]
    fn rope_rows_advance_the_position() {
        let (heads, hd, theta) = (1usize, 8usize, 10000.0f32);
        let base: Vec<f32> = (0..hd).map(|i| i as f32 * 0.1).collect();
        let mut rows2 = [base.clone(), base.clone()].concat();
        rope_neox(&mut rows2, 2, heads, hd, 3, theta);
        let mut r0 = base.clone();
        let mut r1 = base.clone();
        rope_neox_row(&mut r0, heads, hd, 3, theta);
        rope_neox_row(&mut r1, heads, hd, 4, theta);
        assert_eq!(&rows2[..hd], &r0[..]);
        assert_eq!(&rows2[hd..], &r1[..]);
    }

    #[test]
    fn matvec_is_x_times_w_transposed() {
        // W = [[1,2],[3,4],[5,6]] (out=3, inn=2), x = [1,1] => [3,7,11]
        let w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0f32, 1.0];
        assert_eq!(matvec(&w, &x, 3, 2), vec![3.0, 7.0, 11.0]);
    }

    #[test]
    fn softmax_normalises_and_is_shift_invariant() {
        let mut a = vec![1.0f32, 2.0, 3.0];
        let mut b = vec![101.0f32, 102.0, 103.0];
        softmax(&mut a);
        softmax(&mut b);
        assert!((a.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6, "softmax must be shift-invariant");
        }
    }
}

/// Seeded standard-normal samples (xorshift64* + Box-Muller) — the shared
/// diffusion latent-noise source (Z-Image, FLUX.2). Deterministic per seed;
/// NOT the torch Philox stream, so cross-framework runs are statistically
/// equivalent rather than bit-identical.
pub fn randn(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        // to (0,1)
        ((s >> 11) as f64 / (1u64 << 53) as f64).clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let (u1, u2) = (next(), next());
        let r = (-2.0 * u1.ln()).sqrt();
        out.push((r * (std::f64::consts::TAU * u2).cos()) as f32);
        if out.len() < n {
            out.push((r * (std::f64::consts::TAU * u2).sin()) as f32);
        }
    }
    out
}

