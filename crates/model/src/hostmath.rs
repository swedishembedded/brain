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

/// Sinusoidal timestep embedding, **cos block first** then sin — the layout
/// every diffusion model in this repo needs:
///
/// ```text
/// half     = dim / 2
/// freq[k]  = max_period^(-k/half)
/// e[k]     = cos(t · freq[k])          k in 0..half
/// e[half+k]= sin(t · freq[k])
/// ```
///
/// This is `flux.modules.layers.timestep_embedding` (BFL) and diffusers'
/// `Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0)` — they agree.
/// `t` is the ALREADY-SCALED time (both references pre-multiply by their
/// `time_factor`, 1000 for the FLUX family), so this function applies no
/// scaling of its own.
///
/// The angle is accumulated in `f64` and rounded once, as the references do.
/// `dim` must be even.
pub fn timestep_embedding(t: f32, dim: usize, max_period: f64) -> Vec<f32> {
    assert!(dim.is_multiple_of(2), "timestep_embedding: dim {dim} must be even");
    let half = dim / 2;
    let mut e = vec![0.0f32; dim];
    for k in 0..half {
        let freq = (-(max_period.ln()) * k as f64 / half as f64).exp();
        let arg = t as f64 * freq;
        e[k] = arg.cos() as f32;
        e[half + k] = arg.sin() as f32;
    }
    e
}

/// NeoX **half-split** rotary embedding, in place.
///
/// `buf` is `rows × heads × head_dim`; row `r` is rotated at absolute position
/// `pos0 + r`. The half-split pairs element `j` with `j + hd/2` (as opposed to
/// the interleaved layout that pairs `2j` with `2j+1`) — matching
/// `crates/kernels/wgsl/rope.wgsl`.
pub fn rope_neox(buf: &mut [f32], rows: usize, heads: usize, hd: usize, pos0: usize, theta: f32) {
    assert!(hd.is_multiple_of(2), "rope: head_dim {hd} must be even");
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
    // Native: the AVX2+FMA `matmul_abt` kernel — `y[o] = W[o]·x` mapped as
    // C[out,1] = A[out,inn]·B[1,inn]ᵀ, so it parallelises over `out` rows (rayon)
    // AND vectorises each dot (8-wide FMA). Strictly beats both the scalar loop
    // and the rayon-scalar `matvec_par`. fp reassociation vs the scalar order is
    // within the models' cosine gates.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut y = vec![0f32; out];
        backend_cpu::fast_ops::matmul_abt(w, &x[..inn], &mut y, out, inn, 1);
        y
    }
    #[cfg(target_arch = "wasm32")]
    {
        (0..out).map(|o| w[o * inn..o * inn + inn].iter().zip(x).map(|(a, b)| a * b).sum()).collect()
    }
}

/// [`matvec`] — since [`matvec`] is now the AVX2+FMA `matmul_abt` (rayon over
/// output rows + vectorised dots), this is an alias kept for call-site clarity at
/// the big LM-head shape (one hidden row against a `[vocab, d]` table), which a
/// scalar single core turned into hundreds of ms per token (measured: 277
/// ms/token in the caption decode).
#[cfg(not(target_arch = "wasm32"))]
pub fn matvec_par(w: &[f32], x: &[f32], out: usize, inn: usize) -> Vec<f32> {
    matvec(w, x, out, inn)
}

/// L2-normalise a vector: `v / ‖v‖`.
///
/// The operation every embedding model's consumer performs before a cosine
/// comparison (ArcFace and ECAPA both emit un-normalised vectors). Host math by
/// the `AGENTS.md` rule — one vector of a few hundred values, once per item —
/// and therefore here rather than in a model crate.
///
/// A zero vector comes back unchanged rather than as NaN; the floor is applied
/// to the norm, not to the values.
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if n <= 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| (*x as f64 / n) as f32).collect()
}

/// Cosine similarity of two equal-length vectors, accumulated in f64.
///
/// f64 accumulation is not decoration: a 512-d ArcFace embedding is gated at
/// seven digits of cosine against a numpy reference, and an f32 dot product
/// loses the last two of them to summation order alone.
///
/// Panics on a length mismatch — a silent 0.0 there would read as "different
/// identity" rather than "you passed the wrong tensor".
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine: length mismatch ({} vs {})", a.len(), b.len());
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    let den = na.sqrt() * nb.sqrt();
    if den <= 0.0 {
        return 0.0;
    }
    (d / den) as f32
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
mod embed_tests {
    use super::*;

    #[test]
    fn l2_normalize_gives_a_unit_vector_and_leaves_zero_alone() {
        let v = l2_normalize(&[3.0, 4.0]);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6, "{v:?}");
        assert_eq!(l2_normalize(&[0.0, 0.0]), vec![0.0, 0.0]);
    }

    #[test]
    fn cosine_is_direction_only_and_zero_safe() {
        let a = [1.0f32, 2.0, 3.0];
        let b: Vec<f32> = a.iter().map(|x| x * 7.5).collect();
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-6);
        let neg: Vec<f32> = a.iter().map(|x| -x).collect();
        assert!((cosine(&a, &neg) + 1.0).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn cosine_refuses_mismatched_lengths() {
        cosine(&[1.0, 2.0], &[1.0]);
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


/// Least-squares 2-D **similarity** transform (Umeyama) mapping `src` onto `dst`.
///
/// `src`/`dst` are `n` point pairs, flat row-major `[n, 2]` as `(x, y)`. Returns
/// the affine matrix `M` as `[a, b, tx, c, d, ty]` (row-major `[2, 3]`), i.e.
///
/// ```text
/// dst ≈ M · [x, y, 1]ᵀ,      M = [ s·cosθ  -s·sinθ  tx ]
///                                [ s·sinθ   s·cosθ  ty ]
/// ```
///
/// This is `skimage.transform.SimilarityTransform().estimate(src, dst)` — the
/// solver insightface's `face_align.estimate_norm` uses to fit 5 facial landmarks
/// to the 112×112 ArcFace template. Four degrees of freedom (uniform scale,
/// rotation, translation) over `n ≥ 2` points, so with 5 points there is an
/// irreducible residual; that is expected, not an error.
///
/// # Why it lives here and not in the face model
///
/// It is host math (a 4-parameter closed-form least squares over ≤ 8 numbers),
/// and AGENTS.md gives host math exactly one home. It is also not face-specific:
/// any point-set registration — ROI alignment, template matching, a fiducial fit
/// — wants the same solve.
///
/// # Reflections are rejected, not silently dropped
///
/// Umeyama's general form allows a reflection when the cross-covariance has a
/// negative determinant, and returns `R = U·diag(1, −1)·Vᵀ` there. A *proper*
/// similarity (`M[0][0] == M[1][1]`, `M[0][1] == −M[1][0]`) cannot express that,
/// so this returns `Err` rather than quietly fitting the nearest non-reflected
/// transform — which would be a plausible-looking, wrong warp. A landmark set
/// that needs a reflection is a mirrored/degenerate input, not a valid fit.
pub fn similarity_transform_2d(src: &[f32], dst: &[f32], n: usize) -> Result<[f32; 6], String> {
    if n < 2 {
        return Err(format!("similarity_transform_2d: need >= 2 points, got {n}"));
    }
    if src.len() != 2 * n || dst.len() != 2 * n {
        return Err(format!(
            "similarity_transform_2d: expected {} values per point set, got src {} dst {}",
            2 * n,
            src.len(),
            dst.len()
        ));
    }
    // f64 throughout: the solve is 8 numbers wide and the reference (numpy/
    // skimage) is f64, so matching it costs nothing and avoids a needless
    // divergence in the last digits of a matrix every later pixel depends on.
    let nf = n as f64;
    let (mut msx, mut msy, mut mdx, mut mdy) = (0.0f64, 0.0, 0.0, 0.0);
    for i in 0..n {
        msx += src[2 * i] as f64;
        msy += src[2 * i + 1] as f64;
        mdx += dst[2 * i] as f64;
        mdy += dst[2 * i + 1] as f64;
    }
    msx /= nf;
    msy /= nf;
    mdx /= nf;
    mdy /= nf;

    // var_src = mean ||s - mu_s||², and the 2x2 cross-covariance
    // cov = mean (d - mu_d)(s - mu_s)ᵀ.
    let (mut var_src, mut c00, mut c01, mut c10, mut c11) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        let sx = src[2 * i] as f64 - msx;
        let sy = src[2 * i + 1] as f64 - msy;
        let dx = dst[2 * i] as f64 - mdx;
        let dy = dst[2 * i + 1] as f64 - mdy;
        var_src += sx * sx + sy * sy;
        c00 += dx * sx;
        c01 += dx * sy;
        c10 += dy * sx;
        c11 += dy * sy;
    }
    var_src /= nf;
    c00 /= nf;
    c01 /= nf;
    c10 /= nf;
    c11 /= nf;
    if var_src <= 0.0 {
        return Err("similarity_transform_2d: degenerate source (all points coincide)".into());
    }
    if c00 * c11 - c01 * c10 < 0.0 {
        return Err("similarity_transform_2d: the best fit is a REFLECTION, which is not a proper similarity".into());
    }

    // For d = 2 with no reflection, Umeyama's `s·R` has the closed form of the
    // least-squares fit over the two free parameters of a scaled rotation
    // (`M = [[a, -b], [b, a]]`): differentiating `Σ‖M·s + t − d‖²` gives
    //   a = Σ(s·d)/Σ‖s‖²,  b = Σ(s_x·d_y − s_y·d_x)/Σ‖s‖².
    // In cross-covariance terms that is `a = (c00 + c11)/var_src` and
    // `b = (c10 − c01)/var_src`, and `√(a² + b²) = trace(D)/var_src` is exactly
    // Umeyama's scale — so this is the same transform, not an approximation.
    let a = (c00 + c11) / var_src;
    let b = (c10 - c01) / var_src;
    let tx = mdx - (a * msx - b * msy);
    let ty = mdy - (b * msx + a * msy);
    Ok([a as f32, -b as f32, tx as f32, b as f32, a as f32, ty as f32])
}

/// Invert an affine `[2, 3]` matrix `[a, b, tx, c, d, ty]`.
///
/// A warp *samples* the source at `M⁻¹ · [x_dst, y_dst, 1]`, so the inverse — not
/// the forward matrix — is what builds a resampling grid.
pub fn invert_affine_2x3(m: &[f32; 6]) -> Result<[f32; 6], String> {
    let (a, b, tx, c, d, ty) = (m[0] as f64, m[1] as f64, m[2] as f64, m[3] as f64, m[4] as f64, m[5] as f64);
    let det = a * d - b * c;
    if det == 0.0 {
        return Err("invert_affine_2x3: singular matrix".into());
    }
    let (ia, ib, ic, id) = (d / det, -b / det, -c / det, a / det);
    Ok([ia as f32, ib as f32, (-(ia * tx + ib * ty)) as f32, ic as f32, id as f32, (-(ic * tx + id * ty)) as f32])
}

#[cfg(test)]
mod geom_tests {
    use super::*;

    /// An exact scaled rotation + translation must be recovered exactly.
    #[test]
    fn recovers_an_exact_similarity() {
        let (s, th, tx, ty) = (1.7f64, 0.4f64, 3.0f64, -2.0f64);
        let src = [0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0, -1.0, 4.0];
        let mut dst = [0.0f32; 10];
        for i in 0..5 {
            let (x, y) = (src[2 * i] as f64, src[2 * i + 1] as f64);
            dst[2 * i] = (s * (th.cos() * x - th.sin() * y) + tx) as f32;
            dst[2 * i + 1] = (s * (th.sin() * x + th.cos() * y) + ty) as f32;
        }
        let m = similarity_transform_2d(&src, &dst, 5).unwrap();
        assert!((m[0] - (s * th.cos()) as f32).abs() < 1e-5, "{m:?}");
        assert!((m[1] + (s * th.sin()) as f32).abs() < 1e-5, "{m:?}");
        assert!((m[2] - tx as f32).abs() < 1e-4, "{m:?}");
        assert!((m[4] - m[0]).abs() < 1e-6, "must be a proper similarity");
        assert!((m[3] + m[1]).abs() < 1e-6, "must be a proper similarity");
    }

    #[test]
    fn inverse_composes_to_the_identity() {
        let m = [1.3f32, -0.7, 5.0, 0.7, 1.3, -4.0];
        let inv = invert_affine_2x3(&m).unwrap();
        // M ∘ M⁻¹ applied to a point is the point.
        for &(x, y) in &[(0.0f32, 0.0f32), (3.0, -2.0), (17.5, 9.25)] {
            let ux = inv[0] * x + inv[1] * y + inv[2];
            let uy = inv[3] * x + inv[4] * y + inv[5];
            let rx = m[0] * ux + m[1] * uy + m[2];
            let ry = m[3] * ux + m[4] * uy + m[5];
            assert!((rx - x).abs() < 1e-3 && (ry - y).abs() < 1e-3, "{rx} {ry} vs {x} {y}");
        }
    }

    #[test]
    fn a_reflected_fit_is_an_error_not_a_silent_wrong_warp() {
        let src = [0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        let dst = [0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0]; // x/y swapped = a reflection
        assert!(similarity_transform_2d(&src, &dst, 3).unwrap_err().contains("REFLECTION"));
    }
}
