// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for `kernels_cuda`'s native bidirectional flash attention
//! (`cu/flash_attn_f32.cu`), which `gpu_core::native_upgrade` substitutes for
//! the WGSL `flash_attn_bidir_reg2` on a CUDA device at `head_dim == 128`.
//!
//! Swedish Embedded AB implements fused attention kernels for GPUs without
//! tensor cores. If your team needs expertise in getting attention close to
//! the fp32 roof of the hardware it actually has, you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! ```text
//! cargo test -p brain-gpu-core --test flash_bidir_native -- --device gpu1 --backend cuda
//! ```
//!
//! The native kernel is NOT bit-identical to the WGSL tier: it sums the
//! `q.k` products and the `p.v` terms in a different order (a register block
//! per thread instead of four lanes' partials, the softmax in base 2 with the
//! scale folded into one multiplier, the row sum reduced once at the end).
//! Attention is a weighted MEAN of the value rows, so an fp32 evaluation's
//! error is bounded in units of the largest value magnitude, not relative to
//! each output. The stated tolerance, held by both tiers against an f64
//! oracle on every shape here:
//!
//! ```text
//! |out - oracle| <= TOL * max|v|        TOL = 2^-17 (about 7.6e-6)
//! ```
//!
//! That is about 64 fp32 ulps of a unit value, far below the noise of the int8
//! GEMMs that feed it, and the native kernel is held to agree with the
//! generated tier within twice that. So that the redirect never LOWERS
//! accuracy, the native kernel's error must also stay within twice the
//! generated tier's own error on the same data, plus `FLOOR`: a kernel that
//! sums each score as one 128-long chain of multiply-adds passes the first
//! bound and fails this one.

use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[
    ("flash_attn_bidir_reg2", kernels::FLASH_ATTN_BIDIR_REG2),
    ("flash_attn_bidir_reg2_ref", kernels::FLASH_ATTN_BIDIR_REG2),
];
const K_REG2: usize = 0;
const K_REF: usize = 1;
/// Query rows per workgroup of the WGSL kernel, which sets its grid.
const WGSL_BR: u32 = 128;
const HD: u32 = 128;
/// The stated tolerance, in units of the largest value magnitude.
const TOL: f64 = 1.0 / 131_072.0;
/// The slack, in units of the largest value magnitude, beside twice the
/// generated tier's own error, within which the native kernel must stay on
/// the same data: a couple of fp32 ulps of a unit value.
const FLOOR: f64 = 1.0 / 4_194_304.0;
/// Written past the end of the output so an out-of-range write shows.
const SENTINEL: u32 = 0x7fc0_dead;

fn device() -> Option<Gpu> {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if gpu.kind() != "cuda" || gpu.caps().arch.compute_capability.is_none() {
        brain_testutil::skip_unavailable("native flash attention needs a CUDA device");
        return None;
    }
    Some(gpu)
}

/// `[bsz, heads, T, head_dim, qkv_stride, q_off, k_off, v_off, d_model]` for
/// the packed `[bsz*T, 3*d_model]` slab every caller of the family uses.
fn params(bsz: u32, heads: u32, t: u32, hd: u32) -> [u32; 9] {
    let d = heads * hd;
    [bsz, heads, t, hd, 3 * d, 0, d, 2 * d, d]
}

/// A packed qkv slab: q and k with entries of standard deviation `sigma` (so
/// the logits `q.k / sqrt(hd)` have standard deviation `sigma^2`), v uniform
/// in [-1, 1).
fn slab(bsz: u32, heads: u32, t: u32, hd: u32, sigma: f32, seed: u64) -> Vec<f32> {
    let d = (heads * hd) as usize;
    let mut r = data::rng::Lcg::new(seed);
    let mut uni = || (r.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
    let rows = (bsz * t) as usize;
    let mut out = vec![0.0f32; rows * 3 * d];
    for row in 0..rows {
        for c in 0..3 * d {
            out[row * 3 * d + c] = if c < 2 * d {
                // Box-Muller from two uniforms.
                let (u1, u2) = (uni().max(1e-7), uni());
                sigma * (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
            } else {
                2.0 * uni() - 1.0
            };
        }
    }
    out
}

/// The f64 attention output of query row `i` of sample `b`, head `h`.
fn oracle_row(qkv: &[f32], p: &[u32; 9], b: u32, h: u32, i: u32) -> Vec<f64> {
    let [_, _, t, hd, stride, q_off, k_off, v_off, _] = p.map(|v| v as usize);
    let (b, h, i) = (b as usize, h as usize, i as usize);
    let at = |row: usize, off: usize, c: usize| f64::from(qkv[(b * t + row) * stride + off + h * hd + c]);
    let scale = 1.0 / (hd as f64).sqrt();
    let s: Vec<f64> = (0..t).map(|j| (0..hd).map(|c| at(i, q_off, c) * at(j, k_off, c)).sum::<f64>() * scale).collect();
    let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
    let l: f64 = e.iter().sum();
    (0..hd).map(|c| (0..t).map(|j| e[j] * at(j, v_off, c)).sum::<f64>() / l).collect()
}

/// Both tiers' outputs for one slab.
fn run_both(gpu: &Gpu, p: &[u32; 9], qkv: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let [bsz, heads, t, _, _, _, _, _, d] = *p;
    let out_len = (bsz * t * d) as usize;
    let qkv_buf = gpu.storage(qkv.len() as u64);
    gpu.write_f32(&qkv_buf, qkv);
    let fresh = || {
        let b = gpu.storage(out_len as u64 + 8);
        gpu.write(&b, &vec![SENTINEL; out_len + 8]);
        b
    };
    let (a, r) = (fresh(), fresh());
    let wg = bsz * heads * t.div_ceil(WGSL_BR);
    let steps = [
        gpu.dispatch(K_REG2, &[&qkv_buf, &a], p, Dispatch::Workgroups(wg)),
        gpu.dispatch(K_REF, &[&qkv_buf, &r], p, Dispatch::Workgroups(wg)),
    ];
    gpu.submit(&[], &steps);
    gpu.poll_wait();
    let read = |b| {
        let v = gpu.read(b, out_len + 8);
        assert!(v[out_len..].iter().all(|x| x.to_bits() == SENTINEL), "a write past the output at {p:?}");
        v[..out_len].to_vec()
    };
    (read(&a), read(&r))
}

/// Check `rows` sampled query rows of every (sample, head) against the f64
/// oracle, and the two tiers against each other on the whole output.
fn check(gpu: &Gpu, bsz: u32, heads: u32, t: u32, sigma: f32, rows: u32, seed: u64) {
    let p = params(bsz, heads, t, HD);
    let qkv = slab(bsz, heads, t, HD, sigma, seed);
    let (native, wgsl) = run_both(gpu, &p, &qkv);
    let d = (heads * HD) as usize;
    let vmax = (0..(bsz * t) as usize)
        .flat_map(|row| qkv[row * 3 * d + 2 * d..row * 3 * d + 3 * d].iter())
        .fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
    let bound = TOL * vmax;
    let pair = native.iter().zip(&wgsl).map(|(a, b)| f64::from(*a - *b).abs()).fold(0.0, f64::max);
    assert!(native.iter().all(|v| v.is_finite()), "non-finite native output at {p:?}");
    assert!(pair <= 2.0 * bound, "native and generated tiers differ by {pair:e} (bound {:e}) at {p:?} sigma {sigma}", 2.0 * bound);
    let mut r = data::rng::Lcg::new(seed ^ 0x5a5a);
    let (mut worst_n, mut worst_w) = (0.0f64, 0.0f64);
    for b in 0..bsz {
        for h in 0..heads {
            // Always the first and last rows (the tile edges), then a sample.
            let picks: Vec<u32> = [0, t - 1].into_iter().chain((0..rows.saturating_sub(2)).map(|_| r.next_u32() % t)).collect();
            for i in picks {
                let want = oracle_row(&qkv, &p, b, h, i);
                let base = ((b * t + i) as usize) * d + (h * HD) as usize;
                for (c, w) in want.iter().enumerate() {
                    worst_n = worst_n.max((f64::from(native[base + c]) - w).abs());
                    worst_w = worst_w.max((f64::from(wgsl[base + c]) - w).abs());
                }
            }
        }
    }
    println!("bsz {bsz} heads {heads} T {t} sigma {sigma}: |native-f64| {worst_n:.2e}  |wgsl-f64| {worst_w:.2e}  |native-wgsl| {pair:.2e}  bound {bound:.2e}");
    assert!(worst_w <= bound, "the generated tier itself misses the stated tolerance at {p:?}: {worst_w:e} > {bound:e}");
    assert!(worst_n <= bound, "native flash attention misses the f64 oracle at {p:?}: {worst_n:e} > {bound:e}");
    let no_worse = 2.0 * worst_w + FLOOR * vmax;
    assert!(worst_n <= no_worse, "native flash attention is less accurate than the generated tier at {p:?}: {worst_n:e} > {no_worse:e}");
}

fn the_native_kernel_serves_head_dim_128() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if gpu.kind() != "cuda" || gpu.caps().arch.compute_capability.is_none() {
        assert_eq!(gpu.native_kernel_for(K_REG2, &params(1, 24, 1792, HD)), None, "only a CUDA device takes a native kernel");
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    for (bsz, heads, t) in [(1u32, 24u32, 1792u32), (1, 24, 3072), (2, 3, 1), (1, 1, 65)] {
        assert_eq!(gpu.native_kernel_for(K_REG2, &params(bsz, heads, t, HD)), Some("flash_bidir_f32"), "{bsz} {heads} {t}");
    }
    assert_eq!(gpu.native_kernel_for(K_REG2, &params(1, 4, 64, 64)), None, "another head width keeps the WGSL tier");
    assert_eq!(gpu.native_kernel_for(K_REG2, &params(1, 4, 0, HD)), None, "no rows");
    assert_eq!(gpu.native_kernel_for(K_REG2, &params(0, 4, 64, HD)), None, "no samples");
    assert_eq!(gpu.native_kernel_for(K_REG2, &params(1, 4, 64, HD)[..8]), None, "short params");
    assert_eq!(gpu.native_kernel_for(K_REF, &params(1, 24, 1792, HD)), None);
}

/// Sequence lengths on both sides of every tile edge, batches, several heads,
/// and two logit spreads (a flat softmax and a peaked one).
fn matches_the_f64_oracle_and_the_generated_tier() {
    let Some(gpu) = device() else { return };
    for (bsz, heads, t) in [(1u32, 1u32, 1u32), (1, 2, 7), (1, 3, 31), (2, 2, 32), (1, 2, 33), (1, 1, 63), (2, 2, 64), (1, 2, 65), (1, 1, 127), (1, 2, 129), (2, 3, 200)] {
        for sigma in [1.0f32, 3.0] {
            check(&gpu, bsz, heads, t, sigma, 16, u64::from(bsz * 1000 + heads * 100 + t));
        }
    }
}

/// Seeded random shapes, every length residue a tile can leave.
fn random_shapes_match() {
    let Some(gpu) = device() else { return };
    let mut r = data::rng::Lcg::new(0xf1a5_4b1d);
    for _ in 0..12 {
        let (bsz, heads, t) = (1 + r.next_u32() % 2, 1 + r.next_u32() % 4, 1 + r.next_u32() % 700);
        check(&gpu, bsz, heads, t, 2.0, 8, u64::from(r.next_u32()));
    }
}

/// FLUX.2 klein's joint sequences: text-to-image at 640x512 (512 text + 1280
/// image tokens) and an edit with one reference of the same size, 24 heads.
fn the_model_shapes_match() {
    let Some(gpu) = device() else { return };
    for t in [1792u32, 3072] {
        check(&gpu, 1, 24, t, 2.0, 4, u64::from(t));
    }
}

gpu_core::card_tests!(
    the_native_kernel_serves_head_dim_128,
    matches_the_f64_oracle_and_the_generated_tier,
    random_shapes_match,
    the_model_shapes_match,
);
