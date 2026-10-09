// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native causal GQA flash attention with an additive
//! per-key mask (`gpu_core::Fused::FlashGqaKmask`, `cu/flash_attn_f32.cu`):
//! one launch in place of the `gqa_scores_kmask` -> `softmax_rows` ->
//! `gqa_apply` chain a right-padded encoder (Qwen3 as FLUX.2's text encoder)
//! runs per layer.
//!
//! Swedish Embedded AB implements fused attention kernels for GPUs without
//! tensor cores. If your team needs expertise in text-encoder latency on the
//! hardware you already have, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! ```text
//! cargo test -p brain-gpu-core --test flash_gqa_kmask_native -- --device gpu1 --backend cuda
//! ```
//!
//! Not bit-identical to the chain (a reordered sum, as for the bidirectional
//! kernel). Both are held to the same stated tolerance against an f64 oracle,
//! `|out - oracle| <= 2^-17 * max|v|`, and the native kernel to no more than
//! twice the chain's own error plus `2^-22 * max|v|`. Masks cover no pads,
//! a short caption padded far (the encoder's case, where the native kernel
//! skips the all-masked key tiles), pads straddling a key-tile edge, and a
//! masked key in the middle of the content.

use gpu_core::{Dispatch, Fused, Gpu};

const KERNELS: &[(&str, &str)] = &[
    ("gqa_scores_kmask", kernels::GQA_SCORES_KMASK),
    ("softmax_rows", kernels::SOFTMAX_ROWS),
    ("gqa_apply", kernels::GQA_APPLY),
];
const K_SCORES: usize = 0;
const K_SOFTMAX: usize = 1;
const K_APPLY: usize = 2;
const HD: u32 = 128;
const TOL: f64 = 1.0 / 131_072.0;
const FLOOR: f64 = 1.0 / 4_194_304.0;
/// The mask value `qwen3::Qwen::arm_pad_kmask` writes for an excluded key.
const MASKED: f32 = -3.4e38;
const SENTINEL: u32 = 0x7fc0_dead;

fn cuda() -> Option<Gpu> {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if gpu.kind() != "cuda" || !gpu.has_fused(Fused::FlashGqaKmask) {
        brain_testutil::skip_unavailable("the native masked GQA attention needs a CUDA device");
        return None;
    }
    Some(gpu)
}

struct Case {
    bsz: u32,
    nh: u32,
    nkv: u32,
    t: u32,
    kmask: Vec<f32>,
}

impl Case {
    fn params(&self) -> [u32; 6] {
        [self.bsz, self.nh, self.nkv, self.t, HD, self.nh / self.nkv]
    }
}

fn gauss(r: &mut data::rng::Lcg, sigma: f32) -> f32 {
    let u1 = ((r.next_u32() >> 8) as f32 / (1u32 << 24) as f32).max(1e-7);
    let u2 = (r.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
    sigma * (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// f64 masked causal attention of query row `i` of sample `b`, head `h`.
fn oracle_row(c: &Case, q: &[f32], k: &[f32], v: &[f32], b: u32, h: u32, i: u32) -> Vec<f64> {
    let (t, hd, nh, nkv) = (c.t as usize, HD as usize, c.nh as usize, c.nkv as usize);
    let hk = h as usize / (nh / nkv);
    let (b, h, i) = (b as usize, h as usize, i as usize);
    let qv = |c_: usize| f64::from(q[(b * t + i) * nh * hd + h * hd + c_]);
    let kv = |j: usize, c_: usize| f64::from(k[(b * t + j) * nkv * hd + hk * hd + c_]);
    let vv = |j: usize, c_: usize| f64::from(v[(b * t + j) * nkv * hd + hk * hd + c_]);
    let scale = 1.0 / (hd as f64).sqrt();
    let s: Vec<f64> = (0..=i).map(|j| (0..hd).map(|x| qv(x) * kv(j, x)).sum::<f64>() * scale + f64::from(c.kmask[j])).collect();
    let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
    let l: f64 = e.iter().sum();
    (0..hd).map(|x| (0..=i).map(|j| e[j] * vv(j, x)).sum::<f64>() / l).collect()
}

fn check(gpu: &Gpu, c: &Case, seed: u64) {
    let p = c.params();
    let (rows, qw, kw) = ((c.bsz * c.t) as usize, (c.nh * HD) as usize, (c.nkv * HD) as usize);
    let mut r = data::rng::Lcg::new(seed);
    let q: Vec<f32> = (0..rows * qw).map(|_| gauss(&mut r, 2.0)).collect();
    let k: Vec<f32> = (0..rows * kw).map(|_| gauss(&mut r, 2.0)).collect();
    let v: Vec<f32> = (0..rows * kw).map(|_| ((r.next_u32() >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0).collect();
    let up = |x: &[f32]| {
        let b = gpu.storage(x.len() as u64);
        gpu.write_f32(&b, x);
        b
    };
    let (qb, kb, vb, mb) = (up(&q), up(&k), up(&v), up(&c.kmask));
    let n = rows * qw;
    let out = |words: usize| {
        let b = gpu.storage(words as u64 + 8);
        gpu.write(&b, &vec![SENTINEL; words + 8]);
        b
    };
    let (native, chain) = (out(n), out(n));
    let srows = c.bsz * c.nh * c.t;
    let scores = gpu.storage(u64::from(srows) * u64::from(c.t));
    let probs = gpu.storage(u64::from(srows) * u64::from(c.t));
    let fused = gpu.fused_step(Fused::FlashGqaKmask, &[&qb, &kb, &mb, &vb, &native], &p).expect("the kernel serves the shape");
    let steps = [
        fused,
        gpu.step(K_SCORES, &[&qb, &kb, &mb, &scores], &p, srows * c.t),
        gpu.dispatch(K_SOFTMAX, &[&scores, &probs], &[srows, c.t], Dispatch::Workgroups(srows)),
        gpu.step(K_APPLY, &[&probs, &vb, &chain], &p, srows * HD),
    ];
    gpu.submit(&[], &steps);
    gpu.poll_wait();
    let read = |b| {
        let x = gpu.read(b, n + 8);
        assert!(x[n..].iter().all(|y| y.to_bits() == SENTINEL), "a write past the output at {p:?}");
        x[..n].to_vec()
    };
    let (got, want_chain) = (read(&native), read(&chain));
    assert!(got.iter().all(|x| x.is_finite()), "non-finite native output at {p:?}");
    let vmax = v.iter().fold(0.0f64, |m, x| m.max(f64::from(*x).abs()));
    let bound = TOL * vmax;
    let (mut worst_n, mut worst_c) = (0.0f64, 0.0f64);
    let mut pick = data::rng::Lcg::new(seed ^ 0x77);
    for b in 0..c.bsz {
        for h in 0..c.nh {
            let rows_to_check: Vec<u32> = [0, c.t - 1].into_iter().chain((0..6).map(|_| pick.next_u32() % c.t)).collect();
            for i in rows_to_check {
                let want = oracle_row(c, &q, &k, &v, b, h, i);
                let base = ((b * c.t + i) * c.nh * HD + h * HD) as usize;
                for (x, w) in want.iter().enumerate() {
                    worst_n = worst_n.max((f64::from(got[base + x]) - w).abs());
                    worst_c = worst_c.max((f64::from(want_chain[base + x]) - w).abs());
                }
            }
        }
    }
    println!("{p:?} live {}: |native-f64| {worst_n:.2e}  |chain-f64| {worst_c:.2e}  bound {bound:.2e}", c.kmask.iter().filter(|m| **m == 0.0).count());
    assert!(worst_c <= bound, "the WGSL chain misses the stated tolerance at {p:?}: {worst_c:e}");
    assert!(worst_n <= bound, "native masked attention misses the f64 oracle at {p:?}: {worst_n:e} > {bound:e}");
    let no_worse = 2.0 * worst_c + FLOOR * vmax;
    assert!(worst_n <= no_worse, "native masked attention is less accurate than the chain at {p:?}: {worst_n:e} > {no_worse:e}");
}

/// `t` keys, the first `content` live and the rest masked like padding.
fn padded(t: u32, content: u32) -> Vec<f32> {
    (0..t).map(|j| if j < content { 0.0 } else { MASKED }).collect()
}

fn the_kernel_is_offered_for_head_dim_128_only() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if gpu.kind() != "cuda" {
        assert!(!gpu.has_fused(Fused::FlashGqaKmask), "only a CUDA device takes a native fused kernel");
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    assert!(gpu.has_fused(Fused::FlashGqaKmask));
    assert!(Fused::FlashGqaKmask.serves(&[1, 32, 8, 512, 128, 4]));
    assert!(!Fused::FlashGqaKmask.serves(&[1, 32, 8, 512, 64, 4]), "another head width");
    assert!(!Fused::FlashGqaKmask.serves(&[1, 32, 8, 512, 128, 3]), "group must be n_heads / n_kv_heads");
    assert!(!Fused::FlashGqaKmask.serves(&[1, 30, 8, 512, 128, 4]), "n_heads must be a multiple of n_kv_heads");
    assert!(!Fused::FlashGqaKmask.serves(&[1, 32, 8, 0, 128, 4]), "no rows");
    assert!(!Fused::FlashGqaKmask.serves(&[1, 32, 8, 512, 128]), "short params");
}

fn matches_the_oracle_and_the_chain() {
    let Some(gpu) = cuda() else { return };
    let mut mid = padded(97, 97);
    mid[40] = MASKED;
    let cases = [
        Case { bsz: 1, nh: 4, nkv: 1, t: 1, kmask: padded(1, 1) },
        Case { bsz: 1, nh: 4, nkv: 2, t: 33, kmask: padded(33, 33) },
        Case { bsz: 2, nh: 4, nkv: 2, t: 64, kmask: padded(64, 64) },
        Case { bsz: 1, nh: 8, nkv: 2, t: 130, kmask: padded(130, 100) },
        Case { bsz: 1, nh: 4, nkv: 1, t: 200, kmask: padded(200, 31) },
        Case { bsz: 1, nh: 4, nkv: 1, t: 200, kmask: padded(200, 32) },
        Case { bsz: 1, nh: 4, nkv: 1, t: 200, kmask: padded(200, 33) },
        Case { bsz: 2, nh: 4, nkv: 4, t: 97, kmask: mid },
    ];
    for (i, c) in cases.iter().enumerate() {
        check(&gpu, c, 100 + i as u64);
    }
}

/// Qwen3-4B's attention (32 query heads over 8 key/value heads) at FLUX.2's
/// 512-token caption length: a short caption padded far, and a full one.
fn the_encoder_shapes_match() {
    let Some(gpu) = cuda() else { return };
    for content in [23u32, 512] {
        check(&gpu, &Case { bsz: 1, nh: 32, nkv: 8, t: 512, kmask: padded(512, content) }, u64::from(content));
    }
}

gpu_core::card_tests!(the_kernel_is_offered_for_head_dim_128_only, matches_the_oracle_and_the_chain, the_encoder_shapes_match);
