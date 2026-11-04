// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The two cross-attention score paths: same numbers, and what the coalesced
//! one is worth at each caller's shape.
//!
//! `attn_scores_cross` parallelises over the KEY index and reduces over
//! `head_dim`, so with K in the caller's key-major fused slab every lane of a
//! warp lands on its own cache line. `kv_k_headt` + `attn_scores_cross_kt`
//! compute the identical scores from a key-minor K, which is the same traffic
//! with the loads coalesced.
//!
//! Two tests, and the first is the one that matters: the swap is only worth
//! anything if it is EXACT, so [`kt_matches_cross`] asserts `max_abs == 0` and
//! not a tolerance. [`ktbench`] then prints what each caller's shape gains, so
//! the claim in the migrated models' comments can be re-checked on any device
//! rather than taken on faith.

use gpu_core::{DeviceBuffer, Gpu, Step};

const PIPES: &[(&str, &str)] = &[
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("kv_k_headt", kernels::KV_K_HEADT),
    ("attn_scores_cross_kt", kernels::ATTN_SCORES_CROSS_KT),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
];

fn time_steps(g: &Gpu, probe: &DeviceBuffer, steps: &[Step], iters: u32) -> f64 {
    g.submit(&[], steps);
    let _ = g.read(probe, 1);
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        g.submit(&[], steps);
    }
    let _ = g.read(probe, 1);
    t0.elapsed().as_secs_f64() * 1e3 / iters as f64
}

/// `(label, heads, head_dim, t_dec, t_enc, kv_stride_mult, chunks)`.
/// `kv_stride_mult` is the fused row width in units of `heads*head_dim`
/// (3 = a fused qkv self-attention buffer, 2 = a fused kv memory, 1 = compact).
/// `chunks` is how many query chunks re-read the SAME K per transpose.
const SHAPES: &[(&str, u32, u32, u32, u32, u32, u32)] = &[
    ("sam1 vit-b global", 12, 64, 4096, 4096, 3, 16),
    ("sam1 vit-b window", 12, 64, 196, 196, 3, 1),
    ("clip vit-l/14 vision", 16, 64, 257, 257, 3, 1),
    ("clip vit-b text", 12, 64, 77, 77, 3, 1),
    ("sam2 hiera s0 window", 1, 128, 4096, 4096, 3, 16),
    ("fastvlm stage-4 attn", 24, 32, 256, 256, 3, 1),
    ("pulid idformer", 12, 64, 32, 609, 2, 1),
    ("instantid resampler", 12, 64, 16, 17, 2, 1),
    ("wan cross (text ctx)", 12, 128, 4096, 512, 2, 1),
];

#[test]
fn ktbench() {
    let g = gpu_core::testgpu::dev(PIPES);
    println!("\nbackend: {}", g.kind());
    println!("{:<24} {:>10} {:>10} {:>8}", "shape", "cross ms", "kt ms", "speedup");
    for &(label, heads, hd, tq, tk, mult, chunks) in SHAPES {
        let dm = heads * hd;
        let stride = mult * dm;
        let q = g.storage(tq as u64 * stride as u64);
        let kv = g.storage(tk as u64 * stride as u64);
        let scores = g.storage(heads as u64 * tq as u64 * tk as u64);
        let kt = g.storage(dm as u64 * tk as u64);
        let qn = tq / chunks;

        let mut a = Vec::new();
        for c in 0..chunks {
            let qo = c * qn * stride;
            a.push(g.step(0, &[&q, &kv, &scores], &[1, heads, qn, tk, hd, stride, stride, qo, 0], heads * qn * tk));
        }
        let mut b = vec![g.step(1, &[&kv, &kt], &[tk, dm, stride, 0], dm * tk)];
        for c in 0..chunks {
            let qo = c * qn * stride;
            b.push(g.step(2, &[&q, &kt, &scores], &[1, heads, qn, tk, hd, stride, qo], heads * qn * tk));
        }
        let it = if (tq as u64 * tk as u64) > 1_000_000 { 3 } else { 30 };
        let ta = time_steps(&g, &scores, &a, it);
        let tb = time_steps(&g, &scores, &b, it);
        println!("{label:<24} {ta:>10.3} {tb:>10.3} {:>7.2}x", ta / tb);
    }
}

/// The two paths must produce the same numbers, not merely similar ones. A
/// score slab feeds a softmax; a tolerance here would let a layout bug hide as
/// rounding.
///
/// The sweep covers what the callers actually pass: a fused `[q|k|v]` qkv slab
/// (`kv_stride = 3*dim`, `k_off = dim`) as well as the compact `k_off = 0`
/// case, and head widths on both sides of the CPU transpose's tile.
#[test]
fn kt_matches_cross() {
    let g = gpu_core::testgpu::dev(PIPES);
    println!("backend: {}", g.kind());
    // (heads, head_dim, t_dec, t_enc, kv_stride_mult, k_off_mult)
    let cases: &[(u32, u32, u32, u32, u32, u32)] = &[
        (3, 8, 7, 11, 2, 0),
        (3, 8, 7, 11, 3, 1),
        (16, 128, 5, 37, 3, 1),
        (12, 64, 33, 197, 3, 1),
        (1, 96, 4, 4, 3, 1),
        (8, 40, 1, 65, 3, 1),
        // WorldMirror-2 DINOv2 tiny: 2 heads x 32, one 21-token frame span.
        (2, 32, 21, 21, 3, 1),
    ];
    let mut worst_all = 0.0f32;
    for &(heads, hd, tq, tk, mult, koff_mult) in cases {
        let dm = heads * hd;
        let stride = mult * dm;
        let k_off = koff_mult * dm;
        let qh: Vec<f32> = (0..tq * stride).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
        let kvh: Vec<f32> = (0..tk * stride).map(|i| ((i * 53 % 97) as f32 - 48.0) / 23.0).collect();
        let q = g.storage_init("q", &qh);
        let kv = g.storage_init("kv", &kvh);
        let n = (heads * tq * tk) as usize;
        let s1 = g.storage(n as u64);
        let s2 = g.storage(n as u64);
        let kt = g.storage(dm as u64 * tk as u64);
        g.submit(
            &[&s1],
            &[g.step(0, &[&q, &kv, &s1], &[1, heads, tq, tk, hd, stride, stride, 0, k_off], heads * tq * tk)],
        );
        g.submit(
            &[&s2],
            &[
                g.step(1, &[&kv, &kt], &[tk, dm, stride, k_off], dm * tk),
                g.step(2, &[&q, &kt, &s2], &[1, heads, tq, tk, hd, stride, 0], heads * tq * tk),
            ],
        );
        let (a, b) = (g.read(&s1, n), g.read(&s2, n));
        let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        println!("  heads={heads} hd={hd} tq={tq} tk={tk} stride={stride} k_off={k_off}: max_abs={worst:e}");
        worst_all = worst_all.max(worst);
    }
    assert!(worst_all == 0.0, "the two score paths disagree by {worst_all:e}");
}

/// The kernels agreeing in isolation is not the same as the two dispatch
/// sequences agreeing: `chunked_bidir_fwd` binds `qkv` SLICED at each span's
/// row offset, folds the region offsets into `k_off`/`q_off`, reuses one `kt`
/// across spans, and hoists the transpose out of the query-chunk loop. Every
/// one of those is a place the two paths can diverge while both kernels are
/// individually correct - so the contract is checked on the whole `ctx`, over a
/// RAGGED multi-span layout with query chunking, which is what the ViT callers
/// actually pass.
#[test]
fn chunked_attn_matches_with_and_without_key_minor() {
    use model::vit::{chunked_attn_fwd, VitKernelIds, VitShape, UNREGISTERED};
    let g = gpu_core::testgpu::dev(PIPES);
    println!("backend: {}", g.kind());
    let base = VitKernelIds {
        layernorm: UNREGISTERED,
        matmul: UNREGISTERED,
        matmul_rows: UNREGISTERED,
        bias_add: UNREGISTERED,
        mlp_act: UNREGISTERED,
        scale_chan: UNREGISTERED,
        add2: UNREGISTERED,
        attn_scores_cross: 0,
        attn_softmax_cross: 3,
        attn_apply_cross: 4,
        kv_k_headt: UNREGISTERED,
        attn_scores_cross_kt: UNREGISTERED,
        ln_head: UNREGISTERED,
        rope2d: UNREGISTERED,
    };
    let kt_ids = VitKernelIds { kv_k_headt: 1, attn_scores_cross_kt: 2, ..base };

    /// One span layout to check both dispatch sequences against.
    struct Case {
        label: &'static str,
        dim: u32,
        heads: u32,
        spans: &'static [(u32, u32)],
        chunk: u32,
    }
    let case = |label, dim, heads, spans, chunk| Case { label, dim, heads, spans, chunk };
    let cases = [
        case("wm2 dino: 3 frame spans", 64, 2, &[(0, 21), (21, 21), (42, 21)], 21),
        case("wm2 trunk: 3 frame spans", 64, 2, &[(0, 23), (23, 23), (46, 23)], 23),
        case("wm2 trunk: one global span", 64, 2, &[(0, 69)], 69),
        case("one span, chunked 3 ways", 64, 2, &[(0, 69)], 23),
        case("ragged spans + chunking", 128, 4, &[(0, 40), (40, 24)], 16),
        case("wm2 cam head: 16x8 over 3", 128, 16, &[(0, 3)], 3),
        case("wm2 cam head: 16x8 over 4", 128, 16, &[(0, 4)], 4),
    ];
    let mut bad = 0;
    for Case { label, dim: c, heads, spans, chunk } in cases {
        let sh = VitShape { dim: c, heads, mlp: 0, eps: 1e-5 };
        let rows: u32 = spans.iter().map(|s| s.1).sum();
        let qkv_h: Vec<f32> = (0..rows * 3 * c).map(|i| ((i * 31 % 89) as f32 - 44.0) / 21.0).collect();
        let qkv = g.storage_init("qkv", &qkv_h);
        let max_span = spans.iter().map(|s| s.1).max().unwrap();
        let slab = (heads * chunk * max_span) as u64;
        let out = |ids: &VitKernelIds| -> Vec<f32> {
            let ctx = g.storage(rows as u64 * c as u64);
            let scores = g.storage(slab);
            let probs = g.storage(slab);
            let kt = g.storage(c as u64 * max_span as u64);
            let mut steps = Vec::new();
            chunked_attn_fwd(&g, ids, &sh, &qkv, &ctx, &scores, &probs, &kt, spans, chunk, &mut steps);
            g.submit(&[], &steps);
            g.read(&ctx, (rows * c) as usize)
        };
        let (a, b) = (out(&base), out(&kt_ids));
        let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
        println!("  {label:<28} max_abs={worst:e}  rel={:e}", worst / scale);
        // NaN-safe: a non-finite ratio is a failure, not an "unordered" pass.
        let rel = worst / scale;
        if rel.is_nan() || rel >= 1e-5 {
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "the two dispatch sequences disagree on {bad} of the layouts above");
}
