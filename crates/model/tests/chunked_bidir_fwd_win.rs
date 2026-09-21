// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `block::chunked_bidir_fwd_win` - the windowed twin of
//! `chunked_bidir_fwd`, driving the two kernels checked at the kernel level
//! in `attn_scores_cross_win.rs` through the SAME span/chunk dispatch
//! sequence `crates/decide` and (soon) `crates/modernbert` actually submit.
//!
//! Kernel-level agreement is not the same as dispatch-sequence agreement:
//! `chunked_bidir_fwd` binds `qkv` sliced at each span's row offset, folds
//! region offsets into `q_off`/`k_off`/`v_off`, and (with a `KeyMinor`) hoists
//! one transpose out of the query-chunk loop - every one of those is a place
//! a window could be applied in the wrong coordinate frame while the bare
//! kernel is still individually correct, mirroring
//! `cross_scores_kt.rs::chunked_attn_matches_with_and_without_key_minor`'s
//! own reasoning for the unwindowed path.

use data::rng::Lcg;
use gpu_core::Gpu;
use model::block::{self, CrossIds, CrossWinIds, KeyMinor, KeyMinorWin};

const PIPES: &[(&str, &str)] = &[
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_scores_cross_win", kernels::ATTN_SCORES_CROSS_WIN),
    ("kv_k_headt", kernels::KV_K_HEADT),
    ("attn_scores_cross_kt", kernels::ATTN_SCORES_CROSS_KT),
    ("attn_scores_cross_kt_win", kernels::ATTN_SCORES_CROSS_KT_WIN),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
];

fn dev() -> Gpu {
    gpu_core::testgpu::dev(PIPES)
}

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

/// Fused `[rows, 3*d]` qkv buffer from independent q/k/v host arrays, the
/// standard convention every caller of `chunked_bidir_fwd` uses
/// (`q_off=0, k_off=d, v_off=2d, stride=3d`).
fn fuse_qkv(q: &[f32], k: &[f32], v: &[f32], rows: u32, d: u32) -> Vec<f32> {
    let mut out = vec![0f32; (rows * 3 * d) as usize];
    for r in 0..rows as usize {
        out[r * 3 * d as usize..r * 3 * d as usize + d as usize].copy_from_slice(&q[r * d as usize..(r + 1) * d as usize]);
        out[r * 3 * d as usize + d as usize..r * 3 * d as usize + 2 * d as usize]
            .copy_from_slice(&k[r * d as usize..(r + 1) * d as usize]);
        out[r * 3 * d as usize + 2 * d as usize..r * 3 * d as usize + 3 * d as usize]
            .copy_from_slice(&v[r * d as usize..(r + 1) * d as usize]);
    }
    out
}

/// Independent host oracle: bidirectional self-attention within one span,
/// masked to `|i-j| <= window` - the same formula the kernel-level test
/// already checked for scores alone, now carried through softmax and the
/// value-weighted sum, computed a completely different way (host f64,
/// nothing shared with either kernel).
fn host_windowed_self_attn(q: &[f32], k: &[f32], v: &[f32], t: u32, n_heads: u32, head_dim: u32, window: u32) -> Vec<f32> {
    let (t, nh, hd) = (t as usize, n_heads as usize, head_dim as usize);
    let mut ctx = vec![0f32; t * nh * hd];
    let scale = 1.0 / (hd as f64).sqrt();
    for h in 0..nh {
        for i in 0..t {
            let lo = i.saturating_sub(window as usize);
            let hi = (i + window as usize).min(t - 1);
            let mut scores = Vec::with_capacity(hi - lo + 1);
            for j in lo..=hi {
                let qb = i * nh * hd + h * hd;
                let kb = j * nh * hd + h * hd;
                let dot: f64 = (0..hd).map(|d| q[qb + d] as f64 * k[kb + d] as f64).sum();
                scores.push(dot * scale);
            }
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|&s| (s - m).exp()).collect();
            let sum: f64 = exps.iter().sum();
            let probs: Vec<f64> = exps.iter().map(|&e| e / sum).collect();
            for d in 0..hd {
                let mut acc = 0f64;
                for (pi, j) in (lo..=hi).enumerate() {
                    acc += probs[pi] * v[j * nh * hd + h * hd + d] as f64;
                }
                ctx[i * nh * hd + h * hd + d] = acc as f32;
            }
        }
    }
    ctx
}

/// `window >= the longest span`: `chunked_bidir_fwd_win` must match
/// `chunked_bidir_fwd`'s output exactly, for both the fused-KV and the
/// key-minor path, over ragged multi-span + chunked layouts - the same case
/// list `cross_scores_kt.rs` uses for the unwindowed dispatch, so this is
/// checking the SAME property the windowed path must also honour.
#[test]
fn window_covering_every_span_matches_the_unwindowed_dispatch_exactly() {
    let g = dev();
    let cross = CrossIds { scores: idx(&g, "attn_scores_cross"), softmax: idx(&g, "attn_softmax_cross"), apply: idx(&g, "attn_apply_cross") };
    let cross_win = CrossWinIds { scores: idx(&g, "attn_scores_cross_win") };
    let (transpose, kt_scores, kt_scores_win) = (idx(&g, "kv_k_headt"), idx(&g, "attn_scores_cross_kt"), idx(&g, "attn_scores_cross_kt_win"));

    struct Case {
        label: &'static str,
        d: u32,
        heads: u32,
        spans: &'static [(u32, u32)],
        chunk: u32,
    }
    // `d` must be a multiple of 64: the attention bind group slices `qkv` at
    // each chunk's row offset (`(row0+q0)*3d*4` bytes), and the device
    // requires that offset 256-byte aligned - see
    // `crates/decide/src/model.rs::set_batch`'s own note on the same
    // constraint for span starts. `d` a multiple of 64 makes `3d` a multiple
    // of 192, itself a multiple of 64, so EVERY row lands on a legal offset
    // regardless of chunking - not just the span starts.
    let cases = [
        Case { label: "one span, unchunked", d: 64, heads: 2, spans: &[(0, 9)], chunk: 9 },
        Case { label: "one span, chunked 3 ways", d: 64, heads: 2, spans: &[(0, 9)], chunk: 3 },
        Case { label: "ragged spans + chunking", d: 128, heads: 4, spans: &[(0, 5), (5, 3)], chunk: 2 },
    ];
    let mut bad = 0;
    for Case { label, d, heads, spans, chunk } in cases {
        let rows: u32 = spans.iter().map(|s| s.1).sum();
        let head_dim = d / heads;
        let mut r = Lcg::new(0x1A7A0002);
        let (qh, kh, vh) = (r.vec_scaled((rows * d) as usize, 0.5), r.vec_scaled((rows * d) as usize, 0.5), r.vec_scaled((rows * d) as usize, 0.5));
        let qkv_h = fuse_qkv(&qh, &kh, &vh, rows, d);
        let max_span = spans.iter().map(|s| s.1).max().unwrap();
        let chunk_cap = chunk.min(max_span).max(1);
        let slab = (heads * chunk_cap * max_span) as u64;

        let run = |windowed: bool, use_km: bool| -> Vec<f32> {
            let qkv = g.storage_init("qkv", &qkv_h);
            let ctx = g.storage(rows as u64 * d as u64);
            let scores = g.storage(slab);
            let probs = g.storage(slab);
            let kt = g.storage(d as u64 * max_span as u64);
            let mut steps = Vec::new();
            if windowed {
                let km = use_km.then_some(KeyMinorWin { transpose, scores: kt_scores_win, kt: &kt });
                block::chunked_bidir_fwd_win(
                    &g, &cross_win, cross.softmax, cross.apply, km.as_ref(), max_span, heads, head_dim, d, &qkv, 3 * d, 0, d, 2 * d, &ctx, &scores, &probs, spans, chunk_cap, &mut steps,
                );
            } else {
                let km = use_km.then_some(KeyMinor { transpose, scores: kt_scores, kt: &kt });
                block::chunked_bidir_fwd(
                    &g, &cross, km.as_ref(), heads, head_dim, d, &qkv, 3 * d, 0, d, 2 * d, &ctx, &scores, &probs, spans, chunk_cap, None, &mut steps,
                );
            }
            g.submit(&[], &steps);
            g.read(&ctx, (rows * d) as usize)
        };
        for use_km in [false, true] {
            let (a, b) = (run(false, use_km), run(true, use_km));
            let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            println!("  {label:<28} km={use_km} max_abs={worst:e}");
            if worst > 1e-5 {
                bad += 1;
            }
        }
    }
    assert_eq!(bad, 0, "{bad} case(s) disagreed between windowed (window=max_span) and unwindowed dispatch");
}

/// `window < span`: the full windowed dispatch (scores, softmax, apply) must
/// match the independent host oracle, and must diverge from the unwindowed
/// dispatch at a row whose full-span context extends past the window -
/// proving the window is load-bearing end to end, not just at the score
/// kernel checked in isolation.
#[test]
fn window_narrower_than_the_span_matches_host_oracle_and_diverges_from_unwindowed() {
    let g = dev();
    let cross = CrossIds { scores: idx(&g, "attn_scores_cross"), softmax: idx(&g, "attn_softmax_cross"), apply: idx(&g, "attn_apply_cross") };
    let cross_win = CrossWinIds { scores: idx(&g, "attn_scores_cross_win") };

    let (heads, head_dim, window) = (2u32, 4u32, 2u32);
    let d = heads * head_dim;
    let t = 10u32;
    let mut r = Lcg::new(0x1A7A0003);
    let (qh, kh, vh) = (r.vec_scaled((t * d) as usize, 0.5), r.vec_scaled((t * d) as usize, 0.5), r.vec_scaled((t * d) as usize, 0.5));
    let qkv_h = fuse_qkv(&qh, &kh, &vh, t, d);
    let spans = [(0u32, t)];
    let slab = (heads * t * t) as u64;

    let qkv = g.storage_init("qkv", &qkv_h);
    let ctx = g.storage(t as u64 * d as u64);
    let scores = g.storage(slab);
    let probs = g.storage(slab);
    let mut steps = Vec::new();
    block::chunked_bidir_fwd_win(&g, &cross_win, cross.softmax, cross.apply, None, window, heads, head_dim, d, &qkv, 3 * d, 0, d, 2 * d, &ctx, &scores, &probs, &spans, t, &mut steps);
    g.submit(&[], &steps);
    let got = g.read(&ctx, (t * d) as usize);

    let want = host_windowed_self_attn(&qh, &kh, &vh, t, heads, head_dim, window);
    let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "windowed dispatch != host oracle, worst abs diff {worst}");

    let qkv2 = g.storage_init("qkv2", &qkv_h);
    let ctx2 = g.storage(t as u64 * d as u64);
    let scores2 = g.storage(slab);
    let probs2 = g.storage(slab);
    let mut steps2 = Vec::new();
    block::chunked_bidir_fwd(&g, &cross, None, heads, head_dim, d, &qkv2, 3 * d, 0, d, 2 * d, &ctx2, &scores2, &probs2, &spans, t, None, &mut steps2);
    g.submit(&[], &steps2);
    let unwindowed = g.read(&ctx2, (t * d) as usize);

    // Row 0 sees only {0,1,2} windowed vs the whole span unwindowed - a real
    // divergence, not a rounding difference.
    let row0 = 0usize..d as usize;
    let diverges = got[row0.clone()].iter().zip(&unwindowed[row0]).any(|(a, b)| (a - b).abs() > 1e-3);
    assert!(diverges, "windowed and unwindowed dispatches must diverge at a row whose context the window actually cuts");
}
