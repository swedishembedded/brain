// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `block::chunked_bidir_bwd_win` - the windowed twin of `chunked_bidir_bwd`.
//!
//! The plan this crate is built against PREDICTS that the windowed backward
//! needs no new gradient kernel: recompute scores/probs through the WINDOWED
//! forward kernel, then feed them into the same UNWINDOWED `CrossBwdIds`
//! gradient kernels `chunked_bidir_bwd` already uses, because softmax's own
//! adjoint is already exactly zero wherever the probability is zero - which
//! is every masked position under a correctly-windowed softmax. That is a
//! plausible claim, not a proven one, and this file is what actually proves
//! it: an independent finite-difference check on `chunked_bidir_bwd_win`'s
//! own `d_qkv` output, isolated from the rest of the trunk (no LayerNorm, no
//! RoPE, no GeGLU in the loop) so a failure here can only be this one
//! function - the same "isolate before composing" discipline
//! `crates/gradcheck/src/decide.rs`'s own module doc argues for.

use data::rng::Lcg;
use gpu_core::Gpu;
use model::block::{self, CrossBwdIds, CrossIds, CrossWinIds};

const PIPES: &[(&str, &str)] = &[
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_scores_cross_win", kernels::ATTN_SCORES_CROSS_WIN),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS),
    ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS),
    ("attn_bwd_dk_cross_acc", kernels::ATTN_BWD_DK_CROSS_ACC),
    ("attn_bwd_dv_cross_acc", kernels::ATTN_BWD_DV_CROSS_ACC),
];

fn dev() -> Gpu {
    gpu_core::testgpu::dev(PIPES)
}

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

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

/// One windowed-backward dispatch: forward to `ctx`, seed `d_ctx = w`, run
/// `chunked_bidir_bwd_win`, return `(ctx, d_qkv)`.
#[allow(clippy::too_many_arguments)]
fn run_win(
    g: &Gpu,
    cross_win: &CrossWinIds,
    softmax: usize,
    apply: usize,
    cross_bwd: &CrossBwdIds,
    window: u32,
    heads: u32,
    head_dim: u32,
    d: u32,
    qkv_h: &[f32],
    w: &[f32],
    spans: &[(u32, u32)],
    chunk: u32,
) -> (Vec<f32>, Vec<f32>) {
    let rows: u32 = spans.iter().map(|s| s.1).sum();
    let max_span = spans.iter().map(|s| s.1).max().unwrap();
    let chunk_cap = chunk.min(max_span).max(1);
    let slab = (heads * chunk_cap * max_span) as u64;

    let qkv = g.storage_init("qkv", qkv_h);
    let ctx = g.storage(rows as u64 * d as u64);
    let scores = g.storage(slab);
    let probs = g.storage(slab);
    let mut steps = Vec::new();
    block::chunked_bidir_fwd_win(g, cross_win, softmax, apply, None, window, heads, head_dim, d, &qkv, 3 * d, 0, d, 2 * d, &ctx, &scores, &probs, spans, chunk_cap, &mut steps);
    g.submit(&[], &steps);
    let ctx_h = g.read(&ctx, (rows * d) as usize);

    let d_ctx = g.storage_init("d_ctx", w);
    let d_qkv = g.storage(rows as u64 * 3 * d as u64);
    let mut bsteps = Vec::new();
    block::chunked_bidir_bwd_win(
        g, cross_win, softmax, None, cross_bwd, window, heads, head_dim, d, &qkv, 3 * d, 0, d, 2 * d, &d_ctx, &d_qkv, &scores, &probs, &scores, spans, chunk_cap, &mut bsteps,
    );
    g.submit(&[], &bsteps);
    let d_qkv_h = g.read(&d_qkv, (rows * 3 * d) as usize);
    (ctx_h, d_qkv_h)
}

/// `window >= the longest span`: the windowed backward must reproduce the
/// unwindowed backward's `d_qkv` exactly (same reasoning as the forward
/// file's first test), over ragged multi-span + chunked layouts.
#[test]
fn window_covering_every_span_matches_the_unwindowed_backward_exactly() {
    let g = dev();
    let cross = CrossIds { scores: idx(&g, "attn_scores_cross"), softmax: idx(&g, "attn_softmax_cross"), apply: idx(&g, "attn_apply_cross") };
    let cross_win = CrossWinIds { scores: idx(&g, "attn_scores_cross_win") };
    let cross_bwd = CrossBwdIds::resolve(&g, idx(&g, "attn_bwd_dscores_cross"), idx(&g, "attn_bwd_dq_cross"), idx(&g, "attn_bwd_dk_cross_acc"), idx(&g, "attn_bwd_dv_cross_acc"));

    struct Case {
        label: &'static str,
        d: u32,
        heads: u32,
        spans: &'static [(u32, u32)],
        chunk: u32,
    }
    let cases = [
        Case { label: "one span, unchunked", d: 64, heads: 2, spans: &[(0, 9)], chunk: 9 },
        Case { label: "one span, chunked 3 ways", d: 64, heads: 2, spans: &[(0, 9)], chunk: 3 },
        Case { label: "ragged spans + chunking", d: 128, heads: 4, spans: &[(0, 5), (5, 3)], chunk: 2 },
    ];
    let mut bad = 0;
    for Case { label, d, heads, spans, chunk } in cases {
        let rows: u32 = spans.iter().map(|s| s.1).sum();
        let head_dim = d / heads;
        let max_span = spans.iter().map(|s| s.1).max().unwrap();
        let chunk_cap = chunk.min(max_span).max(1);
        let slab = (heads * chunk_cap * max_span) as u64;
        let mut r = Lcg::new(0x1B7A0001);
        let (qh, kh, vh) = (r.vec_scaled((rows * d) as usize, 0.5), r.vec_scaled((rows * d) as usize, 0.5), r.vec_scaled((rows * d) as usize, 0.5));
        let qkv_h = fuse_qkv(&qh, &kh, &vh, rows, d);
        let w = r.vec_scaled((rows * d) as usize, 0.5);

        let (_ctx_win, dqkv_win) = run_win(&g, &cross_win, cross.softmax, cross.apply, &cross_bwd, max_span, heads, head_dim, d, &qkv_h, &w, spans, chunk_cap);

        // Unwindowed reference: forward + backward through the plain rungs.
        let qkv = g.storage_init("qkv_ref", &qkv_h);
        let ctx = g.storage(rows as u64 * d as u64);
        let scores = g.storage(slab);
        let probs = g.storage(slab);
        let mut steps = Vec::new();
        block::chunked_bidir_fwd(&g, &cross, None, heads, head_dim, d, &qkv, 3 * d, 0, d, 2 * d, &ctx, &scores, &probs, spans, chunk_cap, None, &mut steps);
        g.submit(&[], &steps);
        let d_ctx = g.storage_init("d_ctx_ref", &w);
        let d_qkv = g.storage(rows as u64 * 3 * d as u64);
        let mut bsteps = Vec::new();
        block::chunked_bidir_bwd(&g, &cross, None, &cross_bwd, heads, head_dim, d, &qkv, 3 * d, 0, d, 2 * d, &d_ctx, &d_qkv, &scores, &probs, &scores, spans, chunk_cap, None, &mut bsteps);
        g.submit(&[], &bsteps);
        let dqkv_ref = g.read(&d_qkv, (rows * 3 * d) as usize);

        let worst = dqkv_win.iter().zip(&dqkv_ref).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("  {label:<28} max_abs={worst:e}");
        if worst > 1e-4 {
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{bad} case(s) disagreed between windowed (window=max_span) and unwindowed d_qkv");
}

/// `window < span`: `chunked_bidir_bwd_win`'s `d_qkv` must agree with a
/// finite-difference of the loss `L = sum(ctx * w)` taken through the SAME
/// windowed forward (`chunked_bidir_fwd_win`) it is the adjoint of. This is
/// the real test of the plan's "no new gradient kernel" claim: if the
/// unwindowed gradient kernels were somehow picking up a masked-out
/// contribution (e.g. because `probs` at a masked position were not actually
/// zero, or the recompute used the wrong window), this is what would catch
/// it - the forward/fwd-win agreement test above cannot, since it never
/// exercises a window narrower than the span.
#[test]
fn window_narrower_than_the_span_matches_finite_differences() {
    let g = dev();
    let cross = CrossIds { scores: idx(&g, "attn_scores_cross"), softmax: idx(&g, "attn_softmax_cross"), apply: idx(&g, "attn_apply_cross") };
    let cross_win = CrossWinIds { scores: idx(&g, "attn_scores_cross_win") };
    let cross_bwd = CrossBwdIds::resolve(&g, idx(&g, "attn_bwd_dscores_cross"), idx(&g, "attn_bwd_dq_cross"), idx(&g, "attn_bwd_dk_cross_acc"), idx(&g, "attn_bwd_dv_cross_acc"));

    let (heads, head_dim, window) = (2u32, 4u32, 2u32);
    let d = heads * head_dim;
    let spans: &[(u32, u32)] = &[(0, 10)];
    let rows: u32 = 10;
    let mut r = Lcg::new(0x1B7A0002);
    let (qh, kh, vh) = (r.vec_scaled((rows * d) as usize, 0.5), r.vec_scaled((rows * d) as usize, 0.5), r.vec_scaled((rows * d) as usize, 0.5));
    let qkv_h = fuse_qkv(&qh, &kh, &vh, rows, d);
    let w = r.vec_scaled((rows * d) as usize, 0.5);

    let loss = |qkv_h: &[f32]| -> f64 {
        let (ctx_h, _) = run_win(&g, &cross_win, cross.softmax, cross.apply, &cross_bwd, window, heads, head_dim, d, qkv_h, &w, spans, rows);
        ctx_h.iter().zip(&w).map(|(&c, &wi)| c as f64 * wi as f64).sum()
    };
    let (_, dqkv) = run_win(&g, &cross_win, cross.softmax, cross.apply, &cross_bwd, window, heads, head_dim, d, &qkv_h, &w, spans, rows);

    // A handful of random ±1 directions over the WHOLE fused qkv buffer (q,
    // k and v regions all participate in the forward, so all three must be
    // perturbed together for the central difference to match the analytic
    // sum over `d_qkv`).
    let eps = 5e-3f32;
    let n = qkv_h.len();
    let mut worst_rel = 0.0f32;
    for seed in 0..4u64 {
        let mut r2 = Lcg::new(0x1B7A1000 + seed);
        let v: Vec<f32> = (0..n).map(|_| if r2.unit() < 0.5 { -1.0 } else { 1.0 }).collect();
        let analytic: f64 = dqkv.iter().zip(&v).map(|(&g, &vi)| g as f64 * vi as f64).sum();
        let plus: Vec<f32> = qkv_h.iter().zip(&v).map(|(&x, &vi)| x + eps * vi).collect();
        let minus: Vec<f32> = qkv_h.iter().zip(&v).map(|(&x, &vi)| x - eps * vi).collect();
        let numeric = (loss(&plus) - loss(&minus)) / (2.0 * eps as f64);
        let abs_err = (analytic - numeric).abs();
        let denom = analytic.abs().max(numeric.abs()).max(1e-3);
        let rel = (abs_err / denom) as f32;
        println!("  dir {seed}: analytic={analytic:+.5e} numeric={numeric:+.5e} rel={rel:.2e}");
        worst_rel = worst_rel.max(rel);
    }
    assert!(worst_rel < 2e-2, "windowed backward disagrees with finite differences: worst rel={worst_rel:e}");
}
