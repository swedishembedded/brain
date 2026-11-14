// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context encoder attention
// that costs O(T * window) instead of O(T^2) for its clients. If your team
// needs expertise in sliding-window attention kernels then you can procure our
// services by sending an email to info@swedishembedded.com.

//! `block::chunked_bidir_{fwd,bwd}_band` - bidirectional sliding-window
//! attention over a `[heads, rows, 2*window + 1]` slab.
//!
//! The oracle is the same window done the dense way: `chunked_bidir_{fwd,bwd}_win`
//! materializes the whole `[heads, rows, span]` slab and masks outside the band,
//! so the two compute the same softmax over the same live keys and may differ
//! only by the order of float additions. The band runs here use a score slab
//! of exactly `heads * chunk * (2*window + 1)` floats, which is the claim under
//! test: a slab any wider would run off the end of the buffer.
//!
//! Compared on the CPU backend and on the default device, at a span many times
//! the window (so interior rows have a full band and the first and last
//! `window` rows have a clipped one), over several chunk sizes including
//! ragged ones, and with a span shorter than the band.

use data::rng::{random_seed, Lcg};
use gpu_core::Gpu;
use model::block::{self, BandBwdIds, BandIds, CrossBwdIds, CrossWinIds};

const PIPES: &[(&str, &str)] = &[
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_scores_cross_win", kernels::ATTN_SCORES_CROSS_WIN),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS),
    ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS),
    ("attn_bwd_dk_cross_acc", kernels::ATTN_BWD_DK_CROSS_ACC),
    ("attn_bwd_dv_cross_acc", kernels::ATTN_BWD_DV_CROSS_ACC),
    ("attn_scores_band", kernels::ATTN_SCORES_BAND),
    ("attn_apply_band", kernels::ATTN_APPLY_BAND),
    ("attn_bwd_dscores_band", kernels::ATTN_BWD_DSCORES_BAND),
    ("attn_bwd_dq_band", kernels::ATTN_BWD_DQ_BAND),
    ("attn_bwd_dk_band_acc", kernels::ATTN_BWD_DK_BAND_ACC),
    ("attn_bwd_dv_band_acc", kernels::ATTN_BWD_DV_BAND_ACC),
];

/// Largest absolute difference tolerated between the banded and the masked
/// dense result. Both sum the same live terms; the dense softmax adds
/// `exp(-3.4e38 - max) == 0` terms and the dense apply/dk/dv add `0 * x`
/// terms, so the only difference is where a zero sits in a float sum.
/// Inputs are O(0.5) over `head_dim` 16, so values are O(1) and f32 round-off
/// per reduction is below 1e-6.
const TOL: f32 = 2e-5;

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn fuse_qkv(rng: &mut Lcg, rows: u32, d: u32) -> Vec<f32> {
    rng.vec_scaled((rows * 3 * d) as usize, 0.5)
}

struct Outputs {
    ctx: Vec<f32>,
    d_qkv: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn run_dense(g: &Gpu, window: u32, heads: u32, d: u32, qkv_h: &[f32], d_ctx_h: &[f32], spans: &[(u32, u32)], chunk: u32) -> Outputs {
    let hd = d / heads;
    let rows: u32 = spans.iter().map(|s| s.1).sum();
    let max_span = spans.iter().map(|s| s.1).max().unwrap();
    let slab = (heads * chunk * max_span) as u64;
    let win = CrossWinIds { scores: idx(g, "attn_scores_cross_win") };
    let (softmax, apply) = (idx(g, "attn_softmax_cross"), idx(g, "attn_apply_cross"));
    let bwd = CrossBwdIds::resolve(g, idx(g, "attn_bwd_dscores_cross"), idx(g, "attn_bwd_dq_cross"), idx(g, "attn_bwd_dk_cross_acc"), idx(g, "attn_bwd_dv_cross_acc"));

    let qkv = g.storage_init("qkv", qkv_h);
    let ctx = g.storage(rows as u64 * d as u64);
    let (scores, probs) = (g.storage(slab), g.storage(slab));
    let mut fwd = Vec::new();
    block::chunked_bidir_fwd_win(g, &win, softmax, apply, None, window, heads, hd, d, &qkv, 3 * d, 0, d, 2 * d, &ctx, &scores, &probs, spans, chunk, &mut fwd);
    g.submit(&[], &fwd);

    let d_ctx = g.storage_init("d_ctx", d_ctx_h);
    let d_qkv = g.storage(rows as u64 * 3 * d as u64);
    let mut back = Vec::new();
    block::chunked_bidir_bwd_win(g, &win, softmax, None, &bwd, window, heads, hd, d, &qkv, 3 * d, 0, d, 2 * d, &d_ctx, &d_qkv, &scores, &probs, &scores, spans, chunk, &mut back);
    g.submit(&[], &back);
    Outputs { ctx: g.read(&ctx, (rows * d) as usize), d_qkv: g.read(&d_qkv, (rows * 3 * d) as usize) }
}

#[allow(clippy::too_many_arguments)]
fn run_band(g: &Gpu, window: u32, heads: u32, d: u32, qkv_h: &[f32], d_ctx_h: &[f32], spans: &[(u32, u32)], chunk: u32) -> Outputs {
    let hd = d / heads;
    let rows: u32 = spans.iter().map(|s| s.1).sum();
    // The whole claim: the score slab is `[heads, chunk, 2*window + 1]`, not `[heads, chunk, span]`.
    let slab = (heads * chunk * (2 * window + 1)) as u64;
    let fwd_ids = BandIds { scores: idx(g, "attn_scores_band"), softmax: idx(g, "attn_softmax_cross"), apply: idx(g, "attn_apply_band") };
    let bwd_ids = BandBwdIds { dscores: idx(g, "attn_bwd_dscores_band"), dq: idx(g, "attn_bwd_dq_band"), dk: idx(g, "attn_bwd_dk_band_acc"), dv: idx(g, "attn_bwd_dv_band_acc") };

    let qkv = g.storage_init("qkv", qkv_h);
    let ctx = g.storage(rows as u64 * d as u64);
    let (scores, probs, d_scores) = (g.storage(slab), g.storage(slab), g.storage(slab));
    let mut fwd = Vec::new();
    block::chunked_bidir_fwd_band(g, &fwd_ids, window, heads, hd, d, &qkv, 3 * d, 0, d, 2 * d, &ctx, &scores, &probs, spans, chunk, &mut fwd);
    g.submit(&[], &fwd);

    let d_ctx = g.storage_init("d_ctx", d_ctx_h);
    // Poisoned: the backward must assign every key row, not rely on a cleared buffer.
    let d_qkv = g.storage_init("d_qkv", &vec![1e9f32; (rows * 3 * d) as usize]);
    let mut back = Vec::new();
    block::chunked_bidir_bwd_band(g, &fwd_ids, &bwd_ids, window, heads, hd, d, &qkv, 3 * d, 0, d, 2 * d, &d_ctx, &d_qkv, &scores, &probs, &d_scores, spans, chunk, &mut back);
    g.submit(&[], &back);
    Outputs { ctx: g.read(&ctx, (rows * d) as usize), d_qkv: g.read(&d_qkv, (rows * 3 * d) as usize) }
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn check(g: &Gpu, label: &str) {
    // d = 64 keeps every sliced row offset a multiple of the 256-byte binding alignment.
    let (heads, d) = (4u32, 64u32);
    // (window, spans, chunk): span 53 and 30 are several times 2*window+1;
    // chunk 8 and 7 are ragged against both; the 5-row span is shorter than the band (7).
    // (window, (start row, rows) spans, chunk)
    type Case = (u32, &'static [(u32, u32)], u32);
    let cases: &[Case] =
        &[(3, &[(0, 53), (53, 30), (83, 5)], 8), (3, &[(0, 53)], 7), (2, &[(0, 41)], 41), (5, &[(0, 64)], 1)];
    for &(window, spans, chunk) in cases {
        let rows: u32 = spans.iter().map(|s| s.1).sum();
        let mut r = Lcg::new(random_seed());
        let qkv_h = fuse_qkv(&mut r, rows, d);
        let d_ctx_h = r.vec_scaled((rows * d) as usize, 0.5);
        let chunk = chunk.min(spans.iter().map(|s| s.1).max().unwrap());
        let dense = run_dense(g, window, heads, d, &qkv_h, &d_ctx_h, spans, chunk);
        let band = run_band(g, window, heads, d, &qkv_h, &d_ctx_h, spans, chunk);
        let (wc, wg) = (worst(&dense.ctx, &band.ctx), worst(&dense.d_qkv, &band.d_qkv));
        println!("{label} window={window} chunk={chunk} spans={spans:?}: ctx {wc:e} d_qkv {wg:e}");
        assert!(wc < TOL, "{label}: band ctx differs from masked dense by {wc:e} (window {window}, chunk {chunk})");
        assert!(wg < TOL, "{label}: band d_qkv differs from masked dense by {wg:e} (window {window}, chunk {chunk})");
    }
}

#[test]
fn banded_attention_equals_masked_dense_on_the_cpu_backend() {
    check(&Gpu::new_cpu(PIPES), "cpu");
}

#[test]
fn banded_attention_equals_masked_dense_on_the_default_device() {
    check(&gpu_core::testgpu::dev(PIPES), "device");
}

/// The window must be load-bearing: the dense unwindowed attention is a different function.
#[test]
fn banded_attention_is_not_full_attention() {
    let g = Gpu::new_cpu(PIPES);
    let (heads, d, window) = (4u32, 64u32, 2u32);
    let spans: &[(u32, u32)] = &[(0, 40)];
    let mut r = Lcg::new(random_seed());
    let qkv_h = fuse_qkv(&mut r, 40, d);
    let d_ctx_h = r.vec_scaled((40 * d) as usize, 0.5);
    let narrow = run_band(&g, window, heads, d, &qkv_h, &d_ctx_h, spans, 8);
    let wide = run_dense(&g, 40, heads, d, &qkv_h, &d_ctx_h, spans, 8);
    assert!(worst(&narrow.ctx, &wide.ctx) > 1e-3, "window 2 over 40 rows must not equal full attention");
}
