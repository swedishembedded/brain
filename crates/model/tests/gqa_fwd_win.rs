// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::block::gqa_fwd_win` — the sliding-window causal sibling of
//! `gqa_fwd`, added for `crates/codec`'s Code2Wav pre-transformer
//! (`sliding_window: 72` was parsed into `CodecConfig` and never applied —
//! see `crates/codec/src/model.rs::transformer`'s doc for the real bug this
//! closes and why it affects both `Codec::decode` and `Codec::decode_omni`).
//!
//! No real checkpoint is available in this environment (`docs/models/omni/
//! status.md`'s M17 testdata audit: the omni checkpoint mirror is the one
//! subtree actually restorable here, and even that is 66 GB — not staged for
//! a routine test run), so this is the tiny-config rung of the parity ladder
//! (`docs/porting-playbook.md` §4-5): synthetic weights, an independent
//! host-side masked-attention oracle, and a mutation-style check that the
//! window is actually load-bearing (not merely "runs without panicking"). A
//! real T>72 golden against the released checkpoint is real, separate
//! follow-up work once the checkpoint is staged — tracked in
//! `.todo/omni-chunked-code2wav.md`.

use data::rng::Lcg;
use gpu_core::Gpu;
use model::block::{Gqa, KernelIds};

const PIPES: &[(&str, &str)] = &[
    ("gqa_scores", kernels::GQA_SCORES),
    ("gqa_scores_win", kernels::GQA_SCORES_WIN),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn dev() -> Gpu {
    gpu_core::testgpu::dev(PIPES)
}

fn ids(g: &Gpu) -> KernelIds {
    KernelIds {
        rmsnorm: usize::MAX,
        rms_inv: usize::MAX,
        rmsnorm_dx: usize::MAX,
        rmsnorm_dw: usize::MAX,
        rope: usize::MAX,
        rope_bwd: usize::MAX,
        gqa_scores: idx(g, "gqa_scores"),
        gqa_apply: idx(g, "gqa_apply"),
        attn_softmax: idx(g, "attn_softmax"),
        gqa_dscores: usize::MAX,
        gqa_dv: usize::MAX,
        gqa_dq: usize::MAX,
        gqa_dk: usize::MAX,
        silu_mul: usize::MAX,
        silu_da: usize::MAX,
        silu_db: usize::MAX,
    }
}

/// Independent host oracle: plain softmax attention restricted to keys
/// `max(0, i-window+1)..=i` for each query row `i` — the same masked-causal
/// formula `gqa_scores_win.wgsl`'s doc states, computed a completely
/// different way (host f64 accumulation, no shared code with the kernel) so
/// this test cannot pass merely because a shared bug exists on both sides.
/// MHA only (`n_kv_heads == n_heads`), matching this test's shapes.
fn host_windowed_attention(q: &[f32], k: &[f32], v: &[f32], t: u32, n_heads: u32, head_dim: u32, window: u32) -> Vec<f32> {
    let (t, nh, hd) = (t as usize, n_heads as usize, head_dim as usize);
    let mut ctx = vec![0f32; t * nh * hd];
    let scale = 1.0 / (hd as f64).sqrt();
    for h in 0..nh {
        for i in 0..t {
            let j0 = i.saturating_sub(window as usize - 1);
            let mut scores = Vec::with_capacity(i - j0 + 1);
            for j in j0..=i {
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
                for (pi, j) in (j0..=i).enumerate() {
                    acc += probs[pi] * v[j * nh * hd + h * hd + d] as f64;
                }
                ctx[i * nh * hd + h * hd + d] = acc as f32;
            }
        }
    }
    ctx
}

/// `window >= t` must degenerate to `gqa_fwd`'s plain causal mask exactly
/// (every `i-j <= t-1 < window` holds) — the property that lets
/// `Codec::transformer` dispatch `gqa_fwd_win` unconditionally instead of
/// keeping a separate unwindowed call site.
#[test]
fn window_covering_the_whole_sequence_matches_plain_causal() {
    let g = dev();
    let k = ids(&g);
    let win_kernel = idx(&g, "gqa_scores_win");
    let (t, n_heads, head_dim) = (6u32, 2u32, 4u32);
    let ga = Gqa { b: 1, t, n_heads, n_kv_heads: n_heads, head_dim };

    let mut r = Lcg::new(0x9E17);
    let qh = r.vec_scaled((t * n_heads * head_dim) as usize, 0.5);
    let kh = r.vec_scaled((t * n_heads * head_dim) as usize, 0.5);
    let vh = r.vec_scaled((t * n_heads * head_dim) as usize, 0.5);

    let run = |windowed: bool| -> Vec<f32> {
        let q = g.storage_init("q", &qh);
        let kb = g.storage_init("k", &kh);
        let v = g.storage_init("v", &vh);
        let scores = g.storage((n_heads * t * t) as u64);
        let probs = g.storage((n_heads * t * t) as u64);
        let ctx = g.storage((t * n_heads * head_dim) as u64);
        let steps = if windowed {
            model::block::gqa_fwd_win(&g, win_kernel, &k, &ga, t, &q, &kb, &v, &scores, &probs, &ctx)
        } else {
            model::block::gqa_fwd(&g, &k, &ga, &q, &kb, &v, &scores, &probs, &ctx)
        };
        g.submit(&[], &steps);
        g.read(&ctx, (t * n_heads * head_dim) as usize).to_vec()
    };

    let causal = run(false);
    let windowed_full = run(true);
    let worst = causal.iter().zip(&windowed_full).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(worst < 1e-6, "window=t must match plain causal exactly, worst abs diff {worst}");
}

/// `window < t`: `gqa_fwd_win` must (a) match the independent host oracle,
/// and (b) actually DIFFER from plain `gqa_fwd`'s output at a query row whose
/// causal context extends further back than the window — proving the window
/// is load-bearing, not silently masked-then-ignored (the exact shape of bug
/// this kernel exists to fix: `sliding_window` was parsed into `CodecConfig`
/// and never reached the attention dispatch at all).
#[test]
fn window_narrower_than_the_sequence_matches_host_oracle_and_diverges_from_plain_causal() {
    let g = dev();
    let k = ids(&g);
    let win_kernel = idx(&g, "gqa_scores_win");
    let (t, n_heads, head_dim, window) = (8u32, 2u32, 4u32, 3u32);
    let ga = Gqa { b: 1, t, n_heads, n_kv_heads: n_heads, head_dim };

    let mut r = Lcg::new(0x51A5);
    let qh = r.vec_scaled((t * n_heads * head_dim) as usize, 0.5);
    let kh = r.vec_scaled((t * n_heads * head_dim) as usize, 0.5);
    let vh = r.vec_scaled((t * n_heads * head_dim) as usize, 0.5);

    let q = g.storage_init("q", &qh);
    let kb = g.storage_init("k", &kh);
    let v = g.storage_init("v", &vh);

    let scores_w = g.storage((n_heads * t * t) as u64);
    let probs_w = g.storage((n_heads * t * t) as u64);
    let ctx_w = g.storage((t * n_heads * head_dim) as u64);
    let steps_w = model::block::gqa_fwd_win(&g, win_kernel, &k, &ga, window, &q, &kb, &v, &scores_w, &probs_w, &ctx_w);
    g.submit(&[], &steps_w);
    let got = g.read(&ctx_w, (t * n_heads * head_dim) as usize).to_vec();

    let want = host_windowed_attention(&qh, &kh, &vh, t, n_heads, head_dim, window);
    let worst = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(worst < 1e-4, "windowed kernel != host oracle, worst abs diff {worst}");

    // mutation check: plain (unwindowed) causal must land somewhere else at
    // the last row, which under window=3 only ever sees keys {5,6,7} but
    // under plain causal sees keys {0..7}.
    let scores_c = g.storage((n_heads * t * t) as u64);
    let probs_c = g.storage((n_heads * t * t) as u64);
    let ctx_c = g.storage((t * n_heads * head_dim) as u64);
    let steps_c = model::block::gqa_fwd(&g, &k, &ga, &q, &kb, &v, &scores_c, &probs_c, &ctx_c);
    g.submit(&[], &steps_c);
    let unwindowed = g.read(&ctx_c, (t * n_heads * head_dim) as usize).to_vec();

    let last_row = ((t - 1) * n_heads * head_dim) as usize..(t * n_heads * head_dim) as usize;
    let diff_at_last_row = got[last_row.clone()].iter().zip(&unwindowed[last_row]).any(|(a, b)| (a - b).abs() > 1e-3);
    assert!(diff_at_last_row, "windowed and unwindowed attention must diverge at the last row when window < t");
}
