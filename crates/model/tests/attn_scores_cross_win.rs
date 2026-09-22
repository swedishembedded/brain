// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `attn_scores_cross_win` / `attn_scores_cross_kt_win` - the BIDIRECTIONAL
//! sliding-window twins of `attn_scores_cross`/`attn_scores_cross_kt`, added
//! for ModernBERT's alternating full/local attention (`crates/modernbert`,
//! Laya). `gqa_scores_win` already covers the CAUSAL case
//! (`crates/model/tests/gqa_fwd_win.rs`); this is the same discipline for a
//! model whose local window looks both directions, not just backward.
//!
//! Same structure as that file: an independent host oracle (bidirectional,
//! f64 accumulation, no shared code with either kernel), a degenerate-window
//! exactness check against the UNWINDOWED kernels, and a mutation check that
//! the window is load-bearing rather than silently masked-then-ignored.

use data::rng::Lcg;
use gpu_core::Gpu;

const PIPES: &[(&str, &str)] = &[
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_scores_cross_win", kernels::ATTN_SCORES_CROSS_WIN),
    ("kv_k_headt", kernels::KV_K_HEADT),
    ("attn_scores_cross_kt", kernels::ATTN_SCORES_CROSS_KT),
    ("attn_scores_cross_kt_win", kernels::ATTN_SCORES_CROSS_KT_WIN),
];

fn dev() -> Gpu {
    gpu_core::testgpu::dev(PIPES)
}

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

/// Bidirectional sliding-window attention scores, host side: `scores[h,i,j] =
/// dot(q_i, k_j)/sqrt(hd)` when `|i-j| <= window`, else `-inf` (represented as
/// f32::NEG_INFINITY so a softmax test could reuse this; here only the
/// in-window entries are compared).
fn host_windowed_scores(q: &[f32], k: &[f32], t: u32, n_heads: u32, head_dim: u32, window: u32) -> Vec<f32> {
    let (t, nh, hd) = (t as usize, n_heads as usize, head_dim as usize);
    let mut out = vec![0f32; nh * t * t];
    for h in 0..nh {
        for i in 0..t {
            for j in 0..t {
                let dist = i.abs_diff(j);
                let idx = (h * t + i) * t + j;
                if dist > window as usize {
                    out[idx] = f32::NEG_INFINITY;
                    continue;
                }
                let qb = i * nh * hd + h * hd;
                let kb = j * nh * hd + h * hd;
                let dot: f64 = (0..hd).map(|d| q[qb + d] as f64 * k[kb + d] as f64).sum();
                out[idx] = (dot / (hd as f64).sqrt()) as f32;
            }
        }
    }
    out
}

/// `window >= t` must degenerate to the UNWINDOWED kernel's output (every
/// `|i-j| <= t-1 <= window` holds), for both the fused-KV and the key-minor
/// path - the property that lets a caller dispatch the windowed kernel
/// unconditionally on every layer, full-attention or local, rather than
/// keeping two call sites per layer type. Compared at a tight tolerance
/// rather than bit-exact: these are two SEPARATELY COMPILED kernel files with
/// identical math, and `gqa_scores_win`'s own degenerate-window test
/// (`gqa_fwd_win.rs`) uses the same `1e-6` bar for the same reason - the CPU
/// Cranelift JIT can codegen a textually-identical reduction loop into
/// slightly different FMA scheduling per kernel, unlike `kt_matches_cross`'s
/// `== 0.0` bar, which is checking one algebraic identity (a transpose), not
/// two independently-compiled kernels.
#[test]
fn window_covering_the_whole_span_matches_the_unwindowed_kernel_exactly() {
    let g = dev();
    let (heads, hd, t) = (3u32, 8u32, 7u32);
    let dm = heads * hd;
    let mut r = Lcg::new(0x1A7A);
    let qh = r.vec_scaled((t * dm) as usize, 0.5);
    let kh = r.vec_scaled((t * dm) as usize, 0.5);
    let q = g.storage_init("q", &qh);
    let kv = g.storage_init("kv", &kh);
    let n = (heads * t * t) as usize;

    let plain = idx(&g, "attn_scores_cross");
    let win = idx(&g, "attn_scores_cross_win");
    let s_plain = g.storage(n as u64);
    let s_win = g.storage(n as u64);
    g.submit(&[&s_plain], &[g.step(plain, &[&q, &kv, &s_plain], &[1, heads, t, t, hd, dm, dm, 0, 0], n as u32)]);
    g.submit(
        &[&s_win],
        &[g.step(win, &[&q, &kv, &s_win], &[1, heads, t, t, hd, dm, dm, 0, 0, 0, t], n as u32)],
    );
    let (a, b) = (g.read(&s_plain, n), g.read(&s_win, n));
    let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    assert!(worst < 1e-6, "window=t must match the unwindowed fused-KV kernel, worst abs diff {worst}");

    // Key-minor path: same claim, against `attn_scores_cross_kt`.
    let transpose = idx(&g, "kv_k_headt");
    let plain_kt = idx(&g, "attn_scores_cross_kt");
    let win_kt = idx(&g, "attn_scores_cross_kt_win");
    let kt = g.storage(dm as u64 * t as u64);
    let s_plain_kt = g.storage(n as u64);
    let s_win_kt = g.storage(n as u64);
    g.submit(
        &[&s_plain_kt],
        &[
            g.step(transpose, &[&kv, &kt], &[t, dm, dm, 0], dm * t),
            g.step(plain_kt, &[&q, &kt, &s_plain_kt], &[1, heads, t, t, hd, dm, 0], n as u32),
        ],
    );
    g.submit(
        &[&s_win_kt],
        &[
            g.step(transpose, &[&kv, &kt], &[t, dm, dm, 0], dm * t),
            g.step(win_kt, &[&q, &kt, &s_win_kt], &[1, heads, t, t, hd, dm, 0, 0, t], n as u32),
        ],
    );
    let (c, d) = (g.read(&s_plain_kt, n), g.read(&s_win_kt, n));
    let worst_kt = c.iter().zip(&d).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    assert!(worst_kt < 1e-6, "window=t must match the unwindowed key-minor kernel, worst abs diff {worst_kt}");
}

/// `window < t`: both windowed kernels must (a) match an independent
/// bidirectional host oracle on the in-window entries, (b) mask every
/// out-of-window entry to a very negative score (so a downstream softmax
/// reads it as 0 probability), and (c) actually DIFFER from the unwindowed
/// kernel's output - proving the window is load-bearing, mirroring
/// `gqa_fwd_win.rs`'s mutation check for the causal case.
#[test]
fn window_narrower_than_the_span_matches_host_oracle_and_masks_out_of_window_entries() {
    let g = dev();
    let (heads, hd, t, window) = (2u32, 4u32, 9u32, 2u32);
    let dm = heads * hd;
    let mut r = Lcg::new(0x0DEC1DE);
    let qh = r.vec_scaled((t * dm) as usize, 0.5);
    let kh = r.vec_scaled((t * dm) as usize, 0.5);
    let q = g.storage_init("q", &qh);
    let kv = g.storage_init("kv", &kh);
    let n = (heads * t * t) as usize;

    let win = idx(&g, "attn_scores_cross_win");
    let s_win = g.storage(n as u64);
    g.submit(
        &[&s_win],
        &[g.step(win, &[&q, &kv, &s_win], &[1, heads, t, t, hd, dm, dm, 0, 0, 0, window], n as u32)],
    );
    let got = g.read(&s_win, n);

    let want = host_windowed_scores(&qh, &kh, t, heads, hd, window);
    let mut worst_in_window = 0f32;
    for h in 0..heads as usize {
        for i in 0..t as usize {
            for j in 0..t as usize {
                let k = (h * t as usize + i) * t as usize + j;
                if i.abs_diff(j) <= window as usize {
                    worst_in_window = worst_in_window.max((got[k] - want[k]).abs());
                } else {
                    assert!(got[k] < -1e30, "out-of-window entry h={h} i={i} j={j} was not masked: {}", got[k]);
                }
            }
        }
    }
    assert!(worst_in_window < 1e-4, "windowed kernel != host oracle on in-window entries, worst abs diff {worst_in_window}");

    // Mutation check: the unwindowed scores must disagree at a masked
    // position (e.g. h=0, i=0, j=t-1, distance t-1 > window).
    let plain = idx(&g, "attn_scores_cross");
    let s_plain = g.storage(n as u64);
    g.submit(&[&s_plain], &[g.step(plain, &[&q, &kv, &s_plain], &[1, heads, t, t, hd, dm, dm, 0, 0], n as u32)]);
    let unwindowed = g.read(&s_plain, n);
    // Spelled out as (head, query, key) rather than folded, so the index
    // still says which entry it is - the zeros are the point, and writing
    // them into the arithmetic is what clippy's `erasing_op` denies.
    let (head, query, key) = (0usize, 0usize, t as usize - 1);
    let masked_pos = (head * t as usize + query) * t as usize + key;
    assert!(
        (got[masked_pos] - unwindowed[masked_pos]).abs() > 1e30,
        "windowed and unwindowed scores must diverge at a masked position"
    );
}
