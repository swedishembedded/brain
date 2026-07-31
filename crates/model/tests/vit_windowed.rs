// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Windowed attention and Hiera `q_pool` on top of `model::vit`, all composed
//! from kernels that already existed:
//!
//! * a window partition is a ROW PERMUTATION (`embed` / `row_scatter`), and the
//!   ViT block commutes with it, so `vit_block_fwd` needs no window parameter;
//! * `q_pool` is `embed -> nlc_nchw -> maxpool2d -> nchw_nlc`, and its adjoint
//!   is those four steps reversed with `maxpool2d_dx` in the middle;
//! * two-length attention is what the `*_cross` kernels already do.
//!
//! The last test is the MEASUREMENT the kernel checklist demands before anyone
//! proposes a dedicated `window_partition`/`window_reverse` kernel: it prints
//! the permutation cost next to the windowed-attention cost it enables.

use gpu_core::{DeviceBuffer, Gpu};
use model::block::CrossIds;
use model::vit::{
    cross_q_bwd, cross_q_fwd, gather_rows, probs_len, probs_offsets, q_pool_bwd, q_pool_fwd,
    region_index, max_slab, row_index_buffer, vit_block_fwd, vit_block_fwd_cached, window_partition,
    window_reverse, AttnSpan, QPoolCache, QPoolPlan, VitBlockCache, VitBlockWeights, VitBwdIds,
    VitKernelIds, VitPermuteIds, VitQPoolIds, VitScratch, VitShape, WindowIndex, WindowPlan,
};

const PIPES: &[(&str, &str)] = &[
    ("layernorm", kernels::LAYERNORM),
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("gelu_erf", kernels::GELU_ERF),
    ("scale_chan", kernels::SCALE_CHAN),
    ("add2", kernels::ADD2),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("ln_head", kernels::LN_HEAD),
    ("rope2d", kernels::ROPE2D),
    ("matmul_rows", kernels::MATMUL_ROWS),
    ("embed", kernels::EMBED),
    ("row_scatter", kernels::ROW_SCATTER),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("maxpool2d", kernels::MAXPOOL2D),
    ("maxpool2d_dx", kernels::MAXPOOL2D_DX),
    ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS),
    ("attn_bwd_dv_cross", kernels::ATTN_BWD_DV_CROSS),
    ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS),
    ("attn_bwd_dk_cross", kernels::ATTN_BWD_DK_CROSS),
    ("axpy", kernels::AXPY),
    ("ln_stats", kernels::LN_STATS),
    ("layernorm_dx", kernels::LAYERNORM_DX),
];

const K_EMBED: usize = 12;
const K_ROW_SCATTER: usize = 13;
const K_NLC_NCHW: usize = 14;
const K_NCHW_NLC: usize = 15;
const K_MAXPOOL: usize = 16;
const K_MAXPOOL_DX: usize = 17;
const K_AXPY: usize = 22;
const K_LN_STATS: usize = 23;
const K_LAYERNORM_DX: usize = 24;

fn fwd_ids() -> VitKernelIds {
    VitKernelIds {
        layernorm: 0,
        matmul: 1,
        bias_add: 2,
        gelu_erf: 3,
        scale_chan: 4,
        add2: 5,
        attn_scores_cross: 6,
        attn_softmax_cross: 7,
        attn_apply_cross: 8,
        ln_head: 9,
        rope2d: 10,
        matmul_rows: 11,
    }
}

/// Only the four `*_cross` backward slots are dispatched by `cross_q_bwd`; the
/// rest are never read on this path, so they carry the sentinel `usize::MAX` —
/// a wrong index here would be a silent miscompute, not a crash.
fn bwd_ids() -> VitBwdIds {
    VitBwdIds {
        layernorm_dx: K_LAYERNORM_DX,
        ln_dgamma: usize::MAX,
        ln_dbeta: usize::MAX,
        matmul_dx: usize::MAX,
        matmul_dw: usize::MAX,
        bias_grad: usize::MAX,
        gelu_erf_bwd: usize::MAX,
        scale_chan_dg: usize::MAX,
        ln_head_dx: usize::MAX,
        ln_head_dgb: usize::MAX,
        attn_bwd_dscores_cross: 18,
        attn_bwd_dv_cross: 19,
        attn_bwd_dq_cross: 20,
        attn_bwd_dk_cross: 21,
        ln_stats: K_LN_STATS,
        region_copy: usize::MAX,
        axpy: K_AXPY,
    }
}

fn perm_ids() -> VitPermuteIds {
    VitPermuteIds { embed: K_EMBED, row_scatter: K_ROW_SCATTER }
}

fn qpool_ids() -> VitQPoolIds {
    VitQPoolIds {
        permute: perm_ids(),
        nlc_nchw: K_NLC_NCHW,
        nchw_nlc: K_NCHW_NLC,
        maxpool2d: K_MAXPOOL,
        maxpool2d_dx: K_MAXPOOL_DX,
    }
}

fn cross_ids() -> CrossIds {
    CrossIds { scores: 6, softmax: 7, apply: 8 }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0) * 0.5
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn dev() -> Gpu {
    gpu_core::testgpu::dev(PIPES)
}

// ---------------------------------------------------------------------------
// 1. The partition is an exact permutation
// ---------------------------------------------------------------------------

#[test]
fn window_plan_partitions_every_row_once() {
    // The (4,4,4,4,2,2) row is the trap: every window is 2x2, i.e. equal
    // LENGTH, but none is win_h x win_w — `is_uniform` must still say false or
    // `QPoolPlan::per_window` pools a 4x4 grid that is really 2x2.
    for (gh, gw, wh, ww, sh, sw) in [
        (8u32, 8u32, 4u32, 4u32, 0u32, 0u32),
        (8, 8, 4, 4, 2, 2),
        (7, 5, 4, 4, 0, 0),
        (12, 8, 4, 4, 2, 1),
        (4, 4, 4, 4, 2, 2),
    ]
    {
        let p = WindowPlan::shifted(gh, gw, wh, ww, sh, sw);
        let rows = (gh * gw) as usize;
        assert_eq!(p.perm().len(), rows);
        let mut seen = vec![false; rows];
        for &r in p.perm() {
            assert!(!seen[r as usize], "row {r} appears twice");
            seen[r as usize] = true;
        }
        assert!(seen.iter().all(|&b| b), "partition does not cover every row");
        // spans tile the window-major buffer back to back
        let mut cur = 0u32;
        for &(r0, len) in p.spans() {
            assert_eq!(r0, cur);
            cur += len;
            assert!(len <= wh * ww);
        }
        assert_eq!(cur, rows as u32);
        // inv really inverts perm
        for (dst, &src) in p.perm().iter().enumerate() {
            assert_eq!(p.inv()[src as usize] as usize, dst);
        }
        // an unshifted, divisible grid is uniform; anything else is ragged
        assert_eq!(p.is_uniform(), sh == 0 && sw == 0 && gh % wh == 0 && gw % ww == 0);
    }
}

#[test]
fn partition_then_reverse_is_bit_exact() {
    let g = dev();
    let ids = perm_ids();
    let (gh, gw, c) = (8u32, 8u32, 16u32);
    let plan = WindowPlan::shifted(gh, gw, 4, 4, 2, 2);
    let wi = WindowIndex::new(&g, &plan);
    let mut r = Lcg(0x51DE);
    let host = r.vec((gh * gw * c) as usize);
    let x = g.storage_init("x", &host);
    let win = g.storage((gh * gw * c) as u64);
    let back = g.storage((gh * gw * c) as u64);
    let steps = vec![window_partition(&g, &ids, &wi, &x, &win, c), window_reverse(&g, &ids, &wi, &win, &back, c)];
    g.submit(&[], &steps);
    let got = g.read(&back, host.len());
    assert_eq!(got.to_vec(), host, "window_partition -> window_reverse must be bit-exact");

    // and the window-major buffer really is the permutation
    let wm = g.read(&win, host.len());
    for (dst, &src) in plan.perm().iter().enumerate() {
        let a = &wm[dst * c as usize..(dst + 1) * c as usize];
        let b = &host[src as usize * c as usize..(src as usize + 1) * c as usize];
        assert_eq!(a, b, "window-major row {dst} != grid row {src}");
    }
}

// ---------------------------------------------------------------------------
// 2. A ViT block commutes with the permutation: windowed == per-window
// ---------------------------------------------------------------------------

struct Blk {
    bufs: Vec<DeviceBuffer>,
}

impl Blk {
    /// `rope: None` deliberately — `rope2d` indexes its table by `row % tmod`,
    /// the one stage of the block that is NOT row-wise and therefore the one
    /// that does not commute with the partition.
    fn new(g: &Gpu, sh: &VitShape, r: &mut Lcg) -> Blk {
        let (c, m) = (sh.dim as usize, sh.mlp as usize);
        let sizes = [c, c, 3 * c * c, 3 * c, c * c, c, c, c, m * c, m, c * m, c];
        let mut bufs = Vec::new();
        for (i, n) in sizes.iter().enumerate() {
            let mut v = r.vec(*n);
            // norm gains near 1, projections scaled so softmax stays unsaturated
            if i == 0 || i == 6 {
                for x in v.iter_mut() {
                    *x = 1.0 + 0.3 * *x;
                }
            } else if i == 2 || i == 4 || i == 8 || i == 10 {
                for x in v.iter_mut() {
                    *x *= 0.35;
                }
            }
            bufs.push(g.storage_init("w", &v));
        }
        Blk { bufs }
    }
    fn w(&self) -> VitBlockWeights<'_> {
        VitBlockWeights {
            norm1_w: &self.bufs[0],
            norm1_b: &self.bufs[1],
            qkv_w: &self.bufs[2],
            qkv_b: &self.bufs[3],
            qk_norm: None,
            rope: None,
            proj_w: &self.bufs[4],
            proj_b: &self.bufs[5],
            ls1: None,
            norm2_w: &self.bufs[6],
            norm2_b: &self.bufs[7],
            fc1_w: &self.bufs[8],
            fc1_b: &self.bufs[9],
            fc2_w: &self.bufs[10],
            fc2_b: &self.bufs[11],
            ls2: None,
        }
    }
}

#[test]
fn windowed_block_equals_per_window_block() {
    let g = dev();
    let k = fwd_ids();
    let ids = perm_ids();
    let sh = VitShape { dim: 16, heads: 2, mlp: 32, eps: 1e-5 };
    let (gh, gw) = (8u32, 8u32);
    let (wh, ww) = (4u32, 4u32);
    let rows = gh * gw;
    let ws = wh * ww;
    let plan = WindowPlan::new(gh, gw, wh, ww);
    let wi = WindowIndex::new(&g, &plan);

    let mut r = Lcg(0xC0FFEE);
    let blk = Blk::new(&g, &sh, &mut r);
    let host = r.vec((rows * sh.dim) as usize);

    // --- path A: permute -> one block over window spans -> unpermute
    let x = g.storage_init("x", &host);
    let win = g.storage((rows * sh.dim) as u64);
    let out = g.storage((rows * sh.dim) as u64);
    let scr = VitScratch::new(&g, &sh, rows, ws, ws);
    let mut steps = vec![window_partition(&g, &ids, &wi, &x, &win, sh.dim)];
    vit_block_fwd(&g, &k, &sh, &blk.w(), &win, rows, plan.spans(), ws, &scr, &mut steps);
    steps.push(window_reverse(&g, &ids, &wi, &win, &out, sh.dim));
    g.submit(&[], &steps);
    let got = g.read(&out, host.len());

    // --- path B: each window on its own, as an independent [ws, C] sequence
    let scr1 = VitScratch::new(&g, &sh, ws, ws, ws);
    let mut want = vec![0f32; host.len()];
    for m in 0..plan.n_windows() as usize {
        let (row0, len) = plan.spans()[m];
        let mut wx = Vec::with_capacity((len * sh.dim) as usize);
        for i in 0..len {
            let src = plan.perm()[(row0 + i) as usize] as usize;
            wx.extend_from_slice(&host[src * sh.dim as usize..(src + 1) * sh.dim as usize]);
        }
        let wb = g.storage_init("wx", &wx);
        let mut s1 = Vec::new();
        vit_block_fwd(&g, &k, &sh, &blk.w(), &wb, len, &[(0, len)], ws, &scr1, &mut s1);
        g.submit(&[], &s1);
        let wo = g.read(&wb, wx.len());
        for i in 0..len {
            let dst = plan.perm()[(row0 + i) as usize] as usize;
            want[dst * sh.dim as usize..(dst + 1) * sh.dim as usize]
                .copy_from_slice(&wo[(i * sh.dim) as usize..((i + 1) * sh.dim) as usize]);
        }
    }

    let mut worst = 0f32;
    for (a, b) in got.iter().zip(want.iter()) {
        worst = worst.max((a - b).abs());
    }
    assert!(worst < 2e-5, "windowed block != per-window block, max abs diff {worst}");
}

// ---------------------------------------------------------------------------
// 2b. The RAGGED (Swin shifted-window) span list through the TRAINING path
// ---------------------------------------------------------------------------

/// The shifted partition is the whole point of `WindowPlan::shifted`, and the
/// cached forward is the only path that can train it. It binds `probs` at a
/// per-span offset, and `heads*len*len` summed over ragged spans is not a
/// multiple of the 64-float storage-binding alignment — so before
/// `probs_offsets` padded its slabs this test died inside the wgpu bind-group
/// validator ("Buffer offset 128 does not respect ... limit 256"), not on a
/// numeric assert. Nothing in the original suite exercised ragged spans through
/// `vit_block_fwd_cached`, which is why the arithmetic-only fix looked
/// complete.
///
/// Correctness oracle: the unchunked `vit_block_fwd` over the SAME ragged
/// spans, which binds `probs` once at offset 0 and therefore never had the
/// problem.
#[test]
fn shifted_window_cached_block_matches_unchunked() {
    let g = dev();
    let k = fwd_ids();
    let kb = bwd_ids();
    // dim 32 keeps every span's `row0*C` 64-aligned, the one binding constraint
    // that survives (see `WindowPlan::ctx_bindable`).
    let sh = VitShape { dim: 32, heads: 2, mlp: 64, eps: 1e-5 };
    let plan = WindowPlan::shifted(8, 8, 4, 4, 2, 2);
    let spans = plan.spans().to_vec();
    // ragged by construction: 4/8/16-row windows, not one uniform length
    assert!(spans.iter().any(|&(_, l)| l != spans[0].1), "this plan is supposed to be ragged");
    assert!(plan.ctx_bindable(sh.dim), "test shape must satisfy the ctx binding rule");
    let (rows, c, ms) = (plan.rows(), sh.dim, plan.max_span());

    let mut r = Lcg(0x5417);
    let blk = Blk::new(&g, &sh, &mut r);
    let host = r.vec((rows * c) as usize);

    // reference: the inference builder over the same ragged spans
    let xref = g.storage_init("x", &host);
    let scr = VitScratch::new(&g, &sh, rows, ms, ms);
    let mut s0 = Vec::new();
    vit_block_fwd(&g, &k, &sh, &blk.w(), &xref, rows, &spans, ms, &scr, &mut s0);
    g.submit(&[], &s0);
    let want = g.read(&xref, host.len());

    // the training builder: caches one probs slab per span at a padded offset
    let cache = VitBlockCache::new(&g, &sh, rows, ms);
    // `Gpu::write` takes the raw words; f32 bits go over unchanged.
    g.write(&cache.x_in, unsafe { core::slice::from_raw_parts(host.as_ptr() as *const u32, host.len()) });
    let x_out = g.storage((rows * c) as u64);
    let scr_tmp = g.storage((rows * c) as u64);
    let scores = g.storage((sh.heads * ms * ms) as u64);
    let mut s1 = Vec::new();
    vit_block_fwd_cached(&g, &k, &kb, &sh, &blk.w(), &cache, &x_out, rows, &spans, &scr_tmp, &scores, &mut s1);
    g.submit(&[&cache.qkv], &s1);
    let got = g.read(&x_out, host.len());

    let worst = got.iter().zip(want.iter()).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(worst < 2e-5, "cached ragged-span block != unchunked, max abs diff {worst}");

    // and every span's cached softmax must be a probability distribution — the
    // check that catches a span reading into its neighbour's slab.
    let att: Vec<AttnSpan> = spans.iter().map(|&(r0, l)| AttnSpan::span(r0, l)).collect();
    let at = probs_offsets(&att, sh.heads);
    let probs = g.read(&cache.probs, probs_len(&att, sh.heads) as usize);
    for (i, &(_, len)) in spans.iter().enumerate() {
        for h in 0..sh.heads as usize {
            for q in 0..len as usize {
                let row = at[i] as usize + (h * len as usize + q) * len as usize;
                let sum: f32 = probs[row..row + len as usize].iter().sum();
                assert!((sum - 1.0).abs() < 1e-4, "span {i} head {h} row {q}: softmax sums to {sum}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. q_pool forward == host max-pool of the q region
// ---------------------------------------------------------------------------

#[test]
fn q_pool_matches_host_maxpool() {
    let g = dev();
    let ids = qpool_ids();
    let c = 16u32;
    let (gh, gw) = (8u32, 4u32);
    let plan = WindowPlan::new(gh, gw, 4, 4); // 2 windows of 4x4
    let pool = QPoolPlan::per_window(&plan, 2, 2, 0);
    assert_eq!((pool.n, pool.ho(), pool.wo()), (2, 2, 2));
    let rows = plan.rows();

    let mut r = Lcg(0xBEEF);
    let qkv_host = r.vec((rows * 3 * c) as usize);
    let qkv = g.storage_init("qkv", &qkv_host);
    let q_idx = row_index_buffer(&g, "q_idx", &region_index(rows, 3, 0));
    let cache = QPoolCache::new(&g, &pool, c);
    let mut steps = Vec::new();
    q_pool_fwd(&g, &ids, &pool, c, &qkv, &q_idx, &cache, &mut steps);
    g.submit(&[], &steps);
    let got = g.read(&cache.q_pooled, (pool.rows_out() * c) as usize);

    // host reference, straight from Hiera's do_pool: max over each 2x2 tile of
    // the WINDOW-LOCAL token grid, per channel.
    let qrow = |t: u32, ch: u32| qkv_host[(t * 3 * c + ch) as usize];
    for m in 0..pool.n {
        for oh in 0..pool.ho() {
            for ow in 0..pool.wo() {
                for ch in 0..c {
                    let mut best = f32::NEG_INFINITY;
                    for dh in 0..pool.k {
                        for dw in 0..pool.k {
                            let ih = oh * pool.stride + dh;
                            let iw = ow * pool.stride + dw;
                            let t = m * pool.h * pool.w + ih * pool.w + iw;
                            best = best.max(qrow(t, ch));
                        }
                    }
                    let o = ((m * pool.ho() + oh) * pool.wo() + ow) * c + ch;
                    assert_eq!(got[o as usize], best, "q_pool mismatch at window {m} ({oh},{ow}) ch {ch}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Gradient check of the whole pooled-query attention chain
// ---------------------------------------------------------------------------

/// `q_pool_fwd` + `cross_q_fwd` on a 2-window 8x4 grid; loss = <ctx, wloss>.
/// Central finite differences over the fused qkv exercise all three of dq (via
/// the max-pool adjoint and the row scatter), dk and dv.
#[test]
fn pooled_attention_gradcheck() {
    let g = dev();
    let sh = VitShape { dim: 16, heads: 2, mlp: 32, eps: 1e-5 };
    let c = sh.dim;
    let plan = WindowPlan::new(8, 4, 4, 4);
    let pool = QPoolPlan::per_window(&plan, 2, 2, 0);
    let spans = AttnSpan::pooled_windows(&plan, &pool);
    let rows = plan.rows();
    let rows_q = pool.rows_out();
    assert_eq!(spans.len(), 2);

    let mut r = Lcg(0x1234_5678);
    let qkv_host = r.vec((rows * 3 * c) as usize);
    let wloss = r.vec((rows_q * c) as usize);

    let q_idx = row_index_buffer(&g, "q_idx", &region_index(rows, 3, 0));
    let cache = QPoolCache::new(&g, &pool, c);
    let ctx = g.storage((rows_q * c) as u64);
    // `scores`/`dscores` are transient (one span at a time) -> the largest
    // single slab; `probs` is CACHED for every span -> the sum.
    let scores = g.storage(max_slab(&spans, sh.heads));
    let probs = g.storage(probs_len(&spans, sh.heads));

    let loss_of = |qkv_h: &[f32]| -> f64 {
        let qkv = g.storage_init("qkv", qkv_h);
        let mut steps = Vec::new();
        q_pool_fwd(&g, &qpool_ids(), &pool, c, &qkv, &q_idx, &cache, &mut steps);
        cross_q_fwd(
            &g,
            &cross_ids(),
            &sh,
            &cache.q_pooled,
            c,
            0,
            &qkv,
            3 * c,
            c,
            2 * c,
            &ctx,
            &scores,
            &probs,
            &spans,
            &mut steps,
        );
        g.submit(&[], &steps);
        let o = g.read(&ctx, (rows_q * c) as usize);
        o.iter().zip(wloss.iter()).map(|(a, b)| *a as f64 * *b as f64).sum()
    };

    // analytic
    let qkv = g.storage_init("qkv", &qkv_host);
    let d_ctx = g.storage_init("d_ctx", &wloss);
    let d_q_pooled = g.storage((rows_q * c) as u64);
    let d_qkv = g.storage((rows * 3 * c) as u64);
    let dscores = g.storage(max_slab(&spans, sh.heads));
    let mut steps = Vec::new();
    q_pool_fwd(&g, &qpool_ids(), &pool, c, &qkv, &q_idx, &cache, &mut steps);
    cross_q_fwd(
        &g,
        &cross_ids(),
        &sh,
        &cache.q_pooled,
        c,
        0,
        &qkv,
        3 * c,
        c,
        2 * c,
        &ctx,
        &scores,
        &probs,
        &spans,
        &mut steps,
    );
    cross_q_bwd(
        &g,
        &bwd_ids(),
        &sh,
        &cache.q_pooled,
        c,
        0,
        &qkv,
        3 * c,
        c,
        2 * c,
        &probs,
        &d_ctx,
        &d_q_pooled,
        &d_qkv,
        &dscores,
        &spans,
        &mut steps,
    );
    q_pool_bwd(&g, &qpool_ids(), &pool, c, &d_q_pooled, &q_idx, &d_qkv, &cache, &mut steps);
    g.submit(&[], &steps);
    let analytic = g.read(&d_qkv, qkv_host.len());

    // central differences on a stratified sample of the qkv buffer
    let eps = 2e-3f32;
    let mut worst = 0f64;
    let mut checked = 0;
    for i in (0..qkv_host.len()).step_by(23) {
        let mut plus = qkv_host.clone();
        plus[i] += eps;
        let mut minus = qkv_host.clone();
        minus[i] -= eps;
        let num = (loss_of(&plus) - loss_of(&minus)) / (2.0 * eps as f64);
        let ana = analytic[i] as f64;
        let rel = (num - ana).abs() / (1.0 + num.abs().max(ana.abs()));
        worst = worst.max(rel);
        checked += 1;
    }
    assert!(checked > 50, "sampled only {checked} entries");
    assert!(worst < 2e-2, "pooled attention gradcheck: worst relative error {worst} over {checked} entries");
}

// ---------------------------------------------------------------------------
// 5. THE MEASUREMENT — is the permutation gather worth its own kernel?
// ---------------------------------------------------------------------------

fn time_steps(g: &Gpu, probe: &DeviceBuffer, steps: &[gpu_core::Step], iters: u32) -> f64 {
    g.submit(&[], steps); // warm-up (pipeline + first-touch)
    let _ = g.read(probe, 1);
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        g.submit(&[], steps);
    }
    let _ = g.read(probe, 1);
    t0.elapsed().as_secs_f64() * 1e3 / iters as f64
}

#[test]
fn measure_gather_vs_windowed_attention() {
    let g = dev();
    let ids = perm_ids();
    let k = fwd_ids();
    println!("\nbackend: {}", g.kind());
    println!("{:>18} {:>10} {:>10} {:>10} {:>10} {:>8}", "config", "perm ms", "attn ms", "qpool ms", "block ms", "perm/blk");

    // (grid, C, heads, window) — the second row is SAM 2 Hiera-B+ stage 0
    // (1024px input, patch stride 4, 8x8 windows) at half the token grid.
    for &(gh, gw, c, heads, win) in &[(64u32, 64u32, 256u32, 4u32, 8u32), (128, 128, 112, 2, 8)] {
        let sh = VitShape { dim: c, heads, mlp: 4 * c, eps: 1e-5 };
        let rows = gh * gw;
        let ws = win * win;
        let plan = WindowPlan::new(gh, gw, win, win);
        let wi = WindowIndex::new(&g, &plan);

        let mut r = Lcg(0xA5A5);
        let x = g.storage_init("x", &r.vec((rows * c) as usize));
        let winbuf = g.storage((rows * c) as u64);
        let out = g.storage((rows * c) as u64);

        let perm_steps =
            vec![window_partition(&g, &ids, &wi, &x, &winbuf, c), window_reverse(&g, &ids, &wi, &winbuf, &out, c)];

        // the windowed attention the permutation enables, on its own
        let qkv = g.storage_init("qkv", &r.vec((rows * 3 * c) as usize));
        let ctx = g.storage((rows * c) as u64);
        let slab = heads as u64 * ws as u64 * ws as u64;
        let scores = g.storage(slab);
        let probs = g.storage(slab);
        let mut attn_steps = Vec::new();
        model::vit::chunked_attn_fwd(&g, &k, &sh, &qkv, &ctx, &scores, &probs, plan.spans(), ws, &mut attn_steps);

        // the q_pool chain (4 dispatches) for the same partition
        let pool = QPoolPlan::per_window(&plan, 2, 2, 0);
        let q_idx = row_index_buffer(&g, "q_idx", &region_index(rows, 3, 0));
        let cache = QPoolCache::new(&g, &pool, c);
        let mut pool_steps = Vec::new();
        q_pool_fwd(&g, &qpool_ids(), &pool, c, &qkv, &q_idx, &cache, &mut pool_steps);

        // the whole block, for scale
        let blk = Blk::new(&g, &sh, &mut r);
        let scr = VitScratch::new(&g, &sh, rows, ws, ws);
        let mut blk_steps = Vec::new();
        vit_block_fwd(&g, &k, &sh, &blk.w(), &winbuf, rows, plan.spans(), ws, &scr, &mut blk_steps);

        let it = 10;
        let tp = time_steps(&g, &out, &perm_steps, it);
        let ta = time_steps(&g, &ctx, &attn_steps, it);
        let tq = time_steps(&g, &cache.q_pooled, &pool_steps, it);
        let tb = time_steps(&g, &winbuf, &blk_steps, it);
        // perm as a fraction OF THE BLOCK it is added to — `tp/(tp+tb)` would
        // flatter the permutation by putting its own cost in the denominator.
        println!(
            "{:>18} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>7.2}%",
            format!("{gh}x{gw}x{c}/w{win}"),
            tp,
            ta,
            tq,
            tb,
            100.0 * tp / tb
        );
    }
}

/// A permutation-only sanity check that `gather_rows` and `scatter_rows` really
/// are each other's inverse, which is what makes the backward free.
#[test]
fn gather_and_scatter_are_inverse() {
    let g = dev();
    let ids = perm_ids();
    let (n, d) = (64u32, 16u32);
    let plan = WindowPlan::new(8, 8, 4, 4);
    let fwd = row_index_buffer(&g, "p", plan.perm());
    let mut r = Lcg(7);
    let host = r.vec((n * d) as usize);
    let src = g.storage_init("src", &host);
    let mid = g.storage((n * d) as u64);
    let back = g.storage((n * d) as u64);
    let steps = vec![
        gather_rows(&g, &ids, &fwd, &src, &mid, n, d),
        model::vit::scatter_rows(&g, &ids, &fwd, &mid, &back, n, d, n),
    ];
    g.submit(&[], &steps);
    assert_eq!(g.read(&back, host.len()).to_vec(), host);
}
