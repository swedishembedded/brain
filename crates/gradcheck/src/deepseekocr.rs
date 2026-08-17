// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference gate for **SAM's decomposed relative-position bias**
//! (`attn_relpos_qr` / `_add` / `_drh` / `_drw` / `_dq` / `_dr`, plus the
//! `embed`+`scale_row`+`add2`+`nlc_nchw`/`emb_bwd` composition that builds and
//! differentiates the dense table) - DeepSeek-OCR's SAM ViT-B tower is the
//! first model that will consume it.
//!
//! ## What this check covers
//!
//! No model crate consumes these kernels yet, so the harness IS the fixture: a
//! two-"block" graph built straight on `model::block::chunked_bidir_{fwd,bwd}`
//! with the rel-pos `Option` engaged, over a fused `[rows, 3C]` qkv buffer that
//! is itself a leaf parameter. Five tensors are checked:
//!
//! | tensor | shape | what its gradient exercises |
//! |---|---|---|
//! | `w.qkv` / `g.qkv` | `[rows, 3C]` | `attn_bwd_d{scores,q,k,v}_cross` **plus** `attn_relpos_dq`'s accumulate-on-top (both axes) - a dropped rel-pos term is a wrong `dq` here and nowhere else |
//! | `w.rel_pos_h` | `[5, 8]` | height table, **upsampled** 5 -> 7 by the half-pixel resample (`2*4-1`) |
//! | `w.rel_pos_w` | `[3, 8]` | width table, **identity** (3 == `2*2-1`, no resample at all) |
//! | `g.rel_pos_h` | `[13, 8]` | height table, **downsampled** 13 -> 9 (`2*5-1`) |
//! | `g.rel_pos_w` | `[7, 8]` | width table, **downsampled** 7 -> 5 (`2*3-1`) |
//!
//! Between them the three `get_rel_pos` cases (identity / upsample /
//! downsample) all run, in one graph.
//!
//! ## The fixture
//!
//! Two blocks, deliberately pairwise-distinct in every extent so a swapped
//! index cannot cancel:
//!
//! * **`w` - windowed.** A 7x3 token grid zero-padded to **8x4** (SAM's own
//!   discipline: pad the grid to a multiple of the window, crop after) and
//!   partitioned by `WindowPlan::new(8, 4, 4, 2)` into **4 windows of 4x2**.
//!   Four spans share ONE pair of tables, which is what makes
//!   `attn_relpos_dr`'s accumulate flag load-bearing.
//! * **`g` - global.** One unwindowed **5x3** span, 15 rows - a query count
//!   that the chunk size does not divide, so the last chunk is short.
//!
//! `heads = 2`, `head_dim = 8` (C = 16, qkv stride 48), and `chunk = 4`
//! everywhere, so **every span is multi-chunk**: the `q0` threading through
//! `attn_relpos_add` / `_drh` / `_drw` is exercised rather than assumed.
//! `(row0 + q0) * C` is a multiple of 64 for every dispatch, which is what
//! `attn_apply_cross`'s offset-less `ctx` binding requires.
//!
//! ## The objective
//!
//! ```text
//! L = <r_w, ctx_w> + <r_g, ctx_g>
//! ```
//! with `r_b` fixed random directions on each block's attention output. `L` is
//! exactly linear in them, so `backward()` seeds `d_ctx` with `r` directly.
//!
//! ## Why there are TWO entry points
//!
//! [`check_deepseekocr_relpos`] is the directional check over all five
//! tensors. [`check_deepseekocr_relpos_elementwise`] is a per-ENTRY check on
//! the four TABLES, and it is not redundant: a table is folded across every
//! window, head, query row and chunk, which is exactly the shape
//! `directional_check` is measurably blind to - T5's `rel_bias.weight` is this
//! repo's recorded instance (see `gradcheck::elementwise_check`'s rustdoc).
//!
//! **Mutation-verified, seed 7, both mutations reverted.** Two breaks were
//! introduced on purpose and the gates' answers recorded rather than assumed:
//!
//! | mutation | directional (per tensor) | per entry |
//! |---|---|---|
//! | `attn_relpos_dr` ASSIGNS instead of accumulating (only the LAST of `w`'s four windows survives) | **RED** - `w.rel_pos_h` rel 2.33e-1, `w.rel_pos_w` rel 6.30e-1 | RED |
//! | the SECOND interpolation tap dropped from the table adjoint (`RelPosAxis::build_bwd`) | `w.rel_pos_h` rel **5.43e-2 - a PASS** inside the `(4e-3, 8e-2)` gate | **RED**: 24 of that tensor's 40 entries fail, worst rel 1.83 (analytic +3.69e-2 vs numeric −4.43e-2, opposite sign) |
//!
//! So the first break is too LARGE to hide (75 % of the gradient missing) and
//! the directional check catches it; the second is the partial, entry-dependent
//! kind - the tap's weight varies per resampled row, so the contraction onto a
//! ±1 direction nearly cancels - and only the per-entry check sees it. That
//! second row is the whole justification for this file's second entry point.

use std::cell::Cell;

use data::rng::Rng;
use gpu_core::{DeviceBuffer, Gpu, Step};
use model::block::{self, CrossBwdIds, CrossIds};
use model::vit::{RelPos, RelPosAxis, RelPosBwd, RelPosIds, RelPosTableIds, WindowPlan};

use crate::{directional_check, CheckModel, Report};

/// Pipeline order is the index space every `*Ids` below refers to.
const PIPES: &[(&str, &str)] = &[
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS),
    ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS),
    ("attn_bwd_dk_cross_acc", kernels::ATTN_BWD_DK_CROSS_ACC),
    ("attn_bwd_dv_cross_acc", kernels::ATTN_BWD_DV_CROSS_ACC),
    ("attn_relpos_qr", kernels::ATTN_RELPOS_QR),
    ("attn_relpos_add", kernels::ATTN_RELPOS_ADD),
    ("attn_relpos_drh", kernels::ATTN_RELPOS_DRH),
    ("attn_relpos_drw", kernels::ATTN_RELPOS_DRW),
    ("attn_relpos_dq", kernels::ATTN_RELPOS_DQ),
    ("attn_relpos_dr", kernels::ATTN_RELPOS_DR),
    ("embed", kernels::EMBED),
    ("scale_row", kernels::SCALE_ROW),
    ("add2", kernels::ADD2),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("emb_bwd", kernels::EMB_BWD),
];

const FWD: CrossIds = CrossIds { scores: 0, softmax: 1, apply: 2 };
const BWD: CrossBwdIds = CrossBwdIds { dscores: 3, dq: 4, dk_acc: 5, dv_acc: 6 };
const REL: RelPosIds = RelPosIds { qr: 7, add: 8, drh: 9, drw: 10, dq: 11, dr: 12 };
const TBL: RelPosTableIds = RelPosTableIds { embed: 13, scale_row: 14, add2: 15, nlc_nchw: 16, emb_bwd: 17 };

const HEADS: u32 = 2;
const HEAD_DIM: u32 = 8;
const C: u32 = HEADS * HEAD_DIM;
const STRIDE: u32 = 3 * C;
const CHUNK: u32 = 4;

/// One block of the fixture: geometry, its two learned tables, and every
/// device buffer the forward and backward bind.
struct Blk {
    name: &'static str,
    rows: u32,
    qh: u32,
    qw: u32,
    spans: Vec<(u32, u32)>,
    ax_h: RelPosAxis,
    ax_w: RelPosAxis,
    table_h: DeviceBuffer,
    table_w: DeviceBuffer,
    d_table_h: DeviceBuffer,
    d_table_w: DeviceBuffer,
    tbl_scratch: DeviceBuffer,
    qkv: DeviceBuffer,
    d_qkv: DeviceBuffer,
    ctx: DeviceBuffer,
    d_ctx: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    d_scores: DeviceBuffer,
    rel_h: DeviceBuffer,
    rel_w: DeviceBuffer,
    d_rel_h: DeviceBuffer,
    d_rel_w: DeviceBuffer,
    d_rh: DeviceBuffer,
    d_rw: DeviceBuffer,
    /// `[rows, C]` proxy direction defining this block's share of `L`.
    r: Vec<f32>,
    lh: u32,
    lw: u32,
}

impl Blk {
    #[allow(clippy::too_many_arguments)]
    fn new(g: &Gpu, name: &'static str, qh: u32, qw: u32, spans: Vec<(u32, u32)>, lh: u32, lw: u32, rng: &mut Rng) -> Blk {
        let rows: u32 = spans.iter().map(|&(_, l)| l).sum();
        let span_qn = qh * qw;
        let max_span = span_qn;
        let slab = HEADS as u64 * CHUNK.min(max_span) as u64 * max_span as u64;
        let ax_h = RelPosAxis::new(g, qh, qh, HEAD_DIM, lh);
        let ax_w = RelPosAxis::new(g, qw, qw, HEAD_DIM, lw);
        let dense_h = (qh * qh * HEAD_DIM) as u64;
        let dense_w = (qw * qw * HEAD_DIM) as u64;
        let mut init = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.next_f32() - 0.5).collect() };
        Blk {
            name,
            rows,
            qh,
            qw,
            ax_h,
            ax_w,
            table_h: g.storage_init("table_h", &init((lh * HEAD_DIM) as usize)),
            table_w: g.storage_init("table_w", &init((lw * HEAD_DIM) as usize)),
            d_table_h: g.storage((lh * HEAD_DIM) as u64),
            d_table_w: g.storage((lw * HEAD_DIM) as u64),
            tbl_scratch: g.storage(dense_h.max(dense_w)),
            qkv: g.storage_init("qkv", &init((rows * STRIDE) as usize)),
            d_qkv: g.storage((rows * STRIDE) as u64),
            ctx: g.storage((rows * C) as u64),
            d_ctx: g.storage((rows * C) as u64),
            scores: g.storage(slab),
            probs: g.storage(slab),
            d_scores: g.storage(slab),
            rel_h: g.storage((HEADS * span_qn * qh) as u64),
            rel_w: g.storage((HEADS * span_qn * qw) as u64),
            d_rel_h: g.storage((HEADS * span_qn * qh) as u64),
            d_rel_w: g.storage((HEADS * span_qn * qw) as u64),
            d_rh: g.storage(dense_h),
            d_rw: g.storage(dense_w),
            r: init((rows * C) as usize),
            spans,
            lh,
            lw,
        }
    }

    fn relpos(&self, bwd: bool) -> RelPos<'_> {
        RelPos {
            ids: REL,
            qh: self.qh,
            qw: self.qw,
            kh: self.qh,
            kw: self.qw,
            rh_t: &self.ax_h.r_t,
            rw_t: &self.ax_w.r_t,
            rel_h: &self.rel_h,
            rel_w: &self.rel_w,
            bwd: bwd.then_some(RelPosBwd {
                rh: &self.ax_h.r,
                rw: &self.ax_w.r,
                d_rh: &self.d_rh,
                d_rw: &self.d_rw,
                d_rel_h: &self.d_rel_h,
                d_rel_w: &self.d_rel_w,
                acc0: false,
            }),
        }
    }

    fn build_tables(&self, g: &Gpu, steps: &mut Vec<Step>) {
        self.ax_h.build_fwd(g, &TBL, &self.table_h, steps);
        self.ax_w.build_fwd(g, &TBL, &self.table_w, steps);
    }

    fn fwd(&self, g: &Gpu) {
        let mut steps = Vec::new();
        self.build_tables(g, &mut steps);
        let rel = self.relpos(false);
        block::chunked_bidir_fwd(
            g, &FWD, None, HEADS, HEAD_DIM, C, &self.qkv, STRIDE, 0, C, 2 * C, &self.ctx, &self.scores, &self.probs,
            &self.spans, CHUNK, Some(&rel), &mut steps,
        );
        g.submit(&[&self.ctx], &steps);
    }

    fn bwd(&self, g: &Gpu) {
        g.write_f32(&self.d_ctx, &self.r);
        let mut steps = Vec::new();
        // The tables are rebuilt here too, so the backward never depends on
        // whichever forward happened to run last.
        self.build_tables(g, &mut steps);
        let rel = self.relpos(true);
        block::chunked_bidir_bwd(
            g, &FWD, None, &BWD, HEADS, HEAD_DIM, C, &self.qkv, STRIDE, 0, C, 2 * C, &self.d_ctx, &self.d_qkv,
            &self.scores, &self.probs, &self.d_scores, &self.spans, CHUNK, Some(&rel), &mut steps,
        );
        // Dense-table adjoint -> learned-table adjoint, `emb_bwd` accumulating
        // over both interpolation taps (hence the zero-clear).
        self.ax_h.build_bwd(g, &TBL, &self.d_rh, &self.d_table_h, &self.tbl_scratch, &mut steps);
        self.ax_w.build_bwd(g, &TBL, &self.d_rw, &self.d_table_w, &self.tbl_scratch, &mut steps);
        g.submit(&[&self.d_table_h, &self.d_table_w], &steps);
    }

    fn param(&self, which: &str) -> (&DeviceBuffer, &DeviceBuffer, usize) {
        match which {
            "qkv" => (&self.qkv, &self.d_qkv, (self.rows * STRIDE) as usize),
            "rel_pos_h" => (&self.table_h, &self.d_table_h, (self.lh * HEAD_DIM) as usize),
            "rel_pos_w" => (&self.table_w, &self.d_table_w, (self.lw * HEAD_DIM) as usize),
            _ => panic!("unknown parameter {which}"),
        }
    }
}

/// The two-block fixture as a [`CheckModel`].
struct RelPosHarness {
    g: Gpu,
    blocks: Vec<Blk>,
    fwd_done: Cell<bool>,
}

impl RelPosHarness {
    /// The pooled test device (`gpu_core::testgpu::dev`), never a fresh
    /// `Gpu::new` - several entry points share one test binary.
    fn new(seed: u64) -> RelPosHarness {
        let g = gpu_core::testgpu::dev(PIPES);
        let mut rng = Rng::new(seed ^ 0xD0C5);
        // Windowed: a 7x3 grid padded to 8x4, window 4x2 -> four uniform spans.
        let plan = WindowPlan::new(8, 4, 4, 2);
        assert!(plan.is_uniform(), "the windowed fixture must be uniform");
        let w = Blk::new(&g, "w", 4, 2, plan.spans().to_vec(), 5, 3, &mut rng);
        // Global: one unwindowed 5x3 span.
        let gl = Blk::new(&g, "g", 5, 3, vec![(0, 15)], 13, 7, &mut rng);
        RelPosHarness { g, blocks: vec![w, gl], fwd_done: Cell::new(false) }
    }

    fn split<'a>(&'a self, name: &'a str) -> (&'a Blk, &'a str) {
        let (b, p) = name.split_once('.').expect("param name is <block>.<tensor>");
        (self.blocks.iter().find(|x| x.name == b).expect("block"), p)
    }
}

impl CheckModel for RelPosHarness {
    fn param_names(&self) -> Vec<String> {
        let mut v = Vec::new();
        for b in &self.blocks {
            for t in ["qkv", "rel_pos_h", "rel_pos_w"] {
                v.push(format!("{}.{t}", b.name));
            }
        }
        v
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        let (b, p) = self.split(name);
        let (buf, _, n) = b.param(p);
        self.g.read(buf, n)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        let (b, p) = self.split(name);
        let (buf, _, n) = b.param(p);
        assert_eq!(data.len(), n, "{name}: size mismatch");
        self.g.write_f32(buf, data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        let (b, p) = self.split(name);
        let (_, grad, n) = b.param(p);
        self.g.read(grad, n)
    }
    fn loss(&self) -> f32 {
        let mut dot = 0f64;
        for b in &self.blocks {
            b.fwd(&self.g);
            let out = self.g.read(&b.ctx, (b.rows * C) as usize);
            // f64 accumulation: `elementwise_check` differences a loss that
            // moves by ~1e-3 of itself, so an f32 accumulator's round-off would
            // land straight in the numerator.
            dot += out.iter().zip(&b.r).map(|(y, r)| *y as f64 * *r as f64).sum::<f64>();
        }
        self.fwd_done.set(true);
        dot as f32
    }
    fn zero_grads(&self) {
        // Every gradient buffer here is fully ASSIGNED by its first writer
        // (`d_qkv` by the cross backward, `d_rh`/`d_rw` by the first span's
        // `attn_relpos_dr`), and the two learned-table grads are cleared inside
        // `Blk::bwd`'s own submit because `emb_bwd` accumulates.
    }
    fn backward(&self) {
        if !self.fwd_done.get() {
            let _ = self.loss();
        }
        for b in &self.blocks {
            b.bwd(&self.g);
        }
        self.g.poll_wait();
    }
}

/// **The gate.** Directional finite differences over the whole rel-pos chain:
/// the `q·R` hoist, the in-place fold, all four backward kernels, and the
/// interpolation gather that builds the dense tables.
///
/// `eps = 5e-4`, not the workspace default `5e-3`: the largest tensor here is
/// `w.qkv` at 1536 entries, where a ±1 direction at `5e-3` is an L2 step of
/// 0.196 in weight space - well outside the region where a softmax is locally
/// linear. [`check_deepseekocr_relpos_eps_sweep`] is the measurement behind
/// that choice.
pub fn check_deepseekocr_relpos(seed: u64) -> Report {
    let h = RelPosHarness::new(seed);
    directional_check(&h, 5e-4, 4, seed ^ 0x1234)
}

/// **The gate that actually covers the shared-table fold.** Per-ENTRY finite
/// differences on all four `rel_pos_*` tables.
///
/// The windowed block's two tables are read by all four of its windows, and
/// `attn_relpos_dr` sums those four contributions under its `acc` flag. That is
/// a *partial* gradient error when broken, and `directional_check` contracts a
/// tensor onto one ±1 direction and keeps the BEST of four - the wrong
/// selection rule for exactly this failure. See the module header for the
/// mutation-verified numbers.
///
/// 224 entries total, so 448 extra forwards of a two-block, 47-row graph.
///
/// `eps = 5e-3`: a single-entry step has no `√numel` amplification, so the loss
/// difference is `eps·|∂L/∂wᵢ|` and fp32 cancellation bites well before
/// [`check_deepseekocr_relpos`]'s `5e-4`.
pub fn check_deepseekocr_relpos_elementwise(seed: u64) -> Report {
    let h = RelPosHarness::new(seed);
    let mut checks = Vec::new();
    for name in h.param_names().into_iter().filter(|n| n.contains("rel_pos")) {
        checks.extend(crate::elementwise_check(&h, &name, 5e-3).checks);
    }
    Report { checks }
}

/// The eps/error table behind [`check_deepseekocr_relpos`]'s `5e-4`, measured
/// rather than assumed - the repo's rule when a gradcheck fails is to PROBE
/// this and report it, never to widen the bound.
pub fn check_deepseekocr_relpos_eps_sweep(seed: u64) -> Vec<(f32, f32)> {
    let h = RelPosHarness::new(seed);
    [5e-3f32, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
        .iter()
        .map(|&eps| (eps, directional_check(&h, eps, 4, seed ^ 0x1234).max_rel()))
        .collect()
}

/// The eps table behind [`check_deepseekocr_relpos_elementwise`]'s `5e-3`.
pub fn check_deepseekocr_relpos_elementwise_eps_sweep(seed: u64) -> Vec<(f32, f32)> {
    let h = RelPosHarness::new(seed);
    [2e-2f32, 1e-2, 5e-3, 2e-3, 1e-3]
        .iter()
        .map(|&eps| {
            let mut worst = 0f32;
            for name in h.param_names().into_iter().filter(|n| n.contains("rel_pos")) {
                worst = worst.max(crate::elementwise_check(&h, &name, eps).max_rel());
            }
            (eps, worst)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// fp32 finite differences on a device: the workspace-standard combined
    /// tolerance.
    const ATOL: f32 = 4e-3;
    const RTOL: f32 = 8e-2;

    fn gate(report: Report, what: &str) {
        report.print();
        let fails = report.failures(ATOL, RTOL);
        assert!(
            fails.is_empty(),
            "{what} gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
        let dead = report.dead_gradients();
        assert!(dead.is_empty(), "{what}: exactly-zero analytic gradients for {:?}", dead.iter().map(|c| &c.param).collect::<Vec<_>>());
    }

    #[test]
    fn relpos_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        gate(check_deepseekocr_relpos(7), "SAM decomposed rel-pos bias");
    }

    /// The shared-table fold, per entry. See
    /// [`check_deepseekocr_relpos_elementwise`] for why the directional check
    /// is not enough.
    #[test]
    fn relpos_table_grads_are_the_sum_over_windows() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        gate(check_deepseekocr_relpos_elementwise(7), "SAM rel-pos tables (per entry)");
    }

    /// The eps probe, run as a gate: it asserts the chosen `5e-4` is not
    /// sitting on a knee, and prints the table if a future change moves it.
    #[test]
    fn relpos_eps_plateau() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let table = check_deepseekocr_relpos_eps_sweep(7);
        for (eps, rel) in &table {
            println!("  eps={eps:.1e}  max_rel={rel:.3e}");
        }
        let at = |e: f32| table.iter().find(|(x, _)| *x == e).expect("eps in table").1;
        assert!(at(5e-4) <= RTOL, "eps 5e-4 max_rel {:.3e} exceeds rtol", at(5e-4));
        assert!(at(5e-4) <= at(5e-3).max(at(1e-4)) * 4.0, "eps 5e-4 is not on the plateau: {table:?}");
    }
}
