// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The SAM-1 / ViTDet tower's forward and backward graph.
//!
//! Composed from the SHARED builders, never from private copies:
//!   * `model::vit` -- `WindowPlan::padded` / `WindowIndex` (the zero-padded
//!     window partition as a row permutation), `RelPosAxis` (the `get_rel_pos`
//!     resample+gather, itself composed from `embed`/`scale_row`/`add2`) and
//!     `RelPos` (the six `attn_relpos_*` dispatches);
//!   * `model::block` -- `chunked_bidir_fwd` / `chunked_bidir_bwd` with the
//!     rel-pos `Option` engaged, and the `LayerNormIds` seam that picks the
//!     coalesced `layernorm_rows` family;
//!   * `vision::blocks` -- `Conv` (patch embed, the two neck convs, the two
//!     compressor convs) and `LayerNorm2d` (channels-first, forward+backward).
//!
//! **It adds no kernel.** Everything below is a dispatch of something that was
//! already in `crates/kernels`.
//!
//! ## Why this block builder is not `model::vit::vit_block_fwd_cached`
//!
//! That builder's attention goes through `vit::cross_q_fwd` / `cross_q_bwd`,
//! which take no `RelPos` -- the rel-pos `Option` lives on
//! `block::chunked_bidir_{fwd,bwd}`, the *other* per-span attention path. SAM's
//! block therefore assembles the same pre-LN stages around that path instead.
//! Everything except the attention call itself is a dispatch of the same shared
//! kernels, in the same order, so the two cannot disagree about the block; they
//! disagree only about which attention builder they can reach. Giving
//! `cross_q_*` a rel-pos parameter (and a query-chunk loop, which it also lacks)
//! is the hoist that would collapse them, and it is a `model` change, not a
//! `sam1` one.
//!
//! ## Window padding -- where the pad happens, and why it is not negotiable
//!
//! `WindowPlan::padded` zero-pads the token grid bottom/right to a multiple of
//! the window, representing an out-of-grid position by the sentinel row
//! `rows()`. The pad is applied to the **post-`norm1`** activations, one step
//! BEFORE the qkv projection: a padded position's input is exactly zero, so its
//! key and value come out as the qkv **bias**, not zero, and it participates in
//! its window's softmax as a real extra key. Padding after the projection would
//! feed exact zeros and is a different model.
//!
//! Concretely: the `ln1` buffer carries `rows + 1` rows and is zero-cleared by
//! its own submit, `window_partition` gathers the sentinel into every pad slot
//! for free, and the projection runs over all `win_rows`. On the way back,
//! `window_reverse` reads only the real rows, so the pad's gradient is dropped
//! with no mask -- and the pad's *bias* gradient is not, because `bias_grad` runs
//! over the full `win_rows`.
//!
//! ## Training vs inference builds
//!
//! [`SamEncoder::new_on`]'s `train` flag decides two independent things, and
//! both matter at the tower's production 1024x1024 shape:
//!
//!  * **Parameter role.** `train` gives every tensor `Role::Trainable` (weight +
//!    gradient + the two AdamW moments, ~4x the raw parameter memory);
//!    `!train` gives every tensor `Role::Frozen`, which allocates the weight
//!    ONLY. At ViT-B that is ~1.4 GB vs ~0.36 GB.
//!  * **Backward scratch.** The SSA *forward* cache ([`Block`]'s `ln1`..`out`)
//!    is what the forward itself writes and is allocated either way. The
//!    backward's own scratch ([`BlockBwd`], [`SamBwd`]) is read by nothing but
//!    [`SamEncoder::backward`] / [`SamEncoder::block_bwd`], so an inference
//!    build does not allocate it. At production shape that is ~256 MB per
//!    windowed block and ~289 MB per global one - ~3.2 GB over the 12 blocks,
//!    the single largest line item in this tower.
//!
//! An inference build therefore has no `d_image`/`d_neck`/per-block adjoint
//! buffers; the accessors that hand those out say so rather than returning a
//! buffer that was never sized.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::block::{self, CrossBwdIds, CrossIds, LayerNormIds};
use model::vit::{
    scatter_rows, window_partition, window_reverse, RelPos, RelPosAxis, RelPosBwd, RelPosIds, RelPosTableIds,
    VitPermuteIds, WindowIndex, WindowPlan,
};
use paramstore::{ParamStore, Role};
use vision::{Act, Conv, ConvNames, ConvSpec, Ctx, LayerNorm2d, Ln2dNames, Norm, Shape};

use crate::config::SamViTConfig;

/// Kernels this crate dispatches, by name. Nothing here holds a positional
/// index: `vision::ConvKernelIds::resolve` and `Gpu::kernel_index` both key on
/// the name, so the order is irrelevant.
pub const PIPELINES: &[(&str, &str)] = &[
    // ---- conv half (patch embed, neck, compressor) ----
    ("conv2d", kernels::CONV2D),
    ("conv2d_dx", kernels::CONV2D_DX),
    ("conv2d_dw", kernels::CONV2D_DW),
    ("conv_bias", kernels::CONV_BIAS),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("nlc_nchw", kernels::NLC_NCHW),
    // ---- LayerNorm family (block norms + the two LayerNorm2d) ----
    ("layernorm", kernels::LAYERNORM),
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("ln_stats", kernels::LN_STATS),
    ("ln_stats_rows", kernels::LN_STATS_ROWS),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    ("layernorm_dx_rows", kernels::LAYERNORM_DX_ROWS),
    ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA),
    ("layernorm_dbeta", kernels::LAYERNORM_DBETA),
    // ---- linear algebra ----
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_rows", kernels::MATMUL_ROWS),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("bias_add", kernels::BIAS_ADD),
    ("bias_grad", kernels::BIAS_GRAD),
    ("add2", kernels::ADD2),
    ("axpy", kernels::AXPY),
    // SAM's ViT MLP is `nn.GELU`, i.e. the EXACT erf form -- NOT the tanh
    // approximation `gelu.wgsl` computes. A swap is gradcheck-invisible.
    ("gelu_erf", kernels::GELU_ERF),
    ("gelu_erf_bwd", kernels::GELU_ERF_BWD),
    // ---- window partition (a row permutation) ----
    ("embed", kernels::EMBED),
    ("row_scatter", kernels::ROW_SCATTER),
    ("emb_bwd", kernels::EMB_BWD),
    ("scale_row", kernels::SCALE_ROW),
    // ---- attention ----
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS),
    ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS),
    ("attn_bwd_dk_cross_acc", kernels::ATTN_BWD_DK_CROSS_ACC),
    ("attn_bwd_dv_cross_acc", kernels::ATTN_BWD_DV_CROSS_ACC),
    // ---- decomposed relative position bias ----
    ("attn_relpos_qr", kernels::ATTN_RELPOS_QR),
    ("attn_relpos_add", kernels::ATTN_RELPOS_ADD),
    ("attn_relpos_drh", kernels::ATTN_RELPOS_DRH),
    ("attn_relpos_drw", kernels::ATTN_RELPOS_DRW),
    ("attn_relpos_dq", kernels::ATTN_RELPOS_DQ),
    ("attn_relpos_dr", kernels::ATTN_RELPOS_DR),
    // ---- coalesced cross-attention scores ----
    // `attn_scores_cross` reads the fused KV slab with the KEY index as the
    // fastest thread index, so every lane of a warp lands on its own cache
    // line. SAM-1 global attention is the worst case for that: one span of
    // 4096 keys re-read once per query chunk. Transposing K to key-minor
    // once per span buys the same sweep coalesced loads.
    ("kv_k_headt", kernels::KV_K_HEADT),
    ("attn_scores_cross_kt", kernels::ATTN_SCORES_CROSS_KT),
];

/// Pipeline indices resolved by NAME.
struct Ids {
    ln: LayerNormIds,
    ln_dgamma: usize,
    ln_dbeta: usize,
    matmul: usize,
    matmul_reg3: usize,
    matmul_dx: usize,
    matmul_dw: usize,
    bias_add: usize,
    bias_grad: usize,
    add2: usize,
    axpy: usize,
    gelu: usize,
    gelu_bwd: usize,
    nlc_nchw: usize,
    nchw_nlc: usize,
    perm: VitPermuteIds,
    cross: CrossIds,
    /// The coalesced score path both the forward and the backward recompute
    /// dispatch through.
    key_minor: (usize, usize),
    cross_bwd: CrossBwdIds,
    rel: RelPosIds,
    tbl: RelPosTableIds,
}

impl Ids {
    fn new(g: &Gpu) -> Ids {
        let k = |n: &str| g.kernel_index(n).unwrap_or_else(|| panic!("sam1: kernel {n:?} not registered"));
        Ids {
            ln: LayerNormIds::resolve(g, k("layernorm"), k("ln_stats"), k("layernorm_dx")),
            ln_dgamma: k("layernorm_dgamma"),
            ln_dbeta: k("layernorm_dbeta"),
            matmul: k("matmul"),
            matmul_reg3: k("matmul_reg3"),
            matmul_dx: k("matmul_dx"),
            matmul_dw: k("matmul_dw"),
            bias_add: k("bias_add"),
            bias_grad: k("bias_grad"),
            add2: k("add2"),
            axpy: k("axpy"),
            gelu: k("gelu_erf"),
            gelu_bwd: k("gelu_erf_bwd"),
            nlc_nchw: k("nlc_nchw"),
            nchw_nlc: k("nchw_nlc"),
            perm: VitPermuteIds { embed: k("embed"), row_scatter: k("row_scatter") },
            cross: CrossIds { scores: k("attn_scores_cross"), softmax: k("attn_softmax_cross"), apply: k("attn_apply_cross") },
            key_minor: (k("kv_k_headt"), k("attn_scores_cross_kt")),
            cross_bwd: CrossBwdIds {
                dscores: k("attn_bwd_dscores_cross"),
                dq: k("attn_bwd_dq_cross"),
                dk_acc: k("attn_bwd_dk_cross_acc"),
                dv_acc: k("attn_bwd_dv_cross_acc"),
            },
            rel: RelPosIds {
                qr: k("attn_relpos_qr"),
                add: k("attn_relpos_add"),
                drh: k("attn_relpos_drh"),
                drw: k("attn_relpos_drw"),
                dq: k("attn_relpos_dq"),
                dr: k("attn_relpos_dr"),
            },
            tbl: RelPosTableIds {
                embed: k("embed"),
                scale_row: k("scale_row"),
                add2: k("add2"),
                nlc_nchw: k("nlc_nchw"),
                emb_bwd: k("emb_bwd"),
            },
        }
    }
}

/// One block's backward scratch -- every buffer written or read ONLY by
/// [`SamEncoder::block_bwd`]. Absent from an inference build; see this module's
/// header for what that saves.
struct BlockBwd {
    d_rel_h: DeviceBuffer,
    d_rel_w: DeviceBuffer,
    d_rh: DeviceBuffer,
    d_rw: DeviceBuffer,
    tbl_scratch: DeviceBuffer,
    d_scores: DeviceBuffer,
    d_res: DeviceBuffer,
    d_ln: DeviceBuffer,
    tmp: DeviceBuffer,
    d_h: DeviceBuffer,
    d_h2: DeviceBuffer,
    d_proj: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_qkv: DeviceBuffer,
    d_wm: DeviceBuffer,
    d_ln1: DeviceBuffer,
    mean: DeviceBuffer,
    inv: DeviceBuffer,
    d_x: DeviceBuffer,
}

/// One block's geometry, relative-position machinery and SSA activation cache.
///
/// Every buffer is allocated once and reused by every forward: a gradient check
/// runs hundreds of forwards, and per-forward allocation would dominate it.
struct Block {
    l: u32,
    /// `(qh, qw)` of one attention span -- the window, or the whole grid.
    qh: u32,
    qw: u32,
    /// Window-major rows the attention runs over. `rows` for a global block,
    /// `pad_h * pad_w` for a windowed one.
    attn_rows: u32,
    spans: Vec<(u32, u32)>,
    /// `None` for a global block.
    win: Option<WindowIndex>,

    ax_h: RelPosAxis,
    ax_w: RelPosAxis,
    rel_h: DeviceBuffer,
    rel_w: DeviceBuffer,

    // ---- forward cache (SSA) ----
    /// `[rows + 1, C]`. The extra row is `WindowPlan::padded`'s sentinel and is
    /// zero-cleared by every forward's own submit.
    ln1: DeviceBuffer,
    /// `[attn_rows, C]` -- window-major `ln1`; absent for a global block.
    wm: Option<DeviceBuffer>,
    qkv: DeviceBuffer,
    ctx: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    /// `[c, max_span]` key-minor K for the coalesced score path.
    kt: DeviceBuffer,
    proj: DeviceBuffer,
    attn_out: DeviceBuffer,
    res: DeviceBuffer,
    ln2: DeviceBuffer,
    h: DeviceBuffer,
    h2: DeviceBuffer,
    mlp_out: DeviceBuffer,
    out: DeviceBuffer,

    /// Backward scratch -- `None` for an inference build.
    bwd: Option<BlockBwd>,
}

impl Block {
    fn new(g: &Gpu, cfg: &SamViTConfig, l: u32, train: bool) -> Block {
        let (c, rows, hd) = (cfg.d_model, cfg.rows(), cfg.head_dim());
        let (qh, qw) = cfg.attn_extent(l);
        let (rh_rows, rw_rows) = cfg.rel_pos_rows(l);
        let (attn_rows, spans, win) = if cfg.is_global(l) {
            (rows, vec![(0u32, rows)], None)
        } else {
            let plan = WindowPlan::padded(cfg.grid_h, cfg.grid_w, cfg.window_h, cfg.window_w);
            assert!(
                plan.ctx_bindable(c),
                "block {l}: a {}x{} window over a {}x{} grid at width {c} produces a span whose \
                 ctx binding offset is not 256 B aligned",
                cfg.window_h,
                cfg.window_w,
                cfg.grid_h,
                cfg.grid_w
            );
            (plan.win_rows(), plan.spans().to_vec(), Some(WindowIndex::new(g, &plan)))
        };
        let span_qn = qh * qw;
        assert_eq!(span_qn, spans[0].1, "block {l}: rel-pos extent {qh}x{qw} != span length {}", spans[0].1);
        let slab = cfg.n_heads as u64 * cfg.attn_chunk.min(span_qn) as u64 * span_qn as u64;
        let ar = attn_rows as u64;
        let rc = rows as u64 * c as u64;
        let m = rows as u64 * cfg.ffn_hidden as u64;
        Block {
            l,
            qh,
            qw,
            attn_rows,
            spans,
            ax_h: RelPosAxis::new(g, qh, qh, hd, rh_rows),
            ax_w: RelPosAxis::new(g, qw, qw, hd, rw_rows),
            rel_h: g.storage((cfg.n_heads * span_qn * qh) as u64),
            rel_w: g.storage((cfg.n_heads * span_qn * qw) as u64),
            bwd: train.then(|| BlockBwd {
                d_rel_h: g.storage((cfg.n_heads * span_qn * qh) as u64),
                d_rel_w: g.storage((cfg.n_heads * span_qn * qw) as u64),
                d_rh: g.storage((qh * qh * hd) as u64),
                d_rw: g.storage((qw * qw * hd) as u64),
                tbl_scratch: g.storage(((qh * qh).max(qw * qw) * hd) as u64),
                d_scores: g.storage(slab),
                d_res: g.storage(rc),
                d_ln: g.storage(rc),
                tmp: g.storage(rc),
                d_h: g.storage(m),
                d_h2: g.storage(m),
                d_proj: g.storage(ar * c as u64),
                d_ctx: g.storage(ar * c as u64),
                d_qkv: g.storage(ar * 3 * c as u64),
                d_wm: g.storage(ar * c as u64),
                d_ln1: g.storage(rc),
                mean: g.storage(rows as u64),
                inv: g.storage(rows as u64),
                d_x: g.storage(rc),
            }),
            ln1: g.storage((rows as u64 + 1) * c as u64),
            wm: win.is_some().then(|| g.storage(ar * c as u64)),
            qkv: g.storage(ar * 3 * c as u64),
            ctx: g.storage(ar * c as u64),
            scores: g.storage(slab),
            probs: g.storage(slab),
            kt: g.storage(c as u64 * span_qn as u64),
            proj: g.storage(ar * c as u64),
            attn_out: g.storage(rc),
            res: g.storage(rc),
            ln2: g.storage(rc),
            h: g.storage(m),
            h2: g.storage(m),
            mlp_out: g.storage(rc),
            out: g.storage(rc),
            win,
        }
    }

    /// The backward scratch, or a panic naming the build that lacks it.
    fn bwd(&self) -> &BlockBwd {
        self.bwd.as_ref().expect("sam1: this is an inference build (train = false); it has no backward scratch")
    }

    fn p(&self, leaf: &str) -> String {
        format!("vision.sam.blocks.{}.{leaf}", self.l)
    }

    /// The block's tensor names, in manifest order -- used to prove this
    /// crate's graph and its config agree about what exists.
    fn param_names(&self) -> Vec<String> {
        [
            "norm1.weight",
            "norm1.bias",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "attn.rel_pos_h",
            "attn.rel_pos_w",
            "norm2.weight",
            "norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ]
        .iter()
        .map(|leaf| self.p(leaf))
        .collect()
    }

    /// This block's coalesced score path, bound to its own `kt` scratch.
    fn key_minor<'a>(&'a self, ids: &Ids) -> block::KeyMinor<'a> {
        block::KeyMinor { transpose: ids.key_minor.0, scores: ids.key_minor.1, kt: &self.kt }
    }

    fn relpos(&self, ids: &Ids, bwd: bool) -> RelPos<'_> {
        RelPos {
            ids: ids.rel,
            qh: self.qh,
            qw: self.qw,
            kh: self.qh,
            kw: self.qw,
            rh_t: &self.ax_h.r_t,
            rw_t: &self.ax_w.r_t,
            rel_h: &self.rel_h,
            rel_w: &self.rel_w,
            bwd: bwd.then(|| RelPosBwd {
                rh: &self.ax_h.r,
                rw: &self.ax_w.r,
                d_rh: &self.bwd().d_rh,
                d_rw: &self.bwd().d_rw,
                d_rel_h: &self.bwd().d_rel_h,
                d_rel_w: &self.bwd().d_rel_w,
                // The FIRST span assigns; every later window of the same block
                // accumulates onto it, which `chunked_bidir_bwd` handles.
                acc0: false,
            }),
        }
    }

    /// `[table_rows, head_dim]` -> the dense `[q_ext, k_ext, head_dim]` tables.
    /// Re-run every pass: the learned tables are leaf parameters and a
    /// gradient check moves them between forwards.
    fn build_tables(&self, g: &Gpu, ids: &Ids, ps: &ParamStore, steps: &mut Vec<Step>) {
        self.ax_h.build_fwd(g, &ids.tbl, ps.w(&self.p("attn.rel_pos_h")), steps);
        self.ax_w.build_fwd(g, &ids.tbl, ps.w(&self.p("attn.rel_pos_w")), steps);
    }
}

/// The tower-level backward scratch: the neck/compressor/patch-embed adjoints
/// and the image gradient. Like [`BlockBwd`], nothing but
/// [`SamEncoder::backward`] touches it, so an inference build omits it.
struct SamBwd {
    /// `[1, 3, image_h, image_w]`.
    d_image: DeviceBuffer,
    /// `[rows, C]` -- the neck's gradient w.r.t. the last block's output. Its own
    /// buffer, NOT that block's `d_x`, so no block ever reads and writes one
    /// buffer in the same submit.
    d_feats: DeviceBuffer,
    d_top: DeviceBuffer,
    d_embed_nchw: DeviceBuffer,
    d_neck: Vec<DeviceBuffer>,
}

/// The SAM-1 tower: image in, `[1, compress_out, grid_h/4, grid_w/4]` out, plus
/// the full analytic backward of a scalar objective on that output.
pub struct SamEncoder {
    pub gpu: Gpu,
    pub cfg: SamViTConfig,
    pub ps: ParamStore,
    ids: Ids,
    conv_ids: vision::ConvKernelIds,

    patch: Conv,
    neck_c1: Conv,
    neck_n1: LayerNorm2d,
    neck_c2: Conv,
    neck_n2: LayerNorm2d,
    comp_c1: Conv,
    comp_c2: Conv,
    blocks: Vec<Block>,

    /// `[1, 3, image_h, image_w]` -- the fixed input.
    image: DeviceBuffer,
    /// `[rows, C]` NLC patch tokens, and the same plus `pos_embed`.
    patch_nlc: DeviceBuffer,
    embed: DeviceBuffer,
    /// `[1, C, grid_h, grid_w]` -- the last block's output, back in NCHW.
    feats: DeviceBuffer,

    /// Fixed unit-scale direction defining the scalar objective.
    dir: Vec<f32>,
    d_out: DeviceBuffer,
    fwd_done: Cell<bool>,
    /// Tower-level backward scratch -- `None` for an inference build.
    bwd: Option<SamBwd>,
}

impl SamEncoder {
    /// Trainable build from an eager map. Exactly
    /// [`Self::new_on`]`(gpu, cfg, init, seed, true)` - the signature and the
    /// behaviour every existing caller already has.
    pub fn new(gpu: Gpu, cfg: SamViTConfig, init: &HashMap<String, Vec<f32>>, seed: u64) -> SamEncoder {
        SamEncoder::new_on(gpu, cfg, init, seed, true)
    }

    /// Build the graph and upload the weights, streaming.
    ///
    /// `src` must name every tensor of [`SamViTConfig::param_list`]. It is a
    /// [`checkpoint::TensorSource`], so an eager `&HashMap<String, Vec<f32>>`
    /// coerces (that is what [`Self::new`] passes) **and** an mmap-backed
    /// `WeightReader` / `MmapGguf` works with peak host allocation of one
    /// tensor - the whole model is never a second host copy.
    ///
    /// `train` picks the parameter role and whether the backward scratch is
    /// allocated at all; see this module's header for the two costs it decides.
    /// Shaped after `DeepseekV2::new_on`, deliberately: one `_on` builder
    /// carrying both, with named wrappers over it.
    pub fn new_on(
        gpu: Gpu,
        cfg: SamViTConfig,
        src: &dyn checkpoint::TensorSource,
        seed: u64,
        train: bool,
    ) -> SamEncoder {
        cfg.check_bindable();
        let ids = Ids::new(&gpu);
        let conv_ids = vision::ConvKernelIds::resolve(PIPELINES);
        let role = if train { Role::Trainable } else { Role::Frozen };
        let roles: Vec<(String, usize, Role)> =
            cfg.param_list().into_iter().map(|(n, numel)| (n, numel, role)).collect();
        let ps = ParamStore::new_with_roles_src(&gpu, roles, src);
        let (c, rows) = (cfg.d_model, cfg.rows());

        let ctx = Ctx::new(&gpu, &conv_ids);
        let img = Shape::new(1, 3, cfg.image_h(), cfg.image_w());
        let raw = |cout: u32, k: u32, stride: u32, pad: u32| ConvSpec {
            cout,
            k,
            stride,
            pad,
            groups: 1,
            dilation: 1,
            norm: Norm::None,
            act: Act::None,
            bias: false,
        };
        let patch = Conv::with_names(
            &ctx,
            "vision.sam.patch_embed",
            ConvNames::torch_flat("vision.sam.patch_embed"),
            img,
            raw(c, cfg.patch_size, cfg.patch_size, 0).with_bias(),
            false,
        );
        assert_eq!(patch.out_shape, Shape::new(1, c, cfg.grid_h, cfg.grid_w), "patch embed must produce the config's grid");

        let grid = Shape::new(1, c, cfg.grid_h, cfg.grid_w);
        let neck_c1 = Conv::with_names(
            &ctx,
            "vision.sam.neck.conv1",
            ConvNames::torch_flat("vision.sam.neck.conv1"),
            grid,
            raw(cfg.neck_channels, 1, 1, 0),
            false,
        );
        let neck_n1 = LayerNorm2d::new(&ctx, Ln2dNames::torch("vision.sam.neck.norm1"), neck_c1.out_shape, cfg.eps);
        let neck_c2 = Conv::with_names(
            &ctx,
            "vision.sam.neck.conv2",
            ConvNames::torch_flat("vision.sam.neck.conv2"),
            neck_c1.out_shape,
            raw(cfg.neck_channels, 3, 1, 1),
            false,
        );
        let neck_n2 = LayerNorm2d::new(&ctx, Ln2dNames::torch("vision.sam.neck.norm2"), neck_c2.out_shape, cfg.eps);
        let comp_c1 = Conv::with_names(
            &ctx,
            "vision.sam.compress.conv1",
            ConvNames::torch_flat("vision.sam.compress.conv1"),
            neck_c2.out_shape,
            raw(cfg.compress_mid, 3, 2, 1),
            false,
        );
        let comp_c2 = Conv::with_names(
            &ctx,
            "vision.sam.compress.conv2",
            ConvNames::torch_flat("vision.sam.compress.conv2"),
            comp_c1.out_shape,
            raw(cfg.compress_out, 3, 2, 1),
            false,
        );
        let (ch, cw) = cfg.compress_grid();
        assert_eq!(comp_c2.out_shape, Shape::new(1, cfg.compress_out, ch, cw), "compressor output disagrees with the config");

        let blocks: Vec<Block> = (0..cfg.n_layers).map(|l| Block::new(&gpu, &cfg, l, train)).collect();

        // Two-way coverage between the graph and the manifest: every tensor the
        // graph reads is declared, and every declared tensor is read. Without
        // this a renamed leaf is a silently frozen parameter, not an error.
        let mut used: Vec<String> = vec!["vision.sam.pos_embed".to_string()];
        for cv in [&patch, &neck_c1, &neck_c2, &comp_c1, &comp_c2] {
            used.extend(cv.param_list().into_iter().map(|(n, _)| n));
        }
        for ln in [&neck_n1, &neck_n2] {
            used.extend(ln.param_list().into_iter().map(|(n, _)| n));
        }
        for b in &blocks {
            used.extend(b.param_names());
        }
        let declared: HashSet<String> = cfg.param_list().into_iter().map(|(n, _)| n).collect();
        let used_set: HashSet<String> = used.iter().cloned().collect();
        assert_eq!(used.len(), used_set.len(), "sam1: a tensor is claimed by two stages");
        let missing: Vec<&String> = declared.difference(&used_set).collect();
        let extra: Vec<&String> = used_set.difference(&declared).collect();
        assert!(missing.is_empty() && extra.is_empty(), "sam1 manifest/graph mismatch: unread {missing:?}, undeclared {extra:?}");

        let mut rng = data::rng::Rng::new(seed ^ 0x5A11);
        let image: Vec<f32> = (0..img.numel() as usize).map(|_| rng.next_f32() - 0.5).collect();
        let dir: Vec<f32> = (0..comp_c2.out_shape.numel() as usize).map(|_| rng.next_f32() - 0.5).collect();

        let bwd = train.then(|| SamBwd {
            d_image: gpu.storage(img.numel() as u64),
            d_feats: gpu.storage(rows as u64 * c as u64),
            d_top: gpu.storage(rows as u64 * c as u64),
            d_embed_nchw: gpu.storage(rows as u64 * c as u64),
            d_neck: vec![
                gpu.storage(comp_c1.out_shape.numel() as u64),
                gpu.storage(neck_n2.shape.numel() as u64),
                gpu.storage(neck_c2.out_shape.numel() as u64),
                gpu.storage(neck_n1.shape.numel() as u64),
                gpu.storage(neck_c1.out_shape.numel() as u64),
            ],
        });
        SamEncoder {
            image: gpu.storage_init("sam1_image", &image),
            patch_nlc: gpu.storage(rows as u64 * c as u64),
            embed: gpu.storage(rows as u64 * c as u64),
            feats: gpu.storage(rows as u64 * c as u64),
            d_out: gpu.storage_init("sam1_dir", &dir),
            bwd,
            dir,
            fwd_done: Cell::new(false),
            gpu,
            cfg,
            ps,
            ids,
            conv_ids,
            patch,
            neck_c1,
            neck_n1,
            neck_c2,
            neck_n2,
            comp_c1,
            comp_c2,
            blocks,
        }
    }

    /// Frozen (forward-only) build: every parameter `Role::Frozen`, no backward
    /// scratch. The forward graph is bit-identically the one [`Self::new`]
    /// records - only what is *allocated* differs.
    pub fn new_inference(gpu: Gpu, cfg: SamViTConfig, src: &dyn checkpoint::TensorSource, seed: u64) -> SamEncoder {
        SamEncoder::new_on(gpu, cfg, src, seed, false)
    }

    /// Convenience: a fresh trainable encoder on `gpu` with deterministic dense
    /// weights.
    pub fn with_dense_init(gpu: Gpu, cfg: SamViTConfig, seed: u64) -> SamEncoder {
        let init = crate::init::init_dense(&cfg, seed);
        SamEncoder::new(gpu, cfg, &init, seed)
    }

    /// Whether this build can run [`Self::backward`] (i.e. was built with
    /// `train = true`).
    pub fn is_trainable(&self) -> bool {
        self.bwd.is_some()
    }

    fn bwd(&self) -> &SamBwd {
        self.bwd.as_ref().expect(
            "sam1: this is an inference build (train = false); it allocated no gradient buffers. \
             Build with SamEncoder::new / new_on(.., train = true) to run the backward.",
        )
    }

    fn ctx(&self) -> Ctx<'_> {
        Ctx::new(&self.gpu, &self.conv_ids)
    }

    /// Forward-GEMM kernel + dispatch threads for a `[m,k]x[k,n]` linear,
    /// picked on the output dims -- `block::pick_gemm`'s measured crossover
    /// (`m >= 8 && n >= 128` takes the 128x128 register-tiled kernel), the same
    /// rule `crates/clip` and `crates/deepseekv2` already apply to their own
    /// GEMMs. Every block's QKV/proj/fc1/fc2 linear in this tower runs at
    /// `m` in the hundreds to thousands (windowed OR global -- both are far
    /// above the crossover), so this was previously ALWAYS the one-thread-
    /// per-output-element naive kernel: the register-tiled sibling
    /// (`matmul_reg3`) was registered by every other model in this crate's own
    /// composite (`crates/clip`, `crates/deepseekv2`) but never wired in here.
    fn gemm(&self, m: u32, n: u32) -> (usize, u32) {
        block::pick_gemm(m as usize, n as usize, self.ids.matmul, self.ids.matmul_reg3, false)
    }

    /// `[rows, C]` NLC -> `[1, C, grid_h, grid_w]` NCHW (and back).
    fn perm_params(&self) -> [u32; 3] {
        let (c, rows) = (self.cfg.d_model, self.cfg.rows());
        [rows * c, c, rows]
    }

    /// The compressor output -- valid after [`Self::forward`].
    pub fn output(&self) -> &DeviceBuffer {
        self.comp_c2.out()
    }

    // -----------------------------------------------------------------------
    // forward
    // -----------------------------------------------------------------------

    /// Run the whole tower on the fixed image and return the scalar objective
    /// `<output, dir>` -- the quantity [`Self::backward`] differentiates.
    pub fn forward(&self) -> f32 {
        let g = &self.gpu;
        let ctx = self.ctx();
        let (c, rows) = (self.cfg.d_model, self.cfg.rows());

        // ---- patch embed + learned absolute position ----
        self.patch.forward(&ctx, &self.ps, &self.image);
        let mut steps = vec![
            g.step(self.ids.nchw_nlc, &[self.patch.out(), &self.patch_nlc], &self.perm_params(), rows * c),
            g.step(self.ids.add2, &[&self.patch_nlc, self.ps.w("vision.sam.pos_embed"), &self.embed], &[rows * c], rows * c),
        ];
        g.submit(&[], &steps);

        // ---- blocks ----
        for i in 0..self.blocks.len() {
            let x = if i == 0 { &self.embed } else { &self.blocks[i - 1].out };
            self.block_fwd(&self.blocks[i], x);
        }

        // ---- NLC -> NCHW, neck, compressor ----
        steps = vec![g.step(self.ids.nlc_nchw, &[&self.blocks[self.blocks.len() - 1].out, &self.feats], &self.perm_params(), rows * c)];
        g.submit(&[], &steps);
        self.neck_c1.forward(&ctx, &self.ps, &self.feats);
        self.neck_n1.forward(&ctx, &self.ps, self.neck_c1.out());
        self.neck_c2.forward(&ctx, &self.ps, self.neck_n1.out());
        self.neck_n2.forward(&ctx, &self.ps, self.neck_c2.out());
        self.comp_c1.forward(&ctx, &self.ps, self.neck_n2.out());
        self.comp_c2.forward(&ctx, &self.ps, self.comp_c1.out());

        self.fwd_done.set(true);
        // f64 accumulation: an element-wise check differences a loss that moves
        // by ~1e-3 of itself, so an f32 accumulator's round-off would land
        // straight in the numerator.
        let out = g.read(self.output(), self.dir.len());
        out.iter().zip(&self.dir).map(|(y, r)| *y as f64 * *r as f64).sum::<f64>() as f32
    }

    fn block_fwd(&self, b: &Block, x: &DeviceBuffer) {
        let g = &self.gpu;
        let cfg = &self.cfg;
        let (c, rows, m) = (cfg.d_model, cfg.rows(), cfg.ffn_hidden);
        let ar = b.attn_rows;
        let w = |leaf: &str| self.ps.w(&b.p(leaf));
        let mut steps: Vec<Step> = Vec::new();

        b.build_tables(g, &self.ids, &self.ps, &mut steps);
        steps.push(block::layernorm_fwd(g, &self.ids.ln, x, w("norm1.weight"), w("norm1.bias"), &b.ln1, c, rows, cfg.eps));

        // Zero-padded window partition. The sentinel row of `ln1` is zero
        // because the submit below clears the whole buffer first, so a pad
        // slot's qkv comes out as the qkv BIAS -- see this module's header.
        let attn_in: &DeviceBuffer = match (&b.win, &b.wm) {
            (Some(wi), Some(wm)) => {
                steps.push(window_partition(g, &self.ids.perm, wi, &b.ln1, wm, c));
                wm
            }
            _ => &b.ln1,
        };
        let (mk, mt) = self.gemm(ar, 3 * c);
        steps.push(g.step(mk, &[attn_in, w("attn.qkv.weight"), &b.qkv], &[ar, c, 3 * c], mt));
        steps.push(g.step(self.ids.bias_add, &[&b.qkv, w("attn.qkv.bias")], &[ar, 3 * c], ar * 3 * c));

        let rel = b.relpos(&self.ids, false);
        let km = b.key_minor(&self.ids);
        block::chunked_bidir_fwd(
            g, &self.ids.cross, Some(&km), cfg.n_heads, cfg.head_dim(), c, &b.qkv, 3 * c, 0, c, 2 * c, &b.ctx,
            &b.scores, &b.probs, &b.spans, cfg.attn_chunk, Some(&rel), &mut steps,
        );

        let (mk, mt) = self.gemm(ar, c);
        steps.push(g.step(mk, &[&b.ctx, w("attn.proj.weight"), &b.proj], &[ar, c, c], mt));
        steps.push(g.step(self.ids.bias_add, &[&b.proj, w("attn.proj.bias")], &[ar, c], ar * c));
        let branch: &DeviceBuffer = match &b.win {
            Some(wi) => {
                steps.push(window_reverse(g, &self.ids.perm, wi, &b.proj, &b.attn_out, c));
                &b.attn_out
            }
            None => &b.proj,
        };
        steps.push(g.step(self.ids.add2, &[x, branch, &b.res], &[rows * c], rows * c));

        steps.push(block::layernorm_fwd(g, &self.ids.ln, &b.res, w("norm2.weight"), w("norm2.bias"), &b.ln2, c, rows, cfg.eps));
        let (mk, mt) = self.gemm(rows, m);
        steps.push(g.step(mk, &[&b.ln2, w("mlp.fc1.weight"), &b.h], &[rows, c, m], mt));
        steps.push(g.step(self.ids.bias_add, &[&b.h, w("mlp.fc1.bias")], &[rows, m], rows * m));
        steps.push(g.step(self.ids.gelu, &[&b.h, &b.h2], &[rows * m], rows * m));
        let (mk, mt) = self.gemm(rows, c);
        steps.push(g.step(mk, &[&b.h2, w("mlp.fc2.weight"), &b.mlp_out], &[rows, m, c], mt));
        steps.push(g.step(self.ids.bias_add, &[&b.mlp_out, w("mlp.fc2.bias")], &[rows, c], rows * c));
        steps.push(g.step(self.ids.add2, &[&b.res, &b.mlp_out, &b.out], &[rows * c], rows * c));
        g.submit(&[&b.ln1], &steps);
    }

    // -----------------------------------------------------------------------
    // backward
    // -----------------------------------------------------------------------

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    /// Analytic gradients of `<output, dir>` into the ParamStore.
    ///
    /// `d_out` is seeded with `dir` directly: the objective is exactly linear
    /// in the compressor output.
    pub fn backward(&self) {
        if !self.fwd_done.get() {
            let _ = self.forward();
        }
        let bw = self.bwd();
        let g = &self.gpu;
        let ctx = self.ctx();
        let (c, rows) = (self.cfg.d_model, self.cfg.rows());

        self.comp_c2.backward(&ctx, &self.ps, self.comp_c1.out(), &self.d_out, &bw.d_neck[0]);
        self.comp_c1.backward(&ctx, &self.ps, self.neck_n2.out(), &bw.d_neck[0], &bw.d_neck[1]);
        self.neck_n2.backward(&ctx, &self.ps, &bw.d_neck[1], &bw.d_neck[2]);
        self.neck_c2.backward(&ctx, &self.ps, self.neck_n1.out(), &bw.d_neck[2], &bw.d_neck[3]);
        self.neck_n1.backward(&ctx, &self.ps, &bw.d_neck[3], &bw.d_neck[4]);
        self.neck_c1.backward(&ctx, &self.ps, &self.feats, &bw.d_neck[4], &bw.d_feats);

        let last = self.blocks.len() - 1;
        g.submit(&[], &[g.step(self.ids.nchw_nlc, &[&bw.d_feats, &bw.d_top], &self.perm_params(), rows * c)]);
        for i in (0..self.blocks.len()).rev() {
            let x = if i == 0 { &self.embed } else { &self.blocks[i - 1].out };
            let d_out: &DeviceBuffer = if i == last { &bw.d_top } else { &self.blocks[i + 1].bwd().d_x };
            self.block_bwd(&self.blocks[i], x, d_out);
        }

        // pos_embed and the patch tokens share the same adjoint (a plain sum).
        let d_embed = &self.blocks[0].bwd().d_x;
        let steps = vec![
            g.step(self.ids.axpy, &[self.ps.g("vision.sam.pos_embed"), d_embed], &[rows * c, f(1.0)], rows * c),
            g.step(self.ids.nlc_nchw, &[d_embed, &bw.d_embed_nchw], &self.perm_params(), rows * c),
        ];
        g.submit(&[], &steps);
        self.patch.backward(&ctx, &self.ps, &self.image, &bw.d_embed_nchw, &bw.d_image);
        g.poll_wait();
    }

    /// One block's adjoint: upstream `d_out` `[rows, C]` -> `b.d_x`, parameter
    /// gradients accumulated into the ParamStore.
    fn block_bwd(&self, b: &Block, x: &DeviceBuffer, d_out: &DeviceBuffer) {
        let g = &self.gpu;
        let cfg = &self.cfg;
        let (c, rows, m) = (cfg.d_model, cfg.rows(), cfg.ffn_hidden);
        let (heads, hd) = (cfg.n_heads, cfg.head_dim());
        let ar = b.attn_rows;
        let bb = b.bwd();
        let w = |leaf: &str| self.ps.w(&b.p(leaf));
        let gr = |leaf: &str| self.ps.g(&b.p(leaf));
        let mut steps: Vec<Step> = Vec::new();

        // Rebuild the dense tables so the backward never depends on whichever
        // forward happened to run last.
        b.build_tables(g, &self.ids, &self.ps, &mut steps);

        // ---- MLP half ----
        steps.push(g.step(self.ids.matmul_dx, &[d_out, w("mlp.fc2.weight"), &bb.d_h2], &[rows, m, c, 0], rows * m));
        steps.push(g.step(self.ids.matmul_dw, &[d_out, &b.h2, gr("mlp.fc2.weight")], &[rows, m, c], c * m));
        steps.push(g.step(self.ids.bias_grad, &[d_out, gr("mlp.fc2.bias")], &[rows, c], c));
        steps.push(g.step(self.ids.gelu_bwd, &[&b.h, &bb.d_h2, &bb.d_h], &[rows * m], rows * m));
        steps.push(g.step(self.ids.matmul_dx, &[&bb.d_h, w("mlp.fc1.weight"), &bb.d_ln], &[rows, c, m, 0], rows * c));
        steps.push(g.step(self.ids.matmul_dw, &[&bb.d_h, &b.ln2, gr("mlp.fc1.weight")], &[rows, c, m], m * c));
        steps.push(g.step(self.ids.bias_grad, &[&bb.d_h, gr("mlp.fc1.bias")], &[rows, m], m));
        steps.push(block::ln_stats_fwd(g, &self.ids.ln, &b.res, &bb.mean, &bb.inv, c, rows, cfg.eps));
        steps.push(g.step(self.ids.ln_dgamma, &[&bb.d_ln, &b.res, &bb.mean, &bb.inv, gr("norm2.weight")], &[c, rows], c));
        steps.push(g.step(self.ids.ln_dbeta, &[&bb.d_ln, gr("norm2.bias")], &[c, rows], c));
        steps.push(block::layernorm_dx_bwd(g, &self.ids.ln, &b.res, w("norm2.weight"), &bb.d_ln, &bb.tmp, c, rows, cfg.eps));
        steps.push(g.step(self.ids.add2, &[d_out, &bb.tmp, &bb.d_res], &[rows * c], rows * c));

        // ---- attention half (upstream d_res) ----
        // Adjoint of `window_reverse`: scatter the real rows back into their
        // window-major slots and leave the pad at ZERO (the pad's projection
        // output reaches no loss). `d_proj` is in the clear list for that.
        let d_branch: &DeviceBuffer = match &b.win {
            Some(wi) => {
                steps.push(scatter_rows(g, &self.ids.perm, &wi.inv, &bb.d_res, &bb.d_proj, rows, c, ar));
                &bb.d_proj
            }
            None => &bb.d_res,
        };
        steps.push(g.step(self.ids.matmul_dx, &[d_branch, w("attn.proj.weight"), &bb.d_ctx], &[ar, c, c, 0], ar * c));
        steps.push(g.step(self.ids.matmul_dw, &[d_branch, &b.ctx, gr("attn.proj.weight")], &[ar, c, c], c * c));
        steps.push(g.step(self.ids.bias_grad, &[d_branch, gr("attn.proj.bias")], &[ar, c], c));

        let rel = b.relpos(&self.ids, true);
        let km = b.key_minor(&self.ids);
        block::chunked_bidir_bwd(
            g, &self.ids.cross, Some(&km), &self.ids.cross_bwd, heads, hd, c, &b.qkv, 3 * c, 0, c, 2 * c, &bb.d_ctx,
            &bb.d_qkv, &b.scores, &b.probs, &bb.d_scores, &b.spans, cfg.attn_chunk, Some(&rel), &mut steps,
        );
        // Dense-table adjoint -> learned-table adjoint. `emb_bwd` ACCUMULATES,
        // which is what sums the two interpolation taps; the destination is a
        // ParamStore gradient and is cleared by `zero_grads`.
        b.ax_h.build_bwd(g, &self.ids.tbl, &bb.d_rh, gr("attn.rel_pos_h"), &bb.tbl_scratch, &mut steps);
        b.ax_w.build_bwd(g, &self.ids.tbl, &bb.d_rw, gr("attn.rel_pos_w"), &bb.tbl_scratch, &mut steps);

        let attn_in: &DeviceBuffer = b.wm.as_ref().unwrap_or(&b.ln1);
        steps.push(g.step(self.ids.matmul_dx, &[&bb.d_qkv, w("attn.qkv.weight"), &bb.d_wm], &[ar, c, 3 * c, 0], ar * c));
        steps.push(g.step(self.ids.matmul_dw, &[&bb.d_qkv, attn_in, gr("attn.qkv.weight")], &[ar, c, 3 * c], 3 * c * c));
        // `attn_rows`, NOT `rows`: for a windowed block the padded positions are
        // real keys/values whose input is zero, so they contribute the qkv BIAS
        // to every window's softmax and their gradient belongs in this sum.
        // Summing over `rows` instead is a ~6% error that the DIRECTIONAL
        // gradient check does not catch (measured: rel 6.28e-2, a pass) -- see
        // this crate's `windowed_pad_rows_contribute_to_the_qkv_bias_gradient`.
        steps.push(g.step(self.ids.bias_grad, &[&bb.d_qkv, gr("attn.qkv.bias")], &[ar, 3 * c], 3 * c));

        let d_ln1: &DeviceBuffer = match &b.win {
            Some(wi) => {
                // The adjoint of the padded gather, restricted to the real rows
                // -- which is exactly what `window_reverse` computes, so the pad's
                // gradient is dropped with no mask.
                steps.push(window_reverse(g, &self.ids.perm, wi, &bb.d_wm, &bb.d_ln1, c));
                &bb.d_ln1
            }
            None => &bb.d_wm,
        };
        steps.push(block::ln_stats_fwd(g, &self.ids.ln, x, &bb.mean, &bb.inv, c, rows, cfg.eps));
        steps.push(g.step(self.ids.ln_dgamma, &[d_ln1, x, &bb.mean, &bb.inv, gr("norm1.weight")], &[c, rows], c));
        steps.push(g.step(self.ids.ln_dbeta, &[d_ln1, gr("norm1.bias")], &[c, rows], c));
        steps.push(block::layernorm_dx_bwd(g, &self.ids.ln, x, w("norm1.weight"), d_ln1, &bb.tmp, c, rows, cfg.eps));
        steps.push(g.step(self.ids.add2, &[&bb.d_res, &bb.tmp, &bb.d_x], &[rows * c], rows * c));
        // `d_proj` is the ONE gradient buffer here that is not fully assigned by
        // its writer: the scatter touches only the real rows, and the pad slots
        // must arrive at the projection adjoint as zeros.
        let clears: Vec<&DeviceBuffer> = b.win.iter().map(|_| &bb.d_proj).collect();
        g.submit(&clears, &steps);
    }

    // -----------------------------------------------------------------------
    // composition seam (what `crates/deepseekocr` drives this tower through)
    // -----------------------------------------------------------------------
    //
    // The tower owns its input image and its output-gradient buffer, exactly
    // like every other model in this tree; a *composite* needs to drive both
    // from outside instead of from the constructor's own RNG. These accessors
    // are additive -- `forward()`/`backward()` are unchanged and still read the
    // same two buffers -- and the per-stage taps are what a composite's parity
    // test compares against a reference dump. Nothing here allocates.

    /// Overwrite the fixed input image, `[1, 3, image_h, image_w]` NCHW.
    pub fn write_image(&self, px: &[f32]) {
        assert_eq!(px.len(), (3 * self.cfg.image_h() * self.cfg.image_w()) as usize, "image size mismatch");
        self.gpu.write_f32(&self.image, px);
        self.fwd_done.set(false);
    }

    /// The seed buffer [`Self::backward`] differentiates: `d<objective>/d output`,
    /// shaped like [`Self::output`]. Constructed holding a fixed random direction
    /// (which is what the gradient check wants); a composite overwrites it with
    /// the real upstream gradient before calling `backward`.
    pub fn d_out(&self) -> &DeviceBuffer {
        &self.d_out
    }

    /// `[3, image_h, image_w]` -- the gradient w.r.t. the input image. Panics on
    /// an inference build, which allocates no gradient buffers.
    pub fn d_image(&self) -> &DeviceBuffer {
        &self.bwd().d_image
    }

    /// `[rows, C]` patch tokens, before `pos_embed` is added.
    pub fn patch_tokens(&self) -> &DeviceBuffer {
        &self.patch_nlc
    }

    /// `[rows, C]` patch tokens + `pos_embed` -- block 0's input.
    pub fn embedded_tokens(&self) -> &DeviceBuffer {
        &self.embed
    }

    /// Block `l`'s `norm1` output. **`rows + 1` rows**: the last is
    /// `WindowPlan::padded`'s zero sentinel, not a token.
    pub fn block_norm1(&self, l: usize) -> &DeviceBuffer {
        &self.blocks[l].ln1
    }

    /// `[rows, C]` -- block `l` after the attention residual add, before `norm2`.
    pub fn block_attn_res(&self, l: usize) -> &DeviceBuffer {
        &self.blocks[l].res
    }

    /// `[rows, C]` -- block `l`'s output.
    pub fn block_out(&self, l: usize) -> &DeviceBuffer {
        &self.blocks[l].out
    }

    /// The six neck/compressor stages in dispatch order, each with its element
    /// count: `conv1, norm1, conv2, norm2, compress1, compress2`. The last is
    /// [`Self::output`].
    pub fn neck_stages(&self) -> Vec<(&'static str, &DeviceBuffer, usize)> {
        let n = |c: &Conv| c.out_shape.numel() as usize;
        vec![
            ("neck_conv1", self.neck_c1.out(), n(&self.neck_c1)),
            ("neck_norm1", self.neck_n1.out(), self.neck_n1.shape.numel() as usize),
            ("neck_conv2", self.neck_c2.out(), n(&self.neck_c2)),
            ("neck_norm2", self.neck_n2.out(), self.neck_n2.shape.numel() as usize),
            ("compress1", self.comp_c1.out(), n(&self.comp_c1)),
            ("compress2", self.comp_c2.out(), n(&self.comp_c2)),
        ]
    }

    /// Element count of [`Self::output`] (`compress_out * grid_h/4 * grid_w/4`).
    pub fn out_len(&self) -> usize {
        self.comp_c2.out_shape.numel() as usize
    }

    // -----------------------------------------------------------------------
    // parameter access (what a `gradcheck::CheckModel` wrapper forwards to)
    // -----------------------------------------------------------------------

    pub fn param_names(&self) -> Vec<String> {
        self.cfg.param_list().into_iter().map(|(n, _)| n).collect()
    }
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        assert_eq!(data.len(), self.ps.numel(name), "{name}: size mismatch");
        self.gpu.write_f32(self.ps.w(name), data);
        // A weight moved between passes invalidates the cached activations.
        self.fwd_done.set(false);
    }
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }
}
