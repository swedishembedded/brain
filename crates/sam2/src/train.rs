// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SAM 2 **mask-decoder training**: an SSA-cached forward plus the device
//! backward, with the Hiera trunk and the FPN neck FROZEN.
//!
//! This is the common SAM 2 finetuning mode (the reference MOSE recipe freezes
//! the image encoder), and it is a self-contained sub-graph: everything the
//! decoder reads from the encoder — `image_embed`, the two high-resolution
//! feature maps and the dense positional encoding — enters as a constant
//! ([`FrozenEncode`]). What is NOT covered is spelled out on
//! [`TRAINABLE_PREFIXES`]; there is no silent partial.
//!
//! ## Why a second forward
//!
//! [`crate::model::Sam2::decode`] is the parity-gated INFERENCE forward: it
//! allocates per stage, drops every intermediate when the call returns, and is
//! free to reuse buffers. Training needs each stage to survive into the backward
//! pass. This module is the SSA twin, in exactly the relationship
//! `model::vit::vit_block_fwd_cached` has to `vit_block_fwd` — same math, same
//! kernels, same `Sam2` dispatch helpers (`linear`, `act_step`, `to_nchw`,
//! `to_nlc`), every stage in its own persistent buffer. It shares the dispatch
//! primitives rather than re-deriving them, and it reuses
//! `vision::ConvTranspose::backward` / `vision::LayerNorm2d::backward` for the
//! upscaling tail rather than open-coding a second copy of those adjoints.
//!
//! ## Loss
//!
//! `training/loss_fns.py::MultiStepMultiMasksAndIous`, single step, single
//! object: sigmoid **focal** + **dice** on the mask logits (reference weights
//! 20 / 1), **MSE** on the IoU head, **BCE-with-logits** on the object score
//! (the reference's `focal_alpha_obj_score = -1`, `focal_gamma_obj_score = 0`
//! focal loss IS plain BCE-with-logits). The reference's best-mask `argmin`
//! selection and its `actual_ious` (computed from THRESHOLDED masks) are both
//! piecewise-constant in the weights, so they are FROZEN into [`MaskTargets`]
//! rather than differentiated — the same discipline `gradcheck::check_moe` uses
//! to keep `top_k` off a selection boundary. Freezing them changes nothing
//! mathematically (their true gradient is zero almost everywhere) and is what
//! makes the objective smooth enough to finite-difference.
//!
//! ## Discipline
//!
//! * every backward is gather-based (one invocation per OUTPUT element) and
//!   atomic-free — it is composed entirely of kernels that already satisfy that;
//! * parameter-gradient buffers ACCUMULATE (`matmul_dw`, `bias_grad`,
//!   `layernorm_dgamma`/`_dbeta`, `convtr2d_dw`), so they are cleared exactly
//!   once per step by `ParamStore::zero_grads` and never appear in a submit's
//!   clear list;
//! * the four `attn_bwd_*_cross` kernels ASSIGN, and each attention here owns
//!   separate `d_q`/`d_k`/`d_v` buffers — the decoder keeps `k` and `v` in
//!   separate buffers (see `Sam2::attention`), so there is no fused `d_kv` for
//!   `attn_bwd_dk_cross` and `attn_bwd_dv_cross` to overwrite each other in;
//! * no new per-channel/per-row reduction is introduced in a backward, so no new
//!   caps-gated cooperative pair is required.

use std::cell::Cell;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::block;
use model::vit::{gather_rows, row_index_buffer, scatter_rows};
use vision::{Act, ConvTrSpec, ConvTranspose, Ctx, LayerNorm2d, Ln2dNames, Shape};

use crate::config::Sam2Config;
use crate::import::Tensors;
use crate::model::{idx, Sam2};

/// Parameter-name prefixes the decoder-finetune role set makes trainable.
///
/// COVERED: the two-way transformer (all projections, all four per-layer
/// LayerNorms, the token MLP, the final token→image attention and
/// `norm_final_attn`), the output tokens (`obj_score_token`, `iou_token`,
/// `mask_tokens`), the `output_upscaling` stack (both `ConvTranspose2d`s and the
/// channels-first `LayerNorm2d` between them), the four hypernetwork MLPs, the
/// IoU-prediction head, the object-score head, and the prompt encoder's
/// `no_mask_embed`.
///
/// NOT covered, deliberately, and left `Role::Frozen`: the whole Hiera trunk and
/// FPN neck (this is a decoder finetune); `conv_s0`/`conv_s1`, which consume
/// frozen FPN levels and are folded into [`FrozenEncode::high_res`]; the prompt
/// encoder's `pe_layer` gaussian matrix, `point_embeddings`, `not_a_point_embed`
/// and `mask_downscaling` — the first three are folded into the sparse embedding
/// by `hostpe::embed_points` on the HOST, which severs their gradient, and
/// moving that add onto the device is a separate task; and `obj_ptr_proj` /
/// `no_obj_ptr`, which feed the video-path object pointer that no image-path
/// loss touches.
pub const TRAINABLE_PREFIXES: &[&str] = &[
    "sam_mask_decoder.transformer.",
    "sam_mask_decoder.obj_score_token.",
    "sam_mask_decoder.iou_token.",
    "sam_mask_decoder.mask_tokens.",
    "sam_mask_decoder.output_upscaling.",
    "sam_mask_decoder.output_hypernetworks_mlps.",
    "sam_mask_decoder.iou_prediction_head.",
    "sam_mask_decoder.pred_obj_score_head.",
    "sam_prompt_encoder.no_mask_embed.",
];

/// True for the parameters [`TRAINABLE_PREFIXES`] covers.
pub fn is_decoder_trainable(name: &str) -> bool {
    TRAINABLE_PREFIXES.iter().any(|p| name.starts_with(p))
}

/// Everything the mask decoder consumes from the (frozen) image encoder.
pub struct FrozenEncode {
    /// `[1, d, side, side]` NCHW — `fpn[lvl] + no_mem_embed`.
    pub image_embed: DeviceBuffer,
    /// `conv_s0(fpn[0])` `[1, d/8, 4*side, 4*side]` and `conv_s1(fpn[1])`
    /// `[1, d/4, 2*side, 2*side]`, in that order.
    pub high_res: [DeviceBuffer; 2],
    /// `[1, d, side, side]` NCHW — `PositionEmbeddingRandom` over the grid.
    pub dense_pe: DeviceBuffer,
    /// `[n_sparse, d]` — the host-computed sparse prompt embedding.
    pub sparse: DeviceBuffer,
    pub n_sparse: u32,
}

/// The frozen targets of one training step.
pub struct MaskTargets {
    /// `[nmt, 16*n_img]` binary ground truth, already broadcast over the mask
    /// channel (the reference's `target_masks.expand_as(src_masks)`).
    pub masks: DeviceBuffer,
    /// `[nmt]` `actual_ious` — frozen, because the reference computes it from
    /// THRESHOLDED masks and it is piecewise constant in the weights.
    pub ious: DeviceBuffer,
    /// `[1]` `target_obj`.
    pub obj: DeviceBuffer,
    /// `[nmt]` per-mask supervision weight. The reference back-props focal+dice
    /// only through the `argmin`-selected channel; that selection is frozen here.
    pub mask_w: DeviceBuffer,
    pub w_focal: f32,
    pub w_dice: f32,
    pub w_iou: f32,
    pub w_class: f32,
    pub focal_alpha: f32,
    pub focal_gamma: f32,
    /// Host copy of `mask_w`, for the scalar reduction in [`MaskDecoderTrainer::loss`].
    pub mask_w_host: Vec<f32>,
}

/// Kernel indices the backward half needs, on top of `Sam2`'s forward set.
struct BwdIds {
    matmul_dx: usize,
    matmul_dw: usize,
    bias_grad: usize,
    ln_dgamma: usize,
    ln_dbeta: usize,
    dscores: usize,
    dv: usize,
    dq: usize,
    dk: usize,
    add_chan_bcast_dv: usize,
    gelu_erf_bwd: usize,
    leaky_relu_bwd: usize,
    sigmoid_bwd: usize,
    focal_stats: usize,
    focal_grad: usize,
    mse_value: usize,
    mse_grad: usize,
    bce: usize,
    bce_grad: usize,
}

impl BwdIds {
    fn resolve(g: &Gpu) -> BwdIds {
        BwdIds {
            matmul_dx: idx(g, "matmul_dx"),
            matmul_dw: idx(g, "matmul_dw"),
            bias_grad: idx(g, "bias_grad"),
            ln_dgamma: idx(g, "layernorm_dgamma"),
            ln_dbeta: idx(g, "layernorm_dbeta"),
            dscores: idx(g, "attn_bwd_dscores_cross"),
            dv: idx(g, "attn_bwd_dv_cross"),
            dq: idx(g, "attn_bwd_dq_cross"),
            dk: idx(g, "attn_bwd_dk_cross"),
            add_chan_bcast_dv: idx(g, "add_chan_bcast_dv"),
            gelu_erf_bwd: idx(g, "gelu_erf_bwd"),
            leaky_relu_bwd: idx(g, "leaky_relu_bwd"),
            sigmoid_bwd: idx(g, "sigmoid_bwd"),
            focal_stats: idx(g, "focal_dice_stats"),
            focal_grad: idx(g, "focal_dice_grad"),
            mse_value: idx(g, "mse_value"),
            mse_grad: idx(g, "mse_grad"),
            bce: idx(g, "bce_logits"),
            bce_grad: idx(g, "bce_logits_grad"),
        }
    }
}

/// One `Attention` module's activation cache + backward scratch. Mirrors
/// [`crate::model::Sam2::attention`] exactly, including the deliberate choice to
/// keep `k` and `v` in SEPARATE buffers.
struct AttnCache {
    prefix: String,
    tq: u32,
    tk: u32,
    io: u32,
    din: u32,
    heads: u32,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    d_q: DeviceBuffer,
    d_k: DeviceBuffer,
    d_v: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_scores: DeviceBuffer,
    /// Gradients w.r.t. the three (independent) module inputs.
    d_q_in: DeviceBuffer,
    d_k_in: DeviceBuffer,
    d_v_in: DeviceBuffer,
}

impl AttnCache {
    fn new(g: &Gpu, prefix: &str, tq: u32, tk: u32, io: u32, din: u32, heads: u32) -> AttnCache {
        assert_eq!(io % heads, 0, "{prefix}: internal dim {io} is not divisible by {heads} heads");
        let sl = heads as u64 * tq as u64 * tk as u64;
        AttnCache {
            prefix: prefix.to_string(),
            tq,
            tk,
            io,
            din,
            heads,
            q: g.storage(tq as u64 * io as u64),
            k: g.storage(tk as u64 * io as u64),
            v: g.storage(tk as u64 * io as u64),
            scores: g.storage(sl),
            probs: g.storage(sl),
            ctx: g.storage(tq as u64 * io as u64),
            d_q: g.storage(tq as u64 * io as u64),
            d_k: g.storage(tk as u64 * io as u64),
            d_v: g.storage(tk as u64 * io as u64),
            d_ctx: g.storage(tq as u64 * io as u64),
            d_scores: g.storage(sl),
            d_q_in: g.storage(tq as u64 * din as u64),
            d_k_in: g.storage(tk as u64 * din as u64),
            d_v_in: g.storage(tk as u64 * din as u64),
        }
    }
    fn hd(&self) -> u32 {
        self.io / self.heads
    }
    /// `[bsz, heads, t_dec, t_enc, head_dim, kv_stride, v_off, d_model]` — the
    /// `attn_apply_cross` / `attn_bwd_{dscores,dv}_cross` uniform. The decoder's
    /// v buffer is compact `[tk, io]`, hence `kv_stride = io`, `v_off = 0`.
    fn p_v(&self) -> [u32; 8] {
        [1, self.heads, self.tq, self.tk, self.hd(), self.io, 0, self.io]
    }
    /// `[bsz, heads, t_dec, t_enc, head_dim, q_stride, kv_stride, q_off, k_off]`
    /// — the `attn_scores_cross` / `attn_bwd_{dq,dk}_cross` uniform.
    fn p_qk(&self) -> [u32; 9] {
        [1, self.heads, self.tq, self.tk, self.hd(), self.io, self.io, 0, 0]
    }
}

/// An `sam2_utils.MLP` cache: one buffer per linear output (the PRE-activation
/// every `*_bwd` kernel wants) and one per activation output (the input the next
/// `matmul_dw` wants).
struct MlpCache {
    prefix: String,
    rows: u32,
    dims: Vec<u32>,
    act: Act,
    sigmoid: bool,
    pre: Vec<DeviceBuffer>,
    post: Vec<DeviceBuffer>,
    d_pre: Vec<DeviceBuffer>,
    d_post: Vec<DeviceBuffer>,
}

impl MlpCache {
    fn new(g: &Gpu, prefix: &str, rows: u32, dims: &[u32], act: Act, sigmoid: bool) -> MlpCache {
        let n_layers = dims.len() - 1;
        let mut pre = Vec::with_capacity(n_layers);
        let mut post = Vec::with_capacity(n_layers);
        let mut d_pre = Vec::with_capacity(n_layers);
        let mut d_post = Vec::with_capacity(n_layers);
        for dim in dims.iter().skip(1) {
            let n = rows as u64 * *dim as u64;
            pre.push(g.storage(n));
            post.push(g.storage(n));
            d_pre.push(g.storage(n));
            d_post.push(g.storage(n));
        }
        MlpCache { prefix: prefix.to_string(), rows, dims: dims.to_vec(), act, sigmoid, pre, post, d_pre, d_post }
    }
    fn last(&self) -> usize {
        self.dims.len() - 2
    }
    fn out(&self) -> &DeviceBuffer {
        if self.sigmoid {
            &self.post[self.last()]
        } else {
            &self.pre[self.last()]
        }
    }
}

/// One `TwoWayAttentionBlock`'s SSA activations.
struct LayerCache {
    p: String,
    self_attn: AttnCache,
    t2i: AttnCache,
    i2t: AttnCache,
    mlp: MlpCache,
    /// `queries + query_pe` feeding the self-attention (layer > 0 only).
    qs: DeviceBuffer,
    o1: DeviceBuffer,
    q_a: DeviceBuffer,
    q_b: DeviceBuffer,
    q2: DeviceBuffer,
    /// `keys + key_pe`. The reference recomputes this twice per layer from the
    /// SAME `keys`, so one buffer serves both the token→image K and the
    /// image→token Q.
    kpe: DeviceBuffer,
    o2: DeviceBuffer,
    q_c: DeviceBuffer,
    q_d: DeviceBuffer,
    q_e: DeviceBuffer,
    q_f: DeviceBuffer,
    q3: DeviceBuffer,
    o3: DeviceBuffer,
    k_a: DeviceBuffer,
    k_b: DeviceBuffer,
}

/// Resolved decoder geometry.
#[derive(Clone, Copy)]
struct Dims {
    d: u32,
    t: u32,
    n_img: u32,
    nmt: u32,
    mask_dim: u32,
    /// Mask-logit pixels per mask (`16 * n_img`, i.e. `(4*side)^2`).
    hi: u32,
    depth: u32,
}

/// The SAM 2 mask decoder, wired for training.
pub struct MaskDecoderTrainer {
    pub sam: Sam2,
    enc: FrozenEncode,
    tgt: MaskTargets,
    dim: Dims,
    bwd: BwdIds,
    eps: f32,

    // ---- forward SSA cache ----
    dense: DeviceBuffer,
    src_in: DeviceBuffer,
    keys0: DeviceBuffer,
    key_pe: DeviceBuffer,
    tokens: DeviceBuffer,
    tok_idx: [DeviceBuffer; 4],
    layers: Vec<LayerCache>,
    final_attn: AttnCache,
    qf: DeviceBuffer,
    kf: DeviceBuffer,
    final_out: DeviceBuffer,
    sum_final: DeviceBuffer,
    hs: DeviceBuffer,
    src_img: DeviceBuffer,
    dc1: ConvTranspose,
    sum1: DeviceBuffer,
    ln2d: LayerNorm2d,
    act1: DeviceBuffer,
    dc2: ConvTranspose,
    sum2: DeviceBuffer,
    upscaled: DeviceBuffer,
    up_nlc: DeviceBuffer,
    hyper_tok: Vec<DeviceBuffer>,
    hyper: Vec<MlpCache>,
    hyper_idx: Vec<DeviceBuffer>,
    hyper_in: DeviceBuffer,
    masks_all: DeviceBuffer,
    iou_tok: DeviceBuffer,
    iou_head: MlpCache,
    obj_tok: DeviceBuffer,
    obj_head: MlpCache,
    /// `hs` row index for each of the `2 + nmt` token reads (all disjoint).
    hs_rows: Vec<DeviceBuffer>,

    // ---- loss + backward scratch ----
    stats: DeviceBuffer,
    mse_out: DeviceBuffer,
    bce_out: DeviceBuffer,
    d_masks: DeviceBuffer,
    d_iou_raw: DeviceBuffer,
    d_iou: DeviceBuffer,
    d_obj_raw: DeviceBuffer,
    d_obj: DeviceBuffer,
    d_tokens: DeviceBuffer,
    d_hs: DeviceBuffer,
    d_hyper_in: DeviceBuffer,
    d_up_nlc: DeviceBuffer,
    d_upscaled: DeviceBuffer,
    d_sum2: DeviceBuffer,
    d_act1: DeviceBuffer,
    d_ln2d: DeviceBuffer,
    d_sum1: DeviceBuffer,
    d_src_img: DeviceBuffer,
    d_keys_up: DeviceBuffer,
    ln_mean: DeviceBuffer,
    ln_inv: DeviceBuffer,
    fwd_done: Cell<bool>,
}

impl MaskDecoderTrainer {
    /// Build the trainer over a fresh [`Sam2`] whose decoder parameters are
    /// `Role::Trainable` and whose trunk/neck are `Role::Frozen`.
    pub fn new(gpu: Gpu, cfg: Sam2Config, weights: &Tensors, enc: FrozenEncode, tgt: MaskTargets) -> MaskDecoderTrainer {
        let sam = Sam2::new_with_roles(gpu, cfg, weights, &is_decoder_trainable);
        MaskDecoderTrainer::over(sam, enc, tgt)
    }

    /// [`Self::new`] over an already-built model, whose `ParamStore` must
    /// already carry gradients for [`TRAINABLE_PREFIXES`].
    pub fn over(sam: Sam2, enc: FrozenEncode, tgt: MaskTargets) -> MaskDecoderTrainer {
        let cfg = sam.cfg.clone();
        assert!(cfg.pred_obj_scores, "the training graph assumes the object-score token is present");
        assert!(cfg.pred_obj_scores_mlp, "the training graph assumes an MLP object-score head");
        let g = &sam.gpu;
        let d = cfg.d_model;
        let io = d / cfg.attention_downsample_rate;
        let heads = cfg.transformer_heads;
        let side = cfg.image_embedding_size();
        let n_img = side * side;
        let nmt = cfg.num_mask_tokens();
        let mask_dim = d / 8;
        let hi = 16 * n_img;
        let n_out_tokens = 2 + nmt;
        let t = n_out_tokens + enc.n_sparse;
        let dim = Dims { d, t, n_img, nmt, mask_dim, hi, depth: cfg.transformer_depth };
        let bwd = BwdIds::resolve(g);
        let qn = t as u64 * d as u64;
        let kn = n_img as u64 * d as u64;

        // Token-buffer row indices: [obj_score | iou | mask x nmt | sparse].
        let tok_idx = [
            row_index_buffer(g, "sam2_tok_obj", &[0]),
            row_index_buffer(g, "sam2_tok_iou", &[1]),
            row_index_buffer(g, "sam2_tok_mask", &(2..2 + nmt).collect::<Vec<u32>>()),
            row_index_buffer(g, "sam2_tok_sparse", &(n_out_tokens..t).collect::<Vec<u32>>()),
        ];
        // `hs` reads: obj = row 0, iou = row 1, mask i = row 2+i — disjoint, so
        // `row_scatter` into a cleared `d_hs` composes them without accumulation.
        let hs_rows: Vec<DeviceBuffer> = (0..2 + nmt).map(|r| row_index_buffer(g, "sam2_hs_row", &[r])).collect();

        let mut layers = Vec::with_capacity(cfg.transformer_depth as usize);
        for l in 0..cfg.transformer_depth {
            let p = format!("sam_mask_decoder.transformer.layers.{l}");
            layers.push(LayerCache {
                self_attn: AttnCache::new(g, &format!("{p}.self_attn"), t, t, d, d, heads),
                t2i: AttnCache::new(g, &format!("{p}.cross_attn_token_to_image"), t, n_img, io, d, heads),
                i2t: AttnCache::new(g, &format!("{p}.cross_attn_image_to_token"), n_img, t, io, d, heads),
                mlp: MlpCache::new(g, &format!("{p}.mlp"), t, &[d, cfg.transformer_mlp_dim, d], Act::Relu, false),
                qs: g.storage(qn),
                o1: g.storage(qn),
                q_a: g.storage(qn),
                q_b: g.storage(qn),
                q2: g.storage(qn),
                kpe: g.storage(kn),
                o2: g.storage(qn),
                q_c: g.storage(qn),
                q_d: g.storage(qn),
                q_e: g.storage(qn),
                q_f: g.storage(qn),
                q3: g.storage(qn),
                o3: g.storage(kn),
                k_a: g.storage(kn),
                k_b: g.storage(kn),
                p,
            });
        }

        let ctx = Ctx::new(&sam.gpu, &sam.conv_ids);
        let dc1 = ConvTranspose::torch(
            &ctx,
            "sam_mask_decoder.output_upscaling.0",
            Shape::new(1, d, side, side),
            ConvTrSpec::new(d / 4, 2, 2, 0),
        );
        let ln2d = LayerNorm2d::new(
            &ctx,
            Ln2dNames::torch("sam_mask_decoder.output_upscaling.1"),
            Shape::new(1, d / 4, 2 * side, 2 * side),
            cfg.ln2d_eps,
        );
        let dc2 = ConvTranspose::torch(
            &ctx,
            "sam_mask_decoder.output_upscaling.3",
            Shape::new(1, d / 4, 2 * side, 2 * side),
            ConvTrSpec::new(d / 8, 2, 2, 0),
        );
        let n1 = (d / 4) as u64 * 4 * n_img as u64;
        let n2 = mask_dim as u64 * hi as u64;

        let hyper: Vec<MlpCache> = (0..nmt)
            .map(|i| {
                MlpCache::new(
                    g,
                    &format!("sam_mask_decoder.output_hypernetworks_mlps.{i}"),
                    1,
                    &[d, d, d, mask_dim],
                    Act::Relu,
                    false,
                )
            })
            .collect();
        let hyper_tok: Vec<DeviceBuffer> = (0..nmt).map(|_| g.storage(d as u64)).collect();
        let hyper_idx: Vec<DeviceBuffer> = (0..nmt).map(|i| row_index_buffer(g, "sam2_hyper_row", &[i])).collect();

        let mut iou_dims = vec![d];
        iou_dims.extend(std::iter::repeat_n(cfg.iou_head_hidden_dim, cfg.iou_head_depth as usize - 1));
        iou_dims.push(nmt);
        let iou_head = MlpCache::new(
            g,
            "sam_mask_decoder.iou_prediction_head",
            1,
            &iou_dims,
            Act::Relu,
            cfg.iou_prediction_use_sigmoid,
        );
        let obj_head = MlpCache::new(g, "sam_mask_decoder.pred_obj_score_head", 1, &[d, d, d, 1], Act::Relu, false);
        let ln_rows = t.max(n_img) as u64;

        let out = MaskDecoderTrainer {
            eps: cfg.ln_eps,
            dense: g.storage(kn),
            src_in: g.storage(kn),
            keys0: g.storage(kn),
            key_pe: g.storage(kn),
            tokens: g.storage(qn),
            tok_idx,
            final_attn: AttnCache::new(
                g,
                "sam_mask_decoder.transformer.final_attn_token_to_image",
                t,
                n_img,
                io,
                d,
                heads,
            ),
            qf: g.storage(qn),
            kf: g.storage(kn),
            final_out: g.storage(qn),
            sum_final: g.storage(qn),
            hs: g.storage(qn),
            src_img: g.storage(kn),
            dc1,
            sum1: g.storage(n1),
            ln2d,
            act1: g.storage(n1),
            dc2,
            sum2: g.storage(n2),
            upscaled: g.storage(n2),
            up_nlc: g.storage(n2),
            hyper_tok,
            hyper,
            hyper_idx,
            hyper_in: g.storage(nmt as u64 * mask_dim as u64),
            masks_all: g.storage(nmt as u64 * hi as u64),
            iou_tok: g.storage(d as u64),
            iou_head,
            obj_tok: g.storage(d as u64),
            obj_head,
            hs_rows,
            stats: g.storage(4 * nmt as u64),
            mse_out: g.storage(nmt as u64),
            bce_out: g.storage(1),
            d_masks: g.storage(nmt as u64 * hi as u64),
            d_iou_raw: g.storage(nmt as u64),
            d_iou: g.storage(nmt as u64),
            d_obj_raw: g.storage(1),
            d_obj: g.storage(1),
            d_tokens: g.storage(qn),
            d_hs: g.storage(qn),
            d_hyper_in: g.storage(nmt as u64 * mask_dim as u64),
            d_up_nlc: g.storage(n2),
            d_upscaled: g.storage(n2),
            d_sum2: g.storage(n2),
            d_act1: g.storage(n1),
            d_ln2d: g.storage(n1),
            d_sum1: g.storage(n1),
            d_src_img: g.storage(kn),
            d_keys_up: g.storage(kn),
            ln_mean: g.storage(ln_rows),
            ln_inv: g.storage(ln_rows),
            fwd_done: Cell::new(false),
            layers,
            dim,
            bwd,
            enc,
            tgt,
            sam,
        };
        // `key_pe` is a constant NLC view of the (frozen) dense positional encoding.
        let mut s = Vec::new();
        out.sam.to_nlc(&mut s, &out.enc.dense_pe, &out.key_pe, out.dim.d, out.dim.n_img);
        out.sam.gpu.submit(&[], &s);
        out
    }

    // -----------------------------------------------------------------------
    // small dispatch helpers
    // -----------------------------------------------------------------------

    fn g(&self) -> &Gpu {
        &self.sam.gpu
    }
    fn ctx(&self) -> Ctx<'_> {
        Ctx::new(&self.sam.gpu, &self.sam.conv_ids)
    }
    fn w(&self, n: &str) -> &DeviceBuffer {
        self.sam.ps.w(n)
    }
    fn gd(&self, n: &str) -> &DeviceBuffer {
        self.sam.ps.g(n)
    }
    fn add2(&self, a: &DeviceBuffer, b: &DeviceBuffer, out: &DeviceBuffer, n: u32) -> Step {
        self.g().step(self.sam.ids.add2, &[a, b, out], &[n], n)
    }
    /// `out += src` — `out` is read-modify-write, so the CALLER owns the clear.
    fn acc(&self, out: &DeviceBuffer, src: &DeviceBuffer, n: u32) -> Step {
        self.g().step(self.sam.ids.axpy, &[out, src], &[n, f(1.0)], n)
    }
    /// `out += s * src`.
    fn acc_s(&self, out: &DeviceBuffer, src: &DeviceBuffer, n: u32, scale: f32) -> Step {
        self.g().step(self.sam.ids.axpy, &[out, src], &[n, f(scale)], n)
    }
    fn act_bwd(&self, act: Act, x: &DeviceBuffer, dy: &DeviceBuffer, dx: &DeviceBuffer, n: u32) -> Step {
        match act {
            // ReLU is `leaky_relu` at slope 0 in BOTH directions.
            Act::Relu => self.g().step(self.bwd.leaky_relu_bwd, &[x, dy, dx], &[n, f(0.0)], n),
            Act::GeluErf => self.g().step(self.bwd.gelu_erf_bwd, &[x, dy, dx], &[n], n),
            Act::Sigmoid => self.g().step(self.bwd.sigmoid_bwd, &[x, dy, dx], &[n], n),
            other => panic!("sam2::train: no backward for activation {other:?}"),
        }
    }
    /// Backward of `out = x @ W^T + b`: `dW += dyᵀx`, `db += Σdy`, `dx = dy·W`.
    /// `matmul_dw` / `bias_grad` accumulate (cleared once by `zero_grads`);
    /// `matmul_dx` is dispatched with `accumulate = 0`, so `dx` is ASSIGNED.
    #[allow(clippy::too_many_arguments)]
    fn linear_bwd(
        &self,
        s: &mut Vec<Step>,
        dy: &DeviceBuffer,
        x: &DeviceBuffer,
        dx: &DeviceBuffer,
        rows: u32,
        k: u32,
        n: u32,
        wname: &str,
        bname: &str,
    ) {
        let g = self.g();
        s.push(g.step(self.bwd.bias_grad, &[dy, self.gd(bname)], &[rows, n], n));
        s.push(g.step(self.bwd.matmul_dw, &[dy, x, self.gd(wname)], &[rows, k, n], n * k));
        s.push(g.step(self.bwd.matmul_dx, &[dy, self.w(wname), dx], &[rows, k, n, 0], rows * k));
    }

    fn ln_fwd(&self, s: &mut Vec<Step>, x: &DeviceBuffer, out: &DeviceBuffer, prefix: &str, rows: u32, d: u32) {
        s.push(block::layernorm_fwd(
            self.g(),
            &self.sam.ids.ln,
            x,
            self.w(&format!("{prefix}.weight")),
            self.w(&format!("{prefix}.bias")),
            out,
            d,
            rows,
            self.eps,
        ));
    }

    /// LayerNorm backward. `mean`/`inv` are recomputed from the cached input (a
    /// `[rows]` pair is cheaper to recompute than to keep — same call the ViT
    /// backward makes).
    #[allow(clippy::too_many_arguments)]
    fn ln_bwd(
        &self,
        s: &mut Vec<Step>,
        x: &DeviceBuffer,
        prefix: &str,
        rows: u32,
        d: u32,
        dy: &DeviceBuffer,
        dx: &DeviceBuffer,
    ) {
        let g = self.g();
        let (wn, bn) = (format!("{prefix}.weight"), format!("{prefix}.bias"));
        s.push(block::ln_stats_fwd(g, &self.sam.ids.ln, x, &self.ln_mean, &self.ln_inv, d, rows, self.eps));
        s.push(g.step(self.bwd.ln_dgamma, &[dy, x, &self.ln_mean, &self.ln_inv, self.gd(&wn)], &[d, rows], d));
        s.push(g.step(self.bwd.ln_dbeta, &[dy, self.gd(&bn)], &[d, rows], d));
        s.push(block::layernorm_dx_bwd(g, &self.sam.ids.ln, x, self.w(&wn), dy, dx, d, rows, self.eps));
    }

    // -----------------------------------------------------------------------
    // attention / MLP
    // -----------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn attn_fwd(
        &self,
        s: &mut Vec<Step>,
        a: &AttnCache,
        q_in: &DeviceBuffer,
        k_in: &DeviceBuffer,
        v_in: &DeviceBuffer,
        out: &DeviceBuffer,
    ) {
        let (p, io, d) = (&a.prefix, a.io, a.din);
        self.sam.linear(s, q_in, &a.q, a.tq, d, io, &format!("{p}.q_proj.weight"), &format!("{p}.q_proj.bias"));
        self.sam.linear(s, k_in, &a.k, a.tk, d, io, &format!("{p}.k_proj.weight"), &format!("{p}.k_proj.bias"));
        self.sam.linear(s, v_in, &a.v, a.tk, d, io, &format!("{p}.v_proj.weight"), &format!("{p}.v_proj.bias"));
        let ids = &self.sam.ids.cross;
        let g = self.g();
        s.push(g.step(ids.scores, &[&a.q, &a.k, &a.scores], &a.p_qk(), a.heads * a.tq * a.tk));
        s.push(g.step(ids.softmax, &[&a.scores, &a.probs], &[1, a.heads, a.tq, a.tk], a.heads * a.tq));
        s.push(g.step(ids.apply, &[&a.probs, &a.v, &a.ctx], &a.p_v(), a.heads * a.tq * a.hd()));
        self.sam.linear(s, &a.ctx, out, a.tq, io, d, &format!("{p}.out_proj.weight"), &format!("{p}.out_proj.bias"));
    }

    /// Backward of one attention module. Writes `a.d_q_in` / `a.d_k_in` /
    /// `a.d_v_in` (all ASSIGNED), leaving the caller to fold them into whatever
    /// the three inputs actually were — which is what lets the same routine serve
    /// the self-attention (q == k == v), the two cross-attentions (q, k and v all
    /// different) and the final attention.
    #[allow(clippy::too_many_arguments)]
    fn attn_bwd(
        &self,
        s: &mut Vec<Step>,
        a: &AttnCache,
        q_in: &DeviceBuffer,
        k_in: &DeviceBuffer,
        v_in: &DeviceBuffer,
        d_out: &DeviceBuffer,
    ) {
        let g = self.g();
        let (p, io, d) = (&a.prefix, a.io, a.din);
        self.linear_bwd(
            s,
            d_out,
            &a.ctx,
            &a.d_ctx,
            a.tq,
            io,
            d,
            &format!("{p}.out_proj.weight"),
            &format!("{p}.out_proj.bias"),
        );
        // The four cross-attention adjoints. All ASSIGN, and d_q / d_k / d_v are
        // distinct buffers, so nothing needs pre-zeroing or an accumulate flag.
        s.push(g.step(self.bwd.dscores, &[&a.d_ctx, &a.v, &a.probs, &a.d_scores], &a.p_v(), a.heads * a.tq));
        s.push(g.step(self.bwd.dv, &[&a.probs, &a.d_ctx, &a.d_v], &a.p_v(), a.heads * a.tk * a.hd()));
        s.push(g.step(self.bwd.dq, &[&a.d_scores, &a.k, &a.d_q], &a.p_qk(), a.heads * a.tq * a.hd()));
        s.push(g.step(self.bwd.dk, &[&a.d_scores, &a.q, &a.d_k], &a.p_qk(), a.heads * a.tk * a.hd()));
        self.linear_bwd(s, &a.d_q, q_in, &a.d_q_in, a.tq, d, io, &format!("{p}.q_proj.weight"), &format!("{p}.q_proj.bias"));
        self.linear_bwd(s, &a.d_k, k_in, &a.d_k_in, a.tk, d, io, &format!("{p}.k_proj.weight"), &format!("{p}.k_proj.bias"));
        self.linear_bwd(s, &a.d_v, v_in, &a.d_v_in, a.tk, d, io, &format!("{p}.v_proj.weight"), &format!("{p}.v_proj.bias"));
    }

    fn mlp_fwd(&self, s: &mut Vec<Step>, m: &MlpCache, x: &DeviceBuffer) {
        let last = m.last();
        for i in 0..=last {
            let (k, n) = (m.dims[i], m.dims[i + 1]);
            let src: &DeviceBuffer = if i == 0 { x } else { &m.post[i - 1] };
            self.sam.linear(
                s,
                src,
                &m.pre[i],
                m.rows,
                k,
                n,
                &format!("{}.layers.{i}.weight", m.prefix),
                &format!("{}.layers.{i}.bias", m.prefix),
            );
            if i < last {
                s.push(self.sam.act_step(&m.pre[i], &m.post[i], m.rows * n, m.act));
            } else if m.sigmoid {
                s.push(self.sam.act_step(&m.pre[i], &m.post[i], m.rows * n, Act::Sigmoid));
            }
        }
    }

    fn mlp_bwd(&self, s: &mut Vec<Step>, m: &MlpCache, x: &DeviceBuffer, d_out: &DeviceBuffer, d_x: &DeviceBuffer) {
        let last = m.last();
        for i in (0..=last).rev() {
            let (k, n) = (m.dims[i], m.dims[i + 1]);
            // Gradient w.r.t. this layer's PRE-activation.
            let d_pre: &DeviceBuffer = if i == last {
                if m.sigmoid {
                    s.push(self.act_bwd(Act::Sigmoid, &m.pre[i], d_out, &m.d_pre[i], m.rows * n));
                    &m.d_pre[i]
                } else {
                    d_out
                }
            } else {
                s.push(self.act_bwd(m.act, &m.pre[i], &m.d_post[i], &m.d_pre[i], m.rows * n));
                &m.d_pre[i]
            };
            let x_in: &DeviceBuffer = if i == 0 { x } else { &m.post[i - 1] };
            let dx: &DeviceBuffer = if i == 0 { d_x } else { &m.d_post[i - 1] };
            self.linear_bwd(
                s,
                d_pre,
                x_in,
                dx,
                m.rows,
                k,
                n,
                &format!("{}.layers.{i}.weight", m.prefix),
                &format!("{}.layers.{i}.bias", m.prefix),
            );
        }
    }

    // =======================================================================
    // forward
    // =======================================================================

    /// SSA training forward of the whole mask decoder. Every stage lands in its
    /// own persistent buffer, which IS the activation cache [`Self::backward`]
    /// reads.
    pub fn forward(&self) {
        let g = self.g();
        let dm = self.dim;
        let (d, t, n_img) = (dm.d, dm.t, dm.n_img);
        let qn = t * d;
        let kn = n_img * d;
        let perm = &self.sam.ids.permute;

        // ---- dense prompt: `no_mask_embed` broadcast over the grid ----
        // The CLEAR is what makes this a pure broadcast (`add_chan_inplace` is
        // read-modify-write); its adjoint is `add_chan_bcast_dv`.
        g.submit(
            &[&self.dense],
            &[g.step(
                self.sam.ids.add_chan_inplace,
                &[&self.dense, self.w("sam_prompt_encoder.no_mask_embed.weight")],
                &[kn, d, n_img],
                kn,
            )],
        );

        let mut s = Vec::new();
        s.push(self.add2(&self.enc.image_embed, &self.dense, &self.src_in, kn));
        self.sam.to_nlc(&mut s, &self.src_in, &self.keys0, d, n_img);
        // tokens = [obj_score | iou | mask x nmt | sparse]; every row is named,
        // so `row_scatter` needs no clear.
        s.push(scatter_rows(g, perm, &self.tok_idx[0], self.w("sam_mask_decoder.obj_score_token.weight"), &self.tokens, 1, d, t));
        s.push(scatter_rows(g, perm, &self.tok_idx[1], self.w("sam_mask_decoder.iou_token.weight"), &self.tokens, 1, d, t));
        s.push(scatter_rows(g, perm, &self.tok_idx[2], self.w("sam_mask_decoder.mask_tokens.weight"), &self.tokens, dm.nmt, d, t));
        s.push(scatter_rows(g, perm, &self.tok_idx[3], &self.enc.sparse, &self.tokens, self.enc.n_sparse, d, t));
        g.submit(&[], &s);

        // ---- two-way transformer ----
        for l in 0..dm.depth as usize {
            let lc = &self.layers[l];
            let (q_in, k_in): (&DeviceBuffer, &DeviceBuffer) = if l == 0 {
                (&self.tokens, &self.keys0)
            } else {
                (&self.layers[l - 1].q_f, &self.layers[l - 1].k_b)
            };
            let mut s = Vec::new();
            // (1) token self-attention
            if l == 0 {
                // `skip_first_layer_pe`: no positional add AND no residual.
                self.attn_fwd(&mut s, &lc.self_attn, q_in, q_in, q_in, &lc.o1);
                s.push(self.acc(&lc.q_a, &lc.o1, qn));
                g.submit(&[&lc.q_a], &s);
            } else {
                s.push(self.add2(q_in, &self.tokens, &lc.qs, qn));
                self.attn_fwd(&mut s, &lc.self_attn, &lc.qs, &lc.qs, q_in, &lc.o1);
                s.push(self.add2(q_in, &lc.o1, &lc.q_a, qn));
                g.submit(&[], &s);
            }
            let mut s = Vec::new();
            self.ln_fwd(&mut s, &lc.q_a, &lc.q_b, &format!("{}.norm1", lc.p), t, d);
            // (2) tokens -> image
            s.push(self.add2(&lc.q_b, &self.tokens, &lc.q2, qn));
            s.push(self.add2(k_in, &self.key_pe, &lc.kpe, kn));
            self.attn_fwd(&mut s, &lc.t2i, &lc.q2, &lc.kpe, k_in, &lc.o2);
            s.push(self.add2(&lc.q_b, &lc.o2, &lc.q_c, qn));
            self.ln_fwd(&mut s, &lc.q_c, &lc.q_d, &format!("{}.norm2", lc.p), t, d);
            // (3) MLP on the tokens
            self.mlp_fwd(&mut s, &lc.mlp, &lc.q_d);
            s.push(self.add2(&lc.q_d, lc.mlp.out(), &lc.q_e, qn));
            self.ln_fwd(&mut s, &lc.q_e, &lc.q_f, &format!("{}.norm3", lc.p), t, d);
            // (4) image -> tokens
            s.push(self.add2(&lc.q_f, &self.tokens, &lc.q3, qn));
            self.attn_fwd(&mut s, &lc.i2t, &lc.kpe, &lc.q3, &lc.q_f, &lc.o3);
            s.push(self.add2(k_in, &lc.o3, &lc.k_a, kn));
            self.ln_fwd(&mut s, &lc.k_a, &lc.k_b, &format!("{}.norm4", lc.p), n_img, d);
            g.submit(&[], &s);
        }
        let last = &self.layers[dm.depth as usize - 1];

        // ---- final token -> image attention + LayerNorm ----
        let mut s = Vec::new();
        s.push(self.add2(&last.q_f, &self.tokens, &self.qf, qn));
        s.push(self.add2(&last.k_b, &self.key_pe, &self.kf, kn));
        self.attn_fwd(&mut s, &self.final_attn, &self.qf, &self.kf, &last.k_b, &self.final_out);
        s.push(self.add2(&last.q_f, &self.final_out, &self.sum_final, qn));
        self.ln_fwd(&mut s, &self.sum_final, &self.hs, "sam_mask_decoder.transformer.norm_final_attn", t, d);
        g.submit(&[], &s);

        // ---- upscaling tail ----
        let mut s = Vec::new();
        self.sam.to_nchw(&mut s, &last.k_b, &self.src_img, d, n_img);
        g.submit(&[], &s);
        let ctx = self.ctx();
        let n1 = d / 4 * 4 * n_img;
        let n2 = dm.mask_dim * dm.hi;
        self.dc1.forward(&ctx, &self.sam.ps, &self.src_img);
        g.submit(&[], &[self.add2(self.dc1.out(), &self.enc.high_res[1], &self.sum1, n1)]);
        self.ln2d.forward(&ctx, &self.sam.ps, &self.sum1);
        g.submit(&[], &[self.sam.act_step(self.ln2d.out(), &self.act1, n1, Act::GeluErf)]);
        self.dc2.forward(&ctx, &self.sam.ps, &self.act1);
        g.submit(&[], &[self.add2(self.dc2.out(), &self.enc.high_res[0], &self.sum2, n2)]);
        g.submit(&[], &[self.sam.act_step(&self.sum2, &self.upscaled, n2, Act::GeluErf)]);

        // ---- hypernetwork MLPs -> per-mask dynamic dot product ----
        let mut s = Vec::new();
        self.sam.to_nlc(&mut s, &self.upscaled, &self.up_nlc, dm.mask_dim, dm.hi);
        for i in 0..dm.nmt as usize {
            // mask token i is `hs` row 2+i ([obj_score, iou, mask x nmt, ...]).
            s.push(gather_rows(g, perm, &self.hs_rows[2 + i], &self.hs, &self.hyper_tok[i], 1, d));
            self.mlp_fwd(&mut s, &self.hyper[i], &self.hyper_tok[i]);
            s.push(scatter_rows(g, perm, &self.hyper_idx[i], self.hyper[i].out(), &self.hyper_in, 1, dm.mask_dim, dm.nmt));
        }
        // masks = hyper_in @ upscaled.view(C, HW): `matmul` computes x @ W^T, so
        // the NLC view of `upscaled` IS the W it wants.
        s.push(g.step(
            self.sam.ids.matmul,
            &[&self.hyper_in, &self.up_nlc, &self.masks_all],
            &[dm.nmt, dm.mask_dim, dm.hi],
            dm.nmt * dm.hi,
        ));
        // ---- IoU head + object-score head ----
        s.push(gather_rows(g, perm, &self.hs_rows[1], &self.hs, &self.iou_tok, 1, d));
        self.mlp_fwd(&mut s, &self.iou_head, &self.iou_tok);
        s.push(gather_rows(g, perm, &self.hs_rows[0], &self.hs, &self.obj_tok, 1, d));
        self.mlp_fwd(&mut s, &self.obj_head, &self.obj_tok);
        g.submit(&[], &s);
        self.fwd_done.set(true);
    }

    /// Mask logits `[nmt, 16*n_img]` from the last forward.
    pub fn masks(&self) -> &DeviceBuffer {
        &self.masks_all
    }

    // =======================================================================
    // loss
    // =======================================================================

    /// Run the forward and reduce the scalar objective.
    ///
    /// The three per-term reductions run on the DEVICE; only `4*nmt + nmt + 1`
    /// floats (21 for SAM 2) cross the bus, and the host arithmetic that
    /// combines them is the constant-size algebra `focal_dice_stats`'s header
    /// specifies — not a per-pixel host loop.
    pub fn loss(&self) -> f32 {
        let g = self.g();
        self.forward();
        let dm = self.dim;
        let t = &self.tgt;
        let s = vec![
            g.step(
                self.bwd.focal_stats,
                &[&self.masks_all, &t.masks, &self.stats],
                &[dm.nmt, dm.hi, f(t.focal_alpha), f(t.focal_gamma)],
                dm.nmt,
            ),
            g.step(self.bwd.mse_value, &[self.iou_head.out(), &t.ious, &self.mse_out], &[dm.nmt], dm.nmt),
            g.step(self.bwd.bce, &[self.obj_head.out(), &t.obj, &self.bce_out], &[1], 1),
        ];
        g.submit(&[], &s);

        let st = g.read(&self.stats, 4 * dm.nmt as usize);
        let mut l = 0.0f32;
        for m in 0..dm.nmt as usize {
            let mw = t.mask_w_host[m];
            if mw == 0.0 {
                continue;
            }
            let focal = st[4 * m] / dm.hi as f32;
            let den = st[4 * m + 1] + st[4 * m + 3] + 1.0;
            let dice = 1.0 - (2.0 * st[4 * m + 2] + 1.0) / den;
            l += mw * (t.w_focal * focal + t.w_dice * dice);
        }
        l += t.w_iou * g.read(&self.mse_out, dm.nmt as usize).iter().sum::<f32>();
        l += t.w_class * g.read(&self.bce_out, 1)[0];
        l
    }

    // =======================================================================
    // backward
    // =======================================================================

    /// Reverse pass over the whole decoder. Parameter gradients ACCUMULATE into
    /// the `ParamStore`, so [`Self::zero_grads`] must have run for this step.
    pub fn backward(&self) {
        if !self.fwd_done.get() {
            self.loss();
        }
        let g = self.g();
        let dm = self.dim;
        let (d, t, n_img) = (dm.d, dm.t, dm.n_img);
        let qn = t * d;
        let kn = n_img * d;
        let n2 = dm.mask_dim * dm.hi;
        let n1 = d / 4 * 4 * n_img;
        let perm = &self.sam.ids.permute;
        let tg = &self.tgt;

        // ---- loss gradients ----
        // `d_iou` / `d_obj` are `axpy` targets (read-modify-write) so they are in
        // this submit's clear list; `d_tokens` and `d_hs` are cleared here too,
        // being the two accumulators the rest of the pass adds into.
        let s = vec![
            g.step(
                self.bwd.focal_grad,
                &[&self.masks_all, &tg.masks, &self.stats, &tg.mask_w, &self.d_masks],
                &[dm.nmt, dm.hi, f(tg.focal_alpha), f(tg.focal_gamma), f(tg.w_focal), f(tg.w_dice)],
                dm.nmt * dm.hi,
            ),
            g.step(self.bwd.mse_grad, &[self.iou_head.out(), &tg.ious, &self.d_iou_raw], &[dm.nmt], dm.nmt),
            self.acc_s(&self.d_iou, &self.d_iou_raw, dm.nmt, tg.w_iou),
            g.step(self.bwd.bce_grad, &[self.obj_head.out(), &tg.obj, &self.d_obj_raw], &[1], 1),
            self.acc_s(&self.d_obj, &self.d_obj_raw, 1, tg.w_class),
        ];
        g.submit(&[&self.d_iou, &self.d_obj, &self.d_tokens, &self.d_hs], &s);

        // ---- masks = hyper_in @ up_nlc^T ----
        // `matmul_dw` ACCUMULATES, and `d_up_nlc` is an activation grad (not a
        // ParamStore buffer), so it is cleared here — the one place it is written.
        let s = vec![
            g.step(
                self.bwd.matmul_dx,
                &[&self.d_masks, &self.up_nlc, &self.d_hyper_in],
                &[dm.nmt, dm.mask_dim, dm.hi, 0],
                dm.nmt * dm.mask_dim,
            ),
            g.step(
                self.bwd.matmul_dw,
                &[&self.d_masks, &self.hyper_in, &self.d_up_nlc],
                &[dm.nmt, dm.mask_dim, dm.hi],
                dm.hi * dm.mask_dim,
            ),
        ];
        g.submit(&[&self.d_up_nlc], &s);

        // ---- upscaling tail (reverse) ----
        let mut s = Vec::new();
        // adjoint of nchw_nlc is nlc_nchw with the same params.
        self.sam.to_nchw(&mut s, &self.d_up_nlc, &self.d_upscaled, dm.mask_dim, dm.hi);
        s.push(self.act_bwd(Act::GeluErf, &self.sum2, &self.d_upscaled, &self.d_sum2, n2));
        g.submit(&[], &s);
        let ctx = self.ctx();
        // `sum2 = dc2_out + high_res[0]` — the high-res branch is frozen.
        self.dc2.backward(&ctx, &self.sam.ps, &self.act1, &self.d_sum2, &self.d_act1);
        g.submit(&[], &[self.act_bwd(Act::GeluErf, self.ln2d.out(), &self.d_act1, &self.d_ln2d, n1)]);
        self.ln2d.backward(&ctx, &self.sam.ps, &self.d_ln2d, &self.d_sum1);
        self.dc1.backward(&ctx, &self.sam.ps, &self.src_img, &self.d_sum1, &self.d_src_img);
        let mut s = Vec::new();
        self.sam.to_nlc(&mut s, &self.d_src_img, &self.d_keys_up, d, n_img);
        g.submit(&[], &s);

        // ---- the three `hs` heads ----
        // Their rows are disjoint, so `row_scatter` into the cleared `d_hs`
        // composes them with no accumulation.
        let d_tok: Vec<DeviceBuffer> = (0..2 + dm.nmt).map(|_| g.storage(d as u64)).collect();
        let d_hyper_row: Vec<DeviceBuffer> = (0..dm.nmt).map(|_| g.storage(dm.mask_dim as u64)).collect();
        let mut s = Vec::new();
        for i in 0..dm.nmt as usize {
            s.push(gather_rows(g, perm, &self.hyper_idx[i], &self.d_hyper_in, &d_hyper_row[i], 1, dm.mask_dim));
            self.mlp_bwd(&mut s, &self.hyper[i], &self.hyper_tok[i], &d_hyper_row[i], &d_tok[2 + i]);
            s.push(scatter_rows(g, perm, &self.hs_rows[2 + i], &d_tok[2 + i], &self.d_hs, 1, d, t));
        }
        self.mlp_bwd(&mut s, &self.iou_head, &self.iou_tok, &self.d_iou, &d_tok[1]);
        s.push(scatter_rows(g, perm, &self.hs_rows[1], &d_tok[1], &self.d_hs, 1, d, t));
        self.mlp_bwd(&mut s, &self.obj_head, &self.obj_tok, &self.d_obj, &d_tok[0]);
        s.push(scatter_rows(g, perm, &self.hs_rows[0], &d_tok[0], &self.d_hs, 1, d, t));
        g.submit(&[], &s);

        // ---- final attention + norm_final_attn ----
        let last = &self.layers[dm.depth as usize - 1];
        let d_sum_final = g.storage(qn as u64);
        let d_q_last = g.storage(qn as u64);
        let d_k_last = g.storage(kn as u64);
        let tmp_k = g.storage(kn as u64);
        let mut s = Vec::new();
        self.ln_bwd(&mut s, &self.sum_final, "sam_mask_decoder.transformer.norm_final_attn", t, d, &self.d_hs, &d_sum_final);
        // sum_final = q_f + final_out, so both branches receive d_sum_final.
        self.attn_bwd(&mut s, &self.final_attn, &self.qf, &self.kf, &last.k_b, &d_sum_final);
        // qf = q_f + tokens ; kf = k_b + key_pe (key_pe frozen) ; v_in = k_b.
        s.push(self.add2(&d_sum_final, &self.final_attn.d_q_in, &d_q_last, qn));
        s.push(self.acc(&self.d_tokens, &self.final_attn.d_q_in, qn));
        s.push(self.add2(&self.final_attn.d_k_in, &self.final_attn.d_v_in, &tmp_k, kn));
        s.push(self.add2(&tmp_k, &self.d_keys_up, &d_k_last, kn));
        g.submit(&[], &s);

        // ---- two-way layers, in reverse ----
        let mut d_qout = d_q_last;
        let mut d_kout = d_k_last;
        for l in (0..dm.depth as usize).rev() {
            let (d_qin, d_kin) = (g.storage(qn as u64), g.storage(kn as u64));
            self.layer_bwd(l, &d_qout, &d_kout, &d_qin, &d_kin);
            d_qout = d_qin;
            d_kout = d_kin;
        }

        // ---- layer 0's inputs: queries = tokens, keys = keys0 ----
        let d_src_in = g.storage(kn as u64);
        let dv_no_mask = g.storage(d as u64);
        let mut s = Vec::new();
        s.push(self.acc(&self.d_tokens, &d_qout, qn));
        // keys0 = nchw_nlc(src_in); src_in = image_embed + dense; dense is the
        // `no_mask_embed` broadcast, whose adjoint is a per-channel sum.
        self.sam.to_nchw(&mut s, &d_kout, &d_src_in, d, n_img);
        s.push(g.step(self.bwd.add_chan_bcast_dv, &[&d_src_in, &dv_no_mask], &[1, d, n_img], d));
        s.push(self.acc(self.gd("sam_prompt_encoder.no_mask_embed.weight"), &dv_no_mask, d));
        g.submit(&[], &s);

        // ---- the output tokens ----
        // `row_scatter`'s adjoint is the `embed` row gather; the gathered rows go
        // through `axpy` rather than being assigned, because a ParamStore grad
        // must ACCUMULATE (it is cleared once per step, not once per use).
        let g_obj = g.storage(d as u64);
        let g_iou = g.storage(d as u64);
        let g_mask = g.storage(dm.nmt as u64 * d as u64);
        let s = vec![
            gather_rows(g, perm, &self.tok_idx[0], &self.d_tokens, &g_obj, 1, d),
            self.acc(self.gd("sam_mask_decoder.obj_score_token.weight"), &g_obj, d),
            gather_rows(g, perm, &self.tok_idx[1], &self.d_tokens, &g_iou, 1, d),
            self.acc(self.gd("sam_mask_decoder.iou_token.weight"), &g_iou, d),
            gather_rows(g, perm, &self.tok_idx[2], &self.d_tokens, &g_mask, dm.nmt, d),
            self.acc(self.gd("sam_mask_decoder.mask_tokens.weight"), &g_mask, dm.nmt * d),
        ];
        g.submit(&[], &s);
    }

    /// One `TwoWayAttentionBlock`, in reverse. `d_qout` is the gradient reaching
    /// this layer's `queries` OUTPUT from everything after it, `d_kout` the same
    /// for its `keys` output; `d_qin` / `d_kin` receive the gradients w.r.t. its
    /// two inputs.
    fn layer_bwd(&self, l: usize, d_qout: &DeviceBuffer, d_kout: &DeviceBuffer, d_qin: &DeviceBuffer, d_kin: &DeviceBuffer) {
        let g = self.g();
        let dm = self.dim;
        let (d, t, n_img) = (dm.d, dm.t, dm.n_img);
        let (qn, kn) = (t * d, n_img * d);
        let lc = &self.layers[l];
        let (q_in, k_in): (&DeviceBuffer, &DeviceBuffer) = if l == 0 {
            (&self.tokens, &self.keys0)
        } else {
            (&self.layers[l - 1].q_f, &self.layers[l - 1].k_b)
        };

        let d_k_a = g.storage(kn as u64);
        let d_q_f1 = g.storage(qn as u64);
        let d_q_f = g.storage(qn as u64);
        let d_q_e = g.storage(qn as u64);
        let d_mlp_x = g.storage(qn as u64);
        let d_q_d = g.storage(qn as u64);
        let d_q_c = g.storage(qn as u64);
        let d_q_b = g.storage(qn as u64);
        let d_q_a = g.storage(qn as u64);
        let d_kpe = g.storage(kn as u64);
        let d_k_mid = g.storage(kn as u64);

        // ---- (4) image -> tokens ----
        let mut s = Vec::new();
        self.ln_bwd(&mut s, &lc.k_a, &format!("{}.norm4", lc.p), n_img, d, d_kout, &d_k_a);
        // k_a = k_in + o3
        self.attn_bwd(&mut s, &lc.i2t, &lc.kpe, &lc.q3, &lc.q_f, &d_k_a);
        // q3 = q_f + tokens; the i2t V input IS q_f.
        s.push(self.add2(d_qout, &lc.i2t.d_v_in, &d_q_f1, qn));
        s.push(self.add2(&d_q_f1, &lc.i2t.d_k_in, &d_q_f, qn));
        s.push(self.acc(&self.d_tokens, &lc.i2t.d_k_in, qn));

        // ---- (3) MLP ----
        self.ln_bwd(&mut s, &lc.q_e, &format!("{}.norm3", lc.p), t, d, &d_q_f, &d_q_e);
        self.mlp_bwd(&mut s, &lc.mlp, &lc.q_d, &d_q_e, &d_mlp_x);
        s.push(self.add2(&d_q_e, &d_mlp_x, &d_q_d, qn));

        // ---- (2) tokens -> image ----
        self.ln_bwd(&mut s, &lc.q_c, &format!("{}.norm2", lc.p), t, d, &d_q_d, &d_q_c);
        self.attn_bwd(&mut s, &lc.t2i, &lc.q2, &lc.kpe, k_in, &d_q_c);
        // q2 = q_b + tokens
        s.push(self.add2(&d_q_c, &lc.t2i.d_q_in, &d_q_b, qn));
        s.push(self.acc(&self.d_tokens, &lc.t2i.d_q_in, qn));
        // kpe = k_in + key_pe, read by BOTH the t2i K projection and the i2t Q one.
        s.push(self.add2(&lc.i2t.d_q_in, &lc.t2i.d_k_in, &d_kpe, kn));
        // k_in also feeds the t2i V projection and the (4) residual.
        s.push(self.add2(&d_kpe, &lc.t2i.d_v_in, &d_k_mid, kn));
        s.push(self.add2(&d_k_mid, &d_k_a, d_kin, kn));

        // ---- (1) token self-attention ----
        self.ln_bwd(&mut s, &lc.q_a, &format!("{}.norm1", lc.p), t, d, &d_q_b, &d_q_a);
        if l == 0 {
            // `skip_first_layer_pe`: q_a == o1 and q == k == v == q_in.
            self.attn_bwd(&mut s, &lc.self_attn, q_in, q_in, q_in, &d_q_a);
            let acc = g.storage(qn as u64);
            s.push(self.add2(&lc.self_attn.d_q_in, &lc.self_attn.d_k_in, &acc, qn));
            s.push(self.add2(&acc, &lc.self_attn.d_v_in, d_qin, qn));
            g.submit(&[], &s);
        } else {
            // qs = q_in + tokens; q_a = q_in + o1; the V input is q_in itself.
            self.attn_bwd(&mut s, &lc.self_attn, &lc.qs, &lc.qs, q_in, &d_q_a);
            let d_qs = g.storage(qn as u64);
            let acc = g.storage(qn as u64);
            s.push(self.add2(&lc.self_attn.d_q_in, &lc.self_attn.d_k_in, &d_qs, qn));
            s.push(self.acc(&self.d_tokens, &d_qs, qn));
            s.push(self.add2(&d_q_a, &d_qs, &acc, qn));
            s.push(self.add2(&acc, &lc.self_attn.d_v_in, d_qin, qn));
            g.submit(&[], &s);
        }
    }

    // =======================================================================
    // CheckModel surface
    // =======================================================================

    pub fn param_names(&self) -> Vec<String> {
        self.sam.ps.trainable.iter().map(|(n, _)| n.clone()).collect()
    }
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.sam.ps.read_weight(&self.sam.gpu, name)
    }
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.sam.gpu.write(self.sam.ps.w(name), bytemuck::cast_slice(data));
        self.fwd_done.set(false);
    }
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.sam.ps.read_grad(&self.sam.gpu, name)
    }
    pub fn zero_grads(&self) {
        self.sam.ps.zero_grads(&self.sam.gpu);
    }
}
