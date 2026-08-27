// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SAM 2 image-path forward graph.
//!
//! Composed from the SHARED blocks, never from private copies:
//!   * `model::vit` - `WindowPlan`/`WindowIndex` (window partition/reverse as a
//!     row permutation), `QPoolPlan`/`QPoolCache`/`q_pool_fwd` (Hiera's
//!     `MaxPool2d` on the attention query only) and `cross_q_fwd` (attention
//!     with independent q and kv extents - the reason `q_pool` works at all);
//!   * `model::block` - `chunked_bidir_fwd` (span + query-chunked bidirectional
//!     attention, used for every block whose query is NOT pooled) and the
//!     `LayerNormIds` seam that picks the coalesced `layernorm_rows`;
//!   * `vision::blocks` - `Conv` (patch embed, FPN laterals, `conv_s0`/`conv_s1`,
//!     the mask-prompt downsampling), `ConvTranspose` (the mask decoder's two 2x
//!     upscalings) and `LayerNorm2d` (channels-first, eps 1e-6).
//!
//! SSA: every Hiera block writes a FRESH `[rows_out, dim_out]` output buffer -
//! which is exactly what the backward will need to cache - and the neck, prompt
//! encoder and decoder allocate a fresh buffer per stage throughout. Intra-block
//! temporaries are per-block allocations that drop when the block ends.
//!
//! ## Window padding
//!
//! `window_partition` zero-pads the token grid bottom/right when the window does
//! not divide it (hiera-tiny's 14/7 windows on 64²/32² grids; hiera-large at
//! 1024 never pads). The pad happens AFTER `norm1`, so the padded tokens are
//! zeros entering `qkv` - which means their keys and values are the qkv BIAS,
//! not zero, and they genuinely participate in their window's softmax. That is
//! reproduced exactly here: `WindowPlan` deliberately never pads (it emits a
//! short final window, which is a DIFFERENT operator), so this module zero-pads
//! the token buffer with a cleared `row_scatter`, partitions the padded grid
//! uniformly, and gathers the valid rows back out afterwards.

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::block::{self, CrossIds, LayerNormIds};
use model::vit::{
    cross_q_fwd, q_pool_fwd, row_index_buffer, scatter_rows, window_partition, window_reverse,
    AttnSpan, QPoolCache, QPoolPlan, VitPermuteIds, VitQPoolIds, VitShape, WindowIndex, WindowPlan,
};
use paramstore::{ParamStore, Role};
use vision::{Act, Conv, ConvNames, ConvSpec, ConvTrSpec, ConvTranspose, Ctx, LayerNorm2d, Ln2dNames, Norm, Shape};

use crate::config::{BlockSpec, Sam2Config};
use crate::hostpe;
use crate::import::{self, Scope, Tensors, NO_MEM_EMBED};

/// Kernels this model dispatches, by name. `vision::ConvKernelIds::resolve` and
/// `Gpu::kernel_index` both key on the NAME, so the order here is irrelevant -
/// nothing in this crate holds a positional kernel index.
pub const PIPELINES: &[(&str, &str)] = &[
    ("conv_bias", kernels::CONV_BIAS),
    ("convtr2d", kernels::CONVTR2D),
    ("add_chan_inplace", kernels::ADD_CHAN_INPLACE),
    ("layernorm", kernels::LAYERNORM),
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("ln_stats", kernels::LN_STATS),
    ("ln_stats_rows", kernels::LN_STATS_ROWS),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    ("layernorm_dx_rows", kernels::LAYERNORM_DX_ROWS),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("gelu_erf", kernels::GELU_ERF),
    ("gelu_erf_bwd", kernels::GELU_ERF_BWD),
    ("leaky_relu", kernels::LEAKY_RELU),
    ("leaky_relu_bwd", kernels::LEAKY_RELU_BWD),
    ("sigmoid", kernels::SIGMOID),
    ("sigmoid_bwd", kernels::SIGMOID_BWD),
    ("matmul", kernels::MATMUL),
    ("matmul_rows", kernels::MATMUL_ROWS),
    ("bias_add", kernels::BIAS_ADD),
    ("add2", kernels::ADD2),
    ("embed", kernels::EMBED),
    ("row_scatter", kernels::ROW_SCATTER),
    ("maxpool2d", kernels::MAXPOOL2D),
    ("maxpool2d_dx", kernels::MAXPOOL2D_DX),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("resize_bicubic", kernels::RESIZE_BICUBIC),
    ("resize_bilinear", kernels::RESIZE_BILINEAR),
    ("resize_nearest", kernels::RESIZE_NEAREST),
    ("axpy", kernels::AXPY),
    // `imaging::Ctx`'s per-channel affine - brain's ONE normalise/denormalise
    // kernel, used by [`Sam2::preprocess`].
    ("film_chan", kernels::FILM_CHAN),
    // ---- backward half (dispatched by [`crate::train`] only) ----
    // Registering them here rather than in a second pipeline list keeps ONE
    // kernel index space for the crate: `vision::ConvKernelIds::resolve` and
    // `Gpu::kernel_index` both key on the name, so the forward is unaffected.
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("bias_grad", kernels::BIAS_GRAD),
    ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA),
    ("layernorm_dbeta", kernels::LAYERNORM_DBETA),
    ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS),
    ("attn_bwd_dv_cross", kernels::ATTN_BWD_DV_CROSS),
    ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS),
    ("attn_bwd_dk_cross", kernels::ATTN_BWD_DK_CROSS),
    ("convtr2d_dx", kernels::CONVTR2D_DX),
    ("convtr2d_dw", kernels::CONVTR2D_DW),
    ("add_chan_bcast_dv", kernels::ADD_CHAN_BCAST_DV),
    // loss heads: focal + dice on the mask logits, MSE on the IoU head,
    // BCE-with-logits on the object score.
    ("focal_dice_stats", kernels::FOCAL_DICE_STATS),
    ("focal_dice_grad", kernels::FOCAL_DICE_GRAD),
    ("mse_value", kernels::MSE_VALUE),
    ("mse_grad", kernels::MSE_GRAD),
    ("bce_logits", kernels::BCE_LOGITS),
    ("bce_logits_grad", kernels::BCE_LOGITS_GRAD),
    // ---- coalesced cross-attention scores ----
    // `attn_scores_cross` reads the fused KV slab with the KEY index as the
    // fastest thread index, so every lane of a warp lands on its own cache
    // line. Transposing K to key-minor once per span buys the same sweep
    // coalesced loads.
    ("kv_k_headt", kernels::KV_K_HEADT),
    ("attn_scores_cross_kt", kernels::ATTN_SCORES_CROSS_KT),
    // ---- video memory bank (`crate::video`) ----
    // The memory encoder's ConvNeXt fuser is DEPTHWISE, and `backend-cpu` binds
    // its dense fast path to the NAME `conv2d` - a grouped conv must reach
    // `conv2d_gd`, or it would silently convolve as if dense.
    ("conv2d", kernels::CONV2D),
    ("conv2d_gd", kernels::CONV2D_GD),
    ("conv2d_gd_reg", kernels::CONV2D_GD_REG),
    // SAM 2's axial 2D RoPE rotates the INTERLEAVED pair (2j, 2j+1) against a
    // host table - `rope2d` is the rotate-half sibling and is a different
    // operator, not a faster spelling of this one.
    ("rope_interleave_table", kernels::ROPE_INTERLEAVE_TABLE),
];

/// Query rows per `chunked_bidir_fwd` dispatch. A multiple of 64 is REQUIRED,
/// not preferred: that path binds `qkv` sliced at `(row0 + q0) * 3 * dim_out`
/// floats, and wgpu rejects a storage binding whose offset is not 256-byte
/// aligned. 256 also caps the largest score slab (a 4096-token global-attention
/// block at 8 heads) at 32 MiB.
const ATTN_CHUNK: u32 = 256;

/// Largest `N*C*H*W` one `maxpool2d` dispatch may cover. The kernel records the
/// winning input index in an f32 - exact only below 2^24 - and `QPoolCache`
/// asserts it. hiera-large's block 2 pools 65536 rows x 288 channels = 18.9 M,
/// over the bound, so its `q_pool` runs in WINDOW CHUNKS: same shared
/// `q_pool_fwd`, one call per chunk, each under the limit.
const MAXPOOL_ELEMS: u64 = 1 << 24;

/// Pipeline indices resolved by NAME (never by position - see `vision::ids`).
pub(crate) struct Ids {
    pub(crate) matmul: usize,
    pub(crate) matmul_rows: usize,
    pub(crate) bias_add: usize,
    pub(crate) add2: usize,
    pub(crate) axpy: usize,
    pub(crate) gelu_erf: usize,
    pub(crate) leaky_relu: usize,
    pub(crate) sigmoid: usize,
    pub(crate) nchw_nlc: usize,
    pub(crate) nlc_nchw: usize,
    maxpool2d: usize,
    pub(crate) add_chan_inplace: usize,
    resize_bicubic: usize,
    resize_bilinear: usize,
    resize_nearest: usize,
    pub(crate) cross: CrossIds,
    /// `(kv_k_headt, attn_scores_cross_kt)` - the coalesced score path.
    pub(crate) key_minor: (usize, usize),
    pub(crate) permute: VitPermuteIds,
    qpool: VitQPoolIds,
    pub(crate) ln: LayerNormIds,
}

pub(crate) fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("sam2: kernel {name} is not in PIPELINES"))
}

impl Ids {
    /// The coalesced score path bound to `kt` (`[dim_out, max_kn]`).
    pub(crate) fn key_minor<'a>(&self, kt: &'a DeviceBuffer) -> block::KeyMinor<'a> {
        block::KeyMinor { transpose: self.key_minor.0, scores: self.key_minor.1, kt }
    }
    fn resolve(g: &Gpu) -> Ids {
        let permute = VitPermuteIds { embed: idx(g, "embed"), row_scatter: idx(g, "row_scatter") };
        Ids {
            matmul: idx(g, "matmul"),
            matmul_rows: idx(g, "matmul_rows"),
            bias_add: idx(g, "bias_add"),
            add2: idx(g, "add2"),
            axpy: idx(g, "axpy"),
            gelu_erf: idx(g, "gelu_erf"),
            leaky_relu: idx(g, "leaky_relu"),
            sigmoid: idx(g, "sigmoid"),
            nchw_nlc: idx(g, "nchw_nlc"),
            nlc_nchw: idx(g, "nlc_nchw"),
            maxpool2d: idx(g, "maxpool2d"),
            add_chan_inplace: idx(g, "add_chan_inplace"),
            resize_bicubic: idx(g, "resize_bicubic"),
            resize_bilinear: idx(g, "resize_bilinear"),
            resize_nearest: idx(g, "resize_nearest"),
            cross: CrossIds {
                scores: idx(g, "attn_scores_cross"),
                softmax: idx(g, "attn_softmax_cross"),
                apply: idx(g, "attn_apply_cross"),
            },
            key_minor: (idx(g, "kv_k_headt"), idx(g, "attn_scores_cross_kt")),
            permute,
            qpool: VitQPoolIds {
                permute,
                nlc_nchw: idx(g, "nlc_nchw"),
                nchw_nlc: idx(g, "nchw_nlc"),
                maxpool2d: idx(g, "maxpool2d"),
                maxpool2d_dx: idx(g, "maxpool2d_dx"),
            },
            // Full resolve (forward + backward): `ln_stats`, `layernorm_dx` and
            // the coalesced `*_rows` twins are all in PIPELINES, so the
            // forward-only variant would leave the training path's LN backward
            // pointing at the forward kernel.
            ln: LayerNormIds::resolve(g, idx(g, "layernorm"), idx(g, "ln_stats"), idx(g, "layernorm_dx")),
        }
    }
}

/// Host copies of the few small tensors the prompt encoder and object-pointer
/// head need on the CPU (none is larger than 4x256 floats).
struct HostConsts {
    gauss: Vec<f32>,
    point_embeddings: [Vec<f32>; 4],
    not_a_point: Vec<f32>,
    iou_token: Vec<f32>,
    mask_tokens: Vec<f32>,
    obj_score_token: Vec<f32>,
    no_obj_ptr: Vec<f32>,
}

/// Hiera trunk + FPN neck + SAM prompt encoder + two-way mask decoder.
pub struct Sam2 {
    pub gpu: Gpu,
    pub cfg: Sam2Config,
    /// Which half of the checkpoint this instance's [`ParamStore`] holds. The
    /// video path asserts [`Scope::Video`] rather than letting `ps.w()` panic on
    /// a name that was never allocated.
    pub scope: Scope,
    pub ps: ParamStore,
    pub(crate) ids: Ids,
    pub(crate) conv_ids: vision::ConvKernelIds,
    host: HostConsts,
}

/// Every image-encoder tap, one buffer per stage so the parity ladder can be
/// climbed rung by rung.
pub struct Encoded {
    /// `[H*W, C]` NLC - the interpolated + tiled Hiera position embedding.
    pub pos_embed: DeviceBuffer,
    /// `[H*W, C]` NLC - `patch_embed`'s output.
    pub patch_embed: DeviceBuffer,
    /// Per-block output `[rows_out, dim_out]` NLC (SSA: one buffer per block).
    pub blocks: Vec<DeviceBuffer>,
    /// The 4 stage outputs, NCHW.
    pub trunk_feats: Vec<DeviceBuffer>,
    /// FPN lateral conv outputs by LEVEL (0 = highest resolution).
    pub lateral: Vec<DeviceBuffer>,
    /// FPN outputs after top-down fusion, by level.
    pub fpn: Vec<DeviceBuffer>,
    /// `PositionEmbeddingSine` per level. The image path never consumes these
    /// (they feed the video memory attention) but they are part of the neck's
    /// output contract and are goldened.
    pub pos_sine: Vec<DeviceBuffer>,
    /// `conv_s0(fpn0)` and `conv_s1(fpn1)`.
    pub high_res: Vec<DeviceBuffer>,
    /// `fpn[2] + no_mem_embed`, NCHW `[1, C, h, w]`.
    pub image_embed: DeviceBuffer,
}

/// A prompt for one image. A BOX is two points with labels 2 and 3 - the image
/// path never passes `boxes=` to the reference prompt encoder, so there is one
/// code path, not two.
pub struct Prompt {
    /// `(x, y)` in the `image_size` frame.
    pub coords: Vec<(f32, f32)>,
    /// 1 = foreground, 0 = background, 2/3 = box corners.
    pub labels: Vec<f32>,
    /// `[1, 1, 256, 256]` mask logits, ALREADY at `mask_input_size`. The
    /// reference downsamples a full-resolution mask with
    /// `bilinear, antialias=True`, which brain has no kernel for - see the
    /// crate docs.
    pub mask_lowres: Option<Vec<f32>>,
    pub multimask_output: bool,
}

/// Every mask-decoder tap, named as in the goldens.
pub struct Decoded {
    pub sparse: DeviceBuffer,
    pub dense: DeviceBuffer,
    pub dense_pe: DeviceBuffer,
    pub tokens: DeviceBuffer,
    pub src_in: DeviceBuffer,
    /// `(queries, keys)` leaving each two-way layer.
    pub twoway: Vec<(DeviceBuffer, DeviceBuffer)>,
    pub final_attn_out: DeviceBuffer,
    pub hs: DeviceBuffer,
    pub src_out: DeviceBuffer,
    pub dc1_out: DeviceBuffer,
    pub dc2_out: DeviceBuffer,
    pub upscaled_embedding: DeviceBuffer,
    pub hyper_in: DeviceBuffer,
    pub masks_all: DeviceBuffer,
    pub iou_all: DeviceBuffer,
    pub object_score_logits: DeviceBuffer,
    pub low_res_multimasks: DeviceBuffer,
    pub high_res_multimasks: DeviceBuffer,
    pub obj_ptr: DeviceBuffer,
    pub ious: Vec<f32>,
    pub best_iou_index: usize,
    pub n_masks: u32,
}

impl Sam2 {
    /// Build on an existing device handle from the map [`crate::import::import`]
    /// produced. Every parameter is `Role::Frozen`: this is a forward-parity
    /// port, and the trainable roles arrive with the backward workstream.
    pub fn new(gpu: Gpu, cfg: Sam2Config, weights: &Tensors) -> Sam2 {
        Sam2::new_with_roles(gpu, cfg, weights, &|_| false)
    }

    /// [`Sam2::new`] holding the WHOLE checkpoint - image path plus memory bank.
    /// Required by [`crate::video`]; `weights` must come from
    /// `import::import_scoped(.., Scope::Video)`.
    pub fn new_video(gpu: Gpu, cfg: Sam2Config, weights: &Tensors) -> Sam2 {
        Sam2::new_scoped(gpu, cfg, weights, Scope::Video, &|_| false)
    }

    /// [`Sam2::new`] with a per-tensor trainability predicate. `trainable(name)`
    /// picks the parameters that get gradient + AdamW buffers; everything else
    /// stays `Role::Frozen` (weight buffer only). `crate::train` uses this to
    /// build the mask-decoder-finetune role set - the trunk and neck stay frozen
    /// and allocate no optimiser state at all.
    pub fn new_with_roles(gpu: Gpu, cfg: Sam2Config, weights: &Tensors, trainable: &dyn Fn(&str) -> bool) -> Sam2 {
        Sam2::new_scoped(gpu, cfg, weights, Scope::Image, trainable)
    }

    /// [`Sam2::new_with_roles`] at an explicit [`Scope`].
    pub fn new_scoped(
        gpu: Gpu,
        cfg: Sam2Config,
        weights: &Tensors,
        scope: Scope,
        trainable: &dyn Fn(&str) -> bool,
    ) -> Sam2 {
        let params: Vec<(String, usize, Role)> = import::param_list_scoped(&cfg, scope)
            .into_iter()
            .map(|(n, c)| {
                let r = if trainable(&n) { Role::Trainable } else { Role::Frozen };
                (n, c, r)
            })
            .collect();
        let init: HashMap<String, Vec<f32>> = import::init_map(weights);
        let ps = ParamStore::new_with_roles(&gpu, params, &init);
        let ids = Ids::resolve(&gpu);
        let conv_ids = vision::ConvKernelIds::resolve(PIPELINES);
        let take = |n: &str| -> Vec<f32> { weights.get(n).unwrap_or_else(|| panic!("sam2: missing {n}")).1.clone() };
        let host = HostConsts {
            gauss: take("sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"),
            point_embeddings: [
                take("sam_prompt_encoder.point_embeddings.0.weight"),
                take("sam_prompt_encoder.point_embeddings.1.weight"),
                take("sam_prompt_encoder.point_embeddings.2.weight"),
                take("sam_prompt_encoder.point_embeddings.3.weight"),
            ],
            not_a_point: take("sam_prompt_encoder.not_a_point_embed.weight"),
            iou_token: take("sam_mask_decoder.iou_token.weight"),
            mask_tokens: take("sam_mask_decoder.mask_tokens.weight"),
            obj_score_token: take("sam_mask_decoder.obj_score_token.weight"),
            no_obj_ptr: take(NO_OBJ_PTR),
        };
        assert_eq!(host.gauss.len(), cfg.d_model as usize, "gaussian matrix is [2, d/2]");
        Sam2 { gpu, cfg, scope, ps, ids, conv_ids, host }
    }

    pub(crate) fn ctx(&self) -> Ctx<'_> {
        Ctx::new(&self.gpu, &self.conv_ids)
    }

    // -----------------------------------------------------------------------
    // dispatch helpers
    // -----------------------------------------------------------------------

    /// `out[rows, n] = x[rows, k] @ W[n, k]^T + b[n]` - the `nn.Linear` pair.
    /// `matmul_rows` is bit-identical to `matmul` (per its own header) and loads
    /// each weight row once per 8 output rows; the trunk's `[65536, C]` inputs
    /// are exactly the shape that motivated it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear(&self, s: &mut Vec<Step>, x: &DeviceBuffer, out: &DeviceBuffer, rows: u32, k: u32, n: u32, w: &str, b: &str) {
        s.push(self.gpu.step(self.ids.matmul_rows, &[x, self.ps.w(w), out], &[rows, k, n], rows.div_ceil(8) * n));
        s.push(self.gpu.step(self.ids.bias_add, &[out, self.ps.w(b)], &[rows, n], rows * n));
    }

    pub(crate) fn act_step(&self, x: &DeviceBuffer, y: &DeviceBuffer, n: u32, act: Act) -> Step {
        match act {
            // ReLU is `leaky_relu` at slope 0 - identical, and costs no kernel.
            Act::Relu => self.gpu.step(self.ids.leaky_relu, &[x, y], &[n, f(0.0)], n),
            Act::GeluErf => self.gpu.step(self.ids.gelu_erf, &[x, y], &[n], n),
            Act::Sigmoid => self.gpu.step(self.ids.sigmoid, &[x, y], &[n], n),
            other => panic!("sam2: unexpected activation {other:?}"),
        }
    }

    /// `sam2_utils.MLP`: `layers.{0..L-1}`, activation on every layer but the
    /// last, optional sigmoid on the output.
    fn mlp(&self, prefix: &str, x: &DeviceBuffer, rows: u32, dims: &[u32], act: Act, sigmoid: bool) -> DeviceBuffer {
        let mut owned: Vec<DeviceBuffer> = Vec::new();
        let mut steps = Vec::new();
        let last = dims.len() - 2;
        let mut cur_is_input = true;
        for i in 0..=last {
            let (k, n) = (dims[i], dims[i + 1]);
            let y = self.gpu.storage(rows as u64 * n as u64);
            let src: &DeviceBuffer = if cur_is_input { x } else { owned.last().unwrap() };
            self.linear(&mut steps, src, &y, rows, k, n, &format!("{prefix}.layers.{i}.weight"), &format!("{prefix}.layers.{i}.bias"));
            cur_is_input = false;
            if i < last {
                let a = self.gpu.storage(rows as u64 * n as u64);
                steps.push(self.act_step(&y, &a, rows * n, act));
                owned.push(y);
                owned.push(a);
            } else if sigmoid {
                let a = self.gpu.storage(rows as u64 * n as u64);
                steps.push(self.act_step(&y, &a, rows * n, Act::Sigmoid));
                owned.push(y);
                owned.push(a);
            } else {
                owned.push(y);
            }
        }
        self.gpu.submit(&[], &steps);
        owned.pop().expect("mlp produced no output")
    }

    /// `[N, L, C] -> [N, C, H, W]`.
    pub(crate) fn to_nchw(&self, s: &mut Vec<Step>, x: &DeviceBuffer, y: &DeviceBuffer, c: u32, hw: u32) {
        let t = c * hw;
        s.push(self.gpu.step(self.ids.nlc_nchw, &[x, y], &[t, c, hw], t));
    }
    /// `[N, C, H, W] -> [N, L, C]`.
    pub(crate) fn to_nlc(&self, s: &mut Vec<Step>, x: &DeviceBuffer, y: &DeviceBuffer, c: u32, hw: u32) {
        let t = c * hw;
        s.push(self.gpu.step(self.ids.nchw_nlc, &[x, y], &[t, c, hw], t));
    }

    pub(crate) fn upload(&self, label: &str, data: &[f32]) -> DeviceBuffer {
        self.gpu.storage_init(label, data)
    }

    /// SSA copy into a fresh buffer. `axpy` is `out += s*in` (read-modify-write),
    /// so the destination goes in the submit's CLEAR list - relying on a fresh
    /// allocation being zeroed would be a backend-dependent assumption.
    pub(crate) fn copy_of(&self, src: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let out = self.gpu.storage(n as u64);
        self.gpu.submit(&[&out], &[self.gpu.step(self.ids.axpy, &[&out, src], &[n, f(1.0)], n)]);
        out
    }

    // =======================================================================
    // Hiera trunk + FPN neck
    // =======================================================================

    /// SAM 2's preprocessing tail: `(x - pixel_mean) / pixel_std` over an RGB
    /// image already in `[0, 1]` and already at `image_size` (letterboxing and
    /// decode belong to `crates/imaging` and to the serving workstream).
    ///
    /// Dispatched through `imaging::Ctx`, which owns the workspace's single
    /// per-channel affine (`film_chan`) - there is no host pixel loop here.
    pub fn preprocess(&self, rgb01: &[f32]) -> DeviceBuffer {
        let s = self.cfg.image_size;
        assert_eq!(rgb01.len(), 3 * (s * s) as usize, "preprocess wants a [3,{s},{s}] RGB image in [0,1]");
        let ctx = imaging::Ctx::new(&self.gpu);
        let x = ctx.upload("sam2_rgb01", rgb01);
        let norm = imaging::Normalization { mean: self.cfg.pixel_mean, std: self.cfg.pixel_std };
        ctx.normalize(&x, Shape::new(1, 3, s, s), &norm)
    }

    /// `image` is the NORMALIZED model input, NCHW `[1, 3, S, S]`.
    pub fn encode_image(&self, image: &[f32]) -> Encoded {
        let s = self.cfg.image_size;
        assert_eq!(image.len(), 3 * (s * s) as usize, "image must be a normalized [1,3,{s},{s}] map");
        self.encode(&self.upload("sam2_image", image))
    }

    /// [`Self::encode_image`] from a device buffer - what [`Self::preprocess`]
    /// hands back.
    pub fn encode(&self, img: &DeviceBuffer) -> Encoded {
        let cfg = &self.cfg;
        let s = cfg.image_size;
        let e = cfg.embed_dim;
        let grid = cfg.trunk_grid();
        let rows0 = grid * grid;

        // ---- patch embed: Conv2d(3, C, 7, stride 4, pad 3) ----
        let ctx = self.ctx();
        let pfx = "image_encoder.trunk.patch_embed.proj";
        let patch = Conv::with_names(
            &ctx,
            pfx,
            ConvNames::torch_flat(pfx),
            Shape::new(1, 3, s, s),
            ConvSpec::relu(e, cfg.patch_kernel, cfg.patch_stride, cfg.patch_pad)
                .with_norm(Norm::None)
                .with_act(Act::None)
                .with_bias(),
            false,
        );
        patch.forward(&ctx, &self.ps, img);

        let mut steps = Vec::new();
        let patch_embed = self.gpu.storage(rows0 as u64 * e as u64);
        self.to_nlc(&mut steps, patch.out(), &patch_embed, e, rows0);

        // ---- position embedding: bicubic upsample + tiled window embed ----
        let (ph, pw) = cfg.window_pos_embed_bkg_spatial_size;
        let interp = self.gpu.storage(e as u64 * rows0 as u64);
        steps.push(self.gpu.step(
            self.ids.resize_bicubic,
            &[self.ps.w("image_encoder.trunk.pos_embed"), &interp],
            // torch's F.interpolate(mode="bicubic") defaults to align_corners=False.
            &[1, e, ph, pw, grid, grid, 0],
            e * rows0,
        ));
        // `pos_embed_window.tile(...)` is index arithmetic over a CONSTANT, so it
        // is expanded once on the host and uploaded; the add is a dispatch.
        let win = self.ps.read_weight(&self.gpu, "image_encoder.trunk.pos_embed_window");
        let w0 = cfg.window_spec[0];
        let tiled = self.upload("sam2_pos_window_tiled", &hostpe::tile_chw(&win, e, w0, w0, grid, grid));
        let pos_nchw = self.gpu.storage(e as u64 * rows0 as u64);
        steps.push(self.gpu.step(self.ids.add2, &[&interp, &tiled, &pos_nchw], &[e * rows0], e * rows0));
        let pos_embed = self.gpu.storage(rows0 as u64 * e as u64);
        self.to_nlc(&mut steps, &pos_nchw, &pos_embed, e, rows0);

        let x0 = self.gpu.storage(rows0 as u64 * e as u64);
        steps.push(self.gpu.step(self.ids.add2, &[&patch_embed, &pos_embed, &x0], &[rows0 * e], rows0 * e));
        self.gpu.submit(&[], &steps);
        drop(patch);

        // ---- the MultiScaleBlock stack ----
        let specs = cfg.blocks();
        let mut blocks: Vec<DeviceBuffer> = Vec::with_capacity(specs.len());
        for (i, b) in specs.iter().enumerate() {
            let rows_out = b.out_hw.0 * b.out_hw.1;
            let out = self.gpu.storage(rows_out as u64 * b.dim_out as u64);
            {
                let input: &DeviceBuffer = if i == 0 { &x0 } else { &blocks[i - 1] };
                self.hiera_block(b, input, &out);
            }
            blocks.push(out);
        }

        // ---- stage outputs, permuted to NCHW ----
        let mut steps = Vec::new();
        let mut trunk_feats = Vec::new();
        for &end in cfg.stage_ends().iter() {
            let b = specs[end as usize];
            let hw = b.out_hw.0 * b.out_hw.1;
            let t = self.gpu.storage(hw as u64 * b.dim_out as u64);
            self.to_nchw(&mut steps, &blocks[end as usize], &t, b.dim_out, hw);
            trunk_feats.push(t);
        }
        self.gpu.submit(&[], &steps);

        self.neck(patch_embed, pos_embed, blocks, trunk_feats)
    }

    /// One `MultiScaleBlock`. Submits as it goes (three phases: pre-window,
    /// zero-pad, everything else) because the zero-pad needs its own submit with
    /// a clear list.
    fn hiera_block(&self, b: &BlockSpec, x: &DeviceBuffer, out: &DeviceBuffer) {
        let g = &self.gpu;
        let cfg = &self.cfg;
        let p = format!("image_encoder.trunk.blocks.{}", b.index);
        let (c, co) = (b.dim, b.dim_out);
        let (h, w) = b.in_hw;
        let (ho, wo) = b.out_hw;
        let (rows_in, rows_out) = (h * w, ho * wo);
        let ws = b.window_size;
        let (hp, wp) = b.pad_hw();
        let rows_p = hp * wp;
        let ws_out = b.out_window();
        let (hpo, wpo) = b.out_pad_hw();
        let rows_po = hpo * wpo;
        let heads = b.num_heads;
        let sh = VitShape { dim: co, heads, mlp: co * cfg.mlp_ratio, eps: cfg.trunk_eps };

        // ---- phase 1: norm1, and the (projected, pooled) residual shortcut ----
        let mut steps = Vec::new();
        let ln1 = g.storage(rows_in as u64 * c as u64);
        steps.push(block::layernorm_fwd(
            g,
            &self.ids.ln,
            x,
            self.ps.w(&format!("{p}.norm1.weight")),
            self.ps.w(&format!("{p}.norm1.bias")),
            &ln1,
            c,
            rows_in,
            cfg.trunk_eps,
        ));
        // `shortcut = do_pool(self.proj(norm1(x)), self.pool)` - note the
        // projection reads norm1's OUTPUT, not the block input, and the pool runs
        // over the FULL grid (not per window).
        let mut sc_owned: Option<DeviceBuffer> = None;
        if c != co {
            let proj = g.storage(rows_in as u64 * co as u64);
            self.linear(&mut steps, &ln1, &proj, rows_in, c, co, &format!("{p}.proj.weight"), &format!("{p}.proj.bias"));
            if b.q_pool {
                let nchw = g.storage(rows_in as u64 * co as u64);
                self.to_nchw(&mut steps, &proj, &nchw, co, rows_in);
                let pooled = g.storage(rows_out as u64 * co as u64);
                // `argmax` is written but never read on the forward path; it is
                // exact only below 2^24 elements, so the BACKWARD will have to
                // chunk this dispatch the way `q_pool` below already does.
                let argmax = g.storage(rows_out as u64 * co as u64);
                steps.push(g.step(
                    self.ids.maxpool2d,
                    &[&nchw, &pooled, &argmax],
                    &[1, co, h, w, cfg.q_stride, cfg.q_stride, 0, ho, wo],
                    rows_out * co,
                ));
                let sc = g.storage(rows_out as u64 * co as u64);
                self.to_nlc(&mut steps, &pooled, &sc, co, rows_out);
                g.submit(&[], &steps);
                steps = Vec::new();
                sc_owned = Some(sc);
            } else {
                sc_owned = Some(proj);
            }
        }
        if !steps.is_empty() {
            g.submit(&[], &steps);
            steps = Vec::new();
        }
        let shortcut: &DeviceBuffer = sc_owned.as_ref().unwrap_or(x);

        // ---- phase 2: zero-pad the token grid if the window does not divide it ----
        let mut padded_owned: Option<DeviceBuffer> = None;
        if (hp, wp) != (h, w) {
            let padded = g.storage(rows_p as u64 * c as u64);
            let map: Vec<u32> = (0..rows_in).map(|t| (t / w) * wp + (t % w)).collect();
            let pad_idx = row_index_buffer(g, "sam2_pad_idx", &map);
            // The clear is what makes the pad ZEROS; `row_scatter` leaves unnamed
            // rows untouched by design.
            g.submit(&[&padded], &[scatter_rows(g, &self.ids.permute, &pad_idx, &ln1, &padded, rows_in, c, rows_p)]);
            padded_owned = Some(padded);
        }
        let grid_src: &DeviceBuffer = padded_owned.as_ref().unwrap_or(&ln1);

        // ---- phase 3: window partition -> qkv -> attention -> proj -> reverse ----
        let plan = (ws > 0).then(|| WindowPlan::new(hp, wp, ws, ws));
        let mut wm_owned: Option<DeviceBuffer> = None;
        if let Some(pl) = &plan {
            let widx = WindowIndex::new(g, pl);
            let wm = g.storage(rows_p as u64 * c as u64);
            steps.push(window_partition(g, &self.ids.permute, &widx, grid_src, &wm, c));
            wm_owned = Some(wm);
            // `widx` must outlive the submit; it is rebuilt for the reverse below,
            // so keep this one alive by submitting here.
            g.submit(&[], &steps);
            steps = Vec::new();
        }
        let attn_in: &DeviceBuffer = wm_owned.as_ref().unwrap_or(grid_src);

        let qkv = g.storage(rows_p as u64 * 3 * co as u64);
        self.linear(&mut steps, attn_in, &qkv, rows_p, c, 3 * co, &format!("{p}.attn.qkv.weight"), &format!("{p}.attn.qkv.bias"));

        let ctxb = g.storage(rows_po as u64 * co as u64);
        // Keep every transient the recorded steps bind alive until the submit.
        let mut keep: Vec<DeviceBuffer> = Vec::new();
        let mut keep_caches: Vec<QPoolCache> = Vec::new();
        if b.q_pool {
            let (n_win, pwh, pww) = match &plan {
                Some(pl) => (pl.n_windows(), ws, ws),
                None => (1, hp, wp),
            };
            let full = QPoolPlan { n: n_win, h: pwh, w: pww, k: cfg.q_stride, stride: cfg.q_stride, pad: 0 };
            assert_eq!(full.rows_out(), rows_po, "pooled query rows must tile the padded output grid");
            let q_pooled = g.storage(rows_po as u64 * co as u64);
            let per_win = co as u64 * (pwh * pww) as u64;
            let chunks = (n_win as u64 * per_win).div_ceil(MAXPOOL_ELEMS - 1).max(1) as u32;
            assert!(
                chunks == 1 || n_win > 1,
                "q_pool over a single {pwh}x{pww} grid needs {} elements, past the f32-argmax bound",
                n_win as u64 * per_win
            );
            let per_chunk = n_win.div_ceil(chunks);
            let qw = full.win_rows_out();
            let mut start = 0u32;
            while start < n_win {
                let nc = per_chunk.min(n_win - start);
                let plan_c = QPoolPlan { n: nc, ..full };
                let cache = QPoolCache::new(g, &plan_c, co);
                // The q region of the fused `[rows, 3*co]` qkv, for this chunk's
                // rows: `region_index` with the chunk's absolute row base.
                let row0 = start * pwh * pww;
                let iv: Vec<u32> = (0..plan_c.rows_in()).map(|t| (row0 + t) * 3).collect();
                let qidx = row_index_buffer(g, "sam2_qpool_q", &iv);
                q_pool_fwd(g, &self.ids.qpool, &plan_c, co, &qkv, &qidx, &cache, &mut steps);
                let ov: Vec<u32> = (0..plan_c.rows_out()).map(|t| start * qw + t).collect();
                let oidx = row_index_buffer(g, "sam2_qpool_out", &ov);
                steps.push(scatter_rows(g, &self.ids.permute, &oidx, &cache.q_pooled, &q_pooled, plan_c.rows_out(), co, rows_po));
                keep.push(qidx);
                keep.push(oidx);
                keep_caches.push(cache);
                start += nc;
            }
            let spans: Vec<AttnSpan> = match &plan {
                Some(pl) => AttnSpan::pooled_windows(pl, &full),
                None => vec![AttnSpan { q0: 0, qn: rows_po, k0: 0, kn: rows_p }],
            };
            let scores = g.storage(model::vit::max_slab(&spans, heads));
            let probs = g.storage(model::vit::probs_len(&spans, heads));
            let max_kn = spans.iter().map(|s| s.kn).max().unwrap_or(0);
            let kt = g.storage(co as u64 * max_kn as u64);
            let km = self.ids.key_minor(&kt);
            cross_q_fwd(
                g, &self.ids.cross, Some(&km), &sh, &q_pooled, co, 0, &qkv, 3 * co, co, 2 * co, &ctxb, &scores, &probs,
                &spans, &mut steps,
            );
            keep.push(q_pooled);
            keep.push(scores);
            keep.push(probs);
            keep.push(kt);
        } else {
            let spans: Vec<(u32, u32)> = match &plan {
                Some(pl) => pl.spans().to_vec(),
                None => vec![(0, rows_p)],
            };
            let max_span = spans.iter().map(|&(_, l)| l).max().unwrap_or(0);
            let slab = heads as u64 * ATTN_CHUNK.min(max_span).max(1) as u64 * max_span as u64;
            let scores = g.storage(slab);
            let probs = g.storage(slab);
            let kt = g.storage(co as u64 * max_span as u64);
            let km = self.ids.key_minor(&kt);
            block::chunked_bidir_fwd(
                g,
                &self.ids.cross,
                Some(&km),
                heads,
                sh.head_dim(),
                co,
                &qkv,
                3 * co,
                0,
                co,
                2 * co,
                &ctxb,
                &scores,
                &probs,
                &spans,
                ATTN_CHUNK,
                None,
                &mut steps,
            );
            keep.push(scores);
            keep.push(probs);
            keep.push(kt);
        }

        let attn = g.storage(rows_po as u64 * co as u64);
        self.linear(&mut steps, &ctxb, &attn, rows_po, co, co, &format!("{p}.attn.proj.weight"), &format!("{p}.attn.proj.bias"));
        g.submit(&[], &steps);
        steps = Vec::new();
        drop(keep);
        drop(keep_caches);

        // Reverse the (output-resolution) window partition, then drop the pad.
        let mut grid_owned: Option<DeviceBuffer> = None;
        if ws_out > 0 && plan.is_some() {
            let oplan = WindowPlan::new(hpo, wpo, ws_out, ws_out);
            let oidx = WindowIndex::new(g, &oplan);
            let back = g.storage(rows_po as u64 * co as u64);
            g.submit(&[], &[window_reverse(g, &self.ids.permute, &oidx, &attn, &back, co)]);
            grid_owned = Some(back);
        }
        let padded_grid: &DeviceBuffer = grid_owned.as_ref().unwrap_or(&attn);
        let mut crop_owned: Option<DeviceBuffer> = None;
        if (hpo, wpo) != (ho, wo) {
            let cropped = g.storage(rows_out as u64 * co as u64);
            let map: Vec<u32> = (0..rows_out).map(|t| (t / wo) * wpo + (t % wo)).collect();
            let cidx = row_index_buffer(g, "sam2_crop_idx", &map);
            g.submit(&[], &[model::vit::gather_rows(g, &self.ids.permute, &cidx, padded_grid, &cropped, rows_out, co)]);
            crop_owned = Some(cropped);
        }
        let branch: &DeviceBuffer = crop_owned.as_ref().unwrap_or(padded_grid);

        // ---- residual + MLP ----
        let res = g.storage(rows_out as u64 * co as u64);
        steps.push(g.step(self.ids.add2, &[shortcut, branch, &res], &[rows_out * co], rows_out * co));
        let ln2 = g.storage(rows_out as u64 * co as u64);
        steps.push(block::layernorm_fwd(
            g,
            &self.ids.ln,
            &res,
            self.ps.w(&format!("{p}.norm2.weight")),
            self.ps.w(&format!("{p}.norm2.bias")),
            &ln2,
            co,
            rows_out,
            cfg.trunk_eps,
        ));
        let m = co * cfg.mlp_ratio;
        let h1 = g.storage(rows_out as u64 * m as u64);
        self.linear(&mut steps, &ln2, &h1, rows_out, co, m, &format!("{p}.mlp.layers.0.weight"), &format!("{p}.mlp.layers.0.bias"));
        let a1 = g.storage(rows_out as u64 * m as u64);
        steps.push(self.act_step(&h1, &a1, rows_out * m, Act::GeluErf));
        let h2 = g.storage(rows_out as u64 * co as u64);
        self.linear(&mut steps, &a1, &h2, rows_out, m, co, &format!("{p}.mlp.layers.1.weight"), &format!("{p}.mlp.layers.1.bias"));
        steps.push(g.step(self.ids.add2, &[&res, &h2, out], &[rows_out * co], rows_out * co));
        g.submit(&[], &steps);
    }

    /// FPN neck + the SAM-side projections (`conv_s0`/`conv_s1`) and the
    /// `no_mem_embed` add that turns FPN level 2 into the SAM image embedding.
    fn neck(
        &self,
        patch_embed: DeviceBuffer,
        pos_embed: DeviceBuffer,
        blocks: Vec<DeviceBuffer>,
        trunk_feats: Vec<DeviceBuffer>,
    ) -> Encoded {
        let cfg = &self.cfg;
        let ctx = self.ctx();
        let d = cfg.d_model;
        let chans = cfg.trunk_channel_list(); // reverse-resolution order
        // `ImageEncoder.__init__`'s own assertion. The two lists come from
        // different halves of the config (the derived block table vs the
        // transcribed neck list); if they ever disagree, every lateral conv
        // still BINDS (a [256, C, 1, 1] weight is only checked for element
        // count at import) and silently convolves the wrong map.
        assert_eq!(
            chans, cfg.backbone_channel_list,
            "trunk channel list {chans:?} != neck backbone_channel_list {:?}",
            cfg.backbone_channel_list
        );
        let n = chans.len() - 1;
        let grid = cfg.trunk_grid();

        // convs[n - i] is applied to trunk LEVEL i.
        let mut lateral = Vec::new();
        let mut convs = Vec::new();
        for i in 0..=n {
            let side = grid >> i;
            let cin = chans[n - i];
            let pfx = format!("image_encoder.neck.convs.{}.conv", n - i);
            let cv = Conv::with_names(
                &ctx,
                &pfx,
                ConvNames::torch_flat(&pfx),
                Shape::new(1, cin, side, side),
                ConvSpec::relu(d, 1, 1, 0).with_norm(Norm::None).with_act(Act::None).with_bias(),
                false,
            );
            cv.forward(&ctx, &self.ps, &trunk_feats[i]);
            lateral.push(self.copy_of(cv.out(), d * side * side));
            convs.push(cv);
        }
        drop(convs);

        // Top-down: only the levels in `fpn_top_down_levels` fuse; the reference
        // walks from the LOWEST resolution up.
        let mut fpn: Vec<Option<DeviceBuffer>> = (0..=n).map(|_| None).collect();
        let mut prev: Option<usize> = None;
        for i in (0..=n).rev() {
            let side = grid >> i;
            let total = d * side * side;
            if let (true, Some(pi)) = (cfg.fpn_top_down_levels.contains(&(i as u32)), prev) {
                let ps = grid >> pi;
                let up = self.gpu.storage(total as u64);
                let fused = self.gpu.storage(total as u64);
                let prev_buf = fpn[pi].as_ref().unwrap();
                self.gpu.submit(
                    &[],
                    &[
                        self.gpu.step(self.ids.resize_nearest, &[prev_buf, &up], &[1, d, ps, ps, side, side], total),
                        self.gpu.step(self.ids.add2, &[&lateral[i], &up, &fused], &[total], total),
                    ],
                );
                fpn[i] = Some(fused);
            } else {
                fpn[i] = Some(self.copy_of(&lateral[i], total));
            }
            prev = Some(i);
        }
        let fpn: Vec<DeviceBuffer> = fpn.into_iter().map(|f| f.unwrap()).collect();

        // PositionEmbeddingSine per level - a constant table per (h, w).
        let pos_sine: Vec<DeviceBuffer> = (0..=n)
            .map(|i| {
                let side = grid >> i;
                let t = hostpe::sine(cfg.pos_sine_num_pos_feats, cfg.pos_sine_temperature, side, side);
                self.upload("sam2_possine", &t)
            })
            .collect();

        // conv_s0 / conv_s1 project FPN levels 0 and 1 into the decoder's
        // high-resolution features (this happens in `SAM2Base.forward_image`,
        // not in the neck).
        let mut high_res = Vec::new();
        for (lvl, (name, cout)) in [("conv_s0", d / 8), ("conv_s1", d / 4)].into_iter().enumerate() {
            let side = grid >> lvl;
            let pfx = format!("sam_mask_decoder.{name}");
            let cv = Conv::with_names(
                &ctx,
                &pfx,
                ConvNames::torch_flat(&pfx),
                Shape::new(1, d, side, side),
                ConvSpec::relu(cout, 1, 1, 0).with_norm(Norm::None).with_act(Act::None).with_bias(),
                false,
            );
            cv.forward(&ctx, &self.ps, &fpn[lvl]);
            high_res.push(self.copy_of(cv.out(), cout * side * side));
        }

        // `directly_add_no_mem_embed`: the image path adds the no-memory
        // embedding to the lowest-resolution KEPT level - `ImageEncoder.forward`
        // discards `scalp` levels off the bottom, so that is level
        // `n - scalp`, not a hardcoded 2. Cross-checked against the SAM side,
        // which fixes the embedding grid at `image_size / backbone_stride`.
        let lvl = n - cfg.scalp as usize;
        let side = grid >> lvl;
        assert_eq!(
            side,
            cfg.image_embedding_size(),
            "scalp {} leaves FPN level {lvl} at {side}x{side}, but the SAM heads want {}x{}",
            cfg.scalp,
            cfg.image_embedding_size(),
            cfg.image_embedding_size()
        );
        let total = d * side * side;
        let image_embed = self.copy_of(&fpn[lvl], total);
        self.gpu.submit(
            &[],
            &[self.gpu.step(self.ids.add_chan_inplace, &[&image_embed, self.ps.w(NO_MEM_EMBED)], &[total, d, side * side], total)],
        );

        Encoded { pos_embed, patch_embed, blocks, trunk_feats, lateral, fpn, pos_sine, high_res, image_embed }
    }

    // =======================================================================
    // prompt encoder + two-way mask decoder
    // =======================================================================

    pub fn decode(&self, enc: &Encoded, prompt: &Prompt) -> Decoded {
        self.decode_with(enc, &enc.image_embed, prompt)
    }

    /// [`Sam2::decode`] against an EXPLICIT backbone feature map, `[1, d, h, w]`
    /// NCHW, in place of `enc.image_embed`.
    ///
    /// The image path passes `enc.image_embed` (`fpn[2] + no_mem_embed`); the
    /// video path passes the memory-conditioned feature the memory attention
    /// produced ([`crate::video`]). Everything else - the prompt encoder, the
    /// two-way transformer, the high-resolution features `enc.high_res` - is
    /// identical, which is why this is one function and not two.
    pub fn decode_with(&self, enc: &Encoded, backbone: &DeviceBuffer, prompt: &Prompt) -> Decoded {
        let cfg = &self.cfg;
        let g = &self.gpu;
        let d = cfg.d_model;
        let side = cfg.image_embedding_size();
        let n_img = side * side;
        let feats = (d / 2) as usize;

        // ---- sparse prompt embeddings (points; a box is two labelled points) ----
        let sparse_host = hostpe::embed_points(
            &self.host.gauss,
            feats,
            &self.host.point_embeddings,
            &self.host.not_a_point,
            &prompt.coords,
            &prompt.labels,
            (cfg.image_size, cfg.image_size),
        );
        let n_sparse = prompt.coords.len() as u32 + 1;
        let sparse = self.upload("sam2_sparse", &sparse_host);

        // ---- dense prompt embedding ----
        let dense = g.storage(d as u64 * n_img as u64);
        match &prompt.mask_lowres {
            None => {
                // `no_mask_embed.weight.reshape(1,-1,1,1).expand(...)`: a cleared
                // buffer plus the per-channel add IS the broadcast.
                g.submit(
                    &[&dense],
                    &[g.step(
                        self.ids.add_chan_inplace,
                        &[&dense, self.ps.w("sam_prompt_encoder.no_mask_embed.weight")],
                        &[d * n_img, d, n_img],
                        d * n_img,
                    )],
                );
            }
            Some(m) => self.mask_downscaling(m, &dense),
        }

        // ---- dense positional encoding (constant per grid size) ----
        let dense_pe = self.upload("sam2_dense_pe", &hostpe::dense_pe(&self.host.gauss, feats, side, side));

        // ---- output tokens ‖ sparse ----
        let mut tok_host: Vec<f32> = Vec::new();
        if cfg.pred_obj_scores {
            tok_host.extend_from_slice(&self.host.obj_score_token);
        }
        tok_host.extend_from_slice(&self.host.iou_token);
        tok_host.extend_from_slice(&self.host.mask_tokens);
        tok_host.extend_from_slice(&sparse_host);
        let n_out_tokens = if cfg.pred_obj_scores { 2 } else { 1 } + cfg.num_mask_tokens();
        let t = n_out_tokens + n_sparse;
        assert_eq!(tok_host.len(), (t * d) as usize);
        let tokens = self.upload("sam2_tokens", &tok_host);

        // ---- src = image_embed + dense; keys/pe as NLC token rows ----
        let src_in = g.storage(d as u64 * n_img as u64);
        let keys0 = g.storage(n_img as u64 * d as u64);
        let key_pe = g.storage(n_img as u64 * d as u64);
        let mut steps = Vec::new();
        steps.push(g.step(self.ids.add2, &[backbone, &dense, &src_in], &[d * n_img], d * n_img));
        self.to_nlc(&mut steps, &src_in, &keys0, d, n_img);
        self.to_nlc(&mut steps, &dense_pe, &key_pe, d, n_img);
        g.submit(&[], &steps);

        // ---- two-way transformer ----
        let (hs, src_out, twoway, final_attn_out) = self.two_way(&tokens, &keys0, &key_pe, t, n_img);

        // ---- upscaling tail ----
        let ctx = self.ctx();
        let src_img = g.storage(d as u64 * n_img as u64);
        let mut steps = Vec::new();
        self.to_nchw(&mut steps, &src_out, &src_img, d, n_img);
        g.submit(&[], &steps);

        let dc1 = ConvTranspose::torch(
            &ctx,
            "sam_mask_decoder.output_upscaling.0",
            Shape::new(1, d, side, side),
            ConvTrSpec::new(d / 4, 2, 2, 0),
        );
        dc1.forward(&ctx, &self.ps, &src_img);
        let n1 = d / 4 * 4 * n_img;
        let dc1_out = self.copy_of(dc1.out(), n1);

        let sum1 = g.storage(n1 as u64);
        g.submit(&[], &[g.step(self.ids.add2, &[&dc1_out, &enc.high_res[1], &sum1], &[n1], n1)]);
        let ln1 = LayerNorm2d::new(
            &ctx,
            Ln2dNames::torch("sam_mask_decoder.output_upscaling.1"),
            Shape::new(1, d / 4, 2 * side, 2 * side),
            cfg.ln2d_eps,
        );
        ln1.forward(&ctx, &self.ps, &sum1);
        let act1 = g.storage(n1 as u64);
        g.submit(&[], &[self.act_step(ln1.out(), &act1, n1, Act::GeluErf)]);

        let dc2 = ConvTranspose::torch(
            &ctx,
            "sam_mask_decoder.output_upscaling.3",
            Shape::new(1, d / 4, 2 * side, 2 * side),
            ConvTrSpec::new(d / 8, 2, 2, 0),
        );
        dc2.forward(&ctx, &self.ps, &act1);
        let n2 = d / 8 * 16 * n_img;
        let dc2_out = self.copy_of(dc2.out(), n2);
        let sum2 = g.storage(n2 as u64);
        g.submit(&[], &[g.step(self.ids.add2, &[&dc2_out, &enc.high_res[0], &sum2], &[n2], n2)]);
        let upscaled = g.storage(n2 as u64);
        g.submit(&[], &[self.act_step(&sum2, &upscaled, n2, Act::GeluErf)]);
        drop(dc1);
        drop(dc2);
        drop(ln1);

        // ---- hypernetwork MLPs -> per-mask dynamic dot product ----
        let nmt = cfg.num_mask_tokens();
        let mask_dim = d / 8;
        let s0 = if cfg.pred_obj_scores { 1 } else { 0 };
        let hyper_in = g.storage(nmt as u64 * mask_dim as u64);
        for i in 0..nmt {
            // mask token i is hs row `s + 1 + i` - with `pred_obj_scores` the
            // token order is [obj_score, iou, mask x 4, ...prompt], so this is 2,
            // not 1. (The reference dumper found the same off-by-one.)
            let row = s0 + 1 + i;
            let tok = self.row_copy(&hs, row, d);
            let out = self.mlp(
                &format!("sam_mask_decoder.output_hypernetworks_mlps.{i}"),
                &tok,
                1,
                &[d, d, d, mask_dim],
                Act::Relu,
                false,
            );
            let iv = row_index_buffer(g, "sam2_hyper_row", &[i]);
            g.submit(&[], &[scatter_rows(g, &self.ids.permute, &iv, &out, &hyper_in, 1, mask_dim, nmt)]);
        }
        let up_nlc = g.storage(n2 as u64);
        let mut steps = Vec::new();
        self.to_nlc(&mut steps, &upscaled, &up_nlc, mask_dim, 16 * n_img);
        // masks = hyper_in @ upscaled.view(C, HW): `matmul` computes x @ W^T, so
        // the NLC view of `upscaled` IS the W it wants.
        let masks_all = g.storage(nmt as u64 * (16 * n_img) as u64);
        steps.push(g.step(self.ids.matmul, &[&hyper_in, &up_nlc, &masks_all], &[nmt, mask_dim, 16 * n_img], nmt * 16 * n_img));
        g.submit(&[], &steps);

        // ---- IoU head, object score ----
        let iou_tok = self.row_copy(&hs, s0, d);
        // `MLP(d, iou_head_hidden_dim, num_mask_tokens, iou_head_depth)` -
        // derived, not a hardcoded 3, so the model and `tensor_manifest` (which
        // already reads `iou_head_depth`) cannot drift apart.
        let mut iou_dims = vec![d];
        iou_dims.extend(std::iter::repeat_n(cfg.iou_head_hidden_dim, cfg.iou_head_depth as usize - 1));
        iou_dims.push(nmt);
        let iou_all = self.mlp(
            "sam_mask_decoder.iou_prediction_head",
            &iou_tok,
            1,
            &iou_dims,
            Act::Relu,
            cfg.iou_prediction_use_sigmoid,
        );
        let obj_tok = self.row_copy(&hs, 0, d);
        let object_score_logits = self.mlp("sam_mask_decoder.pred_obj_score_head", &obj_tok, 1, &[d, d, d, 1], Act::Relu, false);

        // ---- mask selection, object gate, hi-res upsample ----
        let iou_host = g.read(&iou_all, nmt as usize);
        let obj = g.read(&object_score_logits, 1)[0];
        let is_obj = obj > 0.0;
        let (start, n_masks) = if prompt.multimask_output { (1u32, nmt - 1) } else { (0u32, 1u32) };
        let per = 16 * n_img;
        let low_res_multimasks = g.storage(n_masks as u64 * per as u64);
        if is_obj {
            let off = start as u64 * per as u64;
            g.submit(
                &[&low_res_multimasks],
                &[g.step_sliced(
                    self.ids.axpy,
                    &[&low_res_multimasks, &masks_all],
                    &[(0, 0), (off, 0)],
                    &[n_masks * per, f(1.0)],
                    n_masks * per,
                )],
            );
        } else {
            // NO_OBJ_SCORE: the reference replaces the whole map with -1024.
            g.write(&low_res_multimasks, bytemuck::cast_slice(&vec![-1024.0f32; (n_masks * per) as usize]));
        }
        let hi = 16 * n_img * 16;
        let high_res_multimasks = g.storage(n_masks as u64 * hi as u64);
        g.submit(
            &[],
            &[g.step(
                self.ids.resize_bilinear,
                &[&low_res_multimasks, &high_res_multimasks],
                &[1, n_masks, 4 * side, 4 * side, cfg.image_size, cfg.image_size, 0],
                n_masks * hi,
            )],
        );

        let ious: Vec<f32> = iou_host[start as usize..(start + n_masks) as usize].to_vec();
        let best = ious
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // ---- object pointer ----
        let multi_ptr = prompt.multimask_output && cfg.use_multimask_token_for_obj_ptr;
        let tok_row = s0 + 1 + if multi_ptr { 1 + best as u32 } else { 0 };
        let ptr_tok = self.row_copy(&hs, tok_row, d);
        let obj_ptr = if is_obj {
            self.mlp("obj_ptr_proj", &ptr_tok, 1, &[d, d, d, d], Act::Relu, false)
        } else {
            // `fixed_no_obj_ptr` with a hard 0/1 lambda: the pointer IS
            // `no_obj_ptr` when no object is predicted.
            self.upload("sam2_no_obj_ptr", &self.host.no_obj_ptr)
        };

        Decoded {
            sparse,
            dense,
            dense_pe,
            tokens,
            src_in,
            twoway,
            final_attn_out,
            hs,
            src_out,
            dc1_out,
            dc2_out,
            upscaled_embedding: upscaled,
            hyper_in,
            masks_all,
            iou_all,
            object_score_logits,
            low_res_multimasks,
            high_res_multimasks,
            obj_ptr,
            ious,
            best_iou_index: best,
            n_masks,
        }
    }

    /// `PromptEncoder.mask_downscaling`: two stride-2 convs, each followed by a
    /// channels-first LayerNorm and GELU, then a 1x1 projection to `d_model`.
    fn mask_downscaling(&self, mask: &[f32], out: &DeviceBuffer) {
        let cfg = &self.cfg;
        let ctx = self.ctx();
        let d = cfg.d_model;
        let mc = cfg.mask_in_chans;
        let s = 4 * cfg.image_embedding_size();
        assert_eq!(mask.len(), (s * s) as usize, "mask prompt must be [1,1,{s},{s}]");
        let x = self.upload("sam2_mask_in", mask);

        let mk = |pfx: &str, cin: u32, cout: u32, side: u32| {
            Conv::with_names(
                &ctx,
                pfx,
                ConvNames::torch_flat(pfx),
                Shape::new(1, cin, side, side),
                ConvSpec::relu(cout, 2, 2, 0).with_norm(Norm::None).with_act(Act::None).with_bias(),
                false,
            )
        };
        let c0 = mk("sam_prompt_encoder.mask_downscaling.0", 1, mc / 4, s);
        c0.forward(&ctx, &self.ps, &x);
        let ln0 = LayerNorm2d::new(
            &ctx,
            Ln2dNames::torch("sam_prompt_encoder.mask_downscaling.1"),
            Shape::new(1, mc / 4, s / 2, s / 2),
            cfg.ln2d_eps,
        );
        ln0.forward(&ctx, &self.ps, c0.out());
        let n0 = mc / 4 * (s / 2) * (s / 2);
        let a0 = self.gpu.storage(n0 as u64);
        self.gpu.submit(&[], &[self.act_step(ln0.out(), &a0, n0, Act::GeluErf)]);

        let c1 = mk("sam_prompt_encoder.mask_downscaling.3", mc / 4, mc, s / 2);
        c1.forward(&ctx, &self.ps, &a0);
        let ln1 = LayerNorm2d::new(
            &ctx,
            Ln2dNames::torch("sam_prompt_encoder.mask_downscaling.4"),
            Shape::new(1, mc, s / 4, s / 4),
            cfg.ln2d_eps,
        );
        ln1.forward(&ctx, &self.ps, c1.out());
        let n1 = mc * (s / 4) * (s / 4);
        let a1 = self.gpu.storage(n1 as u64);
        self.gpu.submit(&[], &[self.act_step(ln1.out(), &a1, n1, Act::GeluErf)]);

        let pfx = "sam_prompt_encoder.mask_downscaling.6";
        let c2 = Conv::with_names(
            &ctx,
            pfx,
            ConvNames::torch_flat(pfx),
            Shape::new(1, mc, s / 4, s / 4),
            ConvSpec::relu(d, 1, 1, 0).with_norm(Norm::None).with_act(Act::None).with_bias(),
            false,
        );
        c2.forward(&ctx, &self.ps, &a1);
        let n2 = d * (s / 4) * (s / 4);
        self.gpu.submit(&[out], &[self.gpu.step(self.ids.axpy, &[out, c2.out()], &[n2, f(1.0)], n2)]);
    }

    /// Copy one row of `[rows, d]` into a fresh `[1, d]` buffer. The binding
    /// offset is `row*d` floats, and every call site has `d % 64 == 0`.
    fn row_copy(&self, x: &DeviceBuffer, row: u32, d: u32) -> DeviceBuffer {
        let off = row as u64 * d as u64;
        assert_eq!(off % 64, 0, "row binding offset {off} is not 64-float aligned");
        let out = self.gpu.storage(d as u64);
        self.gpu.submit(
            &[&out],
            &[self.gpu.step_sliced(self.ids.axpy, &[&out, x], &[(0, 0), (off, 0)], &[d, f(1.0)], d)],
        );
        out
    }

    /// One `Attention` module: three projections into their OWN buffers, the
    /// materialised cross-attention trio, then `out_proj`.
    ///
    /// `k` and `v` deliberately stay in separate buffers. `vit::cross_q_fwd`
    /// binds ONE fused kv buffer, which SAM 2's decoder cannot supply: in a
    /// non-first two-way layer `q`/`k` read `queries + query_pe` while `v` reads
    /// `queries`, so no single matmul produces both. The cross trio itself takes
    /// the kv buffer separately per kernel (`attn_scores_cross` binds k,
    /// `attn_apply_cross` binds v), so two buffers at stride `internal` with
    /// offset 0 is exact and copy-free.
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, prefix: &str, q_in: &DeviceBuffer, k_in: &DeviceBuffer, v_in: &DeviceBuffer, tq: u32, tk: u32, io: u32, out: &DeviceBuffer) {
        let g = &self.gpu;
        let d = self.cfg.d_model;
        let heads = self.cfg.transformer_heads;
        let hd = io / heads;
        let mut steps = Vec::new();
        let q = g.storage(tq as u64 * io as u64);
        let k = g.storage(tk as u64 * io as u64);
        let v = g.storage(tk as u64 * io as u64);
        self.linear(&mut steps, q_in, &q, tq, d, io, &format!("{prefix}.q_proj.weight"), &format!("{prefix}.q_proj.bias"));
        self.linear(&mut steps, k_in, &k, tk, d, io, &format!("{prefix}.k_proj.weight"), &format!("{prefix}.k_proj.bias"));
        self.linear(&mut steps, v_in, &v, tk, d, io, &format!("{prefix}.v_proj.weight"), &format!("{prefix}.v_proj.bias"));
        let scores = g.storage(heads as u64 * tq as u64 * tk as u64);
        let probs = g.storage(heads as u64 * tq as u64 * tk as u64);
        let ctxb = g.storage(tq as u64 * io as u64);
        // K to key-minor once, then the coalesced sweep - `k` here is a plain
        // `[tk, io]` buffer, so `kv_stride` is `io` and `k_off` is 0.
        let kt = g.storage(io as u64 * tk as u64);
        steps.push(g.step(self.ids.key_minor.0, &[&k, &kt], &[tk, io, io, 0], io * tk));
        steps.push(g.step(self.ids.key_minor.1, &[&q, &kt, &scores], &[1, heads, tq, tk, hd, io, 0], heads * tq * tk));
        steps.push(g.step(self.ids.cross.softmax, &[&scores, &probs], &[1, heads, tq, tk], heads * tq));
        steps.push(g.step(self.ids.cross.apply, &[&probs, &v, &ctxb], &[1, heads, tq, tk, hd, io, 0, io], heads * tq * hd));
        self.linear(&mut steps, &ctxb, out, tq, io, d, &format!("{prefix}.out_proj.weight"), &format!("{prefix}.out_proj.bias"));
        g.submit(&[], &steps);
    }

    pub(crate) fn add(&self, a: &DeviceBuffer, b: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let out = self.gpu.storage(n as u64);
        self.gpu.submit(&[], &[self.gpu.step(self.ids.add2, &[a, b, &out], &[n], n)]);
        out
    }

    pub(crate) fn layernorm(&self, x: &DeviceBuffer, prefix: &str, rows: u32, d: u32) -> DeviceBuffer {
        let out = self.gpu.storage(rows as u64 * d as u64);
        self.gpu.submit(
            &[],
            &[block::layernorm_fwd(
                &self.gpu,
                &self.ids.ln,
                x,
                self.ps.w(&format!("{prefix}.weight")),
                self.ps.w(&format!("{prefix}.bias")),
                &out,
                d,
                rows,
                self.cfg.ln_eps,
            )],
        );
        out
    }

    /// `TwoWayTransformer`. Returns `(hs, keys, per-layer (queries, keys),
    /// final_attn_out)`.
    #[allow(clippy::type_complexity)]
    fn two_way(
        &self,
        tokens: &DeviceBuffer,
        keys0: &DeviceBuffer,
        key_pe: &DeviceBuffer,
        t: u32,
        n_img: u32,
    ) -> (DeviceBuffer, DeviceBuffer, Vec<(DeviceBuffer, DeviceBuffer)>, DeviceBuffer) {
        let cfg = &self.cfg;
        let g = &self.gpu;
        let d = cfg.d_model;
        let io = d / cfg.attention_downsample_rate;
        let qn = t * d;
        let kn = n_img * d;

        // `queries` starts as the tokens themselves; `query_pe` is the SAME
        // buffer throughout (the reference passes `point_embedding` as both).
        let mut queries = self.copy_of(tokens, qn);
        let mut keys = self.copy_of(keys0, kn);

        let mut taps = Vec::new();
        for l in 0..cfg.transformer_depth {
            let p = format!("sam_mask_decoder.transformer.layers.{l}");
            // (1) token self-attention
            if l == 0 {
                // `skip_first_layer_pe`: no positional add AND no residual -
                // `queries = self_attn(q=k=v=queries)`.
                let o = g.storage(qn as u64);
                self.attention(&format!("{p}.self_attn"), &queries, &queries, &queries, t, t, d, &o);
                queries = o;
            } else {
                let q = self.add(&queries, tokens, qn);
                let o = g.storage(qn as u64);
                self.attention(&format!("{p}.self_attn"), &q, &q, &queries, t, t, d, &o);
                queries = self.add(&queries, &o, qn);
            }
            queries = self.layernorm(&queries, &format!("{p}.norm1"), t, d);

            // (2) tokens -> image
            let q = self.add(&queries, tokens, qn);
            let k = self.add(&keys, key_pe, kn);
            let o = g.storage(qn as u64);
            self.attention(&format!("{p}.cross_attn_token_to_image"), &q, &k, &keys, t, n_img, io, &o);
            queries = self.add(&queries, &o, qn);
            queries = self.layernorm(&queries, &format!("{p}.norm2"), t, d);

            // (3) MLP on the tokens
            let m = self.mlp(&format!("{p}.mlp"), &queries, t, &[d, cfg.transformer_mlp_dim, d], Act::Relu, false);
            queries = self.add(&queries, &m, qn);
            queries = self.layernorm(&queries, &format!("{p}.norm3"), t, d);

            // (4) image -> tokens
            let q = self.add(&queries, tokens, qn);
            let k = self.add(&keys, key_pe, kn);
            let o = g.storage(kn as u64);
            self.attention(&format!("{p}.cross_attn_image_to_token"), &k, &q, &queries, n_img, t, io, &o);
            keys = self.add(&keys, &o, kn);
            keys = self.layernorm(&keys, &format!("{p}.norm4"), n_img, d);

            taps.push((self.copy_of(&queries, qn), self.copy_of(&keys, kn)));
        }

        // final token -> image attention + LayerNorm
        let q = self.add(&queries, tokens, qn);
        let k = self.add(&keys, key_pe, kn);
        let final_attn_out = g.storage(qn as u64);
        self.attention("sam_mask_decoder.transformer.final_attn_token_to_image", &q, &k, &keys, t, n_img, io, &final_attn_out);
        let sum = self.add(&queries, &final_attn_out, qn);
        let hs = self.layernorm(&sum, "sam_mask_decoder.transformer.norm_final_attn", t, d);
        (hs, keys, taps, final_attn_out)
    }
}

/// `no_obj_ptr` is a top-level checkpoint tensor with no module prefix.
const NO_OBJ_PTR: &str = "no_obj_ptr";
