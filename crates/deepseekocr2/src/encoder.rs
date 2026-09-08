// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The resampler: SAM's token grid for one view, plus that view's learned
//! query bank, run through the shared Qwen2-shaped GQA tower under a
//! **prefix-LM mask**, then projected into the decoder's width.
//!
//! ```text
//! sam_tokens[n_query, d] ++ query_bank[n_query, d]   -> x[2*n_query, d]
//! per layer:
//!   rmsnorm -> q/k/v (+bias) -> RoPE(q,k) -> GQA->MHA head replication
//!     -> scores -> PREFIX-LM MASK -> softmax -> apply -> out-proj -> +res
//!   rmsnorm -> gate/up -> SiLU*up -> down -> +res
//! rmsnorm (final) -> keep rows [n_query, 2*n_query) -> Linear -> [n_query, decoder_hidden]
//! ```
//!
//! ## Why the mask needs no new kernel
//!
//! `attn_prefix_mask.wgsl` already computes `allow(i,j) = (i<P && j<P) ||
//! (j<=i)`, which is exactly this tower's mask with `P = n_query` - image
//! rows attend to every image row, query rows attend causally over the whole
//! sequence (every image row plus earlier queries). It composes with the
//! mask-agnostic bidirectional attention family
//! ([`model::block::bidir_fwd`]'s constituent kernels, dispatched here one at
//! a time rather than through that bundled builder so the mask step can sit
//! between the scores and softmax dispatches - `crates/moondream3`'s decoder
//! is the in-tree precedent for this exact insertion point), which is MHA.
//! This tower is GQA (14 query heads, 2 KV heads), so K and V are widened
//! into the fused qkv buffer with [`model::block::kv_expand_fwd`] - the same
//! builder places Q too, with `group = 1` (an identity replication, i.e. a
//! strided copy), so one function places every one of the buffer's three
//! regions. `crates/lfm2`'s attention mixer is the in-tree precedent for this
//! GQA-into-fused-qkv wiring (minus the mask step, which it has no need of).
//!
//! ## Scope of this module
//!
//! Forward only - training and the backward-through-the-mask question belong
//! to a later milestone. SAM's own forward (`sam1::SamEncoder`) is NOT
//! invoked here: this module takes a view's already-produced `[n_query,
//! d_model]` image-token grid as a plain host slice, exactly as
//! [`resample_view`]'s golden reference does (SAM has its own gate; nothing
//! here re-derives it). Composing this resampler with a real `SamEncoder` and
//! splicing its multi-view output into the DeepSeek-V2 decoder are later
//! milestones' work.

use gpu_core::{DeviceBuffer, Gpu};
use model::block::{self, KernelIds, UNREGISTERED};
use paramstore::{ParamStore, Role};

use crate::config::DeepseekOcr2VisionConfig;

/// Kernels this crate dispatches, by name (`Gpu::kernel_index` resolves by
/// name, so table order carries no meaning).
pub const PIPELINES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("add_inplace", kernels::ADD_INPLACE),
    ("rmsnorm", kernels::RMSNORM),
    ("rope_base", kernels::ROPE_BASE),
    ("kv_expand", kernels::KV_EXPAND),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_prefix_mask", kernels::ATTN_PREFIX_MASK),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("silu_mul", kernels::SILU_MUL),
];

/// Resolved pipeline indices, looked up once at construction.
#[derive(Clone, Copy)]
struct Ids {
    matmul: usize,
    bias_add: usize,
    add_inplace: usize,
    rmsnorm: usize,
    rope: usize,
    kv_expand: usize,
    scores: usize,
    mask: usize,
    softmax: usize,
    apply: usize,
    silu_mul: usize,
}

impl Ids {
    fn resolve(g: &Gpu) -> Ids {
        let idx = |name: &str| g.kernel_index(name).unwrap_or_else(|| panic!("deepseekocr2: {name} not registered - is it missing from PIPELINES?"));
        Ids {
            matmul: idx("matmul"),
            bias_add: idx("bias_add"),
            add_inplace: idx("add_inplace"),
            rmsnorm: idx("rmsnorm"),
            rope: idx("rope_base"),
            kv_expand: idx("kv_expand"),
            scores: idx("attn_scores_bidir"),
            mask: idx("attn_prefix_mask"),
            softmax: idx("attn_softmax_bidir"),
            apply: idx("attn_apply_bidir"),
            silu_mul: idx("silu_mul"),
        }
    }

    /// The subset [`block::rmsnorm_fwd`]/[`block::rope_fwd`] read; every other
    /// slot is [`UNREGISTERED`] because this module never dispatches through
    /// them (forward-only, no backward, no coalesced-row variant registered).
    fn as_block_ids(&self) -> KernelIds {
        KernelIds {
            rmsnorm: self.rmsnorm,
            rms_inv: UNREGISTERED,
            rmsnorm_dx: UNREGISTERED,
            rmsnorm_dw: UNREGISTERED,
            rope: self.rope,
            rope_bwd: UNREGISTERED,
            gqa_scores: UNREGISTERED,
            gqa_apply: UNREGISTERED,
            attn_softmax: UNREGISTERED,
            gqa_dscores: UNREGISTERED,
            gqa_dv: UNREGISTERED,
            gqa_dq: UNREGISTERED,
            gqa_dk: UNREGISTERED,
            silu_mul: self.silu_mul,
            silu_da: UNREGISTERED,
            silu_db: UNREGISTERED,
            rmsnorm_rows: UNREGISTERED,
            rmsnorm_dx_rows: UNREGISTERED,
        }
    }
}

/// One layer's attention-and-mask scores, kept around for taps ([`LayerTrace`])
/// and, later, backward.
pub struct LayerTrace {
    /// `[n_heads, t, t]` before the mask.
    pub scores_pre_mask: Vec<f32>,
    /// `[n_heads, t, t]` after it.
    pub scores_post_mask: Vec<f32>,
    /// `[n_heads, t, t]` post-softmax.
    pub probs: Vec<f32>,
    /// `[t, d_model]` - the block's residual output.
    pub out: Vec<f32>,
}

/// Everything one call to [`Resampler::resample_view`] produced, named after
/// the golden fixture's own taps.
pub struct ViewTrace {
    /// `[2*n_query, d_model]` - SAM tokens then the query bank.
    pub concat_in: Vec<f32>,
    pub layers: Vec<LayerTrace>,
    /// `[n_query, d_model]` - the post-final-norm query half.
    pub query_slice: Vec<f32>,
    /// `[n_query, decoder_hidden]`.
    pub projected: Vec<f32>,
}

/// The shared Qwen2-shaped GQA tower plus the projector: everything past
/// SAM's own output. One instance serves every view and every tile - the real
/// model has no per-view parameters at all, only the query bank selection
/// differs (see [`Self::resample_view`]'s `local` flag).
pub struct Resampler {
    gpu: Gpu,
    ps: ParamStore,
    ids: Ids,
    cfg: DeepseekOcr2VisionConfig,
}

impl Resampler {
    /// `init` must name every tensor in [`Qwen2EncoderConfig::param_list`] and
    /// [`DeepseekOcr2VisionConfig::projector_param_list`] - one flat source,
    /// since the two name spaces are already disjoint
    /// (`vision.encoder.*`/`vision.query_*` vs `vision.projector.*`/
    /// `vision.view_separator`).
    pub fn new_on(gpu: Gpu, cfg: DeepseekOcr2VisionConfig, init: &dyn checkpoint::TensorSource, train: bool) -> Resampler {
        cfg.check();
        let ids = Ids::resolve(&gpu);
        let role = if train { Role::Trainable } else { Role::Frozen };
        let mut params = cfg.encoder.param_list();
        params.extend(cfg.projector_param_list());
        let roles: Vec<(String, usize, Role)> = params.into_iter().map(|(n, numel)| (n, numel, role)).collect();
        let ps = ParamStore::new_with_roles_src(&gpu, roles, init);
        Resampler { gpu, ps, ids, cfg }
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }

    /// Run the shared tower on one view's SAM tokens, keep the query half,
    /// project it. `sam_tokens` is `[n_query, d_model]` where `n_query` is
    /// `cfg.encoder.n_query_local` or `n_query_global`, selected by `local`.
    pub fn resample_view(&self, sam_tokens: &[f32], local: bool) -> ViewTrace {
        let e = &self.cfg.encoder;
        let n_query = if local { e.n_query_local } else { e.n_query_global };
        assert_eq!(sam_tokens.len(), (n_query * e.d_model) as usize, "sam_tokens must be [n_query, d_model]");

        let query_bank = self.ps.read_weight(&self.gpu, if local { "vision.query_local.weight" } else { "vision.query_global.weight" });
        let mut concat_in = Vec::with_capacity(sam_tokens.len() + query_bank.len());
        concat_in.extend_from_slice(sam_tokens);
        concat_in.extend_from_slice(&query_bank);

        let t = 2 * n_query;
        let d = e.d_model;
        let x = self.gpu.storage((t * d) as u64);
        self.gpu.write_f32(&x, &concat_in);

        let mut layers = Vec::with_capacity(e.n_layers as usize);
        for l in 0..e.n_layers {
            layers.push(self.layer_fwd(l, &x, t, n_query));
        }

        let final_norm = self.gpu.storage((t * d) as u64);
        self.gpu.submit(&[], &[block::rmsnorm_fwd(&self.gpu, &self.ids.as_block_ids(), &x, self.ps.w("vision.encoder.norm.weight"), &final_norm, d, t)]);
        let normed = self.gpu.read(&final_norm, (t * d) as usize);
        let query_slice = normed[(n_query * d) as usize..].to_vec();

        let (pin, pout) = (self.cfg.encoder.d_model, self.cfg.decoder_hidden);
        let q_dev = self.gpu.storage((n_query * pin) as u64);
        self.gpu.write_f32(&q_dev, &query_slice);
        let proj = self.gpu.storage((n_query * pout) as u64);
        let steps = vec![
            self.gpu.step(self.ids.matmul, &[&q_dev, self.ps.w("vision.projector.fc.weight"), &proj], &[n_query, pin, pout], n_query * pout),
            self.gpu.step(self.ids.bias_add, &[&proj, self.ps.w("vision.projector.fc.bias")], &[n_query, pout], n_query * pout),
        ];
        self.gpu.submit(&[], &steps);
        let projected = self.gpu.read(&proj, (n_query * pout) as usize);

        ViewTrace { concat_in, layers, query_slice, projected }
    }

    /// One prefix-LM GQA block, in place on the `[t, d_model]` residual `x`.
    /// Returns this layer's taps, read back before the next layer's dispatch
    /// overwrites the scratch buffers.
    fn layer_fwd(&self, l: u32, x: &DeviceBuffer, t: u32, n_query: u32) -> LayerTrace {
        let e = &self.cfg.encoder;
        let (d, kv, ff, hd, nh, nkv) = (e.d_model, e.kv_dim(), e.ffn_hidden, e.head_dim(), e.n_heads, e.n_kv_heads);
        let g = &self.gpu;
        let ids = &self.ids;
        let bids = ids.as_block_ids();
        let w = |leaf: &str| self.ps.w(&format!("vision.encoder.blocks.{l}.{leaf}"));

        // ---- attention ----
        let xn1 = g.storage((t * d) as u64);
        g.submit(&[], &[block::rmsnorm_fwd(g, &bids, x, w("norm1.weight"), &xn1, d, t)]);

        let q = g.storage((t * d) as u64);
        let k = g.storage((t * kv) as u64);
        let v = g.storage((t * kv) as u64);
        let steps = vec![
            g.step(ids.matmul, &[&xn1, w("attn.q.weight"), &q], &[t, d, d], t * d),
            g.step(ids.bias_add, &[&q, w("attn.q.bias")], &[t, d], t * d),
            g.step(ids.matmul, &[&xn1, w("attn.k.weight"), &k], &[t, d, kv], t * kv),
            g.step(ids.bias_add, &[&k, w("attn.k.bias")], &[t, kv], t * kv),
            g.step(ids.matmul, &[&xn1, w("attn.v.weight"), &v], &[t, d, kv], t * kv),
            g.step(ids.bias_add, &[&v, w("attn.v.bias")], &[t, kv], t * kv),
        ];
        g.submit(&[], &steps);
        g.submit(&[], &[block::rope_fwd(g, &bids, &q, t, nh, hd, d, t, e.rope_theta), block::rope_fwd(g, &bids, &k, t, nkv, hd, kv, t, e.rope_theta)]);

        let qkv = g.storage((t * 3 * d) as u64);
        let group = e.group();
        g.submit(
            &[],
            &[
                block::kv_expand_fwd(g, ids.kv_expand, &q, &qkv, t, nh, 1, hd, 3 * d, 0),
                block::kv_expand_fwd(g, ids.kv_expand, &k, &qkv, t, nh, group, hd, 3 * d, d),
                block::kv_expand_fwd(g, ids.kv_expand, &v, &qkv, t, nh, group, hd, 3 * d, 2 * d),
            ],
        );

        let scores = g.storage((nh * t * t) as u64);
        g.submit(&[], &[g.step(ids.scores, &[&qkv, &scores], &[1, nh, t, hd, 3 * d, 0, d], nh * t * t)]);
        let scores_pre_mask = g.read(&scores, (nh * t * t) as usize);

        g.submit(&[], &[g.step(ids.mask, &[&scores], &[1, nh, t, n_query], nh * t * t)]);
        let scores_post_mask = g.read(&scores, (nh * t * t) as usize);

        let probs_buf = g.storage((nh * t * t) as u64);
        g.submit(&[], &[g.step(ids.softmax, &[&scores, &probs_buf], &[1, nh, t], nh * t)]);
        let probs = g.read(&probs_buf, (nh * t * t) as usize);

        let ctx = g.storage((t * d) as u64);
        g.submit(&[], &[g.step(ids.apply, &[&probs_buf, &qkv, &ctx], &[1, nh, t, hd, 3 * d, 2 * d, d], nh * t * hd)]);

        let attn_out = g.storage((t * d) as u64);
        g.submit(&[], &[g.step(ids.matmul, &[&ctx, w("attn.out.weight"), &attn_out], &[t, d, d], t * d)]);
        g.submit(&[], &[g.step(ids.add_inplace, &[x, &attn_out], &[t * d], t * d)]);

        // ---- SwiGLU MLP ----
        let xn2 = g.storage((t * d) as u64);
        g.submit(&[], &[block::rmsnorm_fwd(g, &bids, x, w("norm2.weight"), &xn2, d, t)]);
        let gate = g.storage((t * ff) as u64);
        let up = g.storage((t * ff) as u64);
        let steps = vec![
            g.step(ids.matmul, &[&xn2, w("mlp.gate.weight"), &gate], &[t, d, ff], t * ff),
            g.step(ids.matmul, &[&xn2, w("mlp.up.weight"), &up], &[t, d, ff], t * ff),
        ];
        g.submit(&[], &steps);
        let h = g.storage((t * ff) as u64);
        g.submit(&[], &[g.step(ids.silu_mul, &[&gate, &up, &h], &[t * ff], t * ff)]);
        let mlp_out = g.storage((t * d) as u64);
        g.submit(&[], &[g.step(ids.matmul, &[&h, w("mlp.down.weight"), &mlp_out], &[t, ff, d], t * d)]);
        g.submit(&[], &[g.step(ids.add_inplace, &[x, &mlp_out], &[t * d], t * d)]);

        let out = g.read(x, (t * d) as usize);
        LayerTrace { scores_pre_mask, scores_post_mask, probs, out }
    }
}

/// Gather every local tile's projected output (in caller-supplied order - the
/// real model's order is row-major over the tile grid, width-first), then the
/// global view's, then one learned separator row. Pure host concatenation:
/// nothing here is a device dispatch, since it is nothing more than placing
/// already-computed rows one after another (the DEVICE-side splice into the
/// decoder's own row layout, with its own backward, is a later milestone -
/// see `crates/deepseek2ocr/src/layout.rs`'s `RowGather` for the shape that
/// work will follow).
pub fn gather_rows(local_tiles: &[Vec<f32>], global: &[f32], separator: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    for tile in local_tiles {
        out.extend_from_slice(tile);
    }
    out.extend_from_slice(global);
    out.extend_from_slice(separator);
    out
}
