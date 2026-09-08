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
use model::block::{self, Bidir, BidirIds, KernelIds, UNREGISTERED};
use paramstore::{ParamStore, Role};

use crate::config::DeepseekOcr2VisionConfig;

/// Kernels this crate dispatches, by name (`Gpu::kernel_index` resolves by
/// name, so table order carries no meaning).
pub const PIPELINES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("bias_add", kernels::BIAS_ADD),
    ("bias_grad", kernels::BIAS_GRAD),
    ("add_inplace", kernels::ADD_INPLACE),
    ("rmsnorm", kernels::RMSNORM),
    ("rms_inv", kernels::RMS_INV),
    ("rmsnorm_dx", kernels::RMSNORM_DX),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
    ("rope_base", kernels::ROPE_BASE),
    ("rope_base_bwd", kernels::ROPE_BASE_BWD),
    ("kv_expand", kernels::KV_EXPAND),
    ("kv_expand_bwd", kernels::KV_EXPAND_BWD),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_prefix_mask", kernels::ATTN_PREFIX_MASK),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR),
    ("attn_bwd_dv_bidir", kernels::ATTN_BWD_DV_BIDIR),
    ("attn_bwd_dq_bidir", kernels::ATTN_BWD_DQ_BIDIR),
    ("attn_bwd_dk_bidir", kernels::ATTN_BWD_DK_BIDIR),
    ("silu_mul", kernels::SILU_MUL),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
];

/// Resolved pipeline indices, looked up once at construction.
#[derive(Clone, Copy)]
struct Ids {
    matmul: usize,
    matmul_dx: usize,
    matmul_dw: usize,
    bias_add: usize,
    bias_grad: usize,
    add_inplace: usize,
    rmsnorm: usize,
    rms_inv: usize,
    rmsnorm_dx: usize,
    rmsnorm_dw: usize,
    rope: usize,
    rope_bwd: usize,
    kv_expand: usize,
    kv_expand_bwd: usize,
    scores: usize,
    mask: usize,
    softmax: usize,
    apply: usize,
    dscores: usize,
    dv: usize,
    dq: usize,
    dk: usize,
    silu_mul: usize,
    silu_da: usize,
    silu_db: usize,
}

impl Ids {
    fn resolve(g: &Gpu) -> Ids {
        let idx = |name: &str| g.kernel_index(name).unwrap_or_else(|| panic!("deepseekocr2: {name} not registered - is it missing from PIPELINES?"));
        Ids {
            matmul: idx("matmul"),
            matmul_dx: idx("matmul_dx"),
            matmul_dw: idx("matmul_dw"),
            bias_add: idx("bias_add"),
            bias_grad: idx("bias_grad"),
            add_inplace: idx("add_inplace"),
            rmsnorm: idx("rmsnorm"),
            rms_inv: idx("rms_inv"),
            rmsnorm_dx: idx("rmsnorm_dx"),
            rmsnorm_dw: idx("rmsnorm_dw"),
            rope: idx("rope_base"),
            rope_bwd: idx("rope_base_bwd"),
            kv_expand: idx("kv_expand"),
            kv_expand_bwd: idx("kv_expand_bwd"),
            scores: idx("attn_scores_bidir"),
            mask: idx("attn_prefix_mask"),
            softmax: idx("attn_softmax_bidir"),
            apply: idx("attn_apply_bidir"),
            dscores: idx("attn_bwd_dscores_bidir"),
            dv: idx("attn_bwd_dv_bidir"),
            dq: idx("attn_bwd_dq_bidir"),
            dk: idx("attn_bwd_dk_bidir"),
            silu_mul: idx("silu_mul"),
            silu_da: idx("silu_bwd_da"),
            silu_db: idx("silu_bwd_db"),
        }
    }

    /// The subset [`block::rmsnorm_fwd`]/[`block::rmsnorm_bwd`]/
    /// [`block::rope_fwd`]/[`block::rope_bwd`] read. The GQA-family slots stay
    /// [`UNREGISTERED`]: this tower never dispatches through them, since its
    /// mask needs the bidirectional family instead (see this module's header).
    fn as_block_ids(&self) -> KernelIds {
        KernelIds {
            rmsnorm: self.rmsnorm,
            rms_inv: self.rms_inv,
            rmsnorm_dx: self.rmsnorm_dx,
            rmsnorm_dw: self.rmsnorm_dw,
            rope: self.rope,
            rope_bwd: self.rope_bwd,
            gqa_scores: UNREGISTERED,
            gqa_apply: UNREGISTERED,
            attn_softmax: UNREGISTERED,
            gqa_dscores: UNREGISTERED,
            gqa_dv: UNREGISTERED,
            gqa_dq: UNREGISTERED,
            gqa_dk: UNREGISTERED,
            silu_mul: self.silu_mul,
            silu_da: self.silu_da,
            silu_db: self.silu_db,
            rmsnorm_rows: UNREGISTERED,
            rmsnorm_dx_rows: UNREGISTERED,
        }
    }

    /// The bidirectional-attention family's ids, forward and backward.
    fn as_bidir_ids(&self) -> BidirIds {
        BidirIds { scores: self.scores, softmax: self.softmax, apply: self.apply, dscores: self.dscores, dv: self.dv, dq: self.dq, dk: self.dk }
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

/// Everything one layer's forward needs to run its own backward - never
/// exposed outside this crate. `x` is mutated in place by both residual adds,
/// so the pre-attention and post-attention residual values are copied into
/// their own buffers here rather than re-derived (there is nothing left to
/// re-derive them FROM once `x` has moved on to the next layer).
struct LayerFwdState {
    x_in: DeviceBuffer,
    x_mid: DeviceBuffer,
    xn1: DeviceBuffer,
    qkv: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    xn2: DeviceBuffer,
    gate: DeviceBuffer,
    up: DeviceBuffer,
    h: DeviceBuffer,
}

/// A resampled view's forward, held open for [`Resampler::backward_train`] -
/// the training-side counterpart of [`ViewTrace`] (which is read-back-and-
/// discard, the inference/gradcheck-tap shape). Opaque to callers outside
/// this crate: hold it by value between a `forward_train`/`backward_train`
/// pair and nothing else.
pub struct TrainState {
    n_query: u32,
    t: u32,
    /// The residual stream after the last layer, before the final norm -
    /// still valid: nothing after the layer loop mutates it further.
    x_final: DeviceBuffer,
    layers: Vec<LayerFwdState>,
    /// The final norm's post-norm query-half rows, uploaded back to device -
    /// the projector's own `x` operand, needed by its `matmul_dw`.
    q_dev: DeviceBuffer,
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

    /// Write `vision.view_separator`'s own gradient. Unlike every other
    /// parameter here, the composite's row layout ([`crate::rows`]) places
    /// this vector on exactly ONE row of the assembled sequence, never more -
    /// so, exactly as [`Self::backward_train`] already does for the query
    /// bank, there is nothing to accumulate: the caller's `d_separator` slice
    /// (`crate::rows`'s `separator_row` of the decoder's splice gradient) IS
    /// the whole gradient, and this simply places it.
    pub fn write_view_separator_grad(&self, d_separator: &[f32]) {
        self.gpu.write_f32(self.ps.g("vision.view_separator"), d_separator);
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
            layers.push(self.layer_fwd(l, &x, t, n_query).0);
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
    /// Returns this layer's taps (read back before the next layer's dispatch
    /// overwrites the scratch buffers) plus the retained device buffers
    /// [`Resampler::backward_train`] needs - a snapshot of `x` at the two
    /// residual points is taken explicitly, since `x` itself is about to be
    /// mutated twice more and there is nothing left to recover them from
    /// afterward.
    fn layer_fwd(&self, l: u32, x: &DeviceBuffer, t: u32, n_query: u32) -> (LayerTrace, LayerFwdState) {
        let e = &self.cfg.encoder;
        let (d, kv, ff, hd, nh, nkv) = (e.d_model, e.kv_dim(), e.ffn_hidden, e.head_dim(), e.n_heads, e.n_kv_heads);
        let g = &self.gpu;
        let ids = &self.ids;
        let bids = ids.as_block_ids();
        let w = |leaf: &str| self.ps.w(&format!("vision.encoder.blocks.{l}.{leaf}"));

        let x_in = self.snapshot(x, (t * d) as usize);

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

        let x_mid = self.snapshot(x, (t * d) as usize);

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
        let trace = LayerTrace { scores_pre_mask, scores_post_mask, probs, out };
        let state = LayerFwdState { x_in, x_mid, xn1, qkv, probs: probs_buf, ctx, xn2, gate, up, h };
        (trace, state)
    }

    /// A fresh device buffer holding an independent copy of `buf`'s current
    /// contents - a host round trip, the simplest way to snapshot a buffer
    /// this crate's forward is about to mutate further. Correctness-first:
    /// an optimisation pass (M12) can replace this with a device-side copy
    /// once the shape this whole backward composes is settled.
    fn snapshot(&self, buf: &DeviceBuffer, n: usize) -> DeviceBuffer {
        let data = self.gpu.read(buf, n);
        let copy = self.gpu.storage(n as u64);
        self.gpu.write_f32(&copy, &data);
        copy
    }

    /// One prefix-LM GQA block's backward, mirroring [`Self::layer_fwd`] in
    /// reverse. `d_x_out` is the gradient w.r.t. this layer's OUTPUT (from the
    /// next layer, or from the final norm for the last one); returns the
    /// gradient w.r.t. this layer's INPUT, to drive the previous layer (or
    /// [`Self::backward_train`]'s split into `d_sam`/the query bank's grad,
    /// for layer 0). Every parameter gradient this layer owns is accumulated
    /// into the `ParamStore` directly, the same accumulate-into-a-pre-zeroed-
    /// buffer contract every kernel here already carries. Takes no `n_query`:
    /// `attn_prefix_mask` needs no backward of its own (see the dispatch
    /// below), so nothing in this function needs the prefix boundary.
    fn layer_bwd(&self, l: u32, st: &LayerFwdState, t: u32, d_x_out: &DeviceBuffer) -> DeviceBuffer {
        let e = &self.cfg.encoder;
        let (d, kv, ff, hd, nh, nkv) = (e.d_model, e.kv_dim(), e.ffn_hidden, e.head_dim(), e.n_heads, e.n_kv_heads);
        let g = &self.gpu;
        let ids = &self.ids;
        let bids = ids.as_block_ids();
        let bidir = Bidir { b: 1, t, n_heads: nh, head_dim: hd, stride: 3 * d, q_off: 0, k_off: d, v_off: 2 * d };
        let bidir_ids = ids.as_bidir_ids();
        let w = |leaf: &str| self.ps.w(&format!("vision.encoder.blocks.{l}.{leaf}"));
        let gr = |leaf: &str| self.ps.g(&format!("vision.encoder.blocks.{l}.{leaf}"));

        // ---- MLP residual: x_out = x_mid + mlp_out ----
        let d_mlp_out = d_x_out; // identity through the residual add
        let d_h = g.storage((t * ff) as u64);
        g.submit(&[], &[g.step(ids.matmul_dx, &[d_mlp_out, w("mlp.down.weight"), &d_h], &[t, ff, d, 0], t * ff)]);
        g.submit(&[], &[g.step(ids.matmul_dw, &[d_mlp_out, &st.h, gr("mlp.down.weight")], &[t, ff, d], d * ff)]);

        let d_gate = g.storage((t * ff) as u64);
        let d_up = g.storage((t * ff) as u64);
        g.submit(&[], &[g.step(ids.silu_da, &[&st.gate, &st.up, &d_h, &d_gate], &[t * ff], t * ff), g.step(ids.silu_db, &[&st.gate, &d_h, &d_up], &[t * ff], t * ff)]);

        let d_xn2 = g.storage((t * d) as u64);
        g.submit(
            &[],
            &[
                g.step(ids.matmul_dx, &[&d_up, w("mlp.up.weight"), &d_xn2], &[t, d, ff, 0], t * d),
                g.step(ids.matmul_dx, &[&d_gate, w("mlp.gate.weight"), &d_xn2], &[t, d, ff, 1], t * d),
            ],
        );
        g.submit(
            &[],
            &[
                g.step(ids.matmul_dw, &[&d_up, &st.xn2, gr("mlp.up.weight")], &[t, d, ff], ff * d),
                g.step(ids.matmul_dw, &[&d_gate, &st.xn2, gr("mlp.gate.weight")], &[t, d, ff], ff * d),
            ],
        );

        let d_x_mid_from_norm2 = g.storage((t * d) as u64);
        let inv2 = g.storage(t as u64);
        g.submit(&[], block::rmsnorm_bwd(g, &bids, &st.x_mid, w("norm2.weight"), &d_xn2, &d_x_mid_from_norm2, &inv2, Some(gr("norm2.weight")), d, t).as_slice());

        // Fresh storage is not guaranteed zeroed on every backend, and both
        // dispatches below are `+=` (`add_inplace`) with no prior `=` write -
        // clear explicitly rather than relying on an allocator convention.
        let d_x_mid = g.storage((t * d) as u64);
        g.submit(&[&d_x_mid], &[g.step(ids.add_inplace, &[&d_x_mid, d_x_out], &[t * d], t * d), g.step(ids.add_inplace, &[&d_x_mid, &d_x_mid_from_norm2], &[t * d], t * d)]);

        // ---- attention residual: x_mid = x_in + attn_out ----
        let d_attn_out = &d_x_mid; // identity through the residual add
        let d_ctx = g.storage((t * d) as u64);
        g.submit(&[], &[g.step(ids.matmul_dx, &[d_attn_out, w("attn.out.weight"), &d_ctx], &[t, d, d, 0], t * d)]);
        g.submit(&[], &[g.step(ids.matmul_dw, &[d_attn_out, &st.ctx, gr("attn.out.weight")], &[t, d, d], d * d)]);

        let d_scores = g.storage((nh * t * t) as u64);
        let d_qkv = g.storage((t * 3 * d) as u64);
        g.submit(&[], block::bidir_bwd(g, &bidir_ids, &bidir, &st.qkv, &st.probs, &d_ctx, &d_scores, &d_qkv).as_slice());
        // `attn_prefix_mask` is a constant additive mask folded into `scores`
        // before the softmax this backward's `probs` already reflects - the
        // softmax jacobian carries the right (~0) gradient into every masked
        // entry on its own, so the mask needs no backward dispatch of its own
        // (`crates/moondream3`'s decoder is the in-tree precedent, and this
        // module's own `tests/tiny_ref.rs` mutation check already proves the
        // forward masking is the thing that's correct here).

        let d_q = g.storage((t * d) as u64);
        let d_k = g.storage((t * kv) as u64);
        let d_v = g.storage((t * kv) as u64);
        g.submit(
            &[],
            &[
                block::kv_expand_bwd(g, ids.kv_expand_bwd, &d_qkv, &d_q, t, nh, 1, hd, 3 * d, 0),
                block::kv_expand_bwd(g, ids.kv_expand_bwd, &d_qkv, &d_k, t, nh, e.group(), hd, 3 * d, d),
                block::kv_expand_bwd(g, ids.kv_expand_bwd, &d_qkv, &d_v, t, nh, e.group(), hd, 3 * d, 2 * d),
            ],
        );
        // RoPE's adjoint: the same per-pair rotation, applied to the
        // gradient buffer in place - it needs no forward-value operand.
        g.submit(&[], &[block::rope_bwd(g, &bids, &d_q, t, nh, hd, d, t, e.rope_theta), block::rope_bwd(g, &bids, &d_k, t, nkv, hd, kv, t, e.rope_theta)]);

        let d_xn1 = g.storage((t * d) as u64);
        g.submit(
            &[],
            &[
                g.step(ids.matmul_dx, &[&d_q, w("attn.q.weight"), &d_xn1], &[t, d, d, 0], t * d),
                g.step(ids.matmul_dx, &[&d_k, w("attn.k.weight"), &d_xn1], &[t, d, kv, 1], t * d),
                g.step(ids.matmul_dx, &[&d_v, w("attn.v.weight"), &d_xn1], &[t, d, kv, 1], t * d),
            ],
        );
        g.submit(
            &[],
            &[
                g.step(ids.matmul_dw, &[&d_q, &st.xn1, gr("attn.q.weight")], &[t, d, d], d * d),
                g.step(ids.matmul_dw, &[&d_k, &st.xn1, gr("attn.k.weight")], &[t, d, kv], kv * d),
                g.step(ids.matmul_dw, &[&d_v, &st.xn1, gr("attn.v.weight")], &[t, d, kv], kv * d),
                g.step(ids.bias_grad, &[&d_q, gr("attn.q.bias")], &[t, d], d),
                g.step(ids.bias_grad, &[&d_k, gr("attn.k.bias")], &[t, kv], kv),
                g.step(ids.bias_grad, &[&d_v, gr("attn.v.bias")], &[t, kv], kv),
            ],
        );

        let d_x_in_from_norm1 = g.storage((t * d) as u64);
        let inv1 = g.storage(t as u64);
        g.submit(&[], block::rmsnorm_bwd(g, &bids, &st.x_in, w("norm1.weight"), &d_xn1, &d_x_in_from_norm1, &inv1, Some(gr("norm1.weight")), d, t).as_slice());

        let d_x_in = g.storage((t * d) as u64);
        g.submit(&[&d_x_in], &[g.step(ids.add_inplace, &[&d_x_in, &d_x_mid], &[t * d], t * d), g.step(ids.add_inplace, &[&d_x_in, &d_x_in_from_norm1], &[t * d], t * d)]);
        d_x_in
    }

    /// Names of every trainable parameter this tower owns - the encoder
    /// stack plus the projector. Does not include SAM's or the decoder's own
    /// parameters, since neither is this crate's concern.
    pub fn param_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.cfg.encoder.param_list().into_iter().map(|(n, _)| n).collect();
        names.extend(self.cfg.projector_param_list().into_iter().map(|(n, _)| n));
        names
    }
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write_f32(self.ps.w(name), data);
    }
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }
    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    /// Forward one view, exactly as [`Self::resample_view`], but keeping every
    /// buffer [`Self::backward_train`] will need instead of discarding them.
    /// Returns the projected output (host) and the retained state.
    pub fn forward_train(&self, sam_tokens: &[f32], local: bool) -> (Vec<f32>, TrainState) {
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
            layers.push(self.layer_fwd(l, &x, t, n_query).1);
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

        (projected, TrainState { n_query, t, x_final: x, layers, q_dev })
    }

    /// Backward through [`Self::forward_train`]'s whole tower, given the
    /// loss's gradient w.r.t. the projected output. Accumulates every
    /// parameter's gradient into the `ParamStore` (call [`Self::zero_grads`]
    /// first) and returns the gradient w.r.t. `sam_tokens` - the input a real
    /// caller would backpropagate into SAM's own output. The query bank's own
    /// gradient (the OTHER half of the concat this tower's input is) is
    /// written directly into its `ParamStore` entry, the same as any other
    /// parameter here.
    pub fn backward_train(&self, st: TrainState, d_projected: &[f32], local: bool) -> Vec<f32> {
        let e = &self.cfg.encoder;
        let (d, t, n_query) = (e.d_model, st.t, st.n_query);
        let (pin, pout) = (e.d_model, self.cfg.decoder_hidden);

        let d_proj = self.gpu.storage((n_query * pout) as u64);
        self.gpu.write_f32(&d_proj, d_projected);
        self.gpu.submit(&[], &[self.gpu.step(self.ids.bias_grad, &[&d_proj, self.ps.g("vision.projector.fc.bias")], &[n_query, pout], pout)]);
        self.gpu.submit(&[], &[self.gpu.step(self.ids.matmul_dw, &[&d_proj, &st.q_dev, self.ps.g("vision.projector.fc.weight")], &[n_query, pin, pout], pout * pin)]);
        let d_q_dev = self.gpu.storage((n_query * pin) as u64);
        self.gpu.submit(&[], &[self.gpu.step(self.ids.matmul_dx, &[&d_proj, self.ps.w("vision.projector.fc.weight"), &d_q_dev], &[n_query, pin, pout, 0], n_query * pin)]);
        let d_query_slice = self.gpu.read(&d_q_dev, (n_query * pin) as usize);

        // Only the query-half rows of `normed` feed the projector, so the
        // image-half rows of `d_normed` are exactly zero.
        let mut d_normed = vec![0.0f32; (t * d) as usize];
        d_normed[(n_query * d) as usize..].copy_from_slice(&d_query_slice);
        let d_normed_buf = self.gpu.storage((t * d) as u64);
        self.gpu.write_f32(&d_normed_buf, &d_normed);

        let d_x_final = self.gpu.storage((t * d) as u64);
        let inv = self.gpu.storage(t as u64);
        self.gpu.submit(
            &[],
            block::rmsnorm_bwd(&self.gpu, &self.ids.as_block_ids(), &st.x_final, self.ps.w("vision.encoder.norm.weight"), &d_normed_buf, &d_x_final, &inv, Some(self.ps.g("vision.encoder.norm.weight")), d, t).as_slice(),
        );

        let mut d_x = d_x_final;
        for l in (0..e.n_layers).rev() {
            d_x = self.layer_bwd(l, &st.layers[l as usize], t, &d_x);
        }
        let d_concat_in = self.gpu.read(&d_x, (t * d) as usize);

        let (d_sam, d_query_bank) = d_concat_in.split_at((n_query * d) as usize);
        let query_bank_grad = self.ps.g(if local { "vision.query_local.weight" } else { "vision.query_global.weight" });
        self.gpu.write_f32(query_bank_grad, d_query_bank);

        d_sam.to_vec()
    }
}

/// Gather every local tile's projected output (in caller-supplied order - the
/// real model's order is row-major over the tile grid, width-first, per
/// `crate::rows`'s `row_plan`, and is a design assumption pending M6's
/// empirical check), then the global view's, then one learned separator row.
///
/// Pure host concatenation, on purpose: unlike v1's `RowGather`
/// (`crates/deepseek2ocr/src/layout.rs`), which exists to avoid a host round
/// trip for a layout that interleaves many `image_newline` rows between
/// token rows, v2's layout has no interleaving at all - every view's rows
/// are already one contiguous run (`crate::rows::RowPlan::runs`), so the
/// assembly is nothing more than placing already-computed `Vec<f32>` blocks
/// one after another. A device gather kernel here would cost a dispatch to
/// do exactly what `Vec::extend_from_slice` already does for free.
pub fn gather_rows(local_tiles: &[Vec<f32>], global: &[f32], separator: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    for tile in local_tiles {
        out.extend_from_slice(tile);
    }
    out.extend_from_slice(global);
    out.extend_from_slice(separator);
    out
}

/// The adjoint of [`gather_rows`]: split the decoder's splice gradient
/// (`[n_rows, d_model]`, `crate::model::DeepseekOcr2::backward`'s
/// `read_d_img_embeds()`) back into per-tile gradients, the global view's,
/// and the separator's own gradient - in the same order `gather_rows`
/// concatenated them.
///
/// A pure split, not a gather: nothing in [`gather_rows`] permutes a row or
/// shares one row across two destinations, so the adjoint of a concatenation
/// is exactly a concatenation's inverse - slicing, with no index table and no
/// accumulation (contrast v1's `RowGather::build_bwd`, whose `image_newline`
/// gradient sums over several rows because several rows read it; v2's
/// separator is read by exactly one row, so its own gradient IS that row -
/// see [`Resampler::write_view_separator_grad`]).
pub fn scatter_rows(d_block: &[f32], n_tiles: usize, n_query_local: u32, n_query_global: u32, d_model: u32) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
    let d = d_model as usize;
    let (local_len, global_len) = (n_query_local as usize * d, n_query_global as usize * d);
    assert_eq!(
        d_block.len(),
        n_tiles * local_len + global_len + d,
        "d_block's length does not match n_tiles={n_tiles}, n_query_local={n_query_local}, n_query_global={n_query_global}, d_model={d_model}"
    );
    let mut off = 0usize;
    let d_local_tiles: Vec<Vec<f32>> = (0..n_tiles)
        .map(|_| {
            let tile = d_block[off..off + local_len].to_vec();
            off += local_len;
            tile
        })
        .collect();
    let d_global = d_block[off..off + global_len].to_vec();
    off += global_len;
    let d_separator = d_block[off..off + d].to_vec();
    (d_local_tiles, d_global, d_separator)
}
