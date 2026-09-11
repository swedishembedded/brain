// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! TimesFM-3 **training** graph: SSA forward (this file, for now) plus a
//! hand-written reverse to follow, gated by `gradcheck::check_timesfm3`.
//!
//! # Scope: `core_forward` only, patch-aligned, one output patch
//!
//! This trains `pre_transformer_resblock` through the raw output-head logits
//! - exactly [`crate::model::Timesfm3::core_forward`]'s own boundary.
//! `preprocess.rs` (patching, RevIN, CPM refinement, linear detrending,
//! stitching) holds no learnable parameters and its output-side map is
//! affine with data-dependent, parameter-independent coefficients, so a loss
//! in original units reduces to one host-side rescale of the logits'
//! gradient - not a second backward. Consequently a training example is
//! restricted to `horizon <= output_patch_len`: no stitching, no
//! autoregressive CPM feedback, both outside the trainable graph.
//!
//! # Every checkpoint tensor stays separate - the PerDimScale fold is NOT baked in
//!
//! [`crate::model::Timesfm3::from_weights`] folds `per_dim_scale` into an
//! EFFECTIVE `query_ln.weight` once, at load time, because inference never
//! changes either tensor again. A trainer cannot do that: `query_ln.weight`
//! and `per_dim_scale.per_dim_scale` are both live, independently trained
//! parameters, and the saved checkpoint must keep the reference's own 445
//! tensors under their own names so it stays loadable by the SAME importer a
//! non-fine-tuned checkpoint uses. So [`Timesfm3Train`]'s `ParamStore` holds
//! `Timesfm3Config::param_list()` verbatim (unfolded), and every forward
//! recomputes the effective query gain (`query_ln.weight * 1.442695 *
//! softplus(per_dim_scale)`) fresh via [`folded_query_gain`] - a `[head_dim]`
//! host round-trip, negligible next to the matmuls it feeds. The backward
//! (a later commit) differentiates the UNFOLDED form back to `query_ln.weight`
//! and `per_dim_scale.per_dim_scale` separately, per `model.rs`'s own module
//! doc ("a later backward must differentiate the unfolded reference form").
//!
//! # Why the forward is recorded again here
//!
//! Same reason as `t5encoder::train`: `Timesfm3::core_forward` holds its SSA
//! buffers as function-locals and returns only the final logits, so a
//! sibling module cannot drive a reverse pass through them. This file
//! duplicates `core_forward`'s dispatch sequence, buffer for buffer, so the
//! reverse (to follow) can read every intermediate activation the forward
//! leaves behind. `tests::trainer_forward_matches_inference_core_forward`
//! asserts the two agree BITWISE on the raw logits for the same weights and
//! input, so a drift between them is a test failure, not a silent one.
//!
//! # The one structural change from `core_forward`: non-destructive ReLU
//!
//! `core_forward` calls `relu_inplace` directly on the GEMM output it will
//! never read again. The backward's `leaky_relu_bwd` (ReLU has no dedicated
//! backward kernel - `slope=0.0` makes leaky ReLU exact) needs the
//! PRE-activation value, so every ReLU here is `region_copy` (whole-buffer)
//! into a fresh SSA buffer, THEN `relu_inplace` on the copy - the pattern
//! `clip::model`'s own `region_copy` use documents. `add_inplace` needs no
//! such treatment: an addition's backward needs neither addend's value, only
//! the output gradient copied to both, so overwriting one addend in place
//! costs the reverse nothing.

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::block;
use paramstore::{ParamStore, Role};

use crate::config::Timesfm3Config;
use crate::model::{softplus, MASK_NEG, ROPE_THETA};

// ---- forward kernels (same set, same order, as `crate::model::PIPELINES`) ----
const K_MATMUL: usize = 0;
const K_MATMUL_REG3: usize = 1;
const K_BIAS_ADD: usize = 2;
const K_RELU: usize = 3;
const K_ADD: usize = 4;
const K_RMSNORM: usize = 5;
const K_RMSNORM_ROWS: usize = 6;
const K_ROPE_PARTIAL: usize = 7;
const K_ATTN_SCORES_QK_KMASK: usize = 8;
const K_ATTN_SOFTMAX_FULL: usize = 9;
const K_ATTN_APPLY_FULL: usize = 10;
const K_SWAP_AXES12_VEC: usize = 11;
const K_SOFTMAX_ROWS: usize = 12;
// ---- one addition for the forward's own non-destructive ReLU ----
const K_REGION_COPY: usize = 13;

/// Forward kernels only, for now - the backward half is appended (not
/// concatenated) in a later commit, same reasoning as
/// `t5encoder::train::TRAIN_PIPELINES`'s own module doc: every name must
/// appear exactly once, or the CPU backend's JIT rejects the duplicate.
pub const TRAIN_PIPELINES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("bias_add", kernels::BIAS_ADD),
    ("relu_inplace", kernels::RELU_INPLACE),
    ("add_inplace", kernels::ADD_INPLACE),
    ("rmsnorm", kernels::RMSNORM),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("rope_partial", kernels::ROPE_PARTIAL),
    ("attn_scores_qk_kmask", kernels::ATTN_SCORES_QK_KMASK),
    ("attn_softmax_full", kernels::ATTN_SOFTMAX_FULL),
    ("attn_apply_full", kernels::ATTN_APPLY_FULL),
    ("swap_axes12_vec", kernels::SWAP_AXES12_VEC),
    ("softmax_rows", kernels::SOFTMAX_ROWS),
    ("region_copy", kernels::REGION_COPY),
];

/// One mixing layer's SSA activations - every buffer the reverse (a later
/// commit) needs to read back. Named after the `core_forward` local variable
/// it mirrors, so the two stay easy to diff against each other by eye.
struct TrainLayer {
    /// This layer's own input hidden state (`pre_seq_attn_ln`'s `x`).
    h_in: DeviceBuffer,
    seq_in: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    vv: DeviceBuffer,
    qn: DeviceBuffer,
    kn: DeviceBuffer,
    probs_seq: DeviceBuffer,
    ctx: DeviceBuffer,
    seq_out: DeviceBuffer,
    /// `post_seq_attn_ln(seq_out) + h_in` - this layer's variate-attention input.
    h1: DeviceBuffer,
    var_in: DeviceBuffer,
    q2: DeviceBuffer,
    k2: DeviceBuffer,
    v2: DeviceBuffer,
    q2n: DeviceBuffer,
    k2n: DeviceBuffer,
    q2t: DeviceBuffer,
    k2t: DeviceBuffer,
    v2t: DeviceBuffer,
    probs_var: DeviceBuffer,
    ctx2t: DeviceBuffer,
    ctx2: DeviceBuffer,
    var_out: DeviceBuffer,
    /// `post_var_attn_ln(var_out) + h1` - this layer's feedforward input.
    h2: DeviceBuffer,
    ff_in: DeviceBuffer,
    /// The feedforward hidden GEMM output BEFORE `relu_inplace` - the one
    /// buffer `core_forward` itself does not keep, see the module doc.
    ff_h_pre: DeviceBuffer,
    ff_h_act: DeviceBuffer,
    ff_out: DeviceBuffer,
    /// `post_ff_ln(ff_out) + h2` - this layer's output, and the next layer's `h_in`.
    h_out: DeviceBuffer,
}

/// A trainable TimesFM-3 core: the SSA forward (backward to follow). See the
/// module doc for exactly what "trainable" covers.
pub struct Timesfm3Train {
    pub gpu: Gpu,
    pub cfg: Timesfm3Config,
    pub ps: ParamStore,
    b: usize,
    v: usize,
    n: usize,
    resblock_input: DeviceBuffer,
    seq_kmask: DeviceBuffer,
    var_kmask: DeviceBuffer,
    resblock_hidden_pre: DeviceBuffer,
    resblock_hidden_act: DeviceBuffer,
    resblock_resid: DeviceBuffer,
    /// `pre_transformer_resblock`'s own output - the first layer's `h_in`.
    resblock_out: DeviceBuffer,
    layers: Vec<TrainLayer>,
    logits: DeviceBuffer,
    steps: Vec<Step>,
}

impl Timesfm3Train {
    /// Build on an existing device (tests pass `gpu_core::testgpu::dev`).
    /// `resblock_input`/`patch_mask` are exactly `core_forward`'s own inputs
    /// (`[b*v*n, resblock_in_dim]` / `[b, v, n]`, `true` = masked); `init` is
    /// keyed by `Timesfm3Config::param_list()`'s own (checkpoint) names,
    /// UNFOLDED - see the module doc.
    pub fn new_on(gpu: Gpu, cfg: Timesfm3Config, resblock_input: &[f32], patch_mask: &[bool], b: usize, v: usize, n: usize, init: &HashMap<String, Vec<f32>>) -> Timesfm3Train {
        let rows = b * v * n;
        assert_eq!(resblock_input.len(), rows * cfg.resblock_in_dim());
        assert_eq!(patch_mask.len(), rows);

        let roles: Vec<(String, usize, Role)> = cfg.param_list().into_iter().map(|(name, shape)| (name, shape.iter().product::<usize>(), Role::Trainable)).collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, init);

        let d = cfg.model_dims;
        let g = &gpu;
        let mut steps = Vec::new();

        let resblock_input_buf = g.storage_init("resblock_input", resblock_input);

        // ---- pre_transformer_resblock: out = output(relu(hidden(x))) + residual(x) ----
        let hidden_pre = g.storage((rows * d) as u64);
        steps.push(linear(g, &ps, &resblock_input_buf, "pre_transformer_resblock.hidden_layer.weight", &hidden_pre, rows, cfg.resblock_in_dim(), d));
        let hidden_act = g.storage((rows * d) as u64);
        steps.push(g.step(K_REGION_COPY, &[&hidden_pre, &hidden_act], &[rows as u32, d as u32, d as u32, 0], (rows * d) as u32));
        steps.push(g.step(K_RELU, &[&hidden_act], &[(rows * d) as u32], (rows * d) as u32));
        let mut resblock_out = g.storage((rows * d) as u64);
        steps.push(linear(g, &ps, &hidden_act, "pre_transformer_resblock.output_layer.weight", &resblock_out, rows, d, d));
        let resid = g.storage((rows * d) as u64);
        steps.push(linear(g, &ps, &resblock_input_buf, "pre_transformer_resblock.residual_layer.weight", &resid, rows, cfg.resblock_in_dim(), d));
        steps.push(g.step(K_ADD, &[&resblock_out, &resid], &[(rows * d) as u32], (rows * d) as u32));

        // Additive key masks, both layouts - built once, reused by every
        // layer, exactly like `core_forward`.
        let seq_kmask_host: Vec<f32> = patch_mask.iter().map(|&m| if m { MASK_NEG } else { 0.0 }).collect();
        let mut var_kmask_host = vec![0.0f32; b * n * v];
        for bi in 0..b {
            for vi in 0..v {
                for ni in 0..n {
                    if patch_mask[(bi * v + vi) * n + ni] {
                        var_kmask_host[(bi * n + ni) * v + vi] = MASK_NEG;
                    }
                }
            }
        }
        let seq_kmask = g.storage_init("seq_kmask", &seq_kmask_host);
        let var_kmask = g.storage_init("var_kmask", &var_kmask_host);

        let mut layers = Vec::with_capacity(cfg.num_layers);
        let mut h_in = resblock_out.clone();
        for l in 0..cfg.num_layers {
            let p = format!("transformer_stack.layers.{l}");

            // ---- sequence attention: causal, per-(b,v) sequence over n, RoPE ----
            let seq_in = g.storage((rows * d) as u64);
            steps.push(rmsnorm(g, &ps, &h_in, &format!("{p}.pre_seq_attn_ln.weight"), &seq_in, d, rows, cfg.rms_norm_eps));
            let (q, k, vv) = (g.storage((rows * d) as u64), g.storage((rows * d) as u64), g.storage((rows * d) as u64));
            steps.push(linear(g, &ps, &seq_in, &format!("{p}.seq_attn.query_proj.weight"), &q, rows, d, d));
            steps.push(linear(g, &ps, &seq_in, &format!("{p}.seq_attn.key_proj.weight"), &k, rows, d, d));
            steps.push(linear(g, &ps, &seq_in, &format!("{p}.seq_attn.value_proj.weight"), &vv, rows, d, d));
            steps.push(rope_seq(g, &cfg, &q, rows, n));
            steps.push(rope_seq(g, &cfg, &k, rows, n));
            let (qn, kn) = (g.storage((rows * d) as u64), g.storage((rows * d) as u64));
            let q_gain = folded_query_gain(g, &ps, &format!("{p}.seq_attn.query_ln.weight"), &format!("{p}.seq_attn.per_dim_scale.per_dim_scale"));
            steps.push(rmsnorm_w(g, &q, &q_gain, &qn, cfg.head_dim, rows * cfg.num_heads, cfg.rms_norm_eps));
            steps.push(rmsnorm(g, &ps, &k, &format!("{p}.seq_attn.key_ln.weight"), &kn, cfg.head_dim, rows * cfg.num_heads, cfg.rms_norm_eps));
            let ctx = g.storage((rows * d) as u64);
            let probs_seq = g.storage((b * v * cfg.num_heads * n * n) as u64);
            steps.extend(attention(g, &cfg, &qn, &kn, &vv, &seq_kmask, &probs_seq, &ctx, b * v, n, true));
            let seq_out = g.storage((rows * d) as u64);
            steps.push(linear(g, &ps, &ctx, &format!("{p}.seq_attn.out_proj.weight"), &seq_out, rows, d, d));
            let seq_normed = g.storage((rows * d) as u64);
            steps.push(rmsnorm(g, &ps, &seq_out, &format!("{p}.post_seq_attn_ln.weight"), &seq_normed, d, rows, cfg.rms_norm_eps));
            steps.push(g.step(K_ADD, &[&seq_normed, &h_in], &[(rows * d) as u32], (rows * d) as u32));
            let h1 = seq_normed;

            // ---- variate attention: non-causal, per-(b,position) over v, no RoPE ----
            let var_in = g.storage((rows * d) as u64);
            steps.push(rmsnorm(g, &ps, &h1, &format!("{p}.pre_var_attn_ln.weight"), &var_in, d, rows, cfg.rms_norm_eps));
            let (q2, k2, v2) = (g.storage((rows * d) as u64), g.storage((rows * d) as u64), g.storage((rows * d) as u64));
            steps.push(linear(g, &ps, &var_in, &format!("{p}.var_attn.query_proj.weight"), &q2, rows, d, d));
            steps.push(linear(g, &ps, &var_in, &format!("{p}.var_attn.key_proj.weight"), &k2, rows, d, d));
            steps.push(linear(g, &ps, &var_in, &format!("{p}.var_attn.value_proj.weight"), &v2, rows, d, d));
            let (q2n, k2n) = (g.storage((rows * d) as u64), g.storage((rows * d) as u64));
            let q2_gain = folded_query_gain(g, &ps, &format!("{p}.var_attn.query_ln.weight"), &format!("{p}.var_attn.per_dim_scale.per_dim_scale"));
            steps.push(rmsnorm_w(g, &q2, &q2_gain, &q2n, cfg.head_dim, rows * cfg.num_heads, cfg.rms_norm_eps));
            steps.push(rmsnorm(g, &ps, &k2, &format!("{p}.var_attn.key_ln.weight"), &k2n, cfg.head_dim, rows * cfg.num_heads, cfg.rms_norm_eps));
            let (q2t, k2t, v2t) = (g.storage((rows * d) as u64), g.storage((rows * d) as u64), g.storage((rows * d) as u64));
            steps.push(swap12(g, &q2n, &q2t, b, v, n, d));
            steps.push(swap12(g, &k2n, &k2t, b, v, n, d));
            steps.push(swap12(g, &v2, &v2t, b, v, n, d));
            let ctx2t = g.storage((rows * d) as u64);
            let probs_var = g.storage((b * n * cfg.num_heads * v * v) as u64);
            steps.extend(attention(g, &cfg, &q2t, &k2t, &v2t, &var_kmask, &probs_var, &ctx2t, b * n, v, false));
            let ctx2 = g.storage((rows * d) as u64);
            steps.push(swap12(g, &ctx2t, &ctx2, b, n, v, d));
            let var_out = g.storage((rows * d) as u64);
            steps.push(linear(g, &ps, &ctx2, &format!("{p}.var_attn.out_proj.weight"), &var_out, rows, d, d));
            let var_normed = g.storage((rows * d) as u64);
            steps.push(rmsnorm(g, &ps, &var_out, &format!("{p}.post_var_attn_ln.weight"), &var_normed, d, rows, cfg.rms_norm_eps));
            steps.push(g.step(K_ADD, &[&var_normed, &h1], &[(rows * d) as u32], (rows * d) as u32));
            let h2 = var_normed;

            // ---- feedforward: ReLU, hidden width == model_dims (no 4x) ----
            let ff_in = g.storage((rows * d) as u64);
            steps.push(rmsnorm(g, &ps, &h2, &format!("{p}.pre_ff_ln.weight"), &ff_in, d, rows, cfg.rms_norm_eps));
            let ff_h_pre = g.storage((rows * cfg.hidden_dims) as u64);
            steps.push(linear(g, &ps, &ff_in, &format!("{p}.ff0.weight"), &ff_h_pre, rows, d, cfg.hidden_dims));
            let ff_h_act = g.storage((rows * cfg.hidden_dims) as u64);
            steps.push(g.step(K_REGION_COPY, &[&ff_h_pre, &ff_h_act], &[rows as u32, cfg.hidden_dims as u32, cfg.hidden_dims as u32, 0], (rows * cfg.hidden_dims) as u32));
            steps.push(g.step(K_RELU, &[&ff_h_act], &[(rows * cfg.hidden_dims) as u32], (rows * cfg.hidden_dims) as u32));
            let ff_out = g.storage((rows * d) as u64);
            steps.push(linear(g, &ps, &ff_h_act, &format!("{p}.ff1.weight"), &ff_out, rows, cfg.hidden_dims, d));
            let ff_normed = g.storage((rows * d) as u64);
            steps.push(rmsnorm(g, &ps, &ff_out, &format!("{p}.post_ff_ln.weight"), &ff_normed, d, rows, cfg.rms_norm_eps));
            steps.push(g.step(K_ADD, &[&ff_normed, &h2], &[(rows * d) as u32], (rows * d) as u32));
            let h_out = ff_normed;

            layers.push(TrainLayer {
                h_in: h_in.clone(),
                seq_in,
                q,
                k,
                vv,
                qn,
                kn,
                probs_seq,
                ctx,
                seq_out,
                h1,
                var_in,
                q2,
                k2,
                v2,
                q2n,
                k2n,
                q2t,
                k2t,
                v2t,
                probs_var,
                ctx2t,
                ctx2,
                var_out,
                h2,
                ff_in,
                ff_h_pre,
                ff_h_act,
                ff_out,
                h_out: h_out.clone(),
            });
            h_in = h_out;
        }

        // ---- output head: biased linear, no norm ----
        let head_out = cfg.head_out_dim();
        let logits = g.storage((rows * head_out) as u64);
        steps.push(linear(g, &ps, &h_in, "output_head.weight", &logits, rows, d, head_out));
        steps.push(g.step(K_BIAS_ADD, &[&logits, ps.w("output_head.bias")], &[rows as u32, head_out as u32], (rows * head_out) as u32));

        Timesfm3Train {
            gpu,
            cfg,
            ps,
            b,
            v,
            n,
            resblock_input: resblock_input_buf,
            seq_kmask,
            var_kmask,
            resblock_hidden_pre: hidden_pre,
            resblock_hidden_act: hidden_act,
            resblock_resid: resid,
            resblock_out,
            layers,
            logits,
            steps,
        }
    }

    pub fn forward(&self) {
        self.gpu.submit(&[], &self.steps);
    }

    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    /// The raw (pre-RevIN-reverse) output-head logits - `[b*v*n,
    /// output_patch_len*num_quantiles]`, exactly `core_forward`'s own return.
    pub fn read_logits(&self) -> Vec<f32> {
        self.gpu.read(&self.logits, self.b * self.v * self.n * self.cfg.head_out_dim())
    }
}

/// `out[m,n] = x[m,k] @ w[n,k]^T` (`w` is a PyTorch `nn.Linear` weight,
/// `[out_features, in_features]`), no bias, weight read from `ps`.
fn linear(g: &Gpu, ps: &ParamStore, x: &DeviceBuffer, weight_name: &str, out: &DeviceBuffer, m: usize, k: usize, n: usize) -> Step {
    let (kind, threads) = block::pick_gemm(m, n, K_MATMUL, K_MATMUL_REG3, false);
    g.step(kind, &[x, ps.w(weight_name), out], &[m as u32, k as u32, n as u32], threads)
}

fn rmsnorm_w(g: &Gpu, x: &DeviceBuffer, weight: &DeviceBuffer, out: &DeviceBuffer, dim: usize, rows: usize, eps: f32) -> Step {
    let coop = Some(K_RMSNORM_ROWS);
    let (kind, threads) = block::rms_variant(g, K_RMSNORM, coop, rows as u32, dim as u32);
    g.step(kind, &[x, weight, out], &[dim as u32, rows as u32, f(eps)], threads)
}

fn rmsnorm(g: &Gpu, ps: &ParamStore, x: &DeviceBuffer, weight_name: &str, out: &DeviceBuffer, dim: usize, rows: usize, eps: f32) -> Step {
    rmsnorm_w(g, x, ps.w(weight_name), out, dim, rows, eps)
}

/// RoPE, NeoX/half-split, applied to the full `head_dim` - identical dispatch
/// to `crate::model::Timesfm3::rope_seq`.
fn rope_seq(g: &Gpu, cfg: &Timesfm3Config, buf: &DeviceBuffer, rows: usize, tcols: usize) -> Step {
    let hd = cfg.head_dim as u32;
    g.step(
        K_ROPE_PARTIAL,
        &[buf],
        &[rows as u32, cfg.num_heads as u32, hd, hd * cfg.num_heads as u32, 0, tcols as u32, f(ROPE_THETA), hd],
        rows as u32 * cfg.num_heads as u32 * (hd / 2),
    )
}

/// One attention sublayer's scores->softmax->apply, identical dispatch to
/// `crate::model::Timesfm3::attention` except `probs` is a caller-owned SSA
/// buffer (the reverse needs it) rather than function-local scratch.
#[allow(clippy::too_many_arguments)]
fn attention(g: &Gpu, cfg: &Timesfm3Config, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, kmask: &DeviceBuffer, probs: &DeviceBuffer, ctx: &DeviceBuffer, bsz: usize, tcols: usize, causal: bool) -> Vec<Step> {
    let (h, hd, d) = (cfg.num_heads, cfg.head_dim, cfg.model_dims);
    let scores = g.storage((bsz * h * tcols * tcols) as u64);
    let (sk, st) = block::softmax_variant(g, K_ATTN_SOFTMAX_FULL, Some(K_SOFTMAX_ROWS), (bsz * h * tcols) as u32, tcols as u32);
    let softmax_step = if sk == K_SOFTMAX_ROWS {
        g.step(sk, &[&scores, probs], &[(bsz * h * tcols) as u32, tcols as u32], st)
    } else {
        g.step(sk, &[&scores, probs], &[bsz as u32, h as u32, tcols as u32], st)
    };
    vec![
        g.step(
            K_ATTN_SCORES_QK_KMASK,
            &[q, k, kmask, &scores],
            &[bsz as u32, h as u32, tcols as u32, hd as u32, d as u32, causal as u32, f(1.0)],
            (bsz * h * tcols * tcols) as u32,
        ),
        softmax_step,
        g.step(K_ATTN_APPLY_FULL, &[probs, v, ctx], &[bsz as u32, h as u32, tcols as u32, hd as u32, d as u32, d as u32], (bsz * h * tcols * hd) as u32),
    ]
}

/// `[b,a1,a2,d] -> [b,a2,a1,d]`, identical dispatch to
/// `crate::model::Timesfm3::swap12`.
fn swap12(g: &Gpu, src: &DeviceBuffer, dst: &DeviceBuffer, a0: usize, a1: usize, a2: usize, d: usize) -> Step {
    g.step(K_SWAP_AXES12_VEC, &[src, dst], &[a0 as u32, a1 as u32, a2 as u32, d as u32], (a0 * a1 * a2 * d) as u32)
}

/// The effective query gain (`query_ln.weight * 1.442695 *
/// softplus(per_dim_scale)`), recomputed fresh via a `[head_dim]` host
/// round-trip - see the module doc for why this cannot be folded once like
/// inference does. `gain_name`/`scale_name` are the checkpoint's own
/// `query_ln.weight` / `per_dim_scale.per_dim_scale` names for one attention
/// sublayer.
fn folded_query_gain(g: &Gpu, ps: &ParamStore, gain_name: &str, scale_name: &str) -> DeviceBuffer {
    let gain = g.read(ps.w(gain_name), ps.numel(gain_name));
    let scale = g.read(ps.w(scale_name), ps.numel(scale_name));
    let folded: Vec<f32> = gain.iter().zip(&scale).map(|(&gi, &si)| gi * 1.442_695_1 * softplus(si)).collect();
    g.storage_init("q_gain_eff", &folded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Timesfm3;

    fn skip() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    fn tiny_weights(cfg: &Timesfm3Config) -> HashMap<String, Vec<f32>> {
        cfg.param_list()
            .into_iter()
            .enumerate()
            .map(|(i, (k, s))| {
                let n: usize = s.iter().product();
                let data: Vec<f32> = (0..n).map(|j| (((i * 131 + j * 17) % 23) as f32 - 11.0) * 0.01).collect();
                (k, data)
            })
            .collect()
    }

    #[test]
    fn trainer_forward_matches_inference_core_forward() {
        if skip() {
            return;
        }
        let cfg = Timesfm3Config::tiny();
        let weights = tiny_weights(&cfg);
        let (b, v, n) = (2usize, 3usize, 4usize);
        let rows = b * v * n;
        let resblock_input: Vec<f32> = (0..rows * cfg.resblock_in_dim()).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        // A non-trivial mask (some leading patches masked for one (b,v) row)
        // so the trainer's kmask construction is exercised too, not just the
        // all-visible smoke shape `model.rs`'s own test uses.
        let mut mask = vec![false; rows];
        mask[0] = true;
        mask[1] = true;

        let inf = Timesfm3::from_weights(cfg.clone(), &weights).unwrap();
        let want = inf.core_forward(&resblock_input, &mask, b, v, n);

        let tr = Timesfm3Train::new_on(gpu_core::testgpu::dev(TRAIN_PIPELINES), cfg.clone(), &resblock_input, &mask, b, v, n, &weights);
        tr.forward();
        tr.poll_wait();
        let got = tr.read_logits();

        assert_eq!(got.len(), want.len());
        assert_eq!(got, want, "Timesfm3Train's forward must match Timesfm3::core_forward bitwise");
    }
}
