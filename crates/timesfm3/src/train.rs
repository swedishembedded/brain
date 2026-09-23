// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! TimesFM-3 **training** graph: the SSA forward and a hand-written reverse.
//!
//! Verified by [`tests::backward_matches_finite_difference_across_representative_tensors`],
//! a whole-model directional FD gradcheck, tight across every tensor kind
//! this model has, at 1/2/3 layers, on the default (Vulkan) backend AND on
//! `BRAIN_DEVICE=cpu`.
//!
//! # Scope: `core_forward` only, patch-aligned, one output patch
//!
//! This trains `pre_transformer_resblock` through the raw output-head
//! logits, exactly [`crate::model::Timesfm3::core_forward`]'s own boundary.
//! `preprocess.rs` (patching, RevIN, CPM refinement, linear detrending,
//! stitching) holds no learnable parameters and its output-side map is
//! affine with data-dependent, parameter-independent coefficients, so a loss
//! in original units reduces to one host-side rescale of the logits'
//! gradient, not a second backward. Consequently a training example is
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
//! differentiates the UNFOLDED form back to `query_ln.weight` and
//! `per_dim_scale.per_dim_scale` separately (also via a small host
//! round-trip - see [`Timesfm3Train::backward`]), per `model.rs`'s own
//! module doc ("a later backward must differentiate the unfolded reference
//! form").
//!
//! # Why the forward is recorded again here
//!
//! Same reason as `t5encoder::train`: `Timesfm3::core_forward` holds its SSA
//! buffers as function-locals and returns only the final logits, so a
//! sibling module cannot drive a reverse pass through them. This file
//! duplicates `core_forward`'s dispatch sequence, buffer for buffer, so the
//! reverse can read every intermediate activation the forward leaves behind.
//! `tests::trainer_forward_matches_inference_core_forward`
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
use optim::Optim;
use paramstore::{ParamStore, Role};

use crate::config::Timesfm3Config;
use crate::model::{softplus, MASK_NEG, ROPE_THETA};

// ---- forward kernels (same set, same order, as `crate::model::PIPELINES`) ----
const K_MATMUL: usize = 0;
const K_MATMUL_REG3: usize = 1;
const K_BIAS_ADD: usize = 2;
const K_RELU: usize = 3;
const K_ADD: usize = 4;
const K_RMSNORM_EPS: usize = 5;
const K_RMSNORM_ROWS: usize = 6;
const K_ROPE_PARTIAL: usize = 7;
const K_ATTN_SCORES_QK_KMASK: usize = 8;
const K_ATTN_SOFTMAX_FULL: usize = 9;
const K_ATTN_APPLY_FULL: usize = 10;
const K_SWAP_AXES12_VEC: usize = 11;
const K_SOFTMAX_ROWS: usize = 12;
// ---- one addition for the forward's own non-destructive ReLU ----
const K_REGION_COPY: usize = 13;
// ---- backward, APPENDED so every index above is unchanged (t5encoder::train's
// own rule: a kernel name registered twice is rejected by the CPU backend's JIT) ----
const K_ADD2: usize = 14;
const K_PACK_QKV: usize = 15;
const K_RMS_INV_EPS: usize = 16;
const K_RMSNORM_DW: usize = 17;
const K_RMSNORM_DX_EPS: usize = 18;
const K_MATMUL_DX: usize = 19;
const K_MATMUL_DW: usize = 20;
const K_ROPE_PARTIAL_BWD: usize = 21;
const K_LEAKY_RELU_BWD: usize = 22;
const K_BIAS_GRAD: usize = 23;
const K_DSCORES: usize = 24;
const K_DV: usize = 25;
const K_DQ_BIAS: usize = 26;
const K_DK_BIAS: usize = 27;
const K_DSCORES_BIDIR: usize = 28;
const K_DV_BIDIR: usize = 29;
const K_UNPACK_QKV: usize = 30;
// ---- LoRA: the fused adapter delta on a projection's OUTPUT, and the scaled
// copy its backward needs (the `attn_bwd_*`/`matmul_*` family has no scale
// parameter, so the factor is applied to the intermediate instead) ----
const K_AXPY: usize = 31;
const K_GRAD_SCALE: usize = 32;
// ---- the optimiser step (`optim::Optim`), so a trainer is still one device
// handle ----
const K_ADAMW: usize = 33;
const K_GRADNORM_SQ: usize = 34;
const K_CLIP_COEF: usize = 35;
const K_GRAD_SCALE_BUF: usize = 36;

/// Forward **and** backward kernels in one list, so a trainer is one device
/// handle. Every name appears exactly once.
pub const TRAIN_PIPELINES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("bias_add", kernels::BIAS_ADD),
    ("relu_inplace", kernels::RELU_INPLACE),
    ("add_inplace", kernels::ADD_INPLACE),
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("rope_partial", kernels::ROPE_PARTIAL),
    ("attn_scores_qk_kmask", kernels::ATTN_SCORES_QK_KMASK),
    ("attn_softmax_full", kernels::ATTN_SOFTMAX_FULL),
    ("attn_apply_full", kernels::ATTN_APPLY_FULL),
    ("swap_axes12_vec", kernels::SWAP_AXES12_VEC),
    ("softmax_rows", kernels::SOFTMAX_ROWS),
    ("region_copy", kernels::REGION_COPY),
    ("add2", kernels::ADD2),
    ("pack_qkv", kernels::PACK_QKV),
    ("rms_inv_eps", kernels::RMS_INV_EPS),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
    ("rmsnorm_dx_eps", kernels::RMSNORM_DX_EPS),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("rope_partial_bwd", kernels::ROPE_PARTIAL_BWD),
    ("leaky_relu_bwd", kernels::LEAKY_RELU_BWD),
    ("bias_grad", kernels::BIAS_GRAD),
    ("attn_bwd_dscores", kernels::ATTN_BWD_DSCORES),
    ("attn_bwd_dv", kernels::ATTN_BWD_DV),
    ("attn_bwd_dq_bias", kernels::ATTN_BWD_DQ_BIAS),
    ("attn_bwd_dk_bias", kernels::ATTN_BWD_DK_BIAS),
    ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR),
    ("attn_bwd_dv_bidir", kernels::ATTN_BWD_DV_BIDIR),
    ("unpack_qkv", kernels::UNPACK_QKV),
    ("axpy", kernels::AXPY),
    ("grad_scale", kernels::GRAD_SCALE),
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
];

/// LoRA hyper-parameters for [`Timesfm3Train`]. `alpha/rank` is the delta
/// scale, matching every other adapter in the workspace.
///
/// `targets` are matched by weight-name SUFFIX against
/// [`Timesfm3Config::param_list`]'s own checkpoint names, so one entry covers
/// the same leaf in every layer (`kronos::train::LoraCfg`'s rule; the
/// bare-leaf matching `qwen3` uses does not transfer, because this model's
/// leaves repeat across two different attention sublayers).
#[derive(Clone, Debug)]
pub struct LoraCfg {
    pub rank: usize,
    pub alpha: f32,
    pub targets: Vec<String>,
}

/// Every projection [`LoraCfg::attn`] adapts: the four square `[D, D]`
/// projections of BOTH mixing sublayers, plus the two feedforward matrices.
/// Ten per layer, each a whole-matrix placement - this architecture fuses no
/// QKV, so there is no packed region to slice.
pub const LORA_TARGETS: &[&str] = &[
    "seq_attn.query_proj.weight",
    "seq_attn.key_proj.weight",
    "seq_attn.value_proj.weight",
    "seq_attn.out_proj.weight",
    "var_attn.query_proj.weight",
    "var_attn.key_proj.weight",
    "var_attn.value_proj.weight",
    "var_attn.out_proj.weight",
    "ff0.weight",
    "ff1.weight",
];

impl LoraCfg {
    /// The default surface: [`LORA_TARGETS`].
    ///
    /// The norm gains (`pre_*_ln`, `post_*_ln`, `query_ln`, `key_ln`) and
    /// `per_dim_scale` are deliberately absent and are not adaptable at all.
    /// A LoRA factorisation of a `[head_dim]` or `[D]` VECTOR is not a
    /// low-rank anything - the product `B·A` for `out = 1` is a rank-1
    /// reparameterisation with more parameters than the tensor it replaces.
    /// The PerDimScale fold additionally makes `query_ln.weight` and
    /// `per_dim_scale` a product of two live tensors rather than a linear
    /// map, which is not the object LoRA is defined over.
    pub fn attn(rank: usize, alpha: f32) -> LoraCfg {
        LoraCfg { rank, alpha, targets: LORA_TARGETS.iter().map(|s| s.to_string()).collect() }
    }

    /// [`LoraCfg::attn`] plus the quantile output head. Optional because the
    /// head is the one target whose output width is the horizon rather than
    /// `model_dims`: adapting it changes how quantiles are read out, which is
    /// what a fine-tune onto a new quantile regime wants and what a fine-tune
    /// that only re-mixes an existing representation does not.
    pub fn with_output_head(mut self) -> LoraCfg {
        self.targets.push("output_head.weight".into());
        self
    }

    /// The delta scale, `alpha/rank`.
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    fn hits(&self, weight_name: &str) -> bool {
        self.targets.iter().any(|t| weight_name.ends_with(t.as_str()))
    }
}

/// The two additive key masks one `patch_mask` implies: `[b*v*n]` in the
/// sequence sublayer's own `(b,v,n)` order, and `[b*n*v]` transposed into the
/// variate sublayer's. `MASK_NEG` where masked, `0.0` where visible - an
/// ADDITIVE mask, so a visible key contributes nothing.
fn kmasks(patch_mask: &[bool], b: usize, v: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let seq: Vec<f32> = patch_mask.iter().map(|&m| if m { MASK_NEG } else { 0.0 }).collect();
    let mut var = vec![0.0f32; b * n * v];
    for bi in 0..b {
        for vi in 0..v {
            for ni in 0..n {
                if patch_mask[(bi * v + vi) * n + ni] {
                    var[(bi * n + ni) * v + vi] = MASK_NEG;
                }
            }
        }
    }
    (seq, var)
}

/// `[rank, in]` / `[out, rank]` under `{weight}.lora_a` / `{weight}.lora_b` -
/// the workspace-wide adapter spelling (`model::adapter::device::
/// is_adapter_param` is the one predicate that recognises them).
fn adapter_names(weight: &str) -> (String, String) {
    (format!("{weight}.lora_a"), format!("{weight}.lora_b"))
}

/// Every parameter's role under `lc`: the checkpoint's own tensors Frozen,
/// two adapters Trainable for each targeted weight. The adapters are
/// interleaved directly after the weight they adapt so the store's iteration
/// order stays readable next to `param_list`'s.
fn adapter_roles(cfg: &Timesfm3Config, lc: &LoraCfg) -> Vec<(String, usize, Role)> {
    let mut roles = Vec::new();
    for (name, shape) in cfg.param_list() {
        let numel: usize = shape.iter().product();
        roles.push((name.clone(), numel, Role::Frozen));
        if lc.hits(&name) {
            // `param_list` gives every adapted weight as `[out, in]`, which is
            // a PyTorch `nn.Linear` weight and exactly what `linear` expects.
            let (out, inn) = (shape[0], shape[1]);
            let (a, bnm) = adapter_names(&name);
            roles.push((a, lc.rank * inn, Role::Trainable));
            roles.push((bnm, out * lc.rank, Role::Trainable));
        }
    }
    roles
}

/// Fill in any adapter tensor `init` does not already carry. `B` is exactly
/// zero, so `W + (alpha/r)·B·A == W` and a freshly built adapter is a
/// bit-exact no-op; `A` is a small deterministic draw, because a zero `A`
/// alongside a zero `B` would leave `dB` identically zero and the adapter
/// would never leave the origin.
fn seed_adapters(cfg: &Timesfm3Config, lc: &LoraCfg, init: &mut HashMap<String, Vec<f32>>) {
    let mut state = 0x51a7_7143_0000_0001u64;
    for (name, shape) in cfg.param_list() {
        if !lc.hits(&name) {
            continue;
        }
        let (out, inn) = (shape[0], shape[1]);
        let (a, bnm) = adapter_names(&name);
        init.entry(a).or_insert_with(|| (0..lc.rank * inn).map(|_| lcg_next(&mut state) * 0.01).collect());
        init.entry(bnm).or_insert_with(|| vec![0.0f32; out * lc.rank]);
    }
}

/// The adapter wiring both passes share: the config, and the three scratch
/// buffers they reuse.
///
/// Shared rather than SSA, matching `kronos::train` and `qwen3::model`: the
/// reverse RECOMPUTES `x·Aᵀ` from the forward's own (SSA) input activation
/// instead of reading a cached copy, so no adapter intermediate has to
/// survive past the three consecutive steps that produce and consume it.
struct LoraWiring {
    cfg: Option<LoraCfg>,
    /// `[rows * rank]` - `x·Aᵀ`, and in the reverse the same product scaled.
    mid: DeviceBuffer,
    /// `[rows * rank]` - `d(x·Aᵀ)`.
    dmid: DeviceBuffer,
    /// `[rows * max_out]` - the adapter's own contribution to a projection's
    /// output, before the `axpy` folds it in.
    out: DeviceBuffer,
}

impl LoraWiring {
    /// `(rank, alpha/rank)` if `weight` carries an adapter.
    fn of(&self, weight: &str) -> Option<(u32, f32)> {
        self.cfg.as_ref().filter(|lc| lc.hits(weight)).map(|lc| (lc.rank as u32, lc.scale()))
    }
}

/// `y += (alpha/r)·(x·Aᵀ)·Bᵀ` for an adapted projection; nothing at all
/// otherwise, so a non-LoRA graph is bit-unchanged. Must be wired at exactly
/// the projections [`proj_bwd`] adapts, or the forward and the reverse
/// disagree about which weights are frozen.
#[allow(clippy::too_many_arguments)]
fn lora_fwd(g: &Gpu, ps: &ParamStore, lw: &LoraWiring, weight: &str, x: &DeviceBuffer, y: &DeviceBuffer, m: usize, k: usize, nout: usize) -> Vec<Step> {
    let Some((r, scale)) = lw.of(weight) else { return Vec::new() };
    let (a, bnm) = adapter_names(weight);
    vec![
        g.step(K_MATMUL, &[x, ps.w(&a), &lw.mid], &[m as u32, k as u32, r], (m * r as usize) as u32),
        g.step(K_MATMUL, &[&lw.mid, ps.w(&bnm), &lw.out], &[m as u32, r, nout as u32], (m * nout) as u32),
        g.step(K_AXPY, &[y, &lw.out], &[(m * nout) as u32, f(scale)], (m * nout) as u32),
    ]
}

/// One mixing layer's SSA activations - every buffer the reverse pass needs
/// to read back (`build_backward` takes these as its own local parameters
/// rather than reading `self.layers[l].foo`, so a few fields the reverse
/// recomputes instead of rereading - e.g. `q2t`/`k2t`/`v2t`, redone inside
/// `attn_sublayer_bwd` from `q2n`/`k2n`/`v2` - go unread; kept anyway so this
/// struct stays a complete, literal record of the forward, matching
/// `core_forward`'s own local variable it mirrors one for one). Named after
/// that local variable, so the two stay easy to diff against each other by eye.
#[allow(dead_code)]
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

/// A trainable TimesFM-3 core: the SSA forward and its reverse. See the
/// module doc for exactly what "trainable" covers.
///
/// The activation fields (`resblock_hidden_*`/`resblock_resid`/
/// `resblock_out`/`layers`) are never read via `self.` - `build_backward` is
/// called with the LOCAL variables from the constructor directly, before this
/// struct exists, rather than as a `&self` method the way
/// `t5encoder::train::T5Trainer::build_bwd_steps` is. They stay struct fields
/// anyway because something must own these buffers for as long as
/// `self.steps`/`self.bwd_steps` can still be resubmitted (repeated
/// `forward()`/`backward()` calls across training steps).
///
/// `resblock_input`/`seq_kmask`/`var_kmask` are different: they are the
/// model's INPUT rather than an activation, and [`Timesfm3Train::set_input`]
/// rewrites all three in place so a training loop can move to the next batch
/// without rebuilding the graph.
#[allow(dead_code)]
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
    /// The objective's gradient w.r.t. the raw output-head logits - the
    /// backward's seed, uploaded fresh by [`Timesfm3Train::backward`].
    d_logits: DeviceBuffer,
    bwd_steps: Vec<Step>,
    /// Per query-attention-sublayer (one seq + one var per layer): the raw
    /// `d(effective query gain)` buffer `rmsnorm_dw` wrote, plus the
    /// checkpoint names of the two ORIGINAL tensors it must be split back
    /// into. Processed host-side, after `bwd_steps` runs - see
    /// [`Timesfm3Train::backward`] and the module doc's PerDimScale section.
    qgain_fold_temps: Vec<(String, String, DeviceBuffer)>,
    /// Per query-attention-sublayer: `(query_ln name, per_dim_scale name, the
    /// ONE buffer holding their fold)`. The effective query gain is not a
    /// tensor of its own - it is a function of two live parameters - so it
    /// cannot be bound once and left alone the way every other weight in this
    /// graph is: [`Timesfm3Train::forward`] recomputes all of them from the
    /// current `ps` before submitting. One buffer per sublayer, shared by the
    /// forward's `rmsnorm_w` and the reverse's `rmsnorm_bwd`, so a refresh
    /// cannot leave the two disagreeing about which weights they differentiate.
    q_gains: Vec<(String, String, DeviceBuffer)>,
    /// The adapter config, when this is a LoRA trainer. `None` is a full
    /// fine-tune and every dispatch below is bit-unchanged from before LoRA
    /// existed.
    lora: Option<LoraCfg>,
    /// The on-device AdamW step. Held rather than constructed per call: it
    /// caches the optimiser graph and rebuilds it only when the clip mode or
    /// the trainable-parameter count changes.
    opt: Optim,
}

impl Timesfm3Train {
    /// Build on an existing device (tests pass `gpu_core::testgpu::dev`).
    /// `resblock_input`/`patch_mask` are exactly `core_forward`'s own inputs
    /// (`[b*v*n, resblock_in_dim]` / `[b, v, n]`, `true` = masked); `init` is
    /// keyed by `Timesfm3Config::param_list()`'s own (checkpoint) names,
    /// UNFOLDED - see the module doc.
    pub fn new_on(gpu: Gpu, cfg: Timesfm3Config, resblock_input: &[f32], patch_mask: &[bool], b: usize, v: usize, n: usize, init: &HashMap<String, Vec<f32>>) -> Timesfm3Train {
        Timesfm3Train::build(gpu, cfg, None, resblock_input, patch_mask, b, v, n, init)
    }

    /// A LoRA trainer over the same graph: every checkpoint tensor is FROZEN
    /// and only the `.lora_a`/`.lora_b` adapters on `lora.targets` are
    /// trainable. `init` supplies the base checkpoint; the adapters are
    /// seeded here if absent (`A` small and deterministic, `B` exactly zero,
    /// so the adapter is an exact no-op at construction), which is what lets
    /// a caller hand this the same weight map an inference load would use.
    #[allow(clippy::too_many_arguments)]
    pub fn new_lora_on(gpu: Gpu, cfg: Timesfm3Config, lora: LoraCfg, resblock_input: &[f32], patch_mask: &[bool], b: usize, v: usize, n: usize, init: &HashMap<String, Vec<f32>>) -> Timesfm3Train {
        Timesfm3Train::build(gpu, cfg, Some(lora), resblock_input, patch_mask, b, v, n, init)
    }

    #[allow(clippy::too_many_arguments)]
    fn build(gpu: Gpu, cfg: Timesfm3Config, lora: Option<LoraCfg>, resblock_input: &[f32], patch_mask: &[bool], b: usize, v: usize, n: usize, init: &HashMap<String, Vec<f32>>) -> Timesfm3Train {
        let rows = b * v * n;
        assert_eq!(resblock_input.len(), rows * cfg.resblock_in_dim());
        assert_eq!(patch_mask.len(), rows);

        // Under LoRA the base is Frozen (weights only, no gradient and no
        // optimiser moments) and the adapters are the whole trainable set -
        // `kronos::train`'s and `qwen3::model`'s identical split. Without it
        // every tensor is Trainable and this is an ordinary full fine-tune.
        let mut init2;
        let init: &HashMap<String, Vec<f32>> = match &lora {
            None => init,
            Some(lc) => {
                init2 = init.clone();
                seed_adapters(&cfg, lc, &mut init2);
                &init2
            }
        };
        let roles: Vec<(String, usize, Role)> = match &lora {
            None => cfg.param_list().into_iter().map(|(name, shape)| (name, shape.iter().product::<usize>(), Role::Trainable)).collect(),
            Some(lc) => adapter_roles(&cfg, lc),
        };
        let ps = ParamStore::new_with_roles(&gpu, roles, init);

        let d = cfg.model_dims;
        let g = &gpu;
        let mut steps = Vec::new();

        // One set of adapter scratch buffers for the whole graph. `out` is
        // sized for the WIDEST adapted projection, which is the quantile head
        // when it is targeted and the feedforward hidden otherwise.
        let rank = lora.as_ref().map(|lc| lc.rank).unwrap_or(0);
        let widest = d.max(cfg.hidden_dims).max(cfg.head_out_dim());
        let lw = LoraWiring {
            cfg: lora.clone(),
            mid: g.storage((rows * rank.max(1)) as u64),
            dmid: g.storage((rows * rank.max(1)) as u64),
            out: g.storage((rows * widest) as u64),
        };

        let resblock_input_buf = g.storage_init("resblock_input", resblock_input);

        // ---- pre_transformer_resblock: out = output(relu(hidden(x))) + residual(x) ----
        let hidden_pre = g.storage((rows * d) as u64);
        steps.push(linear(g, &ps, &resblock_input_buf, "pre_transformer_resblock.hidden_layer.weight", &hidden_pre, rows, cfg.resblock_in_dim(), d));
        let hidden_act = g.storage((rows * d) as u64);
        steps.push(g.step(K_REGION_COPY, &[&hidden_pre, &hidden_act], &[rows as u32, d as u32, d as u32, 0], (rows * d) as u32));
        steps.push(g.step(K_RELU, &[&hidden_act], &[(rows * d) as u32], (rows * d) as u32));
        let resblock_out = g.storage((rows * d) as u64);
        steps.push(linear(g, &ps, &hidden_act, "pre_transformer_resblock.output_layer.weight", &resblock_out, rows, d, d));
        let resid = g.storage((rows * d) as u64);
        steps.push(linear(g, &ps, &resblock_input_buf, "pre_transformer_resblock.residual_layer.weight", &resid, rows, cfg.resblock_in_dim(), d));
        steps.push(g.step(K_ADD, &[&resblock_out, &resid], &[(rows * d) as u32], (rows * d) as u32));

        // Additive key masks, both layouts - reused by every layer, exactly
        // like `core_forward`. Rebuilt (not reallocated) by `set_input`.
        let (seq_kmask_host, var_kmask_host) = kmasks(patch_mask, b, v, n);
        let seq_kmask = g.storage_init("seq_kmask", &seq_kmask_host);
        let var_kmask = g.storage_init("var_kmask", &var_kmask_host);

        // One folded-gain buffer per query-attention sublayer, allocated here
        // and refreshed by every `forward()` - see the `q_gains` field doc.
        // The reverse pass looks its own up by `query_ln` name rather than
        // folding a second time, so forward and backward can never drift onto
        // different gains.
        let mut q_gains: Vec<(String, String, DeviceBuffer)> = Vec::with_capacity(2 * cfg.num_layers);
        for l in 0..cfg.num_layers {
            let p = format!("transformer_stack.layers.{l}");
            for kind in ["seq_attn", "var_attn"] {
                let (gain, scale) = (format!("{p}.{kind}.query_ln.weight"), format!("{p}.{kind}.per_dim_scale.per_dim_scale"));
                let buf = g.storage(cfg.head_dim as u64);
                refresh_query_gain(g, &ps, &gain, &scale, &buf);
                q_gains.push((gain, scale, buf));
            }
        }

        let mut layers = Vec::with_capacity(cfg.num_layers);
        let mut h_in = resblock_out.clone();
        for l in 0..cfg.num_layers {
            let p = format!("transformer_stack.layers.{l}");

            // ---- sequence attention: causal, per-(b,v) sequence over n, RoPE ----
            let seq_in = g.storage((rows * d) as u64);
            steps.push(rmsnorm(g, &ps, &h_in, &format!("{p}.pre_seq_attn_ln.weight"), &seq_in, d, rows, cfg.rms_norm_eps));
            let (q, k, vv) = (g.storage((rows * d) as u64), g.storage((rows * d) as u64), g.storage((rows * d) as u64));
            steps.push(linear(g, &ps, &seq_in, &format!("{p}.seq_attn.query_proj.weight"), &q, rows, d, d));
            steps.extend(lora_fwd(g, &ps, &lw, &format!("{p}.seq_attn.query_proj.weight"), &seq_in, &q, rows, d, d));
            steps.push(linear(g, &ps, &seq_in, &format!("{p}.seq_attn.key_proj.weight"), &k, rows, d, d));
            steps.extend(lora_fwd(g, &ps, &lw, &format!("{p}.seq_attn.key_proj.weight"), &seq_in, &k, rows, d, d));
            steps.push(linear(g, &ps, &seq_in, &format!("{p}.seq_attn.value_proj.weight"), &vv, rows, d, d));
            steps.extend(lora_fwd(g, &ps, &lw, &format!("{p}.seq_attn.value_proj.weight"), &seq_in, &vv, rows, d, d));
            steps.push(rope_seq(g, &cfg, &q, rows, n));
            steps.push(rope_seq(g, &cfg, &k, rows, n));
            let (qn, kn) = (g.storage((rows * d) as u64), g.storage((rows * d) as u64));
            let q_gain = query_gain_of(&q_gains, &format!("{p}.seq_attn.query_ln.weight"));
            steps.push(rmsnorm_w(g, &q, q_gain, &qn, cfg.head_dim, rows * cfg.num_heads, cfg.rms_norm_eps));
            steps.push(rmsnorm(g, &ps, &k, &format!("{p}.seq_attn.key_ln.weight"), &kn, cfg.head_dim, rows * cfg.num_heads, cfg.rms_norm_eps));
            let ctx = g.storage((rows * d) as u64);
            let probs_seq = g.storage((b * v * cfg.num_heads * n * n) as u64);
            steps.extend(attention(g, &cfg, &qn, &kn, &vv, &seq_kmask, &probs_seq, &ctx, b * v, n, true));
            let seq_out = g.storage((rows * d) as u64);
            steps.push(linear(g, &ps, &ctx, &format!("{p}.seq_attn.out_proj.weight"), &seq_out, rows, d, d));
            steps.extend(lora_fwd(g, &ps, &lw, &format!("{p}.seq_attn.out_proj.weight"), &ctx, &seq_out, rows, d, d));
            let seq_normed = g.storage((rows * d) as u64);
            steps.push(rmsnorm(g, &ps, &seq_out, &format!("{p}.post_seq_attn_ln.weight"), &seq_normed, d, rows, cfg.rms_norm_eps));
            steps.push(g.step(K_ADD, &[&seq_normed, &h_in], &[(rows * d) as u32], (rows * d) as u32));
            let h1 = seq_normed;

            // ---- variate attention: non-causal, per-(b,position) over v, no RoPE ----
            let var_in = g.storage((rows * d) as u64);
            steps.push(rmsnorm(g, &ps, &h1, &format!("{p}.pre_var_attn_ln.weight"), &var_in, d, rows, cfg.rms_norm_eps));
            let (q2, k2, v2) = (g.storage((rows * d) as u64), g.storage((rows * d) as u64), g.storage((rows * d) as u64));
            steps.push(linear(g, &ps, &var_in, &format!("{p}.var_attn.query_proj.weight"), &q2, rows, d, d));
            steps.extend(lora_fwd(g, &ps, &lw, &format!("{p}.var_attn.query_proj.weight"), &var_in, &q2, rows, d, d));
            steps.push(linear(g, &ps, &var_in, &format!("{p}.var_attn.key_proj.weight"), &k2, rows, d, d));
            steps.extend(lora_fwd(g, &ps, &lw, &format!("{p}.var_attn.key_proj.weight"), &var_in, &k2, rows, d, d));
            steps.push(linear(g, &ps, &var_in, &format!("{p}.var_attn.value_proj.weight"), &v2, rows, d, d));
            steps.extend(lora_fwd(g, &ps, &lw, &format!("{p}.var_attn.value_proj.weight"), &var_in, &v2, rows, d, d));
            let (q2n, k2n) = (g.storage((rows * d) as u64), g.storage((rows * d) as u64));
            let q2_gain = query_gain_of(&q_gains, &format!("{p}.var_attn.query_ln.weight"));
            steps.push(rmsnorm_w(g, &q2, q2_gain, &q2n, cfg.head_dim, rows * cfg.num_heads, cfg.rms_norm_eps));
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
            steps.extend(lora_fwd(g, &ps, &lw, &format!("{p}.var_attn.out_proj.weight"), &ctx2, &var_out, rows, d, d));
            let var_normed = g.storage((rows * d) as u64);
            steps.push(rmsnorm(g, &ps, &var_out, &format!("{p}.post_var_attn_ln.weight"), &var_normed, d, rows, cfg.rms_norm_eps));
            steps.push(g.step(K_ADD, &[&var_normed, &h1], &[(rows * d) as u32], (rows * d) as u32));
            let h2 = var_normed;

            // ---- feedforward: ReLU, hidden width == model_dims (no 4x) ----
            let ff_in = g.storage((rows * d) as u64);
            steps.push(rmsnorm(g, &ps, &h2, &format!("{p}.pre_ff_ln.weight"), &ff_in, d, rows, cfg.rms_norm_eps));
            let ff_h_pre = g.storage((rows * cfg.hidden_dims) as u64);
            steps.push(linear(g, &ps, &ff_in, &format!("{p}.ff0.weight"), &ff_h_pre, rows, d, cfg.hidden_dims));
            steps.extend(lora_fwd(g, &ps, &lw, &format!("{p}.ff0.weight"), &ff_in, &ff_h_pre, rows, d, cfg.hidden_dims));
            let ff_h_act = g.storage((rows * cfg.hidden_dims) as u64);
            steps.push(g.step(K_REGION_COPY, &[&ff_h_pre, &ff_h_act], &[rows as u32, cfg.hidden_dims as u32, cfg.hidden_dims as u32, 0], (rows * cfg.hidden_dims) as u32));
            steps.push(g.step(K_RELU, &[&ff_h_act], &[(rows * cfg.hidden_dims) as u32], (rows * cfg.hidden_dims) as u32));
            let ff_out = g.storage((rows * d) as u64);
            steps.push(linear(g, &ps, &ff_h_act, &format!("{p}.ff1.weight"), &ff_out, rows, cfg.hidden_dims, d));
            steps.extend(lora_fwd(g, &ps, &lw, &format!("{p}.ff1.weight"), &ff_h_act, &ff_out, rows, cfg.hidden_dims, d));
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
        steps.extend(lora_fwd(g, &ps, &lw, "output_head.weight", &h_in, &logits, rows, d, head_out));
        steps.push(g.step(K_BIAS_ADD, &[&logits, ps.w("output_head.bias")], &[rows as u32, head_out as u32], (rows * head_out) as u32));

        let d_logits = g.storage((rows * head_out) as u64);
        let (bwd_steps, qgain_fold_temps) = build_backward(g, &cfg, &ps, &lw, &q_gains, &layers, &resblock_input_buf, &hidden_pre, &hidden_act, &h_in, &d_logits, b, v, n);

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
            d_logits,
            bwd_steps,
            qgain_fold_temps,
            q_gains,
            lora,
            opt: Optim::new(K_ADAMW, K_GRADNORM_SQ, K_GRAD_SCALE, K_CLIP_COEF, K_GRAD_SCALE_BUF),
        }
    }

    /// Refold every effective query gain from the weights `ps` holds RIGHT
    /// NOW. Called by [`Timesfm3Train::forward`]; the only reason it is
    /// separate is that the reverse pass reads the same buffers, so the
    /// contract is "the gain matches the weights the last forward saw".
    fn refresh_query_gains(&self) {
        for (gain, scale, buf) in &self.q_gains {
            refresh_query_gain(&self.gpu, &self.ps, gain, scale, buf);
        }
    }

    /// Point the graph at a NEW batch, in place: the same shape `(b, v, n)`
    /// this trainer was built for, a fresh `resblock_input` and a fresh
    /// `patch_mask`.
    ///
    /// This is what makes a training loop possible at all. Every other buffer
    /// in the graph is an activation the forward recomputes, so re-uploading
    /// the model's own INPUT (and the two additive key masks its mask
    /// implies) is the whole of "next batch"; rebuilding the trainer per
    /// batch would rebuild several thousand `Step`s and reallocate every
    /// activation to change two buffers.
    pub fn set_input(&self, resblock_input: &[f32], patch_mask: &[bool]) {
        let rows = self.b * self.v * self.n;
        assert_eq!(resblock_input.len(), rows * self.cfg.resblock_in_dim(), "resblock_input is the shape this trainer was built for");
        assert_eq!(patch_mask.len(), rows, "patch_mask is one flag per (b, v, patch)");
        self.gpu.write_f32(&self.resblock_input, resblock_input);
        let (seq, var) = kmasks(patch_mask, self.b, self.v, self.n);
        self.gpu.write_f32(&self.seq_kmask, &seq);
        self.gpu.write_f32(&self.var_kmask, &var);
    }

    /// One AdamW step over the trainable set, `t` 1-based. `clip`, when
    /// given, is `clip_grad_norm_` semantics over the global gradient norm.
    /// Entirely on-device, one submit, no host readback.
    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>) {
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.999, 1e-8, clip, 1.0);
    }

    pub fn forward(&self) {
        self.refresh_query_gains();
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

    /// Zero every parameter gradient. Call once per training step, BEFORE
    /// [`Timesfm3Train::backward`] - the reverse pass accumulates into them
    /// (`matmul_dw`/`rmsnorm_dw`/`bias_grad` all `+=`).
    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    /// Run the reverse pass for the objective whose gradient w.r.t. the raw
    /// output-head logits is `d_logits` (`[b*v*n, head_out_dim]` - the
    /// caller's job to produce, e.g. from `pinball_grad_w` composed with the
    /// host-side RevIN-reverse/trend-readd rescale the module doc describes).
    ///
    /// The forward must already have run on the current weights/input: the
    /// backward reads the SSA activation buffers it left behind. Two phases:
    /// the device reverse pass (everything except the PerDimScale fold),
    /// then a host-side cleanup that splits each query-attention sublayer's
    /// `d(effective gain)` back into `d(query_ln.weight)` and
    /// `d(per_dim_scale)` - see the module doc for why this cannot be a
    /// device kernel step (it targets checkpoint tensors the main pass never
    /// touches directly).
    pub fn backward(&self, d_logits: &[f32]) {
        let head_out = self.cfg.head_out_dim();
        assert_eq!(d_logits.len(), self.b * self.v * self.n * head_out);
        self.gpu.write_f32(&self.d_logits, d_logits);
        // Clear the fold temps HERE, per backward - not in `zero_grads`.
        // `rmsnorm_dw` accumulates, and these buffers are not `ParamStore`
        // tensors, so nothing else clears them. They also must not accumulate
        // across the several backwards of one gradient-accumulation step the
        // way a real grad buffer does: the host split below is linear in this
        // temp and ADDS its result into the parameter gradient, so letting the
        // temp carry over would fold each microbatch's running total in again
        // instead of its own contribution.
        let temps: Vec<&DeviceBuffer> = self.qgain_fold_temps.iter().map(|(_, _, t)| t).collect();
        self.gpu.submit(&temps, &self.bwd_steps);
        self.gpu.poll_wait();

        for (gain_name, scale_name, d_temp) in &self.qgain_fold_temps {
            let head_dim = self.ps.numel(gain_name);
            let d_gain_eff = self.gpu.read(d_temp, head_dim);
            let gain = self.gpu.read(self.ps.w(gain_name), head_dim);
            let scale = self.gpu.read(self.ps.w(scale_name), head_dim);
            // g_eff = g * 1.442695 * softplus(w) => dg = D*1.442695*softplus(w),
            // dw = D*g*1.442695*sigmoid(w) (d/dw softplus(w) = sigmoid(w)).
            let d_gain: Vec<f32> = d_gain_eff.iter().zip(&scale).map(|(&di, &si)| di * 1.442_695_1 * softplus(si)).collect();
            let d_scale: Vec<f32> = d_gain_eff
                .iter()
                .zip(&gain)
                .zip(&scale)
                .map(|((&di, &gi), &si)| di * gi * 1.442_695_1 * (1.0 / (1.0 + (-si).exp())))
                .collect();
            let existing_dg = self.ps.read_grad(&self.gpu, gain_name);
            let new_dg: Vec<f32> = existing_dg.iter().zip(&d_gain).map(|(&e, &d)| e + d).collect();
            self.gpu.write_f32(self.ps.g(gain_name), &new_dg);
            let existing_ds = self.ps.read_grad(&self.gpu, scale_name);
            let new_ds: Vec<f32> = existing_ds.iter().zip(&d_scale).map(|(&e, &d)| e + d).collect();
            self.gpu.write_f32(self.ps.g(scale_name), &new_ds);
        }
    }

    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        assert_eq!(data.len(), self.ps.numel(name), "{name}: weight size");
        self.gpu.write_f32(self.ps.w(name), data);
    }

    /// The names the optimiser actually steps: every checkpoint tensor for a
    /// full fine-tune, and ONLY `*.lora_a`/`*.lora_b` under LoRA. A frozen
    /// base has no gradient buffer, so this is also the exact set
    /// [`Timesfm3Train::read_grad`] may be called with.
    pub fn param_names(&self) -> Vec<String> {
        self.ps.opt_params().iter().map(|(n, _)| n.clone()).collect()
    }

    /// The adapter config, if this is a LoRA trainer.
    pub fn lora(&self) -> Option<&LoraCfg> {
        self.lora.as_ref()
    }

    /// Every tensor of [`Timesfm3Config::param_list`] under its own
    /// CHECKPOINT name, with any LoRA delta folded in
    /// (`W += (alpha/r)·B·A`) and no adapter tensors left over.
    ///
    /// This, not an adapter sidecar, is what a TimesFM-3 fine-tune writes:
    /// this module's own doc records that the saved checkpoint has to keep
    /// the reference's tensors under the reference's names so the SAME
    /// importer loads a fine-tuned and a stock checkpoint. `kronos::train`
    /// saves through the identical shape for the identical reason.
    pub fn to_reference_weights(&self) -> HashMap<String, Vec<f32>> {
        let mut out = HashMap::new();
        for (name, shape) in self.cfg.param_list() {
            let mut w = self.read_weight(&name);
            if let Some(lc) = self.lora.as_ref().filter(|lc| lc.hits(&name)) {
                let (a_name, b_name) = adapter_names(&name);
                let (a, b) = (self.read_weight(&a_name), self.read_weight(&b_name));
                fold_delta(&mut w, &a, &b, lc.rank, shape[1], lc.scale());
            }
            out.insert(name, w);
        }
        out
    }
}

/// `W[o,i] += scale·Σ_k B[o,k]·A[k,i]` over a `[out, in]` row-major weight,
/// `A` `[r, in]`, `B` `[out, r]`. The same contraction
/// `model::lora::device_adapter::fold_adapter_into` performs when it reads an
/// adapter file back, kept here because this model folds from LIVE device
/// tensors (there is no adapter sidecar - see
/// [`Timesfm3Train::to_reference_weights`]).
fn fold_delta(w: &mut [f32], a: &[f32], b: &[f32], r: usize, inn: usize, scale: f32) {
    let out = w.len() / inn;
    for o in 0..out {
        for i in 0..inn {
            let mut acc = 0.0f32;
            for k in 0..r {
                acc += b[o * r + k] * a[k * inn + i];
            }
            w[o * inn + i] += scale * acc;
        }
    }
}

/// `out[m,n] = x[m,k] @ w[n,k]^T` (`w` is a PyTorch `nn.Linear` weight,
/// `[out_features, in_features]`), no bias, weight read from `ps`.
fn linear(g: &Gpu, ps: &ParamStore, x: &DeviceBuffer, weight_name: &str, out: &DeviceBuffer, m: usize, k: usize, n: usize) -> Step {
    let (kind, threads) = block::pick_gemm(m, n, K_MATMUL, K_MATMUL_REG3, false);
    g.dispatch(kind, &[x, ps.w(weight_name), out], &[m as u32, k as u32, n as u32], threads)
}

/// The reference index is `rmsnorm_eps`, never the fixed-1e-6 `rmsnorm` - see
/// `crate::model::Timesfm3::rmsnorm` for why pairing the latter with
/// `rmsnorm_rows` makes the epsilon depend on the device.
fn rmsnorm_w(g: &Gpu, x: &DeviceBuffer, weight: &DeviceBuffer, out: &DeviceBuffer, dim: usize, rows: usize, eps: f32) -> Step {
    let coop = Some(K_RMSNORM_ROWS);
    let (kind, threads) = block::rms_variant(g, K_RMSNORM_EPS, coop, rows as u32, dim as u32);
    g.dispatch(kind, &[x, weight, out], &[dim as u32, rows as u32, f(eps)], threads)
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
        g.dispatch(sk, &[&scores, probs], &[(bsz * h * tcols) as u32, tcols as u32], st)
    } else {
        g.dispatch(sk, &[&scores, probs], &[bsz as u32, h as u32, tcols as u32], st)
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

/// Recompute the effective query gain (`query_ln.weight * 1.442695 *
/// softplus(per_dim_scale)`) from the CURRENT weights, into an
/// already-allocated `[head_dim]` buffer - see the module doc for why this
/// cannot be folded once at load time like inference does, and the `q_gains`
/// field doc for why it has to be redone per forward rather than per
/// construction. `gain_name`/`scale_name` are the checkpoint's own
/// `query_ln.weight` / `per_dim_scale.per_dim_scale` names for one attention
/// sublayer. A `[head_dim]` host round-trip, negligible next to the matmuls
/// it feeds.
fn refresh_query_gain(g: &Gpu, ps: &ParamStore, gain_name: &str, scale_name: &str, out: &DeviceBuffer) {
    let gain = g.read(ps.w(gain_name), ps.numel(gain_name));
    let scale = g.read(ps.w(scale_name), ps.numel(scale_name));
    let folded: Vec<f32> = gain.iter().zip(&scale).map(|(&gi, &si)| gi * 1.442_695_1 * softplus(si)).collect();
    g.write_f32(out, &folded);
}

/// The one folded-gain buffer belonging to the sublayer whose query norm is
/// `gain_name`. Panics rather than folding a fresh one: a miss means the
/// forward and the reverse disagree about which sublayers exist, which is a
/// construction bug, not a cache miss.
fn query_gain_of<'a>(q_gains: &'a [(String, String, DeviceBuffer)], gain_name: &str) -> &'a DeviceBuffer {
    &q_gains.iter().find(|(n, _, _)| n == gain_name).unwrap_or_else(|| panic!("no folded query gain for {gain_name}")).2
}

// ==================================== fixtures ====================================
//
// Shared by this module's own forward-parity test and by
// `gradcheck::timesfm3`'s harness, for the reason `t5encoder::train` exposes
// the same pair: a gradient check whose fixture is written out a second time
// in the checker is a check of a DIFFERENT model than the one the crate's own
// tests lock down, and the two drift silently.

/// A dependency-free LCG returning `[-1, 1)`. Enough determinism for a test
/// fixture, and no new crate dependency for one helper.
fn lcg_next(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (((*state >> 33) as u32 % 20000) as f32 / 10000.0) - 1.0
}

/// Small deterministic weights for every tensor in `cfg.param_list()`, under
/// the checkpoint's own names, UNFOLDED (see the module doc). `0.15` keeps
/// the stack in its locally-linear region, which is what a central-difference
/// check needs; `per_dim_scale` lands near zero, where `softplus` is ~0.69
/// and its sigmoid derivative ~0.5, so neither half of the query-gain fold is
/// numerically degenerate.
pub fn init_weights(cfg: &Timesfm3Config, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut state = seed ^ 0x7143_f43d_0000_0001u64;
    cfg.param_list()
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|_| lcg_next(&mut state) * 0.15).collect();
            (name, data)
        })
        .collect()
}

/// A fixed `[b*v*n, resblock_in_dim]` resblock input - `core_forward`'s own
/// first argument, i.e. what `preprocess.rs` hands the trainable graph.
pub fn fixed_input(cfg: &Timesfm3Config, b: usize, v: usize, n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed ^ 0x1234_5678_9abc_def0u64;
    (0..b * v * n * cfg.resblock_in_dim()).map(|_| lcg_next(&mut state) * 0.5).collect()
}

// ==================================== backward ====================================
//
// Everything below builds the reverse pass, walked in the SAME order this
// file's own module doc lists the forward: output head, then each layer in
// REVERSE (feedforward, then variate attention, then sequence attention),
// then `pre_transformer_resblock`. Every residual junction in the forward
// (`h_in`/`h1`/`h2` each read by BOTH a residual add and a pre-LN) becomes a
// two-term `add2` here - never an in-place accumulate, so neither term needs
// defensive copying first (see individual comments for why a value that is
// only ever READ, never mutated, is safely shared by several backward calls
// with no copy at all).

/// `dX[m,k] = Σ_n dY[m,n]*W[n,k]`, the adjoint of [`linear`]'s `x` argument.
/// `accumulate`: false assigns, true adds - for an `x` read by more than one
/// forward linear (e.g. `seq_in` feeds query/key/value_proj all three).
#[allow(clippy::too_many_arguments)]
fn linear_dx(g: &Gpu, dy: &DeviceBuffer, weight: &DeviceBuffer, dx: &DeviceBuffer, m: usize, k: usize, n: usize, accumulate: bool) -> Step {
    g.step(K_MATMUL_DX, &[dy, weight, dx], &[m as u32, k as u32, n as u32, accumulate as u32], (m * k) as u32)
}

/// `dW[n,k] += Σ_m dY[m,n]*X[m,k]`, the adjoint of [`linear`]'s weight -
/// ALWAYS accumulates (the grad buffer is zeroed once per step by
/// [`Timesfm3Train::zero_grads`]).
fn linear_dw(g: &Gpu, dy: &DeviceBuffer, x: &DeviceBuffer, dw: &DeviceBuffer, m: usize, k: usize, n: usize) -> Step {
    g.step(K_MATMUL_DW, &[dy, x, dw], &[m as u32, k as u32, n as u32], (n * k) as u32)
}

/// `name`'s gradient buffer, or `None` when it has none. A `Role::Frozen`
/// parameter allocates no gradient at all, so every `_dw`/`bias_grad`
/// dispatch below has to ask first rather than assume: under LoRA that is
/// EVERY checkpoint tensor.
fn grad_of<'a>(ps: &'a ParamStore, name: &str) -> Option<&'a DeviceBuffer> {
    ps.grad.get(name)
}

/// The adjoint of [`linear`] for a weight that may be frozen and may carry an
/// adapter. Three cases, and `dX` flows in all of them because a frozen
/// weight still has to pass gradient to whatever produced its input:
///
/// * **trainable, no adapter** - `dW` and `dX`, exactly as before LoRA;
/// * **frozen, no adapter** - `dX` only;
/// * **adapted** - `dX` through the frozen base, plus `dA`/`dB`. The scale
///   `alpha/r` belongs on the adapter's intermediate rather than on either
///   gradient: `matmul_dw`/`matmul_dx` take no scale parameter, and putting
///   it on `mid` (for `dB`) and on `dmid` (for `dA` and for the adapter's own
///   contribution to `dX`) reaches all three with one `grad_scale` each.
#[allow(clippy::too_many_arguments)]
fn proj_bwd(g: &Gpu, ps: &ParamStore, lw: &LoraWiring, weight: &str, dy: &DeviceBuffer, x: &DeviceBuffer, dx: &DeviceBuffer, m: usize, k: usize, nout: usize, accumulate: bool) -> Vec<Step> {
    let mut steps = Vec::new();
    match lw.of(weight) {
        Some((r, scale)) => {
            let (a, bnm) = adapter_names(weight);
            let (mr, rn) = ((m * r as usize) as u32, f(scale));
            // Frozen base: dX only.
            steps.push(linear_dx(g, dy, ps.w(weight), dx, m, k, nout, accumulate));
            // dB = dYᵀ·(scale·x·Aᵀ)
            steps.push(g.step(K_MATMUL, &[x, ps.w(&a), &lw.mid], &[m as u32, k as u32, r], mr));
            steps.push(g.step(K_GRAD_SCALE, &[&lw.mid], &[mr, rn], mr));
            steps.push(linear_dw(g, dy, &lw.mid, ps.g(&bnm), m, r as usize, nout));
            // dmid = scale·dY·B, then dA = dmidᵀ·x and dX += dmid·A
            steps.push(linear_dx(g, dy, ps.w(&bnm), &lw.dmid, m, r as usize, nout, false));
            steps.push(g.step(K_GRAD_SCALE, &[&lw.dmid], &[mr, rn], mr));
            steps.push(linear_dw(g, &lw.dmid, x, ps.g(&a), m, k, r as usize));
            steps.push(linear_dx(g, &lw.dmid, ps.w(&a), dx, m, k, r as usize, true));
        }
        None => {
            if let Some(dw) = grad_of(ps, weight) {
                steps.push(linear_dw(g, dy, x, dw, m, k, nout));
            }
            steps.push(linear_dx(g, dy, ps.w(weight), dx, m, k, nout, accumulate));
        }
    }
    steps
}

/// `out[i] = a[i] + b[i]`, out-of-place - the two-path convergence every
/// residual junction needs (see the section doc).
fn add2(g: &Gpu, a: &DeviceBuffer, b: &DeviceBuffer, out: &DeviceBuffer, total: usize) -> Step {
    g.step(K_ADD2, &[a, b, out], &[total as u32], total as u32)
}

/// The adjoint of [`swap12`]`(src, dst, a0, a1, a2, d)` (which maps
/// `[a0,a1,a2,d] -> [a0,a2,a1,d]`): given `d_dst` (shaped like `dst`),
/// produce `d_src` (shaped like `src`) - the SAME kernel, called with `a1`
/// and `a2` swapped from the ORIGINAL forward call's own arguments (passed
/// here unswapped; the swap happens inside).
fn swap12_adjoint(g: &Gpu, d_dst: &DeviceBuffer, d_src: &DeviceBuffer, a0: usize, a1: usize, a2: usize, d: usize) -> Step {
    swap12(g, d_dst, d_src, a0, a2, a1, d)
}

/// The `rms_inv_eps -> rmsnorm_dw -> rmsnorm_dx_eps` trio, via
/// `model::block`'s shared helper. `gw`, when given, accumulates into it
/// (matching `rmsnorm_dw`'s own `+=`); omit it for a norm whose weight is
/// NOT one of `ps`'s own tensors (the query gain fold - see
/// [`Timesfm3Train::backward`]).
#[allow(clippy::too_many_arguments)]
fn rmsnorm_bwd(g: &Gpu, x: &DeviceBuffer, w: &DeviceBuffer, dy: &DeviceBuffer, dx: &DeviceBuffer, gw: Option<&DeviceBuffer>, dim: usize, rows: usize, eps: f32) -> Vec<Step> {
    let inv = g.storage(rows as u64);
    block::rmsnorm_eps_bwd(g, K_RMS_INV_EPS, K_RMSNORM_DW, K_RMSNORM_DX_EPS, x, w, dy, dx, &inv, gw, dim as u32, rows as u32, eps)
}

/// `rope_partial_bwd`: the transpose rotation, in place on `buf` - same
/// Params as [`rope_seq`]'s forward call.
fn rope_seq_bwd(g: &Gpu, cfg: &Timesfm3Config, buf: &DeviceBuffer, rows: usize, tcols: usize) -> Step {
    let hd = cfg.head_dim as u32;
    g.step(
        K_ROPE_PARTIAL_BWD,
        &[buf],
        &[rows as u32, cfg.num_heads as u32, hd, hd * cfg.num_heads as u32, 0, tcols as u32, f(ROPE_THETA), hd],
        rows as u32 * cfg.num_heads as u32 * (hd / 2),
    )
}

/// One attention sublayer's backward: given the forward's own `q`/`k`/`v`
/// (the values actually fed to `attn_scores_qk_kmask` - POST qk-norm for
/// q/k) and `probs`, plus `d_ctx` (gradient into the attention OUTPUT),
/// returns the steps and fresh `(d_q, d_k, d_v)` buffers. `causal` selects
/// the causal kernel family (`attn_bwd_dscores`/`dv`/`dq_bias`/`dk_bias`,
/// scale=1.0 always - the fold's whole point) or the bidir family
/// (`_bidir`, no scale/causal params - hardcoded non-causal, scale=1.0).
/// `qkv_stride`/`q_off`/`k_off`/`v_off` follow [`kernels::PACK_QKV`]'s own
/// fixed layout, which every `attn_bwd_*` kernel reads from.
#[allow(clippy::too_many_arguments)]
fn attention_bwd(g: &Gpu, cfg: &Timesfm3Config, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, probs: &DeviceBuffer, d_ctx: &DeviceBuffer, bsz: usize, tcols: usize, causal: bool) -> (Vec<Step>, DeviceBuffer, DeviceBuffer, DeviceBuffer) {
    let (h, hd, d) = (cfg.num_heads, cfg.head_dim, cfg.model_dims);
    let rows = bsz * tcols;
    let stride3 = 3 * d;
    let (q_off, k_off, v_off) = (0u32, d as u32, 2 * d as u32);

    let qkv = g.storage((rows * stride3) as u64);
    let mut steps = vec![g.step(K_PACK_QKV, &[q, k, v, &qkv], &[rows as u32, d as u32], (rows * stride3) as u32)];

    let d_qkv = g.storage((rows * stride3) as u64);
    let d_scores = g.storage((bsz * h * tcols * tcols) as u64);
    let pv = [bsz as u32, h as u32, tcols as u32, hd as u32, stride3 as u32, v_off, d as u32];
    // dq_bias/dk_bias, NOT dq_bidir/dk_bidir, for BOTH causal and non-causal:
    // the _bidir pair hardcodes the conventional `1/sqrt(head_dim)` attention
    // scale internally (no scale param at all), but TimesFM-3 always uses
    // scale=1.0 (folded into the query gain - see the module doc). Only
    // _bias exposes `scale` as a parameter, so it is the only correct choice
    // regardless of causality - the same reason t5encoder::train (also
    // scale=1.0, also non-causal) uses `_bias` rather than `_bidir` for
    // exactly these two kernels while still using `_bidir` for dscores/dv,
    // which have no scale-dependence to get wrong.
    let pqk = [bsz as u32, h as u32, tcols as u32, hd as u32, stride3 as u32, q_off, k_off, f(1.0), causal as u32];
    if causal {
        steps.push(g.step(K_DSCORES, &[d_ctx, &qkv, probs, &d_scores], &pv, (bsz * h * tcols) as u32));
        steps.push(g.step(K_DV, &[probs, d_ctx, &d_qkv], &pv, (bsz * h * tcols * hd) as u32));
    } else {
        steps.push(g.step(K_DSCORES_BIDIR, &[d_ctx, &qkv, probs, &d_scores], &pv, (bsz * h * tcols) as u32));
        steps.push(g.step(K_DV_BIDIR, &[probs, d_ctx, &d_qkv], &pv, (bsz * h * tcols * hd) as u32));
    }
    steps.push(g.step(K_DQ_BIAS, &[&d_scores, &qkv, &d_qkv], &pqk, (bsz * h * tcols * hd) as u32));
    steps.push(g.step(K_DK_BIAS, &[&d_scores, &qkv, &d_qkv], &pqk, (bsz * h * tcols * hd) as u32));

    // `region_copy` does NOT fit here: it reuses the SAME strided index for
    // both buffers (a matching-shape in-place-style sub-region copy), so
    // feeding it a dense `[rows,d]` destination against a `row_stride=3*d`
    // source computes destination indices that run far past that buffer's
    // own size for every row but the first - a real bug this function used
    // to have. `unpack_qkv` is the actual, purpose-built inverse of
    // `pack_qkv`: one dispatch splits the fused `[rows,3*d]` buffer into
    // three independently-indexed dense `[rows,d]` buffers.
    let (d_q, d_k, d_v) = (g.storage((rows * d) as u64), g.storage((rows * d) as u64), g.storage((rows * d) as u64));
    steps.push(g.step(K_UNPACK_QKV, &[&d_qkv, &d_q, &d_k, &d_v], &[rows as u32, d as u32], (rows * stride3) as u32));
    (steps, d_q, d_k, d_v)
}

/// One attention sublayer's full backward, from `d_h_next` (this sublayer's
/// output gradient, already the sum of its own residual path and whatever
/// read the sublayer's output downstream) down to `d_h_prev` (this
/// sublayer's INPUT gradient, still needing its own residual-path term
/// added by the caller). `is_seq` selects sequence (causal, RoPE'd,
/// `seq_kmask`-shaped bsz/tcols) vs variate (bidir, no RoPE, needs the
/// swap12 pair) - the two attention sublayers are identical past this
/// branch. Returns the steps and `d_h_prev`, plus any query-gain fold temps
/// this sublayer produced (one, always - both attention kinds have their
/// own PerDimScale).
#[allow(clippy::too_many_arguments)]
fn attn_sublayer_bwd(
    g: &Gpu,
    cfg: &Timesfm3Config,
    ps: &ParamStore,
    lw: &LoraWiring,
    q_gains: &[(String, String, DeviceBuffer)],
    prefix: &str,
    is_seq: bool,
    h_prev_val: &DeviceBuffer,
    sub_in: &DeviceBuffer,
    q_pre_norm: &DeviceBuffer,
    k_pre_norm: &DeviceBuffer,
    v_val: &DeviceBuffer,
    q_normed: &DeviceBuffer,
    k_normed: &DeviceBuffer,
    probs: &DeviceBuffer,
    ctx_val: &DeviceBuffer,
    sub_out: &DeviceBuffer,
    d_h_next: &DeviceBuffer,
    b: usize,
    v: usize,
    n: usize,
    rows: usize,
) -> (Vec<Step>, DeviceBuffer, Vec<(String, String, DeviceBuffer)>) {
    let d = cfg.model_dims;
    let kind = if is_seq { "seq_attn" } else { "var_attn" };
    let (query_ln, key_ln, per_dim_scale) = (format!("{prefix}.{kind}.query_ln.weight"), format!("{prefix}.{kind}.key_ln.weight"), format!("{prefix}.{kind}.per_dim_scale.per_dim_scale"));
    let (query_proj, key_proj, value_proj, out_proj) = (format!("{prefix}.{kind}.query_proj.weight"), format!("{prefix}.{kind}.key_proj.weight"), format!("{prefix}.{kind}.value_proj.weight"), format!("{prefix}.{kind}.out_proj.weight"));
    let post_ln = format!("{prefix}.post_{kind}_ln.weight");
    let pre_ln = format!("{prefix}.pre_{kind}_ln.weight");

    let mut steps = Vec::new();
    let mut folds = Vec::new();
    let (bsz, tcols) = if is_seq { (b * v, n) } else { (b * n, v) };

    // sub_out = post_ln(attn_out), where attn_out = linear(ctx_val, out_proj.weight)
    // (`attn_out` = `seq_out`/`var_out` in the forward's own naming; `ctx_val`
    // = `ctx`/`ctx2`, a DIFFERENT buffer earlier in the same chain - conflating
    // the two here was a real bug this function used to have).
    let d_attn_out = g.storage((rows * d) as u64);
    steps.extend(rmsnorm_bwd(g, sub_out, ps.w(&post_ln), d_h_next, &d_attn_out, grad_of(ps, &post_ln), d, rows, cfg.rms_norm_eps));

    let d_ctx = g.storage((rows * d) as u64);
    steps.extend(proj_bwd(g, ps, lw, &out_proj, &d_attn_out, ctx_val, &d_ctx, rows, d, d, false));

    // Variate attention swaps to (b,n,v) order around its own attention
    // call; sequence attention does not.
    let (d_ctx_attn, attn_q_in, attn_k_in, attn_v_in): (DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer);
    if is_seq {
        d_ctx_attn = d_ctx;
        attn_q_in = q_normed.clone();
        attn_k_in = k_normed.clone();
        attn_v_in = v_val.clone();
    } else {
        // ctx2 = swap12(ctx2t, b, n, v, d) - adjoint produces d_ctx2t.
        let d_ctx2t = g.storage((rows * d) as u64);
        steps.push(swap12_adjoint(g, &d_ctx, &d_ctx2t, b, n, v, d));
        // q2t/k2t/v2t = swap12(q2n/k2n/v2, b, v, n, d) - the attention's own
        // inputs are the POST-swap buffers.
        let (q2t, k2t, v2t) = (g.storage((rows * d) as u64), g.storage((rows * d) as u64), g.storage((rows * d) as u64));
        steps.push(swap12(g, q_normed, &q2t, b, v, n, d));
        steps.push(swap12(g, k_normed, &k2t, b, v, n, d));
        steps.push(swap12(g, v_val, &v2t, b, v, n, d));
        d_ctx_attn = d_ctx2t;
        attn_q_in = q2t;
        attn_k_in = k2t;
        attn_v_in = v2t;
    }

    let (attn_steps, d_qn, d_kn, mut d_v) = attention_bwd(g, cfg, &attn_q_in, &attn_k_in, &attn_v_in, probs, &d_ctx_attn, bsz, tcols, is_seq);
    steps.extend(attn_steps);

    // Variate attention's q/k/v were transposed before the attention call -
    // d_qn/d_kn/d_v are in (b,n,v) order and must swap back to (b,v,n) to
    // line up with q2n/k2n/v2's own layout before continuing.
    let (d_qn, d_kn, d_v) = if is_seq {
        (d_qn, d_kn, d_v)
    } else {
        let (d_q2n, d_k2n, d_v2) = (g.storage((rows * d) as u64), g.storage((rows * d) as u64), g.storage((rows * d) as u64));
        steps.push(swap12_adjoint(g, &d_qn, &d_q2n, b, v, n, d));
        steps.push(swap12_adjoint(g, &d_kn, &d_k2n, b, v, n, d));
        steps.push(swap12_adjoint(g, &d_v, &d_v2, b, v, n, d));
        d_v = d_v2;
        (d_q2n, d_k2n, d_v)
    };

    // q_normed = rmsnorm(q_pre_norm, fold(query_ln, per_dim_scale)) - the
    // fold-split happens host-side later; here we only need d_q_pre_norm.
    // The gain is the forward's OWN buffer, not a second fold of the same two
    // weights, so `forward()`'s refresh reaches this dispatch too.
    let q_gain = query_gain_of(q_gains, &query_ln);
    let d_q_dx = g.storage((rows * d) as u64);
    // Both halves of the fold are trainable together or frozen together (they
    // are never LoRA targets, so under LoRA they are simply frozen). When
    // frozen, the `rmsnorm_dw` half of the trio is skipped outright and no
    // fold temp is registered, which is what keeps `backward`'s host-side
    // split from reaching for a gradient buffer that was never allocated.
    let gain_trainable = grad_of(ps, &query_ln).is_some();
    let d_gain_eff = g.storage(cfg.head_dim as u64);
    let dw = gain_trainable.then_some(&d_gain_eff);
    steps.extend(rmsnorm_bwd(g, q_pre_norm, q_gain, &d_qn, &d_q_dx, dw, cfg.head_dim, rows * cfg.num_heads, cfg.rms_norm_eps));
    if gain_trainable {
        folds.push((query_ln.clone(), per_dim_scale.clone(), d_gain_eff));
    }

    // k_normed = rmsnorm(k_pre_norm, key_ln.weight) - a normal trainable weight.
    let d_k_dx = g.storage((rows * d) as u64);
    steps.extend(rmsnorm_bwd(g, k_pre_norm, ps.w(&key_ln), &d_kn, &d_k_dx, grad_of(ps, &key_ln), cfg.head_dim, rows * cfg.num_heads, cfg.rms_norm_eps));

    // Sequence attention RoPE'd q/k AFTER the projection, BEFORE qk-norm -
    // the backward undoes qk-norm first, then RoPE, in place.
    if is_seq {
        steps.push(rope_seq_bwd(g, cfg, &d_q_dx, rows, n));
        steps.push(rope_seq_bwd(g, cfg, &d_k_dx, rows, n));
    }

    // q/k/v = linear(sub_in, {query,key,value}_proj.weight) - sub_in is read
    // by all three, so their dx accumulates (matmul_dx's own accumulate flag).
    // `sub_in` feeds all three projections, so their `dX` accumulate onto one
    // buffer - the first assigns, the other two add.
    let d_sub_in = g.storage((rows * d) as u64);
    steps.extend(proj_bwd(g, ps, lw, &query_proj, &d_q_dx, sub_in, &d_sub_in, rows, d, d, false));
    steps.extend(proj_bwd(g, ps, lw, &key_proj, &d_k_dx, sub_in, &d_sub_in, rows, d, d, true));
    steps.extend(proj_bwd(g, ps, lw, &value_proj, &d_v, sub_in, &d_sub_in, rows, d, d, true));

    // sub_in = rmsnorm(h_prev_val, pre_ln.weight) - the sublayer's own input norm.
    let d_h_prev_from_norm = g.storage((rows * d) as u64);
    steps.extend(rmsnorm_bwd(g, h_prev_val, ps.w(&pre_ln), &d_sub_in, &d_h_prev_from_norm, grad_of(ps, &pre_ln), d, rows, cfg.rms_norm_eps));

    // h_prev is read by BOTH this sublayer's own residual add (contributing
    // d_h_next unchanged) and pre_ln above - the two-path convergence.
    let d_h_prev = g.storage((rows * d) as u64);
    steps.push(add2(g, d_h_next, &d_h_prev_from_norm, &d_h_prev, rows * d));

    (steps, d_h_prev, folds)
}

/// The full reverse pass: output head, every layer in reverse, then
/// `pre_transformer_resblock`. `h_final` is the LAST layer's own output
/// (`core_forward`'s local `h` just before the head's linear); `d_logits`
/// is the (not-yet-uploaded) seed buffer [`Timesfm3Train::backward`] writes
/// into on every call.
#[allow(clippy::too_many_arguments)]
fn build_backward(
    g: &Gpu,
    cfg: &Timesfm3Config,
    ps: &ParamStore,
    lw: &LoraWiring,
    q_gains: &[(String, String, DeviceBuffer)],
    layers: &[TrainLayer],
    resblock_input: &DeviceBuffer,
    resblock_hidden_pre: &DeviceBuffer,
    resblock_hidden_act: &DeviceBuffer,
    h_final: &DeviceBuffer,
    d_logits: &DeviceBuffer,
    b: usize,
    v: usize,
    n: usize,
) -> (Vec<Step>, Vec<(String, String, DeviceBuffer)>) {
    let rows = b * v * n;
    let d = cfg.model_dims;
    let head_out = cfg.head_out_dim();
    let mut steps = Vec::new();
    let mut folds = Vec::new();

    // ---- output head: logits = linear(h_final, output_head.weight) + bias ----
    if let Some(db) = grad_of(ps, "output_head.bias") {
        steps.push(g.step(K_BIAS_GRAD, &[d_logits, db], &[rows as u32, head_out as u32], head_out as u32));
    }
    let mut d_h = g.storage((rows * d) as u64);
    steps.extend(proj_bwd(g, ps, lw, "output_head.weight", d_logits, h_final, &d_h, rows, d, head_out, false));

    for l in (0..cfg.num_layers).rev() {
        let ly = &layers[l];
        let prefix = format!("transformer_stack.layers.{l}");

        // ---- feedforward, in reverse ----
        let d_ff_out = g.storage((rows * d) as u64);
        steps.extend(rmsnorm_bwd(g, &ly.ff_out, ps.w(&format!("{prefix}.post_ff_ln.weight")), &d_h, &d_ff_out, grad_of(ps, &format!("{prefix}.post_ff_ln.weight")), d, rows, cfg.rms_norm_eps));
        let d_ff_h_act = g.storage((rows * cfg.hidden_dims) as u64);
        steps.extend(proj_bwd(g, ps, lw, &format!("{prefix}.ff1.weight"), &d_ff_out, &ly.ff_h_act, &d_ff_h_act, rows, cfg.hidden_dims, d, false));
        let d_ff_h_pre = g.storage((rows * cfg.hidden_dims) as u64);
        steps.push(g.step(K_LEAKY_RELU_BWD, &[&ly.ff_h_pre, &d_ff_h_act, &d_ff_h_pre], &[(rows * cfg.hidden_dims) as u32, f(0.0)], (rows * cfg.hidden_dims) as u32));
        let d_ff_in = g.storage((rows * d) as u64);
        steps.extend(proj_bwd(g, ps, lw, &format!("{prefix}.ff0.weight"), &d_ff_h_pre, &ly.ff_in, &d_ff_in, rows, d, cfg.hidden_dims, false));
        let d_ff_in_dx = g.storage((rows * d) as u64);
        steps.extend(rmsnorm_bwd(g, &ly.h2, ps.w(&format!("{prefix}.pre_ff_ln.weight")), &d_ff_in, &d_ff_in_dx, grad_of(ps, &format!("{prefix}.pre_ff_ln.weight")), d, rows, cfg.rms_norm_eps));
        let d_h2 = g.storage((rows * d) as u64);
        steps.push(add2(g, &d_h, &d_ff_in_dx, &d_h2, rows * d));

        // ---- variate attention, in reverse ----
        let (var_steps, d_h1, var_folds) = attn_sublayer_bwd(g, cfg, ps, lw, q_gains, &prefix, false, &ly.h1, &ly.var_in, &ly.q2, &ly.k2, &ly.v2, &ly.q2n, &ly.k2n, &ly.probs_var, &ly.ctx2, &ly.var_out, &d_h2, b, v, n, rows);
        steps.extend(var_steps);
        folds.extend(var_folds);

        // ---- sequence attention, in reverse ----
        let (seq_steps, d_h_in, seq_folds) = attn_sublayer_bwd(g, cfg, ps, lw, q_gains, &prefix, true, &ly.h_in, &ly.seq_in, &ly.q, &ly.k, &ly.vv, &ly.qn, &ly.kn, &ly.probs_seq, &ly.ctx, &ly.seq_out, &d_h1, b, v, n, rows);
        steps.extend(seq_steps);
        folds.extend(seq_folds);

        d_h = d_h_in;
    }

    // ---- pre_transformer_resblock, in reverse: resblock_out = output(relu(hidden(x))) + residual(x) ----
    // d_h is now the gradient into resblock_out; both branches read it
    // directly (neither is mutated by what follows), no copy needed.
    let resblock_in_dim = cfg.resblock_in_dim();
    if let Some(dw) = grad_of(ps, "pre_transformer_resblock.residual_layer.weight") {
        steps.push(linear_dw(g, &d_h, resblock_input, dw, rows, resblock_in_dim, d));
    }
    let d_hidden_act = g.storage((rows * d) as u64);
    steps.extend(proj_bwd(g, ps, lw, "pre_transformer_resblock.output_layer.weight", &d_h, resblock_hidden_act, &d_hidden_act, rows, d, d, false));
    let d_hidden_pre = g.storage((rows * d) as u64);
    steps.push(g.step(K_LEAKY_RELU_BWD, &[resblock_hidden_pre, &d_hidden_act, &d_hidden_pre], &[(rows * d) as u32, f(0.0)], (rows * d) as u32));
    if let Some(dw) = grad_of(ps, "pre_transformer_resblock.hidden_layer.weight") {
        steps.push(linear_dw(g, &d_hidden_pre, resblock_input, dw, rows, resblock_in_dim, d));
    }
    // No d_resblock_input needed: it is the model's own input, not a trainable weight.

    (steps, folds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Timesfm3;

    fn skip() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    #[test]
    fn trainer_forward_matches_inference_core_forward() {
        if skip() {
            return;
        }
        let cfg = Timesfm3Config::tiny();
        let weights = init_weights(&cfg, 7);
        let (b, v, n) = (2usize, 3usize, 4usize);
        let rows = b * v * n;
        let resblock_input = fixed_input(&cfg, b, v, n, 7);
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

    /// Weights move BETWEEN forwards - that is what an optimiser step is, and
    /// what a finite-difference harness's perturbation is. The effective query
    /// gain is a FOLD of two live parameters rather than a tensor of its own,
    /// so it has to be recomputed from them each time; a gain captured once at
    /// construction silently pins `query_ln.weight` and `per_dim_scale` to
    /// their initial values for the rest of the run, while their gradients keep
    /// being computed and their optimiser state keeps moving.
    ///
    /// Asserted BITWISE against a trainer freshly constructed from the same
    /// updated weights, which is the only definition of "up to date" that does
    /// not restate the fold's formula a second time.
    /// Moving to the next batch must be indistinguishable from having built
    /// the trainer for that batch in the first place - BITWISE, because
    /// anything less means a training run's later batches are being evaluated
    /// on a graph subtly different from the one its first batch was.
    ///
    /// The mask is the part that can silently go stale: it is not uploaded
    /// as-is but expanded host-side into two additive key masks in two
    /// different axis orders, so a `set_input` that refreshed the input and
    /// forgot the masks would keep training every later batch against the
    /// FIRST batch's padding.
    #[test]
    fn set_input_moves_the_graph_to_a_new_batch() {
        if skip() {
            return;
        }
        let cfg = Timesfm3Config::tiny();
        let weights = init_weights(&cfg, 7);
        let (b, v, n) = (1usize, 2usize, 3usize);
        let rows = b * v * n;
        let first = fixed_input(&cfg, b, v, n, 7);
        let first_mask = vec![false; rows];

        let tr = Timesfm3Train::new_on(gpu_core::testgpu::dev(TRAIN_PIPELINES), cfg.clone(), &first, &first_mask, b, v, n, &weights);

        // A different batch AND a different mask - masked leading patches on
        // one (b, v) row only, so the two key-mask layouts disagree about
        // which entries move and a single-layout refresh cannot pass.
        let second = fixed_input(&cfg, b, v, n, 99);
        let mut second_mask = vec![false; rows];
        second_mask[0] = true;
        second_mask[n] = true;
        tr.set_input(&second, &second_mask);
        tr.forward();
        tr.poll_wait();
        let got = tr.read_logits();

        let fresh = Timesfm3Train::new_on(gpu_core::testgpu::dev(TRAIN_PIPELINES), cfg.clone(), &second, &second_mask, b, v, n, &weights);
        fresh.forward();
        fresh.poll_wait();
        let want = fresh.read_logits();
        assert_ne!(got, tr_logits_of(&cfg, &weights, &first, &first_mask, b, v, n), "the fixture is degenerate: the two batches produce the same logits");
        assert_eq!(got, want, "a forward after set_input must match a trainer built for that batch");
    }

    fn tr_logits_of(cfg: &Timesfm3Config, weights: &HashMap<String, Vec<f32>>, x: &[f32], mask: &[bool], b: usize, v: usize, n: usize) -> Vec<f32> {
        let tr = Timesfm3Train::new_on(gpu_core::testgpu::dev(TRAIN_PIPELINES), cfg.clone(), x, mask, b, v, n, weights);
        tr.forward();
        tr.poll_wait();
        tr.read_logits()
    }

    /// The on-device AdamW step has to actually move the loss downhill on
    /// this graph. Cheap to get wrong silently: `Optim` is driven by five
    /// kernel INDICES into `TRAIN_PIPELINES`, and a mis-ordered index list
    /// dispatches a real kernel with the wrong Params rather than failing.
    #[test]
    fn the_adamw_step_lowers_the_loss() {
        if skip() {
            return;
        }
        let cfg = Timesfm3Config::tiny();
        let weights = init_weights(&cfg, 7);
        let (b, v, n) = (1usize, 2usize, 3usize);
        let rows = b * v * n;
        let x = fixed_input(&cfg, b, v, n, 7);
        let mask = vec![false; rows];
        let tr = Timesfm3Train::new_on(gpu_core::testgpu::dev(TRAIN_PIPELINES), cfg.clone(), &x, &mask, b, v, n, &weights);

        let mut state = 0x0bad_c0de_0bad_c0deu64;
        let c: Vec<f32> = (0..rows * cfg.head_out_dim()).map(|_| lcg_next(&mut state) * 0.1).collect();
        let loss = || {
            tr.forward();
            tr.poll_wait();
            tr.read_logits().iter().zip(&c).map(|(&y, &ci)| y as f64 * ci as f64).sum::<f64>() as f32
        };

        let l0 = loss();
        for t in 1..=8u32 {
            tr.forward();
            tr.poll_wait();
            tr.zero_grads();
            tr.backward(&c);
            tr.adamw_step(t, 1e-2, 0.0, Some(1.0));
            tr.poll_wait();
        }
        let l1 = loss();
        assert!(l1 < l0, "8 AdamW steps did not lower the loss: {l0} -> {l1}");
    }

    /// `zero_grads()` then `backward()` is the unit of a training step, and
    /// running it twice over unchanged weights has to produce the same
    /// gradients twice.
    ///
    /// The reverse pass writes one device temp per query-attention sublayer -
    /// the `rmsnorm_dw` output for the EFFECTIVE query gain, which is not a
    /// `ParamStore` tensor (it is split host-side into `query_ln.weight` and
    /// `per_dim_scale` afterwards) and which `rmsnorm_dw` ACCUMULATES into.
    /// `ParamStore::zero_grads` cannot reach a buffer it does not own, so a
    /// temp left dirty makes every step after the first see a fold gradient
    /// that is the running SUM over all steps so far.
    ///
    /// Both halves of the fold, on both attention kinds, at every layer, plus
    /// two ordinary tensors as controls: the controls accumulate into
    /// `ParamStore` buffers that `zero_grads` does clear, so if THEY regressed
    /// the mechanism at fault would be a different one.
    #[test]
    fn repeated_backward_passes_are_not_cumulative() {
        if skip() {
            return;
        }
        let cfg = Timesfm3Config::tiny();
        let weights = init_weights(&cfg, 7);
        let (b, v, n) = (1usize, 2usize, 3usize);
        let rows = b * v * n;
        let resblock_input = fixed_input(&cfg, b, v, n, 7);
        let mask = vec![false; rows];

        let tr = Timesfm3Train::new_on(gpu_core::testgpu::dev(TRAIN_PIPELINES), cfg.clone(), &resblock_input, &mask, b, v, n, &weights);
        let mut state = 0xfeed_face_dead_beefu64;
        let c: Vec<f32> = (0..rows * cfg.head_out_dim()).map(|_| lcg_next(&mut state) * 0.1).collect();

        let step = || {
            tr.forward();
            tr.poll_wait();
            tr.zero_grads();
            tr.backward(&c);
            let mut g: Vec<(String, Vec<f32>)> = Vec::new();
            for l in 0..cfg.num_layers {
                for kind in ["seq_attn", "var_attn"] {
                    for leaf in ["query_ln.weight", "per_dim_scale.per_dim_scale"] {
                        let name = format!("transformer_stack.layers.{l}.{kind}.{leaf}");
                        let v = tr.read_grad(&name);
                        g.push((name, v));
                    }
                }
            }
            for name in ["transformer_stack.layers.0.seq_attn.key_ln.weight", "output_head.weight"] {
                g.push((name.to_string(), tr.read_grad(name)));
            }
            g
        };

        let first = step();
        let second = step();
        for ((name, a), (_, b)) in first.iter().zip(&second) {
            assert!(a.iter().any(|x| x.abs() > 1e-9), "{name}: gradient is identically zero, the comparison below would be vacuous");
            assert_eq!(a, b, "{name}: a second zero_grads+backward must reproduce the first, not accumulate onto it");
        }
    }

    #[test]
    fn forward_tracks_weights_written_after_construction() {
        if skip() {
            return;
        }
        let cfg = Timesfm3Config::tiny();
        let mut weights = init_weights(&cfg, 7);
        let (b, v, n) = (1usize, 2usize, 2usize);
        let rows = b * v * n;
        let resblock_input = fixed_input(&cfg, b, v, n, 7);
        let mask = vec![false; rows];

        let tr = Timesfm3Train::new_on(gpu_core::testgpu::dev(TRAIN_PIPELINES), cfg.clone(), &resblock_input, &mask, b, v, n, &weights);

        // Both halves of the fold, on both attention kinds, plus one ordinary
        // tensor as a control: the ordinary one is read live from the
        // ParamStore already, so if IT regressed the mechanism at fault would
        // be something else entirely.
        let moved = [
            "transformer_stack.layers.0.seq_attn.query_ln.weight",
            "transformer_stack.layers.0.seq_attn.per_dim_scale.per_dim_scale",
            "transformer_stack.layers.1.var_attn.query_ln.weight",
            "transformer_stack.layers.1.var_attn.per_dim_scale.per_dim_scale",
            "transformer_stack.layers.2.ff0.weight",
        ];
        for name in moved {
            let updated: Vec<f32> = weights[name].iter().enumerate().map(|(i, &w)| w + 0.25 + 0.01 * i as f32).collect();
            tr.write_weight(name, &updated);
            weights.insert(name.to_string(), updated);
        }
        tr.forward();
        tr.poll_wait();
        let got = tr.read_logits();

        let fresh = Timesfm3Train::new_on(gpu_core::testgpu::dev(TRAIN_PIPELINES), cfg.clone(), &resblock_input, &mask, b, v, n, &weights);
        fresh.forward();
        fresh.poll_wait();
        let want = fresh.read_logits();

        assert_eq!(got, want, "a forward after write_weight must use the written weights, including the PerDimScale fold");
    }


}


