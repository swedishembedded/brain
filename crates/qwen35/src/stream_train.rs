// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Streaming LoRA fine-tuning over the real 64-layer `Qwen/Qwen3.8-27B-FP8`
//! decoder - the training counterpart of [`crate::stream`]'s inference-only
//! sliding-window forward.
//!
//! # The problem this module solves
//!
//! [`crate::stream`] proves a real forward pass can run with only a small
//! window of layers' weights resident at once, dropping each layer's weights
//! (and its own internal activations) immediately after use. Training needs a
//! BACKWARD pass too: to get a gradient into a LoRA adapter in an early
//! layer, the loss gradient must flow back THROUGH every later (frozen)
//! layer, which needs `dx = d_out @ W_frozen` at each one - the actual weight
//! value, not just its shape. Since nothing stays resident, this means
//! re-streaming all 64 layers' weights a SECOND time, in reverse order, for
//! the backward pass - roughly doubling the per-step weight-I/O cost of a
//! forward-only pass. That is the accepted cost model here, not something
//! this module tries to avoid.
//!
//! What DOES stay resident in host RAM across a whole step, cheaply: not
//! weights, but the residual stream at every layer boundary
//! (`xres_cache[l]`, `[n, d_model]`, one per layer - a few hundred KB to a
//! few MB at a realistic tiny training `n`, well under a GB for all 64
//! layers combined at real `d_model = 5120`). The backward pass RECOMPUTES
//! each layer's own internal forward activations (the mixer's chunked-
//! recurrence internals, the MLP's gate/up/h) fresh from that cached residual
//! plus the freshly re-streamed weight, rather than caching those larger
//! per-layer internals across the whole run. This is a standard activation-
//! checkpointing trade: backward already re-streams every layer's weights
//! regardless (the dominant, unavoidable cost per this module's own doc
//! above), so recomputing a layer's own forward internals from its cached
//! input residual costs no extra weight I/O, only cheap extra compute paid
//! on hardware that is already I/O-bound on this workload - and it needs
//! LESS host memory than caching every targeted linear's own `x`/`a = x@A^T`
//! activation individually would (one `[n, d_model]` buffer per layer instead
//! of several smaller ones per targeted leaf).
//!
//! # Backward through a streamed weight: any tier, packed
//!
//! The frozen base's input gradient `dx = d_out @ W` and its forward `y = x W^T` go through
//! `model::ops::Ops` (`matmul_rows`, `matmul_dx`): the shared quantized-execution contract, which decodes a
//! bounded tile of rows of a packed weight (FP8, NF4, int8, K-quant, bf16, f32) and runs the tensor-core GEMM
//! on it. So a layer is resident in the encoding the checkpoint stores it in
//! ([`crate::fp8_source::PackedBase`] preserves it through the checkpoint crate's tiled loader and
//! `model::ops::DeviceWeightSink`), and no layer is ever expanded to fp32 on the host or the device. The base
//! gets no `dW` here (it is frozen, there is no full-fine-tune path); the adapter gradients come from the
//! LoRA-internal activations as in `crate::model::Qwen35::proj_bwd`.
//!
//! The one weight kept resident for the whole call (never re-streamed per layer: it is read and
//! differentiated once per step, not once per layer) is the shared `lm_head`/`embed_tokens` table, at
//! stored encoding (bf16 for the real release) - see [`StreamTrainer::new_real`].
//!
//! # What is reused, unchanged
//!
//! - [`crate::stream::get`] - the host-map accessor of the tiny-scale gates.
//! - `model::gdn_mixer::{gdn_mixer_fwd, gdn_mixer_bwd}` / `model::gqa_mixer::
//!   {gqa_mixer_fwd, gqa_mixer_bwd}` - the SAME hoisted mixer math
//!   `crate::model::Qwen35`'s own resident trainer and `crate::stream`'s own
//!   inference forward both already drive.
//! - `model::ops::{Ops, Weight}` for the forward and input-gradient dispatch of
//!   every streamed linear (frozen base), float-read at any tier.
//! - `crate::model::Qwen35::proj_bwd`'s exact LoRA-branch math (frozen base:
//!   `dx` only, no `dW`; adapter grads `gA`/`gB` from the LoRA-internal
//!   activations) - transcribed kernel-dispatch-for-kernel-dispatch in
//!   [`proj_bwd_streamed`], not reinvented.
//! - `optim::Optim` / `paramstore::ParamStore` - the SAME AdamW dispatch
//!   graph and resident-tensor store `Qwen35::new_train_on` already builds,
//!   here sized for ONLY the LoRA adapter tensors (`crate::init::
//!   init_lora_only`) - tiny (rank × a handful of leaves × 64 layers),
//!   nowhere near the memory concern the frozen base is.
//!
//! # Weight residency
//!
//! Same [`weightset::WeightSet`] drop/rebuild-per-slot discipline
//! [`crate::stream`] already established (see that module's own doc) - a
//! forward pass over [`weightset::Schedule::cyclic`] ascending, a backward
//! pass over the REVERSE order (`(0..n_layers).rev()`), each its own
//! [`weightset::WeightSet`] instance (a schedule is a single fixed order;
//! re-streaming in the other direction is a different schedule, not the same
//! one run twice).

use std::collections::HashMap;
use std::path::Path;

use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use gpu_core::select::Dtype;
use gpu_core::{DeviceBuffer, Gpu, Step};
use model::block::{rmsnorm_bwd, rmsnorm_fwd, swiglu_bwd, swiglu_fwd, FlashGqaIds, KernelIds};
use model::gdn::{gdn_chunk_size, GdnShape};
use model::head_ce::{HeadCe, HeadCeConfig, HeadKernels};
use model::gdn_mixer::{gdn_mixer_bwd, gdn_mixer_fwd, GdnMixerActs, GdnMixerGrads, GdnMixerIds, GdnMixerShape, GdnMixerWeights};
use model::gqa_mixer::{gqa_mixer_bwd, gqa_mixer_fwd_tc, GqaMixerActs, GqaMixerGrads, GqaMixerIds, GqaMixerShape, GqaMixerWeights};
use model::ops::{Ops, Weight};
use optim::Optim;
use model::adapter::exec::{AdapterRuntime, BoundBase};
use paramstore::ParamStore;

use crate::config::Qwen35Config;
use crate::stream::{gdn_mixer_ids, get, gqa_mixer_ids, idx, kernel_ids, OwnedGdnLayer, OwnedGqaLayer, OwnedStreamedLayer};

/// Kernel indices this module dispatches, resolved once against a `Gpu`
/// built from [`crate::model::pipelines`] - the same full façade kernel set
/// `crate::stream::StreamState` and `Qwen35::new_train_on` both register
/// (the training tier, `matmul_dx`/`matmul_dw`/`adamw`/... is part of that
/// SAME shared list, see `crate::model::pipelines`'s own doc), so nothing
/// extra needs registering for this module to run on a `Gpu` either of those
/// already builds.
struct TrainIds {
    kernels: KernelIds,
    gdn_mixer: GdnMixerIds,
    gqa_mixer: GqaMixerIds,
    /// The causal GQA flash trio, when this device's pipelines carry it: attention without the `[H,T,T]` score
    /// slabs, from [`FlashGqaIds::MIN_TRAIN_T`] tokens up, and (with a tensor-core provider) as bf16 GEMMs.
    flash: Option<FlashGqaIds>,
    add2: usize,
}

fn train_ids(g: &Gpu, ops: &Ops) -> TrainIds {
    let flash = match (g.kernel_index("flash_attn_causal_gqa"), g.kernel_index("flash_attn_causal_gqa_bwd_dq"), g.kernel_index("flash_attn_causal_gqa_bwd_dkv")) {
        (Some(fwd), Some(bwd_dq), Some(bwd_dkv)) => Some(FlashGqaIds { fwd, bwd_dq, bwd_dkv, native: !ops.tensor_gemm().is_reference_pinned() }),
        _ => None,
    };
    TrainIds {
        kernels: kernel_ids(g),
        gdn_mixer: gdn_mixer_ids(g),
        gqa_mixer: gqa_mixer_ids(g),
        flash,
        add2: idx(g, "add2"),
    }
}

/// Logit words one chunk of the chunked head cross-entropy may hold at once (the resident trainer's budget).
const HEAD_BUDGET_WORDS: u64 = 1 << 28;

fn build_head_ce(ops: &Ops, gpu: &Gpu, cfg: &Qwen35Config, head: &Weight, n: u32) -> HeadCe {
    let rows = model::head_ce::rows_for_budget(n as u64, cfg.vocab as u64, HEAD_BUDGET_WORDS) as u32;
    let kernels = HeadKernels::from_gpu(gpu).unwrap_or_else(|e| panic!("stream_train: LM head: {e}"));
    HeadCe::new(ops, cfg.d_model, cfg.vocab, head, HeadCeConfig { rows_per_chunk: rows, tile_budget_words: None }, kernels).unwrap_or_else(|e| panic!("stream_train: LM head: {e}"))
}

/// Where one streamed layer's tensors come from: its norm gains and small vectors as f32, its weight
/// matrices as resident [`Weight`]s of whatever tier the source holds them in.
trait LayerTensors {
    fn small(&self, name: &str) -> DeviceBuffer;
    fn weight(&self, name: &str, n: usize, k: usize) -> Weight;
}

/// A host map of f32 tensors (the tiny-scale gates), uploading every matrix at `tier`.
struct HostLayerTensors<'a> {
    ops: &'a Ops,
    gpu: &'a Gpu,
    l: usize,
    map: &'a HashMap<String, Vec<f32>>,
    tier: Dtype,
}

impl LayerTensors for HostLayerTensors<'_> {
    fn small(&self, name: &str) -> DeviceBuffer {
        self.gpu.storage_init(name, get(self.map, name, self.l))
    }
    fn weight(&self, name: &str, n: usize, k: usize) -> Weight {
        Weight::upload(self.ops, get(self.map, name, self.l), n, k, self.tier)
    }
}

/// A checkpoint directory: matrices preserved in their stored encoding (FP8 stays FP8), no host or device
/// fp32 copy of a layer.
struct PackedLayerTensors<'a> {
    ops: &'a Ops,
    gpu: &'a Gpu,
    base: &'a crate::fp8_source::PackedBase,
}

impl LayerTensors for PackedLayerTensors<'_> {
    fn small(&self, name: &str) -> DeviceBuffer {
        let v = self.base.small_f32(name).unwrap_or_else(|e| panic!("stream_train: {e}"));
        self.gpu.storage_init(name, &v)
    }
    fn weight(&self, name: &str, n: usize, k: usize) -> Weight {
        self.base.weight(self.ops, name, n, k).unwrap_or_else(|e| panic!("stream_train: {e}"))
    }
}

/// Build one decoder layer as [`OwnedStreamedLayer`] from `t`: every matrix a resident [`Weight`] of the
/// source's own tier (training multiplies and differentiates through any tier with `Ops`, see this module's
/// doc), every other tensor a plain fp32 buffer. Field-for-field the layout of
/// `crate::stream::StreamState::build_layer`.
fn build_layer(cfg: &Qwen35Config, l: usize, t: &dyn LayerTensors) -> OwnedStreamedLayer {
    use crate::config::LayerType;
    let ty = cfg.layer_types()[l];
    let p = |s: &str| format!("blocks.{l}.{s}");
    let d = cfg.d_model as usize;
    let ff = cfg.intermediate_size as usize;
    let (mlp_gate, mlp_up, mlp_down) =
        (t.weight(&p("mlp.gate.weight"), ff, d), t.weight(&p("mlp.up.weight"), ff, d), t.weight(&p("mlp.down.weight"), d, ff));

    match ty {
        LayerType::Linear => {
            let conv_dim = cfg.linear_conv_dim() as usize;
            let value_dim = cfg.linear_value_dim() as usize;
            let nvh = cfg.linear_num_value_heads as usize;
            OwnedStreamedLayer::Linear(OwnedGdnLayer {
                ln1: t.small(&p("ln1.weight")),
                ln2: t.small(&p("ln2.weight")),
                conv1d_weight: t.small(&p("linear_attn.conv1d.weight")),
                a_log: t.small(&p("linear_attn.A_log")),
                dt_bias: t.small(&p("linear_attn.dt_bias")),
                norm_weight: t.small(&p("linear_attn.norm.weight")),
                in_proj_qkv: t.weight(&p("linear_attn.in_proj_qkv.weight"), conv_dim, d),
                in_proj_z: t.weight(&p("linear_attn.in_proj_z.weight"), value_dim, d),
                in_proj_b: t.weight(&p("linear_attn.in_proj_b.weight"), nvh, d),
                in_proj_a: t.weight(&p("linear_attn.in_proj_a.weight"), nvh, d),
                out_proj: t.weight(&p("linear_attn.out_proj.weight"), d, value_dim),
                mlp_gate,
                mlp_up,
                mlp_down,
            })
        }
        LayerType::Full => {
            let hqp = cfg.q_proj_dim() as usize;
            let hkv = cfg.kv_dim() as usize;
            let hq = cfg.q_dim() as usize;
            OwnedStreamedLayer::Full(OwnedGqaLayer {
                ln1: t.small(&p("ln1.weight")),
                ln2: t.small(&p("ln2.weight")),
                q_norm: t.small(&p("self_attn.q_norm.weight")),
                k_norm: t.small(&p("self_attn.k_norm.weight")),
                q_proj: t.weight(&p("self_attn.q_proj.weight"), hqp, d),
                k_proj: t.weight(&p("self_attn.k_proj.weight"), hkv, d),
                v_proj: t.weight(&p("self_attn.v_proj.weight"), hkv, d),
                o_proj: t.weight(&p("self_attn.o_proj.weight"), d, hq),
                mlp_gate,
                mlp_up,
                mlp_down,
            })
        }
    }
}

/// [`build_layer`] over a host map, every matrix uploaded at `tier` (`Dtype::F32` reproduces the exact
/// reference; a quantized tier trains through it packed).
pub(crate) fn build_layer_host(ops: &Ops, gpu: &Gpu, cfg: &Qwen35Config, l: usize, w: &HashMap<String, Vec<f32>>, tier: Dtype) -> OwnedStreamedLayer {
    build_layer(cfg, l, &HostLayerTensors { ops, gpu, l, map: w, tier })
}

/// [`build_layer`] over a checkpoint directory ([`crate::fp8_source::PackedBase`]): the training counterpart
/// of `crate::stream::StreamState::load_layer`, with the matrices kept in their stored encoding.
pub(crate) fn load_layer_packed(ops: &Ops, gpu: &Gpu, base: &crate::fp8_source::PackedBase, cfg: &Qwen35Config, l: usize) -> OwnedStreamedLayer {
    build_layer(cfg, l, &PackedLayerTensors { ops, gpu, base })
}

/// The small, fully-resident LoRA adapter store for a whole training run:
/// `.lora_a`/`.lora_b` weights, gradients, and Adam moments for every
/// targeted leaf in every layer (`crate::init::init_lora_only`'s own tensor
/// set) - reusing `paramstore::ParamStore` + `optim::Optim` exactly as
/// `Qwen35::new_train_on` does for its own (much larger) trainable set,
/// never reinvented here.
pub struct LoraStore {
    pub ps: ParamStore,
    opt: Optim,
    /// The adapter's forward and backward, shared with the resident trainer.
    rt: AdapterRuntime,
}

impl LoraStore {
    /// `cfg.lora` must be set (the rank/alpha/target-leaf configuration -
    /// `crate::config::lora_cfg`). `init` supplies every `.lora_a`/
    /// `.lora_b` tensor `cfg.param_list()` expects (every other entry is
    /// ignored) - a caller passes the SAME map it seeds the frozen base
    /// from (`crate::init::init_weights`, for the tiny-scale equivalence
    /// gate) or a purpose-built adapter-only map (`crate::init::
    /// init_lora_only`, for a real-checkpoint run whose base comes from the
    /// checkpoint itself, never from a fresh-weight init). Taking the
    /// values as data rather than generating them from a seed here is
    /// deliberate: `init_weights` and `init_lora_only` draw from
    /// DIFFERENT RNG streams even at the same seed (`init_lora_only`'s own
    /// doc: "the two were never meant to agree value-for-value"), so a
    /// caller that wants two trainers to start from byte-identical
    /// adapters (the equivalence gate's whole point) must control that by
    /// passing the SAME map to both, not by passing the same seed to two
    /// independent generators.
    pub fn new(gpu: &Gpu, cfg: &Qwen35Config, init: &HashMap<String, Vec<f32>>, rows: u32) -> LoraStore {
        assert!(cfg.lora.is_some(), "stream_train::LoraStore::new: cfg.lora must be set");
        let lora_init: HashMap<String, Vec<f32>> = init.iter().filter(|(k, _)| model::adapter::device::is_adapter_param(k)).map(|(k, v)| (k.clone(), v.clone())).collect();
        let expect_n = cfg.param_list().iter().filter(|(k, _)| model::adapter::device::is_adapter_param(k)).count();
        assert_eq!(lora_init.len(), expect_n, "stream_train::LoraStore::new: init map is missing some of cfg.param_list()'s adapter tensors");
        let hp = cfg.lora.as_ref().expect("asserted above").hp();
        let roles: Vec<(String, usize, paramstore::Role)> =
            lora_init.iter().map(|(k, v)| (k.clone(), v.len(), model::adapter::device::role_of(k, &hp))).collect();
        let mut ps = ParamStore::new_with_roles(gpu, roles, &lora_init);
        model::adapter::device::apply_groups(&mut ps, &hp);
        let opt = Optim::new(idx(gpu, "adamw"), idx(gpu, "gradnorm_sq"), idx(gpu, "grad_scale"), idx(gpu, "clip_coef"), idx(gpu, "grad_scale_buf"));
        let rt = AdapterRuntime::new(gpu, cfg.lora.as_ref().expect("asserted above"), rows, cfg.adapter_max_out());
        LoraStore { ps, opt, rt }
    }

    pub fn adamw_step(&self, gpu: &Gpu, t: u32, lr: f32) {
        self.opt.step(gpu, &self.ps, t, lr, 0.0, optim::Adam::default(), None, 1.0);
    }

    pub fn zero_grads(&self, gpu: &Gpu) {
        self.ps.zero_grads(gpu);
    }
}

/// The adapter runtime's view of one streamed linear: the adapter tensors in `lora` and the frozen `base` in
/// the tier it was streamed at, so a kind that reads its base (DoRA) works over any tier.
fn bound<'a>(lora: &'a LoraStore, ops: &'a Ops, base: &'a Weight) -> BoundBase<'a> {
    BoundBase { params: &lora.ps, ops, weight: base }
}

/// Backward for one (possibly-LoRA) streamed linear `y = x*Wᵀ`, accumulating
/// the input gradient into `dx` (flag `acc`) - a direct kernel-dispatch
/// transcription of `Qwen35::proj_bwd`'s own body (see this module's own
/// doc), with two substitutions: `self.w(wname)` (the frozen base) becomes
/// `base` (this module's own streamed `Weight::F32`, read via [`f32_buf`]),
/// and `self.w`/`self.g` for the adapter become `lora.ps.w`/`lora.ps.g`. The
/// base weight NEVER gets a `dW` here (it is always frozen in this module -
/// there is no full-fine-tune path), matching `proj_bwd`'s own `None` arm
/// restricted to `dx`-only for exactly that reason.
#[allow(clippy::too_many_arguments)]
fn proj_bwd_streamed(ops: &Ops, lora: &LoraStore, leaf: &str, d_out: &DeviceBuffer, x: &DeviceBuffer, base: &Weight, wname: &str, dx: &DeviceBuffer, m: u32, k: u32, nout: u32, acc: u32, s: &mut Vec<Step>) {
    // base: dx += d_out*W (frozen weight - no dW), through the shared quantized-execution contract: any
    // resident tier (f32, bf16, NF4, FP8, ...) with no full-precision copy. An untargeted leaf, or the
    // shared head, has only this.
    debug_assert_eq!((base.n(), base.k()), (nout, k), "proj_bwd_streamed: weight shape does not match the linear");
    let base_dx = |s: &mut Vec<Step>, dy: &DeviceBuffer| ops.matmul_dx(s, base, dy, m, dx, acc != 0);
    if lora.rt.hits(leaf) {
        lora.rt.backward(ops.tensor_gemm(), s, &bound(lora, ops, base), wname, d_out, x, dx, m, k, nout, &base_dx);
    } else {
        base_dx(s, d_out);
    }
}

/// MLP forward activations this layer's backward needs beyond what it
/// recomputes fresh - `xn2`/`gate_pre`/`up`/`h`, the same set `crate::model::
/// Qwen35`'s own `MlpLayerActs` caches, just per-recompute rather than
/// per-run here (see this module's own doc, "recompute" design).
struct MlpActsL {
    xn2: DeviceBuffer,
    gate_pre: DeviceBuffer,
    up: DeviceBuffer,
    h: DeviceBuffer,
}

/// Dense SwiGLU MLP forward with LoRA - a direct transcription of `Qwen35::
/// mlp_fwd`, `gate`/`up`/`down` streamed `Weight::F32`s instead of `self.w`.
#[allow(clippy::too_many_arguments)]
fn mlp_forward_lora(gpu: &Gpu, ops: &Ops, ids: &TrainIds, cfg: &Qwen35Config, lora: &LoraStore, prefix: &str, gate: &Weight, up_w: &Weight, down_w: &Weight, xn2: &DeviceBuffer, n: u32) -> (DeviceBuffer, MlpActsL) {
    let d = cfg.d_model;
    let ff = cfg.intermediate_size;
    let p = |s: &str| format!("{prefix}.{s}");

    let gate_pre = gpu.storage((n * ff) as u64);
    let up = gpu.storage((n * ff) as u64);
    {
        let mut s = Vec::new();
        let act1 = ops.act_f32(xn2, 0, n, d);
        ops.matmul_rows(&mut s, gate, (0, gate.n()), &act1, &gate_pre, 0);
        ops.matmul_rows(&mut s, up_w, (0, up_w.n()), &act1, &up, 0);
        gpu.submit(&[], &s);
    }
    if lora.rt.hits("gate") {
        let mut s = Vec::new();
        lora.rt.forward(ops.tensor_gemm(), &mut s, &bound(lora, ops, gate), &p("gate.weight"), xn2, &gate_pre, n, d, ff);
        gpu.submit(&[], &s);
    }
    if lora.rt.hits("up") {
        let mut s = Vec::new();
        lora.rt.forward(ops.tensor_gemm(), &mut s, &bound(lora, ops, up_w), &p("up.weight"), xn2, &up, n, d, ff);
        gpu.submit(&[], &s);
    }

    let h = gpu.storage((n * ff) as u64);
    gpu.submit(&[], &[swiglu_fwd(gpu, &ids.kernels, &gate_pre, &up, &h, n * ff)]);

    let down = gpu.storage((n * d) as u64);
    {
        let mut s = Vec::new();
        let act2 = ops.act_f32(&h, 0, n, ff);
        ops.matmul_rows(&mut s, down_w, (0, down_w.n()), &act2, &down, 0);
        gpu.submit(&[], &s);
    }
    if lora.rt.hits("down") {
        let mut s = Vec::new();
        lora.rt.forward(ops.tensor_gemm(), &mut s, &bound(lora, ops, down_w), &p("down.weight"), &h, &down, n, ff, d);
        gpu.submit(&[], &s);
    }

    (down, MlpActsL { xn2: xn2.clone(), gate_pre, up, h })
}

/// Reverse of [`mlp_forward_lora`] - a direct transcription of `Qwen35::
/// mlp_bwd`.
#[allow(clippy::too_many_arguments)]
fn mlp_backward_lora(gpu: &Gpu, ops: &Ops, ids: &TrainIds, lora: &LoraStore, cfg: &Qwen35Config, prefix: &str, la: &MlpActsL, gate: &Weight, up_w: &Weight, down_w: &Weight, d_mlp_out: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let d = cfg.d_model;
    let ff = cfg.intermediate_size;
    let p = |s: &str| format!("{prefix}.{s}");

    let d_h = gpu.storage((n * ff) as u64);
    {
        let mut s = Vec::new();
        proj_bwd_streamed(ops, lora, "down", d_mlp_out, &la.h, down_w, &p("down.weight"), &d_h, n, ff, d, 0, &mut s);
        gpu.submit(&[], &s);
    }

    let d_gate_pre = gpu.storage((n * ff) as u64);
    let d_up = gpu.storage((n * ff) as u64);
    gpu.submit(&[], &swiglu_bwd(gpu, &ids.kernels, &la.gate_pre, &la.up, &d_h, &d_gate_pre, &d_up, n * ff));

    let d_xn2 = gpu.storage((n * d) as u64);
    {
        let mut s = Vec::new();
        proj_bwd_streamed(ops, lora, "up", &d_up, &la.xn2, up_w, &p("up.weight"), &d_xn2, n, d, ff, 0, &mut s);
        proj_bwd_streamed(ops, lora, "gate", &d_gate_pre, &la.xn2, gate, &p("gate.weight"), &d_xn2, n, d, ff, 1, &mut s);
        gpu.submit(&[], &s);
    }
    d_xn2
}

/// RMSNorm backward against a frozen (never-trainable) gain - `gw: None`
/// always in this module (no norm is ever a LoRA target). Small helper so
/// call sites don't repeat the throwaway `inv` scratch allocation.
fn rmsnorm_bwd_frozen(gpu: &Gpu, ids: &TrainIds, x: &DeviceBuffer, w: &DeviceBuffer, dy: &DeviceBuffer, dx: &DeviceBuffer, dim: u32, rows: u32, eps: f32) -> Vec<Step> {
    let inv = gpu.storage(rows as u64);
    rmsnorm_bwd(gpu, &ids.kernels, x, w, dy, dx, &inv, None, dim, rows, eps)
}

/// One Gated DeltaNet layer's activations this layer's backward needs beyond
/// what it recomputes fresh.
struct GdnLayerActsL {
    xn1: DeviceBuffer,
    gated: DeviceBuffer,
    mixer: GdnMixerActs,
    xmid: DeviceBuffer,
    mlp: MlpActsL,
}

/// One Gated DeltaNet layer forward, LoRA-aware - a direct transcription of
/// `Qwen35::layer_gdn_fwd` fused with its own residual/ln2/mlp caller loop
/// body (`run_forward`'s per-layer body), streaming `l`'s weights instead of
/// reading `self.w`/`self.weights`. Always captures the activations its own
/// backward twin needs (`is_train` is always effectively `true` here - see
/// this module's own doc, "recompute" design: the forward CALL SITE that
/// only wants the output residual simply drops the returned acts, the SAME
/// function serves both the caching forward pass and the backward-recompute
/// call).
#[allow(clippy::too_many_arguments)]
fn gdn_layer_forward_lora(gpu: &Gpu, ops: &Ops, ids: &TrainIds, cfg: &Qwen35Config, lora: &LoraStore, layer_idx: usize, layer: &OwnedGdnLayer, xres: &DeviceBuffer, n: u32, ones_khd: &DeviceBuffer) -> (DeviceBuffer, GdnLayerActsL) {
    let d = cfg.d_model;
    let xn1 = gpu.storage((n * d) as u64);
    gpu.submit(&[], &[rmsnorm_fwd(gpu, &ids.kernels, xres, &layer.ln1, &xn1, d, n, cfg.rms_eps)]);

    let conv_dim = cfg.linear_conv_dim();
    let value_dim = cfg.linear_value_dim();
    let nvh = cfg.linear_num_value_heads;
    let khd = cfg.linear_key_head_dim;
    let vhd = cfg.linear_value_head_dim;
    let p = |s: &str| format!("blocks.{layer_idx}.linear_attn.{s}");

    let mixed_qkv = gpu.storage((n * conv_dim) as u64);
    {
        let mut s = Vec::new();
        let act1 = ops.act_f32(&xn1, 0, n, d);
        ops.matmul_rows(&mut s, &layer.in_proj_qkv, (0, layer.in_proj_qkv.n()), &act1, &mixed_qkv, 0);
        gpu.submit(&[], &s);
    }
    if lora.rt.hits("in_proj_qkv") {
        let mut s = Vec::new();
        lora.rt.forward(ops.tensor_gemm(), &mut s, &bound(lora, ops, &layer.in_proj_qkv), &p("in_proj_qkv.weight"), &xn1, &mixed_qkv, n, d, conv_dim);
        gpu.submit(&[], &s);
    }

    let bproj = gpu.storage((n * nvh) as u64);
    let aproj = gpu.storage((n * nvh) as u64);
    let z = gpu.storage((n * value_dim) as u64);
    {
        let mut s = Vec::new();
        let act1 = ops.act_f32(&xn1, 0, n, d);
        ops.matmul_rows(&mut s, &layer.in_proj_b, (0, layer.in_proj_b.n()), &act1, &bproj, 0);
        ops.matmul_rows(&mut s, &layer.in_proj_a, (0, layer.in_proj_a.n()), &act1, &aproj, 0);
        ops.matmul_rows(&mut s, &layer.in_proj_z, (0, layer.in_proj_z.n()), &act1, &z, 0);
        gpu.submit(&[], &s);
    }
    if lora.rt.hits("in_proj_b") {
        let mut s = Vec::new();
        lora.rt.forward(ops.tensor_gemm(), &mut s, &bound(lora, ops, &layer.in_proj_b), &p("in_proj_b.weight"), &xn1, &bproj, n, d, nvh);
        gpu.submit(&[], &s);
    }
    if lora.rt.hits("in_proj_a") {
        let mut s = Vec::new();
        lora.rt.forward(ops.tensor_gemm(), &mut s, &bound(lora, ops, &layer.in_proj_a), &p("in_proj_a.weight"), &xn1, &aproj, n, d, nvh);
        gpu.submit(&[], &s);
    }
    if lora.rt.hits("in_proj_z") {
        let mut s = Vec::new();
        lora.rt.forward(ops.tensor_gemm(), &mut s, &bound(lora, ops, &layer.in_proj_z), &p("in_proj_z.weight"), &xn1, &z, n, d, value_dim);
        gpu.submit(&[], &s);
    }

    let shape = GdnMixerShape { gdn: GdnShape { b: 1, h: nvh, t: n, dk: khd, dv: vhd, chunk: gdn_chunk_size(n) }, nkh: cfg.linear_num_key_heads, conv_kernel: cfg.linear_conv_kernel_dim, rms_eps: cfg.rms_eps };
    let mix_w = GdnMixerWeights { conv1d_weight: &layer.conv1d_weight, a_log: &layer.a_log, dt_bias: &layer.dt_bias, norm_weight: &layer.norm_weight, ones_khd };
    let (gated, mixer_internals) = gdn_mixer_fwd(gpu, &ids.gdn_mixer, &shape, &mix_w, &mixed_qkv, &bproj, &aproj, &z, n, true);
    let mixer_internals = mixer_internals.expect("stream_train: gdn_mixer_fwd(is_train=true) must return acts");

    let mixer_out = gpu.storage((n * d) as u64);
    {
        let mut s = Vec::new();
        let act3 = ops.act_f32(&gated, 0, n, value_dim);
        ops.matmul_rows(&mut s, &layer.out_proj, (0, layer.out_proj.n()), &act3, &mixer_out, 0);
        gpu.submit(&[], &s);
    }
    if lora.rt.hits("out_proj") {
        let mut s = Vec::new();
        lora.rt.forward(ops.tensor_gemm(), &mut s, &bound(lora, ops, &layer.out_proj), &p("out_proj.weight"), &gated, &mixer_out, n, value_dim, d);
        gpu.submit(&[], &s);
    }

    let xmid = gpu.storage((n * d) as u64);
    gpu.submit(&[], &[gpu.step(ids.add2, &[xres, &mixer_out, &xmid], &[n * d], n * d)]);

    let xn2 = gpu.storage((n * d) as u64);
    gpu.submit(&[], &[rmsnorm_fwd(gpu, &ids.kernels, &xmid, &layer.ln2, &xn2, d, n, cfg.rms_eps)]);

    let (mlp_out, mlp_acts) = mlp_forward_lora(gpu, ops, ids, cfg, lora, &format!("blocks.{layer_idx}.mlp"), &layer.mlp_gate, &layer.mlp_up, &layer.mlp_down, &xn2, n);

    let out = gpu.storage((n * d) as u64);
    gpu.submit(&[], &[gpu.step(ids.add2, &[&xmid, &mlp_out, &out], &[n * d], n * d)]);

    (out, GdnLayerActsL { xn1, gated, mixer: mixer_internals, xmid, mlp: mlp_acts })
}

/// Reverse of the Gated DeltaNet mixer (out_proj + `gdn_mixer_bwd` + the 4
/// in_proj leaves) - a direct transcription of `Qwen35::gdn_mixer_bwd`, mixer
/// aux weights (`conv1d_weight`/`a_log`/`dt_bias`/`norm_weight`) always
/// ungraded (`GdnMixerGrads` all `None` - no aux tensor is ever a LoRA
/// target in this module, see `crate::config::lora_targets`).
#[allow(clippy::too_many_arguments)]
fn gdn_mixer_backward_lora(gpu: &Gpu, ops: &Ops, ids: &TrainIds, cfg: &Qwen35Config, lora: &LoraStore, layer_idx: usize, layer: &OwnedGdnLayer, xn1: &DeviceBuffer, gated: &DeviceBuffer, mixer_acts: &GdnMixerActs, d_out: &DeviceBuffer, d_xn1: &DeviceBuffer, n: u32, ones_khd: &DeviceBuffer) {
    let d = cfg.d_model;
    let conv_dim = cfg.linear_conv_dim();
    let value_dim = cfg.linear_value_dim();
    let nvh = cfg.linear_num_value_heads;
    let khd = cfg.linear_key_head_dim;
    let vhd = cfg.linear_value_head_dim;
    let p = |s: &str| format!("blocks.{layer_idx}.linear_attn.{s}");

    let d_gated = gpu.storage((n * value_dim) as u64);
    {
        let mut s = Vec::new();
        proj_bwd_streamed(ops, lora, "out_proj", d_out, gated, &layer.out_proj, &p("out_proj.weight"), &d_gated, n, value_dim, d, 0, &mut s);
        gpu.submit(&[], &s);
    }

    let shape = GdnMixerShape { gdn: GdnShape { b: 1, h: nvh, t: n, dk: khd, dv: vhd, chunk: gdn_chunk_size(n) }, nkh: cfg.linear_num_key_heads, conv_kernel: cfg.linear_conv_kernel_dim, rms_eps: cfg.rms_eps };
    let weights = GdnMixerWeights { conv1d_weight: &layer.conv1d_weight, a_log: &layer.a_log, dt_bias: &layer.dt_bias, norm_weight: &layer.norm_weight, ones_khd };
    let grads = GdnMixerGrads { conv1d_weight: None, a_log: None, dt_bias: None, norm_weight: None };
    let (d_mixed_qkv, d_bproj, d_aproj, d_z) = gdn_mixer_bwd(gpu, &ids.gdn_mixer, &shape, &weights, &grads, mixer_acts, &d_gated, n);

    {
        let mut s = Vec::new();
        proj_bwd_streamed(ops, lora, "in_proj_b", &d_bproj, xn1, &layer.in_proj_b, &p("in_proj_b.weight"), d_xn1, n, d, nvh, 0, &mut s);
        proj_bwd_streamed(ops, lora, "in_proj_a", &d_aproj, xn1, &layer.in_proj_a, &p("in_proj_a.weight"), d_xn1, n, d, nvh, 1, &mut s);
        proj_bwd_streamed(ops, lora, "in_proj_z", &d_z, xn1, &layer.in_proj_z, &p("in_proj_z.weight"), d_xn1, n, d, value_dim, 1, &mut s);
        gpu.submit(&[], &s);
    }
    {
        let mut s = Vec::new();
        proj_bwd_streamed(ops, lora, "in_proj_qkv", &d_mixed_qkv, xn1, &layer.in_proj_qkv, &p("in_proj_qkv.weight"), d_xn1, n, d, conv_dim, 1, &mut s);
        gpu.submit(&[], &s);
    }
}

/// One Gated DeltaNet layer backward: recomputes this layer's own forward
/// (fresh `xn1`/mixer internals/MLP internals - see this module's own doc,
/// "recompute" design) from `xres_l` (the cached residual INPUT to this
/// layer) and the just-re-streamed weights, then runs the reverse of
/// `run_forward`'s own per-layer body (second residual add -> MLP -> ln2 ->
/// first residual add -> mixer -> ln1) - a direct transcription of the
/// per-layer body inside `Qwen35::backward`'s own reverse loop.
#[allow(clippy::too_many_arguments)]
fn gdn_layer_backward_lora(gpu: &Gpu, ops: &Ops, ids: &TrainIds, cfg: &Qwen35Config, lora: &LoraStore, layer_idx: usize, layer: &OwnedGdnLayer, xres_l: &DeviceBuffer, d_res_next: &DeviceBuffer, n: u32, ones_khd: &DeviceBuffer) -> DeviceBuffer {
    let d = cfg.d_model;
    let (_out, acts) = gdn_layer_forward_lora(gpu, ops, ids, cfg, lora, layer_idx, layer, xres_l, n, ones_khd);

    let d_xn2 = mlp_backward_lora(gpu, ops, ids, lora, cfg, &format!("blocks.{layer_idx}.mlp"), &acts.mlp, &layer.mlp_gate, &layer.mlp_up, &layer.mlp_down, d_res_next, n);

    let d_xmid = gpu.storage((n * d) as u64);
    {
        let mut s = Vec::new();
        let d_ln2_dx = gpu.storage((n * d) as u64);
        s.extend(rmsnorm_bwd_frozen(gpu, ids, &acts.xmid, &layer.ln2, &d_xn2, &d_ln2_dx, d, n, cfg.rms_eps));
        s.push(gpu.step(ids.add2, &[d_res_next, &d_ln2_dx, &d_xmid], &[n * d], n * d));
        gpu.submit(&[], &s);
    }

    let d_xn1 = gpu.storage((n * d) as u64);
    gdn_mixer_backward_lora(gpu, ops, ids, cfg, lora, layer_idx, layer, &acts.xn1, &acts.gated, &acts.mixer, &d_xmid, &d_xn1, n, ones_khd);

    let d_res_l = gpu.storage((n * d) as u64);
    {
        let mut s = Vec::new();
        let d_ln1_dx = gpu.storage((n * d) as u64);
        s.extend(rmsnorm_bwd_frozen(gpu, ids, xres_l, &layer.ln1, &d_xn1, &d_ln1_dx, d, n, cfg.rms_eps));
        s.push(gpu.step(ids.add2, &[&d_xmid, &d_ln1_dx, &d_res_l], &[n * d], n * d));
        gpu.submit(&[], &s);
    }
    d_res_l
}

/// GQA layer forward activations needed by backward.
struct GqaLayerActsL {
    xn1: DeviceBuffer,
    ctx_gated: DeviceBuffer,
    mixer: GqaMixerActs,
    xmid: DeviceBuffer,
    mlp: MlpActsL,
}

/// One GQA layer forward, LoRA-aware - a direct transcription of `Qwen35::
/// layer_gqa_fwd` fused with `run_forward`'s own per-layer body, same shape
/// as [`gdn_layer_forward_lora`].
#[allow(clippy::too_many_arguments)]
fn gqa_layer_forward_lora(gpu: &Gpu, ops: &Ops, ids: &TrainIds, cfg: &Qwen35Config, lora: &LoraStore, layer_idx: usize, layer: &OwnedGqaLayer, xres: &DeviceBuffer, n: u32, cos: &DeviceBuffer, sin: &DeviceBuffer) -> (DeviceBuffer, GqaLayerActsL) {
    let d = cfg.d_model;
    let xn1 = gpu.storage((n * d) as u64);
    gpu.submit(&[], &[rmsnorm_fwd(gpu, &ids.kernels, xres, &layer.ln1, &xn1, d, n, cfg.rms_eps)]);

    let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    let (qpd, kvd) = (cfg.q_proj_dim(), cfg.kv_dim());
    let p = |s: &str| format!("blocks.{layer_idx}.self_attn.{s}");

    let q_full = gpu.storage((n * qpd) as u64);
    let k = gpu.storage((n * kvd) as u64);
    let v = gpu.storage((n * kvd) as u64);
    {
        let mut s = Vec::new();
        let act1 = ops.act_f32(&xn1, 0, n, d);
        ops.matmul_rows(&mut s, &layer.q_proj, (0, layer.q_proj.n()), &act1, &q_full, 0);
        ops.matmul_rows(&mut s, &layer.k_proj, (0, layer.k_proj.n()), &act1, &k, 0);
        ops.matmul_rows(&mut s, &layer.v_proj, (0, layer.v_proj.n()), &act1, &v, 0);
        gpu.submit(&[], &s);
    }
    if lora.rt.hits("q_proj") {
        let mut s = Vec::new();
        lora.rt.forward(ops.tensor_gemm(), &mut s, &bound(lora, ops, &layer.q_proj), &p("q_proj.weight"), &xn1, &q_full, n, d, qpd);
        gpu.submit(&[], &s);
    }
    if lora.rt.hits("k_proj") {
        let mut s = Vec::new();
        lora.rt.forward(ops.tensor_gemm(), &mut s, &bound(lora, ops, &layer.k_proj), &p("k_proj.weight"), &xn1, &k, n, d, kvd);
        gpu.submit(&[], &s);
    }
    if lora.rt.hits("v_proj") {
        let mut s = Vec::new();
        lora.rt.forward(ops.tensor_gemm(), &mut s, &bound(lora, ops, &layer.v_proj), &p("v_proj.weight"), &xn1, &v, n, d, kvd);
        gpu.submit(&[], &s);
    }

    cfg.require_plain_decoder("the layer-streamed trainer");
    let shape = GqaMixerShape { b: 1, t: n, n_heads: nh, n_kv_heads: nkv, head_dim: hd, rotary_half: cfg.rotary_dim() / 2, rms_eps: cfg.rms_eps };
    let mix_w = GqaMixerWeights { q_norm: &layer.q_norm, k_norm: &layer.k_norm, cos, sin };
    let flash = ids.flash.and_then(|f| f.for_training(gpu, hd, n));
    let (ctx_gated, mixer_internals) = gqa_mixer_fwd_tc(gpu, &ids.gqa_mixer, flash, Some(ops.tensor_gemm()), &shape, &mix_w, &q_full, &k, &v, n, true);
    let mixer_internals = mixer_internals.expect("stream_train: gqa_mixer_fwd(is_train=true) must return acts");

    let mixer_out = gpu.storage((n * d) as u64);
    {
        let mut s = Vec::new();
        let act2 = ops.act_f32(&ctx_gated, 0, n, shape.qd());
        ops.matmul_rows(&mut s, &layer.o_proj, (0, layer.o_proj.n()), &act2, &mixer_out, 0);
        gpu.submit(&[], &s);
    }
    if lora.rt.hits("o_proj") {
        let mut s = Vec::new();
        lora.rt.forward(ops.tensor_gemm(), &mut s, &bound(lora, ops, &layer.o_proj), &p("o_proj.weight"), &ctx_gated, &mixer_out, n, shape.qd(), d);
        gpu.submit(&[], &s);
    }

    let xmid = gpu.storage((n * d) as u64);
    gpu.submit(&[], &[gpu.step(ids.add2, &[xres, &mixer_out, &xmid], &[n * d], n * d)]);

    let xn2 = gpu.storage((n * d) as u64);
    gpu.submit(&[], &[rmsnorm_fwd(gpu, &ids.kernels, &xmid, &layer.ln2, &xn2, d, n, cfg.rms_eps)]);

    let (mlp_out, mlp_acts) = mlp_forward_lora(gpu, ops, ids, cfg, lora, &format!("blocks.{layer_idx}.mlp"), &layer.mlp_gate, &layer.mlp_up, &layer.mlp_down, &xn2, n);

    let out = gpu.storage((n * d) as u64);
    gpu.submit(&[], &[gpu.step(ids.add2, &[&xmid, &mlp_out, &out], &[n * d], n * d)]);

    (out, GqaLayerActsL { xn1, ctx_gated, mixer: mixer_internals, xmid, mlp: mlp_acts })
}

/// Reverse of the GQA mixer (o_proj + `gqa_mixer_bwd` + q/k/v proj) - a
/// direct transcription of `Qwen35::gqa_mixer_bwd`, `q_norm`/`k_norm` always
/// ungraded (never a LoRA target).
#[allow(clippy::too_many_arguments)]
fn gqa_mixer_backward_lora(gpu: &Gpu, ops: &Ops, ids: &TrainIds, cfg: &Qwen35Config, lora: &LoraStore, layer_idx: usize, layer: &OwnedGqaLayer, xn1: &DeviceBuffer, ctx_gated: &DeviceBuffer, mixer_acts: &GqaMixerActs, d_out: &DeviceBuffer, d_xn1: &DeviceBuffer, n: u32, cos: &DeviceBuffer, sin: &DeviceBuffer) {
    let d = cfg.d_model;
    let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    let (qpd, qd, kvd) = (cfg.q_proj_dim(), cfg.q_dim(), cfg.kv_dim());
    let p = |s: &str| format!("blocks.{layer_idx}.self_attn.{s}");

    let d_ctx_gated = gpu.storage((n * qd) as u64);
    {
        let mut s = Vec::new();
        proj_bwd_streamed(ops, lora, "o_proj", d_out, ctx_gated, &layer.o_proj, &p("o_proj.weight"), &d_ctx_gated, n, qd, d, 0, &mut s);
        gpu.submit(&[], &s);
    }

    cfg.require_plain_decoder("the layer-streamed trainer");
    let shape = GqaMixerShape { b: 1, t: n, n_heads: nh, n_kv_heads: nkv, head_dim: hd, rotary_half: cfg.rotary_dim() / 2, rms_eps: cfg.rms_eps };
    let weights = GqaMixerWeights { q_norm: &layer.q_norm, k_norm: &layer.k_norm, cos, sin };
    let grads = GqaMixerGrads { q_norm: None, k_norm: None };
    let (d_q_full, d_k, d_v) = gqa_mixer_bwd(gpu, &ids.gqa_mixer, &shape, &weights, &grads, mixer_acts, &d_ctx_gated, n);

    {
        let mut s = Vec::new();
        proj_bwd_streamed(ops, lora, "q_proj", &d_q_full, xn1, &layer.q_proj, &p("q_proj.weight"), d_xn1, n, d, qpd, 0, &mut s);
        proj_bwd_streamed(ops, lora, "k_proj", &d_k, xn1, &layer.k_proj, &p("k_proj.weight"), d_xn1, n, d, kvd, 1, &mut s);
        proj_bwd_streamed(ops, lora, "v_proj", &d_v, xn1, &layer.v_proj, &p("v_proj.weight"), d_xn1, n, d, kvd, 1, &mut s);
        gpu.submit(&[], &s);
    }
}

/// One GQA layer backward - the GQA sibling of [`gdn_layer_backward_lora`].
#[allow(clippy::too_many_arguments)]
fn gqa_layer_backward_lora(gpu: &Gpu, ops: &Ops, ids: &TrainIds, cfg: &Qwen35Config, lora: &LoraStore, layer_idx: usize, layer: &OwnedGqaLayer, xres_l: &DeviceBuffer, d_res_next: &DeviceBuffer, n: u32, cos: &DeviceBuffer, sin: &DeviceBuffer) -> DeviceBuffer {
    let d = cfg.d_model;
    let (_out, acts) = gqa_layer_forward_lora(gpu, ops, ids, cfg, lora, layer_idx, layer, xres_l, n, cos, sin);

    let d_xn2 = mlp_backward_lora(gpu, ops, ids, lora, cfg, &format!("blocks.{layer_idx}.mlp"), &acts.mlp, &layer.mlp_gate, &layer.mlp_up, &layer.mlp_down, d_res_next, n);

    let d_xmid = gpu.storage((n * d) as u64);
    {
        let mut s = Vec::new();
        let d_ln2_dx = gpu.storage((n * d) as u64);
        s.extend(rmsnorm_bwd_frozen(gpu, ids, &acts.xmid, &layer.ln2, &d_xn2, &d_ln2_dx, d, n, cfg.rms_eps));
        s.push(gpu.step(ids.add2, &[d_res_next, &d_ln2_dx, &d_xmid], &[n * d], n * d));
        gpu.submit(&[], &s);
    }

    let d_xn1 = gpu.storage((n * d) as u64);
    gqa_mixer_backward_lora(gpu, ops, ids, cfg, lora, layer_idx, layer, &acts.xn1, &acts.ctx_gated, &acts.mixer, &d_xmid, &d_xn1, n, cos, sin);

    let d_res_l = gpu.storage((n * d) as u64);
    {
        let mut s = Vec::new();
        let d_ln1_dx = gpu.storage((n * d) as u64);
        s.extend(rmsnorm_bwd_frozen(gpu, ids, xres_l, &layer.ln1, &d_xn1, &d_ln1_dx, d, n, cfg.rms_eps));
        s.push(gpu.step(ids.add2, &[&d_xmid, &d_ln1_dx, &d_res_l], &[n * d], n * d));
        gpu.submit(&[], &s);
    }
    d_res_l
}

/// Everything a streaming LoRA training step needs that is NOT per-layer:
/// the device handle, the fp32 façade, resolved kernel indices, the model-
/// wide RoPE/L2-norm scratch [`crate::stream::StreamState`] also builds, the
/// resident LoRA adapter store, and the resident shared head (`lm_head`/
/// `embed_tokens`, tied or not) + final norm.
pub struct StreamTrainer {
    pub gpu: Gpu,
    ops: Ops,
    ids: TrainIds,
    ones_khd: DeviceBuffer,
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    pub lora: LoraStore,
    head: Weight,
    head_ce: HeadCe,
    final_norm: DeviceBuffer,
    n: u32,
    window_budget: u32,
    /// The host staging every load of this trainer draws on: one tile of the destination encoding, plus the
    /// pinned ring the layer uploads are copied through ([`Self::new_real_staged`]).
    budget: checkpoint::packed::MemoryBudget,
    /// Seconds the compute thread has waited for the next layer to arrive, and spent computing, over every pass
    /// of this trainer ([`Self::pass_timing`]).
    waited_for_load: std::cell::Cell<f64>,
    computed: std::cell::Cell<f64>,
}

/// Pinned host memory the layer uploads of [`StreamTrainer::new_real`] stage through: a few slots, each big enough
/// to keep the copy engine streaming (the loader's tiles are a few MiB).
const UPLOAD_STAGING_BYTES: u64 = 64 << 20;

impl StreamTrainer {
    /// The most pinned host memory this trainer's uploads held at once (0 without a staging ring).
    pub fn pinned_staging_high_water(&self) -> u64 {
        self.budget.stats(checkpoint::packed::Pool::Pinned).high_water
    }

    /// `(seconds waiting for a layer to be loaded, seconds computing)` summed over every layer pass so far. A step
    /// whose wait dominates is bound by streaming the base; one whose compute dominates is bound by the device.
    pub fn pass_timing(&self) -> (f64, f64) {
        (self.waited_for_load.get(), self.computed.get())
    }

    /// Build against an already-resident host weight map (`init_weights`'s
    /// own shape) - no disk I/O, no real checkpoint needed. This is the
    /// tiny-scale equivalence gate's own construction path: the SAME
    /// `HashMap` a resident `Qwen35::new_train_on` build reads feeds this
    /// trainer's `lm_head`/`embed_tokens`/`norm.weight` too, so the two
    /// trainers start from byte-identical weights.
    pub fn new_synthetic(gpu: Gpu, cfg: &Qwen35Config, n: u32, window_budget: u32, init: &HashMap<String, Vec<f32>>) -> StreamTrainer {
        let ops = Ops::new(gpu.share()).unwrap_or_else(|e| panic!("stream_train: Ops::new: {e}"));
        let ids = train_ids(&gpu, &ops);
        let d = cfg.d_model as usize;
        let v = cfg.vocab as usize;
        let ones_khd = gpu.storage_init("stream_train.ones_khd", &vec![1.0f32; cfg.linear_key_head_dim as usize]);
        let positions: Vec<[u32; 3]> = (0..n).map(|ti| [ti, ti, ti]).collect();
        let (cos, sin) = qwen3vl::mrope::mrope_tables(&positions, cfg.mrope_section, cfg.rotary_dim(), cfg.rope_theta);
        let cos = gpu.storage_init("stream_train.rope_cos", &cos);
        let sin = gpu.storage_init("stream_train.rope_sin", &sin);

        let head_name = if cfg.tie_embeddings { "tok.weight" } else { "lm_head.weight" };
        let head_raw = init.get(head_name).unwrap_or_else(|| panic!("stream_train: missing {head_name}"));
        let head = Weight::upload(&ops, head_raw, v, d, Dtype::F32);
        let norm_raw = init.get("norm.weight").unwrap_or_else(|| panic!("stream_train: missing norm.weight"));
        let final_norm = gpu.storage_init("stream_train.final_norm", norm_raw);

        let head_ce = build_head_ce(&ops, &gpu, cfg, &head, n);
        let lora = LoraStore::new(&gpu, cfg, init, n);
        let budget = checkpoint::packed::MemoryBudget::new(crate::fp8_source::WEIGHT_STAGING_BYTES, 0);
        StreamTrainer { gpu, ops, ids, ones_khd, cos, sin, lora, head, head_ce, final_norm, n, window_budget, budget, waited_for_load: Default::default(), computed: Default::default() }
    }

    /// Real per-token embedding rows for `ids` from a host-resident `tok.
    /// weight` (tiny-scale only - see `crate::stream::embed_rows` for the
    /// real-checkpoint, mmap-row-range sibling this deliberately does NOT
    /// need at this scale).
    pub fn embed_synthetic(init: &HashMap<String, Vec<f32>>, ids: &[u32], d: usize) -> Vec<f32> {
        let tok = init.get("tok.weight").unwrap_or_else(|| panic!("stream_train: missing tok.weight"));
        let mut out = Vec::with_capacity(ids.len() * d);
        for &id in ids {
            let row0 = id as usize * d;
            out.extend_from_slice(&tok[row0..row0 + d]);
        }
        out
    }

    /// Build against a real checkpoint directory (`dir`, the SAME
    /// `Qwen/Qwen3.8-27B-FP8` layout `crate::stream::generate` reads):
    /// `lm_head.weight`/`embed_tokens.weight` (tied per `cfg.tie_embeddings`)
    /// and `norm.weight` are loaded ONCE (the head in its stored encoding)
    /// and kept resident for the whole run, never re-streamed per layer. The 64 main-stack layers are NOT
    /// loaded here - a caller streams those per step via [`Self::
    /// real_loader`].
    pub fn new_real(gpu: Gpu, cfg: &Qwen35Config, dir: &Path, n: u32, window_budget: u32, lora_init: &HashMap<String, Vec<f32>>) -> Result<StreamTrainer, String> {
        Self::new_real_staged(gpu, cfg, dir, n, window_budget, lora_init, UPLOAD_STAGING_BYTES)
    }

    /// [`Self::new_real`] with an explicit pinned staging ring for the layer uploads: `staging_bytes` of page-locked
    /// host memory the packed layers are copied through and uploaded from on a transfer stream, so layer `i + 1`
    /// moves to the device while layer `i` computes (`Ops::enable_streamed_uploads`). `0` keeps the plain
    /// synchronous upload on the compute stream - the reference the overlap is held bit-identical to.
    pub fn new_real_staged(gpu: Gpu, cfg: &Qwen35Config, dir: &Path, n: u32, window_budget: u32, lora_init: &HashMap<String, Vec<f32>>, staging_bytes: u64) -> Result<StreamTrainer, String> {
        // The planner refuses before the head is read: a card whose memory is known must hold the footprint.
        if let memauth::Device::Gpu(i) = gpu.memory_device() {
            if let Some(card) = gpu_core::devices::gpus().get(i as usize).map(|d| d.identity.vram_bytes).filter(|&b| b > 0) {
                Self::plan_before_load(cfg, window_budget, lora_init, card)?;
            }
        }
        let ops = Ops::new(gpu.share()).unwrap_or_else(|e| panic!("stream_train: Ops::new: {e}"));
        // The layers' weights are allocated by the loader and freed by the consumer once per pass: keep the freed
        // blocks for the next layer instead of paying a device-wide wait per free.
        ops.gpu().hold_freed_blocks(true);
        let budget = checkpoint::packed::MemoryBudget::new(crate::fp8_source::WEIGHT_STAGING_BYTES, staging_bytes);
        if staging_bytes > 0 {
            ops.enable_streamed_uploads(&budget, staging_bytes)?;
        }
        let ids = train_ids(&gpu, &ops);
        let d = cfg.d_model as usize;
        let v = cfg.vocab as usize;
        let ones_khd = gpu.storage_init("stream_train.ones_khd", &vec![1.0f32; cfg.linear_key_head_dim as usize]);
        let positions: Vec<[u32; 3]> = (0..n).map(|ti| [ti, ti, ti]).collect();
        let (cos, sin) = qwen3vl::mrope::mrope_tables(&positions, cfg.mrope_section, cfg.rotary_dim(), cfg.rope_theta);
        let cos = gpu.storage_init("stream_train.rope_cos", &cos);
        let sin = gpu.storage_init("stream_train.rope_sin", &sin);

        // The shared head and final norm stay resident for the whole call: the head in the encoding the
        // checkpoint stores it in (bf16 for the real release), never expanded to fp32.
        let base = crate::fp8_source::PackedBase::open(dir, cfg, crate::fp8_source::hf_fp8_block_size(dir)?)?.with_budget(budget.clone());
        let head_name = if cfg.tie_embeddings { "tok.weight" } else { "lm_head.weight" };
        let head = base.weight(&ops, head_name, v, d)?;
        let final_norm = gpu.storage_init("stream_train.final_norm", &base.small_f32("norm.weight")?);

        let head_ce = build_head_ce(&ops, &gpu, cfg, &head, n);
        let lora = LoraStore::new(&gpu, cfg, lora_init, n);
        Ok(StreamTrainer { gpu, ops, ids, ones_khd, cos, sin, lora, head, head_ce, final_norm, n, window_budget, budget, waited_for_load: Default::default(), computed: Default::default() })
    }

    /// The streamed trainer's resident footprint as planner categories: `window_budget` layers of the stored
    /// base (the larger layer type, int8-sized as the fp8 release is), the head and final norm, and the LoRA
    /// values, gradients and AdamW state. Activations are NOT modelled here - the streamed backward owns them.
    pub fn footprint_demands(cfg: &Qwen35Config, window_budget: u32, lora_init: &HashMap<String, Vec<f32>>) -> Vec<model::memplan::Demand> {
        use model::memplan::{training_demands, Category, Demand, ParamPrecision, TrainingFootprint};
        let tier = model::ops::TierPolicy::uniform(Dtype::I8);
        let layer = [crate::config::LayerType::Linear, crate::config::LayerType::Full].into_iter().map(|t| cfg.layer_weight_bytes(t, &tier)).max().unwrap_or(0);
        let head = cfg.vocab as u64 * cfg.d_model as u64 * 2 + cfg.d_model as u64 * 4;
        let trainable: Vec<(String, usize)> = lora_init.iter().filter(|(k, _)| model::adapter::device::is_adapter_param(k)).map(|(k, v)| (k.clone(), v.len())).collect();
        let footprint = TrainingFootprint {
            frozen_weight_bytes: window_budget as u64 * layer + head,
            trainable,
            precision: ParamPrecision::FP32,
            moment_slots: 2,
            int8_block: 256,
            optimizer_state_on_host: false,
            activation_bytes: 0,
            recompute_activation_bytes: None,
            saved_boundaries: None,
        };
        let mut demands = training_demands(&footprint).expect("the training categories are consistent");
        demands.retain(|d| d.category != Category::Activations);
        demands.push(Demand::fixed(Category::Activations, "not modelled here", 0));
        demands
    }

    /// Refuse before loading when [`Self::footprint_demands`] cannot fit `device_bytes`, with the arithmetic.
    pub fn plan_before_load(cfg: &Qwen35Config, window_budget: u32, lora_init: &HashMap<String, Vec<f32>>, device_bytes: u64) -> Result<(), String> {
        let demands = Self::footprint_demands(cfg, window_budget, lora_init);
        model::memplan::plan(&demands, model::memplan::Capacity { device_bytes, host_bytes: u64::MAX }, &model::memplan::Authorized::none()).map(drop).map_err(|e| format!("stream_train: refusing to load: {e}"))
    }

    /// Real per-token embedding rows for `ids`, off `dir/outside.
    /// safetensors`'s `model.language_model.embed_tokens.weight` - the
    /// real-checkpoint counterpart of [`Self::embed_synthetic`], reusing
    /// `crate::stream::embed_rows` (unchanged) directly.
    pub fn embed_real(dir: &Path, ids: &[u32], d: usize) -> Result<Vec<f32>, String> {
        // The release keeps the embedding in `outside.safetensors`; a base converted by
        // `fp8_source::write_converted_base` is one `model.safetensors` holding it under the same name.
        let file = ["outside.safetensors", "model.safetensors"].into_iter().map(|f| dir.join(f)).find(|p| p.exists()).ok_or_else(|| format!("{}: neither outside.safetensors nor model.safetensors holds the embedding", dir.display()))?;
        let outside = crate::stream::open_outside(&file)?;
        crate::stream::embed_rows(&outside, "model.language_model.embed_tokens.weight", ids, d)
    }

    /// Greedy generation reflecting THIS trainer's own current LoRA adapter
    /// values - a fresh (zero-`.lora_b`) [`LoraStore`] reproduces the base
    /// model's own behavior exactly (LoRA's own "starts as an exact no-op"
    /// invariant); a trained one reflects the fine-tune. Reuses the SAME
    /// per-layer forward functions ([`gdn_layer_forward_lora`]/
    /// [`gqa_layer_forward_lora`]) [`Self::forward_backward`]'s own forward
    /// pass calls - the "before" and "after" transcripts this milestone's
    /// own gate calls for differ ONLY in which `LoraStore` state this call
    /// is made against, never in which CODE path runs. Single-token-at-a-
    /// time, non-incremental (re-streams the whole growing sequence every
    /// step, same design as `crate::stream::generate` - see that function's
    /// own doc for why), GREEDY ONLY (temperature 0) - this milestone's own
    /// scope is a qualitative before/after check, not a sampling-quality
    /// study.
    pub fn generate_greedy<F: Fn(usize) -> OwnedStreamedLayer>(&self, cfg: &Qwen35Config, loader: F, dir: &Path, tokenizer_path: &Path, prompt: &str, max_new: usize) -> Result<String, String> {
        let tok_path = tokenizer_path.to_str().ok_or_else(|| "stream_train::generate_greedy: tokenizer path is not valid UTF-8".to_string())?;
        let tok = QwenBpe::from_file(tok_path)?;
        let mut ids = tok.encode(prompt);
        if ids.is_empty() {
            return Err("stream_train::generate_greedy: prompt encoded to zero tokens".to_string());
        }
        let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|s| tok.encode(s).first().copied()).collect();

        let g = &self.gpu;
        let d = cfg.d_model as usize;
        let mut generated: Vec<u32> = Vec::with_capacity(max_new);

        for _ in 0..max_new {
            let t = ids.len() as u32;
            let padded_t = crate::stream::pad_to_gdn_chunk(t);
            let mut padded_ids = ids.clone();
            padded_ids.resize(padded_t as usize, 0);
            let x0 = Self::embed_real(dir, &padded_ids, d)?;

            // Every device buffer this step makes - the mrope tables, the
            // residual chain, and the whole streamed layer window below - is
            // scoped to this closure, so the step's footprint is dropped and
            // THEN reclaimed before the next step allocates. Reclaiming
            // before the drop (a `poll_wait` at the end of the loop body,
            // say) would reclaim nothing: see `gpu_core::transient`.
            let next = gpu_core::reclaiming(g, || {
                let positions: Vec<[u32; 3]> = (0..padded_t).map(|ti| [ti, ti, ti]).collect();
                let (cos, sin) = qwen3vl::mrope::mrope_tables(&positions, cfg.mrope_section, cfg.rotary_dim(), cfg.rope_theta);
                let cos = g.storage_init("stream_train.gen.cos", &cos);
                let sin = g.storage_init("stream_train.gen.sin", &sin);

                let mut xres_buf = g.storage_init("stream_train.gen.x0", &x0);
                let n_layers = cfg.n_layers;
                let sched = weightset::Schedule::cyclic(n_layers, 1);
                let mut ws = weightset::WeightSet::build(n_layers, self.window_budget, sched, Box::new(weightset::CyclicScan { lookahead: 1 }))
                    .unwrap_or_else(|e| panic!("stream_train::generate_greedy: WeightSet::build: {e}"));
                let mut slots: Vec<Option<OwnedStreamedLayer>> = (0..self.window_budget).map(|_| None).collect();
                for (i, slot) in ws.slot_contents().iter().enumerate() {
                    if let Some(gid) = slot {
                        slots[i] = Some(loader(gid.0 as usize));
                    }
                }
                for cursor in 0..n_layers as usize {
                    let (slot_id, miss) = ws.advance(cursor);
                    let idx = slot_id.0 as usize;
                    if miss {
                        // Evict, reclaim, THEN upload the replacement - the
                        // outgoing layer's bytes are what the incoming one's
                        // upload needs back (`gpu_core::transient`).
                        gpu_core::reclaiming(g, || slots[idx] = None);
                        slots[idx] = Some(loader(cursor));
                    }
                    let layer = slots[idx].as_ref().expect("stream_train::generate_greedy: WeightSet says this slot is resident");
                    let out = match layer {
                        OwnedStreamedLayer::Linear(l) => gdn_layer_forward_lora(g, &self.ops, &self.ids, cfg, &self.lora, cursor, l, &xres_buf, padded_t, &self.ones_khd).0,
                        OwnedStreamedLayer::Full(l) => gqa_layer_forward_lora(g, &self.ops, &self.ids, cfg, &self.lora, cursor, l, &xres_buf, padded_t, &cos, &sin).0,
                    };
                    xres_buf = out;
                }

                let hidden_all = g.read(&xres_buf, (padded_t as usize) * d);
                let last = (t - 1) as usize;
                let last_row = &hidden_all[last * d..(last + 1) * d];

                let x = g.storage_init("stream_train.gen.row", last_row);
                let normed = g.storage(d as u64);
                g.submit(&[], &[rmsnorm_fwd(g, &self.ids.kernels, &x, &self.final_norm, &normed, d as u32, 1, cfg.rms_eps)]);
                let vocab = cfg.vocab;
                let logits_buf = g.storage(vocab as u64);
                {
                    let mut s = Vec::new();
                    let act = self.ops.act(&mut s, &normed, 0, 1, d as u32);
                    self.ops.matmul(&mut s, &self.head, &act, &logits_buf, 0);
                    g.submit(&[], &s);
                }
                let logits = g.read(&logits_buf, vocab as usize);
                logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0 as u32
            });

            if eos.contains(&next) {
                break;
            }
            generated.push(next);
            ids.push(next);
        }

        Ok(tok.decode(&generated))
    }

    /// A ready-made [`Self::step`]/[`Self::forward_backward`] `loader` that
    /// builds each layer via [`build_layer_f32`] against an already-resident
    /// host weight map - the tiny-scale equivalence gate's own loader (no
    /// disk I/O).
    pub fn synthetic_loader<'a>(&'a self, cfg: &'a Qwen35Config, init: &'a HashMap<String, Vec<f32>>) -> impl Fn(usize) -> OwnedStreamedLayer + Sync + 'a {
        self.synthetic_loader_at(cfg, init, Dtype::F32)
    }

    /// [`Self::synthetic_loader`] with every matrix of the layer resident at `tier` (quantized from the
    /// host map): the trainer then runs forward and backward through that tier packed.
    pub fn synthetic_loader_at<'a>(&'a self, cfg: &'a Qwen35Config, init: &'a HashMap<String, Vec<f32>>, tier: Dtype) -> impl Fn(usize) -> OwnedStreamedLayer + Sync + 'a {
        let (ops, gpu) = (&self.ops, &self.gpu);
        move |l| build_layer_host(ops, gpu, cfg, l, init, tier)
    }

    /// The loader of a real checkpoint directory: every layer's matrices go to the device in the encoding the
    /// checkpoint stores them in (FP8 stays FP8 with its block scales), one tile of host staging at a time.
    /// The FP8 block size is the checkpoint's own `weight_block_size`.
    pub fn real_loader<'a>(&'a self, cfg: &'a Qwen35Config, dir: &'a Path) -> Result<impl Fn(usize) -> OwnedStreamedLayer + Sync + 'a, String> {
        self.real_loader_as(cfg, dir, None)
    }

    /// [`Self::real_loader`] converting every layer matrix to `convert` (an explicit tiled requantization such as
    /// FP8 to NF4) instead of keeping the stored encoding.
    pub fn real_loader_as<'a>(&'a self, cfg: &'a Qwen35Config, dir: &'a Path, convert: Option<Dtype>) -> Result<impl Fn(usize) -> OwnedStreamedLayer + Sync + 'a, String> {
        let mut base = crate::fp8_source::PackedBase::open(dir, cfg, crate::fp8_source::hf_fp8_block_size(dir)?)?.with_budget(self.budget.clone());
        if let Some(tier) = convert {
            base = base.converting_to(tier);
        }
        let (ops, gpu) = (&self.ops, &self.gpu);
        Ok(move |l| load_layer_packed(ops, gpu, &base, cfg, l))
    }

    pub fn step<F: Fn(usize) -> OwnedStreamedLayer + Sync>(&self, cfg: &Qwen35Config, loader: F, x0: &[f32], targets: &[u32], lr: f32, adam_t: u32) -> f32 {
        self.lora.zero_grads(&self.gpu);
        let loss = self.forward_backward(cfg, loader, x0, targets);
        self.lora.adamw_step(&self.gpu, adam_t, lr);
        self.gpu.poll_wait();
        loss
    }

    /// Run `f(layer, weights)` over `order`, loading each layer on producer threads while the previous one
    /// computes ([`model::layer_stream::for_each_prefetched`]): the packed read and device upload of the next
    /// layers overlap layer `i`'s kernels. At most `window_budget` layers are resident at once (the one `f`
    /// holds, the ones loaded ahead, the ones being loaded) and `window_budget - 1` threads load, so a pass
    /// bound by the host's read rate scales with the window. The device is synchronised before a finished
    /// layer is dropped, so no queued kernel still reads it.
    fn for_each_streamed<L: Fn(usize) -> OwnedStreamedLayer + Sync>(&self, order: Vec<usize>, loader: &L, mut f: impl FnMut(usize, &OwnedStreamedLayer)) {
        let window = self.window_budget.max(2) as usize;
        let stats = model::layer_stream::for_each_prefetched(&order, window - 1, window, loader, |l, layer| {
            let computing = std::time::Instant::now();
            f(l, layer);
            self.gpu.poll_wait();
            self.computed.set(self.computed.get() + computing.elapsed().as_secs_f64());
        });
        self.waited_for_load.set(self.waited_for_load.get() + stats.waited.as_secs_f64());
    }

    /// Forward + backward only (no gradient zeroing, no optimizer step) -
    /// split out from [`Self::step`] so a caller (the tiny-scale equivalence
    /// gate) can read the LoRA adapters' own gradients between backward and
    /// the AdamW step, the same window `Qwen35::backward` leaves open to its
    /// own caller. `loader(l)` must produce layer `l`'s streamed weights
    /// (via [`build_layer_f32`] against a synthetic host map for the
    /// tiny-scale gate, or [`load_layer_f32`] against a real checkpoint
    /// directory) - called ONCE per layer per PASS (twice per layer total:
    /// once forward, once backward - see this module's own doc). `x0` is
    /// the (already-embedded) input residual, `[n, d_model]` row-major;
    /// `targets` (`[n]`, `model::IGNORE` for a masked position) mirrors
    /// `Qwen35::set_batch`'s own target convention. Returns the mean
    /// cross-entropy loss. A caller MUST call [`LoraStore::zero_grads`]
    /// itself before this if it wants a fresh (not accumulated-on-top-of-
    /// the-previous-call's) gradient - [`Self::step`] does this for you.
    pub fn forward_backward<F: Fn(usize) -> OwnedStreamedLayer + Sync>(&self, cfg: &Qwen35Config, loader: F, x0: &[f32], targets: &[u32]) -> f32 {
        let g = &self.gpu;
        let n = self.n;
        let d = cfg.d_model;
        let n_layers = cfg.n_layers;
        assert_eq!(x0.len(), (n * d) as usize, "stream_train::forward_backward: x0 length mismatch");
        assert_eq!(targets.len(), n as usize, "stream_train::forward_backward: targets length mismatch");

        // ---- FORWARD: stream all layers ascending, caching the residual at
        // every layer boundary host-side (this module's own doc, "recompute"
        // design). ----
        let mut xres_cache: Vec<Vec<f32>> = Vec::with_capacity(n_layers as usize + 1);
        xres_cache.push(x0.to_vec());
        let mut xres_buf = g.storage_init("stream_train.x0", x0);

        self.for_each_streamed((0..n_layers as usize).collect(), &loader, |cursor, layer| {
            // The scratch of a layer is the same shape sequence in every layer of its kind: replay one arena
            // per kind instead of allocating and freeing ~30 buffers a layer. The layer is drained before the
            // next scope opens (`for_each_streamed` polls), and the output the next layer reads is held by
            // `xres_buf`, so the arena allocates a fresh slot for it.
            let _scratch = g.scratch_scope_in(layer.scratch_arena(false));
            let out = match layer {
                OwnedStreamedLayer::Linear(l) => gdn_layer_forward_lora(g, &self.ops, &self.ids, cfg, &self.lora, cursor, l, &xres_buf, n, &self.ones_khd).0,
                OwnedStreamedLayer::Full(l) => gqa_layer_forward_lora(g, &self.ops, &self.ids, cfg, &self.lora, cursor, l, &xres_buf, n, &self.cos, &self.sin).0,
            };
            xres_cache.push(g.read(&out, (n * d) as usize));
            xres_buf = out;
        });

        // ---- head forward + loss (resident, no re-streaming - read/
        // differentiated once per step, not once per layer). ----
        let xn_final = g.storage((n * d) as u64);
        g.submit(&[], &[rmsnorm_fwd(g, &self.ids.kernels, &xres_buf, &self.final_norm, &xn_final, d, n, cfg.rms_eps)]);
        // The shared chunked head cross-entropy (`model::head_ce`): the logits of a chunk of rows at a time, the loss
        // and its gradient from the softmax statistics, never the whole `[n, vocab]` matrix and no per-element row rescan.
        let count = targets.iter().filter(|&&t| t != model::IGNORE).count().max(1) as f32;
        let loss = (self.head_ce.forward(&self.ops, &xn_final, n, targets, None, 1.0).loss_sum / count as f64) as f32;
        let d_xn_final = g.storage((n * d) as u64);
        self.head_ce.backward(&self.ops, &xn_final, &d_xn_final, n, targets, None, 1.0 / count, 1.0, None);

        let mut d_res_next = g.storage((n * d) as u64);
        {
            let mut s = Vec::new();
            s.extend(rmsnorm_bwd_frozen(g, &self.ids, &xres_buf, &self.final_norm, &d_xn_final, &d_res_next, d, n, cfg.rms_eps));
            g.submit(&[], &s);
        }

        // ---- BACKWARD: re-stream all layers in reverse, recomputing each
        // one's own forward internals from the cached residual. ----
        self.for_each_streamed((0..n_layers as usize).rev().collect(), &loader, |layer_idx, layer| {
            let _scratch = g.scratch_scope_in(layer.scratch_arena(true));
            let xres_l = g.storage_init("stream_train.xres_l", &xres_cache[layer_idx]);
            let d_res_l = match layer {
                OwnedStreamedLayer::Linear(l) => gdn_layer_backward_lora(g, &self.ops, &self.ids, cfg, &self.lora, layer_idx, l, &xres_l, &d_res_next, n, &self.ones_khd),
                OwnedStreamedLayer::Full(l) => gqa_layer_backward_lora(g, &self.ops, &self.ids, cfg, &self.lora, layer_idx, l, &xres_l, &d_res_next, n, &self.cos, &self.sin),
            };
            d_res_next = d_res_l;
        });

        g.poll_wait();
        loss
    }
}

impl OwnedStreamedLayer {
    /// The scratch arena a layer's forward (or backward) draws its buffers from: one per layer kind and
    /// direction, because the buffer sequence is identical within a kind and different between kinds.
    fn scratch_arena(&self, backward: bool) -> &'static str {
        match (self, backward) {
            (OwnedStreamedLayer::Linear(_), false) => "stream_train.gdn.fwd",
            (OwnedStreamedLayer::Linear(_), true) => "stream_train.gdn.bwd",
            (OwnedStreamedLayer::Full(_), false) => "stream_train.gqa.fwd",
            (OwnedStreamedLayer::Full(_), true) => "stream_train.gqa.bwd",
        }
    }
}

#[cfg(test)]
mod preflight_tests {
    use super::*;

    /// The streamed trainer asks the planner before it loads anything: a card too small is refused with the
    /// per-category arithmetic, and a card that holds the footprint is accepted.
    #[test]
    fn the_preflight_refuses_a_card_that_cannot_hold_the_resident_footprint() {
        let cfg = Qwen35Config::tiny_i8();
        let init = crate::init::init_weights(&cfg, 11);
        let demands = StreamTrainer::footprint_demands(&cfg, 2, &init);
        let need = model::memplan::device_total(&demands);
        assert!(need > 0);
        assert!(StreamTrainer::plan_before_load(&cfg, 2, &init, need).is_ok(), "a card of exactly the footprint holds it");
        let err = StreamTrainer::plan_before_load(&cfg, 2, &init, need - 1).unwrap_err();
        assert!(err.contains("refusing to load") && err.contains("Weights: frozen base"), "{err}");
    }
}
