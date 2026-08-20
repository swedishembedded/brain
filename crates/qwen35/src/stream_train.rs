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
//! # Backward through a streamed weight: the real API available (see the
//! milestone's own "Step 0" research)
//!
//! `model::ops::Ops::matmul_dx`/`Ops::matmul_dw` (the `dX`/`dW` backward-of-
//! matmul dispatch pair, B10) exist, but are deliberately scoped to `F32`/
//! `BF16` weights only - `Ops::matmul_dx`'s own doc: "F16/I8/Q4 backward-
//! through-the-weight is a real, reachable follow-up, not attempted here".
//! Backward through an INT8-quantized weight is not wired up anywhere in
//! this tree (`Ops`, `model::dispatch`, or any model crate). Extending that
//! kernel family to int8 would be genuinely new kernel work, out of
//! proportion for this module.
//!
//! This module avoids the problem structurally instead of extending that
//! kernel family: [`crate::import::import_layer`] already dequantizes every
//! FP8 tensor to f32 in host RAM BEFORE [`crate::stream::StreamState::
//! build_layer`] optionally packs it down to int8 for the (inference-only)
//! streaming forward - the f32 values already exist as a byproduct of every
//! streamed layer load, at zero extra host compute. This module's own layer
//! loader ([`build_layer_f32`]) simply skips that packing step and uploads
//! the SAME already-dequantized f32 values as `model::ops::Weight::F32`
//! instead - trading a ~4x larger streamed-weight footprint (fp32 vs the
//! inference path's int8) for exact correctness through a kernel family
//! (`matmul_dx`/`matmul_dw`, plus plain `matmul` for the tiny LoRA-internal
//! math) that is already proven and gradient-checked
//! (`crates/qwen35/src/model.rs::proj_bwd`'s own LoRA branch dispatches
//! these exact kernels by name, gated by `gradcheck::check_qwen35_lora`).
//! The frozen base's own dX/dW dispatch here is a direct kernel-name
//! transcription of that LoRA branch (not the `Ops` façade, which offers no
//! benefit once the weight is already known to be plain f32) - see
//! [`proj_bwd_streamed`].
//!
//! The one place this module DOES need a resident (never re-streamed)
//! weight at more than int8 precision is the shared `lm_head`/`embed_tokens`
//! table: like `crate::stream::generate`, this module keeps it resident for
//! the whole call (never re-streamed per layer - it is read/differentiated
//! once per step, not once per layer), but at `Weight::F32` or `Weight::BF16`
//! (never int8, for the same `matmul_dx` reason above) - see
//! [`StreamTrainer::new_real`]'s own doc for the memory tradeoff.
//!
//! # What is reused, unchanged
//!
//! - [`crate::import::import_layer`] / [`crate::stream::get`] - the proven
//!   per-shard host loader (M10) and its accessor.
//! - `model::gdn_mixer::{gdn_mixer_fwd, gdn_mixer_bwd}` / `model::gqa_mixer::
//!   {gqa_mixer_fwd, gqa_mixer_bwd}` - the SAME hoisted mixer math
//!   `crate::model::Qwen35`'s own resident trainer and `crate::stream`'s own
//!   inference forward both already drive.
//! - `model::ops::{Ops, Weight}` for the FORWARD dispatch of every streamed
//!   linear (frozen base) - exactly `crate::stream`'s own convention, just
//!   requesting `Dtype::F32` instead of `Dtype::I8`.
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

use checkpoint::mmap::MmapSafetensors;
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use gpu_core::select::Dtype;
use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::block::{rmsnorm_bwd, rmsnorm_fwd, swiglu_bwd, swiglu_fwd, KernelIds};
use model::gdn::{gdn_chunk_size, GdnShape};
use model::gdn_mixer::{gdn_mixer_bwd, gdn_mixer_fwd, GdnMixerActs, GdnMixerGrads, GdnMixerIds, GdnMixerShape, GdnMixerWeights};
use model::gqa_mixer::{gqa_mixer_bwd, gqa_mixer_fwd, GqaMixerActs, GqaMixerGrads, GqaMixerIds, GqaMixerShape, GqaMixerWeights};
use model::ops::{Ops, Weight};
use optim::Optim;
use paramstore::ParamStore;

use crate::config::Qwen35Config;
use crate::import::import_layer;
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
    add2: usize,
    matmul: usize,
    matmul_dx: usize,
    matmul_dw: usize,
    axpy: usize,
    grad_scale: usize,
    ce_value: usize,
    ce_grad: usize,
}

fn train_ids(g: &Gpu) -> TrainIds {
    TrainIds {
        kernels: kernel_ids(g),
        gdn_mixer: gdn_mixer_ids(g),
        gqa_mixer: gqa_mixer_ids(g),
        add2: idx(g, "add2"),
        matmul: idx(g, "matmul"),
        matmul_dx: idx(g, "matmul_dx"),
        matmul_dw: idx(g, "matmul_dw"),
        axpy: idx(g, "axpy"),
        grad_scale: idx(g, "grad_scale"),
        ce_value: idx(g, "ce_value"),
        ce_grad: idx(g, "ce_grad"),
    }
}

/// The raw device buffer under an `F32`-tier [`Weight`] - every streamed
/// frozen-base weight in this module is always this tier (see this module's
/// own doc for why); any other tier reaching here would be a real bug in
/// this module's own layer loader, not a normal runtime condition.
fn f32_buf(w: &Weight) -> &DeviceBuffer {
    match w {
        Weight::F32 { w, .. } => w,
        other => panic!(
            "stream_train: expected an F32 streamed weight (this module's frozen-base backward only \
             supports F32 - matmul_dx has no int8 backward, see this module's own doc), got {:?}",
            other.dtype()
        ),
    }
}

/// Build one decoder layer's weights as [`OwnedStreamedLayer`], every
/// quantizable leaf uploaded as `Weight::F32` (not the int8 DP4A tier
/// `crate::stream::StreamState::build_layer` uses for inference) - see this
/// module's own doc for why training needs the f32 tier specifically.
/// Otherwise field-for-field identical to `build_layer` (same naming, same
/// non-quantized aux tensors as plain fp32 buffers).
pub(crate) fn build_layer_f32(ops: &Ops, gpu: &Gpu, cfg: &Qwen35Config, l: usize, w: &HashMap<String, Vec<f32>>) -> OwnedStreamedLayer {
    use crate::config::LayerType;
    let ty = cfg.layer_types()[l];
    let p = |s: &str| format!("blocks.{l}.{s}");
    let f32buf = |name: &str| gpu.storage_init(name, get(w, name, l));
    let f32w = |name: &str, n: usize, k: usize| Weight::upload(ops, get(w, name, l), n, k, Dtype::F32);
    let d = cfg.d_model as usize;
    let ff = cfg.intermediate_size as usize;
    let (mlp_gate, mlp_up, mlp_down) =
        (f32w(&p("mlp.gate.weight"), ff, d), f32w(&p("mlp.up.weight"), ff, d), f32w(&p("mlp.down.weight"), d, ff));

    match ty {
        LayerType::Linear => {
            let conv_dim = cfg.linear_conv_dim() as usize;
            let value_dim = cfg.linear_value_dim() as usize;
            let nvh = cfg.linear_num_value_heads as usize;
            OwnedStreamedLayer::Linear(OwnedGdnLayer {
                ln1: f32buf(&p("ln1.weight")),
                ln2: f32buf(&p("ln2.weight")),
                conv1d_weight: f32buf(&p("linear_attn.conv1d.weight")),
                a_log: f32buf(&p("linear_attn.A_log")),
                dt_bias: f32buf(&p("linear_attn.dt_bias")),
                norm_weight: f32buf(&p("linear_attn.norm.weight")),
                in_proj_qkv: f32w(&p("linear_attn.in_proj_qkv.weight"), conv_dim, d),
                in_proj_z: f32w(&p("linear_attn.in_proj_z.weight"), value_dim, d),
                in_proj_b: f32w(&p("linear_attn.in_proj_b.weight"), nvh, d),
                in_proj_a: f32w(&p("linear_attn.in_proj_a.weight"), nvh, d),
                out_proj: f32w(&p("linear_attn.out_proj.weight"), d, value_dim),
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
                ln1: f32buf(&p("ln1.weight")),
                ln2: f32buf(&p("ln2.weight")),
                q_norm: f32buf(&p("self_attn.q_norm.weight")),
                k_norm: f32buf(&p("self_attn.k_norm.weight")),
                q_proj: f32w(&p("self_attn.q_proj.weight"), hqp, d),
                k_proj: f32w(&p("self_attn.k_proj.weight"), hkv, d),
                v_proj: f32w(&p("self_attn.v_proj.weight"), hkv, d),
                o_proj: f32w(&p("self_attn.o_proj.weight"), d, hq),
                mlp_gate,
                mlp_up,
                mlp_down,
            })
        }
    }
}

/// [`build_layer_f32`]'s disk-backed half - streams `dir/layers-{l}.safetensors`
/// via [`import_layer`] (unchanged, M10) then uploads at f32. The training
/// counterpart of `crate::stream::StreamState::load_layer`.
pub(crate) fn load_layer_f32(ops: &Ops, gpu: &Gpu, dir: &Path, cfg: &Qwen35Config, l: usize) -> OwnedStreamedLayer {
    let shard = dir.join(format!("layers-{l}.safetensors"));
    let reader = MmapSafetensors::open(&shard).unwrap_or_else(|e| panic!("stream_train: open {}: {e}", shard.display()));
    let w = import_layer(&reader, cfg, l, 128).unwrap_or_else(|e| panic!("stream_train: import_layer({l}): {e}"));
    drop(reader);
    build_layer_f32(ops, gpu, cfg, l, &w)
}

/// Build a resident `Weight::F32` for `name` (`[n, k]`, plain BF16 on disk -
/// `lm_head.weight`/`model.language_model.embed_tokens.weight`) straight
/// from the mmap, in row-chunks, WITHOUT ever holding the whole dequantized
/// `[n, k]` f32 array in host RAM at once - the same bounded-host-memory
/// design `crate::stream::quantize_i8_from_mmap_rows` already established
/// for its own (int8) resident head, minus the quantization step (this
/// module's frozen-base backward needs F32/BF16, never int8 - see this
/// module's own doc). At the real checkpoint's `[248320, 5120]` shape this
/// is ~4.74 GiB resident (vs int8's ~1.18 GiB) - the real memory cost of
/// training through this head at all, paid once per run, never per layer.
fn build_head_f32_from_mmap(gpu: &Gpu, reader: &MmapSafetensors, name: &str, n: usize, k: usize, rows_per_chunk: usize) -> Weight {
    let w = gpu.storage((n * k) as u64);
    let mut any = false;
    let found = reader.with_tensor_chunks(name, rows_per_chunk * k, &mut |off, chunk| {
        any = true;
        assert_eq!(off as usize % k, 0, "build_head_f32_from_mmap: chunk offset {off} is not row-aligned (k={k})");
        gpu.write_f32_at(&w, off, chunk);
    });
    assert!(found && any, "build_head_f32_from_mmap: {name} not found or empty");
    Weight::F32 { w, n: n as u32, k: k as u32 }
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
    pub fn new(gpu: &Gpu, cfg: &Qwen35Config, init: &HashMap<String, Vec<f32>>) -> LoraStore {
        assert!(cfg.lora.is_some(), "stream_train::LoraStore::new: cfg.lora must be set");
        let lora_init: HashMap<String, Vec<f32>> = init.iter().filter(|(k, _)| k.ends_with(".lora_a") || k.ends_with(".lora_b")).map(|(k, v)| (k.clone(), v.clone())).collect();
        let expect_n = cfg.param_list().iter().filter(|(k, _)| k.ends_with(".lora_a") || k.ends_with(".lora_b")).count();
        assert_eq!(lora_init.len(), expect_n, "stream_train::LoraStore::new: init map is missing some of cfg.param_list()'s .lora_a/.lora_b tensors");
        let params: Vec<(String, usize)> = lora_init.iter().map(|(k, v)| (k.clone(), v.len())).collect();
        let ps = ParamStore::new(gpu, params, &lora_init);
        let opt = Optim::new(idx(gpu, "adamw"), idx(gpu, "gradnorm_sq"), idx(gpu, "grad_scale"), idx(gpu, "clip_coef"), idx(gpu, "grad_scale_buf"));
        LoraStore { ps, opt }
    }

    /// `Some((rank, alpha/rank))` if `leaf` is LoRA-targeted under `cfg`;
    /// `None` otherwise. Mirrors `Qwen35::lora_for` exactly.
    fn lora_for(cfg: &Qwen35Config, leaf: &str) -> Option<(u32, f32)> {
        cfg.lora.as_ref().filter(|lc| lc.targets_leaf(leaf)).map(|lc| (lc.rank, lc.alpha / lc.rank as f32))
    }

    pub fn adamw_step(&self, gpu: &Gpu, t: u32, lr: f32) {
        self.opt.step(gpu, &self.ps, t, lr, 0.0, 0.9, 0.999, 1e-8, None, 1.0);
    }

    pub fn zero_grads(&self, gpu: &Gpu) {
        self.ps.zero_grads(gpu);
    }
}

/// Forward LoRA delta for a targeted linear: `y += (alpha/r)*(x*A^t)*B^t` -
/// a direct kernel-dispatch transcription of `Qwen35::lora_fwd`, reading the
/// adapter weights from a [`LoraStore`] instead of `self.w`.
#[allow(clippy::too_many_arguments)]
fn lora_fwd_streamed(gpu: &Gpu, ids: &TrainIds, lora: &ParamStore, wname: &str, r: u32, scale: f32, x: &DeviceBuffer, y: &DeviceBuffer, m: u32, k: u32, nout: u32, s: &mut Vec<Step>) {
    let a = format!("{wname}.lora_a");
    let bnm = format!("{wname}.lora_b");
    let lora_a_buf = gpu.storage((m * r) as u64);
    let lora_out_buf = gpu.storage((m * nout) as u64);
    s.push(gpu.step(ids.matmul, &[x, lora.w(&a), &lora_a_buf], &[m, k, r], m * r));
    s.push(gpu.step(ids.matmul, &[&lora_a_buf, lora.w(&bnm), &lora_out_buf], &[m, r, nout], m * nout));
    s.push(gpu.step(ids.axpy, &[y, &lora_out_buf], &[m * nout, f(scale)], m * nout));
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
fn proj_bwd_streamed(gpu: &Gpu, ids: &TrainIds, lora: &LoraStore, cfg: &Qwen35Config, leaf: &str, d_out: &DeviceBuffer, x: &DeviceBuffer, base: &Weight, wname: &str, dx: &DeviceBuffer, m: u32, k: u32, nout: u32, acc: u32, s: &mut Vec<Step>) {
    let base_buf = f32_buf(base);
    match LoraStore::lora_for(cfg, leaf) {
        Some((r, scale)) => {
            // base: dx += d_out*W (frozen weight - no dW).
            s.push(gpu.step(ids.matmul_dx, &[d_out, base_buf, dx], &[m, k, nout, acc], m * k));

            let a = format!("{wname}.lora_a");
            let bnm = format!("{wname}.lora_b");
            // a = (alpha/r)*(x*A^t)  -> gB += d_out^t*a
            let lora_a_buf = gpu.storage((m * r) as u64);
            s.push(gpu.step(ids.matmul, &[x, lora.ps.w(&a), &lora_a_buf], &[m, k, r], m * r));
            s.push(gpu.step(ids.grad_scale, &[&lora_a_buf], &[m * r, f(scale)], m * r));
            s.push(gpu.step(ids.matmul_dw, &[d_out, &lora_a_buf, lora.ps.g(&bnm)], &[m, r, nout], nout * r));
            // da = (alpha/r)*(d_out*B) -> gA += da^t*x ; dx += da*A
            let lora_da_buf = gpu.storage((m * r) as u64);
            s.push(gpu.step(ids.matmul_dx, &[d_out, lora.ps.w(&bnm), &lora_da_buf], &[m, r, nout, 0], m * r));
            s.push(gpu.step(ids.grad_scale, &[&lora_da_buf], &[m * r, f(scale)], m * r));
            s.push(gpu.step(ids.matmul_dw, &[&lora_da_buf, x, lora.ps.g(&a)], &[m, k, r], r * k));
            s.push(gpu.step(ids.matmul_dx, &[&lora_da_buf, lora.ps.w(&a), dx], &[m, k, r, 1], m * k));
        }
        None => {
            // untargeted leaf, or the shared head: base is always frozen in
            // this module, so dx only, never dW.
            s.push(gpu.step(ids.matmul_dx, &[d_out, base_buf, dx], &[m, k, nout, acc], m * k));
        }
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
        let act1 = ops.act(&mut s, xn2, 0, n, d);
        ops.matmul(&mut s, gate, &act1, &gate_pre, 0);
        ops.matmul(&mut s, up_w, &act1, &up, 0);
        gpu.submit(&[], &s);
    }
    if let Some((r, scale)) = LoraStore::lora_for(cfg, "gate") {
        let mut s = Vec::new();
        lora_fwd_streamed(gpu, ids, &lora.ps, &p("gate.weight"), r, scale, xn2, &gate_pre, n, d, ff, &mut s);
        gpu.submit(&[], &s);
    }
    if let Some((r, scale)) = LoraStore::lora_for(cfg, "up") {
        let mut s = Vec::new();
        lora_fwd_streamed(gpu, ids, &lora.ps, &p("up.weight"), r, scale, xn2, &up, n, d, ff, &mut s);
        gpu.submit(&[], &s);
    }

    let h = gpu.storage((n * ff) as u64);
    gpu.submit(&[], &[swiglu_fwd(gpu, &ids.kernels, &gate_pre, &up, &h, n * ff)]);

    let down = gpu.storage((n * d) as u64);
    {
        let mut s = Vec::new();
        let act2 = ops.act(&mut s, &h, 0, n, ff);
        ops.matmul(&mut s, down_w, &act2, &down, 0);
        gpu.submit(&[], &s);
    }
    if let Some((r, scale)) = LoraStore::lora_for(cfg, "down") {
        let mut s = Vec::new();
        lora_fwd_streamed(gpu, ids, &lora.ps, &p("down.weight"), r, scale, &h, &down, n, ff, d, &mut s);
        gpu.submit(&[], &s);
    }

    (down, MlpActsL { xn2: xn2.clone(), gate_pre, up, h })
}

/// Reverse of [`mlp_forward_lora`] - a direct transcription of `Qwen35::
/// mlp_bwd`.
#[allow(clippy::too_many_arguments)]
fn mlp_backward_lora(gpu: &Gpu, ids: &TrainIds, lora: &LoraStore, cfg: &Qwen35Config, prefix: &str, la: &MlpActsL, gate: &Weight, up_w: &Weight, down_w: &Weight, d_mlp_out: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let d = cfg.d_model;
    let ff = cfg.intermediate_size;
    let p = |s: &str| format!("{prefix}.{s}");

    let d_h = gpu.storage((n * ff) as u64);
    {
        let mut s = Vec::new();
        proj_bwd_streamed(gpu, ids, lora, cfg, "down", d_mlp_out, &la.h, down_w, &p("down.weight"), &d_h, n, ff, d, 0, &mut s);
        gpu.submit(&[], &s);
    }

    let d_gate_pre = gpu.storage((n * ff) as u64);
    let d_up = gpu.storage((n * ff) as u64);
    gpu.submit(&[], &swiglu_bwd(gpu, &ids.kernels, &la.gate_pre, &la.up, &d_h, &d_gate_pre, &d_up, n * ff));

    let d_xn2 = gpu.storage((n * d) as u64);
    {
        let mut s = Vec::new();
        proj_bwd_streamed(gpu, ids, lora, cfg, "up", &d_up, &la.xn2, up_w, &p("up.weight"), &d_xn2, n, d, ff, 0, &mut s);
        proj_bwd_streamed(gpu, ids, lora, cfg, "gate", &d_gate_pre, &la.xn2, gate, &p("gate.weight"), &d_xn2, n, d, ff, 1, &mut s);
        gpu.submit(&[], &s);
    }
    d_xn2
}

/// RMSNorm backward against a frozen (never-trainable) gain - `gw: None`
/// always in this module (no norm is ever a LoRA target). Small helper so
/// call sites don't repeat the throwaway `inv` scratch allocation.
fn rmsnorm_bwd_frozen(gpu: &Gpu, ids: &TrainIds, x: &DeviceBuffer, w: &DeviceBuffer, dy: &DeviceBuffer, dx: &DeviceBuffer, dim: u32, rows: u32) -> Vec<Step> {
    let inv = gpu.storage(rows as u64);
    rmsnorm_bwd(gpu, &ids.kernels, x, w, dy, dx, &inv, None, dim, rows)
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
    gpu.submit(&[], &[rmsnorm_fwd(gpu, &ids.kernels, xres, &layer.ln1, &xn1, d, n)]);

    let conv_dim = cfg.linear_conv_dim();
    let value_dim = cfg.linear_value_dim();
    let nvh = cfg.linear_num_value_heads;
    let khd = cfg.linear_key_head_dim;
    let vhd = cfg.linear_value_head_dim;
    let p = |s: &str| format!("blocks.{layer_idx}.linear_attn.{s}");

    let mixed_qkv = gpu.storage((n * conv_dim) as u64);
    {
        let mut s = Vec::new();
        let act1 = ops.act(&mut s, &xn1, 0, n, d);
        ops.matmul(&mut s, &layer.in_proj_qkv, &act1, &mixed_qkv, 0);
        gpu.submit(&[], &s);
    }
    if let Some((r, scale)) = LoraStore::lora_for(cfg, "in_proj_qkv") {
        let mut s = Vec::new();
        lora_fwd_streamed(gpu, ids, &lora.ps, &p("in_proj_qkv.weight"), r, scale, &xn1, &mixed_qkv, n, d, conv_dim, &mut s);
        gpu.submit(&[], &s);
    }

    let bproj = gpu.storage((n * nvh) as u64);
    let aproj = gpu.storage((n * nvh) as u64);
    let z = gpu.storage((n * value_dim) as u64);
    {
        let mut s = Vec::new();
        let act1 = ops.act(&mut s, &xn1, 0, n, d);
        ops.matmul(&mut s, &layer.in_proj_b, &act1, &bproj, 0);
        ops.matmul(&mut s, &layer.in_proj_a, &act1, &aproj, 0);
        ops.matmul(&mut s, &layer.in_proj_z, &act1, &z, 0);
        gpu.submit(&[], &s);
    }
    if let Some((r, scale)) = LoraStore::lora_for(cfg, "in_proj_b") {
        let mut s = Vec::new();
        lora_fwd_streamed(gpu, ids, &lora.ps, &p("in_proj_b.weight"), r, scale, &xn1, &bproj, n, d, nvh, &mut s);
        gpu.submit(&[], &s);
    }
    if let Some((r, scale)) = LoraStore::lora_for(cfg, "in_proj_a") {
        let mut s = Vec::new();
        lora_fwd_streamed(gpu, ids, &lora.ps, &p("in_proj_a.weight"), r, scale, &xn1, &aproj, n, d, nvh, &mut s);
        gpu.submit(&[], &s);
    }
    if let Some((r, scale)) = LoraStore::lora_for(cfg, "in_proj_z") {
        let mut s = Vec::new();
        lora_fwd_streamed(gpu, ids, &lora.ps, &p("in_proj_z.weight"), r, scale, &xn1, &z, n, d, value_dim, &mut s);
        gpu.submit(&[], &s);
    }

    let shape = GdnMixerShape { gdn: GdnShape { b: 1, h: nvh, t: n, dk: khd, dv: vhd, chunk: gdn_chunk_size(n) }, nkh: cfg.linear_num_key_heads, conv_kernel: cfg.linear_conv_kernel_dim };
    let mix_w = GdnMixerWeights { conv1d_weight: &layer.conv1d_weight, a_log: &layer.a_log, dt_bias: &layer.dt_bias, norm_weight: &layer.norm_weight, ones_khd };
    let (gated, mixer_internals) = gdn_mixer_fwd(gpu, &ids.gdn_mixer, &shape, &mix_w, &mixed_qkv, &bproj, &aproj, &z, n, true);
    let mixer_internals = mixer_internals.expect("stream_train: gdn_mixer_fwd(is_train=true) must return acts");

    let mixer_out = gpu.storage((n * d) as u64);
    {
        let mut s = Vec::new();
        let act3 = ops.act(&mut s, &gated, 0, n, value_dim);
        ops.matmul(&mut s, &layer.out_proj, &act3, &mixer_out, 0);
        gpu.submit(&[], &s);
    }
    if let Some((r, scale)) = LoraStore::lora_for(cfg, "out_proj") {
        let mut s = Vec::new();
        lora_fwd_streamed(gpu, ids, &lora.ps, &p("out_proj.weight"), r, scale, &gated, &mixer_out, n, value_dim, d, &mut s);
        gpu.submit(&[], &s);
    }

    let xmid = gpu.storage((n * d) as u64);
    gpu.submit(&[], &[gpu.step(ids.add2, &[xres, &mixer_out, &xmid], &[n * d], n * d)]);

    let xn2 = gpu.storage((n * d) as u64);
    gpu.submit(&[], &[rmsnorm_fwd(gpu, &ids.kernels, &xmid, &layer.ln2, &xn2, d, n)]);

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
fn gdn_mixer_backward_lora(gpu: &Gpu, ids: &TrainIds, cfg: &Qwen35Config, lora: &LoraStore, layer_idx: usize, layer: &OwnedGdnLayer, xn1: &DeviceBuffer, gated: &DeviceBuffer, mixer_acts: &GdnMixerActs, d_out: &DeviceBuffer, d_xn1: &DeviceBuffer, n: u32, ones_khd: &DeviceBuffer) {
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
        proj_bwd_streamed(gpu, ids, lora, cfg, "out_proj", d_out, gated, &layer.out_proj, &p("out_proj.weight"), &d_gated, n, value_dim, d, 0, &mut s);
        gpu.submit(&[], &s);
    }

    let shape = GdnMixerShape { gdn: GdnShape { b: 1, h: nvh, t: n, dk: khd, dv: vhd, chunk: gdn_chunk_size(n) }, nkh: cfg.linear_num_key_heads, conv_kernel: cfg.linear_conv_kernel_dim };
    let weights = GdnMixerWeights { conv1d_weight: &layer.conv1d_weight, a_log: &layer.a_log, dt_bias: &layer.dt_bias, norm_weight: &layer.norm_weight, ones_khd };
    let grads = GdnMixerGrads { conv1d_weight: None, a_log: None, dt_bias: None, norm_weight: None };
    let (d_mixed_qkv, d_bproj, d_aproj, d_z) = gdn_mixer_bwd(gpu, &ids.gdn_mixer, &shape, &weights, &grads, mixer_acts, &d_gated, n);

    {
        let mut s = Vec::new();
        proj_bwd_streamed(gpu, ids, lora, cfg, "in_proj_b", &d_bproj, xn1, &layer.in_proj_b, &p("in_proj_b.weight"), d_xn1, n, d, nvh, 0, &mut s);
        proj_bwd_streamed(gpu, ids, lora, cfg, "in_proj_a", &d_aproj, xn1, &layer.in_proj_a, &p("in_proj_a.weight"), d_xn1, n, d, nvh, 1, &mut s);
        proj_bwd_streamed(gpu, ids, lora, cfg, "in_proj_z", &d_z, xn1, &layer.in_proj_z, &p("in_proj_z.weight"), d_xn1, n, d, value_dim, 1, &mut s);
        gpu.submit(&[], &s);
    }
    {
        let mut s = Vec::new();
        proj_bwd_streamed(gpu, ids, lora, cfg, "in_proj_qkv", &d_mixed_qkv, xn1, &layer.in_proj_qkv, &p("in_proj_qkv.weight"), d_xn1, n, d, conv_dim, 1, &mut s);
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

    let d_xn2 = mlp_backward_lora(gpu, ids, lora, cfg, &format!("blocks.{layer_idx}.mlp"), &acts.mlp, &layer.mlp_gate, &layer.mlp_up, &layer.mlp_down, d_res_next, n);

    let d_xmid = gpu.storage((n * d) as u64);
    {
        let mut s = Vec::new();
        let d_ln2_dx = gpu.storage((n * d) as u64);
        s.extend(rmsnorm_bwd_frozen(gpu, ids, &acts.xmid, &layer.ln2, &d_xn2, &d_ln2_dx, d, n));
        s.push(gpu.step(ids.add2, &[d_res_next, &d_ln2_dx, &d_xmid], &[n * d], n * d));
        gpu.submit(&[], &s);
    }

    let d_xn1 = gpu.storage((n * d) as u64);
    gdn_mixer_backward_lora(gpu, ids, cfg, lora, layer_idx, layer, &acts.xn1, &acts.gated, &acts.mixer, &d_xmid, &d_xn1, n, ones_khd);

    let d_res_l = gpu.storage((n * d) as u64);
    {
        let mut s = Vec::new();
        let d_ln1_dx = gpu.storage((n * d) as u64);
        s.extend(rmsnorm_bwd_frozen(gpu, ids, xres_l, &layer.ln1, &d_xn1, &d_ln1_dx, d, n));
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
    gpu.submit(&[], &[rmsnorm_fwd(gpu, &ids.kernels, xres, &layer.ln1, &xn1, d, n)]);

    let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    let (qpd, kvd) = (cfg.q_proj_dim(), cfg.kv_dim());
    let p = |s: &str| format!("blocks.{layer_idx}.self_attn.{s}");

    let q_full = gpu.storage((n * qpd) as u64);
    let k = gpu.storage((n * kvd) as u64);
    let v = gpu.storage((n * kvd) as u64);
    {
        let mut s = Vec::new();
        let act1 = ops.act(&mut s, &xn1, 0, n, d);
        ops.matmul(&mut s, &layer.q_proj, &act1, &q_full, 0);
        ops.matmul(&mut s, &layer.k_proj, &act1, &k, 0);
        ops.matmul(&mut s, &layer.v_proj, &act1, &v, 0);
        gpu.submit(&[], &s);
    }
    if let Some((r, scale)) = LoraStore::lora_for(cfg, "q_proj") {
        let mut s = Vec::new();
        lora_fwd_streamed(gpu, ids, &lora.ps, &p("q_proj.weight"), r, scale, &xn1, &q_full, n, d, qpd, &mut s);
        gpu.submit(&[], &s);
    }
    if let Some((r, scale)) = LoraStore::lora_for(cfg, "k_proj") {
        let mut s = Vec::new();
        lora_fwd_streamed(gpu, ids, &lora.ps, &p("k_proj.weight"), r, scale, &xn1, &k, n, d, kvd, &mut s);
        gpu.submit(&[], &s);
    }
    if let Some((r, scale)) = LoraStore::lora_for(cfg, "v_proj") {
        let mut s = Vec::new();
        lora_fwd_streamed(gpu, ids, &lora.ps, &p("v_proj.weight"), r, scale, &xn1, &v, n, d, kvd, &mut s);
        gpu.submit(&[], &s);
    }

    let shape = GqaMixerShape { b: 1, t: n, n_heads: nh, n_kv_heads: nkv, head_dim: hd, rotary_half: cfg.rotary_dim() / 2 };
    let mix_w = GqaMixerWeights { q_norm: &layer.q_norm, k_norm: &layer.k_norm, cos, sin };
    let (ctx_gated, mixer_internals) = gqa_mixer_fwd(gpu, &ids.gqa_mixer, &shape, &mix_w, &q_full, &k, &v, n, true);
    let mixer_internals = mixer_internals.expect("stream_train: gqa_mixer_fwd(is_train=true) must return acts");

    let mixer_out = gpu.storage((n * d) as u64);
    {
        let mut s = Vec::new();
        let act2 = ops.act(&mut s, &ctx_gated, 0, n, shape.qd());
        ops.matmul(&mut s, &layer.o_proj, &act2, &mixer_out, 0);
        gpu.submit(&[], &s);
    }
    if let Some((r, scale)) = LoraStore::lora_for(cfg, "o_proj") {
        let mut s = Vec::new();
        lora_fwd_streamed(gpu, ids, &lora.ps, &p("o_proj.weight"), r, scale, &ctx_gated, &mixer_out, n, shape.qd(), d, &mut s);
        gpu.submit(&[], &s);
    }

    let xmid = gpu.storage((n * d) as u64);
    gpu.submit(&[], &[gpu.step(ids.add2, &[xres, &mixer_out, &xmid], &[n * d], n * d)]);

    let xn2 = gpu.storage((n * d) as u64);
    gpu.submit(&[], &[rmsnorm_fwd(gpu, &ids.kernels, &xmid, &layer.ln2, &xn2, d, n)]);

    let (mlp_out, mlp_acts) = mlp_forward_lora(gpu, ops, ids, cfg, lora, &format!("blocks.{layer_idx}.mlp"), &layer.mlp_gate, &layer.mlp_up, &layer.mlp_down, &xn2, n);

    let out = gpu.storage((n * d) as u64);
    gpu.submit(&[], &[gpu.step(ids.add2, &[&xmid, &mlp_out, &out], &[n * d], n * d)]);

    (out, GqaLayerActsL { xn1, ctx_gated, mixer: mixer_internals, xmid, mlp: mlp_acts })
}

/// Reverse of the GQA mixer (o_proj + `gqa_mixer_bwd` + q/k/v proj) - a
/// direct transcription of `Qwen35::gqa_mixer_bwd`, `q_norm`/`k_norm` always
/// ungraded (never a LoRA target).
#[allow(clippy::too_many_arguments)]
fn gqa_mixer_backward_lora(gpu: &Gpu, ids: &TrainIds, cfg: &Qwen35Config, lora: &LoraStore, layer_idx: usize, layer: &OwnedGqaLayer, xn1: &DeviceBuffer, ctx_gated: &DeviceBuffer, mixer_acts: &GqaMixerActs, d_out: &DeviceBuffer, d_xn1: &DeviceBuffer, n: u32, cos: &DeviceBuffer, sin: &DeviceBuffer) {
    let d = cfg.d_model;
    let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    let (qpd, qd, kvd) = (cfg.q_proj_dim(), cfg.q_dim(), cfg.kv_dim());
    let p = |s: &str| format!("blocks.{layer_idx}.self_attn.{s}");

    let d_ctx_gated = gpu.storage((n * qd) as u64);
    {
        let mut s = Vec::new();
        proj_bwd_streamed(gpu, ids, lora, cfg, "o_proj", d_out, ctx_gated, &layer.o_proj, &p("o_proj.weight"), &d_ctx_gated, n, qd, d, 0, &mut s);
        gpu.submit(&[], &s);
    }

    let shape = GqaMixerShape { b: 1, t: n, n_heads: nh, n_kv_heads: nkv, head_dim: hd, rotary_half: cfg.rotary_dim() / 2 };
    let weights = GqaMixerWeights { q_norm: &layer.q_norm, k_norm: &layer.k_norm, cos, sin };
    let grads = GqaMixerGrads { q_norm: None, k_norm: None };
    let (d_q_full, d_k, d_v) = gqa_mixer_bwd(gpu, &ids.gqa_mixer, &shape, &weights, &grads, mixer_acts, &d_ctx_gated, n);

    {
        let mut s = Vec::new();
        proj_bwd_streamed(gpu, ids, lora, cfg, "q_proj", &d_q_full, xn1, &layer.q_proj, &p("q_proj.weight"), d_xn1, n, d, qpd, 0, &mut s);
        proj_bwd_streamed(gpu, ids, lora, cfg, "k_proj", &d_k, xn1, &layer.k_proj, &p("k_proj.weight"), d_xn1, n, d, kvd, 1, &mut s);
        proj_bwd_streamed(gpu, ids, lora, cfg, "v_proj", &d_v, xn1, &layer.v_proj, &p("v_proj.weight"), d_xn1, n, d, kvd, 1, &mut s);
        gpu.submit(&[], &s);
    }
}

/// One GQA layer backward - the GQA sibling of [`gdn_layer_backward_lora`].
#[allow(clippy::too_many_arguments)]
fn gqa_layer_backward_lora(gpu: &Gpu, ops: &Ops, ids: &TrainIds, cfg: &Qwen35Config, lora: &LoraStore, layer_idx: usize, layer: &OwnedGqaLayer, xres_l: &DeviceBuffer, d_res_next: &DeviceBuffer, n: u32, cos: &DeviceBuffer, sin: &DeviceBuffer) -> DeviceBuffer {
    let d = cfg.d_model;
    let (_out, acts) = gqa_layer_forward_lora(gpu, ops, ids, cfg, lora, layer_idx, layer, xres_l, n, cos, sin);

    let d_xn2 = mlp_backward_lora(gpu, ids, lora, cfg, &format!("blocks.{layer_idx}.mlp"), &acts.mlp, &layer.mlp_gate, &layer.mlp_up, &layer.mlp_down, d_res_next, n);

    let d_xmid = gpu.storage((n * d) as u64);
    {
        let mut s = Vec::new();
        let d_ln2_dx = gpu.storage((n * d) as u64);
        s.extend(rmsnorm_bwd_frozen(gpu, ids, &acts.xmid, &layer.ln2, &d_xn2, &d_ln2_dx, d, n));
        s.push(gpu.step(ids.add2, &[d_res_next, &d_ln2_dx, &d_xmid], &[n * d], n * d));
        gpu.submit(&[], &s);
    }

    let d_xn1 = gpu.storage((n * d) as u64);
    gqa_mixer_backward_lora(gpu, ids, cfg, lora, layer_idx, layer, &acts.xn1, &acts.ctx_gated, &acts.mixer, &d_xmid, &d_xn1, n, cos, sin);

    let d_res_l = gpu.storage((n * d) as u64);
    {
        let mut s = Vec::new();
        let d_ln1_dx = gpu.storage((n * d) as u64);
        s.extend(rmsnorm_bwd_frozen(gpu, ids, xres_l, &layer.ln1, &d_xn1, &d_ln1_dx, d, n));
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
    final_norm: DeviceBuffer,
    n: u32,
    window_budget: u32,
}

impl StreamTrainer {
    /// Build against an already-resident host weight map (`init_weights`'s
    /// own shape) - no disk I/O, no real checkpoint needed. This is the
    /// tiny-scale equivalence gate's own construction path: the SAME
    /// `HashMap` a resident `Qwen35::new_train_on` build reads feeds this
    /// trainer's `lm_head`/`embed_tokens`/`norm.weight` too, so the two
    /// trainers start from byte-identical weights.
    pub fn new_synthetic(gpu: Gpu, cfg: &Qwen35Config, n: u32, window_budget: u32, init: &HashMap<String, Vec<f32>>) -> StreamTrainer {
        let ops = Ops::new(gpu.share()).unwrap_or_else(|e| panic!("stream_train: Ops::new: {e}"));
        let ids = train_ids(&gpu);
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

        let lora = LoraStore::new(&gpu, cfg, init);
        StreamTrainer { gpu, ops, ids, ones_khd, cos, sin, lora, head, final_norm, n, window_budget }
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
    /// and `norm.weight` are streamed ONCE from `dir/outside.safetensors`
    /// and kept resident for the whole run (never re-streamed per layer -
    /// see [`build_head_f32_from_mmap`]'s own doc for the memory cost this
    /// pays, and why it is paid once). The 64 main-stack layers are NOT
    /// loaded here - a caller streams those per step via [`Self::
    /// real_loader`].
    pub fn new_real(gpu: Gpu, cfg: &Qwen35Config, dir: &Path, n: u32, window_budget: u32, lora_init: &HashMap<String, Vec<f32>>) -> Result<StreamTrainer, String> {
        let ops = Ops::new(gpu.share()).unwrap_or_else(|e| panic!("stream_train: Ops::new: {e}"));
        let ids = train_ids(&gpu);
        let d = cfg.d_model as usize;
        let v = cfg.vocab as usize;
        let ones_khd = gpu.storage_init("stream_train.ones_khd", &vec![1.0f32; cfg.linear_key_head_dim as usize]);
        let positions: Vec<[u32; 3]> = (0..n).map(|ti| [ti, ti, ti]).collect();
        let (cos, sin) = qwen3vl::mrope::mrope_tables(&positions, cfg.mrope_section, cfg.rotary_dim(), cfg.rope_theta);
        let cos = gpu.storage_init("stream_train.rope_cos", &cos);
        let sin = gpu.storage_init("stream_train.rope_sin", &sin);

        let outside_path = dir.join("outside.safetensors");
        let outside = MmapSafetensors::open(&outside_path)?;
        let embed_name = "model.language_model.embed_tokens.weight";
        let head_name = if cfg.tie_embeddings { embed_name } else { "lm_head.weight" };
        let head = build_head_f32_from_mmap(&gpu, &outside, head_name, v, d, 4096);
        let norm_raw = crate::stream::read_final_norm(&outside, d)?;
        let final_norm = gpu.storage_init("stream_train.final_norm", &norm_raw);
        drop(outside);

        let lora = LoraStore::new(&gpu, cfg, lora_init);
        Ok(StreamTrainer { gpu, ops, ids, ones_khd, cos, sin, lora, head, final_norm, n, window_budget })
    }

    /// Real per-token embedding rows for `ids`, off `dir/outside.
    /// safetensors`'s `model.language_model.embed_tokens.weight` - the
    /// real-checkpoint counterpart of [`Self::embed_synthetic`], reusing
    /// `crate::stream::embed_rows` (unchanged, M16) directly.
    pub fn embed_real(dir: &Path, ids: &[u32], d: usize) -> Result<Vec<f32>, String> {
        let outside = MmapSafetensors::open(dir.join("outside.safetensors"))?;
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
                    slots[idx] = None;
                    g.read(&xres_buf, 1);
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
            g.submit(&[], &[rmsnorm_fwd(g, &self.ids.kernels, &x, &self.final_norm, &normed, d as u32, 1)]);
            let vocab = cfg.vocab;
            let logits_buf = g.storage(vocab as u64);
            {
                let mut s = Vec::new();
                let act = self.ops.act(&mut s, &normed, 0, 1, d as u32);
                self.ops.matmul(&mut s, &self.head, &act, &logits_buf, 0);
                g.submit(&[], &s);
            }
            let logits = g.read(&logits_buf, vocab as usize);
            let next = logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0 as u32;

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
    pub fn synthetic_loader<'a>(&'a self, cfg: &'a Qwen35Config, init: &'a HashMap<String, Vec<f32>>) -> impl Fn(usize) -> OwnedStreamedLayer + 'a {
        move |l| build_layer_f32(&self.ops, &self.gpu, cfg, l, init)
    }

    /// A ready-made [`Self::step`]/[`Self::forward_backward`] `loader` that
    /// streams each layer from a real checkpoint directory via
    /// [`load_layer_f32`] - the real-checkpoint run's own loader.
    pub fn real_loader<'a>(&'a self, cfg: &'a Qwen35Config, dir: &'a Path) -> impl Fn(usize) -> OwnedStreamedLayer + 'a {
        move |l| load_layer_f32(&self.ops, &self.gpu, dir, cfg, l)
    }

    /// One forward+backward+AdamW step (the normal training-loop entry
    /// point): zeroes the LoRA adapter gradients, runs [`Self::
    /// forward_backward`], then one AdamW step on the resident adapters.
    /// Returns the mean cross-entropy loss. See [`Self::forward_backward`]'s
    /// own doc for `loader`/`x0`/`targets`.
    pub fn step<F: Fn(usize) -> OwnedStreamedLayer>(&self, cfg: &Qwen35Config, loader: F, x0: &[f32], targets: &[u32], lr: f32, adam_t: u32) -> f32 {
        self.lora.zero_grads(&self.gpu);
        let loss = self.forward_backward(cfg, loader, x0, targets);
        self.lora.adamw_step(&self.gpu, adam_t, lr);
        self.gpu.poll_wait();
        loss
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
    pub fn forward_backward<F: Fn(usize) -> OwnedStreamedLayer>(&self, cfg: &Qwen35Config, loader: F, x0: &[f32], targets: &[u32]) -> f32 {
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

        {
            let sched = weightset::Schedule::cyclic(n_layers, 1);
            let mut ws = weightset::WeightSet::build(n_layers, self.window_budget, sched, Box::new(weightset::CyclicScan { lookahead: 1 }))
                .unwrap_or_else(|e| panic!("stream_train: WeightSet::build (forward): {e}"));
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
                    slots[idx] = None;
                    g.read(&xres_buf, 1);
                    slots[idx] = Some(loader(cursor));
                }
                let layer = slots[idx].as_ref().expect("stream_train: WeightSet says this slot is resident");
                let out = match layer {
                    OwnedStreamedLayer::Linear(l) => gdn_layer_forward_lora(g, &self.ops, &self.ids, cfg, &self.lora, cursor, l, &xres_buf, n, &self.ones_khd).0,
                    OwnedStreamedLayer::Full(l) => gqa_layer_forward_lora(g, &self.ops, &self.ids, cfg, &self.lora, cursor, l, &xres_buf, n, &self.cos, &self.sin).0,
                };
                let out_host = g.read(&out, (n * d) as usize);
                xres_cache.push(out_host);
                xres_buf = out;
            }
        }

        // ---- head forward + loss (resident, no re-streaming - read/
        // differentiated once per step, not once per layer). ----
        let v = cfg.vocab;
        let xn_final = g.storage((n * d) as u64);
        g.submit(&[], &[rmsnorm_fwd(g, &self.ids.kernels, &xres_buf, &self.final_norm, &xn_final, d, n)]);
        let logits = g.storage((n * v) as u64);
        {
            let mut s = Vec::new();
            let act = self.ops.act(&mut s, &xn_final, 0, n, d);
            self.ops.matmul(&mut s, &self.head, &act, &logits, 0);
            g.submit(&[], &s);
        }
        let targets_buf = g.storage(n as u64);
        g.write(&targets_buf, targets);
        let ce_buf = g.storage(n as u64);
        g.submit(&[], &[g.step(self.ids.ce_value, &[&logits, &targets_buf, &ce_buf], &[n, v, model::IGNORE], n)]);
        let ce_vals = g.read(&ce_buf, n as usize);
        let count = targets.iter().filter(|&&t| t != model::IGNORE).count().max(1) as f32;
        let loss = ce_vals.iter().sum::<f32>() / count;

        // ---- head backward ----
        let ce_grad_uni = g.uniform_dynamic(4);
        g.write(&ce_grad_uni, &[n, v, model::IGNORE, f(count)]);
        let d_logits = g.storage((n * v) as u64);
        g.submit(&[], &[g.step_buf(self.ids.ce_grad, &ce_grad_uni, &[&logits, &targets_buf, &d_logits], n * v)]);

        let d_xn_final = g.storage((n * d) as u64);
        g.submit(&[], &[g.step(self.ids.matmul_dx, &[&d_logits, f32_buf(&self.head), &d_xn_final], &[n, d, v, 0], n * d)]);

        let mut d_res_next = g.storage((n * d) as u64);
        {
            let mut s = Vec::new();
            s.extend(rmsnorm_bwd_frozen(g, &self.ids, &xres_buf, &self.final_norm, &d_xn_final, &d_res_next, d, n));
            g.submit(&[], &s);
        }

        // ---- BACKWARD: re-stream all layers in reverse, recomputing each
        // one's own forward internals from the cached residual. ----
        {
            let rev_order: Vec<weightset::GroupId> = (0..n_layers).rev().map(weightset::GroupId).collect();
            let sched = weightset::Schedule { order: rev_order.clone() };
            let mut ws = weightset::WeightSet::build(n_layers, self.window_budget, sched, Box::new(weightset::CyclicScan { lookahead: 1 }))
                .unwrap_or_else(|e| panic!("stream_train: WeightSet::build (backward): {e}"));
            let mut slots: Vec<Option<OwnedStreamedLayer>> = (0..self.window_budget).map(|_| None).collect();
            for (i, slot) in ws.slot_contents().iter().enumerate() {
                if let Some(gid) = slot {
                    slots[i] = Some(loader(gid.0 as usize));
                }
            }
            for (cursor, gid) in rev_order.iter().enumerate() {
                let layer_idx = gid.0 as usize;
                let (slot_id, miss) = ws.advance(cursor);
                let idx = slot_id.0 as usize;
                if miss {
                    slots[idx] = None;
                    g.read(&d_res_next, 1);
                    slots[idx] = Some(loader(layer_idx));
                }
                let layer = slots[idx].as_ref().expect("stream_train: WeightSet says this slot is resident");
                let xres_l = g.storage_init("stream_train.xres_l", &xres_cache[layer_idx]);
                let d_res_l = match layer {
                    OwnedStreamedLayer::Linear(l) => gdn_layer_backward_lora(g, &self.ops, &self.ids, cfg, &self.lora, layer_idx, l, &xres_l, &d_res_next, n, &self.ones_khd),
                    OwnedStreamedLayer::Full(l) => gqa_layer_backward_lora(g, &self.ops, &self.ids, cfg, &self.lora, layer_idx, l, &xres_l, &d_res_next, n, &self.cos, &self.sin),
                };
                d_res_next = d_res_l;
            }
        }

        g.poll_wait();
        loss
    }
}
