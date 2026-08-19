// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Streaming forward pass over the real 64-layer `Qwen/Qwen3.8-27B-FP8`
//! decoder, holding only a small SLIDING WINDOW of layers' weights resident
//! at once - the piece that lets this model run a real chain of real layer
//! weights on a box with far less RAM than a whole-model build would need
//! (`crate::model::Qwen35::new_*` all require every layer's weights resolved
//! in one host `HashMap` before building anything, which M14's own doc
//! already flags as impossible at this config on a constrained machine).
//!
//! # Scope
//!
//! This module proves the STREAMING PLUMBING itself: real weights, real
//! mixer/MLP math, a real windowed load/evict schedule, bounded host and
//! device memory. It deliberately does NOT wire into CLI/serve, does not run
//! a tokenizer or a real generation loop, and does not touch MTP - those are
//! separate, later concerns. The residual stream this module drives a single
//! forward pass over is either caller-supplied or a small fixed-seed
//! synthetic vector (see [`seed_residual`]) - never the real embedding
//! table, which this module never materializes (see that function's own
//! doc for why).
//!
//! # Weight residency
//!
//! [`weightset::WeightSet`] is pure host-side slot bookkeeping (hit/miss +
//! eviction), never a `Weight` factory or fixed-shape slot buffer pool -
//! `crates/s3dit/src/dev.rs`'s `WindowedPhase` (the other precedent in this
//! tree) can pre-allocate ONE fixed-shape slot buffer set and overwrite it
//! in place on a miss because every one of its blocks has the SAME shape.
//! This model's layers do not: `full_attention_interval = 4` means 3 of
//! every 4 layers are Gated DeltaNet (`LayerType::Linear`, 5 quantizable
//! leaves) and 1 of every 4 is GQA (`LayerType::Full`, 4 quantizable
//! leaves), with different byte sizes per leaf. So each [`WeightSet`] slot
//! here instead holds an `Option<`[`OwnedStreamedLayer`]`>` that is DROPPED
//! and rebuilt fresh on every miss - the same shape `qwen3omnimoe::generate`
//! already uses for its own (homogeneous-enough-in-practice, but still
//! per-call-allocated) `OwnedLayer`.
//!
//! # Two hard-won device-memory lessons this module inherits
//!
//! Both are `qwen3omnimoe::generate`'s own module-doc lessons, and both are
//! load-bearing here too, at a smaller per-layer scale:
//!
//! 1. Dropping a `DeviceBuffer` does not by itself reclaim device memory -
//!    the commands that referenced it must retire first. [`run`] forces
//!    that retirement with one throwaway `gpu.read(still_live_buffer, 1)`
//!    call right after dropping an evicted slot's buffers and before the
//!    next slot's weights are uploaded - any already-live buffer works,
//!    this is purely a submit+wait, never a real read of useful data.
//! 2. A layer's computed output is always a FRESH `gpu.storage` allocation
//!    (never a view into that layer's own weight buffers), so capturing it
//!    into the residual-stream variable before the layer is later dropped
//!    is automatic here, not something a caller has to arrange by hand -
//!    still called out explicitly at the one call site in [`run`] where the
//!    ordering matters.
//!
//! # What is reused, unchanged, from earlier milestones
//!
//! - [`crate::import::import_layer`] - the already-proven, already-measured
//!   (2.37-2.45 GiB peak RSS per layer, M10) per-shard host loader. Nothing
//!   in this module re-decodes a checkpoint tensor by hand.
//! - `model::gdn_mixer::gdn_mixer_fwd` / `model::gqa_mixer::gqa_mixer_fwd` -
//!   the shared mixer math both `crate::model::Qwen35`'s own
//!   `layer_gdn_fwd`/`layer_gqa_fwd` and `tests/real_weight_streaming.rs`
//!   already drive. This module's own `gdn_layer_forward`/`gqa_layer_forward`
//!   below are close copies of those two methods, differing only in reading
//!   weights from a per-layer [`OwnedStreamedLayer`] instead of `self.w`/
//!   `self.weights` (`Qwen35`'s own instance-wide stores, which this module
//!   exists precisely because we cannot afford to build).
//! - `model::ops::{Ops, Weight}` - the same int8 (DP4A) façade `Qwen35::
//!   new_impl_on`'s own `upload` closure drives (M14); the 12 quantizable
//!   leaf names and their `(n, k)` shapes below mirror that closure exactly.
//!
//! `paramstore::upload::Uploader` (the chunked-decode-from-mmap host-RAM
//! bounder `qwen3omnimoe::generate` needs) is deliberately NOT used here -
//! `import_layer` already solves the same problem for this crate's own
//! naming convention, at a measured RSS this module's own gate 2 test
//! re-confirms at full-chain scale.

use std::collections::HashMap;
use std::path::Path;

use checkpoint::mmap::MmapSafetensors;
use gpu_core::select::Dtype;
use gpu_core::{DeviceBuffer, Gpu};
use model::block::{rmsnorm_fwd, swiglu_fwd, KernelIds};
use model::gdn::{GdnBwdIds, GdnIds, GdnShape};
use model::gdn_mixer::{gdn_mixer_fwd, GdnMixerIds, GdnMixerShape, GdnMixerWeights};
use model::gqa_mixer::{gqa_mixer_fwd, GqaMixerIds, GqaMixerShape, GqaMixerWeights};
use model::ops::{Act, Ops, Weight};

use crate::config::{LayerType, Qwen35Config};
use crate::import::import_layer;

/// This model's own [`weightset::ResidencyPlan`] choice for a single
/// streaming forward pass: the fully-known schedule ([`weightset::Schedule::
/// cyclic`], one pass over all 64 layers) makes [`weightset::CyclicScan`]
/// Bélády-optimal, not a heuristic - see that type's own doc. `lookahead: 1`
/// is the minimum rotating reserve a schedule narrower than the model needs
/// at all (a window that already fits every layer, `budget >= n_layers`,
/// never evicts regardless of this number).
const LOOKAHEAD: u32 = 1;

/// One Gated DeltaNet (`LayerType::Linear`) layer's weights: the 4 plain
/// fp32 aux tensors (norms, GDN's own `A_log`/`dt_bias`/gated-norm weight)
/// plus the 5 quantized (int8 DP4A) mixer/MLP-adjacent leaves this layer
/// type owns - `in_proj_{qkv,z,b,a}`/`out_proj` - and the 3 quantized MLP
/// leaves every layer type owns.
pub struct OwnedGdnLayer {
    pub ln1: DeviceBuffer,
    pub ln2: DeviceBuffer,
    pub conv1d_weight: DeviceBuffer,
    pub a_log: DeviceBuffer,
    pub dt_bias: DeviceBuffer,
    pub norm_weight: DeviceBuffer,
    pub in_proj_qkv: Weight,
    pub in_proj_z: Weight,
    pub in_proj_b: Weight,
    pub in_proj_a: Weight,
    pub out_proj: Weight,
    pub mlp_gate: Weight,
    pub mlp_up: Weight,
    pub mlp_down: Weight,
}

/// One GQA (`LayerType::Full`) layer's weights: the 2 plain fp32 aux tensors
/// (norms) plus `q_norm`/`k_norm` plus the 4 quantized attention leaves this
/// layer type owns and the 3 quantized MLP leaves every layer type owns.
pub struct OwnedGqaLayer {
    pub ln1: DeviceBuffer,
    pub ln2: DeviceBuffer,
    pub q_norm: DeviceBuffer,
    pub k_norm: DeviceBuffer,
    pub q_proj: Weight,
    pub k_proj: Weight,
    pub v_proj: Weight,
    pub o_proj: Weight,
    pub mlp_gate: Weight,
    pub mlp_up: Weight,
    pub mlp_down: Weight,
}

/// One decoder layer's streamed weights, whichever shape it turned out to
/// be - a [`weightset::WeightSet`] slot's occupant. Dropped and rebuilt
/// fresh on every miss (see this module's own doc); never mutated in place.
pub enum OwnedStreamedLayer {
    Linear(OwnedGdnLayer),
    Full(OwnedGqaLayer),
}

fn get<'a>(w: &'a HashMap<String, Vec<f32>>, name: &str, l: usize) -> &'a [f32] {
    w.get(name).unwrap_or_else(|| panic!("stream: layer {l}: import_layer did not produce {name}")).as_slice()
}

/// Kernel indices this module's own per-layer forward dispatches, resolved
/// ONCE against a `Gpu` built from [`crate::model::pipelines`] - the same
/// full façade kernel set `Qwen35::new_i8` registers, so both `Ops::new`'s
/// own required-kernel check and every mixer-internal kernel this module's
/// `gdn_layer_forward`/`gqa_layer_forward` dispatch resolve from the same
/// build.
struct StreamIds {
    kernels: KernelIds,
    gdn_mixer: GdnMixerIds,
    gqa_mixer: GqaMixerIds,
    add2: usize,
}

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("stream: kernel '{name}' not registered"))
}

fn kernel_ids(g: &Gpu) -> KernelIds {
    KernelIds {
        rmsnorm: idx(g, "rmsnorm"),
        rms_inv: idx(g, "rms_inv"),
        rmsnorm_dx: idx(g, "rmsnorm_dx"),
        rmsnorm_dw: idx(g, "rmsnorm_dw"),
        rope: idx(g, "rmsnorm"),
        rope_bwd: idx(g, "rmsnorm"),
        gqa_scores: idx(g, "gqa_scores"),
        gqa_apply: idx(g, "gqa_apply"),
        attn_softmax: idx(g, "attn_softmax"),
        gqa_dscores: idx(g, "gqa_bwd_dscores"),
        gqa_dv: idx(g, "gqa_bwd_dv"),
        gqa_dq: idx(g, "gqa_bwd_dq"),
        gqa_dk: idx(g, "gqa_bwd_dk"),
        silu_mul: idx(g, "silu_mul"),
        silu_da: idx(g, "silu_bwd_da"),
        silu_db: idx(g, "silu_bwd_db"),
    }
}

fn gdn_mixer_ids(g: &Gpu) -> GdnMixerIds {
    GdnMixerIds {
        kernels: kernel_ids(g),
        conv: audio::conv::ConvKernels { fwd: idx(g, "conv1d"), dx: idx(g, "conv1d_dx"), dw: idx(g, "conv1d_dw") },
        chunk: GdnIds {
            bmm: idx(g, "bmm"),
            bmm_acc: idx(g, "bmm_acc"),
            cumsum_step: idx(g, "gdn_chunk_cumsum_step"),
            decay_mask: idx(g, "gdn_decay_mask"),
            mask_strict_lower: idx(g, "gdn_mask_strict_lower"),
            ut_step: idx(g, "gdn_ut_step"),
            add_identity: idx(g, "gdn_add_identity"),
            row_scale: idx(g, "scale_row"),
            row_scale_off: idx(g, "gdn_row_scale_off"),
            decay_scale: idx(g, "gdn_decay_scale"),
            state_decay: idx(g, "gdn_state_decay"),
            exp: idx(g, "exp"),
            sub: idx(g, "sub"),
            mul: idx(g, "mul"),
            region_copy: idx(g, "region_copy"),
        },
        chunk_bwd: GdnBwdIds {
            splice_add: idx(g, "splice_add"),
            row_dot: idx(g, "row_dot"),
            scale_add: idx(g, "scale_add"),
            reverse_cumsum_step: idx(g, "gdn_chunk_reverse_cumsum_step"),
            ut_bwd_dattn0: idx(g, "gdn_ut_bwd_dattn0"),
            ut_bwd_dtmat: idx(g, "gdn_ut_bwd_dtmat"),
            mask_strict_lower_bwd: idx(g, "gdn_mask_strict_lower_bwd"),
            decay_mask_bwd: idx(g, "gdn_decay_mask_bwd"),
            decay_scale_bwd: idx(g, "gdn_decay_scale_bwd"),
            decay_scale_bwd_last: idx(g, "gdn_decay_scale_bwd_last"),
            state_decay_bwd_dscale: idx(g, "gdn_state_decay_bwd_dscale"),
        },
        nlc_nchw: idx(g, "nlc_nchw"),
        nchw_nlc: idx(g, "nchw_nlc"),
        silu: idx(g, "silu"),
        silu_bwd: idx(g, "silu_bwd"),
        concat_split: idx(g, "concat_split"),
        concat2: idx(g, "concat2"),
        l2norm_scale: idx(g, "l2norm_scale"),
        l2norm_scale_dx: idx(g, "l2norm_scale_dx"),
        sigmoid: idx(g, "sigmoid"),
        sigmoid_bwd: idx(g, "sigmoid_bwd"),
        gdn_decay_gate: idx(g, "gdn_decay_gate"),
        gdn_decay_gate_bwd: idx(g, "gdn_decay_gate_bwd"),
        kv_expand: idx(g, "kv_expand"),
        kv_expand_bwd: idx(g, "kv_expand_bwd"),
        gdn_layout_permute: idx(g, "gdn_layout_permute"),
        mul: idx(g, "mul"),
        bias_grad: idx(g, "bias_grad"),
    }
}

fn gqa_mixer_ids(g: &Gpu) -> GqaMixerIds {
    GqaMixerIds {
        kernels: kernel_ids(g),
        concat_split: idx(g, "concat_split"),
        concat2: idx(g, "concat2"),
        sigmoid: idx(g, "sigmoid"),
        sigmoid_bwd: idx(g, "sigmoid_bwd"),
        mul: idx(g, "mul"),
        rope2d_partial: idx(g, "rope2d_partial"),
    }
}

/// Everything a streaming forward pass needs that is NOT per-layer: the
/// device handle, the int8 façade, resolved kernel indices, and the two
/// small model-wide buffers every layer of either type reads
/// (`ones_khd` for GDN's query/key L2-norm, `cos`/`sin` M-RoPE tables for
/// GQA) - built once for a fixed row count `n`, reused by every one of the
/// 64 layers a [`run`] call visits.
pub struct StreamState {
    pub gpu: Gpu,
    pub ops: Ops,
    ids: StreamIds,
    ones_khd: DeviceBuffer,
    cos: DeviceBuffer,
    sin: DeviceBuffer,
}

impl StreamState {
    /// Build on a fresh `Gpu` (via [`crate::model::pipelines`] - the full
    /// façade kernel set, same as `Qwen35::new_i8`). `n` is the (fixed, for
    /// the whole run) row count every layer's forward call uses - text-only
    /// sequential positions `0..n`, matching `Qwen35::new_impl_on`'s own
    /// single-sequence M-RoPE table construction.
    pub fn new(gpu: Gpu, cfg: &Qwen35Config, n: u32) -> StreamState {
        let ops = Ops::new(gpu.share()).unwrap_or_else(|e| panic!("stream: Ops::new: {e}"));
        let ids = StreamIds { kernels: kernel_ids(&gpu), gdn_mixer: gdn_mixer_ids(&gpu), gqa_mixer: gqa_mixer_ids(&gpu), add2: idx(&gpu, "add2") };
        let ones_khd = gpu.storage_init("stream.ones_khd", &vec![1.0f32; cfg.linear_key_head_dim as usize]);
        let positions: Vec<[u32; 3]> = (0..n).map(|ti| [ti, ti, ti]).collect();
        let (cos, sin) = qwen3vl::mrope::mrope_tables(&positions, cfg.mrope_section, cfg.rotary_dim(), cfg.rope_theta);
        let cos = gpu.storage_init("stream.rope_cos", &cos);
        let sin = gpu.storage_init("stream.rope_sin", &sin);
        StreamState { gpu, ops, ids, ones_khd, cos, sin }
    }

    /// Stream layer `l`'s own weights from `dir` (one `layers-{l}.safetensors`
    /// shard, via [`import_layer`]) straight into device buffers: the 8-12
    /// quantizable leaves as int8 (DP4A) [`Weight`]s via [`Weight::upload`],
    /// everything else (norms, GDN's `A_log`/`dt_bias`/gated-norm weight,
    /// GQA's `q_norm`/`k_norm`) as plain fp32 `storage_init` buffers - the
    /// same quantizable/non-quantizable split `crate::model::is_i8_linear`
    /// and `Qwen35::new_impl_on`'s own `upload` closure already establish
    /// (M14), just for one layer at a time instead of every layer up front.
    pub fn load_layer(&self, dir: &Path, cfg: &Qwen35Config, l: usize) -> OwnedStreamedLayer {
        let ty = cfg.layer_types()[l];
        let shard = dir.join(format!("layers-{l}.safetensors"));
        let reader = MmapSafetensors::open(&shard).unwrap_or_else(|e| panic!("stream: open {}: {e}", shard.display()));
        let w = import_layer(&reader, cfg, l, 128).unwrap_or_else(|e| panic!("stream: import_layer({l}): {e}"));
        drop(reader);

        let p = |s: &str| format!("blocks.{l}.{s}");
        let f32buf = |name: &str| self.gpu.storage_init(name, get(&w, name, l));
        let i8w = |name: &str, n: usize, k: usize| Weight::upload(&self.ops, get(&w, name, l), n, k, Dtype::I8);
        let d = cfg.d_model as usize;
        let ff = cfg.intermediate_size as usize;
        let (mlp_gate, mlp_up, mlp_down) =
            (i8w(&p("mlp.gate.weight"), ff, d), i8w(&p("mlp.up.weight"), ff, d), i8w(&p("mlp.down.weight"), d, ff));

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
                    in_proj_qkv: i8w(&p("linear_attn.in_proj_qkv.weight"), conv_dim, d),
                    in_proj_z: i8w(&p("linear_attn.in_proj_z.weight"), value_dim, d),
                    in_proj_b: i8w(&p("linear_attn.in_proj_b.weight"), nvh, d),
                    in_proj_a: i8w(&p("linear_attn.in_proj_a.weight"), nvh, d),
                    out_proj: i8w(&p("linear_attn.out_proj.weight"), d, value_dim),
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
                    q_proj: i8w(&p("self_attn.q_proj.weight"), hqp, d),
                    k_proj: i8w(&p("self_attn.k_proj.weight"), hkv, d),
                    v_proj: i8w(&p("self_attn.v_proj.weight"), hkv, d),
                    o_proj: i8w(&p("self_attn.o_proj.weight"), d, hq),
                    mlp_gate,
                    mlp_up,
                    mlp_down,
                })
            }
        }
    }

    /// One full layer forward: `rmsnorm -> mixer -> residual add -> rmsnorm
    /// -> mlp -> residual add` - the exact per-layer shape `crate::model::
    /// Qwen35::run_forward`'s own loop body uses (see this module's own doc),
    /// dispatched here against a streamed [`OwnedStreamedLayer`] instead of
    /// an instance-wide `ParamStore`/weights map. Returns a FRESH residual
    /// buffer (never a view into `layer`'s own weight buffers - see this
    /// module's doc, lesson 2).
    pub fn layer_forward(&self, cfg: &Qwen35Config, layer: &OwnedStreamedLayer, xres: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let d = cfg.d_model;
        let (ln1, ln2) = match layer {
            OwnedStreamedLayer::Linear(l) => (&l.ln1, &l.ln2),
            OwnedStreamedLayer::Full(l) => (&l.ln1, &l.ln2),
        };

        let xn1 = g.storage((n * d) as u64);
        g.submit(&[], &[rmsnorm_fwd(g, &self.ids.kernels, xres, ln1, &xn1, d, n)]);

        let mixer_out = match layer {
            OwnedStreamedLayer::Linear(l) => self.gdn_layer_forward(cfg, l, &xn1, n),
            OwnedStreamedLayer::Full(l) => self.gqa_layer_forward(cfg, l, &xn1, n),
        };

        let xmid = g.storage((n * d) as u64);
        g.submit(&[], &[g.step(self.ids.add2, &[xres, &mixer_out, &xmid], &[n * d], n * d)]);

        let xn2 = g.storage((n * d) as u64);
        g.submit(&[], &[rmsnorm_fwd(g, &self.ids.kernels, &xmid, ln2, &xn2, d, n)]);

        let (mlp_gate, mlp_up, mlp_down) = match layer {
            OwnedStreamedLayer::Linear(l) => (&l.mlp_gate, &l.mlp_up, &l.mlp_down),
            OwnedStreamedLayer::Full(l) => (&l.mlp_gate, &l.mlp_up, &l.mlp_down),
        };
        let mlp_out = self.mlp_forward(cfg, mlp_gate, mlp_up, mlp_down, &xn2, n);

        let out = g.storage((n * d) as u64);
        g.submit(&[], &[g.step(self.ids.add2, &[&xmid, &mlp_out, &out], &[n * d], n * d)]);
        out
    }

    /// One Gated DeltaNet mixer - a close copy of `crate::model::Qwen35::
    /// layer_gdn_fwd`, reading `l`'s streamed weights instead of `self.w`/
    /// `self.weights`, and always dispatching through `Ops::act`+`Ops::
    /// matmul` (this module has no LoRA branch to skip, unlike `ops_linear`).
    fn gdn_layer_forward(&self, cfg: &Qwen35Config, l: &OwnedGdnLayer, xn1: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let ops = &self.ops;
        let d = cfg.d_model;
        let conv_dim = cfg.linear_conv_dim();
        let value_dim = cfg.linear_value_dim();
        let nvh = cfg.linear_num_value_heads;
        let khd = cfg.linear_key_head_dim;
        let vhd = cfg.linear_value_head_dim;

        let mixed_qkv = g.storage((n * conv_dim) as u64);
        let mut s1 = Vec::new();
        let act1: Act = ops.act(&mut s1, xn1, 0, n, d);
        ops.matmul(&mut s1, &l.in_proj_qkv, &act1, &mixed_qkv, 0);
        g.submit(&[], &s1);

        let bproj = g.storage((n * nvh) as u64);
        let aproj = g.storage((n * nvh) as u64);
        let z = g.storage((n * value_dim) as u64);
        let mut s2 = Vec::new();
        ops.matmul(&mut s2, &l.in_proj_b, &act1, &bproj, 0);
        ops.matmul(&mut s2, &l.in_proj_a, &act1, &aproj, 0);
        ops.matmul(&mut s2, &l.in_proj_z, &act1, &z, 0);
        g.submit(&[], &s2);

        let shape = GdnMixerShape {
            gdn: GdnShape { b: 1, h: nvh, t: n, dk: khd, dv: vhd, chunk: model::gdn::gdn_chunk_size(n) },
            nkh: cfg.linear_num_key_heads,
            conv_kernel: cfg.linear_conv_kernel_dim,
        };
        let weights = GdnMixerWeights {
            conv1d_weight: &l.conv1d_weight,
            a_log: &l.a_log,
            dt_bias: &l.dt_bias,
            norm_weight: &l.norm_weight,
            ones_khd: &self.ones_khd,
        };
        let (gated, _acts) = gdn_mixer_fwd(g, &self.ids.gdn_mixer, &shape, &weights, &mixed_qkv, &bproj, &aproj, &z, n, false);

        let out = g.storage((n * d) as u64);
        let mut s3 = Vec::new();
        let act3 = ops.act(&mut s3, &gated, 0, n, value_dim);
        ops.matmul(&mut s3, &l.out_proj, &act3, &out, 0);
        g.submit(&[], &s3);
        out
    }

    /// One GQA mixer - a close copy of `crate::model::Qwen35::layer_gqa_fwd`,
    /// reading `l`'s streamed weights instead of `self.w`/`self.weights`.
    fn gqa_layer_forward(&self, cfg: &Qwen35Config, l: &OwnedGqaLayer, xn1: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let ops = &self.ops;
        let d = cfg.d_model;
        let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let (qpd, kvd) = (cfg.q_proj_dim(), cfg.kv_dim());

        let q_full = g.storage((n * qpd) as u64);
        let k = g.storage((n * kvd) as u64);
        let v = g.storage((n * kvd) as u64);
        let mut s1 = Vec::new();
        let act1 = ops.act(&mut s1, xn1, 0, n, d);
        ops.matmul(&mut s1, &l.q_proj, &act1, &q_full, 0);
        ops.matmul(&mut s1, &l.k_proj, &act1, &k, 0);
        ops.matmul(&mut s1, &l.v_proj, &act1, &v, 0);
        g.submit(&[], &s1);

        let shape = GqaMixerShape { b: 1, t: n, n_heads: nh, n_kv_heads: nkv, head_dim: hd, rotary_half: cfg.rotary_dim() / 2 };
        let weights = GqaMixerWeights { q_norm: &l.q_norm, k_norm: &l.k_norm, cos: &self.cos, sin: &self.sin };
        let (ctx_gated, _acts) = gqa_mixer_fwd(g, &self.ids.gqa_mixer, &shape, &weights, &q_full, &k, &v, n, false);

        let out = g.storage((n * d) as u64);
        let mut s2 = Vec::new();
        let act2 = ops.act(&mut s2, &ctx_gated, 0, n, shape.qd());
        ops.matmul(&mut s2, &l.o_proj, &act2, &out, 0);
        g.submit(&[], &s2);
        out
    }

    /// The universal dense SwiGLU MLP - a close copy of `crate::model::
    /// Qwen35::mlp_fwd`, taking the 3 streamed [`Weight`]s directly (both
    /// layer types own an MLP of the same shape).
    fn mlp_forward(&self, cfg: &Qwen35Config, gate: &Weight, up: &Weight, down: &Weight, xn2: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let ops = &self.ops;
        let d = cfg.d_model;
        let ff = cfg.intermediate_size;

        let gate_pre = g.storage((n * ff) as u64);
        let up_buf = g.storage((n * ff) as u64);
        let mut s1 = Vec::new();
        let act1 = ops.act(&mut s1, xn2, 0, n, d);
        ops.matmul(&mut s1, gate, &act1, &gate_pre, 0);
        ops.matmul(&mut s1, up, &act1, &up_buf, 0);
        g.submit(&[], &s1);

        let h = g.storage((n * ff) as u64);
        g.submit(&[], &[swiglu_fwd(g, &self.ids.kernels, &gate_pre, &up_buf, &h, n * ff)]);

        let down_out = g.storage((n * d) as u64);
        let mut s2 = Vec::new();
        let act2 = ops.act(&mut s2, &h, 0, n, ff);
        ops.matmul(&mut s2, down, &act2, &down_out, 0);
        g.submit(&[], &s2);
        down_out
    }
}

/// A small, fixed-seed synthetic `[n, d_model]` residual stream - this
/// module's stand-in for a real embedding lookup. `Qwen35Config::qwen38_27b`'s
/// `embed_tokens`/`lm_head` are each `[248320, 5120]`, ~5.09 GB (4.74 GiB)
/// dequantized to f32 (`248320 * 5120 * 4` bytes) - materializing even ONE of
/// those tables would burn most of this box's ~11 GiB available RAM before a
/// single layer has streamed, for a value this milestone's gate (streaming
/// LAYER plumbing, not end-to-end generation - see this module's own doc)
/// does not need: a real chain of real layer WEIGHTS still transforms this
/// input exactly as it would transform a real embedded token row. Real
/// embedding/tokenizer/lm_head/sampling is explicitly out of scope here
/// (left to a later milestone).
pub fn seed_residual(n: u32, d_model: u32, seed: u64) -> Vec<f32> {
    let mut rng = data::rng::Lcg::new(seed);
    rng.vec_scaled((n * d_model) as usize, 0.5)
}

/// Run one streaming forward pass over every real layer of `cfg` (in
/// practice always [`Qwen35Config::qwen38_27b`]), reading `dir` on demand
/// (`layers-{l}.safetensors` per layer, `import_layer` per miss), holding at
/// most `window_budget` layers' weights resident on device at once. `n` is
/// the row count of the (synthetic - see [`seed_residual`]) input residual
/// stream; `seed` is its RNG seed.
///
/// Schedule: [`weightset::Schedule::cyclic`]`(cfg.n_layers, 1)` - a SINGLE
/// pass, not a multi-token decode loop (that is later work, see this
/// module's own doc). Eviction: [`weightset::CyclicScan`] with
/// [`LOOKAHEAD`] - Bélády-optimal for this fully-known schedule.
///
/// Returns the final residual, read back host-side, `[n, d_model]` row-major.
pub fn run(dir: &Path, cfg: &Qwen35Config, n: u32, window_budget: u32, seed: u64) -> Vec<f32> {
    let state = StreamState::new(Gpu::new(crate::model::pipelines()), cfg, n);
    let d = cfg.d_model;

    let x0 = seed_residual(n, d, seed);
    let mut xres = state.gpu.storage_init("stream.x0", &x0);

    let n_layers = cfg.n_layers;
    let sched = weightset::Schedule::cyclic(n_layers, 1);
    let mut ws = weightset::WeightSet::build(n_layers, window_budget, sched, Box::new(weightset::CyclicScan { lookahead: LOOKAHEAD }))
        .unwrap_or_else(|e| panic!("stream: WeightSet::build: {e}"));

    let mut slots: Vec<Option<OwnedStreamedLayer>> = (0..window_budget).map(|_| None).collect();
    // Load every group WeightSet pinned up front - these never surface as an
    // `advance` miss (see `WeightSet::slot_contents`'s own doc), so the
    // caller must load them here, once, before the first `advance` call.
    for (i, slot) in ws.slot_contents().iter().enumerate() {
        if let Some(gid) = slot {
            slots[i] = Some(state.load_layer(dir, cfg, gid.0 as usize));
        }
    }

    for cursor in 0..n_layers as usize {
        let (slot_id, miss) = ws.advance(cursor);
        let idx = slot_id.0 as usize;
        if miss {
            // Drop the evicted occupant's device buffers, then force the
            // submit+wait that actually reclaims them (lesson 1, this
            // module's own doc) BEFORE the new layer's weights upload -
            // otherwise both the outgoing and incoming layer's device
            // memory can be live at once.
            slots[idx] = None;
            state.gpu.read(&xres, 1);
            slots[idx] = Some(state.load_layer(dir, cfg, cursor));
        }
        let layer = slots[idx].as_ref().expect("stream: WeightSet says this slot is resident");
        // A fresh buffer, never a view into `layer`'s own weights (lesson 2,
        // this module's own doc) - safe to keep across `layer`'s later drop.
        xres = state.layer_forward(cfg, layer, &xres, n);
    }

    state.gpu.read(&xres, (n * d) as usize)
}
