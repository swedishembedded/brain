// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Streaming forward pass over the real 64-layer `Qwen/Qwen3.8-27B-FP8`
//! decoder, holding only a small SLIDING WINDOW of layers' weights resident
//! at once - the piece that lets this model run a real chain of real layer
//! weights on a box with far less RAM than a whole-model build would need
//! (`crate::model::Qwen35::new_*` all require every layer's weights resolved
//! in one host `HashMap` before building anything, which is impossible at
//! this config's real 27B size on a constrained machine).
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
//! # What is reused, unchanged
//!
//! - [`crate::import::import_layer`] - the already-proven, already-measured
//!   (2.37-2.45 GiB peak RSS per layer) per-shard host loader. Nothing in
//!   this module re-decodes a checkpoint tensor by hand.
//! - `model::gdn_mixer::gdn_mixer_fwd` / `model::gqa_mixer::gqa_mixer_fwd` -
//!   the shared mixer math both `crate::model::Qwen35`'s own
//!   `layer_gdn_fwd`/`layer_gqa_fwd` and `tests/real_weight_streaming.rs`
//!   already drive. This module's own `gdn_layer_forward`/`gqa_layer_forward`
//!   below are close copies of those two methods, differing only in reading
//!   weights from a per-layer [`OwnedStreamedLayer`] instead of `self.w`/
//!   `self.weights` (`Qwen35`'s own instance-wide stores, which this module
//!   exists precisely because we cannot afford to build).
//! - `model::ops::{Ops, Weight}` - the same int8 (DP4A) façade `Qwen35::
//!   new_impl_on`'s own `upload` closure drives; the 12 quantizable leaf
//!   names and their `(n, k)` shapes below mirror that closure exactly.
//!
//! `paramstore::upload::Uploader` (the chunked-decode-from-mmap host-RAM
//! bounder `qwen3omnimoe::generate` needs) is deliberately NOT used here -
//! `import_layer` already solves the same problem for this crate's own
//! naming convention, at a measured RSS this module's own gate 2 test
//! re-confirms at full-chain scale.

use std::collections::HashMap;
use std::path::Path;

use checkpoint::mmap::MmapSafetensors;
use data::qwen_tokenizer::QwenBpe;
use data::rng::Rng;
use data::tokenizer::Tokenizer;
use gpu_core::select::Dtype;
use gpu_core::{DeviceBuffer, Gpu};
use model::block::{rmsnorm_fwd, swiglu_fwd, KernelIds};
use model::gdn::{GdnBwdIds, GdnIds, GdnShape};
use model::gdn_mixer::{gdn_mixer_fwd, GdnMixerIds, GdnMixerShape, GdnMixerWeights};
use model::gqa_mixer::{gqa_mixer_fwd, GqaMixerIds, GqaMixerShape, GqaMixerWeights};
use model::ops::{Act, Ops, Weight};

use crate::config::{LayerType, Qwen35Config};
use crate::import::{import_layer, import_mtp};
use crate::sample::{argmax, sample_logits};

/// This model's own [`weightset::ResidencyPlan`] choice for a single
/// streaming forward pass: the fully-known schedule ([`weightset::Schedule::
/// cyclic`], one pass over all 64 layers) makes [`weightset::CyclicScan`]
/// Bélády-optimal, not a heuristic - see that type's own doc. `lookahead: 1`
/// is the minimum rotating reserve a schedule narrower than the model needs
/// at all (a window that already fits every layer, `budget >= n_layers`,
/// never evicts regardless of this number).
///
/// **Measured, not merely asserted**: `crates/perf`'s `weights-qwen35`
/// scenario drives this exact `CyclicScan`/`Lru`/`AllResident` code against
/// this model's real 64-layer int8 byte-cost profile
/// (`Qwen35Config::layer_i8_bytes`, ~372-383 MB depending on GDN vs GQA layer
/// type - a real but small, ~3%, heterogeneity). At every budget tested (2,
/// 4, 8, 16, 32 slots, 8 passes), `CyclicScan`'s `churn_overhead` is exactly
/// `1.0` on BOTH the plain reload-count metric and a byte-weighted one - the
/// real per-layer size spread does not change which policy wins, because
/// `CyclicScan`'s pinned/tail split is fixed by the schedule and identical
/// every pass regardless of what each pinned or evicted group actually
/// costs. `Lru` measures strictly worse at every budget, and the gap widens
/// with budget (relative to `Lru`'s own fixed 64-reloads/pass, which never
/// improves): +4.9%/+5.0% (count/bytes) at budget 4, +12.3% at 8, +30.6% at
/// 16, +93.9%/+94.1% at 32.
///
/// **The honest caveat those numbers need**: that whole comparison is a
/// MULTI-pass benefit (`CyclicScan`'s persistent pin only pays off when the
/// SAME [`WeightSet`](weightset::WeightSet) survives across repeated passes
/// over the schedule), and [`stream_all_layers`] below builds a brand-new
/// `WeightSet` on EVERY call, with `passes=1` always - see its own doc. A
/// single, non-repeating pass never revisits any group, so at `passes=1`
/// every policy loads every one of the 64 layers from disk exactly once,
/// with ZERO measured difference between `CyclicScan` and `Lru` in real I/O,
/// consistent with [`generate`]'s own already-documented design ("every
/// decode step re-pays the SAME fixed per-pass weight-streaming cost").
/// `CyclicScan` therefore costs nothing over `Lru` today (matches the
/// theoretically-correct choice for any future design that DOES persist a
/// window across decode steps, at zero downside now) but does not currently
/// buy anything either - there is no cross-call persistence yet for it to
/// exploit. Given that, and `qwen35_bench.rs`'s own profiling finding that
/// this model's real measured wall-clock on this iGPU is dispatch-
/// latency-bound (Gated DeltaNet's sequential per-chunk dispatches, not
/// weight-loading I/O), there is no measured case today for moving
/// `WINDOW_BUDGET` off its current value (`crate::caps`'s `run_streaming`,
/// `4`) - a wider window would only shrink an I/O cost this model's own
/// profiling already found is not the bottleneck.
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

/// The MTP head's own weights, loaded ONCE per [`generate`] call (never
/// re-streamed per decode step, unlike the 64 main-stack layers whose
/// weights `stream_all_layers` drops and rebuilds on every pass) - see
/// [`generate`]'s own doc, "MTP wiring", for why: every decode pass already
/// re-pays the ~3-4 minute weight-STREAMING cost of the 64 main layers
/// regardless of how many token positions it yields, so MTP's own weights
/// (a few hundred MB, dominated by the dense MLP's `[17408, 5120]`-class
/// leaves - genuinely small next to a single main layer) are cheap enough to
/// keep resident for the whole call rather than re-streamed.
///
/// `layer` reuses [`OwnedGqaLayer`] directly (not a hand-duplicated copy):
/// `mtp.layers.0.*`'s real dims match a main-stack `Full` layer's exactly
/// (this milestone's own planning note, confirmed against the real
/// checkpoint header). `fc_e`/`fc_h`/the three norm vectors stay fp32
/// (`Weight::F32` for `fc_e`/`fc_h`) - combined `~210 MB` at fp32,
/// genuinely negligible next to a single main layer's own streamed
/// footprint, so quantizing them too was judged not worth the extra
/// complexity here.
pub struct OwnedMtpLayer {
    pub pre_fc_norm_embedding: DeviceBuffer,
    pub pre_fc_norm_hidden: DeviceBuffer,
    pub fc_e: Weight,
    pub fc_h: Weight,
    pub layer: OwnedGqaLayer,
    pub norm: DeviceBuffer,
}

pub(crate) fn get<'a>(w: &'a HashMap<String, Vec<f32>>, name: &str, l: usize) -> &'a [f32] {
    w.get(name).unwrap_or_else(|| panic!("stream: layer {l}: import_layer did not produce {name}")).as_slice()
}

fn get_mtp<'a>(w: &'a HashMap<String, Vec<f32>>, name: &str) -> &'a [f32] {
    w.get(name).unwrap_or_else(|| panic!("stream: import_mtp did not produce {name}")).as_slice()
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

pub(crate) fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("stream: kernel '{name}' not registered"))
}

pub(crate) fn kernel_ids(g: &Gpu) -> KernelIds {
    KernelIds {
        rmsnorm: idx(g, "rmsnorm"),
        rms_inv: idx(g, "rms_inv"),
        rmsnorm_dx: idx(g, "rmsnorm_dx"),
        rmsnorm_dw: idx(g, "rmsnorm_dw"),
        // M-RoPE (`rope2d`) is what rotates here, never `block::rope_fwd` -
        // so UNREGISTERED, not a stand-in `rmsnorm` index (see
        // `model::block::UNREGISTERED`).
        rope: model::block::UNREGISTERED,
        rope_bwd: model::block::UNREGISTERED,
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

pub(crate) fn gdn_mixer_ids(g: &Gpu) -> GdnMixerIds {
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

pub(crate) fn gqa_mixer_ids(g: &Gpu) -> GqaMixerIds {
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
    /// shard, via [`import_layer`]) straight into device buffers - the
    /// on-disk-per-layer half of the split; [`Self::build_layer`] is the
    /// device-upload half, taking an already-imported host weight map so a
    /// test can drive it against synthetic weights with no checkpoint on disk.
    pub fn load_layer(&self, dir: &Path, cfg: &Qwen35Config, l: usize) -> OwnedStreamedLayer {
        let shard = dir.join(format!("layers-{l}.safetensors"));
        let reader = MmapSafetensors::open(&shard).unwrap_or_else(|e| panic!("stream: open {}: {e}", shard.display()));
        let w = import_layer(&reader, cfg, l, 128).unwrap_or_else(|e| panic!("stream: import_layer({l}): {e}"));
        drop(reader);
        self.build_layer(cfg, l, &w)
    }

    /// Upload layer `l`'s already-imported weights (`w`, `blocks.{l}.*`
    /// naming - real, from [`import_layer`], or synthetic, from
    /// `crate::init::init_weights`) straight into device buffers: the 8-12
    /// quantizable leaves as int8 (DP4A) [`Weight`]s via [`Weight::upload`],
    /// everything else (norms, GDN's `A_log`/`dt_bias`/gated-norm weight,
    /// GQA's `q_norm`/`k_norm`) as plain fp32 `storage_init` buffers - the
    /// same quantizable/non-quantizable split `crate::model::is_i8_linear`
    /// and `Qwen35::new_impl_on`'s own `upload` closure already establish,
    /// just for one layer at a time instead of every layer up front.
    fn build_layer(&self, cfg: &Qwen35Config, l: usize, w: &HashMap<String, Vec<f32>>) -> OwnedStreamedLayer {
        let ty = cfg.layer_types()[l];
        let p = |s: &str| format!("blocks.{l}.{s}");
        let f32buf = |name: &str| self.gpu.storage_init(name, get(w, name, l));
        let i8w = |name: &str, n: usize, k: usize| Weight::upload(&self.ops, get(w, name, l), n, k, Dtype::I8);
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

    /// Load the MTP head's own weights from `dir/mtp.safetensors`
    /// (`crate::import::import_mtp`, this milestone's real-weight MTP
    /// import - the counterpart of [`Self::load_layer`], but for the ONE
    /// head loaded once per [`generate`] call, never per decode step - see
    /// [`OwnedMtpLayer`]'s own doc). The self-attn/MLP leaves go through the
    /// SAME int8 (DP4A) [`Weight::upload`] path [`Self::build_layer`]
    /// already uses for every main-stack `Full` layer; `fc_e`/`fc_h` stay
    /// fp32 (also via `Weight::upload`, just requesting `Dtype::F32` - the
    /// same façade, not a second code path).
    pub fn load_mtp(&self, dir: &Path, cfg: &Qwen35Config) -> OwnedMtpLayer {
        let shard = dir.join("mtp.safetensors");
        let reader = MmapSafetensors::open(&shard).unwrap_or_else(|e| panic!("stream: open {}: {e}", shard.display()));
        let w = import_mtp(&reader, cfg, 128).unwrap_or_else(|e| panic!("stream: import_mtp: {e}"));
        drop(reader);

        let d = cfg.d_model as usize;
        let f32buf = |name: &str| self.gpu.storage_init(name, get_mtp(&w, name));
        let f32w = |name: &str, n: usize, k: usize| Weight::upload(&self.ops, get_mtp(&w, name), n, k, Dtype::F32);
        let i8w = |name: &str, n: usize, k: usize| Weight::upload(&self.ops, get_mtp(&w, name), n, k, Dtype::I8);

        let hqp = cfg.q_proj_dim() as usize;
        let hkv = cfg.kv_dim() as usize;
        let hq = cfg.q_dim() as usize;
        let ff = cfg.intermediate_size as usize;

        OwnedMtpLayer {
            pre_fc_norm_embedding: f32buf("mtp.pre_fc_norm_embedding.weight"),
            pre_fc_norm_hidden: f32buf("mtp.pre_fc_norm_hidden.weight"),
            fc_e: f32w("mtp.fc_e.weight", d, d),
            fc_h: f32w("mtp.fc_h.weight", d, d),
            layer: OwnedGqaLayer {
                ln1: f32buf("mtp.layers.0.ln1.weight"),
                ln2: f32buf("mtp.layers.0.ln2.weight"),
                q_norm: f32buf("mtp.layers.0.self_attn.q_norm.weight"),
                k_norm: f32buf("mtp.layers.0.self_attn.k_norm.weight"),
                q_proj: i8w("mtp.layers.0.self_attn.q_proj.weight", hqp, d),
                k_proj: i8w("mtp.layers.0.self_attn.k_proj.weight", hkv, d),
                v_proj: i8w("mtp.layers.0.self_attn.v_proj.weight", hkv, d),
                o_proj: i8w("mtp.layers.0.self_attn.o_proj.weight", d, hq),
                mlp_gate: i8w("mtp.layers.0.mlp.gate.weight", ff, d),
                mlp_up: i8w("mtp.layers.0.mlp.up.weight", ff, d),
                mlp_down: i8w("mtp.layers.0.mlp.down.weight", d, ff),
            },
            norm: f32buf("mtp.norm.weight"),
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

// ---------------------------------------------------------------------------
// MTP head: reimplements `crate::model::Qwen35::run_mtp_forward`'s own math
// (see that function's doc) for a SINGLE row rather than the whole training
// batch that function computes over, against a [`StreamState`]'s own
// `Ops`/`Weight` façade - matching this module's own established pattern of
// reimplementing per-layer forward directly, never instantiating a `Qwen35`
// (this module's own doc, "What is reused, unchanged").
// ---------------------------------------------------------------------------

/// A one-row M-RoPE table at absolute position `pos` - [`StreamState::cos`]/
/// [`StreamState::sin`] are sized for the WHOLE padded main-stack sequence
/// (`0..n-1`), never a single arbitrary absolute position, so an MTP-head
/// call (always exactly one row, at whichever absolute position its own
/// `hidden` row came from - see [`mtp_forward`]'s own doc) needs its own
/// small table instead.
fn single_position_mrope(cfg: &Qwen35Config, g: &Gpu, pos: u32) -> (DeviceBuffer, DeviceBuffer) {
    let (cos, sin) = qwen3vl::mrope::mrope_tables(&[[pos, pos, pos]], cfg.mrope_section, cfg.rotary_dim(), cfg.rope_theta);
    (g.storage_init("mtp.rope_cos", &cos), g.storage_init("mtp.rope_sin", &sin))
}

/// [`StreamState::gqa_layer_forward`]'s own math, parameterized on an
/// explicit `cos`/`sin` table instead of `state.cos`/`state.sin` - the one
/// difference an MTP-head call needs (see [`single_position_mrope`]'s doc).
/// A close copy, same reason `gqa_layer_forward` is itself a close copy of
/// `crate::model::Qwen35::layer_gqa_fwd` (this module's own doc).
fn mtp_mixer_forward(state: &StreamState, cfg: &Qwen35Config, l: &OwnedGqaLayer, xn1: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let g = &state.gpu;
    let ops = &state.ops;
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
    let weights = GqaMixerWeights { q_norm: &l.q_norm, k_norm: &l.k_norm, cos, sin };
    let (ctx_gated, _acts) = gqa_mixer_fwd(g, &state.ids.gqa_mixer, &shape, &weights, &q_full, &k, &v, n, false);

    let out = g.storage((n * d) as u64);
    let mut s2 = Vec::new();
    let act2 = ops.act(&mut s2, &ctx_gated, 0, n, shape.qd());
    ops.matmul(&mut s2, &l.o_proj, &act2, &out, 0);
    g.submit(&[], &s2);
    out
}

/// One MTP-head forward pass at a SINGLE position: `candidate_embed` (a
/// speculative or just-confirmed token's own real embedding row) combined
/// with `hidden_row` (`res_last` - see `crate::model::Qwen35::
/// run_mtp_forward`'s own doc - the pre-final-norm residual stream, the SAME
/// row [`generate`]'s main head reads its own logits from) at absolute
/// position `pos` - the position `hidden_row` itself came from (`crate::
/// model::Qwen35::set_batch`'s own MTP row convention: row `i`'s hidden
/// comes from position `i`, its embedding input from position `i+1`, its
/// RoPE position is `i` too - `hidden_row`'s own position, never the
/// embedded candidate's).
///
/// Reimplements `run_mtp_forward`'s own math (embed -> two pre-norms ->
/// `fc_e(en)+fc_h(hn)` -> one GQA-shaped decoder layer -> `mtp.norm` -> the
/// shared, already-resident int8 `lm_head`) for a single row rather than the
/// whole training batch that function computes over. This is sound for
/// [`generate`]'s own decode loop specifically because of that loop's own
/// causality argument (this module's own doc, "the per-pass confirm/
/// advance/speculate decode loop"): the MTP head's own self-attention seeing
/// only itself (no other rows to attend to at `n=1`, unlike training's
/// whole-batch causal self-attention over every prior MTP row) never affects
/// CORRECTNESS - every candidate this produces is independently re-verified
/// by a real main-head forward the very next pass, and a wrong guess is
/// simply discarded, never trusted blindly. It only affects how often a
/// guess happens to be right (measured, not assumed - gate 2, this
/// milestone's own report).
///
/// Returns the raw `[vocab]` logits, read back host-side (small - `vocab *
/// 4` bytes - next to a whole streaming pass; no reason to keep this
/// on-device only to read it back one call later).
fn mtp_forward(state: &StreamState, cfg: &Qwen35Config, mtp: &OwnedMtpLayer, lm_head: &Weight, candidate_embed: &[f32], hidden_row: &DeviceBuffer, pos: u32) -> Vec<f32> {
    let g = &state.gpu;
    let ops = &state.ops;
    let d = cfg.d_model;
    let n = 1u32;

    let e = g.storage_init("mtp.e", candidate_embed);
    let en = g.storage(d as u64);
    let hn = g.storage(d as u64);
    g.submit(
        &[],
        &[
            rmsnorm_fwd(g, &state.ids.kernels, &e, &mtp.pre_fc_norm_embedding, &en, d, n),
            rmsnorm_fwd(g, &state.ids.kernels, hidden_row, &mtp.pre_fc_norm_hidden, &hn, d, n),
        ],
    );

    let ehp_e = g.storage(d as u64);
    let ehp_h = g.storage(d as u64);
    let mut s1 = Vec::new();
    let act_en = ops.act(&mut s1, &en, 0, n, d);
    ops.matmul(&mut s1, &mtp.fc_e, &act_en, &ehp_e, 0);
    let act_hn = ops.act(&mut s1, &hn, 0, n, d);
    ops.matmul(&mut s1, &mtp.fc_h, &act_hn, &ehp_h, 0);
    g.submit(&[], &s1);

    let ehp = g.storage(d as u64);
    g.submit(&[], &[g.step(state.ids.add2, &[&ehp_e, &ehp_h, &ehp], &[d], d)]);

    let xn1 = g.storage(d as u64);
    g.submit(&[], &[rmsnorm_fwd(g, &state.ids.kernels, &ehp, &mtp.layer.ln1, &xn1, d, n)]);

    let (cos, sin) = single_position_mrope(cfg, g, pos);
    let mixer_out = mtp_mixer_forward(state, cfg, &mtp.layer, &xn1, &cos, &sin, n);

    let xmid = g.storage(d as u64);
    g.submit(&[], &[g.step(state.ids.add2, &[&ehp, &mixer_out, &xmid], &[d], d)]);

    let xn2 = g.storage(d as u64);
    g.submit(&[], &[rmsnorm_fwd(g, &state.ids.kernels, &xmid, &mtp.layer.ln2, &xn2, d, n)]);

    let mlp_out = state.mlp_forward(cfg, &mtp.layer.mlp_gate, &mtp.layer.mlp_up, &mtp.layer.mlp_down, &xn2, n);

    let block_out = g.storage(d as u64);
    g.submit(&[], &[g.step(state.ids.add2, &[&xmid, &mlp_out, &block_out], &[d], d)]);

    let final_h = g.storage(d as u64);
    g.submit(&[], &[rmsnorm_fwd(g, &state.ids.kernels, &block_out, &mtp.norm, &final_h, d, n)]);

    let mut s2 = Vec::new();
    let act = ops.act(&mut s2, &final_h, 0, n, d);
    let logits_buf = g.storage(cfg.vocab as u64);
    ops.matmul(&mut s2, lm_head, &act, &logits_buf, 0);
    g.submit(&[], &s2);
    g.read(&logits_buf, cfg.vocab as usize)
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

/// One streaming forward pass over every real layer of `cfg`, reading `dir`
/// on demand (`layers-{l}.safetensors` per layer, `import_layer` per miss),
/// holding at most `window_budget` layers' weights resident on device at
/// once. `xres0` is the input residual stream (`[n, d_model]`, `n` =
/// `xres0`'s own row count as sized by the caller - either [`run`]'s
/// synthetic seed or [`generate`]'s real, padded embedding); `n` is passed
/// separately since a `DeviceBuffer` carries no shape of its own.
///
/// Schedule: [`weightset::Schedule::cyclic`]`(cfg.n_layers, 1)` - a SINGLE
/// pass over the decoder stack, not a multi-token decode loop by itself (see
/// [`generate`] for the loop that re-invokes this once per new token, and
/// this module's own doc for why there is no persistent incremental state
/// carried between those calls). Eviction: [`weightset::CyclicScan`] with
/// [`LOOKAHEAD`] - Bélády-optimal for this fully-known schedule.
///
/// Returns the final residual as a fresh device buffer (never [`xres0`]
/// itself, nor a view into any layer's own weights - lesson 2, this module's
/// own doc) - NOT yet read back host-side, NOT yet through the model's final
/// `norm.weight` RMSNorm or the `lm_head` projection; both [`run`] and
/// [`generate`] apply those two steps themselves, since only [`generate`]
/// needs the second one at all.
fn stream_all_layers(state: &StreamState, dir: &Path, cfg: &Qwen35Config, xres0: DeviceBuffer, n: u32, window_budget: u32) -> DeviceBuffer {
    let mut xres = xres0;
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
    xres
}

/// Run one streaming forward pass over every real layer of `cfg` (in
/// practice always [`Qwen35Config::qwen38_27b`]), reading `dir` on demand,
/// holding at most `window_budget` layers' weights resident on device at
/// once. `n` is the row count of the (synthetic - see [`seed_residual`])
/// input residual stream; `seed` is its RNG seed.
///
/// Returns the final residual, read back host-side, `[n, d_model]` row-major.
/// This is [`stream_all_layers`]'s own gate: proving the streaming PLUMBING
/// (real weights, real mixer math, bounded memory) over a synthetic input -
/// see this module's own doc, "Scope". [`generate`] is the real-input,
/// real-output sibling built on the same [`stream_all_layers`] helper.
pub fn run(dir: &Path, cfg: &Qwen35Config, n: u32, window_budget: u32, seed: u64) -> Vec<f32> {
    let state = StreamState::new(Gpu::new(crate::model::pipelines()), cfg, n);
    let d = cfg.d_model;
    let x0 = seed_residual(n, d, seed);
    let xres0 = state.gpu.storage_init("stream.x0", &x0);
    let xres = stream_all_layers(&state, dir, cfg, xres0, n, window_budget);
    state.gpu.read(&xres, (n * d) as usize)
}

// ---------------------------------------------------------------------------
// Real generation (real prompt -> real embeddings -> streaming forward ->
// real lm_head logits -> real sampling -> real decoded text).
//
// **No persistent incremental KV/GDN state across decode steps** (design
// decision 1 of the milestone this landed in): the non-streaming `Qwen35::
// step`'s decode path threads persistent per-layer `gqa_kcache`/`gdn_state`/
// `gdn_hist` fields through the whole generation; carrying that same state
// through a streaming window whose layer buffers are dropped and rebuilt
// every pass would need it to somehow survive eviction. Instead, every
// decode step below re-runs a full chunked/non-incremental forward
// ([`stream_all_layers`], the SAME per-layer forward [`run`] uses for
// prefill) over the WHOLE growing sequence (prompt + generated-so-far).
// This is deliberately the SIMPLER choice, not a lesser one: every decode
// step already re-streams all `cfg.n_layers` layers' weights from disk
// regardless (measured ~75 minutes for one streaming pass over all 64 real
// layers) - the marginal cost of recomputing a short growing prefix via
// chunked/quadratic attention is small next to that fixed per-step
// weight-streaming cost.
//
// **This means multi-token generation is extremely slow.** A caller MUST
// keep `max_new` small (2-4 tokens) against the real 64-layer checkpoint -
// this is the honest, correctly-scoped consequence of the measured
// per-decode-step cost above, not a shortcut. `crate::caps`'s `streaming`
// param and this crate's own integration test both scope generation this
// way for exactly that reason.
//
// ## MTP wiring
//
// The observation above - every decode pass re-pays the SAME fixed
// per-pass weight-streaming cost regardless of how many token positions it
// yields - is exactly the leverage the MTP head ([`crate::model::Qwen35::
// run_mtp_forward`], training-only until this landed) gives for free: it
// predicts a token from the SAME final-layer hidden state a normal forward
// pass already produces, given a (true or speculatively-assumed) next
// token's own embedding. Running it as one extra, CHEAP (resident weights,
// never re-streamed - [`OwnedMtpLayer`]'s own doc), computation within the
// SAME pass that already produces the main next-token prediction lets one
// streaming pass yield genuine progress on TWO tokens instead of one - not
// a FLOPs-bound speculative-decode win (`qwen3::serve::spec_decode`'s own
// kind), an I/O-bound-regime-specific one: the win comes from amortizing
// the fixed per-pass weight-streaming cost over more confirmed tokens, not
// from doing less arithmetic.
//
// ## The per-pass confirm/advance/speculate decode loop
//
// [`generate`]'s `use_mtp: true` path ([`generate_mtp_accelerated`]) - see
// that function's own doc for the exact mechanics. GREEDY ONLY
// (`temperature <= 0.0`) - see [`generate`]'s own doc for why sampled
// decoding is explicitly out of scope, not merely deferred.
// ---------------------------------------------------------------------------

/// The Gated DeltaNet chunk size [`generate`]'s decode loop always pads a
/// growing sequence up to a multiple of: the reference default
/// (`torch_chunk_gated_delta_rule`'s `chunk_size=64`, `model::gdn`'s own doc)
/// and what `model::gdn::gdn_chunk_size` always lands on at real
/// hybrid-decoder scale. Pinned here directly rather than re-derived from
/// `gdn_chunk_size` on a not-yet-padded `t`: `gdn_chunk_size`'s smaller
/// candidates (`32,16,...,1`) exist for PREFILL callers that cannot pad a
/// real sequence length at all (see that function's own doc) - this caller
/// CAN pad, and always wants the same chunk size real production inference
/// uses, not whatever smaller divisor an unpadded `t` happens to have.
pub(crate) const GDN_DECODE_CHUNK: u32 = 64;

/// Pad `t` up to the next multiple of [`GDN_DECODE_CHUNK`] (`t=64` stays
/// `64`; `t=65` pads to `128`) - design decision 2: `model::gdn`'s chunked
/// recurrence asserts `t % chunk == 0`
/// (`model::gdn::GdnShape::n_chunks`), and a growing decode-step sequence
/// will not generally already be a multiple of 64. Padding happens at the
/// END ONLY (GDN is strictly causal - see
/// `gdn_end_padding_does_not_change_real_position_outputs` below for the
/// direct proof this relies on), never the start or middle.
pub(crate) fn pad_to_gdn_chunk(t: u32) -> u32 {
    t.div_ceil(GDN_DECODE_CHUNK) * GDN_DECODE_CHUNK
}

/// Real per-token embedding rows for `ids`, read directly off `name`
/// (`model.language_model.embed_tokens.weight`, `[vocab, d_model]`, plain
/// BF16 - confirmed no `.weight_scale_inv` sibling, so a row read has zero
/// cross-row dependency, unlike FP8's block-128 scale coupling) via
/// [`MmapSafetensors::tensor_f32_range`] (design decision 3) - one targeted,
/// `O(d_model)` read per token, never a whole-`[vocab, d_model]`-table decode
/// or scan. Stacked `[ids.len(), d_model]` row-major - [`generate`]'s real
/// replacement for [`seed_residual`]'s synthetic input.
pub(crate) fn embed_rows(reader: &MmapSafetensors, name: &str, ids: &[u32], d: usize) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(ids.len() * d);
    for &id in ids {
        let row = reader.tensor_f32_range(name, id as usize * d, d).ok_or_else(|| format!("stream::generate: token id {id} out of range for {name}"))?;
        out.extend_from_slice(&row);
    }
    Ok(out)
}

/// Quantize `name` (`[n, k]`, plain BF16 - `lm_head.weight`/
/// `model.language_model.embed_tokens.weight` when tied) to int8 (DP4A)
/// straight from the mmap, WITHOUT ever holding the whole dequantized
/// `[n, k]` f32 array in host RAM at once (design decision 4). `model::
/// int8::quantize_weight`'s scale is per ROW, so quantizing `rows_per_chunk`
/// rows at a time via [`MmapSafetensors::with_tensor_chunks`] (chunk size a
/// multiple of `k`, so every chunk boundary lands on a row boundary) and
/// writing each chunk's packed int8 words / per-row scales straight into a
/// pre-sized device buffer via [`Gpu::write_at`]/[`Gpu::write_f32_at`] is
/// byte-identical to `Weight::upload(ops, whole_tensor, n, k, Dtype::I8)`,
/// bounding peak EXTRA host allocation to `O(rows_per_chunk * k)` (tens of
/// MB) instead of `O(n * k)` (~4.74 GiB for the real `lm_head` shape,
/// `248320 * 5120 * 4` bytes). Chosen over the one-shot dequant-then-quantize
/// after checking `free -h` on this (shared) machine at the time this was
/// written: available RAM was too close to that 4.74 GiB peak for comfort.
/// The resulting `Weight::I8` (~1.18 GiB packed) is what [`generate`] keeps
/// resident on device for the whole call - built ONCE, never re-quantized
/// per decode step.
fn quantize_i8_from_mmap_rows(gpu: &Gpu, reader: &MmapSafetensors, name: &str, n: usize, k: usize, rows_per_chunk: usize) -> Weight {
    assert!(k.is_multiple_of(4), "quantize_i8_from_mmap_rows: k must be a multiple of 4 (got {k})");
    assert!(rows_per_chunk > 0, "quantize_i8_from_mmap_rows: rows_per_chunk must be > 0");
    let kg = k / 4;
    let w = gpu.storage((n * kg) as u64);
    let s = gpu.storage(n as u64);
    let mut any = false;
    let found = reader.with_tensor_chunks(name, rows_per_chunk * k, &mut |off, chunk| {
        any = true;
        assert_eq!(off as usize % k, 0, "quantize_i8_from_mmap_rows: chunk offset {off} is not row-aligned (k={k})");
        assert_eq!(chunk.len() % k, 0, "quantize_i8_from_mmap_rows: chunk length {} is not a whole number of rows (k={k})", chunk.len());
        let rows = chunk.len() / k;
        let row0 = off as usize / k;
        let (packed, scales) = model::int8::quantize_weight(chunk, rows, k);
        gpu.write_at(&w, (row0 * kg) as u64, &packed);
        gpu.write_f32_at(&s, row0 as u64, &scales);
    });
    assert!(found && any, "quantize_i8_from_mmap_rows: {name} not found or empty");
    Weight::I8 { w, s, n: n as u32, k: k as u32 }
}

/// The model's final `norm.weight` (`model.language_model.norm.weight` in
/// `outside.safetensors`), folded by [`crate::import::fold_plain_rmsnorm_weights`]
/// exactly like every other plain-RMSNorm weight this crate imports (the
/// `(1+w)` reparameterization - see `crate::import`'s own module doc). A
/// small (`[d_model]`) whole-tensor read, unlike [`embed_rows`]'s per-row one
/// - there is only ever one of these per call.
pub(crate) fn read_final_norm(reader: &MmapSafetensors, d: usize) -> Result<Vec<f32>, String> {
    let raw = reader
        .tensor_f32("model.language_model.norm.weight")
        .ok_or_else(|| "stream::generate: model.language_model.norm.weight missing from outside.safetensors".to_string())?;
    if raw.len() != d {
        return Err(format!("stream::generate: norm.weight has {} elements, expected {d}", raw.len()));
    }
    let mut m: HashMap<String, Vec<f32>> = HashMap::new();
    m.insert("norm.weight".to_string(), raw);
    crate::import::fold_plain_rmsnorm_weights(&mut m);
    Ok(m.remove("norm.weight").expect("just inserted"))
}

/// The main head epilogue shared by every decode path: `hidden_row` (one
/// `[d_model]` residual-stream row, pre-final-norm - `res_last` at whichever
/// position the caller wants a prediction for) -> `norm.weight` RMSNorm ->
/// the resident int8 `lm_head` matmul -> raw `[vocab]` logits, read back
/// host-side. Used by both [`generate`]'s plain per-token loop and
/// [`generate_mtp_accelerated`]'s multi-position reads from a single pass -
/// calling the SAME function from both is what makes gate 1's byte-identical
/// claim (this milestone's own report) true by construction, not just by
/// argument: there is only one implementation of "apply the final norm and
/// project to vocab logits" for either path to possibly diverge on.
fn head_logits(state: &StreamState, cfg: &Qwen35Config, final_norm_buf: &DeviceBuffer, head: &Weight, hidden_row: &[f32]) -> Vec<f32> {
    let g = &state.gpu;
    let d = cfg.d_model as usize;
    let x = g.storage_init("stream.generate.row", hidden_row);
    let normed = g.storage(d as u64);
    g.submit(&[], &[rmsnorm_fwd(g, &state.ids.kernels, &x, final_norm_buf, &normed, d as u32, 1)]);

    let mut s = Vec::new();
    let act = state.ops.act(&mut s, &normed, 0, 1, d as u32);
    let logits_buf = g.storage(cfg.vocab as u64);
    state.ops.matmul(&mut s, head, &act, &logits_buf, 0);
    g.submit(&[], &s);
    g.read(&logits_buf, cfg.vocab as usize)
}

/// Real end-to-end generation: tokenize `prompt` (`tokenizer_path`, `data::
/// qwen_tokenizer::QwenBpe` - mirrors `crate::caps::GenerateAction::run`'s
/// own tokenizer-present path), embed it with REAL rows off `dir/outside.
/// safetensors` (design decision 3), stream all `cfg.n_layers` real decoder
/// layers ([`stream_all_layers`]) once per decode step over the growing
/// (prompt + generated-so-far) sequence - end-padded to the next
/// [`GDN_DECODE_CHUNK`] multiple with token id `0` for GDN's chunk-
/// divisibility constraint (design decision 2) - apply the model's final
/// `norm.weight` RMSNorm and a resident int8 `lm_head` (design decision 4,
/// quantized once before the loop, never per step) to just the last REAL
/// position's hidden state, and sample the next token via `crate::sample::
/// sample_logits` (design decision 5: greedy when `temperature <= 0`,
/// otherwise temperature/top-k/top-p). Stops at `max_new` tokens or the
/// first id in `{<|im_end|>, <|endoftext|>}` (mirrors `crate::caps`'s own
/// default EOS list). Returns the newly generated text ONLY (not the
/// prompt's own text) - the same convention `crate::caps::GenerateAction`'s
/// non-streaming `Plan::Raw` output already uses.
///
/// See this module's own doc (just above) for why multi-token generation
/// through this path is inherently slow, and why that is not a bug.
///
/// `use_mtp` opts into [`generate_mtp_accelerated`]'s "confirm, advance,
/// speculate" decode loop instead of the plain per-token loop below - see
/// that function's own doc for the mechanics and for why it is GREEDY ONLY
/// (`temperature <= 0.0`; a non-zero `temperature` with `use_mtp` is an
/// `Err`, not a silent fallback to the plain path - verifying a
/// STOCHASTIC draft against a STOCHASTIC target needs rejection-sampling
/// machinery this milestone deliberately does not build, unlike `qwen3::
/// serve::spec_decode`, which never needs it either since it drafts
/// deterministically then verifies against the target's own greedy choice -
/// but a temperature-sampled TARGET makes "the correct next token"
/// ill-defined for a single verification check here).
///
/// Thin wrapper over [`generate_with_stats`], discarding the real pass count
/// that function also returns (kept `pub` here for every existing caller;
/// `generate_with_stats` exists so a test can observe the real number of
/// [`stream_all_layers`] calls either path issues - this milestone's own
/// gate 2).
#[allow(clippy::too_many_arguments)]
pub fn generate(
    dir: &Path,
    cfg: &Qwen35Config,
    tokenizer_path: &Path,
    prompt: &str,
    max_new: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    window_budget: u32,
    seed: u64,
    use_mtp: bool,
) -> Result<String, String> {
    generate_with_stats(dir, cfg, tokenizer_path, prompt, max_new, temperature, top_k, top_p, window_budget, seed, use_mtp).map(|(text, _passes)| text)
}

/// [`generate`]'s own real implementation, additionally returning the real
/// number of [`stream_all_layers`] calls (full 64-layer streaming passes)
/// the call issued - see [`generate`]'s own doc.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_stats(
    dir: &Path,
    cfg: &Qwen35Config,
    tokenizer_path: &Path,
    prompt: &str,
    max_new: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    window_budget: u32,
    seed: u64,
    use_mtp: bool,
) -> Result<(String, u32), String> {
    if use_mtp && temperature > 0.0 {
        return Err(
            "stream::generate: use_mtp requires greedy decoding (temperature <= 0.0) - sampled generation needs \
             rejection-sampling machinery this milestone deliberately does not build (see `generate`'s own doc)"
                .to_string(),
        );
    }

    let tok_path = tokenizer_path.to_str().ok_or_else(|| "stream::generate: tokenizer path is not valid UTF-8".to_string())?;
    let tok = QwenBpe::from_file(tok_path)?;
    let ids = tok.encode(prompt);
    if ids.is_empty() {
        return Err("stream::generate: prompt encoded to zero tokens".to_string());
    }
    // Mirrors `crate::caps::GenerateAction::run`'s own tokenizer-present EOS
    // fallback exactly (design decision 6): both Qwen3 EOS ids when no
    // explicit list is given.
    let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|s| tok.encode(s).first().copied()).collect();

    let d = cfg.d_model as usize;
    let outside_path = dir.join("outside.safetensors");
    let outside = MmapSafetensors::open(&outside_path)?;

    let embed_name = "model.language_model.embed_tokens.weight";
    let head_name = if cfg.tie_embeddings { embed_name } else { "lm_head.weight" };

    let gpu = Gpu::new(crate::model::pipelines());
    // The int8 lm_head: quantized ONCE here, kept resident on device for the
    // whole call (design decision 4) - never re-quantized per decode step.
    let head = quantize_i8_from_mmap_rows(&gpu, &outside, head_name, cfg.vocab as usize, d, 4096);
    let final_norm = read_final_norm(&outside, d)?;
    let final_norm_buf = gpu.storage_init("stream.generate.final_norm", &final_norm);

    if use_mtp {
        return generate_mtp_accelerated(dir, cfg, &tok, &eos, &outside, embed_name, &gpu, &head, &final_norm_buf, ids, max_new, window_budget);
    }

    let mut ids = ids;
    let mut rng = Rng::new(seed);
    let mut generated: Vec<u32> = Vec::with_capacity(max_new);
    let mut passes = 0u32;

    for _ in 0..max_new {
        let t = ids.len() as u32;
        let padded_t = pad_to_gdn_chunk(t);
        let mut padded_ids = ids.clone();
        padded_ids.resize(padded_t as usize, 0); // end-only padding, dummy token id 0
        let x0 = embed_rows(&outside, embed_name, &padded_ids, d)?;

        // A fresh `StreamState` per decode step (fresh mrope tables sized
        // for THIS step's padded length, fresh kernel-id resolution) but the
        // SAME underlying device throughout (`gpu.share()`, not a fresh
        // `Gpu::new` - device/adapter/pipeline creation is the one per-call
        // cost worth hoisting out of this loop; everything else a
        // `StreamState` builds is cheap next to one layer's own
        // weight-streaming cost, design decision 1).
        let state = StreamState::new(gpu.share(), cfg, padded_t);
        let xres0 = state.gpu.storage_init("stream.generate.x0", &x0);
        let xres = stream_all_layers(&state, dir, cfg, xres0, padded_t, window_budget);
        passes += 1;

        // Only the LAST REAL (non-padding) position's hidden state matters
        // (design decision 2) - pull just that one row host-side rather than
        // computing a `[padded_t, vocab]` logits matrix nobody reads past
        // row `t-1`.
        let hidden_all = state.gpu.read(&xres, (padded_t as usize) * d);
        let last = (t - 1) as usize;
        let last_row = hidden_all[last * d..(last + 1) * d].to_vec();
        let logits = head_logits(&state, cfg, &final_norm_buf, &head, &last_row);

        let next = sample_logits(&logits, temperature, top_k, top_p, &mut rng);
        if eos.contains(&next) {
            break;
        }
        generated.push(next);
        ids.push(next);
    }

    Ok((tok.decode(&generated), passes))
}

/// The MTP-accelerated "confirm, advance, speculate" GREEDY decode loop -
/// [`generate`]'s `use_mtp: true` path. See [`generate`]'s own doc for the
/// scoping decision (greedy only) and this module's own doc for the
/// per-pass mechanics this reimplements; the short version:
///
/// The growing sequence this loop feeds [`stream_all_layers`] each pass ends
/// in at most one CONFIRMED-but-unverified... no - ends in a confirmed
/// prefix plus at most one PENDING (speculative, MTP-guessed, not yet
/// verified) tail token. Each pass reads main-head logits at TWO row
/// positions from the SAME forward pass (free - both rows are already
/// computed as part of one causal forward): the last-confirmed position
/// (always) and, only when the previous pass's pending guess matches what
/// this row independently predicts, the pending token's own position too
/// (a second, free confirmation - causal attention guarantees that row's
/// output is identical to what a genuine serial continuation would have
/// produced there, since its own input actually was the now-known-correct
/// token). A fresh pending guess for the next round comes from one MTP-head
/// call against THIS pass's own already-computed hidden state - never an
/// extra pass. No persistent-KV rollback is needed on a mismatch (unlike
/// `qwen3::serve::spec_decode`'s `model::paged::truncate`): a rejected
/// pending token is simply never included in the next pass's input
/// sequence, and `stream_all_layers`' growing-prefix-recompute architecture
/// (no persistent KV/GDN state across passes at all - this module's own
/// doc) already makes that trivial, a deliberate simplification this
/// specific architecture enables, not a shortcut or a missing piece.
#[allow(clippy::too_many_arguments)]
fn generate_mtp_accelerated(
    dir: &Path,
    cfg: &Qwen35Config,
    tok: &QwenBpe,
    eos: &[u32],
    outside: &MmapSafetensors,
    embed_name: &str,
    gpu: &Gpu,
    head: &Weight,
    final_norm_buf: &DeviceBuffer,
    prompt_ids: Vec<u32>,
    max_new: usize,
    window_budget: u32,
) -> Result<(String, u32), String> {
    let d = cfg.d_model as usize;

    // Loaded ONCE, kept resident for the whole call (`OwnedMtpLayer`'s own
    // doc) - built against a throwaway minimal `StreamState` (`n=1`; only
    // its `ops`/`gpu` matter for loading weights, not its `cos`/`sin`/
    // `ones_khd`, which this call never uses).
    let loader_state = StreamState::new(gpu.share(), cfg, 1);
    let mtp = loader_state.load_mtp(dir, cfg);
    drop(loader_state);

    let mut generated: Vec<u32> = Vec::new();
    let mut pending: Option<u32> = None;
    let mut passes = 0u32;
    let mut hit_eos = false;

    while generated.len() < max_new && !hit_eos {
        let mut seq = prompt_ids.clone();
        seq.extend_from_slice(&generated);
        let last_confirmed_pos = (seq.len() - 1) as u32;
        if let Some(p) = pending {
            seq.push(p);
        }
        let t = seq.len() as u32;
        let padded_t = pad_to_gdn_chunk(t);
        let mut padded_ids = seq.clone();
        padded_ids.resize(padded_t as usize, 0);
        let x0 = embed_rows(outside, embed_name, &padded_ids, d)?;

        let state = StreamState::new(gpu.share(), cfg, padded_t);
        let xres0 = state.gpu.storage_init("stream.generate.x0", &x0);
        let xres = stream_all_layers(&state, dir, cfg, xres0, padded_t, window_budget);
        passes += 1;

        // Reading multiple real-position rows out of one pass is free (both
        // already computed by the same causal forward) - this module's own
        // doc, "the per-pass confirm/advance/speculate decode loop".
        let hidden_all = state.gpu.read(&xres, (padded_t as usize) * d);
        let row_of = |pos: u32| hidden_all[pos as usize * d..(pos as usize + 1) * d].to_vec();

        // 1. The model's own true prediction for what follows the confirmed
        // history - independent of whatever `pending` guessed.
        let row_last_confirmed = row_of(last_confirmed_pos);
        let candidate_1 = argmax(&head_logits(&state, cfg, final_norm_buf, head, &row_last_confirmed)) as u32;

        if eos.contains(&candidate_1) {
            generated.push(candidate_1);
            break;
        }
        generated.push(candidate_1);

        if pending == Some(candidate_1) {
            // MATCH: the pending token is now confirmed FOR FREE - read a
            // SECOND main-head prediction at its own (now-confirmed)
            // position, from the SAME pass.
            let pending_pos = last_confirmed_pos + 1;
            let row_pending = row_of(pending_pos);
            let candidate_2 = argmax(&head_logits(&state, cfg, final_norm_buf, head, &row_pending)) as u32;

            if eos.contains(&candidate_2) {
                generated.push(candidate_2);
                hit_eos = true;
                pending = None;
            } else {
                generated.push(candidate_2);
                pending = if generated.len() < max_new {
                    let embed_c2 = embed_rows(outside, embed_name, &[candidate_2], d)?;
                    let hidden_buf = state.gpu.storage_init("mtp.hidden", &row_pending);
                    let mtp_logits = mtp_forward(&state, cfg, &mtp, head, &embed_c2, &hidden_buf, pending_pos);
                    Some(argmax(&mtp_logits) as u32)
                } else {
                    None
                };
            }
        } else {
            // MISMATCH (or the bootstrap pass, `pending == None`): only
            // `candidate_1` is confirmed - discard whatever `pending` was
            // (never included in a future pass's input sequence at all, the
            // "no persistent-KV rollback needed" simplification this
            // function's own doc calls out). A fresh pending guess reuses
            // THIS pass's own `row_last_confirmed` (already computed above,
            // no extra pass) combined with the newly-known `candidate_1`.
            pending = if generated.len() < max_new {
                let embed_c1 = embed_rows(outside, embed_name, &[candidate_1], d)?;
                let hidden_buf = state.gpu.storage_init("mtp.hidden", &row_last_confirmed);
                let mtp_logits = mtp_forward(&state, cfg, &mtp, head, &embed_c1, &hidden_buf, last_confirmed_pos);
                Some(argmax(&mtp_logits) as u32)
            } else {
                None
            };
        }
    }

    generated.truncate(max_new);
    Ok((tok.decode(&generated), passes))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Design decision 2 rests on GDN's chunked forward being strictly
    /// causal: a real position's output must not depend on whatever dummy
    /// content fills LATER, end-padded positions, since [`generate`] pads a
    /// growing sequence and reads back only the last real position. Both
    /// `model::gdn`'s own module doc (intra-chunk masking is strict-lower
    /// `j < i`; cross-chunk state only ever flows forward through a
    /// sequential loop) and the depthwise causal conv1d `model::gdn_mixer`
    /// dispatches (`pad: kw - 1`, left-padded) already say this holds - this
    /// test proves it directly against the REAL code path [`generate`]
    /// drives ([`StreamState::build_layer`] + [`StreamState::layer_forward`],
    /// not a hand-rolled parallel replay), rather than trusting the doc
    /// comments alone.
    ///
    /// If this ever fails, `generate`'s whole end-padding strategy is
    /// unsound and must not ship as-is - see this module's own doc.
    #[test]
    fn gdn_end_padding_does_not_change_real_position_outputs() {
        let cfg = Qwen35Config::tiny();
        assert_eq!(cfg.layer_types()[0], LayerType::Linear, "test assumes layer 0 is Gated DeltaNet");
        let init = crate::init::init_weights(&cfg, 42);
        let d = cfg.d_model;

        let t_real = 5u32;
        let t_pad = GDN_DECODE_CHUNK;

        let state = StreamState::new(Gpu::new_cpu(crate::model::pipelines()), &cfg, t_pad);
        let layer = state.build_layer(&cfg, 0, &init);

        // Two residual streams that agree on the first t_real rows but
        // differ arbitrarily after - two different choices for "what fills
        // the padding".
        let mut rng_shared = data::rng::Lcg::new(1);
        let real_rows = rng_shared.vec_scaled((t_real * d) as usize, 0.5);
        let mut rng_a = data::rng::Lcg::new(2);
        let mut rng_b = data::rng::Lcg::new(3);
        let mut xa = real_rows.clone();
        xa.extend(rng_a.vec_scaled(((t_pad - t_real) * d) as usize, 0.5));
        let mut xb = real_rows.clone();
        xb.extend(rng_b.vec_scaled(((t_pad - t_real) * d) as usize, 0.5));

        let xa_buf = state.gpu.storage_init("xa", &xa);
        let xb_buf = state.gpu.storage_init("xb", &xb);
        let out_a = state.layer_forward(&cfg, &layer, &xa_buf, t_pad);
        let out_b = state.layer_forward(&cfg, &layer, &xb_buf, t_pad);

        let got_a = state.gpu.read(&out_a, (t_pad * d) as usize);
        let got_b = state.gpu.read(&out_b, (t_pad * d) as usize);

        let real_len = (t_real * d) as usize;
        let mut max_diff = 0f32;
        for i in 0..real_len {
            max_diff = max_diff.max((got_a[i] - got_b[i]).abs());
        }
        assert!(max_diff < 1e-5, "real-position outputs differ (max_diff={max_diff}) when only the END padding content changes - GDN end-padding is NOT causal");

        // Sanity: the padded TAIL genuinely differs between the two runs, so
        // this test can actually detect a leak (not vacuously pass because
        // the two padding fills happened to coincide).
        assert_ne!(&got_a[real_len..], &got_b[real_len..], "test setup bug: the two padding fills produced identical padded output");
    }
}
