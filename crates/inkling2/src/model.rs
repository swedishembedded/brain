// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements looped mixture-of-experts language models with depth-attention residuals for its
// clients. If your team needs expertise in running such a model from its published release on any device then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The Inkling 2 forward: the layer recipe over brain's shared operators, in the reference's order.
//!
//! Per token the forward embeds and RMS-normalises, then runs the visits of the [`VisitPlan`] (prelude, the core once
//! per pass, coda). The carried hidden state is the partial sum of the current depth-attention block; a layer at a
//! block boundary retains its incoming partial as a block state and restarts the residual from its attention output.
//! A visit:
//!
//! 1. reads its attention input by depth attention over the retained block states and the partial (three queries,
//!    q/k/v, on a full layer; layer 0 reads the embedding as it is);
//! 2. on a core pass after the first, mixes in the previous pass's messages through the message gates;
//! 3. attends: queries scaled and L2-clipped, keys and values through their short convolution (static on sliding
//!    layers, with per-token taps on full layers) then L2-clipped / RMS-normalised, a hidden-conditioned distance-bank
//!    bias and sink logits, a window on sliding layers and the log-position scale on full ones, a sigmoid output gate;
//! 4. convolves the attention output, adds it to the residual (or restarts there), reads the MoE input by depth
//!    attention (message-gated on later passes), runs the router and the experts, convolves their output and adds it.
//!
//! Passes after the first restart from the prelude's output and block states; the previous pass reaches them only
//! through messages: per core layer, four learned queries attend over that pass's own block states and its final
//! partial. The readout mixes every block state of the last pass with the final partial, normalises, and applies the
//! untied head with its soft cap over the valid vocabulary.
//!
//! Numerics are the reference's (I2.4): expert weights are read as published (NVFP4, MXFP8, or float) and multiplied
//! with unquantised f32 activations; every other weight keeps its stored tier. On request
//! ([`LinearActivations::Declared`], `--linear-activations declared`) the expert products take their activations as the
//! release declares them (`required_activation_quantization`: NVFP4 per-token four-over-six on the NVFP4 layers, MXFP8
//! on the MXFP8 ones), the numerics the model was trained with, through [`model::act_quant`].

use std::path::Path;

use checkpoint::packed::{ActivationScheme, BoundedWorkspace, CodecRegistry, HfQuant, MemoryBudget, PackedReader, PackedSource, PackedTensorReader, QuantEncoding, SlicedReader};
use checkpoint::TensorSource;
use gpu_core::{f, DeviceBuffer, Dispatch, Gpu, Step};
use model::depth_attn::{DepthAttn, DepthAttnIds};
use model::flash_ext::{self, AttnInputs, AttnOptions, Contiguous, DistanceBias, FlashExtIds, HeadShape, LogPositionScale};
use model::glu::{expert_fwd_glu, glu_act_step, GatedAct, GluIds, GluLayout, GluScratch, GluSpec, GluWeights};
use model::head_transform::HeadTransform;
use model::looped::{Attention, LoopShape, Segment, Visit, VisitPlan};
use model::message_gate::{MessageGate, MessageGateIds, MessageGateWeights};
use model::moe::{MoeIds, MoeShape};
use model::ops::{upload_micro_bank, upload_preserved, MicroBank, MicroRoute, Ops, Weight};
use model::paged::BlockTable;
use model::policy::LinearActivations;
use model::saved::RetainedStates;
use model::serve::visit_state::VisitPool;
use model::short_conv::{conv_state_tail_step, short_conv_fwd, short_conv_step, BaseLayout, DynTaps, ShortConvIds, ShortConvShape};

use crate::config::{Inkling2Config, LayerKind};
use crate::names::{self, GateSite, LayerNames};

/// The RMS norm epsilon of the depth-attention stacks (the reference's module default, not the config's).
pub(crate) const DEPTH_EPS: f32 = 1e-6;
/// Added to a vector's L2 norm before the clip divides by it.
pub(crate) const CLIP_EPS: f32 = 1e-6;
/// Host staging a single tile upload may take.
const STAGING_BYTES: u64 = 64 << 20;

/// Every kernel the forward and the served engine ([`crate::serve::Engine`]) dispatch, for the `Gpu` its [`Ops`] is
/// built on.
pub fn kernels() -> Vec<(&'static str, &'static str)> {
    let mut list = model::ops::kernel_list().to_vec();
    let mut more: Vec<(&'static str, &'static str)> = vec![
        ("dyn_group_conv1d", kernels::DYN_GROUP_CONV1D),
        ("dyn_group_conv1d_dx", kernels::DYN_GROUP_CONV1D_DX),
        ("dyn_group_conv1d_dbase", kernels::DYN_GROUP_CONV1D_DBASE),
        ("dyn_group_conv1d_ddyn", kernels::DYN_GROUP_CONV1D_DDYN),
        ("dyn_group_conv1d_step", kernels::DYN_GROUP_CONV1D_STEP),
        ("conv_state_tail", kernels::CONV_STATE_TAIL),
        ("glu_act", kernels::GLU_ACT),
        ("glu_act_bwd", kernels::GLU_ACT_BWD),
        ("router_gate_sigmoid", kernels::ROUTER_GATE_SIGMOID),
        ("router_topk_compact", kernels::ROUTER_TOPK_COMPACT),
        ("moe_group_combine", kernels::MOE_GROUP_COMBINE),
        ("scale_add", kernels::SCALE_ADD),
        ("silu_mul", kernels::SILU_MUL),
        ("rmsnorm", kernels::RMSNORM),
        ("l2clip_scale", kernels::L2CLIP_SCALE),
        ("sigmoid", kernels::SIGMOID),
    ];
    more.extend(flash_ext::pipelines());
    more.extend_from_slice(model::depth_attn::kernels());
    more.extend_from_slice(model::saved::retained_kernels());
    more.extend_from_slice(model::message_gate::kernels());
    more.extend(model::ops::micro_bank_kernels());
    more.extend(model::act_quant::KERNELS);
    more.extend(model::head_transform::KERNELS);
    // The served engine's: the visit-state pool and the device readout.
    more.extend(model::serve::visit_state::VisitPool::<Vec<f32>>::pipelines());
    more.extend([("argmax_part", kernels::ARGMAX_PART), ("argmax_final", kernels::ARGMAX_FINAL), ("topk_extract_step", kernels::TOPK_EXTRACT_STEP)]);
    model::ops::extend_kernel_list(&mut list, &more);
    list
}

/// Receives the intermediate tensors of a forward, keyed as the reference golden names them (`embed`,
/// `visit{v}.in`, `visit{v}.attn_in`, ..., `final_norm`, `logits`), each as row-major f32.
pub trait Probe {
    /// Whether `key` is wanted; reading a tensor back synchronises the device, so a probe that wants little costs
    /// little.
    fn wants(&self, _key: &str) -> bool {
        true
    }
    fn record(&mut self, key: &str, values: Vec<f32>);
}

/// The per-sequence state of the served engine: what a prefill entry carries is the final normalised row of its last
/// position.
pub(crate) type Pool = VisitPool<Vec<f32>>;

/// Which rows a pass over the decoder runs and where their attention and convolution state lives. Everything else in
/// the recipe is row-wise (depth attention, messages, router, experts), so one recipe serves all three.
pub(crate) enum Rows<'a> {
    /// One whole sequence from position 0 that keeps no state: the reference forward.
    Whole,
    /// One prompt chunk of one served sequence at positions `start ..`.
    Chunk { pool: &'a Pool, table: &'a BlockTable, start: u32 },
    /// One new token of each served sequence, at the last position its table backs.
    Decode { pool: &'a Pool, tables: &'a [&'a BlockTable] },
}

/// The taps of one short convolution.
enum ConvTaps<'b> {
    /// A published depthwise weight `[channels, taps]`: `(conv(x) + x) / sqrt(taps + 1)`.
    Static(&'b DeviceBuffer),
    /// Static taps `[taps, channels]` plus per-token taps read from a projection output: `(dyn(x) + x) / sqrt(2 taps + 1)`.
    Dynamic { base: &'b DeviceBuffer, taps: &'b DeviceBuffer, layout: DynTaps },
}

struct Kernels {
    flash: FlashExtIds,
    conv: ShortConvIds,
    depth: DepthAttnIds,
    gate: MessageGateIds,
    glu: GluIds,
    moe: MoeIds,
    topk: usize,
    combine: usize,
    rms: usize,
    rms_nogain: usize,
    l2clip: usize,
    sigmoid: usize,
    mul: usize,
    add: usize,
    copy: usize,
    head: usize,
}

impl Kernels {
    fn resolve(g: &Gpu) -> Result<Kernels, String> {
        let k = |name: &str| g.kernel_index(name).ok_or_else(|| format!("inkling2: kernel {name} is not registered (build the Gpu on inkling2::model::kernels())"));
        Ok(Kernels {
            flash: FlashExtIds::resolve(g),
            conv: ShortConvIds {
                fwd: k("dyn_group_conv1d")?,
                dx: k("dyn_group_conv1d_dx")?,
                dbase: k("dyn_group_conv1d_dbase")?,
                ddyn: k("dyn_group_conv1d_ddyn")?,
                step: k("dyn_group_conv1d_step")?,
                state_tail: k("conv_state_tail")?,
            },
            depth: DepthAttnIds::resolve(g)?,
            gate: MessageGateIds::resolve(g)?,
            glu: GluIds { act: k("glu_act")?, act_bwd: k("glu_act_bwd")? },
            moe: MoeIds { router_gate: k("router_gate_sigmoid")?, linear_gated: k("moe_linear_gated")?, silu_mul: k("silu_mul")?, scale_add: k("scale_add")? },
            topk: k("router_topk_compact")?,
            combine: k("moe_group_combine")?,
            rms: k("rmsnorm")?,
            rms_nogain: k("rmsnorm_nogain")?,
            l2clip: k("l2clip_scale")?,
            sigmoid: k("sigmoid")?,
            mul: k("mul")?,
            add: k("add2")?,
            copy: k("splice_at")?,
            head: k("logit_cap_mask")?,
        })
    }
}

/// A K or V projection's short convolution.
enum KvConv {
    /// A depthwise weight `[channels, taps]` as published (last tap on the current token).
    Static(DeviceBuffer),
    /// Static taps `[taps, channels]` (tap 0 the current token) plus per-token taps from `proj`, whose output rows are
    /// reordered at load to the `[tap][kv head]` layout the convolution reads.
    Dynamic { taps: DeviceBuffer, proj: Weight },
}

/// A layer's routed experts: dense f32 banks, or the published microscaled banks read in place with the activation
/// quantization the release declares for each product (taken only under [`LinearActivations::Declared`]).
enum Experts {
    Dense { w13: DeviceBuffer, w2: DeviceBuffer },
    Micro { w13: MicroBank, w2: MicroBank, act: [Option<ActivationScheme>; 2] },
}

struct GateWeights {
    down: DeviceBuffer,
    forget: DeviceBuffer,
    gain: DeviceBuffer,
}

struct LayerWeights {
    /// Depth queries of the attention input (`[nq, hidden]`), none at layer 0.
    attn_queries: Option<DeviceBuffer>,
    mlp_queries: DeviceBuffer,
    /// The input norm gain: one on a sliding layer, the q, k and v paths' on a full one.
    input_norms: Vec<DeviceBuffer>,
    wq: Weight,
    wk: Weight,
    wv: Weight,
    wo: Weight,
    gate_attn: Weight,
    sink: Weight,
    wr: Weight,
    q_scale: DeviceBuffer,
    rel_scale: DeviceBuffer,
    rel_bank: DeviceBuffer,
    kv_conv: [KvConv; 2],
    attn_conv: DeviceBuffer,
    mlp_norm: DeviceBuffer,
    mlp_conv: DeviceBuffer,
    router: Weight,
    router_bias: DeviceBuffer,
    experts: Experts,
    /// The q, k, v and MLP message gates of a core layer.
    gates: Option<Vec<GateWeights>>,
}

struct Weights {
    embed: Weight,
    embed_norm: DeviceBuffer,
    final_norm: DeviceBuffer,
    unembed: Weight,
    output_query: DeviceBuffer,
    layers: Vec<LayerWeights>,
    /// Per core layer, its `[4, hidden]` message queries in q, k, v, MLP order.
    message_queries: Vec<DeviceBuffer>,
}

/// An Inkling 2 decoder resident on one device.
pub struct Inkling2 {
    cfg: Inkling2Config,
    plan: VisitPlan,
    ops: Ops,
    k: Kernels,
    w: Weights,
    /// The declared activation quantizations brain does not implement, by tensor: they refuse
    /// [`LinearActivations::Declared`] and nothing else.
    unsupported_activations: Vec<String>,
}

/// The visit plan of `cfg`: the looped core between its prelude and coda, or the whole stack visited once.
pub fn visit_plan(cfg: &Inkling2Config) -> Result<VisitPlan, String> {
    let n = cfg.n_layers as usize;
    let shape = match cfg.looped {
        Some(l) => LoopShape { prelude: l.start as usize, core: l.layers() as usize, coda: n - l.end as usize, passes: l.passes as usize },
        None => LoopShape::whole_stack(n, 1),
    };
    VisitPlan::new(shape, |layer| match cfg.kind(layer as u32) {
        LayerKind::Sliding => Attention::Window(cfg.window),
        LayerKind::Full => Attention::Full,
    })
}

/// How each boundary visit retains its incoming partial: `Some((slot, fresh))`, `fresh` false when the slot already
/// holds it (the first core layer of a pass after the first sees the prelude's output again), and the slots in use.
pub(crate) fn block_slots(cfg: &Inkling2Config, plan: &VisitPlan) -> (Vec<Option<(u32, bool)>>, u32) {
    let mut next = 0u32;
    let mut core_first = None;
    let out = plan
        .visits()
        .iter()
        .map(|v| {
            if !cfg.is_block_boundary(v.layer as u32) {
                return None;
            }
            let first_core = v.segment == Segment::Core && v.layer == plan.shape().prelude;
            if first_core && v.pass > 0 {
                return Some((core_first.expect("pass 0 retained the prelude's output"), false));
            }
            let slot = next;
            next += 1;
            if first_core {
                core_first = Some(slot);
            }
            Some((slot, true))
        })
        .collect();
    (out, next)
}

/// The checkpoint directory `dir` opened for reading: its validated config, the tensors (held to the name map first, so
/// a file the config does not describe is refused by name before any weight is read) with every microscaled expert
/// bank also viewed expert by expert (`{bank}.{e}`), the names of those banks, and the quantization declaration.
pub(crate) fn open_checkpoint(dir: &Path) -> Result<(Inkling2Config, PackedSource, Vec<String>, Option<HfQuant>), String> {
    let cfg = Inkling2Config::from_dir(dir)?;
    let reader = PackedReader::open(dir)?;
    names::check(&cfg, &reader)?;
    let declared = reader.declaration().cloned();
    let mut micro = Vec::new();
    for l in 0..cfg.n_layers {
        let n = LayerNames::new(l);
        for name in [n.w13(), n.w2()] {
            let e = reader.descriptor(&name)?.encoding;
            if e == QuantEncoding::nvfp4_per_slab() || e == QuantEncoding::mxfp8() {
                micro.push(name);
            }
        }
    }
    let mut sliced = SlicedReader::new(Box::new(reader));
    for name in &micro {
        sliced = sliced.with_slab_views(name, |e| format!("{name}.{e}"))?;
    }
    let src = PackedSource::identity(Box::new(sliced), CodecRegistry::reference())?;
    Ok((cfg, src, micro, declared))
}

/// A dynamic convolution's tap projection rows reordered between the published `[kv head][tap]` order and the
/// `[tap][kv head]` order the convolution reads (`to_tap_major`), or back. Each row is `width` values.
pub(crate) fn reorder_tap_rows(rows: &[f32], kv_heads: usize, taps: usize, width: usize, to_tap_major: bool) -> Vec<f32> {
    let mut out = vec![0f32; rows.len()];
    for h in 0..kv_heads {
        for j in 0..taps {
            let (published, tap_major) = ((h * taps + j) * width, (j * kv_heads + h) * width);
            let (from, to) = if to_tap_major { (published, tap_major) } else { (tap_major, published) };
            out[to..to + width].copy_from_slice(&rows[from..from + width]);
        }
    }
    out
}

/// Values a host read takes from the source at a time.
const READ_CHUNK: usize = 1 << 20;

/// `name` as a fresh f32 device buffer, streamed from the source a bounded chunk at a time.
pub(crate) fn upload_f32(g: &Gpu, src: &dyn TensorSource, name: &str) -> Result<DeviceBuffer, String> {
    let n = src.numel(name).ok_or_else(|| format!("inkling2: tensor {name} is missing"))?;
    let buf = g.storage(n as u64);
    if !src.with_tensor_chunks(name, READ_CHUNK, &mut |off, chunk| g.write_f32_at(&buf, off, chunk)) {
        return Err(format!("inkling2: tensor {name} is missing"));
    }
    src.advise_drop(name);
    Ok(buf)
}

/// A small tensor's values on the host, for a reordering at load (the dynamic convolutions' tap projections and the
/// message queries), read a bounded chunk at a time.
pub(crate) fn read_small(src: &dyn TensorSource, name: &str) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(src.numel(name).unwrap_or(0));
    if !src.with_tensor_chunks(name, READ_CHUNK, &mut |_, chunk| out.extend_from_slice(chunk)) {
        return Err(format!("inkling2: tensor {name} is missing"));
    }
    src.advise_drop(name);
    Ok(out)
}

/// The activation quantization the release declares for each expert projection's product (`w13` reads the hidden rows,
/// `w2` the expert activation), by tensor name, and the declarations brain does not implement (or whose block does not
/// divide the product's input width), by name. A release without a per-tensor declaration declares none.
fn declared_activations(declared: Option<&HfQuant>, cfg: &Inkling2Config) -> (std::collections::HashMap<String, ActivationScheme>, Vec<String>) {
    let (mut schemes, mut unsupported) = (std::collections::HashMap::new(), Vec::new());
    for l in 0..cfg.n_layers {
        let n = LayerNames::new(l);
        for (name, width) in [(n.w13(), cfg.hidden), (n.w2(), cfg.moe_ff)] {
            let Some(act) = declared.and_then(|q| q.declaration(&name)).and_then(|d| d.activation.as_ref()) else { continue };
            match act.scheme() {
                Ok(scheme) if width.is_multiple_of(scheme.block()) => {
                    schemes.insert(name, scheme);
                }
                Ok(scheme) => unsupported.push(format!("{name}: {scheme:?} activations need a multiple of {} values, the product reads {width}", scheme.block())),
                Err(e) => unsupported.push(format!("{name}: {e}")),
            }
        }
    }
    (schemes, unsupported)
}

impl Inkling2 {
    /// Load the checkpoint directory `dir` (`config.json`, the safetensors and, when present, `hf_quant_config.json`)
    /// onto `ops`, whose `Gpu` registers [`kernels()`]. The header is held to the name map first, so a file the config
    /// does not describe is refused by name before any weight is read. The vision tower's tensors are checked, not
    /// loaded. The activation operand `ops` is set to ([`Ops::set_linear_activations`]) is held to what this model can
    /// run ([`Self::set_linear_activations`]).
    pub fn load(ops: Ops, dir: &Path) -> Result<Inkling2, String> {
        let (cfg, src, micro, declared) = open_checkpoint(dir)?;
        let (activations, unsupported_activations) = declared_activations(declared.as_ref(), &cfg);
        let k = Kernels::resolve(ops.gpu())?;
        let w = Weights::load(&ops, &cfg, &src, &micro, &activations)?;
        let plan = visit_plan(&cfg)?;
        let choice = ops.linear_activations();
        let mut m = Inkling2 { cfg, plan, ops, k, w, unsupported_activations };
        m.set_linear_activations(choice)?;
        Ok(m)
    }

    /// Choose the activation operand of the expert products: [`LinearActivations::Auto`] keeps the reference numerics
    /// (f32 activations), [`LinearActivations::Declared`] quantizes them as the release declares. `Declared` on a
    /// checkpoint that declares no activation quantization of its experts, or one brain does not implement, is refused
    /// by name, as are `bf16` and `e4m3`, which this model has no product for.
    pub fn set_linear_activations(&mut self, choice: LinearActivations) -> Result<(), String> {
        match choice {
            LinearActivations::Auto => {}
            LinearActivations::Declared => {
                if let Some(first) = self.unsupported_activations.first() {
                    return Err(format!("inkling2: --linear-activations declared: {first}"));
                }
                let declares = self.w.layers.iter().any(|l| matches!(&l.experts, Experts::Micro { act, .. } if act.iter().any(Option::is_some)));
                if !declares {
                    return Err("inkling2: --linear-activations declared: this checkpoint declares no activation quantization for its experts".into());
                }
            }
            other => return Err(format!("inkling2: --linear-activations {other} is not supported: the expert products take f32 activations (auto) or the declared quantization (declared)")),
        }
        self.ops.set_linear_activations(choice);
        Ok(())
    }

    pub fn config(&self) -> &Inkling2Config {
        &self.cfg
    }

    pub fn plan(&self) -> &VisitPlan {
        &self.plan
    }

    pub fn ops(&self) -> &Ops {
        &self.ops
    }

    /// Choose the arithmetic of the dense bf16-weight products (see [`Ops::set_gemm_precision`]): `Auto`, the default,
    /// lets a tensor-core device round their operands, `Fp32Reference` keeps every product in binary32, what a parity
    /// gate against the f32 reference needs. The expert banks' products are f32 either way.
    pub fn set_gemm_precision(&mut self, precision: gpu_core::provider::library::GemmPrecision) {
        self.ops.set_gemm_precision(precision);
    }

    /// The logits of every position of `tokens` (one sequence, from position 0), row-major `[tokens, valid_vocab]`,
    /// soft-capped as the head declares.
    pub fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        self.run(tokens, None)
    }

    /// [`Self::forward`], handing every intermediate the reference golden records to `probe`.
    pub fn forward_probed(&self, tokens: &[u32], probe: &mut dyn Probe) -> Result<Vec<f32>, String> {
        self.run(tokens, Some(probe))
    }

    fn g(&self) -> &Gpu {
        self.ops.gpu()
    }

    /// Run the queued steps and hand `buf`'s first `n` values to the probe.
    fn tap(&self, s: &mut Vec<Step>, probe: &mut Option<&mut dyn Probe>, key: &str, buf: &DeviceBuffer, n: u32) {
        if let Some(p) = probe.as_deref_mut().filter(|p| p.wants(key)) {
            self.flush(s);
            p.record(key, self.g().read(buf, n as usize));
        }
    }

    pub(crate) fn flush(&self, s: &mut Vec<Step>) {
        if !s.is_empty() {
            self.g().submit(&[], s);
            s.clear();
        }
    }

    /// `tokens` as a row count, refused when empty, past the model's positions or outside its embedding rows.
    pub(crate) fn check_tokens(&self, tokens: &[u32]) -> Result<u32, String> {
        let cfg = &self.cfg;
        let t = u32::try_from(tokens.len()).map_err(|_| "inkling2: too many tokens".to_string())?;
        if t == 0 {
            return Err("inkling2: an empty sequence has no logits".into());
        }
        if u64::from(t) > cfg.max_positions {
            return Err(format!("inkling2: {t} tokens exceed the model's {} positions", cfg.max_positions));
        }
        if let Some(bad) = tokens.iter().find(|&&id| id >= cfg.vocab) {
            return Err(format!("inkling2: token {bad} is outside the {} embedding rows", cfg.vocab));
        }
        Ok(t)
    }

    fn run(&self, tokens: &[u32], mut probe: Option<&mut dyn Probe>) -> Result<Vec<f32>, String> {
        let t = self.check_tokens(tokens)?;
        let (cfg, d) = (&self.cfg, self.cfg.hidden);
        let mut s = Vec::new();
        let x = self.embed(&mut s, tokens);
        self.tap(&mut s, &mut probe, "embed", &x, t * d);
        let normed = self.trunk(&mut s, &Rows::Whole, x, t, &mut probe, None)?;
        let vocab = cfg.vocab;
        let logits = self.head(&mut s, &normed, t);
        self.flush(&mut s);
        let all = self.g().read(&logits, (t * vocab) as usize);
        let valid = cfg.valid_vocab as usize;
        let out: Vec<f32> = all.chunks(vocab as usize).flat_map(|row| row[..valid].iter().copied()).collect();
        if let Some(p) = probe {
            p.record("logits", out.clone());
        }
        Ok(out)
    }

    /// The normalised embedding rows `[t, hidden]` of `tokens` (checked by the caller).
    pub(crate) fn embed(&self, s: &mut Vec<Step>, tokens: &[u32]) -> DeviceBuffer {
        let (g, d, t) = (self.g(), self.cfg.hidden, tokens.len() as u32);
        let tok = g.storage(u64::from(t));
        g.write(&tok, tokens);
        let raw = g.storage(u64::from(t * d));
        self.ops.embed(s, &self.w.embed, &tok, t, &raw);
        self.rms(s, &raw, &self.w.embed_norm, t)
    }

    /// The head over `t` final normalised rows: logits `[t, vocab]` (the stored rows), soft-capped, the padding rows
    /// floored so no readout picks them.
    pub(crate) fn head(&self, s: &mut Vec<Step>, normed: &DeviceBuffer, t: u32) -> DeviceBuffer {
        let logits = self.proj(s, &self.w.unembed, normed, t);
        s.extend(self.head_transform().step(self.g(), self.k.head, &logits, t, self.cfg.vocab));
        logits
    }

    /// The soft cap and the valid-vocabulary bound the head applies.
    pub(crate) fn head_transform(&self) -> HeadTransform {
        HeadTransform { softcap: Some(self.cfg.logit_cap), valid_vocab: Some(self.cfg.valid_vocab) }
    }

    /// Every visit over the `t` rows `x` (normalised embeddings) and the readout's depth mix and final norm: the final
    /// normalised rows `[t, hidden]`. With `routes`, each visit's `[t, top_k]` expert ids (ascending per row) are pushed
    /// in visit order.
    pub(crate) fn trunk(&self, s: &mut Vec<Step>, rows: &Rows, x: DeviceBuffer, t: u32, probe: &mut Option<&mut dyn Probe>, mut routes: Option<&mut Vec<DeviceBuffer>>) -> Result<DeviceBuffer, String> {
        let (cfg, g, d) = (&self.cfg, self.g(), self.cfg.hidden);
        let mut x = x;
        let (retain, capacity) = block_slots(cfg, &self.plan);
        let store = RetainedStates::new(g, capacity.max(1), t, d, false)?;
        let mut stack: Vec<u32> = Vec::new();
        let mut prelude = (Vec::new(), x.clone());
        // The current pass's own block states (after the first core block, which is the prelude's output) and,
        // once a pass ends, the previous pass's with its final partial: what the messages attend over.
        let mut pass_blocks: Vec<u32> = Vec::new();
        let mut previous: Option<(Vec<u32>, DeviceBuffer)> = None;
        let core_start = self.plan.shape().prelude;
        for v in self.plan.visits() {
            if v.segment == Segment::Core && v.layer == core_start {
                if v.pass == 0 {
                    prelude = (stack.clone(), x.clone());
                } else {
                    previous = Some((std::mem::take(&mut pass_blocks), x.clone()));
                    (stack, x) = (prelude.0.clone(), prelude.1.clone());
                }
                pass_blocks.clear();
            }
            let messages = match (&previous, v.segment == Segment::Core && v.pass > 0) {
                (Some((blocks, partial)), true) => {
                    let c = v.layer - core_start;
                    let da = DepthAttn::new(&store, blocks, 4, DEPTH_EPS)?;
                    let out = g.storage(da.out_len());
                    let stats = g.storage(da.stats_len());
                    s.push(da.forward(g, &self.k.depth, &store, partial, &self.w.message_queries[c], &out, &stats));
                    self.tap(s, probe, &format!("visit{}.messages", v.index), &out, 4 * t * d);
                    Some(out)
                }
                _ => None,
            };
            let key = |name: &str| format!("visit{}.{name}", v.index);
            self.tap(s, probe, &key("in"), &x, t * d);
            if probe.as_deref().is_some_and(|p| p.wants(&key("block_states"))) {
                self.flush(s);
                let states: Vec<f32> = stack.iter().flat_map(|&slot| store.read_value(g, slot)).collect();
                if let Some(p) = probe.as_deref_mut() {
                    p.record(&key("block_states"), states);
                }
            }
            let boundary = retain[v.index];
            x = self.visit(s, rows, v, &x, &store, &mut stack, boundary, messages.as_ref(), t, probe, routes.as_deref_mut())?;
            if let Some((slot, _)) = boundary {
                if v.segment == Segment::Core && v.layer != core_start {
                    pass_blocks.push(slot);
                }
            }
            self.tap(s, probe, &key("out"), &x, t * d);
        }

        let da = DepthAttn::new(&store, &stack, 1, DEPTH_EPS)?;
        let mixed = g.storage(da.out_len());
        let stats = g.storage(da.stats_len());
        s.push(da.forward(g, &self.k.depth, &store, &x, &self.w.output_query, &mixed, &stats));
        self.tap(s, probe, "final_in", &mixed, t * d);
        let normed = self.rms(s, &mixed, &self.w.final_norm, t);
        self.tap(s, probe, "final_norm", &normed, t * d);
        Ok(normed)
    }

    /// One visit: the layer recipe over the partial `x`, returning the next partial.
    #[allow(clippy::too_many_arguments)]
    fn visit(
        &self,
        s: &mut Vec<Step>,
        rows: &Rows,
        v: &Visit,
        x: &DeviceBuffer,
        store: &RetainedStates,
        stack: &mut Vec<u32>,
        boundary: Option<(u32, bool)>,
        messages: Option<&DeviceBuffer>,
        t: u32,
        probe: &mut Option<&mut dyn Probe>,
        routes: Option<&mut Vec<DeviceBuffer>>,
    ) -> Result<DeviceBuffer, String> {
        let (cfg, g, d) = (&self.cfg, self.g(), self.cfg.hidden);
        let layer = v.layer as u32;
        let lw = &self.w.layers[v.layer];
        let key = |name: &str| format!("visit{}.{name}", v.index);
        let n = t * d;

        // 1. The attention inputs: q, k, v rows (one shared input on a sliding layer).
        let nq = cfg.attn_queries(layer);
        let mut inputs: Vec<DeviceBuffer> = match &lw.attn_queries {
            None => vec![x.clone()],
            Some(q) => {
                let da = DepthAttn::new(store, stack, nq, DEPTH_EPS)?;
                let out = g.storage(da.out_len());
                let stats = g.storage(da.stats_len());
                s.push(da.forward(g, &self.k.depth, store, x, q, &out, &stats));
                self.tap(s, probe, &key("attn_in"), &out, nq * n);
                (0..nq).map(|i| self.rows(s, &out, i * n, n)).collect()
            }
        };
        let message = |i: u32, s: &mut Vec<Step>| messages.map(|m| self.rows(s, m, i * n, n));
        let gated = |s: &mut Vec<Step>, probe: &mut Option<&mut dyn Probe>, site: usize, h: &DeviceBuffer, name: &str| -> DeviceBuffer {
            match (message(site as u32, s), &lw.gates) {
                (Some(m), Some(gates)) => {
                    let out = self.message_gate(s, &gates[site], h, &m, t);
                    self.tap(s, probe, &key(name), &out, n);
                    out
                }
                _ => h.clone(),
            }
        };
        let (qh, kh, vh) = match cfg.kind(layer) {
            LayerKind::Sliding => {
                let h = inputs.swap_remove(0);
                if messages.is_some() {
                    let q = gated(s, probe, 0, &h, "gated_q");
                    let k = gated(s, probe, 1, &h, "gated_k");
                    let v = gated(s, probe, 2, &h, "gated_v");
                    let norm = &lw.input_norms[0];
                    (self.rms(s, &q, norm, t), self.rms(s, &k, norm, t), self.rms(s, &v, norm, t))
                } else {
                    let hn = self.rms(s, &h, &lw.input_norms[0], t);
                    (hn.clone(), hn.clone(), hn)
                }
            }
            LayerKind::Full => {
                let pick = |i: usize| inputs.get(i).unwrap_or(&inputs[0]).clone();
                let q = gated(s, probe, 0, &pick(0), "gated_q");
                let k = gated(s, probe, 1, &pick(1), "gated_k");
                let v = gated(s, probe, 2, &pick(2), "gated_v");
                (self.rms(s, &q, &lw.input_norms[0], t), self.rms(s, &k, &lw.input_norms[1], t), self.rms(s, &v, &lw.input_norms[2], t))
            }
        };

        // 2. Attention, its output convolution and the residual.
        let attn = self.attention(s, rows, v, &qh, &kh, &vh, t, probe)?;
        self.tap(s, probe, &key("attn"), &attn, n);
        let attn_conv = self.conv(s, rows, v.index, 2, &attn, ConvTaps::Static(&lw.attn_conv), t, d);
        self.conv_state(s, probe, &key("conv2"), &attn, t, d);
        self.tap(s, probe, &key("attn_conv"), &attn_conv, n);
        let residual = match boundary {
            Some((slot, fresh)) => {
                if fresh {
                    s.push(store.retain_step(g, slot, x));
                }
                stack.push(slot);
                attn_conv
            }
            None => {
                let r = g.storage(u64::from(n));
                s.push(g.step(self.k.add, &[x, &attn_conv, &r], &[n], n));
                r
            }
        };

        // 3. The MoE input, the experts, their output convolution and the residual.
        let da = DepthAttn::new(store, stack, 1, DEPTH_EPS)?;
        let mlp_in = g.storage(da.out_len());
        let stats = g.storage(da.stats_len());
        s.push(da.forward(g, &self.k.depth, store, &residual, &lw.mlp_queries, &mlp_in, &stats));
        self.tap(s, probe, &key("mlp_in"), &mlp_in, n);
        let mlp_in = gated(s, probe, 3, &mlp_in, "gated_mlp");
        let hn = self.rms(s, &mlp_in, &lw.mlp_norm, t);
        let moe = self.moe(s, v, &hn, t, !matches!(rows, Rows::Decode { .. }), probe, routes);
        self.tap(s, probe, &key("mlp"), &moe, n);
        let mlp_conv = self.conv(s, rows, v.index, 3, &moe, ConvTaps::Static(&lw.mlp_conv), t, d);
        self.conv_state(s, probe, &key("conv3"), &moe, t, d);
        self.tap(s, probe, &key("mlp_conv"), &mlp_conv, n);
        let out = g.storage(u64::from(n));
        s.push(g.step(self.k.add, &[&residual, &mlp_conv, &out], &[n], n));
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(&self, s: &mut Vec<Step>, rows: &Rows, v: &Visit, qh: &DeviceBuffer, kh: &DeviceBuffer, vh: &DeviceBuffer, t: u32, probe: &mut Option<&mut dyn Probe>) -> Result<DeviceBuffer, String> {
        let (cfg, g) = (&self.cfg, self.g());
        let layer = v.layer as u32;
        let lw = &self.w.layers[v.layer];
        let a = cfg.attn(layer);
        let key = |name: &str| format!("visit{}.{name}", v.index);
        let full = cfg.kind(layer) == LayerKind::Full;

        let q = self.proj(s, &lw.wq, qh, t);
        let q = self.l2clip(s, &q, Some(&lw.q_scale), t * a.heads, a.heads, a.head_dim, cfg.qk_clip);
        let mut kv = Vec::with_capacity(2);
        for (j, (input, w)) in [(kh, &lw.wk), (vh, &lw.wv)].into_iter().enumerate() {
            let pre = self.proj(s, w, input, t);
            self.conv_state(s, probe, &key(&format!("conv{j}")), &pre, t, a.kv_dim());
            let conv = match &lw.kv_conv[j] {
                KvConv::Static(base) => self.conv(s, rows, v.index, j, &pre, ConvTaps::Static(base), t, a.kv_dim()),
                KvConv::Dynamic { taps, proj } => {
                    let dyn_taps = self.proj(s, proj, input, t);
                    let layout = DynTaps { group_size: a.head_dim, stride: a.kv_heads * cfg.conv_taps, off: 0 };
                    self.conv(s, rows, v.index, j, &pre, ConvTaps::Dynamic { base: taps, taps: &dyn_taps, layout }, t, a.kv_dim())
                }
            };
            let normed = if j == 0 {
                self.l2clip(s, &conv, None, t * a.kv_heads, a.kv_heads, a.head_dim, cfg.qk_clip)
            } else {
                let y = g.storage(u64::from(t * a.kv_dim()));
                s.push(g.dispatch(self.k.rms_nogain, &[&conv, &y], &[a.head_dim, t * a.kv_heads, f(cfg.eps)], Dispatch::Workgroups(t * a.kv_heads)));
                y
            };
            kv.push(normed);
        }
        let rel = self.proj(s, &lw.wr, qh, t);
        let rel = self.l2clip(s, &rel, Some(&lw.rel_scale), t * a.heads, a.heads, cfg.d_rel, cfg.rel_clip);
        self.tap(s, probe, &key("rel_states"), &rel, t * a.heads * cfg.d_rel);
        let sink = self.proj(s, &lw.sink, qh, t);

        let heads = HeadShape { n_heads: a.heads, n_kv_heads: a.kv_heads, head_dim: a.head_dim };
        let opts = AttnOptions {
            scale: (a.head_dim as f32).powf(-0.5),
            window: (!full).then_some(cfg.window),
            sink: true,
            bias: Some(DistanceBias { extent: cfg.bias_extent(layer), d_rel: cfg.d_rel }),
            tau: full.then_some(LogPositionScale { alpha: cfg.log_alpha, n_floor: cfg.log_floor as f32, scales_bias: true }),
            documents: false,
        };
        let inputs = AttnInputs { sink: Some(&sink), rel: Some(&rel), bank: Some(&lw.rel_bank), doc_pos: None };
        let out = match rows {
            Rows::Whole => {
                let out = g.storage(u64::from(t * a.q_dim()));
                let lse = g.storage(u64::from(t * a.heads));
                s.extend(flash_ext::flash_ext_fwd(g, &self.k.flash, &heads, &opts, &inputs, Contiguous { b: 1, t, pos0: 0 }, &q, &kv[0], &kv[1], &out, &lse));
                out
            }
            Rows::Chunk { pool, table, start } => {
                self.flush(s);
                pool.attend_chunk(g, v.index, table, *start, &heads, &opts, &inputs, &q, &kv[0], &kv[1], t)
            }
            Rows::Decode { pool, tables } => {
                self.flush(s);
                pool.attend_decode(g, v.index, tables, &heads, &opts, &inputs, &q, &kv[0], &kv[1])
            }
        };

        // The output gate: sigmoid(gate_attn x) * attention, then the output projection.
        let gate = self.proj(s, &lw.gate_attn, qh, t);
        let n = t * a.q_dim();
        let sig = g.storage(u64::from(n));
        s.push(g.step(self.k.sigmoid, &[&gate, &sig], &[n], n));
        let gated = g.storage(u64::from(n));
        s.push(g.step(self.k.mul, &[&out, &sig, &gated], &[n], n));
        Ok(self.proj(s, &lw.wo, &gated, t))
    }

    /// The routed experts over `t` rows. `grouped`: the published banks multiply slots grouped by expert (one weight
    /// pass per expert, the prefill regime) instead of one gather per slot (decode); the arithmetic is the same.
    #[allow(clippy::too_many_arguments)]
    fn moe(&self, s: &mut Vec<Step>, v: &Visit, x: &DeviceBuffer, t: u32, grouped: bool, probe: &mut Option<&mut dyn Probe>, routes: Option<&mut Vec<DeviceBuffer>>) -> DeviceBuffer {
        let (cfg, g, d) = (&self.cfg, self.g(), self.cfg.hidden);
        let lw = &self.w.layers[v.layer];
        let key = |name: &str| format!("visit{}.{name}", v.index);
        let (e, k, ff) = (cfg.experts, cfg.top_k, cfg.moe_ff);
        let shape = MoeShape { rows: t, d_model: d, moe_ff: ff, n_experts: e, top_k: k };
        // Routing in binary32 at every precision tier: a top-k turns a rounded operand into a different expert.
        let logits = model::ops::fp32_kernels(|| self.proj(s, &lw.router, x, t));
        self.tap(s, probe, &key("router_logits"), &logits, t * e);
        let gate = g.storage(u64::from(t * e));
        let probs = g.storage(u64::from(t * e));
        let kind = model::moe::RouterKind::SigmoidNoAuxTc { n_group: 1, topk_group: 1, norm_topk_prob: true, routed_scaling: cfg.route_scale };
        s.push(model::moe::router_fwd_kind(g, &self.k.moe, kind, &shape, &logits, Some(&lw.router_bias), &gate, Some(&probs)));
        if let Some(p) = probe.as_deref_mut().filter(|p| p.wants(&key("router_indices"))) {
            self.flush(s);
            let dense = g.read(&gate, (t * e) as usize);
            let (mut ids, mut weights) = (Vec::new(), Vec::new());
            for row in dense.chunks(e as usize) {
                for (i, w) in row.iter().enumerate().filter(|(_, w)| **w != 0.0) {
                    ids.push(i as f32);
                    weights.push(*w);
                }
            }
            p.record(&key("router_indices"), ids);
            p.record(&key("router_weights"), weights);
        }
        let spec = GluSpec { act: GatedAct::SoftcapSigmoid { gate_cap: cfg.gate_cap, up_cap: cfg.up_cap }, layout: GluLayout::Interleaved };
        let out = g.storage(u64::from(t * d));
        // One slot per (token, chosen expert), in ascending expert order per token: the reference's own accumulation
        // order, and what a routing log records.
        let slots = t * k;
        let ids = (routes.is_some() || matches!(lw.experts, Experts::Micro { .. })).then(|| {
            let ids = g.storage(u64::from(slots));
            s.push(g.step(self.k.topk, &[&gate, &ids], &[t, e, k], t));
            ids
        });
        if let (Some(routes), Some(ids)) = (routes, &ids) {
            routes.push(ids.clone());
        }
        match &lw.experts {
            Experts::Dense { w13, w2 } => {
                let (gate_up, h, expert_out) = (g.storage(u64::from(t * 2 * ff)), g.storage(u64::from(t * ff)), g.storage(u64::from(t * d)));
                let scratch = GluScratch { gate_up: &gate_up, up: None, h: &h, expert_out: &expert_out };
                for ex in 0..e {
                    let w = GluWeights { gate_w: w13, up_w: None, down_w: w2, gate_off: ex * 2 * ff * d, down_off: ex * d * ff };
                    s.extend(expert_fwd_glu(g, &self.k.moe, &self.k.glu, &spec, &shape, x, &gate, &w, &scratch, &out, ex, ex > 0));
                }
            }
            Experts::Micro { w13, w2, act } => {
                let ids = ids.expect("the slots' experts are chosen for a published bank");
                // Under the declared tier each product reads its activation rows round-tripped through the declared
                // format: the token rows once (every chosen expert reads the same quantized row), each slot's
                // activation before the down product.
                let declared = self.ops.linear_activations().is_declared();
                let quantized = |s: &mut Vec<Step>, scheme: Option<ActivationScheme>, x: &DeviceBuffer, rows: u32, width: u32| -> Option<DeviceBuffer> {
                    let scheme = scheme.filter(|_| declared)?;
                    let q = g.storage(u64::from(rows * width));
                    s.push(model::act_quant::qdq_step(g, scheme, x, &q, rows, width).expect("a declared scheme's block divides the width (checked at load)"));
                    Some(q)
                };
                let (gate_up, h, y) = (g.storage(u64::from(slots * 2 * ff)), g.storage(u64::from(slots * ff)), g.storage(u64::from(slots * d)));
                const REGISTERED: &str = "the bank's kernels are registered";
                let xq = quantized(s, act[0], x, t, d);
                let x = xq.as_ref().unwrap_or(x);
                if grouped {
                    let route = MicroRoute::new(g, slots, e);
                    s.extend(route.steps(g, &ids).expect(REGISTERED));
                    s.push(w13.grouped(g, x, &route.tab, &route.perm, &gate_up, route.tiles(), k, e).expect(REGISTERED));
                    s.push(glu_act_step(g, &self.k.glu, &spec, slots, ff, &gate_up, None, &h));
                    let hq = quantized(s, act[1], &h, slots, ff);
                    s.push(w2.grouped(g, hq.as_ref().unwrap_or(&h), &route.tab, &route.perm, &y, route.tiles(), 1, e).expect(REGISTERED));
                } else {
                    s.push(w13.gather(g, x, &ids, &gate_up, slots, k).expect(REGISTERED));
                    s.push(glu_act_step(g, &self.k.glu, &spec, slots, ff, &gate_up, None, &h));
                    let hq = quantized(s, act[1], &h, slots, ff);
                    s.push(w2.gather(g, hq.as_ref().unwrap_or(&h), &ids, &y, slots, 1).expect(REGISTERED));
                }
                let identity = g.storage(u64::from(slots));
                g.write(&identity, &(0..slots).collect::<Vec<u32>>());
                s.push(g.step(self.k.combine, &[&gate, &ids, &identity, &y, &out], &[t, d, e, k], t * d));
            }
        }
        out
    }

    fn message_gate(&self, s: &mut Vec<Step>, w: &GateWeights, x: &DeviceBuffer, m: &DeviceBuffer, t: u32) -> DeviceBuffer {
        let cfg = &self.cfg;
        let rank = cfg.looped.expect("message gates belong to a loop").gate_rank;
        let gate = MessageGate::new(t, cfg.hidden, rank, cfg.eps).expect("a validated shape");
        let acts = gate.acts(self.g());
        let y = self.g().storage(u64::from(t * cfg.hidden));
        gate.forward(s, self.g(), &self.k.gate, &self.ops, &MessageGateWeights { down: &w.down, forget: &w.forget, gain: &w.gain }, x, m, &acts, &y);
        y
    }

    /// `y = x @ W^T` for `t` rows.
    fn proj(&self, s: &mut Vec<Step>, w: &Weight, x: &DeviceBuffer, t: u32) -> DeviceBuffer {
        let y = self.g().storage(u64::from(t) * u64::from(w.n()));
        self.ops.matmul(s, w, &self.ops.act_f32(x, 0, t, w.k()), &y, 0);
        y
    }

    /// RMS norm with a gain over `t` rows of the hidden width.
    fn rms(&self, s: &mut Vec<Step>, x: &DeviceBuffer, gain: &DeviceBuffer, t: u32) -> DeviceBuffer {
        let d = self.cfg.hidden;
        let y = self.g().storage(u64::from(t * d));
        let (idx, grid) = model::block::rms_variant(self.g(), self.k.rms, None, t, d);
        s.push(self.g().dispatch(idx, &[x, gain, &y], &[d, t, f(self.cfg.eps)], grid));
        y
    }

    /// Each of `n_vec` vectors of width `d` (vector `i` of head `i % heads`) scaled by `scale` when given, then clipped
    /// to L2 norm `thr`.
    #[allow(clippy::too_many_arguments)]
    fn l2clip(&self, s: &mut Vec<Step>, x: &DeviceBuffer, scale: Option<&DeviceBuffer>, n_vec: u32, heads: u32, d: u32, thr: f32) -> DeviceBuffer {
        let y = self.g().storage(u64::from(n_vec * d));
        s.push(self.g().dispatch(self.k.l2clip, &[x, scale.unwrap_or(x), &y], &[n_vec, heads, d, scale.is_some() as u32, f(thr), f(CLIP_EPS)], Dispatch::Workgroups(n_vec)));
        y
    }

    /// Convolution state `state` (0 K, 1 V, 2 the attention output, 3 the MoE output) of visit `visit` over the `t`
    /// rows `x [t, c]`: a whole sequence from zero history, a prompt chunk continuing and leaving its sequence's state,
    /// or one token of each decoding sequence stepping its state.
    #[allow(clippy::too_many_arguments)]
    fn conv(&self, s: &mut Vec<Step>, rows: &Rows, visit: usize, state: usize, x: &DeviceBuffer, taps: ConvTaps, t: u32, c: u32) -> DeviceBuffer {
        let (g, k) = (self.g(), self.cfg.conv_taps);
        let (shape, dynamic, base) = match taps {
            ConvTaps::Static(base) => (ShortConvShape::static_taps(t, c, k, BaseLayout::ChannelMajor).residual_scaled(1.0 / ((k + 1) as f32).sqrt()), None, base),
            ConvTaps::Dynamic { base, taps, layout } => (ShortConvShape::dynamic_taps(t, c, k, layout).residual_scaled(1.0 / ((2 * k + 1) as f32).sqrt()), Some(taps), base),
        };
        let y = g.storage(u64::from(t * c));
        match rows {
            Rows::Whole => s.push(short_conv_fwd(g, &self.k.conv, &shape, x, dynamic, base, None, None, &y)),
            Rows::Chunk { pool, table, .. } => {
                self.flush(s);
                let hist = pool.conv_rows(g, visit, state, &[table]);
                let tail = g.storage(u64::from(c * (k - 1)));
                g.submit(&[], &[short_conv_fwd(g, &self.k.conv, &shape.with_history(), x, dynamic, base, None, Some(&hist), &y), conv_state_tail_step(g, &self.k.conv, t, c, k, x, &hist, &tail)]);
                pool.store_conv_rows(g, visit, state, &[table], &tail);
            }
            Rows::Decode { pool, tables } => {
                self.flush(s);
                let hist = pool.conv_rows(g, visit, state, tables);
                g.submit(&[], &[short_conv_step(g, &self.k.conv, &shape, x, dynamic, base, &hist, &y)]);
                pool.store_conv_rows(g, visit, state, tables, &hist);
            }
        }
        y
    }

    /// When probing, the convolution state a cached prefill of `x` (`[t, c]`, the convolution's input) leaves:
    /// `[c, taps - 1]`, the last inputs.
    fn conv_state(&self, s: &mut Vec<Step>, probe: &mut Option<&mut dyn Probe>, key: &str, x: &DeviceBuffer, t: u32, c: u32) {
        let taps = self.cfg.conv_taps;
        if !probe.as_deref().is_some_and(|p| p.wants(key)) || taps < 2 {
            return;
        }
        let g = self.g();
        let zero = g.storage(u64::from(c * (taps - 1)));
        let state = g.storage(u64::from(c * (taps - 1)));
        s.push(model::short_conv::conv_state_tail_step(g, &self.k.conv, t, c, taps, x, &zero, &state));
        self.tap(s, probe, key, &state, c * (taps - 1));
    }

    /// A copy of `n` values of `src` from `offset`.
    fn rows(&self, s: &mut Vec<Step>, src: &DeviceBuffer, offset: u32, n: u32) -> DeviceBuffer {
        let y = self.g().storage(u64::from(n));
        s.push(self.g().step(self.k.copy, &[src, &y], &[n, offset, 0], n));
        y
    }
}

impl Weights {
    fn load(ops: &Ops, cfg: &Inkling2Config, src: &dyn TensorSource, micro: &[String], activations: &std::collections::HashMap<String, ActivationScheme>) -> Result<Weights, String> {
        let g = ops.gpu();
        let staging = BoundedWorkspace::new(MemoryBudget::new(STAGING_BYTES, 0));
        let host = |name: &str| upload_f32(g, src, name);
        let dense = |name: &str| -> Result<Weight, String> {
            let desc = src.packed_descriptor(name).ok_or_else(|| format!("inkling2: no packed view of {name}"))?;
            upload_preserved(ops, src, name, desc.shape[0] as usize, desc.shape[1] as usize, &staging)
        };
        let d = cfg.hidden;
        let mut layers = Vec::with_capacity(cfg.n_layers as usize);
        for l in 0..cfg.n_layers {
            let n = LayerNames::new(l);
            let a = cfg.attn(l);
            let full = cfg.kind(l) == LayerKind::Full;
            let kv_conv = |v: bool| -> Result<KvConv, String> {
                if !full {
                    return host(&n.kv_conv(v)).map(KvConv::Static);
                }
                // Published rows are `[kv head][tap]`; the convolution reads `[tap][kv head]`.
                let taps = cfg.conv_taps as usize;
                let permuted = reorder_tap_rows(&read_small(src, &n.kv_conv_proj(v))?, a.kv_heads as usize, taps, d as usize, true);
                let tier = src.packed_descriptor(&n.kv_conv_proj(v)).map(|desc| if desc.encoding == QuantEncoding::bf16() { gpu_core::select::Dtype::BF16 } else { gpu_core::select::Dtype::F32 });
                let proj = Weight::upload(ops, &permuted, a.kv_heads as usize * taps, d as usize, tier.unwrap_or(gpu_core::select::Dtype::F32));
                Ok(KvConv::Dynamic { taps: host(&n.kv_conv_static(v))?, proj })
            };
            let experts = if micro.contains(&n.w13()) != micro.contains(&n.w2()) {
                return Err(format!("inkling2: layer {l}'s gate/up and down banks are stored in different kinds of encoding"));
            } else if micro.contains(&n.w13()) {
                let names = |base: String| (0..cfg.experts).map(|e| format!("{base}.{e}")).collect::<Vec<_>>();
                Experts::Micro {
                    w13: upload_micro_bank(ops, src, &names(n.w13()), 2 * cfg.moe_ff as usize, d as usize, &staging)?,
                    w2: upload_micro_bank(ops, src, &names(n.w2()), d as usize, cfg.moe_ff as usize, &staging)?,
                    act: [activations.get(&n.w13()).copied(), activations.get(&n.w2()).copied()],
                }
            } else {
                Experts::Dense { w13: host(&n.w13())?, w2: host(&n.w2())? }
            };
            let gates = if cfg.is_core(l) {
                Some(
                    GateSite::ALL
                        .iter()
                        .map(|&site| {
                            let [down, forget, gain] = n.gate(site);
                            Ok(GateWeights { down: host(&down)?, forget: host(&forget)?, gain: host(&gain)? })
                        })
                        .collect::<Result<Vec<_>, String>>()?,
                )
            } else {
                None
            };
            layers.push(LayerWeights {
                attn_queries: if l == 0 { None } else { Some(host(&names::attn_queries(l))?) },
                mlp_queries: host(&names::mlp_queries(l))?,
                input_norms: if full { (0..3).map(|i| host(&n.qkv_norm(i))).collect::<Result<_, _>>()? } else { vec![host(&n.attn_norm())?] },
                wq: dense(&n.wq())?,
                wk: dense(&n.wk())?,
                wv: dense(&n.wv())?,
                wo: dense(&n.wo())?,
                gate_attn: dense(&n.gate_attn())?,
                sink: dense(&n.sink())?,
                wr: dense(&n.wr())?,
                q_scale: host(&n.q_scale())?,
                rel_scale: host(&n.rel_scale())?,
                rel_bank: host(&n.rel_bank())?,
                kv_conv: [kv_conv(false)?, kv_conv(true)?],
                attn_conv: host(&n.attn_conv())?,
                mlp_norm: host(&n.mlp_norm())?,
                mlp_conv: host(&n.mlp_conv())?,
                // The reference keeps the router in binary32 (`_keep_in_fp32_modules_strict`): its bf16-stored weight is
                // widened once here and its product never rounds an operand (see `moe`).
                router: Weight::dense_f32(host(&n.router())?, cfg.experts, d),
                router_bias: host(&n.router_bias())?,
                experts,
                gates,
            });
        }
        let message_queries = match cfg.looped {
            None => Vec::new(),
            Some(lp) => {
                let (qm, kv) = (read_small(src, &names::message_queries())?, read_small(src, &names::message_kv_queries())?);
                (0..lp.layers())
                    .map(|c| {
                        let rows: Vec<f32> = names::message_query_rows(c)
                            .iter()
                            .flat_map(|&(is_kv, r)| {
                                let t = if is_kv { &kv } else { &qm };
                                t[(r * d) as usize..((r + 1) * d) as usize].iter().copied()
                            })
                            .collect();
                        g.storage_init("inkling2.message_queries", &rows)
                    })
                    .collect()
            }
        };
        Ok(Weights {
            embed: dense(&names::embed())?,
            embed_norm: host(&names::embed_norm())?,
            final_norm: host(&names::final_norm())?,
            unembed: dense(&names::unembed())?,
            output_query: host(&names::output_query())?,
            layers,
            message_queries,
        })
    }
}
