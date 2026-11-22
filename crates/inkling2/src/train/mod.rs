// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training of looped mixture-of-experts language models on their published quantized
// weights for its clients. If your team needs expertise in fine-tuning such a model (full, LoRA over the NVFP4/MXFP8
// base, preference or policy-gradient objectives) then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Inkling 2 as a trainable [`model::Model`]: the forward recipe of [`crate::model`] and its backward, over the shared
//! operators' backward passes, in two modes.
//!
//! - **Full fine-tuning** ([`model::Model::new`], [`Inkling2Train::from_dir`] without an adapter): every parameter is
//!   an f32 tensor of the [`ParamStore`], under its published name, and trains (the router's selection bias excepted:
//!   it is a load-balancing heuristic with no gradient). The expert banks must be stored as floats.
//! - **Layer-streamed full fine-tuning** ([`model::Model::set_resident`], [`model::Model::new_resident`],
//!   [`Inkling2Train::from_dir_resident`]): the expert banks (93 % of the released 26B's parameters) leave the device
//!   for [`model::streamed::StreamedStore`]: their stored value on the host (bf16 under `weights=bf16`, the published
//!   NVFP4/MXFP8 banks dequantized once at load), the f32 master and AdamW state there or in a state file, and a
//!   window of f32 slots on the device that each visit loads its layer's banks into. Every other tensor trains in f32
//!   on the device. The core's banks are visited once per pass: their gradients are carried across the passes and
//!   updated once the backward is complete ([`model::streamed::LayeredPass::visits`]), every other layer's as its
//!   only visit ends. A trained build saves the banks in their stored encoding ([`model::Model::save`]).
//! - **LoRA** (a [`LoraCfg`]): the base stays frozen as published - every linear at the run's tier (bf16 as stored, or
//!   f32), the expert banks in their NVFP4/MXFP8 encoding read in place ([`MicroBank`]) - and only adapters train:
//!   linear sites through [`AdapterRuntime`], stacked expert sites per expert on their routed rows
//!   ([`model::moe::ExpertAdapters`]), and the small f32 matrices that no linear reads (depth and message queries,
//!   message-gate factors) through a merged pair ([`MergedLora`]).
//!
//! **Saved activations.** The forward keeps the partial sum entering every visit, the retained block states (one
//! [`model::saved::RetainedStates`] slot per block; the prelude's output slot is shared by every core pass, the previous pass's own
//! core block states are what each later pass's messages read), and nothing inside a visit: the backward walks the
//! visits in reverse and recomputes each one from its saved input before differentiating it. The routing a visit
//! recomputes is the routing its forward took (the same inputs, or the same replayed record). Gradients of the shared
//! core accumulate across its passes in the parameters' gradient buffers; a retained state's gradient is the sum over
//! every consumer (the depth attentions of later visits, the messages of the next pass, the readout) and is collected
//! into its producer's partial when the backward reaches the visit that retained it.

pub mod image_text;
mod recipe;
pub mod objectives;
pub mod sites;

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use checkpoint::packed::{BoundedWorkspace, MemoryBudget, QuantEncoding};
use checkpoint::TensorSource;
use gpu_core::provider::library::GemmPrecision;
use gpu_core::select::Dtype;
use gpu_core::{DeviceBuffer, Gpu};
use model::adapter::device::{lora_names, LoraCfg};
use model::adapter::exec::AdapterRuntime;
use model::adapter::merged::MergedLora;
use model::head_ce::{HeadCe, HeadCeConfig, HeadKernels};
use model::head_transform::HeadTransform;
use model::looped::VisitPlan;
use model::ops::{upload_micro_bank, upload_preserved, MicroBank, Ops, TierPolicy, Weight};
use model::routing::RoutingReplay;
use model::resident_ft::ResidentFtConfig;
use model::segmented::SegmentEdge;
use model::streamed::{LayeredPass, StreamedStore};
use model::{Adam, Batch, ModelConfig};
use optim::Optim;
use paramstore::{ParamStore, Role};

use crate::config::{Inkling2Config, LayerKind};
use crate::model::{open_checkpoint, reorder_tap_rows, upload_f32, visit_plan};
use crate::names::{self, LayerNames};

pub use sites::{expert_sites, lora_sites, SiteKind, DEFAULT_TARGETS};

/// The card family an Inkling 2 adapter or brain checkpoint carries.
pub const FAMILY: &str = "inkling2";

/// Where an in-memory LoRA build's save records its adapter configuration and its checkpoint configuration.
const SAVED_LORA_KEY: &str = "brain_inkling2_lora";
const SAVED_CONFIG_KEY: &str = "config";

/// Rows the adapter runtime is sized for at least: what one scored sequence may hold.
const SCORING_ROWS: u32 = 4096;

/// Host staging a single tile upload may take.
const STAGING_BYTES: u64 = 64 << 20;

/// Every kernel a training build dispatches.
pub fn kernels() -> Vec<(&'static str, &'static str)> {
    let mut list = crate::model::kernels();
    let more: Vec<(&'static str, &'static str)> = vec![
        ("rms_inv", kernels::RMS_INV),
        ("rmsnorm_dw", kernels::RMSNORM_DW),
        ("rmsnorm_dx", kernels::RMSNORM_DX),
        ("rmsnorm_nogain_dx", kernels::RMSNORM_NOGAIN_DX),
        ("l2clip_scale_dx", kernels::L2CLIP_SCALE_DX),
        ("l2clip_scale_ds", kernels::L2CLIP_SCALE_DS),
        ("sigmoid_bwd", kernels::SIGMOID_BWD),
        ("emb_bwd", kernels::EMB_BWD),
        ("router_bwd_sigmoid", kernels::ROUTER_BWD_SIGMOID),
        model::moe::ROUTER_REPLAY_KERNEL,
        ("moe_linear_gated_dx", kernels::MOE_LINEAR_GATED_DX),
        ("moe_linear_gated_dw", kernels::MOE_LINEAR_GATED_DW),
        ("scale_add_dexp", kernels::SCALE_ADD_DEXP),
        ("scale_add_dgate", kernels::SCALE_ADD_DGATE),
        ("silu_bwd_da", kernels::SILU_BWD_DA),
        ("silu_bwd_db", kernels::SILU_BWD_DB),
        ("adamw", kernels::ADAMW),
        ("gradnorm_sq", kernels::GRADNORM_SQ),
        ("grad_scale", kernels::GRAD_SCALE),
        ("clip_coef", kernels::CLIP_COEF),
        ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
        ("axpy", kernels::AXPY),
    ];
    let mut all = more;
    all.extend_from_slice(model::moe::COMPACT_TRAIN_KERNELS);
    all.extend_from_slice(model::moe::EXPERT_LORA_KERNELS);
    all.extend_from_slice(model::head_ce::kernels());
    all.extend_from_slice(model::adapter::merged::KERNELS);
    // A layer-streamed full fine-tune reads and writes its window and measures its gradients through these.
    all.extend_from_slice(model::ops::store_device_kernels());
    // The vision tower trains on the decoder's device ([`image_text`]).
    all.extend_from_slice(crate::vision::train_pipelines());
    model::ops::extend_kernel_list(&mut list, &all);
    list
}

/// The default adapter targets of a checkpoint: every site, the stacked expert sites only over microscaled banks
/// (`micro_banks`; float banks of a small checkpoint take no expert adapter).
pub fn targets(micro_banks: bool) -> Vec<String> {
    DEFAULT_TARGETS.iter().filter(|t| micro_banks || !t.starts_with("w13") && !t.starts_with("w2")).map(|t| t.to_string()).collect()
}

/// The configuration a training build is made from: the checkpoint's `config.json` (kept as written, so a saved
/// checkpoint carries it back), the validated geometry, and the adapter when the run trains one.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainConfig {
    pub json: serde_json::Value,
    pub cfg: Inkling2Config,
    pub lora: Option<LoraCfg>,
}

impl TrainConfig {
    /// Parse and validate a `config.json` value.
    pub fn from_value(json: serde_json::Value) -> Result<TrainConfig, String> {
        let cfg = Inkling2Config::from_json(&json.to_string()).map_err(|e| e.to_string())?;
        Ok(TrainConfig { json, cfg, lora: None })
    }

    /// `<dir>/config.json`.
    pub fn from_dir(dir: &Path) -> Result<TrainConfig, String> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        TrainConfig::from_value(json).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// This configuration training the adapter `lora` over a frozen base. Every target must name a site of the model;
    /// only the LoRA kind is offered.
    pub fn with_lora(mut self, lora: LoraCfg) -> Result<TrainConfig, String> {
        if lora.kind != model::adapter::DeviceKind::Lora {
            return Err(format!("inkling2: a {} adapter is not offered; train a LoRA adapter", lora.kind.name()));
        }
        let leaves: Vec<&str> = lora_sites(&self.cfg).iter().map(|(s, _)| s.leaf).chain(expert_sites(&self.cfg).iter().map(|s| s.leaf)).collect();
        if let Some(bad) = lora.targets.iter().find(|t| !leaves.contains(&t.as_str())) {
            return Err(format!("inkling2: LoRA target {bad:?} is not a site of this model (sites: {})", DEFAULT_TARGETS.join(", ")));
        }
        if !lora.modules_to_save.is_empty() {
            return Err("inkling2: modules_to_save is not offered; adapt the tensor through its LoRA site".into());
        }
        self.lora = Some(lora);
        Ok(self)
    }

    /// The adapter tensors of [`Self::lora`]: `(name, numel)`, linear and merged sites as `[rank, in]` / `[out, rank]`
    /// pairs, expert sites as `[experts, rank, in]` / `[experts, out, rank]`.
    pub fn adapter_params(&self) -> Vec<(String, usize)> {
        let Some(lora) = &self.lora else { return Vec::new() };
        let mut out = Vec::new();
        for (site, _) in lora_sites(&self.cfg) {
            if lora.targets_leaf(site.leaf) {
                out.extend(lora.param_list(&site.name, site.spec.out, site.spec.inn));
            }
        }
        for site in expert_sites(&self.cfg) {
            if lora.targets_leaf(site.leaf) {
                let r = lora.hp_for(&site.name).rank;
                let (a, b) = lora_names(&site.name);
                out.push((a, site.experts * r * site.inn));
                out.push((b, site.experts * site.out * r));
            }
        }
        out
    }
}

impl ModelConfig for TrainConfig {
    /// Every trained parameter of the published decoder, then the adapter's tensors. The routers' selection biases are
    /// held but not listed: they only order the top-k selection (a load-balancing heuristic with no gradient), so no
    /// objective trains them.
    fn param_list(&self) -> Vec<(String, usize)> {
        let biases: Vec<String> = (0..self.cfg.n_layers).map(|l| LayerNames::new(l).router_bias()).collect();
        names::text_manifest(&self.cfg).into_iter().filter(|e| !biases.contains(&e.name)).map(|e| (e.name, e.shape.iter().product::<u64>() as usize)).chain(self.adapter_params()).collect()
    }
    fn to_json(&self) -> serde_json::Value {
        self.json.clone()
    }
    fn from_json(v: &serde_json::Value) -> Self {
        TrainConfig::from_value(v.clone()).unwrap_or_else(|e| panic!("inkling2: {e}"))
    }
    /// The tokens the model scores: the head's valid rows (the stored tables carry padding rows beyond them).
    #[allow(clippy::misnamed_getters)] // the scored vocabulary is the valid one, not the stored row count
    fn vocab(&self) -> u32 {
        self.cfg.valid_vocab
    }
    fn block_size(&self) -> u32 {
        u32::try_from(self.cfg.max_positions).unwrap_or(u32::MAX)
    }
    /// The geometry is the checkpoint's: a dataset does not resize it.
    fn finalize_for_dataset(self, _vocab: u32, _block_size: u32) -> Self {
        self
    }
}

/// Whether `name` is a full layer's dynamic-convolution tap projection, held in the store in the tap-major row order
/// the convolution reads (and translated at every host boundary).
fn tap_rows(cfg: &Inkling2Config, name: &str) -> Option<(usize, usize, usize)> {
    (0..cfg.n_layers).filter(|&l| cfg.kind(l) == LayerKind::Full).find_map(|l| {
        let n = LayerNames::new(l);
        (name == n.kv_conv_proj(false) || name == n.kv_conv_proj(true)).then_some((cfg.full.kv_heads as usize, cfg.conv_taps as usize, cfg.hidden as usize))
    })
}

/// A layer's routed experts.
enum Bank {
    /// f32 banks `[E, 2ff, d]` / `[E, d, ff]`, in the store (full fine-tune) or frozen.
    Dense { w13: DeviceBuffer, w2: DeviceBuffer },
    /// The published microscaled banks, frozen, read in place.
    Micro { w13: MicroBank, w2: MicroBank },
}

/// A layer's routed experts as the recipe reads them.
enum BankRef<'a> {
    Dense { w13: &'a DeviceBuffer, w2: &'a DeviceBuffer },
    Micro { w13: &'a MicroBank, w2: &'a MicroBank },
}

/// The frozen base of a LoRA build.
struct Frozen {
    /// Every linear at the run's tier.
    lins: HashMap<String, Weight>,
    /// Every other float tensor as f32.
    vals: HashMap<String, DeviceBuffer>,
    banks: Vec<Bank>,
}

/// The batch most recently set, as the host holds it.
#[derive(Clone, Default)]
struct HostBatch {
    tokens: Vec<u32>,
    targets: Vec<u32>,
    /// Packed rows: each token's document id.
    segments: Option<Vec<u32>>,
}

/// An Inkling 2 decoder built for training on one device.
pub struct Inkling2Train {
    tc: TrainConfig,
    plan: VisitPlan,
    ops: Ops,
    k: recipe::Kernels,
    ps: ParamStore,
    opt: Optim,
    shapes: HashMap<String, Vec<u64>>,
    frozen: Option<Frozen>,
    /// Full fine-tune: the dense banks are in the store; one entry per layer naming nothing.
    runtime: Option<AdapterRuntime>,
    merged: HashMap<String, MergedLora>,
    /// Where gradients of frozen small tensors go (operators that always write a weight gradient); never read.
    discard: DeviceBuffer,
    head: HeadCe,
    b: u32,
    t: u32,
    batch: RefCell<HostBatch>,
    count: Cell<f32>,
    saved: RefCell<Option<recipe::Saved>>,
    weighted: RefCell<Option<(DeviceBuffer, DeviceBuffer)>>,
    logit_scale: Cell<f32>,
    replay: RefCell<Option<Arc<RoutingReplay>>>,
    /// The f32 base of a LoRA build made from in-memory tensors, kept so [`model::Model::save`] writes a file
    /// [`Inkling2Train::load`] rebuilds the same model from (a build over a checkpoint directory saves its adapter).
    host_base: Option<HashMap<String, Vec<f32>>>,
    /// Embedding rows spliced over the normed token embeddings of the batch (an image's features at its placeholder
    /// positions, positions counting the batch's flattened rows), and the gradient the last backward found at them.
    image_rows: RefCell<Option<model::serve::PromptEmbeds>>,
    image_grads: RefCell<Option<Vec<f32>>>,
    /// Per visit, the experts every row of the last forward selected (`[rows, top_k]`).
    taken: RefCell<Vec<Vec<u32>>>,
    detach_router: Cell<bool>,
    /// A layer-streamed full fine-tune: the expert banks' state and window ([`StreamedStore`]).
    streamed: Option<StreamedStore>,
    /// The forward a layer-streamed backward differentiates inside [`model::Model::adamw_step`].
    pending: RefCell<Option<recipe::Saved>>,
}

/// The layer whose expert bank `name` is: the tensors a layer-streamed full fine-tune streams.
fn bank_layer(cfg: &Inkling2Config, name: &str) -> Option<u32> {
    (0..cfg.n_layers).find(|&l| {
        let n = LayerNames::new(l);
        name == n.w13() || name == n.w2()
    })
}

/// The backward of one forward as a [`LayeredPass`]: each visit's recompute and backward between the edges of its
/// layer, which the streamed update loads before and finishes (or carries, for the core) after.
struct VisitBackward<'a> {
    m: &'a Inkling2Train,
    fwd: &'a recipe::Saved,
}

impl LayeredPass for VisitBackward<'_> {
    fn submit_hooked(&self, _gpu: &Gpu, edge: &mut dyn FnMut(SegmentEdge)) {
        self.m.backward_hooked(self.fwd, edge).unwrap_or_else(|e| panic!("inkling2 backward: {e}"));
    }

    fn visits(&self, layer: usize) -> u32 {
        self.m.plan.visits().iter().filter(|v| v.layer == layer).count() as u32
    }
}

/// How a training build is made.
pub struct BuildOptions<'a> {
    pub batch: u32,
    pub seq: u32,
    /// The tier of the frozen linears of a LoRA build (`f32` or `bf16`; bf16 keeps a bf16 release as stored).
    pub base: &'a TierPolicy,
    /// The seed of the adapters' `A` draws.
    pub seed: u64,
    /// An adapter file whose tensors training continues from (the same rank and targets).
    pub continue_from: Option<&'a Path>,
}

fn shapes_of(cfg: &Inkling2Config) -> HashMap<String, Vec<u64>> {
    names::text_manifest(cfg).into_iter().map(|e| (e.name, e.shape)).collect()
}

/// Whether the published tensor `name` is read by a linear product.
pub fn is_linear(cfg: &Inkling2Config, name: &str) -> bool {
    if [names::embed(), names::unembed()].iter().any(|n| n == name) {
        return true;
    }
    (0..cfg.n_layers).any(|l| {
        let n = LayerNames::new(l);
        [n.wq(), n.wk(), n.wv(), n.wo(), n.gate_attn(), n.sink(), n.wr(), n.router(), n.kv_conv_proj(false), n.kv_conv_proj(true)].iter().any(|x| x == name)
    })
}

impl Inkling2Train {
    /// A full fine-tune over the checkpoint directory `dir` (published names, float experts).
    pub fn from_dir(dir: &Path, b: u32, t: u32) -> Result<Inkling2Train, String> {
        let tc = TrainConfig::from_dir(dir)?;
        let (_, src, micro, _) = open_checkpoint(dir)?;
        if !micro.is_empty() {
            return Err(format!(
                "inkling2: a full fine-tune holding every expert in f32 on the device, but {} expert banks are stored microscaled (NVFP4/MXFP8) - the released 26B; stream its banks through a window (`weights=bf16,window=1`, Inkling2Train::from_dir_resident) or train a LoRA adapter over the published banks",
                micro.len()
            ));
        }
        Inkling2Train::build(tc, &src, &[], b, t, None, GemmPrecision::Auto)
    }

    /// A layer-streamed full fine-tune over the checkpoint directory `dir` ([`model::streamed`]): the expert banks are
    /// built straight into the resident state under `resident` (decoded once from the stored encoding, the published
    /// NVFP4/MXFP8 banks included, into the state's stored value and f32 master) and never all reach the device; every
    /// other tensor trains in f32 on the device. A release larger than the device trains so.
    pub fn from_dir_resident(dir: &Path, b: u32, t: u32, resident: &ResidentFtConfig) -> Result<Inkling2Train, String> {
        let tc = TrainConfig::from_dir(dir)?;
        let (_, src, micro, _) = open_checkpoint(dir)?;
        // The microscaled banks are readable as their per-expert views (each with its expert's global scale).
        let (d, ff, experts) = (tc.cfg.hidden as usize, tc.cfg.moe_ff as usize, tc.cfg.experts as usize);
        let stacks = (0..tc.cfg.n_layers)
            .flat_map(|l| {
                let n = LayerNames::new(l);
                [(n.w13(), 2 * ff * d), (n.w2(), d * ff)]
            })
            .filter(|(name, _)| micro.contains(name))
            .map(|(name, slab)| (name, experts, slab));
        let banks = checkpoint::packed::SlabStacks::new(&src, stacks);
        Inkling2Train::build_resident(tc, &banks, b, t, resident, GemmPrecision::Auto)
    }

    /// A full fine-tune whose banks are built as [`Role::Streamed`] and attached to a [`StreamedStore`] under
    /// `resident`, their initial values read from `src`.
    fn build_resident(tc: TrainConfig, src: &dyn TensorSource, b: u32, t: u32, resident: &ResidentFtConfig, gemm: GemmPrecision) -> Result<Inkling2Train, String> {
        if tc.lora.is_some() {
            return Err("inkling2: a layer-streamed build is a full fine-tune; an adapter trains over the frozen base".into());
        }
        let mut m = Inkling2Train::build_with(tc, src, src, &[], b, t, None, gemm, true)?;
        let cfg = m.tc.cfg.clone();
        let gpu = m.ops.gpu().share();
        let stream = StreamedStore::attach(&gpu, &mut m.ps, resident.clone(), src, &|n| bank_layer(&cfg, n)).map_err(|e| format!("inkling2: {e}"))?;
        m.streamed = Some(stream);
        Ok(m)
    }

    /// A LoRA build over the checkpoint directory `dir`: the base frozen as published (`opts.base` names the linears'
    /// tier), only the adapter `lora` trains.
    pub fn from_dir_lora(dir: &Path, lora: LoraCfg, opts: &BuildOptions) -> Result<Inkling2Train, String> {
        let tc = TrainConfig::from_dir(dir)?.with_lora(lora)?;
        let (_, src, micro, _) = open_checkpoint(dir)?;
        let continued = match opts.continue_from {
            Some(path) => {
                let tensors = checkpoint::safetensors::read(&path.to_string_lossy()).map_err(|e| format!("{}: {e}", path.display()))?;
                let map: HashMap<String, Vec<f32>> = tensors.into_iter().map(|t| (t.name, t.data)).collect();
                for (name, numel) in tc.adapter_params() {
                    match map.get(&name) {
                        Some(v) if v.len() == numel => {}
                        Some(v) => return Err(format!("{}: {name} holds {} values, this run's adapter {numel}", path.display(), v.len())),
                        None => return Err(format!("{}: the adapter being continued has no {name} (a different rank or target list)", path.display())),
                    }
                }
                Some(map)
            }
            None => None,
        };
        let init: &dyn TensorSource = match &continued {
            Some(map) => map,
            None => &src,
        };
        Inkling2Train::build_with(tc, &src, init, &micro, opts.batch, opts.seq, Some((opts.base, opts.seed)), GemmPrecision::Auto, false)
    }

    /// A build over in-memory f32 tensors in the published names (`init`): a full fine-tune, or a LoRA build when
    /// `tc` carries an adapter (its tensors drawn from `seed` unless `init` holds them). Its products run the exact fp32
    /// reference tier ([`GemmPrecision::Fp32Reference`]) on every device.
    pub fn from_tensors(tc: TrainConfig, init: &HashMap<String, Vec<f32>>, b: u32, t: u32, base: &TierPolicy, seed: u64) -> Result<Inkling2Train, String> {
        let lora = tc.lora.is_some();
        let mut m = Inkling2Train::build(tc, init, &[], b, t, lora.then_some((base, seed)), GemmPrecision::Fp32Reference)?;
        if lora {
            let names: Vec<String> = names::text_manifest(&m.tc.cfg).into_iter().map(|e| e.name).collect();
            m.host_base = Some(names.into_iter().filter_map(|n| init.get(&n).map(|v| (n, v.clone()))).collect());
        }
        Ok(m)
    }

    /// Rebuild a model from a file [`model::Model::save`] wrote, sized for `b x t`: a full fine-tune's checkpoint, or
    /// an in-memory LoRA build's base and adapter (held at f32).
    pub fn load(path: &str, b: u32, t: u32) -> Result<Inkling2Train, String> {
        let reader = checkpoint::weightio::WeightReader::open(path).map_err(|e| format!("{path}: {e}"))?;
        let header = reader.config();
        let tensors: HashMap<String, Vec<f32>> = checkpoint::safetensors::read(path).map_err(|e| format!("{path}: {e}"))?.into_iter().map(|t| (t.name, t.data)).collect();
        match header.get(SAVED_LORA_KEY) {
            Some(lora) => {
                let lora = LoraCfg::from_json(lora.as_object().ok_or_else(|| format!("{path}: {SAVED_LORA_KEY} is not an object"))?);
                let tc = TrainConfig::from_value(header.get(SAVED_CONFIG_KEY).cloned().ok_or_else(|| format!("{path}: no {SAVED_CONFIG_KEY}"))?)?.with_lora(lora)?;
                // The adapter's tensors are in the file: no draw happens, so the seed is never read.
                Inkling2Train::from_tensors(tc, &tensors, b, t, &TierPolicy::uniform(Dtype::F32), u64::MAX)
            }
            None => Inkling2Train::build(TrainConfig::from_value(header)?, &tensors, &[], b, t, None, GemmPrecision::Fp32Reference),
        }
    }

    fn build(tc: TrainConfig, src: &dyn TensorSource, micro: &[String], b: u32, t: u32, lora: Option<(&TierPolicy, u64)>, gemm: GemmPrecision) -> Result<Inkling2Train, String> {
        Inkling2Train::build_with(tc, src, src, micro, b, t, lora, gemm, false)
    }

    /// [`Self::build`] with the adapter's starting tensors read from `adapter_src` (where it holds them).
    ///
    /// `gemm` is the arithmetic of the dense products: a build over in-memory tensors (the tiny configurations the
    /// gradient checks, conformance and parity tests train) pins the exact fp32 reference tier those checks are stated
    /// against, a build over a checkpoint directory takes the device's tensor cores ([`GemmPrecision::Auto`]).
    /// `stream_banks` builds a full fine-tune's expert banks as [`Role::Streamed`] (no device buffer: the caller
    /// attaches them to a [`StreamedStore`]).
    #[allow(clippy::too_many_arguments)]
    fn build_with(tc: TrainConfig, src: &dyn TensorSource, adapter_src: &dyn TensorSource, micro: &[String], b: u32, t: u32, lora: Option<(&TierPolicy, u64)>, gemm: GemmPrecision, stream_banks: bool) -> Result<Inkling2Train, String> {
        let cfg = tc.cfg.clone();
        if b == 0 || t == 0 {
            return Err("inkling2: a training batch needs rows and tokens".into());
        }
        if u64::from(t) > cfg.max_positions {
            return Err(format!("inkling2: rows of {t} tokens exceed the model's {} positions", cfg.max_positions));
        }
        if tc.lora.is_some() != lora.is_some() {
            return Err("inkling2: a LoRA build needs both the adapter configuration and its base tier".into());
        }
        let static_list: &'static [(&'static str, &'static str)] = {
            static LIST: std::sync::OnceLock<&'static [(&'static str, &'static str)]> = std::sync::OnceLock::new();
            LIST.get_or_init(|| Box::leak(kernels().into_boxed_slice()))
        };
        let gpu = Gpu::new(static_list);
        let mut ops = Ops::new(gpu.share())?;
        ops.set_gemm_precision(gemm);
        let k = recipe::Kernels::resolve(&gpu)?;
        let need = |name: &str| gpu.kernel_index(name).ok_or_else(|| format!("inkling2: kernel {name} is not registered"));
        let opt = Optim::new(need("adamw")?, need("gradnorm_sq")?, need("grad_scale")?, need("clip_coef")?, need("grad_scale_buf")?);
        let shapes = shapes_of(&cfg);
        let n_k = |name: &str| -> (u32, u32) {
            let s = &shapes[name];
            (s[0] as u32, s[1..].iter().product::<u64>() as u32)
        };

        let (ps, frozen, runtime, merged) = match lora {
            None => {
                // Every parameter trains except the router's selection bias.
                let roles: Vec<(String, usize, Role)> = names::text_manifest(&cfg)
                    .into_iter()
                    .map(|e| {
                        let role = if (0..cfg.n_layers).any(|l| LayerNames::new(l).router_bias() == e.name) {
                            Role::Frozen
                        } else if stream_banks && bank_layer(&cfg, &e.name).is_some() {
                            Role::Streamed
                        } else {
                            Role::Trainable
                        };
                        (e.name, e.shape.iter().product::<u64>() as usize, role)
                    })
                    .collect();
                let ps = ParamStore::new_with_roles_src(&gpu, roles, src);
                for name in ps.params.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>() {
                    if let Some((h, taps, w)) = tap_rows(&cfg, &name) {
                        let rows = ps.read_weight(&gpu, &name);
                        gpu.write_f32(ps.w(&name), &reorder_tap_rows(&rows, h, taps, w, true));
                    }
                }
                (ps, None, None, HashMap::new())
            }
            Some((policy, seed)) => {
                let lcfg = tc.lora.clone().expect("checked above");
                let adapters = tc.adapter_params();
                let init = adapter_init(&tc, adapter_src, seed);
                let roles = adapters.iter().map(|(n, c)| (n.clone(), *c, model::adapter::device::role_of(n, &lcfg.hp_for(n)))).collect();
                let ps = ParamStore::new_with_roles(&gpu, roles, &init);
                let frozen = load_frozen(&ops, &cfg, &shapes, src, micro, policy)?;
                for (bi, bank) in frozen.banks.iter().enumerate() {
                    let n = LayerNames::new(bi as u32);
                    for name in [n.w13(), n.w2()] {
                        if matches!(bank, Bank::Dense { .. }) && ps.weight.contains_key(&lora_names(&name).0) {
                            return Err(format!("inkling2: {name} is stored as a float bank; expert adapters ride beside a microscaled (NVFP4/MXFP8) bank only - drop the w13_weight/w2_weight targets for this checkpoint"));
                        }
                    }
                }
                let max_out = lora_sites(&cfg).iter().filter(|(_, k)| *k == SiteKind::Linear).map(|(s, _)| s.spec.out as u32).max().unwrap_or(1);
                // Scoring a single sequence ([`Self::logits`]) may run more rows than a training batch.
                let runtime = AdapterRuntime::new(&gpu, &lcfg, (b * t).max(SCORING_ROWS), max_out);
                let mut merged = HashMap::new();
                for (site, kind) in lora_sites(&cfg) {
                    if kind == SiteKind::Merged && lcfg.targets_leaf(site.leaf) {
                        let hp = lcfg.hp_for(&site.name);
                        merged.insert(site.name.clone(), MergedLora::new(&gpu, site.spec.out as u32, site.spec.inn as u32, hp.rank as u32, hp.scale())?);
                    }
                }
                (ps, Some(frozen), Some(runtime), merged)
            }
        };
        let discard = gpu.storage(shapes.iter().filter(|(n, _)| !is_linear(&cfg, n)).map(|(_, s)| s.iter().product::<u64>()).max().unwrap_or(1));
        let unembed = match &frozen {
            Some(f) => f.lins[&names::unembed()].clone(),
            None => {
                let (n, kk) = n_k(&names::unembed());
                Weight::dense_f32(ps.w(&names::unembed()).clone(), n, kk)
            }
        };
        let rows = b * t;
        let chunk = if rows <= 256 { rows } else { 256 };
        let mut head = HeadCe::new(&ops, cfg.hidden, cfg.vocab, &unembed, HeadCeConfig { rows_per_chunk: chunk, tile_budget_words: None }, HeadKernels::from_gpu(&gpu)?)?;
        head.set_head_transform(&gpu, HeadTransform { softcap: Some(cfg.logit_cap), valid_vocab: Some(cfg.valid_vocab) })?;
        let plan = visit_plan(&cfg)?;
        Ok(Inkling2Train {
            tc,
            plan,
            ops,
            k,
            ps,
            opt,
            shapes,
            frozen,
            runtime,
            merged,
            discard,
            head,
            b,
            t,
            batch: RefCell::new(HostBatch::default()),
            count: Cell::new(1.0),
            saved: RefCell::new(None),
            weighted: RefCell::new(None),
            logit_scale: Cell::new(1.0),
            replay: RefCell::new(None),
            taken: RefCell::new(Vec::new()),
            image_rows: RefCell::new(None),
            image_grads: RefCell::new(None),
            host_base: None,
            detach_router: Cell::new(false),
            streamed: None,
            pending: RefCell::new(None),
        })
    }

    pub fn config(&self) -> &Inkling2Config {
        &self.tc.cfg
    }

    /// Choose the arithmetic of the dense products ([`model::ops::Ops::set_gemm_precision`]): a build over a checkpoint
    /// directory starts at [`GemmPrecision::Auto`], one over in-memory tensors at the fp32 reference tier.
    pub fn set_gemm_precision(&mut self, precision: GemmPrecision) {
        self.ops.set_gemm_precision(precision);
        *self.saved.borrow_mut() = None;
    }

    pub fn train_config(&self) -> &TrainConfig {
        &self.tc
    }

    pub fn plan(&self) -> &VisitPlan {
        &self.plan
    }

    pub fn ops(&self) -> &Ops {
        &self.ops
    }

    /// The device the model runs on (a vision tower trained with it is built on the same one).
    pub fn gpu(&self) -> &Gpu {
        self.ops.gpu()
    }

    /// Splice `rows` over the normed token embeddings of the batches trained from now on (`None` stops): row `i` of
    /// `rows` replaces position `positions[i]` of the batch's flattened `[b * t]` rows, after the embedding norm, as the
    /// reference scatters image features. The backward takes the gradient at those positions out of the embedding's
    /// ([`Self::take_image_gradients`]).
    pub fn set_image_rows(&self, rows: Option<model::serve::PromptEmbeds>) -> Result<(), String> {
        if let Some(r) = &rows {
            if r.width() != self.tc.cfg.hidden as usize {
                return Err(format!("inkling2: embedding rows of width {}, the decoder's is {}", r.width(), self.tc.cfg.hidden));
            }
            if let Some(&p) = r.positions().iter().find(|&&p| p >= self.rows()) {
                return Err(format!("inkling2: an embedding row at position {p}, the batch holds {} rows", self.rows()));
            }
        }
        *self.image_rows.borrow_mut() = rows;
        *self.saved.borrow_mut() = None;
        Ok(())
    }

    /// The gradient of the last backward at the spliced rows (`[rows, hidden]` in their order), taken once.
    pub fn take_image_gradients(&self) -> Option<Vec<f32>> {
        self.image_grads.borrow_mut().take()
    }

    fn g(&self) -> &Gpu {
        self.ops.gpu()
    }

    /// Hold the routing weights constant in the backward (a stop-gradient through every router): the experts' share
    /// of the input gradient only. Off by default: the router is differentiated, and in a full fine-tune it trains.
    pub fn set_detached_router(&self, detached: bool) {
        self.detach_router.set(detached);
    }

    /// The routing the last forward took, as a replay of the batch: forced back on with
    /// [`model::training::TokenObjectives::set_routing_replay`], every later forward selects the same experts (with
    /// the router's own weights over its current logits), so a perturbation of the weights cannot flip a selection -
    /// what a finite-difference check of the router's gradient needs.
    pub fn taken_routing(&self) -> Result<RoutingReplay, String> {
        let taken = self.taken.borrow();
        let (k, visits) = (self.tc.cfg.top_k as usize, self.plan.visits().len());
        if taken.len() != visits || taken.iter().any(|c| c.len() != self.rows() as usize * k) {
            return Err("inkling2: no forward of a training batch has routed yet".into());
        }
        let t = self.t as usize;
        let rows = (0..self.b as usize)
            .map(|b| {
                let calls = taken.iter().map(|c| model::routing::RoutedCall { top_k: k as u32, experts: c[b * t * k..(b + 1) * t * k].to_vec() }).collect();
                Some((Arc::new(model::routing::RoutingRecord { policy_version: 0, positions: self.t, recorded: vec![true; t], calls }), 0))
            })
            .collect();
        Ok(RoutingReplay { batch: self.b, seq: self.t, rows })
    }

    /// The f32 value of the non-linear tensor `name` (a merged adapter's effective matrix where one targets it).
    fn val(&self, name: &str) -> &DeviceBuffer {
        if let Some(m) = self.merged.get(name) {
            return m.eff();
        }
        match &self.frozen {
            Some(f) => f.vals.get(name).unwrap_or_else(|| panic!("inkling2: no frozen tensor {name}")),
            None => self.ps.w(name),
        }
    }

    /// Where the gradient of the non-linear tensor `name` is added: its store gradient, a merged adapter's `dW_eff`,
    /// or the discard buffer of a frozen tensor.
    fn grad(&self, name: &str) -> &DeviceBuffer {
        if let Some(m) = self.merged.get(name) {
            return m.d_eff();
        }
        match &self.frozen {
            Some(_) => &self.discard,
            None => self.ps.grad.get(name).unwrap_or(&self.discard),
        }
    }

    /// The linear `name` as its products read it.
    fn lin(&self, name: &str) -> Weight {
        match &self.frozen {
            Some(f) => f.lins.get(name).unwrap_or_else(|| panic!("inkling2: no frozen linear {name}")).clone(),
            None => {
                let s = &self.shapes[name];
                Weight::dense_f32(self.ps.w(name).clone(), s[0] as u32, s[1] as u32)
            }
        }
    }

    /// The store gradient of the linear `name` (a full fine-tune), `None` when it is frozen.
    fn lin_grad(&self, name: &str) -> Option<&DeviceBuffer> {
        match &self.frozen {
            Some(_) => None,
            None => self.ps.grad.get(name),
        }
    }

    /// Whether an adapter corrects the linear `name` at run time.
    fn adapted(&self, name: &str) -> bool {
        self.runtime.is_some() && self.ps.weight.contains_key(&lora_names(name).0)
    }

    /// Layer `l`'s expert banks.
    fn bank(&self, l: usize) -> BankRef<'_> {
        match &self.frozen {
            Some(f) => match &f.banks[l] {
                Bank::Dense { w13, w2 } => BankRef::Dense { w13, w2 },
                Bank::Micro { w13, w2 } => BankRef::Micro { w13, w2 },
            },
            None => {
                let n = LayerNames::new(l as u32);
                BankRef::Dense { w13: self.ps.w(&n.w13()), w2: self.ps.w(&n.w2()) }
            }
        }
    }

    fn rows(&self) -> u32 {
        self.b * self.t
    }

    /// Upload a batch: `tokens`/`targets` `[b * t]`, and for packed rows each token's document id.
    fn set_host_batch(&self, tokens: &[u32], targets: &[u32], segments: Option<&[u32]>) {
        let n = self.rows() as usize;
        assert_eq!(tokens.len(), n, "inkling2: {} tokens for a batch of {} x {}", tokens.len(), self.b, self.t);
        assert_eq!(targets.len(), n, "inkling2: {} targets for a batch of {} x {}", targets.len(), self.b, self.t);
        let cfg = &self.tc.cfg;
        if let Some(bad) = tokens.iter().find(|&&x| x >= cfg.vocab) {
            panic!("inkling2: token {bad} is outside the {} embedding rows", cfg.vocab);
        }
        if let Some(bad) = targets.iter().find(|&&x| x != model::IGNORE && x >= cfg.valid_vocab) {
            panic!("inkling2: target {bad} is outside the {} valid tokens", cfg.valid_vocab);
        }
        let count = targets.iter().filter(|&&x| x != model::IGNORE).count();
        self.count.set(count.max(1) as f32);
        *self.batch.borrow_mut() = HostBatch { tokens: tokens.to_vec(), targets: targets.to_vec(), segments: segments.map(<[u32]>::to_vec) };
        if let Some((w, _)) = self.weighted.borrow().as_ref() {
            self.g().write_f32(w, &vec![1.0; n]);
        }
        *self.saved.borrow_mut() = None;
    }

    /// The logits of every position of one sequence over the valid vocabulary (`[len, valid_vocab]`, soft-capped),
    /// under the current logit scale.
    pub fn logits(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let len = u32::try_from(tokens.len()).map_err(|_| "inkling2: too many tokens".to_string())?;
        if len == 0 || u64::from(len) > self.tc.cfg.max_positions {
            return Err(format!("inkling2: {len} tokens is not a sequence this model scores"));
        }
        if self.runtime.is_some() && len > (self.b * self.t).max(SCORING_ROWS) {
            return Err(format!("inkling2: {len} tokens exceed the {} rows this adapter build scores at once", (self.b * self.t).max(SCORING_ROWS)));
        }
        self.refresh_merged();
        let host = HostBatch { tokens: tokens.to_vec(), targets: vec![model::IGNORE; tokens.len()], segments: None };
        let fwd = self.trunk(&host, 1, len, false)?;
        let (v, valid) = (self.tc.cfg.vocab as usize, self.tc.cfg.valid_vocab as usize);
        Ok(self.full_logits(&fwd.normed, len).chunks(v).flat_map(|row| row[..valid].iter().copied()).collect())
    }

    /// Record and run `W_eff = W + scale B A` for every merged adapter.
    fn refresh_merged(&self) {
        if self.merged.is_empty() {
            return;
        }
        let f = self.frozen.as_ref().expect("merged adapters belong to a LoRA build");
        let mut s = Vec::new();
        for (name, m) in &self.merged {
            let (a, b) = lora_names(name);
            m.refresh(&self.ops, &mut s, &f.vals[name], self.ps.w(&a), self.ps.w(&b));
        }
        self.g().submit(&[], &s);
    }
}

/// Initial adapter tensors of `tc`'s adapter: the source's when it holds them, else `A` drawn from `seed`
/// (Kaiming-uniform over the corrected matrix's input width) and `B` zero.
fn adapter_init(tc: &TrainConfig, src: &dyn TensorSource, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = data::rng::Lcg::new(seed);
    let fan_in: HashMap<String, usize> = lora_sites(&tc.cfg).into_iter().map(|(s, _)| (s.name, s.spec.inn)).chain(expert_sites(&tc.cfg).into_iter().map(|s| (s.name, s.inn))).collect();
    tc.adapter_params()
        .into_iter()
        .map(|(name, numel)| {
            let mut v = Vec::with_capacity(numel);
            if src.with_tensor_chunks(&name, 1 << 20, &mut |_, c| v.extend_from_slice(c)) && v.len() == numel {
                return (name, v);
            }
            let values = match model::adapter::device::init_of(&name) {
                Some(model::adapter::ParamInit::Drawn) => {
                    let base = name.trim_end_matches(model::adapter::kind::LORA_A);
                    let bound = (1.0 / fan_in.get(base).copied().unwrap_or(1).max(1) as f32).sqrt();
                    (0..numel).map(|_| rng.scaled(bound)).collect()
                }
                _ => vec![0.0; numel],
            };
            (name, values)
        })
        .collect()
}

/// The frozen base of a LoRA build: linears at `policy`'s tier (bf16 as stored where it asks for bf16), every other
/// float tensor as f32, expert banks as published.
fn load_frozen(ops: &Ops, cfg: &Inkling2Config, shapes: &HashMap<String, Vec<u64>>, src: &dyn TensorSource, micro: &[String], policy: &TierPolicy) -> Result<Frozen, String> {
    let g = ops.gpu();
    let staging = BoundedWorkspace::new(MemoryBudget::new(STAGING_BYTES, 0));
    let (mut lins, mut vals) = (HashMap::new(), HashMap::new());
    for (name, shape) in shapes {
        if micro.contains(name) {
            continue;
        }
        let is_bank = (0..cfg.n_layers).any(|l| {
            let n = LayerNames::new(l);
            *name == n.w13() || *name == n.w2()
        });
        if is_bank {
            continue;
        }
        if !is_linear(cfg, name) {
            vals.insert(name.clone(), upload_f32(g, src, name)?);
            continue;
        }
        let (n, k) = (shape[0] as usize, shape[1..].iter().product::<u64>() as usize);
        // The reference keeps the router in binary32 at every tier (a rounded operand turns into another expert).
        let router = (0..cfg.n_layers).any(|l| LayerNames::new(l).router() == *name);
        let tier = if router { Dtype::F32 } else { policy.want(name) };
        if !matches!(tier, Dtype::F32 | Dtype::BF16) {
            return Err(format!("inkling2: the frozen linears are held as f32 or bf16, not {}", tier.name()));
        }
        let w = match tap_rows(cfg, name) {
            Some((h, taps, width)) => Weight::upload(ops, &reorder_tap_rows(&crate::model::read_small(src, name)?, h, taps, width, true), n, k, tier),
            None => {
                let stored_bf16 = src.packed_descriptor(name).is_some_and(|d| d.encoding == QuantEncoding::bf16());
                if tier == Dtype::BF16 && stored_bf16 {
                    upload_preserved(ops, src, name, n, k, &staging)?
                } else {
                    Weight::upload(ops, &crate::model::read_small(src, name)?, n, k, tier)
                }
            }
        };
        lins.insert(name.clone(), w);
    }
    let mut banks = Vec::with_capacity(cfg.n_layers as usize);
    for l in 0..cfg.n_layers {
        let n = LayerNames::new(l);
        let (w13, w2) = (n.w13(), n.w2());
        banks.push(match (micro.contains(&w13), micro.contains(&w2)) {
            (true, true) => {
                let views = |base: &str| (0..cfg.experts).map(|e| format!("{base}.{e}")).collect::<Vec<_>>();
                Bank::Micro {
                    w13: upload_micro_bank(ops, src, &views(&w13), 2 * cfg.moe_ff as usize, cfg.hidden as usize, &staging)?,
                    w2: upload_micro_bank(ops, src, &views(&w2), cfg.hidden as usize, cfg.moe_ff as usize, &staging)?,
                }
            }
            (false, false) => Bank::Dense { w13: upload_f32(g, src, &w13)?, w2: upload_f32(g, src, &w2)? },
            _ => return Err(format!("inkling2: layer {l}'s gate/up and down banks are stored in different kinds of encoding")),
        });
    }
    Ok(Frozen { lins, vals, banks })
}

impl model::Model for Inkling2Train {
    type Config = TrainConfig;

    /// A full fine-tune, or a LoRA build at f32 when the config carries an adapter (its tensors drawn from a random
    /// seed unless `init` holds them).
    fn new(cfg: TrainConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        let draw = data::rng::Lcg::random().next_u32() as u64;
        Inkling2Train::from_tensors(cfg, init, b, t, &TierPolicy::uniform(Dtype::F32), draw).unwrap_or_else(|e| panic!("inkling2: {e}"))
    }

    /// The decoder's weights and, for a LoRA configuration, its adapter (`A` drawn, `B` zero).
    fn init_weights(cfg: &TrainConfig, seed: u64) -> HashMap<String, Vec<f32>> {
        let mut out = init_weights(&cfg.cfg, seed);
        if cfg.lora.is_some() {
            out.extend(adapter_init(cfg, &HashMap::<String, Vec<f32>>::new(), seed));
        }
        out
    }

    fn config(&self) -> &TrainConfig {
        &self.tc
    }

    fn config_json(&self) -> serde_json::Value {
        self.tc.json.clone()
    }

    fn set_batch(&self, batch: Batch) {
        match batch {
            Batch::Lm { tokens, targets } => self.set_host_batch(tokens, targets, None),
            Batch::LmWeighted { tokens, targets, weights } => {
                self.set_host_batch(tokens, targets, None);
                objectives::TokenObjectives::set_loss_weights(self, weights);
            }
            Batch::LmPacked { tokens, targets, segments } => {
                data::packing::document_positions(segments, self.t as usize).unwrap_or_else(|e| panic!("inkling2: packed rows: {e}"));
                self.set_host_batch(tokens, targets, Some(segments));
            }
            _ => panic!("inkling2: only causal-LM batches (plain, weighted or packed) train this model"),
        }
    }

    fn reads_packed_rows(&self) -> bool {
        true
    }

    /// The backward always recomputes each visit from its saved f32 input partial (the retained block states are
    /// f32 too): [`model::saved::SavedPolicy::Recompute`] is what it does; keeping every visit and narrow boundaries
    /// are refused by name.
    fn set_saved_policy(&mut self, policy: model::saved::SavedPolicy) -> Result<(), String> {
        model::saved::recompute_only("inkling2", policy)
    }

    fn token_objectives(&self) -> Option<&dyn model::training::TokenObjectives> {
        Some(self)
    }

    fn token_objectives_mut(&mut self) -> Option<&mut dyn model::training::TokenObjectives> {
        Some(self)
    }

    fn forward(&self) -> f32 {
        self.refresh_merged();
        let host = self.batch.borrow().clone();
        let fwd = self.trunk(&host, self.b, self.t, true).unwrap_or_else(|e| panic!("inkling2 forward: {e}"));
        let loss = self.head_loss(&fwd);
        *self.saved.borrow_mut() = Some(fwd);
        loss
    }

    fn backward(&self) {
        let fwd = self.saved.borrow_mut().take().expect("inkling2: backward without a forward of the current batch");
        if self.streamed.as_ref().is_some_and(StreamedStore::is_layer_streamed) {
            // The banks are not all on the device: the backward runs visit by visit inside `adamw_step`, where the
            // layers it loads are the ones it updates.
            let earlier = self.pending.borrow_mut().replace(fwd);
            assert!(earlier.is_none(), "inkling2 layer-streamed full fine-tune: a second backward before the update (gradient accumulation) is not supported; every micro-batch is its own step");
            return;
        }
        self.backward_from(&fwd).unwrap_or_else(|e| panic!("inkling2 backward: {e}"));
    }

    fn zero_grads(&self) {
        self.ps.zero_grads(self.g());
    }

    fn adamw_step(&self, t: u32, lr: f32, wd: f32, adam: Adam, clip: Option<f32>, extra_scale: f32) {
        let Some(stream) = &self.streamed else {
            self.opt.step(self.g(), &self.ps, t, lr, wd, adam, clip, extra_scale);
            return;
        };
        let args = model::resident_ft::StepArgs { step: t, lr, wd, adam, clip, extra_scale };
        let pending = self.pending.borrow_mut().take();
        if stream.is_layer_streamed() && pending.is_none() {
            panic!("inkling2 layer-streamed full fine-tune: adamw_step without a backward");
        }
        // A build holding every bank ran its backward already; the pass is then never submitted.
        match &pending {
            Some(fwd) => stream.adamw_step(self.g(), &self.ps, &self.opt, &VisitBackward { m: self, fwd }, &args),
            None => stream.adamw_step(self.g(), &self.ps, &self.opt, &NoPass, &args),
        }
        .unwrap_or_else(|e| panic!("inkling2 layer-streamed update: {e}"));
    }

    fn last_grad_norm(&self) -> Option<f32> {
        None
    }

    fn poll_wait(&self) {
        self.g().poll_wait();
    }

    fn off_device_state(&self) -> Option<&dyn model::OffDeviceState> {
        self.streamed.as_ref().map(|_| self as &dyn model::OffDeviceState)
    }

    /// Stream the expert banks through a device window under `cfg` ([`model::streamed`]); see the module doc.
    fn set_resident(&mut self, cfg: &ResidentFtConfig) -> Result<(), String> {
        if self.tc.lora.is_some() {
            return Err("inkling2: a layer-streamed build is a full fine-tune; an adapter trains over the frozen base".into());
        }
        if self.streamed.is_some() {
            return Err("inkling2: the expert banks already stream".into());
        }
        let tc = self.tc.cfg.clone();
        let gpu = self.ops.gpu().share();
        let stream = StreamedStore::convert(&gpu, &mut self.ps, cfg.clone(), &|n| bank_layer(&tc, n)).map_err(|e| format!("inkling2: {e}"))?;
        self.opt.invalidate();
        self.streamed = Some(stream);
        *self.saved.borrow_mut() = None;
        Ok(())
    }

    fn new_resident(cfg: TrainConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, resident: &ResidentFtConfig) -> Result<Self, String> {
        Inkling2Train::build_resident(cfg, init, b, t, resident, GemmPrecision::Fp32Reference)
    }

    fn optimizer_state(&mut self) -> Option<model::OptimizerState<'_>> {
        Some(model::OptimizerState { gpu: self.ops.gpu(), store: &mut self.ps, optim: &self.opt })
    }

    fn optimizer_parts(&self) -> Option<model::OptimizerParts<'_>> {
        Some(model::OptimizerParts { gpu: self.ops.gpu(), store: &self.ps, optim: &self.opt })
    }

    /// The trained tensors: every parameter but the routers' selection biases (full), the adapter (LoRA).
    fn param_names(&self) -> Vec<String> {
        let streamed = self.streamed.iter().flat_map(|s| s.names().map(str::to_string));
        self.ps.opt_params().iter().map(|(n, _)| n.clone()).chain(streamed).collect()
    }

    fn frozen_adapter_param_names(&self) -> Vec<String> {
        model::adapter::device::frozen_adapter_params(&self.ps)
    }

    fn adapter_sites(&self) -> Vec<model::adapter::LinearSite> {
        lora_sites(&self.tc.cfg).into_iter().map(|(s, _)| s).collect()
    }

    fn adapter_expert_sites(&self) -> Vec<model::adapter::sites::ExpertSite> {
        expert_sites(&self.tc.cfg)
    }

    fn read_weight(&self, name: &str) -> Vec<f32> {
        if let Some(stream) = self.streamed.as_ref().filter(|s| s.holds(name)) {
            self.g().flush();
            return stream.read_weight(self.g(), name).unwrap_or_else(|e| panic!("inkling2: reading {name}: {e}"));
        }
        let v = self.ps.read_weight(self.g(), name);
        match tap_rows(&self.tc.cfg, name) {
            Some((h, taps, w)) => reorder_tap_rows(&v, h, taps, w, false),
            None => v,
        }
    }

    fn write_weight(&self, name: &str, data: &[f32]) {
        if let Some(stream) = self.streamed.as_ref().filter(|s| s.holds(name)) {
            self.g().flush();
            stream.write_weight(self.g(), name, data).unwrap_or_else(|e| panic!("inkling2: writing {name}: {e}"));
            *self.saved.borrow_mut() = None;
            return;
        }
        match tap_rows(&self.tc.cfg, name) {
            Some((h, taps, w)) => self.g().write_f32(self.ps.w(name), &reorder_tap_rows(data, h, taps, w, true)),
            None => self.g().write_f32(self.ps.w(name), data),
        }
        *self.saved.borrow_mut() = None;
    }

    fn read_grad(&self, name: &str) -> Vec<f32> {
        assert!(!self.streamed.as_ref().is_some_and(|s| s.holds(name) && s.is_layer_streamed()), "inkling2: {name} streams through a layer window; no gradient outlives its layer");
        let v = self.ps.read_grad(self.g(), name);
        match tap_rows(&self.tc.cfg, name) {
            Some((h, taps, w)) => reorder_tap_rows(&v, h, taps, w, false),
            None => v,
        }
    }

    fn write_grad(&self, name: &str, data: &[f32]) -> Result<(), String> {
        let g = self.ps.grad.get(name).ok_or_else(|| format!("inkling2: {name} has no gradient"))?;
        match tap_rows(&self.tc.cfg, name) {
            Some((h, taps, w)) => self.g().write_f32(g, &reorder_tap_rows(data, h, taps, w, true)),
            None => self.g().write_f32(g, data),
        }
        Ok(())
    }

    fn logits_all(&self, tokens: &[u32]) -> Option<Vec<f32>> {
        Some(self.logits(tokens).unwrap_or_else(|e| panic!("{e}")))
    }

    /// A full fine-tune writes the whole decoder under its published names with the checkpoint's config (a base again);
    /// a LoRA build over a checkpoint directory writes its adapter (the frozen base is the release itself); a LoRA
    /// build from in-memory tensors writes its base and adapter, which [`Inkling2Train::load`] rebuilds.
    fn save(&self, path: &str) {
        let card = checkpoint::st::ModelCard::new("brain/inkling2", FAMILY);
        if let Some(stream) = &self.streamed {
            // The banks in their stored encoding (the values the forward read), everything else in f32.
            let rest: Vec<(String, Vec<u64>)> = names::text_manifest(&self.tc.cfg).into_iter().filter(|e| !stream.holds(&e.name)).map(|e| (e.name, e.shape)).collect();
            self.g().flush();
            stream
                .save_checkpoint(self.g(), Path::new(path), &|n| self.shapes.get(n).cloned(), &rest, &|n| self.read_weight(n), &self.tc.json, Some(&card))
                .unwrap_or_else(|e| panic!("inkling2: saving {path}: {e}"));
            return;
        }
        let Some(lora) = &self.tc.lora else {
            let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = names::text_manifest(&self.tc.cfg).into_iter().map(|e| { let v = self.read_weight(&e.name); (e.name, e.shape, v) }).collect();
            checkpoint::save_carded(path, self.tc.json.clone(), &tensors, &card);
            return;
        };
        let Some(base) = &self.host_base else {
            model::lora::device_adapter::save_lora_with_lineage(path, self, lora, "brain/inkling2-adapter", "inkling2-base", FAMILY, None, None)
                .unwrap_or_else(|e| panic!("inkling2: saving the adapter to {path}: {e}"));
            return;
        };
        let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = names::text_manifest(&self.tc.cfg).into_iter().map(|e| { let v = base[&e.name].clone(); (e.name, e.shape, v) }).collect();
        for (name, _) in self.tc.adapter_params() {
            let v = self.ps.read_weight(self.g(), &name);
            tensors.push((name, vec![v.len() as u64], v));
        }
        let config = serde_json::json!({ SAVED_CONFIG_KEY: self.tc.json.clone(), SAVED_LORA_KEY: lora.to_json() });
        checkpoint::save_carded(path, config, &tensors, &card);
    }
}

impl model::OffDeviceState for Inkling2Train {
    fn commit(&self, dir: &Path, step: u32) -> std::io::Result<()> {
        let stream = self.streamed.as_ref().ok_or_else(|| std::io::Error::other("inkling2: no streamed state to commit"))?;
        self.g().flush();
        stream.commit(self.g(), dir, step)
    }

    fn restore(&self, dir: &Path, step: u32) -> std::io::Result<()> {
        let stream = self.streamed.as_ref().ok_or_else(|| std::io::Error::other("inkling2: no streamed state to restore"))?;
        self.g().flush();
        stream.restore(self.g(), dir, step)?;
        *self.saved.borrow_mut() = None;
        Ok(())
    }
}

/// The pass of a streamed build that holds every bank: its backward already ran, nothing is submitted.
struct NoPass;

impl LayeredPass for NoPass {
    fn submit_hooked(&self, _gpu: &Gpu, _edge: &mut dyn FnMut(SegmentEdge)) {}
}

/// Fresh weights of `cfg` drawn from `seed`, in the published names: small Gaussian-scale matrices, unit norm gains,
/// queries and scales near their trained magnitudes. For tests and from-scratch runs.
pub fn init_weights(cfg: &Inkling2Config, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = data::rng::Lcg::new(seed);
    names::text_manifest(cfg)
        .into_iter()
        .map(|e| {
            let numel = e.shape.iter().product::<u64>() as usize;
            let fan_in = *e.shape.last().unwrap_or(&1) as f32;
            let values: Vec<f32> = if e.name.ends_with("norm.weight") {
                (0..numel).map(|_| 1.0 + rng.scaled(0.1)).collect()
            } else if e.name.ends_with("l2_clip.scale") {
                (0..numel).map(|_| 1.0 + rng.scaled(0.2)).collect()
            } else if e.shape.len() == 1 || e.name.contains("attn_res") || e.name.contains("cross_pass") {
                (0..numel).map(|_| rng.scaled(0.5)).collect()
            } else {
                let bound = (3.0 / fan_in).sqrt();
                (0..numel).map(|_| rng.scaled(bound)).collect()
            };
            (e.name, values)
        })
        .collect()
}

/// A small looped geometry with every structural feature of the release: 8 layers in blocks of 2, a core of two
/// blocks (layers 2..6) run 3 times so the messages attend over a block state and the final partial, sliding and full
/// layers in every segment, a window and relative extents shorter than a row, 4 experts of which 2 route, padding rows
/// in the vocabulary, soft caps in their nonlinear range. For gradient checks, conformance suites and smoke runs.
pub fn tiny_config() -> TrainConfig {
    let json: serde_json::Value = serde_json::from_str(TINY_CONFIG).expect("the tiny configuration parses");
    TrainConfig::from_value(json).expect("the tiny configuration is valid")
}

/// [`tiny_config`] with every token routed to both of two experts: no top-k selection exists to flip, so a finite
/// difference of any size stays on one branch of the function (the router still weighs the experts and is
/// differentiated). For suites that cannot pin a routing ([`Inkling2Train::taken_routing`]).
pub fn tiny_config_every_expert_routed() -> TrainConfig {
    let mut json: serde_json::Value = serde_json::from_str(TINY_CONFIG).expect("the tiny configuration parses");
    json["text_config"]["n_routed_experts"] = serde_json::json!(2);
    json["text_config"]["num_experts_per_tok"] = serde_json::json!(2);
    TrainConfig::from_value(json).expect("the tiny configuration is valid")
}

/// [`tiny_config`]'s `config.json`.
const TINY_CONFIG: &str = r#"{
        "dtype": "float32",
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "model_type": "inkling2_loop",
        "eos_token_id": 30,
        "text_config": {
            "model_type": "inkling2_loop_text",
            "model_max_length": 64,
            "torch_dtype": "float32",
            "hidden_size": 32,
            "num_hidden_layers": 8,
            "vocab_size": 40,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "d_rel": 4,
            "rel_extent": 5,
            "q_bias": false,
            "o_bias": false,
            "log_scaling_n_floor": 2,
            "log_scaling_alpha": 0.1,
            "rms_norm_eps": 1e-6,
            "use_value_norm_no_gamma": true,
            "qk_l2_clip_threshold": 3.0,
            "rel_l2_clip_threshold": 3.0,
            "local_layer_ids": [0, 2, 4, 6],
            "layer_types": ["hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid"],
            "sconv_kernel_size": 3,
            "unpadded_vocab_size": 36,
            "final_logit_softcapping": 4.0,
            "swa_head_dim": 16,
            "swa_num_attention_heads": 2,
            "swa_num_key_value_heads": 1,
            "sliding_window_size": 3,
            "n_layers_per_attn_res_block": 2,
            "attn_res_normalize_values": true,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_shared_experts": 0,
            "intermediate_size": 16,
            "mlp_gate_softcap": 1.5,
            "mlp_up_softcap": 3.0,
            "route_scale": 2.5,
            "use_gate_bias": true,
            "eos_token_id": 30,
            "loop_config": { "core_start": 2, "core_end": 6, "num_passes": 3, "gate_rank": 4, "is_varloop": false }
        }
    }"#;
