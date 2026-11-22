// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements serving of looped mixture-of-experts language models from their published releases
// for its clients. If your team needs expertise in putting such a model behind one continuous-batching scheduler and
// the same HTTP/D-Bus contract as any other decoder then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Inkling 2 behind the residency executor and the capability interface: the directory `brain pull
//! thinkingmachines/Inkling-2-Mini` leaves in the store, served through the generic scheduler by
//! [`inkling2::serve::Engine`].
//!
//! - **Weights** are read as published: expert banks in their NVFP4/MXFP8 encodings, the dense weights at their
//!   stored tier, activations in f32 (the reference numerics, inkling2 ledger I2.4).
//! - **Chat** is the checkpoint's own template: typed messages, `reasoning_effort` (named or 0..1) as a template
//!   variable ([`qwen3::chat::template_params`]), tool calls and reasoning read back by the typed-message scanner. A
//!   generation stops on the sampling eos `generation_config.json` declares (which the template also ends a reply
//!   with), never on the message boundary that closes the thinking message before an answer.
//! - **K/V** rows are f32 unless `--kv-cache bf16` asks for half the bytes per token, and `--linear-activations declared`
//!   gives the expert products the release's declared activation quantization ([`INKLING2_POLICY`]).
//! - **Capacity**: one sequence holds at most `ctx` tokens; the pool holds `max_batch` such sequences in every full
//!   visit's planes, one state row (40 windowed rings and every visit's convolution states) per running sequence and
//!   as many snapshot rows for cached prompts. [`Inkling2Resident::estimate`] is the weights, that pool and the
//!   engine's step buffers.
//!
//! - **Speculation** is opt-in (`--speculative`): `mtp` drafts with the release's own drafter (`mtp.safetensors`, a
//!   DSpark block drafter over the target's hidden states at four logical depths, `inkling2::mtp`), `ngram` by prompt
//!   lookup. Either adds an undo state row per running sequence (a verify is rolled back over the rings and convolution
//!   states), the drafter's weights and caches, and host memory for the retained hidden states. `auto` and `off`
//!   decode plainly: what speculation buys on a device is not measured yet (inkling2 ledger, Phase 7); `dflash` is
//!   refused by name.
//!
//! - **Images**: a checkpoint with a vision tower reads the request's image blobs (one per image part of its
//!   messages) through [`images::Inkling2Images`], the tower on the decoder's card; the rows stand at the template's
//!   placeholder tokens, expanded to one per 40x40-pixel region, and the engine splices them after its embedding norm.
//!
//! - **Adapters**: with an adapter bank ([`Inkling2Resident::with_adapter_bank`]) the LoRA adapters the trainer saves
//!   for this base are served as models of their own ([`AdapterHost`]: each an [`llmserve::adapter_registry::AdapterView`]
//!   whose id is the adapter's card id): every site the trainer adapts is applied beside the published weights in the
//!   same batches as the base ([`inkling2::serve::ServedSites`]), never folded into them.
//!
//! [`provider`] is the same resident behind `brain do inkling2 generate`, built on first use.

pub mod images;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use capability::{Action, ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, Progress, Provider};
use data::qwen_chat::TemplateFlavor;
use data::qwen_tokenizer::QwenBpe;
use inkling2::serve::{Engine, EngineOptions};
use inkling2::vision::{ImageProcessorSpec, VisionTower, VisionWeights};
use inkling2::Inkling2Config;
use llmserve::adapter_registry::{AdaptedInstance, AdapterBankSettings, AdapterHost, AdapterRegistry, ADAPTER_PARAM};
use llmserve::adapter_release::{LoadedAdapter, ServedBase, ServedFamily};
use llmserve::image_frontend::ImageFrontend;
use llmserve::{generate_spec, TextFrontend};
use model::kv_tier::KvTier;
use model::serve::visit_state::VisitPoolSizing;
use model::serve::{NgramDrafter, Scheduler, SpeculativeMode, TapPool};
use residency::{on_device, Device, Instance, InstanceKey, MemCost, ResidentModel};

use crate::resident_qwen35::Qwen35Sizing;

/// The card family of an Inkling 2 checkpoint and of the adapters its trainer saves.
pub const FAMILY: &str = inkling2::train::FAMILY;
/// The catalog id of the in-process capability (`brain do inkling2 generate`).
pub const MODEL: &str = "brain/inkling2";
/// Tokens one sequence may hold when the server was not told (`--ctx`): the model's positions reach 1M, but a pool
/// of `max_batch` such contexts would not fit a card.
pub const DEFAULT_CTX: u32 = 16384;
/// Rows of a KV block of the full visits.
pub const BLOCK: u32 = 16;
/// Prompt rows of one prefill pass.
pub const PREFILL_CHUNK: u32 = 512;
/// What the Inkling 2 resident honours of a [`model::policy::ServePolicy`]: K/V rows in f32 (the reference numerics,
/// the default) or bf16 (half the bytes per token; its distance from f32 is measured in the engine's tests and recorded
/// in the inkling2 ledger), and the expert products' activation operand (`--linear-activations declared`: the
/// release's declared W4A4/W8A8 numerics; the model refuses `bf16`/`e4m3`, for which it has no product, by name).
/// The weights are served as published, so no weight format is chosen.
pub const INKLING2_POLICY: model::policy::PolicySupport = model::policy::PolicySupport {
    model: "inkling2",
    weights: &[],
    weight_rules: false,
    kv_cache: &["f32", "bf16"],
    gemm: false,
    fuse_norm: false,
    kv_calibration: false,
    exact_decode: false,
    act_precision: false,
    gdn_state: false,
    linear_activations: true,
    sparse_attention: false,
    draft_model: false,
};

/// Tokens the prompt-lookup drafter matches on (`--speculative ngram`).
const NGRAM_ORDER: usize = 3;

/// Where a resident's proposals come from (`--speculative`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Drafting {
    /// Plain decoding.
    Off,
    /// The release's drafter (`mtp.safetensors`).
    Mtp,
    /// Prompt lookup: no model.
    Ngram,
}

impl Drafting {
    /// The drafting `--speculative` asks for, for the checkpoint in `dir`.
    fn resolve(mode: Option<SpeculativeMode>, dir: &Path, cfg: &Inkling2Config, id: &str) -> Result<Drafting, String> {
        match mode {
            None | Some(SpeculativeMode::Off) => Ok(Drafting::Off),
            Some(SpeculativeMode::Auto) => {
                eprintln!("brain: {id}: --speculative auto decodes plainly: speculation is opt-in for this model (--speculative mtp or ngram) until it is measured on a device");
                Ok(Drafting::Off)
            }
            Some(SpeculativeMode::Ngram) => Ok(Drafting::Ngram),
            Some(SpeculativeMode::Dflash) => Err(format!("{id}: --speculative dflash: Inkling 2 has no DFlash2 draft model; its drafter is the release's own (--speculative mtp)")),
            Some(SpeculativeMode::Mtp) => {
                if !dir.join(inkling2::mtp::MTP_FILE).is_file() {
                    return Err(format!("{id}: --speculative mtp: {} has no {}", dir.display(), inkling2::mtp::MTP_FILE));
                }
                inkling2::mtp::draft_config(dir, cfg)?.ok_or_else(|| format!("{id}: --speculative mtp: config.json declares no mtp_config"))?;
                Ok(Drafting::Mtp)
            }
        }
    }
}

const SUMMARY: &str = "text generation (Inkling 2: a looped mixture-of-experts decoder with depth-attention residuals, sliding and full attention, served as released; typed-message chat with reasoning and tool calls; continuous batching with prompt-prefix reuse)";

/// The static manifest: safe to build with no weights on the machine. The released checkpoint has a vision tower, so
/// the in-process action reads images.
pub fn manifest() -> Manifest {
    Manifest::new(MODEL, SUMMARY, vec![action_spec(true)])
}

/// The `generate` action; `reads_images` declares the image blobs (`image`, `image1`, ... in the order of the
/// messages' image parts), so the API serves a request with images only to a checkpoint with a tower.
fn action_spec(reads_images: bool) -> ActionSpec {
    let spec = generate_spec("generate text (Inkling 2; the checkpoint's typed-message chat template)", true).params(qwen3::chat::template_params(TemplateFlavor::TypedMessages));
    (0..if reads_images { capability::blob::MAX_IMAGES } else { 0 }).fold(spec, |spec, i| {
        let help = if i == 0 {
            "an image of the conversation, in the order its image parts appear: raw HWC f32 pixels in [0,1], meta {w,h}; at most 4096 pixels a side, one decoder token per 40x40-pixel region"
        } else {
            "a further image, read only when every earlier one is present"
        };
        spec.input(BlobSpec::new(&capability::blob::image_key(i), Media::Image, help))
    })
}

/// A pulled Inkling 2 directory served through the scheduler.
pub struct Inkling2Resident {
    id: String,
    dir: PathBuf,
    cfg: Inkling2Config,
    opts: EngineOptions,
    /// The expert products' activation operand (`--linear-activations`).
    linear_activations: model::policy::LinearActivations,
    drafting: Drafting,
    /// The sampling defaults the checkpoint's `generation_config.json` declares, on every manifest it answers (an
    /// adapter view's too).
    sampling: data::generation::SamplingDefaults,
    /// The adapters offered on this base, and the bank they run on (none unless [`Self::with_adapter_bank`]).
    adapters: Arc<AdapterRegistry>,
}

/// Inkling 2 as an adapter-serving family: the card family its trainer writes and every site it serves.
pub fn family(cfg: &Inkling2Config) -> ServedFamily {
    ServedFamily::of(FAMILY, "Inkling 2", &inkling2::serve::ServedSites::new(cfg, true))
}

impl Inkling2Resident {
    /// The resident for the directory `dir` (`config.json`, `hf_quant_config.json`, the safetensors, `tokenizer.json`,
    /// the chat template) under the model id `id`: `sizing.ctx` caps a sequence when it was asked for
    /// ([`DEFAULT_CTX`] otherwise, never past the model's positions), `sizing.max_batch` sequences run at once, K/V rows
    /// are kept at `sizing.kv_cache` and the expert products take `sizing.linear_activations` ([`INKLING2_POLICY`]).
    pub fn from_hf_dir(dir: &str, id: &str, sizing: Qwen35Sizing) -> Result<Inkling2Resident, String> {
        let path = Path::new(dir);
        let cfg = Inkling2Config::from_dir(path)?;
        for needed in ["tokenizer.json", "generation_config.json"] {
            if !path.join(needed).is_file() {
                return Err(format!("{id}: {dir} has no {needed}"));
            }
        }
        let sizing = sizing.sanitized();
        let policy = model::policy::ServePolicy { kv_cache: sizing.kv_cache, linear_activations: sizing.linear_activations, ..Default::default() };
        INKLING2_POLICY.check(&policy).map_err(|e| format!("{id}: {e}"))?;
        let kv_tier = sizing.kv_cache.unwrap_or(KvTier::F32);
        let ctx = if sizing.ctx_explicit { sizing.ctx } else { DEFAULT_CTX };
        let ctx = u64::from(ctx).min(cfg.max_positions).max(u64::from(BLOCK)) as u32;
        let live = sizing.max_batch.max(1);
        let blocks = ctx.div_ceil(BLOCK).checked_mul(live).ok_or_else(|| format!("{id}: {live} sequences of {ctx} tokens overflow the block count"))?;
        let drafting = Drafting::resolve(sizing.speculative, path, &cfg, id)?;
        let undo_rows = drafting != Drafting::Off;
        let opts = EngineOptions { kv_tier, sizing: VisitPoolSizing { block_size: BLOCK, num_blocks: blocks, live_rows: live, snapshot_rows: live, undo_rows }, max_seq_len: ctx as usize, prefill_chunk: PREFILL_CHUNK };
        // The layout is held to the engine's own check now, so a configuration it cannot serve is refused at the scan.
        Engine::layout(&cfg, kv_tier)?;
        let sampling = data::generation::SamplingDefaults::from_generation_config(path).map_err(|e| format!("{id}: {e}"))?;
        let adapters = AdapterRegistry::new(None, ServedBase::of_family(family(&cfg), path, Some(id.to_string())));
        Ok(Inkling2Resident { id: id.to_string(), dir: path.to_path_buf(), cfg, opts, linear_activations: sizing.linear_activations.unwrap_or_default(), drafting, sampling, adapters })
    }

    /// This resident with an adapter bank of `bank` (none for `None`): its engine serves the adapters offered on it
    /// ([`AdapterHost`]), and the bank's device bytes are part of its estimate. A size the engine cannot build is refused
    /// by the flag that sets it.
    pub fn with_adapter_bank(mut self, bank: Option<AdapterBankSettings>) -> Result<Inkling2Resident, String> {
        if let Some(b) = &bank {
            b.validate().map_err(|e| format!("{}: {e}", self.id))?;
            b.bank_config(&inkling2::serve::ServedSites::new(&self.cfg, true)).map_err(|e| format!("{}: {e}", self.id))?;
        }
        self.adapters = AdapterRegistry::new(bank, ServedBase::of_family(family(&self.cfg), &self.dir, Some(self.id.clone())));
        Ok(self)
    }

    /// Where this resident's proposals come from.
    pub fn drafting(&self) -> Drafting {
        self.drafting
    }

    /// Device bytes of the drafter (`--speculative mtp`): its weights and its caches, one slot per running sequence.
    fn draft_bytes(&self) -> u64 {
        if self.drafting != Drafting::Mtp {
            return 0;
        }
        inkling2::mtp::device_bytes(&self.dir, &self.cfg, self.opts.sizing.live_rows, self.opts.max_seq_len as u32).expect("the drafter's declaration was checked at construction")
    }

    /// Host bytes of the hidden states the drafter reads, retained per cached position.
    fn tap_bytes(&self) -> u64 {
        if self.drafting != Drafting::Mtp {
            return 0;
        }
        let depths = inkling2::mtp::draft_config(&self.dir, &self.cfg).ok().flatten().map_or(0, |c| c.target_layers.len());
        let s = self.opts.sizing;
        TapPool::bytes(depths, self.cfg.hidden as usize, s.num_blocks, s.block_size)
    }

    /// The engine sizing this resident serves at.
    pub fn options(&self) -> EngineOptions {
        self.opts
    }

    /// Bytes of the decoder's weights as stored (the vision tower and the optional drafter file are not loaded).
    fn weight_bytes(&self) -> u64 {
        std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| {
                let p = e.path();
                p.extension().is_some_and(|x| x == "safetensors") && p.file_name().is_some_and(|n| n != "mtp.safetensors")
            })
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum()
    }

    /// Device bytes of the pool: every full visit's blocks and every state row.
    fn pool_bytes(&self) -> u64 {
        let layout = Engine::layout(&self.cfg, self.opts.kv_tier).expect("checked at construction");
        let s = self.opts.sizing;
        u64::from(s.num_blocks) * layout.block_bytes(s.block_size) + s.state_rows() * layout.row_bytes() + layout.zero_row_bytes()
    }

    /// The engine's step buffers at their widest: one prefill pass of [`PREFILL_CHUNK`] rows or one decode step of the
    /// whole batch. Per row: the logits, every routed slot's expert products, the hidden-width activations of a visit
    /// (inputs, depth stacks, messages and gates, about 64 rows of the width) and the attention's query-width rows.
    fn step_bytes(&self) -> u64 {
        let c = &self.cfg;
        let rows = u64::from(PREFILL_CHUNK.max(self.opts.sizing.live_rows));
        let per_row = u64::from(c.vocab) + u64::from(c.top_k) * (3 * u64::from(c.moe_ff) + u64::from(c.hidden)) + 64 * u64::from(c.hidden) + 8 * u64::from(c.attn(0).q_dim());
        rows * per_row * 4
    }

    /// Device bytes of the adapter bank (none without one): an upper bound before the weights are opened, the expert
    /// sites counted as though the banks were published microscaled.
    fn bank_bytes(&self) -> u64 {
        let rows = PREFILL_CHUNK.max(self.opts.sizing.live_rows);
        self.adapters.settings().map_or(0, |b| b.device_bytes(&inkling2::serve::ServedSites::new(&self.cfg, true), rows))
    }

    /// The vision tower on the decoder's card: its weights in f32 and the activations of its largest group, a context
    /// of regions (four patch rows each, plus class rows), about two dozen rows of the tower's width per patch row
    /// (inputs, attention, the MLP's four-wide hidden, the scratch of a block). Zero without a tower.
    fn tower_bytes(&self) -> u64 {
        let Some(v) = &self.cfg.vision else { return 0 };
        let params: u64 = inkling2::names::vision_manifest_of(v).iter().map(|e| e.shape.iter().product::<u64>()).sum();
        let merge = u64::from(v.adapter_merge_size * v.adapter_merge_size);
        let rows = (self.opts.max_seq_len as u64 * merge + capability::blob::MAX_IMAGES as u64).min(inkling2::vision::MAX_GROUP_ROWS as u64);
        (params + rows * 24 * u64::from(v.hidden_size)) * 4
    }
}

impl ResidentModel for Inkling2Resident {
    /// Declares its architecture and the checkpoint's sampling defaults itself: the adapter views served over this
    /// resident restate its manifest.
    fn manifest(&self) -> Manifest {
        let spec = llmserve::with_sampling_defaults(action_spec(self.cfg.vision.is_some()), &self.id, &self.sampling);
        Manifest::new(&self.id, SUMMARY, vec![spec]).with_max_context_tokens(self.opts.max_seq_len as u64).with_architecture(FAMILY)
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(&self.id, "default")
    }
    fn batch_limit(&self) -> Option<usize> {
        Some(self.opts.sizing.live_rows as usize)
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::new(self.weight_bytes() + self.pool_bytes() + self.step_bytes() + self.tower_bytes() + self.draft_bytes() + self.bank_bytes(), self.tap_bytes())
    }
    /// The base answers as the base: the reserved adapter param is the views' to set ([`ResidentModel::stamp`]).
    fn stamp(&self, inv: &mut Invocation) {
        if let Some(params) = inv.params.as_object_mut() {
            params.remove(ADAPTER_PARAM);
        }
    }
    /// The request's prompt keyed for the prefix cache in the namespace of the adapter it names: a prefix cached under
    /// one adapter is not another's, and the base is the empty namespace.
    fn prefix_chain(&self, inv: &Invocation) -> residency::affinity::PrefixChain {
        llmserve::request_prefix_chain(inv, &inv.get_str(ADAPTER_PARAM).unwrap_or_default())
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let tok = QwenBpe::from_file(&self.dir.join("tokenizer.json").to_string_lossy())?;
        let format = qwen3::chat::ChatFormat::own_template(&self.dir)?;
        let typed = matches!(&format, qwen3::chat::ChatFormat::Template(t) if t.reply_flavor() == Some(TemplateFlavor::TypedMessages));
        if !typed {
            return Err(format!("{}: the chat template does not write typed messages, which this model replies in", self.id));
        }
        let stop = data::generation::stop_ids(Some(&self.dir), &tok, None, format.end_of_turn()).map_err(|e| format!("inkling2: {e}"))?;
        if stop.is_empty() {
            return Err(format!("{}: the checkpoint declares no token that ends a generation (generation_config.json eos_token_id)", self.id));
        }
        let format = Arc::new(format);
        let vision = self.cfg.vision.clone();
        // The processor names the placeholder its template writes; the tokenizer must hold it as one token.
        let processor = match &vision {
            Some(v) => {
                let path = self.dir.join("processor_config.json");
                let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: the checkpoint has a vision tower but {}: {e}", self.id, path.display()))?;
                let spec = ImageProcessorSpec::from_json(&text)?;
                let tiling = spec.tiling(v)?;
                let image_token = tok.special_id(&spec.image_token).ok_or_else(|| format!("{}: the tokenizer has no token {:?}, the processor's image placeholder", self.id, spec.image_token))?;
                Some((tiling, image_token))
            }
            None => None,
        };
        let reads_images = processor.is_some();
        // The image builder runs on its own thread with a second instance of the tokenizer.
        let image_tok = if reads_images { Some(QwenBpe::from_file(&self.dir.join("tokenizer.json").to_string_lossy())?) } else { None };
        let text_format = format.clone();
        let text = TextFrontend::new(
            tok,
            stop,
            "inkling2",
            Arc::new(move |tok: &QwenBpe, inv: &Invocation| {
                images::refuse_unplaced_image_parts(inv, reads_images)?;
                qwen3::chat::parse_request_as(tok, &text_format, inv)
            }),
        );
        let (dir, opts, linear_activations, drafting, id) = (self.dir.clone(), self.opts, self.linear_activations, self.drafting, self.id.clone());
        let adapters = self.adapters.clone();
        let live = opts.sizing.live_rows as usize;
        on_device(device, move || -> Result<Box<dyn Instance>, String> {
            let mut ops = inkling2::serve::device_ops()?;
            ops.set_linear_activations(linear_activations);
            let mut engine = Engine::open(ops, &dir, opts)?;
            if drafting == Drafting::Mtp {
                engine.set_drafter(&dir)?;
            }
            if let Some(bank) = adapters.settings() {
                // The `all` group over this checkpoint: its expert sites only when its banks are published microscaled.
                let cfg = bank.bank_config(&engine.served_sites()).map_err(|e| format!("{id}: {e}"))?;
                engine.enable_adapters(cfg).map_err(|e| format!("{id}: {e}"))?;
            }
            let scheduler = |engine: Engine| {
                let mut sched = Scheduler::new(engine, live);
                if drafting == Drafting::Ngram {
                    sched.set_drafter(Box::new(NgramDrafter::new(NGRAM_ORDER)));
                }
                sched
            };
            let (Some(v), Some((tiling, image_token)), Some(tok)) = (vision, processor, image_tok) else {
                return Ok(Box::new(AdaptedInstance::new(text, scheduler(engine), live, adapters)));
            };
            // The tower shares the decoder's card: its own kernel set on the same device.
            let gpu = engine.model().ops().gpu().new_like(inkling2::vision::pipelines());
            let tower = VisionTower::new(&gpu, &v, &VisionWeights::open(&dir, &v)?)?;
            let sched = scheduler(engine);
            let build = images::Inkling2Images::new(tok, format, gpu, tower, tiling, image_token, sched.sequence_capacity());
            let front = ImageFrontend::new(text, build, "inkling2-vision")?;
            Ok(Box::new(AdaptedInstance::new(front, sched, live, adapters)))
        })?
    }
}

impl AdapterHost for Inkling2Resident {
    fn adapters(&self) -> &Arc<AdapterRegistry> {
        &self.adapters
    }

    fn register_view(self: &Arc<Self>, executor: &residency::Executor, adapter: LoadedAdapter) -> Result<(), String> {
        executor.register(Arc::new(self.adapter_view(adapter)?));
        Ok(())
    }
}

/// `brain do inkling2 generate` and the SDK: the resident of the resolved model directory, built on the emptiest card
/// (the host tier without one) on the first request and kept for the next.
pub fn provider(assembly: &capability::Assembly) -> Result<Arc<dyn Provider>, String> {
    Ok(Arc::new(Inkling2Provider { state: Arc::new(ProviderState { weights: assembly.roles.get("weights").cloned(), live: Mutex::new(None) }) }))
}

struct Inkling2Provider {
    state: Arc<ProviderState>,
}

struct ProviderState {
    weights: Option<PathBuf>,
    live: Mutex<Option<Box<dyn Instance>>>,
}

impl Provider for Inkling2Provider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "generate").then(|| Arc::new(GenerateAction { state: self.state.clone() }) as Arc<dyn Action>)
    }
}

struct GenerateAction {
    state: Arc<ProviderState>,
}

impl Action for GenerateAction {
    fn spec(&self) -> ActionSpec {
        action_spec(true)
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let mut live = self.state.live.lock().map_err(|_| "inkling2: an earlier request panicked while holding the model".to_string())?;
        if live.is_none() {
            let dir = self.state.weights.as_ref().ok_or("inkling2: no model on this machine (brain pull thinkingmachines/Inkling-2-Mini)")?;
            let resident = Inkling2Resident::from_hf_dir(&dir.to_string_lossy(), MODEL, Qwen35Sizing::default())?;
            let device = gpu_core::capacity::available_gpus().into_iter().max_by_key(|&(_, free)| free).map_or(Device::Cpu, |(i, _)| Device::Gpu(i));
            *live = Some(resident.activate(&InstanceKey::new(MODEL, "default"), device)?);
        }
        live.as_mut().expect("built above").run("generate", inv, progress)
    }
}
