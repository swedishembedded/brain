// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving very large language
// models on the accelerators a customer owns, sized to the byte before
// anything is loaded, for its clients. If your team needs expertise in
// placing long-context decoders over one or several cards then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Ripple-65B (`RicursiveForCausalLM`) behind the residency [`Executor`](residency) and the capability interface:
//! the Hugging Face directory a pull of `ricursiveAI/Ripple-65B-Preview` (bf16) or
//! `ricursiveAI/Ripple-65B-Preview-NVFP4` leaves in the store, or any local directory of the same layout, served
//! through the generic continuous-batching scheduler by [`ricursive::RicursiveEngine`].
//!
//! Everything a Qwen3 model gets from the shared paths, Ripple gets from the same ones: the request front end is
//! `llmserve::TextFrontend` over `qwen3::chat::parse_request_as` with the checkpoint's own `chat_template.jinja`
//! ([`qwen3::chat::ChatFormat::own_template`]), the tokenizer is the checkpoint's `tokenizer.json`, replies are
//! scanned into reasoning, content and OpenAI tool calls by the shared scanner, and the model-agnostic policy
//! (`--weight-format`, `--kv-cache`, `--sparse-attention`, `--ctx`, `--max-batch`) is checked against what
//! [`ricursive::POLICY`] declares.
//!
//! # Sizing
//!
//! The memory the deployment costs is stated by [`ricursive::placement::footprint`], the same functions the engine
//! allocates with: the linears at the tier the checkpoint stores them in (or `--weight-format`), the rings of the
//! sliding layers for every live and cached sequence, the block pool of the global layers for the context asked
//! for, the index of latent sparse attention when `--sparse-attention` names one, and the fixed bytes of each
//! stage. With no `--ctx` the context is the longest that places on the cards the device selection offers (at most
//! the model's own limit); an explicit one that does not place is refused with the arithmetic, never shortened.
//!
//! # Devices
//!
//! The device selection (`--device`) decides how many cards the layers are cut over: the fewest of the selected
//! cards that hold the deployment, one stage each ([`ricursive::placement::stream_plan`]). A server with cards
//! registers the resident as a multi-device model; a server on the host (`--device cpu`) registers it as a plain
//! one, a single stage on the host tier.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Invocation, Manifest, ParamSpec, ParamType, Progress, Provider};
use data::generation::SamplingDefaults;
use data::qwen_tokenizer::QwenBpe;
use llmserve::{ScheduledInstance, TextFrontend};
use model::kv_tier::KvTier;
use model::lsa::indexer::Indexer;
use model::ops::TierPolicy;
use model::policy::ServePolicy;
use model::serve::{NgramDrafter, Scheduler};
use qwen3::chat::{parse_request_as, ChatFormat};
use qwen35::int8_gguf_resident::SpeculativeMode;
use residency::multi::{MultiDeviceCost, MultiDeviceResidentModel};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use ricursive::placement::{card_budget, footprint, place, stream_plan, Footprint};
use ricursive::{LsaCharge, RicursiveConfig, RicursiveEngine, ServeConfig};

use crate::adapter_release::{LoadedAdapter, ServedBase, RICURSIVE_FAMILY};
use crate::resident_llm::{generate_spec, on_device};

/// Rows per global KV block.
const BLOCK_SIZE: u32 = 16;
/// Prompt rows per prefill forward, which also sizes the ring slack of the sliding layers.
const PREFILL_CHUNK: u32 = 256;
/// Sequences decoded together when `--max-batch` names none.
const DEFAULT_MAX_BATCH: u32 = 4;
/// Cached prompts beside the live sequences when none is asked for.
const DEFAULT_PREFIX_SLOTS: u32 = 2;
/// The context a server on the host tier is sized for when `--ctx` names none: the weights, not the context, fill a
/// host, and a context past this is asked for by name.
const HOST_DEFAULT_CTX: u32 = 4096;
/// The shortest context the automatic sizing will settle for before it reports that the deployment does not place.
const MIN_CTX: u32 = 2048;
/// Tokens of context the prompt-lookup drafter matches to find what to propose.
const NGRAM_ORDER: usize = 3;
/// Tokens a companion sequence is assumed to hold beside the one that fills the context, when sizing the block pool.
const COMPANION_TOKENS: u32 = 4096;

/// Bytes the host side of a staged instance holds regardless of its cards (the mapped checkpoint is page cache, the
/// embedding rows are gathered from it): the token buffers and the scheduler's own state.
const HOST_OVERHEAD_BYTES: u64 = 256 << 20;

/// The catalog id of the in-process capability (`brain do ricursive generate`).
pub const MODEL: &str = "brain/ricursive";

/// The serving knobs of Ripple, parsed from the model-agnostic policy (`apply_policy`) and nothing else.
#[derive(Clone, Debug, Default)]
pub struct RicursiveServeConfig {
    /// `--ctx`: per-sequence token ceiling. `None`: the longest that places (see the module doc).
    pub ctx: Option<u32>,
    /// `--max-batch`: sequences decoded together.
    pub max_batch: Option<u32>,
    /// `--weight-format`: the format each linear is held in. `None`: as the checkpoint stores it.
    pub weights: Option<TierPolicy>,
    /// `--kv-cache`: how K and V are stored. `None`: bf16, or int8 with `--sparse-attention`.
    pub kv_cache: Option<KvTier>,
    /// `--sparse-attention`: the indexer file the global layers read through.
    pub sparse_attention: Option<String>,
    /// Cached prompts kept for prefix reuse (0 turns reuse off).
    pub prefix_slots: Option<u32>,
    /// `--speculative`: `None`, `Off` and `Auto` decode plainly (the release offers no drafter this engine runs),
    /// `Ngram` drafts by prompt lookup and verifies in one batched step. `Mtp` and `Dflash` are refused.
    pub speculative: Option<SpeculativeMode>,
    /// Why the policy could not be honoured, when it could not (the resident then refuses to build, naming it).
    pub refused: Option<String>,
}

impl RicursiveServeConfig {
    /// Apply the model-agnostic serving policy. A choice Ripple cannot honour is refused by name
    /// ([`ricursive::POLICY`]) and recorded in [`Self::refused`]. Returns notes for settings that have no meaning here.
    pub fn apply_policy(&mut self, p: &ServePolicy) -> Result<Vec<String>, String> {
        if let Err(e) = ricursive::POLICY.check(p) {
            self.refused = Some(e.clone());
            return Err(e);
        }
        if let Some(w) = &p.weights {
            self.weights = Some(w.clone());
        }
        if let Some(kv) = p.kv_cache {
            self.kv_cache = Some(kv);
        }
        if let Some(path) = &p.sparse_attention {
            self.sparse_attention = Some(path.clone());
        }
        let l = &p.limits;
        self.ctx = l.ctx.or(self.ctx);
        self.max_batch = l.max_batch.or(self.max_batch);
        self.prefix_slots = l.prefix_slots.or(self.prefix_slots);
        let mut notes = Vec::new();
        for (set, name) in [(l.kv_tokens.is_some(), "kv-tokens"), (l.kv_offload_gb.is_some(), "kv-offload-gb"), (l.max_prefill.is_some(), "max-prefill"), (l.prefill_slice.is_some(), "prefill-slice"), (l.prefill_batch.is_some(), "prefill-batch"), (l.itl_target_ms.is_some(), "itl-target-ms")] {
            if set {
                notes.push(format!("ricursive: the {name} setting has no meaning for this engine and is ignored"));
            }
        }
        Ok(notes)
    }
}

/// What the sizing settled on.
#[derive(Clone)]
struct Deployment {
    serve: ServeConfig,
    footprint: Footprint,
    /// What the defaults chose, one line each, for the startup log.
    chosen: Vec<String>,
}

/// The block pool for `ctx`: one sequence that fills the context, companions of a few thousand tokens, and the
/// cached prompts, rounded up to whole blocks.
fn pool_blocks(ctx: u32, max_batch: u32, prefix_slots: u32) -> u32 {
    let short = ctx.min(COMPANION_TOKENS);
    let tokens = u64::from(ctx) + u64::from(max_batch.saturating_sub(1) + prefix_slots) * u64::from(short);
    tokens.div_ceil(u64::from(BLOCK_SIZE)).min(u64::from(u32::MAX - 2)) as u32 + 2
}

/// Everything about the checkpoint the sizing reads.
struct Checkpoint {
    cfg: RicursiveConfig,
    tiers: TierPolicy,
    max_positions: u32,
}

/// The directory's `config.json`, with the model's own position limit (which the numerical config does not carry).
fn read_config(dir: &Path) -> Result<(RicursiveConfig, u32), String> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let cfg = RicursiveConfig::from_json(&text)?;
    let raw: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let max_positions = raw.get("max_position_embeddings").and_then(|v| v.as_u64()).filter(|&n| n > 0).ok_or_else(|| format!("{}: no max_position_embeddings", path.display()))?;
    Ok((cfg, u32::try_from(max_positions).unwrap_or(u32::MAX)))
}

/// Ids that end a reply beyond the tokenizer's own end of turn: the checkpoint's `eos_token_id` (an id or a list).
fn config_eos(dir: &Path) -> Vec<u32> {
    let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else { return Vec::new() };
    let Ok(raw) = serde_json::from_str::<serde_json::Value>(&text) else { return Vec::new() };
    match &raw["eos_token_id"] {
        serde_json::Value::Number(n) => n.as_u64().map(|v| vec![v as u32]).unwrap_or_default(),
        serde_json::Value::Array(ids) => ids.iter().filter_map(|v| v.as_u64().map(|v| v as u32)).collect(),
        _ => Vec::new(),
    }
}

/// The deployment `knobs` ask for on `cards` (`(device index, usable bytes)`; empty for the host tier).
fn size(ck: &Checkpoint, knobs: &RicursiveServeConfig, lsa: Option<&Indexer>, cards: &[(usize, u64)]) -> Result<Deployment, String> {
    let mut chosen = Vec::new();
    let kv = match knobs.kv_cache {
        Some(kv) => kv,
        None => {
            // The default cache is bf16; sparse attention reads an f32, int8 or fp8 cache, so the default there is int8.
            let kv = if lsa.is_some() { KvTier::INT8 } else { KvTier::BF16 };
            chosen.push(format!("kv-cache {kv}"));
            kv
        }
    };
    let max_batch = knobs.max_batch.unwrap_or(DEFAULT_MAX_BATCH).max(1);
    let prefix_slots = knobs.prefix_slots.unwrap_or(DEFAULT_PREFIX_SLOTS);
    let build = |ctx: u32| {
        let mut serve = ServeConfig::new(BLOCK_SIZE, pool_blocks(ctx, max_batch, prefix_slots), ctx, max_batch, prefix_slots, PREFILL_CHUNK);
        serve.kv_cache = kv;
        serve.weights = knobs.weights.clone();
        serve.sparse_attention = lsa.is_some();
        serve
    };
    let fit = |serve: &ServeConfig| {
        let charge = lsa.map_or(LsaCharge::default(), |ix| LsaCharge::of(ix, serve));
        let fp = footprint(&ck.cfg, serve, &ck.tiers, charge);
        (cards.is_empty() || stream_plan(&fp, cards).is_some()).then_some(fp)
    };
    let ctx_limit = ck.max_positions;
    let (serve, fp) = match knobs.ctx {
        Some(ctx) => {
            let ctx = ctx.min(ctx_limit);
            let serve = build(ctx);
            let fp = fit(&serve).ok_or_else(|| does_not_place(&ck.cfg, &serve, &ck.tiers, lsa, cards))?;
            (serve, fp)
        }
        None if cards.is_empty() => {
            let serve = build(HOST_DEFAULT_CTX.min(ctx_limit));
            let fp = fit(&serve).expect("the host tier has no capacity to exceed");
            chosen.push(format!("ctx {}", serve.max_seq_len));
            (serve, fp)
        }
        None => {
            let mut ctx = ctx_limit;
            loop {
                let serve = build(ctx);
                if let Some(fp) = fit(&serve) {
                    chosen.push(format!("ctx {ctx}"));
                    break (serve, fp);
                }
                if ctx <= MIN_CTX.min(ctx_limit) {
                    return Err(does_not_place(&ck.cfg, &serve, &ck.tiers, lsa, cards));
                }
                ctx = (ctx / 2).max(MIN_CTX.min(ctx_limit));
            }
        }
    };
    Ok(Deployment { serve, footprint: fp, chosen })
}

/// The refusal for a deployment no leading subset of `cards` holds: what it needs and what is offered.
fn does_not_place(cfg: &RicursiveConfig, serve: &ServeConfig, tiers: &TierPolicy, lsa: Option<&Indexer>, cards: &[(usize, u64)]) -> String {
    let fp = footprint(cfg, serve, tiers, lsa.map_or(LsaCharge::default(), |ix| LsaCharge::of(ix, serve)));
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    format!(
        "ricursive: a context of {} tokens needs {:.1} GiB ({:.1} weights, {:.1} global KV, {:.1} rings, {:.1} index, {:.1} fixed per stage) and the {} card(s) offer {:.1} GiB usable; use a narrower --weight-format or --kv-cache, a smaller --ctx or --max-batch, or more cards",
        serve.max_seq_len,
        gib(fp.single_card_bytes()),
        gib(fp.weights),
        gib(fp.kv_global),
        gib(fp.kv_rings),
        gib(fp.lsa),
        gib(fp.per_stage),
        cards.len(),
        gib(cards.iter().map(|c| c.1).sum())
    )
}

/// The catalog manifest of the in-process capability: static, safe to build with no weights on the machine.
pub fn manifest() -> Manifest {
    Manifest::new(MODEL, SUMMARY, vec![generate_action()])
}

const SUMMARY: &str = "text generation (Ripple-65B hybrid sliding-window/global-NoPE decoder with value residual and gated attention; bf16 or NVFP4 weights, f32/bf16/int8/fp8 KV, latent sparse attention over the global layers, layers staged over the selected cards, continuous batching with prompt-prefix reuse)";

/// The ceilings of the sampling numbers `generate_spec` leaves open: every served number has one (a request for a
/// billion is refused, not tried).
const SAMPLING_CEILINGS: [(&str, f64); 9] = [
    ("temp", 100.0),
    ("top_p", capability::bounds::UNIT),
    ("min_p", capability::bounds::UNIT),
    ("presence_penalty", 10.0),
    ("frequency_penalty", 10.0),
    ("repetition_penalty", 10.0),
    ("dry_multiplier", 100.0),
    ("dry_base", 100.0),
    ("seed", 9_007_199_254_740_991.0),
];

fn generate_action() -> ActionSpec {
    let mut spec = generate_spec("generate text (Ripple; the checkpoint's own chat template, reasoning and tool calls)", true).param(ParamSpec::new(
        "chat_template_kwargs",
        ParamType::Str,
        "JSON object of extra variables the checkpoint's chat template reads (Ripple's `clear_thinking` drops earlier turns' reasoning from the prompt)",
    ));
    for param in spec.params.iter_mut().filter(|p| p.max.is_none()) {
        if let Some(&(_, ceiling)) = SAMPLING_CEILINGS.iter().find(|(name, _)| *name == param.name) {
            param.max = Some(ceiling);
        }
    }
    spec
}

/// The Ripple directory served through the scheduler.
#[derive(Clone)]
pub struct RicursiveResident {
    id: String,
    dir: PathBuf,
    cfg: RicursiveConfig,
    deployment: Deployment,
    /// The indexer file `--sparse-attention` named, loaded again for each instance.
    sparse_attention: Option<String>,
    /// Draft by prompt lookup (`--speculative ngram`).
    ngram: bool,
    /// `(device index, usable bytes)` of the cards the deployment may be placed over; empty for the host tier.
    cards: Vec<(usize, u64)>,
    sampling: SamplingDefaults,
    /// The LoRA adapter this model serves on top of the checkpoint, applied at run time beside the resident base
    /// (`RicursiveEngine::attach_adapter`); `None` for the base itself.
    adapter: Option<Arc<LoadedAdapter>>,
}

impl RicursiveResident {
    /// The resident for the directory `dir` (a pulled `config.json`, `tokenizer.json` and safetensors) under the model id
    /// `id`, sized for `knobs` over `gpus` (`(index, TOTAL bytes)` of every card the process may use, `reserved` bytes
    /// kept free on each; empty for a host-tier server). The checkpoint's tensors are validated against the manifest the
    /// config declares, and a sparse-attention indexer that does not fit the model is refused here, at startup.
    pub fn from_hf_dir(dir: &str, id: &str, knobs: &RicursiveServeConfig, gpus: &[(u32, u64)], reserved: u64) -> Result<RicursiveResident, String> {
        if let Some(why) = &knobs.refused {
            return Err(why.clone());
        }
        match knobs.speculative {
            Some(SpeculativeMode::Mtp) => return Err(format!("{id}: --speculative mtp: Ripple has no multi-token-prediction head")),
            Some(SpeculativeMode::Dflash) => {
                return Err(format!("{id}: --speculative dflash: the Ripple-65B-Preview-DFlash2 drafter needs the target's hidden states at its layers 1, 20, 39, 58 and 77, which this engine does not retain, and its block forward is not ported; use ngram or off"))
            }
            Some(SpeculativeMode::Ngram) if knobs.sparse_attention.is_some() => {
                return Err(format!("{id}: --speculative ngram cannot be combined with --sparse-attention: a rejected row would leave index pages built over rolled-back keys"))
            }
            _ => {}
        }
        let path = Path::new(dir);
        if !path.join("tokenizer.json").exists() {
            return Err(format!("{id}: {dir} has no tokenizer.json"));
        }
        let (cfg, max_positions) = read_config(path)?;
        let layers: Vec<usize> = (0..cfg.num_hidden_layers).collect();
        let src = ricursive::open_shared_checkpoint(dir, &cfg, &layers).map_err(|e| format!("{id}: {e}"))?;
        let tiers = ricursive::resolve_tiers(&cfg, src.as_ref(), knobs.weights.as_ref());
        drop(src);
        let indexer = knobs.sparse_attention.as_deref().map(Indexer::load).transpose().map_err(|e| format!("{id}: --sparse-attention: {e}"))?;
        let cards: Vec<(usize, u64)> = gpus.iter().map(|&(i, total)| (i as usize, card_budget(total).saturating_sub(reserved))).collect();
        let ck = Checkpoint { cfg: cfg.clone(), tiers, max_positions };
        let deployment = size(&ck, knobs, indexer.as_ref(), &cards).map_err(|e| format!("{id}: {e}"))?;
        if let Some(ix) = &indexer {
            deployment.serve.check_indexer(&cfg, ix).map_err(|e| format!("{id}: --sparse-attention: {e}"))?;
        }
        let sampling = SamplingDefaults::from_generation_config(path)?;
        Ok(RicursiveResident { id: id.to_string(), dir: path.to_path_buf(), cfg, deployment, sparse_attention: knobs.sparse_attention.clone(), ngram: knobs.speculative == Some(SpeculativeMode::Ngram), cards, sampling, adapter: None })
    }

    /// The values the defaults chose, as `name value` lines for the startup log.
    pub fn chosen_defaults(&self) -> &[String] {
        &self.deployment.chosen
    }

    /// The per-stage description of the placement over the offered cards, or the host tier's single stage.
    pub fn placement(&self) -> String {
        match place(&self.deployment.footprint, &self.cards) {
            Some(placed) if !self.cards.is_empty() => ricursive::placement::describe(&self.deployment.footprint, &placed),
            _ => format!("host tier, {:.1} GiB", self.deployment.footprint.single_card_bytes() as f64 / (1u64 << 30) as f64),
        }
    }

    /// The base this resident serves, what an adapter release is bound to.
    pub fn served_base(&self) -> ServedBase {
        ServedBase::of_family(RICURSIVE_FAMILY, self.dir.clone(), Some(self.id.clone()))
    }

    /// Whether the placement of this resident is over cards (registered with `register_multi`) or on the host.
    pub fn on_cards(&self) -> bool {
        !self.cards.is_empty()
    }

    /// This checkpoint served with `loaded` applied: a model of its own under the adapter's id, with its own instance
    /// (the engine applies one adapter at run time, so an adapter is a second copy of the weights, not a slot in a
    /// bank). The adapter is checked against this model's configuration here, at startup, and refused by name when it
    /// does not fit: another architecture, a layer the model does not have, a linear it does not have, rank 0.
    pub fn with_adapter(&self, loaded: LoadedAdapter) -> Result<RicursiveResident, String> {
        if self.adapter.is_some() {
            return Err(format!("{}: is already an adapter model; offer the adapter on the base {}", self.id, self.id));
        }
        ricursive::train::adapter::check_fits(&self.cfg, &loaded.adapter).map_err(|e| format!("adapter {}: {e}", loaded.release.path.display()))?;
        let mut served = self.clone();
        served.id = loaded.release.id.clone();
        served.adapter = Some(Arc::new(loaded));
        Ok(served)
    }

    /// The release this model serves on top of its checkpoint, if it is an adapter model.
    pub fn adapter_release(&self) -> Option<&crate::adapter_release::AdapterRelease> {
        self.adapter.as_ref().map(|a| &a.release)
    }

    /// The most tokens a sequence may hold.
    pub fn max_seq_len(&self) -> u32 {
        self.deployment.serve.max_seq_len
    }

    fn manifest_for(&self) -> Manifest {
        let generate = llmserve::with_sampling_defaults(generate_action(), &self.id, &self.sampling);
        Manifest::new(&self.id, SUMMARY, vec![generate]).with_max_context_tokens(u64::from(self.max_seq_len()))
    }

    /// The scheduler and front end over an engine opened on `plan`, in the process's current device scope.
    fn build_instance(&self, plan: &model::shard::StreamPlan) -> Result<Box<dyn Instance>, String> {
        let tok = QwenBpe::from_file(&self.dir.join("tokenizer.json").to_string_lossy())?;
        let mut stop = data::generation::stop_ids(Some(&self.dir), &tok, None, Some("<|im_end|>")).map_err(|e| format!("ricursive: {e}"))?;
        for id in config_eos(&self.dir) {
            if !stop.contains(&id) {
                stop.push(id);
            }
        }
        let format = match ChatFormat::own_template(&self.dir) {
            Ok(f) => f,
            Err(why) => {
                residency::log::warn(&format!("ricursive: {}: no usable chat template ({why}); only raw prompts (`chat: false`) are accepted", self.id));
                ChatFormat::None
            }
        };
        let front = Arc::new(TextFrontend::new(tok, stop, "ricursive", Arc::new(move |tok, inv| parse_request_as(tok, &format, inv))));
        let layers: Vec<usize> = (0..self.cfg.num_hidden_layers).collect();
        let src = ricursive::open_shared_checkpoint(&self.dir.to_string_lossy(), &self.cfg, &layers)?;
        let serve = self.deployment.serve.clone();
        let live = serve.max_batch as usize;
        let mut engine = RicursiveEngine::from_source_staged(self.cfg.clone(), src, serve, None, plan)?;
        if let Some(path) = &self.sparse_attention {
            let indexer = Indexer::load(path).map_err(|e| format!("ricursive: --sparse-attention {path}: {e}"))?;
            engine.set_lsa(&indexer).map_err(|e| format!("ricursive: --sparse-attention {path}: {e}"))?;
        }
        if let Some(adapter) = &self.adapter {
            engine.attach_adapter(&adapter.adapter).map_err(|e| format!("{}: {e}", self.id))?;
        }
        let mut sched = Scheduler::new(engine, live);
        if self.ngram {
            sched.set_drafter(Box::new(NgramDrafter::new(NGRAM_ORDER)));
        }
        Ok(Box::new(ScheduledInstance::new(front, sched, live)))
    }
}

impl ResidentModel for RicursiveResident {
    fn manifest(&self) -> Manifest {
        self.manifest_for()
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(&self.id, "default")
    }
    fn batch_limit(&self) -> Option<usize> {
        Some(self.deployment.serve.max_batch as usize)
    }
    /// A server with cards places Ripple through [`MultiDeviceResidentModel`]; this is the host tier's footprint
    /// (the whole deployment as one stage), which is what a server with no card registers the resident for.
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::new(self.deployment.footprint.single_card_bytes(), HOST_OVERHEAD_BYTES)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let plan = model::shard::StreamPlan::single(self.cfg.num_hidden_layers);
        on_device(device, || self.build_instance(&plan))?
    }
}

impl MultiDeviceResidentModel for RicursiveResident {
    fn estimate_multi(&self, _key: &InstanceKey) -> MultiDeviceCost {
        match place(&self.deployment.footprint, &self.cards) {
            Some(placed) => MultiDeviceCost::new(placed.iter().map(|p| (Device::Gpu(p.shard.gpu_index as u32), p.bytes + self.deployment.footprint.per_stage)).collect(), HOST_OVERHEAD_BYTES),
            None => MultiDeviceCost::default(),
        }
    }
    fn unavailable_reason(&self, _key: &InstanceKey) -> Option<String> {
        place(&self.deployment.footprint, &self.cards).is_none().then(|| format!("ricursive: {}: the deployment no longer places on the offered cards", self.id))
    }
    fn activate_multi(&self, _key: &InstanceKey, devices: &[Device]) -> Result<Box<dyn Instance>, String> {
        let wanted: Vec<usize> = devices.iter().filter_map(|d| if let Device::Gpu(i) = d { Some(*i as usize) } else { None }).collect();
        let cards: Vec<(usize, u64)> = self.cards.iter().copied().filter(|(i, _)| wanted.contains(i)).collect();
        let plan = stream_plan(&self.deployment.footprint, &cards).ok_or_else(|| format!("ricursive: {} does not place on cards {wanted:?}", self.id))?;
        self.build_instance(&plan)
    }
}

/// `brain do ricursive generate` and the SDK: the resident of the resolved model directory, built over every card the
/// process may use (the host tier without one) on the first request and kept for the next.
pub struct RicursiveProvider {
    state: Arc<ProviderState>,
}

struct ProviderState {
    weights: Option<PathBuf>,
    live: std::sync::Mutex<Option<Box<dyn Instance>>>,
}

impl RicursiveProvider {
    /// A provider over the model directory the resolver chose (`None`: nothing is pulled).
    pub fn new(weights: Option<PathBuf>) -> RicursiveProvider {
        RicursiveProvider { state: Arc::new(ProviderState { weights, live: std::sync::Mutex::new(None) }) }
    }
}

impl Provider for RicursiveProvider {
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
        generate_action()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let mut live = self.state.live.lock().map_err(|_| "ricursive: an earlier request panicked while holding the model".to_string())?;
        if live.is_none() {
            let dir = self.state.weights.as_ref().ok_or("ricursive: no model on this machine (brain pull ricursiveAI/Ripple-65B-Preview-NVFP4)")?;
            let gpus = gpu_core::capacity::available_gpus();
            let resident = RicursiveResident::from_hf_dir(&dir.to_string_lossy(), MODEL, &RicursiveServeConfig::default(), &gpus, 0)?;
            let key = InstanceKey::new(MODEL, "default");
            *live = Some(if gpus.is_empty() {
                resident.activate(&key, Device::Cpu)?
            } else {
                let devices: Vec<Device> = resident.estimate_multi(&key).devices().collect();
                resident.activate_multi(&key, &devices)?
            });
        }
        live.as_mut().expect("built above").run("generate", inv, progress)
    }
}

/// Ripple's [`ArchSpec`](brain_modelstore::resolve::ArchSpec): the Hugging Face directory a pull leaves in the store
/// (a `config.json` declaring `RicursiveForCausalLM`, safetensors shards and the root `tokenizer.json`), read in place.
pub struct RicursiveSpec;

impl brain_modelstore::resolve::ArchSpec for RicursiveSpec {
    fn arch(&self) -> &'static str {
        "ricursive"
    }

    fn roles(&self) -> &'static [&'static str] {
        &["weights", "tokenizer"]
    }

    fn classify(&self, records: &[brain_modelstore::inventory::ArtifactRecord], inventory_root: &Path) -> Vec<(usize, String, brain_modelstore::resolve::Confidence)> {
        use brain_modelstore::inventory::ArtifactKind;
        use brain_modelstore::resolve::{classify_tokenizer_role, Confidence};
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate().filter(|(_, r)| r.usable() && r.kind == ArtifactKind::HfDir) {
            let Ok(bytes) = std::fs::read(rec.path.join("config.json")) else { continue };
            let Ok(config) = serde_json::from_slice::<serde_json::Value>(&bytes) else { continue };
            if brain_modelstore::family_of_config(&config) == Some("ricursive") {
                out.push((idx, "weights".to_string(), Confidence::Declared));
            }
        }
        let weights_candidates: Vec<&Path> = out.iter().map(|(idx, ..)| records[*idx].path.as_path()).collect();
        classify_tokenizer_role(records, inventory_root, "tokenizer", &weights_candidates, &mut out);
        out
    }

    /// A pulled directory is named for the repo it came from (`<vendor>/<repo>`), so a served model answers to what
    /// the client asked for.
    fn assemble(
        &self,
        chosen: &std::collections::BTreeMap<String, usize>,
        records: &[brain_modelstore::inventory::ArtifactRecord],
        _overrides: &std::collections::BTreeMap<String, String>,
    ) -> Result<brain_modelstore::resolve::AssembleOutcome, String> {
        use brain_modelstore::resolve::{AssembleOutcome, AssembledVariant};
        let weights = *chosen.get("weights").ok_or("ricursive assemble: no weights chosen")?;
        chosen.get("tokenizer").ok_or("ricursive assemble: no tokenizer chosen")?;
        let path = &records[weights].path;
        let id = path.file_name().zip(path.parent().and_then(Path::file_name)).map(|(repo, vendor)| format!("{}/{}", vendor.to_string_lossy(), repo.to_string_lossy()));
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: id.unwrap_or_else(|| "local/ricursive".to_string()), variant: None }))
    }

    fn validate(&self, assembly: &capability::Assembly) -> Result<(), String> {
        assembly.roles.get("weights").ok_or("ricursive validate: assembly has no weights role")?;
        assembly.roles.get("tokenizer").ok_or("ricursive validate: assembly has no tokenizer role")?;
        Ok(())
    }
}
