// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident-model adapter putting Qwen3.8-27B's paged serving engine
//! (`qwen35::serve::Engine`) behind the residency [`Executor`]. Like every
//! brain LLM resident it is driven by the generic continuous-batching
//! `model::serve::Scheduler` through `llmserve` (all admitted sequences'
//! decode rows go through ONE batched forward per iteration); see
//! `crates/qwen35/src/serve.rs`'s own module doc for the engine's scope
//! (single GPU, fp32 weights, fp32 KV, one KV block per sequence).
//!
//! Two ways in. [`Qwen35Resident::from_card`] serves a checkpoint the
//! model-dir scan found (family `"qwen35"` - what `Qwen35::save` stamps,
//! distinct from the MoE sibling's own `"qwen35moe"` family), under its own
//! card id and with its sibling `tokenizer.json`. [`Qwen35Resident::
//! from_assembly`] is the other: `weights`/`tokenizer` resolved through
//! `qwen35::spec::Qwen35Spec` (the model-store resolver scanning the whole
//! models directory) rather than one specific already-discovered card - what
//! `crates/serving/src/executor.rs::build_executor` calls.
//!
//! Sizing knobs (`ctx`, `max_batch`) arrive as plain constructor data -
//! [`Qwen35Sizing`], what `build_executor` threads down from the parsed CLI -
//! never read from the environment: an env var is global, late-binding state
//! a `serve` flag parses once instead.
//!
//! `activate()` falls back to a GGUF checkpoint's own embedded
//! `tokenizer.ggml.*` KV when no `tokenizer` role resolved (see
//! `crate::resident_llm::QwenResident::activate`, the sibling this mirrors)
//! before giving up - `Engine` itself still never touches a `.gguf` file at
//! all, so a GGUF checkpoint gets a real tokenizer but no further than that
//! today.
//!
//! A THIRD way in lives at the bottom of this file:
//! [`gguf_resident`], the factory for
//! `qwen35::int8_gguf_resident::Qwen35GgufResident` - INT8 weights read
//! straight from a released GGUF and layer-sharded across as many cards as
//! they need, NOT a mode of the one above. The two coexist: this one still
//! serves an fp32 brain checkpoint that fits one card, that one serves a
//! released quant on a box with enough cards and no room on disk for a
//! ~108 GB fp32 conversion. The model-dir scan (`serving::model_dir`) builds
//! one per released `qwen35` GGUF it finds, under that file's store id; they
//! are configured by [`Qwen35GgufServeConfig`] - server-wide sizing and tier
//! defaults, parsed data from the CLI, never environment variables.

use capability::{Invocation, Manifest};
use data::qwen_tokenizer::QwenBpe;
use qwen3::chat::parse_request;
use qwen35::caps::with_template_flavor_default;
use qwen35::config::{LayerType, Qwen35Config};
use qwen35::serve::{Engine, Scheduler};
use std::sync::Arc;

use residency::multi::{MultiDeviceCost, MultiDeviceResidentModel};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

use crate::adapter_registry::{AdapterBankSettings, AdapterHost, AdapterRegistry, AdaptedInstance};
use crate::adapter_release::{LoadedAdapter, ServedBase, QWEN35_FAMILY};

use crate::resident_llm::{est_vram, generate_spec, on_device};
use llmserve::{ScheduledInstance, TextFrontend};

// Re-exported tier types the config struct speaks in, so a CLI builder (or
// a test) names ONE path for each.
pub use model::kv_tier::KvTier as Qwen35KvTier;
pub use model::ops::TierPolicy as Qwen35TierPolicy;

/// Everything `brain serve` parses for the Qwen3.8 family, threaded into
/// `serving::build_executor` as one value: [`fp32`] sizes the
/// brain-checkpoint resident, [`gguf`] configures every multi-card INT8
/// released-GGUF resident (see [`Qwen35GgufServeConfig`]). One struct per
/// family in the CLI's parsed output keeps the two residents' knobs from
/// being confused with each other's.
#[derive(Clone, Debug, Default)]
pub struct Qwen35ServeConfig {
    pub fp32: Qwen35Sizing,
    pub gguf: Qwen35GgufServeConfig,
}

/// Catalog id: a `brain/`-prefixed synthetic id, matching `qwen35::caps::MODEL`
/// exactly.
const MODEL: &str = qwen35::caps::MODEL;

/// The single-GPU fp32 resident's sizing, as parsed data. Historical
/// defaults, byte for byte (ctx 4096, max_batch 4 - what the removed
/// `BRAIN_QWEN35_CTX`/`BRAIN_QWEN35_MAX_BATCH` readers defaulted to) so an
/// operator who sets nothing keeps the prior behavior exactly.
#[derive(Clone, Copy, Debug)]
pub struct Qwen35Sizing {
    /// The hard `prompt + max_new` cap for any ONE sequence
    /// (`Engine::from_map`'s `max_seq_len`, which this engine also uses as
    /// its per-sequence block size).
    pub ctx: u32,
    /// How many sequences may be RESIDENT at once (`Engine::from_map`'s
    /// `max_concurrent`, i.e. `num_blocks`) and so decode together in one
    /// batched step.
    pub max_batch: u32,
}

impl Default for Qwen35Sizing {
    fn default() -> Self {
        Qwen35Sizing { ctx: 4096, max_batch: 4 }
    }
}

impl Qwen35Sizing {
    /// Clamp both knobs to their 1 floor - what the old env readers did to
    /// any parsed value, now applied to the parsed struct once, here.
    pub fn sanitized(self) -> Qwen35Sizing {
        Qwen35Sizing { ctx: self.ctx.max(1), max_batch: self.max_batch.max(1) }
    }
}

/// The Qwen3.8-27B dense hybrid Gated-DeltaNet/GQA decoder behind the
/// scheduler. `path`/`tokenizer` are resolved through `qwen35::spec::
/// Qwen35Spec`'s `weights`/`tokenizer` roles (see [`Self::from_assembly`]).
/// See this module's own doc for the exact (single-GPU, fp32-only) scope.
pub struct Qwen35Resident {
    id: String,
    path: String,
    tokenizer: String,
    sizing: Qwen35Sizing,
}

impl Qwen35Resident {
    /// `None` when the resolver names no `weights`, or names one this
    /// engine cannot load: `Engine` only ever opens brain-native
    /// safetensors (this module's own doc) - a resolved `.gguf` is the OTHER
    /// resident's checkpoint (`crate::resident_qwen35::gguf_resident`),
    /// not a mode of this one.
    pub fn from_assembly(assembly: &capability::Assembly) -> Option<Qwen35Resident> {
        Self::from_assembly_sized(assembly, Qwen35Sizing::default())
    }

    /// [`from_assembly`](Self::from_assembly) with explicit sizing - what
    /// `build_executor` calls with the CLI-parsed [`Qwen35Sizing`].
    pub fn from_assembly_sized(assembly: &capability::Assembly, sizing: Qwen35Sizing) -> Option<Qwen35Resident> {
        let path = assembly.roles.get("weights").map(|p| p.to_string_lossy().into_owned()).filter(|p| !p.ends_with(".gguf"))?;
        let tokenizer = assembly.roles.get("tokenizer").map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
        Some(Qwen35Resident { id: MODEL.to_string(), path, tokenizer, sizing: sizing.sanitized() })
    }

    /// The model-dir counterpart of [`from_assembly`](Self::from_assembly): a
    /// checkpoint discovered by `serving::model_dir` (family `"qwen35"` - what
    /// `Qwen35::save` stamps on its `ModelCard`) served under its OWN card id
    /// rather than the resolver's [`MODEL`].
    ///
    /// `tokenizer` is the sibling `tokenizer.json` the scan found. It is
    /// required: `Engine` never opens a `.gguf`, so there is no
    /// embedded-tokenizer fallback - a checkpoint without one is declined at
    /// discovery instead.
    pub fn from_card(path: &str, card: &checkpoint::st::ModelCard, tokenizer: Option<&str>) -> Result<Qwen35Resident, String> {
        Self::from_card_sized(path, card, tokenizer, Qwen35Sizing::default())
    }

    /// [`from_card`](Self::from_card) with explicit sizing - what the
    /// model-dir scan calls with the CLI-parsed [`Qwen35Sizing`].
    pub fn from_card_sized(path: &str, card: &checkpoint::st::ModelCard, tokenizer: Option<&str>, sizing: Qwen35Sizing) -> Result<Qwen35Resident, String> {
        let tokenizer = tokenizer.filter(|t| !t.is_empty()).ok_or("qwen35: no sibling tokenizer.json")?;
        Ok(Qwen35Resident { id: card.id.clone(), path: path.to_string(), tokenizer: tokenizer.to_string(), sizing: sizing.sanitized() })
    }

    /// The hard per-sequence `prompt + max_new` cap, which this engine also
    /// uses as its physical block size (`Engine::from_map`'s `max_seq_len`
    /// parameter).
    fn ctx(&self) -> u32 {
        self.sizing.ctx
    }

    /// How many sequences may be resident at once (`Engine::from_map`'s
    /// `max_concurrent`, i.e. its `num_blocks`), all decoded in one batched
    /// step.
    fn max_concurrent(&self) -> u32 {
        self.sizing.max_batch
    }
}

impl ResidentModel for Qwen35Resident {
    fn manifest(&self) -> Manifest {
        // `self.ctx()` is exactly the `max_seq_len` `activate()` below builds
        // the engine with, so advertising it is a safe, never-overstated
        // floor on real serving capacity.
        Manifest::new(
            &self.id,
            "text generation (Qwen3.8-27B dense hybrid Gated-DeltaNet/GQA decoder; single-GPU, fp32 weights + fp32 KV; continuous batching: every resident sequence decodes in one batched step per iteration -- see crate::resident_qwen35's module doc)",
            vec![generate_spec("generate text (Qwen3.8-27B; chat template optional)", true)],
        )
        .with_max_context_tokens(self.ctx() as u64)
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(&self.id, "default")
    }
    fn batch_limit(&self) -> Option<usize> {
        Some(self.max_concurrent() as usize)
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        let cost = est_vram(&self.path);
        let Ok(reader) = checkpoint::weightio::WeightReader::open(&self.path) else {
            return cost;
        };
        let cfg = Qwen35Config::from_json(&reader.config());
        let ctx = self.ctx() as u64;
        let max_concurrent = self.max_concurrent() as u64;
        let n_full = cfg.layer_types().iter().filter(|t| **t == LayerType::Full).count() as u64;
        // The GQA side of `Engine::kv_pool_bytes()` ONLY, computed
        // independently via `Qwen35Config`'s own public accessors rather than
        // calling into `qwen35::serve` (whose `GdnSlot` sizing helper is
        // crate-private, by design). Genuinely missing: the GDN recurrent-
        // state/conv-history slot pool is real but small (O(1) state per
        // sequence, not O(context length) like the GQA side), so this
        // estimate UNDER-counts by that (bounded, much smaller) amount
        // rather than over-claiming a number this crate cannot cheaply
        // verify from outside `serve.rs`.
        let gqa_bytes = n_full * max_concurrent * 2 * ctx * cfg.kv_dim() as u64 * 4;
        MemCost::new(cost.vram + gqa_bytes, cost.ram)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // Open first so a GGUF can supply its own embedded tokenizer, exactly
        // like crate::resident_llm::QwenResident::activate. `Engine` itself
        // still only loads brain-native safetensors below (see this module's
        // own doc), so this reader exists purely for the tokenizer fallback.
        let reader = checkpoint::weightio::WeightReader::open(&self.path).map_err(|e| format!("qwen35: {e}"))?;
        // Tokenizer precedence: an explicit sibling tokenizer.json wins; else
        // a .gguf builds from its embedded tokenizer.ggml.* KV; else there is
        // nothing to tokenize with.
        let (tok, tok_dir, gguf_eos) = if !self.tokenizer.is_empty() {
            (QwenBpe::from_file(&self.tokenizer)?, std::path::Path::new(&self.tokenizer).parent(), None)
        } else if let Some(gt) = reader.tokenizer() {
            (QwenBpe::from_gguf(&gt).map_err(|e| format!("qwen35: {e}"))?, None, gt.eos)
        } else {
            return Err("qwen35: no tokenizer (serve a checkpoint with a sibling tokenizer.json, or a GGUF with an embedded tokenizer)".to_string());
        };
        // The checkpoint's own eos ids, plus ChatML's end-of-turn marker: this
        // resident renders every prompt as ChatML.
        let stop = data::generation::stop_ids(tok_dir, &tok, gguf_eos, Some("<|im_end|>")).map_err(|e| format!("qwen35: {e}"))?;
        let ctx = self.ctx();
        let max_concurrent = self.max_concurrent();
        let path = self.path.clone();
        let sched = on_device(device, move || -> Scheduler {
            // `Engine` only loads brain-native safetensors (`checkpoint::load`)
            // - see this module's own doc on why there is no GGUF arm here.
            let container = checkpoint::load(&path);
            let cfg = Qwen35Config::from_json(&container.header["config"]);
            let weights = container.by_role("");
            let engine = Engine::from_map(cfg, &weights, ctx, max_concurrent);
            Scheduler::new(engine, max_concurrent as usize)
        })?;
        Ok(Box::new(ScheduledInstance::new(TextFrontend { tok, stop, label: "qwen35", parse: parse_qwen35 }, sched, max_concurrent as usize)))
    }
}

/// This model renders the 3.8 chat template by default - `parse_request`
/// alone defaults to the Qwen3-era flavor, which would silently mis-render
/// its prompt.
fn parse_qwen35(tok: &QwenBpe, inv: &Invocation) -> Result<qwen3::chat::ParsedRequest, String> {
    parse_request(tok, &with_template_flavor_default(inv))
}

/// Server-wide defaults for every released-GGUF Qwen3.8 resident the model
/// scan registers, as parsed data - what `brain serve`'s `--qwen35-*` flags
/// build and `build_executor` hands down. No environment variables, and no
/// per-model switch: which GGUFs are served is decided by what the store
/// holds, how they are served by these.
#[derive(Clone, Debug, Default)]
pub struct Qwen35GgufServeConfig {
    /// Per-sequence `prompt + max_new` ceiling (also every stage's KV/GDN
    /// cache capacity, charged into the placement). `None` sizes it from the
    /// memory the placement leaves ([`qwen35::int8_gguf_resident::auto_cap`]).
    pub ctx: Option<u32>,
    /// Per-leaf weight-tier policy (`model::ops::TierPolicy::parse`'s
    /// grammar). `None` keeps the historical default (uniform INT8).
    pub tier: Option<model::ops::TierPolicy>,
    /// GQA KV-cache storage tier. `None` keeps the historical default
    /// (f32).
    pub kv: Option<model::kv_tier::KvTier>,
    /// How many sequences one batched decode step carries (the resident's
    /// decode slots, charged into the placement). `None` takes
    /// [`DEFAULT_MAX_BATCH`].
    pub max_batch: Option<u32>,
    /// Cached prompts kept for prefix reuse (see
    /// `Qwen35GgufResident::with_prefix_slots`). `None` takes
    /// [`DEFAULT_PREFIX_SLOTS`]; 0 turns the cache off.
    pub prefix_slots: Option<u32>,
    /// Speculative decoding (`--speculative`). `None` takes
    /// [`SpeculativeMode::Auto`](qwen35::int8_gguf_resident::SpeculativeMode::Auto):
    /// the best drafter the checkpoint carries, plain decoding when it has none.
    pub speculative: Option<qwen35::int8_gguf_resident::SpeculativeMode>,
    /// Prompt tokens one sequence gets of each scheduler iteration's prefill
    /// while others wait (`--qwen35-prefill-slice`). `None` takes the engine's
    /// default ([`PrefillPolicy::default`](qwen35::int8_gguf_resident::PrefillPolicy)).
    pub prefill_slice: Option<u32>,
    /// Prompt rows one iteration prefills across sequences in a single forward
    /// (`--qwen35-prefill-batch`); 0 prefills one sequence's slice at a time.
    /// `None` takes the engine's default.
    pub prefill_batch: Option<u32>,
}

impl Qwen35GgufServeConfig {
    /// The weight-tier knob with its historical default filled in.
    pub fn tier_or_default(&self) -> model::ops::TierPolicy {
        self.tier.clone().unwrap_or_else(|| model::ops::TierPolicy::uniform(gpu_core::select::Dtype::I8))
    }

    /// The KV-tier knob with its historical default filled in.
    pub fn kv_or_default(&self) -> model::kv_tier::KvTier {
        self.kv.unwrap_or(model::kv_tier::KvTier::F32)
    }

    /// The prefix-cache knob with its default filled in.
    pub fn prefix_slots_or_default(&self) -> u32 {
        self.prefix_slots.unwrap_or(DEFAULT_PREFIX_SLOTS)
    }

    /// The prefill knobs with their defaults filled in.
    pub fn prefill_policy_or_default(&self) -> qwen35::int8_gguf_resident::PrefillPolicy {
        let default = qwen35::int8_gguf_resident::PrefillPolicy::default();
        qwen35::int8_gguf_resident::PrefillPolicy {
            slice_tokens: self.prefill_slice.map_or(default.slice_tokens, |n| n as usize),
            batch_tokens: self.prefill_batch.map_or(default.batch_tokens, |n| n as usize),
        }
    }

    /// The speculation knob with its default filled in.
    pub fn speculative_or_default(&self) -> qwen35::int8_gguf_resident::SpeculativeMode {
        self.speculative.unwrap_or(qwen35::int8_gguf_resident::SpeculativeMode::Auto)
    }

    /// The batch knob with its default filled in.
    pub fn max_batch_or_default(&self) -> u32 {
        self.max_batch.unwrap_or(DEFAULT_MAX_BATCH).max(1)
    }
}

/// Decode slots a GGUF resident is built with when `--qwen35-max-batch` is not
/// given: concurrent requests share every weight read, so a single slot would
/// serve them one at a time. Small enough that the decode state (about 0.4 GiB
/// a sequence at the default 2048 context with an f32 cache) leaves the
/// two-24-GiB-card placement unchanged.
pub const DEFAULT_MAX_BATCH: u32 = 8;

/// Prompts a GGUF resident keeps for prefix reuse when `--qwen35-prefix-slots`
/// is not given. An agent resends its whole history every turn; each cached
/// prompt costs one sequence slot of KV (charged into the placement), so a few
/// conversations stay warm without crowding out the batch.
pub const DEFAULT_PREFIX_SLOTS: u32 = 4;

/// Build the multi-GPU resident for one released GGUF, to be registered with
/// `Executor::register_multi` under `id`.
///
/// `gpus` is `build_executor`'s own budgeted GPU list as `(index, TOTAL
/// bytes)` - already narrowed to the cards the process's `--device` selection
/// allows - and `reserved` the per-card headroom it keeps free, so what is
/// handed on is each card's genuinely USABLE capacity, the same figure the
/// scheduler budgets against. Passing capacity (not just identity) is what
/// lets `model::shard::plan_fewest_devices` size the split to the hardware
/// instead of assuming matched cards.
///
/// The context length is `cfg.ctx` when the operator fixed one, else the
/// longest the cards hold once the weights are placed. `Err` names why the
/// model could never load (no GPU budgeted, an unreadable or non-Qwen3.8
/// file, weights that do not fit the cards, no embedded tokenizer), so the
/// caller can leave it out of the served set and say so: a model that is
/// listed is one whose first request can load.
pub fn gguf_resident(gguf_path: &str, id: &str, gpus: &[(u32, u64)], reserved: u64, cfg: &Qwen35GgufServeConfig) -> Result<Qwen35GgufServed, String> {
    gguf_resident_banked(gguf_path, id, gpus, reserved, cfg, None)
}

/// [`gguf_resident`] with an adapter bank of `bank` built beside the weights
/// (its device bytes are part of the placement), so adapters can be offered
/// on it ([`AdapterHost`]).
pub fn gguf_resident_banked(gguf_path: &str, id: &str, gpus: &[(u32, u64)], reserved: u64, cfg: &Qwen35GgufServeConfig, bank: Option<AdapterBankSettings>) -> Result<Qwen35GgufServed, String> {
    use qwen35::int8_gguf_resident::{auto_cap, resident_config, Qwen35GgufResident, SpecServing};
    let (tier, kv, max_batch, prefix_slots) = (cfg.tier_or_default(), cfg.kv_or_default(), cfg.max_batch_or_default(), cfg.prefix_slots_or_default());
    let spec = SpecServing { mode: cfg.speculative_or_default(), ..SpecServing::default() };
    let mg = checkpoint::gguf::MmapGguf::open(gguf_path).map_err(|e| format!("cannot open '{gguf_path}': {e}"))?;
    let model_cfg = resident_config(&mg, 1).map_err(|e| format!("'{gguf_path}' is not a servable Qwen3.8 GGUF: {e}"))?;
    let devices_for = |cap: u32| -> Vec<(Device, u64)> {
        // Speculation's scratch and drafter live outside the planned placement,
        // so the budget handed to the planner leaves their bytes free.
        let spec_bytes = spec.reserve_bytes(&model_cfg, cap, max_batch + prefix_slots, &tier, kv);
        gpus.iter().map(|&(i, total)| (Device::Gpu(i), total.saturating_sub(reserved).saturating_sub(spec_bytes))).collect()
    };
    let cap = match cfg.ctx {
        Some(ctx) => ctx.max(1),
        None => {
            let devices = devices_for(1);
            if devices.is_empty() {
                return Err("no GPU is budgeted for it (its weights are device buffers; there is no CPU path)".to_string());
            }
            // The speculation reserve grows with the context (the drafter's KV), so
            // the context is the largest one that fits beside ITS OWN reserve.
            let upper = auto_cap(&model_cfg, &devices, &tier, kv, max_batch, prefix_slots).ok_or_else(|| format!("{gguf_path} does not fit across the {} budgeted device(s) even with a one-token context", devices.len()))?;
            
            largest_fitting(upper, |c| auto_cap(&model_cfg, &devices_for(c), &tier, kv, max_batch, prefix_slots).is_some_and(|fit| fit >= c))
        }
    };
    let mut resident = Qwen35GgufResident::new(gguf_path.to_string(), devices_for(cap), cap, tier).with_kv_tier(kv).with_max_batch(max_batch).with_prefix_slots(prefix_slots).with_speculation(spec).with_prefill_policy(cfg.prefill_policy_or_default()).with_model_id(id);
    if let Some(bank) = &bank {
        resident = resident.with_adapter_bank(bank.request());
    }
    match resident.unservable_reason() {
        Some(reason) => Err(reason),
        None => Ok(Qwen35GgufServed::new(resident, bank, gguf_path, id)),
    }
}

/// The largest `c` in `1..=upper` for which `fits(c)` holds, given that it
/// holds for small `c` and never again once it fails (a context that does not
/// fit does not fit with a bigger reserve either). 1 when nothing larger fits.
fn largest_fitting(upper: u32, fits: impl Fn(u32) -> bool) -> u32 {
    let (mut lo, mut hi) = (1, upper.max(1));
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// A released Qwen3.8 GGUF resident with the adapters offered beside it.
///
/// Everything about the model is [`Qwen35GgufResident`]'s (it derefs to it:
/// placement, activation, the engine); this adds the registry of adapters the
/// operator wants served and the instance that applies them. Each served
/// adapter is a model of its own ([`crate::adapter_registry::AdapterView`])
/// registered with the executor beside this one, sharing its instance, so a
/// batch mixes base and adapter requests in one batched decode step.
pub struct Qwen35GgufServed {
    inner: qwen35::int8_gguf_resident::Qwen35GgufResident,
    adapters: Arc<AdapterRegistry>,
}

impl Qwen35GgufServed {
    fn new(inner: qwen35::int8_gguf_resident::Qwen35GgufResident, bank: Option<AdapterBankSettings>, gguf_path: &str, id: &str) -> Qwen35GgufServed {
        // An adapter trains on the full-precision release and serves on its
        // quantisation: the card names the release, not this file.
        let base = ServedBase::of_family(QWEN35_FAMILY, gguf_path, Some(id.to_string())).quantizing(&QWEN35_SOURCES);
        Qwen35GgufServed { inner, adapters: AdapterRegistry::new(bank, base) }
    }
}

/// Ids of the full-precision releases the Qwen3.8-27B quantisations are made
/// from, which an adapter's card names as its base: the repo ids and the bare
/// directory names `brain qwen35 finetune` records for a base it was pointed at.
const QWEN35_SOURCES: [&str; 4] = ["Qwen/Qwen3.8-27B-FP8", "Qwen/Qwen3.8-27B", "Qwen3.8-27B-FP8", "Qwen3.8-27B"];

impl std::ops::Deref for Qwen35GgufServed {
    type Target = qwen35::int8_gguf_resident::Qwen35GgufResident;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl AdapterHost for Qwen35GgufServed {
    fn adapters(&self) -> &Arc<AdapterRegistry> {
        &self.adapters
    }

    fn register_view(self: &Arc<Self>, executor: &residency::Executor, adapter: LoadedAdapter) -> Result<(), String> {
        executor.register_multi(Arc::new(self.adapter_view(adapter)?));
        Ok(())
    }
}

impl ResidentModel for Qwen35GgufServed {
    fn manifest(&self) -> Manifest {
        self.inner.manifest()
    }
    fn instance_key(&self, action: &str, inv: &Invocation) -> InstanceKey {
        self.inner.instance_key(action, inv)
    }
    fn estimate(&self, key: &InstanceKey) -> MemCost {
        self.inner.estimate(key)
    }
    fn batch_limit(&self) -> Option<usize> {
        self.inner.batch_limit()
    }
    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        self.inner.activate(key, device)
    }
}

impl MultiDeviceResidentModel for Qwen35GgufServed {
    fn unavailable_reason(&self, key: &InstanceKey) -> Option<String> {
        self.inner.unavailable_reason(key)
    }
    fn estimate_multi(&self, key: &InstanceKey) -> MultiDeviceCost {
        self.inner.estimate_multi(key)
    }

    /// The served instance: the checkpoint as a `Qwen35GgufEngine` (with its
    /// adapter bank, when one was asked for) under the generic
    /// continuous-batching scheduler, so up to `max_batch` requests of the
    /// base and of any adapters decode in ONE batched step per iteration.
    fn activate_multi(&self, _key: &InstanceKey, devices: &[Device]) -> Result<Box<dyn Instance>, String> {
        let (engine, tok, stop) = self.inner.activate_engine(devices)?;
        let capacity = self.inner.max_batch() as usize;
        let sched = model::serve::Scheduler::new(engine, capacity);
        let front = llmserve::TextFrontend { tok, stop, label: "qwen35", parse: qwen35::int8_gguf_resident::parse_request };
        Ok(Box::new(AdaptedInstance::new(front, sched, capacity, self.adapters.clone())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sizing knobs are constructor data with the historical defaults,
    /// sanitized to their 1 floor - never read from the environment.
    #[test]
    fn sizing_defaults_and_sanitization() {
        let d = Qwen35Sizing::default();
        assert_eq!((d.ctx, d.max_batch), (4096, 4));
        let s = Qwen35Sizing { ctx: 0, max_batch: u32::MAX }.sanitized();
        assert_eq!((s.ctx, s.max_batch), (1, u32::MAX));
    }

    /// No resolved `weights` role at all (an empty `Assembly`, matching "the
    /// resolver found nothing") must decline cleanly rather than build a
    /// resident whose every request then fails placement.
    #[test]
    fn from_assembly_is_none_without_a_resolved_weights_role() {
        let empty = capability::Assembly { id: String::new(), arch: "qwen35".to_string(), variant: None, roles: Default::default(), provenance: Vec::new() };
        assert!(Qwen35Resident::from_assembly(&empty).is_none());
    }

    /// A resolved `.gguf` `weights` role is the OTHER resident's checkpoint
    /// ([`gguf_resident`]'s `Qwen35GgufResident`) - `Engine` never opens a
    /// `.gguf` at all, so this must decline rather than try.
    #[test]
    fn from_assembly_declines_a_resolved_gguf_weights_role() {
        let assembly = capability::Assembly {
            id: String::new(),
            arch: "qwen35".to_string(),
            variant: None,
            roles: std::collections::BTreeMap::from([("weights".to_string(), std::path::PathBuf::from("/models/unsloth/Qwen3.8-27B-Q8_0.gguf"))]),
            provenance: Vec::new(),
        };
        assert!(Qwen35Resident::from_assembly(&assembly).is_none());
    }

    /// REGRESSION: this engine used to call `parse_request` directly, so it
    /// rendered the Qwen3-era chat template by default while `caps.rs` and
    /// the GGUF resident both correctly default to Qwen3.8's own template
    /// (`with_template_flavor_default`'s own doc). Mirrors `qwen35::caps`'s
    /// own `tokenizer_present_renders_the_qwen38_flavor_without_asking` test
    /// exactly - same env-gated real tokenizer, same assertion - because
    /// this engine must render identically to every other entry point into
    /// this model, not merely "some" template.
    #[test]
    fn parse_request_renders_the_qwen38_flavor_without_asking() {
        let Ok(tok_path) = std::env::var("QWEN_TOKENIZER") else {
            brain_testutil::skip("set QWEN_TOKENIZER to a real tokenizer.json to run this test");
            return;
        };
        let tok = QwenBpe::from_file(&tok_path).expect("load Qwen3 tokenizer");
        let inv = Invocation::new().set("prompt", serde_json::json!("what's the weather in Paris?")).set("chat", serde_json::json!(true));
        let req = parse_request(&tok, &with_template_flavor_default(&inv)).unwrap();
        assert_eq!(req.flavor, data::qwen_chat::TemplateFlavor::Qwen38, "this engine must default to the Qwen3.8 flavor, same as caps.rs and the GGUF resident");
    }

    /// REGRESSION for the whole point of this file: registering
    /// `Qwen35Resident` with the residency `Executor` must be enough, with NO
    /// `crates/apiserve`-side code, for `brain/qwen35` to appear on BOTH the
    /// OpenAI and the Anthropic `/v1/models` list with the chat capability.
    /// Drives a REAL `axum` router over a REAL `Executor`, not a
    /// hand-inspected manifest.
    ///
    /// Uses a resident whose weights path does not exist: `GET /v1/models`
    /// never activates a model, so this proves the auto-exposure wiring
    /// without needing a real checkpoint/tokenizer.
    #[test]
    fn brain_qwen35_is_auto_exposed_on_openai_and_anthropic_model_lists() {
        let resident = Qwen35Resident { id: MODEL.to_string(), path: "/nonexistent/qwen35.safetensors".to_string(), tokenizer: String::new(), sizing: Qwen35Sizing::default() };
        let models: Vec<std::sync::Arc<dyn ResidentModel>> = vec![std::sync::Arc::new(resident)];
        let mut budgets = residency::budget::Budgets::new();
        budgets.set(Device::Cpu, 8 << 30, 0);
        let exec = residency::Executor::start(models, budgets, residency::scheduler::Policy::default());

        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        for provider in [apiserve::Provider::OpenAI, apiserve::Provider::Anthropic] {
            let key = "sk-brain-test-key".to_string();
            let state = apiserve::AppState::new(exec.clone(), key.clone(), provider);
            let app = apiserve::router(state);
            let mut req = axum::http::Request::builder().method(axum::http::Method::GET).uri("/v1/models");
            req = match provider {
                apiserve::Provider::Anthropic => req.header("x-api-key", &key),
                _ => req.header(axum::http::header::AUTHORIZATION, format!("Bearer {key}")),
            };
            let req = req.body(axum::body::Body::empty()).unwrap();

            let (status, body): (axum::http::StatusCode, serde_json::Value) = rt.block_on(async {
                use tower::ServiceExt;
                let resp = app.oneshot(req).await.unwrap();
                let status = resp.status();
                let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
                (status, serde_json::from_slice(&bytes).unwrap())
            });
            assert_eq!(status, axum::http::StatusCode::OK, "{provider:?} GET /v1/models failed: {body}");
            let ids: Vec<&str> = body["data"].as_array().unwrap().iter().filter_map(|c| c["id"].as_str()).collect();
            assert!(ids.contains(&MODEL), "{provider:?}: expected {MODEL:?} in /v1/models, got {ids:?}");
        }
    }

    /// A minimal single-file GGUF carrying a real gpt2-scheme
    /// `tokenizer.ggml.*` KV block, same shape `model_dir.rs`'s own
    /// `write_gguf_qwen` test helper builds, written through the real
    /// `checkpoint::gguf_write::write` encoder rather than hand-assembled
    /// bytes.
    fn write_gguf_with_gpt2_tokenizer(path: &std::path::Path) {
        use checkpoint::gguf::GgufValue as V;
        let kv = vec![
            ("general.architecture".to_string(), V::String("qwen35".to_string())),
            ("tokenizer.ggml.model".to_string(), V::String("gpt2".to_string())),
            (
                "tokenizer.ggml.tokens".to_string(),
                V::Array(["<|endoftext|>", "<|im_start|>", "<|im_end|>", "h", "i", "hi"].into_iter().map(|s| V::String(s.to_string())).collect()),
            ),
            ("tokenizer.ggml.merges".to_string(), V::Array(vec![V::String("h i".to_string())])),
            ("tokenizer.ggml.token_type".to_string(), V::Array([3, 3, 3, 1, 1, 1].into_iter().map(V::I32).collect())),
            ("tokenizer.ggml.bos_token_id".to_string(), V::U32(0)),
            ("tokenizer.ggml.eos_token_id".to_string(), V::U32(2)),
        ];
        let tensor = checkpoint::gguf_write::TensorOut { name: "w".to_string(), shape: vec![4], ty: checkpoint::gguf::GgmlType::F32.id(), data: [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect() };
        checkpoint::gguf_write::write(path.to_str().unwrap(), &kv, std::slice::from_ref(&tensor), 32).unwrap();
    }

    /// With no sibling `tokenizer.json`, `activate()` must fall back to the
    /// GGUF's own embedded `tokenizer.ggml.*` KV
    /// (`crate::resident_llm::QwenResident::activate`'s already-working
    /// pattern) instead of immediately returning the "no tokenizer" error.
    /// `Engine` itself has no GGUF arm (this module's own doc), so activation
    /// still cannot fully succeed on a `.gguf` checkpoint -- it panics inside
    /// `checkpoint::load` (a pre-existing, documented limitation, not
    /// introduced here) once it gets past tokenizer construction. Catching
    /// that panic and asserting it is the safetensors-parse failure (not the
    /// tokenizer error) proves the fallback ran and reached the real next
    /// step.
    #[test]
    fn activate_falls_back_to_the_gguf_embedded_tokenizer() {
        let dir = std::env::temp_dir().join(format!("brain-qwen35-gguf-tok-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qwen35.gguf");
        write_gguf_with_gpt2_tokenizer(&path);

        let resident = Qwen35Resident { id: MODEL.to_string(), path: path.to_str().unwrap().to_string(), tokenizer: String::new(), sizing: Qwen35Sizing::default() };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| resident.activate(&InstanceKey::new(MODEL, "default"), Device::Cpu)));
        match result {
            // Getting a clean Err this early would mean the tokenizer step
            // itself rejected something; the fallback must at least get
            // past it.
            Ok(Err(e)) => panic!("expected the GGUF tokenizer fallback to succeed and activation to fail later inside checkpoint::load, got an early Err instead: {e}"),
            Ok(Ok(_)) => panic!("activate() unexpectedly succeeded fully on a .gguf checkpoint -- Engine has no GGUF arm (see this module's own doc); if that changed, update this test"),
            Err(payload) => {
                let msg = payload.downcast_ref::<String>().cloned().or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                assert!(msg.contains("cannot read"), "expected the checkpoint::load safetensors-parse panic (proving tokenizer construction succeeded first), got: {msg:?}");
                assert!(!msg.to_ascii_lowercase().contains("no tokenizer"), "the panic must not be the tokenizer-missing path: {msg:?}");
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    // ========== the multi-GPU GGUF factory: `gguf_resident` ==========

    const GIB: u64 = 1 << 30;

    /// A complete (weights + tokenizer) synthetic Qwen3.8 GGUF.
    fn write_servable_gguf(tag: &str) -> String {
        let path = std::env::temp_dir().join(format!("brain-qwen35-gguf-factory-{tag}-{}.gguf", std::process::id()));
        qwen35::gguf_import::testing::write_synthetic_servable_gguf(path.to_str().unwrap());
        path.to_string_lossy().into_owned()
    }

    /// Two 24 GiB cards, 2 GiB kept free on each.
    fn gpu_budget() -> Vec<(u32, u64)> {
        vec![(0, 24 * GIB), (1, 24 * GIB)]
    }

    const RESERVED: u64 = 2 * GIB;

    /// The resident is served under the id it is handed (the file's store id),
    /// not any constant, and manifest and instance key agree on it.
    #[test]
    fn the_resident_is_served_under_the_id_it_is_given() {
        let path = write_servable_gguf("id");
        let r = gguf_resident(&path, "unsloth/Qwen3.8-27B-GGUF", &gpu_budget(), RESERVED, &Qwen35GgufServeConfig::default()).expect("a complete GGUF must build");
        assert_eq!(r.model_id(), "unsloth/Qwen3.8-27B-GGUF");
        assert_eq!(r.manifest().model, "unsloth/Qwen3.8-27B-GGUF");
        assert_eq!(r.gguf_path(), path);
        std::fs::remove_file(path).ok();
    }

    /// Without a fixed context the cards decide: far more than any development
    /// default, placed within what the budget leaves after the reserve.
    #[test]
    fn an_unset_context_is_sized_from_the_memory_the_placement_leaves() {
        let path = write_servable_gguf("auto-ctx");
        let gpus = vec![(0, 4 * GIB)];
        let r = gguf_resident(&path, "t/auto", &gpus, RESERVED, &Qwen35GgufServeConfig::default()).expect("builds");
        assert!(r.cap() > 1_000_000, "a tiny model on a 2 GiB budget leaves room for a huge context: {}", r.cap());
        let placed: u64 = r.placement().iter().map(|p| p.3).sum();
        assert!(placed <= 2 * GIB, "the context must be charged into a placement that fits the usable bytes: {placed}");
        std::fs::remove_file(path).ok();
    }

    /// The knobs travel as data into the built resident; an explicit context is
    /// used verbatim.
    #[test]
    fn config_knobs_reach_the_resident_as_data() {
        let path = write_servable_gguf("knobs");
        let cfg = Qwen35GgufServeConfig {
            ctx: Some(8192),
            tier: Some(model::ops::TierPolicy::parse("i8").unwrap()),
            kv: Some(model::kv_tier::KvTier::INT8),
            max_batch: Some(4),
            prefix_slots: Some(2),
            speculative: Some(qwen35::int8_gguf_resident::SpeculativeMode::Off),
            ..Qwen35GgufServeConfig::default()
        };
        let r = gguf_resident(&path, "t/knobs", &gpu_budget(), RESERVED, &cfg).expect("builds");
        assert_eq!(r.cap(), 8192);
        assert_eq!(r.kv_tier(), model::kv_tier::KvTier::INT8);
        assert_eq!(r.max_batch(), 4);
        assert_eq!(r.prefix_slots(), 2);
        assert_eq!(r.speculation().mode, qwen35::int8_gguf_resident::SpeculativeMode::Off);
        // What `/v1/models` reports is the context the resident was built with.
        assert_eq!(r.manifest().max_context_tokens, Some(8192));
        std::fs::remove_file(path).ok();
    }

    /// An all-`None` config: the context sized from the memory the placement
    /// leaves, uniform INT8 weights, an f32 KV cache, [`DEFAULT_MAX_BATCH`]
    /// decode slots so concurrent requests share every weight read, and
    /// [`DEFAULT_PREFIX_SLOTS`] cached prompts.
    #[test]
    fn config_defaults_size_the_context_and_batch_the_requests() {
        let path = write_servable_gguf("defaults");
        let r = gguf_resident(&path, "t/defaults", &gpu_budget(), RESERVED, &Qwen35GgufServeConfig::default()).expect("builds");
        assert!(r.cap() > 2048, "sized by memory, not a fixed default: {}", r.cap());
        assert_eq!(r.kv_tier(), model::kv_tier::KvTier::F32);
        assert_eq!(r.max_batch(), DEFAULT_MAX_BATCH);
        assert_eq!(r.prefix_slots(), DEFAULT_PREFIX_SLOTS);
        assert_eq!(r.speculation().mode, qwen35::int8_gguf_resident::SpeculativeMode::Auto, "speculation is on by default and falls back to plain decode without a drafter");
        assert_eq!(r.manifest().max_context_tokens, Some(r.cap() as u64));
        std::fs::remove_file(path).ok();
    }

    /// A reserve that grows with the context must not drag the sized context
    /// down to a fixed point of "shrink to what fit at the previous size": the
    /// result is the largest context that fits beside its own reserve.
    #[test]
    fn the_context_is_the_largest_that_fits_beside_its_own_reserve() {
        // A budget of 2000 units; a context costs one per token plus a reserve of a quarter more.
        let fits = |c: u32| c + c / 4 <= 2000;
        let upper = 2000; // what fits with no reserve at all
        let best = largest_fitting(upper, fits);
        assert_eq!(best, 1600);
        assert!(fits(best) && !fits(best + 1));
        assert_eq!(largest_fitting(upper, |c| c <= 1), 1, "nothing beyond one token fits");
    }

    /// Everything that would make the first request fail is refused here,
    /// with the reason, instead of producing a model that is listed and never
    /// loads.
    #[test]
    fn a_resident_that_could_never_load_is_refused_with_the_reason() {
        let path = write_servable_gguf("refused");
        let no_gpu = gguf_resident(&path, "t/x", &[], RESERVED, &Qwen35GgufServeConfig::default()).err().expect("must be refused");
        assert!(no_gpu.contains("no GPU"), "{no_gpu}");
        let too_small = gguf_resident(&path, "t/x", &[(0, RESERVED + 64)], RESERVED, &Qwen35GgufServeConfig::default()).err().expect("must be refused");
        assert!(too_small.contains("does not fit"), "{too_small}");
        let fixed_ctx_too_small = gguf_resident(&path, "t/x", &[(0, RESERVED + 64)], RESERVED, &Qwen35GgufServeConfig { ctx: Some(1024), ..Default::default() }).err().expect("must be refused");
        assert!(fixed_ctx_too_small.contains("budgeted GPU(s) offer"), "{fixed_ctx_too_small}");
        let missing = gguf_resident("/nonexistent/m.gguf", "t/x", &gpu_budget(), RESERVED, &Qwen35GgufServeConfig::default()).err().expect("must be refused");
        assert!(missing.contains("/nonexistent/m.gguf"), "{missing}");
        std::fs::remove_file(path).ok();
    }
}
