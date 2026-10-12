// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident-model adapter putting Qwen3.8-27B's paged serving engine
//! (`qwen35::serve::Engine`/`Scheduler`) behind the residency [`Executor`].
//! Mirrors `crate::resident_qwen35moe::Qwen35Resident` almost exactly - the
//! GDN/GQA decode-step orchestration `Engine` wraps is architecture-
//! identical between the two crates, so the scope (single-GPU, fp32 weights
//! only, fp32 KV, one truly-active sequence at a time on the GPU) and the
//! "deliberately deferred" list are identical too; see
//! `crates/qwen35/src/serve.rs`'s own module doc for the authoritative list.
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
//! [`multi_gpu_gguf`], the factory for
//! `qwen35::int8_gguf_resident::Qwen35GgufResident` - INT8 weights read
//! straight from a released Q8_0 GGUF and layer-sharded across as many cards
//! as they need, NOT a mode of the one above. The two coexist: this one still
//! serves an fp32 brain checkpoint that fits one card, that one serves the
//! real 27B release on a box with enough cards and no room on disk for a
//! ~108 GB fp32 conversion - both reached through the SAME
//! `qwen35::spec::Qwen35Spec` resolved `weights` role, told apart by its
//! extension (`.safetensors` vs `.gguf`). The GGUF factory is configured by
//! [`Qwen35GgufServeConfig`] - the checkpoint to serve (a `<vendor>/<repo>[-
//! <QUANT>]` reference or a path), its sizing, and its tiers - parsed data
//! from the CLI, never environment variables.

use capability::{ActionResult, Invocation, Manifest, Progress};
use data::qwen_tokenizer::QwenBpe;
use qwen3::chat::{parse_request, SeqState};
use qwen35::caps::with_template_flavor_default;
use qwen35::config::{LayerType, Qwen35Config};
use qwen35::serve::{Engine, Scheduler};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

use crate::resident_llm::{est_vram, generate_spec, on_device};

// Re-exported tier types the config struct speaks in, so a CLI builder (or
// a test) names ONE path for each.
pub use model::kv_tier::KvTier as Qwen35KvTier;
pub use model::ops::TierPolicy as Qwen35TierPolicy;

/// Everything `brain serve` parses for the Qwen3.8 family, threaded into
/// `serving::build_executor` as one value: [`fp32`] sizes the
/// brain-checkpoint resident, [`gguf`] configures the multi-card INT8
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
    /// `max_concurrent`, i.e. `num_blocks` - NOT how many are dispatched
    /// together on the GPU per step, which is always 1 for this engine).
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
    /// resident's checkpoint (`crate::resident_qwen35::multi_gpu_gguf`),
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
    /// `max_concurrent`, i.e. its `num_blocks`) - NOT how many run on the GPU
    /// concurrently per step (always 1; see this module's own doc).
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
            "text generation (Qwen3.8-27B dense hybrid Gated-DeltaNet/GQA decoder; single-GPU, fp32 weights + fp32 KV; multiple sequences may be admitted and scheduled concurrently on a shared engine, but the GPU dispatch itself runs one sequence's step at a time -- see crate::resident_qwen35's module doc)",
            vec![generate_spec("generate text (Qwen3.8-27B; chat template optional)", true)],
        )
        .with_max_context_tokens(self.ctx() as u64)
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(&self.id, "default")
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
        Ok(Box::new(Qwen35Instance { tok, stop, sched }))
    }
}

struct Qwen35Instance {
    tok: QwenBpe,
    /// The ids that end a generation (`data::generation::stop_ids`).
    stop: Vec<u32>,
    sched: Scheduler,
}

impl Instance for Qwen35Instance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.run_batch(action, std::slice::from_ref(inv), &mut |_i, p| progress(p)).pop().unwrap()
    }

    /// Every invocation in `invs` is submitted into the SAME persistent
    /// `Scheduler` (built once at `activate`, so the KV pool and prefix cache
    /// are shared and reused across calls) and driven to completion together
    /// - mirroring `crate::resident_qwen35moe::run_batch_scheduled` exactly.
    fn run_batch(&mut self, _action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        run_batch_scheduled(&mut self.sched, &self.tok, &self.stop, invs, progress)
    }

    /// The paged engine's prefix-cache effectiveness.
    fn metrics(&self) -> Vec<(String, serde_json::Value)> {
        let (hit, looked, cached) = self.sched.prefix_stats();
        let rate = if looked > 0 { hit as f64 / looked as f64 } else { 0.0 };
        vec![
            ("kv_prefix_hit_rate".to_string(), serde_json::json!(rate)),
            ("kv_prefix_hit_tokens".to_string(), serde_json::json!(hit)),
            ("kv_prefix_lookup_tokens".to_string(), serde_json::json!(looked)),
            ("kv_prefix_cached_blocks".to_string(), serde_json::json!(cached)),
        ]
    }
}

/// Drive every invocation in `invs` to completion on the SAME persistent
/// `Scheduler` - a near-identical copy of
/// `crate::resident_qwen35moe::run_batch_scheduled`, specialized to
/// `qwen35::serve::Scheduler` rather than generalized over the engine type.
fn run_batch_scheduled(sched: &mut Scheduler, tok: &QwenBpe, stop: &[u32], invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
    let mut results: Vec<Option<ActionResult>> = vec![None; invs.len()];
    let mut seq_for_bi: Vec<Option<SeqState>> = Vec::with_capacity(invs.len());
    let mut id_for_bi: Vec<Option<u64>> = Vec::with_capacity(invs.len());

    for (bi, inv) in invs.iter().enumerate() {
        // This IS Qwen3.8-27B, so it renders the 3.8 template by default -
        // `parse_request` alone defaults to the Qwen3-era flavor, which
        // would silently mis-render this model's prompt (the GGUF resident
        // and `caps.rs` both already apply this; this engine did not).
        match parse_request(tok, &with_template_flavor_default(inv)) {
            Ok(req) => {
                let sample = model::serve::SampleParams { temp: req.temp, top_k: req.top_k, top_p: req.top_p };
                let seed = req.seed;
                let max_new = req.max_new;
                let seq = SeqState::new(&req, inv.cancel.clone());
                let id = sched.submit_sampled(model::serve::Request { prompt: req.ids, max_new, stop: stop.to_vec() }, sample, seed);
                progress(bi, Progress::step(0, max_new as u32, "generating"));
                seq_for_bi.push(Some(seq));
                id_for_bi.push(Some(id));
            }
            Err(e) => {
                results[bi] = Some(Err(e));
                seq_for_bi.push(None);
                id_for_bi.push(None);
            }
        }
    }

    let mut remaining: std::collections::HashSet<usize> = (0..invs.len()).filter(|&bi| id_for_bi[bi].is_some()).collect();
    while !remaining.is_empty() {
        let mut just_finished = Vec::new();
        for &bi in &remaining {
            let id = id_for_bi[bi].unwrap();
            let Some(all_tokens) = sched.tokens_of(id) else { continue };
            let seq = seq_for_bi[bi].as_mut().unwrap();
            if seq.advance(tok, all_tokens, &mut |p| progress(bi, p)) {
                let toks = sched.cancel(id).unwrap_or_default();
                let seq = seq_for_bi[bi].take().unwrap();
                results[bi] = Some(Ok(seq.finish(tok, &toks, &mut |p| progress(bi, p))));
                just_finished.push(bi);
            }
        }
        for bi in just_finished {
            remaining.remove(&bi);
        }
        if remaining.is_empty() {
            break;
        }
        let report = sched.step_report();
        // A request the scheduler refuses at admission never appears in
        // `completed` and never will; without handling it here its `bi`
        // would stay in `remaining` forever.
        for (id, reason) in report.rejected {
            let bi = id_for_bi.iter().position(|x| *x == Some(id)).expect("rejected id must belong to this batch");
            seq_for_bi[bi] = None;
            results[bi] = Some(Err(format!("qwen35: {reason}")));
            remaining.remove(&bi);
        }
        for (id, toks) in report.completed {
            let bi = id_for_bi.iter().position(|x| *x == Some(id)).expect("completed id must belong to this batch");
            if let Some(seq) = seq_for_bi[bi].take() {
                results[bi] = Some(Ok(seq.finish(tok, &toks, &mut |p| progress(bi, p))));
            }
            remaining.remove(&bi);
        }
    }
    results.into_iter().map(|r| r.expect("every batch index resolved")).collect()
}

/// The INT8, multi-GPU, load-straight-from-GGUF Qwen3.8-27B
/// (`qwen35::int8_gguf_resident::Qwen35GgufResident`). Reachable ONLY via
/// `Executor::register_multi`: it occupies real bytes on SEVERAL cards at
/// once, and a plain `register` would budget only one of them (see
/// `crates/residency/src/multi.rs`' module doc).
///
/// The `.gguf` comes from `qwen35::spec::Qwen35Spec`'s resolved `weights`
/// role (see [`resolve_qwen35_gguf`]) - anywhere under `models_dir`, the
/// serving process's resolved models directory, never a hardcoded machine
/// path. Nothing resolving ⇒ not served, and this returns `None`.
///
/// `gpus` is `build_executor`'s own budgeted GPU list as `(index, TOTAL
/// bytes)` and `reserved` the per-card headroom it keeps free, so what is
/// handed on is each card's genuinely USABLE capacity - the same figure the
/// scheduler budgets against. Passing capacity (not just identity) is what
/// lets `model::shard::plan_fewest_devices` size the split to the hardware
/// instead of assuming matched cards.
/// The GGUF-resident's configuration, as parsed data - what `brain serve`'s
/// CLI builds from its `--qwen35-*` flags and hands down through
/// `build_executor`. No environment variables: an operator names one
/// checkpoint and its sizing on the command line, and the same flags are
/// what a config file or supervisor passes.
#[derive(Clone, Debug, Default)]
pub struct Qwen35GgufServeConfig {
    /// WHICH GGUF to serve, overriding whatever the model-store resolver
    /// would otherwise pick: a `<vendor>/<repo>[-<QUANT>]` reference
    /// (`unsloth/Qwen3.8-27B-GGUF-Q8_0`) or a direct path (absolute, or
    /// relative to the store root - the override matcher accepts both).
    /// `None` resolves over the whole store, which serves nothing when two
    /// real `qwen35` GGUFs make the choice genuinely ambiguous.
    pub gguf: Option<String>,
    /// Per-sequence `prompt + max_new` ceiling (also every stage's KV/GDN
    /// cache capacity, charged into the placement). `None` keeps the
    /// historical default (2048).
    pub ctx: Option<u32>,
    /// Per-leaf weight-tier policy (`model::ops::TierPolicy::parse`'s
    /// grammar). `None` keeps the historical default (uniform INT8).
    pub tier: Option<model::ops::TierPolicy>,
    /// GQA KV-cache storage tier. `None` keeps the historical default
    /// (f32).
    pub kv: Option<model::kv_tier::KvTier>,
    /// How many sequences one dispatch may decode together. `None` keeps
    /// the historical default (1).
    pub max_batch: Option<u32>,
}

impl Qwen35GgufServeConfig {
    /// The ctx knob with its historical default filled in.
    pub fn ctx_or_default(&self) -> u32 {
        self.ctx.unwrap_or(qwen35::int8_gguf_resident::DEFAULT_CTX).max(1)
    }

    /// The weight-tier knob with its historical default filled in.
    pub fn tier_or_default(&self) -> model::ops::TierPolicy {
        self.tier.clone().unwrap_or_else(|| model::ops::TierPolicy::uniform(gpu_core::select::Dtype::I8))
    }

    /// The KV-tier knob with its historical default filled in.
    pub fn kv_or_default(&self) -> model::kv_tier::KvTier {
        self.kv.unwrap_or(model::kv_tier::KvTier::F32)
    }

    /// The batch knob with its historical default filled in.
    pub fn max_batch_or_default(&self) -> u32 {
        self.max_batch.unwrap_or(1).max(1)
    }
}

/// Build the multi-GPU GGUF resident from a [`Qwen35GgufServeConfig`]:
/// resolve the configured (or store-implied) `.gguf`, then hand its path,
/// the budgeted devices and the sizing/tier knobs to
/// `qwen35::int8_gguf_resident::Qwen35GgufResident` as plain data.
///
/// `gpus` is `build_executor`'s own budgeted GPU list as `(index, TOTAL
/// bytes)` and `reserved` the per-card headroom it keeps free, so what is
/// handed on is each card's genuinely USABLE capacity - the same figure the
/// scheduler budgets against. Passing capacity (not just identity) is what
/// lets `model::shard::plan_fewest_devices` size the split to the hardware
/// instead of assuming matched cards.
///
/// Returns the resident alongside the id it must be served under - derived
/// from the RESOLVED artifact's position in the store
/// (`modelstore::resolve::model_id_from_path`, e.g.
/// `unsloth/Qwen3.8-27B-Q8_0`), not from any constant: two different quants
/// on disk are two different models with two different names, each
/// addressable under the name its files actually carry.
///
/// `None` - not served - when the store resolves no `.gguf` for this
/// architecture, the config names one that does not exist, or no GPU is
/// budgeted (the weights are int8 device buffers; there is no CPU path).
/// The id accompanying a `Some` is `model_id_from_path`'s answer for the
/// resolved path, falling back to [`GGUF_FALLBACK_ID`] only when the
/// artifact sits outside the store the resolver scanned (a direct path
/// pointing elsewhere), where no store-relative id exists to derive.
pub fn multi_gpu_gguf(models_dir: Option<&std::path::Path>, gpus: &[(u32, u64)], reserved: u64, cfg: &Qwen35GgufServeConfig) -> Option<(qwen35::int8_gguf_resident::Qwen35GgufResident, String)> {
    let (path, id) = resolve_qwen35_gguf(models_dir, cfg)?;
    if gpus.is_empty() {
        eprintln!("brain: {id} not served (no GPU budgeted -- its weights are int8 device buffers, there is no CPU path)");
        return None;
    }
    let devices: Vec<(Device, u64)> = gpus.iter().map(|&(i, total)| (Device::Gpu(i), total.saturating_sub(reserved))).collect();
    let r = qwen35::int8_gguf_resident::Qwen35GgufResident::new(path, devices, cfg.ctx_or_default(), cfg.tier_or_default())
        .with_kv_tier(cfg.kv_or_default())
        .with_max_batch(cfg.max_batch_or_default())
        .with_model_id(id.clone());
    Some((r, id))
}

/// The served id a resolved artifact OUTSIDE the scanned store falls back
/// to - a direct-path configuration has no store-relative
/// `<vendor>/<repo>` to derive, so the resident keeps the well-known
/// release id it has always advertised.
pub const GGUF_FALLBACK_ID: &str = qwen35::int8_gguf_resident::MODEL;

/// [`multi_gpu_gguf`]'s artifact resolution: WHERE the served GGUF lives and
/// WHAT id it is served under. Returns `(weights path, served id)`, or
/// `None` (never a guessed path) when the configuration names nothing the
/// store can answer for.
///
/// Precedence when [`Qwen35GgufServeConfig::gguf`] names an artifact:
///   1. An existing `.gguf` FILE - absolute, or relative to the store root -
///      is used verbatim. An operator who names a path gets exactly that
///      path, including one outside the store entirely.
///   2. A `<vendor>/<repo>[-<QUANT>]` reference is resolved through the
///      store's own lookup ([`brain_modelstore::Store::local`]) - the same
///      mechanism `brain pull`'s quant-suffix convention writes
///      (`<repo>/<QUANT>.gguf`).
///   3. A reference whose repo DIRECTORY exists but which resolved nothing
///      above (an unsuffixed repo name holding one classifiable GGUF, say)
///      narrows the model-store resolver's scan
///      ([`loader::resolver::resolve_reference`]) to that directory - the
///      same content-based classification the unnamed case uses, never a
///      filename guess. A reference that names a quant is accepted only
///      when the resolved weights ARE that quant's file: the scan ignores
///      the reference's quant, so without this check a missing quant would
///      silently resolve to whatever the repo holds.
///   4. Anything else is a loud refusal, not a fallback to some other
///      artifact.
///
/// Unnamed, the resolver classifies the whole store
/// ([`loader::resolver::resolve_structured`]); a resolved brain-format
/// `.safetensors` is `Qwen35Resident`'s own checkpoint
/// (`crate::resident_qwen35::Qwen35Resident::from_assembly`), so a
/// non-`.gguf` `weights` resolution is simply not this factory's model.
/// More than one real `qwen35`-architecture GGUF with nothing naming one is
/// a genuine ambiguity: refused, naming every candidate as the
/// `--qwen35-gguf` value that would select it (a daemon's startup log being
/// the only place its operator can read it).
///
/// The served id is derived from the resolved artifact's own position in
/// the store ([`brain_modelstore::resolve::model_id_from_path`]) - the
/// `<vendor>/<repo>` (or, for the flat back-compat layout,
/// `<vendor>/<file-stem>`) name its files actually carry, so two different
/// quants on disk are two different servable models, not two aliases of one
/// constant. A path outside the store has no store-relative id and keeps
/// [`GGUF_FALLBACK_ID`].
fn resolve_qwen35_gguf(models_dir: Option<&std::path::Path>, cfg: &Qwen35GgufServeConfig) -> Option<(String, String)> {
    let store_id = |path: &std::path::Path| -> String {
        models_dir
            .and_then(|root| brain_modelstore::resolve::model_id_from_path(path, root))
            .unwrap_or_else(|| GGUF_FALLBACK_ID.to_string())
    };

    if let Some(named) = cfg.gguf.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        // 1. An existing file, absolute or store-relative.
        let as_file = {
            let p = std::path::Path::new(named);
            (p.is_absolute() && p.is_file()).then(|| p.to_path_buf()).or_else(|| {
                let in_store = models_dir?.join(named);
                in_store.is_file().then_some(in_store)
            })
        };
        if let Some(path) = as_file {
            if !named.ends_with(".gguf") {
                eprintln!("brain: qwen35 GGUF not served: {named} is not a .gguf file");
                return None;
            }
            let id = store_id(&path);
            return Some((path.to_string_lossy().into_owned(), id));
        }
        // 2./3. A `<vendor>/<repo>[-<QUANT>]` reference: the store's own
        // lookup first, then a scan narrowed to that repo directory.
        match brain_modelref::ModelRef::parse(named) {
            Ok(r) => {
                let store = brain_modelstore::Store::new(models_dir?.to_path_buf());
                if let Some(local) = store.local(&r).filter(|m| m.weights.to_string_lossy().ends_with(".gguf")) {
                    let id = store_id(&local.weights);
                    return Some((local.weights.to_string_lossy().into_owned(), id));
                }
                if store.repo_dir(&r).is_dir() {
                    return match loader::resolver::resolve_reference(models_dir, "qwen35", &qwen35::spec::Qwen35Spec, &r, &Default::default()) {
                        Ok(resolution) => {
                            // The store scan narrows to the reference's repo
                            // directory but does not itself honor the
                            // reference's quant, so a named-but-absent quant
                            // would otherwise resolve to whatever GGUF the
                            // repo does hold - exactly the silent fallback
                            // step 4 forbids. When a quant is named, the
                            // resolved weights must BE that quant; anything
                            // else is a refusal naming the miss.
                            let wanted = r.quant().map(|q| format!("{}.gguf", q.as_str()));
                            let ok = match (&wanted, resolved_gguf_from(resolution, models_dir, &store_id)) {
                                (Some(want), Some((path, id))) if std::path::Path::new(&path).file_name().map(|f| f.to_string_lossy().ends_with(want.as_str())).unwrap_or(false) => Some((path, id)),
                                (Some(_), found) => {
                                    eprintln!(
                                        "brain: qwen35 GGUF not served: {named} names quant {} but the store's {} directory holds no {}.gguf{}",
                                        r.quant().map(|q| q.as_str()).unwrap_or(""),
                                        store.repo_dir(&r).display(),
                                        r.quant().map(|q| q.as_str()).unwrap_or(""),
                                        found.map(|(p, _)| format!(" (found {} instead)", std::path::Path::new(&p).file_name().unwrap_or_default().to_string_lossy())).unwrap_or_default()
                                    );
                                    None
                                }
                                (None, found) => found,
                            };
                            ok
                        }
                        Err(e) => {
                            eprintln!("brain: qwen35 GGUF not served: {e}");
                            None
                        }
                    };
                }
                let tried_file = models_dir.map(|root| root.join(named)).unwrap_or_else(|| std::path::PathBuf::from(named));
                eprintln!(
                    "brain: qwen35 GGUF not served: {named} names neither an existing .gguf file nor a model-store artifact (looked for {} and {})",
                    tried_file.display(),
                    store.repo_dir(&r).display()
                );
                return None;
            }
            Err(e) => {
                eprintln!("brain: qwen35 GGUF not served: {named} is neither an existing .gguf file nor a parsable <vendor>/<repo>[-<QUANT>] reference ({e})");
                return None;
            }
        }
    }

    // Unnamed: the whole store, classified.
    match loader::resolver::resolve_structured(models_dir, "qwen35", &qwen35::spec::Qwen35Spec, &Default::default()) {
        Ok(resolution) => resolved_gguf_from(resolution, models_dir, &store_id),
        Err(e) => {
            eprintln!("brain: qwen35 GGUF not served over the scheduler ({e})");
            None
        }
    }
}

/// The `.gguf` `weights` path and derived id out of a resolver outcome -
/// [`resolve_qwen35_gguf`]'s shared tail. An ambiguity is refused loudly,
/// naming every real candidate as the `--qwen35-gguf` value that would
/// select it.
fn resolved_gguf_from(resolution: brain_modelstore::resolve::Resolution, _models_dir: Option<&std::path::Path>, store_id: &dyn Fn(&std::path::Path) -> String) -> Option<(String, String)> {
    use brain_modelstore::resolve::Resolution as R;
    match resolution {
        R::Resolved(assembly) => {
            let weights = assembly.roles.get("weights")?;
            if !weights.to_string_lossy().ends_with(".gguf") {
                return None;
            }
            Some((weights.to_string_lossy().into_owned(), store_id(weights)))
        }
        R::Ambiguous(a) => {
            let mut s = String::from("brain: more than one qwen35 GGUF candidate and nothing named which - pass --qwen35-gguf with one of:\n");
            for c in &a.choices {
                for (_, value) in &c.selector {
                    s.push_str(&format!("  --qwen35-gguf {value}\n"));
                }
            }
            eprintln!("{s}");
            None
        }
        R::Missing(_) => None,
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
    /// ([`multi_gpu_gguf`]'s `Qwen35GgufResident`) - `Engine` never opens a
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

    // ========== the multi-GPU GGUF factory: `multi_gpu_gguf` ==========
    //
    // Fixtures use the SAME real-GGUF helper `activate_falls_back_to_the_
    // gguf_embedded_tokenizer` uses above, plus the store-layout directories
    // the resolver classifies through: a candidate is found by CONTENT
    // (`general.architecture`), never by filename.

    fn gguf_store(tag: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("brain-qwen35-gguf-factory-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        // A second, unrelated vendor directory: without one, every record
        // lives under the fixture's single vendor alone, so `resolve()`'s
        // `common_root` collapses onto it and the vendor-co-location checks
        // (`classify_tokenizer_role`, the GGUF's own tokenizer self-
        // satisfaction) compute a different "vendor" per role - the same
        // fixture shape `qwen35::spec`'s and `resident_zimage`'s tests
        // work around. The placeholder must be a SCANNABLE artifact (the
        // inventory records only recognized model extensions), and it must
        // classify as a foreign architecture so the qwen35 resolver never
        // considers it a candidate.
        std::fs::create_dir_all(root.join("other-vendor")).unwrap();
        checkpoint::st::save_safetensors(
            root.join("other-vendor").join("unrelated.safetensors").to_str().unwrap(),
            &[("w".to_string(), vec![1], vec![0.0f32])],
            &serde_json::json!({"architectures": ["UnrelatedModel"]}),
            None,
        )
        .unwrap();
        root
    }

    /// A real classifiable `qwen35`-architecture GGUF at the pulled-quant
    /// store position `<root>/<vendor>/<repo>/<QUANT>.gguf`.
    fn write_q35_gguf(root: &std::path::Path, vendor: &str, repo: &str, quant: &str) -> std::path::PathBuf {
        let dir = root.join(vendor).join(repo);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{quant}.gguf"));
        write_gguf_with_gpt2_tokenizer(&path);
        path
    }

    fn gpu_budget() -> Vec<(u32, u64)> {
        vec![(0, 24 << 30), (1, 24 << 30)]
    }

    fn named_cfg(named: &str) -> Qwen35GgufServeConfig {
        Qwen35GgufServeConfig { gguf: Some(named.to_string()), ..Default::default() }
    }

    /// THE case this change exists for: two real `qwen35` GGUFs in the store
    /// (the old env-var answer was "set the one GGUF variable or serve
    /// nothing") and a `--qwen35-gguf <vendor>/<repo>[-<QUANT>]` reference
    /// naming ONE - exactly the reference `brain pull`'s quant-suffix
    /// convention writes. The named artifact must resolve, and the resident
    /// must be served under the id the artifact's own store position
    /// carries - not any hardcoded constant.
    #[test]
    fn a_named_reference_resolves_one_gguf_of_several_under_its_own_id() {
        let root = gguf_store("named-ref");
        let q8 = write_q35_gguf(&root, "unsloth", "Qwen3.8-27B-GGUF", "Q8_0");
        let _q4 = write_q35_gguf(&root, "Abiray", "Qwen3.8-27B-Q4_K_M-GGUF", "Q4_K_M");

        let (r, id) = multi_gpu_gguf(Some(&root), &gpu_budget(), 2 << 30, &named_cfg("unsloth/Qwen3.8-27B-GGUF-Q8_0")).expect("the named reference must resolve");
        assert_eq!(id, "unsloth/Qwen3.8-27B-GGUF", "the id comes from the artifact's own store position");
        assert_eq!(r.manifest().model, id, "the manifest advertises the derived id, not a hardcoded const");
        assert_eq!(q8.to_string_lossy(), r.gguf_path(), "the resolved weights are the named artifact");

        std::fs::remove_dir_all(&root).ok();
    }

    /// The same disambiguation against the OTHER candidate - proving the
    /// reference, not directory order, decides which one is served.
    #[test]
    fn naming_the_other_reference_serves_the_other_artifact() {
        let root = gguf_store("named-ref-other");
        let _q8 = write_q35_gguf(&root, "unsloth", "Qwen3.8-27B-GGUF", "Q8_0");
        let q4 = write_q35_gguf(&root, "Abiray", "Qwen3.8-27B-Q4_K_M-GGUF", "Q4_K_M");

        let (_, id) = multi_gpu_gguf(Some(&root), &gpu_budget(), 2 << 30, &named_cfg("Abiray/Qwen3.8-27B-Q4_K_M-GGUF-Q4_K_M")).expect("the named reference must resolve");
        assert_eq!(id, "Abiray/Qwen3.8-27B-Q4_K_M-GGUF");
        assert_eq!(q4.to_string_lossy().contains("Q4_K_M"), true);

        std::fs::remove_dir_all(&root).ok();
    }

    /// An absolute path is used verbatim - including one outside the store
    /// - and keeps the well-known release id (no store-relative name
    /// exists to derive).
    #[test]
    fn a_direct_path_outside_the_store_is_used_verbatim_under_the_fallback_id() {
        let outside = gguf_store("direct-path");
        let path = outside.join("hand-placed.gguf");
        write_gguf_with_gpt2_tokenizer(&path);

        let (r, id) = multi_gpu_gguf(Some(&gguf_store("direct-store")), &gpu_budget(), 2 << 30, &named_cfg(path.to_str().unwrap())).expect("a direct path must resolve");
        assert_eq!(r.gguf_path(), path.to_str().unwrap(), "an operator who names a path gets exactly that path");
        assert_eq!(id, GGUF_FALLBACK_ID, "an out-of-store artifact keeps the release id");

        std::fs::remove_dir_all(&outside).ok();
    }

    /// A direct path pointing at an existing non-GGUF file is a loud
    /// refusal, never a silent serve-nothing: the operator asked for that
    /// file by name.
    #[test]
    fn a_direct_path_to_a_non_gguf_file_is_refused_by_name() {
        let dir = gguf_store("not-gguf");
        let path = dir.join("weights.safetensors");
        std::fs::write(&path, b"not a gguf").unwrap();
        // The file check precedes the extension check by design (an
        // operator naming an existing file gets THAT file or a refusal
        // naming it); a non-gguf name is refused either way.
        assert!(
            multi_gpu_gguf(Some(&dir), &gpu_budget(), 2 << 30, &named_cfg(path.to_str().unwrap())).is_none(),
            "an existing non-.gguf file must not be served as one"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Unnamed and exactly one real `qwen35` GGUF in the store: served on
    /// its own, under the id its store position carries - the zero-config
    /// path staying as automatic as before.
    #[test]
    fn one_unnamed_gguf_is_served_on_its_own() {
        let root = gguf_store("unnamed-single");
        let path = write_q35_gguf(&root, "unsloth", "Qwen3.8-27B-GGUF", "Q8_0");

        let (r, id) = multi_gpu_gguf(Some(&root), &gpu_budget(), 2 << 30, &Qwen35GgufServeConfig::default()).expect("a single candidate must resolve on its own");
        assert_eq!(id, "unsloth/Qwen3.8-27B-GGUF");
        assert_eq!(r.gguf_path(), path.to_str().unwrap());

        std::fs::remove_dir_all(&root).ok();
    }

    /// Unnamed with TWO real candidates and nothing naming one: genuinely
    /// ambiguous, so not served - and the refusal names every candidate as
    /// the `--qwen35-gguf` value that would select it, since a daemon's
    /// startup log is the only place its operator can read it. Naming one
    /// candidate resolves it (the first test above pins that half).
    #[test]
    fn unnamed_ambiguity_refuses_and_names_the_flag_that_selects() {
        let root = gguf_store("unnamed-ambiguous");
        let a = write_q35_gguf(&root, "unsloth", "Qwen3.8-27B-GGUF", "Q8_0");
        let b = write_q35_gguf(&root, "Abiray", "Qwen3.8-27B-Q4_K_M-GGUF", "Q4_K_M");

        assert!(
            multi_gpu_gguf(Some(&root), &gpu_budget(), 2 << 30, &Qwen35GgufServeConfig::default()).is_none(),
            "two candidates with nothing naming one must not be silently collapsed"
        );
        // The resolution behind the refusal is the resolver's own, so assert
        // its message names both candidate paths and the selecting flag.
        let records = brain_modelstore::inventory::scan(&root);
        let resolution = brain_modelstore::resolve::resolve("qwen35", &records, &[&qwen35::spec::Qwen35Spec], &Default::default());
        let brain_modelstore::resolve::Resolution::Ambiguous(amb) = resolution else { panic!("expected ambiguity") };
        let msg = {
            let mut s = String::new();
            for c in &amb.choices {
                for (_, value) in &c.selector {
                    s.push_str(&format!("  --qwen35-gguf {value}\n"));
                }
            }
            s
        };
        assert!(msg.contains("--qwen35-gguf"), "{msg}");
        assert!(msg.contains(a.to_str().unwrap()) && msg.contains(b.to_str().unwrap()), "{msg}");

        std::fs::remove_dir_all(&root).ok();
    }

    /// A config naming an artifact the store does not hold is a loud,
    /// named refusal - never a silent not-served and never a fallback to a
    /// different artifact.
    #[test]
    fn a_reference_the_store_does_not_hold_is_refused_by_name() {
        let root = gguf_store("missing-ref");
        let _ = write_q35_gguf(&root, "unsloth", "Qwen3.8-27B-GGUF", "Q8_0");
        assert!(
            multi_gpu_gguf(Some(&root), &gpu_budget(), 2 << 30, &named_cfg("unsloth/Qwen3.8-27B-GGUF-Q4_K_M")).is_none(),
            "a quant the store does not hold must not fall back to another quant"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    /// The sizing and tier knobs travel as data into the built resident:
    /// each knob lands where its accessor reports it.
    #[test]
    fn config_knobs_reach_the_resident_as_data() {
        let root = gguf_store("knobs");
        let _ = write_q35_gguf(&root, "unsloth", "Qwen3.8-27B-GGUF", "Q8_0");
        let cfg = Qwen35GgufServeConfig {
            gguf: Some("unsloth/Qwen3.8-27B-GGUF-Q8_0".to_string()),
            ctx: Some(8192),
            tier: Some(model::ops::TierPolicy::parse("i8").unwrap()),
            kv: Some(model::kv_tier::KvTier::Int8),
            max_batch: Some(4),
        };
        let (r, _) = multi_gpu_gguf(Some(&root), &gpu_budget(), 2 << 30, &cfg).expect("resolve");
        assert_eq!(r.cap(), 8192);
        assert_eq!(r.kv_tier(), model::kv_tier::KvTier::Int8);
        assert_eq!(r.max_batch(), 4);
        std::fs::remove_dir_all(&root).ok();
    }

    /// Historical defaults survive an all-`None` config: ctx 2048, uniform
    /// INT8 weights, f32 KV, batch 1 - byte for byte what the removed env
    /// readers defaulted to.
    #[test]
    fn config_defaults_match_the_historical_env_defaults() {
        let root = gguf_store("defaults");
        let _ = write_q35_gguf(&root, "unsloth", "Qwen3.8-27B-GGUF", "Q8_0");
        let cfg = Qwen35GgufServeConfig { gguf: Some("unsloth/Qwen3.8-27B-GGUF-Q8_0".to_string()), ..Default::default() };
        let (r, _) = multi_gpu_gguf(Some(&root), &gpu_budget(), 2 << 30, &cfg).expect("resolve");
        assert_eq!(r.cap(), 2048);
        assert_eq!(r.kv_tier(), model::kv_tier::KvTier::F32);
        assert_eq!(r.max_batch(), 1);
        std::fs::remove_dir_all(&root).ok();
    }

    /// No GPU budgeted: not served - the weights are int8 device buffers,
    /// there is no CPU path (same refusal the old factory gave, now naming
    /// the derived id).
    #[test]
    fn no_gpu_budgeted_is_not_served() {
        let root = gguf_store("no-gpu");
        let _ = write_q35_gguf(&root, "unsloth", "Qwen3.8-27B-GGUF", "Q8_0");
        assert!(multi_gpu_gguf(Some(&root), &[], 2 << 30, &named_cfg("unsloth/Qwen3.8-27B-GGUF-Q8_0")).is_none());
        std::fs::remove_dir_all(&root).ok();
    }

    /// No models directory at all: not served, quietly - the same policy
    /// every other resolved-at-startup model follows.
    #[test]
    fn no_models_dir_is_not_served() {
        assert!(multi_gpu_gguf(None, &gpu_budget(), 2 << 30, &Qwen35GgufServeConfig::default()).is_none());
    }
}
