// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving long-context language
// models at a fixed memory budget for its clients. If your team needs
// expertise in bounded-state inference and continuous batching then you can
// procure our services by sending an email to info@swedishembedded.com.

//! EdgeRock behind the residency [`Executor`](residency) and the capability interface: the model directory
//! `brain pull rockmatrix-ai/EdgeRock` leaves in the store, served through the generic continuous-batching
//! scheduler by [`qwen35::serve_ring::RingEngine`].
//!
//! The directory is read in place: [`qwen35::fp8_source::open_dir`] serves each tensor lazily, the linears are
//! uploaded at the bf16 tier the checkpoint stores them in (no fp32 copy of the model on a card or on the host
//! tier), and the tied embedding and the anchor memories' projections are held in fp32 by the model
//! ([`qwen35::serve_ring::memory_param_bytes`] and the embedding are part of [`EdgeRockResident::estimate`]).
//!
//! Sizing is memory-aware and does not involve the context: a sequence costs the same fixed state at any length
//! ([`qwen35::model::slot_state_bytes`]), so `--max-batch` live slots and as many cached prompts are sized to the
//! free memory of the emptiest card at construction ([`qwen35::serve_ring::ring_slots`]). The context a request
//! may use is the model's position limit; `--ctx` only lowers it.
//!
//! [`EdgeRockProvider`] is the same resident behind `brain do edgerock generate`, built on first use.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use capability::{Action, ActionResult, ActionSpec, Invocation, Manifest, Progress, Provider};
use data::qwen_tokenizer::QwenBpe;
use gpu_core::select::Dtype;
use llmserve::{ScheduledInstance, TextFrontend};
use model::ops::TierPolicy;
use qwen3::chat::parse_request;
use qwen35::caps::with_template_flavor_default;
use qwen35::config::Qwen35Config;
use qwen35::serve_ring::{ring_slots, RingEngine, PREFILL_ROUND_TOKENS};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel, Tier};

use crate::parkable::{Build, HostWeights, ParkableInstance};
use crate::resident_llm::{generate_spec, on_device};
use crate::resident_qwen35::Qwen35Sizing;
use model::serve::Scheduler;

/// The catalog id of the in-process capability (`brain do edgerock generate`).
pub const MODEL: &str = "brain/edgerock";

/// The weight tier EdgeRock is served at: the precision it is released in.
fn weight_tier() -> TierPolicy {
    TierPolicy::uniform(Dtype::BF16)
}

/// How each sequence's ring of keys and values is stored: bf16, with the attention accumulating in fp32.
fn ring_tier() -> model::kv_tier::KvTier {
    model::kv_tier::KvTier::BF16
}

/// Device bytes kept free on the card beyond the weights, the slots and the prefill workspace: allocator slack
/// and the kernels' own scratch.
const HEADROOM_BYTES: u64 = 1 << 30;

/// The static manifest: safe to build with no weights on the machine.
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "text generation (EdgeRock windowed hybrid Gated-DeltaNet/GQA decoder with a bounded anchor memory; state of fixed size per sequence, continuous batching with prompt-prefix reuse)",
        vec![generate_spec("generate text (EdgeRock; chat template optional)", true)],
    )
}

/// The EdgeRock model directory served through the scheduler.
pub struct EdgeRockResident {
    id: String,
    dir: PathBuf,
    cfg: Qwen35Config,
    max_seq_len: u32,
    live_slots: u32,
    prefix_slots: u32,
}

impl EdgeRockResident {
    /// The resident for the directory `dir` (a pulled `config.json`, `tokenizer.json` and safetensors shards)
    /// under the model id `id`. Slots are sized from the free memory of the emptiest card now; with no card
    /// they are `sizing.max_batch` live slots and as many cached prompts.
    pub fn from_hf_dir(dir: &str, id: &str, sizing: Qwen35Sizing) -> Result<EdgeRockResident, String> {
        let path = Path::new(dir);
        let cfg = qwen35::fp8_source::hf_config(path)?;
        if cfg.swa_window.is_none() {
            return Err(format!("{id}: {dir} holds a plain Qwen3.5 decoder, not a windowed EdgeRock; it is not served from its directory"));
        }
        if !path.join("tokenizer.json").exists() {
            return Err(format!("{id}: {dir} has no tokenizer.json"));
        }
        let sizing = sizing.sanitized();
        let max_seq_len = if sizing.ctx_explicit { sizing.ctx.min(cfg.max_position_embeddings) } else { cfg.max_position_embeddings };
        let fixed = Self::fixed_bytes(&cfg);
        let budget = gpu_core::capacity::available_gpus().iter().map(|&(_, free)| free).max().map_or(u64::MAX, |free| free.saturating_sub(fixed + HEADROOM_BYTES));
        let (live_slots, prefix_slots) = ring_slots(&cfg, ring_tier(), budget, sizing.max_batch);
        Ok(EdgeRockResident { id: id.to_string(), dir: path.to_path_buf(), cfg, max_seq_len, live_slots, prefix_slots })
    }

    /// Device bytes that do not depend on the slot count: every weight (embedding and the memories' fp32 projections
    /// included) and the prefill workspace of one round.
    fn fixed_bytes(cfg: &Qwen35Config) -> u64 {
        cfg.infer_footprint_bytes(&weight_tier(), 0) + qwen35::int8_gguf_resident::prefill_workspace_bytes(cfg, PREFILL_ROUND_TOKENS)
    }

    /// Slots this resident was sized to: `(live sequences, cached prompts)`.
    pub fn slots(&self) -> (u32, u32) {
        (self.live_slots, self.prefix_slots)
    }

    fn shard_bytes(&self) -> u64 {
        std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum()
    }
}

impl ResidentModel for EdgeRockResident {
    fn manifest(&self) -> Manifest {
        Manifest::new(
            &self.id,
            "text generation (EdgeRock windowed hybrid Gated-DeltaNet/GQA decoder with a bounded anchor memory; state of fixed size per sequence, continuous batching with prompt-prefix reuse)",
            vec![generate_spec("generate text (EdgeRock; chat template optional)", true)],
        )
        .with_max_context_tokens(u64::from(self.max_seq_len))
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(&self.id, "default")
    }
    fn batch_limit(&self) -> Option<usize> {
        Some(self.live_slots as usize)
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        let slots = u64::from(self.live_slots + self.prefix_slots);
        MemCost::new(Self::fixed_bytes(&self.cfg) + slots * qwen35::model::slot_state_bytes(&self.cfg, ring_tier()), 0)
    }
    /// A demoted instance keeps the shards as a mapping of the files (evictable page cache): no f32 copy, no card.
    fn estimate_at(&self, key: &InstanceKey, tier: Tier) -> MemCost {
        match tier {
            Tier::Hot => self.estimate(key),
            Tier::Warm | Tier::Cold => MemCost::new(0, 0).with_mapped(self.shard_bytes()),
        }
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let tok = QwenBpe::from_file(&self.dir.join("tokenizer.json").to_string_lossy())?;
        let stop = data::generation::stop_ids(Some(&self.dir), &tok, None, Some("<|im_end|>")).map_err(|e| format!("edgerock: {e}"))?;
        let front = Arc::new(TextFrontend { tok, stop, label: "edgerock", parse: Arc::new(|tok, inv| parse_request(tok, &with_template_flavor_default(inv))) });
        let src = qwen35::fp8_source::open_dir(&self.dir, &self.cfg, qwen35::fp8_source::hf_fp8_block_size(&self.dir)?)?;
        let names = src.names().into_iter().map(str::to_string).collect();
        let host = HostWeights::from_source(Box::new(src), names);
        let (cfg, max_seq_len, live, prefix) = (self.cfg.clone(), self.max_seq_len, self.live_slots, self.prefix_slots);
        let build: Build = Box::new(move |host, device| {
            let sched = on_device(device, || Scheduler::new(RingEngine::from_source_tier(cfg.clone(), host.source(), max_seq_len, live, prefix, &weight_tier(), ring_tier()), live as usize))?;
            Ok(Box::new(ScheduledInstance::new(front.clone(), sched, live as usize)))
        });
        Ok(Box::new(ParkableInstance::activate(Arc::new(host), device, build)?))
    }
}

/// `brain do edgerock generate` and the SDK: the resident of the resolved model directory, built on the
/// emptiest card (the host tier without one) on the first request and kept for the next.
pub struct EdgeRockProvider {
    state: Arc<ProviderState>,
}

struct ProviderState {
    weights: Option<PathBuf>,
    live: Mutex<Option<Box<dyn Instance>>>,
}

impl EdgeRockProvider {
    /// A provider over the model directory the resolver chose (`None`: nothing is pulled).
    pub fn new(weights: Option<PathBuf>) -> EdgeRockProvider {
        EdgeRockProvider { state: Arc::new(ProviderState { weights, live: Mutex::new(None) }) }
    }
}

impl Provider for EdgeRockProvider {
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
        manifest().actions.into_iter().find(|a| a.name == "generate").expect("known action")
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let mut live = self.state.live.lock().map_err(|_| "edgerock: an earlier request panicked while holding the model".to_string())?;
        if live.is_none() {
            let dir = self.state.weights.as_ref().ok_or("edgerock: no model on this machine (brain pull rockmatrix-ai/EdgeRock)")?;
            let resident = EdgeRockResident::from_hf_dir(&dir.to_string_lossy(), MODEL, Qwen35Sizing::default())?;
            let device = gpu_core::capacity::available_gpus().into_iter().max_by_key(|&(_, free)| free).map_or(Device::Cpu, |(i, _)| Device::Gpu(i));
            *live = Some(resident.activate(&InstanceKey::new(MODEL, "default"), device)?);
        }
        live.as_mut().expect("built above").run("generate", inv, progress)
    }
}
