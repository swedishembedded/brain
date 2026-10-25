// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements serving for recurrent-depth language models
// for its clients. If your team needs expertise in putting a prelude/core/coda
// decoder behind one continuous-batching engine, a tokenizer and the same
// HTTP/D-Bus contract as a conventional decoder, then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Resident adapter for Huginn (Raven, a prelude/recurrent-core/coda decoder).
//!
//! One [`huginn::serve::Engine`] behind `model::serve::Scheduler`, with the
//! checkpoint's own byte-level BPE tokenizer and its own Jinja chat template
//! rendered by [`data::chat_template`] - the template ships with the
//! checkpoint, so no per-family Rust transcription exists here.
//!
//! The released template has no slot for tool schemas and the model was not
//! post-trained for tool calling, so a request's `tools` are dropped: a
//! schema block injected into a system turn was measured to derail the
//! model's answer, while the agent's own system text already describes its
//! tools. An answer that nevertheless carries a `<tool_call>` block is parsed
//! by the scanner as usual.
//!
//! Capacity: `ctx` is the shared KV capacity in tokens, and every stage owns a
//! plane of it. There are `n_prelude + steps * n_recurrent + n_coda` stages,
//! so the pool is that many conventional decoders' worth: the default keeps a
//! four-step run of the 3.5B checkpoint inside one 24 GB card.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use capability::{Invocation, Manifest, Provider};
use checkpoint::st::ModelCard;
use llmserve::{ParsedRequest, ScheduledInstance, TextFrontend};
use model::serve::Scheduler;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

use crate::resident_llm::{est_vram, generate_spec, on_device};

/// Default shared KV capacity (tokens), page size and batch.
pub const DEFAULT_CTX: u32 = 2048;
pub const DEFAULT_BLOCK: u32 = 16;
pub const DEFAULT_BATCH: u32 = 4;

/// Recurrent steps served unless the checkpoint's own `mean_recurrence` is
/// smaller. Each step costs one core block per layer in compute and one more
/// attention history per block in memory, so on a 24 GB card the depth and
/// the serving context trade against each other: 6 steps at 6144 tokens is
/// about 20 GiB beside fp32 weights. The released model is trained to accept
/// any depth (its card: fewer than 4 gives coarse answers, gains continue to
/// about 64) and its own `compress-sN` KV sharing halves the pool again, at a
/// divergence the served correctness gate would have to characterise first -
/// `huginn::serve::Engine::open_with_sharing` exposes it.
pub const DEFAULT_STEPS: u32 = 12;

pub const MODEL_ID: &str = "brain/huginn";

/// The catalog's manifest entry for the Huginn family.
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL_ID,
        "text generation (Huginn recurrent-depth decoder: prelude, a shared core re-applied with the input re-injected, coda; one KV history per stage; checkpoint chat template; continuous batching)",
        vec![generate_spec("generate text (Huginn recurrent-depth decoder; checkpoint chat template, tools described in text)", true)],
    )
}

pub struct HuginnProvider;

impl HuginnProvider {
    pub fn new() -> HuginnProvider {
        HuginnProvider
    }
}

impl Provider for HuginnProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }

    /// Huginn is served through the residency executor; a one-shot provider
    /// action would build a second engine from a path the invocation does not
    /// carry. The manifest is what `/v1/models` and `brain caps` read.
    fn action(&self, _name: &str) -> Option<Arc<dyn capability::Action>> {
        None
    }
}

pub struct HuginnResident {
    id: String,
    path: String,
    tokenizer: String,
    ctx: u32,
    block_size: u32,
    max_batch: u32,
    steps: Option<u32>,
}

impl HuginnResident {
    /// A resident over the brain checkpoint or HF directory at `weights`.
    pub fn from_card(weights: &str, card: &ModelCard, tokenizer: Option<&str>) -> Result<HuginnResident, String> {
        let tokenizer = tokenizer
            .filter(|t| !t.is_empty())
            .map(str::to_string)
            .or_else(|| sibling_tokenizer(weights))
            .ok_or_else(|| format!("huginn: no tokenizer.json next to {weights}"))?;
        Ok(HuginnResident {
            id: card.id.clone(),
            path: weights.to_string(),
            tokenizer,
            ctx: DEFAULT_CTX,
            block_size: DEFAULT_BLOCK,
            max_batch: DEFAULT_BATCH,
            steps: None,
        })
    }

    /// Override the shared KV capacity / page / batch (serving policy).
    pub fn with_sizing(mut self, ctx: u32, block_size: u32, max_batch: u32) -> HuginnResident {
        self.block_size = block_size.max(1);
        self.ctx = ctx.max(self.block_size).div_ceil(self.block_size) * self.block_size;
        self.max_batch = max_batch.max(1);
        self
    }

    /// Recurrent steps to serve (default: the checkpoint's `mean_recurrence`,
    /// capped at [`DEFAULT_STEPS`]).
    pub fn with_steps(mut self, steps: Option<u32>) -> HuginnResident {
        self.steps = steps.filter(|s| *s > 0);
        self
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    fn cfg(&self) -> Result<huginn::HuginnConfig, String> {
        config_of(&self.path)
    }

    fn steps_for(&self, cfg: &huginn::HuginnConfig) -> u32 {
        self.steps.unwrap_or(cfg.mean_recurrence.min(DEFAULT_STEPS))
    }
}

fn config_of(path: &str) -> Result<huginn::HuginnConfig, String> {
    if Path::new(path).is_dir() {
        let json = std::fs::read_to_string(Path::new(path).join("config.json")).map_err(|e| format!("{path}/config.json: {e}"))?;
        huginn::import::config_from_hf(&json)
    } else {
        let reader = checkpoint::weightio::WeightReader::open(path).map_err(|e| format!("{path}: {e}"))?;
        huginn::HuginnConfig::from_json(&reader.config())
    }
}

fn sibling_tokenizer(weights: &str) -> Option<String> {
    let p = PathBuf::from(weights);
    let dir = if p.is_dir() { p } else { p.parent()?.to_path_buf() };
    let t = dir.join("tokenizer.json");
    t.exists().then(|| t.to_string_lossy().to_string())
}

/// Render a request through the checkpoint's template. Tool schemas are
/// dropped (see the module doc) - the model has no tool protocol to render
/// them into.
fn parse(tok: &data::qwen_tokenizer::QwenBpe, format: &qwen3::chat::ChatFormat, inv: &Invocation) -> Result<ParsedRequest, String> {
    let mut req = if inv.get_str("tools").is_some_and(|t| !t.is_empty()) {
        let mut bare = inv.clone();
        let params = bare.params.as_object_mut().ok_or("huginn: params must be an object")?;
        params.remove("tools");
        params.remove("tool_choice");
        qwen3::chat::parse_request_as(tok, format, &bare)?
    } else {
        qwen3::chat::parse_request_as(tok, format, inv)?
    };
    // The checkpoint's template closes an assistant turn with `<|end_turn|>`.
    req.stops.push("<|end_turn|>".to_string());
    Ok(req)
}

impl ResidentModel for HuginnResident {
    fn manifest(&self) -> Manifest {
        Self::manifest_static(&self.id, self.ctx)
    }

    fn instance_key(&self, _action: &str, _inv: &capability::Invocation) -> InstanceKey {
        InstanceKey::new(&self.id, "default")
    }

    fn batch_limit(&self) -> Option<usize> {
        Some(self.max_batch as usize)
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        let Ok(cfg) = self.cfg() else { return est_vram(&self.path) };
        // The engine holds every parameter as fp32 whatever the file was
        // written as, so the weight side is exact from the parameter list -
        // the generic file-size estimate's margin would put the released
        // checkpoint over a 24 GB card's budget at a serving context it fits.
        let weights: u64 = cfg.param_list().iter().map(|(_, n)| *n as u64).sum::<u64>() * 4;
        // Every stage owns a plane of the whole context: one per prelude block,
        // one per core block per recurrent step, one per coda block. The
        // prefill slab is prompt-sized and is not charged here - a prompt
        // close to the context capacity needs that much headroom on top.
        let steps = self.steps_for(&cfg) as u64;
        let planes = cfg.n_prelude as u64 + steps * cfg.n_recurrent as u64 + cfg.n_coda as u64;
        let kv = self.ctx as u64 * cfg.d_model as u64 * 2 * planes * 4;
        MemCost::new(weights + kv, 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let tok = data::qwen_tokenizer::QwenBpe::from_file(&self.tokenizer)?;
        let dir = Path::new(&self.tokenizer).parent().map(Path::to_path_buf);
        let stop = data::generation::stop_ids(dir.as_deref(), &tok, None, Some("<|end_text|>"))?;
        if stop.is_empty() {
            return Err("huginn: the checkpoint declares no end-of-text token".to_string());
        }
        let format = qwen3::chat::ChatFormat::for_checkpoint(dir.as_deref(), &tok);
        let front = Arc::new(TextFrontend {
            tok,
            stop,
            label: "huginn",
            parse: Arc::new(move |tok: &data::qwen_tokenizer::QwenBpe, inv: &Invocation| parse(tok, &format, inv)),
        });
        let path = self.path.clone();
        let (ctx, block, batch) = (self.ctx, self.block_size, self.max_batch);
        // The depth is resolved here, not left to the engine's own default:
        // `mean_recurrence` is the training-time mean (32 for the released
        // checkpoint), and each iteration is another core stage of KV.
        let steps = self.steps_for(&self.cfg()?);
        let inst = on_device(device, move || {
            let engine = huginn::serve::Engine::open_with_steps(&path, ctx, block, Some(steps))?;
            let sched = Scheduler::new(engine, batch as usize);
            Ok::<Box<dyn Instance>, String>(Box::new(ScheduledInstance::new(front, sched, batch as usize)))
        })??;
        Ok(inst)
    }
}

impl HuginnResident {
    fn manifest_static(id: &str, ctx: u32) -> Manifest {
        Manifest::new(
            id,
            "text generation (Huginn recurrent-depth decoder: prelude, a shared core re-applied with the input re-injected, coda; one KV history per stage; checkpoint chat template; continuous batching)",
            vec![generate_spec("generate text (Huginn recurrent-depth decoder; checkpoint chat template, tools described in text)", true)],
        )
        .with_max_context_tokens(ctx as u64)
    }
}
