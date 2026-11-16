// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements serving for recurrent-depth language models
// for its clients. If your team needs expertise in putting a looped decoder
// behind one continuous-batching engine, a tokenizer and the same HTTP/D-Bus
// contract as a conventional decoder, then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Resident adapter for Ouro (looped decoder).
//!
//! One [`ouro::serve::Engine`] behind `model::serve::Scheduler`, with the
//! checkpoint's own byte-level BPE tokenizer and its own chat template ([`chat_format`]), tool calls included. The
//! manifest declares the same `generate` contract every text resident serves,
//! so the HTTP, D-Bus and SDK fronts need no Ouro-specific code.
//!
//! Capacity: `ctx` caps one sequence (never past the model's positions); the KV pool is shared by every running
//! sequence and holds `pool` tokens in every pass's planes. A server that knows its card ([`OuroResident::
//! with_card_budget`]) plans the pool from the card's usable bytes, less the weights and the decode step's buffers,
//! and the batch from the pool: as many sequences decode in one step as the pool holds at a quarter of `ctx` each,
//! up to [`MAX_PLANNED_BATCH`]. Without one the pool holds `max_batch` sequences of `ctx`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use capability::{ActionResult, ActionSpec, Invocation, Manifest, ParamSpec, ParamType, Progress, Provider};
use serde_json::json;
use checkpoint::st::ModelCard;
use llmserve::{ScheduledInstance, TextFrontend};
use model::serve::Scheduler;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

use crate::resident_llm::est_vram;
use llmserve::generate_spec;
use residency::on_device;

/// Default per-sequence context (tokens), page size and batch.
pub const DEFAULT_CTX: u32 = 4096;
pub const DEFAULT_BLOCK: u32 = 16;
pub const DEFAULT_BATCH: u32 = 8;
/// The most sequences a card-planned decode step carries.
pub const MAX_PLANNED_BATCH: u32 = 256;
/// The share of a card's usable bytes a planned pool may take: the rest is the prefill's transient buffers.
const POOL_SHARE: f64 = 0.85;

/// The catalog's manifest entry for the Ouro family.
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL_ID,
        "text generation (Ouro looped decoder: one 48-layer stack run for four recurrent passes per token, per-pass KV; chat template optional; continuous batching)",
        vec![generate_spec("generate text (Ouro looped decoder; ChatML chat template optional)", true)],
    )
}

pub const MODEL_ID: &str = "brain/ouro";

/// Most rows or tokens per row a training run may ask for.
const MAX_ROWS: f64 = 65_536.0;
/// Largest gradient-norm clip a run may ask for.
const MAX_CLIP: f64 = 1_000_000.0;

/// The `train` action's defaults.
const DEFAULT_STEPS: u32 = 32;
const DEFAULT_SEQ: u32 = 128;
const DEFAULT_BATCH_ROWS: u32 = 2;
const DEFAULT_LR: f32 = 1e-5;
const DEFAULT_CLIP: f32 = 1.0;

fn train_spec() -> ActionSpec {
    ActionSpec::new("train", "a short causal-LM run on the checkpoint itself (the released exit-probability-weighted objective, AdamW at a constant rate, through the shared training loop over windows of the text), on text or raw token ids")
        .training()
        .streaming()
        .cancellable()
        .param(ParamSpec::new("text", ParamType::Str, "the training text, tokenized with the checkpoint's tokenizer (default: a short passage of prose)"))
        .param(ParamSpec::new("ids", ParamType::Str, "comma-separated token ids instead of text, for a vocabulary no tokenizer produces"))
        .param(ParamSpec::new("steps", ParamType::Int, "optimizer steps").default(json!(DEFAULT_STEPS)).min(1.0).max(capability::bounds::TRAIN_STEPS))
        .param(ParamSpec::new("seq", ParamType::Int, "tokens per row").default(json!(DEFAULT_SEQ)).min(2.0).max(MAX_ROWS))
        .param(ParamSpec::new("batch", ParamType::Int, "rows per step").default(json!(DEFAULT_BATCH_ROWS)).min(1.0).max(MAX_ROWS))
        .param(ParamSpec::new("lr", ParamType::Float, "AdamW learning rate").default(json!(DEFAULT_LR)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("clip", ParamType::Float, "global gradient-norm clip; 0 disables it").default(json!(DEFAULT_CLIP)).min(0.0).max(MAX_CLIP))
        .param(ParamSpec::new("seed", ParamType::Int, "seed of the window draws (random when omitted; the result carries the one used)").min(0.0).max(capability::bounds::EXACT_INTEGER))
        .param(ParamSpec::new("save", ParamType::Str, "where to write the trained checkpoint (.safetensors); omit to keep nothing").host_resolved())
        .param(ParamSpec::new(
            "activations",
            ParamType::Enum(model::saved::SavedPolicy::ACTIVATIONS_CHOICES.map(String::from).to_vec()),
            "what the backward keeps of the forward (default: the model's own)",
        ))
}

/// Default training text: enough distinct prose to make a few rows of
/// causal-LM targets without repeating one sentence.
const DEFAULT_TRAIN_TEXT: &str = "The city of Paris is the capital of France. \
Its history reaches back more than two thousand years, from a small settlement \
on the banks of the Seine to the seat of government, learning and art that it \
is today. The Eiffel Tower was completed in 1889 for the world's fair, and the \
Louvre holds one of the largest collections of art anywhere. The cathedral of \
Notre-Dame stands on the island in the middle of the river, and the wide \
boulevards that cross the city were laid out in the nineteenth century.";

/// What `brain ouro <action>` and a sample run: `generate` with the engine
/// sizing a local run sets, `train` and `import`.
pub fn direct_manifest() -> Manifest {
    let mut m = manifest();
    for a in &mut m.actions {
        a.params.extend(crate::resident_provider::kv_sizing_params(DEFAULT_CTX, DEFAULT_BLOCK));
    }
    m.actions.push(train_spec());
    m.actions.push(crate::resident_provider::import_spec("convert a Hugging Face Ouro checkpoint into brain's checkpoint"));
    m
}

struct ImportAction;

impl capability::Action for ImportAction {
    fn spec(&self) -> ActionSpec {
        direct_manifest().actions.into_iter().find(|a| a.name == "import").expect("declared")
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let source = inv.get_str("source").ok_or("ouro import: missing required param 'source'")?;
        let dest = inv.get_str("dest").ok_or("ouro import: missing required param 'dest'")?;
        ouro::import::import_as(&source, &dest, None, &mut inv.keep_going(progress, 0, "tensor"))?;
        Ok(capability::Outcome::new().set("dest", json!(dest)))
    }
}

struct TrainAction {
    weights: Option<String>,
    tokenizer: Option<String>,
}

impl capability::Action for TrainAction {
    fn spec(&self) -> ActionSpec {
        train_spec()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let weights = self.weights.as_deref().ok_or("ouro train: no checkpoint (brain pull ByteDance/Ouro-2.6B-Thinking, or name one with --weights)")?;
        let corpus: Vec<u32> = match inv.get_str("ids") {
            Some(raw) => raw.split(',').map(|t| t.trim().parse::<u32>().map_err(|_| format!("ouro train: ids takes comma-separated token ids (got {raw:?})"))).collect::<Result<_, _>>()?,
            None => {
                let tokenizer = self.tokenizer.clone().or_else(|| sibling_tokenizer(weights)).ok_or("ouro train: no tokenizer.json beside the checkpoint; name one with --tokenizer or give ids")?;
                let tok = data::qwen_tokenizer::QwenBpe::from_file(&tokenizer)?;
                data::tokenizer::Tokenizer::encode(&tok, &inv.get_str("text").unwrap_or_else(|| DEFAULT_TRAIN_TEXT.to_string()))
            }
        };
        let run = ouro::train::TrainRun {
            steps: inv.get_i64("steps").map_or(DEFAULT_STEPS, |v| v.max(1) as u32),
            seq: inv.get_i64("seq").map_or(DEFAULT_SEQ, |v| v.max(2) as u32),
            batch: inv.get_i64("batch").map_or(DEFAULT_BATCH_ROWS, |v| v.max(1) as u32),
            lr: inv.get_f64("lr").map_or(DEFAULT_LR, |v| v as f32),
            clip: inv.get_f64("clip").map_or(DEFAULT_CLIP, |v| v as f32),
            activations: inv.get_str("activations").map(|a| model::saved::SavedPolicy::parse_activations(&a)).transpose()?,
            seed: inv.seed()?.unwrap_or_else(data::rng::random_seed),
        };
        let save = inv.get_str("save");
        let t = ouro::train::train(Path::new(weights), &corpus, &run, save.as_deref().map(Path::new), &mut |step, loss| {
            progress(Progress::step(step, run.steps, format!("loss {loss:.4}")));
            !inv.cancel.is_cancelled()
        })?;
        if t.steps < run.steps {
            progress(Progress::step(t.steps, run.steps, format!("cancelled after step {}", t.steps)));
            return Err(capability::failure::CANCELLED.to_string());
        }
        Ok(capability::Outcome::new().set("first_loss", json!(t.first_loss)).set("final_loss", json!(t.final_loss)).set("steps", json!(t.steps)).set("seed", json!(run.seed)).set("saved", json!(save)))
    }
}

/// Ouro over `assembly`'s checkpoint and tokenizer, run in this process
/// through its own serving adapter (`crate::resident_provider`).
pub fn provider(assembly: &capability::Assembly) -> Result<Arc<dyn Provider>, String> {
    let weights = assembly.role_path("weights").ok();
    let tokenizer = assembly.role_path("tokenizer").ok();
    let id = if assembly.id.is_empty() { MODEL_ID.to_string() } else { assembly.id.clone() };
    let train = TrainAction { weights: weights.clone(), tokenizer: tokenizer.clone() };
    let build = move |inv: &Invocation| -> Result<Arc<dyn ResidentModel>, String> {
        let weights = weights.as_deref().ok_or("ouro: no checkpoint (brain pull ByteDance/Ouro-2.6B-Thinking, or name one with --weights)")?;
        let n = |name: &str, default: u32| inv.get_i64(name).map_or(default, |v| v.max(1) as u32);
        Ok(Arc::new(OuroResident::from_card(weights, &ModelCard::new(&id, "ouro"), tokenizer.as_deref())?.with_sizing(n("ctx", DEFAULT_CTX), n("block", DEFAULT_BLOCK), DEFAULT_BATCH)))
    };
    let sizing = |inv: &Invocation| format!("{:?}:{:?}", inv.get_i64("ctx"), inv.get_i64("block"));
    Ok(Arc::new(crate::resident_provider::ResidentProvider::new(direct_manifest(), Box::new(build), Box::new(sizing), vec![Arc::new(ImportAction), Arc::new(train)])))
}

pub struct OuroResident {
    id: String,
    path: String,
    tokenizer: String,
    /// Tokens one sequence may hold.
    ctx: u32,
    block_size: u32,
    max_batch: u32,
    /// Tokens of the shared pool, in every pass's planes.
    pool: u32,
}

impl OuroResident {
    /// A resident over the brain checkpoint or HF directory at `weights`.
    pub fn from_card(weights: &str, card: &ModelCard, tokenizer: Option<&str>) -> Result<OuroResident, String> {
        let tokenizer = tokenizer
            .filter(|t| !t.is_empty())
            .map(str::to_string)
            .or_else(|| sibling_tokenizer(weights))
            .ok_or_else(|| format!("ouro: no tokenizer.json next to {weights}"))?;
        Ok(OuroResident {
            id: card.id.clone(),
            path: weights.to_string(),
            tokenizer,
            ctx: DEFAULT_CTX,
            block_size: DEFAULT_BLOCK,
            max_batch: DEFAULT_BATCH,
            pool: DEFAULT_CTX * DEFAULT_BATCH,
        })
    }

    /// Override the per-sequence context / page / batch (serving policy); the pool holds `max_batch` sequences of
    /// `ctx`. A context past the model's positions is capped at them.
    pub fn with_sizing(mut self, ctx: u32, block_size: u32, max_batch: u32) -> OuroResident {
        self.block_size = block_size.max(1);
        let positions = self.cfg().map_or(ctx, |c| c.block_size);
        // Whole pages; round a policy cap up to one.
        self.ctx = ctx.min(positions).max(self.block_size).div_ceil(self.block_size) * self.block_size;
        self.max_batch = max_batch.max(1);
        self.pool = self.ctx.saturating_mul(self.max_batch);
        self
    }

    /// Plan the pool and the batch from `card_bytes`, the usable bytes of the card the model is placed on: the
    /// pool takes [`POOL_SHARE`] of what the weights and the decode step's buffers leave, and the batch is as many
    /// sequences as the pool holds at a quarter of the context each (at least the batch already set, at most
    /// [`MAX_PLANNED_BATCH`]). A card too small for the planned pool keeps the sizing it had.
    pub fn with_card_budget(mut self, card_bytes: u64) -> OuroResident {
        let Ok(cfg) = self.cfg() else { return self };
        let per_token = ouro::serve::Engine::kv_bytes_per_token(&cfg);
        let fixed = est_vram(&self.path).vram;
        let batch_for = |pool: u64| (pool / u64::from((self.ctx / 4).max(1))).clamp(u64::from(self.max_batch), u64::from(MAX_PLANNED_BATCH)) as u32;
        // The decode buffers grow with the batch the pool affords, so the batch is planned on the pool left after
        // the largest batch's buffers.
        let room = card_bytes.saturating_sub(fixed).saturating_sub(decode_bytes(&cfg, MAX_PLANNED_BATCH, self.ctx));
        let pool = ((room as f64 * POOL_SHARE) as u64 / per_token.max(1)).min(u64::from(u32::MAX));
        let pool = pool / u64::from(self.block_size) * u64::from(self.block_size);
        if pool >= u64::from(self.pool) {
            self.pool = pool as u32;
            self.max_batch = batch_for(pool);
        }
        self
    }

    /// The engine at this sizing under the continuous-batching scheduler, built on the device the calling thread is
    /// scoped to: what [`ResidentModel::activate`] serves, and what a measurement of the served path drives.
    pub fn scheduler(&self) -> Result<Scheduler<ouro::serve::Engine>, String> {
        let engine = ouro::serve::Engine::open(&self.path, self.pool, self.block_size, self.max_batch)?;
        Ok(Scheduler::new(engine, self.max_batch as usize))
    }

    /// The planned sizing: (per-sequence context, pool tokens, batch).
    pub fn sizing(&self) -> (u32, u32, u32) {
        (self.ctx, self.pool, self.max_batch)
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    fn cfg(&self) -> Result<ouro::OuroConfig, String> {
        config_of(&self.path)
    }
}

/// Device bytes of the engine's step buffers for `rows` sequences of up to `ctx` tokens: a decode step's per-row
/// activations of every pass, the attention's score and weight slabs and the logits, and the prompt buffers a
/// prefill of `ctx` tokens runs in.
fn decode_bytes(cfg: &ouro::OuroConfig, rows: u32, ctx: u32) -> u64 {
    let (r, t, c) = (u64::from(rows), u64::from(cfg.total_ut_steps), u64::from(ctx));
    let (d, ff, v) = (u64::from(cfg.d_model), u64::from(cfg.d_ff), u64::from(cfg.vocab));
    let layer = d * 8 + 3 * ff + 2 * u64::from(cfg.q_dim()) + 2 * u64::from(cfg.kv_dim());
    let decode = r * (layer + d * t + v) + 2 * r * u64::from(cfg.n_heads) * c;
    let prompt = c * (layer + d * (2 + t) + 3 * t + u64::from(cfg.n_heads));
    (decode + prompt) * 4
}

fn config_of(path: &str) -> Result<ouro::OuroConfig, String> {
    if Path::new(path).is_dir() {
        let json = std::fs::read_to_string(Path::new(path).join("config.json")).map_err(|e| format!("{path}/config.json: {e}"))?;
        ouro::import::config_from_hf(&json)
    } else {
        let reader = checkpoint::weightio::WeightReader::open(path).map_err(|e| format!("{path}: {e}"))?;
        ouro::OuroConfig::from_json(&reader.config())
    }
}

/// How the checkpoint whose tokenizer files are in `dir` renders a chat: its own chat template whenever it ships one
/// (whatever markup it writes, ChatML included), so a request is rendered exactly as the chat fine-tune rendered the
/// training records; with none, ChatML when the vocabulary is ChatML's, else raw prompts only. Tool calls are read back
/// in the form the template writes them (`<tool_call>{"name", "arguments"}</tool_call>`, or `<function=...>`).
pub fn chat_format(dir: Option<&Path>, tok: &data::qwen_tokenizer::QwenBpe) -> qwen3::chat::ChatFormat {
    match dir.map(qwen3::chat::ChatFormat::own_template) {
        Some(Ok(own)) => own,
        _ => qwen3::chat::ChatFormat::for_checkpoint(dir, tok),
    }
}

fn sibling_tokenizer(weights: &str) -> Option<String> {
    let p = PathBuf::from(weights);
    let dir = if p.is_dir() { p } else { p.parent()?.to_path_buf() };
    let t = dir.join("tokenizer.json");
    t.exists().then(|| t.to_string_lossy().to_string())
}

impl ResidentModel for OuroResident {
    /// The context a request may use is the shared pool's capacity, and never more positions than the model has.
    fn manifest(&self) -> Manifest {
        let positions = self.cfg().map_or(self.ctx, |c| c.block_size);
        Self::manifest_static(&self.id, self.ctx.min(positions))
    }

    fn instance_key(&self, _action: &str, _inv: &capability::Invocation) -> InstanceKey {
        InstanceKey::new(&self.id, "default")
    }

    fn batch_limit(&self) -> Option<usize> {
        Some(self.max_batch as usize)
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        let cost = est_vram(&self.path);
        let Ok(cfg) = self.cfg() else { return cost };
        // Every pass owns a K and a V plane per layer over the whole pool, beside one decode step's buffers.
        let kv = u64::from(self.pool) * ouro::serve::Engine::kv_bytes_per_token(&cfg);
        MemCost::new(cost.vram + kv + decode_bytes(&cfg, self.max_batch, self.ctx), cost.ram)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let tok = data::qwen_tokenizer::QwenBpe::from_file(&self.tokenizer)?;
        let dir = Path::new(&self.tokenizer).parent();
        let format = chat_format(dir, &tok);
        let stop = data::generation::stop_ids(dir, &tok, None, format.end_of_turn()).map_err(|e| format!("ouro: {e}"))?;
        if stop.is_empty() {
            return Err("ouro: the checkpoint declares no token that ends a generation (eos_token in tokenizer_config.json or generation_config.json)".to_string());
        }
        let front = Arc::new(TextFrontend::new(tok, stop, "ouro", Arc::new(move |tok: &data::qwen_tokenizer::QwenBpe, inv: &Invocation| qwen3::chat::parse_request_as(tok, &format, inv))));
        let batch = self.max_batch as usize;
        let inst = on_device(device, move || Ok::<Box<dyn Instance>, String>(Box::new(ScheduledInstance::new(front, self.scheduler()?, batch))))??;
        Ok(inst)
    }
}

impl OuroResident {
    fn manifest_static(id: &str, ctx: u32) -> Manifest {
        Manifest::new(
            id,
            "text generation (Ouro looped decoder: one shared stack run for four recurrent passes per token, per-pass KV; chat template optional; continuous batching)",
            vec![generate_spec("generate text (Ouro looped decoder; ChatML chat template optional)", true)],
        )
        .with_max_context_tokens(ctx as u64)
    }
}
