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
//! checkpoint's own byte-level BPE tokenizer and ChatML chat rendering. The
//! manifest declares the same `generate` contract every text resident serves,
//! so the HTTP, D-Bus and SDK fronts need no Ouro-specific code.
//!
//! Capacity: `ctx` is the shared KV capacity in tokens, including every pass's
//! share (the engine charges one block per pass per page). The default keeps a
//! 2.6B model's four-pass KV pool inside one 24 GB card; a larger context is a
//! serving setting, not a model fact.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use capability::{ActionResult, ActionSpec, Invocation, Manifest, ParamSpec, ParamType, Progress, Provider};
use serde_json::json;
use checkpoint::st::ModelCard;
use llmserve::{ScheduledInstance, TextFrontend};
use model::serve::Scheduler;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

use crate::resident_llm::{est_vram, generate_spec, on_device};

/// Default shared KV capacity (tokens) and page size.
pub const DEFAULT_CTX: u32 = 4096;
pub const DEFAULT_BLOCK: u32 = 16;
pub const DEFAULT_BATCH: u32 = 8;

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

fn train_spec() -> ActionSpec {
    let d = ouro::train::TrainRun { steps: 32, seq: 128, batch: 2, lr: 1e-5, clip: 1.0, activations: None };
    ActionSpec::new("train", "a short causal-LM run on the checkpoint itself (the released exit-probability-weighted objective, AdamW), on text or raw token ids")
        .training()
        .streaming()
        .param(ParamSpec::new("text", ParamType::Str, "the training text, tokenized with the checkpoint's tokenizer (default: a short passage of prose)"))
        .param(ParamSpec::new("ids", ParamType::Str, "comma-separated token ids instead of text, for a vocabulary no tokenizer produces"))
        .param(ParamSpec::new("steps", ParamType::Int, "optimizer steps").default(json!(d.steps)).min(1.0).max(capability::bounds::TRAIN_STEPS))
        .param(ParamSpec::new("seq", ParamType::Int, "tokens per row").default(json!(d.seq)).min(2.0).max(MAX_ROWS))
        .param(ParamSpec::new("batch", ParamType::Int, "rows per step").default(json!(d.batch)).min(1.0).max(MAX_ROWS))
        .param(ParamSpec::new("lr", ParamType::Float, "AdamW learning rate").default(json!(d.lr)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("clip", ParamType::Float, "global gradient-norm clip; 0 disables it").default(json!(d.clip)).min(0.0).max(MAX_CLIP))
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
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let source = inv.get_str("source").ok_or("ouro import: missing required param 'source'")?;
        let dest = inv.get_str("dest").ok_or("ouro import: missing required param 'dest'")?;
        ouro::import::import(&source, &dest)?;
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
            steps: inv.get_i64("steps").unwrap_or(32).max(1) as u32,
            seq: inv.get_i64("seq").unwrap_or(128).max(2) as u32,
            batch: inv.get_i64("batch").unwrap_or(2).max(1) as u32,
            lr: inv.get_f64("lr").unwrap_or(1e-5) as f32,
            clip: inv.get_f64("clip").unwrap_or(1.0) as f32,
            activations: inv.get_str("activations").map(|a| model::saved::SavedPolicy::parse_activations(&a)).transpose()?,
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
        Ok(capability::Outcome::new().set("first_loss", json!(t.first_loss)).set("final_loss", json!(t.final_loss)).set("steps", json!(t.steps)).set("saved", json!(save)))
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
    ctx: u32,
    block_size: u32,
    max_batch: u32,
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
        })
    }

    /// Override the shared KV capacity / page / batch (serving policy).
    pub fn with_sizing(mut self, ctx: u32, block_size: u32, max_batch: u32) -> OuroResident {
        self.block_size = block_size.max(1);
        // The pool is a whole number of pages; round a policy cap up to one.
        self.ctx = ctx.max(self.block_size).div_ceil(self.block_size) * self.block_size;
        self.max_batch = max_batch.max(1);
        self
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    fn cfg(&self) -> Result<ouro::OuroConfig, String> {
        config_of(&self.path)
    }
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

fn sibling_tokenizer(weights: &str) -> Option<String> {
    let p = PathBuf::from(weights);
    let dir = if p.is_dir() { p } else { p.parent()?.to_path_buf() };
    let t = dir.join("tokenizer.json");
    t.exists().then(|| t.to_string_lossy().to_string())
}

impl ResidentModel for OuroResident {
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
        let cost = est_vram(&self.path);
        let Ok(cfg) = self.cfg() else { return cost };
        // Every pass owns a plane per layer: the pool is
        // `passes × layers × 2 × ctx × kv_dim` floats.
        let kv = self.ctx as u64
            * cfg.kv_dim() as u64
            * 2
            * cfg.n_layers as u64
            * cfg.total_ut_steps as u64
            * 4;
        MemCost::new(cost.vram + kv, cost.ram)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let tok = data::qwen_tokenizer::QwenBpe::from_file(&self.tokenizer)?;
        let dir = Path::new(&self.tokenizer).parent();
        let stop = data::generation::stop_ids(dir, &tok, None, Some("<|im_end|>")).map_err(|e| format!("ouro: {e}"))?;
        let front = Arc::new(TextFrontend::new(tok, stop, "ouro", Arc::new(qwen3::chat::parse_request)));
        let path = self.path.clone();
        let (ctx, block, batch) = (self.ctx, self.block_size, self.max_batch);
        let inst = on_device(device, move || {
            let engine = ouro::serve::Engine::open(&path, ctx, block)?;
            let sched = Scheduler::new(engine, batch as usize);
            Ok::<Box<dyn Instance>, String>(Box::new(ScheduledInstance::new(front, sched, batch as usize)))
        })??;
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
