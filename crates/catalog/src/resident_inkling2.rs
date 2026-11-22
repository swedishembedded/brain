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
//! - **K/V** rows are f32 unless `--kv-cache bf16` asks for half the bytes per token ([`KV_TIERS`]).
//! - **Capacity**: one sequence holds at most `ctx` tokens; the pool holds `max_batch` such sequences in every full
//!   visit's planes, one state row (40 windowed rings and every visit's convolution states) per running sequence and
//!   as many snapshot rows for cached prompts. [`Inkling2Resident::estimate`] is the weights, that pool and the
//!   engine's step buffers.
//!
//! - **Images**: a checkpoint with a vision tower reads the request's image blobs (one per image part of its
//!   messages) through [`images::Inkling2Images`], the tower on the decoder's card; the rows stand at the template's
//!   placeholder tokens, expanded to one per 40x40-pixel region, and the engine splices them after its embedding norm.
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
use llmserve::image_frontend::ImageFrontend;
use llmserve::{generate_spec, ScheduledInstance, TextFrontend};
use model::kv_tier::KvTier;
use model::serve::visit_state::VisitPoolSizing;
use model::serve::Scheduler;
use residency::{on_device, Device, Instance, InstanceKey, MemCost, ResidentModel};

use crate::resident_qwen35::Qwen35Sizing;

/// The catalog id of the in-process capability (`brain do inkling2 generate`).
pub const MODEL: &str = "brain/inkling2";
/// Tokens one sequence may hold when the server was not told (`--ctx`): the model's positions reach 1M, but a pool
/// of `max_batch` such contexts would not fit a card.
pub const DEFAULT_CTX: u32 = 16384;
/// Rows of a KV block of the full visits.
pub const BLOCK: u32 = 16;
/// Prompt rows of one prefill pass.
pub const PREFILL_CHUNK: u32 = 512;
/// The K/V tiers the engine keeps rows in (`--kv-cache`): f32, the reference numerics and the default, or bf16, half
/// the bytes per token (its distance from f32 is measured in the engine's tests and recorded in the inkling2 ledger).
pub const KV_TIERS: [KvTier; 2] = [KvTier::F32, KvTier::BF16];

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
}

impl Inkling2Resident {
    /// The resident for the directory `dir` (`config.json`, `hf_quant_config.json`, the safetensors, `tokenizer.json`,
    /// the chat template) under the model id `id`: `sizing.ctx` caps a sequence when it was asked for
    /// ([`DEFAULT_CTX`] otherwise, never past the model's positions), `sizing.max_batch` sequences run at once, K/V rows
    /// are kept at `sizing.kv_cache` (one of [`KV_TIERS`]; f32 unless asked).
    pub fn from_hf_dir(dir: &str, id: &str, sizing: Qwen35Sizing) -> Result<Inkling2Resident, String> {
        let path = Path::new(dir);
        let cfg = Inkling2Config::from_dir(path)?;
        for needed in ["tokenizer.json", "generation_config.json"] {
            if !path.join(needed).is_file() {
                return Err(format!("{id}: {dir} has no {needed}"));
            }
        }
        let sizing = sizing.sanitized();
        let kv_tier = sizing.kv_cache.unwrap_or(KvTier::F32);
        if !KV_TIERS.contains(&kv_tier) {
            let offered: Vec<&str> = KV_TIERS.iter().map(|t| t.as_str()).collect();
            return Err(format!("{id}: KV cache {} is not supported; supported: {}", kv_tier.as_str(), offered.join(", ")));
        }
        let ctx = if sizing.ctx_explicit { sizing.ctx } else { DEFAULT_CTX };
        let ctx = u64::from(ctx).min(cfg.max_positions).max(u64::from(BLOCK)) as u32;
        let live = sizing.max_batch.max(1);
        let blocks = ctx.div_ceil(BLOCK).checked_mul(live).ok_or_else(|| format!("{id}: {live} sequences of {ctx} tokens overflow the block count"))?;
        let opts = EngineOptions { kv_tier, sizing: VisitPoolSizing { block_size: BLOCK, num_blocks: blocks, live_rows: live, snapshot_rows: live, undo_rows: false }, max_seq_len: ctx as usize, prefill_chunk: PREFILL_CHUNK };
        // The layout is held to the engine's own check now, so a configuration it cannot serve is refused at the scan.
        Engine::layout(&cfg, kv_tier)?;
        Ok(Inkling2Resident { id: id.to_string(), dir: path.to_path_buf(), cfg, opts })
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
        u64::from(s.num_blocks) * layout.block_bytes(s.block_size) + u64::from(s.live_rows + s.snapshot_rows) * layout.row_bytes() + layout.zero_row_bytes()
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
    fn manifest(&self) -> Manifest {
        Manifest::new(&self.id, SUMMARY, vec![action_spec(self.cfg.vision.is_some())]).with_max_context_tokens(self.opts.max_seq_len as u64)
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(&self.id, "default")
    }
    fn batch_limit(&self) -> Option<usize> {
        Some(self.opts.sizing.live_rows as usize)
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::new(self.weight_bytes() + self.pool_bytes() + self.step_bytes() + self.tower_bytes(), 0)
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
        let (dir, opts) = (self.dir.clone(), self.opts);
        let live = opts.sizing.live_rows as usize;
        on_device(device, move || -> Result<Box<dyn Instance>, String> {
            let engine = Engine::open(inkling2::serve::device_ops()?, &dir, opts)?;
            let (Some(v), Some((tiling, image_token)), Some(tok)) = (vision, processor, image_tok) else {
                return Ok(Box::new(ScheduledInstance::new(text, Scheduler::new(engine, live), live)));
            };
            // The tower shares the decoder's card: its own kernel set on the same device.
            let gpu = engine.model().ops().gpu().new_like(inkling2::vision::pipelines());
            let tower = VisionTower::new(&gpu, &v, &VisionWeights::open(&dir, &v)?)?;
            let sched = Scheduler::new(engine, live);
            let build = images::Inkling2Images::new(tok, format, gpu, tower, tiling, image_token, sched.sequence_capacity());
            let front = ImageFrontend::new(text, build, "inkling2-vision")?;
            Ok(Box::new(ScheduledInstance::new(front, sched, live)))
        })?
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
