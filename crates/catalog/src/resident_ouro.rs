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

use capability::{Manifest, Provider};
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

pub struct OuroProvider;

impl OuroProvider {
    pub fn new() -> OuroProvider {
        OuroProvider
    }
}

impl Provider for OuroProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }

    /// Ouro is served through the residency executor; a one-shot provider
    /// action would build a second engine from a path the invocation does not
    /// carry. The manifest is what `/v1/models` and `brain caps` read.
    fn action(&self, _name: &str) -> Option<Arc<dyn capability::Action>> {
        None
    }
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
        let front = Arc::new(TextFrontend { tok, stop, label: "ouro", parse: Arc::new(qwen3::chat::parse_request) });
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
