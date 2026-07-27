// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident-model adapters for brain's text-generation LLMs — GPT (dense
//! char-level baseline), GLM (MLA + noaux_tc MoE decoder), and Qwen3 (BPE
//! decoder) — behind the residency [`Executor`], mirroring the yolo/z-image
//! adapters in [`crate::resident`].
//!
//! Each model family is one [`ResidentModel`] with a single `"generate"` action.
//! Unlike yolo (which pins itself to the CPU via `Gpu::new_cpu` unless a
//! `--device` was chosen), these models load through `gpu_core::Gpu::new`, i.e.
//! the process-default backend — **wgpu (GPU) unless `BRAIN_DEVICE=cpu`**. So the
//! resident instance holds the model on a GPU (VRAM); dropping it frees the card.
//! `activate` pins the assigned card via `BRAIN_GPU_INDEX`, exactly like z-image.
//!
//! Config is env-only: `BRAIN_GPT_WEIGHTS`, `BRAIN_GLM_WEIGHTS`,
//! `BRAIN_QWEN_WEIGHTS` + `BRAIN_QWEN_TOKENIZER` (and an optional
//! `BRAIN_QWEN_CTX`, default 2048, sizing Qwen's built context length). Each
//! `from_env` returns `None` when its primary weights var is unset/empty.

use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::json;

use data::rng::Rng;
use data::tokenizer::{CharTokenizer, Tokenizer};

// ---------------------------------------------------------------- shared

/// The shared `"generate"` action spec. `chat` adds Qwen's chat-template toggle.
fn generate_spec(summary: &str, chat: bool) -> ActionSpec {
    let mut s = ActionSpec::new("generate", summary)
        .param(ParamSpec::new("prompt", ParamType::Str, "the prompt to continue (or chat message)"))
        .param(ParamSpec::new("max_new", ParamType::Int, "number of new tokens to generate").default(json!(128)))
        .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature (<= 0 = greedy)").default(json!(0.8)))
        .param(ParamSpec::new("top_k", ParamType::Int, "top-k filter (0 = disabled)").default(json!(40)))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed").default(json!(0)));
    if chat {
        s = s.param(ParamSpec::new("chat", ParamType::Bool, "apply the chat template to the prompt").default(json!(true)));
    }
    s.output(BlobSpec::new("text", Media::Text, "the generated text"))
}

/// Read the four shared sampling params (with the spec defaults).
fn sampling_params(inv: &Invocation) -> (usize, f32, usize, u64) {
    let max_new = inv.get_i64("max_new").unwrap_or(128).max(0) as usize;
    let temp = inv.get_f64("temp").unwrap_or(0.8) as f32;
    let top_k = inv.get_i64("top_k").unwrap_or(40).max(0) as usize;
    let seed = inv.get_i64("seed").unwrap_or(0).max(0) as u64;
    (max_new, temp, top_k, seed)
}

/// Wrap generated text as a text-output [`Outcome`] (`text` value + `text` blob).
fn text_outcome(text: String) -> Outcome {
    Outcome::new().set("text", json!(text)).blob("text", Blob::new(Media::Text, text.into_bytes()))
}

/// Estimate the Hot VRAM footprint of a checkpoint as ~1.3x its file size.
fn est_vram(path: &str) -> MemCost {
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0).saturating_mul(13) / 10;
    MemCost::new(bytes, 0)
}

/// Pin the physical GPU (`BRAIN_GPU_INDEX`) the model's `Gpu::new` will select.
fn pin_gpu(device: Device) {
    if let Device::Gpu(i) = device {
        std::env::set_var("BRAIN_GPU_INDEX", i.to_string());
    }
}

// ---------------------------------------------------------------- gpt

/// The dense char-level GPT baseline behind the scheduler (`BRAIN_GPT_WEIGHTS`).
/// The checkpoint must embed its char vocab (trained with vocab embedding).
pub struct GptResident {
    path: String,
}

impl GptResident {
    pub fn from_env() -> Option<GptResident> {
        std::env::var("BRAIN_GPT_WEIGHTS").ok().filter(|p| !p.is_empty()).map(|path| GptResident { path })
    }
}

impl ResidentModel for GptResident {
    fn manifest(&self) -> Manifest {
        Manifest::new("gpt", "text generation (dense char-level GPT)", vec![generate_spec("generate text continuing a prompt (char-level GPT)", false)])
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new("gpt", "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        est_vram(&self.path)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        pin_gpu(device);
        let itos = gpt::model::Gpt::load_itos(&self.path)
            .ok_or("gpt: checkpoint has no embedded char vocab (BRAIN_GPT_WEIGHTS)")?;
        let tok = CharTokenizer::from_itos(itos);
        let block = gpt::GptConfig::from_json(&checkpoint::load(&self.path).header["config"]).block_size;
        let model = gpt::model::Gpt::load(&self.path, 1, block);
        Ok(Box::new(GptInstance { model, tok }))
    }
}

struct GptInstance {
    model: gpt::model::Gpt,
    tok: CharTokenizer,
}

impl Instance for GptInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (max_new, temp, top_k, seed) = sampling_params(inv);
        let prompt = inv.get_str("prompt").unwrap_or_default();
        let prompt_text = if prompt.is_empty() { "\n".to_string() } else { prompt };
        let ids = self.tok.encode(&prompt_text);
        let mut rng = Rng::new(seed);
        progress(Progress { step: 0, total: max_new as u32, message: "generating".to_string() });
        let gen = gpt::sample::generate(&self.model, &ids, max_new, temp, top_k, &mut rng);
        let text = self.tok.decode(&gen);
        progress(Progress { step: max_new as u32, total: max_new as u32, message: "done".to_string() });
        Ok(text_outcome(text))
    }
}

// ---------------------------------------------------------------- glm

/// The GLM decoder (MLA + sigmoid noaux_tc MoE) behind the scheduler
/// (`BRAIN_GLM_WEIGHTS`). Char-level: the checkpoint must embed its vocab.
pub struct GlmResident {
    path: String,
}

impl GlmResident {
    pub fn from_env() -> Option<GlmResident> {
        std::env::var("BRAIN_GLM_WEIGHTS").ok().filter(|p| !p.is_empty()).map(|path| GlmResident { path })
    }
}

impl ResidentModel for GlmResident {
    fn manifest(&self) -> Manifest {
        Manifest::new("glm", "text generation (GLM MLA + MoE decoder)", vec![generate_spec("generate text continuing a prompt (GLM decoder)", false)])
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new("glm", "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        est_vram(&self.path)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        pin_gpu(device);
        let itos = glm::model::Glm::load_itos(&self.path)
            .ok_or("glm: checkpoint has no embedded char vocab (BRAIN_GLM_WEIGHTS)")?;
        let tok = CharTokenizer::from_itos(itos);
        let block = glm::config::GlmConfig::from_json(&checkpoint::load(&self.path).header["config"]).block_size;
        let model = glm::model::Glm::load_inference(&self.path, 1, block);
        Ok(Box::new(GlmInstance { model, tok }))
    }
}

struct GlmInstance {
    model: glm::model::Glm,
    tok: CharTokenizer,
}

impl Instance for GlmInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (max_new, temp, top_k, seed) = sampling_params(inv);
        let prompt = inv.get_str("prompt").unwrap_or_default();
        let prompt_text = if prompt.is_empty() { "\n".to_string() } else { prompt };
        let ids = self.tok.encode(&prompt_text);
        let mut rng = Rng::new(seed);
        progress(Progress { step: 0, total: max_new as u32, message: "generating".to_string() });
        let gen = glm::sample::generate(&self.model, &ids, max_new, temp, top_k, None, &mut rng);
        let text = self.tok.decode(&gen);
        progress(Progress { step: max_new as u32, total: max_new as u32, message: "done".to_string() });
        Ok(text_outcome(text))
    }
}

// ---------------------------------------------------------------- qwen

/// The Qwen3 BPE decoder behind the scheduler (`BRAIN_QWEN_WEIGHTS` +
/// `BRAIN_QWEN_TOKENIZER`). Runs the CPU/GPU forward `generate` path (never the
/// NPU branch). `BRAIN_QWEN_CTX` (default 2048) sizes the built context length.
pub struct QwenResident {
    path: String,
    tokenizer: String,
}

impl QwenResident {
    pub fn from_env() -> Option<QwenResident> {
        let path = std::env::var("BRAIN_QWEN_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        let tokenizer = std::env::var("BRAIN_QWEN_TOKENIZER").ok().unwrap_or_default();
        Some(QwenResident { path, tokenizer })
    }

    fn ctx() -> u32 {
        std::env::var("BRAIN_QWEN_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(2048u32).max(1)
    }
}

impl ResidentModel for QwenResident {
    fn manifest(&self) -> Manifest {
        Manifest::new("qwen", "text generation (Qwen3 BPE decoder)", vec![generate_spec("generate text (Qwen3; chat template optional)", true)])
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new("qwen", "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        est_vram(&self.path)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        pin_gpu(device);
        if self.tokenizer.is_empty() {
            return Err("qwen: set BRAIN_QWEN_TOKENIZER to the tokenizer.json path".to_string());
        }
        let tok = data::qwen_tokenizer::QwenBpe::from_file(&self.tokenizer)?;
        let eos = tok.encode("<|im_end|>").first().copied();
        let model = qwen::model::Qwen::load_inference(&self.path, 1, Self::ctx());
        Ok(Box::new(QwenInstance { model, tok, eos }))
    }
}

struct QwenInstance {
    model: qwen::model::Qwen,
    tok: data::qwen_tokenizer::QwenBpe,
    eos: Option<u32>,
}

impl Instance for QwenInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (max_new, temp, top_k, seed) = sampling_params(inv);
        let chat = inv.get_bool("chat").unwrap_or(true);
        let prompt = inv.get_str("prompt").unwrap_or_default();
        let text = if chat {
            self.tok.apply_chat_template(&[("user", &prompt)], true)
        } else {
            prompt
        };
        let ids = self.tok.encode(&text);
        if ids.is_empty() {
            return Err("qwen: empty prompt".to_string());
        }
        let mut rng = Rng::new(seed);
        progress(Progress { step: 0, total: max_new as u32, message: "generating".to_string() });
        let gen = qwen::sample::generate(&self.model, &ids, max_new, temp, top_k, self.eos, &mut rng);
        let out = self.tok.decode(&gen);
        progress(Progress { step: max_new as u32, total: max_new as u32, message: "done".to_string() });
        Ok(text_outcome(out))
    }
}
