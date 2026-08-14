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
//! `activate` places the build on the assigned card via a scoped device-registry
//! selection ([`on_device`]), exactly like z-image.
//!
//! Config is env-only: `BRAIN_GPT2_WEIGHTS`, `BRAIN_GLMDSA_WEIGHTS`,
//! `BRAIN_QWEN_WEIGHTS` + `BRAIN_QWEN_TOKENIZER` (and an optional
//! `BRAIN_QWEN_CTX` sizing Qwen's built context length — default in `QwenResident::ctx`,
//! currently 24576). Each `from_env` returns `None` when its primary weights
//! var is unset/empty.

use capability::{ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, ParamSpec, ParamType, Progress};
use checkpoint::st::ModelCard;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::json;

use data::rng::Rng;
use data::tokenizer::{CharTokenizer, Tokenizer};
use qwen3::chat::{parse_request, sampling_params, text_outcome, SeqState};

// ---------------------------------------------------------------- shared

/// The shared `"generate"` action spec. `chat` adds Qwen's chat contract: the
/// chat-template toggle plus `messages`/`system`/`top_p`/`stop` and per-token
/// streaming (one `Progress::token` delta each accepted token).
///
/// `pub(crate)`: reused as-is by [`crate::resident_qwen35moe::Qwen35Resident`]
/// (same chat contract, same `qwen3::chat` shared parse underneath) rather
/// than duplicated a third time.
pub(crate) fn generate_spec(summary: &str, chat: bool) -> ActionSpec {
    let mut s = ActionSpec::new("generate", summary)
        .param(ParamSpec::new("prompt", ParamType::Str, "the prompt to continue (or chat message)"))
        .param(ParamSpec::new("max_new", ParamType::Int, "number of new tokens to generate").default(json!(128)))
        .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature (<= 0 = greedy)").default(json!(0.8)))
        .param(ParamSpec::new("top_k", ParamType::Int, "top-k filter (40 = standard; 1 = greedy; 0 or negative = disabled)").default(json!(40)))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed").default(json!(0)));
    if chat {
        s = s
            .streaming()
            .param(ParamSpec::new("chat", ParamType::Bool, "apply the chat template to the prompt").default(json!(true)))
            .param(ParamSpec::new(
                "messages",
                ParamType::Str,
                "JSON array of {role,content,reasoning_content?,tool_calls?,tool_call_id?} chat turns (overrides prompt)",
            ))
            .param(ParamSpec::new("system", ParamType::Str, "optional system prompt prepended to the chat"))
            .param(ParamSpec::new("top_p", ParamType::Float, "nucleus sampling threshold (>= 1 = disabled)").default(json!(1.0)))
            .param(ParamSpec::new("stop", ParamType::Str, "JSON array of stop strings"))
            .param(ParamSpec::new("tools", ParamType::Str, "JSON array of tool definitions (OpenAI function-calling schema)"))
            .param(ParamSpec::new("tool_choice", ParamType::Str, "tool_choice directive, raw JSON text (\"auto\"|\"none\"|\"required\"|{...})"))
            .param(ParamSpec::new("enable_thinking", ParamType::Bool, "allow the model to emit a <think> reasoning block").default(json!(true)));
    }
    s.output(BlobSpec::new("text", Media::Text, "the generated text"))
}

/// Estimate the Hot VRAM footprint of a checkpoint as ~1.3x its file size.
/// `pub(crate)`: reused by [`crate::resident_qwen35moe::Qwen35Resident`] too.
pub(crate) fn est_vram(path: &str) -> MemCost {
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0).saturating_mul(13) / 10;
    MemCost::new(bytes, 0)
}

/// The KV pool alone must not exceed this before `QwenResident::activate`
/// refuses outright, rather than let `Engine::from_map_with_gpu` attempt a
/// device allocation that fails with wgpu's cryptic per-buffer byte count
/// (`resident_llm.rs`'s own `pool_sizing` doc comment has the historical
/// crash). Matches `run_cli.rs::build_serving_executor`'s iGPU policy budget
/// (`(8u64 << 30).min(ram / 2)`) -- duplicated here because that budget is
/// computed once at server startup and not threaded down to a single
/// resident's `activate`, and this guard must fire before any allocation,
/// not after querying a live budget that may not exist yet (e.g. `brain qwen
/// serve`'s direct-engine CLI path never builds a residency executor at
/// all). Only ever checked against the FP32 pool: at `BRAIN_QWEN_CTX`'s new
/// 24576 default, int8 is comfortably under this on its own
/// (`kv_pool_bytes_at_the_new_ctx_default_fits_the_igpu_budget`); fp32 is not.
const MAX_FP32_KV_POOL_BYTES: u64 = 8 << 30;

/// Run `f` placed on the residency-assigned device: a GPU assignment becomes a
/// scoped (thread-local) selection in the canonical device registry, so every
/// `Gpu::new` inside `f` binds that physical card — race-free across the
/// executor's concurrent activation lanes. Shared by the resident adapters.
///
/// `f` always builds a wgpu (GPU/CPU backend) engine, so a `Device::Npu`
/// assignment must never reach here silently -- a resident that wants NPU
/// placement (`MemCost::with_npu`) must branch on `Device::Npu` in its own
/// `activate` *before* calling this (see `resident_depth.rs`), the same way
/// every other NPU-capable resident already does.
pub(crate) fn on_device<R>(device: Device, f: impl FnOnce() -> R) -> Result<R, String> {
    match device {
        Device::Gpu(i) => gpu_core::devices::with_gpu(i, f),
        Device::Cpu => Ok(f()),
        Device::Npu(i) => Err(format!(
            "on_device: got Device::Npu({i}) but this resident has no NPU activation path -- \
             a resident declaring MemCost::with_npu must branch on Device::Npu in its own \
             activate() before calling on_device, not fall through to the wgpu build"
        )),
    }
}

// ---------------------------------------------------------------- gpt

/// The dense char-level GPT baseline behind the scheduler (`BRAIN_GPT2_WEIGHTS`).
/// The checkpoint must embed its char vocab (trained with vocab embedding).
pub struct GptResident {
    /// Catalog id (the model-card id): the manifest/instance-key key, so two
    /// checkpoints of the same family are two distinct selectable models.
    id: String,
    path: String,
}

impl GptResident {
    pub fn from_env() -> Option<GptResident> {
        let path = std::env::var("BRAIN_GPT2_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        // Back-compat: synthesize a card whose id is the canonical brain/
        // fallback (see crates/modelref/src/alias.rs's module docs) -- a
        // checkpoint loaded straight from an env var carries no upstream
        // vendor/repo provenance to build a fully-qualified ref from.
        Some(Self::from_card(&path, &ModelCard::new("brain/gpt", "gpt"), None))
    }

    /// Construct under the card's id. `_tokenizer` is unused — GPT is char-level
    /// (its vocab is embedded in the checkpoint).
    pub fn from_card(path: &str, card: &ModelCard, _tokenizer: Option<&str>) -> GptResident {
        GptResident { id: card.id.clone(), path: path.to_string() }
    }
}

impl ResidentModel for GptResident {
    fn manifest(&self) -> Manifest {
        Manifest::new(&self.id, "text generation (dense char-level GPT)", vec![generate_spec("generate text continuing a prompt (char-level GPT)", false)])
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(self.id.as_str(), "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        est_vram(&self.path)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // Stream weights from the mmap: peak host allocation is ~one tensor, not
        // a whole-model f32 copy on top of the device weights. One reader serves
        // the vocab, the config, and the tensor upload.
        let reader = checkpoint::weightio::WeightReader::open(&self.path).map_err(|e| format!("gpt: {e}"))?;
        let itos = gpt2::model::Gpt::itos_from_config(&reader.config())
            .ok_or("gpt: checkpoint has no embedded char vocab (BRAIN_GPT2_WEIGHTS)")?;
        let tok = CharTokenizer::from_itos(itos);
        let block = gpt2::GptConfig::from_json(&reader.config()).block_size;
        let model = on_device(device, || gpt2::model::Gpt::from_reader(&reader, 1, block))?;
        Ok(Box::new(GptInstance { model, tok }))
    }
}

struct GptInstance {
    model: gpt2::model::Gpt,
    tok: CharTokenizer,
}

impl Instance for GptInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (max_new, temp, top_k, seed) = sampling_params(inv);
        let prompt = inv.get_str("prompt").unwrap_or_default();
        let prompt_text = if prompt.is_empty() { "\n".to_string() } else { prompt };
        let ids = self.tok.encode(&prompt_text);
        let mut rng = Rng::new(seed);
        progress(Progress::step(0, max_new as u32, "generating"));
        let gen = gpt2::sample::generate(&self.model, &ids, max_new, temp, top_k, &mut rng);
        let text = self.tok.decode(&gen);
        progress(Progress::step(max_new as u32, max_new as u32, "done"));
        Ok(text_outcome(text))
    }
}

// ---------------------------------------------------------------- glm

/// The GLM decoder (MLA + sigmoid noaux_tc MoE) behind the scheduler
/// (`BRAIN_GLMDSA_WEIGHTS`). Char-level: the checkpoint must embed its vocab.
pub struct GlmResident {
    id: String,
    path: String,
}

impl GlmResident {
    pub fn from_env() -> Option<GlmResident> {
        let path = std::env::var("BRAIN_GLMDSA_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        // See GptResident::from_env's comment: env-loaded, no upstream provenance.
        Some(Self::from_card(&path, &ModelCard::new("brain/glm", "glm"), None))
    }

    /// Construct under the card's id. `_tokenizer` is unused — GLM is char-level.
    pub fn from_card(path: &str, card: &ModelCard, _tokenizer: Option<&str>) -> GlmResident {
        GlmResident { id: card.id.clone(), path: path.to_string() }
    }
}

impl ResidentModel for GlmResident {
    fn manifest(&self) -> Manifest {
        Manifest::new(&self.id, "text generation (GLM MLA + MoE decoder)", vec![generate_spec("generate text continuing a prompt (GLM decoder)", false)])
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(self.id.as_str(), "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        est_vram(&self.path)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // Stream weights from the mmap (see GptResident::activate).
        let reader = checkpoint::weightio::WeightReader::open(&self.path).map_err(|e| format!("glm: {e}"))?;
        let itos = glmdsa::model::Glm::itos_from_config(&reader.config())
            .ok_or("glm: checkpoint has no embedded char vocab (BRAIN_GLMDSA_WEIGHTS)")?;
        let tok = CharTokenizer::from_itos(itos);
        let block = glmdsa::config::GlmConfig::from_json(&reader.config()).block_size;
        let model = on_device(device, || glmdsa::model::Glm::from_reader_inference(&reader, 1, block))?;
        Ok(Box::new(GlmInstance { model, tok }))
    }
}

struct GlmInstance {
    model: glmdsa::model::Glm,
    tok: CharTokenizer,
}

impl Instance for GlmInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (max_new, temp, top_k, seed) = sampling_params(inv);
        let prompt = inv.get_str("prompt").unwrap_or_default();
        let prompt_text = if prompt.is_empty() { "\n".to_string() } else { prompt };
        let ids = self.tok.encode(&prompt_text);
        let mut rng = Rng::new(seed);
        progress(Progress::step(0, max_new as u32, "generating"));
        let gen = glmdsa::sample::generate(&self.model, &ids, max_new, temp, top_k, None, &mut rng);
        let text = self.tok.decode(&gen);
        progress(Progress::step(max_new as u32, max_new as u32, "done"));
        Ok(text_outcome(text))
    }
}

// ---------------------------------------------------------------- qwen

/// The Qwen3 BPE decoder behind the scheduler (`BRAIN_QWEN_WEIGHTS` +
/// `BRAIN_QWEN_TOKENIZER`). Runs the CPU/GPU forward `generate` path (never the
/// NPU branch). `BRAIN_QWEN_CTX` sizes the built context length (default in
/// `QwenResident::ctx`, currently 24576 — do not restate the number here,
/// it drifted once already).
pub struct QwenResident {
    id: String,
    path: String,
    tokenizer: String,
    /// A named LoRA adapter's own weight file (`qwen3::lora::save_adapter`'s
    /// output), when this resident is the ADAPTER's catalog entry rather than
    /// the base's -- folded into the base tensors at `activate` (see that
    /// method's doc). `None` for a plain base/quant resident.
    adapter: Option<String>,
}

impl QwenResident {
    pub fn from_env() -> Option<QwenResident> {
        let path = std::env::var("BRAIN_QWEN_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        let tokenizer = std::env::var("BRAIN_QWEN_TOKENIZER").ok().unwrap_or_default();
        // See GptResident::from_env's comment: env-loaded, no upstream provenance.
        Some(Self::from_card(&path, &ModelCard::new("brain/qwen", "qwen"), Some(&tokenizer), None))
    }

    /// Construct under the card's id. `tokenizer` is the sibling `tokenizer.json`
    /// (empty/None defers the "set a tokenizer" error to `activate`). `adapter`
    /// is the adapter's own weight file when `card.id` names one
    /// (`brain_modelstore::LocalModel::adapter`) -- `None` for a plain base.
    pub fn from_card(path: &str, card: &ModelCard, tokenizer: Option<&str>, adapter: Option<&str>) -> QwenResident {
        QwenResident {
            id: card.id.clone(),
            path: path.to_string(),
            tokenizer: tokenizer.unwrap_or_default().to_string(),
            adapter: adapter.filter(|a| !a.is_empty()).map(str::to_string),
        }
    }

    /// Default 24576 (12x the old 2048), sized to what int8 KV actually
    /// buys: at real Qwen3-0.6B (`head_dim=128`), the pool + scores/probs
    /// scratch at this `ctx` is ~7.0 GiB measured
    /// (`kv_pool_bytes_at_the_new_ctx_default_fits_the_igpu_budget`) against
    /// the iGPU policy budget's 8 GiB (`run_cli.rs::build_serving_executor`),
    /// ~1 GiB of margin. This default is ONLY safe because int8 KV is the
    /// serving default (this workstream) -- the fp32 pool at the SAME `ctx`
    /// is ~10.6 GiB, which is why `activate()` refuses rather than attempts
    /// it (see `MAX_FP32_KV_POOL_BYTES`).
    fn ctx() -> u32 {
        std::env::var("BRAIN_QWEN_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(24576u32).max(1)
    }

    /// Concurrent decode slots for the batched (paged-KV) serving path.
    /// `BRAIN_QWEN_MAX_BATCH` overrides; the default favors real concurrency
    /// over a huge per-slot context, matching `perf_cli.rs::pool_for`'s
    /// "size so admission, not allocation failure, limits concurrency".
    fn max_batch() -> u32 {
        std::env::var("BRAIN_QWEN_MAX_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(16u32).max(1)
    }

    /// Whether the paged engine's KV pool is packed int8 (online per-token
    /// absmax, ~3.9x smaller pool at Qwen3's head_dim) rather than fp32.
    /// Default ON: measured on the real Qwen3-0.6B checkpoint (`brain qwen
    /// eval --kv fp32,int8`) at +0.0154
    /// loss vs fp32 (token-acc actually slightly HIGHER) -- close enough to
    /// free that the memory win is a clear default. `BRAIN_QWEN_KV_INT8=0`
    /// (also `false`/`off`, case-insensitive, matching `BRAIN_AUTO_FETCH`'s
    /// convention) opts back out to fp32 KV.
    ///
    /// Deliberately NOT the calibrated variant (`model::kvcalib::KvCalib`,
    /// `--kv-calib`): the same measurement pass found a p99.9-calibrated
    /// clip built from a small (10-prompt) calibration set measurably WORSE
    /// than plain online-absmax (+1.34 loss, -14pp token-acc) -- real
    /// signal on held-out data gets truncated by an under-calibrated
    /// ceiling. That is evidence against defaulting to calibration with a
    /// small calibration set, not against int8 KV itself, and not against
    /// calibration once a properly-sized calibration corpus exists.
    fn kv_int8() -> bool {
        Self::kv_int8_from(std::env::var("BRAIN_QWEN_KV_INT8").ok().as_deref())
    }

    /// Pure parsing logic for [`Self::kv_int8`], dependency-injected so a
    /// test can check every spelling deterministically without mutating
    /// process-global environment state (which would race against any other
    /// test reading the same var concurrently in the same test binary).
    fn kv_int8_from(v: Option<&str>) -> bool {
        !v.is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "off"))
    }

    /// Whether to opt INTO a `kv_calib.json` beside the checkpoint, when the
    /// engine is int8 and one exists there with a matching shape. Default
    /// OFF (unlike `kv_int8`): P12's own measurement found a small (10-
    /// prompt) calibration set makes serving quality measurably WORSE, so
    /// this is opt-in only, matching `brain qwen serve --kv-calib`'s CLI
    /// equivalent -- see that flag's doc comment.
    fn kv_calib_opt_in() -> bool {
        Self::kv_calib_opt_in_from(std::env::var("BRAIN_QWEN_KV_CALIB").ok().as_deref())
    }

    /// Pure parsing logic for [`Self::kv_calib_opt_in`] -- same
    /// dependency-injection reasoning as `kv_int8_from`.
    fn kv_calib_opt_in_from(v: Option<&str>) -> bool {
        v.is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on"))
    }

    /// KV-pool geometry for the batched serving engine — the ONE place
    /// `estimate()` and `activate()` derive `block_size`/`max_batch`/
    /// `max_blocks_per_seq`/`num_blocks`/`max_prefill` from, so the residency
    /// budget's PREDICTION of what a pool will cost cannot silently drift
    /// from what activation actually allocates.
    ///
    /// Aggregate pool sized to ~`ctx` tokens total (+ `max_batch` blocks of
    /// headroom for prefix-cache reuse across concurrent requests) - NOT
    /// `ctx * max_batch`. The old formula reserved one full worst-case
    /// `ctx`-sized allocation PER concurrent batch slot simultaneously (e.g.
    /// 7 GiB of fp32 KV at the default ctx=2048, max_batch=16 -
    /// `Manifest::max_context_tokens`'s doc comment has the arithmetic),
    /// which is exactly what paged attention exists to avoid: concurrent
    /// sequences share one pool, they don't each get a private worst-case
    /// reservation. A shared pool this size still admits one `ctx`-length
    /// request (or several smaller concurrent ones); `max_batch` simultaneous
    /// near-`ctx`-length requests now correctly queues/rejects via the
    /// scheduler's existing admission control (`RejectReason::
    /// ExceedsCapacity`) instead of being pre-reserved for at ~17x the memory
    /// cost for the common case where that never happens.
    ///
    /// `max_prefill`: lower than the old fixed 2048 -- `Engine::
    /// from_map_with_gpu`'s scores/probs scratch buffers are sized
    /// `bcap = max(max_batch, max_prefill) * n_heads * (max_blocks_per_seq *
    /// block_size)` (serve.rs) - `max_prefill` sits in that product, so
    /// raising `ctx` (and therefore `max_blocks_per_seq`) enough to serve a
    /// real agent prompt pushes `bcap * 4 bytes` past the GPU's single-
    /// binding ceiling (2047 MiB) at the OLD 2048 value. Empirically
    /// confirmed on real hardware: ctx=16384 with max_prefill=2048 crashed
    /// with wgpu's "Buffer size 2147483648 is greater than the maximum
    /// buffer size (2147483647)" at Qwen3's 16 attention heads - exactly one
    /// byte over. 512 keeps that product comfortably under the ceiling at
    /// the context sizes this fix is meant to unlock, confirmed against the
    /// same scenario post-fix.
    fn pool_sizing(ctx: u32) -> (u32, u32, u32, u32, u32) {
        let block_size = 16u32;
        let max_batch = QwenResident::max_batch();
        let max_blocks_per_seq = ctx.div_ceil(block_size);
        let num_blocks = max_blocks_per_seq * 2 + max_batch;
        let max_prefill = ctx.min(512);
        (block_size, max_batch, max_blocks_per_seq, num_blocks, max_prefill)
    }

    /// `Err` naming the checkpoint, the requested `ctx`, the computed byte
    /// count and the safety ceiling when a FP32 KV pool at this sizing would
    /// exceed [`MAX_FP32_KV_POOL_BYTES`] -- called from `activate()` before
    /// any device allocation is attempted, so the failure is specific and
    /// actionable instead of wgpu's own bare-byte-count error (the exact
    /// crash `pool_sizing`'s doc comment records: "Buffer size 2147483648 is
    /// greater than the maximum buffer size (2147483647)", no context on
    /// which buffer or why). A free function of its inputs (no I/O, no env
    /// read) so a test can drive it directly at whatever `num_blocks` trips
    /// the ceiling, without needing a real multi-GiB allocation to prove it.
    fn check_fp32_kv_pool_fits(cfg: &qwen3::config::QwenConfig, block_size: u32, num_blocks: u32, ctx: u32, path: &str) -> Result<(), String> {
        let pool_bytes = qwen3::serve::kv_pool_bytes(cfg, block_size, num_blocks, false);
        if pool_bytes <= MAX_FP32_KV_POOL_BYTES {
            return Ok(());
        }
        let int8_bytes = qwen3::serve::kv_pool_bytes(cfg, block_size, num_blocks, true);
        Err(format!(
            "qwen: {path}: fp32 KV pool at ctx={ctx} would be {:.2} GiB, over the {:.0} GiB safety ceiling \
             (MAX_FP32_KV_POOL_BYTES) -- lower BRAIN_QWEN_CTX, or drop --kv-fp32/BRAIN_QWEN_KV_INT8=0 \
             so int8 KV (~{:.2} GiB at this ctx) is used instead",
            pool_bytes as f64 / (1u64 << 30) as f64,
            MAX_FP32_KV_POOL_BYTES as f64 / (1u64 << 30) as f64,
            int8_bytes as f64 / (1u64 << 30) as f64,
        ))
    }
}

impl ResidentModel for QwenResident {
    fn manifest(&self) -> Manifest {
        // `Self::ctx()` is exactly the value `activate()` below builds the
        // engine's KV-cache sizing from (`max_blocks_per_seq = ctx.div_ceil(16)`,
        // `Engine::max_seq_len() = max_blocks_per_seq * 16 >= ctx`), so
        // advertising `ctx` itself is a safe, never-overstated floor on real
        // serving capacity - see `Manifest::max_context_tokens`'s doc comment
        // on why this must be the actual engine capacity, not the
        // checkpoint's architectural `max_position_embeddings`.
        Manifest::new(&self.id, "text generation (Qwen3 BPE decoder)", vec![generate_spec("generate text (Qwen3; chat template optional)", true)])
            .with_max_context_tokens(Self::ctx() as u64)
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(self.id.as_str(), "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        let cost = est_vram(&self.path);
        // A .gguf checkpoint uses the Legacy (non-paged) decode path -- no KV
        // pool to add. The header peek is cheap (WeightReader never loads
        // tensors); any failure here just defers to the real, specific error
        // `activate()` raises -- `estimate()` must never itself hard-fail.
        if self.path.to_ascii_lowercase().ends_with(".gguf") {
            return cost;
        }
        let Ok(reader) = checkpoint::weightio::WeightReader::open(&self.path) else {
            return cost;
        };
        let cfg = qwen3::config::QwenConfig::from_json(&reader.config());
        let (block_size, _max_batch, _max_blocks_per_seq, num_blocks, _max_prefill) = Self::pool_sizing(Self::ctx());
        let kv_int8 = Self::kv_int8() && qwen3::serve::kv_int8_supported(&cfg);
        let kv_bytes = qwen3::serve::kv_pool_bytes(&cfg, block_size, num_blocks, kv_int8);
        MemCost::new(cost.vram + kv_bytes, cost.ram)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // Coarse stage progress -- NOT per-tensor (the actual weight upload
        // loop lives in `paramstore`, a lower-level crate that must not
        // depend on `residency` just for this), a handful of stage markers so
        // `-v -v` shows SOMETHING moving during a cold activate that can take
        // over a minute, without turning into a per-layer scroll.
        residency::log::info(&format!("{}: step 1/3 opening checkpoint", self.id));
        // Stream weights from the mmap (see GptResident::activate). Open first so
        // a GGUF can supply its own embedded tokenizer.
        let reader = checkpoint::weightio::WeightReader::open(&self.path).map_err(|e| format!("qwen: {e}"))?;
        residency::log::info(&format!("{}: step 2/3 loading tokenizer", self.id));
        // Tokenizer precedence: an explicit sibling `tokenizer.json` (safetensors
        // path, or an override) wins; else a `.gguf` builds from its embedded
        // `tokenizer.ggml.*` KV; else there is nothing to tokenize with.
        let tok = if !self.tokenizer.is_empty() {
            data::qwen_tokenizer::QwenBpe::from_file(&self.tokenizer)?
        } else if let Some(gt) = reader.tokenizer() {
            data::qwen_tokenizer::QwenBpe::from_gguf(&gt).map_err(|e| format!("qwen: {e}"))?
        } else {
            return Err("qwen: no tokenizer (set BRAIN_QWEN_TOKENIZER, or use a GGUF with an embedded tokenizer)".to_string());
        };
        let eos = tok.encode("<|im_end|>").first().copied();
        residency::log::info(&format!("{}: step 3/3 building engine (uploading weights to {device:?})", self.id));
        let ctx = Self::ctx();
        // `qwen3::serve::Engine` (the paged, continuous-batching serving engine --
        // see this plan's W2/W3/W5) reads checkpoints via `checkpoint::load`,
        // which is SAFETENSORS-ONLY (`checkpoint::parse` -> `st::parse_safetensors`).
        // A `.gguf` checkpoint therefore cannot build an `Engine` today -- this is
        // a real, pre-existing gap (not introduced here), so `.gguf` keeps the
        // original single-sequence decode-only path rather than silently losing
        // GGUF support. Everything else (the common case: a `.brain.safetensors`
        // checkpoint, with or without a named LoRA adapter) gets the batched engine.
        let is_gguf = self.path.to_ascii_lowercase().ends_with(".gguf");
        let engine = on_device(device, || -> Result<QwenEngineKind, String> {
            if is_gguf {
                let model = qwen3::model::Qwen::from_reader_decode(&reader, ctx);
                // Read the (tied-embedding) LM head ONCE here, not per request:
                // the fix `generate_kv_stream_with_head`'s doc comment asks for
                // (594 MiB device->host re-read at real vocab/d_model, otherwise
                // paid on every single chat request).
                let head = model.read_weight(model.cfg.head_weight());
                return Ok(QwenEngineKind::Legacy { model: Box::new(model), head });
            }
            // See QwenResident::pool_sizing's doc comment for the arithmetic
            // -- the same derivation `estimate()` predicts a budget from, so
            // the two cannot silently drift apart.
            let (block_size, max_batch, max_blocks_per_seq, num_blocks, max_prefill) = QwenResident::pool_sizing(ctx);
            // Both branches MUST pass the same kv_int8 -- a base and its
            // folded-adapter sibling serving on numerically different KV
            // paths would be a confusing, undocumented split.
            //
            // A DEFAULT-selecting caller degrades loudly rather than hitting
            // `Engine::from_map_with_gpu`'s hard assert: nobody explicitly
            // asked this checkpoint's unusual `head_dim` for int8, so a
            // serving-process panic on activation would be the wrong failure
            // mode -- see `qwen3::serve::kv_int8_supported`'s doc comment.
            let checkpoint_cfg = qwen3::config::QwenConfig::from_json(&reader.config());
            let kv_int8 = QwenResident::kv_int8() && qwen3::serve::kv_int8_supported(&checkpoint_cfg);
            if QwenResident::kv_int8() && !kv_int8 {
                eprintln!(
                    "serve: {}: int8 KV requested (the default) but head_dim={} is not a multiple of 4; falling back to fp32 KV",
                    self.path, checkpoint_cfg.head_dim
                );
            }
            // Boundary guard: refuse a fp32 KV pool over the safety ceiling,
            // loudly and specifically, BEFORE Engine::from_map_with_gpu
            // attempts the device allocation. Only reachable via an explicit
            // opt-out (--kv-fp32/BRAIN_QWEN_KV_INT8=0) or an unsupported
            // head_dim -- the int8 default at the SAME ctx does not come
            // close.
            if !kv_int8 {
                QwenResident::check_fp32_kv_pool_fits(&checkpoint_cfg, block_size, num_blocks, ctx, &self.path)?;
            }
            let mut eng = match &self.adapter {
                None => qwen3::serve::Engine::load(&self.path, block_size, num_blocks, max_batch, max_blocks_per_seq, max_prefill, kv_int8, false),
                // Fold the adapter's delta into the base tensors first (the same
                // fold `qwen3::eval::score_chat` uses to score one) -- the result
                // is an ordinary frozen base, zero extra inference cost versus
                // the base once folded.
                Some(a) => {
                    let mut tensors = checkpoint::load(&self.path).by_role("");
                    let mut cfg = qwen3::config::QwenConfig::from_json(&reader.config());
                    qwen3::lora::fold_adapter_into(&mut tensors, a).map_err(|e| format!("qwen: folding adapter {a}: {e}"))?;
                    cfg.lora = None;
                    qwen3::serve::Engine::from_map(cfg, &tensors, block_size, num_blocks, max_batch, max_blocks_per_seq, max_prefill, kv_int8, false)
                }
            };
            // BRAIN_QWEN_KV_CALIB=1: opt IN to a kv_calib.json beside the
            // BASE checkpoint (self.path, not the adapter's own file, since
            // the adapter is folded into the base's K/V distribution) --
            // KvCalib::from_model_dir already warns and returns None on a
            // missing file or a shape mismatch, so opting in without a real
            // file just serves uncalibrated, same as not opting in.
            if kv_int8 && QwenResident::kv_calib_opt_in() {
                if let Some(dir) = std::path::Path::new(&self.path).parent() {
                    let calib = model::kvcalib::KvCalib::from_model_dir(dir, checkpoint_cfg.n_layers as usize, checkpoint_cfg.n_kv_heads as usize, checkpoint_cfg.head_dim as usize);
                    eng.set_kv_calib(calib);
                }
            }
            Ok(QwenEngineKind::Batched(Box::new(model::serve::Scheduler::new(eng, max_batch as usize))))
        })??;
        Ok(Box::new(QwenInstance { tok, eos, engine }))
    }
}

/// Which serving path this instance drives — see [`QwenResident::activate`]'s
/// `.gguf` note for why both still exist.
enum QwenEngineKind {
    /// The original single-sequence KV-cache decode path (`Qwen::
    /// from_reader_decode` + `generate_kv_stream_with_head`) -- GGUF only.
    /// `model` boxed: `qwen3::model::Qwen` is ~1.6 KB by value, which would
    /// otherwise size every `QwenEngineKind` (even a `Batched` one) to it.
    Legacy { model: Box<qwen3::model::Qwen>, head: Vec<f32> },
    /// The paged, continuous-batching serving engine (this plan's W2/W3) --
    /// every safetensors checkpoint, the common case. Boxed for the same
    /// reason as `Legacy.model`: `Scheduler<Engine>` is large by value too.
    Batched(Box<model::serve::Scheduler<qwen3::serve::Engine>>),
}

struct QwenInstance {
    tok: data::qwen_tokenizer::QwenBpe,
    eos: Option<u32>,
    engine: QwenEngineKind,
}

impl Instance for QwenInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.run_batch(action, std::slice::from_ref(inv), &mut |_i, p| progress(p)).pop().unwrap()
    }

    /// `Legacy` (GGUF): the original sequential loop, one full generation per
    /// invocation, unchanged from before this rewiring.
    ///
    /// `Batched`: every invocation in `invs` is submitted into the SAME
    /// persistent `Scheduler` (built once at `activate`, so the paged KV pool
    /// and prefix cache are shared and reused across calls, not rebuilt) and
    /// driven to completion together — real continuous batching for
    /// whatever the dispatcher grouped into this one call (admitting MORE
    /// work into an ALREADY-running call is a separate, known gap this does
    /// not yet do).
    fn run_batch(&mut self, _action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        match &mut self.engine {
            QwenEngineKind::Legacy { model, head } => invs
                .iter()
                .enumerate()
                .map(|(i, inv)| run_one_legacy(model, head, &self.tok, self.eos, inv, &mut |p| progress(i, p)))
                .collect(),
            QwenEngineKind::Batched(sched) => run_batch_scheduled(sched, &self.tok, self.eos, invs, progress),
        }
    }

    /// `Batched`'s prefix-cache effectiveness, surfaced through
    /// `Executor::stats().metrics` — reachable from HTTP/D-Bus for the first
    /// time (previously only observable from `brain perf`'s in-process
    /// `PagedLlmTarget`, which bypasses the served path entirely). `Legacy`
    /// (GGUF) has no prefix cache, so it reports nothing extra.
    fn metrics(&self) -> Vec<(String, serde_json::Value)> {
        match &self.engine {
            QwenEngineKind::Legacy { .. } => Vec::new(),
            QwenEngineKind::Batched(sched) => {
                let (hit, looked, cached) = sched.prefix_stats();
                let rate = if looked > 0 { hit as f64 / looked as f64 } else { 0.0 };
                vec![
                    ("kv_prefix_hit_rate".to_string(), serde_json::json!(rate)),
                    ("kv_prefix_hit_tokens".to_string(), serde_json::json!(hit)),
                    ("kv_prefix_lookup_tokens".to_string(), serde_json::json!(looked)),
                    ("kv_prefix_cached_blocks".to_string(), serde_json::json!(cached)),
                ]
            }
        }
    }
}

/// One full generation on the legacy single-sequence decode path — the exact
/// logic `QwenInstance::run` had before this rewiring, extracted so
/// `run_batch`'s sequential loop and the (unlikely, but possible) direct
/// `run` call share one implementation.
fn run_one_legacy(
    model: &qwen3::model::Qwen,
    head: &[f32],
    tok: &data::qwen_tokenizer::QwenBpe,
    eos: Option<u32>,
    inv: &Invocation,
    progress: &mut dyn FnMut(Progress),
) -> ActionResult {
    let req = parse_request(tok, inv)?;
    let mut rng = Rng::new(req.seed);
    let total = req.max_new as u32;
    progress(Progress::step(0, total, "generating"));

    // `generate_kv_stream_with_head` wants a stop-id SET, not an `Option`.
    let eos_arr: [u32; 1];
    let eos_slice: &[u32] = match eos {
        Some(e) => {
            eos_arr = [e];
            &eos_arr
        }
        None => &[],
    };

    let mut seq = SeqState::new(&req, inv.cancel.clone());
    let mut ids_out: Vec<u32> = Vec::with_capacity(req.max_new);
    let gen = qwen3::sample::generate_kv_stream_with_head(model, &req.ids, req.max_new, req.temp, req.top_k, req.top_p, eos_slice, &mut rng, head, &mut |_i, t| {
        ids_out.push(t);
        !seq.advance(tok, &ids_out, progress)
    });

    Ok(seq.finish(tok, &gen, progress))
}

/// Drive every invocation in `invs` to completion on the SAME persistent
/// `Scheduler` — see [`QwenInstance::run_batch`]'s doc.
fn run_batch_scheduled(
    sched: &mut model::serve::Scheduler<qwen3::serve::Engine>,
    tok: &data::qwen_tokenizer::QwenBpe,
    eos: Option<u32>,
    invs: &[Invocation],
    progress: &mut dyn FnMut(usize, Progress),
) -> Vec<ActionResult> {
    let mut results: Vec<Option<ActionResult>> = vec![None; invs.len()];
    let mut seq_for_bi: Vec<Option<SeqState>> = Vec::with_capacity(invs.len());
    let mut id_for_bi: Vec<Option<u64>> = Vec::with_capacity(invs.len());

    for (bi, inv) in invs.iter().enumerate() {
        match parse_request(tok, inv) {
            Ok(req) => {
                let sample = model::serve::SampleParams { temp: req.temp, top_k: req.top_k, top_p: req.top_p };
                let seed = req.seed;
                let max_new = req.max_new;
                let seq = SeqState::new(&req, inv.cancel.clone());
                let id = sched.submit_sampled(model::serve::Request { prompt: req.ids, max_new, eos }, sample, seed);
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
        // Stream each still-open sequence's newly generated suffix and check
        // its stop-string/cancellation; a triggered sequence is cancelled
        // (and finalised) immediately rather than waiting for the scheduler
        // to reap it naturally — `Scheduler::cancel` returns its tokens so
        // far synchronously and reclaims its blocks right away.
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
        // A request the scheduler refuses at admission (a prompt token
        // outside its vocabulary, or one that can never fit its per-sequence
        // capacity — see `model::serve::RejectReason`) never appears in
        // `completed` and never will; without handling it here its `bi`
        // would stay in `remaining` forever, spinning this loop on an
        // otherwise-empty scheduler.
        for (id, reason) in report.rejected {
            let bi = id_for_bi.iter().position(|x| *x == Some(id)).expect("rejected id must belong to this batch");
            seq_for_bi[bi] = None;
            results[bi] = Some(Err(format!("qwen: {reason}")));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// REGRESSION: nothing previously asserted what the serving default IS
    /// (only that specific engines built with an explicit `true`/`false`
    /// behaved correctly) -- this pins int8 KV as the default and every
    /// documented off-spelling, matching `BRAIN_AUTO_FETCH`'s convention
    /// (`build_auto_fetch_supplier`) for case/whitespace handling.
    #[test]
    fn kv_int8_defaults_on_and_recognizes_every_off_spelling() {
        assert!(QwenResident::kv_int8_from(None), "int8 KV must be the default with no env override");
        for off in ["0", "false", "off", "FALSE", "Off", " off ", "OFF"] {
            assert!(!QwenResident::kv_int8_from(Some(off)), "{off:?} must opt out of int8 KV");
        }
        for on in ["1", "true", "yes", "anything-else", ""] {
            assert!(QwenResident::kv_int8_from(Some(on)), "{on:?} must not disable int8 KV");
        }
    }

    /// Opposite default from `kv_int8`: calibration is OFF unless explicitly
    /// requested (P12's own measurement found a small calibration set makes
    /// things worse), so this must default to `false` with no env override
    /// and recognize the same on-spellings `kv_int8_from` recognizes for off.
    #[test]
    fn kv_calib_opt_in_defaults_off_and_recognizes_every_on_spelling() {
        assert!(!QwenResident::kv_calib_opt_in_from(None), "calibration must default OFF with no env override");
        for on in ["1", "true", "on", "TRUE", "On", " on ", "ON"] {
            assert!(QwenResident::kv_calib_opt_in_from(Some(on)), "{on:?} must opt in to calibration");
        }
        for off in ["0", "false", "yes", "anything-else", ""] {
            assert!(!QwenResident::kv_calib_opt_in_from(Some(off)), "{off:?} must not opt in to calibration");
        }
    }

    /// The fp32 KV boundary guard must refuse (not attempt) a pool over the
    /// ceiling, and pass through a pool comfortably under it -- pure
    /// arithmetic, no engine, no allocation, so a multi-GiB refusal case
    /// costs nothing to test.
    #[test]
    fn fp32_kv_pool_guard_refuses_over_ceiling_and_passes_under_it() {
        let cfg = qwen3::config::QwenConfig::tiny(); // head_dim=8, n_kv_heads=2, hkv=16
        // Small, ordinary sizing: comfortably under the ceiling.
        assert!(QwenResident::check_fp32_kv_pool_fits(&cfg, 16, 64, 2048, "test.safetensors").is_ok());
        // num_blocks chosen so this TINY config's fp32 pool (256 bytes/slot)
        // alone exceeds MAX_FP32_KV_POOL_BYTES (8 GiB) -- num_blocks=3,000,000
        // -> slots=48,000,000 -> ~11.4 GiB, comfortably over.
        let err = QwenResident::check_fp32_kv_pool_fits(&cfg, 16, 3_000_000, 100_000_000, "test.safetensors")
            .expect_err("an 11+ GiB fp32 pool must be refused, not attempted");
        assert!(err.contains("test.safetensors"), "error must name the checkpoint: {err}");
        assert!(err.contains("100000000"), "error must name the requested ctx: {err}");
        assert!(err.contains("GiB"), "error must name the computed size: {err}");
    }

    /// The plan's ctx=24576 sizing table (from the planning notes)
    /// was a HAND ESTIMATE before this test -- this replaces
    /// it with the real number, computed through the exact same
    /// `pool_sizing`/`kv_pool_bytes` the resident actually calls, at the
    /// REAL Qwen3-0.6B config (not `tiny()` -- lesson 18, a toy-fitted
    /// number can't predict the real shape). Pure arithmetic, no device, no
    /// checkpoint file, so it runs always.
    #[test]
    fn ctx_24576_int8_kv_pool_fits_but_fp32_kv_pool_would_be_refused() {
        let cfg = qwen3::config::QwenConfig::qwen3_0_6b();
        let ctx = 24576u32;
        assert_eq!(QwenResident::ctx(), ctx, "this test's premise is the new default -- update it if the default changes");
        let (block_size, _max_batch, _max_blocks_per_seq, num_blocks, _max_prefill) = QwenResident::pool_sizing(ctx);

        let int8_bytes = qwen3::serve::kv_pool_bytes(&cfg, block_size, num_blocks, true);
        let fp32_bytes = qwen3::serve::kv_pool_bytes(&cfg, block_size, num_blocks, false);
        eprintln!(
            "ctx={ctx}: int8 KV pool = {:.2} GiB, fp32 KV pool = {:.2} GiB (ceiling {:.0} GiB)",
            int8_bytes as f64 / (1u64 << 30) as f64,
            fp32_bytes as f64 / (1u64 << 30) as f64,
            MAX_FP32_KV_POOL_BYTES as f64 / (1u64 << 30) as f64,
        );

        assert!(int8_bytes <= MAX_FP32_KV_POOL_BYTES, "int8 KV pool at the new ctx default must fit the iGPU policy budget on its own");

        // `check_fp32_kv_pool_fits` always checks the FP32 pool regardless of
        // what dtype the caller actually intends to serve -- it's only ever
        // CALLED from `activate()` on the `!kv_int8` arm (the fp32 opt-out).
        // Driving it directly here re-derives the exact refusal `activate()`
        // would hit under `--kv-fp32`/`BRAIN_QWEN_KV_INT8=0` at this ctx.
        assert!(fp32_bytes > MAX_FP32_KV_POOL_BYTES, "fp32 KV pool at ctx=24576 must exceed the ceiling -- this is WHY the default is safe only under int8");
        let err = QwenResident::check_fp32_kv_pool_fits(&cfg, block_size, num_blocks, ctx, "qwen3-0.6b")
            .expect_err("the fp32 opt-out at the new ctx default must be refused, not attempted");
        assert!(err.contains("qwen3-0.6b") && err.contains("24576"), "refusal must name the checkpoint and the requested ctx: {err}");
    }

    /// Real-checkpoint version of the arithmetic test above: actually builds
    /// the paged engine at ctx=24576 against the real Qwen3-0.6B checkpoint
    /// (`Engine::kv_pool_bytes()` must equal the pure function above -- no
    /// drift between "what we predicted" and "what got allocated"), and
    /// measures real host RSS before/after via `host_mem_mb()` -- on this
    /// iGPU box host RSS *is* device memory (see `est_vram`'s doc / D1), so
    /// this is the honest number, not a second guess. `BRAIN_DEVICE=cpu` so
    /// this never additionally carves the iGPU's own 8 GiB policy budget out
    /// of the same shared RAM while the box is also running everything else.
    ///
    /// Gated on `QWEN3_DIR` (needs `qwen.brain.safetensors`), skips loudly if
    /// unset -- same convention as `crates/cli/tests/qwen_eval.rs`.
    #[test]
    #[ignore = "slow: real checkpoint"]
    fn kv_pool_bytes_at_the_new_ctx_default_fits_the_igpu_budget() {
        let Ok(dir) = std::env::var("QWEN3_DIR") else {
            eprintln!("SKIP: set QWEN3_DIR to a real Qwen3-0.6B checkpoint dir to run this test");
            return;
        };
        let path = std::path::Path::new(&dir).join("qwen.brain.safetensors");
        if !path.is_file() {
            eprintln!("SKIP: {} not found under QWEN3_DIR", path.display());
            return;
        }
        // SAFETY: this test is `#[ignore]`d (never runs under the default
        // `TEST_THREADS=8` fast lane) and is invoked alone, single-threaded,
        // for exactly this measurement -- see the module doc's OOM-budget
        // discipline (never build a real-shape engine alongside other tests).
        unsafe { std::env::set_var("BRAIN_DEVICE", "cpu") };

        let cfg = qwen3::config::QwenConfig::qwen3_0_6b();
        let ctx = QwenResident::ctx();
        assert_eq!(ctx, 24576, "this test's premise is the new default");
        let (block_size, max_batch, max_blocks_per_seq, num_blocks, max_prefill) = QwenResident::pool_sizing(ctx);
        let expected_pool_bytes = qwen3::serve::kv_pool_bytes(&cfg, block_size, num_blocks, true);

        let before = perf::scenarios::soak::host_mem_mb();
        let eng = qwen3::serve::Engine::load(path.to_str().unwrap(), block_size, num_blocks, max_batch, max_blocks_per_seq, max_prefill, true, false);
        let after = perf::scenarios::soak::host_mem_mb();

        assert_eq!(eng.kv_pool_bytes(), expected_pool_bytes, "the engine must allocate exactly what the pure function predicts -- no drift");

        let (before_mb, after_mb) = (before.unwrap_or(0.0), after.unwrap_or(0.0));
        eprintln!(
            "ctx=24576, real Qwen3-0.6B: kv_pool_bytes = {:.2} GiB, host RSS {:.0} MiB -> {:.0} MiB (delta {:.0} MiB)",
            expected_pool_bytes as f64 / (1u64 << 30) as f64,
            before_mb,
            after_mb,
            after_mb - before_mb,
        );
        // The iGPU policy budget is 8 GiB (`run_cli.rs::build_serving_executor`);
        // total resident host footprint (weights + KV pool + scratch) must
        // stay comfortably under that at the new default, not just the pool
        // alone -- this is the number the plan's hand-estimate table stood
        // in for.
        assert!(after_mb < 8192.0, "total host RSS after activation at ctx=24576 must stay under the 8 GiB iGPU policy budget, got {after_mb:.0} MiB");
    }

    /// The residency budget must PREDICT the memory the KV pool actually
    /// costs, not just the checkpoint file size -- before this, `estimate()`
    /// was `est_vram` alone (file size * 1.3), so switching KV dtype changed
    /// the RESIDENCY BUDGET, `crates/stats` and braintop by exactly zero,
    /// however much the real pool shrank.
    ///
    /// Doesn't mutate `BRAIN_QWEN_KV_INT8` (see `kv_int8_from`'s doc comment
    /// on why a test shouldn't race other tests over process-global env
    /// state): checks `estimate()` matches an independent recomputation at
    /// whatever the process's CURRENT default is, and checks the dtype-
    /// shrink property directly against the pure `kv_pool_bytes` function
    /// (parameterized over `kv_int8`, no engine, no env needed).
    #[test]
    fn estimate_counts_the_kv_pool_and_shrinks_under_int8() {
        let path = write_tiny_checkpoint(11, "estimate");
        let cfg = qwen3::config::QwenConfig { vocab: 151936, ..qwen3::config::QwenConfig::tiny() };
        let card = checkpoint::st::ModelCard::new("brain/qwen", "qwen");
        let resident = QwenResident::from_card(path.to_str().unwrap(), &card, Some("unused.json"), None);
        let key = InstanceKey::new("brain/qwen", "default");

        let (block_size, _max_batch, _max_blocks_per_seq, num_blocks, _max_prefill) = QwenResident::pool_sizing(QwenResident::ctx());
        let kv_int8 = QwenResident::kv_int8();
        let expected_kv_bytes = qwen3::serve::kv_pool_bytes(&cfg, block_size, num_blocks, kv_int8);
        let file_only = est_vram(path.to_str().unwrap()).vram;

        let got = resident.estimate(&key);
        assert_eq!(got.vram, file_only + expected_kv_bytes, "estimate() must equal file size + the KV pool, no more and no less");
        assert!(expected_kv_bytes > 0, "the KV pool must contribute a nonzero amount at these test dims");

        // The shrink property itself: pure function, both dtypes, no engine.
        let fp32_bytes = qwen3::serve::kv_pool_bytes(&cfg, block_size, num_blocks, false);
        let int8_bytes = qwen3::serve::kv_pool_bytes(&cfg, block_size, num_blocks, true);
        assert!(int8_bytes < fp32_bytes, "int8 must cost fewer bytes than fp32 at the same num_blocks");
    }

    /// Build a tiny (fast-CPU-testable) but REAL safetensors checkpoint whose
    /// vocab covers the real tokenizer's full range (so chat-template special
    /// tokens like `<|im_start|>` never index outside the embedding table —
    /// same reasoning as `rejected_admission_resolves_promptly_instead_of_hanging`),
    /// and write it to a scratch dir. Returns the checkpoint path.
    fn write_tiny_checkpoint(seed: u64, tag: &str) -> std::path::PathBuf {
        let cfg = qwen3::config::QwenConfig { vocab: 151936, ..qwen3::config::QwenConfig::tiny() };
        let init = qwen3::init_weights(&cfg, seed);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
            .param_list()
            .into_iter()
            .map(|(name, n)| {
                let v = init.get(&name).unwrap_or_else(|| panic!("init missing {name}")).clone();
                (name, vec![n as u64], v)
            })
            .collect();
        let dir = std::env::temp_dir().join(format!("qwen-resident-http-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.safetensors");
        checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
        path
    }

    /// REGRESSION coverage for the "no prefix/KV-cache reuse across HTTP
    /// requests" finding from the serving-performance audit, at the layer
    /// that had never been observable before this workstream: through the
    /// real HTTP router.
    /// `qwen3::serve::Engine`'s `PrefixCache` was already gated in isolation
    /// (`serve.rs::random_shared_prefixes_stay_exact`) but nothing surfaced
    /// its hit rate through a served request until `Instance::metrics` +
    /// `Executor::stats().metrics` (this pass) gave it a path out.
    ///
    /// Two chat requests share a long system prompt (differing only in the
    /// short user turn) submitted SEQUENTIALLY against the SAME resident
    /// (`QwenResident` persists its `Scheduler`/KV pool/prefix cache across
    /// `activate` — see that method's doc), so the second's prefill should
    /// find the first's system-prompt blocks already cached.
    ///
    /// Needs a real tokenizer (`QWEN_TOKENIZER=/path/to/tokenizer.json`) --
    /// self-skips loudly when unset.
    #[test]
    fn prefix_cache_hit_rate_is_observable_through_the_real_http_router() {
        let Ok(tok_path) = std::env::var("QWEN_TOKENIZER") else {
            eprintln!("SKIP: set QWEN_TOKENIZER to a real tokenizer.json to run this test");
            return;
        };
        let path = write_tiny_checkpoint(5, "prefix");

        let card = checkpoint::st::ModelCard::new("brain/qwen", "qwen");
        let resident = QwenResident::from_card(path.to_str().unwrap(), &card, Some(&tok_path), None);
        let models: Vec<std::sync::Arc<dyn ResidentModel>> = vec![std::sync::Arc::new(resident)];
        let mut budgets = residency::budget::Budgets::new();
        budgets.set(Device::Cpu, 8 << 30, 0);
        let exec = residency::Executor::start(models, budgets, residency::scheduler::Policy::default());

        let key = "sk-brain-test-key".to_string();
        let state = apiserve::AppState::new(exec.clone(), key.clone(), apiserve::Provider::OpenAI);
        let app = apiserve::router(state);

        // A long, shared system prompt (many tokens) is the exact shape the
        // audit named: a real agent's system prompt + tool-schema block,
        // repeated across turns while only the user turn changes.
        let system: String = (0..200).map(|i| format!("rule {i}: always answer politely. ")).collect();
        let post = |user: &str| {
            let body = serde_json::json!({
                "model": "brain/qwen",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 4,
                "temperature": 0,
            });
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri("/v1/chat/completions")
                .header(axum::http::header::AUTHORIZATION, format!("Bearer {key}"))
                .header(axum::http::header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap()
        };

        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            use tower::ServiceExt;
            let r1 = app.clone().oneshot(post("first turn")).await.unwrap();
            assert_eq!(r1.status(), axum::http::StatusCode::OK, "first request must succeed");
            let r2 = app.clone().oneshot(post("second turn")).await.unwrap();
            assert_eq!(r2.status(), axum::http::StatusCode::OK, "second (prefix-sharing) request must succeed");
        });

        let key = InstanceKey::new("brain/qwen", "default");
        let stats = exec.stats();
        let m = stats.metrics.get(&key).unwrap_or_else(|| panic!("no metrics for {key:?}; stats={stats:?}"));
        let rate = m.iter().find(|(k, _)| k == "kv_prefix_hit_rate").map(|(_, v)| v.as_f64().unwrap_or(0.0)).unwrap_or(0.0);
        assert!(rate > 0.0, "second request must hit the first's cached system-prompt prefix; metrics={m:?}");
    }

    /// REGRESSION: `run_batch_scheduled` must resolve EVERY batch index,
    /// including one the scheduler REJECTS at admission (`model::serve::
    /// RejectReason`) rather than completes. Before this test's fix, a
    /// rejected sequence's `bi` was never removed from the pending set —
    /// `report.rejected` was silently ignored — so `run_batch_scheduled`
    /// spun forever calling `step_report()` on an otherwise-empty scheduler.
    /// Reproduced live: `http:qwen-synth:<small vocab>` against a REAL
    /// tokenizer whose chat-template special tokens exceed the synth
    /// model's vocab hung indefinitely until this fix landed.
    ///
    /// Needs a real tokenizer (`QWEN_TOKENIZER=/path/to/tokenizer.json`) --
    /// self-skips loudly when unset.
    #[test]
    fn rejected_admission_resolves_promptly_instead_of_hanging() {
        let Ok(tok_path) = std::env::var("QWEN_TOKENIZER") else {
            eprintln!("SKIP: set QWEN_TOKENIZER to a real tokenizer.json to run this test");
            return;
        };
        let tok = data::qwen_tokenizer::QwenBpe::from_file(&tok_path).expect("load tokenizer");

        // A deliberately tiny vocab: real chat-template special tokens
        // (`<|im_start|>` etc.) encode to ids from the REAL tokenizer's full
        // (~151936) vocabulary, so a `chat: true` render is certain to
        // include at least one id outside this vocab -- the exact class
        // this test guards.
        let cfg = qwen3::config::QwenConfig { vocab: 64, ..qwen3::config::QwenConfig::tiny() };
        let weights = qwen3::init_weights(&cfg, 3);
        let eng = qwen3::serve::Engine::from_map(cfg, &weights, 8, 16, 4, 4, 16, false, false);
        let mut sched = model::serve::Scheduler::new(eng, 4);

        let inv = Invocation::new().set("prompt", json!("hello")).set("chat", json!(true)).set("max_new", json!(4));
        let start = std::time::Instant::now();
        let results = run_batch_scheduled(&mut sched, &tok, None, std::slice::from_ref(&inv), &mut |_i, _p| {});
        assert!(start.elapsed() < std::time::Duration::from_secs(5), "rejected admission must resolve promptly, not hang");
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err(), "an out-of-vocab prompt must be reported as an error, not silently dropped");
    }
}
