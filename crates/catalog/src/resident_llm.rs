// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident-model adapters for brain's text-generation LLMs - GPT (dense
//! char-level baseline), GLM (MLA + noaux_tc MoE decoder), and Qwen3 (BPE
//! decoder) - behind the residency [`Executor`], mirroring the yolo/z-image
//! adapters in [`crate::resident_yolo`] and [`crate::resident_zimage`].
//!
//! Each model family is one [`ResidentModel`] with a single `"generate"` action.
//! Unlike yolo (which pins itself to the CPU via `Gpu::new_cpu` unless a
//! `--device` was chosen), these models load through `gpu_core::Gpu::new`, i.e.
//! the process-default backend - **wgpu (GPU) unless `BRAIN_DEVICE=cpu`**. So the
//! resident instance holds the model on a GPU (VRAM); dropping it frees the card.
//! `activate` places the build on the assigned card via a scoped device-registry
//! selection ([`on_device`]), exactly like z-image.
//!
//! Model SELECTION: `--checkpoint` (+ `--tokenizer`) names the Qwen3
//! checkpoint to serve; `BRAIN_GPT2_WEIGHTS` and `BRAIN_GLMDSA_WEIGHTS` still
//! name theirs, and each `from_env` returns `None` when its variable is
//! unset/empty. HOW to serve Qwen3 - context length, batching, KV/weight
//! precision - is [`QwenServeConfig`], which `run_cli.rs`'s serving-policy
//! flags build; see that struct's own doc for every field and its default.
//! With no `--ctx` given, context auto-sizes to the target device's
//! real free VRAM (see [`QwenResident::resolve_ctx`]) rather than a single
//! fixed number for every box.

use capability::{ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, ParamSpec, ParamType, Progress};
use checkpoint::st::ModelCard;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::json;

use crate::adapter_registry::{AdapterBankSettings, AdapterHost, AdapterRegistry, AdapterSync, QwenAdapterView, ADAPTER_PARAM};
use crate::adapter_release::{LoadedAdapter, ServedBase};
use llmserve::{run_batch_scheduled, run_open_scheduled, OpenGroup, scheduler_metrics, Frontend};

use std::sync::Arc;

use data::rng::Rng;
use data::tokenizer::{CharTokenizer, Tokenizer};
use qwen3::chat::{parse_request_as, sampling_params, text_outcome, ChatFormat};

// ---------------------------------------------------------------- shared

/// The shared `"generate"` action spec. `chat` adds Qwen's chat contract: the
/// chat-template toggle plus `messages`/`system`/`top_p`/`stop` and per-token
/// streaming (one `Progress::token` delta each accepted token).
///
/// `pub(crate)`: reused as-is by [`crate::resident_qwen35moe::Qwen35Resident`]
/// (same chat contract, same `qwen3::chat` shared parse underneath) rather
/// than duplicated a third time.
pub fn generate_spec(summary: &str, chat: bool) -> ActionSpec {
    let mut s = ActionSpec::new("generate", summary)
        .param(ParamSpec::new("prompt", ParamType::Str, "the prompt to continue (or chat message)"))
        .param(ParamSpec::new("max_new", ParamType::Int, "number of new tokens to generate").default(json!(128)))
        .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature (<= 0 = greedy)").default(json!(0.8)))
        .param(ParamSpec::new("top_k", ParamType::Int, "top-k filter (40 = standard; 1 = greedy; 0 or negative = disabled)").default(json!(40)))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (omit for random)"));
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
            .param(ParamSpec::new(
                "suffix",
                ParamType::Str,
                "fill-in-the-middle: the text after the insertion point, with the raw prompt (chat false) the text before it (needs FIM tokens in the vocabulary)",
            ))
            .param(ParamSpec::new("top_p", ParamType::Float, "nucleus sampling threshold (>= 1 = disabled)").default(json!(1.0)))
            .param(ParamSpec::new("stop", ParamType::Str, "JSON array of stop strings"))
            .param(ParamSpec::new("tools", ParamType::Str, "JSON array of tool definitions (OpenAI function-calling schema)"))
            .param(ParamSpec::new("tool_choice", ParamType::Str, "tool_choice directive, raw JSON text (\"auto\"|\"none\"|\"required\"|{...})"))
            .param(ParamSpec::new("enable_thinking", ParamType::Bool, "allow the model to emit a <think> reasoning block").default(json!(true)))
            .param(ParamSpec::new("thinking_budget", ParamType::Int, "tokens the think block may take; after that the closing tag is forced and the model answers (default: unbounded)"))
            .param(ParamSpec::new("min_p", ParamType::Float, "drop tokens below this fraction of the best token's probability (0 = off)"))
            .param(ParamSpec::new("presence_penalty", ParamType::Float, "subtracted from the logit of every token already generated (0 = off)"))
            .param(ParamSpec::new("frequency_penalty", ParamType::Float, "subtracted from a token's logit once per time it was generated (0 = off)"))
            .param(ParamSpec::new("repetition_penalty", ParamType::Float, "divides the positive logit of every token in the context, prompt included (1 = off)"))
            .param(ParamSpec::new("dry_multiplier", ParamType::Float, "DRY repetition penalty strength over the whole context (0 = off)"))
            .param(ParamSpec::new("dry_base", ParamType::Float, "DRY penalty growth per extra repeated token (default 1.75)"))
            .param(ParamSpec::new("dry_allowed_length", ParamType::Int, "DRY: repeats up to this many tokens are not penalised (default 2)"))
            .param(ParamSpec::new("dry_sequence_breakers", ParamType::Str, "JSON array of strings that end a DRY sequence (default newline, colon, quote, asterisk)"));
    }
    llmserve::with_token_capture(s.output(BlobSpec::new("text", Media::Text, "the generated text")))
}

/// Estimate the Hot VRAM footprint of a checkpoint as ~1.3x its file size.
/// `pub(crate)`: reused by [`crate::resident_qwen35moe::Qwen35Resident`] too.
///
/// Measured against a real `Qwen/Qwen3-8B` checkpoint at the historical
/// perf-number: memory-footprint ratio of the int8 weights, not a timing
/// 24576 default context (`--ctx`), int8 KV: weights (1.3x a ~16 GiB bf16 file)
/// ~20.8 GiB + KV pool ~3.5 GiB + paged-attention scratch ~3.0 GiB totals
/// ~27.3 GiB - more than a single 24 GiB P40's ENTIRE capacity, before
/// `--reserve-gb` even enters the picture. `weights_int8_bytes` below is the
/// lever that actually closes that gap: quantizing the 7 per-layer linears
/// (the dominant term) to int8 cuts the weights term to roughly a quarter,
/// not a `--reserve-gb` adjustment.
pub(crate) fn est_vram(path: &str) -> MemCost {
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0).saturating_mul(13) / 10;
    MemCost::new(bytes, 0)
}

/// The device byte footprint of a checkpoint matching `cfg`'s decoder
/// weights UNDER int8 quantization of the 7 per-layer linears
/// (`qwen3::q8::Q8::LINEARS`) - everything else (the token embedding, the
/// LM head, norms) stays fp32, exactly as `Engine::from_map_with_gpu`'s own
/// `w8_on` branch builds it.
///
/// Deliberately computed from `QwenConfig::param_list()` - the SAME
/// config-derived, canonical-name list `qwen3::serve`'s own
/// `decoder_param_list` builds the engine from - and NOT from
/// `WeightReader::names()`/`shape()` (the checkpoint's raw on-disk tensor
/// names). REGRESSION this closes: `checkpoint::load(path)`'s `Container::
/// find` resolves/remaps a real checkpoint's raw tensor names onto these
/// SAME canonical names before `Engine::load` ever sees them, so
/// `Q8::is_i8_linear` matching against the RAW names (what `WeightReader`
/// exposes) matched NOTHING on a real checkpoint whose on-disk names differ
/// from brain's canonical `"blocks.N.leaf"` form - every tensor silently
/// fell through to the fp32 branch, so this estimate reported the FULL
/// fp32 total even with int8 weights requested and correctly applied at
/// activation time (`37899 MiB` observed as `30.51 GiB` weights + KV +
/// scratch - exactly the fp32 total, not the ~7.28 GiB int8 one). Deriving
/// this from the config instead of the file makes it structurally
/// impossible for the estimate to disagree with what `decoder_param_list`
/// actually builds, regardless of the checkpoint's own raw naming.
///
/// A linear's total element count IS `n*k`; the packed layout's `n*(k/32)*4`
/// group-scale term equals `elems/32*4` for exactly the same reason,
/// whenever `k` (every real head_dim/d_model/d_ff here) is a multiple of 32
/// -- true for every shipped Qwen3 config, so no separate `n`/`k` split is
/// needed to compute it.
/// The device byte footprint of `cfg`'s weights at fp32, with the same 1.3x
/// allowance [`est_vram`] gives a file - derived from the config, since the
/// checkpoint on disk may be a directory, or bf16 at half the device size.
pub fn weights_fp32_bytes(cfg: &qwen3::config::QwenConfig) -> u64 {
    cfg.param_list().into_iter().map(|(_, elems)| elems as u64 * 4).sum::<u64>() * 13 / 10
}

/// The device bytes of `cfg`'s weights with the linears and the head stored as bf16 (`--weight-format bf16`):
/// a bf16 checkpoint's own values at half the fp32 size; embeddings and norms stay fp32.
pub fn weights_bf16_bytes(cfg: &qwen3::config::QwenConfig) -> u64 {
    cfg.param_list()
        .into_iter()
        .map(|(name, elems)| {
            let elems = elems as u64;
            if qwen3::q8::Q8::is_i8_linear(&name) || name == cfg.head_weight() {
                elems * 2
            } else {
                elems * 4
            }
        })
        .sum()
}

pub fn weights_int8_bytes(cfg: &qwen3::config::QwenConfig) -> u64 {
    cfg.param_list()
        .into_iter()
        .map(|(name, elems)| {
            let elems = elems as u64;
            if qwen3::q8::Q8::is_i8_linear(&name) {
                elems + (elems / 32) * 4
            } else {
                elems * 4
            }
        })
        .sum()
}

/// The KV pool alone must not exceed this before `QwenResident::activate`
/// refuses outright, rather than let `Engine::from_map_with_gpu` attempt a
/// device allocation that fails with wgpu's cryptic per-buffer byte count
/// (`resident_llm.rs`'s own `pool_sizing` doc comment has the historical
/// crash). This guard must fire before any allocation, not after querying a
/// live budget that may not exist yet (e.g. `brain qwen serve`'s
/// direct-engine CLI path never builds a residency executor at all), so it
/// is a fixed, explicit ceiling of its own rather than one threaded down
/// from `run_cli.rs::build_serving_executor`'s per-run device budgets - a
/// prior version of this doc claimed it matched a `(8u64 << 30).min(ram / 2)`
/// formula there; that formula no longer exists (`build_serving_executor`'s
/// iGPU fallback now budgets the fallback GPU at the FULL `host_ram_available()`,
/// no fixed cap or halving), so this ceiling is NOT device-budget-aware -
/// it is the FLOOR of the ceiling: a server that knows its card budget
/// ([`QwenServeConfig::card_budget_bytes`]) allows an fp32 pool up to that
/// budget (the placer's own memory check still decides whether it fits next
/// to the weights), so a 96 GB card is not held to a limit sized for a
/// small-RAM box.
/// Only ever checked against the FP32 pool: at the historical 24576
/// default context, int8 is comfortably under this on its own
/// (`kv_pool_bytes_at_the_new_ctx_default_fits_the_igpu_budget`); fp32 is not.
const MAX_FP32_KV_POOL_BYTES: u64 = 8 << 30;

/// The largest per-binding size ANY device can report: wgpu clamps
/// `max_storage_buffer_binding_size` to `i32::MAX` on every backend, so a
/// buffer over this is unservable everywhere and can be refused without
/// knowing which card will run it.
///
/// This is an upper bound, and only an upper bound. It was previously
/// documented as the opposite - the size "every backend this engine runs on
/// guarantees AT LEAST" - which conflated two different numbers. WebGPU
/// guarantees 128 MiB (`maxStorageBufferBindingSize`'s spec default); real
/// adapters report anywhere from that floor, on software rasterisers and
/// mobile, up to 4 GiB on desktop discrete cards. A Tesla P40 on Vulkan
/// reports 2047 MiB; llvmpipe reports 128 MiB.
///
/// So no constant can decide whether a given sizing fits: checking the floor
/// refuses sizings real cards run fine, and checking the ceiling accepts
/// sizings that die on a software fallback. This one is deliberately the
/// ceiling, because its job is only to catch the absurd early, before
/// weights are read. The question that actually matters - does THIS device
/// accept this binding - is answered against the device's own queried limit
/// where the device exists, in `qwen3::serve::check_kv_bindings_fit`.
const WGPU_MAX_REPORTABLE_BINDING_BYTES: u64 = 2047 * (1 << 20);

/// Run `f` placed on the residency-assigned device: a GPU assignment becomes a
/// scoped (thread-local) selection in the canonical device registry, so every
/// `Gpu::new` inside `f` binds that physical card - race-free across the
/// executor's concurrent activation lanes. Shared by the resident adapters.
///
/// `f` always builds a wgpu (GPU/CPU backend) engine, so a `Device::Npu`
/// assignment must never reach here silently -- a resident that wants NPU
/// placement (`MemCost::with_npu`) must branch on `Device::Npu` in its own
/// `activate` *before* calling this (see `resident_depth.rs`), the same way
/// every other NPU-capable resident already does.
///
/// `Device::Cpu` scopes with [`gpu_core::devices::with_host_tier`], not a
/// bare `f()`: an unscoped `f()` leaves every `Gpu::new` inside bound to
/// whatever card is ambient (the memoized `auto_home()` answer from server
/// startup, still resolving to the SAME GPU residency just decided this part
/// could not have) - the exact "`Home::Cpu` was a label, not a placement"
/// defect `gpu_core::devices::Homes::run` was already fixed for (see that
/// function's own doc and `crates/gpu-core/tests/host_tier.rs`). This
/// resident path had never adopted the fix, so a residency fallback to the
/// host tier double-booked the GPU it was falling back FROM instead of
/// relieving it, and panicked with a `wgpu` OOM on a thread literally named
/// `brain-lane-Cpu`.
pub(crate) fn on_device<R>(device: Device, f: impl FnOnce() -> R) -> Result<R, String> {
    match device {
        Device::Gpu(i) => gpu_core::devices::with_gpu(i, f),
        Device::Cpu => Ok(gpu_core::devices::with_host_tier(f)),
        Device::Npu(i) => Err(format!(
            "on_device: got Device::Npu({i}) but this resident has no NPU activation path -- \
             a resident declaring MemCost::with_npu must branch on Device::Npu in its own \
             activate() before calling on_device, not fall through to the wgpu build"
        )),
    }
}

/// A brain safetensors checkpoint opened for serving: its config and a source that serves every tensor in the
/// encoding it is stored in ([`checkpoint::packed::stored_source`]), so a bf16 or int8 checkpoint reaches the
/// engine - on a card or on the host tier - without a whole-model fp32 copy.
pub(crate) fn open_stored_checkpoint(path: &str) -> Result<(serde_json::Value, Box<dyn checkpoint::TensorSource + Send + Sync>), String> {
    let reader = checkpoint::weightio::WeightReader::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
    if reader.gguf().is_some() {
        return Err(format!("{path}: a GGUF is not a brain-native checkpoint (serve it through the GGUF resident of its family)"));
    }
    let config = reader.config();
    Ok((config, checkpoint::packed::stored_source(reader)))
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

    /// Construct under the card's id. `_tokenizer` is unused - GPT is char-level
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
        let (max_new, temp, top_k, seed) = sampling_params(inv)?;
        let prompt = inv.get_str("prompt").unwrap_or_default();
        let prompt_text = if prompt.is_empty() { "\n".to_string() } else { prompt };
        let ids = self.tok.encode(&prompt_text);
        let mut rng = Rng::new(seed);
        progress(Progress::step(0, max_new as u32, "generating"));
        let gen = gpt2::sample::generate(&self.model, &ids, max_new, temp, top_k, &mut rng);
        let text = self.tok.decode(&gen);
        progress(Progress::step(max_new as u32, max_new as u32, "done"));
        Ok(text_outcome(text).set("seed", json!(seed)))
    }
}

// ---------------------------------------------------------------- glm

/// The GLM decoder (MLA + sigmoid noaux_tc MoE) behind the scheduler
/// (`BRAIN_GLMDSA_WEIGHTS`). Char-level: the checkpoint must embed its vocab.
///
/// Decoding is [`glmdsa::sample::generate_kv`] (the KV-cached fast path) -
/// the same one `glmdsa::caps::GenerateAction` (the direct `brain glmdsa
/// generate` path) calls, so the served and direct surfaces sample
/// identically rather than drifting (they used to: this adapter called the
/// slower cache-free `generate` while the direct path had already moved to
/// `generate_kv`). Bit-identity between `generate` and `generate_kv` only
/// holds at `temperature=0` (greedy) - `generate_kv` applies GLM's untied
/// `lm_head` on the host in a scalar loop, agreeing with the device path to
/// only ~1e-3, which is enough to flip a sampled token when two candidates
/// are close (see `generate_kv`'s own doc comment). The served default is
/// `temp=0.8`, so this is a real, disclosed numerical difference from the
/// pre-KV behaviour, not a regression: the two paths were never
/// bit-identical at temperature>0, they now just agree with EACH OTHER
/// (served == direct) instead of the served path being the slow, no-longer-
/// canonical one.
///
/// # Batching: deliberately serial, and here is why
///
/// `GlmInstance` runs the residency default `run_batch` (a loop over `run`,
/// `residency::model`'s `Instance` default) rather than an override - GLM has
/// no batch axis to exploit yet. [`glmdsa::model::Glm::step`] takes one
/// token id and one KV-cache, [`glmdsa::model::Glm::logits_all_compact`]
/// takes one window, and [`glmdsa::model::Glm::set_batch`] builds one
/// TRAINING sequence - there is no N dimension threaded through the MLA
/// attention or the sigmoid `noaux_tc` MoE dispatch anywhere in this model.
/// The two options this repo's serving contract names for a served decoder
/// (batching the prefill, or adopting `model::serve::PagedDecoder`) both
/// assume a batched forward already exists to drive - GLM's does not, and
/// building one (batched MLA + batched sigmoid `noaux_tc` MoE) is new
/// kernel/architecture work, tracked as its own out-of-scope roadmap item,
/// not something this adapter can shortcut. Concurrent requests are still
/// served today: the residency scheduler's own default loop interleaves
/// them across separate `run` calls on this one resident instance, exactly
/// like `crate::resident_restore`'s CodeFormer/VQGAN graphs (see that
/// module's doc for the same reasoning applied to a fixed-shape recorded
/// graph instead of a missing batch axis).
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

    /// Construct under the card's id. `path` is a brain checkpoint (char-level,
    /// carrying its own vocabulary) or a Hugging Face GLM checkpoint directory,
    /// served as downloaded with its own `tokenizer.json`; `_tokenizer` is
    /// unused either way.
    pub fn from_card(path: &str, card: &ModelCard, _tokenizer: Option<&str>) -> GlmResident {
        GlmResident { id: card.id.clone(), path: path.to_string() }
    }

    fn is_hf_dir(&self) -> bool {
        std::path::Path::new(&self.path).is_dir()
    }
}

impl ResidentModel for GlmResident {
    /// One definition, shared with the direct `brain glmdsa generate` path:
    /// `glmdsa::caps::manifest_resident` is `manifest()` minus the `weights`
    /// param this adapter supplies from `BRAIN_GLMDSA_WEIGHTS`. Building an
    /// `ActionSpec` here instead is how the served and direct surfaces drift
    /// into advertising different parameters for the same action.
    fn manifest(&self) -> Manifest {
        let mut m = glmdsa::caps::manifest_resident();
        m.model = self.id.clone();
        m
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(self.id.as_str(), "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        if !self.is_hf_dir() {
            return est_vram(&self.path);
        }
        // fp32 on the device, whatever the download's dtype; the same 1.3x
        // allowance `est_vram` gives a file.
        let cfg = std::fs::read_to_string(std::path::Path::new(&self.path).join("config.json")).ok().and_then(|j| glmdsa::import::config_from_hf(&j).ok());
        let bytes = cfg.map_or(0, |c| c.param_list().into_iter().map(|(_, n)| n as u64 * 4).sum::<u64>() * 13 / 10);
        MemCost::new(bytes, 0)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if self.is_hf_dir() {
            let dir = std::path::Path::new(&self.path);
            let (cfg, src) = glmdsa::import::open_dir(dir).map_err(|e| format!("glm: {e}"))?;
            let tok = data::qwen_tokenizer::QwenBpe::from_file(dir.join("tokenizer.json").to_str().ok_or("glm: non-UTF8 path")?).map_err(|e| format!("glm: {e}"))?;
            let block = cfg.block_size;
            let model = on_device(device, || glmdsa::model::Glm::from_source_inference(cfg, &src, 1, block))?;
            return Ok(Box::new(GlmInstance { model, tok: Box::new(tok) }));
        }
        // Stream weights from the mmap (see GptResident::activate).
        let reader = checkpoint::weightio::WeightReader::open(&self.path).map_err(|e| format!("glm: {e}"))?;
        let itos = glmdsa::model::Glm::itos_from_config(&reader.config())
            .ok_or("glm: checkpoint has no embedded char vocab (BRAIN_GLMDSA_WEIGHTS)")?;
        let tok = CharTokenizer::from_itos(itos);
        let block = glmdsa::config::GlmConfig::from_json(&reader.config()).block_size;
        let model = on_device(device, || glmdsa::model::Glm::from_reader_inference(&reader, 1, block))?;
        Ok(Box::new(GlmInstance { model, tok: Box::new(tok) }))
    }
}

struct GlmInstance {
    model: glmdsa::model::Glm,
    /// The checkpoint's own vocabulary: brain's char-level one, or a
    /// downloaded checkpoint's `tokenizer.json`.
    tok: Box<dyn Tokenizer + Send>,
}

impl Instance for GlmInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (max_new, temp, top_k, seed) = sampling_params(inv)?;
        let prompt = inv.get_str("prompt").unwrap_or_default();
        let prompt_text = if prompt.is_empty() { "\n".to_string() } else { prompt };
        let ids = self.tok.encode(&prompt_text);
        let mut rng = Rng::new(seed);
        progress(Progress::step(0, max_new as u32, "generating"));
        let gen = glmdsa::sample::generate_kv(&self.model, &ids, max_new, temp, top_k, None, &mut rng);
        let text = self.tok.decode(&gen);
        progress(Progress::step(max_new as u32, max_new as u32, "done"));
        Ok(text_outcome(text).set("seed", json!(seed)))
    }
    // `run_batch` is deliberately the residency-default serial loop - GLM has
    // no batch axis to exploit yet, see `GlmResident`'s module doc for why.
}

// ---------------------------------------------------------------- qwen

/// Every Qwen3 serving choice `brain serve`'s flags build and pass down:
/// which checkpoint to serve (`--checkpoint`, `--tokenizer`) and how to
/// serve it (the serving-policy flags). `Default` reproduces every historical env-var
/// default byte for byte, so a caller that builds `QwenServeConfig::
/// default()` (every test, `perf_cli.rs`, `model_dir.rs`'s catalog scan,
/// `continuous_train.rs`) is unaffected by this existing at all.
#[derive(Clone, Debug)]
pub struct QwenServeConfig {
    /// `--checkpoint PATH`: the checkpoint (file, or the repo directory
    /// holding it) the Qwen3 resident serves. `None` serves no such resident;
    /// the model store's Qwen3 checkpoints are served from the store scan.
    pub weights: Option<String>,
    /// `--tokenizer PATH`: the `tokenizer.json` for a `--checkpoint` that has
    /// none beside it. `None` uses the one beside the checkpoint (or a GGUF's
    /// embedded tokenizer).
    pub tokenizer: Option<String>,
    /// `--ctx N`. `None` means "size it automatically" - see
    /// [`QwenResident::resolve_ctx`]. Historical default when NEITHER this
    /// nor `auto_budget_bytes` is given: 24576.
    pub ctx: Option<u32>,
    /// The device budget to auto-size `ctx` against, when `ctx` itself is
    /// `None` - real free VRAM (minus `--reserve-gb`) for the card `brain
    /// serve` would actually place this on, known only to the caller of
    /// `serving::build_executor`. `None` (every caller except the live
    /// server) means "unknown", which keeps the historical 24576 fallback
    /// rather than guessing - auto-sizing is a property of the live
    /// serving path, not of this struct's mere existence.
    pub auto_budget_bytes: Option<u64>,
    /// The usable bytes of the largest card this server may place the model on
    /// (total minus `--reserve-gb`), when known. Unlike `auto_budget_bytes` it
    /// never sizes `ctx`: it only bounds the default KV pool
    /// ([`QwenResident::resolve_kv_tokens`]).
    pub card_budget_bytes: Option<u64>,
    /// `--max-batch N`, default 16.
    pub max_batch: u32,
    /// `--kv-cache f32` (presence opts OUT), default `true` (int8 KV is the
    /// serving default - see [`QwenResident::activate`]'s own doc for the
    /// measurement that earned it).
    pub kv_int8: bool,
    /// `--kv-cache fp8` (presence opts IN), default `false`: store the packed KV pools as FP8 E4M3
    /// instead of int8 (same bytes per token). Ignored, with a message, when the pools are fp32.
    pub kv_fp8: bool,
    /// `--kv-calibration` (presence opts IN), default `false`.
    pub kv_calib_opt_in: bool,
    /// `--kv-offload-gb N` (fractional allowed), default `0.0` (off).
    pub kv_offload_gb: f64,
    /// `--weight-format int8` (`Some(true)`) or `--weight-format f32`
    /// (`Some(false)`); `None` (the default) quantizes the linears of a
    /// checkpoint of `qwen3::footprint::INT8_WEIGHTS_FROM_PARAMS` parameters
    /// or more, which at fp32 does not fit one 24 GB card.
    pub weights_int8: Option<bool>,
    /// `--weight-format bf16`: the linears and the head are stored as bf16 and decoded inline (rounded to
    /// bf16, exact for a checkpoint that ships bf16; half the fp32 bytes). Takes precedence over `weights_int8`.
    pub weights_bf16: bool,
    /// `the prefill row cap N`, clamped to `1..=512`, default 512.
    pub max_prefill_cap: u32,
    /// `the KV pool size N`: the KV pool's size in tokens, shared by every
    /// sequence and by the prompt-prefix cache. `None` (the default) sizes it
    /// for `max_batch` half-context histories (never under two contexts). An
    /// agent fleet resending long histories every turn needs the pool to hold
    /// all of their histories at once (clients x history tokens), or the
    /// prefix cache evicts a conversation before its next turn.
    pub kv_tokens: Option<u32>,
    /// `--gemm fp32` (`Fp32Reference`) pins exact fp32 arithmetic. Default `Auto`: the
    /// fp32-weight linears of 8 or more rows (prefill, batched decode) multiply on TF32 tensor cores where the device has them
    /// (operands rounded to a 10-bit mantissa, fp32 accumulation).
    pub gemm_precision: gpu_core::provider::library::GemmPrecision,
    /// `the norm-fusion switch` turns off the residual-add + norm fusion (the A/B switch of that launch fusion); default on.
    pub fuse_residual_norm: bool,
    /// `--adapter-slots`, `--adapter-rank-cap`, `--adapter-targets`: the
    /// size of the adapter bank built beside the base weights, which lets
    /// requests run under LoRA adapters ([`crate::adapter_registry`]). `None`
    /// (the default) builds no bank and costs nothing; `Some` costs
    /// [`AdapterBankSettings::device_bytes`] of device memory, which the
    /// resident's footprint estimate includes. A model directory that holds
    /// adapters for a base turns the bank on for that base with the defaults
    /// of [`AdapterBankSettings`] when this is `None`.
    pub adapter_bank: Option<AdapterBankSettings>,
    /// Whether the operator named `--adapter-targets`. When they sized the
    /// bank by other flags only (`--adapter-slots`, `--adapter-rank-cap`), the
    /// targets are not theirs: a model directory's adapters widen them, as they
    /// do when no bank flag is given.
    pub adapter_targets_named: bool,
    /// Why the serving policy this server was started with cannot be honoured
    /// by the Qwen3 resident (set by [`Self::apply_policy`]); activation
    /// refuses with it instead of serving something else.
    pub refused: Option<String>,
}

/// What the Qwen3 paged engine honours of a [`model::policy::ServePolicy`].
pub const QWEN3_POLICY: model::policy::PolicySupport = model::policy::PolicySupport {
    model: "qwen3",
    weights: &[gpu_core::select::Dtype::F32, gpu_core::select::Dtype::BF16, gpu_core::select::Dtype::I8],
    weight_rules: false,
    kv_cache: &["f32", "int8", "fp8"],
    gemm: true,
    fuse_norm: true,
    kv_calibration: true,
    exact_decode: false,
};

impl QwenServeConfig {
    /// Apply the model-agnostic serving policy. A precision choice this engine
    /// cannot honour is recorded in [`Self::refused`] (and returned) rather
    /// than ignored. Returns notes for limits that have no meaning here.
    pub fn apply_policy(&mut self, p: &model::policy::ServePolicy) -> Result<Vec<String>, String> {
        if let Err(e) = QWEN3_POLICY.check(p) {
            self.refused = Some(e.clone());
            return Err(e);
        }
        if let Some(w) = &p.weights {
            use gpu_core::select::Dtype;
            let (int8, bf16) = match w.default_dtype() {
                Dtype::I8 => (Some(true), false),
                Dtype::BF16 => (None, true),
                _ => (Some(false), false),
            };
            self.weights_int8 = int8;
            self.weights_bf16 = bf16;
        }
        if let Some(t) = p.kv_cache {
            self.kv_int8 = t.as_str() != "f32";
            self.kv_fp8 = t.as_str() == "fp8";
        }
        self.kv_calib_opt_in |= p.kv_calibration;
        if let Some(g) = p.gemm {
            self.gemm_precision = g;
        }
        if let Some(f) = p.fuse_norm {
            self.fuse_residual_norm = f;
        }
        let l = &p.limits;
        self.ctx = l.ctx.or(self.ctx);
        self.max_batch = l.max_batch.unwrap_or(self.max_batch);
        self.kv_tokens = l.kv_tokens.or(self.kv_tokens);
        self.kv_offload_gb = l.kv_offload_gb.unwrap_or(self.kv_offload_gb);
        self.max_prefill_cap = l.max_prefill.unwrap_or(self.max_prefill_cap);
        let mut notes = Vec::new();
        for (set, name) in [(l.prefix_slots, "prefix-slots"), (l.prefill_slice, "prefill-slice"), (l.prefill_batch, "prefill-batch")] {
            if set.is_some() {
                notes.push(format!("qwen3: the {name} setting has no meaning for this engine and is ignored"));
            }
        }
        Ok(notes)
    }
}

impl Default for QwenServeConfig {
    fn default() -> QwenServeConfig {
        QwenServeConfig { weights: None, tokenizer: None, ctx: None, auto_budget_bytes: None, card_budget_bytes: None, max_batch: 16, kv_int8: true, kv_fp8: false, kv_calib_opt_in: false, kv_offload_gb: 0.0, weights_int8: None, weights_bf16: false, max_prefill_cap: 512, kv_tokens: None, gemm_precision: gpu_core::provider::library::GemmPrecision::Auto, fuse_residual_norm: true, adapter_bank: None, adapter_targets_named: false, refused: None }
    }
}

/// Whether `checkpoint`'s linears are served int8: the operator's explicit choice, else
/// `qwen3::footprint::int8_weights_default` for the largest card's free memory (the one rule the SDK shares).
fn weights_int8_for(cfg: &QwenServeConfig, checkpoint: &qwen3::config::QwenConfig) -> bool {
    !cfg.weights_bf16 && cfg.weights_int8.unwrap_or_else(|| qwen3::footprint::int8_weights_default(checkpoint, cfg.auto_budget_bytes))
}

/// Device bytes of the weights at the tier `cfg` asks for.
fn serve_weight_bytes(cfg: &QwenServeConfig, checkpoint: &qwen3::config::QwenConfig) -> u64 {
    if cfg.weights_bf16 {
        weights_bf16_bytes(checkpoint)
    } else if weights_int8_for(cfg, checkpoint) {
        weights_int8_bytes(checkpoint)
    } else {
        weights_fp32_bytes(checkpoint)
    }
}

/// Context-length tiers [`QwenResident::resolve_ctx`]'s auto-sizing path
/// picks from, ascending - round numbers an operator recognizes in a log
/// line, not an arbitrary byte-granularity sweep result.
const AUTO_CTX_TIERS: &[u32] = &[2048, 4096, 8192, 16384, 24576, 32768, 40960, 49152, 65536, 98304, 131072, 196608, 262144];

/// The context length a checkpoint was pretrained on, the `original` of a YaRN
/// extension, from its declared `max_position_embeddings`: the largest power
/// of two not above it. Qwen3 declares 40960 for a model pretrained on 32768
/// (the extra 8192 is generation headroom); every other window this serves is
/// itself a power of two, which this leaves unchanged.
fn pretrained_window(max_position_embeddings: u32) -> u32 {
    1 << (u32::BITS - 1 - max_position_embeddings.max(1).leading_zeros())
}

/// Upper bound, in tokens, of the default pool's room for `max_batch`
/// half-context histories (see [`QwenResident::fleet_pool_tokens`]): 1M
/// tokens, about 74 GB of int8 KV on the 8B, far beyond any card's budget
/// beside the weights; it only keeps the arithmetic in range.
const MAX_DEFAULT_POOL_TOKENS: u32 = 1 << 20;

/// The default KV pool of a live server (see [`QwenResident::resolve_kv_tokens`]):
/// room for `max_batch` half-context histories, or the largest half of that
/// (repeatedly) whose whole footprint fits `budget` beside `weight_bytes`;
/// `None` keeps the two-context pool when not even the smallest does.
fn fleet_pool_tokens(checkpoint: &qwen3::config::QwenConfig, weight_bytes: u64, budget: u64, ctx: u32, cfg: &QwenServeConfig) -> Option<u32> {
    let max_batch = cfg.max_batch.max(1);
    let mut fleet = (u64::from(ctx) * u64::from(max_batch) / 2).min(u64::from(MAX_DEFAULT_POOL_TOKENS)) as u32;
    while fleet > ctx.saturating_mul(2) {
        let profile = PoolProfile::at_ctx(checkpoint, ctx, max_batch, cfg.max_prefill_cap.clamp(1, 512), Some(fleet), cfg.kv_int8, cfg.kv_offload_gb > 0.0, qwen3::serve::FusedAttention::default());
        // The pool grows only the KV buffers. The scratch binding, which the
        // unfused worst case of `PoolProfile` overstates at ctx 32768, does not
        // change with it, so only the KV binding is checked beside the total.
        let kv_binding = qwen3::serve::kv_binding_bytes(checkpoint, 16, fleet.div_ceil(16) + max_batch, cfg.kv_int8);
        if profile.fits_budget(weight_bytes, budget) && kv_binding <= WGPU_MAX_REPORTABLE_BINDING_BYTES {
            return Some(fleet);
        }
        fleet /= 2;
    }
    None
}

/// What one `(ctx, batch, prefill)` sizing costs on the device: the totals a
/// budget has to cover, and the largest SINGLE binding a driver has to accept.
///
/// One definition, because three used to exist. `auto_ctx_for_budget` summed
/// the totals, `check_fp32_kv_pool_fits` recomputed the KV pool, and
/// `check_buffers_fit_one_binding` recomputed both and divided out a
/// per-buffer worst case - each from the same underlying formulas, none
/// obliged to agree with the others. They didn't: the sizer accepted a ctx
/// whose per-layer KV binding the guard's own arithmetic would have refused,
/// which is how a serving lane ended up panicking inside `create_bind_group`.
/// Sharing the arithmetic makes that class of disagreement unrepresentable
/// rather than merely fixed.
struct PoolProfile {
    /// Whole KV pool: every layer's K and V, plus int8 scales.
    kv_bytes: u64,
    /// Paged-attention scratch (`scores` + `probs`).
    scratch_bytes: u64,
    /// Device staging for host-RAM KV offload (`0` with offload off): the
    /// buffers a preemption moves blocks through, allocated up front with the
    /// pool so that demoting a sequence can never need memory the budget did
    /// not count.
    swap_staging_bytes: u64,
    /// The largest single storage binding this sizing implies.
    ///
    /// The KV term is `qwen3::serve::kv_binding_bytes` (that crate owns the
    /// pool layout and its own doc explains the per-layer split);
    /// `scores`/`probs` are each exactly half of the scratch total.
    largest_binding: u64,
}

impl PoolProfile {
    fn new(cfg: &qwen3::config::QwenConfig, block_size: u32, num_blocks: u32, max_batch: u32, max_prefill: u32, cap: u32, kv_int8: bool, fused: qwen3::serve::FusedAttention) -> PoolProfile {
        let kv_bytes = qwen3::serve::kv_pool_bytes(cfg, block_size, num_blocks, kv_int8);
        let scratch_bytes = qwen3::serve::paged_attn_scratch_bytes(cfg, max_batch, max_prefill, cap, fused);
        // The per-layer KV figure comes from `qwen3::serve`, which owns the
        // pool layout -- the same function the engine itself checks against
        // the real device limit, so this pre-flight and that authoritative
        // check cannot disagree about the size of the buffer in question.
        let per_layer_kv = qwen3::serve::kv_binding_bytes(cfg, block_size, num_blocks, kv_int8);
        PoolProfile { kv_bytes, scratch_bytes, swap_staging_bytes: 0, largest_binding: per_layer_kv.max(scratch_bytes / 2) }
    }

    /// This sizing with the swap staging a host-RAM KV offload needs counted
    /// in `total`.
    fn with_swap_staging(mut self, cfg: &qwen3::config::QwenConfig, block_size: u32, num_blocks: u32, kv_int8: bool) -> PoolProfile {
        self.swap_staging_bytes = qwen3::serve::kv_swap_staging_bytes(cfg, block_size, num_blocks, kv_int8);
        self
    }

    /// The profile for a whole `ctx`, deriving the pool geometry itself;
    /// `offload` counts the swap staging of host-RAM KV offload.
    fn at_ctx(cfg: &qwen3::config::QwenConfig, ctx: u32, max_batch: u32, max_prefill_cap: u32, kv_tokens: Option<u32>, kv_int8: bool, offload: bool, fused: qwen3::serve::FusedAttention) -> PoolProfile {
        let (block_size, mb, max_blocks_per_seq, num_blocks, max_prefill) = QwenResident::pool_sizing(ctx, max_batch, max_prefill_cap, kv_tokens);
        let profile = PoolProfile::new(cfg, block_size, num_blocks, mb, max_prefill, max_blocks_per_seq * block_size, kv_int8, fused);
        if offload {
            profile.with_swap_staging(cfg, block_size, num_blocks, kv_int8)
        } else {
            profile
        }
    }

    /// This sizing as the global planner's categories (`model::memplan`): the one place the serving
    /// footprint is summed, so a budget decision, an estimate and a pre-load refusal cannot disagree.
    /// `weight_bytes` is the ctx-independent weight cost, `bank_bytes` the adapter bank beside it.
    fn demands(&self, weight_bytes: u64, bank_bytes: u64) -> Vec<model::memplan::Demand> {
        use model::memplan::{Category, Demand};
        vec![
            Demand::fixed(Category::Weights, "weights", weight_bytes),
            Demand::fixed(Category::Weights, "adapter bank", bank_bytes),
            Demand::fixed(Category::Kv, "paged pool", self.kv_bytes),
            Demand::fixed(Category::Activations, "attention scratch", self.scratch_bytes),
            Demand::fixed(Category::Transfers, "kv swap staging", self.swap_staging_bytes),
        ]
    }

    /// Whether the planner places this sizing beside `weight_bytes` within `budget_bytes` of device memory.
    fn fits_budget(&self, weight_bytes: u64, budget_bytes: u64) -> bool {
        self.plan(weight_bytes, 0, budget_bytes).is_ok()
    }

    /// The planner's verdict (and, refused, its arithmetic) for this sizing in `budget_bytes`.
    fn plan(&self, weight_bytes: u64, bank_bytes: u64, budget_bytes: u64) -> Result<model::memplan::PlanReport, model::memplan::PlanError> {
        model::memplan::plan(&self.demands(weight_bytes, bank_bytes), model::memplan::Capacity { device_bytes: budget_bytes, host_bytes: u64::MAX }, &model::memplan::Authorized::none())
    }

    /// Whether every single binding fits what a compliant device guarantees.
    fn bindings_fit(&self) -> bool {
        self.largest_binding <= WGPU_MAX_REPORTABLE_BINDING_BYTES
    }
}

/// The largest tier in [`AUTO_CTX_TIERS`] that both fits `budget_bytes` and
/// stays bindable, given `weight_bytes` is fixed (independent of `ctx`).
///
/// Auto-sizing never goes past the window the checkpoint declares
/// (`max_position_embeddings`; the smallest tier is always allowed): a context
/// beyond it is served with a derived YaRN scaling that rescales every
/// request, short ones included, so it is something an operator asks for with
/// `--ctx`, never something a large card hands out silently (knowledge
/// #338).
///
/// Fitting the budget was once the only test, which let this choose a ctx no
/// driver would bind - total VRAM and per-binding size are independent limits,
/// and a KV pool spread over many layers can clear the first while any single
/// layer's buffer breaks the second. Both are monotonically increasing in
/// `ctx`, so the first tier that fails either still ends the search.
///
/// `FusedAttention::default()` (nothing fused) is passed here on purpose: this runs before a
/// device is chosen, so the never-fused scratch worst case is the only honest
/// assumption. `activate()` re-checks with the real answer.
fn auto_ctx_for_budget(weight_bytes: u64, budget_bytes: u64, cfg: &qwen3::config::QwenConfig, max_batch: u32, max_prefill_cap: u32, kv_tokens: Option<u32>, kv_int8: bool, offload: bool) -> u32 {
    let mut best = AUTO_CTX_TIERS[0];
    for &ctx in AUTO_CTX_TIERS.iter().take_while(|&&t| t <= cfg.max_position_embeddings.max(AUTO_CTX_TIERS[0])) {
        let profile = PoolProfile::at_ctx(cfg, ctx, max_batch, max_prefill_cap, kv_tokens, kv_int8, offload, qwen3::serve::FusedAttention::default());
        if profile.fits_budget(weight_bytes, budget_bytes) && profile.bindings_fit() {
            best = ctx;
        } else {
            break;
        }
    }
    best
}

/// The Qwen3 BPE decoder behind the scheduler ([`QwenServeConfig`] names
/// WHICH checkpoint, `--checkpoint`, and HOW to serve it). Runs the CPU/GPU forward
/// `generate` path (never the NPU branch).
///
/// One resident is one copy of the weights, however many LoRA adapters ride
/// on it: each adapter is its own model id ([`QwenAdapterView`]) that shares
/// this resident's [`InstanceKey`], so base and adapter requests run in the
/// same batches on the same engine. See [`crate::adapter_registry`].
pub struct QwenResident {
    id: String,
    path: String,
    tokenizer: String,
    /// The adapters this base offers and the bank size they run on.
    adapters: Arc<AdapterRegistry>,
    /// Resolved once at construction (see [`Self::resolve_ctx`]) - explicit,
    /// auto-sized, or the historical 24576 fallback. Fixed for this
    /// resident's whole lifetime, exactly like every other field below.
    ctx: u32,
    max_batch: u32,
    kv_int8_requested: bool,
    kv_fp8_requested: bool,
    kv_calib_opt_in: bool,
    kv_offload_bytes: u64,
    weights_int8_requested: bool,
    weights_bf16_requested: bool,
    max_prefill_cap: u32,
    kv_tokens: Option<u32>,
    card_budget_bytes: Option<u64>,
    gemm_precision: gpu_core::provider::library::GemmPrecision,
    fuse_residual_norm: bool,
    refused: Option<String>,
}

/// What `--checkpoint` names, as the `(path to open, repo directory,
/// base id)` a [`QwenResident`] serves. A directory is what every other
/// weights variable takes, so accept one here too: `resolve_base` turns a repo
/// directory into the checkpoint inside it (for a Hugging Face directory its
/// `config.json`, which `qwen3::store_checkpoint_path` turns back into the
/// directory, the form `open_checkpoint` reads), and leaves a plain file path
/// alone. Unresolvable is not fatal - the path travels on verbatim so
/// `activate` reports it against the real open, which is where every other
/// bad path is reported.
fn resolve_served_base(spec: &str) -> (String, Option<std::path::PathBuf>, Option<String>) {
    match loader::model_dir::resolve_base(spec, None) {
        Ok((weights, dir, id)) => (qwen3::store_checkpoint_path(&weights, &dir).to_string_lossy().into_owned(), Some(dir), Some(id)),
        Err(_) => (spec.to_string(), None, None),
    }
}

impl AdapterHost for QwenResident {
    fn adapters(&self) -> &Arc<AdapterRegistry> {
        &self.adapters
    }

    fn register_view(self: &Arc<Self>, executor: &residency::Executor, adapter: LoadedAdapter) -> Result<(), String> {
        executor.register(Arc::new(self.adapter_view(adapter)?));
        Ok(())
    }
}

impl QwenResident {
    /// The resident `--checkpoint` selects, `None` when no checkpoint was
    /// named. `cfg` carries every serving-policy flag `brain serve` parsed.
    /// The tokenizer is `--tokenizer`, else the one beside the checkpoint.
    pub fn from_config(cfg: QwenServeConfig) -> Option<QwenResident> {
        let spec = cfg.weights.clone().filter(|p| !p.is_empty())?;

        let (path, dir, base_id) = resolve_served_base(&spec);

        let tokenizer = match cfg.tokenizer.clone().filter(|t| !t.is_empty()) {
            Some(t) => t,
            None => dir
                .map(|d| d.join("tokenizer.json"))
                .filter(|t| t.is_file())
                .map(|t| t.to_string_lossy().into_owned())
                .unwrap_or_default(),
        };

        // See GptResident::from_env's comment: named by path, no upstream
        // provenance. Adapters are bound to the id the spelling resolved to, the id a
        // trainer resolving the same spelling writes on an adapter's card.
        Some(Self::build(&path, "brain/qwen3".to_string(), base_id, &tokenizer, cfg))
    }

    /// [`Self::from_card_configured`] at every historical default -
    /// `QwenServeConfig::default()` reproduces the old env-var defaults
    /// exactly, so every existing caller that has no opinion about serving
    /// tunables (tests, `perf_cli.rs`, `model_dir.rs`'s catalog scan,
    /// `continuous_train.rs`) is unaffected by this constructor's own
    /// existence.
    pub fn from_card(path: &str, card: &ModelCard, tokenizer: Option<&str>) -> QwenResident {
        Self::from_card_configured(path, card, tokenizer, QwenServeConfig::default())
    }

    /// Construct under the card's id, which is also the id adapters trained
    /// on this base name as theirs. `tokenizer` is the sibling
    /// `tokenizer.json` (empty/None defers the "set a tokenizer" error to
    /// `activate`). `cfg` resolves `ctx` once, here (see
    /// [`Self::resolve_ctx`]), and is otherwise copied straight into fields
    /// `estimate`/`activate` read.
    pub fn from_card_configured(path: &str, card: &ModelCard, tokenizer: Option<&str>, cfg: QwenServeConfig) -> QwenResident {
        Self::build(path, card.id.clone(), Some(card.id.clone()), tokenizer.unwrap_or_default(), cfg)
    }

    fn build(path: &str, id: String, base_id: Option<String>, tokenizer: &str, cfg: QwenServeConfig) -> QwenResident {
        let ctx = Self::resolve_ctx(path, &cfg);
        // A checkpoint that cannot be read yet keeps the operator's explicit choice or fp32; `activate`
        // reports the real error.
        let weights_int8_requested = match qwen3::checkpoint_config(path) {
            Ok(checkpoint) => weights_int8_for(&cfg, &checkpoint),
            Err(_) => !cfg.weights_bf16 && cfg.weights_int8.unwrap_or(false),
        };
        let kv_tokens = Self::resolve_kv_tokens(path, &cfg, ctx);
        QwenResident {
            id,
            path: path.to_string(),
            tokenizer: tokenizer.to_string(),
            adapters: AdapterRegistry::new(cfg.adapter_bank, ServedBase::new(path, base_id)),
            ctx,
            max_batch: cfg.max_batch.max(1),
            kv_int8_requested: cfg.kv_int8,
            kv_fp8_requested: cfg.kv_fp8,
            kv_calib_opt_in: cfg.kv_calib_opt_in,
            kv_offload_bytes: (cfg.kv_offload_gb.max(0.0) * (1u64 << 30) as f64) as u64,
            weights_int8_requested,
            weights_bf16_requested: cfg.weights_bf16,
            max_prefill_cap: cfg.max_prefill_cap.clamp(1, 512),
            kv_tokens,
            card_budget_bytes: cfg.card_budget_bytes.or(cfg.auto_budget_bytes),
            gemm_precision: cfg.gemm_precision,
            fuse_residual_norm: cfg.fuse_residual_norm,
            refused: cfg.refused.clone(),
        }
    }

    /// The KV pool size in tokens: `the KV pool size` verbatim when named;
    /// otherwise, on a live server (a device budget is known), room for
    /// `max_batch` half-context histories - capped at
    /// [`MAX_DEFAULT_POOL_TOKENS`] - when that pool fits the budget beside
    /// the weights, so a fleet of agents resending their histories finds each
    /// one cached at its next turn without a flag. `None` keeps the
    /// two-context pool: no budget known, nothing to add beyond two contexts,
    /// or the larger pool would not fit.
    fn resolve_kv_tokens(path: &str, cfg: &QwenServeConfig, ctx: u32) -> Option<u32> {
        if cfg.kv_tokens.is_some() {
            return cfg.kv_tokens;
        }
        let budget = cfg.card_budget_bytes.or(cfg.auto_budget_bytes)?;
        let checkpoint_cfg = qwen3::checkpoint_config(path).ok()?;
        let weight_bytes = serve_weight_bytes(cfg, &checkpoint_cfg);
        fleet_pool_tokens(&checkpoint_cfg, weight_bytes, budget, ctx, cfg)
    }

    /// `cfg.ctx` verbatim when the operator named one explicitly
    /// (`--ctx`); else, when `cfg.auto_budget_bytes` names a real
    /// device budget (only `brain serve`'s own caller ever does), the
    /// largest [`AUTO_CTX_TIERS`] entry whose real total fits it, capped at the
    /// checkpoint's own `max_position_embeddings` (serving past it is an
    /// explicit `--ctx`, which [`apply_yarn_if_serving_past_native`],
    /// called from `activate`, backs with YaRN); else the historical 24576 default, unchanged. Must never
    /// itself hard-fail - a checkpoint this can't open just falls back to
    /// the historical default and defers the real, specific error to
    /// `activate`'s own open, exactly like `estimate` already does.
    fn resolve_ctx(path: &str, cfg: &QwenServeConfig) -> u32 {
        const HISTORICAL_DEFAULT: u32 = 24576;
        if let Some(explicit) = cfg.ctx {
            return explicit.max(1);
        }
        let Some(budget) = cfg.auto_budget_bytes else { return HISTORICAL_DEFAULT };
        let Ok(checkpoint_cfg) = qwen3::checkpoint_config(path) else { return HISTORICAL_DEFAULT };
        let weight_bytes = serve_weight_bytes(cfg, &checkpoint_cfg);
        auto_ctx_for_budget(weight_bytes, budget, &checkpoint_cfg, cfg.max_batch.max(1), cfg.max_prefill_cap.clamp(1, 512), cfg.kv_tokens, cfg.kv_int8, cfg.kv_offload_gb > 0.0)
    }

    /// Give `cfg` a derived [`model::yarn::YarnConfig`] when `ctx` (the KV
    /// pool `activate` is actually about to build - `self.ctx`, whether from
    /// `--ctx` or auto-sizing) exceeds the checkpoint's declared
    /// `max_position_embeddings` and nothing already named a scaling. A
    /// checkpoint that already declares its own `rope_scaling` (a real
    /// long-context release) is never second-guessed.
    ///
    /// The scaling is Qwen's published recipe (`rope_type: yarn`, `factor:
    /// 4.0`, `original_max_position_embeddings: 32768` for 131072 tokens on
    /// Qwen3): `original` is the window the model was PRETRAINED on
    /// ([`pretrained_window`]), not the declared `max_position_embeddings`
    /// (Qwen3 declares 40960 = 32768 + 8192 of generation headroom), and
    /// `factor` is `ctx / original`, so `--ctx 131072` on a Qwen3 is
    /// exactly the recipe. Deriving against 40960 instead gave factor 3.2 and
    /// a different correction range (knowledge #338).
    ///
    /// A DELIBERATE, resident-level decision, not something `qwen3::serve`
    /// derives on its own from its pool-sizing parameters: an `Engine`
    /// constructed directly (a test fixture, `qwen_bench`, `perf_cli`) sizes
    /// its pool however is numerically convenient for THAT caller, with no
    /// relation to "please extend this checkpoint's real context" - inferring
    /// intent from bare pool geometry there would auto-opt every such caller
    /// into YaRN by accident. Only `activate`'s own `ctx` - `resolve_ctx`'s
    /// real output, `--ctx`/auto-sizing's own answer - means that.
    fn apply_yarn_if_serving_past_native(cfg: &mut qwen3::config::QwenConfig, ctx: u32) {
        if cfg.rope_scaling.is_some() || ctx <= cfg.max_position_embeddings.max(1) {
            return;
        }
        let original = pretrained_window(cfg.max_position_embeddings);
        cfg.rope_scaling = Some(model::rope_scaling::RopeScaling::Yarn(model::yarn::YarnConfig::new(ctx as f32 / original as f32, original)));
    }

    /// The adapters this base offers.
    pub fn adapters(&self) -> &Arc<AdapterRegistry> {
        &self.adapters
    }

    /// The base checkpoint adapters are bound to, for
    /// `crate::adapter_release` to verify an adapter against before it is
    /// offered.
    /// The checkpoint this resident serves: two residents with the same path
    /// are one set of weights and must be one instance.
    pub fn weights_path(&self) -> &str {
        &self.path
    }

    pub fn served_base(&self) -> ServedBase {
        self.adapters.base().clone()
    }

    /// Offer `adapter` as a model of its own and return the view to register
    /// with the executor. The view shares this resident's instance: nothing
    /// is built, and the adapter reaches the engine's bank when the instance
    /// next starts a batch. Dropping the view withdraws the adapter.
    ///
    /// # Errors
    ///
    /// The adapter's id is not an adapter model reference, or the bank this
    /// server was started with cannot hold the adapter
    /// ([`AdapterRegistry::attach`] says which flag to change).
    pub fn adapter_view(self: &Arc<Self>, adapter: LoadedAdapter) -> Result<QwenAdapterView, String> {
        let lease = self.adapters.attach(adapter)?;
        Ok(QwenAdapterView::new(self.clone(), lease))
    }

    /// KV-pool geometry for the batched serving engine - the ONE place
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
    /// ExceedsCapacity`) instead of being pre-reserved for at an order of
    /// magnitude more memory in the common case where that never happens.
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
    /// A free function of its inputs (no `self`, no I/O) - both a test and
    /// [`auto_ctx_for_budget`]'s own sweep can drive it directly, and
    /// `estimate`/`activate` pass `self.max_batch`/`self.max_prefill_cap`
    /// explicitly so the two can never silently disagree about which
    /// resident's tuning they're sizing for. `max_prefill_cap`: see
    /// [`QwenServeConfig::max_prefill_cap`]'s own doc for why lowering it
    /// shrinks `paged_attn_scratch_bytes`'s pre-placement (never-under)
    /// estimate for free, and why the clamp only ever SHRINKS the historical
    /// 512 default, never raises it past the value the comment below records
    /// once caused a real wgpu buffer-size crash. On a device that runs the
    /// fused prefill kernel the real allocation is `max_batch`-sized and does
    /// not depend on `max_prefill` at all - `lowering_max_prefill_shrinks_
    /// scratch_linearly_at_the_real_config`'s own doc has the split.
    ///
    /// `kv_tokens`: the pool's size in tokens when the operator named one
    /// (`the KV pool size`); `None` is two contexts (a live server
    /// resolves a larger default first, see [`QwenResident::fleet_kv_tokens`]).
    /// A pool is never smaller than one full context, so any admitted request
    /// can run alone.
    fn pool_sizing(ctx: u32, max_batch: u32, max_prefill_cap: u32, kv_tokens: Option<u32>) -> (u32, u32, u32, u32, u32) {
        let block_size = 16u32;
        let max_blocks_per_seq = ctx.div_ceil(block_size);
        let num_blocks = match kv_tokens {
            Some(tokens) => tokens.div_ceil(block_size).max(max_blocks_per_seq) + max_batch,
            None => max_blocks_per_seq * 2 + max_batch,
        };
        let max_prefill = ctx.min(max_prefill_cap);
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
    fn check_fp32_kv_pool_fits(cfg: &qwen3::config::QwenConfig, block_size: u32, num_blocks: u32, ctx: u32, ceiling: u64, path: &str) -> Result<(), String> {
        let pool_bytes = PoolProfile::new(cfg, block_size, num_blocks, 1, 1, block_size, false, qwen3::serve::FusedAttention::default()).kv_bytes;
        if pool_bytes <= ceiling {
            return Ok(());
        }
        let int8_bytes = qwen3::serve::kv_pool_bytes(cfg, block_size, num_blocks, true);
        Err(format!(
            "qwen: {path}: fp32 KV pool at ctx={ctx} would be {:.2} GiB, over the {:.0} GiB safety ceiling \
             -- lower --ctx, or drop --kv-cache f32 \
             so int8 KV (~{:.2} GiB at this ctx) is used instead",
            pool_bytes as f64 / (1u64 << 30) as f64,
            ceiling as f64 / (1u64 << 30) as f64,
            int8_bytes as f64 / (1u64 << 30) as f64,
        ))
    }

    /// The device bytes of the weights this resident will load, for the tier it requests:
    /// `--weight-format int8` reads them off `cfg` (the same canonical-name derivation
    /// `decoder_param_list` builds the engine from), so the estimate and `activate()`
    /// cannot disagree about which tensors are quantized.
    fn weight_bytes(&self, cfg: &qwen3::config::QwenConfig) -> u64 {
        if self.weights_bf16_requested {
            weights_bf16_bytes(cfg)
        } else if self.weights_int8_requested {
            weights_int8_bytes(cfg)
        } else {
            weights_fp32_bytes(cfg)
        }
    }

    /// `Err` naming the checkpoint, the requested `ctx`/`max_prefill` and the
    /// worst single device buffer this sizing would allocate, when that
    /// buffer would exceed WebGPU's own spec-mandated
    /// `maxStorageBufferBindingSize` floor -- called from `activate()`
    /// BEFORE any device allocation, for BOTH the int8 and fp32 KV paths
    /// (unlike [`Self::check_fp32_kv_pool_fits`], which only ever runs on
    /// the fp32 branch and checks the pool's TOTAL against a much smaller
    /// sanity ceiling, not any one buffer's real size).
    ///
    /// The int8 (default) path had NO equivalent guard at all: a context
    /// large enough to push one buffer past this ceiling reached wgpu's own
    /// uncaptured-error panic directly (the exact crash class
    /// `check_fp32_kv_pool_fits`'s own doc comment already records once
    /// happening for real, on the fp32 path, before that guard existed) --
    /// this closes the same gap for the path everybody actually runs.
    ///
    /// Checked against the WebGPU FLOOR, not this box's own (possibly
    /// larger) queried limit: no `DeviceCaps` query happens here, so a bound
    /// every compliant device guarantees is the only one that is safe to
    /// assume without one - never-under (for the KV term; see
    /// the `fused` argument's note on the scratch term), the same
    /// discipline every other pre-flight number in this file already
    /// follows.
    ///
    /// `fused`: unlike `estimate()` (truly pre-placement,
    /// no device chosen yet, always nothing fused), `activate()` already has a
    /// concrete `device` by the time it calls this - `device != Device::Cpu`
    /// is a safe proxy for the fused kernel's real gate
    /// (`caps.workgroup_reductions`, true on every GPU backend, false only
    /// on the CPU JIT), so passing nothing-fused here unconditionally caused a
    /// real, oversized-sounding refusal for a buffer the engine would never
    /// actually allocate once the fused kernel shrinks the scratch term.
    ///
    /// `Engine::from_map_with_gpu` allocates one K buffer and one V buffer
    /// PER LAYER (never one combined buffer for the whole model), so the
    /// worst per-layer K/V buffer is `kv_pool_bytes / n_layers / 2` -
    /// slightly an OVER-estimate (it also divides in the much smaller
    /// per-slot int8 scale buffers), which only makes this check stricter,
    /// never looser. `scores`/`probs` are each exactly half of
    /// `paged_attn_scratch_bytes`'s combined total.
    fn check_buffers_fit_one_binding(
        cfg: &qwen3::config::QwenConfig,
        block_size: u32,
        num_blocks: u32,
        max_batch: u32,
        max_prefill: u32,
        cap: u32,
        kv_int8: bool,
        ctx: u32,
        path: &str,
        fused: qwen3::serve::FusedAttention,
    ) -> Result<(), String> {
        let profile = PoolProfile::new(cfg, block_size, num_blocks, max_batch, max_prefill, cap, kv_int8, fused);
        if profile.bindings_fit() {
            return Ok(());
        }
        Err(format!(
            "qwen: {path}: at ctx={ctx} (kv_int8={kv_int8}) the largest single device buffer this engine \
             would allocate is {:.0} MiB, over the {:.0} MiB no device can exceed -- lower \
             --ctx or the prefill row cap",
            profile.largest_binding as f64 / (1u64 << 20) as f64,
            WGPU_MAX_REPORTABLE_BINDING_BYTES as f64 / (1u64 << 20) as f64,
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
            .with_max_context_tokens(self.ctx as u64)
    }
    /// One key for the base and every adapter riding on it: a constant, because
    /// the instance holds the base weights once and the adapters beside them
    /// in a bank, so a request for any of them runs on it, in the same batch as
    /// the others ([`QwenAdapterView`] answers with this same key).
    fn batch_limit(&self) -> Option<usize> {
        Some(self.max_batch as usize)
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(self.id.as_str(), "default")
    }
    /// A caller reaching the base never selects an adapter: the reserved
    /// param is the host's, and only a [`QwenAdapterView`] writes it.
    fn stamp(&self, inv: &mut Invocation) {
        if let Some(params) = inv.params.as_object_mut() {
            params.remove(ADAPTER_PARAM);
        }
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        let cost = est_vram(&self.path);
        // Any failure here just defers to the real, specific error
        // `activate()` raises -- `estimate()` must never itself hard-fail.
        let Ok(cfg) = qwen3::checkpoint_config(&self.path) else {
            return cost;
        };
        let (block_size, max_batch, max_blocks_per_seq, num_blocks, max_prefill) = Self::pool_sizing(self.ctx, self.max_batch, self.max_prefill_cap, self.kv_tokens);
        let kv_int8 = self.kv_int8_requested && qwen3::serve::kv_int8_supported(&cfg);
        // The paged-attention scores/probs scratch (`Scratch::{scores,probs}`) is a real, often
        // multi-GiB device allocation `activate()` makes alongside the KV pool. The fused-attention
        // answer needs the target device's own `DeviceCaps`, unknown at this pre-placement point, so
        // `FusedAttention::default()` (nothing fused) is the larger size `paged_attn_scratch_bytes`
        // can return: a worst-case, never-under bound.
        let cap = max_blocks_per_seq * block_size;
        let weight_bytes = self.weight_bytes(&cfg);
        // The adapter bank is allocated beside the weights at activation, sized by the same settings,
        // over the larger of the decode batch and the prefill chunk.
        let bank_bytes = self.adapters.settings().map_or(0, |bank| bank.device_bytes(&cfg, max_batch.max(max_prefill)));
        // Host-RAM KV offload moves blocks through device staging buffers the engine allocates when
        // offload is enabled; they come out of the same budget as the pool.
        let profile = PoolProfile::new(&cfg, block_size, num_blocks, max_batch, max_prefill, cap, kv_int8, qwen3::serve::FusedAttention::default());
        let profile = if self.kv_offload_bytes > 0 { profile.with_swap_staging(&cfg, block_size, num_blocks, kv_int8) } else { profile };
        let demands = profile.demands(weight_bytes, bank_bytes);
        residency::log::info(&format!("{}: estimate {}", self.id, model::memplan::summary(&demands)));
        MemCost::new(model::memplan::device_total(&demands), cost.ram)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // Coarse stage progress -- NOT per-tensor (the actual weight upload
        // loop lives in `paramstore`, a lower-level crate that must not
        // depend on `residency` just for this), a handful of stage markers so
        // `-v -v` shows SOMETHING moving during a cold activate that can take
        // over a minute, without turning into a per-layer scroll.
        if let Some(why) = &self.refused {
            return Err(why.clone());
        }
        let stage_t0 = std::time::Instant::now();
        residency::log::info(&format!("{}: step 1/3 opening checkpoint", self.id));
        // The checkpoint is read as it is on disk (a Hugging Face directory, a
        // GGUF or a brain file); only a GGUF carries its own tokenizer.
        let is_gguf = self.path.to_ascii_lowercase().ends_with(".gguf");
        let gguf_tok = if is_gguf { checkpoint::weightio::WeightReader::open(&self.path).map_err(|e| format!("qwen: {e}"))?.tokenizer() } else { None };
        gpu_core::profile::stage_time(&format!("{}: open checkpoint", self.id), stage_t0);
        let stage_t0 = std::time::Instant::now();
        residency::log::info(&format!("{}: step 2/3 loading tokenizer", self.id));
        // Tokenizer precedence: an explicit sibling `tokenizer.json` (safetensors
        // path, or an override) wins; else a `.gguf` builds from its embedded
        // `tokenizer.ggml.*` KV; else there is nothing to tokenize with.
        let (tok, tok_dir, gguf_eos) = if !self.tokenizer.is_empty() {
            (data::qwen_tokenizer::QwenBpe::from_file(&self.tokenizer)?, std::path::Path::new(&self.tokenizer).parent(), None)
        } else if let Some(gt) = gguf_tok {
            (data::qwen_tokenizer::QwenBpe::from_gguf(&gt).map_err(|e| format!("qwen: {e}"))?, None, gt.eos)
        } else {
            return Err("qwen: no tokenizer (put tokenizer.json beside the checkpoint, pass --tokenizer, or use a GGUF with an embedded tokenizer)".to_string());
        };
        // Prompts render in the checkpoint's own chat format; generation stops
        // on its eos ids and on that format's end-of-turn marker.
        let format = ChatFormat::for_checkpoint(tok_dir, &tok);
        let stop = data::generation::stop_ids(tok_dir, &tok, gguf_eos, format.end_of_turn()).map_err(|e| format!("qwen: {e}"))?;
        gpu_core::profile::stage_time(&format!("{}: load tokenizer", self.id), stage_t0);
        let stage_t0 = std::time::Instant::now();
        residency::log::info(&format!("{}: step 3/3 building engine (uploading weights to {device:?})", self.id));
        let ctx = self.ctx;
        // Every checkpoint - a Hugging Face directory, a GGUF or a brain file -
        // gets the paged, continuous-batching engine, which holds f32 or int8
        // weights and, when the server was started with an adapter bank, the
        // LoRA adapters it offers beside them.
        let engine = on_device(device, || -> Result<Box<model::serve::Scheduler<qwen3::serve::Engine>>, String> {
            // See QwenResident::pool_sizing's doc comment for the arithmetic
            // -- the same derivation `estimate()` predicts a budget from, so
            // the two cannot silently drift apart.
            let (block_size, max_batch, max_blocks_per_seq, num_blocks, max_prefill) = QwenResident::pool_sizing(ctx, self.max_batch, self.max_prefill_cap, self.kv_tokens);
            // A DEFAULT-selecting caller degrades loudly rather than hitting
            // `Engine::from_map_with_gpu`'s hard assert: nobody explicitly
            // asked this checkpoint's unusual `head_dim` for int8, so a
            // serving-process panic on activation would be the wrong failure
            // mode -- see `qwen3::serve::kv_int8_supported`'s doc comment.
            let checkpoint_cfg = qwen3::checkpoint_config(&self.path).map_err(|e| format!("qwen: {e}"))?;
            let kv_int8 = self.kv_int8_requested && qwen3::serve::kv_int8_supported(&checkpoint_cfg);
            if self.kv_int8_requested && !kv_int8 {
                eprintln!(
                    "serve: {}: int8 KV requested (the default) but head_dim={} is not a multiple of 4; falling back to fp32 KV",
                    self.path, checkpoint_cfg.head_dim
                );
            }
            // Boundary guard: refuse a fp32 KV pool over the safety ceiling,
            // loudly and specifically, BEFORE Engine::from_map_with_gpu
            // attempts the device allocation. Only reachable via an explicit
            // opt-out (--kv-cache f32) or an unsupported head_dim -- the
            // int8 default at the SAME ctx does not come close.
            if !kv_int8 {
                QwenResident::check_fp32_kv_pool_fits(&checkpoint_cfg, block_size, num_blocks, ctx, self.card_budget_bytes.unwrap_or(0).max(MAX_FP32_KV_POOL_BYTES), &self.path)?;
            }
            // Runs on BOTH branches (unlike the fp32-only guard above): a
            // per-buffer overflow is a different failure mode than the
            // fp32 pool's total-size sanity ceiling, and the int8 (default)
            // path had no equivalent guard at all before this existed.
            let cap = max_blocks_per_seq * block_size;
            // `device` is already concrete here (unlike `estimate()`'s truly
            // pre-placement call): every GPU backend has workgroup
            // reductions (only the CPU JIT lacks them, the same gate
            // `Op::PagedAttentionFused` itself checks), so this predicts the
            // real dispatch instead of assuming the never-fused worst case.
            let fused = if device == Device::Cpu { qwen3::serve::FusedAttention::default() } else { qwen3::serve::fused_attention_on_gpu(checkpoint_cfg.head_dim, kv_int8) };
            QwenResident::check_buffers_fit_one_binding(
                &checkpoint_cfg,
                block_size,
                num_blocks,
                max_batch,
                max_prefill,
                cap,
                kv_int8,
                ctx,
                &self.path,
                fused,
            )?;
            // Refuse BEFORE loading when the whole footprint (weights, adapter bank, KV pool, attention
            // scratch, swap staging) cannot fit the card this server was told it may use, with the
            // planner's per-category arithmetic. `estimate()` charged the same demands for placement.
            if let Some(card) = self.card_budget_bytes.filter(|&b| b > 0) {
                let profile = PoolProfile::new(&checkpoint_cfg, block_size, num_blocks, max_batch, max_prefill, cap, kv_int8, fused);
                let profile = if self.kv_offload_bytes > 0 { profile.with_swap_staging(&checkpoint_cfg, block_size, num_blocks, kv_int8) } else { profile };
                let bank_bytes = self.adapters.settings().map_or(0, |bank| bank.device_bytes(&checkpoint_cfg, max_batch.max(max_prefill)));
                profile.plan(self.weight_bytes(&checkpoint_cfg), bank_bytes, card).map_err(|e| format!("qwen: {}: refusing to load, {e}; lower --ctx, --max-batch or the KV pool size", self.path))?;
            }
            // `--weight-format int8` -- requested, not asserted: the real
            // capability gate (`caps.numeric.int8_dot`) lives inside
            // `Engine::from_map_with_gpu` itself, which degrades to fp32
            // weights with its own loud fallback message when a device (the
            // CPU JIT, or an unusual GPU) has no packed-int8 path, exactly
            // the same shape `kv_int8`'s own request/degrade split already
            // uses. `estimate()` above computes its budget from this SAME
            // field, so the two cannot silently disagree about what was
            // asked for.
            let weights_int8 = self.weights_int8_requested;
            // `Engine::load` re-derives its own `QwenConfig` straight from the
            // file, with no seam for `ctx`'s own YaRN decision below to reach
            // it - open the checkpoint here and build from its source, so the config is
            // mutated in one place. The engine reads each tensor as it uploads (a packed
            // source streams into the tier tile by tile), never a whole-model host map.
            let load_t0 = std::time::Instant::now();
            let (_, src) = qwen3::open_checkpoint(&self.path).map_err(|e| format!("qwen: {e}"))?;
            gpu_core::profile::stage_time(&format!("{}: open checkpoint source for engine build", self.id), load_t0);
            let mut cfg = checkpoint_cfg.clone();
            Self::apply_yarn_if_serving_past_native(&mut cfg, ctx);
            let tier = if self.weights_bf16_requested {
                qwen3::Dtype::BF16
            } else if weights_int8 {
                qwen3::Dtype::I8
            } else {
                qwen3::Dtype::F32
            };
            let mut eng = qwen3::serve::Engine::from_source_tier(cfg, &*src, block_size, num_blocks, max_batch, max_blocks_per_seq, max_prefill, kv_int8, tier);
            if self.kv_fp8_requested {
                if kv_int8 {
                    eng.set_kv_fp8().map_err(|e| format!("qwen: --kv-cache fp8: {e}"))?;
                } else {
                    eprintln!("serve: {}: --kv-cache fp8 needs the packed KV pools but the KV cache is fp32; ignoring", self.path);
                }
            }
            eng.set_gemm_precision(self.gemm_precision);
            eng.set_fuse_residual_norm(self.fuse_residual_norm);
            // The adapter bank, once per engine, from the settings the
            // estimate above budgeted. Refusing here is deliberate: a server
            // asked for adapters that cannot give them must say so, not serve
            // the base under their names.
            if let Some(bank) = self.adapters.settings() {
                eng.enable_adapters(bank.bank_config(&checkpoint_cfg)?).map_err(|e| format!("qwen: adapter bank: {e}"))?;
            }
            // --kv-calibration: opt IN to a kv_calib.json beside the
            // BASE checkpoint (measured on the base's K/V distribution, which
            // an adapter's differs from: the engine warns when the two are
            // combined) --
            // KvCalib::from_model_dir already warns and returns None on a
            // missing file or a shape mismatch, so opting in without a real
            // file just serves uncalibrated, same as not opting in.
            if kv_int8 && self.kv_calib_opt_in {
                if let Some(dir) = std::path::Path::new(&self.path).parent() {
                    let calib = model::kvcalib::KvCalib::from_model_dir(dir, checkpoint_cfg.n_layers as usize, checkpoint_cfg.n_kv_heads as usize, checkpoint_cfg.head_dim as usize);
                    eng.set_kv_calib(calib);
                }
            }
            // Host-RAM KV offload (`model::kv_offload`): with a pool sized for
            // about two full contexts (see `pool_sizing`) but a batch of
            // `--max-batch` slots, a busy server can admit far more
            // sessions than the pool can hold cached at once. Without host RAM
            // the scheduler preempts by recompute; given host RAM, it parks the
            // sessions it is not advancing and brings them back byte-identical,
            // which skips the re-prefill. Off by default (0): it spends host
            // memory, and a box that has none to spare must not be made to.
            // Enabling it builds the device staging `estimate()` budgeted.
            eng.set_kv_offload_bytes(self.kv_offload_bytes);
            Ok(Box::new(model::serve::Scheduler::new(eng, max_batch as usize)))
        })??;
        gpu_core::profile::stage_time(&format!("{}: build engine (total, on_device)", self.id), stage_t0);
        let mut instance = QwenInstance { max_batch: Self::pool_sizing(self.ctx, self.max_batch, self.max_prefill_cap, self.kv_tokens).1 as usize, decoding: Decoding { tok, format, stop }, engine, adapters: AdapterSync::new(self.adapters.clone()) };
        // Every adapter offered so far is registered with the engine before
        // the first request, so a cold instance answers an adapter request
        // as soon as it answers a base one.
        instance.adapters.sync(instance.engine.decoder_mut());
        Ok(Box::new(instance))
    }
}

/// What turns requests into prompts and tokens back into text.
struct Decoding {
    tok: data::qwen_tokenizer::QwenBpe,
    /// How requests become prompt text (`ChatFormat`).
    format: ChatFormat,
    /// The ids that end a generation (`data::generation::stop_ids`).
    stop: Vec<u32>,
}

struct QwenInstance {
    /// Requests the scheduler runs at once (an open group admits up to this).
    max_batch: usize,
    decoding: Decoding,
    /// The paged, continuous-batching serving engine, whatever the
    /// checkpoint's format. Boxed: `Scheduler<Engine>` is large by value.
    engine: Box<model::serve::Scheduler<qwen3::serve::Engine>>,
    /// Which adapters the engine's bank holds, and the route to each.
    adapters: AdapterSync,
}

impl Instance for QwenInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.run_batch(action, std::slice::from_ref(inv), &mut |_i, p| progress(p)).pop().unwrap()
    }

    /// Every invocation in `invs` is submitted into the SAME
    /// persistent `Scheduler` (built once at `activate`, so the paged KV pool
    /// and prefix cache are shared and reused across calls, not rebuilt) and
    /// driven to completion together - real continuous batching for
    /// whatever the dispatcher grouped into this one call (admitting MORE
    /// work into an ALREADY-running call is a separate, known gap this does
    /// not yet do).
    ///
    /// The group may mix the base and any number of adapters: a request's
    /// adapter is the reserved param its model id stamped
    /// ([`ADAPTER_PARAM`]), and the engine applies each row's own adapter in
    /// the one batched forward. The engine's bank is brought in line with the
    /// registry first, between calls, when no sequence is running, so a hot
    /// swap never changes what a running sequence reads.
    fn run_batch(&mut self, _action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        self.adapters.sync(self.engine.decoder_mut());
        run_batch_scheduled(&mut self.engine, &QwenFrontend { decoding: &self.decoding, adapters: &self.adapters }, invs, progress)
    }

    /// The same group as [`Self::run_batch`], open to requests that arrive
    /// while it runs: a long generation no longer holds back a short one that
    /// came after it. The adapter bank is brought in line with the registry
    /// when the group starts (no sequence is running then); an adapter offered
    /// while a group runs is routable from the next group, a request naming it
    /// earlier is answered "no model".
    fn run_open(&mut self, _action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress), feed: &mut dyn residency::JobFeed) -> bool {
        self.adapters.sync(self.engine.decoder_mut());
        let capacity = self.max_batch;
        let mut invs = invs;
        // A group that idles while the registry has moved hands back: refresh the
        // bank and routes, then go on with the jobs still queued on the feed.
        let mut numbered = 0;
        while let OpenGroup::Refresh { numbered: n } = run_open_scheduled(&mut self.engine, &QwenFrontend { decoding: &self.decoding, adapters: &self.adapters }, invs, capacity, numbered, progress, feed) {
            self.adapters.sync(self.engine.decoder_mut());
            (invs, numbered) = (&[], n);
        }
        true
    }

    /// The prefix cache's effectiveness, surfaced through
    /// `Executor::stats().metrics` (reachable from HTTP/D-Bus).
    fn metrics(&self) -> Vec<(String, serde_json::Value)> {
        scheduler_metrics(&self.engine)
    }
}

/// Qwen3's request front end: its own chat format, and the adapter a
/// request's model id stamped (see [`ADAPTER_PARAM`]).
struct QwenFrontend<'a> {
    decoding: &'a Decoding,
    adapters: &'a AdapterSync,
}

impl Frontend for QwenFrontend<'_> {
    fn tok(&self) -> &data::qwen_tokenizer::QwenBpe {
        &self.decoding.tok
    }
    fn stop(&self) -> &[u32] {
        &self.decoding.stop
    }
    fn label(&self) -> &str {
        "Qwen"
    }
    fn parse(&self, inv: &Invocation) -> Result<qwen3::chat::ParsedRequest, String> {
        parse_request_as(&self.decoding.tok, &self.decoding.format, inv)
    }
    fn route(&self, inv: &Invocation) -> Result<Option<(model::serve::AdapterId, residency::AdapterIdentity)>, String> {
        inv.get_str(ADAPTER_PARAM).map(|name| self.adapters.route(&name)).transpose()
    }
    fn stale(&self) -> bool {
        self.adapters.stale()
    }
    fn base_digest(&self) -> Option<String> {
        self.adapters.base_digest()
    }
    fn rejection(&self, inv: &Invocation, reason: &model::serve::RejectReason) -> String {
        match (reason, inv.get_str(ADAPTER_PARAM)) {
            // An adapter the engine unloaded between routing and admission:
            // as an unknown model, with no engine detail.
            (model::serve::RejectReason::AdapterUnavailable { .. }, Some(name)) => format!("no model '{name}'"),
            _ => format!("qwen: {reason}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_serving_policy_maps_onto_the_qwen3_engine_and_refusals_name_it() {
        let mut p = model::policy::ServePolicy::default();
        p.set("--weight-format", "bf16").unwrap();
        p.set("--kv-cache", "fp8").unwrap();
        p.set("--gemm", "fp32").unwrap();
        p.set("--ctx", "4096").unwrap();
        p.set("--max-batch", "32").unwrap();
        p.set("--kv-offload-gb", "2").unwrap();
        p.set("prefix-slots", "16").unwrap();
        let mut cfg = QwenServeConfig::default();
        let notes = cfg.apply_policy(&p).unwrap();
        assert!(cfg.weights_bf16 && cfg.kv_int8 && cfg.kv_fp8);
        assert_eq!((cfg.ctx, cfg.max_batch, cfg.kv_offload_gb), (Some(4096), 32, 2.0));
        assert_eq!(cfg.gemm_precision, gpu_core::provider::library::GemmPrecision::Fp32Reference);
        assert!(notes.iter().any(|n| n.contains("prefix-slots")), "{notes:?}");

        let mut p = model::policy::ServePolicy::default();
        p.set("--kv-cache", "bf16").unwrap();
        let mut cfg = QwenServeConfig::default();
        let e = cfg.apply_policy(&p).unwrap_err();
        assert!(e.contains("qwen3") && e.contains("f32, int8, fp8"), "{e}");
        assert_eq!(cfg.refused.as_deref(), Some(e.as_str()));
    }

    /// `--checkpoint` naming a Hugging Face repo directory serves the
    /// directory itself, the form `open_checkpoint` reads (its `config.json`
    /// holds the architecture); a carded brain checkpoint serves as its
    /// file, and a path that resolves to nothing travels on verbatim.
    #[test]
    fn a_served_hugging_face_directory_is_opened_as_the_directory() {
        let root = std::env::temp_dir().join(format!("catalog-served-base-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let hf = root.join("deepseek-ai").join("DeepSeek-R1-Distill-Qwen-1.5B");
        let native = root.join("Qwen").join("Qwen3-0.6B");
        for dir in [&hf, &native] {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(dir.join("config.json"), br#"{"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2"}"#).unwrap();
            std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        }
        // A complete shard set: a real (tiny) safetensors file, as an inventory sees one.
        checkpoint::st::save_safetensors(hf.join("model.safetensors").to_str().unwrap(), &[("weight".to_string(), vec![2], vec![1.0, 2.0])], &serde_json::json!({}), None).unwrap();
        let card = checkpoint::st::ModelCard::new("Qwen/Qwen3-0.6B", "qwen");
        checkpoint::st::save_safetensors(native.join("model.brain.safetensors").to_str().unwrap(), &[("weight".to_string(), vec![2], vec![1.0, 2.0])], &serde_json::json!({"vocab_size": 23}), Some(&card)).unwrap();

        let (path, dir, id) = resolve_served_base(hf.to_str().unwrap());
        assert_eq!(id.as_deref(), Some("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"), "resolved, not passed on verbatim");
        assert_eq!(dir.as_deref(), Some(hf.as_path()));
        assert_eq!(path, hf.to_str().unwrap());

        let (path, _, _) = resolve_served_base(native.to_str().unwrap());
        assert_eq!(path, native.join("model.brain.safetensors").to_str().unwrap());

        let missing = root.join("absent");
        assert_eq!(resolve_served_base(missing.to_str().unwrap()), (missing.to_str().unwrap().to_string(), None, None));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// REGRESSION: `on_device(Device::Cpu, f)` used to run `f` completely
    /// unscoped, so a residency fallback to the host tier left every
    /// `Gpu::new` inside `f` bound to whatever card was ambient - the same
    /// "`Home::Cpu` was a label, not a placement" defect
    /// `gpu_core::devices::Homes::run` was already fixed for (see
    /// `crates/gpu-core/tests/host_tier.rs`). `Device::Gpu`/`Device::Npu`
    /// must be unaffected: a GPU assignment still scopes to that card, and
    /// NPU is still refused here (never silently built as wgpu).
    #[test]
    fn on_device_cpu_scopes_to_the_real_host_tier_not_an_unscoped_call() {
        assert!(!gpu_core::devices::on_host_tier(), "no scope must be active before this test starts");
        let was_host_tier = on_device(Device::Cpu, gpu_core::devices::on_host_tier).unwrap();
        assert!(was_host_tier, "Device::Cpu must build f() inside with_host_tier, not bare");
        assert!(!gpu_core::devices::on_host_tier(), "the scope must not outlive the call");

        let was_host_tier = on_device(Device::Gpu(0), gpu_core::devices::on_host_tier).unwrap();
        assert!(!was_host_tier, "Device::Gpu must not be dragged onto the host tier");

        assert!(on_device(Device::Npu(0), || ()).is_err(), "Device::Npu must be refused here, never silently built as wgpu");
    }

    /// `QwenServeConfig::default()` reproduces every historical env-var
    /// default byte for byte, and `from_card_configured` clamps
    /// `max_prefill_cap` the same way the old static method did (never
    /// past the historical 512 default `pool_sizing`'s own doc comment
    /// records a real crash at a larger value; a nonsense/zero override
    /// clamps up to the minimum, not disabling prefill).
    /// A checkpoint whose fp32 weights do not fit one card is served with
    /// int8 linears unless the operator says otherwise; a small one stays
    /// fp32; either flag overrides the size.
    #[test]
    fn a_7b_checkpoint_defaults_to_int8_weights() {
        let dir = std::env::temp_dir().join(format!("brain-qwen-int8-default-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let shaped = |name: &str, cfg: &qwen3::config::QwenConfig| {
            let p = dir.join(name);
            checkpoint::st::save_safetensors(p.to_str().unwrap(), &[("w".to_string(), vec![1], vec![0.0])], &cfg.to_json(), None).unwrap();
            p.to_string_lossy().into_owned()
        };
        let seven_b = qwen3::config::QwenConfig { vocab: 102_400, d_model: 4096, n_layers: 30, n_heads: 32, n_kv_heads: 32, head_dim: 128, d_ff: 11_008, qk_norm: false, tie_embeddings: false, ..qwen3::config::QwenConfig::tiny() };
        let (big, small) = (shaped("7b.safetensors", &seven_b), shaped("tiny.safetensors", &qwen3::config::QwenConfig::tiny()));
        let card = checkpoint::st::ModelCard::new("deepseek-ai/deepseek-llm-7b-chat", "qwen");
        let with = |path: &str, w: Option<bool>| QwenResident::from_card_configured(path, &card, None, QwenServeConfig { weights_int8: w, ..QwenServeConfig::default() }).weights_int8_requested;
        assert!(with(&big, None), "6.9B parameters default to int8 linears");
        assert!(!with(&small, None), "a tiny checkpoint stays fp32");
        assert!(!with(&big, Some(false)) && with(&small, Some(true)), "the flags win over the size");
        // A card that holds the fp32 weights and the headroom serves them fp32 (the more accurate tier); a smaller one stays int8.
        let fp32_need = qwen3::footprint::fp32_weight_bytes(&qwen3::config::QwenConfig::qwen3_8b()) + qwen3::footprint::FP32_SERVE_HEADROOM_BYTES;
        let eight_b = shaped("8b.safetensors", &qwen3::config::QwenConfig::qwen3_8b());
        let on = |budget: Option<u64>| QwenResident::from_card_configured(&eight_b, &card, None, QwenServeConfig { auto_budget_bytes: budget, ..QwenServeConfig::default() }).weights_int8_requested;
        assert!(on(None), "no budget known: the size rule stands");
        assert!(on(Some(24 << 30)), "a 24 GB card cannot hold the 8B at fp32");
        assert!(!on(Some(fp32_need)), "a card that holds the fp32 weights and the headroom serves them fp32");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn qwen_serve_config_default_matches_every_historical_default() {
        let cfg = QwenServeConfig::default();
        assert_eq!(cfg.ctx, None, "None means auto-size - see resolve_ctx");
        assert_eq!(cfg.max_batch, 16);
        assert!(cfg.kv_int8, "int8 KV must be the default");
        assert!(!cfg.kv_calib_opt_in, "calibration must default OFF");
        assert_eq!(cfg.kv_offload_gb, 0.0);
        assert_eq!(cfg.weights_int8, None, "int8 weights default by size - see weights_int8_for");
        assert_eq!(cfg.max_prefill_cap, 512);
    }

    /// `ctx` at or under the checkpoint's own trained window is a pure no-op
    /// -- the overwhelming default (auto-sizing rarely needs to go past
    /// native, an explicit `--ctx` usually doesn't either) must never
    /// perturb `rope_scaling` from `None`.
    #[test]
    fn apply_yarn_if_serving_past_native_is_a_no_op_at_or_under_the_native_window() {
        let mut cfg = qwen3::config::QwenConfig::qwen3_8b();
        let native = cfg.max_position_embeddings;
        for ctx in [1u32, native / 2, native] {
            cfg.rope_scaling = None;
            QwenResident::apply_yarn_if_serving_past_native(&mut cfg, ctx);
            assert!(cfg.rope_scaling.is_none(), "ctx={ctx} is within the native window - must not opt into YaRN");
        }
    }

    /// The real Qwen3 shape: declared window 40960 (32768 pretrained + 8192 of
    /// generation headroom). Serving it at 131072 must be Qwen's published
    /// recipe - factor 4.0 over `original_max_position_embeddings` 32768, not
    /// 131072 / 40960 = 3.2 - and that recipe's frequency table must be the
    /// reference one (`model::yarn`'s pinned oracle, attention factor
    /// 0.1 ln 4 + 1). `--ctx 131072` on a checkpoint that spells the
    /// recipe out in its own config.json (b8's needle run) and on one that
    /// does not must therefore compute the same rotation.
    #[test]
    fn serving_a_qwen3_at_128k_derives_qwens_published_yarn_recipe() {
        let mut cfg = qwen3::config::QwenConfig::qwen3_8b();
        cfg.max_position_embeddings = 40960;
        QwenResident::apply_yarn_if_serving_past_native(&mut cfg, 131072);
        let Some(model::rope_scaling::RopeScaling::Yarn(y)) = cfg.rope_scaling else { panic!("ctx past native must derive a YaRN scaling") };
        let published = model::yarn::YarnConfig::new(4.0, 32768);
        assert_eq!(y, published);
        let (freq, af) = model::yarn::scaled_inv_freq(cfg.head_dim, cfg.rope_theta, &y);
        let (want_freq, want_af) = model::yarn::scaled_inv_freq(128, 1.0e6, &published);
        assert_eq!(freq, want_freq);
        assert_eq!(af, want_af);
        assert!((af - 1.1386294).abs() < 1e-5, "attention factor {af}");
    }

    /// Other extensions scale the pretrained window by `ctx / original`, and a
    /// declared window that is itself a power of two is the pretrained one.
    #[test]
    fn yarn_factor_is_ctx_over_the_pretrained_window() {
        for (declared, ctx, want_factor, want_original) in [(40960u32, 65536u32, 2.0f32, 32768u32), (32768, 131072, 4.0, 32768), (131072, 262144, 2.0, 131072), (4096, 8192, 2.0, 4096)] {
            let mut cfg = qwen3::config::QwenConfig::qwen3_8b();
            cfg.max_position_embeddings = declared;
            QwenResident::apply_yarn_if_serving_past_native(&mut cfg, ctx);
            let Some(model::rope_scaling::RopeScaling::Yarn(y)) = cfg.rope_scaling else { panic!("declared {declared} ctx {ctx}: no YaRN") };
            assert_eq!((y.factor, y.original_max_position_embeddings), (want_factor, want_original), "declared {declared} ctx {ctx}");
        }
    }

    /// Auto-sizing must not hand out a context past the declared window: that
    /// would switch YaRN on for every request of a server nobody asked to
    /// extend (128k stays an explicit `--ctx` until verified).
    #[test]
    fn auto_sizing_stops_at_the_declared_window_however_big_the_budget() {
        let mut cfg = qwen3::config::QwenConfig::qwen3_0_6b();
        cfg.max_position_embeddings = 40960;
        let ctx = auto_ctx_for_budget(0, u64::MAX / 4, &cfg, 16, 512, None, true, false);
        assert_eq!(ctx, 40960);
    }

    /// A checkpoint that already names its own `rope_scaling` (a real
    /// published long-context release) must never be second-guessed, even
    /// when `ctx` also exceeds its native window.
    #[test]
    fn apply_yarn_if_serving_past_native_never_overrides_an_already_declared_scaling() {
        let mut cfg = qwen3::config::QwenConfig::qwen3_8b();
        let native = cfg.max_position_embeddings;
        let declared = model::rope_scaling::RopeScaling::Yarn(model::yarn::YarnConfig::new(2.0, native));
        cfg.rope_scaling = Some(declared.clone());
        QwenResident::apply_yarn_if_serving_past_native(&mut cfg, native * 10);
        assert_eq!(cfg.rope_scaling, Some(declared), "an already-declared rope_scaling must survive untouched");
    }

    /// The fp32 KV ceiling is a floor, not a limit: a server that knows its
    /// card budget accepts an fp32 pool up to it (a 96 GB card must not be held
    /// to the 8 GiB a small-RAM box needs), while the unknown-budget ceiling
    /// still refuses the same pool.
    #[test]
    fn a_known_card_budget_raises_the_fp32_kv_ceiling() {
        let cfg = qwen3::config::QwenConfig::qwen3_0_6b();
        let (block_size, num_blocks) = (16u32, 3_000u32);
        let bytes = qwen3::serve::kv_pool_bytes(&cfg, block_size, num_blocks, false);
        assert!(bytes > MAX_FP32_KV_POOL_BYTES / 2 && bytes < 80 << 30);
        let tight = QwenResident::check_fp32_kv_pool_fits(&cfg, block_size, num_blocks * 8, 1, MAX_FP32_KV_POOL_BYTES, "x");
        assert!(tight.is_err(), "the default ceiling refuses a pool over 8 GiB");
        assert!(QwenResident::check_fp32_kv_pool_fits(&cfg, block_size, num_blocks * 8, 1, 90 << 30, "x").is_ok(), "a 90 GiB card budget accepts it");
    }

    /// `the KV pool size` sizes the KV pool in tokens independently of the
    /// per-sequence context, so many long agent histories can stay cached at
    /// once; unnamed it holds `max_batch` half-context histories (at least two
    /// contexts), and it never drops below one context (a lone admitted
    /// request must fit).
    #[test]
    fn kv_tokens_sets_the_pool_size_apart_from_the_context() {
        let (_, _, per_seq, default_blocks, _) = QwenResident::pool_sizing(4096, 8, 512, None);
        assert_eq!(default_blocks, per_seq * 2 + 8, "few slots: two contexts");
        let (_, _, _, big, _) = QwenResident::pool_sizing(4096, 8, 512, Some(400_000));
        assert_eq!(big, 400_000 / 16 + 8);
        let (_, _, _, small, _) = QwenResident::pool_sizing(4096, 8, 512, Some(16));
        assert_eq!(small, per_seq + 8);
    }

    /// At the served 8B's real shape (ctx 32768, 32 slots) on a 94 GB card the
    /// default pool is the full fleet: the unfused scratch worst case that makes
    /// the plain two-context profile "unbindable" must not veto it.
    #[test]
    fn the_real_8b_on_a_big_card_gets_the_fleet_pool() {
        let cfg = qwen3::config::QwenConfig::qwen3_8b();
        let serve = QwenServeConfig { max_batch: 32, ..QwenServeConfig::default() };
        let weights = weights_fp32_bytes(&cfg);
        assert_eq!(fleet_pool_tokens(&cfg, weights, 94 << 30, 32768, &serve), Some(32768 * 32 / 2), "32 slots x half a context");
        let tighter = fleet_pool_tokens(&cfg, weights, 60 << 30, 32768, &serve).expect("a smaller fleet fits");
        assert!(tighter < 32768 * 32 / 2 && tighter > 65536, "a card with less room gets a halved fleet, got {tighter}");
        assert_eq!(fleet_pool_tokens(&cfg, weights, 40 << 30, 32768, &serve), None, "no room beside the weights");
    }

    /// A live server (budget known) keeps `max_batch` half-context histories
    /// cached by default, when that fits; with no budget, no room, or an
    /// operator's own number, the pool is the plain one.
    #[test]
    fn a_live_server_sizes_the_pool_for_a_fleet_of_histories_when_it_fits() {
        let path = write_tiny_checkpoint(5, "fleet");
        let path = path.to_str().unwrap();
        let live = |budget: Option<u64>, kv_tokens: Option<u32>| QwenServeConfig { ctx: Some(4096), max_batch: 16, card_budget_bytes: budget, kv_tokens, ..QwenServeConfig::default() };
        let resolve = |cfg: &QwenServeConfig| QwenResident::resolve_kv_tokens(path, cfg, 4096);
        assert_eq!(resolve(&live(Some(u64::MAX / 4), None)), Some(4096 * 16 / 2), "16 slots x half a context");
        assert_eq!(resolve(&live(None, None)), None, "no card budget: the two-context pool");
        assert_eq!(resolve(&live(Some(1 << 20), None)), None, "does not fit the budget");
        assert_eq!(resolve(&live(Some(u64::MAX / 4), Some(1234))), Some(1234), "the operator's number wins");
        let few = QwenServeConfig { max_batch: 2, ..live(Some(u64::MAX / 4), None) };
        assert_eq!(resolve(&few), None, "two contexts already hold two half-contexts");
    }

    #[test]
    fn max_prefill_cap_is_clamped_at_construction() {
        let card = checkpoint::st::ModelCard::new("brain/qwen3", "qwen");
        for (requested, expected) in [(128u32, 128u32), (0, 1), (999_999, 512)] {
            let cfg = QwenServeConfig { max_prefill_cap: requested, ..QwenServeConfig::default() };
            let resident = QwenResident::from_card_configured("unused.safetensors", &card, Some("unused.json"), cfg);
            assert_eq!(resident.max_prefill_cap, expected, "requested {requested}");
        }
    }

    /// End to end at the REAL Qwen3-8B config: lowering `the prefill row cap`
    /// shrinks `paged_attn_scratch_bytes` (3.00 of an 18.41 GiB device total
    /// at the operator's real ctx=24576) linearly, with zero change to
    /// `kv_pool_bytes` (prefill chunk size never enters the KV pool's own
    /// sizing) and zero change to decode-step shape (`max_prefill` only ever
    /// appears in the causal-chunk PREFILL term).
    ///
    /// This is the nothing-fused branch, which is what
    /// every call site in THIS file deliberately prices (see `estimate`'s own
    /// comment: no target `DeviceCaps` exists pre-placement, so the never-
    /// under bound is the only honest one). On a device that can actually run
    /// the fused kernels, `Engine::from_map_with_gpu` allocates the OTHER
    /// branch, where `max_prefill` does not appear at all and the buffer is
    /// `max_batch`-sized instead - `qwen3::serve`'s own
    /// `paged_attn_scratch_at_the_real_8b_serving_default_shape` pins both
    /// arms at this same shape. So `the prefill row cap` moves the pre-
    /// placement budget and the triad-path allocation; it does not move a
    /// fused-path one.
    #[test]
    fn lowering_max_prefill_shrinks_scratch_linearly_at_the_real_config() {
        let cfg = qwen3::config::QwenConfig::qwen3_8b();
        let ctx = 24576u32;
        let max_batch = 16u32;

        let (block_size, max_batch, max_blocks_per_seq, num_blocks, max_prefill_default) = QwenResident::pool_sizing(ctx, max_batch, 512, None);
        assert_eq!(max_prefill_default, 512);
        let cap = max_blocks_per_seq * block_size;
        let scratch_default = qwen3::serve::paged_attn_scratch_bytes(&cfg, max_batch, max_prefill_default, cap, qwen3::serve::FusedAttention::default());
        let kv_default = qwen3::serve::kv_pool_bytes(&cfg, block_size, num_blocks, true);

        let (_, max_batch2, _, num_blocks2, max_prefill_128) = QwenResident::pool_sizing(ctx, max_batch, 128, None);
        assert_eq!(max_prefill_128, 128);
        let scratch_128 = qwen3::serve::paged_attn_scratch_bytes(&cfg, max_batch2, max_prefill_128, cap, qwen3::serve::FusedAttention::default());
        let kv_128 = qwen3::serve::kv_pool_bytes(&cfg, block_size, num_blocks2, true);

        assert_eq!(kv_128, kv_default, "the KV pool must be completely unaffected by max_prefill");
        assert_eq!(scratch_default * 128 / 512, scratch_128, "scratch must shrink exactly linearly with the chunk cap");
        let saved_gib = (scratch_default - scratch_128) as f64 / (1u64 << 30) as f64;
        assert!(saved_gib > 2.0, "expected roughly the measured 2.25 GiB saving at 512->128, got {saved_gib:.2} GiB");
    }

    /// The 7 per-layer linears (`Q8::LINEARS`) must shrink to the real
    /// packed-int8 + group-scale byte count; every OTHER tensor (the token
    /// embedding and LM head, at this untied real config) must stay at its
    /// full fp32 size.
    ///
    /// REGRESSION: this used to read a checkpoint's RAW on-disk tensor
    /// names via `WeightReader`, which `checkpoint::load`'s `Container::find`
    /// remaps onto these SAME canonical names before `Engine::load` ever
    /// sees them - so on a real checkpoint whose raw names differ from
    /// brain's canonical `"blocks.N.leaf"` form, nothing ever matched
    /// `Q8::is_i8_linear` and this silently returned the full fp32 total
    /// even with int8 correctly requested and applied at activation time.
    /// Deriving from `QwenConfig::param_list()` (the same canonical-name
    /// source `decoder_param_list` uses) makes that class of drift
    /// structurally impossible - there is no raw file to disagree with.
    #[test]
    fn weights_int8_bytes_quantizes_only_the_declared_linears() {
        let cfg = qwen3::config::QwenConfig::qwen3_8b();
        let mut linear_elems = 0u64;
        let mut other_elems = 0u64;
        for (name, n) in cfg.param_list() {
            if qwen3::q8::Q8::is_i8_linear(&name) {
                linear_elems += n as u64;
            } else {
                other_elems += n as u64;
            }
        }
        assert!(linear_elems > 0 && other_elems > 0, "the real config must exercise both branches");
        let expected = (linear_elems + (linear_elems / 32) * 4) + other_elems * 4;
        assert_eq!(weights_int8_bytes(&cfg), expected);
        assert!(weights_int8_bytes(&cfg) < other_elems * 4 + linear_elems * 4, "quantizing the linears must shrink the total below the full fp32 size");
    }

    /// End to end through `estimate()`: opting in shrinks the total, opting
    /// out (the default) leaves it at the config's fp32 weight bytes
    /// - same env-gated shape `kv_int8`'s own estimate switch already uses.
    #[test]
    fn estimate_shrinks_the_weight_term_when_weights_int8_is_requested() {
        let path = write_tiny_checkpoint(29, "estimate-weights-int8");
        let card = checkpoint::st::ModelCard::new("brain/qwen3", "qwen");
        let key = InstanceKey::new("brain/qwen3", "default");

        let fp32_resident = QwenResident::from_card(path.to_str().unwrap(), &card, Some("unused.json"));
        let fp32_total = fp32_resident.estimate(&key).vram;

        let int8_cfg = QwenServeConfig { weights_int8: Some(true), ..QwenServeConfig::default() };
        let int8_resident = QwenResident::from_card_configured(path.to_str().unwrap(), &card, Some("unused.json"), int8_cfg);
        let int8_total = int8_resident.estimate(&key).vram;

        assert!(int8_total < fp32_total, "requesting int8 weights must shrink the estimate: fp32={fp32_total} int8={int8_total}");
    }

    /// The fp32 KV boundary guard must refuse (not attempt) a pool over the
    /// ceiling, and pass through a pool comfortably under it -- pure
    /// arithmetic, no engine, no allocation, so a multi-GiB refusal case
    /// costs nothing to test.
    #[test]
    fn fp32_kv_pool_guard_refuses_over_ceiling_and_passes_under_it() {
        let cfg = qwen3::config::QwenConfig::tiny(); // head_dim=8, n_kv_heads=2, hkv=16
        // Small, ordinary sizing: comfortably under the ceiling.
        assert!(QwenResident::check_fp32_kv_pool_fits(&cfg, 16, 64, 2048, MAX_FP32_KV_POOL_BYTES, "test.safetensors").is_ok());
        // num_blocks chosen so this TINY config's fp32 pool (256 bytes/slot)
        // alone exceeds MAX_FP32_KV_POOL_BYTES (8 GiB) -- num_blocks=3,000,000
        // -> slots=48,000,000 -> ~11.4 GiB, comfortably over.
        let err = QwenResident::check_fp32_kv_pool_fits(&cfg, 16, 3_000_000, 100_000_000, MAX_FP32_KV_POOL_BYTES, "test.safetensors")
            .expect_err("an 11+ GiB fp32 pool must be refused, not attempted");
        assert!(err.contains("test.safetensors"), "error must name the checkpoint: {err}");
        assert!(err.contains("100000000"), "error must name the requested ctx: {err}");
        assert!(err.contains("GiB"), "error must name the computed size: {err}");
    }

    /// REGRESSION target: the int8 (default) KV path had NO per-buffer
    /// guard at all before this existed - only the fp32 branch's much
    /// coarser total-pool ceiling did. `Engine::from_map_with_gpu` allocates
    /// one K and one V buffer PER LAYER, so a large enough `num_blocks`
    /// (driving the per-layer buffer size, independent of `check_fp32_kv_
    /// pool_fits`'s own total-pool check) must be refused on kv_int8=true
    /// too, never left to reach wgpu's own uncaptured-error panic.
    #[test]
    fn per_buffer_guard_refuses_an_oversized_per_layer_kv_buffer_even_under_int8() {
        let cfg = qwen3::config::QwenConfig::tiny(); // n_layers=2, head_dim=8, n_kv_heads=2, hkv=16
        // Small, ordinary sizing on both dtypes: comfortably under.
        assert!(QwenResident::check_buffers_fit_one_binding(&cfg, 16, 64, 4, 16, 256, true, 2048, "test.safetensors", Default::default()).is_ok());
        assert!(QwenResident::check_buffers_fit_one_binding(&cfg, 16, 64, 4, 16, 256, false, 2048, "test.safetensors", Default::default()).is_ok());
        // num_blocks=3,000,000 -> per-layer fp32 KV buffer ~2.86 GiB, far over
        // the 128 MiB floor - even though this SAME sizing's fp32 TOTAL
        // pool (~11.4 GiB) is what `check_fp32_kv_pool_fits`'s own test
        // above already catches; this checks the independent per-buffer path.
        // The KV term does not depend on the fused-attention answer at all,
        // so this must still refuse it under `true` too.
        for fused in [qwen3::serve::FusedAttention::default(), qwen3::serve::FusedAttention { prefill: true, decode: true }] {
            let err = QwenResident::check_buffers_fit_one_binding(&cfg, 16, 3_000_000, 4, 16, 256, false, 100_000_000, "test.safetensors", fused)
                .expect_err("a ~2.86 GiB single buffer must be refused, not attempted");
            // Names the checkpoint and the requested ctx -- the two things an
            // operator needs to act. Deliberately not the size unit: that is
            // presentation, and pinning it here broke this test when the
            // message moved from GiB to MiB without any behaviour changing.
            assert!(err.contains("test.safetensors") && err.contains("100000000"), "{err}");
        }
    }

    /// The OTHER buffer this same guard must catch: a large enough
    /// `max_prefill`/`cap` drives the scores/probs scratch buffer over the
    /// floor independent of the KV pool entirely (small `num_blocks` here).
    #[test]
    fn per_buffer_guard_refuses_an_oversized_scratch_buffer() {
        let cfg = qwen3::config::QwenConfig::tiny(); // n_heads=4
        // max_prefill=20_000 -> the causal-chunk term (max_prefill^2 * n_heads)
        // dominates: 20_000^2 * 4 = 1.6e9 words -> per-buffer ~5.96 GiB.
        let err = QwenResident::check_buffers_fit_one_binding(&cfg, 16, 64, 4, 20_000, 320_000, true, 100_000_000, "test.safetensors", Default::default())
            .expect_err("a ~5.96 GiB scratch buffer must be refused, not attempted");
        assert!(err.contains("test.safetensors") && err.contains("100000000"), "{err}");
    }

    /// REGRESSION target: this exact shape (Qwen3-8B, ctx=32768, the default
    /// max_prefill=512) reached a real production refusal - "largest single
    /// device buffer ... 2.00 GiB, over WebGPU's own 2.00 GiB per-buffer
    /// floor" - on a live GPU that was actually about to dispatch the fused
    /// prefill kernel (a much smaller scratch buffer), because `activate()`
    /// passed nothing-fused unconditionally even once it
    /// already knew a concrete, non-CPU `device`. Proves the SAME sizing
    /// that this guard must refuse under `false` (the pre-fix behavior) it
    /// must now ACCEPT under `true`.
    #[test]
    fn fused_attention_shrinks_the_scratch_term_enough_to_admit_the_real_qwen3_8b_ctx_32768_shape() {
        let cfg = qwen3::config::QwenConfig::qwen3_8b();
        let (block_size, max_batch, max_blocks_per_seq, num_blocks, max_prefill) = QwenResident::pool_sizing(32768, 16, 512, None);
        let cap = max_blocks_per_seq * block_size;
        assert!(
            QwenResident::check_buffers_fit_one_binding(&cfg, block_size, num_blocks, max_batch, max_prefill, cap, true, 32768, "test.safetensors", qwen3::serve::FusedAttention::default()).is_err(),
            "this test's premise is the real reported refusal under the never-fused worst case"
        );
        assert!(
            QwenResident::check_buffers_fit_one_binding(&cfg, block_size, num_blocks, max_batch, max_prefill, cap, true, 32768, "test.safetensors", qwen3::serve::fused_attention_on_gpu(cfg.head_dim, true)).is_ok(),
            "the fused kernel's real scratch buffer must fit, once activate() actually knows the device supports it"
        );
    }

    /// The pre-flight constant is an upper bound, not a floor, and the two
    /// must not be confused again: it exists to catch sizings no device could
    /// ever accept, while whether a sizing fits THIS device is decided
    /// against that device's own queried limit.
    #[test]
    fn the_preflight_constant_is_the_ceiling_no_device_can_exceed() {
        assert_eq!(WGPU_MAX_REPORTABLE_BINDING_BYTES, 2047 * (1 << 20));
        const {
            assert!(
                WGPU_MAX_REPORTABLE_BINDING_BYTES > 128 * (1 << 20),
                "a real adapter may report as little as WebGPU's 128 MiB guarantee; this constant is \
                 deliberately NOT that number, so it must never be used to decide a real device fits"
            );
        }
    }

    /// The auto-sizer and the pre-flight guard must not disagree about what
    /// fits, because between them they decide whether a server runs at all.
    ///
    /// They did. Auto-sizing picked ctx=98304 for a real Qwen3-0.6B on a
    /// 24 GiB card; the guard waved it through against the wrong floor; and
    /// the engine then died inside create_bind_group with "Buffer binding 5
    /// range 201588736 exceeds max_*_buffer_binding_size limit 134217728" --
    /// a wgpu validation panic on a lane, mid-request, rather than a refusal.
    #[test]
    fn auto_sizing_never_picks_a_ctx_the_binding_guard_would_refuse() {
        let mut cfg = qwen3::config::QwenConfig::qwen3_0_6b();
        cfg.max_position_embeddings = 262144;
        let (max_batch, max_prefill_cap) = (16u32, 512u32);
        // A budget far past any real card, so the binding limit is the only
        // thing that can bound the answer.
        let ctx = auto_ctx_for_budget(0, u64::MAX / 4, &cfg, max_batch, max_prefill_cap, None, true, false);

        let (block_size, mb, max_blocks_per_seq, num_blocks, max_prefill) =
            QwenResident::pool_sizing(ctx, max_batch, max_prefill_cap, None);
        let cap = max_blocks_per_seq * block_size;
        QwenResident::check_buffers_fit_one_binding(
            &cfg, block_size, num_blocks, mb, max_prefill, cap, true, ctx, "qwen3-0.6b", qwen3::serve::fused_attention_on_gpu(cfg.head_dim, true),
        )
        .unwrap_or_else(|e| panic!("auto-sizing chose ctx={ctx}, which its own guard refuses: {e}"));
    }

    /// The swap staging of host-RAM KV offload is device memory like the pool:
    /// a budget that fits a context without it must not be allowed to choose
    /// that context with it, or the first preemption allocates past the budget.
    #[test]
    fn a_sizing_that_does_not_fit_is_refused_with_the_per_category_arithmetic() {
        let cfg = qwen3::config::QwenConfig::qwen3_0_6b();
        let profile = PoolProfile::at_ctx(&cfg, 8192, 16, 512, None, true, true, qwen3::serve::FusedAttention::default());
        let weights = 1u64 << 30;
        let need = model::memplan::device_total(&profile.demands(weights, 0));
        assert!(profile.plan(weights, 0, need).is_ok(), "a budget of exactly the footprint fits");
        let err = profile.plan(weights, 0, need - 1).unwrap_err().to_string();
        for needle in ["does not fit", "Weights: weights", "Kv: paged pool", "Activations: attention scratch", "Transfers: kv swap staging"] {
            assert!(err.contains(needle), "the refusal must show {needle:?}: {err}");
        }
    }

    #[test]
    fn auto_sizing_leaves_room_for_the_swap_staging_when_offload_is_on() {
        let mut cfg = qwen3::config::QwenConfig::qwen3_0_6b();
        cfg.max_position_embeddings = 262144;
        let (max_batch, max_prefill_cap) = (16u32, 512u32);
        let fused = qwen3::serve::FusedAttention::default();
        let tier = AUTO_CTX_TIERS[3];
        // A budget that holds `tier` exactly, with nothing to spare.
        let budget = model::memplan::device_total(&PoolProfile::at_ctx(&cfg, tier, max_batch, max_prefill_cap, None, true, false, fused).demands(0, 0));
        assert_eq!(auto_ctx_for_budget(0, budget, &cfg, max_batch, max_prefill_cap, None, true, false), tier);
        let with_offload = auto_ctx_for_budget(0, budget, &cfg, max_batch, max_prefill_cap, None, true, true);
        assert!(with_offload < tier, "offload staging must come out of the budget: chose ctx {with_offload}");
    }

    /// The sizing that crashed a real server is device-dependent, and that is
    /// the whole point: one layer's K pool at ctx=98304 on Qwen3-0.6B is
    /// 201588736 bytes, which a Tesla P40 (2047 MiB limit) binds happily and
    /// llvmpipe (128 MiB) refuses. No constant can answer this, so the
    /// pre-flight ceiling must NOT refuse it and the per-device check must.
    #[test]
    fn a_sizing_only_some_devices_accept_is_left_to_the_device() {
        let cfg = qwen3::config::QwenConfig::qwen3_0_6b();
        let ctx = 98304u32;
        let (block_size, mb, max_blocks_per_seq, num_blocks, max_prefill) =
            QwenResident::pool_sizing(ctx, 16, 512, None);
        let cap = max_blocks_per_seq * block_size;
        let profile = PoolProfile::new(&cfg, block_size, num_blocks, mb, max_prefill, cap, true, qwen3::serve::FusedAttention { prefill: true, decode: false });
        assert!(profile.largest_binding > 128 * (1 << 20), "too big for a 128 MiB device");
        assert!(profile.largest_binding < WGPU_MAX_REPORTABLE_BINDING_BYTES, "but fine on a desktop card");
        QwenResident::check_buffers_fit_one_binding(
            &cfg, block_size, num_blocks, mb, max_prefill, cap, true, ctx, "qwen3-0.6b", qwen3::serve::fused_attention_on_gpu(cfg.head_dim, true),
        )
        .expect("the early ceiling must not refuse a sizing real hardware runs");
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
        let card = checkpoint::st::ModelCard::new("brain/qwen3", "qwen");
        let resident = QwenResident::from_card("unused.safetensors", &card, Some("unused.json"));
        assert_eq!(resident.ctx, ctx, "this test's premise is the new default -- update it if the default changes");
        let (block_size, _max_batch, _max_blocks_per_seq, num_blocks, _max_prefill) = QwenResident::pool_sizing(ctx, 16, 512, None);

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
        // would hit under `--kv-cache f32` at this ctx.
        assert!(fp32_bytes > MAX_FP32_KV_POOL_BYTES, "fp32 KV pool at ctx=24576 must exceed the ceiling -- this is WHY the default is safe only under int8");
        let err = QwenResident::check_fp32_kv_pool_fits(&cfg, block_size, num_blocks, ctx, MAX_FP32_KV_POOL_BYTES, "qwen3-0.6b")
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
            brain_testutil::skip("set QWEN3_DIR to a real Qwen3-0.6B checkpoint dir to run this test");
            return;
        };
        let path = std::path::Path::new(&dir).join("qwen.brain.safetensors");
        if !path.is_file() {
            brain_testutil::skip(&format!("{} not found under QWEN3_DIR", path.display()));
            return;
        }
        // SAFETY: this test is `#[ignore]`d (never runs under the default
        // `TEST_THREADS=8` fast lane) and is invoked alone, single-threaded,
        // for exactly this measurement -- see the module doc's OOM-budget
        // discipline (never build a real-shape engine alongside other tests).
        unsafe { std::env::set_var("BRAIN_DEVICE", "cpu") };

        let cfg = qwen3::config::QwenConfig::qwen3_0_6b();
        let card = checkpoint::st::ModelCard::new("brain/qwen3", "qwen");
        let resident = QwenResident::from_card("unused.safetensors", &card, Some("unused.json"));
        let ctx = resident.ctx;
        assert_eq!(ctx, 24576, "this test's premise is the new default");
        let (block_size, max_batch, max_blocks_per_seq, num_blocks, max_prefill) = QwenResident::pool_sizing(ctx, 16, 512, None);
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
    /// No env/global state involved at all now that `kv_int8` is a field
    /// resolved once at construction: checks `estimate()` matches an
    /// independent recomputation at the resident's own default, and checks
    /// the dtype-shrink property directly against the pure `kv_pool_bytes`
    /// function (parameterized over `kv_int8`, no engine needed).
    #[test]
    fn estimate_counts_the_kv_pool_and_shrinks_under_int8() {
        let path = write_tiny_checkpoint(11, "estimate");
        let cfg = qwen3::config::QwenConfig { vocab: 151936, ..qwen3::config::QwenConfig::tiny() };
        let card = checkpoint::st::ModelCard::new("brain/qwen3", "qwen");
        let resident = QwenResident::from_card(path.to_str().unwrap(), &card, Some("unused.json"));
        let key = InstanceKey::new("brain/qwen3", "default");

        let (block_size, max_batch, max_blocks_per_seq, num_blocks, max_prefill) = QwenResident::pool_sizing(resident.ctx, resident.max_batch, resident.max_prefill_cap, resident.kv_tokens);
        let kv_int8 = resident.kv_int8_requested;
        let expected_kv_bytes = qwen3::serve::kv_pool_bytes(&cfg, block_size, num_blocks, kv_int8);
        let cap = max_blocks_per_seq * block_size;
        let expected_scratch_bytes = qwen3::serve::paged_attn_scratch_bytes(&cfg, max_batch, max_prefill, cap, qwen3::serve::FusedAttention::default());
        let weights = weights_fp32_bytes(&cfg);

        let got = resident.estimate(&key);
        assert_eq!(
            got.vram,
            weights + expected_kv_bytes + expected_scratch_bytes,
            "estimate() must equal the fp32 weights + the KV pool + the paged-attention scratch, no more and no less"
        );
        assert!(expected_kv_bytes > 0, "the KV pool must contribute a nonzero amount at these test dims");
        assert!(expected_scratch_bytes > 0, "the paged-attention scratch must contribute a nonzero amount at these test dims");

        // The shrink property itself: pure function, both dtypes, no engine.
        let fp32_bytes = qwen3::serve::kv_pool_bytes(&cfg, block_size, num_blocks, false);
        let int8_bytes = qwen3::serve::kv_pool_bytes(&cfg, block_size, num_blocks, true);
        assert!(int8_bytes < fp32_bytes, "int8 must cost fewer bytes than fp32 at the same num_blocks");
    }

    /// Build a tiny (fast-CPU-testable) but REAL safetensors checkpoint whose
    /// vocab covers the real tokenizer's full range (so chat-template special
    /// tokens like `<|im_start|>` never index outside the embedding table -
    /// same reasoning as `rejected_admission_resolves_promptly_instead_of_hanging`),
    /// and write it to a scratch dir. Returns the checkpoint path.
    /// A Llama-shaped checkpoint served straight from its Hugging Face
    /// directory, as downloaded: the resident reads `config.json`, the
    /// safetensors under their HF names and the byte-level `tokenizer.json`
    /// in place, and generates exactly what the decoder built from the same
    /// weights generates. Nothing is written beside the checkpoint.
    #[test]
    fn a_hugging_face_directory_is_served_as_downloaded() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("qwen-resident-hf-dir-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let config = r#"{"architectures":["LlamaForCausalLM"],"vocab_size":256,"hidden_size":16,"num_hidden_layers":2,
            "num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":32,"rope_theta":10000,
            "rms_norm_eps":1e-5,"tie_word_embeddings":false,"max_position_embeddings":64}"#;
        std::fs::write(dir.join("config.json"), config).unwrap();
        let cfg = qwen3::hf::decoder_config(config).unwrap();
        let init = qwen3::init_weights(&cfg, 17);
        let names = qwen3::hf::HfNames::CAUSAL_LM;
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
            cfg.param_list().into_iter().map(|(n, numel)| (names.from_brain(&n).unwrap(), vec![numel as u64], init[&n].clone())).collect();
        checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &tensors, &serde_json::Value::Null, None).unwrap();
        let vocab: serde_json::Map<String, serde_json::Value> = data::bpe::bytes_to_unicode().iter().enumerate().map(|(i, c)| (c.to_string(), json!(i))).collect();
        let tokenizer = json!({"model": {"vocab": vocab, "merges": []}, "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}});
        std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
        let before: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();

        let card = checkpoint::st::ModelCard::new("test/tiny-llama", "llama");
        let resident = QwenResident::from_card(dir.to_str().unwrap(), &card, dir.join("tokenizer.json").to_str());
        let mut inst = resident.activate(&InstanceKey::new("test/tiny-llama", "default"), Device::Gpu(0)).expect("activate");
        let inv = Invocation::new().set("prompt", json!("hello")).set("chat", json!(false)).set("max_new", json!(6)).set("temp", json!(0.0));
        let got = inst.run("generate", &inv, &mut |_| {}).expect("generate");

        let tok = data::qwen_tokenizer::QwenBpe::from_file(dir.join("tokenizer.json").to_str().unwrap()).unwrap();
        let model = qwen3::model::Qwen::from_tensors_decode(cfg, &init, 64);
        let ids = qwen3::sample::generate_kv(&model, &tok.encode("hello"), 6, 0.0, 0, 1.0, &[], &mut Rng::random());
        assert_eq!(got.outputs.get("text"), Some(&json!(tok.decode(&ids))));
        let after: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(after, before, "serving writes nothing beside the checkpoint");
        let _ = std::fs::remove_dir_all(&dir);
    }

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
    /// `activate` - see that method's doc), so the second's prefill should
    /// find the first's system-prompt blocks already cached.
    ///
    /// Needs a real tokenizer (`QWEN_TOKENIZER=/path/to/tokenizer.json`) --
    /// self-skips loudly when unset.
    #[test]
    fn prefix_cache_hit_rate_is_observable_through_the_real_http_router() {
        let Ok(tok_path) = std::env::var("QWEN_TOKENIZER") else {
            brain_testutil::skip("set QWEN_TOKENIZER to a real tokenizer.json to run this test");
            return;
        };
        let path = write_tiny_checkpoint(5, "prefix");

        let card = checkpoint::st::ModelCard::new("brain/qwen3", "qwen");
        let resident = QwenResident::from_card(path.to_str().unwrap(), &card, Some(&tok_path));
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
                "model": "brain/qwen3",
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

        let key = InstanceKey::new("brain/qwen3", "default");
        let stats = exec.stats();
        let m = stats.metrics.get(&key).unwrap_or_else(|| panic!("no metrics for {key:?}; stats={stats:?}"));
        let rate = m.iter().find(|(k, _)| k == "kv_prefix_hit_rate").map(|(_, v)| v.as_f64().unwrap_or(0.0)).unwrap_or(0.0);
        assert!(rate > 0.0, "second request must hit the first's cached system-prompt prefix; metrics={m:?}");
    }

    /// REGRESSION: `run_batch_scheduled` must resolve EVERY batch index,
    /// including one the scheduler REJECTS at admission (`model::serve::
    /// RejectReason`) rather than completes. Before this test's fix, a
    /// rejected sequence's `bi` was never removed from the pending set -
    /// `report.rejected` was silently ignored - so `run_batch_scheduled`
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
            brain_testutil::skip("set QWEN_TOKENIZER to a real tokenizer.json to run this test");
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
        let decoding = Decoding { tok, format: ChatFormat::Qwen, stop: Vec::new() };
        let adapters = AdapterSync::new(AdapterRegistry::new(None, ServedBase::new("unused", None)));
        let results = run_batch_scheduled(&mut sched, &QwenFrontend { decoding: &decoding, adapters: &adapters }, std::slice::from_ref(&inv), &mut |_i, _p| {});
        assert!(start.elapsed() < std::time::Duration::from_secs(5), "rejected admission must resolve promptly, not hang");
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err(), "an out-of-vocab prompt must be reported as an error, not silently dropped");
    }
}
