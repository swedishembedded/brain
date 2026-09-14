// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident-model adapters for brain's text-generation LLMs - GPT (dense
//! char-level baseline), GLM (MLA + noaux_tc MoE decoder), and Qwen3 (BPE
//! decoder) - behind the residency [`Executor`], mirroring the yolo/z-image
//! adapters in [`crate::resident`].
//!
//! Each model family is one [`ResidentModel`] with a single `"generate"` action.
//! Unlike yolo (which pins itself to the CPU via `Gpu::new_cpu` unless a
//! `--device` was chosen), these models load through `gpu_core::Gpu::new`, i.e.
//! the process-default backend - **wgpu (GPU) unless `BRAIN_DEVICE=cpu`**. So the
//! resident instance holds the model on a GPU (VRAM); dropping it frees the card.
//! `activate` places the build on the assigned card via a scoped device-registry
//! selection ([`on_device`]), exactly like z-image.
//!
//! Model SELECTION is env-only: `BRAIN_GPT2_WEIGHTS`, `BRAIN_GLMDSA_WEIGHTS`,
//! `BRAIN_QWEN_WEIGHTS` + `BRAIN_QWEN_TOKENIZER` name WHICH checkpoint to
//! serve; each `from_env` returns `None` when its primary weights var is
//! unset/empty. HOW to serve Qwen3 - context length, batching, KV/weight
//! precision - is [`QwenServeConfig`], which `run_cli.rs`'s `--qwen-*`
//! flags build; see that struct's own doc for every field and its default.
//! With no `--qwen-ctx` given, context auto-sizes to the target device's
//! real free VRAM (see [`QwenResident::resolve_ctx`]) rather than a single
//! fixed number for every box.

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
///
/// Measured against a real `Qwen/Qwen3-8B` checkpoint at the historical
/// 24576 default context (`--qwen-ctx`), int8 KV: weights (1.3x a ~16 GiB bf16 file)
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
/// - true for every shipped Qwen3 config, so no separate `n`/`k` split is
/// needed to compute it.
fn weights_int8_bytes(cfg: &qwen3::config::QwenConfig) -> u64 {
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
/// it refuses the same 8 GiB regardless of how much VRAM the target card
/// actually has, which is real headroom to revisit if this ever blocks a
/// large-VRAM discrete card rather than the small-RAM box it was sized for.
/// Only ever checked against the FP32 pool: at the historical 24576
/// default context, int8 is comfortably under this on its own
/// (`kv_pool_bytes_at_the_new_ctx_default_fits_the_igpu_budget`); fp32 is not.
const MAX_FP32_KV_POOL_BYTES: u64 = 8 << 30;

/// WebGPU's own spec-mandated floor for `maxStorageBufferBindingSize`
/// (2047 MiB, not 2048 - the spec's limit is `2^31 - 1` bytes, one byte
/// short of a clean power of two). Every compliant device, including
/// every backend this engine runs on, guarantees AT LEAST this many bytes
/// per single storage-buffer binding; some report more, but a live
/// query needs an actual device, which neither `estimate()` nor this
/// pre-placement point in `activate()` has picked yet. Checking against
/// the floor rather than guessing a real device's own (possibly larger)
/// limit keeps this pre-flight check honest: it can refuse a config that
/// would actually be fine on THIS box's cards, never accept one that
/// would crash on some compliant device.
const WEBGPU_MIN_STORAGE_BINDING_BYTES: u64 = 2047 * (1 << 20);

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

    /// Construct under the card's id. `_tokenizer` is unused - GLM is char-level.
    pub fn from_card(path: &str, card: &ModelCard, _tokenizer: Option<&str>) -> GlmResident {
        GlmResident { id: card.id.clone(), path: path.to_string() }
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
        let gen = glmdsa::sample::generate_kv(&self.model, &ids, max_new, temp, top_k, None, &mut rng);
        let text = self.tok.decode(&gen);
        progress(Progress::step(max_new as u32, max_new as u32, "done"));
        Ok(text_outcome(text))
    }
    // `run_batch` is deliberately the residency-default serial loop - GLM has
    // no batch axis to exploit yet, see `GlmResident`'s module doc for why.
}

// ---------------------------------------------------------------- qwen

/// Every tunable that used to be its own `BRAIN_QWEN_*` env var, now the
/// explicit config `brain serve`'s own `--qwen-*` flags build and pass down
/// - model SELECTION (`BRAIN_QWEN_WEIGHTS`/`BRAIN_QWEN_TOKENIZER`, which
/// checkpoint to serve at all) stays env-only, everything about HOW to
/// serve it is a flag. `Default` reproduces every historical env-var
/// default byte for byte, so a caller that builds `QwenServeConfig::
/// default()` (every test, `perf_cli.rs`, `model_dir.rs`'s catalog scan,
/// `continuous_train.rs`) is unaffected by this existing at all.
#[derive(Clone, Copy, Debug)]
pub struct QwenServeConfig {
    /// `--qwen-ctx N`. `None` means "size it automatically" - see
    /// [`QwenResident::resolve_ctx`]. Historical default when NEITHER this
    /// nor `auto_budget_bytes` is given: 24576.
    pub ctx: Option<u32>,
    /// The device budget to auto-size `ctx` against, when `ctx` itself is
    /// `None` - real free VRAM (minus `--reserve-gb`) for the card `brain
    /// serve` would actually place this on, known only to `resident.rs::
    /// build_executor`'s caller. `None` (every caller except the live
    /// server) means "unknown", which keeps the historical 24576 fallback
    /// rather than guessing - auto-sizing is a property of the live
    /// serving path, not of this struct's mere existence.
    pub auto_budget_bytes: Option<u64>,
    /// `--qwen-max-batch N`, default 16.
    pub max_batch: u32,
    /// `--qwen-kv-fp32` (presence opts OUT), default `true` (int8 KV is the
    /// serving default - see [`QwenResident::activate`]'s own doc for the
    /// measurement that earned it).
    pub kv_int8: bool,
    /// `--qwen-kv-calib` (presence opts IN), default `false`.
    pub kv_calib_opt_in: bool,
    /// `--qwen-kv-offload-gb N` (fractional allowed), default `0.0` (off).
    pub kv_offload_gb: f64,
    /// `--qwen-weights-int8` (presence opts IN), default `false`.
    pub weights_int8: bool,
    /// `--qwen-max-prefill N`, clamped to `1..=512`, default 512.
    pub max_prefill_cap: u32,
}

impl Default for QwenServeConfig {
    fn default() -> QwenServeConfig {
        QwenServeConfig { ctx: None, auto_budget_bytes: None, max_batch: 16, kv_int8: true, kv_calib_opt_in: false, kv_offload_gb: 0.0, weights_int8: false, max_prefill_cap: 512 }
    }
}

/// Context-length tiers [`QwenResident::resolve_ctx`]'s auto-sizing path
/// picks from, ascending - round numbers an operator recognizes in a log
/// line, not an arbitrary byte-granularity sweep result.
const AUTO_CTX_TIERS: &[u32] = &[2048, 4096, 8192, 16384, 24576, 32768, 40960, 49152, 65536, 98304, 131072, 196608, 262144];

/// The largest tier in [`AUTO_CTX_TIERS`] whose real weights+KV+scratch
/// total (the SAME formulas `estimate()`/`activate()` build from) fits
/// `budget_bytes`, given `weight_bytes` is fixed (independent of `ctx`).
/// Never a closed-form approximation - costs are monotonically increasing
/// in `ctx`, so the first tier that overflows ends the search.
fn auto_ctx_for_budget(weight_bytes: u64, budget_bytes: u64, cfg: &qwen3::config::QwenConfig, max_batch: u32, max_prefill_cap: u32, kv_int8: bool) -> u32 {
    let mut best = AUTO_CTX_TIERS[0];
    for &ctx in AUTO_CTX_TIERS {
        let (block_size, mb, max_blocks_per_seq, num_blocks, max_prefill) = QwenResident::pool_sizing(ctx, max_batch, max_prefill_cap);
        let kv_bytes = qwen3::serve::kv_pool_bytes(cfg, block_size, num_blocks, kv_int8);
        let cap = max_blocks_per_seq * block_size;
        let scratch_bytes = qwen3::serve::paged_attn_scratch_bytes(cfg, mb, max_prefill, cap, false);
        if weight_bytes + kv_bytes + scratch_bytes <= budget_bytes {
            best = ctx;
        } else {
            break;
        }
    }
    best
}

/// The Qwen3 BPE decoder behind the scheduler (`BRAIN_QWEN_WEIGHTS` +
/// `BRAIN_QWEN_TOKENIZER` name WHICH checkpoint; [`QwenServeConfig`]'s
/// `--qwen-*` flags configure HOW to serve it). Runs the CPU/GPU forward
/// `generate` path (never the NPU branch).
pub struct QwenResident {
    id: String,
    path: String,
    tokenizer: String,
    /// A named LoRA adapter's own weight file (`qwen3::lora::save_adapter`'s
    /// output), when this resident is the ADAPTER's catalog entry rather than
    /// the base's -- folded into the base tensors at `activate` (see that
    /// method's doc). `None` for a plain base/quant resident.
    ///
    /// `RwLock`, not a plain `Option<String>`: `activate(&self, ..)` only
    /// ever gets `&self` (this resident is registered once as `Arc<dyn
    /// ResidentModel>` and shared), but the self-improve continuous-training
    /// hot-swap path (roadmap P4/P5) needs to point an ALREADY-registered
    /// resident at a newly-trained adapter file without re-registering under
    /// the same model id (which would duplicate the manifest entry --
    /// `Executor::register` has no update-in-place semantics, only
    /// `register`/`register_if_absent`). [`Self::set_adapter`] writes here;
    /// callers pair it with `Executor::evict` (evicts the stale Hot/Warm
    /// instance so the NEXT claim's `activate` reads the new value -- a
    /// currently-running request is never interrupted, `evict`'s own
    /// pinned-refusal contract) to actually take effect.
    adapter: std::sync::RwLock<Option<String>>,
    /// Resolved once at construction (see [`Self::resolve_ctx`]) - explicit,
    /// auto-sized, or the historical 24576 fallback. Fixed for this
    /// resident's whole lifetime, exactly like every other field below.
    ctx: u32,
    max_batch: u32,
    kv_int8_requested: bool,
    kv_calib_opt_in: bool,
    kv_offload_bytes: u64,
    weights_int8_requested: bool,
    max_prefill_cap: u32,
}

impl QwenResident {
    /// `cfg` carries every `--qwen-*` flag `brain serve` parsed; model
    /// SELECTION stays the two env vars named below (see
    /// [`QwenServeConfig`]'s own doc for why that split).
    pub fn from_env(cfg: QwenServeConfig) -> Option<QwenResident> {
        let path = std::env::var("BRAIN_QWEN_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        let tokenizer = std::env::var("BRAIN_QWEN_TOKENIZER").ok().unwrap_or_default();
        // See GptResident::from_env's comment: env-loaded, no upstream provenance.
        Some(Self::from_card_configured(&path, &ModelCard::new("brain/qwen3", "qwen"), Some(&tokenizer), None, cfg))
    }

    /// [`Self::from_card_configured`] at every historical default -
    /// `QwenServeConfig::default()` reproduces the old env-var defaults
    /// exactly, so every existing caller that has no opinion about serving
    /// tunables (tests, `perf_cli.rs`, `model_dir.rs`'s catalog scan,
    /// `continuous_train.rs`) is unaffected by this constructor's own
    /// existence.
    pub fn from_card(path: &str, card: &ModelCard, tokenizer: Option<&str>, adapter: Option<&str>) -> QwenResident {
        Self::from_card_configured(path, card, tokenizer, adapter, QwenServeConfig::default())
    }

    /// Construct under the card's id. `tokenizer` is the sibling `tokenizer.json`
    /// (empty/None defers the "set a tokenizer" error to `activate`). `adapter`
    /// is the adapter's own weight file when `card.id` names one
    /// (`brain_modelstore::LocalModel::adapter`) -- `None` for a plain base.
    /// `cfg` resolves `ctx` once, here (see [`Self::resolve_ctx`]), and is
    /// otherwise copied straight into fields `estimate`/`activate` read.
    pub fn from_card_configured(path: &str, card: &ModelCard, tokenizer: Option<&str>, adapter: Option<&str>, cfg: QwenServeConfig) -> QwenResident {
        let ctx = Self::resolve_ctx(path, &cfg);
        QwenResident {
            id: card.id.clone(),
            path: path.to_string(),
            tokenizer: tokenizer.unwrap_or_default().to_string(),
            adapter: std::sync::RwLock::new(adapter.filter(|a| !a.is_empty()).map(str::to_string)),
            ctx,
            max_batch: cfg.max_batch.max(1),
            kv_int8_requested: cfg.kv_int8,
            kv_calib_opt_in: cfg.kv_calib_opt_in,
            kv_offload_bytes: (cfg.kv_offload_gb.max(0.0) * (1u64 << 30) as f64) as u64,
            weights_int8_requested: cfg.weights_int8,
            max_prefill_cap: cfg.max_prefill_cap.clamp(1, 512),
        }
    }

    /// `cfg.ctx` verbatim when the operator named one explicitly
    /// (`--qwen-ctx`); else, when `cfg.auto_budget_bytes` names a real
    /// device budget (only `brain serve`'s own caller ever does), the
    /// largest [`AUTO_CTX_TIERS`] entry whose real total fits it - no longer
    /// capped at the checkpoint's own trained `max_position_embeddings`
    /// (see [`apply_yarn_if_serving_past_native`], called from `activate`,
    /// for what makes serving past it a real, not just a memory-shaped,
    /// capability); else the historical 24576 default, unchanged. Must never
    /// itself hard-fail - a checkpoint this can't open just falls back to
    /// the historical default and defers the real, specific error to
    /// `activate`'s own open, exactly like `estimate` already does.
    fn resolve_ctx(path: &str, cfg: &QwenServeConfig) -> u32 {
        const HISTORICAL_DEFAULT: u32 = 24576;
        if let Some(explicit) = cfg.ctx {
            return explicit.max(1);
        }
        let Some(budget) = cfg.auto_budget_bytes else { return HISTORICAL_DEFAULT };
        if path.to_ascii_lowercase().ends_with(".gguf") {
            return HISTORICAL_DEFAULT; // the Legacy non-paged decode path - no pool to auto-size
        }
        let Ok(reader) = checkpoint::weightio::WeightReader::open(path) else { return HISTORICAL_DEFAULT };
        let checkpoint_cfg = qwen3::config::QwenConfig::from_json(&reader.config());
        let weight_bytes = if cfg.weights_int8 { weights_int8_bytes(&checkpoint_cfg) } else { est_vram(path).vram };
        auto_ctx_for_budget(weight_bytes, budget, &checkpoint_cfg, cfg.max_batch.max(1), cfg.max_prefill_cap.clamp(1, 512), cfg.kv_int8)
    }

    /// Give `cfg` a derived [`model::yarn::YarnConfig`] when `ctx` (the KV
    /// pool `activate` is actually about to build - `self.ctx`, whether from
    /// `--qwen-ctx` or auto-sizing) exceeds the checkpoint's own trained
    /// `max_position_embeddings` and nothing already named a scaling - the
    /// same `extended / original` ratio HF's own `config.json` producers use
    /// for `rope_scaling.factor`. A checkpoint that already declares its own
    /// `rope_scaling` (a real long-context release) is never second-guessed.
    ///
    /// A DELIBERATE, resident-level decision, not something `qwen3::serve`
    /// derives on its own from its pool-sizing parameters: an `Engine`
    /// constructed directly (a test fixture, `qwen_bench`, `perf_cli`) sizes
    /// its pool however is numerically convenient for THAT caller, with no
    /// relation to "please extend this checkpoint's real context" - inferring
    /// intent from bare pool geometry there would auto-opt every such caller
    /// into YaRN by accident. Only `activate`'s own `ctx` - `resolve_ctx`'s
    /// real output, `--qwen-ctx`/auto-sizing's own answer - means that.
    fn apply_yarn_if_serving_past_native(cfg: &mut qwen3::config::QwenConfig, ctx: u32) {
        if cfg.rope_scaling.is_some() {
            return;
        }
        let native = cfg.max_position_embeddings.max(1);
        if ctx > native {
            cfg.rope_scaling = Some(model::yarn::YarnConfig::new(ctx as f32 / native as f32, native));
        }
    }

    /// Point this ALREADY-registered resident at a different (or no) LoRA
    /// adapter file - the self-improve continuous-training hot-swap write
    /// side (see [`Self::adapter`]'s doc for the full contract). Takes
    /// effect for the next `activate` - pair with `Executor::evict(self.
    /// instance_key(...))` (or just `Executor::evict(InstanceKey::new(&self.
    /// id, "default"))`) so a cached Hot/Warm instance from the OLD adapter
    /// isn't reused first. `crate::continuous_train::swap_in_adapter` does
    /// exactly that pairing.
    pub fn set_adapter(&self, adapter: Option<String>) {
        *self.adapter.write().unwrap() = adapter.filter(|a| !a.is_empty());
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
    fn pool_sizing(ctx: u32, max_batch: u32, max_prefill_cap: u32) -> (u32, u32, u32, u32, u32) {
        let block_size = 16u32;
        let max_blocks_per_seq = ctx.div_ceil(block_size);
        let num_blocks = max_blocks_per_seq * 2 + max_batch;
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
    fn check_fp32_kv_pool_fits(cfg: &qwen3::config::QwenConfig, block_size: u32, num_blocks: u32, ctx: u32, path: &str) -> Result<(), String> {
        let pool_bytes = qwen3::serve::kv_pool_bytes(cfg, block_size, num_blocks, false);
        if pool_bytes <= MAX_FP32_KV_POOL_BYTES {
            return Ok(());
        }
        let int8_bytes = qwen3::serve::kv_pool_bytes(cfg, block_size, num_blocks, true);
        Err(format!(
            "qwen: {path}: fp32 KV pool at ctx={ctx} would be {:.2} GiB, over the {:.0} GiB safety ceiling \
             (MAX_FP32_KV_POOL_BYTES) -- lower --qwen-ctx, or drop --qwen-kv-fp32 \
             so int8 KV (~{:.2} GiB at this ctx) is used instead",
            pool_bytes as f64 / (1u64 << 30) as f64,
            MAX_FP32_KV_POOL_BYTES as f64 / (1u64 << 30) as f64,
            int8_bytes as f64 / (1u64 << 30) as f64,
        ))
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
    /// `fused_prefill_available`'s own note on the scratch term), the same
    /// discipline every other pre-flight number in this file already
    /// follows.
    ///
    /// `fused_prefill_available`: unlike `estimate()` (truly pre-placement,
    /// no device chosen yet, always `false`), `activate()` already has a
    /// concrete `device` by the time it calls this - `device != Device::Cpu`
    /// is a safe proxy for the fused kernel's real gate
    /// (`caps.workgroup_reductions`, true on every GPU backend, false only
    /// on the CPU JIT), so passing `false` here unconditionally caused a
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
        fused_prefill_available: bool,
    ) -> Result<(), String> {
        let kv_bytes = qwen3::serve::kv_pool_bytes(cfg, block_size, num_blocks, kv_int8);
        let per_layer_kv_buffer = kv_bytes / cfg.n_layers as u64 / 2;
        let scratch_bytes = qwen3::serve::paged_attn_scratch_bytes(cfg, max_batch, max_prefill, cap, fused_prefill_available);
        let per_scratch_buffer = scratch_bytes / 2;
        let worst = per_layer_kv_buffer.max(per_scratch_buffer);
        if worst <= WEBGPU_MIN_STORAGE_BINDING_BYTES {
            return Ok(());
        }
        Err(format!(
            "qwen: {path}: at ctx={ctx} (kv_int8={kv_int8}) the largest single device buffer this engine \
             would allocate is {:.2} GiB, over WebGPU's own {:.2} GiB per-buffer floor -- lower --qwen-ctx \
             or --qwen-max-prefill",
            worst as f64 / (1u64 << 30) as f64,
            WEBGPU_MIN_STORAGE_BINDING_BYTES as f64 / (1u64 << 30) as f64,
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
        let (block_size, max_batch, max_blocks_per_seq, num_blocks, max_prefill) = Self::pool_sizing(self.ctx, self.max_batch, self.max_prefill_cap);
        let kv_int8 = self.kv_int8_requested && qwen3::serve::kv_int8_supported(&cfg);
        let kv_bytes = qwen3::serve::kv_pool_bytes(&cfg, block_size, num_blocks, kv_int8);
        // The paged-attention scores/probs scratch (`Scratch::{scores,probs}`)
        // is a real, often multi-GiB device allocation `activate()` makes
        // alongside the KV pool - omitting it here left placement relying on
        // `--reserve-gb`'s flat, model-agnostic headroom to accidentally
        // cover a real, model-specific need it was never sized for, on top
        // of an ALREADY doubly-conservative weight estimate (`est_vram`'s
        // static 1.3x). `fused_prefill_available` needs the target device's
        // own `DeviceCaps`, unknown at this pre-placement point - `false` is
        // the larger of the two sizes `paged_attn_scratch_bytes` can return
        // (see its own doc), so this stays a worst-case, never-under bound.
        let cap = max_blocks_per_seq * block_size;
        let scratch_bytes = qwen3::serve::paged_attn_scratch_bytes(&cfg, max_batch, max_prefill, cap, false);
        // `--weights-int8`'s equivalent (`BRAIN_QWEN_WEIGHTS_INT8`): the
        // weight term is by far the dominant one for any sizeable model, and
        // `weights_int8_bytes` reads it precisely off `cfg` - the SAME
        // canonical-name derivation `decoder_param_list` builds the engine
        // from - rather than a flat fraction of `cost.vram`. `activate()`
        // below requests the identical `weights_int8` value, so the two
        // cannot silently disagree about which tensors are quantized.
        let weight_bytes = if self.weights_int8_requested { weights_int8_bytes(&cfg) } else { cost.vram };
        residency::log::info(&format!(
            "{}: estimate weights={:.2}GiB(int8={}) kv={:.2}GiB scratch={:.2}GiB total={:.2}GiB",
            self.id,
            weight_bytes as f64 / (1u64 << 30) as f64,
            self.weights_int8_requested,
            kv_bytes as f64 / (1u64 << 30) as f64,
            scratch_bytes as f64 / (1u64 << 30) as f64,
            (weight_bytes + kv_bytes + scratch_bytes) as f64 / (1u64 << 30) as f64,
        ));
        MemCost::new(weight_bytes + kv_bytes + scratch_bytes, cost.ram)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // Coarse stage progress -- NOT per-tensor (the actual weight upload
        // loop lives in `paramstore`, a lower-level crate that must not
        // depend on `residency` just for this), a handful of stage markers so
        // `-v -v` shows SOMETHING moving during a cold activate that can take
        // over a minute, without turning into a per-layer scroll.
        let stage_t0 = std::time::Instant::now();
        residency::log::info(&format!("{}: step 1/3 opening checkpoint", self.id));
        // Stream weights from the mmap (see GptResident::activate). Open first so
        // a GGUF can supply its own embedded tokenizer.
        let reader = checkpoint::weightio::WeightReader::open(&self.path).map_err(|e| format!("qwen: {e}"))?;
        gpu_core::profile::stage_time(&format!("{}: open checkpoint", self.id), stage_t0);
        let stage_t0 = std::time::Instant::now();
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
        gpu_core::profile::stage_time(&format!("{}: load tokenizer", self.id), stage_t0);
        let stage_t0 = std::time::Instant::now();
        residency::log::info(&format!("{}: step 3/3 building engine (uploading weights to {device:?})", self.id));
        let ctx = self.ctx;
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
            let (block_size, max_batch, max_blocks_per_seq, num_blocks, max_prefill) = QwenResident::pool_sizing(ctx, self.max_batch, self.max_prefill_cap);
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
            // opt-out (--qwen-kv-fp32) or an unsupported head_dim -- the
            // int8 default at the SAME ctx does not come close.
            if !kv_int8 {
                QwenResident::check_fp32_kv_pool_fits(&checkpoint_cfg, block_size, num_blocks, ctx, &self.path)?;
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
            let fused_prefill_available = device != Device::Cpu && checkpoint_cfg.head_dim <= qwen3::serve::FUSED_MAX_HEAD_DIM;
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
                fused_prefill_available,
            )?;
            // `--qwen-weights-int8` -- requested, not asserted: the real
            // capability gate (`caps.numeric.int8_dot`) lives inside
            // `Engine::from_map_with_gpu` itself, which degrades to fp32
            // weights with its own loud fallback message when a device (the
            // CPU JIT, or an unusual GPU) has no packed-int8 path, exactly
            // the same shape `kv_int8`'s own request/degrade split already
            // uses. `estimate()` above computes its budget from this SAME
            // field, so the two cannot silently disagree about what was
            // asked for.
            let weights_int8 = self.weights_int8_requested;
            // Snapshot the adapter path under the lock, then release it before
            // the (potentially slow) fold/load below -- `set_adapter` must
            // never block on an in-progress activation, and this activation
            // must not hold up a concurrent `set_adapter` either.
            let adapter_path = self.adapter.read().unwrap().clone();
            let mut eng = match &adapter_path {
                None => {
                    // `Engine::load` re-derives its own `QwenConfig` straight
                    // from the file, with no seam for `ctx`'s own YaRN
                    // decision below to reach it -- go through the same
                    // load-tensors-then-`from_map` shape the adapter branch
                    // already uses so both branches share one cfg mutation
                    // point.
                    let load_t0 = std::time::Instant::now();
                    let tensors = checkpoint::load(&self.path).into_by_role("");
                    gpu_core::profile::stage_time(&format!("{}: re-load checkpoint tensors for engine build", self.id), load_t0);
                    let mut cfg = checkpoint_cfg.clone();
                    Self::apply_yarn_if_serving_past_native(&mut cfg, ctx);
                    qwen3::serve::Engine::from_map(cfg, &tensors, block_size, num_blocks, max_batch, max_blocks_per_seq, max_prefill, kv_int8, weights_int8)
                }
                // Fold the adapter's delta into the base tensors first (the same
                // fold `qwen3::eval::score_chat` uses to score one) -- the result
                // is an ordinary frozen base, zero extra inference cost versus
                // the base once folded.
                Some(a) => {
                    let mut tensors = checkpoint::load(&self.path).into_by_role("");
                    let mut cfg = qwen3::config::QwenConfig::from_json(&reader.config());
                    qwen3::lora::fold_adapter_into(&mut tensors, a).map_err(|e| format!("qwen: folding adapter {a}: {e}"))?;
                    cfg.lora = None;
                    Self::apply_yarn_if_serving_past_native(&mut cfg, ctx);
                    qwen3::serve::Engine::from_map(cfg, &tensors, block_size, num_blocks, max_batch, max_blocks_per_seq, max_prefill, kv_int8, weights_int8)
                }
            };
            // --qwen-kv-calib: opt IN to a kv_calib.json beside the
            // BASE checkpoint (self.path, not the adapter's own file, since
            // the adapter is folded into the base's K/V distribution) --
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
            // `--qwen-max-batch` slots, a busy server can admit far more
            // sessions than the pool can hold cached at once -- today that ends
            // as a hard "KV pool exhausted" mid-decode. Given host RAM, the
            // scheduler instead parks the sessions it is not advancing and
            // brings them back byte-identical. Off by default (0): it spends
            // host memory, and a box that has none to spare must not be made
            // to.
            eng.set_kv_offload_bytes(self.kv_offload_bytes);
            Ok(QwenEngineKind::Batched(Box::new(model::serve::Scheduler::new(eng, max_batch as usize))))
        })??;
        gpu_core::profile::stage_time(&format!("{}: build engine (total, on_device)", self.id), stage_t0);
        Ok(Box::new(QwenInstance { tok, eos, engine }))
    }
}

/// Which serving path this instance drives - see [`QwenResident::activate`]'s
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
    /// driven to completion together - real continuous batching for
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
    /// `Executor::stats().metrics` - reachable from HTTP/D-Bus for the first
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

/// One full generation on the legacy single-sequence decode path - the exact
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
/// `Scheduler` - see [`QwenInstance::run_batch`]'s doc.
fn run_batch_scheduled(
    sched: &mut model::serve::Scheduler<qwen3::serve::Engine>,
    tok: &data::qwen_tokenizer::QwenBpe,
    eos: Option<u32>,
    invs: &[Invocation],
    progress: &mut dyn FnMut(usize, Progress),
) -> Vec<ActionResult> {
    // The scheduler-level batch: how many concurrent requests residency
    // bundled onto this SAME persistent `Scheduler` pass, as opposed to
    // `bridge.rs`'s per-HTTP-request log, which knows nothing about
    // scheduler-side batching at all.
    residency::log::info(&format!("Qwen batch: {} request(s) admitted to the scheduler", invs.len()));
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
        // to reap it naturally - `Scheduler::cancel` returns its tokens so
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
        // Per-step scheduling detail: gated at level 3 ("finer scheduling
        // detail" per `run_serve`'s HELP text) and the format cost skipped
        // below it - a continuous-batching decode loop calls this once per
        // TOKEN, so an unconditional `format!` here would cost real
        // throughput even with nothing printed.
        if residency::log::verbosity() >= 3 {
            residency::log::debug(&format!(
                "Qwen batch step: {} running, {} waiting, admitted={}, produced={}, finished={}, rejected={}, demoted={}, promoted={}",
                sched.running_len(),
                sched.waiting_len(),
                report.admitted.len(),
                report.produced.len(),
                report.finished.len(),
                report.rejected.len(),
                report.demoted.len(),
                report.promoted.len(),
            ));
        }
        // A request the scheduler refuses at admission (a prompt token
        // outside its vocabulary, or one that can never fit its per-sequence
        // capacity - see `model::serve::RejectReason`) never appears in
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
    #[test]
    fn qwen_serve_config_default_matches_every_historical_default() {
        let cfg = QwenServeConfig::default();
        assert_eq!(cfg.ctx, None, "None means auto-size - see resolve_ctx");
        assert_eq!(cfg.max_batch, 16);
        assert!(cfg.kv_int8, "int8 KV must be the default");
        assert!(!cfg.kv_calib_opt_in, "calibration must default OFF");
        assert_eq!(cfg.kv_offload_gb, 0.0);
        assert!(!cfg.weights_int8, "int8 weights must default OFF");
        assert_eq!(cfg.max_prefill_cap, 512);
    }

    /// `ctx` at or under the checkpoint's own trained window is a pure no-op
    /// - the overwhelming default (auto-sizing rarely needs to go past
    /// native, an explicit `--qwen-ctx` usually doesn't either) must never
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

    /// `ctx` past the native window derives a real `YarnConfig` whose
    /// `factor` is exactly `ctx / native` (HF's own `config.json` ratio) and
    /// whose `original_max_position_embeddings` is the checkpoint's real
    /// native window, not the requested `ctx`.
    #[test]
    fn apply_yarn_if_serving_past_native_derives_the_extended_over_original_ratio() {
        let mut cfg = qwen3::config::QwenConfig::qwen3_8b();
        let native = cfg.max_position_embeddings;
        cfg.rope_scaling = None;
        let ctx = native * 4;
        QwenResident::apply_yarn_if_serving_past_native(&mut cfg, ctx);
        let y = cfg.rope_scaling.expect("ctx past native must derive a YarnConfig");
        assert_eq!(y.factor, 4.0);
        assert_eq!(y.original_max_position_embeddings, native);
    }

    /// A checkpoint that already names its own `rope_scaling` (a real
    /// published long-context release) must never be second-guessed, even
    /// when `ctx` also exceeds its native window.
    #[test]
    fn apply_yarn_if_serving_past_native_never_overrides_an_already_declared_scaling() {
        let mut cfg = qwen3::config::QwenConfig::qwen3_8b();
        let native = cfg.max_position_embeddings;
        let declared = model::yarn::YarnConfig::new(2.0, native);
        cfg.rope_scaling = Some(declared);
        QwenResident::apply_yarn_if_serving_past_native(&mut cfg, native * 10);
        assert_eq!(cfg.rope_scaling, Some(declared), "an already-declared rope_scaling must survive untouched");
    }

    #[test]
    fn max_prefill_cap_is_clamped_at_construction() {
        let card = checkpoint::st::ModelCard::new("brain/qwen3", "qwen");
        for (requested, expected) in [(128u32, 128u32), (0, 1), (999_999, 512)] {
            let cfg = QwenServeConfig { max_prefill_cap: requested, ..QwenServeConfig::default() };
            let resident = QwenResident::from_card_configured("unused.safetensors", &card, Some("unused.json"), None, cfg);
            assert_eq!(resident.max_prefill_cap, expected, "requested {requested}");
        }
    }

    /// End to end at the REAL Qwen3-8B config: lowering `--qwen-max-prefill`
    /// shrinks `paged_attn_scratch_bytes` (3.00 of an 18.41 GiB device total
    /// at the operator's real ctx=24576) linearly, with zero change to
    /// `kv_pool_bytes` (prefill chunk size never enters the KV pool's own
    /// sizing) and zero change to decode-step shape (`max_prefill` only ever
    /// appears in the causal-chunk PREFILL term).
    ///
    /// This is the `fused_prefill_available = false` branch, which is what
    /// every call site in THIS file deliberately prices (see `estimate`'s own
    /// comment: no target `DeviceCaps` exists pre-placement, so the never-
    /// under bound is the only honest one). On a device that can actually run
    /// the fused kernels, `Engine::from_map_with_gpu` allocates the OTHER
    /// branch, where `max_prefill` does not appear at all and the buffer is
    /// `max_batch`-sized instead - `qwen3::serve`'s own
    /// `paged_attn_scratch_at_the_real_8b_serving_default_shape` pins both
    /// arms at this same shape. So `--qwen-max-prefill` moves the pre-
    /// placement budget and the triad-path allocation; it does not move a
    /// fused-path one.
    #[test]
    fn lowering_max_prefill_shrinks_scratch_linearly_at_the_real_config() {
        let cfg = qwen3::config::QwenConfig::qwen3_8b();
        let ctx = 24576u32;
        let max_batch = 16u32;

        let (block_size, max_batch, max_blocks_per_seq, num_blocks, max_prefill_default) = QwenResident::pool_sizing(ctx, max_batch, 512);
        assert_eq!(max_prefill_default, 512);
        let cap = max_blocks_per_seq * block_size;
        let scratch_default = qwen3::serve::paged_attn_scratch_bytes(&cfg, max_batch, max_prefill_default, cap, false);
        let kv_default = qwen3::serve::kv_pool_bytes(&cfg, block_size, num_blocks, true);

        let (_, max_batch2, _, num_blocks2, max_prefill_128) = QwenResident::pool_sizing(ctx, max_batch, 128);
        assert_eq!(max_prefill_128, 128);
        let scratch_128 = qwen3::serve::paged_attn_scratch_bytes(&cfg, max_batch2, max_prefill_128, cap, false);
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
    /// out (the default) leaves it exactly at `est_vram`'s file-size figure
    /// - same env-gated shape `kv_int8`'s own estimate switch already uses.
    #[test]
    fn estimate_shrinks_the_weight_term_when_weights_int8_is_requested() {
        let path = write_tiny_checkpoint(29, "estimate-weights-int8");
        let card = checkpoint::st::ModelCard::new("brain/qwen3", "qwen");
        let key = InstanceKey::new("brain/qwen3", "default");

        let fp32_resident = QwenResident::from_card(path.to_str().unwrap(), &card, Some("unused.json"), None);
        let fp32_total = fp32_resident.estimate(&key).vram;

        let int8_cfg = QwenServeConfig { weights_int8: true, ..QwenServeConfig::default() };
        let int8_resident = QwenResident::from_card_configured(path.to_str().unwrap(), &card, Some("unused.json"), None, int8_cfg);
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
        assert!(QwenResident::check_buffers_fit_one_binding(&cfg, 16, 64, 4, 16, 256, true, 2048, "test.safetensors", false).is_ok());
        assert!(QwenResident::check_buffers_fit_one_binding(&cfg, 16, 64, 4, 16, 256, false, 2048, "test.safetensors", false).is_ok());
        // num_blocks=3,000,000 -> per-layer fp32 KV buffer ~2.86 GiB, over
        // the 2047 MiB floor - even though this SAME sizing's fp32 TOTAL
        // pool (~11.4 GiB) is what `check_fp32_kv_pool_fits`'s own test
        // above already catches; this checks the independent per-buffer path.
        // The KV term does not depend on `fused_prefill_available` at all,
        // so this must still refuse it under `true` too.
        for fused in [false, true] {
            let err = QwenResident::check_buffers_fit_one_binding(&cfg, 16, 3_000_000, 4, 16, 256, false, 100_000_000, "test.safetensors", fused)
                .expect_err("a ~2.86 GiB single buffer must be refused, not attempted");
            assert!(err.contains("test.safetensors") && err.contains("GiB"), "{err}");
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
        let err = QwenResident::check_buffers_fit_one_binding(&cfg, 16, 64, 4, 20_000, 320_000, true, 100_000_000, "test.safetensors", false)
            .expect_err("a ~5.96 GiB scratch buffer must be refused, not attempted");
        assert!(err.contains("test.safetensors") && err.contains("GiB"), "{err}");
    }

    /// REGRESSION target: this exact shape (Qwen3-8B, ctx=32768, the default
    /// max_prefill=512) reached a real production refusal - "largest single
    /// device buffer ... 2.00 GiB, over WebGPU's own 2.00 GiB per-buffer
    /// floor" - on a live GPU that was actually about to dispatch the fused
    /// prefill kernel (a much smaller scratch buffer), because `activate()`
    /// passed `fused_prefill_available=false` unconditionally even once it
    /// already knew a concrete, non-CPU `device`. Proves the SAME sizing
    /// that this guard must refuse under `false` (the pre-fix behavior) it
    /// must now ACCEPT under `true`.
    #[test]
    fn fused_prefill_available_shrinks_the_scratch_term_enough_to_admit_the_real_qwen3_8b_ctx_32768_shape() {
        let cfg = qwen3::config::QwenConfig::qwen3_8b();
        let (block_size, max_batch, max_blocks_per_seq, num_blocks, max_prefill) = QwenResident::pool_sizing(32768, 16, 512);
        let cap = max_blocks_per_seq * block_size;
        assert!(
            QwenResident::check_buffers_fit_one_binding(&cfg, block_size, num_blocks, max_batch, max_prefill, cap, true, 32768, "test.safetensors", false).is_err(),
            "this test's premise is the real reported refusal under the never-fused worst case"
        );
        assert!(
            QwenResident::check_buffers_fit_one_binding(&cfg, block_size, num_blocks, max_batch, max_prefill, cap, true, 32768, "test.safetensors", true).is_ok(),
            "the fused kernel's real scratch buffer must fit, once activate() actually knows the device supports it"
        );
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
        let resident = QwenResident::from_card("unused.safetensors", &card, Some("unused.json"), None);
        assert_eq!(resident.ctx, ctx, "this test's premise is the new default -- update it if the default changes");
        let (block_size, _max_batch, _max_blocks_per_seq, num_blocks, _max_prefill) = QwenResident::pool_sizing(ctx, 16, 512);

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
        // would hit under `--qwen-kv-fp32` at this ctx.
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
        let resident = QwenResident::from_card("unused.safetensors", &card, Some("unused.json"), None);
        let ctx = resident.ctx;
        assert_eq!(ctx, 24576, "this test's premise is the new default");
        let (block_size, max_batch, max_blocks_per_seq, num_blocks, max_prefill) = QwenResident::pool_sizing(ctx, 16, 512);
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
        let resident = QwenResident::from_card(path.to_str().unwrap(), &card, Some("unused.json"), None);
        let key = InstanceKey::new("brain/qwen3", "default");

        let (block_size, max_batch, max_blocks_per_seq, num_blocks, max_prefill) = QwenResident::pool_sizing(resident.ctx, resident.max_batch, resident.max_prefill_cap);
        let kv_int8 = resident.kv_int8_requested;
        let expected_kv_bytes = qwen3::serve::kv_pool_bytes(&cfg, block_size, num_blocks, kv_int8);
        let cap = max_blocks_per_seq * block_size;
        let expected_scratch_bytes = qwen3::serve::paged_attn_scratch_bytes(&cfg, max_batch, max_prefill, cap, false);
        let file_only = est_vram(path.to_str().unwrap()).vram;

        let got = resident.estimate(&key);
        assert_eq!(
            got.vram,
            file_only + expected_kv_bytes + expected_scratch_bytes,
            "estimate() must equal file size + the KV pool + the paged-attention scratch, no more and no less"
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
        let results = run_batch_scheduled(&mut sched, &tok, None, std::slice::from_ref(&inv), &mut |_i, _p| {});
        assert!(start.elapsed() < std::time::Duration::from_secs(5), "rejected admission must resolve promptly, not hang");
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err(), "an out-of-vocab prompt must be reported as an error, not silently dropped");
    }
}
