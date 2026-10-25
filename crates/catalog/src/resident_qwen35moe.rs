// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident-model adapter putting Qwen3.5-35B-A3B's paged serving engine
//! (`qwen35moe::serve::Engine`/`Scheduler`) behind the residency [`Executor`],
//! mirroring `crate::resident_llm::QwenResident`'s `Batched` arm — SCOPED
//! DOWN to exactly what `qwen35moe::serve::Engine` actually is (see that
//! module's own doc for the authoritative "deliberately deferred" list):
//! **single-GPU**, **f32 or packed-int8 expert weights as the
//! checkpoint stores them** (`serving_engine` reads the file through its packed
//! view; a bf16 checkpoint is decoded one tensor at a time to f32), **fp32 KV** (no int8 KV path exists
//! for this engine, unlike `qwen3::serve::Engine`'s `kv_int8` default).
//! Every resident sequence's decode row goes through one batched forward per
//! scheduler iteration (the generic `model::serve::Scheduler`, driven by
//! `llmserve`).
//!
//! Not present, because they don't exist anywhere in this crate yet: LoRA
//! adapter folding (`qwen3::lora::fold_adapter_into` has no `qwen35moe`
//! counterpart), multi-GPU sharding, a prefix cache that survives past what
//! `Scheduler`'s own `PrefixCache` already provides at the paged-block
//! granularity `Engine`'s single-block-per-sequence layout allows (see
//! `qwen35moe::serve`'s module doc on `block_size == max_seq_len`), and a
//! `.gguf` serving path (`Engine` only loads brain-native safetensors via
//! the packed source view of the file, unlike `QwenResident`'s `Legacy` GGUF arm) - so unlike
//! `QwenResident` there is only ONE engine kind here, not a `Legacy`/`Batched`
//! split.
//!
//! Two ways in. [`Qwen35Resident::from_card`] serves a checkpoint the
//! model-dir scan found (family `"qwen35moe"` - what `brain import-gguf`'s
//! conversion stamps, distinct from the dense sibling's own `"qwen35"`
//! family, `crates/qwen35`), under its own card id and with its sibling
//! `tokenizer.json`; no env vars involved. The env path below is the manual
//! alternative, for a checkpoint outside the models directory.
//!
//! Env config follows `BRAIN_QWEN_*`'s naming convention:
//!   * `BRAIN_QWEN35MOE_WEIGHTS` — a brain-format Qwen3.5-35B-A3B checkpoint
//!     (`.safetensors`, f32, bf16 or packed int8 - what `brain
//!     import-gguf` writes). The primary gate; unset ⇒ not served.
//!   * `BRAIN_QWEN35MOE_TOKENIZER` - the sibling `tokenizer.json`. If unset,
//!     `activate()` falls back to a GGUF checkpoint's own embedded
//!     `tokenizer.ggml.*` KV (see `crate::resident_llm::QwenResident::
//!     activate`, the sibling this mirrors) before giving up - `Engine`
//!     itself still never touches a `.gguf` file at all, so a GGUF
//!     checkpoint gets a real tokenizer but no further than that today.
//!   * `BRAIN_QWEN35MOE_CTX` — the hard `prompt + max_new` cap for any ONE
//!     sequence (`Engine::from_map`'s `max_seq_len`, which this engine also
//!     uses as its per-sequence block size — see `qwen35moe::serve`'s module
//!     doc). Default 4096: far more conservative than `QwenResident`'s 24576
//!     default, since every block here costs a WHOLE `max_seq_len`-sized GQA
//!     KV allocation per full-attention layer (no int8 KV to shrink it, and
//!     no sub-`max_seq_len` paging to share a big context economically across
//!     many short sequences).
//!   * `BRAIN_QWEN35MOE_MAX_BATCH` — how many sequences may be RESIDENT at
//!     once (`Engine::from_map`'s `max_concurrent`, i.e. `num_blocks` — NOT
//!     how many are dispatched together on the GPU per step, which is always
//!     1 for this engine). Default 4.

use capability::{Invocation, Manifest};
use std::sync::Arc;
use data::qwen_tokenizer::QwenBpe;
use qwen3::chat::parse_request;
use qwen35moe::config::{LayerType, Qwen35Config};
use qwen35moe::serve::{Engine, EngineOptions, Scheduler};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel, Tier};

use crate::parkable::{Build, HostWeights, ParkableInstance};
use crate::resident_llm::{est_vram, generate_spec, on_device, open_stored_checkpoint};
use llmserve::{ScheduledInstance, TextFrontend};

/// Catalog id: a `brain/`-prefixed synthetic id, matching `qwen35moe::caps::MODEL`
/// exactly (`GptResident::from_env`'s doc explains why an env-loaded
/// checkpoint with no upstream provenance gets this fallback rather than the
/// real `Qwen/Qwen3.5-35B-A3B` HF id).
const MODEL: &str = qwen35moe::caps::MODEL;

/// The Qwen3.5-35B-A3B hybrid Gated-DeltaNet/GQA sparse-MoE decoder behind
/// the scheduler (`BRAIN_QWEN35MOE_WEIGHTS` + `BRAIN_QWEN35MOE_TOKENIZER`).
/// See this module's own doc for the exact (single-GPU, fp32-only) scope.
pub struct Qwen35Resident {
    id: String,
    path: String,
    tokenizer: String,
    /// `path` is a HuggingFace checkpoint directory read in place (ModelOpt FP8/NVFP4, compressed-tensors or bf16
    /// safetensors, as published), not a brain-native file.
    hf_dir: bool,
}

impl Qwen35Resident {
    pub fn from_env() -> Option<Qwen35Resident> {
        let path = std::env::var("BRAIN_QWEN35MOE_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        let tokenizer = std::env::var("BRAIN_QWEN35MOE_TOKENIZER").ok().unwrap_or_default();
        Some(Qwen35Resident { id: MODEL.to_string(), path, tokenizer, hf_dir: false })
    }

    /// The model-dir counterpart of [`from_env`](Self::from_env): a checkpoint
    /// discovered by `serving::model_dir` (family `"qwen35moe"` - what
    /// `qwen35moe::import`'s conversion stamps on its `ModelCard`) served under
    /// its OWN card id rather than the env fallback [`MODEL`].
    ///
    /// `tokenizer` is the sibling `tokenizer.json` the scan found. It is
    /// required: `Engine` never opens a `.gguf`, so unlike `QwenResident` there
    /// is no embedded-tokenizer fallback (see this module's own doc) - a
    /// checkpoint without one would activate straight into an error, so it is
    /// declined at discovery instead, exactly like the `lfm` family.
    pub fn from_card(path: &str, card: &checkpoint::st::ModelCard, tokenizer: Option<&str>) -> Result<Qwen35Resident, String> {
        let tokenizer = tokenizer.filter(|t| !t.is_empty()).ok_or("qwen35moe: no sibling tokenizer.json")?;
        Ok(Qwen35Resident { id: card.id.clone(), path: path.to_string(), tokenizer: tokenizer.to_string(), hf_dir: false })
    }

    /// A HuggingFace checkpoint directory (`config.json`, `tokenizer.json` and safetensors shards, as pulled), served
    /// where it lies: every tensor reaches the card in the format it is stored in.
    pub fn from_hf_dir(dir: &str, card: &checkpoint::st::ModelCard) -> Result<Qwen35Resident, String> {
        let tokenizer = std::path::Path::new(dir).join("tokenizer.json");
        if !tokenizer.is_file() {
            return Err(format!("qwen35moe: {dir} has no tokenizer.json"));
        }
        Ok(Qwen35Resident { id: card.id.clone(), path: dir.to_string(), tokenizer: tokenizer.to_string_lossy().into_owned(), hf_dir: true })
    }

    /// The hard per-sequence `prompt + max_new` cap, which this engine also
    /// uses as its physical block size (`Engine::from_map`'s `max_seq_len`
    /// parameter — see this module's own doc on why the default is far more
    /// conservative than `QwenResident::ctx`'s).
    fn ctx() -> u32 {
        std::env::var("BRAIN_QWEN35MOE_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(4096u32).max(1)
    }

    /// How many sequences may be resident at once (`Engine::from_map`'s
    /// `max_concurrent`, i.e. its `num_blocks`), all decoded in one batched
    /// step.
    fn max_concurrent() -> u32 {
        std::env::var("BRAIN_QWEN35MOE_MAX_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(4u32).max(1)
    }
}

impl Qwen35Resident {
    /// Device bytes of a directory-served checkpoint: the stored bytes of its shards (FP8 and NVFP4 stay as stored), the
    /// embedding table widened to f32, plus the KV pool and recurrent state the engine allocates.
    fn estimate_hf_dir(&self) -> MemCost {
        let stored: u64 = std::fs::read_dir(&self.path)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();
        let Ok(cfg) = qwen35moe::hf_source::hf_config(std::path::Path::new(&self.path)) else {
            return MemCost::new(stored, 0);
        };
        let weights = stored + cfg.vocab as u64 * cfg.d_model as u64 * 2;
        let kv = qwen35moe::serve::kv_pool_bytes(&cfg, qwen35moe::serve::EngineOptions::new(Self::ctx(), Self::max_concurrent()).kv_tier, Self::max_concurrent(), Self::ctx());
        MemCost::new(weights + kv, 0)
    }

    /// Build the scheduler from the directory, in place: the config and the tensors are read from the published files.
    fn activate_hf_dir(&self, device: Device) -> Result<Box<dyn Instance>, String> {
        let dir = std::path::Path::new(&self.path);
        let cfg = qwen35moe::hf_source::hf_config(dir)?;
        let tok = QwenBpe::from_file(&self.tokenizer)?;
        let stop = data::generation::stop_ids(Some(dir), &tok, None, Some("<|im_end|>")).map_err(|e| format!("qwen35moe: {e}"))?;
        let (ctx, max_concurrent) = (Self::ctx(), Self::max_concurrent());
        let front = Arc::new(TextFrontend { tok, stop, label: "qwen35moe", parse: Arc::new(parse_request) });
        let src = qwen35moe::hf_source::open_dir(dir, &cfg)?;
        let names = src.names().into_iter().map(str::to_string).collect();
        let host = HostWeights::from_source(Box::new(src), names);
        let config = cfg.to_json();
        let build: Build = Box::new(move |host, device| {
            let sched = on_device(device, || engine_from(host.source(), &config, ctx, max_concurrent).map(|engine| Scheduler::new(engine, max_concurrent as usize)))??;
            Ok(Box::new(ScheduledInstance::new(front.clone(), sched, max_concurrent as usize)))
        });
        Ok(Box::new(ParkableInstance::activate(Arc::new(host), device, build)?))
    }
}

impl ResidentModel for Qwen35Resident {
    fn manifest(&self) -> Manifest {
        // `Self::ctx()` is exactly the `max_seq_len` `activate()` below builds
        // the engine with, so advertising it is a safe, never-overstated
        // floor on real serving capacity (`Manifest::max_context_tokens`'s own
        // doc on why this must be real engine capacity, not an architectural
        // maximum) — same reasoning as `QwenResident::manifest`.
        Manifest::new(
            &self.id,
            "text generation (Qwen3.5-35B-A3B hybrid Gated-DeltaNet/GQA sparse-MoE decoder; single-GPU, f32 or packed-int8 weights as the checkpoint stores them + fp32 KV, one active sequence at a time -- see crate::resident_qwen35moe's module doc)",
            vec![generate_spec("generate text (Qwen3.5-35B-A3B; chat template optional)", true)],
        )
        .with_max_context_tokens(Self::ctx() as u64)
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(&self.id, "default")
    }
    fn batch_limit(&self) -> Option<usize> {
        Some(Self::max_concurrent() as usize)
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        if self.hf_dir {
            return self.estimate_hf_dir();
        }
        let cost = est_vram(&self.path);
        let Ok(reader) = checkpoint::weightio::WeightReader::open(&self.path) else {
            return cost;
        };
        let cfg = Qwen35Config::from_json(&reader.config());
        let ctx = Self::ctx() as u64;
        let max_concurrent = Self::max_concurrent() as u64;
        let n_full = cfg.layer_types().iter().filter(|t| **t == LayerType::Full).count() as u64;
        // The GQA side of `Engine::kv_pool_bytes()` ONLY (`n_full * num_blocks
        // * 2 * block_size * kv_dim * 4`, `block_size == ctx` here) --
        // computed independently via `Qwen35Config`'s own public accessors
        // rather than calling into `qwen35moe::serve` (whose `GdnSlot` sizing
        // helper is crate-private, by design: it is an internal accounting
        // detail of `Engine`, not part of this crate's public surface).
        // Genuinely missing: the GDN recurrent-state/conv-history slot pool
        // (`num_blocks` slots, `GdnSlot::bytes` — private) is real but
        // small (O(1) state per sequence, not O(context length) like the GQA
        // side), so this estimate UNDER-counts by that (bounded, much
        // smaller) amount rather than over-claiming a number this crate
        // cannot cheaply verify from outside `serve.rs`.
        let gqa_bytes = n_full * max_concurrent * 2 * ctx * cfg.kv_dim() as u64 * 4;
        MemCost::new(cost.vram + gqa_bytes, cost.ram)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if self.hf_dir {
            return self.activate_hf_dir(device);
        }
        // Open first so a GGUF can supply its own embedded tokenizer, exactly
        // like crate::resident_llm::QwenResident::activate. The engine is built
        // from the stored checkpoint view ([`engine_from`]); this reader
        // exists for the tokenizer fallback.
        let reader = checkpoint::weightio::WeightReader::open(&self.path).map_err(|e| format!("qwen35moe: {e}"))?;
        // Tokenizer precedence: an explicit sibling tokenizer.json (or an
        // env override) wins; else a .gguf builds from its embedded
        // tokenizer.ggml.* KV; else there is nothing to tokenize with.
        let (tok, tok_dir, gguf_eos) = if !self.tokenizer.is_empty() {
            (QwenBpe::from_file(&self.tokenizer)?, std::path::Path::new(&self.tokenizer).parent(), None)
        } else if let Some(gt) = reader.tokenizer() {
            (QwenBpe::from_gguf(&gt).map_err(|e| format!("qwen35moe: {e}"))?, None, gt.eos)
        } else {
            return Err("qwen35moe: no tokenizer (set BRAIN_QWEN35MOE_TOKENIZER, or use a GGUF with an embedded tokenizer)".to_string());
        };
        // The checkpoint's own eos ids, plus ChatML's end-of-turn marker: this
        // resident renders every prompt as ChatML.
        let stop = data::generation::stop_ids(tok_dir, &tok, gguf_eos, Some("<|im_end|>")).map_err(|e| format!("qwen35moe: {e}"))?;
        let ctx = Self::ctx();
        let max_concurrent = Self::max_concurrent();
        let front = Arc::new(TextFrontend { tok, stop, label: "qwen35moe", parse: Arc::new(parse_request) });
        let (config, host) = HostWeights::open(&self.path, |config| Qwen35Config::from_json(config).param_list().into_iter().map(|(name, _)| name).collect())?;
        let build: Build = Box::new(move |host, device| {
            let sched = on_device(device, || engine_from(host.source(), &config, ctx, max_concurrent).map(|engine| Scheduler::new(engine, max_concurrent as usize)))??;
            Ok(Box::new(ScheduledInstance::new(front.clone(), sched, max_concurrent as usize)))
        });
        Ok(Box::new(ParkableInstance::activate(Arc::new(host), device, build)?))
    }
    /// A demoted instance keeps the checkpoint's packed bytes as a mapping of the file (evictable page cache), no
    /// f32 copy and no device memory.
    fn estimate_at(&self, key: &InstanceKey, tier: Tier) -> MemCost {
        match tier {
            Tier::Hot => self.estimate(key),
            Tier::Warm | Tier::Cold if self.hf_dir => MemCost::new(0, 0).with_mapped(self.estimate_hf_dir().vram),
            Tier::Warm | Tier::Cold => MemCost::new(0, 0).with_mapped(std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0)),
        }
    }
}

/// The serving engine over the brain checkpoint at `path` (see [`engine_from`]).
pub fn serving_engine(path: &str, ctx: u32, max_concurrent: u32) -> Result<Engine, String> {
    let (config, src) = open_stored_checkpoint(path)?;
    engine_from(&*src, &config, ctx, max_concurrent)
}

/// The serving engine over `src`, a brain checkpoint of config `config`, read one tensor at a time as the file
/// stores it: the routed experts stay packed int8 when the checkpoint holds them so (the engine implements the f32
/// and int8 tiers), and every other tensor is decoded as it is uploaded, never as a whole-model fp32 host copy.
fn engine_from(src: &dyn checkpoint::TensorSource, config: &serde_json::Value, ctx: u32, max_concurrent: u32) -> Result<Engine, String> {
    let cfg = Qwen35Config::from_json(config);
    let experts_stored = model::ops::stored_tier(src, "blocks.0.mlp.experts.0.gate.weight");
    let tier = model::ops::TierPolicy::uniform(if experts_stored == Some(gpu_core::select::Dtype::I8) { gpu_core::select::Dtype::I8 } else { gpu_core::select::Dtype::F32 });
    Ok(Engine::from_source(cfg, src, EngineOptions::new(ctx, max_concurrent).with_tier(tier)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `BRAIN_QWEN35MOE_CTX`/`BRAIN_QWEN35MOE_MAX_BATCH` parse with sane
    /// defaults and floor at 1 (matching `QwenResident::ctx`/`max_batch`'s own
    /// tests) — pure parsing, no env mutation (this crate's tests run
    /// concurrently; see `resident_llm`'s own tests for why env vars are
    /// never mutated directly in a test).
    #[test]
    fn ctx_and_max_concurrent_have_sane_defaults() {
        // Only assert defaults when the env vars are genuinely unset in this
        // process, mirroring the honesty `QwenResident`'s own env tests need
        // to keep (a real value on the runner would make this a lie).
        if std::env::var("BRAIN_QWEN35MOE_CTX").is_err() {
            assert_eq!(Qwen35Resident::ctx(), 4096);
        }
        if std::env::var("BRAIN_QWEN35MOE_MAX_BATCH").is_err() {
            assert_eq!(Qwen35Resident::max_concurrent(), 4);
        }
    }

    #[test]
    fn from_env_is_none_without_the_weights_var() {
        // SAFETY: no other test in this process reads/writes this exact var.
        unsafe { std::env::remove_var("BRAIN_QWEN35MOE_WEIGHTS") };
        assert!(Qwen35Resident::from_env().is_none());
    }

    /// REGRESSION for the whole point of this file: registering
    /// `Qwen35Resident` with the residency `Executor` must be enough, with NO
    /// `crates/apiserve`-side code, for `brain/qwen35moe` to appear on BOTH
    /// the OpenAI and the Anthropic `/v1/models` list with the chat
    /// capability — `apiserve::catalog::api_caps`/`exposed` derive this
    /// purely from `Manifest` shape (`generate_spec("...", true)` declares a
    /// streaming `generate` action with a `messages` param and a `Text`
    /// output — exactly `api_caps`'s chat-capability shape). Drives a REAL
    /// `axum` router over a REAL `Executor`, not a hand-inspected manifest.
    ///
    /// Uses a resident whose weights path does not exist: `GET /v1/models`
    /// never activates a model (`models.rs`'s own doc: "No generation here;
    /// discovery only"), so this proves the auto-exposure wiring without
    /// needing a real checkpoint/tokenizer.
    #[test]
    fn brain_qwen35moe_is_auto_exposed_on_openai_and_anthropic_model_lists() {
        let resident = Qwen35Resident { id: MODEL.to_string(), path: "/nonexistent/qwen35.safetensors".to_string(), tokenizer: String::new(), hf_dir: false };
        let models: Vec<std::sync::Arc<dyn ResidentModel>> = vec![std::sync::Arc::new(resident)];
        let mut budgets = residency::budget::Budgets::new();
        budgets.set(Device::Cpu, 8 << 30, 0);
        let exec = residency::Executor::start(models, budgets, residency::scheduler::Policy::default());

        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        for provider in [apiserve::Provider::OpenAI, apiserve::Provider::Anthropic] {
            let key = "sk-brain-test-key".to_string();
            let state = apiserve::AppState::new(exec.clone(), key.clone(), provider);
            let app = apiserve::router(state);
            let mut req = axum::http::Request::builder().method(axum::http::Method::GET).uri("/v1/models");
            req = match provider {
                apiserve::Provider::Anthropic => req.header("x-api-key", &key),
                _ => req.header(axum::http::header::AUTHORIZATION, format!("Bearer {key}")),
            };
            let req = req.body(axum::body::Body::empty()).unwrap();

            let (status, body): (axum::http::StatusCode, serde_json::Value) = rt.block_on(async {
                use tower::ServiceExt;
                let resp = app.oneshot(req).await.unwrap();
                let status = resp.status();
                let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
                (status, serde_json::from_slice(&bytes).unwrap())
            });
            assert_eq!(status, axum::http::StatusCode::OK, "{provider:?} GET /v1/models failed: {body}");
            let ids: Vec<&str> = body["data"].as_array().unwrap().iter().filter_map(|c| c["id"].as_str()).collect();
            assert!(ids.contains(&MODEL), "{provider:?}: expected {MODEL:?} in /v1/models, got {ids:?}");
        }
    }
}
