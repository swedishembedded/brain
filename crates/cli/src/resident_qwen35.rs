// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident-model adapter putting Qwen3.8-27B's paged serving engine
//! (`qwen35::serve::Engine`/`Scheduler`) behind the residency [`Executor`].
//! Mirrors `crate::resident_qwen35moe::Qwen35Resident` almost exactly - the
//! GDN/GQA decode-step orchestration `Engine` wraps is architecture-
//! identical between the two crates, so the scope (single-GPU, fp32 weights
//! only, fp32 KV, one truly-active sequence at a time on the GPU) and the
//! "deliberately deferred" list are identical too; see
//! `crates/qwen35/src/serve.rs`'s own module doc for the authoritative list.
//!
//! Two ways in. [`Qwen35Resident::from_card`] serves a checkpoint the
//! model-dir scan found (family `"qwen35"` - what `Qwen35::save` stamps,
//! distinct from the MoE sibling's own `"qwen35moe"` family), under its own
//! card id and with its sibling `tokenizer.json`; no env vars involved. The
//! env path below is the manual alternative, for a checkpoint outside the
//! models directory.
//!
//! Env config follows `BRAIN_QWEN_*`'s naming convention:
//!   * `BRAIN_QWEN35_WEIGHTS` - a brain-format Qwen3.8-27B checkpoint
//!     (`.safetensors`, `checkpoint::load`-compatible). The primary gate;
//!     unset means not served.
//!   * `BRAIN_QWEN35_TOKENIZER` - the sibling `tokenizer.json`. If unset,
//!     `activate()` falls back to a GGUF checkpoint's own embedded
//!     `tokenizer.ggml.*` KV (see `crate::resident_llm::QwenResident::
//!     activate`, the sibling this mirrors) before giving up - `Engine`
//!     itself still never touches a `.gguf` file at all, so a GGUF
//!     checkpoint gets a real tokenizer but no further than that today.
//!   * `BRAIN_QWEN35_CTX` - the hard `prompt + max_new` cap for any ONE
//!     sequence (`Engine::from_map`'s `max_seq_len`, which this engine also
//!     uses as its per-sequence block size). Default 4096.
//!   * `BRAIN_QWEN35_MAX_BATCH` - how many sequences may be RESIDENT at once
//!     (`Engine::from_map`'s `max_concurrent`, i.e. `num_blocks` - NOT how
//!     many are dispatched together on the GPU per step, which is always 1
//!     for this engine). Default 4.
//!
//! A THIRD way in lives at the bottom of this file:
//! [`multi_gpu_gguf_from_env`], the factory for
//! `qwen35::int8_gguf_resident::Qwen35GgufResident` - a genuinely different
//! model (`unsloth/Qwen3.8-27B-Q8_0`, INT8 weights read straight from the
//! released Q8_0 GGUF and layer-sharded across as many cards as they need),
//! not a mode of the one above. The two coexist: this one still serves an
//! fp32 brain checkpoint that fits one card, that one serves the real 27B
//! release on a box with enough cards and no room on disk for a ~108 GB fp32
//! conversion.

use capability::{ActionResult, Invocation, Manifest, Progress};
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use qwen3::chat::{parse_request, SeqState};
use qwen35::caps::with_template_flavor_default;
use qwen35::config::{LayerType, Qwen35Config};
use qwen35::serve::{Engine, Scheduler};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

use crate::resident_llm::{est_vram, generate_spec, on_device};

/// Catalog id: a `brain/`-prefixed synthetic id, matching `qwen35::caps::MODEL`
/// exactly.
const MODEL: &str = qwen35::caps::MODEL;

/// The Qwen3.8-27B dense hybrid Gated-DeltaNet/GQA decoder behind the
/// scheduler (`BRAIN_QWEN35_WEIGHTS` + `BRAIN_QWEN35_TOKENIZER`). See this
/// module's own doc for the exact (single-GPU, fp32-only) scope.
pub struct Qwen35Resident {
    id: String,
    path: String,
    tokenizer: String,
}

impl Qwen35Resident {
    pub fn from_env() -> Option<Qwen35Resident> {
        let path = std::env::var("BRAIN_QWEN35_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        let tokenizer = std::env::var("BRAIN_QWEN35_TOKENIZER").ok().unwrap_or_default();
        Some(Qwen35Resident { id: MODEL.to_string(), path, tokenizer })
    }

    /// The model-dir counterpart of [`from_env`](Self::from_env): a checkpoint
    /// discovered by `crate::model_dir` (family `"qwen35"` - what
    /// `Qwen35::save` stamps on its `ModelCard`) served under its OWN card id
    /// rather than the env fallback [`MODEL`].
    ///
    /// `tokenizer` is the sibling `tokenizer.json` the scan found. It is
    /// required: `Engine` never opens a `.gguf`, so there is no
    /// embedded-tokenizer fallback - a checkpoint without one is declined at
    /// discovery instead.
    pub fn from_card(path: &str, card: &checkpoint::st::ModelCard, tokenizer: Option<&str>) -> Result<Qwen35Resident, String> {
        let tokenizer = tokenizer.filter(|t| !t.is_empty()).ok_or("qwen35: no sibling tokenizer.json")?;
        Ok(Qwen35Resident { id: card.id.clone(), path: path.to_string(), tokenizer: tokenizer.to_string() })
    }

    /// The hard per-sequence `prompt + max_new` cap, which this engine also
    /// uses as its physical block size (`Engine::from_map`'s `max_seq_len`
    /// parameter).
    fn ctx() -> u32 {
        std::env::var("BRAIN_QWEN35_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(4096u32).max(1)
    }

    /// How many sequences may be resident at once (`Engine::from_map`'s
    /// `max_concurrent`, i.e. its `num_blocks`) - NOT how many run on the GPU
    /// concurrently per step (always 1; see this module's own doc).
    fn max_concurrent() -> u32 {
        std::env::var("BRAIN_QWEN35_MAX_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(4u32).max(1)
    }
}

impl ResidentModel for Qwen35Resident {
    fn manifest(&self) -> Manifest {
        // `Self::ctx()` is exactly the `max_seq_len` `activate()` below builds
        // the engine with, so advertising it is a safe, never-overstated
        // floor on real serving capacity.
        Manifest::new(
            &self.id,
            "text generation (Qwen3.8-27B dense hybrid Gated-DeltaNet/GQA decoder; single-GPU, fp32 weights + fp32 KV; multiple sequences may be admitted and scheduled concurrently on a shared engine, but the GPU dispatch itself runs one sequence's step at a time -- see crate::resident_qwen35's module doc)",
            vec![generate_spec("generate text (Qwen3.8-27B; chat template optional)", true)],
        )
        .with_max_context_tokens(Self::ctx() as u64)
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(&self.id, "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        let cost = est_vram(&self.path);
        let Ok(reader) = checkpoint::weightio::WeightReader::open(&self.path) else {
            return cost;
        };
        let cfg = Qwen35Config::from_json(&reader.config());
        let ctx = Self::ctx() as u64;
        let max_concurrent = Self::max_concurrent() as u64;
        let n_full = cfg.layer_types().iter().filter(|t| **t == LayerType::Full).count() as u64;
        // The GQA side of `Engine::kv_pool_bytes()` ONLY, computed
        // independently via `Qwen35Config`'s own public accessors rather than
        // calling into `qwen35::serve` (whose `GdnSlot` sizing helper is
        // crate-private, by design). Genuinely missing: the GDN recurrent-
        // state/conv-history slot pool is real but small (O(1) state per
        // sequence, not O(context length) like the GQA side), so this
        // estimate UNDER-counts by that (bounded, much smaller) amount
        // rather than over-claiming a number this crate cannot cheaply
        // verify from outside `serve.rs`.
        let gqa_bytes = n_full * max_concurrent * 2 * ctx * cfg.kv_dim() as u64 * 4;
        MemCost::new(cost.vram + gqa_bytes, cost.ram)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // Open first so a GGUF can supply its own embedded tokenizer, exactly
        // like crate::resident_llm::QwenResident::activate. `Engine` itself
        // still only loads brain-native safetensors below (see this module's
        // own doc), so this reader exists purely for the tokenizer fallback.
        let reader = checkpoint::weightio::WeightReader::open(&self.path).map_err(|e| format!("qwen35: {e}"))?;
        // Tokenizer precedence: an explicit sibling tokenizer.json (or an
        // env override) wins; else a .gguf builds from its embedded
        // tokenizer.ggml.* KV; else there is nothing to tokenize with.
        let tok = if !self.tokenizer.is_empty() {
            QwenBpe::from_file(&self.tokenizer)?
        } else if let Some(gt) = reader.tokenizer() {
            QwenBpe::from_gguf(&gt).map_err(|e| format!("qwen35: {e}"))?
        } else {
            return Err("qwen35: no tokenizer (set BRAIN_QWEN35_TOKENIZER, or use a GGUF with an embedded tokenizer)".to_string());
        };
        let eos = tok.encode("<|im_end|>").first().copied();
        let ctx = Self::ctx();
        let max_concurrent = Self::max_concurrent();
        let path = self.path.clone();
        let sched = on_device(device, move || -> Scheduler {
            // `Engine` only loads brain-native safetensors (`checkpoint::load`)
            // - see this module's own doc on why there is no GGUF arm here.
            let container = checkpoint::load(&path);
            let cfg = Qwen35Config::from_json(&container.header["config"]);
            let weights = container.by_role("");
            let engine = Engine::from_map(cfg, &weights, ctx, max_concurrent);
            Scheduler::new(engine, max_concurrent as usize)
        })?;
        Ok(Box::new(Qwen35Instance { tok, eos, sched }))
    }
}

struct Qwen35Instance {
    tok: QwenBpe,
    eos: Option<u32>,
    sched: Scheduler,
}

impl Instance for Qwen35Instance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.run_batch(action, std::slice::from_ref(inv), &mut |_i, p| progress(p)).pop().unwrap()
    }

    /// Every invocation in `invs` is submitted into the SAME persistent
    /// `Scheduler` (built once at `activate`, so the KV pool and prefix cache
    /// are shared and reused across calls) and driven to completion together
    /// - mirroring `crate::resident_qwen35moe::run_batch_scheduled` exactly.
    fn run_batch(&mut self, _action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        run_batch_scheduled(&mut self.sched, &self.tok, self.eos, invs, progress)
    }

    /// The paged engine's prefix-cache effectiveness.
    fn metrics(&self) -> Vec<(String, serde_json::Value)> {
        let (hit, looked, cached) = self.sched.prefix_stats();
        let rate = if looked > 0 { hit as f64 / looked as f64 } else { 0.0 };
        vec![
            ("kv_prefix_hit_rate".to_string(), serde_json::json!(rate)),
            ("kv_prefix_hit_tokens".to_string(), serde_json::json!(hit)),
            ("kv_prefix_lookup_tokens".to_string(), serde_json::json!(looked)),
            ("kv_prefix_cached_blocks".to_string(), serde_json::json!(cached)),
        ]
    }
}

/// Drive every invocation in `invs` to completion on the SAME persistent
/// `Scheduler` - a near-identical copy of
/// `crate::resident_qwen35moe::run_batch_scheduled`, specialized to
/// `qwen35::serve::Scheduler` rather than generalized over the engine type.
fn run_batch_scheduled(sched: &mut Scheduler, tok: &QwenBpe, eos: Option<u32>, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
    let mut results: Vec<Option<ActionResult>> = vec![None; invs.len()];
    let mut seq_for_bi: Vec<Option<SeqState>> = Vec::with_capacity(invs.len());
    let mut id_for_bi: Vec<Option<u64>> = Vec::with_capacity(invs.len());

    for (bi, inv) in invs.iter().enumerate() {
        // This IS Qwen3.8-27B, so it renders the 3.8 template by default -
        // `parse_request` alone defaults to the Qwen3-era flavor, which
        // would silently mis-render this model's prompt (the GGUF resident
        // and `caps.rs` both already apply this; this engine did not).
        match parse_request(tok, &with_template_flavor_default(inv)) {
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
        // A request the scheduler refuses at admission never appears in
        // `completed` and never will; without handling it here its `bi`
        // would stay in `remaining` forever.
        for (id, reason) in report.rejected {
            let bi = id_for_bi.iter().position(|x| *x == Some(id)).expect("rejected id must belong to this batch");
            seq_for_bi[bi] = None;
            results[bi] = Some(Err(format!("qwen35: {reason}")));
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

/// The INT8, multi-GPU, load-straight-from-GGUF Qwen3.8-27B
/// (`qwen35::int8_gguf_resident::Qwen35GgufResident`). Reachable ONLY via
/// `Executor::register_multi`: it occupies real bytes on SEVERAL cards at
/// once, and a plain `register` would budget only one of them (see
/// `crates/residency/src/multi.rs`' module doc).
///
/// Where the `.gguf` comes from, in order:
///   1. `BRAIN_QWEN35_GGUF` - an explicit path. The same variable
///      `qwen35::gguf_import`'s own real-checkpoint gate uses, so a box
///      configured to run those tests is configured to serve this.
///   2. the model store's canonical location for the upstream release
///      (`<models-dir>/unsloth/Qwen3.8-27B/Q8_0.gguf`, resolved through
///      `modelstore::Store` - never a hardcoded machine path), when a
///      `brain fetch` put it there.
///
/// Neither present ⇒ not served, and this returns `None`.
///
/// `gpus` is `build_executor`'s own budgeted GPU list as `(index, TOTAL
/// bytes)` and `reserved` the per-card headroom it keeps free, so what is
/// handed on is each card's genuinely USABLE capacity - the same figure the
/// scheduler budgets against. Passing capacity (not just identity) is what
/// lets `model::shard::plan_fewest_devices` size the split to the hardware
/// instead of assuming matched cards.
pub fn multi_gpu_gguf_from_env(gpus: &[(u32, u64)], reserved: u64) -> Option<qwen35::int8_gguf_resident::Qwen35GgufResident> {
    let path = resolve_qwen35_gguf()?;
    if gpus.is_empty() {
        eprintln!(
            "brain: {} not served (no GPU budgeted -- its weights are int8 device buffers, there is no CPU path)",
            qwen35::int8_gguf_resident::MODEL
        );
        return None;
    }
    let devices: Vec<(Device, u64)> = gpus.iter().map(|&(i, total)| (Device::Gpu(i), total.saturating_sub(reserved))).collect();
    let cap = qwen35::int8_gguf_resident::Qwen35GgufResident::ctx_from_env();
    let tier = qwen35::int8_gguf_resident::Qwen35GgufResident::tier_from_env();
    Some(qwen35::int8_gguf_resident::Qwen35GgufResident::new(path, devices, cap, tier))
}

/// [`multi_gpu_gguf_from_env`]'s file resolution: the env var, else the model
/// store's canonical path for the upstream release. Returns `None` (never a
/// guessed path) when neither exists on disk.
fn resolve_qwen35_gguf() -> Option<String> {
    /// The upstream release this resident serves, as a `modelref` id.
    const RELEASE: &str = "unsloth/Qwen3.8-27B-Q8_0";
    if let Some(p) = std::env::var(qwen35::int8_gguf_resident::GGUF_ENV).ok().filter(|p| !p.is_empty()) {
        if !std::path::Path::new(&p).is_file() {
            eprintln!("brain: {}={p} does not name a readable file -- {} not served", qwen35::int8_gguf_resident::GGUF_ENV, qwen35::int8_gguf_resident::MODEL);
            return None;
        }
        return Some(p);
    }
    let r = brain_modelref::ModelRef::parse(RELEASE).ok()?;
    let store = brain_modelstore::Store::new(brain_modelstore::default_root()?);
    Some(store.local(&r)?.weights.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `BRAIN_QWEN35_CTX`/`BRAIN_QWEN35_MAX_BATCH` parse with sane defaults
    /// and floor at 1.
    #[test]
    fn ctx_and_max_concurrent_have_sane_defaults() {
        // Only assert defaults when the env vars are genuinely unset in this
        // process (a real value on the runner would make this a lie).
        if std::env::var("BRAIN_QWEN35_CTX").is_err() {
            assert_eq!(Qwen35Resident::ctx(), 4096);
        }
        if std::env::var("BRAIN_QWEN35_MAX_BATCH").is_err() {
            assert_eq!(Qwen35Resident::max_concurrent(), 4);
        }
    }

    /// The GGUF resident is gated on a REAL file, not on the variable merely
    /// being set: a stale `BRAIN_QWEN35_GGUF` pointing at a deleted download
    /// must decline (and say so) rather than register a resident whose every
    /// `estimate_multi` then reports zero devices and whose every request
    /// fails placement.
    #[test]
    fn multi_gpu_gguf_declines_a_path_that_is_not_a_file() {
        // SAFETY: no other test in this process reads/writes this exact var.
        unsafe { std::env::set_var(qwen35::int8_gguf_resident::GGUF_ENV, "/nonexistent/qwen35.gguf") };
        assert!(multi_gpu_gguf_from_env(&[(0, 24 << 30)], 2 << 30).is_none());
        unsafe { std::env::remove_var(qwen35::int8_gguf_resident::GGUF_ENV) };
    }

    #[test]
    fn from_env_is_none_without_the_weights_var() {
        // SAFETY: no other test in this process reads/writes this exact var.
        unsafe { std::env::remove_var("BRAIN_QWEN35_WEIGHTS") };
        assert!(Qwen35Resident::from_env().is_none());
    }

    /// REGRESSION: this engine used to call `parse_request` directly, so it
    /// rendered the Qwen3-era chat template by default while `caps.rs` and
    /// the GGUF resident both correctly default to Qwen3.8's own template
    /// (`with_template_flavor_default`'s own doc). Mirrors `qwen35::caps`'s
    /// own `tokenizer_present_renders_the_qwen38_flavor_without_asking` test
    /// exactly - same env-gated real tokenizer, same assertion - because
    /// this engine must render identically to every other entry point into
    /// this model, not merely "some" template.
    #[test]
    fn parse_request_renders_the_qwen38_flavor_without_asking() {
        let Ok(tok_path) = std::env::var("QWEN_TOKENIZER") else {
            brain_testutil::skip("set QWEN_TOKENIZER to a real tokenizer.json to run this test");
            return;
        };
        let tok = QwenBpe::from_file(&tok_path).expect("load Qwen3 tokenizer");
        let inv = Invocation::new().set("prompt", serde_json::json!("what's the weather in Paris?")).set("chat", serde_json::json!(true));
        let req = parse_request(&tok, &with_template_flavor_default(&inv)).unwrap();
        assert_eq!(req.flavor, data::qwen_chat::TemplateFlavor::Qwen38, "this engine must default to the Qwen3.8 flavor, same as caps.rs and the GGUF resident");
    }

    /// REGRESSION for the whole point of this file: registering
    /// `Qwen35Resident` with the residency `Executor` must be enough, with NO
    /// `crates/apiserve`-side code, for `brain/qwen35` to appear on BOTH the
    /// OpenAI and the Anthropic `/v1/models` list with the chat capability.
    /// Drives a REAL `axum` router over a REAL `Executor`, not a
    /// hand-inspected manifest.
    ///
    /// Uses a resident whose weights path does not exist: `GET /v1/models`
    /// never activates a model, so this proves the auto-exposure wiring
    /// without needing a real checkpoint/tokenizer.
    #[test]
    fn brain_qwen35_is_auto_exposed_on_openai_and_anthropic_model_lists() {
        let resident = Qwen35Resident { id: MODEL.to_string(), path: "/nonexistent/qwen35.safetensors".to_string(), tokenizer: String::new() };
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

    /// A minimal single-file GGUF carrying a real gpt2-scheme
    /// `tokenizer.ggml.*` KV block, same shape `model_dir.rs`'s own
    /// `write_gguf_qwen` test helper builds, written through the real
    /// `checkpoint::gguf_write::write` encoder rather than hand-assembled
    /// bytes.
    fn write_gguf_with_gpt2_tokenizer(path: &std::path::Path) {
        use checkpoint::gguf::GgufValue as V;
        let kv = vec![
            ("general.architecture".to_string(), V::String("qwen3.8".to_string())),
            ("tokenizer.ggml.model".to_string(), V::String("gpt2".to_string())),
            (
                "tokenizer.ggml.tokens".to_string(),
                V::Array(["<|endoftext|>", "<|im_start|>", "<|im_end|>", "h", "i", "hi"].into_iter().map(|s| V::String(s.to_string())).collect()),
            ),
            ("tokenizer.ggml.merges".to_string(), V::Array(vec![V::String("h i".to_string())])),
            ("tokenizer.ggml.token_type".to_string(), V::Array([3, 3, 3, 1, 1, 1].into_iter().map(V::I32).collect())),
            ("tokenizer.ggml.bos_token_id".to_string(), V::U32(0)),
            ("tokenizer.ggml.eos_token_id".to_string(), V::U32(2)),
        ];
        let tensor = checkpoint::gguf_write::TensorOut { name: "w".to_string(), shape: vec![4], ty: checkpoint::gguf::GgmlType::F32.id(), data: [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect() };
        checkpoint::gguf_write::write(path.to_str().unwrap(), &kv, std::slice::from_ref(&tensor), 32).unwrap();
    }

    /// M21: with no explicit `BRAIN_QWEN35_TOKENIZER`/sibling file, `activate()`
    /// must fall back to the GGUF's own embedded `tokenizer.ggml.*` KV
    /// (`crate::resident_llm::QwenResident::activate`'s already-working
    /// pattern) instead of immediately returning the "no tokenizer" error.
    /// `Engine` itself has no GGUF arm (this module's own doc), so activation
    /// still cannot fully succeed on a `.gguf` checkpoint -- it panics inside
    /// `checkpoint::load` (a pre-existing, documented limitation, not
    /// introduced here) once it gets past tokenizer construction. Catching
    /// that panic and asserting it is the safetensors-parse failure (not the
    /// tokenizer error) proves the fallback ran and reached the real next
    /// step.
    #[test]
    fn activate_falls_back_to_the_gguf_embedded_tokenizer() {
        let dir = std::env::temp_dir().join(format!("brain-qwen35-gguf-tok-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qwen35.gguf");
        write_gguf_with_gpt2_tokenizer(&path);

        let resident = Qwen35Resident { id: MODEL.to_string(), path: path.to_str().unwrap().to_string(), tokenizer: String::new() };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| resident.activate(&InstanceKey::new(MODEL, "default"), Device::Cpu)));
        match result {
            // Getting a clean Err this early would mean the tokenizer step
            // itself rejected something; the fallback must at least get
            // past it.
            Ok(Err(e)) => panic!("expected the GGUF tokenizer fallback to succeed and activation to fail later inside checkpoint::load, got an early Err instead: {e}"),
            Ok(Ok(_)) => panic!("activate() unexpectedly succeeded fully on a .gguf checkpoint -- Engine has no GGUF arm (see this module's own doc); if that changed, update this test"),
            Err(payload) => {
                let msg = payload.downcast_ref::<String>().cloned().or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                assert!(msg.contains("cannot read"), "expected the checkpoint::load safetensors-parse panic (proving tokenizer construction succeeded first), got: {msg:?}");
                assert!(!msg.to_ascii_lowercase().contains("no tokenizer"), "the panic must not be the tokenizer-missing path: {msg:?}");
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
