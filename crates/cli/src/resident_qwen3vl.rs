// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-VL behind the residency scheduler.
//!
//! `activate` builds the whole composite ONCE - the checkpoint (HF directory
//! or two-file GGUF) load, the vision tower + PatchMerger/DeepStack merger
//! upload, the decoder's KV-cache buffers - and the [`Instance`] owns the
//! resulting [`qwen3vl::caps::Resident`], so dropping it frees every buffer.
//! One action, `generate`; its schema and all of its work come from
//! `qwen3vl::caps`, so this file holds no second copy of the preprocessing,
//! the prompt assembly or the token accounting - mirroring
//! `crate::resident_moondream3::Moondream3Resident`'s split almost exactly
//! (same single-request VLM shape: image + text in, streamed text out, no
//! batchable decoder axis).
//!
//! # Placement
//!
//! This model is GPU-placeable. `estimate` reports its footprint as `vram`,
//! so `residency::place::pick_device` prefers a card and falls back to the
//! CPU pool on a machine with no GPU (that fallback is `place`'s own rule for
//! a weight-holding model, not a special case here). `activate` builds on
//! whichever device it was handed, through a SCOPED registry selection
//! (`qwen3vl::caps::Resident::load_on` -> `gpu_core::devices::with_gpu`)
//! rather than an env write, because a server-lifetime resident must not
//! change the backend every other model builds on afterwards.
//!
//! # The footprint, and why it is a derivation, not a measurement
//!
//! No real Qwen3-VL-4B checkpoint has been run through THIS resident on an
//! accelerator on the machine this was written on, so [`FP32_BYTES`] and
//! [`INT8_BYTES`] are derived from the released `Qwen3-VL-4B-Instruct`
//! config arithmetic (`qwen3vl::config::Qwen3VlConfig::qwen3_vl_4b`), not
//! measured - see those constants' own doc comments for the arithmetic.
//! `crate::resident_moondream3`'s own doc explains why an honest derivation,
//! shown, is preferred over a fabricated "measured" figure here.
//!
//! # Batching: the documented serial default
//!
//! `run_batch` is NOT overridden. Unlike Moondream 3 (whose vision tower
//! attends within each crop independently, so N requests' crops batch into
//! one `SiglipEncoder::encode` call), Qwen3-VL's vision tower is spliced
//! directly into the decoder's incremental KV-cache decode
//! (`Qwen3Vl::generate_cb`): every request is its own multi-step decode with
//! its own prompt, its own image-token splice position and its own KV cache,
//! and the block forward has no batch dimension on either half. This matches
//! the pattern this repo's `sdxlunet`/`controlnet`/`flux1`/`pulid` residents
//! already document for their own serial multi-step samplers - the default
//! sequential loop in `residency::model::Instance::run_batch` is correct
//! here, not a shortcut.

use capability::{ActionResult, Invocation, Manifest, Progress};
use qwen3vl::caps::{Precision, Resident, DEFAULT_SERVE_MAX_PIXELS, DIR_VAR, MODEL};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// Host bytes an fp32 build holds while hot.
///
/// DERIVED from `Qwen3VlConfig::qwen3_vl_4b`, not measured (see this module's
/// doc). Decoder: 36 layers, each `q_proj` (2560x4096) + `k_proj`/`v_proj`
/// (2560x1024 each) + `o_proj` (4096x2560) = 26,214,400 attention params,
/// plus a SwiGLU MLP (`d_ff` 9728: gate+up 2560x9728x2 + down 9728x2560) =
/// 74,711,040 - 100,925,440 params/layer x 36 = 3,633,315,840. Tied
/// embedding/`lm_head`: vocab 151936 x d_model 2560 = 388,956,160. Decoder
/// total ~4.02B params x 4 bytes = ~14.98 GiB. Vision tower (ViT depth 24,
/// hidden 1024, intermediate 4096): ~12.58M params/block x 24 =
/// ~301.99M, plus patch embed (1536x1024), the learned 2304x1024 position
/// table, and 4 PatchMergers (1 main + 3 DeepStack, ~17M params each) =
/// ~374M params x 4 bytes = ~1.39 GiB (the vision tower is ALWAYS fp32 -
/// `qwen3vl::caps`'s own doc: "a small fraction of the weights and none of
/// the per-token bandwidth"). Weights alone: ~14.98 + ~1.39 = ~16.37 GiB,
/// consistent with `Qwen3Vl::from_hf`'s own doc comment ("the released 4B
/// checkpoint is ~16 GB in f32"). Add the KV cache at this resident's built
/// context (`qwen3vl::caps::default_ctx_len()`, `$BRAIN_QWEN3VL_CTX` default
/// 24576 - NOT the old fixed `SEQ_LEN=4096` this derivation used before
/// `qwen3vl::caps` made context a checkpoint-clamped, operator-tunable
/// default): 8 kv-heads x 128 head_dim x 2 (K,V) x 4 bytes/elem x 36 layers x
/// 24576 tokens = exactly 6.75 GiB, plus DeepStack/splice scratch and the
/// packed-patch/image-token buffers (~1 GiB, generous at the default
/// `max_pixels`). Total: ~16.37 + 6.75 + ~1 = ~24.12 GiB, rounded up.
const FP32_BYTES: u64 = 25u64 << 30;

/// Host bytes an int8 build holds while hot: same derivation as
/// [`FP32_BYTES`], but the DECODER linears (the ~4.02B params counted above)
/// are one byte each instead of four - ~3.75 GiB - while the vision tower
/// stays fp32 (~1.39 GiB, see [`FP32_BYTES`]'s doc). Weights: ~3.75 + ~1.39 =
/// ~5.14 GiB. The KV cache is NOT quantized by this resident (`Qwen3Vl`'s
/// `Precision` only selects the decoder LINEAR dtype), so it stays the same
/// 6.75 GiB at the default built context, plus the same ~1 GiB of scratch.
/// Total: ~5.14 + 6.75 + 1 = ~12.89 GiB, rounded up.
const INT8_BYTES: u64 = 13u64 << 30;

/// Qwen3-VL behind the scheduler. `BRAIN_QWEN3VL_WEIGHTS` names either an HF
/// checkpoint DIRECTORY (`config.json` + `model.safetensors[.index.json]` +
/// `tokenizer.json`) or a two-file llama.cpp GGUF checkpoint (the language
/// half, or the directory holding it plus its `mmproj-*.gguf` sibling) - see
/// `qwen3vl::caps::classify_source`.
pub struct Qwen3VlResident {
    dir: String,
}

impl Qwen3VlResident {
    /// `None` when the variable is unset or the path is absent - registering
    /// a model whose every call would fail is worse than not serving it.
    pub fn from_env() -> Option<Qwen3VlResident> {
        let dir = std::env::var(DIR_VAR).ok().filter(|p| !p.is_empty())?;
        Self::new(dir)
    }

    /// Direct constructor (no env round-trip) - see
    /// `crate::resident_scrfd::ScrfdResident::new`'s rationale.
    pub fn new(dir: impl Into<String>) -> Option<Qwen3VlResident> {
        let dir = dir.into();
        if !std::path::Path::new(&dir).exists() {
            eprintln!("brain: qwen3vl not served ({dir} does not exist)");
            return None;
        }
        Some(Qwen3VlResident { dir })
    }

    /// The resident capacity a request asks for, defaulting to
    /// [`DEFAULT_SERVE_MAX_PIXELS`] - see `qwen3vl::caps`'s own doc on why
    /// this is a construction-time CAPACITY, not one request's exact size.
    fn max_pixels_of(inv: &Invocation) -> u32 {
        inv.get_i64("max_pixels").unwrap_or(DEFAULT_SERVE_MAX_PIXELS as i64).max(1) as u32
    }

    /// The precision a request asks for, defaulting to the one that fits
    /// (`Precision::default()` is fp32, matching the action spec's own
    /// `"precision"` param default). Unlike `qwen3vl::caps::GenerateAction`'s
    /// direct-provider path, an unrecognised string here falls back rather
    /// than erroring - `instance_key` cannot return a `Result`, and
    /// `activate` will hit the same fallback so the two never disagree about
    /// which instance a bad string builds.
    fn precision_of(inv: &Invocation) -> Precision {
        inv.get_str("precision").as_deref().and_then(|s| Precision::from_name(s).ok()).unwrap_or_default()
    }
}

impl ResidentModel for Qwen3VlResident {
    fn manifest(&self) -> Manifest {
        // The stripped, weights-free spec: this resident's checkpoint path is
        // already resolved (`self.dir`, from `from_env`/`new`), so a served
        // caller must never be told a `weights` param exists to set - see
        // `qwen3vl::caps::manifest_resident`'s doc. `run` below never reads
        // `weights` from `inv` either (`Resident::generate` takes none).
        qwen3vl::caps::manifest_resident()
    }

    fn instance_key(&self, _action: &str, inv: &Invocation) -> InstanceKey {
        // `max_pixels` and `precision` are both part of the identity: each
        // combination is a differently-sized build (DeepStack/splice buffer
        // capacity and decoder linear width respectively), and sharing one
        // key across them would let a stray request evict a working instance
        // to build a differently-shaped one.
        let max_pixels = Self::max_pixels_of(inv);
        let p = Self::precision_of(inv).name();
        InstanceKey::new(MODEL, format!("{}|{max_pixels}|{p}", self.dir))
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        // Reported as VRAM: both towers go wherever this instance is placed,
        // so on a GPU box the whole footprint is device memory.
        // `place::pick_device` falls a weight-holding model back to the CPU
        // pool at the same figure on a machine with no GPU, which is its own
        // rule and the behaviour this model wants.
        let (_, _, precision) = parse_key(key).unwrap_or((String::new(), DEFAULT_SERVE_MAX_PIXELS, Precision::default()));
        let bytes = match precision {
            Precision::I8 => INT8_BYTES,
            Precision::F32 => FP32_BYTES,
        };
        MemCost::new(bytes, 0)
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let (dir, max_pixels, precision) = parse_key(key)?;
        let gpu = match device {
            Device::Cpu => None,
            Device::Gpu(i) => Some(i),
            Device::Npu(i) => {
                // This model advertises no NPU footprint, so the placer never
                // offers one; refuse by name rather than silently building wgpu.
                return Err(format!("qwen3vl: assigned Npu({i}), but this model has no NPU export path"));
            }
        };
        let resident = Resident::load_on(&dir, max_pixels, precision, gpu)?;
        Ok(Box::new(Qwen3VlInstance { resident }))
    }
}

/// `(dir, max_pixels, precision)` out of an [`InstanceKey`]'s
/// `"{dir}|{max_pixels}|{precision}"` config string. Splitting from the
/// RIGHT (`rsplitn`) means a directory path containing `|` still parses
/// correctly, since only the last two fields are peeled off.
fn parse_key(key: &InstanceKey) -> Result<(String, u32, Precision), String> {
    let mut parts = key.config.rsplitn(3, '|');
    let precision_s = parts.next().ok_or("qwen3vl: malformed instance key")?;
    let max_pixels_s = parts.next().ok_or("qwen3vl: malformed instance key")?;
    let dir = parts.next().ok_or("qwen3vl: malformed instance key")?.to_string();
    let precision = Precision::from_name(precision_s)?;
    let max_pixels: u32 = max_pixels_s.parse().map_err(|_| "qwen3vl: malformed instance key (max_pixels)".to_string())?;
    Ok((dir, max_pixels, precision))
}

/// A resident Qwen3-VL: the built composite (vision tower + decoder +
/// tokenizer) behind one action, `generate`.
///
/// `run_batch` is left at the default sequential loop - see this module's
/// doc for why there is no batchable axis to override it with (unlike
/// Moondream 3's vision-tower batching, Qwen3-VL's vision tower feeds
/// directly into the decoder's own incremental splice, so there is no stage
/// that is both shared across requests and independent of each request's
/// prompt/KV state).
struct Qwen3VlInstance {
    resident: Resident,
}

impl Instance for Qwen3VlInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if action != "generate" {
            return Err(format!("qwen3vl: unknown action '{action}'"));
        }
        self.resident.generate(inv, progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An unconfigured checkpoint yields no resident at all, rather than one
    /// that fails every call.
    #[test]
    fn a_missing_checkpoint_is_not_registered() {
        assert!(Qwen3VlResident::new("/definitely/not/a/qwen3vl/dir").is_none());
    }

    /// `max_pixels` and `precision` are both part of the instance identity.
    /// Sharing one key across a precision change would let a single int8
    /// request evict a working fp32 instance built for a different capacity.
    #[test]
    fn max_pixels_and_precision_key_separate_instances() {
        let r = Qwen3VlResident { dir: "/tmp".into() };
        let k_default = r.instance_key("generate", &Invocation::new());
        let k_i8 = r.instance_key("generate", &Invocation::new().set("precision", serde_json::json!("int8")));
        let k_bigger = r.instance_key("generate", &Invocation::new().set("max_pixels", serde_json::json!(2_000_000)));
        assert_ne!(k_default, k_i8);
        assert_ne!(k_default, k_bigger);
        assert_ne!(k_i8, k_bigger);
        // An absent or unparseable precision falls back to fp32, matching
        // `Precision::default()` and the action spec's own default.
        assert_eq!(r.instance_key("generate", &Invocation::new().set("precision", serde_json::json!("bogus"))), k_default);
    }

    /// The two precisions must be budgeted differently, and fp32 must be the
    /// larger one - the whole point of the int8 tier existing.
    #[test]
    fn the_two_precisions_are_budgeted_apart() {
        let r = Qwen3VlResident { dir: "/tmp".into() };
        let c32 = r.estimate(&r.instance_key("generate", &Invocation::new()));
        let c8 = r.estimate(&r.instance_key("generate", &Invocation::new().set("precision", serde_json::json!("int8"))));
        assert!(c32.vram > 0, "this model is GPU-placeable; a zero vram would hide it from the GPU class");
        assert_eq!(c32.npu, 0, "no NPU export path exists");
        assert!(c32.vram > c8.vram, "fp32 should be larger than int8, got {} vs {}", c32.vram, c8.vram);
    }

    /// An NPU assignment is refused by name. The placer never offers one
    /// (npu == 0), but a silent wgpu build under an NPU label is the kind of
    /// wrong-backend failure a sibling model in this directory already paid
    /// for.
    #[test]
    fn an_npu_assignment_is_refused() {
        let r = Qwen3VlResident { dir: "/tmp".into() };
        let e = r.activate(&r.instance_key("generate", &Invocation::new()), Device::Npu(0)).err().unwrap_or_default();
        assert!(e.contains("no NPU export path"), "{e}");
    }

    /// The resident manifest must never advertise `weights` as a settable
    /// param - a served caller cannot point this instance at a different
    /// checkpoint, only the operator (via `BRAIN_QWEN3VL_WEIGHTS`) can.
    #[test]
    fn the_resident_manifest_has_no_weights_param() {
        let r = Qwen3VlResident { dir: "/tmp".into() };
        let m = r.manifest();
        let a = &m.actions[0];
        assert_eq!(a.name, "generate");
        assert!(!a.params.iter().any(|p| p.name == "weights"), "resident manifest must not expose 'weights'");
    }

    /// REGRESSION for the whole point of this file: registering
    /// `Qwen3VlResident` with the residency `Executor` must be enough, with
    /// NO `crates/dbus`/`crates/apiserve`-side code, for `brain/qwen3vl` to
    /// be reachable there - `qwen3vl::caps`'s own module doc says the
    /// `generate` action's shape (streaming, `messages`/`prompt`, `Text`
    /// output) is chosen SPECIFICALLY to satisfy `apiserve::catalog::
    /// api_caps`'s chat classification, and `crates/dbus`'s `Manager::run`/
    /// `subscribe` dispatch is model-agnostic over whatever `exec.manifests()`
    /// lists - the same `Executor` D-Bus reads from. Drives a REAL `axum`
    /// router over a REAL `Executor`, matching
    /// `resident_qwen35moe::brain_qwen35moe_is_auto_exposed_on_openai_and_anthropic_model_lists`.
    ///
    /// Uses a resident whose weights path does not exist: `GET /v1/models`
    /// never activates a model, so this proves the auto-exposure wiring
    /// (residency -> D-Bus's `Manager::manifests`/`list_models` AND HTTP's
    /// `/v1/models`) without needing a real checkpoint.
    #[test]
    fn brain_qwen3vl_is_auto_exposed_on_openai_and_anthropic_model_lists() {
        let resident = Qwen3VlResident { dir: "/nonexistent/qwen3vl".to_string() };
        let models: Vec<std::sync::Arc<dyn ResidentModel>> = vec![std::sync::Arc::new(resident)];
        let mut budgets = residency::budget::Budgets::new();
        budgets.set(Device::Cpu, 8 << 30, 0);
        let exec = residency::Executor::start(models, budgets, residency::scheduler::Policy::default());

        // The exact classification D-Bus's `Manager::manifests`/`list_models`
        // and HTTP's `/v1/models` both build on: this is `exec.manifests()`,
        // the one source both transports read.
        assert!(
            exec.manifests().iter().any(|m| m.model == MODEL),
            "qwen3vl must be registered in the executor's manifest set once BRAIN_QWEN3VL_WEIGHTS names a resident"
        );

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
