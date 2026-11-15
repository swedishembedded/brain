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
//! [`ResidentModel::estimate`] reads `self.dir`'s OWN config
//! (`qwen3vl::caps::config_and_capacity`, a cheap metadata-only read - a GGUF
//! header is a mmap, an HF `config.json` a small JSON parse) and runs it
//! through `qwen3vl::footprint::estimate_vram_bytes` - the same arithmetic
//! `qwen3vl::caps::load_hf_resident`/`load_gguf_resident` use to refuse an
//! over-budget checkpoint on the direct `brain qwen3vl generate` path. A
//! constant sized for one released checkpoint (the 4B) cannot tell an 8B
//! checkpoint's decoder apart from a 4B's - which is exactly the gap that let
//! a real 8B GGUF get budgeted at the 4B's own figure and OOM the driver
//! instead of being refused (see [`FP32_BYTES`]/[`INT8_BYTES`]'s own doc:
//! they are now the FALLBACK for the rare case `self.dir`'s config cannot be
//! read - `estimate()` must never itself hard-fail, see this trait's doc).
//!
//! # Batching: the documented serial default
//!
//! `run_batch` is NOT overridden. Unlike Moondream 3 (whose vision tower
//! attends within each crop independently, so N requests' crops batch into
//! one `ClipVision::encode` call), Qwen3-VL's vision tower is spliced
//! directly into the decoder's incremental KV-cache decode
//! (`Qwen3Vl::generate_cb`): every request is its own multi-step decode with
//! its own prompt, its own image-token splice position and its own KV cache,
//! and the block forward has no batch dimension on either half. This matches
//! the pattern this repo's `sdxlunet`/`controlnet`/`flux1`/`pulid` residents
//! already document for their own serial multi-step samplers - the default
//! sequential loop in `residency::model::Instance::run_batch` is correct
//! here, not a shortcut.

use capability::{ActionResult, Invocation, Manifest, Progress};
use qwen3vl::caps::{Capacity, Precision, Resident, MODEL};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// Fallback fp32 estimate for the rare case `self.dir`'s config cannot be
/// read at `estimate()` time (see this module's doc) - the released
/// `Qwen3-VL-4B-Instruct` figure `qwen3vl::footprint::estimate_vram_bytes`
/// itself now produces for `Qwen3VlConfig::qwen3_vl_4b` (~24 GiB; see that
/// function's own tests for the arithmetic), not a substitute for it.
const FP32_BYTES: u64 = 25u64 << 30;

/// Fallback int8 estimate, same rationale as [`FP32_BYTES`] (~13 GiB for the
/// 4B config).
const INT8_BYTES: u64 = 13u64 << 30;

/// Qwen3-VL behind the scheduler. The resolved `weights` role names either an HF
/// checkpoint DIRECTORY (`config.json` + `model.safetensors[.index.json]` +
/// `tokenizer.json`) or a two-file llama.cpp GGUF checkpoint (the language
/// half, or the directory holding it plus its `mmproj-*.gguf` sibling) - see
/// `qwen3vl::caps::classify_source`.
pub struct Qwen3VlResident {
    /// The served model id: the checkpoint's own `<vendor>/<repo>` when the
    /// store resolved it, [`MODEL`] when the operator pinned it.
    id: String,
    dir: String,
    /// The tier a request that names none builds: the encoding the
    /// checkpoint stores (`qwen3vl::caps::Precision::stored`), so a quantized
    /// GGUF is served as stored, never widened to fp32 by default.
    stored: Precision,
}

impl Qwen3VlResident {
    /// One resident per Qwen3-VL checkpoint the store holds, each under its
    /// own `<vendor>/<repo>` id (`qwen3vl::spec::Qwen3VlSpec::instance_role`):
    /// two repos are two models, not one ambiguous choice. A pinned
    /// `qwen3vl.weights` serves exactly that checkpoint, as [`MODEL`].
    pub fn all_from_store(store: &loader::served::ServedStore) -> Vec<Qwen3VlResident> {
        loader::served::served_assemblies(store, "qwen3vl", &qwen3vl::spec::Qwen3VlSpec, MODEL)
            .into_iter()
            .filter_map(|assembly| Self::new(assembly.id.clone(), assembly.role_path("weights").ok()?))
            .collect()
    }

    /// Direct constructor for a caller that already holds the checkpoint;
    /// `None` when the path is absent or its header cannot say how it is
    /// stored - registering a model whose every call would fail is worse
    /// than not serving it.
    pub fn new(id: impl Into<String>, dir: impl Into<String>) -> Option<Qwen3VlResident> {
        let (id, dir) = (id.into(), dir.into());
        if !std::path::Path::new(&dir).exists() {
            residency::log::not_served(&format!("brain: {id} not served ({dir} does not exist)"));
            return None;
        }
        match Precision::stored(&dir) {
            Ok(stored) => Some(Qwen3VlResident { id, dir, stored }),
            Err(e) => {
                residency::log::not_served(&format!("brain: {id} not served ({e})"));
                None
            }
        }
    }

    /// The tier name a request keys its instance by: the one it names, else
    /// [`Self::stored`]'s. A name that is no tier is kept as given, so
    /// `activate` refuses it by name (`parse_key`) instead of building some
    /// other tier in its place.
    fn precision_of(&self, inv: &Invocation) -> String {
        match inv.get_str("precision").filter(|name| !name.is_empty()) {
            Some(name) => Precision::from_name(&name).map_or(name, |p| p.name().to_string()),
            None => self.stored.name().to_string(),
        }
    }
}

impl ResidentModel for Qwen3VlResident {
    fn manifest(&self) -> Manifest {
        // The stripped, weights-free spec: this resident's checkpoint path is
        // already resolved (`self.dir`, from `from_store`/`new`), so a served
        // caller must never be told a `weights` param exists to set - see
        // `qwen3vl::caps::manifest_resident`'s doc. `run` below never reads
        // `weights` from `inv` either (`Resident::generate` takes none).
        Manifest { model: self.id.clone(), ..qwen3vl::caps::manifest_resident() }
    }

    fn instance_key(&self, _action: &str, inv: &Invocation) -> InstanceKey {
        // The capacity (`max_pixels`, `ctx`) and `precision` are all part of
        // the identity: each combination is a differently-sized build
        // (DeepStack/splice buffers, KV cache, decoder linear width), and
        // sharing one key across them would let a stray request evict a
        // working instance to build a differently-shaped one.
        let Capacity { max_pixels, ctx } = Capacity::of(inv);
        let p = self.precision_of(inv);
        InstanceKey::new(&self.id, format!("{}|{max_pixels}|{ctx}|{p}", self.dir))
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        // Reported as VRAM: both towers go wherever this instance is placed,
        // so on a GPU box the whole footprint is device memory.
        // `place::pick_device` falls a weight-holding model back to the CPU
        // pool at the same figure on a machine with no GPU, which is its own
        // rule and the behaviour this model wants.
        // A key `activate` will refuse is budgeted as this checkpoint's stored tier.
        let (_, capacity, precision) = parse_key(key).unwrap_or((String::new(), Capacity::default(), self.stored));
        // Read THIS checkpoint's own config (never the request's `dir` from
        // `key` -- `self.dir` is the one this resident actually serves) and
        // derive the real figure from it; only a config this resident cannot
        // even read (a bad path, a malformed file) falls back to the
        // hand-derived 4B constant, which `activate` would refuse on anyway.
        let bytes = qwen3vl::caps::config_and_capacity(&self.dir, capacity)
            .map(|(cfg, n_visual_capacity, seq_len)| qwen3vl::footprint::estimate_vram_bytes(&cfg, precision, seq_len, n_visual_capacity))
            .unwrap_or(match precision {
                Precision::I8 => INT8_BYTES,
                Precision::F32 => FP32_BYTES,
            });
        MemCost::new(bytes, 0)
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let (dir, capacity, precision) = parse_key(key)?;
        let gpu = match device {
            Device::Cpu => None,
            Device::Gpu(i) => Some(i),
            Device::Npu(i) => {
                // This model advertises no NPU footprint, so the placer never
                // offers one; refuse by name rather than silently building wgpu.
                return Err(format!("qwen3vl: assigned Npu({i}), but this model has no NPU export path"));
            }
        };
        let resident = Resident::load_on(&dir, capacity, precision, gpu)?;
        Ok(Box::new(Qwen3VlInstance { resident }))
    }
}

/// `(dir, capacity, precision)` out of an [`InstanceKey`]'s
/// `"{dir}|{max_pixels}|{ctx}|{precision}"` config string. Splitting from the
/// RIGHT (`rsplitn`) means a directory path containing `|` still parses
/// correctly, since only the last three fields are peeled off.
fn parse_key(key: &InstanceKey) -> Result<(String, Capacity, Precision), String> {
    let mut parts = key.config.rsplitn(4, '|');
    let malformed = || "qwen3vl: malformed instance key".to_string();
    let precision = Precision::from_name(parts.next().ok_or_else(malformed)?)?;
    let ctx: u32 = parts.next().ok_or_else(malformed)?.parse().map_err(|_| "qwen3vl: malformed instance key (ctx)".to_string())?;
    let max_pixels: u32 = parts.next().ok_or_else(malformed)?.parse().map_err(|_| "qwen3vl: malformed instance key (max_pixels)".to_string())?;
    let dir = parts.next().ok_or_else(malformed)?.to_string();
    Ok((dir, Capacity { max_pixels, ctx }, precision))
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
        let video_frames = qwen3vl::caps::decode_media(inv)?;
        let (tool_choice, tools) = qwen3vl::caps::parse_tool_request(inv)?;
        let seed = inv.seed()?.unwrap_or_else(data::rng::random_seed);
        self.resident.generate(inv, video_frames, tool_choice, &tools, seed, progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A weight-free llama.cpp release at `<store>/<repo>`: the language half
    /// `lm` (one Q8_0 decoder linear) beside its F16 projector, one block each.
    fn write_gguf_release(store: &std::path::Path, repo: &str, lm: &str, mmproj: &str) -> std::path::PathBuf {
        use checkpoint::gguf::{GgmlType, GgufValue};
        let dir = store.join(repo);
        std::fs::create_dir_all(&dir).unwrap();
        let write = |file: &str, kv: &[(&str, &str)], (name, ty): (&str, GgmlType)| {
            let tensor = checkpoint::gguf_write::TensorOut { name: name.to_string(), shape: vec![ty.block_elems()], ty: ty.id(), data: vec![0u8; ty.block_bytes()] };
            let kv: Vec<_> = kv.iter().map(|(k, v)| (k.to_string(), GgufValue::String(v.to_string()))).collect();
            checkpoint::gguf_write::write(dir.join(file).to_str().unwrap(), &kv, &[tensor], 32).unwrap();
        };
        write(lm, &[("general.architecture", qwen3vl::gguf_import::GGUF_ARCHITECTURE)], ("blk.0.attn_q.weight", GgmlType::Q8_0));
        write(mmproj, &[("general.architecture", "clip"), ("clip.projector_type", qwen3vl::gguf_import::PROJECTOR_TYPE)], ("v.patch_embd.weight", GgmlType::F16));
        dir.join(lm)
    }

    /// Two Qwen3-VL repos in one store are two served models, each under its
    /// own `<vendor>/<repo>` id and bound to its own checkpoint, with nothing
    /// pinned. Resolving them as one role would be ambiguous and serve
    /// neither.
    #[test]
    fn every_qwen3vl_repo_in_the_store_is_its_own_served_model() {
        let store = std::env::temp_dir().join(format!("brain-resident-qwen3vl-two-repos-{}", std::process::id()));
        std::fs::remove_dir_all(&store).ok();
        let small = write_gguf_release(&store, "unsloth/Qwen3-VL-4B-Instruct-GGUF", "Q8_0.gguf", "mmproj-F16.gguf");
        let large = write_gguf_release(&store, "Qwen/Qwen3-VL", "Qwen3-VL-8B-Instruct-abliterated-v2.Q8_0.gguf", "Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-Q8_0.gguf");

        let served = crate::residents(&loader::served::ServedStore::new(Some(store.clone())));
        let mut listed: Vec<(String, Option<String>)> = served.iter().map(|m| m.manifest()).filter(|m| m.architecture.as_deref() == Some("qwen3vl")).map(|m| (m.model, m.architecture)).collect();
        listed.sort();
        assert_eq!(
            listed,
            [("Qwen/Qwen3-VL".to_string(), Some("qwen3vl".to_string())), ("unsloth/Qwen3-VL-4B-Instruct-GGUF".to_string(), Some("qwen3vl".to_string()))],
            "both repos are listed, each under its own id"
        );

        let mut bound: Vec<(String, String)> = Qwen3VlResident::all_from_store(&loader::served::ServedStore::new(Some(store.clone()))).into_iter().map(|r| (r.id, r.dir)).collect();
        bound.sort();
        assert_eq!(
            bound,
            [("Qwen/Qwen3-VL".to_string(), large.to_string_lossy().into_owned()), ("unsloth/Qwen3-VL-4B-Instruct-GGUF".to_string(), small.to_string_lossy().into_owned())],
            "each model resolves to its own checkpoint"
        );
        std::fs::remove_dir_all(&store).ok();
    }

    /// An unconfigured checkpoint yields no resident at all, rather than one
    /// that fails every call.
    #[test]
    fn a_missing_checkpoint_is_not_registered() {
        assert!(Qwen3VlResident::new(MODEL, "/definitely/not/a/qwen3vl/dir").is_none());
    }

    /// `max_pixels` and `precision` are both part of the instance identity.
    /// Sharing one key across a precision change would let a single int8
    /// request evict a working fp32 instance built for a different capacity.
    #[test]
    fn max_pixels_and_precision_key_separate_instances() {
        let r = Qwen3VlResident { id: MODEL.into(), dir: "/tmp".into(), stored: Precision::F32 };
        let k_default = r.instance_key("generate", &Invocation::new());
        let k_i8 = r.instance_key("generate", &Invocation::new().set("precision", serde_json::json!("int8")));
        let k_bigger = r.instance_key("generate", &Invocation::new().set("max_pixels", serde_json::json!(2_000_000)));
        assert_ne!(k_default, k_i8);
        assert_ne!(k_default, k_bigger);
        assert_ne!(k_i8, k_bigger);
        // A spelling of the stored tier is the same instance; a name that is
        // no tier is refused when the instance is built, never replaced.
        assert_eq!(r.instance_key("generate", &Invocation::new().set("precision", serde_json::json!("f32"))), k_default);
        let bogus = r.instance_key("generate", &Invocation::new().set("precision", serde_json::json!("bogus")));
        assert_ne!(bogus, k_default);
        let e = r.activate(&bogus, Device::Cpu).err().unwrap_or_default();
        assert!(e.contains("bogus"), "{e}");
    }

    /// A Q8_0 GGUF is served as it stores its decoder: a request naming no
    /// precision keys (and budgets) the int8 instance, not an fp32 upcast,
    /// and fp32 is still built when a request asks for it.
    #[test]
    fn a_q8_0_gguf_is_served_at_its_stored_int8_and_fp32_is_asked_for_by_name() {
        let store = std::env::temp_dir().join(format!("brain-resident-qwen3vl-stored-{}", std::process::id()));
        std::fs::remove_dir_all(&store).ok();
        let lm = write_gguf_release(&store, "unsloth/Qwen3-VL-4B-Instruct-GGUF", "Q8_0.gguf", "mmproj-F16.gguf");
        let r = Qwen3VlResident::new("unsloth/Qwen3-VL-4B-Instruct-GGUF", lm.to_string_lossy()).expect("a resident for the fixture");

        let precision = |inv: &Invocation| parse_key(&r.instance_key("generate", inv)).map(|(_, _, p)| p);
        assert_eq!(precision(&Invocation::new()), Ok(Precision::I8), "no precision asked for keeps the stored Q8_0 as int8");
        assert_eq!(precision(&Invocation::new().set("precision", serde_json::json!("fp32"))), Ok(Precision::F32), "fp32 is an explicit choice");
        std::fs::remove_dir_all(&store).ok();
    }

    /// The two precisions must be budgeted differently, and fp32 must be the
    /// larger one - the whole point of the int8 tier existing.
    #[test]
    fn the_two_precisions_are_budgeted_apart() {
        let r = Qwen3VlResident { id: MODEL.into(), dir: "/tmp".into(), stored: Precision::F32 };
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
        let r = Qwen3VlResident { id: MODEL.into(), dir: "/tmp".into(), stored: Precision::F32 };
        let e = r.activate(&r.instance_key("generate", &Invocation::new()), Device::Npu(0)).err().unwrap_or_default();
        assert!(e.contains("no NPU export path"), "{e}");
    }

    /// The resident manifest must never advertise `weights` as a settable
    /// param - a served caller cannot point this instance at a different
    /// checkpoint, only the operator (the model store or a `qwen3vl.weights` pin) can.
    #[test]
    fn the_resident_manifest_has_no_weights_param() {
        let r = Qwen3VlResident { id: MODEL.into(), dir: "/tmp".into(), stored: Precision::F32 };
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
        let resident = Qwen3VlResident { id: MODEL.into(), dir: "/nonexistent/qwen3vl".to_string(), stored: Precision::F32 };
        let models: Vec<std::sync::Arc<dyn ResidentModel>> = vec![std::sync::Arc::new(resident)];
        let mut budgets = residency::budget::Budgets::new();
        budgets.set(Device::Cpu, 8 << 30, 0);
        let exec = residency::Executor::start(models, budgets, residency::scheduler::Policy::default());

        // The exact classification D-Bus's `Manager::manifests`/`list_models`
        // and HTTP's `/v1/models` both build on: this is `exec.manifests()`,
        // the one source both transports read.
        assert!(
            exec.manifests().iter().any(|m| m.model == MODEL),
            "qwen3vl must be registered in the executor's manifest set once a resident is built for a checkpoint"
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
