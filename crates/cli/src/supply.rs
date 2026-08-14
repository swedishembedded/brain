// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`residency::ModelSupplier`] backed by the model store
//! (`brain_modelstore`): classifies a model name against the naming grammar
//! and reserved-vendor gate (no network for anything `classify` alone can
//! answer), and fetches via the store's resolution ladder on `ensure`.
//!
//! Single-flight: concurrent `ensure` calls for the SAME model share one
//! underlying fetch (via [`std::sync::Condvar`]) -- ten simultaneous
//! requests for a cold model download it once, not ten times.
//!
//! Honest scope note: `ensure` completes any plan whose deferred steps are
//! pure `Download` and/or `Convert` -- covering every base ref, and a quant
//! ref whose upstream `-GGUF` sibling repo already has the file. A plan that
//! still needs LOCAL QUANTIZATION (a quant ref with no upstream artifact) is
//! real but not yet automated end-to-end here -- driving
//! `checkpoint::quant`/`gguf_write` for that step is follow-up work (Phase 9
//! in the design doc: a from-scratch GGUF quantizer). `ensure` fails cleanly
//! with which steps are missing rather than silently producing a wrong
//! checkpoint.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex};

use brain_modelref::ModelRef;
use brain_modelstore::recipe::ZimageRecipe;
use brain_modelstore::{CompoundManifest, Hub, Step, Store, MANIFEST_FILE};
use residency::{Executor, ModelSupplier, Supply};

/// Dispatch a `Step::Convert { vendor, repo, recipe }` to the matching
/// family's finish logic. `recipe` is the `ArtifactRecipe::id` `modelstore::
/// plan` already picked (`brain_modelstore::recipe`) -- routing on it directly
/// rather than re-deriving the family from disk a second time, one
/// implementation of "which family this repo is", not a second guess that
/// could drift from the first.
fn convert(store: &Store, vendor: &str, repo: &str, recipe: &str) -> Result<(), String> {
    match recipe {
        "transformers" => convert_transformers(store, vendor, repo),
        "zimage" => convert_zimage(store, vendor, repo),
        "yolo" => convert_yolo(store, vendor, repo),
        other => Err(format!("{vendor}/{repo}: convert: unknown recipe {other:?} (bug: modelstore::recipe::recipes() and this dispatch have drifted)")),
    }
}

/// The yolo recipe: `YoloRecipe::artifacts` downloaded exactly one
/// `yolov8*.pt` file into the repo dir; run the pure-Rust importer
/// (`yolov8::import::import_yolov8n`, built on `checkpoint::torchpt`) and write
/// the remapped tensors as `model.brain.safetensors` -- the same single-file
/// convention every transformers-family model already uses, so no store or
/// `resident_for` changes were needed for this family.
fn convert_yolo(store: &Store, vendor: &str, repo: &str) -> Result<(), String> {
    let dir = store.repo_dir(&ModelRef::new(vendor, repo, None));
    let pt = std::fs::read_dir(&dir)
        .map_err(|e| format!("{vendor}/{repo}: convert: {}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("yolov8") && n.ends_with(".pt")))
        .ok_or_else(|| format!("{vendor}/{repo}: convert: no downloaded yolov8*.pt file in {}", dir.display()))?;
    let pt_str = pt.to_str().ok_or_else(|| format!("{vendor}/{repo}: convert: non-UTF8 path {}", pt.display()))?;

    let tensors = yolov8::import::import_yolov8n(pt_str)?;
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = tensors.into_iter().map(|(name, shape, data)| (name, shape.into_iter().map(|d| d as u64).collect(), data)).collect();
    let card = checkpoint::st::ModelCard::for_ref(&format!("{vendor}/{repo}"), vendor, repo, None, "yolo");
    let out = dir.join("model.brain.safetensors");
    checkpoint::st::save_safetensors(out.to_str().ok_or_else(|| format!("{vendor}/{repo}: convert: non-UTF8 store path"))?, &tensors, &yolov8::config::YoloConfig::yolov8n().to_json(), Some(&card))
        .map_err(|e| format!("{vendor}/{repo}: convert: write model.brain.safetensors: {e}"))?;
    // The upstream .pt is never read again -- Store::local/scan only ever load
    // model.brain.safetensors (see modelstore::BASE_WEIGHTS_FILE) -- so keeping
    // it around is pure disk waste. Best-effort: a failed cleanup must not fail
    // an otherwise-successful convert.
    std::fs::remove_file(&pt).ok();
    Ok(())
}

/// The zimage recipe: no tensor rewrite is needed (`s3dit::import::
/// import_comfy` already remaps names in memory at load time), so "finish"
/// is just writing the `brain.manifest.json` `Store::local` reads back --
/// naming the SAME four role paths `ZimageRecipe::artifacts` just downloaded,
/// from `ZimageRecipe::ROLES` (one source of truth for z-image's role
/// layout, not a second guess of what landed on disk).
fn convert_zimage(store: &Store, vendor: &str, repo: &str) -> Result<(), String> {
    let dir = store.repo_dir(&ModelRef::new(vendor, repo, None));
    let mut roles = BTreeMap::new();
    for (role, rel) in ZimageRecipe::ROLES {
        if !dir.join(rel).exists() {
            return Err(format!("{vendor}/{repo}: convert: role {role:?} ({rel}) did not download"));
        }
        roles.insert(role.to_string(), rel.to_string());
    }
    let manifest = CompoundManifest { id: format!("{vendor}/{repo}"), family: "zimage".to_string(), roles };
    let bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| format!("{vendor}/{repo}: convert: encode manifest: {e}"))?;
    std::fs::write(dir.join(MANIFEST_FILE), bytes).map_err(|e| format!("{vendor}/{repo}: convert: write manifest: {e}"))
}

/// The original (and still only) family: an HF `transformers`-shaped repo.
/// Reads `<dir>/config.json` to pick the specific qwen/glm/lfm/gpt importer
/// the same way `modelstore::plan`'s `TransformersRecipe` already gated the
/// download on (`family_of_architecture`) -- one implementation of "which
/// families brain can serve", not a second guess that could drift from the
/// first. The produced card's `id` is overridden to `vendor/repo` (each
/// importer otherwise derives it from the output filename) so the resident
/// registers under the fully-qualified reference the client actually asked
/// for, not `"model.brain"`.
fn convert_transformers(store: &Store, vendor: &str, repo: &str) -> Result<(), String> {
    let dir = store.repo_dir(&ModelRef::new(vendor, repo, None));
    let config_bytes = std::fs::read(dir.join("config.json")).map_err(|e| format!("{vendor}/{repo}: read config.json: {e}"))?;
    let config: serde_json::Value = serde_json::from_slice(&config_bytes).map_err(|e| format!("{vendor}/{repo}: config.json: {e}"))?;
    let arch = brain_modelstore::declared_architecture(&config).ok_or_else(|| format!("{vendor}/{repo}: config.json has no architecture"))?;
    let family = brain_modelstore::family_of_architecture(&arch).ok_or_else(|| format!("{vendor}/{repo}: unsupported architecture {arch:?}"))?;

    let hf_dir = dir.to_str().ok_or_else(|| format!("{vendor}/{repo}: non-UTF8 store path"))?;
    let out_path = dir.join("model.brain.safetensors");
    let out = out_path.to_str().ok_or_else(|| format!("{vendor}/{repo}: non-UTF8 store path"))?;
    let id = format!("{vendor}/{repo}");

    let result = match family {
        "qwen3" => qwen3::import::import_as(hf_dir, out, None, Some(&id)),
        "glmdsa" => glmdsa::import::import_as(hf_dir, out, Some(&id)),
        "lfm2" => lfm2::import::import_as(hf_dir, out, Some(&id)),
        // gpt2 is nanogpt-style, trained from scratch -- brain has never had
        // an HF importer for it (unlike glmdsa/qwen3/lfm2, all
        // production-tested). Writing one is real new-crate work, not "wire
        // the dispatch", so this fails cleanly instead of guessing at a
        // Conv1D-transpose import.
        "gpt2" => Err("gpt2 has no HF import path yet -- fetch and convert manually".to_string()),
        // qwen3omnimoe (Qwen3-Omni) is recognized via an exact HF class-name
        // match, so it is never mis-routed to the dense qwen3 importer even
        // though its class name contains "qwen" as a substring. The importer
        // itself streams from the sharded HF dir fine (M3) -- what is NOT yet
        // true is that the resulting unified checkpoint is directly loadable
        // by qwen3tts::mtp::MtpModel/mimi::Codec for the Talker/Code2Wav pieces
        // (two open naming gaps); Thinker-only generation
        // (crate::resident_omni, gated on BRAIN_QWEN3OMNIMOE_HF_DIR, not this
        // converted-checkpoint path) is unaffected by either gap.
        "qwen3omnimoe" => qwen3omnimoe::import::import_as(hf_dir, out, Some(&id)),
        other => Err(format!("architecture {other:?} matched but has no dispatch arm (bug: family_of_architecture and this match have drifted)")),
    };
    result.map_err(|e| format!("{vendor}/{repo}: convert: {e}"))?;
    // The upstream weights (single model.safetensors, or a model-*-of-*.safetensors
    // shard set + its index) are never read again once model.brain.safetensors
    // exists -- Store::local/scan only ever load BASE_WEIGHTS_FILE -- so keeping
    // them is pure disk waste (often larger than the converted file itself, e.g.
    // a bf16 upstream vs. brain's fp32-only format). Best-effort: a failed
    // cleanup must not fail an otherwise-successful convert.
    remove_upstream_weights(&dir);
    Ok(())
}

/// See [`convert_transformers`]'s cleanup note. Handles both shapes
/// `TransformersRecipe::artifacts` can have downloaded: a single
/// `model.safetensors`, or a `model.safetensors.index.json` + its
/// `model-NNNNN-of-NNNNN.safetensors` shard set.
fn remove_upstream_weights(dir: &Path) {
    let single = dir.join("model.safetensors");
    if single.exists() {
        std::fs::remove_file(&single).ok();
        return;
    }
    std::fs::remove_file(dir.join("model.safetensors.index.json")).ok();
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for name in entries.filter_map(|e| e.ok()).map(|e| e.file_name()) {
        let name = name.to_string_lossy();
        if name.starts_with("model-") && name.ends_with(".safetensors") {
            std::fs::remove_file(dir.join(&*name)).ok();
        }
    }
}

/// Constructed by `run_cli.rs::build_auto_fetch_supplier` and threaded into
/// every HTTP/D-Bus surface (`run_apis`), behind `BRAIN_AUTO_FETCH=0` to
/// disable. Wiring it in went through the full watertight-API security pass
/// (AGENTS.md).
#[derive(Clone)]
enum FetchState {
    Running,
    Done(Result<(), String>),
}

/// The 10%-bucket to report for one file's download progress (`got`/`total`
/// bytes), given the last bucket already reported for THIS file (`None` if
/// never reported yet) -- `Some(bucket)` only when `bucket` is a NEW
/// threshold this call just crossed, so a caller logs at most once per 10%
/// rather than once per raw progress tick (the hub's chunk size, not a
/// number an operator watching `-v -v` would choose to see scroll by).
/// `total == 0` (genuinely empty file, or a host that never reported a
/// Content-Length) has no meaningful percentage -- `None`, never a divide.
/// Pure so it's directly testable without a real download.
fn next_download_pct_bucket(got: u64, total: u64, last: Option<u32>) -> Option<u32> {
    if total == 0 {
        return None;
    }
    let bucket = (got.min(total) * 100 / total / 10 * 10) as u32;
    last.is_none_or(|l| bucket > l).then_some(bucket)
}

/// Auto-fetch `arch`'s [`brain_arch::Arch::default_ref`] checkpoint into the
/// model store (fetching + converting exactly as [`StoreSupplier::ensure`]
/// does for a server request), and return the path to its
/// `model.brain.safetensors` -- the one thing every dedicated `_cli.rs`
/// handler's `--weights F` flag already expects. `crate::resolve` calls this
/// to inject `--weights <path>` into an `infer` invocation that named none,
/// so `brain infer zipdepth --in image=x.jpg` (no `--weights`) resolves a
/// concrete checkpoint on its own.
///
/// Not single-flight (unlike [`StoreSupplier`], built for concurrent server
/// requests sharing one long-lived process): a one-shot CLI invocation has
/// exactly one caller, so the plain plan/execute/convert sequence is enough
/// -- two `brain infer` processes racing the same cold fetch is a real but
/// rare case, and each just refetches into the same destination independently
/// rather than corrupting anything (`brain_modelstore::fetch` writes via a
/// temp file + atomic rename).
pub fn ensure_default_weights(arch: &str) -> Result<DefaultWeights, String> {
    let root = crate::model_dir::resolve(None).ok_or_else(|| "no models directory (no $HOME and no $BRAIN_MODELS_DIR)".to_string())?;
    ensure_default_weights_with(arch, &Store::new(root), &brain_modelstore::HfHub::new())
}

/// [`ensure_default_weights`]'s result: the weights path every architecture
/// needs, plus the tokenizer path for the ones that also need one (a fetched
/// HF checkpoint's `tokenizer.json`, when present) -- what lets
/// [`crate::resolve::maybe_inject_default_weights`] inject both `--weights`
/// and `--tokenizer` for a flagless `brain infer <arch>`.
#[derive(Debug)]
pub struct DefaultWeights {
    pub weights: String,
    pub tokenizer: Option<String>,
}

/// [`ensure_default_weights`]'s implementation, taking `store`/`hub`
/// explicitly so it is testable against [`brain_modelstore::FakeHub`] with no
/// real network or `$HOME` -- the same split every other fetch path in this
/// file (`StoreSupplier`, `convert_*`) already uses.
fn ensure_default_weights_with(arch: &str, store: &Store, hub: &dyn Hub) -> Result<DefaultWeights, String> {
    let a = brain_arch::by_id(arch).ok_or_else(|| format!("{arch}: not a registered architecture"))?;
    let default_ref = a.default_ref.ok_or_else(|| format!("{arch}: no default checkpoint known -- pass --weights explicitly"))?;
    let reference = ModelRef::parse(default_ref).map_err(|e| format!("{default_ref}: {e}"))?;

    let plan = brain_modelstore::plan(&reference, store, hub).map_err(|e| format!("{default_ref}: {e}"))?;
    let mut last_pct: HashMap<String, u32> = HashMap::new();
    let deferred = brain_modelstore::execute(store, hub, &plan, &mut |name, got, total| {
        if let Some(total) = total {
            if let Some(bucket) = next_download_pct_bucket(got, total, last_pct.get(name).copied()) {
                residency::log::info(&format!("{default_ref}: downloading {name} {bucket}%"));
                last_pct.insert(name.to_string(), bucket);
            }
        }
    })
    .map_err(|e| format!("{default_ref}: {e}"))?;

    for step in &deferred {
        match step {
            Step::Convert { vendor, repo, recipe } => convert(store, vendor, repo, recipe).map_err(|e| format!("{default_ref}: {e}"))?,
            other => {
                return Err(format!(
                    "{default_ref}: needs an additional step ({other:?}) auto-fetch does not automate yet -- fetch and convert manually"
                ))
            }
        }
    }

    let local = store.local(&reference).ok_or_else(|| format!("{default_ref}: fetched but not found on disk (unexpected)"))?;
    let weights = local.weights.to_str().map(str::to_string).ok_or_else(|| format!("{default_ref}: non-UTF8 store path"))?;
    let tokenizer = local.tokenizer.as_deref().and_then(|p| p.to_str()).map(str::to_string);
    Ok(DefaultWeights { weights, tokenizer })
}

/// One single-flight gate per model: the fetch's outcome plus the condvar
/// that wakes every other concurrent `ensure` call waiting on it.
type FetchGate = Arc<(Mutex<FetchState>, Condvar)>;

pub struct StoreSupplier {
    store: Store,
    hub: Box<dyn Hub>,
    inflight: Mutex<HashMap<String, FetchGate>>,
}

impl StoreSupplier {
    pub fn new(store: Store, hub: Box<dyn Hub>) -> StoreSupplier {
        StoreSupplier { store, hub, inflight: Mutex::new(HashMap::new()) }
    }

    fn do_ensure(&self, model: &str, exec: &Executor, progress: &mut dyn FnMut(&str, u32, u32)) -> Result<(), String> {
        let r = ModelRef::parse(model).map_err(|e| format!("{model}: {e}"))?;
        let plan = brain_modelstore::plan(&r, &self.store, self.hub.as_ref()).map_err(|e| format!("{model}: {e}"))?;
        // The last 10%-bucket logged per downloaded file, so a caller watching
        // `-v -v` sees "downloading X% ... 10% ... 20% ..." instead of either
        // silence (today's complaint: no visibility that a fetch is even
        // happening) or one line per raw progress tick (the hub's chunk size,
        // not a number an operator would choose to watch scroll by).
        let mut last_pct: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        let deferred = brain_modelstore::execute(&self.store, self.hub.as_ref(), &plan, &mut |name, got, total| {
            if let Some(total) = total {
                if let Some(bucket) = next_download_pct_bucket(got, total, last_pct.get(name).copied()) {
                    residency::log::info(&format!("{model}: downloading {name} {bucket}%"));
                    last_pct.insert(name.to_string(), bucket);
                }
            }
            progress(name, got.min(u32::MAX as u64) as u32, total.unwrap_or(0).min(u32::MAX as u64) as u32);
        })
        .map_err(|e| format!("{model}: {e}"))?;

        // Convert is automated (dispatch by architecture, see `convert` above);
        // Quantize is not yet (Phase 9 -- a from-scratch GGUF quantizer). Run
        // every Convert now so a base checkpoint this plan needed is left
        // servable on disk even when the ORIGINAL request was a quant ref this
        // call still can't finish.
        let mut still_missing = Vec::new();
        for step in &deferred {
            match step {
                Step::Convert { vendor, repo, recipe } => convert(&self.store, vendor, repo, recipe).map_err(|e| format!("{model}: {e}"))?,
                other => still_missing.push(other.clone()),
            }
        }
        if !still_missing.is_empty() {
            return Err(format!(
                "{model}: needs {} additional step(s) (local-quantize) that auto-fetch does not automate yet -- fetch and convert manually",
                still_missing.len()
            ));
        }

        // `exec.register_if_absent`, not `register`: the `inflight` single-flight
        // gate above only serializes callers that overlap IN TIME -- a straggler
        // that lands after the leader already finished and tore its gate down
        // starts a fresh, unguarded episode here. `register_if_absent` closes
        // that gap atomically (unlike a separate check-then-`register()`, which
        // is itself a TOCTOU race between episodes), making this call genuinely
        // idempotent per this trait's own "MUST be idempotent" doc.
        let local = self.store.local(&r).ok_or_else(|| format!("{model}: fetched but not found on disk (unexpected)"))?;
        let resident = crate::model_dir::resident_for_local(&local).ok_or_else(|| format!("{model}: family not servable"))?;
        exec.register_if_absent(resident);
        Ok(())
    }
}

impl ModelSupplier for StoreSupplier {
    /// Grammar + reserved-vendor gate only -- zero network/filesystem I/O
    /// beyond what `ModelRef::parse` needs (none). A reserved vendor with
    /// nothing already resident is `Unknown` unconditionally: this is what
    /// keeps a discovery endpoint (`GET /models`) safe to call with an
    /// attacker-chosen name with no risk of an outbound request.
    fn classify(&self, model: &str) -> Supply {
        match ModelRef::parse(model) {
            Ok(r) if r.is_reserved() => Supply::Unknown(format!("{model}: reserved vendor, not on disk")),
            Ok(_) => Supply::Fetchable,
            Err(e) => Supply::Unknown(format!("{model}: {e}")),
        }
    }

    fn ensure(&self, model: &str, exec: &Executor, progress: &mut dyn FnMut(&str, u32, u32)) -> Result<(), String> {
        let existing = {
            let mut map = self.inflight.lock().unwrap();
            match map.get(model) {
                Some(slot) => Some(slot.clone()),
                None => {
                    map.insert(model.to_string(), Arc::new((Mutex::new(FetchState::Running), Condvar::new())));
                    None
                }
            }
        };
        if let Some(slot) = existing {
            // A follower: another thread is already fetching this model --
            // wait for it and share its result rather than fetching again.
            let (lock, cv) = &*slot;
            let mut state = lock.lock().unwrap();
            while matches!(*state, FetchState::Running) {
                state = cv.wait(state).unwrap();
            }
            return match &*state {
                FetchState::Done(r) => r.clone(),
                FetchState::Running => unreachable!("woke from wait while still Running"),
            };
        }

        // The leader: do the real work, then publish the result to every
        // follower waiting on the same slot.
        let result = self.do_ensure(model, exec, progress);
        let slot = self.inflight.lock().unwrap().get(model).cloned();
        if let Some(slot) = slot {
            let (lock, cv) = &*slot;
            *lock.lock().unwrap() = FetchState::Done(result.clone());
            cv.notify_all();
        }
        self.inflight.lock().unwrap().remove(model);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::FakeHub;
    use residency::budget::Budgets;
    use residency::{Device, Policy};

    fn store(name: &str) -> Store {
        let dir = std::env::temp_dir().join(name);
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        Store::new(dir)
    }

    fn exec() -> Executor {
        let mut budgets = Budgets::new();
        budgets.set(Device::Cpu, 1 << 30, 0);
        Executor::start(vec![], budgets, Policy::default())
    }

    #[test]
    fn next_download_pct_bucket_reports_each_new_10pct_threshold_once() {
        // Starts with 0%, per spec -- the very first call (nothing seen yet)
        // at 0 bytes must report Some(0), not silence until the first real
        // threshold.
        assert_eq!(next_download_pct_bucket(0, 1000, None), Some(0));
        // A later call still inside the SAME bucket (0%) reports nothing new.
        assert_eq!(next_download_pct_bucket(50, 1000, Some(0)), None);
        // Crossing into a new bucket reports it.
        assert_eq!(next_download_pct_bucket(105, 1000, Some(0)), Some(10));
        // A coarse jump (e.g. one big chunk) reports the NEW bucket directly,
        // not every threshold it skipped over.
        assert_eq!(next_download_pct_bucket(800, 1000, Some(10)), Some(80));
        // Completion.
        assert_eq!(next_download_pct_bucket(1000, 1000, Some(80)), Some(100));
        // got > total (a defensive clamp -- a host lying about Content-Length,
        // or a stream that overran it) must not compute over 100% or panic.
        assert_eq!(next_download_pct_bucket(1500, 1000, Some(80)), Some(100));
        // total == 0 has no meaningful percentage and must never divide by it.
        assert_eq!(next_download_pct_bucket(0, 0, None), None);
        assert_eq!(next_download_pct_bucket(500, 0, None), None);
    }

    #[test]
    fn ensure_default_weights_fetches_converts_and_returns_the_brain_safetensors_path() {
        let (config, weights) = tiny_qwen3_hf_files();
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "config.json", config);
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model.safetensors", weights);
        let store = store("supply-test-default-weights-qwen3");

        let got = ensure_default_weights_with("qwen3", &store, &hub).unwrap();
        assert!(got.weights.ends_with("Qwen/Qwen3-0.6B/model.brain.safetensors"), "{}", got.weights);
        assert!(std::path::Path::new(&got.weights).exists(), "{} must actually exist on disk", got.weights);
    }

    #[test]
    fn ensure_default_weights_is_a_clean_error_for_an_arch_with_no_default_ref() {
        // t5encoder has no default_ref (no confirmed small upstream repo
        // yet) -- must fail with a clear reason, never panic or silently
        // pick something.
        let store = store("supply-test-default-weights-no-ref");
        let hub = FakeHub::new();
        let err = ensure_default_weights_with("t5encoder", &store, &hub).unwrap_err();
        assert!(err.contains("no default checkpoint known"), "{err}");
    }

    #[test]
    fn ensure_default_weights_is_a_clean_error_for_an_unknown_arch() {
        let store = store("supply-test-default-weights-unknown-arch");
        let hub = FakeHub::new();
        let err = ensure_default_weights_with("totally-bogus", &store, &hub).unwrap_err();
        assert!(err.contains("not a registered architecture"), "{err}");
    }

    #[test]
    fn classify_refuses_a_reserved_vendor_with_no_network() {
        let supplier = StoreSupplier::new(store("supply-test-reserved"), Box::new(FakeHub::new()));
        assert!(matches!(supplier.classify("brain/mock"), Supply::Unknown(_)));
    }

    #[test]
    fn classify_refuses_an_invalid_name() {
        let supplier = StoreSupplier::new(store("supply-test-invalid"), Box::new(FakeHub::new()));
        assert!(matches!(supplier.classify("no-slash-here"), Supply::Unknown(_)));
    }

    #[test]
    fn classify_accepts_a_well_formed_non_reserved_name() {
        let supplier = StoreSupplier::new(store("supply-test-ok"), Box::new(FakeHub::new()));
        assert_eq!(supplier.classify("Qwen/Qwen3-0.6B"), Supply::Fetchable);
    }

    /// A tiny but real 1-layer tied-embedding Qwen3 HF checkpoint -- the same
    /// shape as `crates/qwen3/src/import.rs`'s own `build_tiny_hf_dir` test
    /// fixture, reproduced here as raw bytes for a [`FakeHub`] rather than a
    /// directory, since `ensure` must drive the whole plan -> download ->
    /// convert pipeline, not just call the importer directly.
    fn tiny_qwen3_hf_files() -> (Vec<u8>, Vec<u8>) {
        let config = br#"{"architectures":["Qwen3ForCausalLM"],
            "vocab_size":5,"hidden_size":6,"num_hidden_layers":1,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":8,"rope_theta":1000000,"rms_norm_eps":1e-6,
            "tie_word_embeddings":true}"#
            .to_vec();
        fn seq(base: f32, n: usize) -> Vec<f32> {
            (0..n).map(|i| base + i as f32).collect()
        }
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
            ("model.embed_tokens.weight".into(), vec![30], seq(1_000_000.0, 30)),
            ("model.norm.weight".into(), vec![6], seq(2_000_000.0, 6)),
            ("model.layers.0.input_layernorm.weight".into(), vec![6], seq(10.0, 6)),
            ("model.layers.0.self_attn.q_proj.weight".into(), vec![48], seq(20.0, 48)),
            ("model.layers.0.self_attn.k_proj.weight".into(), vec![24], seq(70.0, 24)),
            ("model.layers.0.self_attn.v_proj.weight".into(), vec![24], seq(100.0, 24)),
            ("model.layers.0.self_attn.q_norm.weight".into(), vec![4], seq(130.0, 4)),
            ("model.layers.0.self_attn.k_norm.weight".into(), vec![4], seq(140.0, 4)),
            ("model.layers.0.self_attn.o_proj.weight".into(), vec![48], seq(150.0, 48)),
            ("model.layers.0.post_attention_layernorm.weight".into(), vec![6], seq(200.0, 6)),
            ("model.layers.0.mlp.gate_proj.weight".into(), vec![48], seq(210.0, 48)),
            ("model.layers.0.mlp.up_proj.weight".into(), vec![48], seq(260.0, 48)),
            ("model.layers.0.mlp.down_proj.weight".into(), vec![48], seq(310.0, 48)),
        ];
        // A unique filename per CALL (not just per-process): two tests running
        // concurrently (the default, non---test-threads=1 harness) each call
        // this helper, and a shared PID-only name raced one test's cleanup
        // `remove_file` against the other's `save_safetensors` write.
        static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let out = std::env::temp_dir().join(format!("brain-supply-tiny-qwen3-{}-{n}.st", std::process::id()));
        checkpoint::st::save_safetensors(out.to_str().unwrap(), &tensors, &serde_json::Value::Null, None).unwrap();
        let weights = std::fs::read(&out).unwrap();
        std::fs::remove_file(&out).ok();
        (config, weights)
    }

    #[test]
    fn ensure_completes_a_base_ref_plan_by_converting_via_the_dispatch() {
        // No upstream GGUF sibling and no quant suffix -> plan_base -> the
        // deferred Convert step must be driven by `do_ensure`'s dispatch, not
        // left for the caller.
        let (config, weights) = tiny_qwen3_hf_files();
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "config.json", config);
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model.safetensors", weights);
        let supplier = StoreSupplier::new(store("supply-test-converts-base"), Box::new(hub));
        let e = exec();
        supplier.ensure("Qwen/Qwen3-0.6B", &e, &mut |_, _, _| {}).unwrap();

        // Registered under the fully-qualified vendor/repo id, NOT the
        // filename-derived default ("model.brain") each importer falls back
        // to when called standalone -- this is exactly what `id_override`
        // exists to fix.
        let names: Vec<String> = e.manifests().iter().map(|m| m.model.clone()).collect();
        assert_eq!(names, vec!["Qwen/Qwen3-0.6B".to_string()]);
    }

    #[test]
    fn ensure_deletes_the_upstream_safetensors_once_converted() {
        // model.safetensors is the download input to convert_transformers;
        // once model.brain.safetensors exists, Store::local never reads it
        // again (see remove_upstream_weights's doc comment) -- it must not
        // survive a successful ensure().
        let (config, weights) = tiny_qwen3_hf_files();
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "config.json", config);
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model.safetensors", weights);
        let dir = store("supply-test-deletes-upstream").root().to_path_buf();
        let supplier = StoreSupplier::new(Store::new(dir.clone()), Box::new(hub));
        let e = exec();
        supplier.ensure("Qwen/Qwen3-0.6B", &e, &mut |_, _, _| {}).unwrap();

        let repo_dir = dir.join("Qwen").join("Qwen3-0.6B");
        assert!(!repo_dir.join("model.safetensors").exists(), "upstream model.safetensors must be cleaned up after a successful convert");
        assert!(repo_dir.join("model.brain.safetensors").exists(), "the converted checkpoint must still be there");
    }

    #[test]
    fn ensure_fails_cleanly_when_the_family_has_no_import_path_yet() {
        // gpt2 is a `family_of_architecture` match (so `plan()` accepts it
        // and schedules a Convert step) but has no HF importer -- `convert`
        // dispatches to an explicit error rather than silently skipping or
        // guessing at an unwritten Conv1D-transpose import.
        let mut hub = FakeHub::new();
        hub.add_file("openai-community", "gpt2", "main", "config.json", br#"{"architectures":["GPT2LMHeadModel"]}"#.to_vec());
        hub.add_file("openai-community", "gpt2", "main", "model.safetensors", vec![0u8; 8]);
        let supplier = StoreSupplier::new(store("supply-test-gpt-unsupported"), Box::new(hub));
        let e = exec();
        let err = supplier.ensure("openai-community/gpt2", &e, &mut |_, _, _| {}).unwrap_err();
        assert!(err.contains("no HF import path yet"), "{err}");
    }

    #[test]
    fn ensure_converts_the_base_but_still_fails_cleanly_on_the_unimplemented_quantize_step() {
        // A quant ref with no upstream -GGUF sibling falls back to base +
        // local-quantize (plan.rs's `plan_quant`). The base Convert must
        // still run (and leave the base servable on disk) even though the
        // ORIGINAL request can't complete, since local quantization isn't
        // automated yet.
        let (config, weights) = tiny_qwen3_hf_files();
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "config.json", config);
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model.safetensors", weights);
        let dir = store("supply-test-quantize-not-automated").root().to_path_buf();
        let supplier = StoreSupplier::new(Store::new(dir.clone()), Box::new(hub));
        let e = exec();
        let err = supplier.ensure("Qwen/Qwen3-0.6B-Q4_K_M", &e, &mut |_, _, _| {}).unwrap_err();
        assert!(err.contains("local-quantize"), "{err}");
        assert!(err.contains("1 additional step"), "{err}");

        // The base got converted anyway -- a second `ensure` for the base ref
        // itself (no quant suffix) needs no network at all.
        let base = ModelRef::new("Qwen", "Qwen3-0.6B", None);
        assert!(Store::new(dir).local(&base).is_some(), "base must be servable even though the quant request failed");
    }

    #[test]
    fn ensure_completes_a_pure_download_plan_and_registers_a_gguf_resident() {
        // A quant ref whose upstream -GGUF sibling repo already has the
        // file resolves to a pure Download plan -- no conversion needed,
        // since a fetched GGUF carries its own tokenizer already.
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B-GGUF", "main", "Qwen3-0.6B-Q8_0.gguf", tiny_qwen3_gguf());
        let supplier = StoreSupplier::new(store("supply-test-pure-download"), Box::new(hub));
        let e = exec();
        let mut progressed = false;
        supplier.ensure("Qwen/Qwen3-0.6B-Q8_0", &e, &mut |_, _, _| progressed = true).unwrap();
        assert!(progressed);
        let names: Vec<String> = e.manifests().iter().map(|m| m.model.clone()).collect();
        assert_eq!(names, vec!["toy-qwen-gguf".to_string()]);
    }

    #[test]
    fn ensure_completes_a_diffusers_pipeline_plan_and_registers_a_zimage_resident() {
        // A Z-Image-shaped repo (no root config.json, four role subdirs) must
        // route to the zimage recipe end to end: plan -> download every role
        // file (subdirectory structure preserved) -> convert_zimage writes
        // brain.manifest.json -> resident_for_local builds a real
        // ZImageResident from the manifest's roles, no BRAIN_ZIMAGE_* env
        // vars involved anywhere in this path.
        let mut hub = FakeHub::new();
        for f in [
            "model_index.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "tokenizer/tokenizer.json",
        ] {
            hub.add_file("Tongyi-MAI", "Z-Image-Turbo", "main", f, b"stub".to_vec());
        }
        let dir = store("supply-test-zimage-compound").root().to_path_buf();
        let supplier = StoreSupplier::new(Store::new(dir.clone()), Box::new(hub));
        let e = exec();
        supplier.ensure("Tongyi-MAI/Z-Image-Turbo", &e, &mut |_, _, _| {}).unwrap();

        let names: Vec<String> = e.manifests().iter().map(|m| m.model.clone()).collect();
        assert_eq!(names, vec!["Tongyi-MAI/Z-Image-Turbo".to_string()], "must register under the fetched ref, not the compiled-in brain/s3dit constant, or the request that triggered the fetch would find nothing");

        // The manifest landed with the exact roles ZimageRecipe declares, and
        // a second ensure() for the same ref needs no network at all.
        let manifest_bytes = std::fs::read(dir.join("Tongyi-MAI").join("Z-Image-Turbo").join(brain_modelstore::MANIFEST_FILE)).unwrap();
        let manifest: brain_modelstore::CompoundManifest = serde_json::from_slice(&manifest_bytes).unwrap();
        assert_eq!(manifest.id, "Tongyi-MAI/Z-Image-Turbo");
        assert_eq!(manifest.family, "zimage");
        assert_eq!(manifest.roles.len(), 4);
        let base = ModelRef::new("Tongyi-MAI", "Z-Image-Turbo", None);
        assert!(Store::new(dir).local(&base).is_some());
    }

    /// Opt-in: exercises the full `ensure()` pipeline (plan -> download ->
    /// `convert_yolo` -> `resident_for_local`) against a REAL, unmodified
    /// `yolov8n.pt`'s bytes served through a `FakeHub` -- the decisive
    /// end-to-end proof that `YoloRecipe` + the new importer + this crate's
    /// finish dispatch compose correctly, complementing
    /// `crates/yolo/tests/import_real.rs`'s narrower importer-only check.
    /// Skips cleanly without `YOLO_RAW_PT` (see that file's module docs).
    #[test]
    fn ensure_completes_a_flat_release_plan_and_registers_a_yolo_resident() {
        let path = match std::env::var("YOLO_RAW_PT") {
            Ok(p) if std::path::Path::new(&p).is_file() => p,
            _ => {
                println!("SKIP ensure_completes_a_flat_release_plan_and_registers_a_yolo_resident: set YOLO_RAW_PT to a real yolov8n.pt");
                return;
            }
        };
        let bytes = std::fs::read(&path).unwrap();

        let mut hub = FakeHub::new();
        hub.add_file("Ultralytics", "YOLOv8", "main", "yolov8n.pt", bytes);
        let dir = store("supply-test-yolo-flat-release").root().to_path_buf();
        let supplier = StoreSupplier::new(Store::new(dir.clone()), Box::new(hub));
        let e = exec();
        supplier.ensure("Ultralytics/YOLOv8", &e, &mut |_, _, _| {}).unwrap();

        let names: Vec<String> = e.manifests().iter().map(|m| m.model.clone()).collect();
        assert_eq!(names, vec!["Ultralytics/YOLOv8".to_string()]);

        // A real, loadable model.brain.safetensors landed -- and a second
        // ensure() for the same ref needs no network at all.
        let base = ModelRef::new("Ultralytics", "YOLOv8", None);
        let local = Store::new(dir).local(&base).expect("converted checkpoint must be servable");
        let card = local.card.expect("save_safetensors wrote a card");
        assert_eq!(card.family, "yolo");
    }

    #[test]
    fn concurrent_ensure_for_the_same_model_shares_one_fetch() {
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B-GGUF", "main", "Qwen3-0.6B-Q8_0.gguf", tiny_qwen3_gguf());
        let supplier = Arc::new(StoreSupplier::new(store("supply-test-concurrent"), Box::new(hub)));
        let e = exec();

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let supplier = supplier.clone();
                let e = e.clone();
                std::thread::spawn(move || supplier.ensure("Qwen/Qwen3-0.6B-Q8_0", &e, &mut |_, _, _| {}))
            })
            .collect();
        for h in handles {
            h.join().unwrap().unwrap();
        }
        // Registered exactly once despite 8 concurrent callers.
        let names: Vec<String> = e.manifests().iter().map(|m| m.model.clone()).collect();
        assert_eq!(names, vec!["toy-qwen-gguf".to_string()]);
    }

    /// A minimal GGUF (one f32 tensor) with a `qwen` family card, mirroring
    /// `model_dir.rs`'s own `write_gguf_qwen` test fixture.
    fn tiny_qwen3_gguf() -> Vec<u8> {
        fn put_str(v: &mut Vec<u8>, s: &str) {
            v.extend((s.len() as u64).to_le_bytes());
            v.extend(s.as_bytes());
        }
        let mut h: Vec<u8> = Vec::new();
        h.extend(b"GGUF");
        h.extend(3u32.to_le_bytes());
        h.extend(1u64.to_le_bytes()); // tensor count
        h.extend(2u64.to_le_bytes()); // kv count
        put_str(&mut h, "general.architecture");
        h.extend(8u32.to_le_bytes());
        put_str(&mut h, "qwen3");
        put_str(&mut h, "general.name");
        h.extend(8u32.to_le_bytes());
        put_str(&mut h, "toy-qwen-gguf");
        // tensor info: "w", 1 dim [4], type F32, offset 0
        put_str(&mut h, "w");
        h.extend(1u32.to_le_bytes());
        h.extend(4u64.to_le_bytes());
        h.extend(0u32.to_le_bytes());
        h.extend(0u64.to_le_bytes());
        let data_start = h.len().div_ceil(32) * 32;
        h.resize(data_start, 0);
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            h.extend(v.to_le_bytes());
        }
        h
    }
}
