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
//!
//! The fetch/convert engine itself (`convert` and its whole per-family finish
//! dispatch, `execute_plan`/`execute_plan_opt`) and the default-checkpoint
//! auto-fetch policy (`ensure_default_weights`) now live in
//! `loader::supply` -- any embedder wants the same "fetch and convert a
//! checkpoint" primitive, not just this CLI. What stays here is everything
//! that constructs or registers a CLI-local `ResidentModel`
//! (`StoreSupplier::ensure` calling `serving::model_dir::resident_for_local`),
//! plus the `BRAIN_AUTO_FETCH` environment gate itself.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex};

use brain_modelref::ModelRef;
use brain_modelstore::{Hub, Step, Store};
use residency::{Executor, ModelSupplier, Supply};

/// Constructed by `run_cli.rs::build_auto_fetch_supplier` -- only when
/// fetching is enabled ([`auto_fetch_enabled`]) -- and threaded into
/// every HTTP/D-Bus surface (`run_apis`). Wiring it in went through the full
/// watertight-API security pass (AGENTS.md).
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
/// Pure so it's directly testable without a real download. `pub(crate)`:
/// the `--model` resolver (`model_flag`) prints the same 10% ladder for the
/// downloads it triggers -- one progress convention, not two.
pub(crate) fn next_download_pct_bucket(got: u64, total: u64, last: Option<u32>) -> Option<u32> {
    if total == 0 {
        return None;
    }
    let bucket = (got.min(total) * 100 / total / 10 * 10) as u32;
    last.is_none_or(|l| bucket > l).then_some(bucket)
}

/// Whether fetching weights on demand is enabled for this process. Default
/// OFF: a CLI run or serve request whose weights are not on disk is an error
/// naming what is missing, never a download -- the network is something the
/// caller asks for, with `--autofetch` (which `main` publishes by setting
/// `BRAIN_AUTO_FETCH=1`) or by exporting the variable directly. The old
/// opt-out spellings (`0`/`false`/`off`) keep reading as off, so an
/// environment written against the previous default keeps the behavior it
/// asked for.
pub fn auto_fetch_enabled() -> bool {
    std::env::var("BRAIN_AUTO_FETCH").ok().is_some_and(|v| {
        matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on")
    })
}

/// Walk `models_dir` for every `<vendor>/<repo>` entry not already served by
/// `exec`, and run [`ModelSupplier::ensure`] on each in one detached
/// background thread - the exact fetch/convert/register path a live request
/// already takes for an unresolved model (an interrupted download re-fetched,
/// a GGUF that needs the one-time import step converted), just run
/// proactively at startup instead of waiting for a client to hit it.
///
/// The ONLY caller (`run_cli.rs::run_apis`) gates this behind `--autofetch`:
/// this does real, unprompted network I/O and disk writes, which is exactly
/// what that flag exists to require consent for - `crate::supply` never
/// calls this on its own.
///
/// Every entry is attempted independently and its own failure is logged and
/// skipped, never aborting the walk: a model with no serving adapter written
/// yet (`Supply`'s own doc: "not fetchable... with a reason a caller can
/// surface directly") fails here exactly as it would on a live request, at
/// startup instead of on first use, and every OTHER entry still gets its
/// turn. `ensure` is documented idempotent and single-flight, so running it
/// again for an already-healthy model (nothing to fetch, already registered)
/// is safe, if slightly wasteful - the model list is not large enough for
/// that cost to matter, and it is simpler and more honest than trying to
/// pre-guess which entries are "broken" ourselves and duplicating `ensure`'s
/// own resolution logic to do it.
pub fn heal_missing_models_in_background(models_dir: std::path::PathBuf, supplier: Arc<dyn ModelSupplier>, exec: Executor) {
    let spawned = std::thread::Builder::new().name("brain-model-healer".to_string()).spawn(move || heal_all(&models_dir, supplier.as_ref(), &exec));
    if let Err(e) = spawned {
        residency::log::warn(&format!("model healer: could not spawn its background thread: {e}"));
    }
}

/// Every `<vendor>/<repo>` id under `models_dir` that is not already in
/// `already` - pulled out of [`heal_all`] so "which ids need healing" is
/// testable without a real `Executor`/[`ModelSupplier`], and returned as an
/// owned, sorted list rather than driven from inside the directory walk, so
/// the order the caller attempts them in is deterministic.
fn heal_candidates(models_dir: &Path, already: &std::collections::HashSet<String>) -> Vec<String> {
    let mut out = Vec::new();
    let Ok(vendors) = std::fs::read_dir(models_dir) else { return out };
    for vendor in vendors.flatten() {
        let vendor_path = vendor.path();
        if !vendor_path.is_dir() {
            continue;
        }
        let Some(vendor_name) = vendor_path.file_name().and_then(|n| n.to_str()) else { continue };
        let Ok(repos) = std::fs::read_dir(&vendor_path) else { continue };
        for repo in repos.flatten() {
            let repo_path = repo.path();
            if !repo_path.is_dir() {
                continue;
            }
            let Some(repo_name) = repo_path.file_name().and_then(|n| n.to_str()) else { continue };
            let id = format!("{vendor_name}/{repo_name}");
            if !already.contains(&id) {
                out.push(id);
            }
        }
    }
    out.sort();
    out
}

/// The synchronous walk [`heal_missing_models_in_background`] runs on its
/// background thread - pulled out on its own so it can be driven directly
/// (a fake [`ModelSupplier`], a temp directory, no real thread/network) rather
/// than only ever observable by joining a background thread.
fn heal_all(models_dir: &Path, supplier: &dyn ModelSupplier, exec: &Executor) {
    let already: std::collections::HashSet<String> = exec.manifests().iter().map(|m| m.model.clone()).collect();
    for id in heal_candidates(models_dir, &already) {
        residency::log::info(&format!("model healer: attempting {id}"));
        match supplier.ensure(&id, exec, &mut |_, _, _| {}) {
            Ok(()) => residency::log::info(&format!("model healer: {id} is now servable")),
            Err(e) => residency::log::warn(&format!("model healer: {id}: {e}")),
        }
    }
}

/// One single-flight gate per model: the fetch's outcome plus the condvar
/// that wakes every other concurrent `ensure` call waiting on it.
type FetchGate = Arc<(Mutex<FetchState>, Condvar)>;

pub struct StoreSupplier {
    store: Store,
    hub: Box<dyn Hub>,
    inflight: Mutex<HashMap<String, FetchGate>>,
    /// Held for the whole of one fetch: models are fetched one at a time, so
    /// requests naming many repos cannot start that many downloads at once.
    fetching: Mutex<()>,
}

impl StoreSupplier {
    pub fn new(store: Store, hub: Box<dyn Hub>) -> StoreSupplier {
        StoreSupplier { store, hub, inflight: Mutex::new(HashMap::new()), fetching: Mutex::new(()) }
    }

    fn do_ensure(&self, model: &str, exec: &Executor, progress: &mut dyn FnMut(&str, u32, u32)) -> Result<(), String> {
        let r = ModelRef::parse(model).map_err(|e| format!("{model}: {e}"))?;
        // `classify` never offers an adapter for fetching, and this is the
        // same refusal for a caller that does not go through it (the heal walk,
        // an embedder): an adapter is loaded by the operator, so reaching the
        // plan, and the hub behind it, with one is never right.
        if r.adapter().is_some() {
            return Err(format!("{model}: an adapter is loaded by the operator and is never fetched"));
        }
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

        // Convert is automated (dispatch by architecture, see
        // `loader::supply::convert`); Quantize is not yet (Phase 9 -- a
        // from-scratch GGUF quantizer). Run every Convert now so a base
        // checkpoint this plan needed is left servable on disk even when the
        // ORIGINAL request was a quant ref this call still can't finish.
        let mut still_missing = Vec::new();
        for step in &deferred {
            match step {
                Step::Convert { vendor, repo, recipe } => loader::supply::convert(&self.store, vendor, repo, recipe).map_err(|e| format!("{model}: {e}"))?,
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
        // `plan.reference`, not the ref parsed from `model`: a plan whose
        // recipe CHOSE between interchangeable artifacts records the choice
        // there (a GGUF release repo resolving to one quantization), and that
        // resolved reference is what `Store::local` finds the fetched bytes
        // under. Identical to `r` for every recipe that makes no choice.
        let local = self.store.local(&plan.reference).ok_or_else(|| format!("{model}: fetched but not found on disk (unexpected)"))?;
        // `QwenServeConfig::default()`, not the live server's real serving policy
        // flags/VRAM budget: this supplier is built before `brain serve`'s own
        // device probe runs (`run_cli::build_auto_fetch_supplier` predates
        // `build_serving_executor`), so a model first served via on-demand
        // auto-fetch gets the historical 24576 default rather than
        // auto-sized context, unlike one already on disk at startup
        // (`model_dir::discover`, which DOES receive the real config).
        let resident = serving::model_dir::resident_for_local(&local, catalog::resident_llm::QwenServeConfig::default(), catalog::resident_qwen35::Qwen35Sizing::default(), None).map_err(|e| format!("{model}: {e}"))?;
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
            // A request can name any adapter reference it likes, and an
            // adapter is something only the operator puts in front of a base
            // (`--adapter`, `--adapter-manifest`, `--watch-adapters`, or a
            // model directory): an adapter that is not already served is an
            // unknown model, and fetching anything for it would let a request
            // choose what this server downloads and runs.
            Ok(r) if r.adapter().is_some() => Supply::Unknown(format!("{model}: an adapter is not fetched on request")),
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
        let result = {
            let _one_at_a_time = self.fetching.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            self.do_ensure(model, exec, progress)
        };
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
pub(crate) mod tests {
    use super::*;
    use brain_modelstore::FakeHub;
    use residency::budget::Budgets;
    use residency::{Device, Policy};
    use brain_testutil::env_lock;

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

    // -- The auto-fetch gate: fetching is opt-in -------------------------

    /// Default OFF: unset never fetches, only a truthy `BRAIN_AUTO_FETCH`
    /// (what `--autofetch` publishes) does. The off spellings stay off so an
    /// operator who exported `=0` under the old opt-out convention keeps
    /// exactly the behavior they asked for.
    #[test]
    fn auto_fetch_is_off_unless_the_environment_opts_in() {
        let _serial = env_lock();
        std::env::remove_var("BRAIN_AUTO_FETCH");
        assert!(!auto_fetch_enabled(), "unset must be OFF: fetching is opt-in");
        for off in ["0", "false", "off", "OFF", "False", " nope "] {
            std::env::set_var("BRAIN_AUTO_FETCH", off);
            assert!(!auto_fetch_enabled(), "{off:?} must not enable fetching");
        }
        for on in ["1", "true", "TRUE", "on", " On "] {
            std::env::set_var("BRAIN_AUTO_FETCH", on);
            assert!(auto_fetch_enabled(), "{on:?} must enable fetching");
        }
        std::env::remove_var("BRAIN_AUTO_FETCH");
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

    /// A hub that fails the test on any contact: what an adapter reference
    /// must never reach.
    struct UntouchableHub;
    impl brain_modelstore::Hub for UntouchableHub {
        fn resolve_revision(&self, v: &str, r: &str, _: &str) -> Result<String, brain_modelstore::HubError> {
            panic!("the hub was asked to resolve {v}/{r}")
        }
        fn list_files(&self, v: &str, r: &str, _: &str) -> Result<Vec<String>, brain_modelstore::HubError> {
            panic!("the hub was asked to list {v}/{r}")
        }
        fn file_sizes(&self, v: &str, r: &str, _: &str) -> Result<std::collections::BTreeMap<String, u64>, brain_modelstore::HubError> {
            panic!("the hub was asked for sizes of {v}/{r}")
        }
        fn read_file(&self, v: &str, r: &str, _: &str, f: &str) -> Result<Vec<u8>, brain_modelstore::HubError> {
            panic!("the hub was asked to read {v}/{r}/{f}")
        }
        fn download(&self, v: &str, r: &str, _: &str, f: &str, _: &Path, _: &mut dyn FnMut(u64, Option<u64>)) -> Result<(), brain_modelstore::HubError> {
            panic!("the hub was asked to download {v}/{r}/{f}")
        }
    }

    /// An adapter reference is never fetched for a request: not classified
    /// fetchable (so no front end calls `ensure`), and refused by `ensure`
    /// itself before the plan, for a caller that does not classify. The hub
    /// here fails the test on any contact, and the adapter's base IS
    /// fetchable, so a regression that stripped the suffix and fetched the
    /// base would be caught too.
    #[test]
    fn an_adapter_reference_is_never_fetched() {
        let supplier = StoreSupplier::new(store("supply-test-adapter"), Box::new(UntouchableHub));
        for adapter in ["Qwen/Qwen3-0.6B:lab:canary:v1", "Qwen/Qwen3-0.6B-Q8_0:lab:canary:v1", "Qwen/Qwen3-0.6B:lab:never-trained:latest"] {
            assert!(matches!(supplier.classify(adapter), Supply::Unknown(_)), "{adapter} must not classify as fetchable");
            let mut progress = |_: &str, _: u32, _: u32| {};
            let err = supplier.ensure(adapter, &exec(), &mut progress).unwrap_err();
            assert!(err.contains("never fetched"), "{err}");
        }
        assert_eq!(supplier.classify("Qwen/Qwen3-0.6B"), Supply::Fetchable, "the base of an adapter is still an ordinary fetchable model");
        // The malformed spellings stay refused with zero I/O as well.
        for bad in ["Qwen/Qwen3-0.6B:lab:../x:v1", "Qwen/Qwen3-0.6B:lab:x", "Qwen/Qwen3-0.6B:a:b:c:d", "../x/y:lab:canary:v1"] {
            assert!(matches!(supplier.classify(bad), Supply::Unknown(_)), "{bad}");
        }
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

        // Registered under the fully-qualified vendor/repo id the manifest
        // carries, not anything derived from a file name.
        let names: Vec<String> = e.manifests().iter().map(|m| m.model.clone()).collect();
        assert_eq!(names, vec!["Qwen/Qwen3-0.6B".to_string()]);
    }

    #[test]
    fn ensure_serves_the_download_and_keeps_it() {
        // The download is what serves: ensure() names it in a manifest,
        // converts nothing on disk and removes nothing.
        let (config, weights) = tiny_qwen3_hf_files();
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "config.json", config);
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model.safetensors", weights);
        let dir = store("supply-test-keeps-upstream").root().to_path_buf();
        let supplier = StoreSupplier::new(Store::new(dir.clone()), Box::new(hub));
        let e = exec();
        supplier.ensure("Qwen/Qwen3-0.6B", &e, &mut |_, _, _| {}).unwrap();

        let repo_dir = dir.join("Qwen").join("Qwen3-0.6B");
        assert!(repo_dir.join("model.safetensors").exists(), "the downloaded model.safetensors is what serves");
        assert!(repo_dir.join(brain_modelstore::MANIFEST_FILE).exists(), "the manifest names the download");
        assert!(!repo_dir.join("model.brain.safetensors").exists(), "nothing is converted on disk");
    }

    #[test]
    fn ensure_fails_cleanly_when_the_family_has_no_import_path_yet() {
        // gpt2 is a `family_of_architecture` match (so `plan()` accepts it
        // and schedules a Convert step) but has no HF checkpoint reader -- `convert`
        // dispatches to an explicit error rather than silently skipping or
        // guessing at an unwritten Conv1D-transpose import.
        let mut hub = FakeHub::new();
        hub.add_file("openai-community", "gpt2", "main", "config.json", br#"{"architectures":["GPT2LMHeadModel"]}"#.to_vec());
        hub.add_file("openai-community", "gpt2", "main", "model.safetensors", vec![0u8; 8]);
        let supplier = StoreSupplier::new(store("supply-test-gpt-unsupported"), Box::new(hub));
        let e = exec();
        let err = supplier.ensure("openai-community/gpt2", &e, &mut |_, _, _| {}).unwrap_err();
        assert!(err.contains("no HF checkpoint reader yet"), "{err}");
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
    fn a_gguf_only_repo_fetches_one_quantization_and_its_header_is_read_back() {
        // The bug this recipe exists to fix: a repo whose whole release is
        // GGUF quantizations of one model had no recipe, fell through to the
        // transformers catch-all and failed with "no config.json in repo".
        // Fetching the listing would have been the other wrong answer -- for
        // the real repo that is over 100 GB of interchangeable copies.
        //
        // End to end: plan -> download exactly ONE file -> `convert_gguf`
        // opens what landed and reads its `general.architecture` back (the
        // first moment that is knowable: `Hub` has no range request) ->
        // `Store::local` resolves it under the quant convention, so a
        // resident is built from the fetched bytes with no env var involved.
        let mut hub = FakeHub::new();
        hub.add_file("unsloth", "Toy-Model-GGUF", "main", "README.md", b"#".to_vec());
        for q in ["Q2_K", "Q4_K_M", "Q6_K", "Q8_0"] {
            hub.add_file("unsloth", "Toy-Model-GGUF", "main", &format!("toy-model-{q}.gguf"), tiny_qwen3_gguf());
        }
        let dir = store("supply-test-gguf-only-repo").root().to_path_buf();
        let supplier = StoreSupplier::new(Store::new(dir.clone()), Box::new(hub));
        let e = exec();
        supplier.ensure("unsloth/Toy-Model-GGUF", &e, &mut |_, _, _| {}).unwrap();

        // Exactly one file, under the store's own quant name -- so this and
        // an explicit `unsloth/Toy-Model-GGUF-Q8_0` are one artifact, and the
        // three quantizations nobody asked for were never fetched.
        let repo_dir = dir.join("unsloth").join("Toy-Model-GGUF");
        let mut ggufs: Vec<String> =
            std::fs::read_dir(&repo_dir).unwrap().filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.ends_with(".gguf")).collect();
        ggufs.sort();
        assert_eq!(ggufs, ["Q8_0.gguf"], "one quantization, defaulted to the highest-fidelity one offered");

        // ... and the fetched bytes are what got registered, read through the
        // GGUF header rather than assumed from the filename.
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
        hub.add_file("Tongyi-MAI", "Z-Image-Turbo", "main", "model_index.json", br#"{"_class_name": "ZImagePipeline"}"#.to_vec());
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

    // `convert_zimage`'s "wrong pipeline" refusal is covered directly by
    // `loader::supply`'s own test now that the function lives there
    // (`convert_zimage_refuses_when_model_index_names_a_different_pipeline`);
    // nothing in this crate calls it directly any more.

    /// The actual bug this whole track exists to fix, end to end:
    /// `black-forest-labs/FLUX.2-klein-4B` is an official BFL repo, shaped
    /// byte-for-byte like `Tongyi-MAI/Z-Image-Turbo` above (same
    /// `model_index.json` + four role dirs), and must land as family
    /// `"flux2"`, never `"zimage"` -- `modelstore::recipe::select`'s
    /// `repos`-pin tiebreak is what routes it to the `flux2` `FilesRecipe`
    /// row, and the `convert` dispatch's generic `files_recipe_roles`
    /// fallback (the same one `sam2`/`kronos`/`timesfm3` already go through)
    /// is what writes the manifest -- no `flux2`-specific finish code is
    /// needed for THIS to hold.
    ///
    /// `ensure()` itself still errors: `flux2` has no `resident_for_compound`
    /// dispatch arm yet (`model_dir.rs`'s own doc note -- it is reachable only
    /// through its own `BRAIN_FLUX2_*` env vars, wiring a real resident for it
    /// is separate work). That is expected and out of scope here; what this
    /// track owns is that the CONVERT step already ran and left the right
    /// manifest on disk before that later, unrelated failure.
    #[test]
    fn ensure_writes_the_flux2_family_manifest_though_no_resident_dispatch_exists_yet() {
        let mut hub = FakeHub::new();
        for f in [
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "tokenizer/tokenizer.json",
        ] {
            hub.add_file("black-forest-labs", "FLUX.2-klein-4B", "main", f, b"stub".to_vec());
        }
        hub.add_file("black-forest-labs", "FLUX.2-klein-4B", "main", "model_index.json", br#"{"_class_name": "Flux2KleinPipeline"}"#.to_vec());
        // Pid-suffixed: a bare literal name is not safe when other worktrees
        // exercise this same track concurrently against a shared temp dir.
        let dir = store(&format!("supply-test-flux2-compound-{}", std::process::id())).root().to_path_buf();
        let supplier = StoreSupplier::new(Store::new(dir.clone()), Box::new(hub));
        let e = exec();
        let err = supplier.ensure("black-forest-labs/FLUX.2-klein-4B", &e, &mut |_, _, _| {}).unwrap_err();
        assert!(err.contains("compound family 'flux2' not servable from the model dir yet"), "{err}");

        let manifest_bytes = std::fs::read(dir.join("black-forest-labs").join("FLUX.2-klein-4B").join(brain_modelstore::MANIFEST_FILE)).unwrap();
        let manifest: brain_modelstore::CompoundManifest = serde_json::from_slice(&manifest_bytes).unwrap();
        assert_eq!(manifest.family, "flux2", "must not be misclassified as zimage");
        assert_eq!(manifest.roles.len(), 4);
    }

    /// `deepseek-ai/DeepSeek-OCR` is a passthrough family: `deepseek2ocr`
    /// reads the upstream safetensors in place, so the finish step must write
    /// a manifest and LEAVE THE WEIGHTS ALONE rather than rewriting tensors
    /// and then deleting the shards, which is what every other
    /// `transformers`-shaped repo's convert step does.
    ///
    /// The role is `dir`, not `weights`: that is the name
    /// `deepseek2ocr::spec::Deepseek2ocrSpec` resolves, and a manifest naming
    /// the other one would download correctly and then fail to resolve.
    #[test]
    fn ensure_passes_the_upstream_deepseek_ocr_checkpoint_through_under_its_own_role() {
        let mut hub = FakeHub::new();
        hub.add_file("deepseek-ai", "DeepSeek-OCR", "main", "config.json", br#"{"architectures":["DeepseekOCRForCausalLM"]}"#.to_vec());
        for f in ["tokenizer.json", "tokenizer_config.json", "model.safetensors"] {
            hub.add_file("deepseek-ai", "DeepSeek-OCR", "main", f, b"stub".to_vec());
        }
        let dir = store(&format!("supply-test-deepseekocr-hf-{}", std::process::id())).root().to_path_buf();
        let supplier = StoreSupplier::new(Store::new(dir.clone()), Box::new(hub));
        let e = exec();
        // Reaching a resident-dispatch complaint means the fetch AND the
        // finish step both completed; the checkpoint here is a stub.
        let _ = supplier.ensure("deepseek-ai/DeepSeek-OCR", &e, &mut |_, _, _| {});

        let repo = dir.join("deepseek-ai").join("DeepSeek-OCR");
        let manifest: brain_modelstore::CompoundManifest = serde_json::from_slice(&std::fs::read(repo.join(brain_modelstore::MANIFEST_FILE)).unwrap()).unwrap();
        assert_eq!(manifest.family, "deepseek2ocr");
        assert_eq!(manifest.roles.get("dir").map(String::as_str), Some("."), "the resolver's role name, not `weights`: {:?}", manifest.roles);
        assert!(repo.join("model.safetensors").exists(), "a passthrough family's upstream weights ARE what gets served, so they must survive convert");
        assert!(!repo.join("model.brain.safetensors").exists(), "nothing is rewritten on this path");
    }

    /// `deepseek-ai/DeepSeek-OCR-2` is the same passthrough family shape as v1:
    /// its upstream safetensors are served in place, so the finish step writes a
    /// manifest under the resolver's `dir` role and leaves the weights alone.
    #[test]
    fn ensure_passes_the_upstream_deepseek_ocr_2_checkpoint_through_under_its_own_role() {
        let mut hub = FakeHub::new();
        hub.add_file("deepseek-ai", "DeepSeek-OCR-2", "main", "config.json", br#"{"architectures":["DeepseekOCR2ForCausalLM"]}"#.to_vec());
        for f in ["tokenizer.json", "tokenizer_config.json", "model-00001-of-000001.safetensors"] {
            hub.add_file("deepseek-ai", "DeepSeek-OCR-2", "main", f, b"stub".to_vec());
        }
        let dir = store(&format!("supply-test-deepseekocr2-hf-{}", std::process::id())).root().to_path_buf();
        let supplier = StoreSupplier::new(Store::new(dir.clone()), Box::new(hub));
        let e = exec();
        let _ = supplier.ensure("deepseek-ai/DeepSeek-OCR-2", &e, &mut |_, _, _| {});

        let repo = dir.join("deepseek-ai").join("DeepSeek-OCR-2");
        let manifest: brain_modelstore::CompoundManifest = serde_json::from_slice(&std::fs::read(repo.join(brain_modelstore::MANIFEST_FILE)).unwrap()).unwrap();
        assert_eq!(manifest.family, "deepseekocr2");
        assert_eq!(manifest.roles.get("dir").map(String::as_str), Some("."), "the resolver's role name, not `weights`: {:?}", manifest.roles);
        assert!(repo.join("model-00001-of-000001.safetensors").exists(), "the upstream weights ARE what gets served, so they must survive convert");
    }

    /// The whole point of the `wan` recipe: `Wan-AI/Wan2.1-T2V-1.3B` has a
    /// root `config.json` declaring `"model_type": "t2v"` and no
    /// `architectures`, so before `WanRecipe` existed this plan reached
    /// `TransformersRecipe` and died with `unsupported architecture "t2v"` --
    /// which is exactly what a flagless `brain wan t2v` reported. This drives
    /// the real path end to end (plan -> download every role -> `convert_wan`
    /// writes brain.manifest.json -> `resident_for_local` builds a real
    /// WanResident from the manifest's roles) with stub bytes, so it needs no
    /// network and none of the 17.6 GB.
    #[test]
    fn ensure_completes_the_native_wan_plan_and_registers_a_wan_resident() {
        let mut hub = FakeHub::new();
        // The real listing, confirmed live via the HF API -- including the
        // README/LICENSE/assets the recipe must NOT fetch.
        for f in [
            "README.md",
            "LICENSE.txt",
            "assets/logo.png",
            "config.json",
            "diffusion_pytorch_model.safetensors",
            "Wan2.1_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "google/umt5-xxl/tokenizer.json",
            "google/umt5-xxl/tokenizer_config.json",
            "google/umt5-xxl/special_tokens_map.json",
            "google/umt5-xxl/spiece.model",
        ] {
            hub.add_file("Wan-AI", "Wan2.1-T2V-1.3B", "main", f, b"stub".to_vec());
        }
        let dir = store("supply-test-wan-compound").root().to_path_buf();
        let supplier = StoreSupplier::new(Store::new(dir.clone()), Box::new(hub));
        let e = exec();
        supplier.ensure("Wan-AI/Wan2.1-T2V-1.3B", &e, &mut |_, _, _| {}).unwrap();

        let names: Vec<String> = e.manifests().iter().map(|m| m.model.clone()).collect();
        assert_eq!(names, vec!["Wan-AI/Wan2.1-T2V-1.3B".to_string()], "must register under the fetched ref, not the compiled-in brain/wan constant");

        let repo = dir.join("Wan-AI").join("Wan2.1-T2V-1.3B");
        let manifest: brain_modelstore::CompoundManifest = serde_json::from_slice(&std::fs::read(repo.join(brain_modelstore::MANIFEST_FILE)).unwrap()).unwrap();
        assert_eq!(manifest.family, "wan");
        assert_eq!(manifest.roles["dit"], "diffusion_pytorch_model.safetensors");
        assert_eq!(manifest.roles["vae"], "Wan2.1_VAE.pth");
        assert_eq!(manifest.roles["text_encoder"], "models_t5_umt5-xxl-enc-bf16.pth");
        assert_eq!(manifest.roles["tokenizer"], "google/umt5-xxl");
        // The 3.6 MB of documentation and screenshots stayed upstream.
        assert!(!repo.join("README.md").exists());
        assert!(!repo.join("assets").exists());

        // ... and the roles resolve to real paths.
        let local = Store::new(dir).local(&ModelRef::new("Wan-AI", "Wan2.1-T2V-1.3B", None)).expect("servable");
        let roles = local.roles.expect("a compound manifest");
        for role in ["dit", "vae", "text_encoder", "tokenizer"] {
            let p = roles.get(role).unwrap_or_else(|| panic!("no role {role:?}"));
            assert!(p.exists(), "{role} -> {} does not exist", p.display());
        }
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
                brain_testutil::skip("ensure_completes_a_flat_release_plan_and_registers_a_yolo_resident: set YOLO_RAW_PT to a real yolov8n.pt");
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

    fn mkdir_all_under(root: &Path, rel: &str) {
        std::fs::create_dir_all(root.join(rel)).unwrap();
    }

    #[test]
    fn heal_candidates_skips_already_served_ids_and_sorts_the_rest() {
        let s = store("brain-supply-heal-candidates");
        mkdir_all_under(s.root(), "vendor/repoA");
        mkdir_all_under(s.root(), "vendor/repoB");
        mkdir_all_under(s.root(), "other/repoC");
        // A loose file directly under a vendor dir (not a repo directory)
        // must not be mistaken for a repo - only real subdirectories count.
        std::fs::write(s.root().join("vendor").join("loose-file.txt"), b"x").unwrap();

        let mut already = std::collections::HashSet::new();
        already.insert("vendor/repoA".to_string());
        let got = heal_candidates(s.root(), &already);
        assert_eq!(got, vec!["other/repoC".to_string(), "vendor/repoB".to_string()]);
    }

    #[test]
    fn heal_candidates_is_empty_for_a_fully_served_store() {
        let s = store("brain-supply-heal-candidates-empty");
        mkdir_all_under(s.root(), "vendor/repoA");
        let mut already = std::collections::HashSet::new();
        already.insert("vendor/repoA".to_string());
        assert!(heal_candidates(s.root(), &already).is_empty());
    }

    /// Records every id it was asked to `ensure`, in call order, and fails
    /// deterministically for any id containing "bad" - the toy stand-in for
    /// a real family with no serving adapter written yet.
    struct RecordingSupplier {
        calls: Mutex<Vec<String>>,
    }
    impl ModelSupplier for RecordingSupplier {
        fn classify(&self, _model: &str) -> Supply {
            Supply::Fetchable
        }
        fn ensure(&self, model: &str, _exec: &Executor, _progress: &mut dyn FnMut(&str, u32, u32)) -> Result<(), String> {
            self.calls.lock().unwrap().push(model.to_string());
            if model.contains("bad") {
                Err(format!("{model}: deliberately broken"))
            } else {
                Ok(())
            }
        }
    }

    /// REGRESSION target: one candidate failing (a family with no import
    /// path, exactly like `Supply`'s own doc describes) must never abort the
    /// walk - every OTHER candidate still gets its own attempt.
    #[test]
    fn heal_all_attempts_every_candidate_even_after_an_earlier_one_fails() {
        let s = store("brain-supply-heal-all");
        mkdir_all_under(s.root(), "vendor/bad-repo");
        mkdir_all_under(s.root(), "vendor/good-repo");
        let sup = RecordingSupplier { calls: Mutex::new(Vec::new()) };
        heal_all(s.root(), &sup, &exec());
        let mut calls = sup.calls.lock().unwrap().clone();
        calls.sort();
        assert_eq!(calls, vec!["vendor/bad-repo".to_string(), "vendor/good-repo".to_string()]);
    }

    /// A minimal GGUF (one f32 tensor) with a `qwen` family card, mirroring
    /// `model_dir.rs`'s own `write_gguf_qwen` test fixture.
    pub(crate) fn tiny_qwen3_gguf() -> Vec<u8> {
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
