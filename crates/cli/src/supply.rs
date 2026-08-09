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

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};

use brain_modelref::ModelRef;
use brain_modelstore::{Hub, Step, Store};
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
        other => Err(format!("{vendor}/{repo}: convert: unknown recipe {other:?} (bug: modelstore::recipe::recipes() and this dispatch have drifted)")),
    }
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
        "qwen" => qwen::import::import_as(hf_dir, out, None, Some(&id)),
        "glm" => glm::import::import_as(hf_dir, out, Some(&id)),
        "lfm" => lfm::import::import_as(hf_dir, out, Some(&id)),
        // gpt is nanogpt-style, trained from scratch -- brain has never had an
        // HF importer for it (unlike glm/qwen/lfm, all production-tested).
        // Writing one is real new-crate work, not "wire the dispatch", so
        // this fails cleanly instead of guessing at a Conv1D-transpose import.
        "gpt" => Err("gpt has no HF import path yet -- fetch and convert manually".to_string()),
        // omni (Qwen3-Omni) is recognized (family_of_architecture checks it
        // before "qwen" specifically so it is never silently mis-routed
        // there). The importer itself streams from the sharded HF dir fine
        // (M3) -- what is NOT yet true is that the resulting unified
        // checkpoint is directly loadable by tts::mtp::MtpModel/codec::Codec
        // for the Talker/Code2Wav pieces (two open naming gaps, see
        // docs/models/omni/status.md's M7b/M8 entries); Thinker-only
        // generation (crate::resident_omni, gated on BRAIN_OMNI_HF_DIR, not
        // this converted-checkpoint path) is unaffected by either gap.
        "omni" => omni::import::import_as(hf_dir, out, Some(&id)),
        other => Err(format!("family {other:?} matched but has no dispatch arm (bug: family_of_architecture and this match have drifted)")),
    };
    result.map_err(|e| format!("{vendor}/{repo}: convert: {e}"))
}

/// Constructed by `run_cli.rs::build_auto_fetch_supplier` and threaded into
/// every HTTP/D-Bus surface (`run_apis`), behind `BRAIN_AUTO_FETCH=0` to
/// disable. Wiring it in went through the full watertight-API security pass
/// (AGENTS.md) -- see `docs/api-security-audit.md`'s auto-fetch entry.
#[derive(Clone)]
enum FetchState {
    Running,
    Done(Result<(), String>),
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
        let deferred = brain_modelstore::execute(&self.store, self.hub.as_ref(), &plan, &mut |name, got, total| {
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

        let local = self.store.local(&r).ok_or_else(|| format!("{model}: fetched but not found on disk (unexpected)"))?;
        let resident = crate::model_dir::resident_for_local(&local).ok_or_else(|| format!("{model}: family not servable"))?;
        exec.register(resident);
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
    /// shape as `crates/qwen/src/import.rs`'s own `build_tiny_hf_dir` test
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
        let names: Vec<String> = e.manifests().into_iter().map(|m| m.model).collect();
        assert_eq!(names, vec!["Qwen/Qwen3-0.6B".to_string()]);
    }

    #[test]
    fn ensure_fails_cleanly_when_the_family_has_no_import_path_yet() {
        // gpt is a `family_of_architecture` match (so `plan()` accepts it and
        // schedules a Convert step) but has no HF importer -- `convert`
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
        let names: Vec<String> = e.manifests().into_iter().map(|m| m.model).collect();
        assert_eq!(names, vec!["toy-qwen-gguf".to_string()]);
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
        let names: Vec<String> = e.manifests().into_iter().map(|m| m.model).collect();
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
