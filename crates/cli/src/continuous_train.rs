// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The hot-swap half of the self-improve loop: point an already-registered
//! `QwenResident` at a new LoRA adapter file and evict its stale instance so
//! the NEXT claim against that resident's key rebuilds with the adapter
//! folded in -- an in-flight request is never interrupted (`evict`'s own
//! pinned-refusal contract).
//!
//! [`swap_in_adapter`] is the pairing itself: `resident_llm::QwenResident::
//! set_adapter` (crate `cli`) + `residency::Executor::evict` (crate
//! `residency`). [`AdapterWatcher`] drives it unattended against a LIVE
//! server: `brain serve --watch-adapters DIR` polls `DIR` for the
//! highest-versioned adapter a publisher has dropped there -- a gated
//! promotion (`rl::improve::cycle`) run elsewhere, a scheduled `lora_train`
//! capability run, a promotion on another machine -- and applies it through
//! [`swap_in_adapter`], with no restart and no re-registration.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::resident_llm::QwenResident;
use capability::Invocation;
use residency::{Executor, InstanceKey, ResidentModel};

/// Point `resident` at `adapter` and drop the stale instance so the NEXT
/// claim against `key` rebuilds with it folded in. Returns whether an
/// instance was actually evicted.
///
/// The order is load-bearing and is why this is one function rather than two
/// calls at each call site: `set_adapter` first, so that any activation from
/// this instant on already reads the new path, and only then `evict`, whose
/// job is limited to getting rid of an instance built BEFORE that write. A
/// `false` return therefore never means the swap was lost - either nothing
/// was resident (the next activation reads the new adapter anyway) or a
/// request is in flight and `evict` refused rather than tearing down a
/// running lane, which is exactly the pinned-safety contract that makes a
/// mid-request swap harmless.
pub fn swap_in_adapter(resident: &QwenResident, key: &InstanceKey, executor: &Executor, adapter: &Path) -> bool {
    resident.set_adapter(adapter.to_str().map(str::to_string));
    executor.evict(key.clone())
}

/// How often [`AdapterWatcher`] looks at its directory. Small enough that a
/// promoted adapter reaches the serving path in well under a second, large
/// enough that an idle server's watcher costs one `read_dir` of a directory
/// holding a handful of files per tick.
const POLL: std::time::Duration = std::time::Duration::from_millis(200);

/// A background thread watching one directory for a newer adapter and
/// applying it to a live `QwenResident` through [`swap_in_adapter`].
///
/// Opt-in (`brain serve --watch-adapters DIR`), never on by default: a
/// serving process that silently reloads its weights because a file appeared
/// on disk is not something an operator should get without asking.
///
/// Stops and joins on drop, so the thread's lifetime is exactly the serving
/// surfaces' lifetime rather than "until the process exits" - a detached
/// thread would keep polling a directory during shutdown drain.
pub struct AdapterWatcher {
    stop: Arc<AtomicBool>,
    swaps: Arc<AtomicU64>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl AdapterWatcher {
    /// How many adapters this watcher has fully applied - the resident
    /// pointed at the new file AND no instance built before that write left
    /// resident. Observable so a caller (and the test) can wait for a swap
    /// to land instead of racing the poll interval.
    pub fn swaps(&self) -> u64 {
        self.swaps.load(Ordering::SeqCst)
    }
}

impl Drop for AdapterWatcher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
        // What this server actually did unattended, said once at shutdown:
        // a hot swap leaves no other trace an operator can go back and read.
        residency::log::info(&format!("adapter watch: stopped after {} swap(s)", self.swaps()));
    }
}

/// Spawn the opt-in watcher, or `None` when there is nothing to watch:
/// `dir` is `None` (the flag was not given - the default) or `resident` is
/// `None` (this process serves no Qwen3, so no adapter could be applied to
/// anything).
///
/// Takes the CONCRETE `Arc<QwenResident>` rather than the erased
/// `Arc<dyn ResidentModel>` the executor holds, because `set_adapter` is
/// inherent - see `crate::resident::Serving`.
pub fn spawn_adapter_watcher(dir: Option<&Path>, resident: Option<Arc<QwenResident>>, executor: &Executor) -> Option<AdapterWatcher> {
    let dir = dir?.to_path_buf();
    let resident = resident?;
    // The resident's own answer to "which instance serves me", not a locally
    // rebuilt `InstanceKey`: the variant string is the resident's business
    // and a second spelling of it here would silently evict nothing the day
    // it changed.
    let key = resident.instance_key("generate", &Invocation::new());
    let stop = Arc::new(AtomicBool::new(false));
    let swaps = Arc::new(AtomicU64::new(0));
    let executor = executor.clone();
    let (t_stop, t_swaps) = (stop.clone(), swaps.clone());
    eprintln!("brain serve: watching {} for promoted LoRA adapters ({})", dir.display(), key.model);
    let join = std::thread::Builder::new()
        .name("brain-adapter-watch".to_string())
        .spawn(move || watch_loop(&dir, &resident, &key, &executor, &t_stop, &t_swaps))
        .ok()?;
    Some(AdapterWatcher { stop, swaps, join: Some(join) })
}

/// The watcher's body: adopt the newest adapter that is not the one already
/// applied, then keep retrying the eviction half for as long as an in-flight
/// request is pinning the stale instance.
///
/// `applied` tracks what was handed to `set_adapter`, so an unchanged
/// directory costs one `read_dir` per tick and nothing else. `pending` is
/// the deferred half: `evict` refuses while a request runs, and the swap is
/// then simply owed, not lost. It is cleared once the stale instance is
/// gone - either evicted, or found not resident at all, which after the
/// `set_adapter` above means every future activation already reads the new
/// adapter and there is nothing left to drop.
fn watch_loop(dir: &Path, resident: &QwenResident, key: &InstanceKey, executor: &Executor, stop: &AtomicBool, swaps: &AtomicU64) {
    let mut applied: Option<PathBuf> = None;
    let mut pending = false;
    while !stop.load(Ordering::SeqCst) {
        match rl::improve::latest_adapter(dir) {
            Ok(Some((_, path))) if applied.as_deref() != Some(path.as_path()) => {
                pending = !swap_in_adapter(resident, key, executor, &path);
                applied = Some(path);
                if !pending {
                    swaps.fetch_add(1, Ordering::SeqCst);
                }
            }
            Ok(_) => {}
            // A directory that has not been created yet is the normal state
            // before the first publish, not a fault: say so once per tick at
            // the same level everything else here reports, and keep polling.
            Err(e) => residency::log::debug(&format!("adapter watch: {}: {e}", dir.display())),
        }
        // Short-circuit order is the contract, not a style choice: ask
        // whether anything stale is resident BEFORE attempting the eviction
        // (see `is_resident`).
        if pending && (!is_resident(executor, key) || executor.evict(key.clone())) {
            pending = false;
            swaps.fetch_add(1, Ordering::SeqCst);
        }
        std::thread::sleep(POLL);
    }
}

/// Whether an instance for `key` is currently resident. Checked BEFORE
/// retrying a refused eviction, never after: an `evict` that fails on a key
/// nothing has claimed is indistinguishable from one refused by a pinned,
/// in-flight request, and retrying the second case forever while treating
/// the first as still-owed would evict a perfectly current instance the
/// moment one is finally built.
fn is_resident(executor: &Executor, key: &InstanceKey) -> bool {
    executor.residency().placements.iter().any(|p| &p.key == key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use checkpoint::st::ModelCard;
    use data::chat_template::ChatTemplate;
    use qwen3::config::QwenConfig;
    use residency::{budget::Budgets, Device, Policy};
    use std::sync::Arc;

    fn skip() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-cli-hot-swap-cycle-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Same tiny (vocab 23, a handful of layers) shape `resident_llm.rs`'s
    /// own tests already use, at a caller-chosen vocab -- the live-serving
    /// test below needs one wide enough for a real byte tokenizer.
    fn write_tiny_base_cfg(path: &std::path::Path, seed: u64, cfg: QwenConfig) {
        let init = qwen3::init_weights(&cfg, seed);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
            .param_list()
            .into_iter()
            .map(|(name, n)| (name.clone(), vec![n as u64], init.get(&name).unwrap_or_else(|| panic!("init missing {name}")).clone()))
            .collect();
        checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
    }

    fn tiny_tmpl() -> ChatTemplate {
        ChatTemplate::compile("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}").unwrap()
    }

    /// A `tokenizer.json` with one token per printable ASCII byte (plus space
    /// and newline) and no merges, written into `dir` and returned. `QwenBpe`
    /// maps each byte through GPT-2's `bytes_to_unicode`, which is the
    /// IDENTITY on `0x21..=0x7E`, so every byte is its own token - enough to
    /// round-trip both this test's fixture text and the chat template's own
    /// `<|im_start|>`/`<|im_end|>` markers, which is what lets the live-serving
    /// test below drive a real chat request with no real checkpoint and no
    /// `QWEN_TOKENIZER` on the box.
    fn write_byte_tokenizer(dir: &std::path::Path) -> std::path::PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let mut vocab = serde_json::Map::new();
        for (i, b) in (0x21u8..=0x7e).enumerate() {
            vocab.insert((b as char).to_string(), serde_json::json!(i));
        }
        let n = vocab.len();
        vocab.insert('\u{0120}'.to_string(), serde_json::json!(n)); // space
        vocab.insert('\u{010a}'.to_string(), serde_json::json!(n + 1)); // newline
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, serde_json::json!({"model": {"vocab": vocab, "merges": []}}).to_string()).unwrap();
        path
    }

    /// One ATIF trajectory with a reward, the input [`stage_adapter`]
    /// ingests -- long enough (and repetitive enough) that a few hundred
    /// LoRA steps on a tiny model move the weights somewhere a greedy decode
    /// can see.
    fn write_trajectory(dir: &std::path::Path) {
        std::fs::create_dir_all(dir).unwrap();
        let traj = serde_json::json!({
            "schema_version": "ATIF-v1.7",
            "agent": {"name": "test-agent", "version": "0.0.1"},
            "final_metrics": {"extra": {"reward": 1.0}},
            "steps": [
                {"step_id": 1, "source": "user", "message": "what colour is the sky"},
                {"step_id": 2, "source": "agent", "message": "the sky is green green green green green green"}
            ]
        });
        std::fs::write(dir.join("t1.json"), serde_json::to_string(&traj).unwrap()).unwrap();
    }

    /// Train and save a real, versioned LoRA adapter file straight to
    /// `adapter_out_dir`, for the live-serve test below to drop into a
    /// watched directory. Deliberately NOT `rl::improve::cycle`: it does
    /// its own gating, and what this fixture stands in for is "some other
    /// process already gated this and published the file" -- the property
    /// under test here is [`AdapterWatcher`]'s live-swap behavior, not
    /// promotion.
    fn stage_adapter(
        trajectories_dir: &std::path::Path,
        base_checkpoint: &std::path::Path,
        training_checkpoint: &std::path::Path,
        adapter_out_dir: &std::path::Path,
        tok: &data::qwen_tokenizer::QwenBpe,
        tmpl: &ChatTemplate,
        lora_rank: u32,
        lora_alpha: f32,
        opts: &model::FitOpts,
    ) -> std::io::Result<PathBuf> {
        let base = checkpoint::load(base_checkpoint.to_str().expect("utf-8 path"));
        let vocab = QwenConfig::from_json(&base.header["config"]).vocab;

        let dataset_dir = training_checkpoint.parent().unwrap_or_else(|| std::path::Path::new(".")).join("dataset");
        let count = rl::atif::ingest_dir(trajectories_dir, tok, tmpl, vocab as usize, &dataset_dir)?;
        assert!(count > 0, "stage_adapter: the fixture trajectory must ingest to a non-empty dataset");

        let mut cfg = QwenConfig::from_json(&base.header["config"]);
        cfg.lora = Some(qwen3::config::LoraCfg::attn(lora_rank, lora_alpha));
        rl::fit_weighted::<qwen3::model::Qwen>(&dataset_dir, cfg, opts, Some(training_checkpoint))?;

        let trained = checkpoint::load(training_checkpoint.to_str().expect("utf-8 path"));
        let trained_cfg = QwenConfig::from_json(&trained.header["config"]);
        let init = trained.by_role("");
        let block = trained_cfg.block_size;
        let model = qwen3::model::Qwen::new(trained_cfg, 1, block, &init);

        std::fs::create_dir_all(adapter_out_dir)?;
        let version = rl::improve::latest_adapter(adapter_out_dir)?.map(|(v, _)| v + 1).unwrap_or(0);
        let adapter_path = adapter_out_dir.join(format!("adapter-{version:06}.safetensors"));
        qwen3::lora::save_adapter(adapter_path.to_str().expect("utf-8 path"), &model, &format!("adapter-{version:06}"), "base", None)?;
        Ok(adapter_path)
    }

    /// The watcher is OPT-IN: `brain serve` without `--watch-adapters` must
    /// spawn no background thread at all, and neither must a run that names a
    /// directory but serves no Qwen3 (nothing to point at a new adapter).
    #[test]
    fn no_watcher_is_spawned_unless_the_flag_names_an_adapter_directory() {
        let mut budgets = Budgets::new();
        budgets.set(Device::Cpu, 1 << 30, 0);
        let executor = Executor::start(Vec::new(), budgets, Policy::default());
        let dir = tmp("watcher-off");

        assert!(spawn_adapter_watcher(None, None, &executor).is_none(), "no --watch-adapters must spawn no watcher");
        assert!(
            spawn_adapter_watcher(Some(&dir), None, &executor).is_none(),
            "a watched directory with no served Qwen3 has nothing to swap, so it must spawn no watcher either"
        );
    }

    /// The gap this milestone closes: the hot swap has been implemented,
    /// pinned-safe and unit-tested for a while, but no cycle had ever run
    /// unattended against a LIVE serving process. So: serve the tiny fixture
    /// over the real OpenAI router, hold a request in flight, drop a real
    /// trained adapter into the watched directory while that request is
    /// running, and check both halves of the contract - the in-flight request
    /// completes uncorrupted, and the NEXT request answers from the new
    /// weights, with no restart and no re-registration.
    ///
    /// The fixture is a randomly initialised tiny model, so neither answer
    /// means anything as TEXT - the checked property is that the served
    /// weights changed, which is what a hot swap is. Mutation-verified:
    /// dropping the `set_adapter` half of `swap_in_adapter` makes the two
    /// answers identical and this test fail.
    #[test]
    fn a_promoted_adapter_changes_a_live_serve_response_without_restart() {
        if skip() {
            return;
        }
        let dir = tmp("live-swap");
        let tok_path = write_byte_tokenizer(&dir.join("tok"));
        // Vocab 128 covers the 96-token byte tokenizer above with room to
        // spare; everything else is the tiny CPU shape.
        let cfg = QwenConfig { vocab: 128, ..QwenConfig::tiny() };
        let base = dir.join("base.safetensors");
        write_tiny_base_cfg(&base, 7, cfg);

        // A REAL promoted adapter, trained OFFLINE into a staging directory --
        // so the "drop" below is a single file appearing in the watched
        // directory (what a publish step does), not a training run the test
        // would be timing.
        let tok = data::qwen_tokenizer::QwenBpe::from_file(tok_path.to_str().unwrap()).expect("byte tokenizer");
        let trajectories = dir.join("trajectories");
        write_trajectory(&trajectories);
        let staging = dir.join("staging");
        let opts = model::FitOpts { steps: 60, batch_size: 4, block_size: 8, lr: 3e-2, warmup: 0, decay_iters: 60, ..Default::default() };
        let promoted =
            stage_adapter(&trajectories, &base, &dir.join("train.safetensors"), &staging, &tok, &tiny_tmpl(), 4, 8.0, &opts).expect("stage_adapter");

        // The live server: one resident, one executor, the real OpenAI router.
        let card = ModelCard::new("brain/qwen3", "qwen");
        let resident = Arc::new(QwenResident::from_card(base.to_str().unwrap(), &card, Some(tok_path.to_str().unwrap()), None));
        let mut budgets = Budgets::new();
        budgets.set(Device::Cpu, 8 << 30, 0);
        let models: Vec<Arc<dyn residency::ResidentModel>> = vec![resident.clone()];
        let executor = Executor::start(models, budgets, Policy::default());

        let watched = dir.join("adapters");
        std::fs::create_dir_all(&watched).unwrap();
        let watcher = spawn_adapter_watcher(Some(&watched), Some(resident.clone()), &executor).expect("a watched directory + a served Qwen3 spawns the watcher");

        let api_key = "sk-brain-test-key".to_string();
        let state = apiserve::AppState::new(executor.clone(), api_key.clone(), apiserve::Provider::OpenAI);
        let app = apiserve::router(state);
        let post = move |user: &str, max_tokens: u32| {
            let body = serde_json::json!({
                "model": "brain/qwen3",
                "messages": [{"role": "user", "content": user}],
                "max_tokens": max_tokens,
                "temperature": 0,
            });
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri("/v1/chat/completions")
                .header(axum::http::header::AUTHORIZATION, format!("Bearer {api_key}"))
                .header(axum::http::header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap()
        };
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build().unwrap();
        let content = |app: axum::Router, req: axum::http::Request<axum::body::Body>| -> (axum::http::StatusCode, String) {
            use tower::ServiceExt;
            rt.block_on(async move {
                let r = app.oneshot(req).await.unwrap();
                let status = r.status();
                let bytes = axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
                let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
                let text = v["choices"][0]["message"]["content"].as_str().unwrap_or_default().to_string();
                (status, text)
            })
        };

        let (status, before) = content(app.clone(), post("what colour is the sky", 16));
        assert_eq!(status, axum::http::StatusCode::OK, "the base model must serve before anything is swapped");
        assert!(!before.is_empty(), "the baseline answer must be non-empty, or 'the answer changed' means nothing");

        // Hold a request in flight, and only drop the adapter once the
        // executor really is running it -- otherwise "mid-flight" would be an
        // assumption, not a fact this test established.
        let (in_app, in_req) = (app.clone(), post("what colour is the sky", 192));
        let in_flight = std::thread::spawn(move || {
            use tower::ServiceExt;
            let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            rt.block_on(async move {
                let r = in_app.oneshot(in_req).await.unwrap();
                let status = r.status();
                let bytes = axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
                (status, bytes)
            })
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while executor.in_flight().is_empty() {
            assert!(std::time::Instant::now() < deadline, "the second request never reached the executor");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        std::fs::copy(&promoted, watched.join(promoted.file_name().unwrap())).expect("drop the promoted adapter into the watched dir");

        let (status, bytes) = in_flight.join().expect("the in-flight request thread must not panic");
        assert_eq!(status, axum::http::StatusCode::OK, "a swap arriving mid-request must never break that request");
        let v: serde_json::Value = serde_json::from_slice(&bytes).expect("the in-flight response must still be well-formed JSON");
        assert!(
            !v["choices"][0]["message"]["content"].as_str().unwrap_or_default().is_empty(),
            "the in-flight request must complete with real content, not a truncated or empty body"
        );

        // The swap itself is deferred while that request pins the instance, so
        // wait for the watcher to report it applied rather than racing it.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while watcher.swaps() == 0 {
            assert!(std::time::Instant::now() < deadline, "the watcher never applied the adapter it was pointed at");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }

        let (status, after) = content(app.clone(), post("what colour is the sky", 16));
        assert_eq!(status, axum::http::StatusCode::OK, "the swapped-in adapter must still serve");
        assert_ne!(after, before, "the next request after a promoted adapter landed must answer from the new weights, with no restart");
    }
}
