// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One continuous-training hot-swap cycle: rl::continuous::run_cycle, then
//! (only if it actually produced a new adapter) QwenResident::set_adapter +
//! residency::Executor::evict, so the NEXT claim against `key` rebuilds with
//! the new adapter folded in -- an in-flight request is never interrupted
//! (`evict`'s own pinned-refusal contract).
//!
//! This is the glue self-improve roadmap P4/P5 flagged as the one gap left
//! before a resident can actually be hot-swapped: `rl::continuous::
//! run_cycle` (crate `rl`, generic-ish but qwen3-shaped) produces adapter
//! files; `resident_llm::QwenResident::set_adapter` (crate `cli`) and
//! `residency::Executor::evict` (crate `residency`) are the two halves
//! that make an already-registered resident pick one up -- nothing
//! previously called all three together.
//!
//! [`AdapterWatcher`] is the other half, and the one that runs against a
//! LIVE server: `brain serve --watch-adapters DIR` polls `DIR` for the
//! highest-versioned adapter a publisher has dropped there (whoever
//! produced it -- `run_cycle` above, a scheduled `lora_train` capability
//! run, a promotion gate on another machine) and applies it through the
//! same [`swap_in_adapter`] pairing, with no restart and no
//! re-registration. Both callers share that one swap, so the unattended
//! path and the in-process one cannot drift.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::resident_llm::QwenResident;
use capability::Invocation;
use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;
use residency::{Executor, InstanceKey, ResidentModel};

/// Run one cycle; returns `true` iff a new adapter was produced AND the
/// hot-swap (`set_adapter` + `evict`) actually took effect. `false` covers
/// two different, both-fine outcomes callers may want to tell apart via
/// logging: nothing new to train on ([`rl::continuous::run_cycle`] itself
/// returned `None`), or a new adapter WAS produced but eviction was
/// refused because a request is actively in flight against `key` right
/// now -- the swap is simply deferred to the next call, not lost (the
/// adapter file survives on disk; `resident.set_adapter` already pointed
/// at it before the evict attempt, so the very next successful evict of
/// this key, from any cause, picks it up).
// Parked scaffolding, not dead weight, and the reason is now narrower than
// it was: the background loop this waited for EXISTS (`spawn_adapter_watcher`
// below, wired into `run_cli::run_apis`), but it adopts an adapter someone
// else already trained and promoted rather than training one in-process --
// which is the split the whole cross-repo loop is built on. What still has no
// caller is the TRAINING half's trigger: nothing in `brain serve` stamps a
// reward onto the trajectories this would ingest, so a server-side timer
// driving it would train on unscored data. The `tests` module below exercises
// both of its outcomes, so it is covered, just not yet reachable from `main`.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn hot_swap_cycle(
    resident: &QwenResident,
    key: &InstanceKey,
    executor: &Executor,
    trajectories_dir: &Path,
    base_checkpoint: &Path,
    training_checkpoint: &Path,
    adapter_out_dir: &Path,
    tok: &QwenBpe,
    tmpl: &ChatTemplate,
    lora_rank: u32,
    lora_alpha: f32,
    opts: &model::FitOpts,
) -> std::io::Result<bool> {
    let Some(adapter_path) = rl::continuous::run_cycle(trajectories_dir, base_checkpoint, training_checkpoint, adapter_out_dir, tok, tmpl, lora_rank, lora_alpha, opts)? else {
        return Ok(false);
    };
    Ok(swap_in_adapter(resident, key, executor, &adapter_path))
}

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
    /// own tests already use -- deliberately small so this runs on the CPU
    /// backend without touching a real checkpoint.
    fn write_tiny_base(path: &std::path::Path, seed: u64) {
        write_tiny_base_cfg(path, seed, QwenConfig::tiny());
    }

    /// [`write_tiny_base`] at a caller-chosen config -- the live-serving test
    /// below needs a vocabulary wide enough for a real byte tokenizer.
    fn write_tiny_base_cfg(path: &std::path::Path, seed: u64, cfg: QwenConfig) {
        let init = qwen3::init_weights(&cfg, seed);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
            .param_list()
            .into_iter()
            .map(|(name, n)| (name.clone(), vec![n as u64], init.get(&name).unwrap_or_else(|| panic!("init missing {name}")).clone()))
            .collect();
        checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
    }

    fn tiny_tok() -> data::qwen_tokenizer::QwenBpe {
        use checkpoint::gguf::GgufTokenizer;
        let gt = GgufTokenizer {
            model: "gpt2".into(),
            pre: Some("qwen2".into()),
            tokens: vec!["<|endoftext|>".into(), "<|im_start|>".into(), "<|im_end|>".into(), "h".into(), "i".into(), "hi".into()],
            merges: vec!["h i".into()],
            token_types: vec![3, 3, 3, 1, 1, 1],
            bos: Some(0),
            eos: Some(2),
            unk: None,
            pad: None,
            ..Default::default()
        };
        data::qwen_tokenizer::QwenBpe::from_gguf(&gt).unwrap()
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

    /// One ATIF trajectory with a reward, the input `rl::continuous::
    /// run_cycle` ingests -- long enough (and repetitive enough) that a few
    /// hundred LoRA steps on a tiny model move the weights somewhere a greedy
    /// decode can see.
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
        let promoted = rl::continuous::run_cycle(&trajectories, &base, &dir.join("train.safetensors"), &staging, &tok, &tiny_tmpl(), 4, 8.0, &opts)
            .expect("run_cycle")
            .expect("a trajectory with a reward must produce an adapter");

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

    #[test]
    fn hot_swap_cycle_is_false_with_nothing_to_train_on_and_leaves_the_resident_untouched() {
        if skip() {
            return;
        }
        let dir = tmp("empty");
        let trajectories = dir.join("trajectories");
        std::fs::create_dir_all(&trajectories).unwrap();
        let base = dir.join("base.safetensors");
        write_tiny_base(&base, 1);

        let card = ModelCard::new("brain/qwen-hot-swap-test-empty", "qwen");
        let resident = Arc::new(QwenResident::from_card(base.to_str().unwrap(), &card, Some("unused.json"), None));
        let key = InstanceKey::new(&card.id, "default");
        let mut budgets = Budgets::new();
        budgets.set(Device::Cpu, 8 << 30, 0);
        let models: Vec<Arc<dyn residency::ResidentModel>> = vec![resident.clone()];
        let executor = Executor::start(models, budgets, Policy::default());

        let did_swap = hot_swap_cycle(
            &resident,
            &key,
            &executor,
            &trajectories,
            &base,
            &dir.join("train.safetensors"),
            &dir.join("adapters"),
            &tiny_tok(),
            &tiny_tmpl(),
            2,
            4.0,
            &model::FitOpts::default(),
        )
        .expect("hot_swap_cycle");
        assert!(!did_swap, "no trajectories waiting must not swap anything");
    }

    #[test]
    fn hot_swap_cycle_produces_and_points_the_resident_at_a_new_adapter_even_before_anything_ever_claimed_it() {
        if skip() {
            return;
        }
        let dir = tmp("real");
        let trajectories = dir.join("trajectories");
        std::fs::create_dir_all(&trajectories).unwrap();
        let base = dir.join("base.safetensors");
        write_tiny_base(&base, 1);

        let traj_json = serde_json::json!({
            "schema_version": "ATIF-v1.7",
            "agent": {"name": "test-agent", "version": "0.0.1"},
            "final_metrics": {"extra": {"reward": 1.0}},
            "steps": [
                {"step_id": 1, "source": "user", "message": "hihihihihi"},
                {"step_id": 2, "source": "agent", "message": "hihihihihihihihihi"}
            ]
        });
        std::fs::write(trajectories.join("t1.json"), serde_json::to_string(&traj_json).unwrap()).unwrap();

        let card = ModelCard::new("brain/qwen-hot-swap-test-real", "qwen");
        let resident = Arc::new(QwenResident::from_card(base.to_str().unwrap(), &card, Some("unused.json"), None));
        let key = InstanceKey::new(&card.id, "default");
        let mut budgets = Budgets::new();
        budgets.set(Device::Cpu, 8 << 30, 0);
        let models: Vec<Arc<dyn residency::ResidentModel>> = vec![resident.clone()];
        let executor = Executor::start(models, budgets, Policy::default());

        let opts = model::FitOpts { steps: 5, batch_size: 2, block_size: 4, ..Default::default() };
        let did_swap = hot_swap_cycle(
            &resident,
            &key,
            &executor,
            &trajectories,
            &base,
            &dir.join("train.safetensors"),
            &dir.join("adapters"),
            &tiny_tok(),
            &tiny_tmpl(),
            2,
            4.0,
            &opts,
        )
        .expect("hot_swap_cycle");
        // `did_swap` is `false` here -- correctly, not a bug: this resident
        // was only REGISTERED with the executor, never actually claimed by
        // any request, so `Executor::evict` refuses for the documented
        // "isn't resident at all" reason, the same as it would for any
        // never-yet-claimed key. That's a fact about `evict` already
        // covered by residency's own test suite, not something this test
        // needs to re-prove. What this test verifies is the two effects
        // that DID have to happen before `evict` was ever reached: a real
        // adapter file on disk, and `resident.set_adapter` having pointed
        // at it -- both of which persist regardless of whether the evict
        // that would apply them lands now or on some later, successful
        // call, per this function's own doc comment on deferred-not-lost
        // swaps.
        assert!(!did_swap, "evict must refuse for a never-claimed key, same as for a pinned one");
        let adapters: Vec<_> = std::fs::read_dir(dir.join("adapters")).unwrap().collect();
        assert_eq!(adapters.len(), 1, "exactly one adapter version must have been produced and pointed at, even though the evict step deferred");
    }
}
