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
//! `--adapter-manifest FILE` follows a release manifest instead, swapping
//! only releases whose digest verifies (`catalog::adapter_release`), and
//! `--adapter FILE` pins one verified release with no watcher at all
//! ([`AdapterMode`], [`start_adapter_mode`]).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use catalog::adapter_release::{verify_adapter, ManifestFollower, ManifestPoll};
use catalog::resident_llm::QwenResident;
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
pub fn swap_in_adapter(resident: &QwenResident, executor: &Executor, adapter: &Path) -> bool {
    // The key is read BEFORE the write, and that ordering is the contract
    // now that a resident may key on the adapter it is pointing at
    // (`QwenResident::instance_key`). What has to be evicted is the instance
    // built from the OLD value; asking afterwards would name the new one,
    // which is not resident yet, and leave the stale instance serving.
    let stale = resident.instance_key("generate", &Invocation::new());
    resident.set_adapter(adapter.to_str().map(str::to_string));
    executor.evict(stale)
}

/// Where `brain serve`'s Qwen3 gets its LoRA adapter from. One source at
/// most: a pinned release combined with anything that could replace it
/// would make "what is served" depend on timing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdapterMode {
    /// No adapter flag: the base alone.
    Base,
    /// `--adapter FILE`: exactly this release, verified at startup, never
    /// replaced.
    Pinned(PathBuf),
    /// `--watch-adapters DIR`: the highest-versioned adapter in `DIR`.
    Latest(PathBuf),
    /// `--adapter-manifest FILE`: the release a manifest names, swapped
    /// only once its digest verifies.
    Manifest(PathBuf),
}

impl AdapterMode {
    /// The mode `--adapter`, `--adapter-manifest` and `--watch-adapters`
    /// select, or why they cannot be combined.
    pub fn from_flags(adapter: Option<PathBuf>, manifest: Option<PathBuf>, watch: Option<PathBuf>) -> Result<AdapterMode, String> {
        match (adapter, manifest, watch) {
            (None, None, None) => Ok(AdapterMode::Base),
            (Some(path), None, None) => Ok(AdapterMode::Pinned(path)),
            (None, Some(path), None) => Ok(AdapterMode::Manifest(path)),
            (None, None, Some(dir)) => Ok(AdapterMode::Latest(dir)),
            _ => Err("--adapter, --adapter-manifest and --watch-adapters each choose the served adapter; give at most one".to_string()),
        }
    }
}

/// What an [`AdapterWatcher`] follows after startup.
pub enum Follow {
    /// The highest-versioned adapter in a directory.
    Latest(PathBuf),
    /// A release manifest.
    Manifest(ManifestFollower),
}

/// Put `mode` into effect on `resident` before anything is served. Returns
/// what a watcher should follow from here on (nothing for a pinned release)
/// and, when a verified release is now served, the line to print for it.
///
/// A pinned adapter that does not verify is an error, and so is a manifest
/// that exists but does not verify: at startup there is no previous release
/// to keep serving, and serving the base in its place would answer requests
/// from a model the operator did not ask for. A manifest that does not
/// exist yet serves the base until one is published.
pub fn start_adapter_mode(mode: AdapterMode, resident: Option<&Arc<QwenResident>>, executor: &Executor) -> Result<(Option<Follow>, Option<String>), String> {
    let required = |flag: &str| resident.ok_or_else(|| format!("{flag} needs a served Qwen3 to fold the adapter into; name its checkpoint with BRAIN_QWEN_WEIGHTS=<dir-or-file>"));
    match mode {
        AdapterMode::Base => Ok((None, None)),
        AdapterMode::Latest(dir) => Ok((Some(Follow::Latest(dir)), None)),
        AdapterMode::Pinned(path) => {
            let resident = required("--adapter")?;
            let release = verify_adapter(&path, &resident.served_base())?;
            swap_in_adapter(resident, executor, &release.path);
            Ok((None, Some(release.serving_line(&served_model(resident)))))
        }
        AdapterMode::Manifest(path) => {
            let resident = required("--adapter-manifest")?;
            let mut follower = ManifestFollower::new(path, resident.served_base());
            let line = match follower.poll() {
                ManifestPoll::Release(release) => {
                    swap_in_adapter(resident, executor, &release.path);
                    Some(release.serving_line(&served_model(resident)))
                }
                ManifestPoll::Rejected(why) => return Err(why),
                ManifestPoll::Missing | ManifestPoll::Unchanged => {
                    eprintln!("brain serve: no release manifest at {} yet; serving the base until one is published", follower.path().display());
                    None
                }
            };
            Ok((Some(Follow::Manifest(follower)), line))
        }
    }
}

/// The model id `resident` serves under.
fn served_model(resident: &QwenResident) -> String {
    resident.instance_key("generate", &Invocation::new()).model
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
    rejections: Arc<AtomicU64>,
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

    /// How many new release manifests this watcher refused, each leaving
    /// the previous adapter served. Always 0 when following a directory.
    pub fn rejections(&self) -> u64 {
        self.rejections.load(Ordering::SeqCst)
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
        residency::log::info(&format!("adapter watch: stopped after {} swap(s) and {} rejected release(s)", self.swaps(), self.rejections()));
    }
}

/// The warning a caller must print when `--watch-adapters` was given but
/// cannot do anything, or `None` when the configuration is coherent.
///
/// Separated from [`spawn_adapter_watcher`] so the decision is testable
/// without capturing stderr or standing up a resident. The distinction it
/// draws is the whole point: no flag is a default and is correctly silent; a
/// flag with nothing to apply an adapter to is a misconfiguration whose only
/// symptom is that promotion silently stops working.
pub(crate) fn ignored_watch_warning(dir: Option<&Path>, has_resident: bool) -> Option<String> {
    let dir = dir?;
    if has_resident {
        return None;
    }
    Some(format!(
        "brain serve: WARNING --watch-adapters {} is being IGNORED: this process serves no Qwen3 \
         resident, so there is nothing for a promoted adapter to apply to. Name the checkpoint \
         (BRAIN_QWEN_WEIGHTS=<dir-or-file>) and restart. Until then every request is answered by \
         the base weights no matter what is promoted into that directory, so a before/after \
         comparison will show no difference and look like a training failure.",
        dir.display()
    ))
}

/// Spawn the opt-in watcher, or `None` when there is nothing to watch:
/// `follow` is `None` (no following flag was given - the default, or a
/// pinned adapter) or `resident` is `None` (this process serves no Qwen3, so
/// no adapter could be applied to anything).
///
/// Takes the CONCRETE `Arc<QwenResident>` rather than the erased
/// `Arc<dyn ResidentModel>` the executor holds, because `set_adapter` is
/// inherent - see `serving::Serving`.
pub fn spawn_adapter_watcher(follow: Option<Follow>, resident: Option<Arc<QwenResident>>, executor: &Executor) -> Option<AdapterWatcher> {
    // No flag: the default, and correctly silent.
    let mut follow = follow?;
    // A directory to watch but nothing to apply an adapter to. This is a
    // misconfiguration, not a default, and it must not be silent: the flag was
    // given deliberately, every request is still answered, and the ONLY
    // observable difference is that promoted adapters never take effect. A
    // caller measuring before/after therefore sees two identical arms and
    // concludes that training achieved nothing -- the process reports success
    // the whole way through.
    let Some(resident) = resident else {
        let watched = match &follow {
            Follow::Latest(dir) => dir.as_path(),
            Follow::Manifest(follower) => follower.path(),
        };
        if let Some(warning) = ignored_watch_warning(Some(watched), false) {
            eprintln!("{warning}");
        }
        return None;
    };
    // Asked of the resident each time rather than captured once: the
    // variant string is the resident's business, and a resident that keys on
    // its current adapter answers differently after every swap. A key held
    // for the life of the process would name the instance that was resident
    // at startup and silently evict nothing from the second swap onward.
    let model = resident.instance_key("generate", &Invocation::new()).model;
    let stop = Arc::new(AtomicBool::new(false));
    let swaps = Arc::new(AtomicU64::new(0));
    let rejections = Arc::new(AtomicU64::new(0));
    let executor = executor.clone();
    let counters = WatchCounters { stop: stop.clone(), swaps: swaps.clone(), rejections: rejections.clone() };
    match &follow {
        Follow::Latest(dir) => eprintln!("brain serve: watching {} for promoted LoRA adapters ({model})", dir.display()),
        Follow::Manifest(follower) => eprintln!("brain serve: following release manifest {} ({model})", follower.path().display()),
    }
    let join = std::thread::Builder::new()
        .name("brain-adapter-watch".to_string())
        .spawn(move || watch_loop(&mut follow, &resident, &executor, &counters))
        .ok()?;
    Some(AdapterWatcher { stop, swaps, rejections, join: Some(join) })
}

/// The watcher thread's half of [`AdapterWatcher`]'s shared state.
struct WatchCounters {
    stop: Arc<AtomicBool>,
    swaps: Arc<AtomicU64>,
    rejections: Arc<AtomicU64>,
}

/// The adapter `follow` names that is not the one already `applied`, if any.
/// A manifest's own refusals are said here, once each, since the previous
/// adapter silently staying served would otherwise be their only symptom.
fn next_adapter(follow: &mut Follow, applied: Option<&Path>, model: &str, rejections: &AtomicU64) -> Option<PathBuf> {
    match follow {
        Follow::Latest(dir) => match rl::improve::latest_adapter(dir) {
            Ok(Some((_, path))) if applied != Some(path.as_path()) => Some(path),
            Ok(_) => None,
            // A directory that has not been created yet is the normal state
            // before the first publish, not a fault: say so once per tick at
            // the same level everything else here reports, and keep polling.
            Err(e) => {
                residency::log::debug(&format!("adapter watch: {}: {e}", dir.display()));
                None
            }
        },
        // The follower compares releases by digest, so a new release written
        // under the path already served is still swapped in.
        Follow::Manifest(follower) => match follower.poll() {
            ManifestPoll::Release(release) => {
                eprintln!("{}", release.serving_line(model));
                Some(release.path)
            }
            ManifestPoll::Rejected(why) => {
                rejections.fetch_add(1, Ordering::SeqCst);
                eprintln!("brain serve: keeping the adapter already served: {why}");
                None
            }
            ManifestPoll::Missing | ManifestPoll::Unchanged => None,
        },
    }
}

/// The watcher's body: adopt each new adapter `follow` names, then keep
/// retrying the eviction half for as long as an in-flight request is
/// pinning the stale instance.
///
/// `applied` tracks what was handed to `set_adapter`, so an unchanged
/// directory costs one `read_dir` per tick and nothing else (an unchanged
/// manifest, one small read). `pending` is the deferred half: `evict`
/// refuses while a request runs, and the swap is then simply owed, not lost. It is cleared once the stale instance is
/// gone - either evicted, or found not resident at all, which after the
/// `set_adapter` above means every future activation already reads the new
/// adapter and there is nothing left to drop.
fn watch_loop(follow: &mut Follow, resident: &QwenResident, executor: &Executor, counters: &WatchCounters) {
    let WatchCounters { stop, swaps, rejections } = counters;
    let model = served_model(resident);
    let mut applied: Option<PathBuf> = None;
    // The key whose instance is owed an eviction, captured at the moment the
    // swap was applied. It cannot be re-derived later: by then the resident
    // answers with the NEW adapter's key, and the instance still holding
    // memory is the old one.
    let mut pending: Option<InstanceKey> = None;
    while !stop.load(Ordering::SeqCst) {
        if let Some(path) = next_adapter(follow, applied.as_deref(), &model, rejections) {
            let stale = resident.instance_key("generate", &Invocation::new());
            pending = (!swap_in_adapter(resident, executor, &path)).then_some(stale);
            applied = Some(path);
            if pending.is_none() {
                swaps.fetch_add(1, Ordering::SeqCst);
            }
        }
        // Short-circuit order is the contract, not a style choice: ask
        // whether anything stale is resident BEFORE attempting the eviction
        // (see `is_resident`).
        if let Some(stale) = pending.clone() {
            if !is_resident(executor, &stale) || executor.evict(stale) {
                pending = None;
                swaps.fetch_add(1, Ordering::SeqCst);
            }
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
        // A ChatML template beside it: a checkpoint with no template serves
        // raw prompts only, and this vocabulary has no `<|im_start|>` special
        // to infer ChatML from.
        let chatml = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        std::fs::write(dir.join("tokenizer_config.json"), serde_json::json!({ "chat_template": chatml }).to_string()).unwrap();
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
        let base_cfg = QwenConfig::from_json_checked(&base.header["config"]).map_err(std::io::Error::other)?;
        let vocab = base_cfg.vocab;

        let dataset_dir = training_checkpoint.parent().unwrap_or_else(|| std::path::Path::new(".")).join("dataset");
        let count = rl::atif::ingest_dir(trajectories_dir, tok, tmpl, vocab as usize, &dataset_dir)?;
        assert!(count > 0, "stage_adapter: the fixture trajectory must ingest to a non-empty dataset");

        let mut cfg = base_cfg;
        cfg.lora = Some(qwen3::config::LoraCfg::attn(lora_rank, lora_alpha));
        rl::fit_weighted::<qwen3::model::Qwen>(&dataset_dir, cfg, opts, Some(training_checkpoint))?;

        let trained = checkpoint::load(training_checkpoint.to_str().expect("utf-8 path"));
        let trained_cfg = QwenConfig::from_json_checked(&trained.header["config"]).map_err(std::io::Error::other)?;
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
            spawn_adapter_watcher(Some(Follow::Latest(dir)), None, &executor).is_none(),
            "a watched directory with no served Qwen3 has nothing to swap, so it must spawn no watcher either"
        );
    }

    /// Not spawning is correct; doing it SILENTLY is not. A directory named
    /// with nothing to apply an adapter to keeps answering every request from
    /// the base weights, so the only symptom is that a promoted adapter never
    /// changes an answer -- which reads as a training failure rather than a
    /// configuration one, and costs whoever hits it a debugging session
    /// pointed at entirely the wrong half of the system.
    #[test]
    fn a_watched_directory_with_nothing_to_apply_it_to_says_so() {
        let dir = tmp("watcher-warns");

        assert!(ignored_watch_warning(None, false).is_none(), "no flag is the default, not a misconfiguration");
        assert!(ignored_watch_warning(Some(&dir), true).is_none(), "a resident to swap under means nothing is wrong");

        let warning = ignored_watch_warning(Some(&dir), false).expect("a flag that cannot work must be reported");
        assert!(warning.contains("IGNORED"), "it must say the flag is not in effect: {warning}");
        assert!(warning.contains("BRAIN_QWEN_WEIGHTS"), "and name the remedy: {warning}");
        assert!(
            warning.contains(&dir.display().to_string()),
            "and name the directory it is ignoring: {warning}"
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
        let watcher = spawn_adapter_watcher(Some(Follow::Latest(watched.clone())), Some(resident.clone()), &executor).expect("a watched directory + a served Qwen3 spawns the watcher");

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

    /// The adapter a resident currently folds in, as its instance key names it.
    fn served_adapter(resident: &QwenResident) -> String {
        resident.instance_key("generate", &Invocation::new()).config
    }

    fn cpu_executor(models: Vec<Arc<dyn residency::ResidentModel>>) -> Executor {
        let mut budgets = Budgets::new();
        budgets.set(Device::Cpu, 1 << 30, 0);
        Executor::start(models, budgets, Policy::default())
    }

    /// The flags pick exactly one source for the served adapter: a pinned
    /// release cannot be combined with anything that would replace it.
    #[test]
    fn a_pinned_adapter_excludes_every_following_mode() {
        let p = || Some(PathBuf::from("x"));
        assert!(matches!(AdapterMode::from_flags(None, None, None), Ok(AdapterMode::Base)));
        assert!(matches!(AdapterMode::from_flags(p(), None, None), Ok(AdapterMode::Pinned(_))));
        assert!(matches!(AdapterMode::from_flags(None, p(), None), Ok(AdapterMode::Manifest(_))));
        assert!(matches!(AdapterMode::from_flags(None, None, p()), Ok(AdapterMode::Latest(_))));
        for (a, m, w) in [(p(), p(), None), (p(), None, p()), (None, p(), p())] {
            assert!(AdapterMode::from_flags(a, m, w).is_err(), "two adapter sources at once must be refused");
        }
    }

    /// `--adapter`: the pinned release is served from startup, reported by
    /// digest, and no watcher is left running that could replace it. A
    /// release for another base is a startup error.
    #[test]
    fn a_pinned_adapter_is_served_and_never_followed() {
        use brain_testutil::adapters::{file_digest as digest, scratch_dir as tmp, write_adapter, write_base};
        let dir = tmp("pinned");
        let (base, other) = (dir.join("base.safetensors"), dir.join("other.safetensors"));
        write_base(&base, 1.0);
        write_base(&other, 2.0);
        let adapter = dir.join("adapter.safetensors");
        write_adapter(&adapter, "pinned", "local/base", Some(digest(&base)), 0.5);

        let card = ModelCard::new("brain/qwen3", "qwen");
        let resident = Arc::new(QwenResident::from_card(base.to_str().unwrap(), &card, None, None));
        let executor = cpu_executor(vec![resident.clone()]);
        let (follow, line) = start_adapter_mode(AdapterMode::Pinned(adapter.clone()), Some(&resident), &executor).expect("the adapter's own base serves it");
        assert!(follow.is_none(), "a pinned adapter must leave nothing to follow");
        assert_eq!(line.as_deref(), Some(format!("brain serve: brain/qwen3 adapter=pinned digest={}", digest(&adapter)).as_str()));
        assert_eq!(served_adapter(&resident), adapter.to_str().unwrap());

        let wrong = Arc::new(QwenResident::from_card(other.to_str().unwrap(), &card, None, None));
        let Err(err) = start_adapter_mode(AdapterMode::Pinned(adapter.clone()), Some(&wrong), &cpu_executor(vec![wrong.clone()])) else {
            panic!("an adapter trained on another base must not start");
        };
        assert!(err.contains(&digest(&base)) && err.contains(&digest(&other)), "{err}");
        assert_eq!(served_adapter(&wrong), "base", "a refused adapter is never set");
        assert!(start_adapter_mode(AdapterMode::Pinned(adapter), None, &executor).is_err(), "an adapter with no Qwen3 to fold it into is an error");
    }

    /// `--adapter-manifest`: a replaced manifest swaps the served adapter
    /// only after its digest verifies; a corrupt one, or one whose digest
    /// does not match its file, keeps the previous adapter served.
    #[test]
    fn a_followed_manifest_swaps_only_verified_releases() {
        use brain_testutil::adapters::{file_digest as digest, scratch_dir as tmp, write_adapter, write_base};
        let dir = tmp("manifest-watch");
        let base = dir.join("base.safetensors");
        write_base(&base, 1.0);
        let (one, two) = (dir.join("adapter-1.safetensors"), dir.join("adapter-2.safetensors"));
        write_adapter(&one, "one", "local/base", Some(digest(&base)), 0.5);
        write_adapter(&two, "two", "local/base", Some(digest(&base)), 0.25);
        let manifest = dir.join("release.json");
        // Replaced the way a release step replaces it: written aside, renamed over.
        let publish = |body: String| {
            let staged = dir.join("release.json.tmp");
            std::fs::write(&staged, body).unwrap();
            std::fs::rename(&staged, &manifest).unwrap();
        };
        let release = |path: &std::path::Path, digest: String| serde_json::json!({"adapter": path, "digest": digest}).to_string();
        publish(release(&one, digest(&one)));

        let card = ModelCard::new("brain/qwen3", "qwen");
        let resident = Arc::new(QwenResident::from_card(base.to_str().unwrap(), &card, None, None));
        let executor = cpu_executor(vec![resident.clone()]);
        let (follow, line) = start_adapter_mode(AdapterMode::Manifest(manifest.clone()), Some(&resident), &executor).expect("a valid manifest serves at startup");
        assert_eq!(line.as_deref(), Some(format!("brain serve: brain/qwen3 adapter=one digest={}", digest(&one)).as_str()));
        assert_eq!(served_adapter(&resident), one.to_str().unwrap());
        let watcher = spawn_adapter_watcher(follow, Some(resident.clone()), &executor).expect("a manifest to follow spawns the watcher");

        let wait = |what: &str, done: &dyn Fn() -> bool| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
            while !done() {
                assert!(std::time::Instant::now() < deadline, "timed out waiting for {what}");
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        };
        publish("{\"adapter\": ".to_string());
        wait("the corrupt manifest to be rejected", &|| watcher.rejections() == 1);
        assert_eq!(served_adapter(&resident), one.to_str().unwrap(), "a corrupt manifest keeps the previous adapter");

        publish(release(&two, digest(&one)));
        wait("the mismatched digest to be rejected", &|| watcher.rejections() == 2);
        assert_eq!(served_adapter(&resident), one.to_str().unwrap(), "a release whose digest does not verify is never swapped in");

        publish(release(&two, digest(&two)));
        wait("the verified release to be swapped in", &|| watcher.swaps() == 1);
        assert_eq!(served_adapter(&resident), two.to_str().unwrap());
    }
}
