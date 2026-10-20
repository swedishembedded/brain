// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The adapters `brain serve` offers beside its Qwen3 base, and the unattended
//! loop that keeps them current: `--adapter FILE` offers one verified release,
//! `--adapter-manifest FILE` follows a release manifest, and `--watch-adapters
//! DIR` follows a directory a training or promotion step publishes into.
//!
//! Every adapter is a model of its own, named by its card id
//! (`<vendor>/<repo>:<owner>:<name>:<tag>`), served on the one resident copy of
//! the base weights ([`catalog::adapter_registry`]). The base keeps answering
//! under its own id. A new release has a new card id (a new tag) and digest:
//! it is offered beside the old one, and only then is the old one withdrawn, so
//! a client that selects the new id never sees a gap, and a request already
//! running under the old one finishes on it. A release that keeps its card id
//! under a new digest replaces the adapter under that id the same way.
//!
//! [`LiveAdapters`](catalog::adapter_registry::LiveAdapters) is the one place
//! an adapter enters or leaves a running executor; [`AdapterWatcher`] drives it
//! unattended against a LIVE server and [`start_adapter_mode`] does the same
//! once, before anything binds ([`AdapterMode`]).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use catalog::adapter_registry::{AdapterHost, LiveAdapters};
use catalog::adapter_release::{load_adapter, verify_adapter, LoadedAdapter, ManifestFollower, ManifestPoll};
use catalog::resident_llm::QwenResident;
use catalog::resident_qwen35::Qwen35GgufServed;
use residency::{Executor, ResidentModel};

/// Where `brain serve`'s Qwen3 gets its LoRA adapter from. One source at
/// most. Each source names its adapters by their own card ids, so combining
/// them would not make "what is served" depend on timing any more, but it would
/// make "which id is the current one" depend on which source moved last; the
/// flags stay exclusive until a deployment needs two sources.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdapterMode {
    /// No adapter flag: the base alone.
    Base,
    /// `--adapter FILE`: exactly this release, verified at startup, never
    /// replaced.
    Pinned(PathBuf),
    /// `--watch-adapters DIR`: the highest-versioned adapter in `DIR`.
    Latest(PathBuf),
    /// `--adapter-manifest FILE`: the release a manifest names, offered
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
pub struct Follow {
    source: FollowSource,
    /// The card id this source has offered so far, which the next release
    /// withdraws once it is offered itself.
    served: Option<String>,
}

enum FollowSource {
    /// The highest-versioned adapter in a directory.
    Latest(PathBuf),
    /// A release manifest.
    Manifest(ManifestFollower),
}

impl Follow {
    /// Follow the highest-versioned adapter in `dir`.
    pub fn latest(dir: PathBuf) -> Follow {
        Follow { source: FollowSource::Latest(dir), served: None }
    }

    fn describe(&self) -> PathBuf {
        match &self.source {
            FollowSource::Latest(dir) => dir.clone(),
            FollowSource::Manifest(follower) => follower.path().to_path_buf(),
        }
    }
}

/// Put `mode` into effect on `resident` before anything is served. Returns
/// what a watcher should follow from here on (nothing for a pinned release)
/// and, when a verified release is now served, the line to print for it.
///
/// A pinned adapter that does not verify is an error, and so is a manifest
/// that exists but does not verify: at startup there is no previous release
/// to keep serving, and serving the base in its place would answer requests
/// for a model the operator did not ask for. A manifest that does not
/// exist yet offers the base alone until one is published.
pub fn start_adapter_mode<R: AdapterHost>(mode: AdapterMode, resident: Option<&Arc<R>>, executor: &Executor) -> Result<(Option<Follow>, Option<String>), String> {
    let required = |flag: &str| resident.ok_or_else(|| format!("{flag} needs a served Qwen3 or Qwen3.8 to put the adapter on; name a Qwen3 checkpoint with --checkpoint <dir-or-file>, or serve one released Qwen3.8 GGUF from the model directory"));
    match mode {
        AdapterMode::Base => Ok((None, None)),
        AdapterMode::Latest(dir) => Ok((Some(Follow::latest(dir)), None)),
        AdapterMode::Pinned(path) => {
            let resident = required("--adapter")?;
            let loaded = verify_adapter(&path, &resident.served_base())?;
            let line = loaded.release.serving_line(&loaded.release.id);
            LiveAdapters::new(resident.clone(), executor.clone()).serve(loaded)?;
            Ok((None, Some(line)))
        }
        AdapterMode::Manifest(path) => {
            let resident = required("--adapter-manifest")?;
            let mut follower = ManifestFollower::new(path, resident.served_base());
            let (served, line) = match follower.poll() {
                ManifestPoll::Release(loaded) => {
                    let line = loaded.release.serving_line(&loaded.release.id);
                    let id = loaded.release.id.clone();
                    LiveAdapters::new(resident.clone(), executor.clone()).serve(*loaded)?;
                    (Some(id), Some(line))
                }
                ManifestPoll::Rejected(why) => return Err(why),
                ManifestPoll::Missing | ManifestPoll::Unchanged => {
                    eprintln!("brain serve: no release manifest at {} yet; serving the base alone until one is published", follower.path().display());
                    (None, None)
                }
            };
            Ok((Some(Follow { source: FollowSource::Manifest(follower), served }), line))
        }
    }
}

/// Where `--adapter`, `--adapter-manifest` and `--watch-adapters` put their
/// adapter: the env-selected Qwen3, else the one released Qwen3.8 GGUF the
/// model scan served. Which of several is ambiguous and is refused by name.
pub enum Host {
    /// Nothing the adapters could ride on.
    None,
    Qwen(Arc<QwenResident>),
    Qwen35(Arc<Qwen35GgufServed>),
    /// Several released Qwen3.8 GGUFs are served and no Qwen3: the ids.
    Ambiguous(Vec<String>),
}

impl Host {
    /// The host for a server that serves `qwen` and `qwen35`.
    pub fn pick(qwen: Option<Arc<QwenResident>>, qwen35: &[Arc<Qwen35GgufServed>]) -> Host {
        match (qwen, qwen35) {
            (Some(q), _) => Host::Qwen(q),
            (None, []) => Host::None,
            (None, [one]) => Host::Qwen35(one.clone()),
            (None, many) => Host::Ambiguous(many.iter().map(|m| m.manifest().model).collect()),
        }
    }

    /// [`start_adapter_mode`] on this host.
    pub fn start(&self, mode: AdapterMode, executor: &Executor) -> Result<(Option<Follow>, Option<String>), String> {
        match self {
            Host::Qwen(q) => start_adapter_mode(mode, Some(q), executor),
            Host::Qwen35(g) => start_adapter_mode(mode, Some(g), executor),
            Host::Ambiguous(ids) if mode != AdapterMode::Base => Err(format!("{} models could take the adapter ({}); serve one Qwen3.8 GGUF, or put the adapter in the model directory under the id of the one it rides on", ids.len(), ids.join(", "))),
            Host::None | Host::Ambiguous(_) => start_adapter_mode::<QwenResident>(mode, None, executor),
        }
    }

    /// [`spawn_adapter_watcher`] on this host.
    pub fn watch(self, follow: Option<Follow>, executor: &Executor) -> Option<AdapterWatcher> {
        match self {
            Host::Qwen(q) => spawn_adapter_watcher(follow, Some(q), executor),
            Host::Qwen35(g) => spawn_adapter_watcher(follow, Some(g), executor),
            Host::None | Host::Ambiguous(_) => spawn_adapter_watcher::<QwenResident>(follow, None, executor),
        }
    }
}

/// How often [`AdapterWatcher`] looks at its directory. Small enough that a
/// promoted adapter reaches the serving path in well under a second, large
/// enough that an idle server's watcher costs one `read_dir` of a directory
/// holding a handful of files per tick.
const POLL: std::time::Duration = std::time::Duration::from_millis(200);

/// A background thread watching one directory or manifest for a newer adapter
/// and offering it on a live `QwenResident` through [`LiveAdapters`].
///
/// Opt-in (`brain serve --watch-adapters DIR`), never on by default: a
/// serving process that silently starts offering a model because a file
/// appeared on disk is not something an operator should get without asking.
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
    /// How many adapters this watcher has offered. Observable so a caller
    /// (and the test) can wait for a swap to land instead of racing the poll
    /// interval.
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
         (--checkpoint <dir-or-file>) and restart. Until then every request is answered by \
         the base weights no matter what is promoted into that directory, so a before/after \
         comparison will show no difference and look like a training failure.",
        dir.display()
    ))
}

/// Spawn the opt-in watcher, or `None` when there is nothing to watch:
/// `follow` is `None` (no following flag was given - the default, or a
/// pinned adapter) or `resident` is `None` (this process serves no Qwen3, so
/// no adapter could be applied to anything).
pub fn spawn_adapter_watcher<R: AdapterHost>(follow: Option<Follow>, resident: Option<Arc<R>>, executor: &Executor) -> Option<AdapterWatcher> {
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
        if let Some(warning) = ignored_watch_warning(Some(&follow.describe()), false) {
            eprintln!("{warning}");
        }
        return None;
    };
    let stop = Arc::new(AtomicBool::new(false));
    let swaps = Arc::new(AtomicU64::new(0));
    let rejections = Arc::new(AtomicU64::new(0));
    let live = LiveAdapters::new(resident, executor.clone());
    let counters = WatchCounters { stop: stop.clone(), swaps: swaps.clone(), rejections: rejections.clone() };
    match &follow.source {
        FollowSource::Latest(dir) => eprintln!("brain serve: watching {} for promoted LoRA adapters", dir.display()),
        FollowSource::Manifest(follower) => eprintln!("brain serve: following release manifest {}", follower.path().display()),
    }
    let join = std::thread::Builder::new()
        .name("brain-adapter-watch".to_string())
        .spawn(move || watch_loop(&mut follow, &live, &counters))
        .ok()?;
    Some(AdapterWatcher { stop, swaps, rejections, join: Some(join) })
}

/// The watcher thread's half of [`AdapterWatcher`]'s shared state.
struct WatchCounters {
    stop: Arc<AtomicBool>,
    swaps: Arc<AtomicU64>,
    rejections: Arc<AtomicU64>,
}

/// A file's size and modification time: whether a file that failed to load
/// has been touched since, so a broken file is read again only when it
/// changed.
#[derive(Clone, PartialEq, Eq)]
struct FileStamp(Option<(u64, std::time::SystemTime)>);

impl FileStamp {
    fn of(path: &Path) -> FileStamp {
        FileStamp(std::fs::metadata(path).ok().and_then(|m| Some((m.len(), m.modified().ok()?))))
    }
}

/// What the watcher remembers between ticks about a directory.
#[derive(Default)]
struct DirMemory {
    /// The newest file already offered.
    offered: Option<PathBuf>,
    /// A file that failed to load, and how it looked: said once, retried when
    /// it changes.
    failed: Option<(PathBuf, FileStamp)>,
}

/// The next adapter `follow` names that is not already offered, if any. A
/// manifest's own refusals are said here, once each, since the previous
/// adapter silently staying served would otherwise be their only symptom.
fn next_adapter(source: &mut FollowSource, memory: &mut DirMemory, rejections: &AtomicU64) -> Option<LoadedAdapter> {
    match source {
        FollowSource::Latest(dir) => match rl::improve::latest_adapter(dir) {
            // This directory's files are not bound to the base (a gated
            // promotion put them there), but each is read once, named by its
            // card id and content digest, and refused if it is not a Qwen3
            // LoRA.
            Ok(Some((_, path))) if memory.offered.as_deref() != Some(path.as_path()) => {
                let stamp = FileStamp::of(&path);
                if memory.failed.as_ref().is_some_and(|(p, s)| *p == path && *s == stamp) {
                    return None;
                }
                match load_adapter(&path) {
                    Ok(loaded) => Some(loaded),
                    Err(why) => {
                        eprintln!("brain serve: not offering the adapter {}: {why}", path.display());
                        memory.failed = Some((path, stamp));
                        None
                    }
                }
            }
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
        // under the path already served is still offered.
        FollowSource::Manifest(follower) => match follower.poll() {
            ManifestPoll::Release(loaded) => Some(*loaded),
            ManifestPoll::Rejected(why) => {
                rejections.fetch_add(1, Ordering::SeqCst);
                eprintln!("brain serve: keeping the adapter already served: {why}");
                None
            }
            ManifestPoll::Missing | ManifestPoll::Unchanged => None,
        },
    }
}

/// The watcher's body: offer each new adapter `follow` names, then withdraw
/// the one it replaces. In that order, so a client that selects the new id
/// never finds neither.
fn watch_loop(follow: &mut Follow, live: &LiveAdapters, counters: &WatchCounters) {
    let WatchCounters { stop, swaps, rejections } = counters;
    let mut memory = DirMemory::default();
    while !stop.load(Ordering::SeqCst) {
        if let Some(loaded) = next_adapter(&mut follow.source, &mut memory, rejections) {
            let release = loaded.release.clone();
            match live.serve(loaded) {
                Ok(()) => {
                    eprintln!("{}", release.serving_line(&release.id));
                    memory.offered = Some(release.path.clone());
                    if let Some(old) = follow.served.replace(release.id.clone()).filter(|old| *old != release.id) {
                        live.retire(&old);
                    }
                    swaps.fetch_add(1, Ordering::SeqCst);
                }
                Err(why) => {
                    eprintln!("brain serve: not offering the adapter {}: {why}", release.path.display());
                    memory.failed = Some((release.path.clone(), FileStamp::of(&release.path)));
                }
            }
        }
        std::thread::sleep(POLL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_testutil::adapters::{file_digest as digest, scratch_dir as tmp, write_adapter, write_base, write_byte_tokenizer, write_lora_adapter, Linear};
    use catalog::adapter_registry::AdapterBankSettings;
    use catalog::resident_llm::QwenServeConfig;
    use checkpoint::st::ModelCard;
    use qwen3::config::QwenConfig;
    use residency::{budget::Budgets, Device, Policy};
    use std::sync::Arc;

    fn skip() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    /// The card id the fixtures' adapters carry: an adapter reference.
    const ONE: &str = "brain/qwen3:test:release:v1";
    const TWO: &str = "brain/qwen3:test:release:v2";

    /// A server config with the default adapter bank.
    fn banked() -> QwenServeConfig {
        QwenServeConfig { adapter_bank: Some(AdapterBankSettings::default()), ..QwenServeConfig::default() }
    }

    fn write_tiny_base(path: &std::path::Path, seed: u64, cfg: &QwenConfig) {
        let init = qwen3::init_weights(cfg, seed);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
            .param_list()
            .into_iter()
            .map(|(name, n)| (name.clone(), vec![n as u64], init.get(&name).unwrap_or_else(|| panic!("init missing {name}")).clone()))
            .collect();
        checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
    }

    /// Every attention linear of the tiny model, in the shapes an adapter
    /// file stores them.
    fn attention_linears(cfg: &QwenConfig) -> Vec<Linear> {
        let (d, hq, hkv) = (cfg.d_model as usize, cfg.q_dim() as usize, cfg.kv_dim() as usize);
        (0..cfg.n_layers)
            .flat_map(|l| [("wq", hq, d), ("wk", hkv, d), ("wv", hkv, d), ("wo", d, hq)].map(|(leaf, out, inn)| Linear { base: format!("blocks.{l}.attn.{leaf}.weight"), out, inn }))
            .collect()
    }

    fn cpu_executor(models: Vec<Arc<dyn residency::ResidentModel>>) -> Executor {
        let mut budgets = Budgets::new();
        budgets.set(Device::Cpu, 8 << 30, 0);
        Executor::start(models, budgets, Policy::default())
    }

    fn listed(executor: &Executor, model: &str) -> bool {
        executor.manifests().iter().any(|m| m.model == model)
    }

    /// The watcher is OPT-IN: `brain serve` without `--watch-adapters` must
    /// spawn no background thread at all, and neither must a run that names a
    /// directory but serves no Qwen3 (nothing to put an adapter on).
    #[test]
    fn no_watcher_is_spawned_unless_the_flag_names_an_adapter_directory() {
        let mut budgets = Budgets::new();
        budgets.set(Device::Cpu, 1 << 30, 0);
        let executor = Executor::start(Vec::new(), budgets, Policy::default());
        let dir = tmp("watcher-off");

        assert!(spawn_adapter_watcher::<QwenResident>(None, None, &executor).is_none(), "no --watch-adapters must spawn no watcher");
        assert!(
            spawn_adapter_watcher::<QwenResident>(Some(Follow::latest(dir)), None, &executor).is_none(),
            "a watched directory with no served Qwen3 has nothing to offer an adapter on, so it must spawn no watcher either"
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
        assert!(ignored_watch_warning(Some(&dir), true).is_none(), "a resident to put an adapter on means nothing is wrong");

        let warning = ignored_watch_warning(Some(&dir), false).expect("a flag that cannot work must be reported");
        assert!(warning.contains("IGNORED"), "it must say the flag is not in effect: {warning}");
        assert!(warning.contains("--checkpoint"), "and name the remedy: {warning}");
        assert!(warning.contains(&dir.display().to_string()), "and name the directory it is ignoring: {warning}");
    }

    /// The flags pick exactly one source for the offered adapter.
    #[test]
    fn the_adapter_flags_pick_exactly_one_source() {
        let p = || Some(PathBuf::from("x"));
        assert!(matches!(AdapterMode::from_flags(None, None, None), Ok(AdapterMode::Base)));
        assert!(matches!(AdapterMode::from_flags(p(), None, None), Ok(AdapterMode::Pinned(_))));
        assert!(matches!(AdapterMode::from_flags(None, p(), None), Ok(AdapterMode::Manifest(_))));
        assert!(matches!(AdapterMode::from_flags(None, None, p()), Ok(AdapterMode::Latest(_))));
        for (a, m, w) in [(p(), p(), None), (p(), None, p()), (None, p(), p())] {
            assert!(AdapterMode::from_flags(a, m, w).is_err(), "two adapter sources at once must be refused");
        }
    }

    /// The live-serving gap: the directory watcher has to work against a
    /// running server over the real OpenAI router. Serve the tiny fixture,
    /// hold a base request in flight, publish an adapter into the watched
    /// directory, and check the contract: the in-flight request completes,
    /// the adapter is a model of its own (listed with its digest, never a
    /// path) that answers differently, the BASE keeps answering exactly as it
    /// did, and a later release replaces the earlier one as a model without a
    /// restart or a gap.
    #[test]
    fn a_promoted_adapter_is_served_beside_the_base_without_restart() {
        if skip() {
            return;
        }
        let dir = tmp("live-swap");
        let tok_path = write_byte_tokenizer(&dir.join("tok"));
        let cfg = QwenConfig { vocab: 128, ..QwenConfig::tiny() };
        let base = dir.join("base.safetensors");
        write_tiny_base(&base, 7, &cfg);

        // Two releases, staged OFF to the side so that publishing one is a
        // single file appearing in the watched directory.
        let staging = dir.join("staging");
        std::fs::create_dir_all(&staging).unwrap();
        let linears = attention_linears(&cfg);
        let (first, second) = (staging.join("adapter-000000.safetensors"), staging.join("adapter-000001.safetensors"));
        write_lora_adapter(&first, ONE, "brain/qwen3", 4, &linears, 31);
        write_lora_adapter(&second, TWO, "brain/qwen3", 4, &linears, 32);

        let card = ModelCard::new("brain/qwen3", "qwen");
        let resident = Arc::new(QwenResident::from_card_configured(base.to_str().unwrap(), &card, Some(tok_path.to_str().unwrap()), banked()));
        let executor = cpu_executor(vec![resident.clone()]);
        let watched = dir.join("adapters");
        std::fs::create_dir_all(&watched).unwrap();
        let watcher = spawn_adapter_watcher(Some(Follow::latest(watched.clone())), Some(resident.clone()), &executor).expect("a watched directory + a served Qwen3 spawns the watcher");

        let api_key = "sk-brain-test-key".to_string();
        let state = apiserve::AppState::new(executor.clone(), api_key.clone(), apiserve::Provider::OpenAI);
        let app = apiserve::router(state);
        let post = {
            let api_key = api_key.clone();
            move |model: &str, user: &str, max_tokens: u32| {
                let body = serde_json::json!({
                    "model": model,
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
            }
        };
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build().unwrap();
        let content = |app: axum::Router, req: axum::http::Request<axum::body::Body>| -> (axum::http::StatusCode, String) {
            use tower::ServiceExt;
            rt.block_on(async move {
                let r = app.oneshot(req).await.unwrap();
                let status = r.status();
                let bytes = axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
                let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
                (status, v["choices"][0]["message"]["content"].as_str().unwrap_or_default().to_string())
            })
        };
        // The `/v1/models` listing: what an agent runtime reads to learn which models, and which fine-tunes, are served.
        let models = |app: axum::Router| -> Vec<serde_json::Value> {
            use tower::ServiceExt;
            let req = axum::http::Request::builder().uri("/v1/models").header(axum::http::header::AUTHORIZATION, format!("Bearer {api_key}")).body(axum::body::Body::empty()).unwrap();
            rt.block_on(async move {
                let r = app.oneshot(req).await.unwrap();
                let bytes = axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
                serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()["data"].as_array().unwrap().clone()
            })
        };
        let card_of = |list: &[serde_json::Value], id: &str| list.iter().find(|m| m["id"] == id).cloned();

        let listing = models(app.clone());
        assert_eq!(listing.len(), 1, "before any release only the base is served: {listing:?}");
        assert_eq!(card_of(&listing, "brain/qwen3").unwrap().get("adapter"), None, "a base names no adapter");

        let (status, before) = content(app.clone(), post("brain/qwen3", "what colour is the sky", 16));
        assert_eq!(status, axum::http::StatusCode::OK, "the base must serve before anything is published");
        assert!(!before.is_empty(), "the baseline answer must be non-empty, or 'the answer changed' means nothing");

        // Hold a request in flight, and only publish once the executor really is running it.
        let (in_app, in_req) = (app.clone(), post("brain/qwen3", "what colour is the sky", 192));
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
            assert!(std::time::Instant::now() < deadline, "the long request never reached the executor");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        std::fs::copy(&first, watched.join(first.file_name().unwrap())).expect("publish the first release");

        let (status, bytes) = in_flight.join().expect("the in-flight request thread must not panic");
        assert_eq!(status, axum::http::StatusCode::OK, "a release arriving mid-request must never break that request");
        let v: serde_json::Value = serde_json::from_slice(&bytes).expect("the in-flight response must still be well-formed JSON");
        assert!(!v["choices"][0]["message"]["content"].as_str().unwrap_or_default().is_empty(), "the in-flight request must complete with real content");

        let wait = |what: &str, done: &dyn Fn() -> bool| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
            while !done() {
                assert!(std::time::Instant::now() < deadline, "timed out waiting for {what}");
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        };
        wait("the watcher to offer the first release", &|| watcher.swaps() == 1);

        let listing = models(app.clone());
        assert_eq!(listing.len(), 2, "the base and the release: {listing:?}");
        let adapter_card = card_of(&listing, ONE).expect("the release is a model of its own");
        assert_eq!(adapter_card["adapter"]["digest"], digest(&first), "named by the digest of the file now served");
        assert_eq!(adapter_card["adapter"]["id"], ONE);
        assert!(!listing.iter().any(|m| m.to_string().contains(".safetensors") || m.to_string().contains("staging") || m.to_string().contains("adapters/")), "and never by where the file lives: {listing:?}");
        assert_eq!(card_of(&listing, "brain/qwen3").unwrap().get("adapter"), None, "the base entry is unchanged");

        let (status, tuned) = content(app.clone(), post(ONE, "what colour is the sky", 16));
        assert_eq!(status, axum::http::StatusCode::OK, "the release serves");
        assert_ne!(tuned, before, "the adapter changes the answer");
        let (_, base_again) = content(app.clone(), post("brain/qwen3", "what colour is the sky", 16));
        assert_eq!(base_again, before, "and the base still answers exactly as it did");

        // The next release replaces the first as a model: offered, then the old one withdrawn.
        std::fs::copy(&second, watched.join(second.file_name().unwrap())).expect("publish the second release");
        wait("the watcher to offer the second release", &|| watcher.swaps() == 2);
        wait("the first release to be withdrawn", &|| !listed(&executor, ONE));
        let (status, newer) = content(app.clone(), post(TWO, "what colour is the sky", 16));
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_ne!(newer, tuned, "the later release answers differently");
        let (status, _) = content(app.clone(), post(ONE, "what colour is the sky", 16));
        assert_eq!(status, axum::http::StatusCode::NOT_FOUND, "the withdrawn release is an unknown model");
        assert_eq!(models(app.clone()).len(), 2, "the base and the newest release");
    }

    /// `--adapter`: the pinned release is a model of its own from startup,
    /// reported by digest, and no watcher is left running that could replace
    /// it. A release for another base is a startup error and offers nothing;
    /// so is one on a server started without a bank.
    #[test]
    fn a_pinned_adapter_is_offered_and_never_followed() {
        let dir = tmp("pinned");
        let (base, other) = (dir.join("base.safetensors"), dir.join("other.safetensors"));
        write_base(&base, 1.0);
        write_base(&other, 2.0);
        let adapter = dir.join("adapter.safetensors");
        write_adapter(&adapter, ONE, "local/base", Some(digest(&base)), 0.5);

        let card = ModelCard::new("brain/qwen3", "qwen");
        let resident = Arc::new(QwenResident::from_card_configured(base.to_str().unwrap(), &card, None, banked()));
        let executor = cpu_executor(vec![resident.clone()]);
        let (follow, line) = start_adapter_mode(AdapterMode::Pinned(adapter.clone()), Some(&resident), &executor).expect("the adapter's own base serves it");
        assert!(follow.is_none(), "a pinned adapter must leave nothing to follow");
        assert_eq!(line.as_deref(), Some(format!("brain serve: {ONE} adapter={ONE} digest={}", digest(&adapter)).as_str()));
        assert!(listed(&executor, ONE) && listed(&executor, "brain/qwen3"), "the adapter is a model beside the base");
        assert_eq!(executor.served_adapter(ONE), Some(residency::AdapterIdentity { id: ONE.to_string(), digest: digest(&adapter) }), "the serving surfaces are told the verified release, by id and digest");
        assert_eq!(executor.served_adapter("brain/qwen3"), None, "the base names none");

        let wrong = Arc::new(QwenResident::from_card_configured(other.to_str().unwrap(), &card, None, banked()));
        let wrong_exec = cpu_executor(vec![wrong.clone()]);
        let Err(err) = start_adapter_mode(AdapterMode::Pinned(adapter.clone()), Some(&wrong), &wrong_exec) else {
            panic!("an adapter trained on another base must not start");
        };
        assert!(err.contains(&digest(&base)) && err.contains(&digest(&other)), "{err}");
        assert!(!listed(&wrong_exec, ONE), "a refused adapter is never offered");

        let bankless = Arc::new(QwenResident::from_card(base.to_str().unwrap(), &card, None));
        let Err(err) = start_adapter_mode(AdapterMode::Pinned(adapter.clone()), Some(&bankless), &cpu_executor(vec![bankless.clone()])) else {
            panic!("a server without an adapter bank cannot offer an adapter");
        };
        assert!(err.contains("--adapter-slots"), "{err}");
        assert!(start_adapter_mode::<QwenResident>(AdapterMode::Pinned(adapter), None, &executor).is_err(), "an adapter with no Qwen3 to put it on is an error");
    }

    /// `--adapter-manifest`: a replaced manifest offers the new release only
    /// after its digest verifies, then withdraws the previous one; a corrupt
    /// one, or one whose digest does not match its file, keeps the previous
    /// adapter served.
    #[test]
    fn a_followed_manifest_offers_only_verified_releases_and_retires_the_old_one() {
        let dir = tmp("manifest-watch");
        let base = dir.join("base.safetensors");
        write_base(&base, 1.0);
        let (one, two) = (dir.join("adapter-1.safetensors"), dir.join("adapter-2.safetensors"));
        write_adapter(&one, ONE, "local/base", Some(digest(&base)), 0.5);
        write_adapter(&two, TWO, "local/base", Some(digest(&base)), 0.25);
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
        let resident = Arc::new(QwenResident::from_card_configured(base.to_str().unwrap(), &card, None, banked()));
        let executor = cpu_executor(vec![resident.clone()]);
        let (follow, line) = start_adapter_mode(AdapterMode::Manifest(manifest.clone()), Some(&resident), &executor).expect("a valid manifest serves at startup");
        assert_eq!(line.as_deref(), Some(format!("brain serve: {ONE} adapter={ONE} digest={}", digest(&one)).as_str()));
        assert!(listed(&executor, ONE));
        assert_eq!(executor.served_adapter(ONE).map(|a| a.digest), Some(digest(&one)));
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
        assert!(listed(&executor, ONE), "a corrupt manifest keeps the previous adapter");
        assert_eq!(executor.served_adapter(ONE).map(|a| a.digest), Some(digest(&one)), "and keeps reporting it");

        publish(release(&two, digest(&one)));
        wait("the mismatched digest to be rejected", &|| watcher.rejections() == 2);
        assert!(listed(&executor, ONE) && !listed(&executor, TWO), "a release whose digest does not verify is never offered");

        publish(release(&two, digest(&two)));
        wait("the verified release to be offered", &|| watcher.swaps() == 1);
        wait("the previous release to be withdrawn", &|| !listed(&executor, ONE));
        assert!(listed(&executor, TWO) && listed(&executor, "brain/qwen3"));
        assert_eq!(executor.served_adapter(TWO), Some(residency::AdapterIdentity { id: TWO.to_string(), digest: digest(&two) }), "a hot swap moves the reported identity with the served file");
        assert_eq!(executor.served_adapter(ONE), None);
    }
}
