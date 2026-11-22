// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training services whose jobs survive the
// death of the job or the server and continue where they stopped, for its
// clients. If your team needs expertise in resumable, idempotent training on
// shared accelerators, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Resumable jobs: the contract of a `resume_key`.
//!
//! A training job (`init_model`, `pretrain`, `full_train`, and the adapter job `train` of every objective) that names
//! a `resume_key` keeps a durable **run record** in the artifact store's directory, `<store>/.runs/<blake3(key)>/`:
//!
//! * `run.json`: `{"format": "brain-train-run-v1", "key", "action", "identity", "seed", "status": "running" |
//!   "completed", "result"}`, written atomically (a temporary file in the same directory, synced, renamed, the
//!   directory synced), so a crash leaves the previous record or the new one, never half of either;
//! * `state.safetensors` (and `state.offdevice/` for a model with off-device state): the fit loop's exact training
//!   state (weights, AdamW moments, schedule position, data order, generator), itself written atomically, every
//!   `checkpoint_every` steps and when the job is cancelled; removed once the checkpoint or adapter is exported;
//! * `evaluations/`: an adapter run's adapter of each evaluation (`keep_evaluations`), kept here so a resumed run
//!   publishes the ones written before it stopped; removed once published;
//! * `lock`: held (`flock`) while a job of the key runs, released by the kernel however the process ends.
//!
//! **Identity.** The run's identity is the blake3 digest of its action, every param but `resume_key` and
//! `checkpoint_every` (digests of artifact inputs included) and the blake3 digest of every attached input blob. A
//! key whose record names another identity is refused: one key is one run. An omitted `seed` is drawn once and kept
//! in the record, so a resubmission continues with the same seed. The record also keeps the run's [`Layout`] (its
//! ranks and sharding), so a resubmission on another layout is refused with both layouts' numbers: each rank's state
//! is its own shard of the stream and of the optimizer, and resumes only on the layout that saved it.
//!
//! **Resubmitting** a key: a `completed` record answers with the result it recorded (`resume.state: "completed"`)
//! once every artifact it names is verified to be in the store, and recomputes nothing; a `running` record continues
//! from its training state (`resume.state: "continued"`, `resumed_at` the step) or, when the run had saved none yet,
//! starts it again; a key whose job is still running is refused. A record or a state that cannot be read, or a
//! state of a different run, is refused loudly and never silently restarted: the caller chooses a new key.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use capability::Invocation;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::store::{digest_of, ArtifactStore, Digest};

/// The longest `resume_key` accepted.
pub const MAX_KEY_LEN: usize = 128;
/// The format of a run record.
pub const RECORD_FORMAT: &str = "brain-train-run-v1";
/// The directory of the run records inside the artifact store's directory.
pub const RUNS_DIR: &str = ".runs";
/// How often a resumable run saves its state unless it names `checkpoint_every`.
pub const DEFAULT_CHECKPOINT_EVERY: u32 = 25;

/// The params of a request that configure resumption rather than the run: not part of its identity.
const RESUME_PARAMS: [&str; 2] = ["resume_key", "checkpoint_every"];

/// `key` is 1 to [`MAX_KEY_LEN`] letters, digits, `.`, `_` and `-`, not starting with a dot.
pub fn check_key(key: &str) -> Result<(), String> {
    if key.is_empty() || key.len() > MAX_KEY_LEN || key.starts_with('.') || !key.chars().all(|c| c.is_ascii_alphanumeric() || "._-".contains(c)) {
        return Err(format!("resume_key must be 1 to {MAX_KEY_LEN} letters, digits, `.`, `_` and `-`, not starting with a dot"));
    }
    Ok(())
}

/// The identity of the run `inv` of `action` asks for (see the module doc).
pub fn identity(action: &str, inv: &Invocation) -> Digest {
    let mut params = inv.params.clone();
    if let Some(map) = params.as_object_mut() {
        for name in RESUME_PARAMS {
            map.remove(name);
        }
    }
    let blobs: serde_json::Map<String, Value> = inv.blobs.iter().map(|(name, blob)| (name.clone(), json!(digest_of(&blob.bytes).hex()))).collect();
    // serde_json's map is ordered by key, so the same request always serializes to the same bytes.
    digest_of(json!({"action": action, "params": params, "blobs": blobs}).to_string().as_bytes())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Running,
    Completed,
}

/// How a run is spread over ranks: what a resumed run must keep (see the module doc).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layout {
    /// Ranks of the whole group, every server's (1 for a run on one card).
    pub world_size: usize,
    /// The ranks this server runs, `[first, end)`.
    pub ranks: [usize; 2],
    /// What the ranks shard, as the request names it (or its job plan does); `None` where the server chooses.
    pub sharding: Option<String>,
}

impl Layout {
    /// A run on one card, not data-parallel.
    pub fn single() -> Layout {
        Layout { world_size: 1, ranks: [0, 1], sharding: None }
    }
}

impl std::fmt::Display for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} rank(s), ranks {}..{} on this server, sharding {}", self.world_size, self.ranks[0], self.ranks[1], self.sharding.as_deref().unwrap_or("chosen by the server"))
    }
}

/// `run.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub format: String,
    pub key: String,
    pub action: String,
    pub identity: String,
    pub seed: Option<u64>,
    pub status: Status,
    pub result: Option<Value>,
    /// The layout the run began on; absent from a record written before layouts were kept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<Layout>,
}

/// A run of a key this job holds the lock of.
pub struct Run {
    dir: PathBuf,
    record: Record,
    /// Held for the job's lifetime: the lock is the open file.
    _lock: File,
    /// The run had a record before this job: a resumption.
    pub resumed: bool,
}

/// What opening a key found.
pub enum Opened {
    /// A run to train, fresh or continued.
    Run(Run),
    /// The key's run completed: its recorded result.
    Completed(Value),
}

fn refuse(key: &str, why: impl std::fmt::Display) -> String {
    format!("resume_key {key:?}: {why}")
}

/// Writes `bytes` to `path` atomically and durably.
fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().ok_or_else(|| std::io::Error::other("no directory"))?;
    let tmp = dir.join(format!(".{}.tmp-{}", path.file_name().and_then(|n| n.to_str()).unwrap_or("record"), std::process::id()));
    {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    File::open(dir)?.sync_all()
}

#[cfg(unix)]
fn lock_exclusive(file: &File) -> bool {
    use std::os::unix::io::AsRawFd;
    // SAFETY: `flock` on a descriptor this process owns; it touches no memory.
    unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) == 0 }
}

#[cfg(not(unix))]
fn lock_exclusive(_file: &File) -> bool {
    true
}

/// Opens the run of `key` for the request whose identity is `identity`, laid out as `layout`, creating its record
/// when there is none.
pub fn open(store: &ArtifactStore, action: &str, key: &str, identity: &Digest, seed: Option<u64>, layout: &Layout) -> Result<Opened, String> {
    check_key(key)?;
    let dir = store.root().join(RUNS_DIR).join(digest_of(key.as_bytes()).hex());
    std::fs::create_dir_all(&dir).map_err(|e| refuse(key, format!("cannot make its run directory: {e}")))?;
    let lock = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(dir.join("lock")).map_err(|e| refuse(key, format!("cannot open its lock: {e}")))?;
    if !lock_exclusive(&lock) {
        return Err(refuse(key, "a job of this key is running; wait for it to end or cancel it"));
    }
    let path = dir.join("run.json");
    let existing = match std::fs::read(&path) {
        Ok(bytes) => Some(serde_json::from_slice::<Record>(&bytes).map_err(|e| refuse(key, format!("its run record cannot be read ({e}); choose a new key")))?),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(refuse(key, format!("its run record cannot be read: {e}"))),
    };
    match existing {
        Some(record) => {
            if record.format != RECORD_FORMAT || record.key != key {
                return Err(refuse(key, "its run record is not one this server wrote for the key; choose a new key"));
            }
            if let Some(began) = record.layout.as_ref().filter(|began| record.action == action && *began != layout) {
                return Err(refuse(key, format!("its run began on {began}, and this request asks {layout}: a run resumes only on the layout it began on (each rank's saved state is its own shard); submit it on that layout, or under a new key")));
            }
            if record.action != action || record.identity != identity.hex() {
                return Err(refuse(key, format!("names another run (a {} job with other inputs or settings); one key is one run", record.action)));
            }
            if record.status == Status::Completed {
                let result = record.result.clone().ok_or_else(|| refuse(key, "its record is completed with no result"))?;
                verify_published(store, key, &result)?;
                return Ok(Opened::Completed(result));
            }
            Ok(Opened::Run(Run { dir, record, _lock: lock, resumed: true }))
        }
        None => {
            let record = Record { format: RECORD_FORMAT.to_string(), key: key.to_string(), action: action.to_string(), identity: identity.hex(), seed: seed.or_else(|| Some(brain::random_seed())), status: Status::Running, result: None, layout: Some(layout.clone()) };
            write_atomic(&path, &serde_json::to_vec_pretty(&record).map_err(|e| e.to_string())?).map_err(|e| refuse(key, format!("cannot write its run record: {e}")))?;
            Ok(Opened::Run(Run { dir, record, _lock: lock, resumed: false }))
        }
    }
}

/// Every artifact a recorded result names is still in the store.
fn verify_published(store: &ArtifactStore, key: &str, result: &Value) -> Result<(), String> {
    // A checkpoint names its manifest and files; an adapter run names the adapter, its training record and the
    // adapters of its evaluations.
    let single = ["checkpoint_digest", "adapter_digest", "record_digest"].into_iter().filter_map(|field| result[field].as_str());
    let listed = ["files", "evaluations"].into_iter().flat_map(|list| result[list].as_array().into_iter().flatten().filter_map(|f| f["digest"].as_str()));
    let digests = single.chain(listed);
    for hex in digests {
        let digest = Digest::parse(hex).map_err(|e| refuse(key, format!("its record names {hex:?}: {e}")))?;
        if !store.contains(&digest) {
            return Err(refuse(key, format!("its run completed, and the artifact {hex} it published is no longer in the artifact store; choose a new key to train it again")));
        }
    }
    Ok(())
}

impl Run {
    /// Where the fit loop keeps this run's exact training state.
    pub fn state_path(&self) -> PathBuf {
        self.dir.join("state.safetensors")
    }

    /// Where an adapter run keeps the adapter of each evaluation until it is published.
    pub fn evaluations_dir(&self) -> PathBuf {
        self.dir.join("evaluations")
    }

    /// The seed the run trains with: the request's, or the one drawn when the run began.
    pub fn seed(&self) -> u64 {
        self.record.seed.expect("a record always carries its seed")
    }

    /// Records the run completed with `result` (atomically) and removes what is left of its training state.
    pub fn complete(mut self, result: &Value) -> Result<(), String> {
        self.record.status = Status::Completed;
        self.record.result = Some(result.clone());
        let path = self.dir.join("run.json");
        write_atomic(&path, &serde_json::to_vec_pretty(&self.record).map_err(|e| e.to_string())?).map_err(|e| refuse(&self.record.key, format!("cannot write its run record: {e}")))?;
        std::fs::remove_file(self.state_path()).ok();
        std::fs::remove_dir_all(self.state_path().with_extension("offdevice")).ok();
        // A run on several cards keeps each rank's state in numbered generations beside the state path.
        std::fs::remove_dir_all(self.state_path().with_extension("safetensors.job")).ok();
        std::fs::remove_dir_all(self.evaluations_dir()).ok();
        Ok(())
    }

    /// Records the run completed with `outputs`, which gain the report's `resume` object first: what every answer of a
    /// resumable run (a server's of its ranks of a job too) is recorded as, so a resubmission answers it again.
    pub fn finish(self, outputs: &mut Value, resumed_at: Option<u32>) -> Result<(), String> {
        outputs["resume"] = self.report(resumed_at);
        self.complete(outputs)
    }

    /// The `resume` object of the report.
    pub fn report(&self, resumed_at: Option<u32>) -> Value {
        let state = match (self.resumed, resumed_at) {
            (_, Some(_)) => "continued",
            (true, None) => "restarted",
            (false, None) => "fresh",
        };
        json!({"key": self.record.key, "state": state, "resumed_at": resumed_at})
    }
}

/// The `resume` object of a result answered from a completed record.
pub fn completed_report(key: &str) -> Value {
    json!({"key": key, "state": "completed", "resumed_at": null})
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::{Blob, Media};

    fn store() -> (tempfile::TempDir, ArtifactStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = ArtifactStore::open(dir.path()).unwrap();
        (dir, store)
    }

    fn inv(params: Value, blob: &[u8]) -> Invocation {
        let mut inv = Invocation::new();
        inv.params = params;
        inv.blobs.insert("dataset".into(), Blob::new(Media::Bytes, blob.to_vec()));
        inv
    }

    /// The identity is the request's, resumption settings aside: a blob's bytes and every other param count.
    #[test]
    fn the_identity_is_the_request_less_its_resumption_settings() {
        let a = identity("pretrain", &inv(json!({"steps": 3, "resume_key": "k", "checkpoint_every": 2}), b"x"));
        assert_eq!(a, identity("pretrain", &inv(json!({"steps": 3, "resume_key": "other", "checkpoint_every": 9}), b"x")));
        assert_ne!(a, identity("pretrain", &inv(json!({"steps": 4}), b"x")));
        assert_ne!(a, identity("pretrain", &inv(json!({"steps": 3}), b"y")));
        assert_ne!(a, identity("full_train", &inv(json!({"steps": 3}), b"x")));
    }

    /// One key is one run; a running key is locked; a completed one answers its result while its artifacts exist; an
    /// unreadable record is refused, never replaced.
    #[test]
    fn a_key_is_one_run_locked_while_it_runs_and_answered_once_complete() {
        let (_dir, store) = store();
        let one = Layout::single();
        let id = digest_of(b"run-a");
        let Opened::Run(run) = open(&store, "pretrain", "k-1", &id, None, &one).unwrap() else { panic!("a fresh run") };
        assert!(!run.resumed);
        let seed = run.seed();
        assert!(open(&store, "pretrain", "k-1", &id, None, &one).err().unwrap().contains("is running"), "the lock is held");
        drop(run);
        let Opened::Run(run) = open(&store, "pretrain", "k-1", &id, Some(5), &one).unwrap() else { panic!("a resumed run") };
        assert!(run.resumed);
        assert_eq!(run.seed(), seed, "the seed drawn when the run began is kept");
        let other = digest_of(b"run-b");
        drop(run);
        assert!(open(&store, "pretrain", "k-1", &other, None, &one).err().unwrap().contains("names another run"));
        assert!(open(&store, "full_train", "k-1", &id, None, &one).err().unwrap().contains("names another run"));
        let artifact = store.publish_bytes(b"checkpoint").unwrap();
        let Opened::Run(run) = open(&store, "pretrain", "k-1", &id, None, &one).unwrap() else { panic!() };
        run.complete(&json!({"status": "completed", "checkpoint_digest": artifact.hex()})).unwrap();
        let Opened::Completed(result) = open(&store, "pretrain", "k-1", &id, None, &one).unwrap() else { panic!("completed") };
        assert_eq!(result["checkpoint_digest"], artifact.hex());
        std::fs::remove_file(store.root().join(artifact.hex())).unwrap();
        assert!(open(&store, "pretrain", "k-1", &id, None, &one).err().unwrap().contains("no longer in the artifact store"));
        let record = store.root().join(RUNS_DIR).join(digest_of(b"k-2").hex());
        std::fs::create_dir_all(&record).unwrap();
        std::fs::write(record.join("run.json"), b"{\"format\": 1").unwrap();
        assert!(open(&store, "pretrain", "k-2", &id, None, &one).err().unwrap().contains("cannot be read"));
        assert_eq!(std::fs::read(record.join("run.json")).unwrap(), b"{\"format\": 1", "a corrupt record is never overwritten");
        for bad in ["", "a/b", "..", "a b", &"k".repeat(MAX_KEY_LEN + 1)] {
            assert!(open(&store, "pretrain", bad, &id, None, &one).is_err(), "{bad:?}");
        }
    }

    /// A run resumes only on the layout it began on: another one is refused with both layouts' numbers, before the
    /// identity (which holds the layout's params too) is compared.
    #[test]
    fn a_run_is_refused_on_another_layout_with_both_layouts() {
        let (_dir, store) = store();
        let id = digest_of(b"run-a");
        let two = Layout { world_size: 2, ranks: [0, 2], sharding: Some("gradients".into()) };
        drop(open(&store, "full_train", "k", &id, Some(7), &two).unwrap());
        let four = Layout { world_size: 4, ranks: [0, 4], sharding: Some("gradients".into()) };
        let refused = open(&store, "full_train", "k", &digest_of(b"run-a on 4"), Some(7), &four).err().unwrap();
        assert!(refused.contains("began on 2 rank(s), ranks 0..2 on this server, sharding gradients") && refused.contains("asks 4 rank(s), ranks 0..4"), "{refused}");
        let replicated = Layout { sharding: Some("replicated".into()), ..two.clone() };
        assert!(open(&store, "full_train", "k", &digest_of(b"replicated"), Some(7), &replicated).err().unwrap().contains("sharding replicated"));
        assert!(matches!(open(&store, "full_train", "k", &id, Some(7), &two), Ok(Opened::Run(_))), "the same layout resumes");
    }
}
