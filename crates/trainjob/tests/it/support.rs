// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The rig the integration tests share: a model store holding a toy base, an artifact store, an executor.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use capability::{Blob, Invocation, Media, Progress};
use residency::budget::Budgets;
use residency::{Device, Executor, Policy};
use serde_json::{json, Value};
use trainjob::{ArtifactStore, Digest, TrainConfig};

use crate::fixture::{qwen35_model_dir, qwen3_model_dir_of, TempDir};

/// (prompt, answer, advantage): two actions of well-rewarded rollouts and two of badly rewarded ones.
const ROLLOUTS: [(&str, &str, f64); 4] = [("cab", "bad", 1.0), ("fed", "dad", 1.0), ("ace", "cab", -1.0), ("bag", "egg", -1.0)];

pub const BASE: &str = "local/toy";

pub struct Rig {
    pub dir: TempDir,
    pub store: ArtifactStore,
    pub exec: Executor,
    pub work_root: PathBuf,
}

/// A server with `memory` bytes of host memory to place runs in, holding the toy base as `local/toy`.
pub fn rig(tag: &str, memory: u64, store: bool) -> Rig {
    rig_with(tag, memory, store, None, Vec::new())
}

/// [`rig`] on a host whose largest card holds `card` bytes (when it has one), serving `extra` beside the training workload.
pub fn rig_with(tag: &str, memory: u64, store: bool, card: Option<u64>, extra: Vec<Arc<dyn residency::ResidentModel>>) -> Rig {
    rig_over(tag, memory, store, card, extra, qwen35_model_dir(&format!("{tag}-base")))
}

/// [`rig_qwen3`] with or without an artifact store.
pub fn rig_with_store_flag(tag: &str, store: bool) -> Rig {
    rig_over(tag, 16 << 30, store, None, Vec::new(), qwen3_model_dir_of(&format!("{tag}-base"), qwen3::QwenConfig::tiny()))
}

/// [`rig`] holding a toy `qwen3` base as `local/toy`.
pub fn rig_qwen3(tag: &str, memory: u64) -> Rig {
    rig_over(tag, memory, true, None, Vec::new(), qwen3_model_dir_of(&format!("{tag}-base"), qwen3::QwenConfig::tiny()))
}

/// A server whose store holds `made` (a model directory) as `local/toy`.
fn rig_over(tag: &str, memory: u64, store: bool, card: Option<u64>, extra: Vec<Arc<dyn residency::ResidentModel>>, made: TempDir) -> Rig {
    let dir = TempDir::new(tag);
    let models = dir.join("models");
    let base = models.join("local").join("toy");
    std::fs::create_dir_all(&base).unwrap();
    std::fs::rename(made.join("model.safetensors"), base.join("model.brain.safetensors")).unwrap();
    for file in ["tokenizer.json", "tokenizer_config.json"] {
        std::fs::rename(made.join(file), base.join(file)).unwrap();
    }
    let artifacts = dir.join("artifacts");
    std::fs::create_dir_all(&artifacts).unwrap();
    let store_handle = ArtifactStore::open(&artifacts).unwrap();
    let work_root = dir.join("work");
    let config = TrainConfig { models_dir: Some(models.clone()), store: store.then(|| store_handle.clone()), work_root: work_root.clone(), largest_card_bytes: card, devices: 1 };
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, memory, 0);
    let mut served = vec![trainjob::resident(config)];
    served.extend(extra);
    let exec = Executor::start(served, budgets, Policy::default());
    Rig { dir, store: store_handle, exec, work_root }
}

pub fn record(prompt: &str, answer: &str, advantage: Option<f64>) -> String {
    let mut record = json!({"messages": [{"role": "user", "content": prompt, "train": false}, {"role": "assistant", "content": answer, "train": true}], "tools": []});
    if let Some(a) = advantage {
        record["advantage"] = json!(a);
    }
    record.to_string()
}

/// The rollout records of the toy task (each carrying an advantage), as a JSONL file's bytes.
pub fn rollouts() -> Vec<u8> {
    ROLLOUTS.iter().map(|(p, a, adv)| record(p, a, Some(*adv))).collect::<Vec<_>>().join("\n").into_bytes()
}

/// One `generic-preference-v1` pair of the toy task.
pub fn pair(user: &str, chosen: &str, rejected: &str) -> String {
    json!({"prompt": [{"role": "user", "content": user}], "chosen": {"role": "assistant", "content": chosen}, "rejected": {"role": "assistant", "content": rejected}}).to_string()
}

/// Preference pairs of the toy task: what the rollouts rewarded is preferred to what they punished.
pub fn pairs() -> Vec<u8> {
    [("cab", "bad", "wig"), ("fed", "dad", "vow"), ("ace", "cab", "jug"), ("bag", "egg", "mop")].iter().map(|(u, c, r)| pair(u, c, r)).collect::<Vec<_>>().join("\n").into_bytes()
}

/// The same records as plain supervised ones.
pub fn plain() -> Vec<u8> {
    ROLLOUTS.iter().map(|(p, a, _)| record(p, a, None)).collect::<Vec<_>>().join("\n").into_bytes()
}

/// `params` the way a surface hands them to the workload: validated against the action's spec as a remote caller sees it.
pub fn invocation(params: Value, blobs: &[(&str, Vec<u8>)]) -> Result<Invocation, String> {
    let mut inv = Invocation::new();
    inv.params = params;
    for (name, bytes) in blobs {
        inv.blobs.insert((*name).to_string(), Blob::new(Media::Bytes, bytes.clone()));
    }
    trainjob::request::action_spec().validate_served(inv)
}

/// What a run reported on the way.
#[derive(Default)]
pub struct Frames(pub Arc<Mutex<Vec<Progress>>>);

impl Frames {
    pub fn sink(&self) -> impl FnMut(Progress) + Send + 'static {
        let frames = Arc::clone(&self.0);
        move |p| frames.lock().unwrap().push(p)
    }

    /// The structured frames, in order.
    pub fn events(&self) -> Vec<Value> {
        self.0.lock().unwrap().iter().filter_map(|p| p.event.clone()).collect()
    }

    pub fn steps(&self) -> Vec<Value> {
        self.events().into_iter().filter(|e| e["type"] == "step").collect()
    }
}

pub fn data_digest(rig: &Rig, bytes: &[u8]) -> Digest {
    rig.store.publish_bytes(bytes).unwrap()
}

/// Runs a job to its end on the rig.
pub fn run(rig: &Rig, params: Value, blobs: &[(&str, Vec<u8>)], frames: &Frames) -> Result<capability::Outcome, String> {
    let inv = invocation(params, blobs)?;
    rig.exec.run_blocking(trainjob::MODEL, trainjob::ACTION, inv, frames.sink())
}

/// Runs `action` of the service (validated against its spec as a remote caller sees it) to its end on the rig.
pub fn run_action(rig: &Rig, action: &str, spec: &capability::ActionSpec, params: Value, blobs: &[(&str, Vec<u8>)], frames: &Frames) -> Result<capability::Outcome, String> {
    let mut inv = Invocation::new();
    inv.params = params;
    for (name, bytes) in blobs {
        inv.blobs.insert((*name).to_string(), Blob::new(Media::Bytes, bytes.clone()));
    }
    let inv = spec.validate_served(inv)?;
    rig.exec.run_blocking(trainjob::MODEL, action, inv, frames.sink())
}

/// Nothing of any job is left in the work directory.
pub fn work_is_clean(work_root: &Path) -> bool {
    std::fs::read_dir(work_root).map(|d| d.count() == 0).unwrap_or(true)
}

/// Waits (bounded) until nothing is placed and no device is charged: a run replies before the dispatcher has
/// processed its end and returned the reservation, so a check made at once would race it.
pub fn assert_reservation_returned(rig: &Rig) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let placed = rig.exec.residency();
        if placed.placements.is_empty() && placed.budgets.iter().all(|b| b.used == 0) {
            return;
        }
        assert!(std::time::Instant::now() < deadline, "a finished run still holds its reservation: {placed:?}");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}
