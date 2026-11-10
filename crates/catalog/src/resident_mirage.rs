// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Mirage visual perception behind the residency scheduler.
//!
//! `activate` loads the released checkpoint ONCE onto the assigned device; the
//! [`Instance`] owns the [`MirageSession`], so dropping the instance frees the
//! weights and the cached decoder memories. Both actions (`describe`,
//! `extract`) and their schema come from `mirage::caps`: this file holds no
//! second copy of the request checks, the options or the batching.
//!
//! # Batching: one device batch per group of queued requests
//!
//! Every request keys the same instance, and the model declares a
//! [`mirage::caps::MAX_BATCH`]-wide batch limit, so the scheduler hands the
//! instance everything queued for it (up to that limit) as one
//! [`Instance::run_batch`] call. The session turns the group into one
//! `describe_batch` per set of equal options: a batched encoder over all the
//! images and one lockstep decoder loop. The decoder is lockstep rather than
//! continuously batched because the whole group is a handful of hundred steps
//! of a small model; a request arriving mid-group waits for the next group
//! (well under a second at the group sizes that matter).
//!
//! # Admission and cancellation
//!
//! [`ResidentModel::check_job`] refuses a request that can never run (an
//! option out of range, a form past its bounds, an image past the pixel bound)
//! at the door, before a device is claimed. The queue itself is bounded by the
//! transports' edge limit and admit deadline. A request cancelled before its
//! group starts takes no slot in it; one cancelled while the group decodes
//! stops costing decoder steps once every request of the group has stopped.

use capability::{ActionResult, Invocation, Manifest, Progress};
use mirage::caps::{MirageSession, MAX_BATCH, MODEL};
use mirage::model::Precision;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// Mirage behind the scheduler.
pub struct MirageResident {
    dir: String,
}

impl MirageResident {
    /// Whatever the model-store resolver finds for [`mirage::resolver::MirageSpec`]'s `weights` role.
    /// `None` (not served, never a startup failure) when the store holds no released checkpoint, or holds more than
    /// one and nothing says which.
    pub fn from_env(models_dir: Option<&std::path::Path>) -> Option<MirageResident> {
        let assembly = loader::served::served_assembly(models_dir, "mirage", &mirage::resolver::MirageSpec, &[])?;
        Self::from_assembly(&assembly)
    }

    /// From an already resolved assembly: its `weights` role is the checkpoint directory (or its weights file).
    pub fn from_assembly(assembly: &capability::Assembly) -> Option<MirageResident> {
        Self::new(crate::role_dir(assembly, "weights").ok()?)
    }

    /// Direct constructor for callers that already hold the directory.
    pub fn new(dir: impl Into<String>) -> Option<MirageResident> {
        let dir = dir.into();
        let d = std::path::Path::new(&dir);
        let missing: Vec<&str> = mirage::resolver::RELEASE_FILES.iter().copied().filter(|f| !d.join(f).is_file()).collect();
        if !missing.is_empty() {
            residency::log::not_served(&format!("brain: mirage not served ({dir} is missing {missing:?})"));
            return None;
        }
        Some(MirageResident { dir })
    }
}

/// Device bytes a served Mirage holds while hot: the weights, the dense matrices the mixers are folded into at
/// upload, and the activations of a full batch.
///
/// MEASURED on a CUDA card through the driver's per-process figure
/// (`mirage`'s `dropping_the_session_releases_device_memory`): the process held 564 MiB with only the CUDA context
/// and 948 MiB with the model loaded and a batch of 64 images described, so the model's own share is about 384 MiB;
/// 1 GiB rounds that up with room for a longer decode.
pub const DEVICE_BYTES: u64 = 1 << 30;

/// Host bytes: the host copy of the model (the oracle operators and the decoder the form scoring runs on) and the
/// decoder-memory cache (at most [`mirage::caps::MEMORY_CACHE_BYTES`]). MEASURED: the same process peaked at 1.26 GiB
/// resident after a batch of 64, the test's own goldens included; 2 GiB leaves room for a full cache.
pub const HOST_BYTES: u64 = 2 << 30;

impl ResidentModel for MirageResident {
    fn manifest(&self) -> Manifest {
        mirage::caps::manifest()
    }

    /// One instance per weight source: a request naming a `precision` is served by the instance holding exactly those
    /// weights (loaded on first use, the scheduler evicts it like any other), a request naming none by the
    /// directory's default.
    fn instance_key(&self, _action: &str, inv: &Invocation) -> InstanceKey {
        InstanceKey::new(MODEL, mirage::caps::precision_of(inv).ok().flatten().map_or("default", config_of))
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::new(DEVICE_BYTES, HOST_BYTES)
    }

    fn batch_limit(&self) -> Option<usize> {
        Some(MAX_BATCH)
    }

    fn check_job(&self, action: &str, inv: &Invocation) -> Result<(), String> {
        mirage::caps::check_request(action, inv)
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let gpu = crate::resident_llm::on_device(device, || gpu_core::Gpu::new(&MirageSession::kernels()))?;
        let session = match precision_of(&key.config) {
            Some(precision) => MirageSession::load_with(&self.dir, gpu, precision)?,
            None => MirageSession::load(&self.dir, gpu)?,
        };
        Ok(Box::new(MirageInstance { session }))
    }
}

/// The instance configuration naming `precision`; the inverse of [`precision_of`].
fn config_of(precision: Precision) -> &'static str {
    precision.name()
}

/// The precision an instance configuration asks for; `None` for the directory's default.
fn precision_of(config: &str) -> Option<Precision> {
    Precision::parse(config)
}

/// A resident Mirage.
struct MirageInstance {
    session: MirageSession,
}

impl Instance for MirageInstance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.session.run_batch(action, std::slice::from_ref(inv)).pop().expect("one result per request")
    }

    fn run_batch(&mut self, action: &str, invs: &[Invocation], _progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        self.session.run_batch(action, invs)
    }

    fn metrics(&self) -> Vec<(String, serde_json::Value)> {
        let stats = self.session.stats();
        let (entries, bytes) = self.session.cached_memories();
        vec![
            ("mirage_batches".to_string(), stats.batches.into()),
            ("mirage_images".to_string(), stats.images.into()),
            ("mirage_widest_batch".to_string(), stats.widest.into()),
            ("mirage_memory_hits".to_string(), stats.memory_hits.into()),
            ("mirage_cached_memories".to_string(), entries.into()),
            ("mirage_cached_memory_bytes".to_string(), bytes.into()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Each precision is its own instance, so a `cq` request is served by the packed archive and never by (or grouped
    /// with) the FP32 weights; a request naming none keeps the directory default.
    #[test]
    fn the_precision_option_selects_the_instance_and_its_weights() {
        let resident = MirageResident { dir: String::new() };
        let key = |precision: Option<&str>| {
            let inv = precision.map_or_else(Invocation::new, |p| Invocation::new().set("precision", json!(p)));
            resident.instance_key("describe", &inv)
        };
        assert_eq!(key(None).config, "default");
        assert_eq!(key(Some("fp32")).config, "fp32");
        assert_eq!(key(Some("cq")).config, "cq");
        assert_eq!(key(Some("cq-a8")).config, "cq-a8", "the int8 activation mode is its own instance, never grouped with cq");
        assert_eq!(precision_of("cq"), Some(Precision::Cq));
        assert_eq!(precision_of("cq-a8"), Some(Precision::CqA8));
        assert_eq!(precision_of("fp32"), Some(Precision::Fp32));
        assert_eq!(precision_of("default"), None);
    }
}
