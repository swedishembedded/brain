// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training services that share accelerators with
// the models being served, giving cards to a job and taking them back, for its
// clients. If your team needs expertise in scheduling fine-tuning beside
// production inference, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The training workload as a residency citizen.
//!
//! A fine-tune is not a model: it holds a card while it runs and nothing
//! after, and what it needs is known from its request before it starts. So
//! [`TrainResident`] registers as a [transient](residency::ResidentModel::transient)
//! model whose [`estimate`](residency::ResidentModel::estimate) is the run's
//! footprint ([`brain::ChatFineTune::footprint`]) and whose
//! [`check_job`](residency::ResidentModel::check_job) refuses at the door
//! what could never run, before any card is claimed. Placing it is then the
//! manager's ordinary work, and it is the policy this crate's docs state:
//!
//! * the job takes the device with the most free room that holds it; where
//!   none does, it evicts the least valuable **idle** serving instances on
//!   the best candidate card, whole replicas at a time, and rebuilds none of
//!   them itself (a replica is rebuilt by the next request for it, on a card
//!   with room);
//! * an instance a lane is running a job on is never evicted, so while
//!   serving is busy on every card that could hold the run it waits, and
//!   past the surface's admission deadline it is refused (HTTP 429, a D-Bus
//!   `error` frame) instead of queueing without bound or taking the memory
//!   serving is using - the caller retries;
//! * one run holds one device at a time (the executor's device lane), and a
//!   run needing more than the largest card holds is refused at the door;
//! * when the run ends, its instance is dropped and its reservation
//!   returned at once ([`residency::ResidentModel::transient`]).

use std::path::PathBuf;
use std::sync::Arc;

use capability::{ActionResult, Invocation, Manifest, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

use crate::consolidate::{self, FullRequest, MergeRequest};
use crate::dryrun;
use crate::job::{self, FootprintCache, JobEnv};
use crate::request::{self, Request};
use crate::store::{ArtifactStore, StoreError};

/// Everything the operator decides about the training service.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    /// The model store the bases are resolved in.
    pub models_dir: Option<PathBuf>,
    /// The artifact store datasets are fetched from and adapters are published to.
    pub store: Option<ArtifactStore>,
    /// Where each job keeps its private files.
    pub work_root: PathBuf,
    /// The memory in bytes of the largest card a run may be placed on; `None` on a host with no accelerator.
    pub largest_card_bytes: Option<u64>,
    /// How many runs may be in flight at once: one per device that can hold one.
    pub devices: usize,
}

/// The training workload. See the module doc.
pub struct TrainResident {
    env: JobEnv,
    largest_card_bytes: Option<u64>,
    devices: usize,
    footprints: FootprintCache,
}

/// The key config prefixes: the bytes of a run that holds a card, and of one that works in host memory.
const CARD_PREFIX: &str = "train:";
const HOST_PREFIX: &str = "host:";

impl TrainResident {
    pub fn new(config: TrainConfig) -> TrainResident {
        TrainResident { env: JobEnv { store: config.store, models_dir: config.models_dir, work_root: config.work_root }, largest_card_bytes: config.largest_card_bytes, devices: config.devices.max(1), footprints: FootprintCache::default() }
    }

    /// Everything about `inv` that can be known without a device: the request, the base, the artifact store it
    /// needs, and what it will reserve.
    fn admit(&self, action: &str, inv: &Invocation) -> Result<Reservation, String> {
        match action {
            request::ACTION => self.admit_train(inv),
            consolidate::FULL_ACTION => self.admit_full(inv),
            consolidate::MERGE_ACTION => self.admit_merge(inv),
            dryrun::ACTION => self.admit_check(inv),
            other => Err(format!("{} has no action '{other}'", request::MODEL)),
        }
    }

    fn admit_train(&self, inv: &Invocation) -> Result<Reservation, String> {
        let request = Request::from_invocation(inv)?;
        if self.env.store.is_none() {
            if !request.data.digests().is_empty() {
                return Err("train: this server has no artifact store (--artifact-store), so a dataset cannot be named by digest".to_string());
            }
            if !request.params.return_adapter {
                return Err("train: this server has no artifact store (--artifact-store), so the adapter can only be returned: set return_adapter".to_string());
            }
            if request.params.keep_evaluations {
                return Err("train: keep_evaluations publishes to the artifact store, and this server has none (--artifact-store)".to_string());
            }
        }
        self.check_digests(&request.data.digests())?;
        let base_dir = job::resolve_base(self.env.models_dir.as_deref(), &request.params.base)?;
        let footprint = self.footprints.of(&base_dir, &request, self.largest_card_bytes)?;
        self.fits_a_card("train", footprint.device_bytes, &footprint.reason, request.params.max_block)?;
        Ok(Reservation::Card(footprint.device_bytes))
    }

    fn admit_full(&self, inv: &Invocation) -> Result<Reservation, String> {
        let request = FullRequest::from_invocation(inv)?;
        let p = &request.params;
        if self.env.store.is_none() && p.install_as.is_none() {
            return Err("full_train: this server has no artifact store (--artifact-store) to publish the checkpoint to: set install_as to keep it in the model store".to_string());
        }
        self.check_digests(&request.data.digests())?;
        let base_dir = job::resolve_base(self.env.models_dir.as_deref(), &p.base)?;
        let mut ft = brain::ChatFullFineTune::from_pretrained(job::utf8(&base_dir)?);
        if let Some(arch) = &p.arch {
            ft = ft.arch(arch.as_str());
        }
        let footprint = ft.footprint(p.max_block).map_err(|e| format!("full_train: {e}"))?;
        self.fits_a_card("full_train", footprint.device_bytes, &footprint.reason, p.max_block)?;
        Ok(Reservation::Card(footprint.device_bytes))
    }

    fn admit_merge(&self, inv: &Invocation) -> Result<Reservation, String> {
        let request = MergeRequest::from_invocation(inv)?;
        if self.env.store.is_none() && request.params.install_as.is_none() {
            return Err("merge: this server has no artifact store (--artifact-store) to publish the checkpoint to: set install_as to keep it in the model store".to_string());
        }
        if let (None, crate::request::Source::Digest(_)) = (&self.env.store, &request.adapter) {
            return Err("merge: this server has no artifact store (--artifact-store), so an adapter cannot be named by digest".to_string());
        }
        if let crate::request::Source::Digest(digest) = &request.adapter {
            self.check_digests(&[*digest])?;
        }
        let base_dir = job::resolve_base(self.env.models_dir.as_deref(), &request.params.base)?;
        // Folding is host work: the base is read and a merged copy written, so about twice the base's own size in memory.
        Ok(Reservation::Host(2 * job::tree_bytes(&base_dir)))
    }

    fn admit_check(&self, inv: &Invocation) -> Result<Reservation, String> {
        let request = dryrun::Request::from_invocation(inv)?;
        match (&self.env.store, &request.dataset) {
            (None, crate::request::Source::Digest(_)) => return Err("check_dataset: this server has no artifact store (--artifact-store), so a dataset cannot be named by digest".to_string()),
            (_, crate::request::Source::Digest(digest)) => self.check_digests(&[*digest])?,
            _ => {}
        }
        job::resolve_base(self.env.models_dir.as_deref(), &request.params.base)?;
        // The tokenizer and the template are read, and the dataset is held while it is measured: host memory only.
        let dataset_bytes = match (&request.dataset, &self.env.store) {
            (crate::request::Source::Blob(name), _) => inv.blobs.get(name).map_or(0, |b| b.bytes.len() as u64),
            (crate::request::Source::Digest(d), Some(store)) => std::fs::metadata(store.root().join(d.hex())).map_or(0, |m| m.len()),
            _ => 0,
        };
        Ok(Reservation::Host((256u64 << 20) + dataset_bytes))
    }

    fn check_digests(&self, digests: &[crate::store::Digest]) -> Result<(), String> {
        if let Some(store) = &self.env.store {
            if let Some(missing) = digests.iter().find(|d| !store.contains(d)) {
                return Err(format!("train: {}", StoreError::Missing(*missing).client_message()));
            }
        }
        Ok(())
    }

    /// Refuses a run that needs more than the largest card holds.
    fn fits_a_card(&self, action: &str, bytes: u64, reason: &str, max_block: u32) -> Result<(), String> {
        let Some(card) = self.largest_card_bytes else { return Ok(()) };
        if bytes > card {
            let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
            return Err(format!(
                "{action}: this run needs about {:.0} GiB ({reason}; rows of up to {max_block} tokens) and the largest card has {:.0} GiB; lower max_block or hold the base narrower",
                gib(bytes),
                gib(card)
            ));
        }
        Ok(())
    }
}

/// What a run reserves: bytes on a card, or bytes of host memory for work that takes no card.
#[derive(Clone, Copy)]
enum Reservation {
    Card(u64),
    Host(u64),
}

impl ResidentModel for TrainResident {
    fn manifest(&self) -> Manifest {
        Manifest::new(request::MODEL, "fine-tune a chat model on the devices serving shares: a LoRA adapter (supervised, policy gradient or preference), every weight (`full_train`), or fold an adapter into its base (`merge`); datasets and results by blob or artifact digest, scheduled with the serving residents (see the trainjob crate)", vec![request::action_spec(), consolidate::full_spec(), consolidate::merge_spec(), dryrun::spec()])
    }

    fn instance_key(&self, action: &str, inv: &Invocation) -> InstanceKey {
        // `check_job` has already refused a job this cannot size; a model that vanished since reads as no room needed
        // and the run fails by name when it starts.
        let key = match self.admit(action, inv) {
            Ok(Reservation::Card(bytes)) => format!("{CARD_PREFIX}{bytes}"),
            Ok(Reservation::Host(bytes)) => format!("{HOST_PREFIX}{bytes}"),
            Err(_) => format!("{CARD_PREFIX}0"),
        };
        InstanceKey::new(request::MODEL, key)
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        let config = &key.base().config;
        let bytes = |prefix: &str| config.strip_prefix(prefix).and_then(|b| b.parse().ok());
        match (bytes(CARD_PREFIX), bytes(HOST_PREFIX)) {
            (_, Some(host)) => MemCost::new(0, host),
            (Some(card), None) => MemCost::new(card, 0),
            (None, None) => MemCost::new(0, 0),
        }
    }

    fn replicas(&self) -> usize {
        self.devices
    }

    fn check_job(&self, action: &str, inv: &Invocation) -> Result<(), String> {
        self.admit(action, inv).map(drop)
    }

    fn transient(&self) -> bool {
        true
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(TrainInstance { env: self.env.clone(), device }))
    }
}

/// A run's claim on a device. It holds no memory itself: the trainer builds its model inside [`Instance::run`], on
/// the device the manager placed it on, and frees it before the run returns.
struct TrainInstance {
    env: JobEnv,
    device: Device,
}

impl Instance for TrainInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let env = &self.env;
        let work: &mut dyn FnMut() -> ActionResult = match action {
            request::ACTION => &mut || job::run(env, inv, progress),
            consolidate::FULL_ACTION => &mut || consolidate::run_full(env, inv, progress),
            consolidate::MERGE_ACTION => &mut || consolidate::run_merge(env, inv),
            dryrun::ACTION => &mut || dryrun::run(env, inv),
            other => return Err(format!("{} has no action '{other}'", request::MODEL)),
        };
        // The trainer reports a device failure (an allocation the card cannot hold) as a panic. One run's failure is
        // that run's error, not the server's.
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| catalog::resident_llm::on_device(self.device, work)));
        match run {
            Ok(placed) => placed?,
            Err(panic) => {
                let why = panic.downcast_ref::<String>().cloned().or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_string())).unwrap_or_default();
                eprintln!("brain train: the run panicked: {why}");
                Err("train: the device failed during the run (see the server log)".to_string())
            }
        }
    }
}

/// The service for `config`, as the executor registers it.
pub fn resident(config: TrainConfig) -> Arc<dyn ResidentModel> {
    Arc::new(TrainResident::new(config))
}
