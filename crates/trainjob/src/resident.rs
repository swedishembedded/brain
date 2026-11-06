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

/// The key config prefix: the footprint in bytes follows.
const KEY_PREFIX: &str = "train:";

impl TrainResident {
    pub fn new(config: TrainConfig) -> TrainResident {
        TrainResident { env: JobEnv { store: config.store, models_dir: config.models_dir, work_root: config.work_root }, largest_card_bytes: config.largest_card_bytes, devices: config.devices.max(1), footprints: FootprintCache::default() }
    }

    /// Everything about `inv` that can be known without a device: the request, the base, the artifact store it
    /// needs, and the bytes it will reserve.
    fn admit(&self, action: &str, inv: &Invocation) -> Result<u64, String> {
        if action != request::ACTION {
            return Err(format!("{} has no action '{action}'", request::MODEL));
        }
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
        if let Some(store) = &self.env.store {
            if let Some(missing) = request.data.digests().into_iter().find(|d| !store.contains(d)) {
                return Err(format!("train: {}", StoreError::Missing(missing).client_message()));
            }
        }
        let base_dir = job::resolve_base(self.env.models_dir.as_deref(), &request.params.base)?;
        let footprint = self.footprints.of(&base_dir, &request, self.largest_card_bytes)?;
        if let Some(card) = self.largest_card_bytes {
            if footprint.device_bytes > card {
                let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
                return Err(format!(
                    "train: this run needs about {:.0} GiB ({}; rows of up to {} tokens) and the largest card has {:.0} GiB; lower max_block or hold the base narrower (base_format)",
                    gib(footprint.device_bytes),
                    footprint.reason,
                    request.params.max_block,
                    gib(card)
                ));
            }
        }
        Ok(footprint.device_bytes)
    }
}

impl ResidentModel for TrainResident {
    fn manifest(&self) -> Manifest {
        Manifest::new(request::MODEL, "fine-tune a chat model's LoRA adapter on the devices serving shares: supervised or by policy gradient, datasets and adapter by blob or artifact digest, scheduled with the serving residents (see the trainjob crate)", vec![request::action_spec()])
    }

    fn instance_key(&self, action: &str, inv: &Invocation) -> InstanceKey {
        // `check_job` has already refused a job this cannot size; a model that vanished since reads as no room needed
        // and the run fails by name when it starts.
        let bytes = self.admit(action, inv).unwrap_or(0);
        InstanceKey::new(request::MODEL, format!("{KEY_PREFIX}{bytes}"))
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        let bytes = key.base().config.strip_prefix(KEY_PREFIX).and_then(|b| b.parse().ok()).unwrap_or(0);
        MemCost::new(bytes, 0)
    }

    fn replicas(&self) -> usize {
        self.devices
    }

    fn check_job(&self, action: &str, inv: &Invocation) -> Result<(), String> {
        self.admit(action, inv).map(|_| ())
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
        if action != request::ACTION {
            return Err(format!("{} has no action '{action}'", request::MODEL));
        }
        let env = &self.env;
        // The trainer reports a device failure (an allocation the card cannot hold) as a panic. One run's failure is
        // that run's error, not the server's.
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| catalog::resident_llm::on_device(self.device, || job::run(env, inv, progress))));
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
