// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving each new version of a policy the moment its training job
// produces it, in the same process, for its clients. If your team needs expertise in online learning where the
// trainer and the server share accelerators then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The serving adapter bank a training job publishes to: a streaming policy-gradient job offers each new adapter under
//! one id as that id's next policy version ([`residency::AdapterIdentity::version`]), in process, so the next request
//! for the id samples from it.
//!
//! The bank belongs to the served bases, which the server builds after the training workload is registered; the
//! workload holds a [`BankSlot`] the server fills once they exist ([`BankSlot::fill`]). A job that asks to publish on a
//! server whose slot is empty is refused by name. Publishing reads the file once, binds it to the served base its card
//! names and verifies it there exactly as `--serve-adapters` does
//! ([`llmserve::adapter_release::verify_adapter_among`]), then offers it ([`LiveAdapters::serve`]); the call returns
//! once the registry lists the new release, the point from which a request for the id is routed to it.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use llmserve::adapter_registry::{AdapterHost, LiveAdapters};
use llmserve::adapter_release::{verify_adapter_among, LoadedAdapter, ServedBase};
use residency::{AdapterIdentity, Executor};

/// Where a job offers the adapters it publishes.
pub trait PolicyBank: Send + Sync {
    /// Offer the adapter in `file`, whose card id must be `id`, as the next release of `id`; the identity it is served
    /// under once a request for `id` is routed to it.
    fn publish(&self, id: &str, file: &Path) -> Result<AdapterIdentity, String>;

    /// The release served under `id` now, if one is.
    fn identity(&self, id: &str) -> Option<AdapterIdentity>;

    /// [`Self::publish`], and also serve the release under an id of its own for its version
    /// ([`version_id`]), so a client that started on it can stay on it; the `retain` newest version ids of `id` stay
    /// served, older ones are withdrawn (requests already accepted for them finish). Returns the identity and the
    /// version id; a bank that cannot keep versions serves none (`None`).
    fn publish_retaining(&self, id: &str, file: &Path, retain: usize) -> Result<(AdapterIdentity, Option<String>), String> {
        let _ = retain;
        Ok((self.publish(id, file)?, None))
    }
}

/// The id version `version` of the adapter published under `id` is served under when its versions are retained:
/// `id` with `-v<version>` added to its tag (`org/repo:owner:name:latest-v7`), never the published id itself.
pub fn version_id(id: &str, version: u64) -> String {
    format!("{id}-v{version}")
}

/// The bank of a server, filled once its served bases exist. Cloned into the training workload when it is built.
#[derive(Clone, Default)]
pub struct BankSlot(Arc<OnceLock<Arc<dyn PolicyBank>>>);

impl std::fmt::Debug for BankSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.get().is_some() { "BankSlot(filled)" } else { "BankSlot(empty)" })
    }
}

impl BankSlot {
    /// Give the workload its bank; a slot is filled once (a second bank is refused).
    pub fn fill(&self, bank: Arc<dyn PolicyBank>) -> Result<(), String> {
        self.0.set(bank).map_err(|_| "the training workload's adapter bank is set already".to_string())
    }

    pub fn get(&self) -> Option<&Arc<dyn PolicyBank>> {
        self.0.get()
    }
}

/// The adapter banks of a server's served bases: each adapter rides on the base its card names.
pub struct ServingBank {
    hosts: Vec<(ServedBase, LiveAdapters)>,
    /// The version ids served for each published id, oldest first.
    retained: Mutex<HashMap<String, VecDeque<(usize, String)>>>,
}

impl ServingBank {
    pub fn new(hosts: Vec<(ServedBase, LiveAdapters)>) -> ServingBank {
        ServingBank { hosts, retained: Mutex::new(HashMap::new()) }
    }

    /// `resident`'s base and its live adapters on `executor`, the executor it is registered with: one host of a bank.
    pub fn host<R: AdapterHost>(resident: Arc<R>, executor: Executor) -> (ServedBase, LiveAdapters) {
        (resident.served_base(), LiveAdapters::new(resident, executor))
    }
}

impl ServingBank {
    /// Verifies `file` as a release of `id` and offers it; returns the host it rides on, what was loaded and the
    /// identity now served.
    fn offer(&self, id: &str, file: &Path) -> Result<(usize, LoadedAdapter, AdapterIdentity), String> {
        let bases: Vec<ServedBase> = self.hosts.iter().map(|(base, _)| base.clone()).collect();
        let (host, loaded) = verify_adapter_among(file, &bases).map_err(|e| format!("publishing adapter {id}: {e}"))?;
        if loaded.release.id != id {
            return Err(format!("publishing adapter {id}: the file's card names {}", loaded.release.id));
        }
        let live = &self.hosts[host].1;
        live.serve(loaded.clone()).map_err(|e| format!("publishing adapter {id}: {e}"))?;
        let identity = live.identity(id).ok_or_else(|| format!("publishing adapter {id}: the bank does not list it after offering it"))?;
        Ok((host, loaded, identity))
    }
}

impl PolicyBank for ServingBank {
    fn publish_retaining(&self, id: &str, file: &Path, retain: usize) -> Result<(AdapterIdentity, Option<String>), String> {
        let (host, mut loaded, identity) = self.offer(id, file)?;
        if retain == 0 {
            return Ok((identity, None));
        }
        let versioned = version_id(id, identity.version);
        loaded.version = identity.version;
        loaded.lineage = identity.lineage.clone();
        let live = &self.hosts[host].1;
        let mut retained = self.retained.lock().unwrap_or_else(|e| e.into_inner());
        let kept = retained.entry(id.to_string()).or_default();
        // The same weights published again keep their version, which is served already.
        if live.identity(&versioned).is_none() {
            live.serve_version(loaded, &versioned).map_err(|e| format!("publishing adapter {id} as {versioned}: {e}"))?;
            kept.push_back((host, versioned.clone()));
        }
        while kept.len() > retain {
            if let Some((host, old)) = kept.pop_front() {
                self.hosts[host].1.retire(&old);
            }
        }
        Ok((identity, Some(versioned)))
    }

    fn publish(&self, id: &str, file: &Path) -> Result<AdapterIdentity, String> {
        self.offer(id, file).map(|(_, _, identity)| identity)
    }

    fn identity(&self, id: &str) -> Option<AdapterIdentity> {
        self.hosts.iter().find_map(|(_, live)| live.identity(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_version_id_is_the_published_id_with_the_version_for_its_tag() {
        assert_eq!(version_id("org/online:run:4f2a9c1e:latest", 7), "org/online:run:4f2a9c1e:latest-v7");
        assert_ne!(version_id("Qwen/Qwen3-0.6B:lab:canary:v1", 1), "Qwen/Qwen3-0.6B:lab:canary:v1");
        assert!(brain_modelref::ModelRef::parse(&version_id("Qwen/Qwen3-0.6B:lab:canary:v1", 12)).is_ok_and(|r| r.adapter().is_some()));
    }
}
