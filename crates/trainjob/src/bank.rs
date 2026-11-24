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

use std::path::Path;
use std::sync::{Arc, OnceLock};

use llmserve::adapter_registry::{AdapterHost, LiveAdapters};
use llmserve::adapter_release::{verify_adapter_among, ServedBase};
use residency::{AdapterIdentity, Executor};

/// Where a job offers the adapters it publishes.
pub trait PolicyBank: Send + Sync {
    /// Offer the adapter in `file`, whose card id must be `id`, as the next release of `id`; the identity it is served
    /// under once a request for `id` is routed to it.
    fn publish(&self, id: &str, file: &Path) -> Result<AdapterIdentity, String>;

    /// The release served under `id` now, if one is.
    fn identity(&self, id: &str) -> Option<AdapterIdentity>;
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
}

impl ServingBank {
    pub fn new(hosts: Vec<(ServedBase, LiveAdapters)>) -> ServingBank {
        ServingBank { hosts }
    }

    /// `resident`'s base and its live adapters on `executor`, the executor it is registered with: one host of a bank.
    pub fn host<R: AdapterHost>(resident: Arc<R>, executor: Executor) -> (ServedBase, LiveAdapters) {
        (resident.served_base(), LiveAdapters::new(resident, executor))
    }
}

impl PolicyBank for ServingBank {
    fn publish(&self, id: &str, file: &Path) -> Result<AdapterIdentity, String> {
        let bases: Vec<ServedBase> = self.hosts.iter().map(|(base, _)| base.clone()).collect();
        let (host, loaded) = verify_adapter_among(file, &bases).map_err(|e| format!("publishing adapter {id}: {e}"))?;
        if loaded.release.id != id {
            return Err(format!("publishing adapter {id}: the file's card names {}", loaded.release.id));
        }
        let live = &self.hosts[host].1;
        live.serve(loaded).map_err(|e| format!("publishing adapter {id}: {e}"))?;
        live.identity(id).ok_or_else(|| format!("publishing adapter {id}: the bank does not list it after offering it"))
    }

    fn identity(&self, id: &str) -> Option<AdapterIdentity> {
        self.hosts.iter().find_map(|(_, live)| live.identity(id))
    }
}
