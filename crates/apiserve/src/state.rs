// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The shared handler state: the executor every request submits to, the guarding
//! key, this surface's provider dialect, and the in-flight job registry.
//!
//! Cloned per request by axum (all fields are cheap `Arc`/`Copy`/`String` clones).
//! The [`JobRegistry`] mirrors `crates/dbus`: armed [`capability::CancelToken`]s
//! keyed by a request id, live from submission until the reply fires. Actively
//! used — `bridge::CancelGuard` fires a registered token when a streaming SSE
//! response is dropped (client disconnect), and the admit-deadline race
//! (`bridge::submit`/`stream_inner`) cancels on a 429/saturated shed.

use std::sync::Arc;
use std::time::Duration;

use capability::CancelToken;
use residency::{Executor, ModelSupplier};
use uuid::Uuid;

use crate::surface::Provider;

/// Default admission deadline — see `residency::admission`'s doc (shared with
/// `crates/dbus` so both transports gate identically).
pub use residency::admission::DEFAULT_ADMIT_DEADLINE;

/// Armed cancel tokens for in-flight requests, keyed by a per-request UUID —
/// `residency::jobs::JobRegistry`, the SAME shared type `crates/dbus`'s
/// `Manager` uses (keyed by `u64` there instead).
pub type JobRegistry = residency::jobs::JobRegistry<Uuid>;

/// Everything a request handler needs. Clone is cheap.
#[derive(Clone)]
pub struct AppState {
    pub exec: Executor,
    pub jobs: JobRegistry,
    pub key: String,
    pub provider: Provider,
    /// Bounded wait for a request to be ADMITTED (work started on a lane) before it
    /// is shed with a 429. Overridable so tests can use a short deadline.
    pub admit_deadline: Duration,
    /// Classifies/fetches a `model` string that isn't already resident (transparent
    /// auto-fetch). `None` — the default, and every test's `AppState::new` — means
    /// an unresolved model is a plain 404 with zero I/O, exactly today's behavior.
    /// Set only by `run_apis` (a live `brain serve`), via [`AppState::with_supplier`].
    pub supplier: Option<Arc<dyn ModelSupplier>>,
}

impl AppState {
    pub fn new(exec: Executor, key: impl Into<String>, provider: Provider) -> AppState {
        AppState { exec, jobs: JobRegistry::new(), key: key.into(), provider, admit_deadline: DEFAULT_ADMIT_DEADLINE, supplier: None }
    }

    /// Override the admission deadline (builder-style). Used by tests to force fast
    /// shedding without slow real-time waits.
    pub fn with_admit_deadline(mut self, deadline: Duration) -> AppState {
        self.admit_deadline = deadline;
        self
    }

    /// Attach a model supplier (builder-style) so an unresolved model auto-fetches
    /// instead of 404ing. `None` restores today's no-auto-fetch behavior.
    pub fn with_supplier(mut self, supplier: Option<Arc<dyn ModelSupplier>>) -> AppState {
        self.supplier = supplier;
        self
    }

    /// Arm and register a fresh cancel token under a new request id. The handler
    /// puts the token in its [`capability::Invocation`] and removes the entry
    /// (via [`AppState::finish`]) when the reply fires.
    pub fn register(&self) -> (Uuid, CancelToken) {
        let id = Uuid::new_v4();
        let token = CancelToken::armed();
        self.jobs.insert(id, token.clone());
        (id, token)
    }

    /// Drop a finished request's cancel token.
    pub fn finish(&self, id: &Uuid) {
        self.jobs.remove(id);
    }

    /// Requests currently in flight.
    pub fn active(&self) -> usize {
        self.jobs.len()
    }
}
