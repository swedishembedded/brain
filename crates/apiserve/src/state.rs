// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The shared handler state: the executor every request submits to, the guarding
//! key, this surface's provider dialect, and the in-flight job registry.
//!
//! Cloned per request by axum (all fields are cheap `Arc`/`Copy`/`String` clones).
//! The [`JobRegistry`] mirrors `crates/dbus`: armed [`capability::CancelToken`]s
//! keyed by a request id, so a future `cancel`/client-disconnect path (P5+) can
//! abort a running generation. It is unused by the P4 skeletons but wired now so
//! the chat/stream handlers can register jobs without a state change.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use capability::CancelToken;
use residency::{Executor, ModelSupplier};
use uuid::Uuid;

use crate::surface::Provider;

/// Default admission deadline: the bounded time a request may wait for work to
/// START on a lane before it is shed with a 429. A running job may then take much
/// longer — only the wait-to-start is bounded.
pub const DEFAULT_ADMIT_DEADLINE: Duration = Duration::from_secs(10);

/// Armed cancel tokens for in-flight requests, keyed by a per-request UUID. An
/// entry lives from submission until the reply fires (copy of the dbus pattern).
pub type JobRegistry = Arc<Mutex<HashMap<Uuid, CancelToken>>>;

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
        AppState {
            exec,
            jobs: Arc::new(Mutex::new(HashMap::new())),
            key: key.into(),
            provider,
            admit_deadline: DEFAULT_ADMIT_DEADLINE,
            supplier: None,
        }
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
        if let Ok(mut m) = self.jobs.lock() {
            m.insert(id, token.clone());
        }
        (id, token)
    }

    /// Drop a finished request's cancel token.
    pub fn finish(&self, id: &Uuid) {
        if let Ok(mut m) = self.jobs.lock() {
            m.remove(id);
        }
    }

    /// Requests currently in flight.
    pub fn active(&self) -> usize {
        self.jobs.lock().map(|m| m.len()).unwrap_or(0)
    }
}
