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
use residency::{DeviceInventory, Executor, ModelSupplier};
use uuid::Uuid;

use crate::auth::{Authenticator, CallerExt, Principal, StaticKey};
use crate::capabilities::AsyncJobs;
use crate::hooks::RequestHooks;
use crate::surface::Provider;
use crate::video::VideoJobs;

/// Default admission deadline — see `residency::admission`'s doc (shared with
/// `crates/dbus` so both transports gate identically).
pub use residency::admission::{DEFAULT_ADMIT_DEADLINE, DEFAULT_COLD_BUILD_ADMIT_DEADLINE};

/// Armed cancel tokens for in-flight requests, keyed by a per-request UUID —
/// `residency::jobs::JobRegistry`, the SAME shared type `crates/dbus`'s
/// `Manager` uses (keyed by `u64` there instead).
pub type JobRegistry = residency::jobs::JobRegistry<Uuid>;

/// Everything a request handler needs. Clone is cheap.
#[derive(Clone)]
pub struct AppState {
    pub exec: Executor,
    pub jobs: JobRegistry,
    /// The background jobs started through `/v1/jobs`.
    pub async_jobs: Arc<AsyncJobs>,
    /// The video jobs started through `/videos`, and their spool.
    pub videos: Arc<VideoJobs>,
    /// Who may call this surface. [`StaticKey`] unless an embedder replaced it.
    pub authenticator: Arc<dyn Authenticator>,
    /// What the embedder wants a say in on every call; `None` for a surface
    /// that serves everything it is asked.
    pub hooks: Option<Arc<dyn RequestHooks>>,
    /// Who the request being handled is from. Set per request by
    /// [`AppState::scoped`]; `None` on the router's own copy.
    pub caller: Option<Principal>,
    pub provider: Provider,
    /// Bounded wait for a request to be ADMITTED (work started on a lane) before it
    /// is shed with a 429. Overridable so tests can use a short deadline.
    pub admit_deadline: Duration,
    /// Bounded wait applied INSTEAD of `admit_deadline` once a request has been
    /// queued past it AND `residency::admission::model_is_cold_building` says its
    /// own model is already running (its first cold activation, almost always) —
    /// see that function's doc. Genuine cross-model contention still sheds at
    /// `admit_deadline`.
    pub cold_build_admit_deadline: Duration,
    /// Classifies/fetches a `model` string that isn't already resident (transparent
    /// auto-fetch). `None` — the default, and every test's `AppState::new` — means
    /// an unresolved model is a plain 404 with zero I/O, exactly today's behavior.
    /// Set only by `run_apis` (a live `brain serve`), via [`AppState::with_supplier`].
    pub supplier: Option<Arc<dyn ModelSupplier>>,
    /// Whether this surface runs training actions ([`capability::ActionSpec::training`]).
    /// Off unless the operator started the server with `--training`: a client
    /// that can call the API must not be able to occupy the cards with a
    /// training run on that alone.
    pub training: bool,
    /// What the host knows of its devices beyond what the executor budgets ([`DeviceInventory`]); `None` reports
    /// the budgets alone.
    pub devices: Option<Arc<dyn DeviceInventory>>,
}

impl AppState {
    pub fn new(exec: Executor, key: impl Into<String>, provider: Provider) -> AppState {
        AppState {
            exec,
            jobs: JobRegistry::new(),
            async_jobs: Arc::default(),
            videos: Arc::default(),
            authenticator: Arc::new(StaticKey::new(key)),
            hooks: None,
            caller: None,
            provider,
            admit_deadline: DEFAULT_ADMIT_DEADLINE,
            cold_build_admit_deadline: DEFAULT_COLD_BUILD_ADMIT_DEADLINE,
            supplier: None,
            training: false,
            devices: None,
        }
    }

    /// Replace the surface's static key with the embedder's own way of telling
    /// callers apart (builder-style).
    pub fn with_authenticator(mut self, authenticator: Arc<dyn Authenticator>) -> AppState {
        self.authenticator = authenticator;
        self
    }

    /// Let this surface run training actions (builder-style); see [`Self::training`].
    pub fn with_training(mut self, training: bool) -> AppState {
        self.training = training;
        self
    }

    /// Give the embedder a say in every call, and a record of its cost
    /// (builder-style).
    pub fn with_hooks(mut self, hooks: Arc<dyn RequestHooks>) -> AppState {
        self.hooks = Some(hooks);
        self
    }

    /// This state as one request sees it: the same shared parts, plus who the
    /// authentication middleware let in. Cheap -- every field is an `Arc` or a
    /// small value.
    pub fn scoped(&self, caller: CallerExt) -> AppState {
        AppState { caller: caller.map(|axum::Extension(principal)| principal), ..self.clone() }
    }

    /// The supplier this request may make fetch a model: the surface's, unless
    /// the request is from a named caller below [`crate::Role::Admin`]. Fetching
    /// downloads gigabytes and fills the store, which is the operator's to
    /// allow, not every client's. Every fetch path asks here, never
    /// `self.supplier` itself.
    pub fn supplier_for_caller(&self) -> Option<Arc<dyn ModelSupplier>> {
        let below_admin = crate::auth::Caller::of(self.caller.as_ref()).is_some_and(|c| c.role < crate::Role::Admin);
        if below_admin {
            return None;
        }
        self.supplier.clone()
    }

    /// What separates this request's caller from every other: its job scope
    /// ([`crate::Authenticator::scope`]), empty for an unnamed caller. Stamped
    /// on every invocation this surface dispatches (`Invocation::scope`).
    pub fn scope(&self) -> String {
        self.caller.as_ref().map(|p| self.authenticator.scope(p)).unwrap_or_default()
    }

    /// Whether the hooks list `model` on this surface.
    pub fn lists(&self, model: &str) -> bool {
        self.hooks.as_ref().is_none_or(|hooks| hooks.listed(model))
    }

    /// Keep video jobs' clips in `spool`, at most `bytes` of them
    /// (builder-style). The default is a private directory under the system's
    /// temporary directory and [`crate::video::DEFAULT_SPOOL_BYTES`].
    pub fn with_video_spool(mut self, spool: std::path::PathBuf, bytes: u64) -> AppState {
        self.videos = Arc::new(VideoJobs::new(spool, bytes));
        self
    }

    /// Keep video jobs and their clips in `dir` across restarts, at most
    /// `bytes` of clips (builder-style): [`VideoJobs::durable`].
    pub fn with_durable_video_jobs(self, dir: std::path::PathBuf, bytes: u64) -> Result<AppState, String> {
        let jobs = VideoJobs::durable(dir, bytes, self.provider)?;
        Ok(self.with_video_jobs(Arc::new(jobs)))
    }

    /// Serve `/videos` from `jobs`, a store other surfaces of the same dialect
    /// may share (builder-style).
    pub fn with_video_jobs(mut self, jobs: Arc<VideoJobs>) -> AppState {
        self.videos = jobs;
        self
    }

    /// Override the admission deadline (builder-style). Used by tests to force fast
    /// shedding without slow real-time waits.
    pub fn with_admit_deadline(mut self, deadline: Duration) -> AppState {
        self.admit_deadline = deadline;
        self
    }

    /// Override the cold-build admission deadline (builder-style) — see the field's
    /// doc. Used by tests to force fast shedding without slow real-time waits.
    pub fn with_cold_build_admit_deadline(mut self, deadline: Duration) -> AppState {
        self.cold_build_admit_deadline = deadline;
        self
    }

    /// Both admission deadlines as the value the admission race takes.
    pub fn deadlines(&self) -> residency::admission::Deadlines {
        residency::admission::Deadlines { admit: self.admit_deadline, cold_build: self.cold_build_admit_deadline }
    }

    /// Describe the devices beyond their budgets (builder-style); see [`Self::devices`].
    pub fn with_devices(mut self, devices: Option<Arc<dyn DeviceInventory>>) -> AppState {
        self.devices = devices;
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
