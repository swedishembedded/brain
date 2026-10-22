// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Provider-shaped HTTP surfaces that let brain act as a backend for external
//! agents. brain speaks three dialects — **Anthropic Messages**, **OpenAI**, and
//! **OpenRouter** — each on its own socket, each behind its own key.
//!
//! Swedish Embedded AB implements self-hosted, API-compatible inference
//! endpoints for teams that want their existing agents and tooling pointed at
//! hardware they control instead of at a vendor. If your team needs expertise
//! in standing up a production inference API - auth, admission control,
//! streaming, cancellation and the security posture that has to hold when the
//! socket is reachable - you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Layering (kept thin — no model code here, only `residency`/`capability`):
//! - [`surface`] — one `(provider, addr, key)` binding + key generation/exposure.
//! - [`state`] — the shared handler state (executor, key, provider, job registry).
//! - [`auth`] - who is calling: the [`Authenticator`] seam, its static-key default
//!   (Anthropic `x-api-key`, others `Bearer`), and the middleware that applies it.
//! - [`capabilities`] - routes that reach every model class: the catalogue, `run`,
//!   and background jobs for work too long to hold a request open.
//! - [`systemone`] - `POST /v1/systemone`: typed decisions in the System One wire
//!   shape, a thin layer over the `decide` capability of whichever decision model
//!   answers.
//! - [`hooks`] - the [`RequestHooks`] seam: a say in whether a call runs, and a
//!   record of what it cost, settled exactly once.
//! - [`error`] — provider-shaped error bodies.
//! - [`catalog`] — deriving API capabilities from a manifest; the `/models` filter.
//! - [`models`] — the `/models` catalog handlers.
//! - [`bridge`] — the async→sync seam to the shared executor (submit/stream + the
//!   cancel-on-disconnect guard) that the chat handlers dispatch through.
//! - [`png`] — a minimal dependency-free PNG encoder for `images/generations`.
//! - [`anthropic`]/[`openai`]/[`openrouter`] — the per-dialect routes: real chat
//!   (non-stream + SSE token streaming), embeddings, and image generation.
//!
//! Every path submits [`residency::Job`]s to the ONE shared [`residency::Executor`],
//! so scheduling/residency/batching stay uniform across the D-Bus surface, the CLI,
//! and these HTTP surfaces.

pub mod anthropic;
pub mod auth;
pub mod b64;
pub mod bridge;
pub mod capabilities;
pub mod catalog;
pub mod error;
pub mod hooks;
pub mod media;
pub mod models;
pub mod openai;
pub mod openrouter;
pub mod png;
pub mod sampling;
pub mod state;
pub mod surface;
pub mod systemone;

use axum::error_handling::HandleErrorLayer;
use axum::extract::{DefaultBodyLimit, State};
use axum::response::IntoResponse;
use axum::Router;
use tower::limit::GlobalConcurrencyLimitLayer;
use tower::load_shed::LoadShedLayer;
use tower::{BoxError, ServiceBuilder};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

pub use auth::{Authenticator, CallerExt, Principal, StaticKey};
pub use catalog::{api_caps, CapSet};
pub use error::{ApiError, Kind};
pub use hooks::{Call, CallResult, RequestHooks, Ticket};
pub use state::{AppState, JobRegistry, DEFAULT_ADMIT_DEADLINE};
pub use surface::{write_keys, Provider, Surface};

use residency::Executor;

/// The edge concurrency ceiling — see `residency::admission`'s doc (shared
/// with `crates/dbus` so both transports gate identically).
pub use residency::admission::EDGE_CONCURRENCY;

/// The maximum accepted request-body size (8 MiB). Sized well above the largest
/// legitimate body (a chat/embeddings request, or a batch of `input` strings) yet
/// bounded so a single huge body cannot be buffered into an OOM. A body over this
/// limit is rejected with `413 Payload Too Large` before any handler runs. This is
/// an explicit ceiling rather than relying on axum's smaller implicit default.
pub const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;

/// Build the axum [`Router`] for one provider surface — the shared `/models`
/// routes plus this provider's dialect routes, all behind the key-enforcing
/// middleware (+ trace/cors). Exposed so tests can drive a provider with
/// `tower::ServiceExt::oneshot`, without binding a socket.
pub fn router(state: AppState) -> Router {
    let provider = state.provider;
    let provider_routes = match state.provider {
        Provider::Anthropic => anthropic::routes(),
        Provider::OpenAI => openai::routes(),
        Provider::OpenRouter => openrouter::routes(),
    };

    // Edge backpressure on the chat/embeddings/image routes only (NOT `/models`):
    // a global concurrency limit + load-shed. When `EDGE_CONCURRENCY` requests are
    // already in flight, the overflow is shed IMMEDIATELY as a 503 ("saturated,
    // rejected at the edge") — a distinct signal from the admit-timeout 429
    // ("accepted, but couldn't start a lane in time"). `HandleErrorLayer` turns the
    // shed's `Overloaded` error back into a normal (Infallible) provider-shaped
    // response so axum's routing contract holds. The layers sit INSIDE auth/trace/
    // cors (applied below) and leave the `oneshot` test path untouched.
    let edge = ServiceBuilder::new()
        .layer(HandleErrorLayer::new(move |_err: BoxError| async move {
            ApiError::saturated(provider, "server saturated: request rejected at the edge").into_response()
        }))
        .layer(LoadShedLayer::new())
        .layer(GlobalConcurrencyLimitLayer::new(EDGE_CONCURRENCY))
        .into_inner();

    Router::new()
        .merge(models::routes())
        .merge(provider_routes.merge(capabilities::routes()).merge(systemone::routes()).layer(edge))
        .fallback(fallback)
        // Auth wraps everything (incl. the fallback): an unauthenticated caller
        // never learns which routes exist.
        .layer(axum::middleware::from_fn_with_state(state.clone(), auth::authenticate))
        // Bound the request body BEFORE it is buffered by any handler's `Bytes`
        // extractor: a body over `MAX_BODY_BYTES` is a 413, not an OOM.
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// The shared 404 for any unrouted path, in the surface's dialect.
async fn fallback(State(state): State<AppState>) -> ApiError {
    ApiError::not_found(state.provider, "no such route")
}

/// Everything [`serve_all`] needs beyond the executor and the surfaces.
///
/// One options struct rather than a `serve_all_with_shutdown_and_supplier…`-style
/// ladder: the previous three-rung ladder had two rungs with zero callers anywhere
/// in the workspace, and every new cross-cutting concern (a shutdown source, an
/// auto-fetch supplier, a readiness gate) was adding another rung and another name.
#[derive(Default)]
pub struct ServeOpts {
    /// The shared shutdown source. `None` installs a private one via
    /// [`brain_shutdown::Shutdown::from_signals`] — right when HTTP is the only
    /// serving surface in the process, WRONG when it runs alongside D-Bus
    /// (`brain serve --dbus --openai ...`): SIGINT/SIGTERM disposition is
    /// process-wide, so if each surface independently registered its own handler,
    /// only one would ever actually see the signal (in practice: whichever runtime
    /// happened to register first, which for the combined case was the D-Bus
    /// thread — and that runtime was the one that used to deadlock in
    /// `Connection::graceful_shutdown()`, so Ctrl-C did nothing at all).
    /// `crates/cli/src/run_cli.rs::run_apis` installs one shared source and passes
    /// it to both surfaces, which fixes both problems: exactly one registration,
    /// and every surface reacts to it.
    ///
    /// Each surface is served with axum's `with_graceful_shutdown`, so in-flight
    /// requests drain instead of being cut off mid-response. The jobs behind
    /// them are cancelled when the signal fires, and the drain is bounded by
    /// [`SHUTDOWN_DRAIN`] so a slow client or an uninterruptible step cannot
    /// hold the process open.
    pub shutdown: Option<brain_shutdown::Shutdown>,
    /// Attached to every surface's [`AppState`] ([`AppState::with_supplier`]) so
    /// an unresolved model auto-fetches instead of 404ing. `None` disables
    /// auto-fetch (today's default behavior with no supplier).
    pub supplier: Option<std::sync::Arc<dyn residency::ModelSupplier>>,
    /// Notified once per surface, immediately after its `TcpListener` binds (and
    /// before it starts serving). Default is [`brain_shutdown::ready::Gate::disabled`]
    /// (a no-op) — see that type's docs for why a marker file, not an HTTP route,
    /// is how `brain serve` reports "every requested surface is up".
    pub ready: brain_shutdown::ready::Gate,
    /// How long a request may wait to be admitted before it is shed with a 429.
    /// Defaults to [`residency::admission::Deadlines::default`]; `brain serve`
    /// passes the operator's `--admit-deadline-ms` values.
    pub deadlines: residency::admission::Deadlines,
    /// Where to publish the connection record ([`surface::record_json`]) once
    /// every surface has bound, removed again when serving ends. `None`
    /// publishes nothing.
    pub record: Option<std::path::PathBuf>,
}

impl ServeOpts {
    pub fn new() -> ServeOpts {
        ServeOpts::default()
    }
    pub fn with_shutdown(mut self, shutdown: brain_shutdown::Shutdown) -> ServeOpts {
        self.shutdown = Some(shutdown);
        self
    }
    pub fn with_supplier(mut self, supplier: Option<std::sync::Arc<dyn residency::ModelSupplier>>) -> ServeOpts {
        self.supplier = supplier;
        self
    }
    pub fn with_ready(mut self, ready: brain_shutdown::ready::Gate) -> ServeOpts {
        self.ready = ready;
        self
    }
    pub fn with_deadlines(mut self, deadlines: residency::admission::Deadlines) -> ServeOpts {
        self.deadlines = deadlines;
        self
    }
    pub fn with_record(mut self, record: Option<std::path::PathBuf>) -> ServeOpts {
        self.record = record;
        self
    }
}

/// The addresses every surface actually bound, gathered as each listener
/// comes up; the last one to bind writes the connection record, so a client
/// that finds the record finds every listener already accepting.
struct Publication {
    path: std::path::PathBuf,
    expected: usize,
    bound: std::sync::Mutex<Vec<(Provider, std::net::SocketAddr, String)>>,
}

impl Publication {
    fn bound(&self, provider: Provider, addr: std::net::SocketAddr, key: &str) {
        let mut bound = self.bound.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        bound.push((provider, addr, key.to_string()));
        if bound.len() != self.expected {
            return;
        }
        let body = surface::record_json(std::process::id(), &bound).to_string();
        if let Err(e) = surface::write_private(&self.path, body.as_bytes()) {
            eprintln!("brain serve: the connection record {} could not be written ({e}); local clients will not find this server", self.path.display());
        }
    }
}


/// How long a stopping server waits for in-flight requests to finish after the
/// shutdown fires and their jobs are cancelled. Cancellation is polled between
/// decode steps, so a healthy job ends well inside it; a step that cannot be
/// interrupted must not hold the process, and the GPUs behind it, any longer.
pub const SHUTDOWN_DRAIN: std::time::Duration = std::time::Duration::from_secs(3);

/// Serve every [`Surface`] over the ONE shared [`Executor`] until `opts.shutdown`
/// fires (or, with `opts.shutdown: None`, until this process's own Ctrl-C/SIGTERM).
/// Builds a single multi-threaded Tokio runtime and one axum router per surface
/// (its own key + provider), binds each `Surface.addr`, and serves them
/// concurrently on a [`tokio::task::JoinSet`]. Blocks the calling thread.
pub fn serve_all(exec: Executor, surfaces: Vec<Surface>, opts: ServeOpts) -> anyhow::Result<()> {
    let shutdown = opts.shutdown.unwrap_or_else(brain_shutdown::Shutdown::from_signals);
    let ServeOpts { supplier, ready, deadlines, record, .. } = opts;
    let publication = record.and_then(|path| {
        if let Err(e) = path.parent().map_or(Ok(()), surface::private_dir) {
            eprintln!("brain serve: no connection record is published ({e}); local clients will not find this server");
            return None;
        }
        Some(std::sync::Arc::new(Publication { path, expected: surfaces.len(), bound: std::sync::Mutex::new(Vec::new()) }))
    });
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let published = publication.clone();
    let result = rt.block_on(async move {
        let mut set = tokio::task::JoinSet::new();
        for s in surfaces {
            let state = AppState::new(exec.clone(), s.api_key.clone(), s.provider)
                .with_admit_deadline(deadlines.admit)
                .with_cold_build_admit_deadline(deadlines.cold_build)
                .with_supplier(supplier.clone());
            // Graceful shutdown waits for every in-flight request, and a
            // generation answers only when it ends: cancel them all the moment
            // shutdown fires so they end at their next poll.
            let jobs = state.jobs.clone();
            let cancel_on = shutdown.clone();
            tokio::spawn(async move {
                cancel_on.wait().await;
                jobs.cancel_all();
            });
            let app = router(state);
            let (addr, provider, key) = (s.addr, s.provider, s.api_key);
            let sd = shutdown.clone();
            let ready = ready.clone();
            let publication = published.clone();
            set.spawn(async move {
                let listener = tokio::net::TcpListener::bind(addr)
                    .await
                    .map_err(|e| anyhow::anyhow!("bind {addr} ({provider}): {e}"))?;
                let addr = listener.local_addr().unwrap_or(addr);
                eprintln!("brain apiserve: {provider} on http://{addr}");
                if let Some(publication) = &publication {
                    publication.bound(provider, addr, &key);
                }
                // AFTER the bind and after the line every existing harness greps
                // for, and BEFORE serving: the marker means "the listener existed",
                // nothing stronger.
                ready.bound(&provider.to_string());
                axum::serve(listener, app)
                    .with_graceful_shutdown(async move { sd.wait().await })
                    .await
                    .map_err(|e| anyhow::anyhow!("serve {provider}: {e}"))?;
                Ok::<(), anyhow::Error>(())
            });
        }
        // The drain bound starts when shutdown fires; whatever has not finished
        // by then is abandoned.
        let drain_expired = async {
            shutdown.wait().await;
            tokio::time::sleep(SHUTDOWN_DRAIN).await;
        };
        tokio::pin!(drain_expired);
        loop {
            tokio::select! {
                joined = set.join_next() => match joined {
                    Some(res) => res??,
                    None => break,
                },
                _ = &mut drain_expired => {
                    set.abort_all();
                    break;
                }
            }
        }
        Ok(())
    });
    // Not a plain drop: that waits for every blocking task, and a request
    // abandoned at the drain bound may still hold one.
    rt.shutdown_background();
    // Serving has ended: withdraw the record before the caller exits. Not
    // left to a drop, since an abandoned task may still hold a handle to it.
    if let Some(publication) = &publication {
        std::fs::remove_file(&publication.path).ok();
    }
    result
}
