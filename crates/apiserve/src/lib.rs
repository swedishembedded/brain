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
//! - [`auth`] — the key-enforcing middleware (Anthropic `x-api-key`, others `Bearer`).
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
pub mod catalog;
pub mod error;
pub mod media;
pub mod models;
pub mod openai;
pub mod openrouter;
pub mod png;
pub mod state;
pub mod surface;

use axum::error_handling::HandleErrorLayer;
use axum::extract::{DefaultBodyLimit, State};
use axum::response::IntoResponse;
use axum::Router;
use tower::limit::GlobalConcurrencyLimitLayer;
use tower::load_shed::LoadShedLayer;
use tower::{BoxError, ServiceBuilder};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

pub use catalog::{api_caps, CapSet};
pub use error::{ApiError, Kind};
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
        .merge(provider_routes.layer(edge))
        .fallback(fallback)
        // Auth wraps everything (incl. the fallback): an unauthenticated caller
        // never learns which routes exist.
        .layer(axum::middleware::from_fn_with_state(state.clone(), auth::require_key))
        // Bound the request body BEFORE it is buffered by any handler's `Bytes`
        // extractor: a body over `MAX_BODY_BYTES` is a 413, not an OOM.
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// The admission deadline for live servers — see `residency::admission`'s doc.
fn admit_deadline_from_env() -> std::time::Duration {
    residency::admission::admit_deadline_from_env()
}

/// The cold-build admission deadline for live servers — see `residency::admission`'s doc.
fn cold_build_admit_deadline_from_env() -> std::time::Duration {
    residency::admission::cold_build_admit_deadline_from_env()
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
    /// requests drain instead of being cut off mid-response. The drain has no
    /// upper bound of its own here; `run_apis` bounds the overall wait so a slow
    /// client cannot hold the process open indefinitely.
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
}

/// Serve every [`Surface`] over the ONE shared [`Executor`] until `opts.shutdown`
/// fires (or, with `opts.shutdown: None`, until this process's own Ctrl-C/SIGTERM).
/// Builds a single multi-threaded Tokio runtime and one axum router per surface
/// (its own key + provider), binds each `Surface.addr`, and serves them
/// concurrently on a [`tokio::task::JoinSet`]. Blocks the calling thread.
pub fn serve_all(exec: Executor, surfaces: Vec<Surface>, opts: ServeOpts) -> anyhow::Result<()> {
    let shutdown = opts.shutdown.unwrap_or_else(brain_shutdown::Shutdown::from_signals);
    let ServeOpts { supplier, ready, .. } = opts;
    // The admission deadline is operator-overridable via `BRAIN_ADMIT_DEADLINE_MS`
    // (default `DEFAULT_ADMIT_DEADLINE`, 10s). Keeping it here (rather than in the
    // pure `router`) leaves the unit-test `oneshot` path on the fixed default while
    // letting a live server be driven with a short deadline (e.g. the conformance
    // harness sets 500ms to force fast 429 shedding).
    let admit_deadline = admit_deadline_from_env();
    let cold_build_admit_deadline = cold_build_admit_deadline_from_env();
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let mut set = tokio::task::JoinSet::new();
        for s in surfaces {
            let state = AppState::new(exec.clone(), s.api_key.clone(), s.provider)
                .with_admit_deadline(admit_deadline)
                .with_cold_build_admit_deadline(cold_build_admit_deadline)
                .with_supplier(supplier.clone());
            let app = router(state);
            let (addr, provider) = (s.addr, s.provider);
            let sd = shutdown.clone();
            let ready = ready.clone();
            set.spawn(async move {
                let listener = tokio::net::TcpListener::bind(addr)
                    .await
                    .map_err(|e| anyhow::anyhow!("bind {addr} ({provider}): {e}"))?;
                eprintln!("brain apiserve: {provider} on http://{addr}");
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
        while let Some(res) = set.join_next().await {
            res??;
        }
        Ok(())
    })
}

