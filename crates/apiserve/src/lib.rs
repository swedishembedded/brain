// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Provider-shaped HTTP surfaces that let brain act as a backend for external
//! agents. brain speaks three dialects — **Anthropic Messages**, **OpenAI**, and
//! **OpenRouter** — each on its own socket, each behind its own key.
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
pub mod bridge;
pub mod catalog;
pub mod error;
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

/// The edge concurrency ceiling: at most this many chat/embeddings/image requests
/// are admitted to the router at once. The overflow is load-shed (503) rather than
/// queued, so a saturated server sheds fast instead of building unbounded latency.
/// Well above the executor's lane count — this guards the HTTP edge, not the lanes.
pub const EDGE_CONCURRENCY: usize = 256;

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

/// The admission deadline for live servers: `BRAIN_ADMIT_DEADLINE_MS` if set to a
/// positive integer, else [`DEFAULT_ADMIT_DEADLINE`]. An empty/invalid/zero value
/// falls back to the default (never an unbounded or zero-length wait).
fn admit_deadline_from_env() -> std::time::Duration {
    parse_admit_deadline(std::env::var("BRAIN_ADMIT_DEADLINE_MS").ok().as_deref())
}

/// Pure parse of the admit-deadline override: a positive integer of milliseconds, or
/// [`DEFAULT_ADMIT_DEADLINE`] for any missing/empty/invalid/zero value.
fn parse_admit_deadline(raw: Option<&str>) -> std::time::Duration {
    match raw.and_then(|v| v.trim().parse::<u64>().ok()).filter(|&ms| ms > 0) {
        Some(ms) => std::time::Duration::from_millis(ms),
        None => DEFAULT_ADMIT_DEADLINE,
    }
}

#[cfg(test)]
mod deadline_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn admit_deadline_override_parses_positive_ms_else_default() {
        assert_eq!(parse_admit_deadline(Some("500")), Duration::from_millis(500));
        assert_eq!(parse_admit_deadline(Some("  250 ")), Duration::from_millis(250));
        assert_eq!(parse_admit_deadline(None), DEFAULT_ADMIT_DEADLINE);
        assert_eq!(parse_admit_deadline(Some("")), DEFAULT_ADMIT_DEADLINE);
        assert_eq!(parse_admit_deadline(Some("0")), DEFAULT_ADMIT_DEADLINE);
        assert_eq!(parse_admit_deadline(Some("nope")), DEFAULT_ADMIT_DEADLINE);
    }
}

/// The shared 404 for any unrouted path, in the surface's dialect.
async fn fallback(State(state): State<AppState>) -> ApiError {
    ApiError::not_found(state.provider, "no such route")
}

/// Serve every [`Surface`] over the ONE shared [`Executor`] until interrupted.
/// Builds a single multi-threaded Tokio runtime and one axum router per surface
/// (its own key + provider), binds each `Surface.addr`, and serves them
/// concurrently on a [`tokio::task::JoinSet`]. Blocks the calling thread.
///
/// Installs its own SIGINT/SIGTERM handling via [`brain_shutdown`]. If HTTP is the
/// only serving surface in the process, that is exactly right; when it runs
/// alongside D-Bus (`brain serve --dbus --openai ...`), use
/// [`serve_all_with_shutdown`] instead so both surfaces share one shutdown
/// source — see that function's docs for why.
pub fn serve_all(exec: Executor, surfaces: Vec<Surface>) -> anyhow::Result<()> {
    serve_all_with_shutdown(exec, surfaces, brain_shutdown::Shutdown::from_signals())
}

/// Like [`serve_all_with_shutdown`], but every surface's [`AppState`] gets
/// `supplier` attached ([`AppState::with_supplier`]) so an unresolved model
/// auto-fetches instead of 404ing. `None` is identical to
/// [`serve_all_with_shutdown`] (today's no-auto-fetch behavior).
pub fn serve_all_with_shutdown_and_supplier(
    exec: Executor,
    surfaces: Vec<Surface>,
    shutdown: brain_shutdown::Shutdown,
    supplier: Option<std::sync::Arc<dyn residency::ModelSupplier>>,
) -> anyhow::Result<()> {
    serve_all_inner(exec, surfaces, shutdown, supplier)
}

/// Serve every [`Surface`] until `shutdown` fires. Identical to [`serve_all`]
/// except the caller supplies the shutdown signal — the shape needed when HTTP
/// runs alongside D-Bus: `tokio::signal::ctrl_c()` claims a process-wide
/// disposition, so if each surface independently registered its own handler,
/// only one would ever actually see the signal (in practice: whichever runtime
/// happened to register first, which for the combined case was the D-Bus
/// thread — and that runtime was the one that used to deadlock in
/// `Connection::graceful_shutdown()`, so Ctrl-C did nothing at all). Sharing one
/// `Shutdown` — installed once, in `crates/cli/src/run_cli.rs::run_apis` — fixes
/// both problems: exactly one registration, and every surface reacts to it.
///
/// Each surface is served with axum's `with_graceful_shutdown`, so in-flight
/// requests drain instead of being cut off mid-response. The drain has no upper
/// bound of its own here; `run_apis` bounds the overall wait so a slow client
/// cannot hold the process open indefinitely.
pub fn serve_all_with_shutdown(exec: Executor, surfaces: Vec<Surface>, shutdown: brain_shutdown::Shutdown) -> anyhow::Result<()> {
    serve_all_inner(exec, surfaces, shutdown, None)
}

fn serve_all_inner(
    exec: Executor,
    surfaces: Vec<Surface>,
    shutdown: brain_shutdown::Shutdown,
    supplier: Option<std::sync::Arc<dyn residency::ModelSupplier>>,
) -> anyhow::Result<()> {
    // The admission deadline is operator-overridable via `BRAIN_ADMIT_DEADLINE_MS`
    // (default `DEFAULT_ADMIT_DEADLINE`, 10s). Keeping it here (rather than in the
    // pure `router`) leaves the unit-test `oneshot` path on the fixed default while
    // letting a live server be driven with a short deadline (e.g. the conformance
    // harness sets 500ms to force fast 429 shedding).
    let admit_deadline = admit_deadline_from_env();
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let mut set = tokio::task::JoinSet::new();
        for s in surfaces {
            let state = AppState::new(exec.clone(), s.api_key.clone(), s.provider).with_admit_deadline(admit_deadline).with_supplier(supplier.clone());
            let app = router(state);
            let (addr, provider) = (s.addr, s.provider);
            let sd = shutdown.clone();
            set.spawn(async move {
                let listener = tokio::net::TcpListener::bind(addr)
                    .await
                    .map_err(|e| anyhow::anyhow!("bind {addr} ({provider}): {e}"))?;
                eprintln!("brain apiserve: {provider} on http://{addr}");
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
