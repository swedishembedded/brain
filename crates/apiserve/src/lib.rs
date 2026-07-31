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
//! - [`anthropic`]/[`openai`]/[`openrouter`] — the per-dialect routes: real chat
//!   (non-stream + SSE token streaming); embeddings/images still 501.
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
pub mod state;
pub mod surface;

use axum::error_handling::HandleErrorLayer;
use axum::extract::State;
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
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// The shared 404 for any unrouted path, in the surface's dialect.
async fn fallback(State(state): State<AppState>) -> ApiError {
    ApiError::not_found(state.provider, "no such route")
}

/// Serve every [`Surface`] over the ONE shared [`Executor`] until interrupted.
/// Builds a single multi-threaded Tokio runtime and one axum router per surface
/// (its own key + provider), binds each `Surface.addr`, and serves them
/// concurrently on a [`tokio::task::JoinSet`]. Blocks the calling thread. (In P4
/// this is unit-testable but not yet called from the `brain` binary — the CLI
/// wiring lands in a later phase.)
pub fn serve_all(exec: Executor, surfaces: Vec<Surface>) -> anyhow::Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let mut set = tokio::task::JoinSet::new();
        for s in surfaces {
            let state = AppState::new(exec.clone(), s.api_key.clone(), s.provider);
            let app = router(state);
            let (addr, provider) = (s.addr, s.provider);
            set.spawn(async move {
                let listener = tokio::net::TcpListener::bind(addr)
                    .await
                    .map_err(|e| anyhow::anyhow!("bind {addr} ({provider}): {e}"))?;
                eprintln!("brain apiserve: {provider} on http://{addr}");
                axum::serve(listener, app).await.map_err(|e| anyhow::anyhow!("serve {provider}: {e}"))?;
                Ok::<(), anyhow::Error>(())
            });
        }
        while let Some(res) = set.join_next().await {
            res??;
        }
        Ok(())
    })
}
