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
//! - [`anthropic`]/[`openai`]/[`openrouter`] — the per-dialect routes (P4: 501 stubs).
//!
//! Every path submits [`residency::Job`]s to the ONE shared [`residency::Executor`],
//! so scheduling/residency/batching stay uniform across the D-Bus surface, the CLI,
//! and these HTTP surfaces.

pub mod anthropic;
pub mod auth;
pub mod catalog;
pub mod error;
pub mod models;
pub mod openai;
pub mod openrouter;
pub mod state;
pub mod surface;

use axum::extract::State;
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

pub use catalog::{api_caps, CapSet};
pub use error::{ApiError, Kind};
pub use state::{AppState, JobRegistry};
pub use surface::{Provider, Surface};

use residency::Executor;

/// Build the axum [`Router`] for one provider surface — the shared `/models`
/// routes plus this provider's dialect routes, all behind the key-enforcing
/// middleware (+ trace/cors). Exposed so tests can drive a provider with
/// `tower::ServiceExt::oneshot`, without binding a socket.
pub fn router(state: AppState) -> Router {
    let provider_routes = match state.provider {
        Provider::Anthropic => anthropic::routes(),
        Provider::OpenAI => openai::routes(),
        Provider::OpenRouter => openrouter::routes(),
    };
    Router::new()
        .merge(models::routes())
        .merge(provider_routes)
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
