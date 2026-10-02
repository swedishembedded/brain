// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Who is calling. Access control is always on: every route behind this layer
//! needs the surface's [`Authenticator`] to accept the request. The default,
//! [`StaticKey`], is the surface's one key (Anthropic reads it from `x-api-key`,
//! OpenAI/OpenRouter from `Authorization: Bearer <key>`); an embedding
//! application replaces it to tell its callers apart. A missing, blank, or wrong
//! key is a provider-shaped 401 and the route is never reached.

use std::any::Any;
use std::sync::Arc;

use axum::extract::{Extension, Request, State};
use axum::http::header::AUTHORIZATION;
use axum::http::HeaderMap;
use axum::middleware::Next;
use axum::response::Response;
use subtle::ConstantTimeEq;

use crate::error::ApiError;
use crate::state::AppState;
use crate::surface::Provider;

/// Whoever an [`Authenticator`] says a request is from. Opaque here: the
/// embedder puts what it needs in it and downcasts it back in its
/// [`RequestHooks`](crate::RequestHooks).
pub type Principal = Arc<dyn Any + Send + Sync>;

/// The caller a route handler receives from the authentication middleware.
pub type CallerExt = Option<Extension<Principal>>;

/// Decides who a request is from, or refuses it.
pub trait Authenticator: Send + Sync {
    /// Identifies the caller from the request's headers.
    ///
    /// # Errors
    /// A provider-shaped refusal (401, 403, 429...), sent to the caller as is.
    fn authenticate(&self, provider: Provider, headers: &HeaderMap) -> Result<Principal, ApiError>;

    /// What separates this caller's background jobs from everyone else's: a
    /// job is only ever visible to the scope that started it. The default is
    /// one scope for everybody, right for a surface with a single key.
    fn scope(&self, _principal: &Principal) -> String {
        String::new()
    }
}

/// The default [`Authenticator`]: one key for the whole surface.
pub struct StaticKey(String);

impl StaticKey {
    pub fn new(key: impl Into<String>) -> StaticKey {
        StaticKey(key.into())
    }
}

/// Constant-time key check: `true` only when the presented key equals the surface's
/// key. Uses [`subtle::ConstantTimeEq`] so acceptance/rejection time does not depend
/// on how many leading bytes matched — closing the timing side channel a plain `==`
/// (which short-circuits at the first differing byte) would open. The keys are a
/// fixed-length `sk-brain-<hex>` format, so the length is not itself a secret; the
/// length-mismatch fast path in `ct_eq` therefore leaks nothing meaningful.
fn key_matches(presented: &str, expected: &str) -> bool {
    presented.as_bytes().ct_eq(expected.as_bytes()).into()
}

/// Pull the presented key out of the request headers in this provider's scheme.
pub fn presented_key(provider: Provider, headers: &HeaderMap) -> Option<&str> {
    match provider {
        Provider::Anthropic => headers.get("x-api-key").and_then(|v| v.to_str().ok()).map(str::trim),
        Provider::OpenAI | Provider::OpenRouter => {
            headers.get(AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|s| s.strip_prefix("Bearer ")).map(str::trim)
        }
    }
}

impl Authenticator for StaticKey {
    fn authenticate(&self, provider: Provider, headers: &HeaderMap) -> Result<Principal, ApiError> {
        match presented_key(provider, headers) {
            Some(k) if !k.is_empty() && key_matches(k, &self.0) => Ok(Arc::new(())),
            _ => Err(ApiError::unauthorized(provider, "missing or invalid API key")),
        }
    }
}

/// axum middleware: refuses the request unless the surface's [`Authenticator`]
/// accepts it, and hands the caller it identified to the handler.
pub async fn authenticate(State(state): State<AppState>, mut req: Request, next: Next) -> Result<Response, ApiError> {
    let principal = state.authenticator.authenticate(state.provider, req.headers())?;
    req.extensions_mut().insert(principal);
    Ok(next.run(req).await)
}
