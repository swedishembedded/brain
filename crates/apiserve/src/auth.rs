// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Key enforcement middleware. Access control is always on: every route behind
//! this layer needs the surface's key. Anthropic reads it from `x-api-key`;
//! OpenAI/OpenRouter read `Authorization: Bearer <key>`. A missing, blank, or
//! wrong key is a provider-shaped 401 (the route is never reached).

use axum::extract::{Request, State};
use axum::http::header::AUTHORIZATION;
use axum::middleware::Next;
use axum::response::Response;
use subtle::ConstantTimeEq;

use crate::error::ApiError;
use crate::state::AppState;
use crate::surface::Provider;

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
fn presented_key<'a>(provider: Provider, req: &'a Request) -> Option<&'a str> {
    let h = req.headers();
    match provider {
        Provider::Anthropic => h.get("x-api-key").and_then(|v| v.to_str().ok()).map(str::trim),
        Provider::OpenAI | Provider::OpenRouter => {
            h.get(AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|s| s.strip_prefix("Bearer ")).map(str::trim)
        }
    }
}

/// axum middleware: 401 unless the presented key matches this surface's key.
pub async fn require_key(State(state): State<AppState>, req: Request, next: Next) -> Result<Response, ApiError> {
    match presented_key(state.provider, &req) {
        Some(k) if !k.is_empty() && key_matches(k, &state.key) => Ok(next.run(req).await),
        _ => Err(ApiError::unauthorized(state.provider, "missing or invalid API key")),
    }
}
