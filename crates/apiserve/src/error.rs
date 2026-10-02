// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Provider-shaped API errors.
//!
//! One [`ApiError`] carries a [`Kind`] (which fixes the HTTP status + the canonical
//! error type name) and the target [`Provider`], and renders the body in that
//! provider's dialect:
//! - Anthropic: `{"type":"error","error":{"type":<t>,"message":<m>}}`
//! - OpenAI:    `{"error":{"message":<m>,"type":<t>,"param":null,"code":<c>}}`
//! - OpenRouter:`{"error":{"code":<http status int>,"message":<m>,"metadata":null}}`
//!
//! Each shape validates against the respective vendored OpenAPI error schema.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::surface::Provider;

/// The class of failure — determines the HTTP status and the per-dialect type name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Unauthorized,
    /// The caller is known but may not do this -- HTTP 403.
    Forbidden,
    /// The caller cannot pay for this call -- HTTP 402. Not retryable: asking
    /// again changes nothing until the balance does.
    PaymentRequired,
    /// The caller is over a limit of its own, as opposed to the server being
    /// over capacity ([`Kind::Overloaded`]) -- HTTP 429, carries a `Retry-After`.
    RateLimited,
    /// A fault on this side that is not the caller's doing and not a capacity
    /// shed -- HTTP 500.
    Internal,
    NotFound,
    ModelNotFound,
    InvalidRequest,
    /// The prompt (plus requested output) exceeds the model instance's
    /// serving capacity. Distinct from the generic `InvalidRequest` so a
    /// client can detect it programmatically (e.g. compact and retry)
    /// instead of treating it as an opaque failure - mirrors OpenAI's real
    /// `context_length_exceeded` error code. Same HTTP status and dialect
    /// `type` as `InvalidRequest`; only the OpenAI `code` differs.
    ContextLengthExceeded,
    NotImplemented,
    /// Accepted at the edge but could not be ADMITTED (started on a lane) within the
    /// admission deadline — HTTP 429, carries a `Retry-After`.
    Overloaded,
    /// Rejected at the edge before admission: the server's global concurrency limit
    /// was saturated and the request was load-shed — HTTP 503, carries a
    /// `Retry-After`.
    Saturated,
}

impl Kind {
    pub fn status(&self) -> StatusCode {
        match self {
            Kind::Unauthorized => StatusCode::UNAUTHORIZED,
            Kind::Forbidden => StatusCode::FORBIDDEN,
            Kind::PaymentRequired => StatusCode::PAYMENT_REQUIRED,
            Kind::RateLimited => StatusCode::TOO_MANY_REQUESTS,
            Kind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
            Kind::NotFound | Kind::ModelNotFound => StatusCode::NOT_FOUND,
            Kind::InvalidRequest | Kind::ContextLengthExceeded => StatusCode::BAD_REQUEST,
            Kind::NotImplemented => StatusCode::NOT_IMPLEMENTED,
            Kind::Overloaded => StatusCode::TOO_MANY_REQUESTS,
            Kind::Saturated => StatusCode::SERVICE_UNAVAILABLE,
        }
    }
    /// `true` when the client should back off and retry — surfaced as `Retry-After`.
    fn retryable(&self) -> bool {
        matches!(self, Kind::Overloaded | Kind::Saturated | Kind::RateLimited)
    }
    /// Anthropic's `error.type` value (one of its discriminated error variants).
    fn anthropic_type(&self) -> &'static str {
        match self {
            Kind::Unauthorized => "authentication_error",
            Kind::Forbidden => "permission_error",
            Kind::PaymentRequired => "billing_error",
            Kind::RateLimited => "rate_limit_error",
            Kind::Internal => "api_error",
            Kind::NotFound | Kind::ModelNotFound => "not_found_error",
            // Anthropic's error taxonomy has no distinct "context length"
            // type; it folds into invalid_request_error same as OpenAI's
            // dialect `type` (only OpenAI's `code` slug is distinct).
            Kind::InvalidRequest | Kind::ContextLengthExceeded => "invalid_request_error",
            // Anthropic has no "not_implemented"; api_error is its catch-all.
            Kind::NotImplemented => "api_error",
            Kind::Overloaded | Kind::Saturated => "overloaded_error",
        }
    }
    /// OpenAI's `error.type` value.
    fn openai_type(&self) -> &'static str {
        match self {
            Kind::Unauthorized => "authentication_error",
            Kind::Forbidden => "permission_error",
            Kind::PaymentRequired => "insufficient_quota",
            Kind::RateLimited => "rate_limit_exceeded",
            Kind::Internal => "server_error",
            Kind::NotFound | Kind::ModelNotFound | Kind::InvalidRequest | Kind::ContextLengthExceeded => {
                "invalid_request_error"
            }
            Kind::NotImplemented => "not_implemented",
            Kind::Overloaded => "rate_limit_exceeded",
            Kind::Saturated => "server_error",
        }
    }
    /// OpenAI's short `error.code` slug.
    fn openai_code(&self) -> &'static str {
        match self {
            Kind::Unauthorized => "invalid_api_key",
            Kind::Forbidden => "forbidden",
            Kind::PaymentRequired => "insufficient_quota",
            Kind::RateLimited => "rate_limit_exceeded",
            Kind::Internal => "server_error",
            Kind::NotFound => "not_found",
            Kind::ModelNotFound => "model_not_found",
            Kind::InvalidRequest => "invalid_request",
            // OpenAI's real, well-known code for this exact scenario.
            Kind::ContextLengthExceeded => "context_length_exceeded",
            Kind::NotImplemented => "not_implemented",
            Kind::Overloaded => "rate_limit_exceeded",
            Kind::Saturated => "server_error",
        }
    }
}

/// A provider-shaped error ready to become an axum [`Response`].
#[derive(Clone, Debug)]
pub struct ApiError {
    pub kind: Kind,
    pub provider: Provider,
    pub message: String,
}

impl ApiError {
    pub fn new(provider: Provider, kind: Kind, message: impl Into<String>) -> ApiError {
        ApiError { kind, provider, message: message.into() }
    }
    pub fn unauthorized(provider: Provider, message: impl Into<String>) -> ApiError {
        ApiError::new(provider, Kind::Unauthorized, message)
    }
    pub fn forbidden(provider: Provider, message: impl Into<String>) -> ApiError {
        ApiError::new(provider, Kind::Forbidden, message)
    }
    pub fn payment_required(provider: Provider, message: impl Into<String>) -> ApiError {
        ApiError::new(provider, Kind::PaymentRequired, message)
    }
    pub fn rate_limited(provider: Provider, message: impl Into<String>) -> ApiError {
        ApiError::new(provider, Kind::RateLimited, message)
    }
    pub fn internal(provider: Provider, message: impl Into<String>) -> ApiError {
        ApiError::new(provider, Kind::Internal, message)
    }
    pub fn not_found(provider: Provider, message: impl Into<String>) -> ApiError {
        ApiError::new(provider, Kind::NotFound, message)
    }
    pub fn model_not_found(provider: Provider, model: &str) -> ApiError {
        ApiError::new(provider, Kind::ModelNotFound, format!("model '{model}' not found"))
    }
    pub fn invalid_request(provider: Provider, message: impl Into<String>) -> ApiError {
        ApiError::new(provider, Kind::InvalidRequest, message)
    }
    pub fn context_length_exceeded(provider: Provider, message: impl Into<String>) -> ApiError {
        ApiError::new(provider, Kind::ContextLengthExceeded, message)
    }
    pub fn not_implemented(provider: Provider, message: impl Into<String>) -> ApiError {
        ApiError::new(provider, Kind::NotImplemented, message)
    }
    pub fn overloaded(provider: Provider, message: impl Into<String>) -> ApiError {
        ApiError::new(provider, Kind::Overloaded, message)
    }
    pub fn saturated(provider: Provider, message: impl Into<String>) -> ApiError {
        ApiError::new(provider, Kind::Saturated, message)
    }

    /// The provider-shaped JSON body (without the HTTP status).
    pub fn body(&self) -> Value {
        match self.provider {
            Provider::Anthropic => json!({
                "type": "error",
                "error": { "type": self.kind.anthropic_type(), "message": self.message },
            }),
            Provider::OpenAI => json!({
                "error": {
                    "message": self.message,
                    "type": self.kind.openai_type(),
                    "param": Value::Null,
                    "code": self.kind.openai_code(),
                },
            }),
            Provider::OpenRouter => json!({
                "error": {
                    "code": self.kind.status().as_u16(),
                    "message": self.message,
                    "metadata": Value::Null,
                },
            }),
        }
    }
}

/// Seconds advertised in `Retry-After` on a shed/overloaded response — a small,
/// fixed back-off hint for clients.
pub const RETRY_AFTER_SECS: u32 = 1;

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.kind.status();
        let retryable = self.kind.retryable();
        let mut resp = (status, Json(self.body())).into_response();
        if retryable {
            if let Ok(v) = axum::http::HeaderValue::from_str(&RETRY_AFTER_SECS.to_string()) {
                resp.headers_mut().insert(axum::http::header::RETRY_AFTER, v);
            }
        }
        resp
    }
}
