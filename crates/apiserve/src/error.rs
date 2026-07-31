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
    NotFound,
    ModelNotFound,
    InvalidRequest,
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
            Kind::NotFound | Kind::ModelNotFound => StatusCode::NOT_FOUND,
            Kind::InvalidRequest => StatusCode::BAD_REQUEST,
            Kind::NotImplemented => StatusCode::NOT_IMPLEMENTED,
            Kind::Overloaded => StatusCode::TOO_MANY_REQUESTS,
            Kind::Saturated => StatusCode::SERVICE_UNAVAILABLE,
        }
    }
    /// `true` when the client should back off and retry — surfaced as `Retry-After`.
    fn retryable(&self) -> bool {
        matches!(self, Kind::Overloaded | Kind::Saturated)
    }
    /// Anthropic's `error.type` value (one of its discriminated error variants).
    fn anthropic_type(&self) -> &'static str {
        match self {
            Kind::Unauthorized => "authentication_error",
            Kind::NotFound | Kind::ModelNotFound => "not_found_error",
            Kind::InvalidRequest => "invalid_request_error",
            // Anthropic has no "not_implemented"; api_error is its catch-all.
            Kind::NotImplemented => "api_error",
            Kind::Overloaded | Kind::Saturated => "overloaded_error",
        }
    }
    /// OpenAI's `error.type` value.
    fn openai_type(&self) -> &'static str {
        match self {
            Kind::Unauthorized => "authentication_error",
            Kind::NotFound | Kind::ModelNotFound | Kind::InvalidRequest => "invalid_request_error",
            Kind::NotImplemented => "not_implemented",
            Kind::Overloaded => "rate_limit_exceeded",
            Kind::Saturated => "server_error",
        }
    }
    /// OpenAI's short `error.code` slug.
    fn openai_code(&self) -> &'static str {
        match self {
            Kind::Unauthorized => "invalid_api_key",
            Kind::NotFound => "not_found",
            Kind::ModelNotFound => "model_not_found",
            Kind::InvalidRequest => "invalid_request",
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
    pub fn not_found(provider: Provider, message: impl Into<String>) -> ApiError {
        ApiError::new(provider, Kind::NotFound, message)
    }
    pub fn model_not_found(provider: Provider, model: &str) -> ApiError {
        ApiError::new(provider, Kind::ModelNotFound, format!("model '{model}' not found"))
    }
    pub fn invalid_request(provider: Provider, message: impl Into<String>) -> ApiError {
        ApiError::new(provider, Kind::InvalidRequest, message)
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
