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

/// What a named caller may do. Each role includes the ones before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    /// Run actions that do not train, on models already served.
    Inference,
    /// Also run training actions ([`capability::ActionSpec::training`]), when the
    /// server serves training at all (`--training`).
    Training,
    /// Also make the server fetch a model it does not hold, when auto-fetch is on.
    Admin,
}

impl Role {
    /// The role named `name` (`inference`, `training`, `admin`).
    pub fn parse(name: &str) -> Option<Role> {
        match name {
            "inference" => Some(Role::Inference),
            "training" => Some(Role::Training),
            "admin" => Some(Role::Admin),
            _ => None,
        }
    }
}

/// One named caller as the operator declares it.
#[derive(Clone, Debug)]
pub struct CallerSpec {
    /// `[a-z0-9_-]{1,64}`; separates the caller's jobs from everyone else's.
    pub name: String,
    pub role: Role,
    /// The caller's own key.
    pub key: String,
    /// How many of its calls may be in flight at once, background jobs
    /// included: a caller that is compromised can hold this share of the
    /// server and no more.
    pub max_in_flight: usize,
}

/// A caller the [`Callers`] authenticator let in: the [`Principal`] a request
/// from it carries.
#[derive(Debug)]
pub struct Caller {
    pub name: String,
    pub role: Role,
    max_in_flight: usize,
    in_flight: std::sync::atomic::AtomicUsize,
}

impl Caller {
    /// The caller a request carries, when it came through [`Callers`]; `None`
    /// for a surface with one static key, whose holder the operator's flags
    /// alone constrain.
    pub fn of(principal: Option<&Principal>) -> Option<&Caller> {
        principal.and_then(|p| p.downcast_ref::<Caller>())
    }

    /// Take one of the caller's in-flight slots, or `None` when all are taken.
    pub(crate) fn begin(self: &Arc<Caller>) -> Option<InFlight> {
        use std::sync::atomic::Ordering;
        let taken = self.in_flight.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| (n < self.max_in_flight).then_some(n + 1));
        taken.ok().map(|_| InFlight(self.clone()))
    }
}

/// One in-flight call of a [`Caller`], given back when dropped.
pub(crate) struct InFlight(Arc<Caller>);

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// The [`Authenticator`] for several named callers, each with its own key, role
/// and in-flight budget (`brain serve --caller`). A caller's jobs are its own:
/// its name is its job scope.
pub struct Callers(Vec<(String, Arc<Caller>)>);

impl Callers {
    /// The callers in `specs`.
    ///
    /// # Errors
    ///
    /// No caller, a name that is not `[a-z0-9_-]{1,64}` or appears twice, an
    /// empty key or one two callers share, or a budget of zero.
    pub fn new(specs: Vec<CallerSpec>) -> Result<Callers, String> {
        if specs.is_empty() {
            return Err("no caller is declared".into());
        }
        let mut out: Vec<(String, Arc<Caller>)> = Vec::with_capacity(specs.len());
        for spec in specs {
            if spec.name.is_empty() || spec.name.len() > 64 || !spec.name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-') {
                return Err(format!("caller name {:?} is not 1-64 of [a-z0-9_-]", spec.name));
            }
            if spec.key.is_empty() {
                return Err(format!("caller {} has an empty key", spec.name));
            }
            if spec.max_in_flight == 0 {
                return Err(format!("caller {} may have nothing in flight", spec.name));
            }
            if out.iter().any(|(_, c)| c.name == spec.name) {
                return Err(format!("caller {} is declared twice", spec.name));
            }
            if out.iter().any(|(key, _)| key_matches(key, &spec.key)) {
                return Err(format!("caller {} shares its key with another caller", spec.name));
            }
            let caller = Caller { name: spec.name, role: spec.role, max_in_flight: spec.max_in_flight, in_flight: Default::default() };
            out.push((spec.key, Arc::new(caller)));
        }
        Ok(Callers(out))
    }
}

impl Authenticator for Callers {
    fn authenticate(&self, provider: Provider, headers: &HeaderMap) -> Result<Principal, ApiError> {
        let presented = presented_key(provider, headers).filter(|k| !k.is_empty());
        // Every key is compared, whichever matches: the time taken says
        // nothing about which caller, if any, a key belongs to.
        let mut found = None;
        for (key, caller) in &self.0 {
            if presented.is_some_and(|k| key_matches(k, key)) {
                found = Some(caller.clone());
            }
        }
        match found {
            Some(caller) => Ok(caller),
            None => Err(ApiError::unauthorized(provider, "missing or invalid API key")),
        }
    }

    fn scope(&self, principal: &Principal) -> String {
        Caller::of(Some(principal)).map(|c| c.name.clone()).unwrap_or_default()
    }
}

/// axum middleware: refuses the request unless the surface's [`Authenticator`]
/// accepts it, and hands the caller it identified to the handler.
pub async fn authenticate(State(state): State<AppState>, mut req: Request, next: Next) -> Result<Response, ApiError> {
    let principal = state.authenticator.authenticate(state.provider, req.headers())?;
    req.extensions_mut().insert(principal);
    Ok(next.run(req).await)
}
