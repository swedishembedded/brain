// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`Error`]: this crate's one error type.
//!
//! Hand-rolled (manual `Display` + `impl std::error::Error`) rather than
//! `thiserror`, for the same reason `brain_modelstore::plan::PlanError`
//! (`crates/modelstore/src/plan.rs`) is: an SDK facade's error surface is
//! part of its PUBLIC API, and a hand-rolled enum keeps that surface exactly
//! what this crate chooses to expose, with no derive-macro-generated trait
//! bound leaking a dependency's own error type into callers who never
//! imported it themselves. The two structured variants
//! ([`Error::Ambiguous`], [`Error::Missing`]) are boxed for the same reason
//! `PlanError` boxes its own wide variants: they carry a
//! [`brain_modelstore::resolve::Ambiguity`]/[`brain_modelstore::resolve::Missing`]
//! in full (every candidate, every missing role's near-misses) rather than
//! flattening the resolver's own structured answer to a bare string, which
//! would throw away exactly the information a caller needs to act on it
//! (which flag to pass, which file is missing).
//!
//! ## Error hygiene, for a caller re-exposing this type
//!
//! [`Error::Backend`]/[`Error::Download`] carry the underlying flux2/s3dit/
//! modelstore crate's own message VERBATIM (see that variant's doc) -- which
//! routinely includes real on-disk paths (a checkpoint file, a models
//! directory). That is the correct, intended contract for an IN-PROCESS
//! caller, who already has filesystem access and needs the real reason a
//! build failed to act on it. It is exactly what a NETWORK-facing surface
//! must never do (`crates/apiserve`/`crates/dbus` both collapse every
//! internal error to a generic, path-free body before it reaches a client)
//! -- this crate has no such collapsing layer, and is not meant to. A caller who
//! turns `Error`'s `Display` straight into a response for an untrusted
//! network client inherits that path-disclosure surface and must add their
//! own translation layer, the same way `apiserve`/`dbus` do in front of
//! everything else in this workspace.

use brain_modelstore::resolve::{describe_ambiguity, describe_missing, Ambiguity, Missing};

/// [`Error::Forecast`]'s payload - see that variant's own doc for why this
/// is a local redefinition of `forecast::ForecastError`'s shape rather than
/// that type itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForecastFailure {
    /// Stable machine-readable slug (`"context_too_long"`, `"missing_variate"`,
    /// `"unsupported_capability"`, `"bad_request"`, `"internal"`, ...).
    pub code: String,
    /// Human-readable detail - already states any numbers a `detail` JSON
    /// blob on the wire-facing type would otherwise repeat.
    pub message: String,
    /// Whether retrying the identical request could plausibly succeed.
    pub retryable: bool,
}

impl std::fmt::Display for ForecastFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

/// Everything a `brain` SDK call can fail with.
#[derive(Debug)]
pub enum Error {
    /// The named model does not exist, or does not name anything the
    /// resolver could ever have been pointed at (an unparseable reference,
    /// a reserved vendor with nothing on disk).
    ModelNotFound(String),
    /// The model-store resolver found more than one candidate for a role (or
    /// could not tell a klein/base variant apart) and nothing picked
    /// automatically -- the full structured answer
    /// [`brain_modelstore::resolve::resolve`] produced, not a flattened
    /// string, so a caller can inspect `question`/`choices` directly instead
    /// of re-parsing [`Error::to_string`]'s rendering of them.
    Ambiguous(Box<Ambiguity>),
    /// The model-store resolver found no usable candidate for one or more
    /// required roles. Structured for the same reason [`Error::Ambiguous`]
    /// is.
    Missing(Box<Missing>),
    /// The resolved (or requested) checkpoint's declared architecture is not
    /// one `brain` can load.
    UnsupportedArchitecture(String),
    /// Fetching a model from the hub failed.
    Download(String),
    /// The resolved checkpoint is gated behind a license the caller has not
    /// accepted (e.g. FLUX.2's 9B weights, Non-Commercial). Named separately
    /// from [`Error::Backend`] because a caller plausibly wants to react to
    /// THIS failure differently -- surface the license terms, or fall back
    /// to an ungated variant -- rather than just reporting a generic error.
    LicenseRequired(String),
    /// A required builder argument was never set (e.g.
    /// [`crate::CreatureBuilder::connectome`]/`body`). Named separately from
    /// [`Error::Backend`] because this is a caller-programming error,
    /// knowable before any backend/GPU/filesystem call runs, not a real
    /// backend failure -- the two are handled differently by any caller that
    /// bothers to `match` on the difference (retry a backend failure,
    /// don't retry a missing argument).
    MissingArgument(String),
    /// A forecasting request failed a capability check or a model's own
    /// validation (context too long, a missing required variate, an
    /// unsupported representation) - [`ForecastFailure`]'s `code`/`message`/
    /// `retryable`, carried structurally rather than flattened into
    /// [`Error::Backend`]'s bare string, for the same reason
    /// [`Error::Ambiguous`] is. Redefined here rather than wrapping
    /// `forecast::ForecastError` directly so `Error`'s own shape does not
    /// depend on the `forecast` feature (this enum is one shape in every
    /// configuration - see this module's doc); the conversion lives in
    /// `crate::forecast`, gated the same way. The wire-only structured
    /// `detail` JSON blob is NOT carried across this boundary - `code`,
    /// `message` and `retryable` are what an in-process Rust caller acts on,
    /// and `message` already states the numbers `detail` would repeat
    /// (e.g. `"context 900 exceeds max_context 512"`).
    Forecast(ForecastFailure),
    /// Any other failure surfaced by the flux2/modelstore backends this
    /// facade sits on -- they return `Result<_, String>` throughout (see
    /// those crates' own module docs for why), and this variant carries that
    /// message verbatim rather than discarding it.
    Backend(String),
    /// A local filesystem operation failed (e.g. [`crate::Image::save`]'s
    /// write, or reading a `--adapter`-style LoRA path).
    Io(std::io::Error),
    /// The caller's [`capability::CancelToken`] fired mid-generation.
    Cancelled,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ModelNotFound(m) => write!(f, "model not found: {m}"),
            Error::Ambiguous(a) => write!(f, "{}", describe_ambiguity(a)),
            Error::Missing(m) => write!(f, "{}", describe_missing(m)),
            Error::UnsupportedArchitecture(a) => write!(f, "unsupported architecture: {a}"),
            Error::Download(m) => write!(f, "download failed: {m}"),
            Error::LicenseRequired(m) => write!(f, "license required: {m}"),
            Error::MissingArgument(m) => write!(f, "{m}"),
            Error::Forecast(e) => write!(f, "{e}"),
            Error::Backend(m) => write!(f, "{m}"),
            Error::Io(e) => write!(f, "{e}"),
            Error::Cancelled => write!(f, "generation cancelled"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// A failed [`brain_modelstore::plan`] call, converted to the variant that
/// names the SAME failure precisely, rather than collapsing every planning
/// failure to [`Error::Backend`] -- a caller told "unsupported architecture"
/// can act on that differently than "the hub is unreachable".
impl From<Box<brain_modelstore::PlanError>> for Error {
    fn from(e: Box<brain_modelstore::PlanError>) -> Self {
        use brain_modelstore::PlanError;
        match *e {
            PlanError::NotFetchable(r) => Error::ModelNotFound(format!("{r}: reserved vendor, not on disk")),
            PlanError::NoUpstreamArtifact(r, why) => Error::ModelNotFound(format!("{r}: {why}")),
            PlanError::UnsupportedArchitecture(r, arch) => Error::UnsupportedArchitecture(format!("{r}: {arch}")),
            PlanError::AmbiguousRecipe(r, ids) => Error::Backend(format!("{r}: ambiguous recipe match -- {}", ids.join(", "))),
            PlanError::Hub(e) => Error::Download(e.to_string()),
        }
    }
}

/// The one place `forecast::ForecastError` becomes [`Error::Forecast`] - see
/// that variant's own doc for why the payload is a local redefinition
/// rather than the wire-facing type itself (`Error`'s shape must not depend
/// on the `forecast` feature).
#[cfg(feature = "forecast")]
impl From<::forecast::ForecastError> for Error {
    fn from(e: ::forecast::ForecastError) -> Self {
        Error::Forecast(ForecastFailure { code: e.code, message: e.message, retryable: e.retryable })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::resolve::{ModelCandidate, Question};
    use capability::Assembly;
    use std::collections::BTreeMap;

    fn candidate(id: &str) -> ModelCandidate {
        ModelCandidate {
            assembly: Box::new(Assembly { id: id.to_string(), arch: "flux2".to_string(), variant: None, roles: BTreeMap::new(), provenance: Vec::new() }),
            selector: vec![("--dit".to_string(), id.to_string())],
            summary: id.to_string(),
        }
    }

    /// The whole point of [`Error::Ambiguous`]: the resolver's own structured
    /// answer survives the trip through this crate's error type -- a caller
    /// can still read `question`/`choices` off it, not just a rendered
    /// string.
    #[test]
    fn ambiguous_preserves_the_resolver_s_structure_rather_than_flattening_it() {
        let a = Ambiguity { arch: "flux2".to_string(), question: Question::Role { role: "text_encoder".to_string() }, choices: vec![candidate("a"), candidate("b")] };
        let err = Error::Ambiguous(Box::new(a));
        let Error::Ambiguous(boxed) = &err else { panic!("must stay Ambiguous") };
        assert_eq!(boxed.arch, "flux2");
        assert_eq!(boxed.question, Question::Role { role: "text_encoder".to_string() });
        assert_eq!(boxed.choices.len(), 2);
        // Display renders the SAME info a `brain flux2` CLI run would print.
        let rendered = err.to_string();
        assert!(rendered.contains("text_encoder"), "{rendered}");
        assert!(rendered.contains("--dit a"), "{rendered}");
    }

    #[test]
    fn backend_carries_the_original_message_verbatim() {
        let err = Error::Backend("flux2: assemble: no dit chosen".to_string());
        assert_eq!(err.to_string(), "flux2: assemble: no dit chosen");
    }

    /// A real `flux2::caps::check_license` gate failure becomes
    /// `Error::LicenseRequired`, not an indistinguishable `Error::Backend` --
    /// a caller can `match` on this to prompt for license acceptance instead
    /// of just reporting a generic failure.
    #[cfg(feature = "image")]
    #[test]
    fn a_real_license_gate_failure_becomes_license_required() {
        let _guard = brain_testutil::env_lock();
        std::env::remove_var("BRAIN_FLUX2_ALLOW_NC");
        let Err(msg) = flux2::caps::check_license("flux2-klein-9b") else {
            panic!("the 9B variant must be gated with BRAIN_FLUX2_ALLOW_NC unset");
        };
        let err = Error::LicenseRequired(msg);
        assert!(err.to_string().starts_with("license required: "), "{err}");
    }

    #[test]
    fn missing_argument_renders_its_own_message_without_a_generic_prefix() {
        let err = Error::MissingArgument("no connectome directory set; call .connectome(dir)".to_string());
        assert_eq!(err.to_string(), "no connectome directory set; call .connectome(dir)");
    }

    /// A real `forecast::ForecastError` (not a hand-built stand-in) survives
    /// the trip through `Error::Forecast` with its code/message/retryable
    /// intact - `code`/`retryable` are exactly what a caller needs to react
    /// differently than to a generic `Error::Backend`.
    #[cfg(feature = "forecast")]
    #[test]
    fn a_real_forecast_error_becomes_error_forecast_with_its_fields_intact() {
        let fe = ::forecast::ForecastError::context_too_long(512, 900);
        let err: Error = fe.into();
        let Error::Forecast(f) = &err else { panic!("expected Error::Forecast, got {err:?}") };
        assert_eq!(f.code, "context_too_long");
        assert!(!f.retryable);
        assert_eq!(err.to_string(), "context_too_long: context 900 exceeds max_context 512");
    }
}
