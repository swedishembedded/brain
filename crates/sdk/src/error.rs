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

use brain_modelstore::resolve::{describe_ambiguity, describe_missing, Ambiguity, Missing};

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
}
