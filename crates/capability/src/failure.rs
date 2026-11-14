// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Machine-readable failure classes for an [`ActionResult`](crate::ActionResult)
//! error, which is a bare `String`.
//!
//! A client that must tell "the weights are not installed" from "the request
//! was malformed" from "the deadline passed" cannot do it by matching prose.
//! [`tag`] puts the class in front of the message as `[<slug>] <message>`;
//! [`classify`] reads it back. The convention is additive: an error with no
//! tag is simply unclassified, so every existing action is unaffected, and a
//! transport that understands classes (HTTP maps them to statuses) falls back
//! to its generic handling for the rest.
//!
//! A tagged message is shown to the caller by transports that recognise the
//! tag, so it must name no filesystem path and no backend internals; log those
//! server-side instead.
//!
//! Swedish Embedded AB implements model-serving contracts whose failures a
//! client can act on rather than parse. If your team needs expertise in
//! error taxonomies for inference services, you can procure our services by
//! sending an email to info@swedishembedded.com.

/// Why a call failed, in the terms a client branches on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureClass {
    /// The model is known but its weights are not on this host. Retrying
    /// helps only after the weights are pulled.
    NotInstalled,
    /// The weights are present but cannot be loaded: an incompatible or
    /// corrupt checkpoint, a hash that does not match the manifest, a device
    /// that could not hold the model.
    FailedToLoad,
    /// The model loaded but cannot do what was asked (a question type it was
    /// not built for, say).
    UnsupportedCapability,
    /// The request's deadline passed before an answer was produced.
    Timeout,
    /// The request itself is malformed or outside the model's published
    /// limits. Retrying unchanged cannot succeed.
    InvalidRequest,
    /// The model was loaded and the request was valid, but inference failed.
    BackendFailure,
    /// The model was loaded and the request was valid, but the host has no room for it right now (a device
    /// out of memory with nothing else left to give back). Nothing is wrong with the request or the server:
    /// retrying later, or with a shorter prompt, can succeed.
    Unavailable,
}

impl FailureClass {
    /// Every class, in documentation order.
    pub const ALL: [FailureClass; 7] = [
        FailureClass::NotInstalled,
        FailureClass::FailedToLoad,
        FailureClass::UnsupportedCapability,
        FailureClass::Timeout,
        FailureClass::InvalidRequest,
        FailureClass::BackendFailure,
        FailureClass::Unavailable,
    ];

    /// The class's stable wire name.
    pub fn slug(self) -> &'static str {
        match self {
            FailureClass::NotInstalled => "not-installed",
            FailureClass::FailedToLoad => "failed-to-load",
            FailureClass::UnsupportedCapability => "unsupported-capability",
            FailureClass::Timeout => "timeout",
            FailureClass::InvalidRequest => "invalid-request",
            FailureClass::BackendFailure => "backend-failure",
            FailureClass::Unavailable => "unavailable",
        }
    }

    /// The class named by `slug`, if it is one.
    pub fn parse(slug: &str) -> Option<FailureClass> {
        FailureClass::ALL.into_iter().find(|c| c.slug() == slug)
    }
}

/// An error string carrying `class`: `[<slug>] <message>`.
pub fn tag(class: FailureClass, message: impl AsRef<str>) -> String {
    let message = message.as_ref();
    if message.is_empty() {
        format!("[{}]", class.slug())
    } else {
        format!("[{}] {message}", class.slug())
    }
}

/// The class and message of a [`tag`]ged error, or `None` for any other string.
///
/// A transport may prefix context to an error as it passes it on
/// (`activate <model>/<config>: [not-installed] ...`), so a tag is also found
/// after any `": "`.
pub fn classify(error: &str) -> Option<(FailureClass, &str)> {
    let mut rest = error;
    loop {
        if let Some(found) = classify_here(rest) {
            return Some(found);
        }
        rest = &rest[rest.find(": ")? + 2..];
    }
}

fn classify_here(error: &str) -> Option<(FailureClass, &str)> {
    let rest = error.strip_prefix('[')?;
    let (slug, message) = rest.split_once(']')?;
    let class = FailureClass::parse(slug)?;
    Some((class, message.strip_prefix(' ').unwrap_or(message)))
}

/// How `residency`'s lane loop words a panic it isolated.
const PANICKED: &str = ": panicked while running: ";

/// How a text model's `generate` reports a `suffix` its vocabulary cannot fill
/// in the middle of (the end of `qwen3::chat::NO_FIM_TOKENS`'s message).
const NO_FIM_TOKENS: &str = "has no fill-in-the-middle tokens";

/// The error a cancelled action or job answers with: the literal every
/// surface forwards unchanged, so a client can map it onto a typed
/// cancellation (see `Registry::run`'s cancellation contract).
pub const CANCELLED: &str = "cancelled";

/// What a failed call may tell the caller, whatever surface it came through.
///
/// A model's or the scheduler's error text can carry a weight path, a panic
/// message with its source location, an OS error or a backend's internals,
/// none of which is the caller's to read. [`ReplyError::of`] sorts an error
/// into what can be said; every surface answers from that and logs the text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplyError {
    /// The model classed its own failure: the tag promises the message names no
    /// path or internals ([`tag`]).
    Classified(FailureClass, String),
    /// A lane panicked while running the call.
    Faulted,
    /// The model or action does not exist.
    NoModel,
    /// A `suffix` the model's vocabulary cannot fill in the middle of.
    NoFim,
    /// The prompt needs more tokens than the model's serving capacity.
    Capacity { need: u64, capacity: u64 },
    /// The job was cancelled before it answered: by its caller, or by the
    /// server draining at shutdown.
    Cancelled,
    /// Anything else: a runtime or activation failure.
    Failed,
}

impl ReplyError {
    /// Sort an executor reply error.
    pub fn of(e: &str) -> ReplyError {
        if let Some((class, message)) = classify(e) {
            return ReplyError::Classified(class, message.to_string());
        }
        if e.contains(PANICKED) {
            return ReplyError::Faulted;
        }
        if e == CANCELLED || e.ends_with(&format!(": {CANCELLED}")) {
            return ReplyError::Cancelled;
        }
        if e.starts_with("no model") || e.contains("no action") {
            return ReplyError::NoModel;
        }
        if e.ends_with(NO_FIM_TOKENS) {
            return ReplyError::NoFim;
        }
        if let Some((need, capacity)) = parse_exceeds_capacity(e) {
            return ReplyError::Capacity { need, capacity };
        }
        ReplyError::Failed
    }

    /// Whether the original text holds what only the operator may read, and
    /// so belongs in the server's log.
    pub fn hides_detail(&self) -> bool {
        matches!(self, ReplyError::Faulted | ReplyError::Capacity { .. } | ReplyError::Failed)
    }

    /// The message a caller is given, naming nothing of this machine.
    pub fn client_message(&self, model: &str) -> String {
        match self {
            ReplyError::Classified(_, message) => message.clone(),
            ReplyError::Faulted => "the model faulted while running the request".into(),
            ReplyError::NoModel => format!("no model '{model}'"),
            ReplyError::NoFim => "'suffix' (fill-in-the-middle) is not supported: this model's vocabulary has no fill-in-the-middle tokens".into(),
            ReplyError::Capacity { need, capacity } => {
                format!("the request needs {need} tokens but this model's serving capacity is {capacity} tokens; reduce the prompt or lower max_tokens")
            }
            ReplyError::Cancelled => CANCELLED.into(),
            ReplyError::Failed => "the model failed to process the request".into(),
        }
    }
}

/// The two counts in a capacity rejection, the one rejection reason whose text
/// is safe to surface: `RejectReason::ExceedsCapacity`'s
/// `"needs {need} tokens, engine capacity is {capacity}"`.
pub fn parse_exceeds_capacity(e: &str) -> Option<(u64, u64)> {
    let rest = e.split_once("needs ")?.1;
    let (need, capacity) = rest.split_once(" tokens, engine capacity is ")?;
    Some((need.trim().parse().ok()?, capacity.trim().parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_class_round_trips_through_its_slug() {
        for class in FailureClass::ALL {
            assert_eq!(FailureClass::parse(class.slug()), Some(class), "{}", class.slug());
        }
        assert_eq!(FailureClass::parse("no-such-class"), None);
    }

    #[test]
    fn slugs_are_the_documented_wire_names() {
        let slugs: Vec<&str> = FailureClass::ALL.iter().map(|c| c.slug()).collect();
        assert_eq!(slugs, ["not-installed", "failed-to-load", "unsupported-capability", "timeout", "invalid-request", "backend-failure", "unavailable"]);
    }

    #[test]
    fn a_tagged_error_is_classified_and_its_message_recovered() {
        let e = tag(FailureClass::NotInstalled, "weights for 'x/y' are not installed: run `brain pull x/y`");
        assert_eq!(classify(&e), Some((FailureClass::NotInstalled, "weights for 'x/y' are not installed: run `brain pull x/y`")));
    }

    #[test]
    fn an_untagged_or_unknown_error_is_not_classified() {
        assert_eq!(classify("cancelled"), None);
        assert_eq!(classify("[nonsense] hi"), None);
        assert_eq!(classify("[timeout"), None);
        assert_eq!(classify(""), None);
        // Only a tag at the start, or right after a "context: " prefix, counts.
        assert_eq!(classify("clip says [timeout] in prose"), None);
    }

    /// A cancellation keeps its load-bearing literal: it names nothing of
    /// the machine, and a client maps exactly that word back onto a typed
    /// cancellation.
    #[test]
    fn a_cancellation_reaches_the_caller_as_cancelled() {
        for e in ["cancelled", "mock/default: cancelled"] {
            let reply = ReplyError::of(e);
            assert_eq!(reply, ReplyError::Cancelled, "{e}");
            assert_eq!(reply.client_message("m"), "cancelled");
            assert!(!reply.hides_detail());
        }
        assert_eq!(ReplyError::of("not cancelled yet: disk full"), ReplyError::Failed);
    }

    #[test]
    fn a_transport_prefix_does_not_hide_the_tag() {
        let e = "activate acme/model/default: [failed-to-load] the checkpoint is corrupt";
        assert_eq!(classify(e), Some((FailureClass::FailedToLoad, "the checkpoint is corrupt")));
    }

    #[test]
    fn a_tag_with_no_message_still_classifies() {
        assert_eq!(classify("[timeout]"), Some((FailureClass::Timeout, "")));
    }
}
