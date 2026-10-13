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
}

impl FailureClass {
    /// Every class, in documentation order.
    pub const ALL: [FailureClass; 6] = [
        FailureClass::NotInstalled,
        FailureClass::FailedToLoad,
        FailureClass::UnsupportedCapability,
        FailureClass::Timeout,
        FailureClass::InvalidRequest,
        FailureClass::BackendFailure,
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
        assert_eq!(slugs, ["not-installed", "failed-to-load", "unsupported-capability", "timeout", "invalid-request", "backend-failure"]);
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
