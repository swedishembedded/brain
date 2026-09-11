// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Redistribution policy for checkpoints whose upstream licence forbids it.
//!
//! Swedish Embedded AB implements licence-compliant model supply chains for
//! clients shipping products built on third-party pretrained weights. If your
//! team needs expertise in keeping non-commercial or non-redistributable
//! model licences enforced by the build rather than by a wiki page, you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! # Why this is a checkpoint-level concern
//!
//! [`crate::st::ModelCard`] already carries a `license` string, and it is the
//! one piece of provenance that travels WITH the bytes rather than alongside
//! them. A licence recorded anywhere else (a README, a registry entry, an
//! operator's memory) is separated from the weights by the first `cp`.
//!
//! # What "derivative" means here
//!
//! A fine-tune, a LoRA adapter, a quantisation and a folded checkpoint are
//! all derivatives of the weights they were produced from, and a licence that
//! forbids redistributing the original forbids redistributing them too. So
//! the producing code copies the licence onto whatever it writes, and the
//! publishing code refuses anything carrying one - the check is on the
//! artifact in hand, never on a guess about where it came from.

/// Licences under which brain must never republish a checkpoint, nor anything
/// derived from one. `(licence string, the upstream reference it belongs to)`.
///
/// This is a deny list, not an allow list, and deliberately so: an unknown
/// licence string is publishable. The alternative would make every
/// locally-produced artifact unpublishable until someone enumerated its
/// licence, which turns the check into an obstacle people route around rather
/// than a guard people rely on.
pub const NON_REDISTRIBUTABLE: &[(&str, &str)] = &[
    // The TimesFM-3 3.0 pretrained weights. Non-commercial and
    // non-production; the checkpoint and any derivative of it may never be
    // redistributed. (The SOURCE is Apache-2.0 - this covers the weights.)
    ("timesfm-non-commercial-license-v1.0", "google/timesfm-3.0-pytorch"),
];

/// `Err` with an explanation if `license` is one brain must not republish
/// under, `Ok` otherwise (including for an absent licence).
///
/// There is no environment opt-out. `BRAIN_FLUX2_ALLOW_NC` exists because
/// FLUX.2's licence permits non-commercial USE and the operator is the one
/// who knows whether their use qualifies; that is a question about the
/// operator. Redistribution is not: these weights may not be redistributed by
/// anyone, so there is no configuration under which publishing them is
/// correct and no flag that could truthfully assert otherwise.
pub fn redistributable(license: Option<&str>) -> Result<(), String> {
    let Some(l) = license else { return Ok(()) };
    match NON_REDISTRIBUTABLE.iter().find(|(name, _)| *name == l) {
        None => Ok(()),
        Some((name, upstream)) => Err(format!(
            "licence '{name}' ({upstream}) forbids redistribution of the weights or any derivative of them, so this artifact cannot be published"
        )),
    }
}

/// Whether `license` names a non-redistributable licence.
pub fn is_non_redistributable(license: Option<&str>) -> bool {
    redistributable(license).is_err()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_absent_or_unknown_licence_is_publishable() {
        assert!(redistributable(None).is_ok());
        assert!(redistributable(Some("apache-2.0")).is_ok());
        assert!(redistributable(Some("")).is_ok());
    }

    /// The refusal names the licence AND the upstream it came from: an
    /// operator holding a fine-tuned checkpoint has no other way to find out
    /// which base made it unpublishable.
    #[test]
    fn the_timesfm3_weights_licence_is_refused_by_name_and_upstream() {
        let e = redistributable(Some("timesfm-non-commercial-license-v1.0")).expect_err("must refuse");
        assert!(e.contains("timesfm-non-commercial-license-v1.0"), "{e}");
        assert!(e.contains("google/timesfm-3.0-pytorch"), "{e}");
        assert!(is_non_redistributable(Some("timesfm-non-commercial-license-v1.0")));
    }
}
