// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which *implementation family* answered an operator - the one fact that
//! distinguishes "this ran on the hand-written kernel the backend exists for"
//! from "this ran on the portable translation of it".
//!
//! Swedish Embedded AB implements accelerator backends whose performance
//! claims are checkable rather than assumed. If your team needs expertise in
//! keeping a fast path honest about when it was actually taken, you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! Numerically the three tiers are interchangeable - that is the whole
//! problem. A generated kernel and an architecture-tuned one produce the
//! same answers, so nothing downstream can tell them apart, and a backend
//! that quietly serves the bottom tier where the top one was promised looks
//! exactly like a backend that is simply slower than hoped. Recording the
//! tier makes that difference a value something can assert on.

/// The implementation family behind one dispatch, ordered worst-to-best by
/// how specialised it is to the device that ran it.
///
/// The ordering is what a tier *policy* compares against
/// (`required <= observed`), so it is part of this type's contract rather
/// than a derive nobody reads: a portable reference never satisfies a
/// requirement for a generated or tuned implementation, and a generated one
/// never satisfies a requirement for a tuned one. It says nothing about
/// measured speed on any particular device - a specialised implementation
/// that turns out to be slower than the reference is a defect to fix, not a
/// reason to renumber the tiers.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum ImplSource {
    /// The portable reference implementation - brain's WGSL kernels, running
    /// through whichever backend compiles them. Always available, never
    /// specialised to anything.
    Reference,
    /// Mechanically derived from the portable reference for this backend's
    /// own language (the WGSL -> CUDA C++ path). It exists for coverage:
    /// without it a native backend cannot run a whole model at all. It is
    /// never evidence that a native backend is worth having.
    Generated,
    /// Hand-written for this backend, optionally specialised to a queried
    /// device capability. The only tier whose existence is an argument for
    /// the backend.
    Tuned,
}

impl ImplSource {
    /// Whether an implementation of this tier satisfies a requirement for
    /// `required` - the single comparison a tier policy makes, spelled once
    /// here so no caller re-derives the ordering's meaning.
    pub fn satisfies(self, required: ImplSource) -> bool {
        self >= required
    }

    /// Stable lowercase spelling for diagnostics and generated tables
    /// (`--trace-impl` output, the CUDA kernel catalogue). Never the `Debug`
    /// formatting, which is free to change.
    pub fn as_str(self) -> &'static str {
        match self {
            ImplSource::Reference => "reference",
            ImplSource::Generated => "generated",
            ImplSource::Tuned => "tuned",
        }
    }

    /// Parse [`Self::as_str`] back. `None` for anything else - callers are
    /// gates and table generators, which must fail loudly on a spelling
    /// nobody defined rather than default to a tier.
    pub fn parse(s: &str) -> Option<ImplSource> {
        match s {
            "reference" => Some(ImplSource::Reference),
            "generated" => Some(ImplSource::Generated),
            "tuned" => Some(ImplSource::Tuned),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ordering IS the policy rule, so it is asserted rather than left
    /// to the derive's field order being obviously right.
    #[test]
    fn a_lower_tier_never_satisfies_a_higher_requirement() {
        assert!(ImplSource::Tuned.satisfies(ImplSource::Tuned));
        assert!(ImplSource::Tuned.satisfies(ImplSource::Generated));
        assert!(ImplSource::Tuned.satisfies(ImplSource::Reference));
        assert!(!ImplSource::Generated.satisfies(ImplSource::Tuned));
        assert!(ImplSource::Generated.satisfies(ImplSource::Reference));
        assert!(!ImplSource::Reference.satisfies(ImplSource::Generated));
        assert!(!ImplSource::Reference.satisfies(ImplSource::Tuned));
    }

    #[test]
    fn spellings_round_trip() {
        for t in [ImplSource::Reference, ImplSource::Generated, ImplSource::Tuned] {
            assert_eq!(ImplSource::parse(t.as_str()), Some(t));
        }
        assert_eq!(ImplSource::parse("fast"), None);
    }
}
