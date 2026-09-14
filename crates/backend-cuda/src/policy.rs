// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The tier policy: which operators are REQUIRED to reach which
//! implementation tier, on which compute capability - stated as code, and
//! checked by a test.
//!
//! Swedish Embedded AB implements performance contracts that fail loudly
//! instead of quietly. If your team needs expertise in making an
//! accelerator's claims about itself machine-checkable, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! # Why this is a `const`, not a configuration file
//!
//! A tier table is a status ledger: "this operator is supposed to be
//! hand-written by now". Ledgers rot exactly when nothing reads them. As a
//! `.toml` this table would be parsed at run time by a backend that has no
//! reason to fail if it disagrees with reality, and a claim nothing checks
//! is indistinguishable from a claim that is false. As a `const` it is
//! compiled, and a test walks it against the real kernel registry and the
//! real dispatch record - the same shape as this engine's kernel
//! cost-coverage ratchet, which is this repo's proven answer to a number
//! going stale.
//!
//! # What a policy entry is, and is not
//!
//! An entry says: *at compute capability `min_cc` and above, operator `op`
//! must be answered by an implementation of at least tier `required`.* The
//! capability is a THRESHOLD the caller supplies from a runtime query
//! (`cuDeviceGetAttribute`), never a device this file assumes exists. A
//! device below every threshold for an operator is held to nothing, which
//! is what lets a requirement that only makes sense on newer silicon be
//! written down without breaking older cards - and equally lets this table
//! be correct on hardware nobody here has, including cards with matrix
//! engines.

use backend_api::select::Op;
use backend_api::ImplSource;

/// A compute capability as `(major, minor)`, the two halves
/// `cuDeviceGetAttribute` reports separately.
pub type Cc = (u32, u32);

/// One requirement: from `min_cc` upward, `op` must reach at least
/// `required`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PolicyEntry {
    pub op: Op,
    pub min_cc: Cc,
    pub required: ImplSource,
}

/// brain's CUDA tier requirements.
///
/// **Still empty, and for a precise reason that is not "no kernel exists".**
/// A hand-written `Op::MatMul` kernel now ships and is dispatched, but it
/// covers exactly ONE weight tier - plain f32. Every quantized tier
/// (`I8`/`Q4`/K-quant) and both backward GEMMs are still answered by the
/// generated tier, correctly and by design.
///
/// A [`PolicyEntry`] has no dtype axis, so the only entry that could be
/// written here - "`Op::MatMul` must reach `Tuned`" - would also demand it of
/// those, and would therefore be false the first time a quantized linear
/// dispatched. Writing it anyway would make this table exactly the list of
/// things that are *supposed* to be true that the whole design exists to
/// avoid.
///
/// So the next change here is not an entry, it is the dtype axis that makes
/// the first entry statable; it lands with the kernel that widens the tuned
/// tier past f32, not before. Until then the mechanism is exercised by its
/// own tests against fixture tables, and the shipped table honestly demands
/// nothing of any device.
pub const POLICY: &[PolicyEntry] = &[];

/// What `policy` requires of `op` on a device of compute capability `cc`:
/// the applicable entry with the HIGHEST threshold, so a stricter
/// requirement written for newer silicon supersedes a looser one without
/// either being deleted. `None` = nothing is required.
///
/// Takes the table as a parameter so the RULE can be tested independently of
/// whatever the shipped table happens to say today.
pub fn required_in(policy: &[PolicyEntry], op: Op, cc: Cc) -> Option<ImplSource> {
    policy.iter().filter(|e| e.op == op && e.min_cc <= cc).max_by_key(|e| e.min_cc).map(|e| e.required)
}

/// [`required_in`] against the shipped [`POLICY`].
pub fn required(op: Op, cc: Cc) -> Option<ImplSource> {
    required_in(POLICY, op, cc)
}

/// The check itself: `Some(explanation)` when an implementation of tier
/// `got` does not satisfy what `policy` requires of `op` at `cc`.
///
/// The explanation names the operator, the capability it was judged at, the
/// tier that ran and the tier that was required - everything needed to act
/// on it, because the failure it describes is invisible in the output
/// (every tier computes the same answer; only the time differs).
pub fn violation_in(policy: &[PolicyEntry], op: Op, cc: Cc, got: ImplSource) -> Option<String> {
    let required = required_in(policy, op, cc)?;
    if got.satisfies(required) {
        return None;
    }
    Some(format!(
        "{op:?} ran as {got:?} on compute capability {}.{}, where the tier policy requires {required:?}. \
         Every tier computes the same answer, so nothing else will report this.",
        cc.0, cc.1
    ))
}

/// [`violation_in`] against the shipped [`POLICY`].
pub fn violation(op: Op, cc: Cc, got: ImplSource) -> Option<String> {
    violation_in(POLICY, op, cc, got)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped table's own invariants. Vacuous while [`POLICY`] is
    /// empty - which is the point: the checks exist before the entries, so
    /// the first entry lands into something that validates it.
    #[test]
    fn the_shipped_policy_is_well_formed() {
        for (i, e) in POLICY.iter().enumerate() {
            assert!(
                !POLICY.iter().take(i).any(|p| p.op == e.op && p.min_cc == e.min_cc),
                "{:?}: two requirements at the same compute capability - one of them silently loses",
                e.op
            );
            assert_ne!(
                e.required,
                ImplSource::Reference,
                "{:?}: requiring the portable reference requires nothing at all; delete the entry instead",
                e.op
            );
            assert!(e.min_cc.0 > 0, "{:?}: a compute capability threshold of 0.x admits nothing", e.op);
        }
    }

    /// A threshold applies upward only, and the highest applicable one wins.
    /// Asserted on a fixture table: the rule must be correct for capability
    /// values no device here has, including ones above every threshold.
    #[test]
    fn the_highest_applicable_threshold_wins() {
        const FIXTURE: &[PolicyEntry] = &[
            PolicyEntry { op: Op::MatMul, min_cc: (6, 1), required: ImplSource::Generated },
            PolicyEntry { op: Op::MatMul, min_cc: (7, 0), required: ImplSource::Tuned },
        ];
        assert_eq!(required_in(FIXTURE, Op::MatMul, (6, 0)), None);
        assert_eq!(required_in(FIXTURE, Op::MatMul, (6, 1)), Some(ImplSource::Generated));
        assert_eq!(required_in(FIXTURE, Op::MatMul, (6, 9)), Some(ImplSource::Generated));
        assert_eq!(required_in(FIXTURE, Op::MatMul, (12, 0)), Some(ImplSource::Tuned));
        assert_eq!(required_in(FIXTURE, Op::Softmax, (12, 0)), None);

        assert!(violation_in(FIXTURE, Op::MatMul, (6, 0), ImplSource::Reference).is_none());
        assert!(violation_in(FIXTURE, Op::MatMul, (6, 1), ImplSource::Reference).is_some());
        assert!(violation_in(FIXTURE, Op::MatMul, (6, 1), ImplSource::Generated).is_none());
        assert!(violation_in(FIXTURE, Op::MatMul, (7, 0), ImplSource::Generated).is_some());
        assert!(violation_in(FIXTURE, Op::MatMul, (7, 0), ImplSource::Tuned).is_none());
    }
}
