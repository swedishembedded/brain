// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The correctness gate — a perf number is only accepted if the computation was
//! still right.
//!
//! Performance benchmarking without an output check actively rewards
//! optimisations that quietly break the model: dropping a normalisation, an
//! off-by-one in a KV index, or a quantisation that shifts the argmax all make
//! the engine faster and the results worthless. So every scenario can carry a
//! gate, and a run that fails it is written with `valid: false` and excluded
//! from `compare` — it is not a "slower but honest" number, it is a measurement
//! of a different computation.
//!
//! The gate here is **greedy token agreement against a reference run of the same
//! prompts**. That is the strongest cheap check for a decoder: greedy decoding
//! is deterministic, so any divergence is a real behavioural change. The
//! reference is normally the same engine at batch size 1 (which exercises a
//! different kernel path than a batched decode) or on a different backend.

use crate::stats::r3;
use serde_json::{json, Value};

/// One gate outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct Fidelity {
    pub gate: String,
    pub reference: String,
    /// Fraction of positions where candidate and reference agree.
    pub token_match: f64,
    /// Required agreement for the run to stay valid.
    pub threshold: f64,
    pub compared: usize,
    pub passed: bool,
    pub detail: Option<String>,
}

impl Fidelity {
    /// Compare two greedy token streams position by position.
    ///
    /// Length mismatch is itself a failure: a run that stopped early produced
    /// different behaviour, and scoring only the common prefix would hide it.
    pub fn greedy(reference: &str, cand: &[Vec<u32>], refr: &[Vec<u32>], threshold: f64) -> Fidelity {
        verdict(GREEDY_GATE, reference, cand, refr, threshold)
    }

    /// Compare two output-**byte** streams position by position.
    ///
    /// The byte-shaped sibling of [`Fidelity::greedy`], for a target whose
    /// output is not token ids: the text a served endpoint actually streamed
    /// back, an encoded image, an embedding blob. It is deliberately the same
    /// comparison with the same discipline - a differing stream count or
    /// length is a failure rather than a scored common prefix, and comparing
    /// **zero** positions fails instead of passing vacuously - because a
    /// second comparator with weaker semantics is exactly how a correctness
    /// gate starts lying.
    pub fn exact_bytes(reference: &str, cand: &[Vec<u8>], refr: &[Vec<u8>], threshold: f64) -> Fidelity {
        verdict(BYTE_GATE, reference, cand, refr, threshold)
    }

    /// The verdict when the probe could not produce output to compare at all -
    /// a probe request failed, or the target refused it. `compared: 0`, which
    /// [`Fidelity::failure_reason`] already words as "nothing to verify"
    /// rather than as a numerically-vacuous inequality.
    ///
    /// This exists so a failing probe can never be mistaken for a passing one:
    /// capturing an error as if it were output would let two identically
    /// broken runs "agree" and score 1.0.
    pub fn probe_failed(gate: &str, reference: &str, detail: String) -> Fidelity {
        Fidelity {
            gate: gate.into(),
            reference: reference.into(),
            // Same convention as a zero-position comparison in `verdict`: 1.0
            // so it can never be read as a low-but-real score, with `passed`
            // false and `failure_reason` naming the real cause.
            token_match: 1.0,
            threshold: EXACT,
            compared: 0,
            passed: false,
            detail: Some(detail),
        }
    }

    /// The artifact's `correctness` block.
    ///
    /// `greedy_token_match` is the schema's fixed name for "fraction of
    /// compared positions that agreed"; which positions those are is named by
    /// `gate` ([`GREEDY_GATE`] = token ids, [`BYTE_GATE`] = output bytes).
    pub fn to_json(&self) -> Value {
        json!({
            "gate": self.gate,
            "reference": self.reference,
            "greedy_token_match": r3(self.token_match),
            "threshold": self.threshold,
            "compared_positions": self.compared,
            "mean_logprob_error": Value::Null,
            "structured_validity": Value::Null,
            "protocol_errors": 0,
            "passed": self.passed,
            "detail": self.detail.clone().map(Value::from).unwrap_or(Value::Null),
        })
    }

    /// Why the run is invalid, for `Artifact::invalidate`.
    pub fn failure_reason(&self) -> String {
        // `compared == 0` defaults `token_match` to 1.0 (see `greedy`'s own
        // comment) precisely so an empty comparison never LOOKS like a passing
        // score -- but that also means the generic "X < Y" phrasing below
        // would print an identical-looking "1.0000 < 1.0000" for a run that
        // compared NOTHING (most often: the probe's own requests were
        // rejected at admission, e.g. an undersized KV pool) as it would for
        // a genuine below-threshold disagreement. Name the real cause instead
        // of a numerically-vacuous inequality.
        if self.compared == 0 {
            return format!(
                "{} compared 0 positions (nothing to verify -- check admission/rejection, not the threshold) vs {}{}",
                self.gate,
                self.reference,
                self.detail.as_ref().map(|d| format!(" ({d})")).unwrap_or_default()
            );
        }
        format!(
            "{} {:.4} < {:.4} vs {}{}",
            self.gate,
            self.token_match,
            self.threshold,
            self.reference,
            self.detail.as_ref().map(|d| format!(" ({d})")).unwrap_or_default()
        )
    }
}

/// The default agreement required: greedy decoding is deterministic, so anything
/// short of exact is a behavioural change.
pub const EXACT: f64 = 1.0;

/// [`Fidelity::greedy`]'s gate name: positions are token ids.
pub const GREEDY_GATE: &str = "greedy_token_match";
/// [`Fidelity::exact_bytes`]'s gate name: positions are output bytes.
pub const BYTE_GATE: &str = "output_byte_match";

/// The ONE comparison both public constructors are: position-by-position over
/// `max(len)` of each pair, a length or count mismatch recorded as its own
/// failure, and zero compared positions treated as failure rather than as a
/// vacuous pass. Generic over the position type purely so the token and byte
/// gates cannot drift apart.
fn verdict<T: PartialEq + std::fmt::Debug>(
    gate: &str,
    reference: &str,
    cand: &[Vec<T>],
    refr: &[Vec<T>],
    threshold: f64,
) -> Fidelity {
    let mut compared = 0usize;
    let mut agree = 0usize;
    let mut detail = None;

    if cand.len() != refr.len() {
        detail = Some(format!("sequence count differs: {} vs reference {}", cand.len(), refr.len()));
    }
    for (i, (c, r)) in cand.iter().zip(refr.iter()).enumerate() {
        if c.len() != r.len() && detail.is_none() {
            detail = Some(format!("sequence {i} length {} vs reference {}", c.len(), r.len()));
        }
        let n = c.len().max(r.len());
        for p in 0..n {
            compared += 1;
            match (c.get(p), r.get(p)) {
                (Some(a), Some(b)) if a == b => agree += 1,
                _ => {
                    if detail.is_none() {
                        detail = Some(format!(
                            "first divergence at sequence {i} position {p}: {:?} vs reference {:?}",
                            c.get(p),
                            r.get(p)
                        ));
                    }
                }
            }
        }
    }
    let token_match = if compared == 0 { 1.0 } else { agree as f64 / compared as f64 };
    // A run that compared nothing has not been verified; treat an empty
    // comparison as a failure rather than a silent pass.
    let passed = compared > 0 && token_match >= threshold && cand.len() == refr.len();
    Fidelity {
        gate: gate.into(),
        reference: reference.into(),
        token_match,
        threshold,
        compared,
        passed,
        detail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_streams_pass_exactly() {
        let a = vec![vec![1u32, 2, 3], vec![4, 5]];
        let f = Fidelity::greedy("cpu", &a, &a, EXACT);
        assert!(f.passed);
        assert_eq!(f.token_match, 1.0);
        assert_eq!(f.compared, 5);
        assert!(f.detail.is_none());
    }

    #[test]
    fn one_differing_token_fails_an_exact_gate() {
        let c = vec![vec![1u32, 2, 3]];
        let r = vec![vec![1u32, 9, 3]];
        let f = Fidelity::greedy("cpu", &c, &r, EXACT);
        assert!(!f.passed, "greedy decoding is deterministic; any divergence is real");
        assert!((f.token_match - 2.0 / 3.0).abs() < 1e-9);
        assert!(f.detail.unwrap().contains("position 1"));
    }

    #[test]
    fn early_stop_is_a_failure_not_a_prefix_match() {
        // Scoring only the common prefix would report 100% here.
        let c = vec![vec![1u32, 2]];
        let r = vec![vec![1u32, 2, 3, 4]];
        let f = Fidelity::greedy("cpu", &c, &r, EXACT);
        assert!(!f.passed);
        assert!(f.token_match < 1.0);
        assert!(f.detail.unwrap().contains("length"));
    }

    #[test]
    fn differing_sequence_count_fails_even_if_common_ones_match() {
        let c = vec![vec![1u32, 2]];
        let r = vec![vec![1u32, 2], vec![3, 4]];
        let f = Fidelity::greedy("cpu", &c, &r, EXACT);
        assert!(!f.passed);
        assert!(f.detail.unwrap().contains("sequence count"));
    }

    #[test]
    fn comparing_nothing_is_not_a_pass() {
        let f = Fidelity::greedy("cpu", &[], &[], EXACT);
        assert!(!f.passed, "an empty comparison has verified nothing");
    }

    #[test]
    fn failure_reason_names_the_gate_and_reference() {
        let c = vec![vec![1u32]];
        let r = vec![vec![2u32]];
        let reason = Fidelity::greedy("gpu0-batch1", &c, &r, EXACT).failure_reason();
        assert!(reason.contains("greedy_token_match"));
        assert!(reason.contains("gpu0-batch1"));
    }

    /// REGRESSION: `compared == 0` defaults `token_match` to 1.0 (so an empty
    /// comparison never LOOKS like a passing score), but the generic "X < Y"
    /// phrasing then printed an identical-looking "1.0000 < 1.0000" for a run
    /// that compared NOTHING (e.g. every request rejected at admission) as it
    /// would for a genuine below-threshold disagreement -- hiding the real
    /// cause. Found via a real perf run whose KV pool was too small to admit
    /// the correctness probe's own requests.
    #[test]
    fn failure_reason_names_zero_comparisons_not_a_vacuous_inequality() {
        let f = Fidelity::greedy("cpu", &[], &[], EXACT);
        let reason = f.failure_reason();
        assert!(reason.contains("compared 0 positions"), "must name the real cause, not a numeric inequality: {reason}");
        assert!(!reason.contains("1.0000 < 1.0000"), "must not print a token_match==threshold inequality for zero comparisons: {reason}");
    }

    /// SPEC: the byte gate is the token gate's discipline applied to a
    /// byte-shaped output - exact agreement, no scored prefixes.
    #[test]
    fn byte_streams_agree_only_when_identical() {
        let a = vec![b"hello".to_vec(), b"world".to_vec()];
        let f = Fidelity::exact_bytes("sequential", &a, &a, EXACT);
        assert!(f.passed);
        assert_eq!(f.compared, 10);
        assert_eq!(f.gate, BYTE_GATE, "the gate must name what a position IS");

        let b = vec![b"hellp".to_vec(), b"world".to_vec()];
        let f = Fidelity::exact_bytes("sequential", &b, &a, EXACT);
        assert!(!f.passed, "one differing byte is a different computation");
        assert!(f.detail.unwrap().contains("position 4"));
    }

    /// REGRESSION-BY-CONSTRUCTION: the byte gate inherits every rule the token
    /// gate exists to enforce. A truncated stream must not score as a matching
    /// prefix, a missing stream must not be ignored, and comparing NOTHING
    /// must never read as a pass - the whole reason this is one shared
    /// implementation and not a second, laxer comparator.
    #[test]
    fn byte_gate_inherits_the_token_gates_discipline() {
        let refr = vec![b"abcd".to_vec()];
        let truncated = vec![b"ab".to_vec()];
        assert!(!Fidelity::exact_bytes("seq", &truncated, &refr, EXACT).passed);

        let missing: Vec<Vec<u8>> = Vec::new();
        let f = Fidelity::exact_bytes("seq", &missing, &refr, EXACT);
        assert!(!f.passed);
        assert!(f.detail.unwrap().contains("sequence count"));

        let f = Fidelity::exact_bytes("seq", &[], &[], EXACT);
        assert!(!f.passed, "an empty comparison has verified nothing");
        assert!(f.failure_reason().contains("compared 0 positions"));
    }

    /// A probe that could not run has verified nothing, and must say WHY.
    /// Capturing the error as if it were output would let two identically
    /// broken runs agree and score a perfect pass.
    #[test]
    fn a_failed_probe_is_a_failure_that_names_its_cause() {
        let f = Fidelity::probe_failed(BYTE_GATE, "sequential", "probe request 1 failed: OOM".into());
        assert!(!f.passed);
        assert_eq!(f.compared, 0);
        let reason = f.failure_reason();
        assert!(reason.contains("compared 0 positions"), "{reason}");
        assert!(reason.contains("OOM"), "the real cause must survive into the artifact: {reason}");
    }

    #[test]
    fn json_carries_the_verdict() {
        let a = vec![vec![7u32, 8]];
        let j = Fidelity::greedy("cpu", &a, &a, EXACT).to_json();
        assert_eq!(j["passed"], true);
        assert_eq!(j["gate"], "greedy_token_match");
        assert_eq!(j["compared_positions"], 2);
    }
}
