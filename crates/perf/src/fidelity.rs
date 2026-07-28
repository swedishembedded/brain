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
            gate: "greedy_token_match".into(),
            reference: reference.into(),
            token_match,
            threshold,
            compared,
            passed,
            detail,
        }
    }

    /// The artifact's `correctness` block.
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

    #[test]
    fn json_carries_the_verdict() {
        let a = vec![vec![7u32, 8]];
        let j = Fidelity::greedy("cpu", &a, &a, EXACT).to_json();
        assert_eq!(j["passed"], true);
        assert_eq!(j["gate"], "greedy_token_match");
        assert_eq!(j["compared_positions"], 2);
    }
}
