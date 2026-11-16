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
    /// The largest gap of the reference's logits a divergence may flip and still pass ([`Fidelity::greedy_tie_aware`]),
    /// as a fraction of the reference's largest logit magnitude at that position; 0 accepts none.
    pub tie_tolerance: f32,
    /// The divergences accepted as near-ties; the sequence each names is not compared past it.
    pub ties: Vec<TieDivergence>,
}

/// A divergence the tie-aware gate accepted: at `position` of `sequence` the candidate chose `token` where the reference
/// chose `reference_token`, and the reference's logits there put the two `gap` apart, as a fraction of their largest
/// magnitude there.
#[derive(Clone, Debug, PartialEq)]
pub struct TieDivergence {
    pub sequence: usize,
    pub position: usize,
    pub token: u32,
    pub reference_token: u32,
    pub gap: f32,
}

impl Fidelity {
    /// Compare two greedy token streams position by position.
    ///
    /// Length mismatch is itself a failure: a run that stopped early produced
    /// different behaviour, and scoring only the common prefix would hide it.
    pub fn greedy(reference: &str, cand: &[Vec<u32>], refr: &[Vec<u32>], threshold: f64) -> Fidelity {
        verdict(GREEDY_GATE, reference, cand, refr, threshold, 0.0, &mut |_, _, _: &u32, _: &u32| None)
    }

    /// [`Self::greedy`] for a target whose numeric contract lets a batching difference flip a near-tie: each sequence
    /// must agree exactly up to its first divergence, and that divergence passes only when `gap(sequence, position,
    /// token, reference_token)` - the reference's logit of its own token less its logit of the candidate's, over its
    /// largest logit magnitude there, measured by the target - is within `tolerance`. The sequence is then not compared past it (the two continuations read
    /// different tokens from there), and the divergence is recorded. A gap over the tolerance, one the target cannot
    /// measure (`None`), or any divergence at a tolerance of 0 fails as in [`Self::greedy`].
    pub fn greedy_tie_aware(reference: &str, cand: &[Vec<u32>], refr: &[Vec<u32>], threshold: f64, tolerance: f32, mut gap: impl FnMut(usize, usize, u32, u32) -> Option<f32>) -> Fidelity {
        verdict(GREEDY_GATE, reference, cand, refr, threshold, tolerance, &mut |seq, pos, c: &u32, r: &u32| gap(seq, pos, *c, *r))
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
        verdict(BYTE_GATE, reference, cand, refr, threshold, 0.0, &mut |_, _, _: &u8, _: &u8| None)
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
            tie_tolerance: 0.0,
            ties: Vec::new(),
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
            "tie_tolerance": self.tie_tolerance,
            "tie_divergences": self.ties.iter().map(|t| json!({
                "sequence": t.sequence,
                "position": t.position,
                "token": t.token,
                "reference_token": t.reference_token,
                "gap": t.gap,
                "note": "accepted as a near-tie of the reference's logits; the sequence is not compared past it",
            })).collect::<Vec<_>>(),
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

/// What one compared position holds: a token id or an output byte, named by its id in a [`TieDivergence`].
trait Position: PartialEq + std::fmt::Debug {
    fn id(&self) -> u32;
}

impl Position for u32 {
    fn id(&self) -> u32 {
        *self
    }
}

impl Position for u8 {
    fn id(&self) -> u32 {
        u32::from(*self)
    }
}

/// The ONE comparison both public constructors are: position-by-position over
/// `max(len)` of each pair, a length or count mismatch recorded as its own
/// failure, and zero compared positions treated as failure rather than as a
/// vacuous pass. Generic over the position type purely so the token and byte
/// gates cannot drift apart.
///
/// `tolerance` and `gap` are the tie-aware rule ([`Fidelity::greedy_tie_aware`]): at a sequence's first divergence of
/// two present positions, a gap within a positive tolerance ends that sequence's comparison as an accepted tie.
fn verdict<T: Position>(
    gate: &str,
    reference: &str,
    cand: &[Vec<T>],
    refr: &[Vec<T>],
    threshold: f64,
    tolerance: f32,
    gap: &mut dyn FnMut(usize, usize, &T, &T) -> Option<f32>,
) -> Fidelity {
    let mut compared = 0usize;
    let mut agree = 0usize;
    let mut detail = None;
    let mut ties = Vec::new();

    if cand.len() != refr.len() {
        detail = Some(format!("sequence count differs: {} vs reference {}", cand.len(), refr.len()));
    }
    for (i, (c, r)) in cand.iter().zip(refr.iter()).enumerate() {
        let n = c.len().max(r.len());
        let first = (0..n).find(|&p| c.get(p) != r.get(p));
        // A near-tie at the first divergence: the streams agree before it and are not comparable after it.
        let tie = match first.map(|p| (p, c.get(p), r.get(p))) {
            Some((p, Some(a), Some(b))) if tolerance > 0.0 => match gap(i, p, a, b) {
                Some(g) if g.abs() <= tolerance => Some((p, g)),
                measured => {
                    if detail.is_none() {
                        let why = measured.map_or("the target could not measure the reference's gap".to_string(), |g| format!("gap {g} over the tie tolerance {tolerance}"));
                        detail = Some(format!("first divergence at sequence {i} position {p}: {a:?} vs reference {b:?}, {why}"));
                    }
                    None
                }
            },
            _ => None,
        };
        if let Some((p, g)) = tie {
            compared += p;
            agree += p;
            ties.push(TieDivergence { sequence: i, position: p, token: c[p].id(), reference_token: r[p].id(), gap: g });
            continue;
        }
        if c.len() != r.len() && detail.is_none() {
            detail = Some(format!("sequence {i} length {} vs reference {}", c.len(), r.len()));
        }
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
        tie_tolerance: tolerance,
        ties,
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

    /// SPEC (tie-aware greedy): the streams must agree exactly up to the first divergence; a divergence passes only
    /// where the reference's logits there show a genuine near-tie between the two tokens (a gap within the tolerance the
    /// target declares), and the sequence is not compared past it. Anything else still fails, and a tolerance of 0 (an
    /// exact-decode target) accepts no divergence at all.
    #[test]
    fn a_near_tie_divergence_passes_and_a_large_gap_flip_fails() {
        let refr = vec![vec![1u32, 2, 3, 4], vec![5, 6, 7]];
        let cand = vec![vec![1u32, 2, 9, 8], vec![5, 6, 7]];
        let gap = |gap: f32| move |seq: usize, pos: usize, token: u32, reference: u32| {
            assert_eq!((seq, pos, token, reference), (0, 2, 9, 3), "the gap is asked once, at the first divergence");
            Some(gap)
        };
        let tie = Fidelity::greedy_tie_aware("seq", &cand, &refr, EXACT, 0.05, gap(0.01));
        assert!(tie.passed, "{:?}", tie.detail);
        assert_eq!(tie.ties, vec![TieDivergence { sequence: 0, position: 2, token: 9, reference_token: 3, gap: 0.01 }]);
        assert_eq!(tie.compared, 5, "positions 0 and 1 of the first sequence and all three of the second; nothing past the tie");
        assert_eq!(tie.token_match, 1.0);
        let j = tie.to_json();
        assert_eq!(j["tie_tolerance"], 0.05f32 as f64);
        assert_eq!(j["tie_divergences"][0]["position"], 2);
        assert_eq!(j["tie_divergences"][0]["token"], 9);
        assert_eq!(j["tie_divergences"][0]["reference_token"], 3);
        assert!(j["tie_divergences"][0]["note"].as_str().unwrap().contains("not compared"));

        let flip = Fidelity::greedy_tie_aware("seq", &cand, &refr, EXACT, 0.05, gap(0.5));
        assert!(!flip.passed, "a flip across a large gap is a different computation");
        assert!(flip.ties.is_empty());
        assert!(flip.detail.as_deref().unwrap().contains("gap 0.5"), "{:?}", flip.detail);

        let exact = Fidelity::greedy_tie_aware("seq", &cand, &refr, EXACT, 0.0, |_, _, _, _| Some(0.0));
        assert!(!exact.passed, "an exact-decode target accepts no divergence, however close the logits");

        let unknown = Fidelity::greedy_tie_aware("seq", &cand, &refr, EXACT, 0.05, |_, _, _, _| None);
        assert!(!unknown.passed, "a gap the target cannot measure is not a tie");
    }

    /// After an accepted tie the rest of that sequence is not compared, so a length difference behind it (the two
    /// continuations stop at different places) is no failure; a length difference in a sequence that agreed is.
    #[test]
    fn nothing_behind_an_accepted_tie_is_compared() {
        let refr = vec![vec![1u32, 2, 3, 4, 5]];
        let cand = vec![vec![1u32, 7]];
        let f = Fidelity::greedy_tie_aware("seq", &cand, &refr, EXACT, 0.1, |_, _, _, _| Some(0.02));
        assert!(f.passed, "{:?}", f.detail);
        let short = Fidelity::greedy_tie_aware("seq", &[vec![1u32, 2]], &refr, EXACT, 0.1, |_, _, _, _| Some(0.02));
        assert!(!short.passed, "an early stop is no divergence of tokens and never a tie");
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
