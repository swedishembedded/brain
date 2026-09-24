// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What is worth learning, and what is garbage.
//!
//! A reader on an unbounded stream cannot train on everything, and should
//! not want to. Four filters in increasing order of cost decide what happens
//! to an episode, so that the cheapest one rejects the most:
//!
//! | | costs | rejects |
//! |---|---|---|
//! | **A** screen | nothing | text with no structure in it |
//! | **B** reach | one forward pass | what the model already predicts, and what it cannot predict at all |
//! | **C** evidence | nothing | an episode too small to be evidence either way |
//! | **D** gate | a training run and two decode passes | a candidate that did not earn promotion |
//!
//! Every outcome is a [`Verdict`] that names the filter it stopped at, so a
//! ledger row can always say where an episode ended and why, and the share
//! of the stream reaching each filter is a reportable number. A triage that
//! passes everything through to D is a triage that is not working.
//!
//! ## B is a property of the model's CURRENT state, which is why it defers
//!
//! The learnable band is bounded on both sides. Below it the model already
//! predicts the text and there is nothing to gain; above it the model cannot
//! model the text at all and one episode will not change that. Neither bound
//! is a property of the document: the same text can be out of reach at
//! episode 10 and squarely in the band at episode 800. So
//! [`Verdict::OutOfReach`] is a DEFERRAL - the episode goes back in the queue
//! for a later attempt - and not a rejection.
//!
//! ## What triage does not decide
//!
//! **Learnable is not true.** An episode can be well-formed, novel, inside
//! the band and wrong. Nothing here infers trustworthiness from content,
//! because content-inferred trust is precisely how a reader ends up
//! confidently absorbing confidently-written nonsense. Trust is an input set
//! per source by whoever pointed the reader at it, and a contradiction with
//! something already learned is surfaced by filter D naming the block that
//! regressed, never adjudicated here.

use flate2::write::DeflateEncoder;
use flate2::Compression;
use promote::gate::{gate, Decision, GateConfig, GateInput, GateReport};
use serde::{Deserialize, Serialize};
use std::io::Write;

/// Why filter A refused an episode.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Unstructured {
    /// Too little text to be worth a forward pass.
    TooShort { chars: usize, floor: usize },
    /// Compression does no better than the text's own symbol frequencies,
    /// so there is no structure in it beyond which characters it happens to
    /// use. This catches high-entropy text the stream's binary screen
    /// cannot: base64 payloads, hex dumps and key material are all valid
    /// UTF-8 and contain no NUL, so they arrive here intact.
    NoStructure { structure: f64, floor: f64 },
}

/// What happened to one episode, and at which filter.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// A. Refused without a model being touched.
    Unstructured(Unstructured),
    /// B. The model already predicts this.
    AlreadyKnown { loss: f64, below: f64 },
    /// B. Out of reach for now. Deferred, not discarded.
    OutOfReach { loss: f64, above: f64 },
    /// C. Too few frozen probes to be evidence either way, so it is
    /// accumulated rather than promoted on a result that could not have been
    /// significant.
    TooSmallToGate { probes: usize, floor: usize },
    /// D. Trained, scored and promoted.
    Promoted(GateReport),
    /// D. Trained, scored and refused, with the gate's own cause.
    Rejected(GateReport),
}

impl Verdict {
    /// Which filter produced this verdict. A ledger row carries it so the
    /// share of the stream stopping at each stage is countable.
    pub fn stage(&self) -> &'static str {
        match self {
            Verdict::Unstructured(_) => "screen",
            Verdict::AlreadyKnown { .. } | Verdict::OutOfReach { .. } => "reach",
            Verdict::TooSmallToGate { .. } => "evidence",
            Verdict::Promoted(_) | Verdict::Rejected(_) => "gate",
        }
    }

    /// The gate's own reason for a refusal, when there was one. A ledger
    /// row carries it so "rejected" is never the whole story.
    pub fn cause(&self) -> Option<&'static str> {
        match self {
            Verdict::Rejected(r) => match r.decision {
                Decision::Reject(c) => Some(match c {
                    promote::gate::Cause::NotSignificant { .. } => "not_significant",
                    promote::gate::Cause::EffectTooSmall { .. } => "effect_too_small",
                    promote::gate::Cause::AnchorRegressed { .. } => "anchor_regressed",
                    promote::gate::Cause::BlockRegressed { .. } => "block_regressed",
                    promote::gate::Cause::Degenerate { .. } => "degenerate",
                }),
                Decision::Promote => None,
            },
            Verdict::Unstructured(Unstructured::TooShort { .. }) => Some("too_short"),
            Verdict::Unstructured(Unstructured::NoStructure { .. }) => Some("no_structure"),
            Verdict::AlreadyKnown { .. } => Some("already_known"),
            Verdict::OutOfReach { .. } => Some("out_of_reach"),
            Verdict::TooSmallToGate { .. } => Some("too_small_to_gate"),
            Verdict::Promoted(_) => None,
        }
    }

    /// Whether this episode should be offered again later. Only the deferral
    /// says yes: everything else is a decision.
    pub fn retry_later(&self) -> bool {
        matches!(self, Verdict::OutOfReach { .. } | Verdict::TooSmallToGate { .. })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriageConfig {
    /// The shortest episode the structure measure can still judge.
    ///
    /// Not a taste threshold: `structure` compares deflate against the
    /// text's own symbol entropy, and deflate finds nothing the frequencies
    /// did not until it has several hundred bytes to work with. Below this a
    /// real manual page and random base64 both score zero, so a lower floor
    /// does not admit more documents - it refuses short ones as
    /// UNSTRUCTURED, which is a claim about the text rather than about the
    /// measure. Calibrated in
    /// `the_length_floor_is_where_structure_can_still_tell_text_from_noise`.
    pub min_chars: usize,
    /// How far compression must beat the text's own symbol entropy before
    /// the text counts as structured. See [`structure`].
    pub min_structure: f64,
    /// Loss at or below which the model already predicts the episode.
    pub known_below: f64,
    /// Loss at or above which one episode will not close the gap.
    pub reach_above: f64,
    /// Fewest frozen probes an episode needs before its gate result could
    /// have been significant at all. See [`MIN_EPISODE_PROBES`].
    pub min_probes: usize,
}

/// An exact paired sign test needs `k` clean wins for `p = 2^-k`, so five are
/// the fewest that can reach `p <= 0.05` at all and a set that cannot supply
/// them cannot produce a significant result however well the model did.
/// Twelve leaves room to lose a few and still clear the bar, rather than
/// requiring a perfect sweep of the minimum.
pub const MIN_EPISODE_PROBES: usize = 12;

impl Default for TriageConfig {
    fn default() -> Self {
        TriageConfig { min_chars: 640, min_structure: 0.15, known_below: 0.20, reach_above: 4.0, min_probes: MIN_EPISODE_PROBES }
    }
}

/// The pre-registered promote/reject bars for a continual reader.
///
/// Deliberately NOT a new set of numbers. It is
/// `promote::document::document_gate_config` with the per-block bar left
/// armed, because a reader is a continual learner in exactly the sense that
/// config was pre-registered for: its anchor suite is earlier episodes'
/// frozen probes, one block each, and a pooled mean goes blind to one of
/// them collapsing as the run lengthens. Restating the thresholds here would
/// be a second place for them to drift.
pub fn reader_gate_config() -> GateConfig {
    promote::document::document_gate_config()
}

/// Filter A. No model is touched.
pub fn screen(text: &str, cfg: &TriageConfig) -> Option<Unstructured> {
    let chars = text.chars().count();
    if chars < cfg.min_chars {
        return Some(Unstructured::TooShort { chars, floor: cfg.min_chars });
    }
    let structure = structure(text);
    (structure < cfg.min_structure).then_some(Unstructured::NoStructure { structure, floor: cfg.min_structure })
}

/// Filter B. `loss` is the episode's own loss under the CURRENT model, which
/// is why this can only be asked once per episode per model state.
pub fn reach(loss: f64, cfg: &TriageConfig) -> Option<Verdict> {
    if loss <= cfg.known_below {
        return Some(Verdict::AlreadyKnown { loss, below: cfg.known_below });
    }
    if loss >= cfg.reach_above {
        return Some(Verdict::OutOfReach { loss, above: cfg.reach_above });
    }
    None
}

/// Filters C and D. `probes` is how many frozen probes the episode yielded.
pub fn adjudicate(probes: usize, input: &GateInput, cfg: &TriageConfig, gate_cfg: &GateConfig) -> Verdict {
    if probes < cfg.min_probes {
        return Verdict::TooSmallToGate { probes, floor: cfg.min_probes };
    }
    let report = gate(input, gate_cfg);
    match report.decision {
        Decision::Promote => Verdict::Promoted(report),
        Decision::Reject(_) => Verdict::Rejected(report),
    }
}

/// How much structure `text` has beyond its own symbol frequencies, in
/// `[0, 1]`.
///
/// A raw compression ratio cannot answer this, and getting that wrong is
/// easy: deflate compresses ANY text over a restricted alphabet down towards
/// `log2(symbols)/8` whether or not there is a pattern in it, so random
/// base64 lands near 0.75 and random hex near 0.5 while containing nothing
/// to learn. Measuring the deflated size against the text's OWN order-0
/// entropy removes that: what is left is how much deflate found that the
/// symbol frequencies alone did not.
///
/// The separation is pinned by a test rather than described here: random
/// base64 must score under 0.05 and a manual page over 0.4, with the default
/// floor between them and clear of both. That test also asserts that a RAW
/// deflate ratio would NOT have separated the two, so the extra term cannot
/// quietly become unnecessary without something failing.
///
/// It is a garbage filter and not a quality judgement. Text that is
/// structured but uninformative - one line repeated five hundred times -
/// scores HIGH here and is caught by the next filter instead, where a model
/// that predicts it trivially reports a low loss.
fn structure(text: &str) -> f64 {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return 0.0;
    }
    let floor = order0_ratio(bytes);
    if floor <= f64::EPSILON {
        // One repeated byte carries no information at all; there is nothing
        // for compression to beat and nothing to learn either.
        return 0.0;
    }
    (1.0 - deflate_ratio(bytes) / floor).clamp(0.0, 1.0)
}

/// Deflated size over original size.
fn deflate_ratio(bytes: &[u8]) -> f64 {
    let mut enc = DeflateEncoder::new(Vec::new(), Compression::fast());
    // Deflate into an in-memory buffer cannot fail, and a screen returning an
    // error would have to be handled at every call site for a case that does
    // not arise. `fast` because this runs on every episode and the question
    // is whether the text compresses AT ALL, which the fast setting answers
    // as well as the slow one.
    if enc.write_all(bytes).is_err() {
        return 1.0;
    }
    enc.finish().map(|out| out.len() as f64 / bytes.len() as f64).unwrap_or(1.0)
}

/// Shannon entropy of the byte distribution, as a share of the eight bits a
/// byte could carry. The size any coder gets to for free, knowing only which
/// symbols appear and how often.
fn order0_ratio(bytes: &[u8]) -> f64 {
    let mut counts = [0u64; 256];
    for &b in bytes {
        counts[b as usize] += 1;
    }
    let n = bytes.len() as f64;
    let h: f64 = counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum();
    h / 8.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Rng;
    use promote::document::MAX_BLOCK_DROP;
    use promote::gate::Cause;

    fn manual(n: usize) -> String {
        (0..n).map(|i| format!("--flag{i:03} VALUE   set the {i:03} option to VALUE\n")).collect()
    }

    /// A manual page whose lines do NOT all share one template.
    ///
    /// `manual` repeats a single sentence with a counter in it, which
    /// deflate reduces far harder than real documentation - it scores 0.45
    /// at 256 bytes where a real page scores 0.00. A length floor calibrated
    /// against it would be calibrated against the easiest input there is,
    /// and would pass while every real short document was refused.
    fn varied_manual(n: usize) -> String {
        const VERBS: [&str; 8] = ["set", "limit", "select", "restrict", "report", "override", "resolve", "expand"];
        const NOUNS: [&str; 8] = ["window", "cutoff", "depth", "origin", "stride", "weight", "anchor", "budget"];
        let mut rng = Rng::new(7);
        (0..n)
            .map(|i| {
                let v = VERBS[(rng.next_u64() % 8) as usize];
                let a = NOUNS[(rng.next_u64() % 8) as usize];
                let b = NOUNS[(rng.next_u64() % 8) as usize];
                format!("  --{a}-{b} VALUE   {v} the {a} of the {b} to VALUE, position {i}\n")
            })
            .collect()
    }

    /// Valid UTF-8 with no structure in it: what a base64 payload or a key
    /// block looks like to a reader. The stream's binary screen passes it,
    /// because it contains no NUL and decodes cleanly.
    fn high_entropy(chars: usize, seed: u64) -> String {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut rng = Rng::new(seed);
        (0..chars).map(|_| ALPHABET[(rng.next_u64() % ALPHABET.len() as u64) as usize] as char).collect()
    }

    /// `n` paired scores where the candidate wins `wins` cleanly and ties the
    /// rest - the shape an episode that taught something produces.
    fn arm(n: usize, wins: usize) -> (Vec<f64>, Vec<f64>) {
        let mut c = vec![0.0f64; n];
        for x in c.iter_mut().take(wins) {
            *x = 1.0;
        }
        (c, vec![0.0f64; n])
    }

    fn input<'a>(c: &'a [f64], i: &'a [f64], blocks: &'a [(f64, f64)]) -> GateInput<'a> {
        GateInput {
            candidate_scores: c,
            incumbent_scores: i,
            anchor_candidate: blocks.iter().map(|(a, _)| a).sum::<f64>() / blocks.len().max(1) as f64,
            anchor_incumbent: blocks.iter().map(|(_, b)| b).sum::<f64>() / blocks.len().max(1) as f64,
            entropy_candidate: 2.0,
            entropy_incumbent: 2.0,
            anchor_blocks: blocks,
        }
    }

    /// The pair that makes filter A worth having: structure passes, the
    /// absence of it does not. The separation is what matters, not either
    /// number alone, so both are asserted against the same floor.
    #[test]
    fn text_with_no_structure_in_it_is_refused_before_any_model_is_touched() {
        let cfg = TriageConfig::default();
        let page = manual(40);
        assert_eq!(screen(&page, &cfg), None, "a manual page (structure {}) must reach the model", structure(&page));

        let noise = high_entropy(4000, 5);
        match screen(&noise, &cfg) {
            Some(Unstructured::NoStructure { structure: st, floor }) => {
                assert!(st < floor, "structure {st} must be under the floor {floor} to have fired");
            }
            other => panic!("expected NoStructure, got {other:?} (noise scores {})", structure(&noise)),
        }
        assert!(matches!(screen("short", &cfg), Some(Unstructured::TooShort { .. })));
    }

    /// `min_chars` is the length at which `structure` can still tell text
    /// from noise, and nothing shorter.
    ///
    /// Deflate needs several hundred bytes before it finds anything a text's
    /// own symbol frequencies did not, so below that a real manual page and
    /// random base64 BOTH score exactly zero. With a floor under that range
    /// every short document was refused as having no structure - a claim
    /// about the text, when the truth was that the screen could not see it.
    /// `TooShort` says the second thing, and is the honest answer.
    #[test]
    fn the_length_floor_is_where_structure_can_still_tell_text_from_noise() {
        let cfg = TriageConfig::default();
        let n = cfg.min_chars;
        let page: String = varied_manual(40).chars().take(n).collect();
        let noise = high_entropy(n, 17);
        assert!(
            structure(&page) > cfg.min_structure,
            "at the floor ({n} chars) a realistic manual page must still clear min_structure {}, got {}",
            cfg.min_structure,
            structure(&page)
        );
        assert!(
            structure(&noise) < cfg.min_structure,
            "and noise at the same length must not, got {}",
            structure(&noise)
        );

        // Below the floor the screen must say it cannot tell rather than
        // that there is nothing there.
        let short: String = varied_manual(40).chars().take(n - 1).collect();
        assert!(
            matches!(screen(&short, &cfg), Some(Unstructured::TooShort { .. })),
            "a page one character under the floor is TooShort, not NoStructure"
        );

        // And the floor is not arbitrary: well under it the measure has
        // nothing to say about EITHER input. That is the range a lower floor
        // was admitting, and refusing as unstructured.
        let quarter: String = varied_manual(40).chars().take(n / 4).collect();
        assert_eq!(
            structure(&quarter),
            structure(&high_entropy(n / 4, 19)),
            "well under the floor a manual page and noise are indistinguishable, which is why the floor is where it is"
        );
    }

    /// A raw compression ratio cannot separate these two, which is why the
    /// screen measures against the text's own symbol entropy instead: deflate
    /// takes random base64 to roughly the same ratio a manual page reaches,
    /// for the entirely different reason that its alphabet is small.
    #[test]
    fn structure_separates_noise_from_text_where_a_raw_compression_ratio_would_not() {
        let page = manual(40);
        let noise = high_entropy(4000, 11);
        assert!(structure(&noise) < 0.05, "random base64 must score near zero, got {}", structure(&noise));
        assert!(structure(&page) > 0.4, "a manual page must score well clear of it, got {}", structure(&page));
        assert!(
            deflate_ratio(noise.as_bytes()) < 0.95,
            "and the raw ratio must NOT have separated them on its own, or this measure is unnecessary: noise ratio {}",
            deflate_ratio(noise.as_bytes())
        );
    }

    /// Filter B is bounded on BOTH sides, and the bounds mean opposite
    /// things: nothing to gain, versus nothing to gain YET.
    #[test]
    fn the_learnable_band_is_bounded_on_both_sides_and_only_the_upper_one_defers() {
        let cfg = TriageConfig::default();
        assert_eq!(reach(1.5, &cfg), None, "a loss inside the band must proceed to the gate");

        let known = reach(0.05, &cfg).expect("below the band");
        assert!(matches!(known, Verdict::AlreadyKnown { .. }));
        assert!(!known.retry_later(), "an episode the model already knows is a decision, not a deferral");

        let far = reach(9.0, &cfg).expect("above the band");
        assert!(matches!(far, Verdict::OutOfReach { .. }));
        assert!(far.retry_later(), "out of reach is about the model's CURRENT state, so it must be offered again");
    }

    /// Filter C. An exact sign test cannot reach `p <= 0.05` on fewer than
    /// five clean wins, so an episode that could not have supplied them must
    /// not be promoted on a result that was never capable of being
    /// significant.
    #[test]
    fn an_episode_too_small_to_be_evidence_is_accumulated_rather_than_promoted() {
        let cfg = TriageConfig::default();
        let (c, i) = arm(4, 4);
        let v = adjudicate(4, &input(&c, &i, &[]), &cfg, &reader_gate_config());
        match v {
            Verdict::TooSmallToGate { probes, floor } => {
                assert_eq!(probes, 4);
                assert_eq!(floor, MIN_EPISODE_PROBES);
            }
            ref other => panic!("expected TooSmallToGate, got {other:?}"),
        }
        assert!(v.retry_later(), "an episode held back for want of evidence must come round again");
    }

    /// The stay-silent half: enough probes, a real win, and it promotes.
    #[test]
    fn an_episode_that_taught_something_promotes() {
        let cfg = TriageConfig::default();
        let (c, i) = arm(20, 14);
        let v = adjudicate(20, &input(&c, &i, &[(0.9, 0.9)]), &cfg, &reader_gate_config());
        assert!(matches!(v, Verdict::Promoted(_)), "14 clean wins in 20 must clear every bar, got {v:?}");
        assert_eq!(v.stage(), "gate");
    }

    /// R0's per-block bar has to be in force HERE, not merely available:
    /// this is the config a reader actually gates with.
    #[test]
    fn the_reader_gate_arms_the_per_block_bar_against_one_collapsing_earlier_episode() {
        let cfg = TriageConfig::default();
        let (c, i) = arm(20, 14);
        // Twenty earlier episodes healthy at 0.9, one down to 0.6: the pooled
        // mean moves 0.015, inside the budget, so only the per-block bar can
        // see it.
        let mut blocks: Vec<(f64, f64)> = (0..20).map(|_| (0.9, 0.9)).collect();
        blocks[7] = (0.6, 0.9);
        match adjudicate(20, &input(&c, &i, &blocks), &cfg, &reader_gate_config()) {
            Verdict::Rejected(r) => match r.decision {
                Decision::Reject(Cause::BlockRegressed { block, max_drop, .. }) => {
                    assert_eq!(block, 7, "the verdict must name WHICH earlier episode regressed");
                    assert!((max_drop - MAX_BLOCK_DROP).abs() < 1e-12);
                }
                other => panic!("expected BlockRegressed, got {other:?}"),
            },
            other => panic!("expected a rejection, got {other:?}"),
        }
    }

    /// Every outcome has to say which filter produced it, or the share of the
    /// stream stopping at each stage is not countable.
    #[test]
    fn every_verdict_names_the_stage_that_produced_it() {
        let cfg = TriageConfig::default();
        let (c, i) = arm(20, 14);
        let stages = [
            Verdict::Unstructured(Unstructured::TooShort { chars: 1, floor: 64 }).stage(),
            reach(0.01, &cfg).unwrap().stage(),
            reach(99.0, &cfg).unwrap().stage(),
            adjudicate(2, &input(&c, &i, &[]), &cfg, &reader_gate_config()).stage(),
            adjudicate(20, &input(&c, &i, &[(0.9, 0.9)]), &cfg, &reader_gate_config()).stage(),
        ];
        assert_eq!(stages, ["screen", "reach", "reach", "evidence", "gate"]);
    }
}
