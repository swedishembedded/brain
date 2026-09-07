// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The document-learning [`Environment`]/[`Verifier`] pair and the
//! pre-registered [`document_gate_config`] a document/fact batch is promoted
//! under (continuous-learning roadmap B2).
//!
//! A batch is N `{fact, probe_question, expected_answer}` triples, frozen at
//! extraction time by the agent that read the document. The FACT is what gets
//! trained; the PROBE is what gets scored; the two live in [`FactSplit::
//! Train`] and [`FactSplit::Probe`] of one [`FactBatch`].
//!
//! ## Verification is programmatic, never model-as-judge
//!
//! [`DocumentVerifier`] decodes the policy's completion, normalises it
//! ([`normalize`]) and compares its digest against the digest carried in
//! [`Task::answer`] - a pure function of a stored run artifact, re-runnable
//! byte-for-byte, exactly the discipline [`crate::env`]'s module doc requires.
//! That is what makes a document-learning claim defensible rather than
//! vibes-based: no second model's opinion enters the number.
//!
//! ## The expected answer is a DIGEST, never the text
//!
//! [`crate::env`] is explicit that `Task::answer` may carry only what a
//! verifier needs to RECOMPUTE correctness - "a target value, a reference
//! program, or a checksum" - never the answer itself, because an answer
//! sitting in a `Task` is indistinguishable from a label and nothing
//! label-shaped may reach training data. A document probe's answer cannot be
//! recomputed from its prompt (it is a fact about the world, not a function
//! of the prompt the way [`crate::curriculum`]'s positional copy is), so the
//! checksum form is the one this family uses: [`answer_digest`].
//!
//! ## The split is a property of the CONTENT
//!
//! A task's id is derived from the row's own normalised text and nothing
//! else: no seed, no split tag, no draw order. So "the probe split is
//! disjoint from the training split" is a real, checkable statement about the
//! batch rather than a tautology about how two id prefixes were spelled, and
//! [`train_probe_split`]'s disjointness assertion (mirroring
//! [`crate::improve::explore_anchor_split`]'s) can actually fail.
//!
//! ## What the batch is checked for, rather than trusted about
//!
//! [`FactBatch::new`] refuses a batch whose probe question appears inside any
//! fact the model will be trained on. A probe whose own question sits in the
//! training span is a memorisation test wearing a generalisation test's
//! clothes, and it would silently inflate every number downstream of it.
//!
//! Note what is deliberately NOT asserted: that a probe's expected ANSWER is
//! absent from the trained rows. It cannot be - the fact "the 3rd relay
//! closes at 13 volts" is exactly the row that teaches the probe answer "13
//! volts", so an answer-absence rule would reject every batch that could ever
//! work. The checkable form of "the probe was frozen, not memorised" is the
//! question, not the answer.
//!
//! Swedish Embedded AB builds the verification machinery that decides whether
//! a model actually learned a document or merely moved - frozen probe sets,
//! programmatic exact-match rewards, and pre-registered statistical bars a
//! promotion has to clear before anything reaches a served model. If your
//! team needs expertise in gating continuous-learning pipelines honestly, you
//! can procure our services by sending an email to info@swedishembedded.com.

use std::collections::{BTreeMap, HashSet};

use data::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::env::{Environment, Reward, Step, Task, Verifier};
use crate::gate::GateConfig;

/// One `{fact, probe_question, expected_answer}` triple, exactly as the
/// extracting agent froze it.
///
/// `deny_unknown_fields` with plain (non-`Option`) members on purpose: this
/// type is the boundary a batch crosses into brain through, and serde itself
/// is then the structural validator - a missing, mistyped or extra field is a
/// loud parse failure rather than a plausible-looking default.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactProbe {
    /// The statement the model is trained on.
    pub fact: String,
    /// The frozen probe question the gate scores against. Never trained on.
    pub probe_question: String,
    /// The answer `probe_question` must elicit, verbatim.
    pub expected_answer: String,
}

/// Which half of a [`FactBatch`] an environment presents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactSplit {
    /// The distinct fact statements - what gets trained.
    Train,
    /// The frozen probes - what gets scored, and never trained.
    Probe,
}

/// How many held-out probes a document-learning cycle must have.
///
/// Matches the measured SFT regime's own `--eval-per-cycle 48`, and puts
/// [`bench::metrics::sign_test`]'s discordant-pair floor (5 net wins for
/// `p <= 0.05`) far out of the danger zone. Batching many facts into one
/// cycle makes the sign test EASIER to satisfy, not harder.
pub const MIN_HELD_OUT_PROBES: usize = 48;

/// The pre-registered promote/reject thresholds for document learning.
///
/// Pre-registered means fixed before the run, not tuned after seeing it. The
/// one deliberate departure from [`GateConfig::default`] is
/// `min_effect_size = 0.15` instead of `0.02`: `effect_size` is
/// `mean(candidate.held_out) - mean(incumbent.held_out)` over the WHOLE
/// scored set, so at the default floor a two percent wobble on a large probe
/// set promotes. The claim being gated is "it learned the document", not "it
/// moved". `alpha`, `anchor_budget` and `min_entropy_ratio` are the defaults
/// and are not re-stated as magic numbers here.
pub fn document_gate_config() -> GateConfig {
    GateConfig { min_effect_size: 0.15, ..GateConfig::default() }
}

/// Normalised form of a row's text: trimmed, internal whitespace runs
/// collapsed to one space, lowercased.
///
/// This is the ONLY latitude an exact match gets. It exists because a decoder
/// controls its own leading space and casing and neither is part of the
/// answer; anything looser - substring containment, edit distance, a second
/// model's opinion - would stop being exact match and take the defensibility
/// of the number with it.
pub fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<&str>>().join(" ").to_lowercase()
}

/// SHA-256 hex digest of [`normalize`]d `text` - the checksum form
/// [`crate::env`] sanctions for a `Task::answer` whose correct completion
/// cannot be recomputed from the prompt.
pub fn answer_digest(text: &str) -> String {
    let mut h = Sha256::new();
    h.update(normalize(text).as_bytes());
    format!("{:x}", h.finalize())
}

/// A validated document fact batch: the triples plus the DISTINCT fact
/// statements they are drawn from (20 facts x 3 probes is 60 triples over 20
/// training rows).
pub struct FactBatch {
    triples: Vec<FactProbe>,
    facts: Vec<String>,
}

impl FactBatch {
    /// Validate `triples` and take ownership of them. Panics, naming the
    /// offending record, on any of:
    ///
    /// - an empty batch, or a blank field in any triple;
    /// - two triples sharing a probe question - two probes with one identity
    ///   is a probe set smaller than it reports, and `continual::run_study`
    ///   asserts pairwise-disjoint probe ids anyway;
    /// - a probe question that appears inside a fact the model will be
    ///   trained on (see this module's doc comment).
    ///
    /// A panic rather than a `Result` for the same reason
    /// [`crate::improve::explore_anchor_split`] panics: every one of these is
    /// a defect in whatever produced the batch, and continuing produces a
    /// flattering number, which is strictly worse than stopping.
    pub fn new(triples: Vec<FactProbe>) -> FactBatch {
        assert!(!triples.is_empty(), "rl::document::FactBatch::new: an empty fact batch has nothing to train and nothing to score");
        for (i, t) in triples.iter().enumerate() {
            for (field, value) in [("fact", &t.fact), ("probe_question", &t.probe_question), ("expected_answer", &t.expected_answer)] {
                assert!(!normalize(value).is_empty(), "rl::document::FactBatch::new: triple {i} has a blank `{field}`");
            }
        }

        let mut seen_questions: HashSet<String> = HashSet::new();
        for (i, t) in triples.iter().enumerate() {
            assert!(
                seen_questions.insert(normalize(&t.probe_question)),
                "rl::document::FactBatch::new: triple {i}'s probe question {:?} already appears in this batch - two probes sharing one identity make the probe set smaller than it reports",
                t.probe_question
            );
        }

        let mut facts: Vec<String> = Vec::new();
        let mut seen_facts: HashSet<String> = HashSet::new();
        for t in &triples {
            if seen_facts.insert(normalize(&t.fact)) {
                facts.push(t.fact.clone());
            }
        }

        for (i, t) in triples.iter().enumerate() {
            let q = normalize(&t.probe_question);
            for f in &facts {
                assert!(
                    !normalize(f).contains(&q),
                    "rl::document::FactBatch::new: triple {i}'s probe question {:?} appears inside the trained fact {f:?} - a probe whose own question is in the training span measures memorisation, not learning",
                    t.probe_question
                );
            }
        }

        FactBatch { triples, facts }
    }

    /// Every frozen probe triple, in batch order.
    pub fn triples(&self) -> &[FactProbe] {
        &self.triples
    }

    /// The DISTINCT fact statements, in first-appearance order - one training
    /// row each.
    pub fn facts(&self) -> &[String] {
        &self.facts
    }
}

/// One half of a [`FactBatch`] presented as an [`Environment`]. Both halves
/// have the same task shape (a text prompt, a digest to verify against), so
/// one [`DocumentVerifier`] scores either.
pub struct DocumentEnv<'a, T: Tokenizer> {
    batch: &'a FactBatch,
    split: FactSplit,
    tok: &'a T,
}

impl<'a, T: Tokenizer> DocumentEnv<'a, T> {
    pub fn new(batch: &'a FactBatch, split: FactSplit, tok: &'a T) -> DocumentEnv<'a, T> {
        DocumentEnv { batch, split, tok }
    }

    /// How many distinct tasks this half holds. [`Environment::tasks`] maps a
    /// seed onto `seed % len`, so `0..len` enumerates the half exactly once.
    pub fn len(&self) -> usize {
        match self.split {
            FactSplit::Train => self.batch.facts.len(),
            FactSplit::Probe => self.batch.triples.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn split(&self) -> FactSplit {
        self.split
    }

    /// `(presented text, expected completion)` for this half's `i`-th row.
    /// A TRAINING row presents the fact statement and expects it back: that
    /// is literally what the teacher-forced SFT row supervises, so the
    /// environment describes the same thing the trainer does rather than a
    /// second, differently-phrased version of it.
    fn row(&self, i: usize) -> (&str, &str) {
        match self.split {
            FactSplit::Train => (self.batch.facts[i].as_str(), self.batch.facts[i].as_str()),
            FactSplit::Probe => (self.batch.triples[i].probe_question.as_str(), self.batch.triples[i].expected_answer.as_str()),
        }
    }
}

/// The task id for a row: `doc-` plus the row's own [`normalize`]d text, and
/// NOTHING else - see this module's doc comment on why the split must not
/// appear in it.
fn task_id(text: &str) -> String {
    format!("doc-{}", normalize(text))
}

impl<T: Tokenizer> Environment for DocumentEnv<'_, T> {
    fn name(&self) -> &str {
        match self.split {
            FactSplit::Train => "document-fact",
            FactSplit::Probe => "document-probe",
        }
    }

    /// Exactly one task, `seed % len` into this half. Deterministic in
    /// `seed`, and `0..len` enumerates the half.
    fn tasks(&self, seed: u64) -> Vec<Task> {
        let n = self.len();
        assert!(n > 0, "rl::document::DocumentEnv::tasks: the {:?} half of this batch is empty", self.split);
        let (presented, expected) = self.row((seed % n as u64) as usize);
        vec![Task {
            id: task_id(presented),
            prompt: self.tok.encode(presented),
            answer: serde_json::json!({ "answer_digest": answer_digest(expected) }),
        }]
    }
}

/// Programmatic exact match against the probe's own expected answer: decode
/// the completion, take its first line, [`normalize`] it and compare digests.
///
/// Stopping at the first line break follows `eval::gpt_exact_match`'s
/// convention for a greedy decode with a token budget - what comes after the
/// answer is the decoder filling its budget, not part of the answer. Reward
/// is 1.0 or 0.0; there is deliberately no partial credit, because a half
/// right fact is a wrong fact.
pub struct DocumentVerifier<'a, T: Tokenizer> {
    tok: &'a T,
}

impl<'a, T: Tokenizer> DocumentVerifier<'a, T> {
    pub fn new(tok: &'a T) -> DocumentVerifier<'a, T> {
        DocumentVerifier { tok }
    }
}

impl<T: Tokenizer> Verifier for DocumentVerifier<'_, T> {
    fn verify(&self, task: &Task, _transcript: &[Step], completion: &[u32]) -> Reward {
        let decoded = self.tok.decode(completion);
        let first_line = decoded.lines().next().unwrap_or("");
        let want = task.answer["answer_digest"]
            .as_str()
            .unwrap_or_else(|| panic!("rl::document::DocumentVerifier: task {} carries no `answer_digest`", task.id));
        let value = if answer_digest(first_line) == want { 1.0 } else { 0.0 };
        Reward { value, parts: BTreeMap::from([("exact_match".to_string(), value)]) }
    }
}

/// Build `batch`'s (training, probe) environment pair, asserting the two
/// structural properties a document-learning cycle's number rests on:
///
/// 1. the probe half carries at least [`MIN_HELD_OUT_PROBES`] probes;
/// 2. no task id appears in both halves - the same check, on the ids that
///    were really generated, that [`crate::improve::explore_anchor_split`]
///    makes for an explore/anchor split. The content-derived id namespace
///    already makes a collision structurally impossible; this is the
///    belt-and-braces check that it actually held.
pub fn train_probe_split<'a, T: Tokenizer>(batch: &'a FactBatch, tok: &'a T) -> (DocumentEnv<'a, T>, DocumentEnv<'a, T>) {
    assert!(
        batch.triples.len() >= MIN_HELD_OUT_PROBES,
        "rl::document::train_probe_split: {} held-out probes is below the pre-registered floor of {MIN_HELD_OUT_PROBES} - below it the sign test has too few discordant pairs to reach p <= alpha at all",
        batch.triples.len()
    );
    let train = DocumentEnv::new(batch, FactSplit::Train, tok);
    let probe = DocumentEnv::new(batch, FactSplit::Probe, tok);
    let train_ids: HashSet<String> = batch.facts.iter().map(|f| task_id(f)).collect();
    let overlap: Vec<&str> = batch
        .triples
        .iter()
        .map(|t| t.probe_question.as_str())
        .filter(|q| train_ids.contains(&task_id(q)))
        .collect();
    assert!(
        overlap.is_empty(),
        "rl::document::train_probe_split: probe question(s) {overlap:?} are also training rows - a held-out score is only honest if the policy never trained on the task it is scored against"
    );
    (train, probe)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::Environment;
    use crate::gate::{gate, Cause, Decision, GateInput};
    use data::tokenizer::CharTokenizer;
    use std::collections::HashSet;

    /// 20 facts x 3 probes = 60 triples, the batch shape B2 sizes its
    /// held-out budget against.
    fn batch_of_twenty_facts() -> FactBatch {
        let triples: Vec<FactProbe> = (0..20)
            .flat_map(|f| {
                (0..3).map(move |p| FactProbe {
                    fact: format!("the {f}th relay closes at {} volts", 10 + f),
                    probe_question: format!("question {p} about relay {f}"),
                    expected_answer: format!("{} volts", 10 + f),
                })
            })
            .collect();
        FactBatch::new(triples)
    }

    fn tokenizer_over(batch: &FactBatch) -> CharTokenizer {
        let corpus: String = batch.triples().iter().map(|t| format!("{}{}{}", t.fact, t.probe_question, t.expected_answer)).collect();
        CharTokenizer::from_corpus(&corpus)
    }

    #[test]
    fn the_probe_split_is_disjoint_from_the_training_split_by_task_id() {
        let batch = batch_of_twenty_facts();
        let tok = tokenizer_over(&batch);
        let (train, probe) = train_probe_split(&batch, &tok);

        let train_ids: HashSet<String> = (0..train.len() as u64).map(|s| train.tasks(s).remove(0).id).collect();
        let probe_ids: HashSet<String> = (0..probe.len() as u64).map(|s| probe.tasks(s).remove(0).id).collect();

        assert_eq!(train_ids.len(), 20, "one training task per DISTINCT fact");
        assert_eq!(probe_ids.len(), 60, "one probe task per frozen triple");
        let overlap: Vec<&String> = probe_ids.intersection(&train_ids).collect();
        assert!(overlap.is_empty(), "a frozen probe is also a trained row: {overlap:?}");
    }

    /// The disjointness above is a property of the CONTENT, not of how the
    /// two id prefixes were spelled: a batch whose probe question IS a
    /// trained fact is refused at construction, so the assertion can fail.
    #[test]
    #[should_panic(expected = "appears inside the trained fact")]
    fn a_probe_question_that_sits_in_the_training_span_is_refused() {
        let mut triples: Vec<FactProbe> = (0..60)
            .map(|i| FactProbe {
                fact: format!("relay {i} closes at {} volts", 10 + i),
                probe_question: format!("question about relay {i}"),
                expected_answer: format!("{} volts", 10 + i),
            })
            .collect();
        triples[7].probe_question = triples[7].fact.clone();
        let _ = FactBatch::new(triples);
    }

    #[test]
    fn a_probe_verifies_by_programmatic_exact_match_against_its_expected_answer() {
        let batch = batch_of_twenty_facts();
        let tok = tokenizer_over(&batch);
        let (_, probe) = train_probe_split(&batch, &tok);
        let verifier = DocumentVerifier::new(&tok);

        let task = probe.tasks(0).remove(0);
        let right = verifier.verify(&task, &[], &tok.encode(&batch.triples()[0].expected_answer));
        assert_eq!(right.value, 1.0);
        assert_eq!(right.parts.get("exact_match"), Some(&1.0));

        // Triples 0..3 all probe fact 0, so they share an expected answer;
        // triple 3 is the first one belonging to a DIFFERENT fact.
        let wrong = verifier.verify(&task, &[], &tok.encode(&batch.triples()[3].expected_answer));
        assert_eq!(wrong.value, 0.0, "a different fact's answer must not verify");

        // Deterministic and re-runnable byte-for-byte: the same artifact
        // re-scores to the same reward, whole breakdown included.
        assert_eq!(right, verifier.verify(&task, &[], &tok.encode(&batch.triples()[0].expected_answer)));

        // Nothing label-shaped sits in the task: the expected answer is
        // present only as a digest.
        assert!(task.answer.get("answer_digest").is_some());
        assert!(
            !task.answer.to_string().contains(&batch.triples()[0].expected_answer),
            "Task::answer must carry a checksum, never the answer text"
        );
    }

    #[test]
    #[should_panic(expected = "below the pre-registered floor of 48")]
    fn a_probe_set_below_the_held_out_floor_is_refused() {
        let triples: Vec<FactProbe> = (0..MIN_HELD_OUT_PROBES - 1)
            .map(|i| FactProbe {
                fact: format!("relay {i} closes at {} volts", 10 + i),
                probe_question: format!("question about relay {i}"),
                expected_answer: format!("{} volts", 10 + i),
            })
            .collect();
        let batch = FactBatch::new(triples);
        let tok = tokenizer_over(&batch);
        let _ = train_probe_split(&batch, &tok);
    }

    /// 250 paired probes, candidate wins 5 and loses none: significant
    /// (`p = 0.5^5 = 0.03125 <= alpha`) yet the whole-set effect size is
    /// `5/250 = 0.02` - a two percent wobble, exactly where
    /// [`GateConfig::default`]'s own `min_effect_size` floor sits.
    #[test]
    fn a_two_percent_wobble_does_not_promote_under_the_document_gate_config() {
        let mut candidate = vec![0.0f64; 250];
        let incumbent = vec![0.0f64; 250];
        for c in candidate.iter_mut().take(5) {
            *c = 1.0;
        }
        let input = GateInput {
            candidate_scores: &candidate,
            incumbent_scores: &incumbent,
            anchor_candidate: 0.9,
            anchor_incumbent: 0.9,
            entropy_candidate: 2.0,
            entropy_incumbent: 2.0,
        };

        // The hazard the pre-registered config exists to close: the DEFAULT
        // gate promotes this. "It moved" is not "it learned the document".
        assert_eq!(
            gate(&input, &GateConfig::default()).decision,
            Decision::Promote,
            "the default GateConfig is expected to promote a two percent wobble - that is why B2 pre-registers its own"
        );

        let report = gate(&input, &document_gate_config());
        match report.decision {
            Decision::Reject(Cause::EffectTooSmall { effect_size, min_effect_size }) => {
                assert!((effect_size - 0.02).abs() < 1e-12, "effect size should be the 2 % wobble itself, got {effect_size}");
                assert!((min_effect_size - 0.15).abs() < 1e-12, "the document gate's pre-registered floor is 0.15, got {min_effect_size}");
            }
            other => panic!("expected Reject(EffectTooSmall), got {other:?}"),
        }
        // The rejection must be about MAGNITUDE, not noise: the sign test
        // still cleared alpha.
        assert!(report.p_value <= document_gate_config().alpha, "p = {} should still clear alpha", report.p_value);
    }

    /// The three thresholds the document config deliberately does NOT move
    /// stay at their defaults - a silent drift in any of them would change
    /// what a promotion means without anything saying so.
    #[test]
    fn the_document_gate_config_moves_only_the_effect_size_floor() {
        let cfg = document_gate_config();
        let def = GateConfig::default();
        assert_eq!(cfg.alpha, def.alpha);
        assert_eq!(cfg.anchor_budget, def.anchor_budget);
        assert_eq!(cfg.min_entropy_ratio, def.min_entropy_ratio);
        assert!(cfg.min_effect_size > def.min_effect_size);
    }
}
