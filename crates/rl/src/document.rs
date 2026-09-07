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
use std::path::{Path, PathBuf};

use model::Model;

use data::chat::{ChatMessage, ChatSample, ENDOFTEXT};
use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;
use data::rng::Rng;
use data::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::continual::{self, Curriculum, GatePolicy, Regime, SftConfig, SftSource, StudyConfig, StudyReport, StudySpec};
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

/// A document/fact [`Curriculum`]: one validated [`FactBatch`] per cycle over
/// one tokenizer, plus the behavioural anchor suite every cycle rehearses
/// (continuous-learning roadmap B5').
///
/// There is deliberately no new training composition here. Brain has already
/// MEASURED that a per-cycle `improve::cycle` composition does not accumulate
/// capability at this model scale, and that [`crate::continual::Regime::Sft`]
/// (teacher-forced, mixed with a rehearsal pool) does; so this milestone
/// implements the trait that regime already drives rather than inventing a
/// cycle beside it. Everything a document study needs beyond this impl (the
/// retention matrix, BWT, the fresh-adapter control, the pre-registered
/// PASS/FAIL block) is [`crate::continual::run_study`]'s, unchanged.
///
/// ## Two row shapes, because a fact and a behaviour are not the same thing
///
/// A CYCLE row is a fact statement, supervised whole: the model has to KNOW
/// the fact, not copy it out of a context window, so the fact is
/// language-modelled with nothing masked in front of it. An ANCHOR row is a
/// behaviour - a prompt and the response it must still produce - so its
/// prompt is masked context and only the response is supervised, the ordinary
/// chat-SFT shape. That is why the anchor suite is presented through
/// [`FactSplit::Probe`]: it is the half of a [`FactBatch`] that carries a
/// `(prompt, response)` pair. Those rows are never scored as probes; nothing
/// in [`crate::continual::run_study`] evaluates a rehearsal environment.
pub struct DocumentCurriculum<'a> {
    cycles: &'a [FactBatch],
    anchors: &'a [FactBatch],
    tok: &'a QwenBpe,
    tmpl: &'a ChatTemplate,
    vocab: usize,
    shape: (usize, usize),
}

impl<'a> DocumentCurriculum<'a> {
    /// Validate the whole study up front and take the borrows.
    ///
    /// Panics, naming what failed, on any of:
    ///
    /// - no cycles, or no anchor suite ([`crate::continual::run_study`] refuses
    ///   a zero-length rehearsal pool under `Regime::Sft` anyway; failing here
    ///   says WHICH input was empty);
    /// - a `vocab` that does not span [`data::chat::ENDOFTEXT`] - every
    ///   dataset [`Self::write_sft_dataset`] writes terminates its records
    ///   with that id, so a model whose embedding is shorter would index off
    ///   the end of its own table;
    /// - anything [`train_probe_split`] refuses on any cycle's batch (the
    ///   held-out floor, and the probe/training id disjointness). Running B2's
    ///   check on every cycle HERE is what makes "explore/eval splits disjoint
    ///   by construction" a checked property of this curriculum rather than a
    ///   promise about how the batches were built.
    ///
    /// [`Curriculum::shape`] is DERIVED, over every row of every environment
    /// this curriculum can present: the longest presented text and the longest
    /// expected completion, in tokens. Deriving it is what makes it identical
    /// across cycles - the invariance the plasticity ratio depends on - instead
    /// of a number a caller has to keep true by hand.
    pub fn new(cycles: &'a [FactBatch], anchors: &'a [FactBatch], tok: &'a QwenBpe, tmpl: &'a ChatTemplate, vocab: usize) -> DocumentCurriculum<'a> {
        assert!(!cycles.is_empty(), "rl::document::DocumentCurriculum::new: a study needs at least one cycle's fact batch");
        assert!(
            !anchors.is_empty(),
            "rl::document::DocumentCurriculum::new: the anchor suite is empty - Regime::Sft mixes it into EVERY cycle's draw, \
             and without it cycle 1's training distribution is one document alone, which is the cue-independent shortcut \
             continual::rehearsal_pool exists to remove"
        );
        assert!(
            vocab > ENDOFTEXT as usize,
            "rl::document::DocumentCurriculum::new: vocab {vocab} does not span data::chat::ENDOFTEXT ({ENDOFTEXT}), the record \
             separator every prepare_chat_samples dataset carries - a model trained on one would index past its own embedding table"
        );
        for batch in cycles {
            let _ = train_probe_split(batch, tok);
        }

        let mut prompt_len = 0usize;
        let mut completion_len = 0usize;
        let envs: Vec<DocumentEnv<'_, QwenBpe>> = cycles
            .iter()
            .flat_map(|b| [DocumentEnv::new(b, FactSplit::Train, tok), DocumentEnv::new(b, FactSplit::Probe, tok)])
            .chain(anchors.iter().map(|b| DocumentEnv::new(b, FactSplit::Probe, tok)))
            .collect();
        for env in &envs {
            for i in 0..env.len() {
                let (presented, expected) = env.row(i);
                prompt_len = prompt_len.max(tok.encode(presented).len());
                completion_len = completion_len.max(tok.encode(expected).len());
            }
        }

        DocumentCurriculum { cycles, anchors, tok, tmpl, vocab, shape: (prompt_len, completion_len) }
    }

    /// Cycle `k`'s batch, panicking with the study's own bounds rather than a
    /// bare index panic - `run_study` drives `0..cfg.cycles`, and a config with
    /// more cycles than batches is a study that would silently repeat one.
    fn batch(&self, cycle: usize) -> &'a FactBatch {
        self.cycles
            .get(cycle)
            .unwrap_or_else(|| panic!("rl::document: cycle {cycle} requested but the curriculum holds {} fact batch(es)", self.cycles.len()))
    }

    /// The environment one [`SftSource`] draws its rows from: a cycle's
    /// TRAINING half, or an anchor suite's `(prompt, response)` half.
    fn env_for_source(&self, source: SftSource) -> DocumentEnv<'a, QwenBpe> {
        match source {
            SftSource::Cycle(k) => self.env_for(k),
            SftSource::Rehearsal(i) => {
                let batch = self
                    .anchors
                    .get(i)
                    .unwrap_or_else(|| panic!("rl::document: rehearsal source {i} but only {} anchor suite(s) exist", self.anchors.len()));
                DocumentEnv::new(batch, FactSplit::Probe, self.tok)
            }
        }
    }
}

/// One row as a supervised chat sample - see [`DocumentCurriculum`]'s doc
/// comment for why the two splits are shaped differently.
fn record(env: &DocumentEnv<'_, QwenBpe>, i: usize) -> ChatSample {
    let (presented, expected) = env.row(i);
    let messages = match env.split() {
        FactSplit::Train => vec![ChatMessage::assistant(presented, true)],
        FactSplit::Probe => vec![ChatMessage::user(presented), ChatMessage::assistant(expected, true)],
    };
    ChatSample { messages, tools: Vec::new() }
}

impl<'a> Curriculum for DocumentCurriculum<'a> {
    type Env = DocumentEnv<'a, QwenBpe>;
    type Ver = DocumentVerifier<'a, QwenBpe>;

    fn env_for(&self, cycle: usize) -> DocumentEnv<'a, QwenBpe> {
        DocumentEnv::new(self.batch(cycle), FactSplit::Train, self.tok)
    }

    fn eval_env_for(&self, cycle: usize) -> DocumentEnv<'a, QwenBpe> {
        DocumentEnv::new(self.batch(cycle), FactSplit::Probe, self.tok)
    }

    fn verifier(&self) -> DocumentVerifier<'a, QwenBpe> {
        DocumentVerifier::new(self.tok)
    }

    fn label(&self, cycle: usize) -> String {
        let batch = self.batch(cycle);
        format!("doc{:02} {} facts/{} probes", cycle + 1, batch.facts().len(), batch.triples().len())
    }

    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn rehearsal_envs(&self) -> Vec<DocumentEnv<'a, QwenBpe>> {
        (0..self.anchors.len()).map(|i| self.env_for_source(SftSource::Rehearsal(i))).collect()
    }

    fn rehearsal_len(&self) -> usize {
        self.anchors.len()
    }

    /// `n` records drawn uniformly over `sources` (and, within a source,
    /// uniformly over its rows), written through
    /// [`data::chat::prepare_chat_samples`] - so the per-token `train.mask.bin`
    /// companion file is what supervises the loss, and
    /// [`model::load_dataset`] aligns every sampled window to a record start
    /// on its own.
    ///
    /// The validation split is deliberately EMPTY: this dataset is consumed by
    /// [`crate::objective::mixture::Anchor`], which takes the training split
    /// only, and `run_study` runs its cycles at `eval_interval: 0`. Writing an
    /// unread val split would be a second, silently-unchecked draw.
    fn write_sft_dataset(&self, sources: &[SftSource], n: usize, seed: u64, out_dir: &Path) -> std::io::Result<()> {
        assert!(!sources.is_empty(), "rl::document::write_sft_dataset: no sources - there is nothing to draw from");
        assert!(n > 0, "rl::document::write_sft_dataset: a zero-record dataset trains nothing");
        let envs: Vec<DocumentEnv<'a, QwenBpe>> = sources.iter().map(|s| self.env_for_source(*s)).collect();
        let mut rng = Rng::new(seed);
        let samples: Vec<ChatSample> = (0..n)
            .map(|_| {
                let env = &envs[(rng.next_u64() % envs.len() as u64) as usize];
                let row = (rng.next_u64() % env.len() as u64) as usize;
                record(env, row)
            })
            .collect();
        data::chat::prepare_chat_samples(&samples, &[], self.tok, self.tmpl, self.vocab, out_dir).map_err(std::io::Error::other)
    }

    /// `None`: the token-level `train.mask.bin` [`Self::write_sft_dataset`]
    /// writes supersedes character-offset masking outright, and
    /// `model::load_dataset` ignores `mask_before` whenever that file is
    /// present. Stated rather than inherited from the trait default, because
    /// on this dataset shape the two are not interchangeable.
    fn sft_mask_before(&self) -> Option<char> {
        None
    }
}

/// Everything a document study chooses, and nothing it may not.
///
/// The regime and the gate are deliberately NOT fields: a document study is
/// [`Regime::Sft`] over [`document_gate_config`], both fixed before any run,
/// and a knob for either would be an invitation to tune after seeing the
/// result. The GRPO-only knobs of [`StudyConfig`] are likewise absent -
/// `run_study` never reads them under `Regime::Sft`, and a config field that
/// did nothing would make a run's own record of itself false.
///
/// [`Default`] IS the pre-registration: one cycle, `MIN_HELD_OUT_PROBES`
/// scored probes, `SftConfig::default`'s measured recipe. A caller normally
/// sets `cycles`/`work_dir` and leaves the rest; a caller that lowers
/// `eval_per_cycle` gets a report that says the run was not pre-registered
/// (see [`DocumentStudyReport::preregistered`]).
#[derive(Clone, Debug)]
pub struct DocumentStudyConfig {
    pub cycles: usize,
    pub steps_per_cycle: u32,
    /// Frozen probes SCORED per cycle - simultaneously the gate's held-out
    /// set and that cycle's permanent retention probe.
    pub eval_per_cycle: usize,
    pub seed: u64,
    /// The coin the null-gate arm flips. Separate from `seed` so the two arms
    /// differ in their GATE, not in their training stream.
    pub null_gate_seed: u64,
    pub sft: SftConfig,
    /// Arm 2, the per-cycle fresh-adapter control. Off by default: it doubles
    /// a study's training cost and answers a plasticity question, which is not
    /// what the MVP's document claim rests on.
    pub plasticity_control: bool,
    pub work_dir: PathBuf,
    pub verbose: bool,
}

impl Default for DocumentStudyConfig {
    fn default() -> Self {
        DocumentStudyConfig {
            cycles: 1,
            steps_per_cycle: 240,
            eval_per_cycle: MIN_HELD_OUT_PROBES,
            seed: 1,
            null_gate_seed: 2,
            sft: SftConfig::default(),
            plasticity_control: false,
            // Repo-relative, and normally overridden: a study writes a
            // checkpoint per cycle per arm.
            work_dir: PathBuf::from("out/document-study"),
            verbose: true,
        }
    }
}

/// Both arms of a document study, and whether the run was entitled to call
/// itself a result.
///
/// This is the surface a per-fact report (roadmap B8) hangs off: it holds the
/// whole [`StudyReport`] of each arm, not a collapsed scalar, so per-fact rows
/// can be derived from the same decodes the gate already made rather than from
/// a second scoring pass.
pub struct DocumentStudyReport {
    /// Arm 1: the real, gated loop. Its `b_base` is Arm 0 - the untrained
    /// base's own score on the first cycle's probes - and its
    /// `heldout_incumbent` column is that arm's zero-shot series.
    pub gated: StudyReport,
    /// The null-gate control ([`GatePolicy::CoinFlip`]): the real gate still
    /// runs and is still recorded, but a coin decides what carries forward. If
    /// the gated arm is not separated from this, the gate is decorative and
    /// the study's number carries no information about it.
    pub null_gate: StudyReport,
    pub eval_per_cycle: usize,
    /// Whether this run met the pre-registered held-out floor
    /// ([`MIN_HELD_OUT_PROBES`]). A run below it is a harness exercise, and
    /// [`Self::summary`] says so in words - reported rather than forbidden,
    /// because a study that cannot be run at all in a test is a study nothing
    /// checks.
    pub preregistered: bool,
}

impl DocumentStudyReport {
    /// `ACC(gated) - ACC(null gate)`: the separation that licenses any claim
    /// that the gate carried information at all.
    pub fn arm_separation(&self) -> f64 {
        self.gated.acc - self.null_gate.acc
    }

    /// Both arms' per-cycle tables plus the gated arm's retention matrix.
    pub fn table(&self) -> String {
        format!(
            "Arm 1 (gated)\n{}\n{}\nnull-gate control arm\n{}",
            self.gated.table(),
            self.gated.matrix_table(),
            self.null_gate.table()
        )
    }

    pub fn summary(&self) -> String {
        let prereg = if self.preregistered {
            format!("{} held-out probes per cycle: at or above the pre-registered floor of {MIN_HELD_OUT_PROBES}", self.eval_per_cycle)
        } else {
            format!(
                "{} held-out probes per cycle is BELOW the pre-registered floor of {MIN_HELD_OUT_PROBES}: this run exercises the \
                 harness and is NOT a pre-registered result",
                self.eval_per_cycle
            )
        };
        format!(
            "document study - Arm 0 baseline (untrained base on probe T1) {:.3}\n\nArm 1 (gated):\n{}\nnull-gate control arm:\n{}\narm separation (gated ACC - null-gate ACC) {:+.3}\n{prereg}\n",
            self.gated.b_base,
            self.gated.summary(),
            self.null_gate.summary(),
            self.arm_separation()
        )
    }
}

/// Run a document study: the gated arm and the null-gate control arm, both
/// [`continual::run_study`] over `curr` under [`Regime::Sft`] and
/// [`document_gate_config`], on the same frozen probes.
///
/// The control arms are not optional here, and that is this repo's own
/// convention rather than a preference: a learning claim with no control arm
/// does not count, and the existing null-gate diff on the synthetic curriculum
/// is what licensed the claim that the gate carries information at all. Arm 0
/// costs nothing extra - `run_study` already scores the untrained base on the
/// first cycle's probes ([`StudyReport::b_base`]) and already records each
/// cycle's zero-shot incumbent column.
///
/// Arm 3 (the joint-training capacity oracle, [`continual::joint_oracle`]) is
/// deliberately NOT run: it is post-MVP, and its verdict is only clean in one
/// direction anyway (see that function's own doc comment).
pub fn run_document_study<M: Model>(spec: &StudySpec, curr: &DocumentCurriculum, cfg: &DocumentStudyConfig) -> std::io::Result<DocumentStudyReport> {
    let arm = |gate_policy: GatePolicy, work_dir: PathBuf| StudyConfig {
        cycles: cfg.cycles,
        steps_per_cycle: cfg.steps_per_cycle,
        // `group_size`/`grad_accum`/`explore_temp` drive GRPO rollouts, which
        // this regime does not draw; `lr`/`min_lr` are GRPO's too - the SFT
        // path reads `SftConfig`'s own. Mirrored rather than left at some
        // unrelated value so a run's serialized config does not read as a
        // learning rate that was never applied.
        group_size: 1,
        grad_accum: 1,
        eval_per_cycle: cfg.eval_per_cycle,
        lr: cfg.sft.lr,
        min_lr: cfg.sft.min_lr,
        explore_temp: 1.0,
        // `run_study` asserts this is 0.0 under `Regime::Sft`: the equivalent
        // knob there is `SftConfig::rehearsal_weight`.
        replay_frac: 0.0,
        seed: cfg.seed,
        gate: document_gate_config(),
        gate_policy,
        plasticity_control: cfg.plasticity_control,
        work_dir,
        verbose: cfg.verbose,
        regime: Regime::Sft(cfg.sft.clone()),
    };

    let gated = continual::run_study::<M, _>(spec, curr, &arm(GatePolicy::Real, cfg.work_dir.join("gated")))?;
    let null_gate = continual::run_study::<M, _>(spec, curr, &arm(GatePolicy::CoinFlip { seed: cfg.null_gate_seed }, cfg.work_dir.join("null-gate")))?;

    Ok(DocumentStudyReport {
        gated,
        null_gate,
        eval_per_cycle: cfg.eval_per_cycle,
        preregistered: cfg.eval_per_cycle >= MIN_HELD_OUT_PROBES,
    })
}

/// One fact's own promote/reject verdict for a cycle, independent of the
/// cycle's aggregate gate decision (continuous-learning roadmap B8).
///
/// A batch of many facts trained and gated together produces ONE verdict for
/// the whole batch - "20 facts in, promote" can silently mean 15 landed and
/// 5 did not, and neither the user nor sven's ledger (`S6′`) can tell which
/// without this.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactVerdict {
    /// The fact statement, exactly as it was trained on.
    pub fact: String,
    /// `true` iff EVERY one of this fact's own probes scored a pass (reward
    /// `1.0`) on the candidate arm. Not a mean: a fact with three probes
    /// where two flip and one does not has NOT landed - averaging is exactly
    /// what would hide the one that failed.
    pub landed: bool,
}

/// Per-fact rows for one [`FactBatch`], from the candidate arm's own
/// `(task id, score)` pairs on its held-out probes - the SAME per-task
/// decodes [`crate::gate::gate`] itself was scored from, never a second
/// pass. `by_task` need not be in any particular order; task ids are
/// [`task_id`] applied to each triple's `probe_question`, exactly what
/// [`DocumentEnv::tasks`] stamps on the [`Task`] the gate decoded.
///
/// A fact's probes are the triples whose fact [`normalize`]s to the same
/// text, NOT the ones whose raw string matches: [`FactBatch::new`] collapses
/// the training rows under exactly that identity, so two spellings of one
/// fact are one row and one verdict. Matching raw here would drop every
/// probe an extractor happened to re-emit with different casing or spacing,
/// and a fact whose only failing probe was written that way would be
/// reported as LANDED - the concealment this whole function exists to stop.
///
/// Panics, naming the offending probe, if `by_task` does not cover every one
/// of `batch`'s probes: a fact this cycle never scored cannot be honestly
/// reported as landed OR failed.
pub fn fact_verdicts(batch: &FactBatch, by_task: &[(String, f64)]) -> Vec<FactVerdict> {
    batch
        .facts()
        .iter()
        .map(|fact| {
            let identity = normalize(fact);
            let landed = batch.triples().iter().filter(|t| normalize(&t.fact) == identity).all(|t| {
                let id = task_id(&t.probe_question);
                let (_, score) = by_task.iter().find(|(tid, _)| *tid == id).unwrap_or_else(|| {
                    panic!(
                        "rl::document::fact_verdicts: probe {:?} (fact {:?}) was not scored this cycle - `by_task` must \
                         cover every one of the batch's probes",
                        t.probe_question, fact
                    )
                });
                *score >= 1.0
            });
            FactVerdict { fact: fact.clone(), landed }
        })
        .collect()
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

    /// [`FactBatch::new`] deduplicates the trained rows by their NORMALISED
    /// text: two triples whose `fact` differs only in casing or internal
    /// whitespace are ONE training row and therefore ONE verdict. So
    /// [`fact_verdicts`] has to group its probes by the same identity. Group
    /// them by the RAW string instead and every probe written under the other
    /// spelling silently leaves the group - `all()` over the survivors then
    /// reports LANDED for a fact whose only failing probe was spelled that
    /// way, which is precisely the "15 landed, 5 did not" concealment this
    /// function exists to prevent.
    #[test]
    fn a_facts_verdict_covers_its_probes_however_that_facts_row_was_spelled() {
        let batch = FactBatch::new(vec![
            FactProbe {
                fact: "The 3rd relay closes at 13 volts".to_string(),
                probe_question: "when does the third relay close".to_string(),
                expected_answer: "13 volts".to_string(),
            },
            FactProbe {
                // The SAME fact, as an extractor re-emitted it for the second
                // probe: different capitalisation, a doubled space.
                fact: "the 3rd  relay closes at 13 volts".to_string(),
                probe_question: "what is the third relay threshold".to_string(),
                expected_answer: "13 volts".to_string(),
            },
        ]);
        assert_eq!(batch.facts().len(), 1, "two spellings of one fact are ONE trained row - that is what makes this a single verdict");

        let by_task: Vec<(String, f64)> =
            vec![(task_id(&batch.triples()[0].probe_question), 1.0), (task_id(&batch.triples()[1].probe_question), 0.0)];

        let verdicts = fact_verdicts(&batch, &by_task);
        assert_eq!(verdicts.len(), 1, "one verdict per DISTINCT fact");
        assert!(
            !verdicts[0].landed,
            "one of this fact's two probes scored zero, so the fact has NOT landed - which spelling that probe's row carried is not a property of the fact"
        );
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

    /// A batch of N facts trained and gated together produces ONE verdict for
    /// the whole batch: "20 facts in, promote" can silently mean 19 landed
    /// and 1 did not. This is the case that verdict would hide - the
    /// aggregate gate promotes (57 of 60 probes flip, an effect size and a
    /// significance no default or document config would reject), but one
    /// fact's own three probes still score zero post-training, and the
    /// per-fact report must name that fact anyway (continuous-learning
    /// roadmap B8).
    #[test]
    fn a_batch_promote_still_names_every_fact_that_did_not_land() {
        let batch = batch_of_twenty_facts();
        let failed_fact = batch.facts()[7].clone();

        let mut by_task: Vec<(String, f64)> = Vec::new();
        let mut candidate_scores: Vec<f64> = Vec::new();
        for t in batch.triples() {
            let score = if t.fact == failed_fact { 0.0 } else { 1.0 };
            by_task.push((task_id(&t.probe_question), score));
            candidate_scores.push(score);
        }
        let incumbent_scores = vec![0.0f64; candidate_scores.len()];

        let input = GateInput {
            candidate_scores: &candidate_scores,
            incumbent_scores: &incumbent_scores,
            anchor_candidate: 0.9,
            anchor_incumbent: 0.9,
            entropy_candidate: 2.0,
            entropy_incumbent: 2.0,
        };
        let report = gate(&input, &document_gate_config());
        assert_eq!(report.decision, Decision::Promote, "57 of 60 probes flipping must promote in aggregate under the document gate config");

        let verdicts = fact_verdicts(&batch, &by_task);
        assert_eq!(verdicts.len(), 20, "one verdict per DISTINCT fact");
        let not_landed: Vec<&str> = verdicts.iter().filter(|v| !v.landed).map(|v| v.fact.as_str()).collect();
        assert_eq!(
            not_landed,
            vec![failed_fact.as_str()],
            "the aggregate gate promoted, but the report must still name the one fact whose own probes did not land"
        );
    }
}
