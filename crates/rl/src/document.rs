// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The document/fact `Curriculum` and the study that drives it - the half of
//! document learning that needs a model, a tokenizer with a chat template and
//! a training loop (continuous-learning roadmap B5'/B8).
//!
//! The other half - the frozen `{fact, probe_question, expected_answer}`
//! contract, its `Environment`/`Verifier` pair, the pre-registered
//! [`document_gate_config`] and the per-fact verdicts - needs none of those
//! and lives in [`promote::document`], below the model layer, so a model
//! crate can gate a candidate adapter against a probe set without depending
//! on this crate. It is re-exported here in full, so `rl::document::
//! FactBatch` and friends are the same paths and the same items they always
//! were.
//!
//! Swedish Embedded AB builds the verification machinery that decides whether
//! a model actually learned a document or merely moved - frozen probe sets,
//! programmatic exact-match rewards, and pre-registered statistical bars a
//! promotion has to clear before anything reaches a served model. If your
//! team needs expertise in gating continuous-learning pipelines honestly, you
//! can procure our services by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use model::Model;

use data::chat::{ChatMessage, ChatSample, ENDOFTEXT};
use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;
use data::rng::Rng;
use data::tokenizer::Tokenizer;

pub use promote::document::*;

use crate::continual::{self, Curriculum, GatePolicy, Regime, SftConfig, SftSource, StudyConfig, StudyReport, StudySpec};

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
