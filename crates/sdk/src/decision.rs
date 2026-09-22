// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain::DecisionPipeline` - calibrated probabilities over options the
//! CALLER supplies, instead of text.
//!
//! ```no_run
//! use brain::{Choice, DecisionPipeline};
//! // `from_pretrained` STARTS A CHAIN, so it hands back a `Flow`; `finish`
//! // is the one place a chain's error surfaces. `builder(..).load()` is the
//! // same construction without the chain, for a caller that only wants to
//! // ask questions.
//! let mut pipe = DecisionPipeline::builder("/path/to/all-MiniLM-L6-v2").load()?;
//! let answer = pipe.choose(
//!     "I am still waiting on my card",
//!     "which banking intent does this message express",
//!     &["card arrival", "exchange rate", "pin blocked"],
//! )?;
//! println!("{} ({:.2} confident)", answer.choice, answer.confidence);
//! # Ok::<(), brain::Error>(())
//! ```
//!
//! Unlike every generative pipeline here, the OPTIONS are part of the call.
//! The same loaded model answers a three-option question and a
//! two-hundred-option one without reloading, because the output space lives in
//! the request rather than in a final layer.
//!
//! **Takes a path, not a hub id**, for the same reason
//! [`crate::TextGenerationPipeline`] does: this architecture has no model-store
//! `ArchSpec` yet, so there is nothing for a resolver to resolve against. A
//! decision model is also two things - an imported encoder and a trained head -
//! and a hub id names only the first.
//!
//! ## Two architectures, one surface
//!
//! `DecisionPipelineBuilder::load` sniffs `dir`'s own directory STRUCTURE
//! (see `resolve_decision_backend`) and routes to one of two backends,
//! mirroring [`crate::embedding`]'s `Backend`/`resolve_text_backend` pattern
//! (a private enum, one variant per architecture, every method matching on
//! it - not a trait object): `crates/decide`'s MiniLM/BERT-family encoder +
//! cross-attention head (the pre-existing, byte-identical path), or
//! `convaiinnovations/laya`'s ModernBERT-large trunk + its own decision head
//! (`crates/modernbert`).
//!
//! `choose`/`probability`/`set_question` and the [`Flow`]/[`Stages`]
//! `train`/`evaluate`/`save`/`turn` chain answer the same way on either
//! backend, because they mean the same thing for both. **One honest
//! asymmetry**: `choose`/`probability` pass exactly ONE question at a time,
//! so `Decide`'s state-encoded-once-many-questions-scored-cheaply economics
//! never actually shows up through this surface on the `Decide` arm either -
//! and on the Laya arm, every call re-encodes the whole packed
//! `(state, question)` sequence through the trunk from scratch (a full
//! ModernBERT-large forward per call), a real, different per-call cost from
//! the `Decide` arm, stated here rather than hidden.
//!
//! **`train_choices`/`save_head` are `Decide`-only for now.** A Laya-backed
//! pipeline returns a clear `Err` naming why (`crates/modernbert` has a
//! gradient-checked backward primitive but no optimizer loop, loss function,
//! or training CLI wiring yet - real future work, not a silent no-op).
//!
//! **[`DecisionLimits`] (`cap_rows`/`cap_slots`/`max_span`/`overlap`) is a
//! `decide`-specific windowing/packing concept and has NO effect on a
//! Laya-backed pipeline** - Laya always sizes and truncates to its own
//! checkpoint's `rl_agent_config.json` (`max_len`/`head_max_len`), read once
//! at import time, rather than a caller-supplied limit.
//!
//! **[`DecisionPipeline::inner`] returns `Option<&mut Decide>`, not
//! `&mut Decide`** - a Laya-backed pipeline has no `Decide` to hand back.

use std::path::Path;

pub use decide::banking77::Banking77;

use decide::banking77::OptionSampler;
use decide::decide::{Decide, Example, Limits};
use decide::loss::{softmax, LossConfig};
use decide::primitives::{confidence, Answer, Opt, Question};

pub use decide::banking77::Row;
pub use decide::decide::Limits as DecisionLimits;

use crate::flow::{EvalReport, Flow, Stages, TrainReport};
use crate::{Device, Error, Result};

/// What a `Choice` question answered.
#[derive(Clone, Debug)]
pub struct Choice {
    /// The highest-probability option, verbatim as it was supplied.
    pub choice: String,
    /// WHERE that option was in the list the caller supplied.
    ///
    /// Not derivable from [`Choice::choice`] in general: two options may carry
    /// the same text, and a caller that matched on the string would silently
    /// act on the first of them. A caller mapping the answer back onto its own
    /// data - an action, a row, an intent id - needs the position.
    pub index: usize,
    /// Every option's probability, in the order they were supplied. Always
    /// returned in full, so a caller who prefers the maximum probability or
    /// the margin over [`Choice::confidence`] can compute it.
    pub probabilities: Vec<(String, f32)>,
    /// How concentrated the distribution is, on `[0, 1]`, independent of how
    /// many options there were.
    pub confidence: f32,
}

/// The encoder arrives pretrained and the head does not, so they move at
/// different rates. One rate would either leave the head too slow to learn or
/// move the encoder fast enough to forget what it was imported for.
const ENCODER_LR: f32 = 2e-5;
const HEAD_LR: f32 = 1e-3;

/// Why [`DecisionPipeline::train_choices`]/[`DecisionPipeline::save_head`]
/// refuse on a Laya-backed pipeline - see this module's doc.
const LAYA_TRAINING_NOT_IMPLEMENTED: &str = "Laya training is not yet implemented in this SDK - \
    crates/modernbert has a gradient-checked backward primitive (a seeded backward through the head \
    into the trunk) but no optimizer loop, loss function, or training CLI wiring yet; only inference \
    (choose/probability) is available for a Laya-backed DecisionPipeline";

pub struct DecisionPipeline {
    backend: Backend,
    /// The option set and question the last training stage used, so the
    /// interactive stages need no second copy of them.
    last_options: Vec<String>,
    last_instructions: String,
    last_eval: Vec<(String, usize)>,
}

/// Which architecture a loaded [`DecisionPipeline`] is actually running -
/// see this module's doc and [`resolve_decision_backend`]'s own doc for how
/// a directory resolves to one or the other. Mirrors
/// [`crate::embedding`]'s own `Backend`/`resolve_text_backend` pattern (an
/// enum with one variant per architecture, every method matching on it) -
/// not a trait object, per this milestone's design review.
enum Backend {
    /// `crates/decide`'s MiniLM/BERT-family encoder + cross-attention head -
    /// the pre-existing, unchanged path ([`load_decide`]).
    Decide(Decide),
    /// `convaiinnovations/laya`'s ModernBERT-large trunk + its own decision
    /// head (`crates/modernbert`). Boxed: [`LayaBackend`] carries two whole
    /// on-device models, and would otherwise make every `Backend::Decide`
    /// significantly larger than the `Decide` it actually holds.
    Laya(Box<LayaBackend>),
}

/// The Laya (`convaiinnovations/laya`) backend: a ModernBERT-large trunk
/// plus its own decision head, both frozen/inference-only at this SDK layer
/// (see [`DecisionPipeline::train_choices`]'s Laya arm for why training is
/// out of scope for this milestone).
///
/// **`choose`/`probability` pass exactly ONE question at a time** (the same
/// SDK contract the `Decide` arm has), so `crates/modernbert`'s trunk is
/// re-encoded from scratch on every call - see this module's doc, "Two
/// architectures, one surface", for the honest cost statement this implies.
struct LayaBackend {
    enc: modernbert::ModernBert,
    head: modernbert::LayaHead,
    tok: data::qwen_tokenizer::QwenBpe,
    cfg: modernbert::ModernBertConfig,
    /// `rl_agent_config.json`'s own `max_len`/`head_max_len` - Laya ALWAYS
    /// truncates to these, read once at import time; see
    /// [`DecisionPipelineBuilder::limits`]'s own doc for why a
    /// caller-supplied [`Limits`] has no effect on this arm.
    max_len: u32,
    head_max_len: u32,
    /// Per-qtype (`choice`/`score`/`noul`, [`modernbert::QType::index`]
    /// order) serving calibration scalar, `rl_agent_config.json`'s own
    /// `temperature` table. The per-cardinality-bucket override
    /// (`rl_agent_config.json`'s `temperature_by_options`, keyed like
    /// `"choice:3-5"`) is deliberately NOT applied here - a real, if minor,
    /// gap: argmax (so `Choice::choice`/`Choice::index`) is unaffected by
    /// any positive temperature, but `Choice::probabilities`/
    /// `Choice::confidence` and `probability`'s own return value are
    /// calibrated only to the coarser per-qtype scalar, not the finer
    /// per-cardinality table the real `rl_agent_api.py` serving path also
    /// consults.
    temperature: Vec<f32>,
}

impl LayaBackend {
    fn temperature_for(&self, qtype: modernbert::QType) -> f32 {
        self.temperature.get(qtype.index() as usize).copied().unwrap_or(1.0)
    }

    /// Run one packed `(state, question)` call through the trunk and head,
    /// returning raw (unsoftmaxed) option logits. The act-head's own output
    /// is discarded here - no SDK surface exposes it yet (M7's job, alongside
    /// the sample that actually needs it).
    fn score(&mut self, state: &str, q: &modernbert::Question) -> Result<Vec<f32>> {
        let state = modernbert::State::Str(state.to_string());
        let (ids, markers) =
            modernbert::build_sequence(&self.tok, &self.cfg, &state, q, self.max_len, self.head_max_len, None, false);
        if ids.is_empty() || markers.is_empty() {
            return Err(Error::Backend("laya: build_sequence produced no option markers for this question".into()));
        }
        let rows = ids.len() as u32;
        let spans = [(0u32, rows)];
        self.enc.set_batch(&ids, &spans);
        self.enc.forward();
        // Cross-`Gpu`-handle synchronization: `enc`/`head` hold SEPARATE `Gpu`
        // handles onto the same device (`gpu.share()`), so a `submit` on one
        // is not ordered against a `submit` on the other - `crates/modernbert`
        // M5's own gradcheck probe hit exactly this (a missing wait here reads
        // a stale/partial hidden state and produces plausible-but-wrong
        // numbers, not a crash). MUST NOT be removed - see
        // `modernbert::ModernBert::poll_wait`'s own doc and
        // `decide::decide::Decide::run_packed`'s identical, independently
        // discovered precedent.
        self.enc.poll_wait();

        let marker_rows: Vec<u32> = markers.iter().map(|&m| m as u32).collect();
        let qtype = [q.qtype().index()];
        let arity = [markers.len()];
        let hidden = self.enc.hidden_buf();
        self.head.set_call(hidden, &spans, &qtype, &marker_rows, &arity);
        let (logits, _act_logits) = self.head.forward();
        self.head.poll_wait();
        Ok(logits)
    }

    fn choose(&mut self, state: &str, instructions: &str, options: &[&str]) -> Result<Choice> {
        if options.is_empty() {
            return Err(Error::MissingArgument("choose needs at least one option".into()));
        }
        let q = modernbert::Question::Choice {
            ins: instructions.to_string(),
            options: options.iter().map(|o| (o.to_string(), None)).collect(),
        };
        let raw = self.score(state, &q)?;
        let t = self.temperature_for(modernbert::QType::Choice);
        let p = softmax(&scaled(&raw, t));
        let index = p.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0);
        Ok(Choice {
            choice: options[index].to_string(),
            index,
            probabilities: options.iter().map(|o| o.to_string()).zip(p.iter().copied()).collect(),
            confidence: confidence(&p),
        })
    }

    fn probability(&mut self, state: &str, proposition: &str) -> Result<f32> {
        let q = modernbert::Question::Noul { ins: proposition.to_string(), false_text: None, true_text: None };
        let raw = self.score(state, &q)?;
        if raw.len() != 2 {
            return Err(Error::Backend(format!(
                "laya: a noul question must score exactly 2 options (false, true), got {}",
                raw.len()
            )));
        }
        let t = self.temperature_for(modernbert::QType::Noul);
        let p = softmax(&scaled(&raw, t));
        // render_options() puts the "true" reading at index 1 - see
        // modernbert::sequence's own module doc.
        Ok(p[1])
    }
}

/// Divide raw scores by a serving-time temperature before the host softmax -
/// a no-op (returns `scores` unchanged) on a non-finite or non-positive `t`,
/// which a malformed `rl_agent_config.json` should not be able to turn into
/// a NaN/Inf softmax input.
fn scaled(scores: &[f32], t: f32) -> Vec<f32> {
    if t.is_finite() && t > 0.0 {
        scores.iter().map(|&s| s / t).collect()
    } else {
        scores.to_vec()
    }
}

/// Load the BANKING77 intent dataset from a directory holding
/// `categories.json`, `train.csv` and `test.csv`.
///
/// Here rather than in the sample that uses it because a sample may depend on
/// no brain crate except this one.
pub fn banking77(dir: impl AsRef<std::path::Path>) -> Result<Banking77> {
    Banking77::load(dir.as_ref()).map_err(Error::Backend)
}

pub use decide::banking77::Banking77 as Banking77Data;
pub use decide::banking77::IntentSplit;
/// The deterministic PRNG the option sampler draws with. Re-exported so a
/// caller can reproduce an evaluation exactly.
pub use data::rng::Rng;

/// Hold `n_unseen` intents back from training, chosen by a fixed seed.
///
/// The point of the split is what it measures: a model trained only on the
/// rest and then asked to pick a held-back intent BY NAME has to have read the
/// option text, because it has never seen an example of that intent and there
/// is no index for it to have learned.
pub fn intent_holdout(n_categories: usize, n_unseen: usize, seed: u64) -> IntentSplit {
    IntentSplit::holdout(n_categories, n_unseen, seed)
}

/// Draw the option set one example is scored against: `gold` plus a random
/// number of distractors from `pool`, gold at a random position.
///
/// The same sampler training uses, exposed so an evaluation can draw from the
/// same distribution. Scoring against all the options when training saw a
/// handful measures a different, easier task.
pub fn draw_options(gold: usize, pool: &[usize], rng: &mut Rng) -> (Vec<usize>, usize) {
    OptionSampler::default().draw(gold, pool, rng)
}

impl std::fmt::Debug for DecisionPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionPipeline").finish_non_exhaustive()
    }
}

impl DecisionPipeline {
    /// Load an encoder checkpoint directory (`config.json`, `model.safetensors`
    /// and `tokenizer.json`), with a freshly initialized head, and START A
    /// STAGE CHAIN on it.
    ///
    /// Returns a [`Flow`] rather than a bare pipeline so the chain reads as
    /// one expression: a load failure becomes the flow's error and every later
    /// stage skips, so there is one error site at `finish` instead of one per
    /// stage. `DecisionPipeline::builder(..).load()` is the same thing without
    /// the chain.
    ///
    /// An untrained head answers, and answers badly: use
    /// [`DecisionPipelineBuilder::head`] to supply trained weights, or
    /// [`Flow::train`] to make some.
    pub fn from_pretrained(dir: impl AsRef<str>) -> Flow<DecisionPipeline> {
        Flow::new(DecisionPipeline::builder(dir).load())
    }

    pub fn builder(dir: impl AsRef<str>) -> DecisionPipelineBuilder {
        DecisionPipelineBuilder {
            dir: dir.as_ref().to_string(),
            head: None,
            device: Device::default(),
            limits: Limits::default(),
            seed: 0,
        }
    }

    /// Pick one of `options` for `state`, given what the question asks.
    pub fn choose(&mut self, state: &str, instructions: &str, options: &[&str]) -> Result<Choice> {
        match &mut self.backend {
            Backend::Decide(model) => {
                let q = Question::Choice {
                    instructions: instructions.to_string(),
                    options: options.iter().map(|o| Opt::new(*o)).collect(),
                };
                let mut answers = model.decide(state, std::slice::from_ref(&q)).map_err(Error::Backend)?;
                match answers.pop() {
                    Some(Answer::Choice { choice, probabilities, confidence }) => {
                        let index = probabilities
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1 .1.total_cmp(&b.1 .1))
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        Ok(Choice { choice, index, probabilities, confidence })
                    }
                    _ => Err(Error::Backend("the model did not return a choice".into())),
                }
            }
            Backend::Laya(l) => l.choose(state, instructions, options),
        }
    }

    /// The probability that a proposition holds, on `[0, 1]`.
    pub fn probability(&mut self, state: &str, proposition: &str) -> Result<f32> {
        match &mut self.backend {
            Backend::Decide(model) => {
                let q = Question::Noul { instructions: proposition.to_string(), yes: None, no: None };
                let mut answers = model.decide(state, std::slice::from_ref(&q)).map_err(Error::Backend)?;
                match answers.pop() {
                    Some(Answer::Noul { noul }) => Ok(noul),
                    _ => Err(Error::Backend("the model did not return a probability".into())),
                }
            }
            Backend::Laya(l) => l.probability(state, proposition),
        }
    }

    /// Tell this pipeline what question its interactive stages ask, and over
    /// which options.
    ///
    /// Needed when a model is loaded from trained weights rather than trained
    /// in this process: the option set lives in the REQUEST, not in the
    /// checkpoint, so weights alone do not say what the model is deciding
    /// between. Training sets the same fields as a side effect.
    pub fn set_question(&mut self, instructions: impl Into<String>, options: Vec<String>) {
        self.last_instructions = instructions.into();
        self.last_options = options;
    }

    /// Fine-tune on labelled examples: `(text, label)` pairs plus the option
    /// text every label maps to.
    ///
    /// Each step scores its example against a random SUBSET of the options,
    /// always containing the correct one, at a random position. That is the
    /// whole difference between training a decision model and training a
    /// classifier: a model always shown the same option list in the same order
    /// can reach the right answer from the POSITION, never having to read an
    /// option at all - and it then cannot answer a question whose options it
    /// has not seen, which is the one thing this model is for.
    ///
    /// Returns the mean loss over the last tenth of the run.
    pub fn train_choices(
        &mut self,
        examples: &[(&str, usize)],
        options: &[String],
        instructions: &str,
        steps: usize,
        seed: u64,
        log: &mut dyn FnMut(usize, f32),
    ) -> Result<f32> {
        let model = match &mut self.backend {
            Backend::Decide(model) => model,
            // Real, deliberate gap - not a silent no-op: see
            // LAYA_TRAINING_NOT_IMPLEMENTED and this module's doc.
            Backend::Laya(_) => return Err(Error::Backend(LAYA_TRAINING_NOT_IMPLEMENTED.into())),
        };
        if examples.is_empty() {
            return Err(Error::MissingArgument("train_choices needs at least one example".into()));
        }
        let pool: Vec<usize> = (0..options.len()).collect();
        let sampler = OptionSampler::default();
        let loss_cfg = LossConfig::cross_entropy();
        let mut rng = data::rng::Rng::new(seed);
        let tail = (steps / 10).max(1);
        let mut tail_sum = 0.0f32;

        for step in 0..steps {
            let (text, label) = examples[(rng.next_u64() % examples.len() as u64) as usize];
            if label >= options.len() {
                return Err(Error::MissingArgument(format!(
                    "example label {label} has no option text ({} supplied)",
                    options.len()
                )));
            }
            let (drawn, gold) = sampler.draw(label, &pool, &mut rng);
            let q = Question::Choice {
                instructions: instructions.to_string(),
                options: drawn.iter().map(|&i| Opt::new(options[i].clone())).collect(),
            };
            let l = model
                .train_step(&Example { state: text, question: &q, gold }, &loss_cfg, ENCODER_LR, HEAD_LR)
                .map_err(Error::Backend)?;
            log(step, l);
            if step >= steps.saturating_sub(tail) {
                tail_sum += l / tail as f32;
            }
        }
        Ok(tail_sum)
    }

    /// Write the trained head to a brain `.safetensors`, for
    /// [`DecisionPipelineBuilder::head`] to load back.
    ///
    /// `Err` on a Laya-backed pipeline - see [`LAYA_TRAINING_NOT_IMPLEMENTED`]
    /// and this module's doc.
    pub fn save_head(&self, path: impl AsRef<str>) -> Result<()> {
        match &self.backend {
            Backend::Decide(model) => model.save_head(path.as_ref()).map_err(Error::Backend),
            Backend::Laya(_) => Err(Error::Backend(LAYA_TRAINING_NOT_IMPLEMENTED.into())),
        }
    }

    /// The underlying `Decide` model, for training and for the question
    /// types this three-line surface does not cover.
    ///
    /// `None` on a Laya-backed pipeline, which has no `Decide` to hand back -
    /// a real, deliberate break from this method's pre-M6 signature
    /// (`&mut Decide`), accepted as part of this milestone's design (see
    /// this module's doc).
    pub fn inner(&mut self) -> Option<&mut Decide> {
        match &mut self.backend {
            Backend::Decide(model) => Some(model),
            Backend::Laya(_) => None,
        }
    }
}

/// What a decision model needs in order to train: labelled examples, the
/// option text each label maps to, and what the question asks.
///
/// Owned rather than borrowed so a chain can be written as one expression
/// without the caller keeping every intermediate alive.
#[derive(Clone, Debug, Default)]
pub struct TrainSpec {
    pub examples: Vec<(String, usize)>,
    pub options: Vec<String>,
    pub instructions: String,
    pub steps: usize,
    pub seed: u64,
    /// Held out for [`Flow::evaluate`]. Empty means evaluation reports nothing
    /// rather than inventing a split.
    pub eval: Vec<(String, usize)>,
}

impl TrainSpec {
    pub fn new(instructions: impl Into<String>) -> TrainSpec {
        TrainSpec { instructions: instructions.into(), steps: 600, ..TrainSpec::default() }
    }

    pub fn examples(mut self, examples: Vec<(String, usize)>) -> TrainSpec {
        self.examples = examples;
        self
    }

    pub fn options(mut self, options: Vec<String>) -> TrainSpec {
        self.options = options;
        self
    }

    pub fn eval(mut self, eval: Vec<(String, usize)>) -> TrainSpec {
        self.eval = eval;
        self
    }

    pub fn steps(mut self, steps: usize) -> TrainSpec {
        self.steps = steps;
        self
    }

    pub fn seed(mut self, seed: u64) -> TrainSpec {
        self.seed = seed;
        self
    }
}

impl Stages for DecisionPipeline {
    type TrainSpec = TrainSpec;

    fn describe(&self) -> String {
        let steps = match &self.backend {
            Backend::Decide(model) => model.steps_taken(),
            // No training loop exists for this arm yet (see this module's
            // doc), so there is nothing to have counted.
            Backend::Laya(_) => 0,
        };
        format!("decision model, {} options in the last call, {steps} training steps so far", self.last_options.len())
    }

    fn run_train(&mut self, spec: &TrainSpec, log: &mut dyn FnMut(usize, f32)) -> Result<TrainReport> {
        let ex: Vec<(&str, usize)> = spec.examples.iter().map(|(t, l)| (t.as_str(), *l)).collect();
        let final_loss =
            self.train_choices(&ex, &spec.options, &spec.instructions, spec.steps, spec.seed, log)?;
        // Remembered so `evaluate` and `ask` need no second copy of the
        // question: a chain should not make the caller repeat itself.
        self.last_options = spec.options.clone();
        self.last_instructions = spec.instructions.clone();
        self.last_eval = spec.eval.clone();
        Ok(TrainReport { steps: spec.steps, final_loss, seconds: 0.0 })
    }

    fn run_eval(&mut self) -> Result<EvalReport> {
        if self.last_eval.is_empty() {
            return Ok(EvalReport::default());
        }
        let options = self.last_options.clone();
        let refs: Vec<&str> = options.iter().map(String::as_str).collect();
        let instructions = self.last_instructions.clone();
        let eval = self.last_eval.clone();
        let (mut hit, mut confidence) = (0usize, 0.0f32);
        for (text, label) in &eval {
            let a = self.choose(text, &instructions, &refs)?;
            confidence += a.confidence / eval.len() as f32;
            if a.choice == options[*label] {
                hit += 1;
            }
        }
        Ok(EvalReport {
            accuracy: hit as f32 / eval.len() as f32,
            items: eval.len(),
            notes: vec![
                ("chance".into(), 1.0 / options.len().max(1) as f32),
                ("mean confidence".into(), confidence),
            ],
        })
    }

    fn run_save(&self, path: &str) -> Result<()> {
        self.save_head(path)
    }

    fn set_output_space(&mut self, instructions: &str, options: Vec<String>) -> Result<()> {
        self.set_question(instructions, options);
        Ok(())
    }

    fn run_turn(&mut self, input: &str) -> Result<String> {
        if self.last_options.is_empty() {
            return Err(Error::MissingArgument(
                "this pipeline has no option set yet - train it, or call `choose` with one".into(),
            ));
        }
        let options = self.last_options.clone();
        let refs: Vec<&str> = options.iter().map(String::as_str).collect();
        let instructions = self.last_instructions.clone();
        let a = self.choose(input, &instructions, &refs)?;
        let mut top = a.probabilities.clone();
        top.sort_by(|x, y| y.1.total_cmp(&x.1));
        let mut out = format!("  -> {}  (confidence {:.2})", a.choice, a.confidence);
        for (name, p) in top.iter().take(3) {
            out.push_str(&format!("\n       {p:>6.3}  {name}"));
        }
        if a.confidence < 0.3 {
            // The reason a decision model returns a distribution and not a
            // label: the CALLER decides what is confident enough to act on.
            out.push_str("\n       (low confidence - a real system would escalate this one)");
        }
        Ok(out)
    }
}

pub struct DecisionPipelineBuilder {
    dir: String,
    head: Option<String>,
    device: Device,
    limits: Limits,
    seed: u64,
}

impl DecisionPipelineBuilder {
    /// Trained head weights, as written by `decide`'s training loop. Without
    /// this the head is random and the model's answers are noise.
    pub fn head(mut self, path: impl AsRef<str>) -> DecisionPipelineBuilder {
        self.head = Some(path.as_ref().to_string());
        self
    }

    pub fn device(mut self, device: Device) -> DecisionPipelineBuilder {
        self.device = device;
        self
    }

    /// How large a request this model is built for. Raising these costs device
    /// memory at build time, not per call.
    ///
    /// **Ignored entirely on a Laya-backed pipeline** - Laya always sizes and
    /// truncates to its own checkpoint's `rl_agent_config.json`
    /// (`max_len`/`head_max_len`), read at import time, rather than a
    /// caller-supplied limit. See this module's doc.
    pub fn limits(mut self, limits: Limits) -> DecisionPipelineBuilder {
        self.limits = limits;
        self
    }

    /// Seed for the head's initialization, when no trained head is supplied.
    pub fn seed(mut self, seed: u64) -> DecisionPipelineBuilder {
        self.seed = seed;
        self
    }

    /// Build a model ready to answer. Trainable on the `Decide` arm, so the
    /// same object a caller loads is the one it can fine-tune (see this
    /// module's doc for the Laya arm's own, narrower, inference-only
    /// contract).
    pub fn load(self) -> Result<DecisionPipeline> {
        let backend = resolve_decision_backend(&self.dir, self.head.as_deref(), &self.device, self.limits, self.seed)?;
        Ok(DecisionPipeline { backend, last_options: Vec::new(), last_instructions: String::new(), last_eval: Vec::new() })
    }
}

/// Build a trainable [`Decide`] from an encoder checkpoint directory.
///
/// Shared by every pipeline over this architecture, because the loading is
/// the part they genuinely have in common - the checkpoint, the tokenizer and
/// the device are the same three things whatever question the head is later
/// asked. What differs is only what gets trained on top.
pub(crate) fn load_decide(
    dir: &str,
    head: Option<&str>,
    device: &Device,
    limits: Limits,
    seed: u64,
) -> Result<Decide> {
    let path = Path::new(dir);
    let cfg_json = std::fs::read_to_string(path.join("config.json"))
        .map_err(|e| Error::Backend(format!("read {dir}/config.json: {e}")))?;
    let cfg = decide::import::config_from_hf(&cfg_json).map_err(Error::Backend)?;
    let weights = path.join("model.safetensors");
    let tensors = checkpoint::safetensors::read(
        weights.to_str().ok_or_else(|| Error::Backend("non-UTF-8 weights path".into()))?,
    )
    .map_err(|e| Error::Backend(format!("read {}: {e}", weights.display())))?;
    let enc_init = decide::import::brain_init_from_hf(tensors, &cfg).map_err(Error::Backend)?;

    let tok_path = path.join("tokenizer.json");
    let tok = data::wordpiece::WordPiece::from_file(
        tok_path.to_str().ok_or_else(|| Error::Backend("non-UTF-8 tokenizer path".into()))?,
    )
    .map_err(Error::Backend)?;

    let head_init = match head {
        Some(p) => {
            let t = checkpoint::safetensors::read(p).map_err(|e| Error::Backend(format!("read {p}: {e}")))?;
            t.into_iter().map(|x| (x.name, x.data)).collect()
        }
        None => decide::init::init_head(&cfg, seed),
    };

    crate::device::resolve(device)?;
    let gpu = gpu_core::Gpu::new(decide::kern::PIPELINES);
    let mut model = Decide::new_on(gpu, cfg, tok, limits, &enc_init, &head_init, true);
    model.set_provenance(decide::decide::Provenance {
        base: base_reference(path),
        ..Default::default()
    });
    Ok(model)
}

/// Resolve `dir` to a [`Backend`]: [`is_laya_dir`] sniffs the directory
/// STRUCTURE and routes to [`load_laya`] when it matches, else falls
/// through to the pre-existing [`load_decide`] path unchanged - mirroring
/// [`crate::embedding::resolve_text_backend`]'s own two-way shape one level
/// up (try the more specific classification first, fall back to the
/// default).
fn resolve_decision_backend(dir: &str, head: Option<&str>, device: &Device, limits: Limits, seed: u64) -> Result<Backend> {
    if is_laya_dir(Path::new(dir)) {
        return Ok(Backend::Laya(Box::new(load_laya(dir, head, device, seed)?)));
    }
    Ok(Backend::Decide(load_decide(dir, head, device, limits, seed)?))
}

/// Sniff `dir`'s own directory STRUCTURE to tell a Laya checkpoint from a
/// `decide`-shaped one - they do NOT share a root `config.json` shape at
/// all, so classifying by "does `config.json` parse as one or the other"
/// (the naive approach) is not available here. `decide::import::
/// config_from_hf` expects a root `config.json` naming `model_type: "bert"`.
/// A Laya checkpoint (what `modernbert::import_dir` reads) has NO root
/// `config.json` whatsoever - only a NESTED `encoder/config.json`
/// (`model_type: "modernbert"`), plus a Laya-specific marker file,
/// `rl_agent_config.json`, at the root. Both conditions are checked (not just
/// the marker file's existence) so an unrelated directory that happens to
/// carry a stray `rl_agent_config.json` cannot be misrouted.
fn is_laya_dir(dir: &Path) -> bool {
    if !dir.join("rl_agent_config.json").is_file() {
        return false;
    }
    let Ok(cfg_text) = std::fs::read_to_string(dir.join("encoder").join("config.json")) else {
        return false;
    };
    let Ok(cfg_v) = serde_json::from_str::<serde_json::Value>(&cfg_text) else {
        return false;
    };
    cfg_v.get("model_type").and_then(serde_json::Value::as_str) == Some("modernbert")
}

/// Build a [`LayaBackend`] from a `convaiinnovations/laya`-shaped checkpoint
/// directory - the arm [`resolve_decision_backend`] routes to when
/// [`is_laya_dir`] recognizes `dir`.
///
/// `limits` plays no part here (see [`DecisionPipelineBuilder::limits`]'s own
/// doc) - Laya always sizes and truncates to `rl_agent_config.json`'s own
/// `max_len`/`head_max_len`, read at import time. `seed` likewise: unlike
/// `load_decide`, there is no fresh-random-head path here to seed - a
/// Laya-backed pipeline has exactly one head, the one the checkpoint shipped
/// (see [`LAYA_TRAINING_NOT_IMPLEMENTED`] for why no trained-head file format
/// exists yet either).
fn load_laya(dir: &str, head: Option<&str>, device: &Device, _seed: u64) -> Result<LayaBackend> {
    if head.is_some() {
        return Err(Error::Backend(
            "a Laya-backed DecisionPipeline has no trained-head file format yet (train_choices/save_head \
             are not implemented) - do not call .head(path) when loading a Laya checkpoint directory"
                .into(),
        ));
    }
    let ckpt = modernbert::import_dir(dir).map_err(Error::Backend)?;

    let tok_path = Path::new(dir).join("tokenizer").join("tokenizer.json");
    let tok = data::qwen_tokenizer::QwenBpe::from_file(
        tok_path.to_str().ok_or_else(|| Error::Backend("non-UTF-8 tokenizer path".into()))?,
    )
    .map_err(Error::Backend)?;

    crate::device::resolve(device)?;
    let gpu = gpu_core::Gpu::new(modernbert::kern::PIPELINES);
    let max_len = ckpt.rl.max_len;
    let head_max_len = ckpt.rl.head_max_len;
    let enc = modernbert::ModernBert::new_on(gpu.share(), ckpt.cfg.clone(), max_len, max_len, &ckpt.encoder_init);
    // One question per call (see this module's "Shared honestly" doc), but
    // the full published option range (`decide::primitives::MAX_OPTIONS`) so
    // a Laya-backed pipeline answers exactly as wide a `choose` as the
    // `Decide` arm does.
    let cap_markers = decide::primitives::MAX_OPTIONS as u32;
    let laya_head =
        modernbert::LayaHead::new_on(gpu, ckpt.laya_cfg.clone(), max_len, max_len, cap_markers, 1, &ckpt.head_init);

    Ok(LayaBackend { enc, head: laya_head, tok, cfg: ckpt.cfg, max_len, head_max_len, temperature: ckpt.rl.temperature })
}

/// The encoder directory, as the Hugging Face reference it came from.
///
/// A head is an adapter and an adapter that does not name its base is
/// unloadable. `brain pull` lays a model down under `<vendor>/<repo>`, so the
/// last two components of the directory ARE the reference; anything else
/// falls back to the leaf, which is at least a name someone can search for.
fn base_reference(dir: &Path) -> String {
    let parts: Vec<&str> = dir
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .filter(|s| !s.is_empty() && *s != "/")
        .collect();
    match parts.len() {
        0 => String::new(),
        1 => parts[0].to_string(),
        n => format!("{}/{}", parts[n - 2], parts[n - 1]),
    }
}
