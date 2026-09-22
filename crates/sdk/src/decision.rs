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
//! backend, because they mean the same thing for both.
//!
//! ## Two shapes of call
//!
//! [`DecisionPipeline::choose`] and [`DecisionPipeline::probability`] are the
//! one-question convenience: a `Choice` over plain option strings, or the
//! probability of a proposition, about a state that is already a string.
//!
//! [`DecisionPipeline::decide`] is the full request both backends implement -
//! several typed [`Question`]s about ONE [`State`], answered positionally.
//! It is the only surface that reaches [`Question::Score`] at all, the only
//! one where a `Choice`'s options carry the descriptions the model is meant
//! to read, and the only one that takes structured state ([`State::Json`],
//! key order preserved) rather than a string the caller flattened itself.
//! `samples/decision/json` is a JEV-style JSON endpoint built on it.
//!
//! **One honest asymmetry, on both shapes**: `choose`/`probability` pass
//! exactly one question, and on the Laya arm even `decide` does - every call
//! re-encodes the whole packed `(state, question)` sequence through the trunk
//! from scratch (a full ModernBERT-large forward per question), where the
//! `Decide` arm encodes the state once and scores every question's options
//! against that one encoding. That is a real, different per-call cost, stated
//! here rather than hidden.
//!
//! **`train_choices`/`save_head` work on BOTH arms**, and the option
//! sampling, the `(text, label)` contract, the per-step `log` callback and
//! the mean-tail-loss return are shared. What genuinely differs is the
//! objective and what moves:
//!
//! | | `Decide` (MiniLM) | Laya (ModernBERT-large) |
//! | --- | --- | --- |
//! | objective | [`rlcd::scoring`] cross-entropy/focal/Brier, minimized directly | [`rlcd::reinforce`] REINFORCE with a group-mean baseline over [`rlcd::proper`]'s strictly proper reward |
//! | what learns | encoder (2e-5) + head (1e-3) | the decision head only |
//! | act/escalate head | none exists | NOT trained - no public source defines its objective |
//!
//! Each arm trains against the rule its own released weights were fitted
//! under, which is why this is two objectives rather than one. **The losses
//! are therefore not comparable between arms**; only each arm's own trend
//! over a run is.
//!
//! The Laya trunk is held fixed, and at 395M parameters that is a memory
//! decision before it is a tuning one - see `modernbert::decision`'s module
//! doc and [`DecisionPipeline::save_head`], which refuses to write a
//! head-only file for a model whose trunk moved.
//!
//! [`Stages::supports_training`] is `true` on both arms now. It is kept
//! rather than removed because it is the general seam a future
//! pretrained-only backend reuses, and a caller that checks it before
//! [`Flow::train`] is doing the right thing whatever is behind it.
//!
//! **[`DecisionPipeline::route`] is the mirror-image asymmetry**: `Err` on
//! the `Decide` arm, because `crates/decide` has no act/escalate head at all
//! (see the architecture table in the plan this crate followed) - only Laya
//! does. It surfaces `LayaHead::forward`'s own `act_logits`, which earlier
//! milestones computed and discarded; see [`RouteVerdict`]'s own doc for what
//! it returns and its honesty caveat about the act head's class order.
//!
//! **The Laya arm is calibrated the way its own serving reference is**: both
//! of `rl_agent_config.json`'s tables are applied, the per-(qtype,
//! option-count) `temperature_by_options` first and the per-qtype
//! `temperature` as the fallback, so published probabilities and confidences
//! match `rl_agent_api.py` number for number rather than only in argmax.
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
use decide::primitives::confidence;

pub use decide::banking77::Row;
pub use decide::decide::Limits as DecisionLimits;
/// The typed request vocabulary, shared by both backends: the three question
/// types a caller may ask ([`Question::Choice`], [`Question::Score`],
/// [`Question::Noul`]), the option they carry, and the answers they come back
/// as. Re-exported rather than mirrored - one definition of what a decision
/// request IS, with `crates/decide`'s own published limits
/// ([`Question::validate`]) attached to it.
pub use decide::primitives::{Answer, Opt, Question};
/// A request's state: free text, or structured JSON whose key order is
/// PRESERVED. Defined in `crates/modernbert` because Laya's own packed
/// sequence is where the order first mattered (`json.dumps` writes a dict in
/// insertion order; a sorted re-serialization tokenizes different bytes), and
/// used on both arms here so one state reaches either backbone as the same
/// text - see [`State::serialize`]. [`write_json`] is the matching writer
/// (`json.dumps(..., ensure_ascii=False)`'s own formatting), for a caller
/// rendering a response the same way it read the request.
pub use modernbert::{write_json, OrderedJson, State};

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

/// What [`DecisionPipeline::route`] returns: a probability plus Laya's own
/// act/escalate head's judgment about it - `Err` on a `Decide`-backed
/// pipeline, which has no such head (see this module's doc).
#[derive(Clone, Debug)]
pub struct RouteVerdict {
    /// `P(true)` on the proposition, exactly [`DecisionPipeline::probability`]'s
    /// own return value for the same call.
    pub probability: f32,
    /// The act head's own softmax, one entry per class (2 on the released
    /// checkpoint: `act.fc2`'s output width, `rl_agent_config.json`'s
    /// `act_costs` names one of them `"escalate"`). **Which INDEX is which
    /// class was not independently confirmed against the real training code
    /// this session** - `rl_common.py`'s own act-label order was not part of
    /// what M3/M5 verified (they verified the head's math, not its output
    /// LABELING). Treat `act_index`/`act_probabilities` as "class 0" vs
    /// "class 1" until a future check pins the mapping, the same honesty bar
    /// this module already applies to the unimplemented per-cardinality
    /// temperature table.
    pub act_probabilities: Vec<f32>,
    /// `argmax(act_probabilities)`.
    pub act_index: usize,
}

/// The encoder arrives pretrained and the head does not, so they move at
/// different rates. One rate would either leave the head too slow to learn or
/// move the encoder fast enough to forget what it was imported for.
const ENCODER_LR: f32 = 2e-5;
const HEAD_LR: f32 = 1e-3;

/// The Laya arm's own head learning rate, and the floor its cosine schedule
/// anneals to.
///
/// **Deliberately not the published Laya fine-tuning loop's `1e-4`**, and the
/// difference is the step BUDGET rather than a disagreement. That loop takes
/// ~7300 updates at an effective batch of 64;
/// [`DecisionPipeline::train_choices`]'s contract is ONE example per step
/// over a few hundred (the same contract the `Decide` arm has). AdamW's step
/// is normalized, so what a run actually moves is roughly `lr * steps`
/// (halved again by the cosine schedule both use). Matching the published
/// run's own budget at a few hundred steps therefore lands near `3e-3`:
///
/// ```text
/// published:  0.5 * 1e-4 * 7300  = 0.37
/// here:       0.5 * 3e-3 * 200   = 0.30
/// ```
///
/// Measured, not only derived: at `3e-4` the real-weight gate in
/// `crates/sdk/tests/decision_pipeline.rs` moved its training loss
/// (1.4319 -> 1.1411 over 120 steps) but left held-out accuracy flat at
/// 0.375, i.e. it was learning far too slowly to finish inside the step
/// budget a caller of this API actually passes. The schedule SHAPE (cosine
/// to a small floor, no warmup) is the published one.
const LAYA_HEAD_LR: f32 = 3e-3;
const LAYA_HEAD_LR_MIN: f32 = 1e-6;
/// Exploration noise, annealed linearly across the run - the published
/// fine-tuning loop's own `0.4 -> 0.1`.
const LAYA_SIGMA_START: f32 = 0.4;
const LAYA_SIGMA_END: f32 = 0.1;

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

/// The Laya (`convaiinnovations/laya`) backend: `crates/modernbert`'s own
/// trunk+head+tokenizer composition, plus the SERVING CALIBRATION this SDK
/// owns on top of it. Trunk frozen, head trainable - see
/// [`DecisionPipeline::train_choices`]'s Laya arm.
///
/// **`choose`/`probability` pass exactly ONE question at a time** (the same
/// SDK contract the `Decide` arm has), so `crates/modernbert`'s trunk is
/// re-encoded from scratch on every call - see this module's doc, "Two
/// architectures, one surface", for the honest cost statement this implies.
struct LayaBackend {
    /// The trunk, head and tokenizer as one model - `crates/modernbert`'s own
    /// composition, the direct counterpart of the [`Decide`] this enum's
    /// other arm holds. This SDK owns the CALIBRATION below and nothing else
    /// about the architecture, which is what lets `crates/modernbert` train
    /// and test a Laya model without the SDK on top of it.
    model: modernbert::LayaDecision,
    /// Per-qtype (`choice`/`score`/`noul`, [`modernbert::QType::index`]
    /// order) serving calibration scalar, `rl_agent_config.json`'s own
    /// `temperature` table - the FALLBACK, consulted only when
    /// [`LayaBackend::temperature_by_options`] has no entry for this call's
    /// own bucket.
    temperature: Vec<f32>,
    /// `rl_agent_config.json`'s finer per-(qtype, option-count) table, keyed
    /// by [`modernbert::temp_bucket`]. Consulted FIRST, which is what
    /// `rl_agent_api.py` does - and it is not a detail: the released
    /// checkpoint fits a 2-option choice at 1.9064 and an 11-or-more-option
    /// one at 0.1006 against a per-qtype 1.6369, so reading only the scalar
    /// publishes a distribution the reference never produces. Argmax is
    /// unaffected by any positive temperature, which is exactly why this was
    /// invisible until the two implementations were compared number by
    /// number.
    temperature_by_options: std::collections::HashMap<String, f32>,
}

impl LayaBackend {
    /// The serving temperature for one call: the per-cardinality bucket if
    /// the checkpoint fitted one, else the per-qtype scalar, else 1.0 - the
    /// same ladder `rl_agent_api.py` walks.
    fn temperature_for(&self, qtype: modernbert::QType, options: usize) -> f32 {
        if let Some(t) = self.temperature_by_options.get(&modernbert::temp_bucket(qtype, options)) {
            return *t;
        }
        self.temperature.get(qtype.index() as usize).copied().unwrap_or(1.0)
    }

    /// Run one packed `(state, question)` call through the trunk and head,
    /// returning raw (unsoftmaxed) option logits AND the act head's own raw
    /// logits (`[n_act]` - always exactly one question per call here, see
    /// this module's "Two architectures, one surface" doc). [`Self::score`]
    /// and [`Self::route`] are both thin callers of this, so the two-wait
    /// cross-`Gpu`-handle dispatch below exists in exactly one place.
    fn score_raw(&mut self, state: &State, q: &modernbert::Question) -> Result<(Vec<f32>, Vec<f32>)> {
        self.model.score(state, q, None).map_err(Error::Backend)
    }

    /// [`Self::score_raw`], option logits only - [`Self::ask`]'s own caller.
    fn score(&mut self, state: &State, q: &modernbert::Question) -> Result<Vec<f32>> {
        self.score_raw(state, q).map(|(logits, _act_logits)| logits)
    }

    /// Answer ONE typed question about `state` - the Laya arm of
    /// [`DecisionPipeline::decide`], and the single place this backend turns
    /// scores into an [`Answer`]. [`Self::choose`]/[`Self::probability`] are
    /// its callers, so the three question types cannot drift apart in how
    /// they are calibrated.
    fn ask(&mut self, state: &State, q: &Question) -> Result<Answer> {
        let mq = laya_question(q);
        let raw = self.score(state, &mq)?;
        // Calibrated by how many options were actually SCORED (markers that
        // survived `build_sequence`'s budget), not by how many the caller
        // asked about - the reference keys its bucket off the same count.
        let p = softmax(&scaled(&raw, self.temperature_for(mq.qtype(), raw.len())));
        match q {
            Question::Choice { options, .. } => {
                if p.len() != options.len() {
                    return Err(Error::Backend(format!("laya: scored {} options for a {}-option choice", p.len(), options.len())));
                }
                Ok(Answer::Choice {
                    choice: options[argmax(&p)].name.clone(),
                    probabilities: options.iter().map(|o| o.name.clone()).zip(p.iter().copied()).collect(),
                    confidence: confidence(&p),
                })
            }
            Question::Score { levels, .. } => {
                if p.len() != levels.len() {
                    return Err(Error::Backend(format!("laya: scored {} levels for a {}-level score", p.len(), levels.len())));
                }
                // The 0-BASED expectation over level indices, the same
                // reading `decide::primitives::Question::answer` applies -
                // all the mass on the first level scores 0, not 1.
                Ok(Answer::Score {
                    score: p.iter().enumerate().map(|(i, &pi)| i as f32 * pi).sum(),
                    legend: levels.clone(),
                    confidence: confidence(&p),
                    probabilities: p,
                })
            }
            Question::Noul { .. } => {
                if p.len() != 2 {
                    return Err(Error::Backend(format!("laya: a noul question must score exactly 2 options (false, true), got {}", p.len())));
                }
                // render_options() puts the "true" reading at index 1 - see
                // modernbert::sequence's own module doc.
                Ok(Answer::Noul { noul: p[1] })
            }
        }
    }

    /// [`DecisionPipeline::route`]'s own Laya-arm implementation: the noul
    /// probability plus the act head's own softmax and argmax.
    fn route(&mut self, state: &str, proposition: &str) -> Result<RouteVerdict> {
        let q = modernbert::Question::Noul { ins: proposition.to_string(), false_text: None, true_text: None };
        let (raw, act_raw) = self.score_raw(&State::Str(state.to_string()), &q)?;
        if raw.len() != 2 {
            return Err(Error::Backend(format!(
                "laya: a noul question must score exactly 2 options (false, true), got {}",
                raw.len()
            )));
        }
        let t = self.temperature_for(modernbert::QType::Noul, raw.len());
        let p = softmax(&scaled(&raw, t));
        // The act head's own calibration is not the option head's - see
        // `RouteVerdict`'s own doc: `rl_agent_config.json`'s temperature
        // table is keyed by qtype/cardinality for the OPTION head, and names
        // nothing for the act head, so no temperature is applied here.
        let act_p = softmax(&act_raw);
        let act_index = act_p.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0);
        Ok(RouteVerdict { probability: p[1], act_probabilities: act_p, act_index })
    }

    fn choose(&mut self, state: &str, instructions: &str, options: &[&str]) -> Result<Choice> {
        if options.is_empty() {
            return Err(Error::MissingArgument("choose needs at least one option".into()));
        }
        let q = Question::Choice {
            instructions: instructions.to_string(),
            options: options.iter().map(|o| Opt::new(*o)).collect(),
        };
        match self.ask(&State::Str(state.to_string()), &q)? {
            Answer::Choice { choice, probabilities, confidence } => {
                let index = argmax(&probabilities.iter().map(|(_, p)| *p).collect::<Vec<f32>>());
                Ok(Choice { choice, index, probabilities, confidence })
            }
            other => Err(Error::Backend(format!("laya: a choice question answered as {other:?}"))),
        }
    }

    fn probability(&mut self, state: &str, proposition: &str) -> Result<f32> {
        let q = Question::Noul { instructions: proposition.to_string(), yes: None, no: None };
        match self.ask(&State::Str(state.to_string()), &q)? {
            Answer::Noul { noul } => Ok(noul),
            other => Err(Error::Backend(format!("laya: a noul question answered as {other:?}"))),
        }
    }
}

/// The typed question, in Laya's own `render_options` vocabulary. One place,
/// so the mapping (a `Choice`'s per-option description, a `Score`'s ordered
/// levels, a `Noul`'s two criteria texts - `yes` is the TRUE reading) cannot
/// be spelled differently by two callers.
fn laya_question(q: &Question) -> modernbert::Question {
    match q {
        Question::Choice { instructions, options } => modernbert::Question::Choice {
            ins: instructions.clone(),
            options: options.iter().map(|o| (o.name.clone(), o.description.clone())).collect(),
        },
        Question::Score { instructions, levels } => {
            modernbert::Question::Score { ins: instructions.clone(), options: levels.clone() }
        }
        Question::Noul { instructions, yes, no } => {
            modernbert::Question::Noul { ins: instructions.clone(), false_text: no.clone(), true_text: yes.clone() }
        }
    }
}

/// First maximal index, so ties resolve to the option the caller listed
/// first - `decide::primitives`' own convention, reproduced here because its
/// `argmax` is private to that crate.
fn argmax(p: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in p.iter().enumerate() {
        if v > p[best] {
            best = i;
        }
    }
    best
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

    /// Answer several typed questions about ONE state, in the order asked.
    ///
    /// The full request shape both backends implement, and the one
    /// [`DecisionPipeline::choose`]/[`DecisionPipeline::probability`] cannot
    /// express: a [`Question::Score`] has no other surface at all, a
    /// [`Question::Choice`]'s options can carry the descriptions the model is
    /// meant to read, and the state may be structured
    /// ([`State::Json`]) rather than a string the caller flattened itself.
    ///
    /// Every question is validated against
    /// [`Question::validate`]'s published limits BEFORE any model runs, so a
    /// malformed request coming off a wire is refused rather than tokenized.
    ///
    /// **What "one call" costs differs by arm, and this is the honest version
    /// of the asymmetry this module's doc opens with.** The `Decide` arm
    /// encodes the state ONCE and scores every question's options against
    /// that encoding - the economics that architecture exists for, reachable
    /// here and nowhere else in this SDK. The Laya arm re-packs and re-encodes
    /// `(state, question)` per question (a full ModernBERT-large forward
    /// each), because its head takes one question's option markers per call;
    /// asking five questions costs five forwards there.
    ///
    /// Questions are independent either way: no answer conditions another,
    /// and the order out is the order in.
    pub fn decide(&mut self, state: &State, questions: &[Question]) -> Result<Vec<Answer>> {
        for (i, q) in questions.iter().enumerate() {
            q.validate().map_err(|e| Error::MissingArgument(format!("question {i}: {e}")))?;
        }
        match &mut self.backend {
            Backend::Decide(model) => {
                let answers = model.decide(&state.serialize(), questions).map_err(Error::Backend)?;
                if answers.len() != questions.len() {
                    return Err(Error::Backend(format!(
                        "the model answered {} of {} questions",
                        answers.len(),
                        questions.len()
                    )));
                }
                Ok(answers)
            }
            Backend::Laya(l) => questions.iter().map(|q| l.ask(state, q)).collect(),
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

    /// The probability of `proposition` AND, on a Laya-backed pipeline, its
    /// own act/escalate head's routing judgment about that call - `Err` on a
    /// `Decide`-backed pipeline, which has no such head at all (see this
    /// module's doc and [`RouteVerdict`]'s own doc for what the act half
    /// means and its honesty caveat).
    pub fn route(&mut self, state: &str, proposition: &str) -> Result<RouteVerdict> {
        match &mut self.backend {
            Backend::Decide(_) => Err(Error::Backend(
                "route() needs Laya's own act/escalate head, which crates/decide has no equivalent \
                 of - a Decide-backed DecisionPipeline has only choose/probability. A decide-backed \
                 conversation router lives in brain::ConversionPipeline instead, which fits its own \
                 semantic/convergence/learned signal ensemble on top of a trained Decide model."
                    .into(),
            )),
            Backend::Laya(l) => l.route(state, proposition),
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

    /// Set what [`Flow::evaluate`]/[`Stages::run_eval`] scores next, WITHOUT
    /// training - [`Stages::run_train`]'s own counterpart for a caller that
    /// is skipping [`Flow::train`] (a backend [`Stages::supports_training`]
    /// says cannot be trained here, or a pipeline already loaded from a
    /// trained head) but still wants a real evaluation number rather than
    /// none. See [`Flow::with_eval`] for the chain stage built on this.
    pub fn set_eval(&mut self, eval: Vec<(String, usize)>) {
        self.last_eval = eval;
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
    ///
    /// **Both arms train, and the option sampling above is shared** - what
    /// differs is the objective, because the two checkpoints were fitted
    /// under different ones and a model should be trained against the rule
    /// its own weights came from. The `Decide` arm minimizes
    /// [`rlcd::scoring`]'s cross-entropy/focal/Brier objective directly. The
    /// Laya arm runs [`rlcd::reinforce`]'s REINFORCE-with-a-group-baseline
    /// over [`rlcd::proper`]'s strictly proper scoring reward, with the
    /// published loop's own annealed exploration and cosine learning rate.
    /// The returned losses are therefore NOT comparable across arms; only
    /// each arm's own trend over a run is.
    ///
    /// On the Laya arm the 395M trunk is held FIXED and only the decision
    /// head learns - see `modernbert::decision`'s module doc for the three
    /// reasons, and [`DecisionPipeline::save_head`] for the one that makes it
    /// load-bearing rather than a tuning choice.
    pub fn train_choices(
        &mut self,
        examples: &[(&str, usize)],
        options: &[String],
        instructions: &str,
        steps: usize,
        seed: u64,
        log: &mut dyn FnMut(usize, f32),
    ) -> Result<f32> {
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
            // ONE sampler, one draw, both arms: each step scores its example
            // against a random SUBSET of the options, always containing the
            // correct one, at a random position.
            let (drawn, gold) = sampler.draw(label, &pool, &mut rng);
            let l = match &mut self.backend {
                Backend::Decide(model) => {
                    let q = Question::Choice {
                        instructions: instructions.to_string(),
                        options: drawn.iter().map(|&i| Opt::new(options[i].clone())).collect(),
                    };
                    model
                        .train_step(&Example { state: text, question: &q, gold }, &loss_cfg, ENCODER_LR, HEAD_LR)
                        .map_err(Error::Backend)?
                }
                Backend::Laya(b) => {
                    let q = modernbert::Question::Choice {
                        ins: instructions.to_string(),
                        options: drawn.iter().map(|&i| (options[i].clone(), None)).collect(),
                    };
                    let target = rlcd::proper::hard_target(drawn.len(), gold);
                    let progress = step as f32 / steps.max(1) as f32;
                    let obj = rlcd::reinforce::RlcdObjective::default()
                        .at(rlcd::reinforce::anneal(LAYA_SIGMA_START, LAYA_SIGMA_END, progress));
                    let lr = rlcd::reinforce::cosine_lr(LAYA_HEAD_LR, LAYA_HEAD_LR_MIN, progress);
                    let state = State::Str(text.to_string());
                    b.model
                        .train_step_with(&state, &q, None, 0.0, lr, |scores| {
                            // A `choice` question is not ordinal, so the
                            // ranked-probability term does not apply - the
                            // same gate `rl_common.py` puts on `qtype`.
                            rlcd::reinforce::rlcd_loss(scores, &target, false, &obj, &mut rng)
                        })
                        .map_err(Error::Backend)?
                }
            };
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
    /// Both arms write a head-only adapter naming the checkpoint it attaches
    /// to; the two files are different formats for different architectures
    /// and are not interchangeable, which is why each carries its own
    /// `architecture`/`adapter.kind` in its card. Feed either back through
    /// [`DecisionPipelineBuilder::head`] alongside the matching checkpoint
    /// directory.
    pub fn save_head(&self, path: impl AsRef<str>) -> Result<()> {
        match &self.backend {
            Backend::Decide(model) => model.save_head(path.as_ref()).map_err(Error::Backend),
            Backend::Laya(b) => b.model.save_head(path.as_ref()).map_err(Error::Backend),
        }
    }

    /// How many optimizer steps this pipeline has taken - the AdamW time
    /// index, on whichever arm is loaded.
    pub fn steps_taken(&self) -> u32 {
        match &self.backend {
            Backend::Decide(model) => model.steps_taken(),
            Backend::Laya(b) => b.model.steps_taken(),
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
        let steps = self.steps_taken();
        format!("decision model, {} options in the last call, {steps} training steps so far", self.last_options.len())
    }

    fn supports_training(&self) -> bool {
        // Both architectures train now. Kept as an explicit `true` rather
        // than removed: this is the general seam a future pretrained-only
        // backend reuses (see this module's doc), and a caller that checks it
        // is doing the right thing whatever is behind it.
        true
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

impl Flow<DecisionPipeline> {
    /// Set what [`Flow::evaluate`] scores next, without training - the chain
    /// stage built on [`DecisionPipeline::set_eval`], for a caller that
    /// checked [`Flow::supports_training`] (or already has a trained head)
    /// and is skipping [`Flow::train`] but still wants a real evaluation
    /// number. Added through the same seam [`Flow::stage`] documents
    /// ([`Flow<ConversionPipeline>::replay`] is the other example) rather
    /// than folded into [`Flow::with_question`]: the two travel together at
    /// most call sites that need this, but `with_question` is shared by
    /// every [`Stages`] architecture and most have no notion of a held-out
    /// eval set shaped like this one's `(text, label)` pairs.
    pub fn with_eval(self, eval: Vec<(String, usize)>) -> Flow<DecisionPipeline> {
        let n = eval.len();
        self.stage("eval set", move |p| {
            p.set_eval(eval);
            Ok(Some(format!("{n} held-out examples")))
        })
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
    /// Trained head weights, as written by [`DecisionPipeline::save_head`].
    ///
    /// The two arms mean slightly different things by this, because their
    /// checkpoints do. On the `Decide` arm the head is RANDOM without it and
    /// the model's answers are noise. A Laya checkpoint SHIPS a trained head,
    /// so this REPLACES a working one - which is also why a head file from
    /// the wrong architecture is refused by name rather than ignored.
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

    /// Build a model ready to answer, and to fine-tune: on BOTH arms the
    /// same object a caller loads is the one it can train (see this module's
    /// doc for what each arm trains, and with what objective).
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
/// `load_decide`, there is no fresh-random-head path to seed here - a Laya
/// checkpoint SHIPS a trained head, so `head` REPLACES it rather than
/// replacing a random initialization.
///
/// Built [`modernbert::Training::HeadOnly`], so the same object a
/// caller loads is the one it can fine-tune, exactly as on the `Decide` arm.
/// That mode is not a tuning preference at this size: a `Role::Trainable`
/// ModernBERT-large trunk would carry a gradient and two AdamW moments for
/// every one of its 395M parameters (~6.3 GB of device memory) before a
/// single activation, so making the trunk trainable *by default* would turn
/// loading a Laya pipeline for INFERENCE into an out-of-memory failure on
/// most hardware. See `modernbert::decision`'s own module doc.
fn load_laya(dir: &str, head: Option<&str>, device: &Device, _seed: u64) -> Result<LayaBackend> {
    let ckpt = modernbert::import_dir(dir).map_err(Error::Backend)?;

    let tok_path = Path::new(dir).join("tokenizer").join("tokenizer.json");
    let tok = data::qwen_tokenizer::QwenBpe::from_file(
        tok_path.to_str().ok_or_else(|| Error::Backend("non-UTF-8 tokenizer path".into()))?,
    )
    .map_err(Error::Backend)?;

    // A supplied head is folded into the init map rather than written after
    // construction, so a reloaded pipeline is built the same way a fresh one
    // is - one construction path, no "loaded" state a later `set_weights`
    // could half-apply.
    let mut head_init = ckpt.head_init;
    if let Some(p) = head {
        let trained = modernbert::LayaDecision::read_head_file(p).map_err(Error::Backend)?;
        for (name, v) in trained {
            match head_init.get(&name) {
                Some(existing) if existing.len() != v.len() => {
                    return Err(Error::Backend(format!(
                        "{p}: tensor {name} has {} floats, this checkpoint's head wants {}",
                        v.len(),
                        existing.len()
                    )))
                }
                // An unknown name is a `decide` head, or a head for a
                // different `d_model` - either way not this architecture's.
                None => {
                    return Err(Error::Backend(format!(
                        "{p}: tensor {name} is not a Laya decision-head parameter - is this a \
                         `decide` head file rather than a Laya one?"
                    )))
                }
                Some(_) => {}
            }
            head_init.insert(name, v);
        }
    }

    crate::device::resolve(device)?;
    let gpu = gpu_core::Gpu::new(modernbert::kern::PIPELINES);
    let max_len = ckpt.rl.max_len;
    let model = modernbert::LayaDecision::new_on(
        gpu,
        ckpt.cfg,
        ckpt.laya_cfg,
        tok,
        max_len,
        ckpt.rl.head_max_len,
        &ckpt.encoder_init,
        &head_init,
        modernbert::Training::HeadOnly,
    );
    let mut model = model;
    model.set_provenance(modernbert::Provenance {
        base: base_reference(Path::new(dir)),
        ..Default::default()
    });

    Ok(LayaBackend {
        model,
        temperature: ckpt.rl.temperature,
        temperature_by_options: ckpt.rl.temperature_by_options,
    })
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
