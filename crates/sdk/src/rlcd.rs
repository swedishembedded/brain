// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain::RlcdPipeline` - training a decision model directly against exact
//! oracle posteriors, and auditing the result on calibration AND
//! cost-sensitive decision regret rather than accuracy alone.
//!
//! ```no_run
//! use brain::{RlcdPipeline, RlcdSpec};
//! RlcdPipeline::from_pretrained("/path/to/all-MiniLM-L6-v2")
//!     .train(RlcdSpec::default())
//!     .evaluate()
//!     .save("out/rlcd-head.safetensors")
//!     .report()
//!     .finish()?;
//! # Ok::<(), brain::Error>(())
//! ```
//!
//! Same stage chain as every other pipeline here; what differs is the
//! training signal. [`crate::DecisionPipeline`] trains on a single gold
//! option index. This one trains on a full target DISTRIBUTION over options
//! (`rlcd::scoring::decision_loss_soft`, via `Decide::train_step_with`'s
//! existing raw-logits-in/gradient-out objective seam - no change to
//! `crates/decide` was needed to support this), which is what lets an
//! application train directly against an exact Bayesian oracle posterior
//! (`P(fault) = 2/3`, not "class 2").
//!
//! **Rendering and worlds are deliberately not this crate's job.** An
//! `RlcdExample` is already a rendered state string paired with a target
//! distribution - turning an application's own executable world
//! (`rlcd::atlas::World`) into that pair is the application's own
//! presentation choice, the same way `ConversionSpec` takes already-rendered
//! `Conversation`s rather than owning a conversation generator. This keeps
//! `RlcdPipeline` domain-agnostic: it knows `(state, target)` pairs and
//! nothing about devices, workflows, or any other concrete domain.
//!
//! Evaluation reports three axes, not one: [`rlcd::metrics::ece`] and
//! [`rlcd::metrics::nll`] audit the PROBABILITY head against calibration
//! (does the model believe correctly), and [`rlcd::cost::regret`] against
//! every named cost matrix in [`RlcdSpec::eval_costs`] audits the DECISION
//! (does the belief, run through an explicit cost matrix, choose correctly)
//! - on cost matrices the model never trained under, which is the number
//! that tells apart a learned belief from a learned policy.

use decide::decide::{Decide, Limits};
use decide::primitives::{confidence, Opt, Question};
use rlcd::cost::{bayes_action, regret};
use rlcd::metrics::{brier_score, ece, nll};
use rlcd::scoring::{decision_loss_soft, softmax};

use crate::flow::{EvalReport, Flow, Stages, TrainReport};
use crate::{Device, Error, Result};

// Re-exported so an application can turn its own executable world into
// `RlcdExample`s and name cost matrices without depending on `brain-rlcd`
// directly - a sample may name only the `brain` SDK facade as its brain
// dependency (see `samples/README.md`'s rules), never an engine crate.
pub use rlcd::atlas::{check_information_refinement, DecisionContract, Distribution, Observation, OracleKind, World};
pub use rlcd::cost::{BayesAction, CostMatrix};
pub use rlcd::metrics::{ada_ece, classwise_ece, coverage_accuracy, failure_auroc, reliability_bins, ReliabilityBin};
pub use rlcd::scoring::LossConfig;
pub use rlcd::witness::{search as witness_search, Learner, WitnessFamily};

/// The encoder arrives pretrained and the head does not - the same
/// discriminative rates every `decide`-backed pipeline in this SDK uses.
const ENCODER_LR: f32 = 2e-5;
const HEAD_LR: f32 = 1e-3;

/// One training or evaluation example: a rendered state, and the EXACT
/// target distribution over `RlcdSpec::options` an oracle assigned it.
/// `target.len()` must equal `RlcdSpec::options.len()`.
#[derive(Clone, Debug)]
pub struct RlcdExample {
    pub state: String,
    pub target: Vec<f32>,
}

impl RlcdExample {
    pub fn new(state: impl Into<String>, target: Vec<f32>) -> RlcdExample {
        RlcdExample { state: state.into(), target }
    }
}

/// What one RLCD training run needs.
#[derive(Clone, Debug)]
pub struct RlcdSpec {
    pub train: Vec<RlcdExample>,
    /// Held out, for [`Flow::evaluate`].
    pub eval: Vec<RlcdExample>,
    /// The question every example answers - e.g. "is this device faulty".
    pub instructions: String,
    /// Option names, in the order `RlcdExample::target` indexes them.
    pub options: Vec<String>,
    pub loss: LossConfig,
    pub steps: usize,
    pub seed: u64,
    /// Named cost matrices [`Flow::evaluate`] reports decision regret
    /// against. Kept OUT of training on purpose: regret measured against a
    /// cost matrix the model never trained under is what shows the model
    /// learned a transferable belief rather than memorizing one policy.
    pub eval_costs: Vec<(String, CostMatrix)>,
    /// Hold the encoder fixed and train only the head. `false` (a full
    /// fine-tune, matching every other `decide`-backed pipeline's default)
    /// needs a training set large enough that the encoder generalizes rather
    /// than memorizes - `crates/decide`'s own BANKING77 run trains on
    /// thousands of examples. A small, purpose-built world (dozens of
    /// examples, not thousands) should set this `true`: it is also the only
    /// way `Flow::save`'s head-only checkpoint format is even writable
    /// (`Decide::save_head` refuses to write one over a moved encoder, since
    /// loading it back would silently reattach the head to the wrong base).
    pub freeze_encoder: bool,
}

impl Default for RlcdSpec {
    fn default() -> RlcdSpec {
        RlcdSpec {
            train: Vec::new(),
            eval: Vec::new(),
            instructions: String::new(),
            options: Vec::new(),
            loss: LossConfig::cross_entropy(),
            steps: 1000,
            seed: 0,
            eval_costs: Vec::new(),
            freeze_encoder: false,
        }
    }
}

impl RlcdSpec {
    pub fn train(mut self, train: Vec<RlcdExample>) -> RlcdSpec {
        self.train = train;
        self
    }
    pub fn eval(mut self, eval: Vec<RlcdExample>) -> RlcdSpec {
        self.eval = eval;
        self
    }
    pub fn instructions(mut self, instructions: impl Into<String>) -> RlcdSpec {
        self.instructions = instructions.into();
        self
    }
    pub fn options(mut self, options: Vec<String>) -> RlcdSpec {
        self.options = options;
        self
    }
    pub fn loss(mut self, loss: LossConfig) -> RlcdSpec {
        self.loss = loss;
        self
    }
    pub fn steps(mut self, steps: usize) -> RlcdSpec {
        self.steps = steps;
        self
    }
    pub fn seed(mut self, seed: u64) -> RlcdSpec {
        self.seed = seed;
        self
    }
    pub fn eval_costs(mut self, eval_costs: Vec<(String, CostMatrix)>) -> RlcdSpec {
        self.eval_costs = eval_costs;
        self
    }
    pub fn freeze_encoder(mut self, freeze: bool) -> RlcdSpec {
        self.freeze_encoder = freeze;
        self
    }
}

pub struct RlcdPipeline {
    model: Decide,
    question: Option<Question>,
    eval: Vec<RlcdExample>,
    eval_costs: Vec<(String, CostMatrix)>,
}

impl std::fmt::Debug for RlcdPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RlcdPipeline").finish_non_exhaustive()
    }
}

impl RlcdPipeline {
    pub fn from_pretrained(dir: impl AsRef<str>) -> Flow<RlcdPipeline> {
        Flow::new(RlcdPipeline::builder(dir).load())
    }

    pub fn builder(dir: impl AsRef<str>) -> RlcdPipelineBuilder {
        RlcdPipelineBuilder {
            dir: dir.as_ref().to_string(),
            head: None,
            device: Device::default(),
            limits: Limits::default(),
            seed: 0,
        }
    }

    fn question(&self, spec: &RlcdSpec) -> Question {
        Question::Choice { instructions: spec.instructions.clone(), options: spec.options.iter().map(Opt::new).collect() }
    }

    /// The model's own belief over `spec.options` for a rendered `state` -
    /// the calibrated probability, with no action attached.
    pub fn probability(&mut self, state: &str) -> Result<Vec<f32>> {
        let q = self.question.clone().ok_or_else(|| Error::MissingArgument("train before asking for a probability".into()))?;
        let scores = self.model.score(state, std::slice::from_ref(&q)).map_err(Error::Backend)?;
        Ok(softmax(&scores[0]))
    }

    /// The model's belief, run through an explicit cost matrix - the
    /// deterministic argmin this whole pipeline exists to keep separate from
    /// the probability itself. See `rlcd::cost`'s own module doc for why.
    pub fn decide_action(&mut self, state: &str, costs: &CostMatrix) -> Result<BayesAction> {
        let p = self.probability(state)?;
        Ok(bayes_action(&p, costs))
    }
}

impl Stages for RlcdPipeline {
    type TrainSpec = RlcdSpec;

    fn describe(&self) -> String {
        format!("RLCD decision model, {} training steps so far, {} options", self.model.steps_taken(), self.question.as_ref().map(Question::arity).unwrap_or(0))
    }

    fn run_train(&mut self, spec: &RlcdSpec, log: &mut dyn FnMut(usize, f32)) -> Result<TrainReport> {
        if spec.train.is_empty() {
            return Err(Error::MissingArgument("no examples to train on".into()));
        }
        for ex in &spec.train {
            if ex.target.len() != spec.options.len() {
                return Err(Error::MissingArgument(format!(
                    "example target has {} entries, but {} options were declared",
                    ex.target.len(),
                    spec.options.len()
                )));
            }
        }
        self.model.set_encoder_frozen(spec.freeze_encoder);
        let q = self.question(spec);
        let mut rng = data::rng::Rng::new(spec.seed);
        let (mut tail, tail_n) = (0.0f32, (spec.steps / 10).max(1));
        for step in 0..spec.steps {
            let ex = &spec.train[rng.gen_range_inclusive(0, spec.train.len() as i64 - 1) as usize];
            let target = ex.target.clone();
            let loss = spec.loss;
            let l = self.model.train_step_with(&ex.state, &q, ENCODER_LR, HEAD_LR, |s| decision_loss_soft(s, &target, &loss)).map_err(Error::Backend)?;
            log(step, l);
            if step + tail_n >= spec.steps {
                tail += l / tail_n as f32;
            }
        }
        self.question = Some(q);
        self.eval = spec.eval.clone();
        self.eval_costs = spec.eval_costs.clone();
        Ok(TrainReport { steps: spec.steps, final_loss: tail, seconds: 0.0 })
    }

    fn run_eval(&mut self) -> Result<EvalReport> {
        if self.eval.is_empty() {
            return Ok(EvalReport::default());
        }
        let q = self.question.clone().ok_or_else(|| Error::MissingArgument("train before evaluating".into()))?;
        let eval = std::mem::take(&mut self.eval);

        let mut probs: Vec<Vec<f32>> = Vec::with_capacity(eval.len());
        let mut labels: Vec<usize> = Vec::with_capacity(eval.len());
        let mut confidences: Vec<f32> = Vec::with_capacity(eval.len());
        let mut correct: Vec<bool> = Vec::with_capacity(eval.len());
        for ex in &eval {
            let scores = self.model.score(&ex.state, std::slice::from_ref(&q)).map_err(Error::Backend)?;
            let p = softmax(&scores[0]);
            let label = argmax(&ex.target);
            let predicted = argmax(&p);
            confidences.push(confidence(&p));
            correct.push(predicted == label);
            probs.push(p);
            labels.push(label);
        }
        self.eval = eval.clone();

        let hit = correct.iter().filter(|&&c| c).count();
        let mut notes = vec![
            ("ECE".into(), ece(&confidences, &correct, 10)),
            ("NLL".into(), nll(&probs, &labels)),
            ("Brier".into(), brier_score(&probs, &labels)),
        ];
        // Decision regret per named held-out cost matrix: the model's own
        // action, scored against the EXACT oracle posterior the example
        // carries (its `target`, which is what an RlcdExample's oracle
        // producer computed it to be) - never against the model's own belief,
        // which would make every model appear to have zero regret.
        for (name, costs) in &self.eval_costs {
            let mut total = 0.0f32;
            for (ex, p) in eval.iter().zip(&probs) {
                let action = bayes_action(p, costs).action;
                total += regret(&ex.target, costs, action);
            }
            notes.push((format!("regret[{name}]"), total / eval.len() as f32));
        }
        Ok(EvalReport { accuracy: hit as f32 / eval.len() as f32, items: eval.len(), notes })
    }

    fn run_save(&self, path: &str) -> Result<()> {
        self.model.save_head(path).map_err(Error::Backend)
    }

    fn run_turn(&mut self, input: &str) -> Result<String> {
        let p = self.probability(input)?;
        let q = self.question.as_ref().expect("probability() above already required a trained question");
        let Question::Choice { options, .. } = q else { unreachable!("RlcdPipeline only ever builds Question::Choice") };
        let mut lines = vec![format!("  belief over {} options:", options.len())];
        for (opt, prob) in options.iter().zip(&p) {
            let filled = (prob * 24.0).round() as usize;
            lines.push(format!("    {:<20} {:.3} [{}{}]", opt.name, prob, "#".repeat(filled), ".".repeat(24 - filled)));
        }
        if let Some((name, costs)) = self.eval_costs.first() {
            let action = bayes_action(&p, costs);
            lines.push(format!("  Bayes action under \"{name}\": {} (risk {:.4})", options[action.action].name, action.risk));
        }
        Ok(lines.join("\n"))
    }

    fn turn_prompt(&self) -> &str {
        "state> "
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0
}

pub struct RlcdPipelineBuilder {
    dir: String,
    head: Option<String>,
    device: Device,
    limits: Limits,
    seed: u64,
}

impl RlcdPipelineBuilder {
    /// Trained head weights. Without this the head is random and every
    /// probability is noise.
    pub fn head(mut self, path: impl AsRef<str>) -> RlcdPipelineBuilder {
        self.head = Some(path.as_ref().to_string());
        self
    }

    pub fn device(mut self, device: Device) -> RlcdPipelineBuilder {
        self.device = device;
        self
    }

    pub fn limits(mut self, limits: Limits) -> RlcdPipelineBuilder {
        self.limits = limits;
        self
    }

    pub fn seed(mut self, seed: u64) -> RlcdPipelineBuilder {
        self.seed = seed;
        self
    }

    pub fn load(self) -> Result<RlcdPipeline> {
        let model = crate::decision::load_decide(&self.dir, self.head.as_deref(), &self.device, self.limits, self.seed)?;
        Ok(RlcdPipeline { model, question: None, eval: Vec::new(), eval_costs: Vec::new() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_the_largest_and_the_first_on_a_tie() {
        assert_eq!(argmax(&[0.1, 0.7, 0.2]), 1);
        assert_eq!(argmax(&[0.5, 0.5]), 0);
    }
}
