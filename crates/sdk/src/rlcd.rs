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
use decide::policy::{choice_loss, Act, PolicyConfig};
use decide::primitives::{confidence, Opt, Question};
use rlcd::metrics::{brier_score, ece, nll};
use rlcd::scoring::{decision_loss_soft, softmax};

use crate::flow::{EvalReport, Flow, Stages, TrainReport};
use crate::{Device, Error, Result};

// Re-exported so an application can turn its own executable world into
// `RlcdExample`s and name cost matrices without depending on `brain-rlcd`
// directly - a sample may name only the `brain` SDK facade as its brain
// dependency (see `samples/README.md`'s rules), never an engine crate.
pub use rlcd::atlas::{check_information_refinement, DecisionContract, Distribution, Observation, OracleKind, World};
pub use rlcd::cost::{bayes_action, bayes_risk, regret, voi, BayesAction, CostMatrix};
pub use rlcd::metrics::{ada_ece, classwise_ece, coverage_accuracy, failure_auroc, reliability_bins, ReliabilityBin};
pub use rlcd::scoring::LossConfig;
pub use rlcd::witness::{search as witness_search, Learner, WitnessFamily};

/// The encoder arrives pretrained and the head does not - the same
/// discriminative rates every `decide`-backed pipeline in this SDK uses.
const ENCODER_LR: f32 = 2e-5;
const HEAD_LR: f32 = 1e-3;

/// `train_voi_policy`'s own, lower rate - matching `ConversionPipeline`'s
/// established precedent (its `POLICY_HEAD_LR` is likewise below its
/// supervised `HEAD_LR`): a policy-gradient update is far noisier than a
/// supervised one, and a fresh head committing to a near-deterministic
/// answer off a handful of single-sample REINFORCE updates leaves little
/// room for exploration to correct an early mistake.
const VOI_HEAD_LR: f32 = 1e-4;

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
    /// The meta-decision question `train_voi_policy` trains and
    /// `voi_policy_action` queries - `{block, release, inspect}` by
    /// construction (see `train_voi_policy`'s own doc). Separate from
    /// `question`: a calibrated belief and a policy over what to DO about it
    /// are two different heads answering two different questions of the same
    /// state, not one repurposed as the other.
    voi_question: Option<Question>,
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

    /// Train a LEARNED policy over `{act_now=0, inspect=1}` at `states` (each
    /// a rendering of "no evidence yet"), through `decide::policy`'s
    /// clipped-choice objective - an evidence-gathering decision, not a
    /// probability, so it is graded by return (`decide::policy::choice_loss`'s
    /// own doc explains why that is the right objective here and
    /// [`crate::flow`]'s calibrated belief is graded by a proper scoring rule
    /// instead).
    ///
    /// **Only the evidence-gathering decision is learned.** What "act now"
    /// MEANS - block or release - is the CLOSED-FORM Bayes action on the
    /// prior (`rlcd::cost::bayes_action`, stage 2, already solved exactly),
    /// computed once and fixed for the whole run - not a second thing left
    /// for the policy to also get right. An earlier version of this function
    /// folded both decisions into one three-way choice `{block, release,
    /// inspect}` and the terminal sub-choice was observed to sometimes
    /// converge to the WRONG one (a real run picked "release" under costs
    /// where "block" was strictly cheaper) even after raising exploration -
    /// a single-example online REINFORCE loop over a handful of states is
    /// good at learning ONE boundary, not two compounded ones, and stage 2
    /// never needed learning in the first place.
    ///
    /// The reward per step is an exact Monte Carlo estimate built ONLY from
    /// what `world: &impl World` already exposes (no new oracle machinery):
    /// sample a true outcome `y ~ prior`; `act_now` realizes
    /// `costs.get(bayes_action(prior, costs).action, y)`; `inspect`
    /// additionally samples which observation the true world would reveal
    /// GIVEN `y` - the joint the world implies, `P(o|y) = posterior_o[y] *
    /// P(o) / prior[y]` by Bayes, derivable from `Observation::posterior`/
    /// `probability` alone - pays `query_cost`, and then takes the
    /// CLOSED-FORM Bayes action on what was revealed.
    ///
    /// Gate this against `rlcd::cost::voi` on the SAME `world`/`costs`/
    /// `query_cost` (see `samples/learning/rlcd`'s own measured run): as
    /// training converges, `voi_policy_action`'s answer should track the
    /// sign of `voi()` - `inspect` when positive, `act_now` when not.
    #[allow(clippy::too_many_arguments)]
    pub fn train_voi_policy(
        &mut self,
        states: &[String],
        world: &impl World,
        costs: &CostMatrix,
        query_cost: f32,
        steps: usize,
        seed: u64,
    ) -> Result<TrainReport> {
        if states.is_empty() {
            return Err(Error::MissingArgument("no states to train the VOI policy on".into()));
        }
        let q = Question::Choice {
            instructions: "should more evidence be gathered before deciding what to do".into(),
            options: vec![Opt::new("act now"), Opt::new("inspect")],
        };
        let prior = world.prior();
        let obs = world.observations();
        let act_now_action = bayes_action(&prior, costs).action;
        let cfg = PolicyConfig::default();
        let mut rng = data::rng::Rng::new(seed);
        // A running baseline (EMA), not a per-batch mean: this pipeline
        // trains one example per step, matching every other RlcdPipeline
        // training loop, so there is no batch to center advantages against.
        let mut baseline = 0.0f32;
        let (mut tail, tail_n) = (0.0f32, (steps / 10).max(1));
        for step in 0..steps {
            let state = &states[rng.gen_range_inclusive(0, states.len() as i64 - 1) as usize];
            let y = sample_categorical(&prior, &mut rng);

            let scores = self.model.score(state, std::slice::from_ref(&q)).map_err(Error::Backend)?;
            let p = softmax(&scores[0]);
            let action = sample_categorical(&p, &mut rng);

            let reward = voi_policy_reward(action, y, act_now_action, &obs, costs, query_cost, &mut rng);

            // Advantage against the baseline BEFORE this step's own reward
            // folds into it - using the just-observed reward to compute its
            // own baseline would leak part of the signal being centered.
            let act = Act { old_prob: p[action], action, advantage: reward - baseline };
            baseline = 0.9 * baseline + 0.1 * reward;
            let l = self
                .model
                .train_step_with(state, &q, ENCODER_LR, VOI_HEAD_LR, |s| choice_loss(s, &act, &cfg))
                .map_err(Error::Backend)?;
            if step + tail_n >= steps {
                tail += l / tail_n as f32;
            }
        }
        self.voi_question = Some(q);
        Ok(TrainReport { steps, final_loss: tail, seconds: 0.0 })
    }

    /// The learned policy's own action at `state` - `(action, probabilities)`
    /// with `action` indexing `{act_now, inspect}` per
    /// [`RlcdPipeline::train_voi_policy`]'s own doc.
    pub fn voi_policy_action(&mut self, state: &str) -> Result<(usize, Vec<f32>)> {
        let q = self.voi_question.clone().ok_or_else(|| Error::MissingArgument("train_voi_policy before querying its action".into()))?;
        let scores = self.model.score(state, std::slice::from_ref(&q)).map_err(Error::Backend)?;
        let p = softmax(&scores[0]);
        let action = p.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0;
        Ok((action, p))
    }
}

/// One Monte Carlo reward sample for [`RlcdPipeline::train_voi_policy`]'s
/// `{act_now=0, inspect=1}` choice, given a true outcome `y` already sampled
/// from `world.prior()` and `act_now_action` (the closed-form Bayes action on
/// the prior, fixed for the whole run) - see that function's own doc for the
/// full derivation. Factored out so the reward computation is checkable on
/// its own, independent of the neural policy that consumes it.
fn voi_policy_reward(action: usize, y: usize, act_now_action: usize, obs: &[Observation], costs: &CostMatrix, query_cost: f32, rng: &mut data::rng::Rng) -> f32 {
    if action == 0 {
        return -costs.get(act_now_action, y);
    }
    // P(o | y) unnormalized: posterior_o[y] * P(o). Bayes' theorem's
    // denominator (prior[y]) is a constant over the choice of o, so it is
    // omitted rather than divided out.
    let weights: Vec<f32> = obs.iter().map(|o| o.posterior[y] * o.probability).collect();
    let revealed = &obs[sample_categorical(&weights, rng)];
    let follow_up = bayes_action(&revealed.posterior, costs).action;
    -query_cost - costs.get(follow_up, y)
}

/// Sample an index from `weights` (need not sum to 1 - normalized here).
fn sample_categorical(weights: &[f32], rng: &mut data::rng::Rng) -> usize {
    let total: f32 = weights.iter().sum();
    assert!(total > 0.0, "sample_categorical: weights sum to {total}, nothing to sample");
    let mut draw = rng.uniform(0.0, total as f64) as f32;
    for (i, &w) in weights.iter().enumerate() {
        if draw < w {
            return i;
        }
        draw -= w;
    }
    weights.len() - 1 // floating-point rounding at the boundary, not a logic error
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
        Ok(RlcdPipeline { model, question: None, eval: Vec::new(), eval_costs: Vec::new(), voi_question: None })
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

    #[test]
    fn sample_categorical_matches_its_weights_over_many_draws() {
        let mut rng = data::rng::Rng::new(7);
        let weights = [0.8f32, 0.2];
        let n = 200_000;
        let ones = (0..n).filter(|_| sample_categorical(&weights, &mut rng) == 1).count();
        let rate = ones as f64 / n as f64;
        assert!((rate - 0.2).abs() < 0.01, "sampled rate {rate:.4} should be close to 0.2");
    }

    /// The worked example this crate's design was built around: `P(fault) =
    /// 0.2`, diagnostic `P(+|fault) = 0.8` / `P(+|healthy) = 0.1`, costs
    /// `(C_FP, C_FN) = (1, 10)`. Verifies `voi_policy_reward`'s expected
    /// value per action matches the closed-form numbers `rlcd::cost`'s own
    /// test suite gates - independent of the neural policy that trains
    /// against it, and independent of `bayes_risk`/`voi` (this recomputes
    /// the same numbers a different way, as a cross-check).
    #[test]
    fn voi_policy_reward_matches_the_worked_example_in_expectation() {
        let obs = vec![
            Observation { name: "diagnostic: positive".into(), probability: 0.24, posterior: vec![1.0 / 3.0, 2.0 / 3.0] },
            Observation { name: "diagnostic: negative".into(), probability: 0.76, posterior: vec![18.0 / 19.0, 1.0 / 19.0] },
        ];
        let costs = CostMatrix::binary(1.0, 10.0);
        let prior = [0.8f32, 0.2];
        let mut rng = data::rng::Rng::new(11);
        let n = 200_000;

        let mut mean_reward = |action: usize, act_now_action: usize, query_cost: f32| -> f64 {
            let mut total = 0.0f64;
            for _ in 0..n {
                let y = sample_categorical(&prior, &mut rng);
                total += voi_policy_reward(action, y, act_now_action, &obs, &costs, query_cost, &mut rng) as f64;
            }
            total / n as f64
        };

        // act_now = block (the actual closed-form answer at this prior):
        // E[cost] = 0.8*1 + 0.2*0 = 0.8, reward = -0.8
        assert!((mean_reward(0, 0, 0.0) - (-0.8)).abs() < 0.01, "act_now=block: {}", mean_reward(0, 0, 0.0));
        // act_now = release (never the real closed-form answer here, but the
        // function must not hard-code which action "act now" means):
        // E[cost] = 0.8*0 + 0.2*10 = 2.0, reward = -2.0
        assert!((mean_reward(0, 1, 0.0) - (-2.0)).abs() < 0.02, "act_now=release: {}", mean_reward(0, 1, 0.0));
        // inspect at query_cost=0: E[cost] = 0.48 (the worked example's own
        // number), reward = -0.48, regardless of act_now_action.
        assert!((mean_reward(1, 0, 0.0) - (-0.48)).abs() < 0.01, "inspect: {}", mean_reward(1, 0, 0.0));

        // The real closed-form act_now (block) is strictly better than the
        // wrong one (release) would have been - confirms the function reads
        // act_now_action rather than silently assuming an index.
        assert!(mean_reward(0, 0, 0.0) > mean_reward(0, 1, 0.0), "the real act_now action must beat the wrong one");
    }
}
