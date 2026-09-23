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
//! Evaluation reports three axes, not one: [`rlcd::metrics::soft_ece`],
//! [`rlcd::metrics::soft_nll`], [`rlcd::metrics::posterior_kl`] and
//! [`rlcd::metrics::posterior_error`] audit the PROBABILITY head against
//! calibration (does the model believe correctly) - every one of them
//! against the example's own exact oracle DISTRIBUTION rather than its
//! argmax, which is the difference between measuring a calibrated belief
//! and rewarding a confident one - and [`rlcd::cost::regret`] against
//! every named cost matrix in [`RlcdSpec::eval_costs`] audits the DECISION
//! (does the belief, run through an explicit cost matrix, choose correctly)
//! - on cost matrices the model never trained under, which is the number
//! that tells apart a learned belief from a learned policy.

use decide::decide::{Decide, Limits};
use decide::policy::{choice_loss, Act, PolicyConfig};
use rlcd::scoring::softmax;

// Re-exported for a caller composing a non-text feature source (an image
// embedding, say) against the same head this pipeline trains - see
// `RlcdPipeline::model_mut`'s own doc for why that needs the raw model
// rather than a widened pipeline surface. `pub use` also brings each name
// into scope unqualified for the rest of this file, the same way
// `CostMatrix`/`World` below already do.
pub use decide::decide::Features;
pub use decide::primitives::{confidence, Answer, Opt, Question};
pub use rlcd::scoring::{decision_loss, decision_loss_soft};

use crate::flow::{EvalReport, Flow, Stages, TrainReport};
use crate::{Device, Error, Result};

// Re-exported so an application can turn its own executable world into
// `RlcdExample`s and name cost matrices without depending on `brain-rlcd`
// directly - a sample may name only the `brain` SDK facade as its brain
// dependency (see `samples/README.md`'s rules), never an engine crate.
pub use rlcd::atlas::{check_information_refinement, validate_distribution, DecisionContract, Distribution, Observation, OracleKind, World};
pub use rlcd::cost::{bayes_action, bayes_risk, regret, voi, BayesAction, CostMatrix};
pub use rlcd::metrics::{
    ada_ece, classwise_ece, coverage_accuracy, ece, failure_auroc, posterior_error, posterior_kl, reliability_bins, soft_ece, soft_nll, ReliabilityBin,
};
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

/// How far an oracle target may be from summing to 1 before it is refused.
/// Loose enough for f32 accumulation over a few hundred options, far tighter
/// than any real mistake - a target built from the wrong denominator, or one
/// option short, misses by percent, not by `1e-4`.
const TARGET_TOLERANCE: f32 = 1e-4;

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

/// The task a saved head was fitted for, written into its checkpoint and
/// read back out of it.
///
/// A head is 445k floats. Without this, everything needed to ASK it
/// anything - the question, the option names the target distribution
/// indexes, the action names its Bayes action indexes, whether its encoder
/// may move - lived only in the `RlcdSpec` of the process that trained it.
/// Loading a head therefore produced a model that could score nothing and
/// reported "train before asking for a probability", which is not a missing
/// feature but a checkpoint that does not describe itself.
#[derive(Clone, Debug, PartialEq)]
pub struct TaskContract {
    pub instructions: String,
    pub options: Vec<String>,
    pub actions: Vec<String>,
    pub eval_costs: Vec<(String, Vec<Vec<f32>>)>,
    pub freeze_encoder: bool,
}

/// The key `Decide::save_head` writes [`Provenance::task`] under.
const TRAINED_FOR: &str = "trained_for";
/// The key it writes the base encoder's reference under.
const ADAPTER_OF: &str = "adapter_of";

impl TaskContract {
    fn from_spec(spec: &RlcdSpec) -> TaskContract {
        TaskContract {
            instructions: spec.instructions.clone(),
            options: spec.options.clone(),
            actions: spec.actions.clone(),
            eval_costs: spec.eval_costs.iter().map(|(n, c)| (n.clone(), c.rows())).collect(),
            freeze_encoder: spec.freeze_encoder,
        }
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "pipeline": "rlcd",
            "instructions": self.instructions,
            "options": self.options,
            "actions": self.actions,
            "freeze_encoder": self.freeze_encoder,
            "eval_costs": self.eval_costs.iter().map(|(n, rows)| serde_json::json!({"name": n, "rows": rows})).collect::<Vec<_>>(),
        })
    }

    /// Parsed strictly: a field that is present but the wrong shape is an
    /// error, not a default. A head that silently loaded with an empty
    /// option list would score, and answer wrongly.
    fn from_json(v: &serde_json::Value) -> std::result::Result<TaskContract, String> {
        let pipeline = v.get("pipeline").and_then(|x| x.as_str()).unwrap_or_default();
        if pipeline != "rlcd" {
            return Err(format!("this head was trained by {pipeline:?}, not by RlcdPipeline"));
        }
        let strings = |key: &str| -> std::result::Result<Vec<String>, String> {
            match v.get(key) {
                None => Ok(Vec::new()),
                Some(a) => a
                    .as_array()
                    .ok_or_else(|| format!("{key} is not a list"))?
                    .iter()
                    .map(|x| x.as_str().map(str::to_string).ok_or_else(|| format!("{key} holds a non-string")))
                    .collect(),
            }
        };
        let options = strings("options")?;
        if options.is_empty() {
            return Err("the saved contract names no options".into());
        }
        let mut eval_costs = Vec::new();
        if let Some(list) = v.get("eval_costs") {
            for entry in list.as_array().ok_or("eval_costs is not a list")? {
                let name = entry.get("name").and_then(|x| x.as_str()).ok_or("a cost matrix has no name")?.to_string();
                let rows = entry.get("rows").and_then(|x| x.as_array()).ok_or_else(|| format!("cost matrix {name:?} has no rows"))?;
                let parsed: std::result::Result<Vec<Vec<f32>>, String> = rows
                    .iter()
                    .map(|r| {
                        r.as_array()
                            .ok_or_else(|| format!("cost matrix {name:?} has a non-list row"))?
                            .iter()
                            .map(|x| x.as_f64().map(|f| f as f32).ok_or_else(|| format!("cost matrix {name:?} has a non-numeric entry")))
                            .collect()
                    })
                    .collect();
                eval_costs.push((name, parsed?));
            }
        }
        Ok(TaskContract {
            instructions: v.get("instructions").and_then(|x| x.as_str()).unwrap_or_default().to_string(),
            options,
            actions: strings("actions")?,
            eval_costs,
            freeze_encoder: v.get("freeze_encoder").and_then(serde_json::Value::as_bool).unwrap_or(true),
        })
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
    /// Names for the ACTION index space of [`RlcdSpec::eval_costs`], in
    /// `CostMatrix` row order - e.g. `["block", "release"]`.
    ///
    /// **A different index space from [`RlcdSpec::options`]**, which names
    /// OUTCOMES. A cost matrix need not even be square, and where it is, the
    /// two orders are unrelated: `CostMatrix::binary` puts the conservative
    /// action ("block") at index 0 while a natural outcome order puts the
    /// benign outcome ("healthy") there, so borrowing the outcome name for
    /// an action does not merely read oddly - it prints the OPPOSITE of what
    /// the model decided. Empty means actions are reported by index.
    pub actions: Vec<String>,
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
            actions: Vec::new(),
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
    pub fn actions(mut self, actions: Vec<String>) -> RlcdSpec {
        self.actions = actions;
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
    /// Names for the cost matrices' ACTION index space - see
    /// [`RlcdSpec::actions`] for why this cannot be `question`'s options.
    actions: Vec<String>,
    eval_costs: Vec<(String, CostMatrix)>,
    /// The meta-decision question `train_voi_policy` trains and
    /// `voi_policy_action` queries - `{act now, inspect}` by construction
    /// (see that function's own doc).
    ///
    /// A different QUESTION from `question`, but **not a different head**.
    /// `Decide` scores (state, option-text) pairs through one option scorer,
    /// which is what lets the two questions coexist at all - and equally
    /// what means training either one moves the other. See
    /// [`RlcdPipeline::train_voi_policy`] for the consequence a caller has
    /// to handle.
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

    /// Install the held-out set this pipeline is measured on.
    ///
    /// [`Stages::run_train`] sets this from its spec, so training needs no
    /// call here. A pipeline built from a SAVED head has weights and a task
    /// contract but no data - the contract records what the head answers,
    /// not what it was scored on - and this is how a caller hands back the
    /// held-out set so a loaded head can be re-measured rather than trusted.
    pub fn eval_set(&mut self, eval: Vec<RlcdExample>) {
        self.eval = eval;
    }

    /// The loaded model itself, for a caller driving `Decide` below this
    /// pipeline's own `(state, target)` surface - e.g. training against
    /// `Features` built from a non-text row source via `Features::from_parts`,
    /// which `RlcdSpec`'s plain-text `RlcdExample` cannot express. Everything
    /// `RlcdPipeline::builder` did (loading the checkpoint, resolving the
    /// device, sizing `Limits`) already happened; this only hands back what
    /// it built.
    pub fn model_mut(&mut self) -> &mut Decide {
        &mut self.model
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
    ///
    /// **This moves the same head [`RlcdPipeline::probability`] answers
    /// from**, and measurably: on the sample's own 400/600-step run it costs
    /// 0.023 of soft ECE and raises safety-critical regret from 0.140 to
    /// 0.200. The states it trains on are the ones it degrades most, since
    /// they are the ones it repeatedly rescores under a different question -
    /// pushed far enough (12k steps there) the belief at "no evidence yet"
    /// falls from 0.20 to 0.04 and crosses the block/release boundary, so a
    /// model that PASSED a witness audit before this call can fail the same
    /// audit after it.
    ///
    /// A caller must therefore **re-evaluate and re-audit after calling
    /// this, and save only then.** Anything measured beforehand describes
    /// weights that no longer exist.
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
        // Both sets, before any step is spent: a target that is not a
        // distribution trains the model toward something that is not one
        // either, and `decision_loss_soft` documents normalization as the
        // caller's invariant without anywhere that establishes it. An
        // evaluation target that is not a distribution is worse still - it
        // is the reference every calibration number is measured against.
        for (set, examples) in [("training", &spec.train), ("evaluation", &spec.eval)] {
            for (i, ex) in examples.iter().enumerate() {
                validate_distribution(&ex.target, spec.options.len(), TARGET_TOLERANCE, &format!("{set} example {i} ({:?})", ex.state))
                    .map_err(Error::MissingArgument)?;
            }
        }
        for (name, costs) in &spec.eval_costs {
            if costs.n_outcomes() != spec.options.len() {
                return Err(Error::MissingArgument(format!(
                    "cost matrix {name:?} scores {} outcomes, but {} options were declared",
                    costs.n_outcomes(),
                    spec.options.len()
                )));
            }
            if !spec.actions.is_empty() && costs.n_actions() != spec.actions.len() {
                return Err(Error::MissingArgument(format!(
                    "cost matrix {name:?} has {} actions, but {} action names were declared",
                    costs.n_actions(),
                    spec.actions.len()
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
        self.actions = spec.actions.clone();
        self.eval_costs = spec.eval_costs.clone();
        // Recorded here rather than in `run_save`, because this is where the
        // contract is known and `Stages::run_save` takes `&self`. A head
        // written without it loads into a model that cannot be asked
        // anything - see `TaskContract`.
        let mut p = self.model.provenance().clone();
        p.task = TaskContract::from_spec(spec).to_json();
        self.model.set_provenance(p);
        Ok(TrainReport { steps: spec.steps, final_loss: tail, seconds: 0.0 })
    }

    fn run_eval(&mut self) -> Result<EvalReport> {
        if self.eval.is_empty() {
            return Ok(EvalReport::default());
        }
        let q = self.question.clone().ok_or_else(|| Error::MissingArgument("train before evaluating".into()))?;
        let eval = std::mem::take(&mut self.eval);

        let mut probs: Vec<Vec<f32>> = Vec::with_capacity(eval.len());
        let mut targets: Vec<Vec<f32>> = Vec::with_capacity(eval.len());
        let mut correct: Vec<bool> = Vec::with_capacity(eval.len());
        for ex in &eval {
            let scores = self.model.score(&ex.state, std::slice::from_ref(&q)).map_err(Error::Backend)?;
            let p = softmax(&scores[0]);
            correct.push(argmax(&p) == argmax(&ex.target));
            probs.push(p);
            targets.push(ex.target.clone());
        }
        self.eval = eval.clone();

        let hit = correct.iter().filter(|&&c| c).count();
        // Every calibration number here is scored against the example's own
        // EXACT oracle distribution, never against its argmax. An RlcdExample
        // carries `[1/3, 2/3]` because that is what the world does; collapsing
        // it to "faulty" and scoring the model on how close it got to
        // `[0, 1]` rewards precisely the overconfidence this pipeline exists
        // to measure. See `rlcd::metrics::soft_ece` for the number an exact
        // oracle scored under the hard-label version that stood here.
        let mut notes = vec![
            ("ECE".into(), soft_ece(&probs, &targets, 10)),
            ("NLL".into(), soft_nll(&probs, &targets)),
            ("KL".into(), posterior_kl(&probs, &targets)),
            ("PostErr".into(), posterior_error(&probs, &targets)),
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
            lines.push(format!("  Bayes action under \"{name}\": {} (risk {:.4})", action_name(&self.actions, action.action), action.risk));
        }
        Ok(lines.join("\n"))
    }

    fn turn_prompt(&self) -> &str {
        "state> "
    }
}

/// What to call action `i`. Falls back to the index rather than to an
/// outcome name: an unnamed action is unhelpful, a MISNAMED one is worse
/// than unhelpful - see [`RlcdSpec::actions`].
fn action_name(actions: &[String], i: usize) -> String {
    actions.get(i).cloned().unwrap_or_else(|| format!("action {i}"))
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0
}

/// The task contract and base-encoder reference a saved head carries, read
/// from its safetensors header alone - no tensor bytes are touched.
fn read_head_contract(path: &str) -> Result<(TaskContract, String)> {
    let meta = checkpoint::st::read_metadata(path).map_err(|e| Error::Backend(format!("read {path} header: {e}")))?;
    let config: serde_json::Value = meta
        .get("brain.config")
        .ok_or_else(|| Error::Backend(format!("{path} carries no brain.config")))
        .and_then(|s| serde_json::from_str(s).map_err(|e| Error::Backend(format!("{path}: brain.config is not JSON: {e}"))))?;
    let task = config.get(TRAINED_FOR).ok_or_else(|| {
        Error::Backend(format!(
            "{path} does not say what it was trained for, so the question, options and costs it needs cannot be recovered - \
             retrain and save with this version, or supply them by training rather than loading"
        ))
    })?;
    let contract = TaskContract::from_json(task).map_err(|e| Error::Backend(format!("{path}: {e}")))?;
    let base = config.get(ADAPTER_OF).and_then(|x| x.as_str()).unwrap_or_default().to_string();
    Ok((contract, base))
}

pub struct RlcdPipelineBuilder {
    dir: String,
    head: Option<String>,
    device: Device,
    limits: Limits,
    seed: u64,
}

impl RlcdPipeline {
    /// Install a [`TaskContract`] on a pipeline that was loaded rather than
    /// trained, so it can be asked the question its weights were fitted for.
    fn adopt(&mut self, c: TaskContract) {
        self.question = Some(Question::Choice { instructions: c.instructions, options: c.options.iter().map(Opt::new).collect() });
        self.actions = c.actions;
        self.eval_costs = c.eval_costs.into_iter().map(|(n, rows)| (n, CostMatrix::from_rows(&rows))).collect();
        // Restored explicitly. `set_encoder_frozen` otherwise keeps
        // `Decide::new_on`'s `false`, so a head saved from a frozen run
        // would come back with a TRAINABLE encoder, any later training would
        // move it, and `save_head` would then refuse to write the result at
        // all - having already changed the model.
        self.model.set_encoder_frozen(c.freeze_encoder);
    }
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
        let mut pipeline = RlcdPipeline { model, question: None, eval: Vec::new(), actions: Vec::new(), eval_costs: Vec::new(), voi_question: None };
        if let Some(head) = &self.head {
            let (contract, base) = read_head_contract(head)?;
            // The head names the encoder it is an adapter TO. Attaching it to
            // a different one loads, scores, and is a different model, which
            // nothing downstream can tell - so it is refused here.
            let want = pipeline.model.provenance().base.clone();
            if !base.is_empty() && !want.is_empty() && base != want {
                return Err(Error::Backend(format!("{head} is an adapter of {base:?}, but it is being loaded onto {want:?}")));
            }
            pipeline.adopt(contract);
        }
        Ok(pipeline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A target that is not a distribution has to be refused where it
    /// enters, not silently trained on. `decision_loss_soft` names
    /// normalization as the caller's invariant and nothing established it.
    #[test]
    fn a_target_that_is_not_a_distribution_is_refused_by_name() {
        let base = || {
            RlcdSpec::default()
                .instructions("q")
                .options(vec!["a".into(), "b".into()])
                .steps(1)
        };
        for bad in [vec![0.5f32, 0.6], vec![0.5, -0.5], vec![f32::NAN, 1.0], vec![1.0]] {
            let spec = base().train(vec![RlcdExample::new("s", bad.clone())]);
            let err = validate_distribution(&spec.train[0].target, spec.options.len(), TARGET_TOLERANCE, "t").expect_err(&format!("{bad:?} is not a distribution"));
            assert!(!err.is_empty());
        }
        validate_distribution(&[1.0 / 3.0, 2.0 / 3.0], 2, TARGET_TOLERANCE, "t").expect("the oracle's own posterior must pass");
    }

    /// The contract has to survive the round trip through a checkpoint's
    /// config, or a loaded head is 445k floats nobody can ask anything.
    #[test]
    fn the_task_contract_round_trips_through_json() {
        let spec = RlcdSpec::default()
            .instructions("is this device faulty")
            .options(vec!["healthy".into(), "faulty".into()])
            .actions(vec!["block".into(), "release".into()])
            .freeze_encoder(true)
            .eval_costs(vec![("safety-critical".into(), CostMatrix::binary(1.0, 10.0))]);
        let written = TaskContract::from_spec(&spec);
        let read = TaskContract::from_json(&written.to_json()).expect("what we wrote must parse");
        assert_eq!(read, written);
        assert_eq!(read.eval_costs[0].1, vec![vec![1.0, 0.0], vec![0.0, 10.0]]);
    }

    /// Refused rather than defaulted. A head that loaded with no options
    /// would score, answer, and be wrong with no indication.
    #[test]
    fn a_contract_that_does_not_describe_the_task_is_refused() {
        for bad in [
            serde_json::json!({"pipeline": "conversion", "options": ["a", "b"]}),
            serde_json::json!({"pipeline": "rlcd", "options": []}),
            serde_json::json!({"pipeline": "rlcd"}),
            serde_json::json!({"pipeline": "rlcd", "options": ["a"], "eval_costs": [{"name": "c"}]}),
            serde_json::json!({"pipeline": "rlcd", "options": ["a"], "actions": [7]}),
        ] {
            assert!(TaskContract::from_json(&bad).is_err(), "should have been refused: {bad}");
        }
    }

    /// The bug this exists to prevent: under `CostMatrix::binary` action 0
    /// is "block" while outcome 0 is "healthy", so naming an action from the
    /// outcome list reports the opposite of the decision. A model believing
    /// `P(faulty) = 0.635` under symmetric costs BLOCKS, and the sample
    /// printed "healthy".
    #[test]
    fn an_action_is_named_from_the_action_list_not_the_outcome_list() {
        let outcomes = ["healthy".to_string(), "faulty".to_string()];
        let actions = ["block".to_string(), "release".to_string()];
        let belief = [0.365f32, 0.635];
        let chosen = bayes_action(&belief, &CostMatrix::binary(1.0, 1.0)).action;
        assert_eq!(chosen, 0, "more likely faulty than not, under equal costs, is a block");
        assert_eq!(action_name(&actions, chosen), "block");
        assert_ne!(action_name(&actions, chosen), outcomes[chosen], "the two index spaces must not be interchangeable");
        assert_eq!(action_name(&[], 1), "action 1", "an unnamed action reports its index, never an outcome");
    }

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
