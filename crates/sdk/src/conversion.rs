// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain::ConversionPipeline` - a live conversation's chance of closing,
//! updated every turn, with a routing decision attached to it.
//!
//! ```no_run
//! use brain::{ConversionPipeline, ConversionSpec};
//! ConversionPipeline::from_pretrained("/path/to/all-MiniLM-L6-v2")
//!     .train(ConversionSpec::default())
//!     .evaluate()
//!     .save("out/sales-head.safetensors")
//!     .tui()
//!     .report()
//!     .finish()?;
//! # Ok::<(), brain::Error>(())
//! ```
//!
//! Same chain as every other pipeline here; what differs is the payload.
//! [`crate::DecisionPipeline`] answers one question about a fixed state. This
//! one carries a conversation that GROWS, and re-answers the same question
//! after every turn - which is what turns a probability into a trajectory.
//!
//! ## The method
//!
//! Two papers, both by Nandakishor M, implemented as they describe:
//!
//! *SalesRLAgent* (arXiv 2503.23303) gives the formulation - conversion
//! prediction as a sequential decision problem - and the training recipe: a
//! supervised phase to initialize the policy, then a reinforcement phase with
//! conservative updates and strong regularization, over a curriculum that
//! starts simple and outcome-balanced batches. [`decide::policy`] carries the
//! objective; [`decide::salesconv::Curriculum`] carries the sampling.
//!
//! *Confidence-Aware Routing* (arXiv 2510.01237) gives the second half: three
//! confidence signals combined into one score, and four pathways keyed off it
//! at the paper's own thresholds. [`decide::routing`] carries that.
//!
//! What is faithful and what is adapted is written down in each of those
//! modules, next to the code that does it.

use std::path::Path;

use decide::decide::{Decide, Limits};
use decide::policy::{self, PolicyConfig, Turn, PROPOSITION};
use decide::primitives::{Answer, Question};
use decide::routing::{self, ConfidenceNet, Projection, Route, Router, Signals};
use decide::salesconv::{Conversation, Curriculum, Message, SalesConversations};

use crate::flow::{EvalReport, Flow, Stages, TrainReport};
use crate::{Device, Error, Result};

pub use decide::routing::Route as RoutingDecision;
pub use decide::salesconv::{Conversation as SalesConversation, Message as SalesMessage};

/// What the model says about a conversation as it stands.
#[derive(Clone, Debug)]
pub struct Verdict {
    /// Probability this conversation converts, in `[0, 1]`.
    pub probability: f32,
    /// The routing paper's unified score, in `[0, 1]`.
    pub confidence: f32,
    /// Its three components, so a caller can see which one was low.
    pub signals: Signals,
    /// What to do about it.
    pub route: Route,
}

/// The question every turn asks. Fixed, because the whole model is trained
/// against this one proposition - unlike a choice pipeline, whose options are
/// genuinely per-request.
const QUESTION_TEXT: &str = "will this sales conversation end in a closed deal";

/// The encoder arrives pretrained and the head does not.
const ENCODER_LR: f32 = 2e-5;
const HEAD_LR: f32 = 1e-3;

/// How hard training leans toward the end of a conversation - see
/// `Conversation::sample_turn` for what a uniform sampler does instead.
const TURN_BIAS: f32 = 0.85;

/// The policy phase moves an already-fitted model, so it moves it slowly: a
/// reinforcement update on a supervised initialization that is already close
/// can undo the calibration it started from.
const POLICY_ENCODER_LR: f32 = 5e-6;
const POLICY_HEAD_LR: f32 = 2e-4;

pub struct ConversionPipeline {
    model: Decide,
    router: Router,
    projection: Option<Projection>,
    phi: Option<ConfidenceNet>,
    /// The conversation the interactive stages are building up.
    live: Vec<Message>,
    eval: Vec<Conversation>,
}

impl std::fmt::Debug for ConversionPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversionPipeline").finish_non_exhaustive()
    }
}

/// Load the sales-conversation dataset from a directory holding
/// `train.jsonl` and `test.jsonl`.
pub fn sales_conversations(dir: impl AsRef<Path>) -> Result<SalesConversations> {
    SalesConversations::load(dir.as_ref()).map_err(Error::Backend)
}

impl ConversionPipeline {
    /// Load an encoder checkpoint directory and start a stage chain on it.
    pub fn from_pretrained(dir: impl AsRef<str>) -> Flow<ConversionPipeline> {
        Flow::new(ConversionPipeline::builder(dir).load())
    }

    pub fn builder(dir: impl AsRef<str>) -> ConversionPipelineBuilder {
        ConversionPipelineBuilder {
            dir: dir.as_ref().to_string(),
            head: None,
            device: Device::default(),
            limits: Limits { cap_rows: 8192, cap_slots: 8, max_span: 256, overlap: 32 },
            seed: 0,
        }
    }

    fn question() -> Question {
        Question::Noul {
            instructions: QUESTION_TEXT.to_string(),
            yes: Some("the deal closes".to_string()),
            no: Some("the deal is lost".to_string()),
        }
    }

    /// P(converts) for a conversation as it stands, and nothing else.
    pub fn probability(&mut self, turns: &[Message]) -> Result<f32> {
        let state = render(turns);
        let q = ConversionPipeline::question();
        let mut a = self.model.decide(&state, std::slice::from_ref(&q)).map_err(Error::Backend)?;
        match a.pop() {
            Some(Answer::Noul { noul }) => Ok(noul),
            _ => Err(Error::Backend("the model did not return a probability".into())),
        }
    }

    /// The probability, its confidence, and where the decision should go.
    ///
    /// Costs more than [`ConversionPipeline::probability`]: the confidence
    /// signals read every layer's hidden states back off the device.
    pub fn verdict(&mut self, turns: &[Message]) -> Result<Verdict> {
        let state = render(turns);
        let q = ConversionPipeline::question();
        let req = self.model.pack_request(&state, std::slice::from_ref(&q)).map_err(Error::Backend)?;
        let scores = self.model.run_packed(&req);
        let probability = decide::primitives::sigmoid(scores[PROPOSITION]);
        let snap = self.model.repr_snapshot(req.state_rows as usize, PROPOSITION);

        let layers: Vec<&[f32]> = snap.layers.iter().map(|v| v.as_slice()).collect();
        let signals = Signals {
            // Before the projection is fitted the alignment is measured
            // directly. The two spaces are the same width and come from the
            // same encoder, so this is a real comparison rather than a
            // placeholder - the fit sharpens it, it does not create it.
            semantic: match &self.projection {
                Some(p) => routing::cosine(&p.apply(&snap.decision), &snap.reference),
                None => routing::cosine(&snap.decision, &snap.reference),
            }
            .clamp(0.0, 1.0),
            convergence: routing::convergence(&layers, 1e-6),
            learned: match &self.phi {
                Some(n) => n.predict(&snap.reference),
                // Until phi is fitted, fall back to how far the committed
                // probability sits from undecided - the sharpness a Bernoulli
                // always has.
                None => (2.0 * (probability - 0.5)).abs(),
            },
        };
        let confidence = self.router.score(&signals);
        Ok(Verdict { probability, confidence, signals, route: self.router.route(confidence) })
    }

    /// The whole trajectory: P(converts) after each turn.
    ///
    /// Each point sees only the turns up to it, which is the only way the
    /// number means anything - a model shown the closing turn can read the
    /// outcome off it.
    pub fn trajectory(&mut self, conv: &Conversation) -> Result<Vec<f32>> {
        (0..conv.len()).map(|t| self.probability(&conv.turns[..=t])).collect()
    }

    pub fn save_head(&self, path: impl AsRef<str>) -> Result<()> {
        self.model.save_head(path.as_ref()).map_err(Error::Backend)
    }

    /// Phase one: fit the per-turn probability the dataset recorded.
    ///
    /// Supervised, against a SOFT target, because the label is a probability
    /// and rounding it to a class would throw away the calibration the policy
    /// phase would then have to rediscover from a binary reward.
    fn warm_start(
        &mut self,
        train: &[Conversation],
        steps: usize,
        seed: u64,
        log: &mut dyn FnMut(usize, f32),
    ) -> Result<f32> {
        let q = ConversionPipeline::question();
        let mut cur = Curriculum::new(train);
        let mut rng = data::rng::Rng::new(seed);
        let (mut tail, tail_n) = (0.0f32, (steps / 10).max(1));
        for step in 0..steps {
            let c = &train[cur.draw(step as f32 / steps.max(1) as f32, &mut rng)];
            let t = c.sample_turn(TURN_BIAS, &mut rng);
            let target = c.trajectory[t].clamp(0.0, 1.0);
            let state = c.prefix(t);
            let l = self
                .model
                .train_step_with(&state, &q, ENCODER_LR, HEAD_LR, |s| policy::bce_loss(s, target))
                .map_err(Error::Backend)?;
            // Reported as the DIVERGENCE from the label, not the raw
            // cross-entropy: a label of 0.5 costs ln 2 even from a perfect
            // model, so the raw number never approaches zero and cannot be
            // read as progress. This one is zero exactly when the model
            // matches the label.
            let l = l - policy::target_entropy(target);
            log(step, l);
            if step + tail_n >= steps {
                tail += l / tail_n as f32;
            }
        }
        Ok(tail)
    }

    /// Phase two: the sequential-decision objective.
    ///
    /// The committed probability is scored by a proper rule against the
    /// outcome, discounted by how far the turn is from it, inside a trust
    /// region around the probability the collecting pass committed to. See
    /// [`decide::policy`] for why the reward has to be a proper scoring rule
    /// and what that leaves of PPO.
    fn policy_phase(
        &mut self,
        train: &[Conversation],
        steps: usize,
        already: usize,
        seed: u64,
        cfg: &PolicyConfig,
        log: &mut dyn FnMut(usize, f32),
    ) -> Result<f32> {
        let q = ConversionPipeline::question();
        let mut cur = Curriculum::new(train);
        let mut rng = data::rng::Rng::new(seed ^ 0x5eed);
        let (mut tail, tail_n) = (0.0f32, (steps / 10).max(1));
        for step in 0..steps {
            let c = &train[cur.draw(step as f32 / steps.max(1) as f32, &mut rng)];
            let t = c.sample_turn(TURN_BIAS, &mut rng);
            let state = c.prefix(t);
            // Collect: the probability as it stands now is what the trust
            // region is measured against, so it is read before the update.
            let scores = self.model.score(&state, std::slice::from_ref(&q)).map_err(Error::Backend)?;
            let turn = Turn {
                old_prob: decide::primitives::sigmoid(scores[0][PROPOSITION]),
                outcome: c.outcome,
                turns_to_end: (c.len() - 1 - t) as u32,
            };
            let l = self
                .model
                .train_step_with(&state, &q, POLICY_ENCODER_LR, POLICY_HEAD_LR, |s| {
                    policy::policy_loss(s, &turn, cfg)
                })
                .map_err(Error::Backend)?;
            log(already + step, l);
            if step + tail_n >= steps {
                tail += l / tail_n as f32;
            }
        }
        Ok(tail)
    }

    /// Phase three: fit the routing paper's projection, confidence network and
    /// signal weights, on conversations the model did not train on.
    ///
    /// On the EVAL split deliberately: a confidence estimator fitted to the
    /// training set learns the training set's accuracy, which is the one
    /// number it must not be calibrated against.
    fn calibrate(&mut self, cal: &[Conversation]) -> Result<usize> {
        let q = ConversionPipeline::question();
        let (mut decision, mut reference, mut correct, mut signals) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for c in cal {
            // The last turn: the point the whole conversation is about, and
            // the only one whose correct answer is not in dispute.
            let t = c.len() - 1;
            let state = c.prefix(t);
            let req = self.model.pack_request(&state, std::slice::from_ref(&q)).map_err(Error::Backend)?;
            let scores = self.model.run_packed(&req);
            let p = decide::primitives::sigmoid(scores[PROPOSITION]);
            let snap = self.model.repr_snapshot(req.state_rows as usize, PROPOSITION);
            let layers: Vec<&[f32]> = snap.layers.iter().map(|v| v.as_slice()).collect();
            let hit = f32::from((p >= 0.5) == c.outcome);
            signals.push(Signals {
                semantic: routing::cosine(&snap.decision, &snap.reference).clamp(0.0, 1.0),
                convergence: routing::convergence(&layers, 1e-6),
                learned: (2.0 * (p - 0.5)).abs(),
            });
            decision.push(snap.decision);
            reference.push(snap.reference);
            correct.push(hit);
        }
        if decision.is_empty() {
            return Err(Error::MissingArgument("nothing to calibrate the router on".into()));
        }
        // (1)'s projection: decision representation -> reference space.
        self.projection = Projection::fit(&decision, &reference, 1.0).ok();
        // (3)'s network, against whether the decision actually held up.
        let mut net = ConfidenceNet::new(reference[0].len(), 17);
        net.fit(&reference, &correct, 300, 0.2, 1e-4);
        self.phi = Some(net);
        // Re-measure the signals through the fitted parts before weighting
        // them, so (4) is fitted on the values it will actually see.
        for (s, (d, r)) in signals.iter_mut().zip(decision.iter().zip(&reference)) {
            if let Some(p) = &self.projection {
                s.semantic = routing::cosine(&p.apply(d), r).clamp(0.0, 1.0);
            }
            if let Some(n) = &self.phi {
                s.learned = n.predict(r);
            }
        }
        // A fit that cannot find a predictive signal leaves the equal weights
        // in place rather than failing the run: the thresholds still work, the
        // score is just less sharp, and that is reported instead of hidden.
        let _ = self.router.fit_weights(&signals, &correct, 1e-3);
        Ok(signals.len())
    }
}

/// The state string for a conversation: one turn per line, speaker first.
fn render(turns: &[Message]) -> String {
    turns.iter().map(Message::render).collect::<Vec<_>>().join("\n")
}

/// What a conversion run needs.
#[derive(Clone, Debug)]
pub struct ConversionSpec {
    pub train: Vec<Conversation>,
    /// Held out, for [`Flow::evaluate`] and for fitting the router.
    pub eval: Vec<Conversation>,
    /// Supervised steps, fitting the recorded per-turn probability.
    pub warmup_steps: usize,
    /// Policy-gradient steps.
    pub policy_steps: usize,
    /// How many held-out conversations the router is fitted on.
    pub calibration: usize,
    pub seed: u64,
    pub policy: PolicyConfig,
}

impl Default for ConversionSpec {
    fn default() -> ConversionSpec {
        ConversionSpec {
            train: Vec::new(),
            eval: Vec::new(),
            warmup_steps: 3000,
            policy_steps: 1000,
            calibration: 200,
            seed: 0,
            policy: PolicyConfig::default(),
        }
    }
}

impl ConversionSpec {
    pub fn train(mut self, train: Vec<Conversation>) -> ConversionSpec {
        self.train = train;
        self
    }
    pub fn eval(mut self, eval: Vec<Conversation>) -> ConversionSpec {
        self.eval = eval;
        self
    }
    pub fn warmup_steps(mut self, n: usize) -> ConversionSpec {
        self.warmup_steps = n;
        self
    }
    pub fn policy_steps(mut self, n: usize) -> ConversionSpec {
        self.policy_steps = n;
        self
    }
    pub fn calibration(mut self, n: usize) -> ConversionSpec {
        self.calibration = n;
        self
    }
    pub fn seed(mut self, seed: u64) -> ConversionSpec {
        self.seed = seed;
        self
    }
}

impl Stages for ConversionPipeline {
    type TrainSpec = ConversionSpec;

    fn describe(&self) -> String {
        format!(
            "conversion model, {} training steps so far, router weights [{:.2}, {:.2}, {:.2}]",
            self.model.steps_taken(),
            self.router.weights[0],
            self.router.weights[1],
            self.router.weights[2],
        )
    }

    fn run_train(&mut self, spec: &ConversionSpec, log: &mut dyn FnMut(usize, f32)) -> Result<TrainReport> {
        if spec.train.is_empty() {
            return Err(Error::MissingArgument(
                "no conversations to train on - run samples/decision/salesagent/fetch-dataset.sh".into(),
            ));
        }
        println!("  phase 1/3: supervised warm start, {} steps", spec.warmup_steps);
        let warm = self.warm_start(&spec.train, spec.warmup_steps, spec.seed, log)?;
        println!("  phase 2/3: policy gradient, {} steps (final supervised loss {warm:.4})", spec.policy_steps);
        let tail =
            self.policy_phase(&spec.train, spec.policy_steps, spec.warmup_steps, spec.seed, &spec.policy, log)?;
        // At most HALF the held-out split, whatever was asked for: the router
        // must be fitted on conversations the model did not train on, and
        // `evaluate` must then score conversations the ROUTER was not fitted
        // on either. Taking the requested slice unconditionally consumed the
        // whole split and left evaluation reporting on nothing.
        let cal = spec.calibration.min(spec.eval.len() / 2);
        if cal == 0 {
            return Err(Error::MissingArgument(
                "the held-out split needs at least two conversations: one to fit the router on, one to score"
                    .into(),
            ));
        }
        if cal < spec.calibration {
            println!(
                "  (calibrating on {cal} rather than {} - half of a {}-conversation held-out split)",
                spec.calibration,
                spec.eval.len()
            );
        }
        println!("  phase 3/3: fitting the router on {cal} held-out conversations");
        let fitted = self.calibrate(&spec.eval[..cal])?;
        println!(
            "    signal weights: semantic {:.2}, convergence {:.2}, learned {:.2}  (over {fitted} conversations)",
            self.router.weights[0], self.router.weights[1], self.router.weights[2]
        );
        // Everything after the calibration slice, so evaluation never scores a
        // conversation the router was fitted on.
        self.eval = spec.eval[cal..].to_vec();
        Ok(TrainReport { steps: spec.warmup_steps + spec.policy_steps, final_loss: tail, seconds: 0.0 })
    }

    fn run_eval(&mut self) -> Result<EvalReport> {
        if self.eval.is_empty() {
            return Ok(EvalReport::default());
        }
        let eval = std::mem::take(&mut self.eval);
        let (mut hit, mut brier, mut traj_err, mut traj_n) = (0usize, 0.0f32, 0.0f32, 0usize);
        let mut scored: Vec<(f32, bool)> = Vec::with_capacity(eval.len());
        let mut routes = [0usize; 4];
        let mut routed_hit = [0usize; 4];
        for c in &eval {
            let v = self.verdict(&c.turns)?;
            let p = v.probability;
            let correct = (p >= 0.5) == c.outcome;
            hit += usize::from(correct);
            let y = f32::from(c.outcome);
            brier += (p - y) * (p - y) / eval.len() as f32;
            scored.push((p, c.outcome));
            let r = match v.route {
                Route::Local => 0,
                Route::Retrieve => 1,
                Route::Escalate => 2,
                Route::Human => 3,
            };
            routes[r] += 1;
            routed_hit[r] += usize::from(correct);
            // Trajectory agreement, on a sample of turns so the eval stays
            // linear in conversations rather than in total turns.
            for t in (0..c.len()).step_by(3) {
                traj_err += (self.probability(&c.turns[..=t])? - c.trajectory[t]).abs();
                traj_n += 1;
            }
        }
        self.eval = eval;
        let n = scored.len();
        let mut notes = vec![
            ("AUC-ROC".into(), auc_roc(&scored)),
            ("Brier".into(), brier),
            ("per-turn MAE".into(), traj_err / traj_n.max(1) as f32),
        ];
        for (i, name) in ["act", "retrieve", "escalate", "human"].iter().enumerate() {
            if routes[i] > 0 {
                notes.push((format!("{name} n"), routes[i] as f32));
                notes.push((format!("{name} acc"), routed_hit[i] as f32 / routes[i] as f32));
            }
        }
        Ok(EvalReport { accuracy: hit as f32 / n as f32, items: n, notes })
    }

    fn run_save(&self, path: &str) -> Result<()> {
        self.save_head(path)
    }

    fn turn_prompt(&self) -> &str {
        "customer/rep> "
    }

    /// One more turn of the live conversation, and the verdict after it.
    ///
    /// Input is `speaker: text`; a line without a speaker is taken as the
    /// customer, since that is who moves a conversation. `reset` starts over.
    fn run_turn(&mut self, input: &str) -> Result<String> {
        if input.trim() == "reset" {
            self.live.clear();
            return Ok("  (conversation reset)".into());
        }
        let (speaker, text) = match input.split_once(':') {
            Some((s, t)) if matches!(s.trim(), "customer" | "rep" | "sales_rep") => {
                (if s.trim() == "customer" { "customer" } else { "sales_rep" }, t.trim())
            }
            _ => ("customer", input.trim()),
        };
        self.live.push(Message { speaker: speaker.to_string(), text: text.to_string() });
        let live = self.live.clone();
        let v = self.verdict(&live)?;
        let bar: String = {
            let filled = (v.probability * 24.0).round() as usize;
            format!("[{}{}]", "#".repeat(filled), ".".repeat(24 - filled))
        };
        Ok(format!(
            "  turn {:>2}  P(closes) {:.3} {bar}\n           confidence {:.3} -> {} ({})\n           \
             semantic {:.2}  convergence {:.2}  learned {:.2}",
            self.live.len(),
            v.probability,
            v.confidence,
            format_args!("{:?}", v.route).to_string().to_lowercase(),
            v.route.advice(),
            v.signals.semantic,
            v.signals.convergence,
            v.signals.learned,
        ))
    }
}

impl Flow<ConversionPipeline> {
    /// Replay whole conversations turn by turn, printing the probability as it
    /// moves - the output the sequential formulation exists to produce.
    ///
    /// A stage of its own rather than part of `evaluate`, because it is the
    /// thing a person reads: `evaluate` returns numbers about a split, this
    /// shows what one conversation looked like from the inside.
    pub fn replay(self, convs: Vec<Conversation>) -> Flow<ConversionPipeline> {
        self.stage("replay", move |p| {
            for (i, c) in convs.iter().enumerate() {
                println!(
                    "
  conversation {}  ({} turns, {}, ended: {})",
                    i + 1,
                    c.len(),
                    c.industry,
                    if c.outcome { "CLOSED" } else { "lost" }
                );
                for t in 0..c.len() {
                    let v = p.verdict(&c.turns[..=t])?;
                    let filled = (v.probability * 20.0).round() as usize;
                    let m = &c.turns[t];
                    let text: String = m.text.chars().take(58).collect();
                    println!(
                        "    {:>2} {:<9} {:.2} [{}{}]  label {:.2}  {:<8} {}",
                        t + 1,
                        m.speaker,
                        v.probability,
                        "#".repeat(filled),
                        ".".repeat(20 - filled),
                        c.trajectory[t],
                        format_args!("{:?}", v.route).to_string().to_lowercase(),
                        text,
                    );
                }
            }
            Ok(Some(format!("{} conversations", convs.len())))
        })
    }
}

/// Area under the ROC curve, by rank - the metric the paper reports alongside
/// accuracy, and the one that does not move when the threshold does.
fn auc_roc(scored: &[(f32, bool)]) -> f32 {
    let mut v = scored.to_vec();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (pos, neg) = (v.iter().filter(|x| x.1).count(), v.iter().filter(|x| !x.1).count());
    if pos == 0 || neg == 0 {
        return 0.5;
    }
    // Mann-Whitney U over average ranks, so ties score 0.5 rather than
    // whichever way the sort happened to break them.
    let mut rank_sum = 0.0f64;
    let mut i = 0usize;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1].0 == v[i].0 {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        rank_sum += avg * v[i..=j].iter().filter(|x| x.1).count() as f64;
        i = j + 1;
    }
    ((rank_sum - pos as f64 * (pos as f64 + 1.0) / 2.0) / (pos as f64 * neg as f64)) as f32
}

pub struct ConversionPipelineBuilder {
    dir: String,
    head: Option<String>,
    device: Device,
    limits: Limits,
    seed: u64,
}

impl ConversionPipelineBuilder {
    /// Trained head weights. Without this the head is random and every
    /// probability is noise.
    pub fn head(mut self, path: impl AsRef<str>) -> ConversionPipelineBuilder {
        self.head = Some(path.as_ref().to_string());
        self
    }

    pub fn device(mut self, device: Device) -> ConversionPipelineBuilder {
        self.device = device;
        self
    }

    pub fn limits(mut self, limits: Limits) -> ConversionPipelineBuilder {
        self.limits = limits;
        self
    }

    pub fn seed(mut self, seed: u64) -> ConversionPipelineBuilder {
        self.seed = seed;
        self
    }

    pub fn load(self) -> Result<ConversionPipeline> {
        let model = crate::decision::load_decide(&self.dir, self.head.as_deref(), &self.device, self.limits, self.seed)?;
        Ok(ConversionPipeline {
            model,
            router: Router::default(),
            projection: None,
            phi: None,
            live: Vec::new(),
            eval: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A perfect ranking is 1.0, a reversed one 0.0, and an uninformative one
    /// 0.5 - the three points that say the metric is the metric.
    #[test]
    fn auc_is_a_ranking_measure() {
        let perfect = [(0.1f32, false), (0.2, false), (0.8, true), (0.9, true)];
        assert!((auc_roc(&perfect) - 1.0).abs() < 1e-6);
        let reversed = [(0.9f32, false), (0.8, false), (0.2, true), (0.1, true)];
        assert!(auc_roc(&reversed).abs() < 1e-6);
        // All-equal scores carry no ranking information at all.
        let tied = [(0.5f32, false), (0.5, true), (0.5, false), (0.5, true)];
        assert!((auc_roc(&tied) - 0.5).abs() < 1e-6, "ties scored {}", auc_roc(&tied));
        // One class only: undefined, reported as chance rather than NaN.
        assert!((auc_roc(&[(0.3f32, true), (0.7, true)]) - 0.5).abs() < 1e-6);
    }
}
