// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain::ControlPipeline` - a decision model in a control loop, trained by
//! reinforcement.
//!
//! ```no_run
//! # use brain::{ControlPipeline, ControlSpec, Env};
//! # fn demo<E: Env + 'static>(env: E) -> brain::Result<()> {
//! ControlPipeline::from_pretrained("/path/to/all-MiniLM-L6-v2", env)
//!     .train(ControlSpec::default())
//!     .evaluate()
//!     .save("out/policy.safetensors")
//!     .play(3)
//!     .report()
//!     .finish()?;
//! # Ok(()) }
//! ```
//!
//! ## What makes this different from the other pipelines here
//!
//! [`crate::DecisionPipeline`] and [`crate::ConversionPipeline`] both learn
//! from a fixed dataset: the examples exist before training starts and nothing
//! the model does changes them. This one generates its own data by acting, and
//! **the environment responds** - an action decides which state the next
//! decision is made from. That is what makes the reinforcement learning here
//! genuine rather than a scoring rule wearing PPO's clothes, and it is why the
//! trust region is load-bearing: the policy is shifting its own data
//! distribution while it learns.
//!
//! ## Why a decision model and not a fixed-head policy network
//!
//! The usual policy network ends in a layer whose WIDTH is the action space,
//! so the actions have to be known when the weights are created and be the
//! same at every step. Here the actions arrive with the observation, as text:
//! [`Env::actions`] is free to return a different set every tick, and their
//! MEANING is read rather than looked up by index. An agent can be offered
//! "shoot the imp" on one step and "grab the medkit, reload, retreat" on the
//! next, and a door it has never seen becomes available simply by appearing in
//! the list.
//!
//! That is the whole architectural claim of this crate's decision surface,
//! exercised where it bites hardest.
//!
//! Swedish Embedded AB builds realtime control policies that run on the
//! customer's own hardware - deciding among actions a system defines at run
//! time, in milliseconds. If your team needs judgment inside a control loop,
//! you can procure our services by sending an email to info@swedishembedded.com.

use decide::decide::{Decide, Limits};
use decide::policy::{self, Act, PolicyConfig};
use decide::value::{gae, Critic};
use decide::primitives::{Opt, Question};

use crate::flow::{EvalReport, Flow, Stages, TrainReport};
use crate::{Device, Error, Result};

/// Something a policy can act in.
///
/// The classic reset/step interface with one deliberate difference:
/// [`Env::actions`] is consulted at **every** step and may return a different
/// set each time. Nothing here is indexed by a global action id, so an
/// environment may invent an action mid-episode.
pub trait Env {
    /// Start a new episode and return the first observation.
    ///
    /// `seed` makes an episode reproducible, which is what lets an evaluation
    /// run be compared against another.
    fn reset(&mut self, seed: u64) -> String;

    /// What may be done RIGHT NOW, as text the model reads.
    ///
    /// Must be non-empty while the episode is live. The text is the only thing
    /// the model gets - "shoot the imp" and "option 3" are the difference
    /// between a policy that can generalize to a new action and one that
    /// cannot.
    ///
    /// Takes `&mut self` because building the list is usually also where an
    /// environment records what each string MEANS, so that [`Env::step`] can
    /// map an index back to a concrete move. Handing out `&self` here would
    /// push every implementation into interior mutability for no gain.
    fn actions(&mut self) -> Vec<String>;

    /// Take `action` (an index into the most recent [`Env::actions`]) and
    /// return the next observation, the reward, and whether the episode ended.
    fn step(&mut self, action: usize) -> (String, f32, bool);

    /// What the model is being asked to do, prepended to every option. Fixed
    /// for the life of an environment.
    fn objective(&self) -> String {
        "choose the best action".to_string()
    }

    /// A human-readable line for [`Flow::play`], if the environment can draw
    /// itself. Purely cosmetic.
    fn render(&self) -> Option<String> {
        None
    }

    /// Whether the episode that just ended counts as a win. Only used for
    /// reporting.
    fn won(&self) -> bool {
        false
    }

    /// How far the episode that just ended GOT, when the environment can say.
    ///
    /// Return is a poor way to rank two runs that both fell short of the
    /// goal, because in a task whose reward is mostly one payment at the end,
    /// neither of them collected it: an episode that crossed nine tenths of a
    /// level and died scores about what one that pressed against a wall
    /// scores. A training run choosing which iteration to keep is then
    /// choosing between numbers that are largely noise.
    ///
    /// An environment that can measure partial progress says so here and it
    /// is used to pick the iteration to keep. `None` - the default, and what
    /// every environment that has not thought about it returns - falls back
    /// to mean return, which is the behaviour this had before.
    fn progress(&self) -> Option<f32> {
        None
    }

    /// A scripted action for the current situation, for the warm-start phase -
    /// an index into the most recent [`Env::actions`].
    ///
    /// `None` (the default) means no warm start: training begins from a random
    /// policy.
    ///
    /// **Supply one if you possibly can.** A policy gradient from a random
    /// start has to discover a good action by sampling it, and over a text
    /// action space that is slow enough to look like a plateau: measured here,
    /// 720 episodes of PPO from scratch moved the win rate 54% -> 56.5%, which
    /// is inside its own noise. The same budget after cloning a one-line
    /// scripted teacher starts where the teacher is and spends its samples
    /// improving on it. This is step one of the method's own recipe -
    /// "initial supervised learning phase to establish reasonable policy
    /// initialization" - and skipping it was a mistake.
    ///
    /// The teacher does not have to be good. It has to be better than random,
    /// so the policy gradient starts somewhere worth improving from.
    fn demo(&mut self) -> Option<usize> {
        None
    }
}

/// The seeds [`Stages::run_eval`] scores on.
///
/// Fixed, disjoint from the rollout seeds (which count up from 0), and PUBLIC
/// so a caller can measure its own reference policies on exactly the same
/// episodes. A learned win rate compared against a baseline measured on
/// different seeds is not a comparison.
///
/// 200 episodes, because 60 was not enough to compare against a baseline: a
/// win rate over 60 episodes carries a standard error of about 6 points, so a
/// policy at 57% and a heuristic at 65% were less than one and a half errors
/// apart and could not be told apart at all. 200 halves that.
pub const EVAL_SEEDS: std::ops::Range<u64> = 1_000_000..1_000_200;

/// One demonstration: the state, the options that were offered, and which of
/// them the teacher took.
type Demo = (String, Vec<String>, usize);
/// One teacher episode: what it scored, and what it did - kept together so the
/// bad ones can be dropped whole rather than a step at a time.
type TeacherRun = (f32, Vec<Demo>);

/// One recorded step of one episode.
struct Step {
    observation: String,
    options: Vec<String>,
    action: usize,
    old_prob: f32,
    advantage: f32,
    /// The encoder's pooled embedding of this observation - the critic's input.
    feature: Vec<f32>,
    /// The encoder's output for this step, when the encoder is frozen. What
    /// makes learning from the step cost the head alone rather than another
    /// six-layer forward pass - see `decide::decide::Features`.
    kept: Option<decide::decide::Features>,
    /// What the critic should have predicted here.
    value_target: f32,
    /// What the policy the warm start produced would have done here. Filled
    /// in once per rollout, after collection. See `PolicyConfig::anchor`.
    reference: Option<Vec<f32>>,
}

/// The running mean of `n` head snapshots, folding in one more.
///
/// Incremental so that only one extra copy of the head is ever held, however
/// many iterates go into it.
fn mean_of(
    so_far: Option<Vec<(String, Vec<f32>)>>,
    next: Vec<(String, Vec<f32>)>,
    n: usize,
) -> Vec<(String, Vec<f32>)> {
    let Some(mut acc) = so_far else {
        return next;
    };
    let k = n as f32 + 1.0;
    for (a, b) in acc.iter_mut().zip(&next) {
        if a.0 != b.0 || a.1.len() != b.1.len() {
            // A head that changed shape mid-run is not a thing that happens,
            // and averaging across one silently would be worse than keeping
            // what we have.
            return next;
        }
        for (x, y) in a.1.iter_mut().zip(&b.1) {
            *x += (*y - *x) / k;
        }
    }
    acc
}

/// How well the policy reproduces the teacher on states the teacher reaches.
#[derive(Clone, Copy, Debug, Default)]
pub struct Agreement {
    /// Decisions compared.
    pub states: usize,
    /// Fraction where the policy's own best action WAS the teacher's.
    pub top1: f32,
    /// Mean probability the policy put on the teacher's action - the same
    /// question without the argmax, so that "nearly right everywhere" and
    /// "right half the time and lost the rest" stop reading the same.
    pub top1_prob: f32,
    /// Mean number of options offered, so `top1` can be read against the
    /// chance level it has to beat.
    pub options: f32,
    /// The largest share any single option POSITION took of the teacher's
    /// choices - what a policy that always answers "the third one" would
    /// score.
    pub majority: f32,
}

impl Agreement {
    /// What top-1 agreement a policy that read nothing would get.
    ///
    /// `1/options` is the floor, and it is the wrong number to compare
    /// against on its own: the teacher does not choose uniformly, and the
    /// options do not arrive in a random order, so a head that reads nothing
    /// useful still beats `1/options` by simply preferring wherever the
    /// teacher's answer usually sits. [`Agreement::majority`] is the baseline
    /// that has to be beaten for a number to mean anything.
    pub fn chance(&self) -> f32 {
        if self.options > 0.0 {
            1.0 / self.options
        } else {
            0.0
        }
    }

    /// The baseline a number has to beat to say anything: the better of
    /// guessing uniformly and always naming the same position.
    pub fn floor(&self) -> f32 {
        self.chance().max(self.majority)
    }
}

impl std::fmt::Display for Agreement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.1}% of {} decisions ({:.2} on the teacher's action; guessing {:.1}%, \
             always-the-same-position {:.1}%)",
            self.top1 * 100.0,
            self.states,
            self.top1_prob,
            self.chance() * 100.0,
            self.majority * 100.0
        )
    }
}

/// What one rollout produced.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rollout {
    pub episodes: usize,
    pub steps: usize,
    pub mean_return: f32,
    pub wins: usize,
    /// Mean of [`Env::progress`] over the episodes, when the environment
    /// measures it.
    pub mean_progress: Option<f32>,
}

pub struct ControlPipeline<E: Env> {
    model: Decide,
    env: E,
    rng: data::rng::Rng,
    /// Advances across every rollout so a run never replays one episode.
    episode_seed: u64,
    /// See [`ControlSpec::gae_lambda`]. Held here because the estimator runs
    /// inside an episode, which does not see the spec.
    gae_lambda: f32,
    /// The head as the warm start left it - the policy the anchor holds to.
    /// `None` when there was no warm start, or no anchor asked for.
    reference: Option<Vec<(String, Vec<f32>)>>,
    last: Rollout,
    critic: Critic,
    /// Mean squared error of the last critic fit - whether the baseline is
    /// worth trusting.
    critic_mse: f32,
    /// Episode horizon for [`Stages::run_eval`] and [`Flow::play`].
    ///
    /// Set from [`ControlSpec::max_steps`] when a run trains, and settable on
    /// the builder for a pipeline that only loads weights. It used to be the
    /// constant 40 in both places, which silently truncated any environment
    /// whose episodes are longer than a toy's: an agent that needs 200
    /// decisions to reach a goal was scored as never reaching it, and the
    /// number looked like a policy failure rather than a harness one.
    max_steps: usize,
}

impl<E: Env> std::fmt::Debug for ControlPipeline<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ControlPipeline").finish_non_exhaustive()
    }
}

/// The encoder arrives pretrained and the head does not.
const ENCODER_LR: f32 = 1e-5;
const HEAD_LR: f32 = 3e-4;
/// Reward discount. Episodes here are short, so this barely discounts - it is
/// present so a long episode does not weight its first move like its last.
const GAMMA: f32 = 0.99;
/// The bias/variance dial on the advantage estimator: 0 is the one-step TD
/// error and 1 is the full return minus the baseline. The usual middle.
const GAE_LAMBDA: f32 = 0.95;
/// The critic's hidden width, over the encoder's 384-d pooled embedding.
const CRITIC_HIDDEN: usize = 64;
/// Transitions per optimizer step. Reference PPO splits a rollout into a
/// handful of minibatches; with a few hundred transitions per iteration this
/// is that handful.
const MINIBATCH: usize = 64;

impl<E: Env> ControlPipeline<E> {
    /// Load an encoder checkpoint and start a stage chain on it, acting in
    /// `env`.
    pub fn from_pretrained(dir: impl AsRef<str>, env: E) -> Flow<ControlPipeline<E>> {
        Flow::new(ControlPipeline::builder(dir, env).load())
    }

    pub fn builder(dir: impl AsRef<str>, env: E) -> ControlPipelineBuilder<E> {
        ControlPipelineBuilder {
            dir: dir.as_ref().to_string(),
            env,
            head: None,
            device: Device::default(),
            // An observation is short and the action list is small, so this is
            // sized for a control loop rather than a document.
            limits: Limits { cap_rows: 1024, cap_slots: 32, max_span: 256, overlap: 32 },
            seed: 0,
            max_steps: ControlSpec::default().max_steps,
        }
    }

    fn question(&self, options: &[String]) -> Question {
        Question::Choice {
            instructions: self.env.objective(),
            options: options.iter().map(Opt::new).collect(),
        }
    }

    /// The policy's distribution over the options offered right now.
    pub fn policy(&mut self, observation: &str, options: &[String]) -> Result<Vec<f32>> {
        Ok(self.policy_and_feature(observation, options)?.0)
    }

    /// The distribution AND the state feature the critic reads, from one
    /// forward pass. Kept together because computing them apart would encode
    /// the observation twice per decision.
    fn policy_and_feature(&mut self, observation: &str, options: &[String]) -> Result<(Vec<f32>, Vec<f32>)> {
        let q = self.question(options);
        let scores = self.model.score(observation, std::slice::from_ref(&q)).map_err(Error::Backend)?;
        Ok((decide::loss::softmax(&scores[0]), self.model.state_embedding()))
    }

    /// As [`Self::policy_and_feature`], and keep the encoder's output so that
    /// learning from this step does not have to compute it again.
    ///
    /// Only worth doing with a frozen encoder, where the output cannot have
    /// changed by the time it is read - otherwise the features would be one
    /// update stale and the ratio PPO clips would be wrong.
    fn policy_keeping(
        &mut self,
        observation: &str,
        options: &[String],
    ) -> Result<(Vec<f32>, Vec<f32>, Option<decide::decide::Features>)> {
        if !self.model.encoder_is_frozen() {
            let (p, f) = self.policy_and_feature(observation, options)?;
            return Ok((p, f, None));
        }
        let q = self.question(options);
        let (scores, kept) =
            self.model.score_keeping(observation, &q).map_err(Error::Backend)?;
        Ok((decide::loss::softmax(&scores), self.model.state_embedding(), Some(kept)))
    }

    /// The highest-probability action - what a deployed agent would take.
    pub fn best(&mut self, observation: &str, options: &[String]) -> Result<(usize, f32)> {
        let p = self.policy(observation, options)?;
        let mut best = 0;
        for (i, &pi) in p.iter().enumerate() {
            if pi > p[best] {
                best = i;
            }
        }
        Ok((best, p[best]))
    }

    /// Run one episode, optionally printing it.
    ///
    /// `greedy` takes the best action rather than sampling: evaluation wants
    /// the policy's actual decision, training wants exploration.
    fn episode(&mut self, seed: u64, greedy: bool, max_steps: usize, trace: bool) -> Result<(Vec<Step>, f32, bool)> {
        let mut obs = self.env.reset(seed);
        let mut steps: Vec<Step> = Vec::new();
        let mut rewards: Vec<f32> = Vec::new();
        let mut total = 0.0f32;
        let mut ended = false;
        for t in 0..max_steps {
            let options = self.env.actions();
            if options.is_empty() {
                ended = true;
                break;
            }
            let (probs, feature, kept) = self.policy_keeping(&obs, &options)?;
            let (action, prob) = if greedy {
                let mut b = 0;
                for (i, &pi) in probs.iter().enumerate() {
                    if pi > probs[b] {
                        b = i;
                    }
                }
                (b, probs[b])
            } else {
                let mut u = self.rng.next_f32();
                let mut chosen = probs.len() - 1;
                for (i, &pi) in probs.iter().enumerate() {
                    if u < pi {
                        chosen = i;
                        break;
                    }
                    u -= pi;
                }
                (chosen, probs[chosen])
            };
            if trace {
                if let Some(line) = self.env.render() {
                    println!("    {:>2} {line}", t + 1);
                }
                println!("       -> {} ({:.0}%)", options[action], prob * 100.0);
            }
            let (next, reward, done) = self.env.step(action);
            steps.push(Step {
                observation: obs,
                options,
                action,
                old_prob: prob,
                advantage: 0.0,
                feature,
                kept,
                value_target: 0.0,
                reference: None,
            });
            rewards.push(reward);
            total += reward;
            obs = next;
            if done {
                ended = true;
                break;
            }
        }

        // --- credit assignment, per episode ---
        //
        // An episode cut off by `max_steps` has NOT ended; its remaining
        // return has to be estimated rather than assumed to be zero, or the
        // critic is taught that surviving to the step limit was worth nothing.
        let truncated_value = if ended {
            None
        } else {
            let options = self.env.actions();
            if options.is_empty() {
                None
            } else {
                let (_, feature) = self.policy_and_feature(&obs, &options)?;
                Some(self.critic.predict(&feature))
            }
        };
        let values: Vec<f32> = steps.iter().map(|s| self.critic.predict(&s.feature)).collect();
        let adv = gae(&rewards, &values, GAMMA, self.gae_lambda, truncated_value);
        for ((s, a), v) in steps.iter_mut().zip(&adv).zip(&values) {
            s.advantage = *a;
            // The critic's regression target is the advantage plus what it
            // already predicted - i.e. the estimated return, which is what a
            // value function is supposed to output.
            s.value_target = *a + *v;
        }
        Ok((steps, total, self.env.won()))
    }

    /// Behaviour cloning: play `episodes` under the scripted teacher and fit
    /// the policy to what it did.
    ///
    /// The episodes are driven BY the teacher, so the states visited are the
    /// ones the teacher reaches - which is the point. Cloning a teacher on
    /// states the learner would visit instead is a different and much harder
    /// problem; this is the cheap half, and PPO handles the rest.
    /// Behaviour cloning: play `episodes` under the scripted teacher and fit
    /// the policy to the best of what it did.
    ///
    /// The episodes are driven BY the teacher, so the states visited are the
    /// ones the teacher reaches - which is the point. Cloning a teacher on
    /// states the learner would visit instead is a different and much harder
    /// problem; this is the cheap half, and PPO handles the rest.
    fn clone_teacher(
        &mut self,
        episodes: usize,
        epochs: usize,
        max_steps: usize,
        keep: f32,
        log: &mut dyn FnMut(usize, f32),
        step: &mut usize,
    ) -> Result<f32> {
        let ce = decide::loss::LossConfig::cross_entropy();
        // Grouped BY EPISODE, with what that episode scored, so the bad ones
        // can be dropped before any of them is learned from.
        let mut runs: Vec<TeacherRun> = Vec::new();
        for _ in 0..episodes {
            self.episode_seed += 1;
            let mut obs = self.env.reset(self.episode_seed);
            let mut demos = Vec::new();
            let mut ret = 0.0f32;
            for _ in 0..max_steps {
                let options = self.env.actions();
                if options.is_empty() {
                    break;
                }
                let Some(teacher) = self.env.demo() else {
                    return Ok(0.0);
                };
                demos.push((obs.clone(), options, teacher));
                let (next, reward, done) = self.env.step(teacher);
                ret += reward;
                obs = next;
                if done {
                    break;
                }
            }
            runs.push((ret, demos));
        }

        // FILTERED behaviour cloning: keep the best `keep` of the teacher's
        // episodes and throw the rest away.
        //
        // A scripted teacher is not uniformly good - it is good in the
        // situations it was written for and arbitrary everywhere else, and a
        // heuristic navigator's bad episodes are bad in a specific, learnable
        // way: it walks into a wall and keeps walking into it. Cloning those
        // teaches the policy exactly that, and the policy gradient then has to
        // spend its samples unlearning something it was deliberately taught.
        // Keeping the top fraction by return is the cheapest form of the
        // filtering the imitation-learning literature calls Filtered BC, and
        // it needs nothing the run does not already have.
        let keep = keep.clamp(0.0, 1.0);
        if keep < 1.0 && runs.len() > 1 {
            runs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let n = ((runs.len() as f32 * keep).round() as usize).clamp(1, runs.len());
            let dropped = runs.len() - n;
            let worst = runs.last().map(|r| r.0).unwrap_or(0.0);
            runs.truncate(n);
            if dropped > 0 {
                println!(
                    "    warm start: kept {n} of {} scripted episodes (best {:+.2}, dropped down to {worst:+.2})",
                    n + dropped,
                    runs.first().map(|r| r.0).unwrap_or(0.0)
                );
            }
        }
        let demos: Vec<Demo> = runs.into_iter().flat_map(|(_, d)| d).collect();
        if demos.is_empty() {
            return Ok(0.0);
        }
        let mut loss = 0.0f32;
        let mut order: Vec<usize> = (0..demos.len()).collect();
        for _ in 0..epochs.max(1) {
            for i in (1..order.len()).rev() {
                let j = (self.rng.next_u64() % (i as u64 + 1)) as usize;
                order.swap(i, j);
            }
            let mut epoch_loss = 0.0f32;
            for chunk in order.chunks(MINIBATCH) {
                self.model.zero_grads();
                for &d in chunk {
                    let (obs, options, teacher) = &demos[d];
                    let q = self.question(options);
                    epoch_loss += self
                        .model
                        .accumulate(obs, &q, |sc| decide::loss::decision_loss(sc, *teacher, &ce))
                        .map_err(Error::Backend)?;
                    *step += 1;
                }
                // A larger rate than the policy phase uses: this is ordinary
                // supervised learning on a fixed set, not a noisy policy
                // gradient, and it has to reach the teacher rather than drift
                // toward it.
                self.model.adamw_scaled(ENCODER_LR, HEAD_LR * 3.0, 1.0 / chunk.len() as f32);
            }
            loss = epoch_loss / demos.len() as f32;
            log(*step, loss);
        }
        Ok(loss)
    }

    /// Collect a batch of episodes under the current policy.
    fn rollout(&mut self, episodes: usize, max_steps: usize) -> Result<(Vec<Step>, Rollout)> {
        let mut batch = Vec::new();
        let (mut total, mut wins) = (0.0f32, 0usize);
        let (mut got, mut measured) = (0.0f32, 0usize);
        for _ in 0..episodes {
            self.episode_seed += 1;
            let seed = self.episode_seed;
            let (steps, ret, won) = self.episode(seed, false, max_steps, false)?;
            total += ret;
            wins += usize::from(won);
            if let Some(p) = self.env.progress() {
                got += p;
                measured += 1;
            }
            batch.extend(steps);
        }
        // Advantages are already centred by the critic; this rescales them so
        // one learning rate works across reward scales, which is what every
        // PPO implementation does before the update.
        let mut adv: Vec<f32> = batch.iter().map(|s| s.advantage).collect();
        policy::normalize(&mut adv);
        for (s, a) in batch.iter_mut().zip(&adv) {
            s.advantage = *a;
        }
        // Fit the critic on THIS batch, against the returns it was used to
        // estimate. On the first iteration it predicts zero everywhere (its
        // output layer starts at zero), so the advantages degrade exactly to
        // return-minus-batch-mean and nothing is worse than it was before the
        // critic existed.
        self.note_reference(&mut batch)?;
        let features: Vec<Vec<f32>> = batch.iter().map(|s| s.feature.clone()).collect();
        let targets: Vec<f32> = batch.iter().map(|s| s.value_target).collect();
        self.critic_mse = self.critic.fit(&features, &targets, 60, 0.02, 1e-5);
        let stats = Rollout {
            episodes,
            steps: batch.len(),
            mean_return: total / episodes.max(1) as f32,
            wins,
            mean_progress: (measured > 0).then(|| got / measured as f32),
        };
        Ok((batch, stats))
    }

    /// What the reference policy would have done at every step of a batch.
    ///
    /// Two weight swaps and one forward pass over the batch, once per
    /// iteration - not per epoch, because the reference does not move. The
    /// head is small and the encoder's output was kept when the step was
    /// taken, so this is cheap next to the rollout that produced the batch.
    fn note_reference(&mut self, batch: &mut [Step]) -> Result<()> {
        let Some(reference) = self.reference.clone() else {
            return Ok(());
        };
        let live = self.model.head_weights();
        self.model.set_head_weights(&reference);
        let mut failed = None;
        for s in batch.iter_mut() {
            let Some(f) = &s.kept else { continue };
            match self.model.score_kept(f) {
                Ok(scores) => s.reference = Some(decide::loss::softmax(&scores)),
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
        }
        // The live weights go back whatever happened: leaving the reference
        // installed would silently undo the whole run.
        self.model.set_head_weights(&live);
        match failed {
            Some(e) => Err(Error::Backend(e)),
            None => Ok(()),
        }
    }

    /// One PPO pass over a collected batch.
    fn update(
        &mut self,
        batch: &[Step],
        cfg: &PolicyConfig,
        order: &mut [usize],
        head_lr: f32,
        lr_scale: f32,
    ) -> Result<(f32, f32, usize, usize)> {
        // Shuffled, because consecutive steps of one episode are correlated
        // and a sequential pass would walk the policy along a trajectory
        // instead of averaging over the batch.
        for i in (1..order.len()).rev() {
            let j = (self.rng.next_u64() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        let mut loss = 0.0f32;
        let mut seen = 0usize;
        // How far this pass moved the policy away from the one that collected
        // the batch. See `PolicyConfig::target_kl`.
        let mut drift = 0.0f64;
        // MINIBATCHES, not single transitions. One optimizer step per
        // transition is the thing this used to do and it is not policy
        // gradient in any recognizable sense: a single step's advantage is an
        // extremely noisy estimate of the gradient, and Adam applied straight
        // to it chases the noise rather than the signal. Reference PPO
        // implementations split a rollout into a handful of minibatches
        // (4 for Atari, 32 for continuous control) and step once per
        // minibatch.
        let mut stopped = false;
        let mut steps_taken = 0usize;
        for chunk in order.chunks(MINIBATCH) {
            if stopped {
                break;
            }
            // Per-MINIBATCH advantage normalization, which is where reference
            // implementations do it - not over the whole rollout.
            let mut adv: Vec<f32> = chunk.iter().map(|&i| batch[i].advantage).collect();
            policy::normalize(&mut adv);
            let mut here = 0.0f64;
            self.model.zero_grads();
            for (slot, &i) in chunk.iter().enumerate() {
                let s = &batch[i];
                let act = Act { old_prob: s.old_prob, action: s.action, advantage: adv[slot] };
                // The encoder's output for this step was kept when the step
                // was taken, and a frozen encoder cannot have changed it - so
                // this is the head alone. See `decide::decide::Features` for what
                // that is worth.
                let reference = s.reference.as_deref();
                let mut moved = 0.0f32;
                loss += match &s.kept {
                    Some(f) => self
                        .model
                        .accumulate_kept(f, |sc| {
                            moved = policy::drift(sc, &act);
                            policy::choice_loss_anchored(sc, &act, reference, cfg)
                        })
                        .map_err(Error::Backend)?,
                    None => {
                        let q = self.question(&s.options);
                        self.model
                            .accumulate(&s.observation, &q, |sc| {
                                moved = policy::drift(sc, &act);
                                policy::choice_loss_anchored(sc, &act, reference, cfg)
                            })
                            .map_err(Error::Backend)?
                    }
                };
                drift += moved as f64;
                here += moved as f64;
                seen += 1;
            }
            // BEFORE this minibatch's step, not after the whole pass. A guard
            // that can only fire once an epoch of thirty-odd steps is done is
            // a report and not a guard: measured on this repository's DOOM
            // sample with it reporting only, one pass moved the policy 0.0403
            // on the first iteration and 0.1718 on the second, against a
            // threshold of 0.02. This is where reference implementations put
            // it, and it bounds the overshoot by one minibatch.
            if cfg.target_kl > 0.0 && here / chunk.len() as f64 > cfg.target_kl as f64 {
                stopped = true;
                continue;
            }
            // The accumulated sum becomes a mean, so one learning rate means
            // the same thing whatever the minibatch happened to hold.
            self.model.adamw_scaled(
                ENCODER_LR * lr_scale,
                head_lr * lr_scale,
                1.0 / chunk.len() as f32,
            );
            steps_taken += 1;
        }
        let n = seen.max(1) as f32;
        let offered = order.len().div_ceil(MINIBATCH);
        Ok((loss / n, (drift / seen.max(1) as f64) as f32, steps_taken, offered))
    }

    /// The environment this pipeline acts in.
    ///
    /// A caller that drives its own loop with [`ControlPipeline::policy`] -
    /// to show the distribution, to score against a scripted baseline on the
    /// same episodes, to record a transcript - needs to reach the environment
    /// it is stepping. Without this the only way to act is [`Flow::play`],
    /// which prints and discards.
    pub fn env(&self) -> &E {
        &self.env
    }

    pub fn env_mut(&mut self) -> &mut E {
        &mut self.env
    }

    /// Score the policy on a FIXED block of episodes, the same block every
    /// time, and say how far they got.
    ///
    /// The rollout cannot answer this. Its episodes come from a seed that
    /// advances, so every iteration is scored on a different set of worlds -
    /// and where the environment generates its world from the seed, the seed
    /// IS the world. Measured on this repository's DOOM scenarios: the
    /// scripted player, which cannot learn or degrade, scores between 0.59
    /// and 0.73 across five blocks of sixteen worlds. That is a standard
    /// deviation of 0.048 at sixteen episodes, or 0.068 at eight - against
    /// which a policy's actual iteration-to-iteration movement of 0.079 is
    /// almost entirely the draw.
    ///
    /// So the alternatives are compared under identical conditions instead:
    /// same worlds, same action-sampler stream. This is the oldest trick in
    /// simulation optimisation - common random numbers - and it does not
    /// reduce the variance of either score, it removes the variance from
    /// their DIFFERENCE, which is the only quantity anybody wanted.
    fn gauge(&mut self, episodes: usize, max_steps: usize) -> Result<Option<f32>> {
        if episodes == 0 {
            return Ok(None);
        }
        // Set aside the stream the rollout is using, so that gauging does not
        // change which episodes training goes on to see.
        let saved_rng = self.rng.clone();
        let saved_seed = self.episode_seed;
        self.rng = data::rng::Rng::new(0x6a11_6e00);
        let (mut got, mut measured) = (0.0f32, 0usize);
        for i in 0..episodes {
            // A block no rollout will ever draw, so the policy is judged on
            // worlds it was not just trained on.
            self.episode_seed = 0x4000_0000 + i as u64;
            self.episode(self.episode_seed, false, max_steps, false)?;
            if let Some(p) = self.env.progress() {
                got += p;
                measured += 1;
            }
        }
        self.rng = saved_rng;
        self.episode_seed = saved_seed;
        Ok((measured > 0).then(|| got / measured as f32))
    }

    /// How often the policy's own choice IS the teacher's, on states the
    /// teacher reaches.
    ///
    /// A supervised question asked of a reinforcement-learning setup, and the
    /// cheapest one there is: no reward, no critic, no advantage estimate.
    /// The episode is driven BY the teacher and the policy is asked, at each
    /// of its states, what it would have done instead.
    ///
    /// Asked on the episodes the head was fitted to and again on episodes it
    /// has never seen, the pair says which of three things is wrong when a
    /// trained policy still does not play well:
    ///
    /// * near chance on both - the observation and the head cannot express
    ///   the decision at all. No policy-gradient budget finds what is not
    ///   there, and the thing to fix is what the agent is allowed to read.
    /// * high on the fitted episodes, near chance on the unseen ones - it
    ///   memorised them, and the fix is more distinct worlds rather than more
    ///   steps in the ones it has.
    /// * high on both - the representation is sufficient and the failure is
    ///   in ACTING: compounding error once the policy leaves the states the
    ///   teacher visits, or credit assignment. That is where labelling the
    ///   states the policy itself reaches is worth the samples and a larger
    ///   PPO budget is not.
    ///
    /// Returns `None` when the environment has no teacher to ask.
    pub fn teacher_agreement(
        &mut self,
        seeds: &[u64],
        max_steps: usize,
    ) -> Result<Option<Agreement>> {
        // The teacher drives, so nothing here consumes the action sampler -
        // but an episode still has to leave the training stream where it
        // found it, or probing would change which worlds training sees.
        let saved = self.episode_seed;
        let (mut hits, mut states, mut on_teacher, mut offered) = (0usize, 0usize, 0.0f64, 0usize);
        // How often the teacher's answer was the first option, the second,
        // and so on - the constant policy this has to beat.
        let mut by_position: Vec<usize> = Vec::new();
        for &seed in seeds {
            let mut obs = self.env.reset(seed);
            for _ in 0..max_steps {
                let options = self.env.actions();
                if options.is_empty() {
                    break;
                }
                let Some(teacher) = self.env.demo() else {
                    self.episode_seed = saved;
                    return Ok(None);
                };
                let probs = self.policy(&obs, &options)?;
                let mut best = 0;
                for (i, &pi) in probs.iter().enumerate() {
                    if pi > probs[best] {
                        best = i;
                    }
                }
                hits += usize::from(best == teacher);
                on_teacher += probs.get(teacher).copied().unwrap_or(0.0) as f64;
                offered += options.len();
                if by_position.len() <= teacher {
                    by_position.resize(teacher + 1, 0);
                }
                by_position[teacher] += 1;
                states += 1;
                let (next, _, done) = self.env.step(teacher);
                obs = next;
                if done {
                    break;
                }
            }
        }
        self.episode_seed = saved;
        if states == 0 {
            return Ok(None);
        }
        let n = states as f32;
        Ok(Some(Agreement {
            states,
            top1: hits as f32 / n,
            top1_prob: (on_teacher / states as f64) as f32,
            options: offered as f32 / n,
            majority: by_position.iter().copied().max().unwrap_or(0) as f32 / n,
        }))
    }

    /// Fit the head to the teacher and measure what it learned, on the
    /// episodes it was fitted to and on episodes it was not.
    ///
    /// The three numbers are reported together because only their SHAPE means
    /// anything - see [`Self::teacher_agreement`]. The first is taken before
    /// any fitting, and it is the control: an untouched head has no reason to
    /// agree with the teacher more often than chance, so a "before" that is
    /// not near chance means the measurement is wrong and the other two
    /// numbers are not worth reading.
    pub fn probe_teacher(
        &mut self,
        episodes: usize,
        epochs: usize,
        max_steps: usize,
        keep: f32,
        holdout: usize,
        freeze_encoder: bool,
    ) -> Result<Option<(Agreement, Agreement, Agreement)>> {
        // The SAME encoder setting training uses, or this answers a question
        // about a different model. Frozen, the fit has 445k parameters to do
        // it with and the observation has already been compressed to 384
        // numbers by weights that never saw DOOM; unfrozen it has 22M and can
        // move the compression itself. Which of those two can reproduce the
        // teacher is exactly the difference between "the head is too small"
        // and "the representation threw the answer away", and running the
        // probe both ways is how they are told apart.
        self.model.set_encoder_frozen(freeze_encoder);
        // The block no rollout and no gauge will ever draw.
        let unseen: Vec<u64> = (0..holdout as u64).map(|i| 0x5000_0000 + i).collect();
        // The seeds `clone_teacher` is about to draw, which is what makes the
        // "fitted on" number a measurement of fitting rather than of luck.
        let fitted: Vec<u64> = (1..=episodes as u64).map(|i| self.episode_seed + i).collect();

        let before = match self.teacher_agreement(&unseen, max_steps)? {
            Some(a) => a,
            None => return Ok(None),
        };
        let mut step = 0usize;
        let mut quiet = |_: usize, _: f32| {};
        self.clone_teacher(episodes, epochs, max_steps, keep, &mut quiet, &mut step)?;
        let on_fitted = match self.teacher_agreement(&fitted, max_steps)? {
            Some(a) => a,
            None => return Ok(None),
        };
        let on_unseen = match self.teacher_agreement(&unseen, max_steps)? {
            Some(a) => a,
            None => return Ok(None),
        };
        Ok(Some((before, on_fitted, on_unseen)))
    }

    /// The episode horizon evaluation and play use.
    pub fn max_steps(&self) -> usize {
        self.max_steps
    }

    pub fn save_head(&self, path: impl AsRef<str>) -> Result<()> {
        self.model.save_head(path.as_ref()).map_err(Error::Backend)
    }

    /// Name this head and say what it was trained for, so the checkpoint it
    /// writes stands on its own. See `decide::decide::Provenance`.
    ///
    /// The base encoder is already known - it is the directory the pipeline
    /// loaded - so a caller supplies only the half it knows: which head this
    /// is, and what it was fitted to do.
    pub fn describe(&mut self, id: impl Into<String>, task: serde_json::Value) {
        let base = self.model.provenance().base.clone();
        self.model.set_provenance(decide::decide::Provenance {
            base,
            id: id.into(),
            task,
        });
    }

    /// Play `n` episodes greedily, printing each step.
    pub fn show(&mut self, n: usize, max_steps: usize) -> Result<Rollout> {
        let (mut total, mut wins) = (0.0f32, 0usize);
        for i in 0..n {
            self.episode_seed += 1;
            let seed = self.episode_seed;
            println!("\n  episode {} (seed {seed})", i + 1);
            let (_, ret, won) = self.episode(seed, true, max_steps, true)?;
            println!("       = return {ret:+.2}, {}", if won { "WON" } else { "lost" });
            total += ret;
            wins += usize::from(won);
        }
        Ok(Rollout {
            episodes: n,
            steps: 0,
            mean_return: total / n.max(1) as f32,
            wins,
            mean_progress: None,
        })
    }
}

/// What a control run needs.
#[derive(Clone, Copy, Debug)]
pub struct ControlSpec {
    /// How many rollout-then-update cycles.
    pub iterations: usize,
    /// Episodes collected per iteration.
    pub episodes: usize,
    /// How many passes each collected batch is reused for. This is what PPO's
    /// trust region buys: without the clipped ratio, a second pass is already
    /// off-policy and unsafe.
    pub epochs: usize,
    pub max_steps: usize,
    pub policy: PolicyConfig,
    pub seed: u64,
    /// Episodes of scripted play to clone before the policy gradient starts.
    /// Zero, or an environment with no [`Env::demo`], skips the phase.
    pub warmup_episodes: usize,
    /// What fraction of the teacher's episodes to actually clone, best first.
    ///
    /// 1.0 clones everything, which is right for a teacher that is uniformly
    /// mediocre and wrong for one that is good in the situations it was
    /// written for and arbitrary elsewhere - see [`ControlPipeline::
    /// clone_teacher`]. Lower it when the teacher has failure modes worth not
    /// teaching.
    pub warmup_keep: f32,
    /// Episodes on a FIXED block of worlds, scored after every iteration, to
    /// decide which iteration to keep. `0` falls back to the rollout's own
    /// numbers, which are measured on worlds that move. See
    /// `ControlPipeline::gauge`.
    pub gauge_episodes: usize,
    /// Average the head over the last `average` iterates and keep that if it
    /// gauges better than the best single one. `0` is off.
    pub average: usize,
    /// The bias/variance dial on the advantage estimator, overriding
    /// [`GAE_LAMBDA`].
    ///
    /// At 1.0 the advantage is the full return minus a baseline: unbiased,
    /// noisy, and - the point here - it carries a reward paid at the end of
    /// an episode all the way back to its first decision. At the usual 0.95
    /// it does not. With `gamma * lambda` at 0.9405 the weight on a reward a
    /// hundred decisions ahead is 0.002, so on a task whose return is mostly
    /// one payment at the exit, every decision before roughly the last thirty
    /// is learning from the dense terms alone and the value function never
    /// sees the goal.
    pub gae_lambda: f32,
    /// The head's learning rate.
    ///
    /// Worth a flag because it trades against the trust region rather than
    /// standing alone. A step size too large for the region exhausts the
    /// divergence budget in a handful of minibatches, and the rest of the
    /// rollout is never used: measured here at 3e-4, an iteration took five
    /// of sixty-four minibatch steps, so ninety-two per cent of the
    /// transitions that had just been collected were discarded. A smaller
    /// step lets the whole batch contribute for the same total travel.
    pub head_lr: f32,
    /// Passes over the collected demonstrations.
    ///
    /// Needed because the demonstrations are a small fixed dataset and one
    /// pass over it is a handful of optimizer steps: at 600 demonstrations and
    /// a minibatch of 64 it is nine. Cloning has to actually CONVERGE, or the
    /// policy gradient starts from something that is neither the teacher nor
    /// random, and the run reports nothing about either.
    pub warmup_epochs: usize,
    /// Hold the imported encoder fixed and train only the head.
    ///
    /// Default ON for control, and the opposite of what the supervised
    /// pipelines here want. A reinforcement signal is far noisier than a
    /// labelled one - a few hundred high-variance gradients per iteration -
    /// and that is not enough to move 22M pretrained parameters anywhere
    /// useful, only enough to damage the language understanding that made the
    /// option text readable. It is also about three times faster per step,
    /// because the encoder's reverse pass is skipped entirely, which buys back
    /// the sample efficiency reinforcement learning spends.
    pub freeze_encoder: bool,
}

impl Default for ControlSpec {
    fn default() -> ControlSpec {
        ControlSpec {
            iterations: 12,
            episodes: 24,
            epochs: 2,
            max_steps: 40,
            // Entropy higher than a supervised run would want: a control policy
            // that commits early stops seeing the states it has not solved.
            policy: PolicyConfig { clip: 0.2, entropy: 0.02, gamma: 0.99, anchor: 0.0, target_kl: 0.02 },
            seed: 0,
            warmup_episodes: 60,
            warmup_epochs: 12,
            warmup_keep: 1.0,
            gauge_episodes: 0,
            average: 0,
            gae_lambda: GAE_LAMBDA,
            head_lr: HEAD_LR,
            freeze_encoder: true,
        }
    }
}

impl ControlSpec {
    pub fn iterations(mut self, n: usize) -> ControlSpec {
        self.iterations = n;
        self
    }
    pub fn episodes(mut self, n: usize) -> ControlSpec {
        self.episodes = n;
        self
    }
    pub fn epochs(mut self, n: usize) -> ControlSpec {
        self.epochs = n;
        self
    }
    pub fn max_steps(mut self, n: usize) -> ControlSpec {
        self.max_steps = n;
        self
    }
    pub fn seed(mut self, seed: u64) -> ControlSpec {
        self.seed = seed;
        self
    }
    pub fn warmup_episodes(mut self, n: usize) -> ControlSpec {
        self.warmup_episodes = n;
        self
    }
    pub fn warmup_epochs(mut self, n: usize) -> ControlSpec {
        self.warmup_epochs = n;
        self
    }
    /// See [`ControlSpec::gauge_episodes`].
    pub fn gauge_episodes(mut self, n: usize) -> ControlSpec {
        self.gauge_episodes = n;
        self
    }

    /// See [`ControlSpec::average`].
    pub fn average(mut self, n: usize) -> ControlSpec {
        self.average = n;
        self
    }

    /// See [`ControlSpec::gae_lambda`].
    pub fn gae_lambda(mut self, l: f32) -> ControlSpec {
        if l > 0.0 {
            self.gae_lambda = l;
        }
        self
    }

    /// See [`ControlSpec::head_lr`].
    pub fn head_lr(mut self, lr: f32) -> ControlSpec {
        if lr > 0.0 {
            self.head_lr = lr;
        }
        self
    }

    /// See [`ControlSpec::warmup_keep`].
    pub fn warmup_keep(mut self, f: f32) -> ControlSpec {
        self.warmup_keep = f;
        self
    }
    /// Fine-tune the encoder as well as the head. See
    /// [`ControlSpec::freeze_encoder`] for why this is off by default.
    pub fn train_encoder(mut self, yes: bool) -> ControlSpec {
        self.freeze_encoder = !yes;
        self
    }
}

impl<E: Env> Stages for ControlPipeline<E> {
    type TrainSpec = ControlSpec;

    fn describe(&self) -> String {
        format!(
            "control policy, {} training steps so far, last rollout {:.2} mean return over {} episodes",
            self.model.steps_taken(),
            self.last.mean_return,
            self.last.episodes
        )
    }

    fn run_train(&mut self, spec: &ControlSpec, log: &mut dyn FnMut(usize, f32)) -> Result<TrainReport> {
        let mut step = 0usize;
        let mut last_loss = 0.0f32;
        self.model.set_encoder_frozen(spec.freeze_encoder);
        self.max_steps = spec.max_steps;
        println!(
            "  {} iterations x {} episodes, {} PPO passes each, encoder {}",
            spec.iterations,
            spec.episodes,
            spec.epochs,
            if spec.freeze_encoder { "frozen" } else { "fine-tuned" }
        );
        // The estimator runs inside an episode, which never sees the spec.
        self.gae_lambda = spec.gae_lambda;
        if spec.warmup_episodes > 0 {
            let bc = self.clone_teacher(
                spec.warmup_episodes,
                spec.warmup_epochs,
                spec.max_steps,
                spec.warmup_keep,
                log,
                &mut step,
            )?;
            if step > 0 {
                println!(
                    "    warm start: {} scripted episodes x {} passes, final loss {bc:.4}",
                    spec.warmup_episodes, spec.warmup_epochs
                );
            }
            // Whatever the cloning produced is what the anchor holds to. Read
            // once, here, because every later update moves the head away from
            // it and the point is to remember where it started.
            if spec.policy.anchor > 0.0 {
                self.reference = Some(self.model.head_weights());
            }
        }
        // The BEST iteration's weights, not the last one's.
        //
        // A policy gradient's return per iteration is not monotone: it steps
        // past a good solution and comes back. Measured on this repository's
        // DOOM sample, one run went 2 wins of 6, then 3 of 6, then 0 of 6 -
        // so the weights that exist when the loop happens to stop are not the
        // weights the run earned, and scoring them measures where the walk
        // ended rather than what was learned. Snapshotting costs one readback
        // of the head per iteration, which is nothing beside a rollout.
        let mut best_rank = f32::NEG_INFINITY;
        let mut last_rank = f32::NEG_INFINITY;
        let mut best_ranked_on = "";
        let mut best_by = Rollout::default();
        let mut best: Option<(usize, Vec<(String, Vec<f32>)>)> = None;
        // The running mean of the iterates, which is a different candidate
        // from the best of them and often a better one. See `mean_of`.
        let mut running: Option<Vec<(String, Vec<f32>)>> = None;
        let mut averaged = 0usize;
        for it in 0..spec.iterations {
            let (batch, stats) = self.rollout(spec.episodes, spec.max_steps)?;
            if batch.is_empty() {
                return Err(Error::Backend("the environment produced no steps to learn from".into()));
            }
            let mut order: Vec<usize> = (0..batch.len()).collect();
            // Linear decay to zero over the run, as reference PPO does. A
            // constant rate keeps taking full-size steps after the policy is
            // good, which is most of how a well-initialized policy gets walked
            // away from.
            let lr_scale = 1.0 - (it as f32 / spec.iterations.max(1) as f32);
            let (mut drift, mut passes) = (0.0f32, 0usize);
            let (mut took, mut offered) = (0usize, 0usize);
            for _ in 0..spec.epochs {
                let (l, d, t, o) =
                    self.update(&batch, &spec.policy, &mut order, spec.head_lr, lr_scale)?;
                last_loss = l;
                drift = d;
                took += t;
                offered += o;
                passes += 1;
                step += batch.len();
                log(step, last_loss);
                // Clipping is not a trust region on its own: it silences the
                // samples that have moved too far while the rest keep pushing.
                // This is the guard that actually binds.
                if spec.policy.target_kl > 0.0 && d > spec.policy.target_kl {
                    break;
                }
            }
            self.last = stats;
            println!(
                "    iter {:>3}  return {:+.2}  wins {:>3}/{:<3}  steps {:>5}  critic mse {:.3}{}",
                it + 1,
                stats.mean_return,
                stats.wins,
                stats.episodes,
                stats.steps,
                self.critic_mse,
                match stats.mean_progress {
                    Some(p) => format!("  progress {p:.2}"),
                    None => String::new(),
                }
            );
            if passes < spec.epochs || took < offered {
                println!(
                    "           took {took} of {offered} minibatch steps over {passes} of \
                     {} passes; the policy had moved {drift:.4}",
                    spec.epochs
                );
            }
            // Ranked on the FIXED block when there is one, because the
            // rollout's own episodes are drawn from worlds that move and the
            // difference between two of those scores is mostly the draw.
            let gauged = self.gauge(spec.gauge_episodes, spec.max_steps)?;
            if let Some(g) = gauged {
                println!("           on the fixed block: {g:.3}");
            }
            // Keeping the BEST iterate selects partly for luck: the score it
            // is chosen on carries noise, so the winner is the iteration that
            // drew the kindest worlds as much as the one that learned most.
            // The MEAN of the iterates has no such failure mode, and it is
            // what two separate traditions point at for a sequence that
            // random-walks around a good solution rather than converging onto
            // one - fictitious play, whose convergence is in the time average
            // of play and not in the last thing played, and Polyak-Ruppert
            // averaging, which attains the optimal asymptotic variance under
            // far less delicate step-size tuning than any single iterate.
            //
            // Averaging weights is usually dismissed as intractable for deep
            // networks. Here the encoder is frozen and the head is small, so
            // it is a few thousand floats.
            if spec.average > 0 && it + 1 > spec.iterations.saturating_sub(spec.average) {
                running = Some(mean_of(running.take(), self.model.head_weights(), averaged));
                averaged += 1;
            }
            let rank = gauged
                .or(stats.mean_progress)
                .unwrap_or(stats.mean_return);
            let ranked_on = if gauged.is_some() {
                "on the fixed block"
            } else if stats.mean_progress.is_some() {
                "over its own episodes"
            } else {
                "in return"
            };
            if best.is_none() || rank > best_rank {
                best_rank = rank;
                best_ranked_on = ranked_on;
                best = Some((it + 1, self.model.head_weights()));
                best_by = stats;
            }
            last_rank = rank;
        }
        // Both candidates, measured on the same worlds, and the better one
        // kept. Averaging is a claim about the shape of the sequence, not a
        // law, so it is checked rather than assumed.
        if let (Some(mean), Some((_, bw))) = (&running, &best) {
            let live = self.model.head_weights();
            self.model.set_head_weights(mean);
            let m = self.gauge(spec.gauge_episodes, spec.max_steps)?;
            self.model.set_head_weights(bw);
            let b = self.gauge(spec.gauge_episodes, spec.max_steps)?;
            self.model.set_head_weights(&live);
            if let (Some(m), Some(b)) = (m, b) {
                println!(
                    "    the mean of the last {averaged} iterates scores {m:.3} on the \
                     fixed block, the best single one {b:.3}"
                );
                if m > b {
                    self.model.set_head_weights(mean);
                    return Ok(TrainReport { steps: step, final_loss: last_loss, seconds: 0.0 });
                }
            }
        }
        if let Some((it, w)) = best {
            if it != spec.iterations {
                // The SAME number the choice was made on. Reporting the
                // rollout's progress next to a decision taken on the fixed
                // block is two different measurements and one decision, which
                // reads as though the wrong iteration was kept.
                println!(
                    "    keeping iteration {it}, which scored {best_rank:.3} \
                     {best_ranked_on} - the last one scored {last_rank:.3}"
                );
                let _ = &best_by;
                self.model.set_head_weights(&w);
            }
        }
        Ok(TrainReport { steps: step, final_loss: last_loss, seconds: 0.0 })
    }

    fn run_eval(&mut self) -> Result<EvalReport> {
        // Greedy, and on seeds no training rollout used: what the policy would
        // actually do, on situations it has not been updated against.
        let (mut total, mut wins, mut steps) = (0.0f32, 0usize, 0usize);
        let n = EVAL_SEEDS.count();
        for seed in EVAL_SEEDS {
            let (st, ret, won) = self.episode(seed, true, self.max_steps, false)?;
            total += ret;
            wins += usize::from(won);
            steps += st.len();
        }
        Ok(EvalReport {
            accuracy: wins as f32 / n as f32,
            items: n,
            notes: vec![
                ("mean return".into(), total / n as f32),
                ("mean episode length".into(), steps as f32 / n as f32),
            ],
        })
    }

    fn run_save(&self, path: &str) -> Result<()> {
        self.save_head(path)
    }

    fn turn_prompt(&self) -> &str {
        "press enter to play an episode> "
    }

    /// One interactive turn plays one episode and prints it.
    fn run_turn(&mut self, _input: &str) -> Result<String> {
        let r = self.show(1, self.max_steps)?;
        Ok(format!("  return {:+.2}", r.mean_return))
    }
}

impl<E: Env> Flow<ControlPipeline<E>> {
    /// Play `n` episodes greedily, printing every decision - the stage that
    /// shows what the policy learned rather than summarizing it.
    pub fn play(self, n: usize) -> Flow<ControlPipeline<E>> {
        self.stage("play", move |p| {
            let n_steps = p.max_steps;
            let r = p.show(n, n_steps)?;
            Ok(Some(format!("{} episodes, mean return {:+.2}, {} won", r.episodes, r.mean_return, r.wins)))
        })
    }
}

pub struct ControlPipelineBuilder<E: Env> {
    dir: String,
    env: E,
    head: Option<String>,
    device: Device,
    limits: Limits,
    seed: u64,
    max_steps: usize,
}

impl<E: Env> ControlPipelineBuilder<E> {
    /// Trained head weights. Without this the policy is random.
    pub fn head(mut self, path: impl AsRef<str>) -> ControlPipelineBuilder<E> {
        self.head = Some(path.as_ref().to_string());
        self
    }

    pub fn device(mut self, device: Device) -> ControlPipelineBuilder<E> {
        self.device = device;
        self
    }

    pub fn limits(mut self, limits: Limits) -> ControlPipelineBuilder<E> {
        self.limits = limits;
        self
    }

    pub fn seed(mut self, seed: u64) -> ControlPipelineBuilder<E> {
        self.seed = seed;
        self
    }

    /// Episode horizon for evaluation and play on a pipeline that is not going
    /// to be trained. A trained one takes it from its [`ControlSpec`].
    pub fn max_steps(mut self, n: usize) -> ControlPipelineBuilder<E> {
        self.max_steps = n;
        self
    }

    pub fn load(self) -> Result<ControlPipeline<E>> {
        let model =
            crate::decision::load_decide(&self.dir, self.head.as_deref(), &self.device, self.limits, self.seed)?;
        let cfg_width = model.cfg.d_model as usize;
        Ok(ControlPipeline {
            model,
            env: self.env,
            rng: data::rng::Rng::new(self.seed ^ 0xc0ffee),
            episode_seed: 0,
            reference: None,
            gae_lambda: GAE_LAMBDA,
            last: Rollout::default(),
            critic: Critic::new(cfg_width, CRITIC_HIDDEN, self.seed ^ 0x1c1),
            critic_mse: 0.0,
            max_steps: self.max_steps,
        })
    }
}

#[cfg(test)]
mod averaging_tests {
    use super::mean_of;

    fn head(v: &[f32]) -> Vec<(String, Vec<f32>)> {
        vec![("w".to_string(), v.to_vec())]
    }

    /// The mean has to be the mean, folded in one iterate at a time and
    /// holding only one extra copy of the head however many go into it.
    #[test]
    fn the_running_mean_is_the_mean() {
        let mut acc = None;
        for (n, v) in [[0.0f32, 4.0], [2.0, 8.0], [4.0, 0.0]].iter().enumerate() {
            acc = Some(mean_of(acc.take(), head(v), n));
        }
        let got = &acc.unwrap()[0].1;
        assert!((got[0] - 2.0).abs() < 1e-6, "{got:?}");
        assert!((got[1] - 4.0).abs() < 1e-6, "{got:?}");
    }

    /// One iterate averages to itself, which is what makes the first fold a
    /// special case worth having a test for rather than a branch to trust.
    #[test]
    fn one_iterate_averages_to_itself() {
        let got = mean_of(None, head(&[1.0, -2.0]), 0);
        assert_eq!(got[0].1, vec![1.0, -2.0]);
    }
}
