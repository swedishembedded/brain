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

/// One recorded step of one episode.
struct Step {
    observation: String,
    options: Vec<String>,
    action: usize,
    old_prob: f32,
    advantage: f32,
    /// The encoder's pooled embedding of this observation - the critic's input.
    feature: Vec<f32>,
    /// What the critic should have predicted here.
    value_target: f32,
}

/// What one rollout produced.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rollout {
    pub episodes: usize,
    pub steps: usize,
    pub mean_return: f32,
    pub wins: usize,
}

pub struct ControlPipeline<E: Env> {
    model: Decide,
    env: E,
    rng: data::rng::Rng,
    /// Advances across every rollout so a run never replays one episode.
    episode_seed: u64,
    last: Rollout,
    critic: Critic,
    /// Mean squared error of the last critic fit - whether the baseline is
    /// worth trusting.
    critic_mse: f32,
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
            let (probs, feature) = self.policy_and_feature(&obs, &options)?;
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
                value_target: 0.0,
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
        let adv = gae(&rewards, &values, GAMMA, GAE_LAMBDA, truncated_value);
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
    fn clone_teacher(
        &mut self,
        episodes: usize,
        epochs: usize,
        max_steps: usize,
        log: &mut dyn FnMut(usize, f32),
        step: &mut usize,
    ) -> Result<f32> {
        let ce = decide::loss::LossConfig::cross_entropy();
        let mut demos: Vec<(String, Vec<String>, usize)> = Vec::new();
        for _ in 0..episodes {
            self.episode_seed += 1;
            let mut obs = self.env.reset(self.episode_seed);
            for _ in 0..max_steps {
                let options = self.env.actions();
                if options.is_empty() {
                    break;
                }
                let Some(teacher) = self.env.demo() else {
                    return Ok(0.0);
                };
                demos.push((obs.clone(), options, teacher));
                let (next, _, done) = self.env.step(teacher);
                obs = next;
                if done {
                    break;
                }
            }
        }
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
        for _ in 0..episodes {
            self.episode_seed += 1;
            let seed = self.episode_seed;
            let (steps, ret, won) = self.episode(seed, false, max_steps, false)?;
            total += ret;
            wins += usize::from(won);
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
        let features: Vec<Vec<f32>> = batch.iter().map(|s| s.feature.clone()).collect();
        let targets: Vec<f32> = batch.iter().map(|s| s.value_target).collect();
        self.critic_mse = self.critic.fit(&features, &targets, 60, 0.02, 1e-5);
        let stats = Rollout {
            episodes,
            steps: batch.len(),
            mean_return: total / episodes.max(1) as f32,
            wins,
        };
        Ok((batch, stats))
    }

    /// One PPO pass over a collected batch.
    fn update(
        &mut self,
        batch: &[Step],
        cfg: &PolicyConfig,
        order: &mut [usize],
        lr_scale: f32,
    ) -> Result<f32> {
        // Shuffled, because consecutive steps of one episode are correlated
        // and a sequential pass would walk the policy along a trajectory
        // instead of averaging over the batch.
        for i in (1..order.len()).rev() {
            let j = (self.rng.next_u64() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        let mut loss = 0.0f32;
        let mut seen = 0usize;
        // MINIBATCHES, not single transitions. One optimizer step per
        // transition is the thing this used to do and it is not policy
        // gradient in any recognizable sense: a single step's advantage is an
        // extremely noisy estimate of the gradient, and Adam applied straight
        // to it chases the noise rather than the signal. Reference PPO
        // implementations split a rollout into a handful of minibatches
        // (4 for Atari, 32 for continuous control) and step once per
        // minibatch.
        for chunk in order.chunks(MINIBATCH) {
            // Per-MINIBATCH advantage normalization, which is where reference
            // implementations do it - not over the whole rollout.
            let mut adv: Vec<f32> = chunk.iter().map(|&i| batch[i].advantage).collect();
            policy::normalize(&mut adv);
            self.model.zero_grads();
            for (slot, &i) in chunk.iter().enumerate() {
                let s = &batch[i];
                let q = self.question(&s.options);
                let act = Act { old_prob: s.old_prob, action: s.action, advantage: adv[slot] };
                loss += self
                    .model
                    .accumulate(&s.observation, &q, |sc| policy::choice_loss(sc, &act, cfg))
                    .map_err(Error::Backend)?;
                seen += 1;
            }
            // The accumulated sum becomes a mean, so one learning rate means
            // the same thing whatever the minibatch happened to hold.
            self.model.adamw_scaled(
                ENCODER_LR * lr_scale,
                HEAD_LR * lr_scale,
                1.0 / chunk.len() as f32,
            );
        }
        Ok(loss / seen.max(1) as f32)
    }

    pub fn save_head(&self, path: impl AsRef<str>) -> Result<()> {
        self.model.save_head(path.as_ref()).map_err(Error::Backend)
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
        Ok(Rollout { episodes: n, steps: 0, mean_return: total / n.max(1) as f32, wins })
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
            policy: PolicyConfig { clip: 0.2, entropy: 0.02, gamma: 0.99 },
            seed: 0,
            warmup_episodes: 60,
            warmup_epochs: 12,
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
        println!(
            "  {} iterations x {} episodes, {} PPO passes each, encoder {}",
            spec.iterations,
            spec.episodes,
            spec.epochs,
            if spec.freeze_encoder { "frozen" } else { "fine-tuned" }
        );
        if spec.warmup_episodes > 0 {
            let bc =
                self.clone_teacher(spec.warmup_episodes, spec.warmup_epochs, spec.max_steps, log, &mut step)?;
            if step > 0 {
                println!(
                    "    warm start: {} scripted episodes x {} passes, final loss {bc:.4}",
                    spec.warmup_episodes, spec.warmup_epochs
                );
            }
        }
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
            for _ in 0..spec.epochs {
                last_loss = self.update(&batch, &spec.policy, &mut order, lr_scale)?;
                step += batch.len();
                log(step, last_loss);
            }
            self.last = stats;
            println!(
                "    iter {:>3}  return {:+.2}  wins {:>3}/{:<3}  steps {:>5}  critic mse {:.3}",
                it + 1,
                stats.mean_return,
                stats.wins,
                stats.episodes,
                stats.steps,
                self.critic_mse
            );
        }
        Ok(TrainReport { steps: step, final_loss: last_loss, seconds: 0.0 })
    }

    fn run_eval(&mut self) -> Result<EvalReport> {
        // Greedy, and on seeds no training rollout used: what the policy would
        // actually do, on situations it has not been updated against.
        let (mut total, mut wins, mut steps) = (0.0f32, 0usize, 0usize);
        let n = EVAL_SEEDS.count();
        for seed in EVAL_SEEDS {
            let (st, ret, won) = self.episode(seed, true, 40, false)?;
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
        let r = self.show(1, 40)?;
        Ok(format!("  return {:+.2}", r.mean_return))
    }
}

impl<E: Env> Flow<ControlPipeline<E>> {
    /// Play `n` episodes greedily, printing every decision - the stage that
    /// shows what the policy learned rather than summarizing it.
    pub fn play(self, n: usize) -> Flow<ControlPipeline<E>> {
        self.stage("play", move |p| {
            let r = p.show(n, 40)?;
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

    pub fn load(self) -> Result<ControlPipeline<E>> {
        let model =
            crate::decision::load_decide(&self.dir, self.head.as_deref(), &self.device, self.limits, self.seed)?;
        let cfg_width = model.cfg.d_model as usize;
        Ok(ControlPipeline {
            model,
            env: self.env,
            rng: data::rng::Rng::new(self.seed ^ 0xc0ffee),
            episode_seed: 0,
            last: Rollout::default(),
            critic: Critic::new(cfg_width, CRITIC_HIDDEN, self.seed ^ 0x1c1),
            critic_mse: 0.0,
        })
    }
}
