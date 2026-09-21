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

    /// Whether the simulation itself has gone wrong, and how.
    ///
    /// A failure is not an ending. `step` can only say "done", so an
    /// environment whose engine died, whose reply would not parse, or which
    /// was handed an action that does not exist had no way to say so: it
    /// reported a terminal transition worth zero reward, and the learner duly
    /// fitted to it. Whatever the policy did before the failure was then
    /// taught as the thing that ended the episode with nothing, which is a
    /// lesson about this code rather than about the game.
    ///
    /// Checked after every step. `None` means the last step was real.
    fn fault(&self) -> Option<String> {
        None
    }

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

    /// What KIND of situation this episode is, when the environment has more
    /// than one kind. `None` when it does not.
    ///
    /// Used to break a measurement down by where it came from. A single
    /// average over a mixed environment answers "is there a signal" and never
    /// "where is the signal", and those want different things done about them:
    /// a number that is the same everywhere is a property of the task, and one
    /// that lives in a tenth of it is an instruction about where to spend the
    /// probe budget.
    fn label(&self) -> Option<String> {
        None
    }

    /// WHICH world this episode is, when [`Env::label`] names a KIND of world
    /// the environment generates many of. `None` when the label already
    /// identifies it.
    ///
    /// A level with fixed geometry is its own instance and needs nothing
    /// here. A generated one does: every maze drawn from one scenario shares
    /// that scenario's label, and anything that keeps "the best run per
    /// label" then holds the best run over all the mazes ever generated -
    /// which is the run that drew the easiest one. Its decisions are about a
    /// layout no other episode has, so cloning them teaches turns that lead
    /// somewhere else.
    fn instance(&self) -> Option<String> {
        None
    }

    /// Whether the episodes from now on COUNT as training.
    ///
    /// An environment that carries state ACROSS episodes - a curriculum that
    /// moves the start as the policy succeeds, a tally of how the last few
    /// went - cannot tell a measurement from the real thing on its own: a
    /// fixed-block score and a hypothetical roll-out both look exactly like an
    /// episode from in there. Left counting, they move what the next TRAINING
    /// episode faces, so how a run trains depends on how often it stopped to
    /// look at itself, and two runs of the same policy on different gauge
    /// budgets are not running the same experiment.
    ///
    /// Counting is the default and the pipeline turns it off around every
    /// measurement it takes, so an environment that keeps nothing across
    /// episodes can ignore this entirely.
    fn set_counting(&mut self, _on: bool) {}

    /// Hold the state this environment is in right now, so that a caller can
    /// come back to it and try something else. `false` when it cannot.
    ///
    /// Default: it cannot. An environment that cannot go back is a complete
    /// environment; what it cannot do is say what a DIFFERENT action would
    /// have been worth, and [`ControlPipeline::counterfactual`] reports that
    /// rather than approximating it by replaying the actions that led here.
    /// Replay is not the same thing: any part of an observation derived from
    /// something outside the simulation - what has been rendered, what a
    /// client has accumulated about the run - does not come back with it.
    ///
    /// Only one state is held at a time; holding again replaces it.
    fn hold(&mut self) -> bool {
        false
    }

    /// Go back to what [`Env::hold`] held, and give the observation there.
    ///
    /// `None` when nothing is held or the environment could not go back, and
    /// in that case the environment is left as it was rather than half
    /// restored.
    fn resume(&mut self) -> Option<String> {
        None
    }

    /// Hold the state in a NUMBERED slot, and say whether it took.
    ///
    /// [`Env::hold`] holds one state, which is what a counterfactual needs -
    /// go back to this decision. A search that returns to promising places
    /// needs many, because its whole advantage is resuming from where it got
    /// to rather than from the start: the depth it can reach stops being
    /// exponential in the length of an episode. `slots` says how many there
    /// are, and zero means the environment cannot do this at all.
    fn hold_at(&mut self, _slot: usize) -> bool {
        false
    }

    /// Go back to what a numbered slot is holding.
    fn resume_from(&mut self, _slot: usize) -> Option<String> {
        None
    }

    /// How many numbered slots there are. Zero disables the search entirely.
    fn slots(&self) -> usize {
        0
    }

    /// A coarse name for WHERE the run is, for an archive to key on.
    ///
    /// The one piece of judgement a search like this needs, and it wants to
    /// be as free of the game as possible: two states with the same name are
    /// treated as the same place and only one of them is kept. Position
    /// rounded to a grid plus what the player is CARRYING is enough, and the
    /// carrying half is what matters - picking up a key makes every cell
    /// reachable with it new, so the search files them and goes on from
    /// there. Nothing has to tell it that keys open doors.
    fn cell(&self) -> Option<String> {
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
/// One decision, with everything needed to reconstruct the question it was
/// an answer to.
///
/// The objective is part of it. Without it a demonstration was replayed
/// against whatever objective the environment happened to be holding when it
/// was fitted, so under `--mix` - where the mission is drawn per episode and
/// prepended to every option - examples gathered under "kill everything"
/// could all be trained as though they had been gathered under "reach the
/// exit". The observation is the same, the right action is not, and nothing
/// in the record said which had been asked.
#[derive(Clone, Debug)]
struct Demo {
    objective: String,
    observation: String,
    options: Vec<String>,
    action: usize,
}
/// One teacher episode: what it scored, and what it did - kept together so the
/// bad ones can be dropped whole rather than a step at a time.
type TeacherRun = (f32, Vec<Demo>);

/// One decision of a student episode, measured against its alternatives.
///
/// There is no prefix here and no replay. Going back is [`Env::hold`] and
/// [`Env::resume`]: replaying the actions that led to a decision reaches the
/// same simulation and not the same OBSERVATION, because an observation can
/// read things the simulation does not own - what has been rendered, what a
/// client has accumulated about the run - and those do not come back with a
/// list of actions. Measured on this repository's DOOM sample before the
/// engine could go back properly, three decisions in ten replayed to a
/// different set of options and had to be thrown away, and the ones that
/// survived were the short prefixes.
struct Branch {
    /// What the trajectory scored, taking this action here and playing the
    /// roll-out out - averaged over the roll-outs if there was more than one.
    score: f32,
    /// How far apart roll-outs of the SAME candidate under the SAME roll-out
    /// policy landed, best to worst.
    ///
    /// The share of a candidate's score that is the path rather than the
    /// action, and the number that says whether a measured margin between two
    /// candidates means anything: a margin smaller than this is inside the
    /// noise of the thing measuring it.
    ///
    /// Within a roll-out policy, never across. A teacher continuation and a
    /// policy continuation differ systematically - the teacher is better -
    /// so comparing one of each would report that difference as noise and
    /// would do it most loudly exactly where the teacher is most worth
    /// beating. Zero unless some roll-out policy was run at least twice,
    /// which is the honest answer when nothing here can tell.
    noise: f32,
    /// Game steps it cost to find out.
    steps: usize,
}

/// The option that means the same as `prev`, if one is on offer.
///
/// An option list rebuilt from the world every step has no stable index, so
/// repeating an action has to be done by intention rather than by number.
/// Options here are sentences whose opening words carry the intention and
/// whose numbers carry the situation - "walk forward, 320 units of open floor
/// ahead" - so the longest shared run of opening words identifies it.
///
/// Two words at minimum, or "turn left and go that way" would count as a
/// repeat of "turn right and go that way".
fn same_again(prev: &str, options: &[String]) -> Option<usize> {
    let words = |s: &str| s.split_whitespace().map(str::to_string).collect::<Vec<_>>();
    let p = words(prev);
    let mut best: Option<(usize, usize)> = None;
    for (i, o) in options.iter().enumerate() {
        let shared = words(o).iter().zip(&p).take_while(|(a, b)| a == b).count();
        if shared >= 2 && best.is_none_or(|(n, _)| shared > n) {
            best = Some((shared, i));
        }
    }
    best.map(|(_, i)| i)
}

/// The best head a phase has seen, INCLUDING the one it was handed.
///
/// Every phase of a control run picks a winner from a score that carries
/// noise - the fixed block, the student's own progress - and "keep the best
/// round" is the right rule for that. What was wrong is where the comparison
/// started: at the first round, never at the policy the phase inherited. So a
/// phase whose every round made things worse still adopted one of them, and a
/// run could leave a phase worse off than it entered it with nothing in the
/// log saying so.
///
/// Seeding the comparison with the incoming policy makes "do nothing" a
/// candidate, which is what it has to be for a phase to be safe to add to a
/// pipeline. See `keep_tests`.
struct Keep {
    at: usize,
    rank: f32,
    weights: Vec<(String, Vec<f32>)>,
}

impl Keep {
    /// Start from what the phase was handed, at whatever it scores.
    fn starting(rank: f32, weights: Vec<(String, Vec<f32>)>) -> Keep {
        Keep { at: 0, rank, weights }
    }

    /// Offer a round's result, and say whether it won. The weights are read
    /// only if it does, because reading them is a device readback and most
    /// rounds lose.
    fn offer(
        &mut self,
        at: usize,
        rank: f32,
        weights: impl FnOnce() -> Vec<(String, Vec<f32>)>,
    ) -> bool {
        if rank <= self.rank {
            return false;
        }
        self.at = at;
        self.rank = rank;
        self.weights = weights();
        true
    }

    fn best(&self) -> (usize, &[(String, Vec<f32>)]) {
        (self.at, &self.weights)
    }
}

/// How many distinct worlds of one KIND the archive keeps a run for.
///
/// One is what a generated scenario used to get, and one is the run that drew
/// the easiest maze: every later round then cloned that single trajectory, so
/// the search kept finding new worlds and the compression kept being handed
/// the same old one. A level with fixed geometry has exactly one instance and
/// is unaffected at any cap above zero.
///
/// It also bounds the FRONTIER, now that a search files one entry per place
/// it reached rather than one per episode. Eight of those is not a frontier,
/// it is eight scattered spots. Large enough to describe the edge of what has
/// been reached, small enough that cloning the archive is still cloning a
/// selection.
const KEEP_PER_KIND: usize = 48;

/// One world's best run, and what kind of world it was.
#[derive(Clone)]
struct Solved {
    /// What KIND - [`Env::label`]. Many instances can share it.
    kind: String,
    /// WHICH one - [`Env::instance`], falling back to the label. Unique.
    instance: String,
    score: f32,
    demos: Vec<Demo>,
}

/// The runs worth cloning: the best one per world, capped per kind of world.
///
/// Keyed by instance, so finding a way through a hard world is never undone
/// by finding a better way through an easy one - that is the failure a greedy
/// top-N archive has, and it deletes exactly the stepping stone worth having.
/// Capped per kind, so an environment that generates a fresh world every
/// episode does not turn self-imitation into cloning everything it has ever
/// played, most of which it played badly.
#[derive(Default)]
struct Archive {
    by_instance: std::collections::HashMap<String, Solved>,
}

impl Archive {
    /// Offer a run. `true` when the archive now holds it.
    fn offer(&mut self, kind: &str, instance: &str, score: f32, demos: Vec<Demo>) -> bool {
        if let Some(had) = self.by_instance.get(instance) {
            if score <= had.score {
                return false;
            }
        }
        self.by_instance.insert(
            instance.to_string(),
            Solved {
                kind: kind.to_string(),
                instance: instance.to_string(),
                score,
                demos,
            },
        );
        // Over the cap, the weakest world of this kind goes - which may be the
        // one just offered, and then nothing was gained.
        let mut of_kind: Vec<(String, f32)> = self
            .by_instance
            .values()
            .filter(|s| s.kind == kind)
            .map(|s| (s.instance.clone(), s.score))
            .collect();
        if of_kind.len() > KEEP_PER_KIND {
            of_kind.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let (drop, _) = of_kind.remove(0);
            self.by_instance.remove(&drop);
            if drop == instance {
                return false;
            }
        }
        true
    }

    fn len(&self) -> usize {
        self.by_instance.len()
    }

    /// The weakest run being cloned, which is the bar the search has to clear.
    fn worst(&self) -> f32 {
        self.by_instance
            .values()
            .map(|s| s.score)
            .fold(f32::INFINITY, f32::min)
    }

    /// The archived decisions worth imitating: those from runs that did at
    /// least as well as `bar`.
    ///
    /// Not everything in here. The archive holds two different kinds of thing
    /// now - whole runs, and the best way to reach each place a search got to
    /// - and only the first is behaviour. A search fragment is a record of
    /// somewhere worth returning to, which is what makes it worth keeping,
    /// and it is mostly a random walk, which is what makes it not worth
    /// copying.
    ///
    /// Imitating it all is what self-imitation is specifically not: the
    /// method is to clone the runs that beat what you usually do, and cloning
    /// the rest teaches the average. Measured, cloning all forty-eight
    /// entries took the policy's score on the fixed block from 0.123 down to
    /// 0.087 over three rounds while the archive itself was improving the
    /// whole time.
    fn demos_above(&self, bar: f32) -> Vec<Demo> {
        self.by_instance
            .values()
            .filter(|s| s.score >= bar)
            .flat_map(|s| s.demos.iter().cloned())
            .collect()
    }

    /// The best run per kind, which is what a reader wants to see: one line
    /// per scenario, not one per world.
    fn by_kind(&self) -> Vec<(String, f32, usize)> {
        let mut acc: std::collections::HashMap<&str, (f32, usize)> =
            std::collections::HashMap::new();
        for s in self.by_instance.values() {
            let e = acc.entry(&s.kind).or_insert((f32::NEG_INFINITY, 0));
            e.0 = e.0.max(s.score);
            e.1 += 1;
        }
        let mut out: Vec<(String, f32, usize)> =
            acc.into_iter().map(|(k, (v, n))| (k.to_string(), v, n)).collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }
}

/// Which of a round's two policies a number was measured on.
///
/// A round measures, then updates. The rollout it scores was drawn by the
/// weights it ENTERED with, and so is any progress or agreement number taken
/// while collecting; a fixed-block gauge is run after the update and measures
/// what the round PRODUCED. Both are handed to the same [`Keep`], and pairing
/// one with the other's weights keeps whatever happens to follow the best
/// score rather than the policy that earned it.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Scored {
    /// The weights that drew the rollout.
    Entering,
    /// The weights the update left behind.
    Produced,
}

/// The number a round is ranked on, and which of its two policies owns it.
///
/// `None` when the round measured nothing comparable, which leaves the caller
/// its own last resort - a fit loss belongs to the weights the fit produced,
/// not to the ones it started from.
fn ranking(gauged: Option<f32>, entering: Option<f32>) -> Option<(f32, Scored)> {
    match (gauged, entering) {
        (Some(g), _) => Some((g, Scored::Produced)),
        (None, Some(e)) => Some((e, Scored::Entering)),
        (None, None) => None,
    }
}

/// The candidate that measured best, or `None` when the decision did not
/// discriminate.
///
/// A probe where every candidate led to the same place has no right answer to
/// score a policy against, and averaging those in measures an arbitrary
/// argmax over a flat set rather than anything about the policy. Measured on
/// this sample, 40% to 52% of probed decisions are that decision - enough to
/// dominate any number they are allowed into.
fn measured_best(scored: &[f32]) -> Option<usize> {
    let hi = scored.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lo = scored.iter().copied().fold(f32::INFINITY, f32::min);
    // The same tolerance `probe` counts `pivotal` with: two trajectories that
    // differ only in which way the player faced for one decision score the
    // same to within rounding.
    if !(hi - lo > 1e-3) {
        return None;
    }
    scored
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(k, _)| k)
}

/// How far the policy has moved, over the candidates a probe measured, from
/// the policy that measured them.
///
/// The forward KL from the collecting policy, which is what `--target-kl`
/// means in the PPO loop - so one flag bounds the movement of both updates
/// and a reader does not have to learn two notions of "too far".
///
/// The outcome fit needs it for a reason the gain limit does not cover.
/// [`OUTCOME_GAIN_LIMIT`] bounds what any ONE decision may pull; nothing
/// bounded how far the whole update travelled, and fitting a few dozen
/// probed states hard enough moves a head that answers for thousands of
/// states nobody measured. Measured: two rounds took a DOOM policy's fixed
/// block from 0.780 to 0.440 while the belief on the best-measured option
/// climbed from 0.09 to 0.21 - the update reaching its target and taking the
/// rest of the policy with it, which is exactly the failure a trust region
/// exists to stop.
fn candidate_drift(old: &[f32], new: &[f32]) -> f32 {
    let mut kl = 0.0f64;
    for (o, n) in old.iter().zip(new) {
        let o = (*o as f64).clamp(1e-8, 1.0);
        let n = (*n as f64).clamp(1e-8, 1.0);
        kl += o * (o / n).ln();
    }
    (kl as f32).max(0.0)
}

/// The cost of each measured candidate, relative to the best of them.
///
/// `c(a) = max Q - Q(a)`, non-negative and zero for the winner, which is the
/// cost vector a cost-sensitive multiclass example carries. LOLS defines it
/// exactly this way; AggreVaTe's reduction is the same quantity up to the
/// constant that the argmin does not see.
///
/// No temperature, no normalisation. The costs are already in the units the
/// run is scored in, so a decision where the candidates differ by 0.003 pulls
/// a hundredth of what one where they differ by 0.3 does, on its own, and a
/// decision where they are equal pulls nothing at all. An earlier version put
/// these through a softmax at a temperature and weighted the result by the
/// spread, which threw the magnitudes away and then tried to reintroduce them.
fn outcome_costs(scored: &[f32]) -> Vec<f32> {
    let best = scored.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    scored.iter().map(|&s| best - s).collect()
}

/// The cost-sensitive classification loss and its gradient on the candidate
/// logits.
///
/// ```text
/// L        = sum_a pi(a) c(a)          the policy's own expected cost
/// dL/dz_k  = pi_k (c_k - L)
/// ```
///
/// This is the objective AggreVaTe reduces to - `argmin_pi sum_i Q(pi(s_i))`
/// - relaxed from a hard argmin to the stochastic policy the head already
/// parameterises. Three properties, all of which the version it replaces had
/// to reach for separately and one of which it never got:
///
/// * **Bounded**, by `min c` below and `max c` above. The construction it
///   replaces was a sum of log-probabilities weighted by signed advantages,
///   which is unbounded below - a fact that cost this sample a training run
///   before it was recognised.
/// * **Silent where nothing was learned.** All costs equal gives `c_k = L`
///   for every k and a gradient of exactly zero, with no threshold. At 40% to
///   52% of probed decisions that is the case.
/// * **Proportionate.** The gradient carries the cost in the units it was
///   measured in rather than a rank sharpened by a temperature.
fn outcome_pull(belief: &[f32], costs: &[f32]) -> (f32, Vec<f32>) {
    let l: f32 = belief.iter().zip(costs).map(|(&p, &c)| p * c).sum();
    let grad = belief.iter().zip(costs).map(|(&p, &c)| p * (c - l)).collect();
    (l, grad)
}

/// How well a critic predicts the score an episode finally reaches.
#[derive(Clone, Copy, Debug, Default)]
pub struct ValueFit {
    /// Error on the episodes it was fitted to.
    pub train_rmse: f32,
    /// Error on episodes it has never seen. The number that matters.
    pub test_rmse: f32,
    /// Standard deviation of the thing being predicted, so the error above
    /// can be read as a fraction of it. An error equal to this is a critic
    /// that has learned the mean and nothing else.
    pub spread: f32,
    pub states: usize,
    pub episodes: usize,
}

/// One decision, the candidates tried there, and what each was actually
/// worth.
///
/// The unit an outcome-fitted update learns from. Unlike a demonstration it
/// carries no opinion about which action was right - only a number per
/// candidate, measured by playing the run out and scoring it.
struct Probe {
    /// What kind of situation it came from, for the breakdown. See
    /// [`Env::label`].
    label: Option<String>,
    /// The objective this decision was taken under, for the same reason a
    /// [`Demo`] carries one: a probe is replayed, and replaying it against
    /// whatever objective the environment holds later asks a different
    /// question of the same observation.
    objective: String,
    observation: String,
    options: Vec<String>,
    /// `(option, what the whole trajectory scored taking it)`. The first is
    /// always the teacher's own choice, so a caller can say what the update
    /// gained over it.
    tried: Vec<(usize, f32)>,
    /// What the policy believed about those same candidates when it measured
    /// them, renormalised over the ones tried. The fit's trust region is
    /// distance from THIS - see [`candidate_drift`] - and it has to be the
    /// policy that collected the probe rather than whatever the head says by
    /// the time an epoch reaches it.
    belief: Vec<f32>,
}

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

/// One roll-out a branch runs: which candidate it tries, and under what.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Trial {
    /// The action being tried - an index into the decision's options.
    candidate: usize,
    /// Where that action sits in the branch's candidate list, so a caller can
    /// accumulate by candidate. Nothing a trial is measured UNDER may depend
    /// on it.
    at: usize,
    /// Which repeat of this candidate this is. It is also the draw stream:
    /// every candidate's repeat `r` is rolled out against the same numbers,
    /// which is what makes the DIFFERENCE between two of them readable.
    repeat: usize,
    /// Rolled out by the reference policy rather than by the learner.
    reference: bool,
    /// Put the held state back before this trial runs, and check that it
    /// arrived.
    restore: bool,
}

/// Every roll-out a branch has to run, in the order it runs them.
///
/// One rule, and it is the whole reason this is a function rather than two
/// nested loops: what a candidate is measured under may not depend on where
/// in the list it sits. Same draws, same roll-out policy, same restored and
/// checked starting state, in whatever order the candidates were assembled -
/// so reordering them reorders the work and changes no number.
fn branch_plan(candidates: &[usize], reference_rollout: &[bool]) -> Vec<Trial> {
    let mut plan = Vec::with_capacity(candidates.len() * reference_rollout.len());
    for (at, &candidate) in candidates.iter().enumerate() {
        for (repeat, &reference) in reference_rollout.iter().enumerate() {
            plan.push(Trial { candidate, at, repeat, reference, restore: true });
        }
    }
    plan
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

/// Which alternatives a counterfactual spends its game steps on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Candidates {
    /// The ones the policy ranks highest after the teacher's own - what
    /// training would actually move toward, and so the right set for
    /// deciding whether to train on measured outcomes.
    ///
    /// It is the wrong set for asking whether room exists at all: a policy
    /// fitted to the teacher ranks the teacher's near-duplicates highest, so
    /// this asks about the actions least likely to lead anywhere different.
    Contested,
    /// Drawn at random from everything on offer. The control for the above.
    Wide,
}

/// Where a probe's wall clock went.
///
/// Kept because a probe is the most expensive thing this pipeline does and
/// its cost is not where one would guess: the policy calls are the obvious
/// candidate and were not the answer.
#[derive(Clone, Copy, Debug, Default)]
pub struct Spend {
    /// How many decisions the policy was asked for, so `deciding` can be read
    /// per call rather than in total.
    pub decisions: usize,
    /// How many packed rows the encoder ran over, and how many overlapping
    /// windows those were split into. A request past `Limits::max_span` costs
    /// one full forward pass PER WINDOW.
    pub tokens: usize,
    pub windows: usize,
    pub deciding: std::time::Duration,
    pub stepping: std::time::Duration,
    pub holding: std::time::Duration,
    pub resuming: std::time::Duration,
    pub scoring: std::time::Duration,
}

impl Spend {
    pub fn total(&self) -> std::time::Duration {
        self.deciding + self.stepping + self.holding + self.resuming + self.scoring
    }
}

impl std::fmt::Display for Spend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let t = self.total().as_secs_f32().max(1e-6);
        let pct = |d: std::time::Duration| d.as_secs_f32() / t * 100.0;
        write!(
            f,
            "{:.0}s over {} decisions ({:.1} ms each, {} rows in {:.2} windows) = deciding {:.0}% \
             stepping {:.0}% holding {:.0}% resuming {:.0}% scoring {:.0}%",
            t,
            self.decisions,
            self.deciding.as_secs_f32() * 1000.0 / self.decisions.max(1) as f32,
            self.tokens / self.decisions.max(1),
            self.windows as f32 / self.decisions.max(1) as f32,
            pct(self.deciding),
            pct(self.stepping),
            pct(self.holding),
            pct(self.resuming),
            pct(self.scoring)
        )
    }
}

/// One kind of situation, and how much room a probe found in it.
#[derive(Clone, Debug, Default)]
pub struct Situation {
    pub label: String,
    pub states: usize,
    /// Share of them where the candidates did not all lead to the same place.
    pub pivotal: f32,
    /// Mean `best - teacher` over them.
    pub regret: f32,
}

/// Whether a different action would have been worth taking.
#[derive(Clone, Debug, Default)]
pub struct Counterfactual {
    /// Decision points actually measured.
    pub states: usize,
    /// Points thrown away because the replay did not land back where it
    /// started. Anything but zero and the method is unsound - see
    /// [`ControlPipeline::counterfactual`].
    pub adrift: usize,
    /// Fraction of points where SOME alternative scored better than the
    /// action the teacher chose.
    pub beaten: f32,
    /// Mean amount by which the best alternative beat the teacher, over the
    /// points where one did.
    pub gain: f32,
    /// Mean over ALL points of `best - teacher`: what a policy that always
    /// picked the best of the candidates offered here would gain over the
    /// teacher. The ceiling on what this signal is worth.
    pub regret: f32,
    /// Mean spread between the best and worst candidate at a point - whether
    /// the choice matters at all, before asking who makes it well.
    ///
    /// A MEAN over decisions, so it hides the shape. A task where one
    /// decision in ten decides the episode and the other nine are free reads
    /// the same here as one where every decision nudges the outcome slightly,
    /// and those two want completely different things done about them - see
    /// [`Counterfactual::pivotal`].
    pub spread: f32,
    /// How far apart a SINGLE candidate's own roll-outs landed, averaged over
    /// candidates and decisions.
    ///
    /// The measuring stick for [`Self::spread`], and the number this study
    /// was missing. A 0.03 margin between two candidates is evidence that one
    /// is better only if re-running the same candidate does not move its
    /// score by more than that; if it does, the ranking is a fact about the
    /// path the roll-out happened to take. Needs `--repeats 2` or more to be
    /// anything but zero.
    pub noise: f32,
    /// Roll-outs run per candidate, so a reader can tell "the noise is zero"
    /// from "nobody measured the noise".
    pub repeats: usize,
    /// Fraction of decisions where the candidates did NOT all lead to the
    /// same place.
    ///
    /// The number to read first. A policy can only be better than a teacher
    /// where the choice has a consequence, so this is the share of decisions
    /// any method could possibly improve - and if it is near zero the task
    /// has no room at the level of single decisions however good the learner
    /// is.
    pub pivotal: f32,
    /// Game steps spent measuring.
    pub steps: usize,
    /// Where the wall clock went.
    pub spend: Spend,
    /// The same numbers split by [`Env::label`], worst-first by room. Empty
    /// when the environment does not label its episodes.
    pub by_situation: Vec<Situation>,
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
    /// Whether this pipeline began from trained weights rather than from
    /// nothing. See the warm start in `run_train`.
    started_from_head: bool,
    /// The cells the search has reached in the world it is exploring, the
    /// slot holding each, and the next free slot. Carried between rounds:
    /// see `explore`, where rebuilding it every round was the difference
    /// between a search that compounds and four copies of the first round.
    explored: Option<(String, std::collections::HashMap<String, (usize, f32, u32)>, usize)>,
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
    /// The reward discount the advantage estimator uses.
    ///
    /// Read from the spec rather than baked in: it was the constant 0.99
    /// while the sample's own reward was built to telescope to the final
    /// score undiscounted, so the estimator was optimising something the run
    /// was not kept on. See [`GAMMA`].
    gamma: f32,
    /// Episode horizon for [`Stages::run_eval`] and [`Flow::play`].
    ///
    /// Set from [`ControlSpec::max_steps`] when a run trains, and settable on
    /// the builder for a pipeline that only loads weights. It used to be the
    /// constant 40 in both places, which silently truncated any environment
    /// whose episodes are longer than a toy's: an agent that needs 200
    /// decisions to reach a goal was scored as never reaching it, and the
    /// number looked like a policy failure rather than a harness one.
    max_steps: usize,
    /// How many measurements are in progress. See [`ControlPipeline::measuring`].
    measurements: usize,
}

impl<E: Env> std::fmt::Debug for ControlPipeline<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ControlPipeline").finish_non_exhaustive()
    }
}

/// The encoder arrives pretrained and the head does not.
const ENCODER_LR: f32 = 1e-5;
const HEAD_LR: f32 = 3e-4;
/// The discount used when a caller has not chosen one.
///
/// ONE, not 0.99, and the difference is the whole task. Under `--reward
/// gauge` a decision is paid what it moved the score the run is finally kept
/// on, so an episode's UNDISCOUNTED return is exactly that score. Discount it
/// and that identity breaks:
///
/// ```text
/// sum_t gamma^t (M_t+1 - M_t)
///     = -M_0 + (1-gamma) sum_t gamma^(t-1) M_t + gamma^(T-1) M_T
/// ```
///
/// The middle term is score held EARLY, which the gauge does not reward and
/// the run is not kept on. At 0.99 the ranking inverts on cases that matter:
/// 0.4 gained at decision 0 and lost again by 200 discounts to about 0.35,
/// while 1.5 gained once at decision 300 discounts to about 0.07 - so a run
/// that ended with nothing outranks one that finished the level.
///
/// A finite-horizon task scored on where the run ended wants no discount at
/// all. `--gamma` is there for a caller whose task genuinely prefers sooner
/// to later.
const GAMMA: f32 = 1.0;
/// The bias/variance dial on the advantage estimator: 0 is the one-step TD
/// error and 1 is the full return minus the baseline. The usual middle.
const GAE_LAMBDA: f32 = 0.95;
/// The critic's hidden width, over the encoder's 384-d pooled embedding.
const CRITIC_HIDDEN: usize = 64;


/// How often an exploring step repeats what it just did.
///
/// Go-Explore's own value for Atari, and the single most load-bearing
/// constant in the search. Without it exploration is a random walk, which
/// covers ground like the square root of the steps taken and therefore
/// covers almost none.
const REPEAT_CHANCE: f32 = 0.95;

/// How often an exploring step does what a competent player would.
///
/// A blind walk in a level full of things that shoot back is mostly a dead
/// one, and a dead walk reaches nothing. A third of the steps taken from the
/// scripted player keeps the walk alive long enough to wander somewhere, and
/// leaves two thirds of them free to do what no competent player would - which
/// is the only way a search finds what its teacher never did.
const GUIDED_CHANCE: f32 = 0.34;
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
            // A control loop rather than a document - but a request is the
            // state split into windows PLUS one span per option, and the
            // option list is rebuilt from the world at every step, so the row
            // count is data and not a constant. Measured on the DOOM sample,
            // a late-episode decision with fifteen options packs just over a
            // thousand rows; at 1024 a run died in its twenty-sixth minute
            // with everything it had learned thrown away, which is the cost
            // of sizing this to the typical request instead of the long one.
            limits: Limits { cap_rows: 4096, cap_slots: 64, max_span: 256, overlap: 32 },
            seed: 0,
            max_steps: ControlSpec::default().max_steps,
        }
    }

    fn question(&self, options: &[String]) -> Question {
        self.question_for(&self.env.objective(), options)
    }

    /// The question as it was ASKED, rather than as the environment would ask
    /// it now. A record that is replayed has to carry its own objective - see
    /// [`Demo`].
    fn question_for(&self, objective: &str, options: &[String]) -> Question {
        Question::Choice {
            instructions: objective.to_string(),
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
            // Before anything is recorded: a failure is not a transition, and
            // the steps taken before one are not evidence about the policy.
            if let Some(why) = self.env.fault() {
                return Err(Error::Backend(format!("the environment failed: {why}")));
            }
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
        let adv = gae(&rewards, &values, self.gamma, self.gae_lambda, truncated_value);
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
                demos.push(Demo { objective: self.env.objective(), observation: obs.clone(), options, action: teacher });
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
        self.fit_demos(&demos, epochs, log, step)
    }

    /// Fit the head to a set of labelled decisions, by ordinary supervised
    /// learning.
    ///
    /// Separate from whoever collected them, because the two sources of
    /// labels this run has - the teacher's own trajectory, and the states the
    /// student reached with the teacher asked at each of them - differ only in
    /// collection and must be learned from identically. A DAgger round that
    /// fitted with its own slightly different loop would be comparing two
    /// things at once.
    fn fit_demos(
        &mut self,
        demos: &[Demo],
        epochs: usize,
        log: &mut dyn FnMut(usize, f32),
        step: &mut usize,
    ) -> Result<f32> {
        if demos.is_empty() {
            return Ok(0.0);
        }
        let ce = decide::loss::LossConfig::cross_entropy();
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
                    let demo = &demos[d];
                    // The objective it was COLLECTED under, not whichever one
                    // the environment is holding now. See [`Demo`].
                    let q = self.question_for(&demo.objective, &demo.options);
                    epoch_loss += self
                        .model
                        .accumulate(&demo.observation, &q, |sc| {
                            decide::loss::decision_loss(sc, demo.action, &ce)
                        })
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

    /// One round of labels from the states the STUDENT reaches.
    ///
    /// The student acts; the teacher is asked, at every state the student
    /// arrived at, what it would have done there - and its answer is recorded
    /// WITHOUT being executed. That is the whole difference from cloning, and
    /// it is the point.
    ///
    /// Behaviour cloning only ever sees the teacher's own trajectory. The
    /// student's first mistake takes it somewhere that trajectory says nothing
    /// about, so it makes a second, which takes it somewhere stranger still:
    /// the errors compound, and the classic bound on them is quadratic in the
    /// episode's length rather than linear. Training on the distribution the
    /// student itself induces is what makes them linear again - it is the
    /// dataset-aggregation loop of Ross, Gordon and Bagnell (2011), and here
    /// it costs nothing but game steps, because this environment's teacher
    /// answers for free at any state.
    ///
    /// The student SAMPLES rather than taking its best action: one greedy
    /// trajectory per world is one path through it, and the round exists to
    /// cover the ways the student can go wrong, not to show off the way it
    /// currently goes right.
    ///
    /// Returns the labelled decisions, how far the student's own episodes got,
    /// and how often it already agreed with the teacher - which is the number
    /// that says whether the round had anything to teach.
    fn label_student(
        &mut self,
        episodes: usize,
        max_steps: usize,
    ) -> Result<Option<(Vec<Demo>, Option<f32>, f32)>> {
        let mut demos: Vec<Demo> = Vec::new();
        let (mut got, mut measured) = (0.0f32, 0usize);
        let (mut agreed, mut seen) = (0usize, 0usize);
        for _ in 0..episodes {
            self.episode_seed += 1;
            let mut obs = self.env.reset(self.episode_seed);
            for _ in 0..max_steps {
                let options = self.env.actions();
                if options.is_empty() {
                    break;
                }
                // Asked, and its answer NOT executed. A teacher that
                // remembers what it has already suggested - this sample's does,
                // to rotate out of a stuck spot - therefore rotates on
                // suggestions the student never took. That makes its label in a
                // stuck state a function of how often it has been asked as well
                // as of the state, which is real label noise; it is the same
                // noise cloning already learns from, because there too the
                // teacher rotates while the player stands still.
                let Some(teacher) = self.env.demo() else {
                    return Ok(None);
                };
                let probs = self.policy(&obs, &options)?;
                let mut best = 0;
                for (i, &pi) in probs.iter().enumerate() {
                    if pi > probs[best] {
                        best = i;
                    }
                }
                agreed += usize::from(best == teacher);
                seen += 1;
                let mut u = self.rng.next_f32();
                let mut chosen = probs.len() - 1;
                for (i, &pi) in probs.iter().enumerate() {
                    if u < pi {
                        chosen = i;
                        break;
                    }
                    u -= pi;
                }
                demos.push(Demo { objective: self.env.objective(), observation: std::mem::take(&mut obs), options, action: teacher });
                let (next, _, done) = self.env.step(chosen);
                obs = next;
                if done {
                    break;
                }
            }
            if let Some(p) = self.env.progress() {
                got += p;
                measured += 1;
            }
        }
        Ok(Some((
            demos,
            (measured > 0).then(|| got / measured as f32),
            agreed as f32 / seen.max(1) as f32,
        )))
    }

    /// Rounds of run-the-student, label-it-with-the-teacher, refit.
    ///
    /// The aggregation is the algorithm: every round's labels are KEPT and
    /// the fit is over all of them, so the dataset grows toward the states
    /// the student actually reaches while never forgetting the ones the
    /// teacher showed it. Fitting on the newest round alone would be a moving
    /// target, and is the variant the original paper shows can cycle.
    ///
    /// The roll-in is the pure student, with no mixing back toward the
    /// teacher. The warm start already IS the mixed first round - it is the
    /// teacher driving, at a mixing weight of one - so the schedule this
    /// implements is the standard one, with beta dropped to zero after it.
    fn dagger(
        &mut self,
        spec: &ControlSpec,
        log: &mut dyn FnMut(usize, f32),
        step: &mut usize,
    ) -> Result<()> {
        let mut aggregate: Vec<Demo> = Vec::new();
        // The warm start's own policy is a candidate, because refitting on a
        // bigger set is not monotone and a phase that only ever compares its
        // own rounds cannot conclude that none of them helped. See [`Keep`].
        let entered = self.gauge(spec.gauge_episodes, spec.max_steps)?;
        let mut keep =
            Keep::starting(entered.unwrap_or(f32::NEG_INFINITY), self.model.head_weights());
        // What one round's worth of labels is, taken from the first round.
        // See the pass count below.
        let mut unit = 0usize;
        for round in 0..spec.dagger {
            // The weights the student is about to be driven with. Its own
            // progress over those episodes is a number about THIS policy, not
            // about whatever the refit below leaves behind. See [`Scored`].
            let entering = self.model.head_weights();
            let Some((demos, progress, agreed)) =
                self.label_student(spec.episodes, spec.max_steps)?
            else {
                println!("    dagger: the environment has no teacher to label with");
                return Ok(());
            };
            let fresh = demos.len();
            if fresh == 0 {
                println!("    dagger: the student produced no decisions to label");
                return Ok(());
            }
            aggregate.extend(demos);
            if unit == 0 {
                unit = aggregate.len();
            }
            // A constant amount of OPTIMIZATION per round, not a constant
            // number of passes. The aggregate grows by one round's labels
            // every round by construction, so a fixed pass count makes the
            // fifth round cost five times the first for no reason: the head
            // is already fitted to all but the newest of what is in there.
            // Passes are set so every round does about the work the first one
            // did, which makes the whole phase linear in rounds instead of
            // quadratic.
            let epochs = (spec.warmup_epochs * unit / aggregate.len().max(1)).max(1);
            let loss = self.fit_demos(&aggregate, epochs, log, step)?;
            let gauged = self.gauge(spec.gauge_episodes, spec.max_steps)?;
            println!(
                "    dagger {:>2}  {fresh} new labels ({} in all, {epochs} passes)  the \
                 student had already agreed {:.0}%  loss {loss:.4}{}{}",
                round + 1,
                aggregate.len(),
                agreed * 100.0,
                match progress {
                    Some(p) => format!("  its own episodes {p:.2}"),
                    None => String::new(),
                },
                match gauged {
                    Some(g) => format!("  fixed block {g:.3}"),
                    None => String::new(),
                }
            );
            // Refitting on a bigger set is not monotone either: a round that
            // adds mostly labels for states the student has stopped visiting
            // can move the head away from the ones it is in now.
            let (rank, whose) =
                ranking(gauged, progress).unwrap_or((-loss, Scored::Produced));
            keep.offer(round + 1, rank, || match whose {
                Scored::Entering => entering.clone(),
                Scored::Produced => self.model.head_weights(),
            });
        }
        let (round, w) = keep.best();
        if round != spec.dagger {
            println!(
                "    keeping {}, which scored {:.3}",
                match round {
                    0 => "the policy the warm start produced".to_string(),
                    n => format!("dagger round {n}"),
                },
                keep.rank
            );
            let w = w.to_vec();
            self.model.set_head_weights(&w);
        }
        Ok(())
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

    /// Run `f` as a MEASUREMENT: nothing the environment sees while it runs
    /// counts as training. See [`Env::set_counting`].
    ///
    /// A scope rather than two calls at the call site because the things that
    /// need it are the things that give up half way - a probe that cannot go
    /// back, an episode whose engine died - and a flag left off by an early
    /// return would silently stop the rest of the run from training. Nested,
    /// because a measurement is allowed to contain one; counting resumes when
    /// the outermost finishes.
    fn measuring<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.measurements += 1;
        self.env.set_counting(false);
        let out = f(self);
        self.measurements -= 1;
        if self.measurements == 0 {
            self.env.set_counting(true);
        }
        out
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
        self.measuring(|p| -> Result<()> {
            for i in 0..episodes {
                // A block no rollout will ever draw, so the policy is judged
                // on worlds it was not just trained on.
                p.episode_seed = 0x4000_0000 + i as u64;
                p.episode(p.episode_seed, false, max_steps, false)?;
                if let Some(scored) = p.env.progress() {
                    got += scored;
                    measured += 1;
                }
            }
            Ok(())
        })?;
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
        let asked = self.measuring(|p| -> Result<bool> {
            for &seed in seeds {
                let mut obs = p.env.reset(seed);
                for _ in 0..max_steps {
                    let options = p.env.actions();
                    if options.is_empty() {
                        break;
                    }
                    let Some(teacher) = p.env.demo() else {
                        return Ok(false);
                    };
                    let probs = p.policy(&obs, &options)?;
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
                    let (next, _, done) = p.env.step(teacher);
                    obs = next;
                    if done {
                        break;
                    }
                }
            }
            Ok(true)
        })?;
        self.episode_seed = saved;
        if !asked || states == 0 {
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

    /// Clone the teacher, as a training run's first phase does.
    ///
    /// Public because a diagnostic needs a policy before it can measure
    /// anything, and there is no reason for each of them to hold its own
    /// slightly different idea of what cloning means.
    pub fn warm_start(
        &mut self,
        spec: &ControlSpec,
        log: &mut dyn FnMut(usize, f32),
    ) -> Result<f32> {
        self.model.set_encoder_frozen(spec.freeze_encoder);
        self.max_steps = spec.max_steps;
        let mut step = 0usize;
        self.clone_teacher(
            spec.warmup_episodes,
            spec.warmup_epochs,
            spec.max_steps,
            spec.warmup_keep,
            log,
            &mut step,
        )
    }

    /// Fit a critic to predict how an episode ends, and report its error on
    /// episodes it has never seen. See [`Self::probe_value`].
    pub fn value_fit(
        &mut self,
        episodes: usize,
        max_steps: usize,
        epochs: usize,
    ) -> Result<Option<ValueFit>> {
        self.probe_value(episodes, max_steps, epochs)
    }

    /// Would a different action have been worth taking?
    ///
    /// The one question imitation cannot ask. Cloning and DAgger both ask
    /// which action the teacher TOOK; neither asks what happens if a
    /// different one is taken instead, and only the second has an answer
    /// that can be better than the teacher's.
    ///
    /// The student plays. At a sampled fraction of its decisions the
    /// environment is asked to HOLD, and then, for each candidate action:
    /// take it, let the TEACHER play the rest of the episode out, score the
    /// whole trajectory - the student's prefix included - with
    /// [`Env::progress`], and resume. The candidates are the teacher's own
    /// choice plus the `alternatives` the policy ranks highest among the
    /// rest, because those are the ones training would actually move toward
    /// and the rest are not worth the game steps.
    ///
    /// Scoring the WHOLE trajectory rather than the continuation is the
    /// point: an action is worth what the run that contains it is worth, and
    /// a continuation scored alone ranks an action by where it happened to
    /// start.
    ///
    /// One repetition per candidate is EXACT rather than a sample, because
    /// nothing in the continuation draws a random number - the teacher is a
    /// script and the engine is lockstep.
    ///
    /// Returns `None` when the environment has no teacher, or cannot go back.
    pub fn counterfactual(
        &mut self,
        episodes: usize,
        states: usize,
        alternatives: usize,
        from: Candidates,
        spec: &ControlSpec,
    ) -> Result<Option<Counterfactual>> {
        Ok(self.probe(episodes, states, alternatives, from, spec)?.map(|(c, _)| c))
    }

    /// As [`Self::counterfactual`], and keep what every branch was worth.
    ///
    /// The summary is what a person reads; the probes are what an update
    /// learns from, and they are the same measurement rather than two runs of
    /// it - so the room a round reports is exactly the room its update had to
    /// work with. See [`Self::fit_outcomes`].
    fn probe(
        &mut self,
        episodes: usize,
        states: usize,
        alternatives: usize,
        from: Candidates,
        spec: &ControlSpec,
    ) -> Result<Option<(Counterfactual, Vec<Probe>)>> {
        let max_steps = spec.max_steps;
        let mut probes: Vec<Probe> = Vec::new();
        let mut spend = Spend::default();
        // How often to stop and branch. Spread over the whole of every
        // episode rather than taken from the front: the first decisions of a
        // level are the ones every run agrees about, and measuring those
        // would measure the start of a level and call it the level.
        let want = states.max(1);
        let total = episodes.max(1) * max_steps.max(1);
        let every = (total / want).max(1);

        let (mut beaten, mut gain, mut regret, mut spread) = (0usize, 0.0f64, 0.0f64, 0.0f64);
        let mut noise = 0.0f64;
        let mut pivotal = 0usize;
        // label -> (decisions, pivotal, summed room)
        let mut per: std::collections::HashMap<String, (usize, usize, f64)> =
            std::collections::HashMap::new();
        let (mut measured, mut adrift, mut spent) = (0usize, 0usize, 0usize);
        let mut since = self.rng.next_u64() as usize % every;

        for _ in 0..episodes {
            self.episode_seed += 1;
            let mut obs = self.env.reset(self.episode_seed);
            for t in 0..max_steps {
                let options = self.env.actions();
                if options.is_empty() {
                    break;
                }
                let Some(teacher) = self.env.demo() else {
                    return Ok(None);
                };
                let mark = std::time::Instant::now();
                let probs = self.policy(&obs, &options)?;
                spend.deciding += mark.elapsed();
                spend.decisions += 1;
                let (rows, wins) = self.model.last_shape();
                spend.tokens += rows;
                spend.windows += wins;

                since += 1;
                if since >= every && measured + adrift < want {
                    since = 0;
                    match self
                        .branch(
                            &options, &probs, teacher, alternatives, from, spec, t, &mut spend,
                        )?
                    {
                        Some((scored, tried)) => {
                            spent += scored.iter().map(|b| b.steps).sum::<usize>();
                            let theirs = scored[0].score;
                            let mut belief: Vec<f32> =
                                tried.iter().map(|&c| probs.get(c).copied().unwrap_or(0.0)).collect();
                            let mass: f32 = belief.iter().sum();
                            if mass > 0.0 {
                                for b in belief.iter_mut() {
                                    *b /= mass;
                                }
                            } else {
                                let flat = 1.0 / belief.len().max(1) as f32;
                                belief.fill(flat);
                            }
                            probes.push(Probe {
                                label: self.env.label(),
                                objective: self.env.objective(),
                                observation: obs.clone(),
                                options: options.clone(),
                                tried: tried
                                    .iter()
                                    .zip(&scored)
                                    .map(|(&c, b)| (c, b.score))
                                    .collect(),
                                belief,
                            });
                            let best =
                                scored.iter().map(|b| b.score).fold(f32::NEG_INFINITY, f32::max);
                            let worst =
                                scored.iter().map(|b| b.score).fold(f32::INFINITY, f32::min);
                            // A tolerance, because two trajectories that
                            // differ only in which way the player faced for
                            // one decision score the same to within rounding,
                            // and calling that an improvement counts noise.
                            if best > theirs + 1e-3 {
                                beaten += 1;
                                gain += (best - theirs) as f64;
                            }
                            regret += (best - theirs) as f64;
                            spread += (best - worst) as f64;
                            noise += scored.iter().map(|b| b.noise as f64).sum::<f64>()
                                / scored.len().max(1) as f64;
                            pivotal += usize::from(best - worst > 1e-3);
                            if let Some(l) = self.env.label() {
                                let e = per.entry(l).or_insert((0, 0, 0.0));
                                e.0 += 1;
                                e.1 += usize::from(best - worst > 1e-3);
                                e.2 += (best - theirs) as f64;
                            }
                            measured += 1;
                        }
                        None => adrift += 1,
                    }
                }

                let mut u = self.rng.next_f32();
                let mut chosen = probs.len() - 1;
                for (i, &pi) in probs.iter().enumerate() {
                    if u < pi {
                        chosen = i;
                        break;
                    }
                    u -= pi;
                }
                let mark = std::time::Instant::now();
                let (next, _, done) = self.env.step(chosen);
                spend.stepping += mark.elapsed();
                obs = next;
                if done {
                    break;
                }
            }
        }
        let mut by_situation: Vec<Situation> = per
            .into_iter()
            .map(|(label, (states, pivot, room))| Situation {
                label,
                states,
                pivotal: pivot as f32 / states.max(1) as f32,
                regret: (room / states.max(1) as f64) as f32,
            })
            .collect();
        by_situation.sort_by(|a, b| {
            b.regret.partial_cmp(&a.regret).unwrap_or(std::cmp::Ordering::Equal)
        });
        if measured == 0 {
            return Ok(Some((
                Counterfactual { adrift, steps: spent, by_situation, spend, ..Default::default() },
                probes,
            )));
        }
        let n = measured as f64;
        Ok(Some((Counterfactual {
            states: measured,
            adrift,
            beaten: (beaten as f64 / n) as f32,
            gain: (gain / beaten.max(1) as f64) as f32,
            regret: (regret / n) as f32,
            spread: (spread / n) as f32,
            noise: (noise / n) as f32,
            repeats: spec.repeats.max(1),
            pivotal: (pivotal as f64 / n) as f32,
            steps: spent,
            by_situation,
            spend,
        }, probes)))
    }

    /// Try each candidate action from where the environment is standing, and
    /// leave it standing exactly there.
    ///
    /// The teacher's own action is always first, because every number the
    /// caller computes is relative to it.
    fn branch(
        &mut self,
        options: &[String],
        probs: &[f32],
        teacher: usize,
        alternatives: usize,
        from: Candidates,
        spec: &ControlSpec,
        at: usize,
        spend: &mut Spend,
    ) -> Result<Option<(Vec<Branch>, Vec<usize>)>> {
        let max_steps = spec.max_steps;
        // How far past the branch point a candidate is scored. Both the
        // AggreVaTe and the LOLS bounds carry the horizon - `Qmax T log T`
        // in one, `sqrt(|A| T)` in the other - and this sample's T is 400
        // against the twenty-odd of a tagging or parsing task. A shorter
        // window is the standard bias-for-variance trade: it stops a
        // candidate's score being decided by what happened three hundred
        // decisions later, at the price of not seeing that far.
        let horizon = if spec.credit == 0 {
            max_steps
        } else {
            (at + spec.credit).min(max_steps)
        };
        let mut pick: Vec<usize> = (0..options.len()).collect();
        match from {
            Candidates::Contested => pick.sort_by(|&a, &b| {
                probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
            }),
            Candidates::Wide => {
                for i in (1..pick.len()).rev() {
                    let j = (self.rng.next_u64() % (i as u64 + 1)) as usize;
                    pick.swap(i, j);
                }
            }
        }
        let mut candidates = vec![teacher];
        for c in pick {
            if candidates.len() > alternatives {
                break;
            }
            if c != teacher {
                candidates.push(c);
            }
        }
        let mark = std::time::Instant::now();
        let held = self.env.hold();
        spend.holding += mark.elapsed();
        if !held {
            return Ok(None);
        }
        // COMMON RANDOM NUMBERS. Every candidate at this decision is rolled
        // out against the same stream of draws, so what separates two of them
        // is the action and not the dice.
        //
        // This is the technique `gauge` already uses, and for the identical
        // reason it gives: it does not reduce the variance of any one score,
        // it removes the variance from their DIFFERENCE, which is the only
        // quantity a counterfactual is asking about. Without it, two rollouts
        // of a four-hundred-decision episode diverge chaotically on their own
        // and the gap between candidates is mostly which draws each happened
        // to get. Measured without it on this sample: re-running ONE candidate
        // moved its own score by 0.073 against a gap between DIFFERENT
        // candidates of 0.048.
        //
        // Drawn once per decision, so different decisions still see different
        // worlds and the probe is not measuring one lucky stream.
        let stream = self.rng.next_u64();
        let outer_rng = self.rng.clone();
        // One draw for the WHOLE state, not one per candidate. LOLS draws per
        // candidate, which is unbiased in expectation over many examples; at
        // one roll-out each it would mean candidate A is scored under the
        // teacher's continuation and candidate B under the policy's, and the
        // difference between them would be mostly the difference between
        // those two continuations. Searn draws per state for the same reason.
        //
        // Stratified rather than drawn once there is more than one roll-out.
        // The engine is deterministic from a restored snapshot, so two
        // roll-outs that happen to draw the same way follow the same
        // trajectory and averaging them is one sample counted twice; splitting
        // them by beta exactly makes k roll-outs k DIFFERENT ones and takes
        // the sampling noise out of the mixture itself.
        let k = spec.repeats.max(1);
        let reference_rollout: Vec<bool> = if k == 1 {
            vec![self.rng.next_f32() < spec.beta]
        } else {
            (0..k).map(|r| (r as f32) < k as f32 * spec.beta).collect()
        };
        // HYPOTHETICAL, all of it: these roll-outs ask what WOULD have
        // happened. An environment that let any of it into what it carries
        // between episodes would be taught by decisions nobody took, and by
        // each of them once per candidate and once per repeat.
        let rolled = self.measuring(|p| {
            p.branch_rollouts(options, &candidates, &reference_rollout, stream, horizon, at, spend)
        })?;
        // The episode's own stream, picked up where the probe interrupted it.
        self.rng = outer_rng;
        // And back to where the student was, so the episode it is in the
        // middle of carries on as if none of this had happened - including
        // when the roll-outs were ABANDONED half way, which used to return
        // from here directly and leave the training episode standing wherever
        // the abandoned roll-out had got to, reading an observation from the
        // branch point and drawing from the branch's stream.
        let mark = std::time::Instant::now();
        let back = self.env.resume();
        spend.resuming += mark.elapsed();
        if back.is_none() {
            return Ok(None);
        }
        Ok(rolled.map(|scored| (scored, candidates)))
    }

    /// Run every trial [`branch_plan`] asks for and score the candidates.
    ///
    /// Separate from [`Self::branch`] so that all of it - including the paths
    /// that give up half way, which is what a probe does whenever the
    /// environment cannot go back - runs inside one measuring scope.
    #[allow(clippy::too_many_arguments)]
    fn branch_rollouts(
        &mut self,
        options: &[String],
        candidates: &[usize],
        reference_rollout: &[bool],
        stream: u64,
        horizon: usize,
        at: usize,
        spend: &mut Spend,
    ) -> Result<Option<Vec<Branch>>> {
        let mut total = vec![0.0f64; candidates.len()];
        let mut steps_all = vec![0usize; candidates.len()];
        // Kept apart by which policy rolled it out, so the spread within each
        // can be read without the other contaminating it.
        let mut by_rollout: Vec<[Vec<f32>; 2]> =
            (0..candidates.len()).map(|_| [Vec::new(), Vec::new()]).collect();
        for trial in branch_plan(candidates, reference_rollout) {
            // The same draws for every candidate, and a different set per
            // repeat so that repeats still measure something.
            self.rng = data::rng::Rng::new(stream ^ ((trial.repeat as u64 + 1) << 32));
            // Every trial, the first one included. The environment is already
            // standing where `hold` was called when the first one starts, so
            // this looks like an engine call for nothing - but restoring is
            // the thing the whole method rests on rather than a free identity,
            // and the check below is how it is known to have worked.
            let mark = std::time::Instant::now();
            let back = self.env.resume();
            spend.resuming += mark.elapsed();
            if back.is_none() {
                return Ok(None);
            }
            // Going back has to arrive where it left, and the options on
            // offer are the cheapest thing that says so - they are derived
            // from most of the state an observation reads. This check is the
            // reason the method is trustworthy: with the actions-replayed
            // version it failed three times in ten, and silently comparing
            // two candidates evaluated from different states is exactly the
            // kind of wrong that looks like a result.
            if self.env.actions() != options {
                return Ok(None);
            }
            let mark = std::time::Instant::now();
            let (mut obs, _, mut done) = self.env.step(trial.candidate);
            spend.stepping += mark.elapsed();
            let mut steps = 1usize;
            let mut t = at + 1;
            // The roll-out: the reference to the end of the window, or the
            // policy to the end of it, chosen with probability `--beta`.
            //
            // LOLS Table 1 is what this implements. Rolling out with the
            // reference alone leaves the learner blind to its own compounding
            // errors - it can be arbitrarily far from locally optimal, since
            // a good teacher undoes whatever one decision did and every
            // candidate then scores what the teacher scores. Rolling out with
            // the learner alone is the cell that paper marks "RL", which is
            // the hard problem this phase was supposed to avoid: it is what
            // this sample did, and it collapsed. The mixture is the cell
            // marked "Good", and beta = 0.5 is the value they report as
            // working and as not being sensitive.
            while !done && t < horizon {
                let options = self.env.actions();
                if options.is_empty() {
                    break;
                }
                let action = if trial.reference {
                    match self.env.demo() {
                        Some(a) => a,
                        None => break,
                    }
                } else {
                    let mark = std::time::Instant::now();
                    let probs = self.policy(&obs, &options)?;
                    spend.deciding += mark.elapsed();
                    spend.decisions += 1;
                    let (rows, wins) = self.model.last_shape();
                    spend.tokens += rows;
                    spend.windows += wins;
                    // SAMPLED, not the argmax. Two reasons, and they agree.
                    // The quantity wanted is the cost-to-go of the policy
                    // that will actually be run, and the policy that will
                    // actually be run samples - `score_policy` does, every
                    // rollout does. And in a deterministic engine an argmax
                    // roll-out is the same trajectory every time, so no
                    // number of repeats would tell us anything about how much
                    // of a candidate's score is the path rather than the
                    // action.
                    let mut u = self.rng.next_f32();
                    let mut chosen = probs.len() - 1;
                    for (idx, &pi) in probs.iter().enumerate() {
                        if u < pi {
                            chosen = idx;
                            break;
                        }
                        u -= pi;
                    }
                    chosen
                };
                let mark = std::time::Instant::now();
                let (next, _, d) = self.env.step(action);
                spend.stepping += mark.elapsed();
                obs = next;
                done = d;
                steps += 1;
                t += 1;
            }
            let mark = std::time::Instant::now();
            let scored = self.env.progress();
            spend.scoring += mark.elapsed();
            let Some(score) = scored else {
                return Ok(None);
            };
            total[trial.at] += score as f64;
            by_rollout[trial.at][usize::from(trial.reference)].push(score);
            steps_all[trial.at] += steps;
        }
        let repeats = reference_rollout.len().max(1) as f64;
        Ok(Some(
            (0..candidates.len())
                .map(|i| Branch {
                    // Averaged over the roll-outs, which is the whole point of
                    // having more than one: a single roll-out of a
                    // four-hundred-decision episode is one draw of a system
                    // where any decision changes everything after it, and its
                    // ordering of the candidates is partly a fact about that
                    // path rather than about this state.
                    score: (total[i] / repeats) as f32,
                    noise: by_rollout[i]
                        .iter()
                        .filter(|g| g.len() > 1)
                        .map(|g| {
                            g.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                                - g.iter().copied().fold(f32::INFINITY, f32::min)
                        })
                        .fold(0.0f32, f32::max),
                    steps: steps_all[i],
                })
                .collect(),
        ))
    }

    /// Can a value function predict how an episode ENDS from where it is?
    ///
    /// The question everything model-based downstream of it depends on, asked
    /// on its own before any of it is built. The score this sample is kept on
    /// is computable on a prefix, so the value of a state IS the final score
    /// reachable from it, and ranking actions needs no reward decomposition
    /// at all:
    ///
    /// ```text
    /// Q(s, a) = V(step(s, a))
    /// ```
    ///
    /// Which matters because of what it replaces. Estimating that same Q by
    /// rolling out to the horizon and reading the score costs a rollout per
    /// candidate and carries the variance of one path through a system where
    /// any decision changes everything after it - measured on this sample, a
    /// noise of 0.073 against a signal of 0.048. A critic is one forward pass
    /// and is deterministic; its error is approximation error, which more
    /// data reduces, rather than path noise, which it does not.
    ///
    /// So this reports the held-out error against the spread of the thing
    /// being predicted. Split BY EPISODE, never by step: consecutive states
    /// of one episode share almost everything including the answer, and a
    /// step-level split would let the training set memorise the test set's
    /// episodes and report an error that means nothing.
    fn probe_value(
        &mut self,
        episodes: usize,
        max_steps: usize,
        epochs: usize,
    ) -> Result<Option<ValueFit>> {
        // (features of every state, what that episode finally scored)
        let mut runs: Vec<(Vec<Vec<f32>>, f32)> = Vec::new();
        for _ in 0..episodes {
            self.episode_seed += 1;
            let mut obs = self.env.reset(self.episode_seed);
            let mut feats = Vec::new();
            for _ in 0..max_steps {
                let options = self.env.actions();
                if options.is_empty() {
                    break;
                }
                let (probs, feature) = self.policy_and_feature(&obs, &options)?;
                feats.push(feature);
                let mut u = self.rng.next_f32();
                let mut chosen = probs.len() - 1;
                for (i, &pi) in probs.iter().enumerate() {
                    if u < pi {
                        chosen = i;
                        break;
                    }
                    u -= pi;
                }
                let (next, _, done) = self.env.step(chosen);
                obs = next;
                if done {
                    break;
                }
            }
            let Some(scored) = self.env.progress() else {
                return Ok(None);
            };
            runs.push((feats, scored));
        }
        if runs.len() < 4 {
            return Ok(None);
        }
        let cut = (runs.len() * 4 / 5).max(1).min(runs.len() - 1);
        let (train, test) = runs.split_at(cut);
        let flat = |rs: &[(Vec<Vec<f32>>, f32)]| {
            let mut x = Vec::new();
            let mut y = Vec::new();
            for (feats, scored) in rs {
                for f in feats {
                    x.push(f.clone());
                    y.push(*scored);
                }
            }
            (x, y)
        };
        let (xtr, ytr) = flat(train);
        let (xte, yte) = flat(test);
        if xtr.is_empty() || xte.is_empty() {
            return Ok(None);
        }
        let d = xtr[0].len();
        let mut critic = Critic::new(d, CRITIC_HIDDEN, self.rng.next_u64());
        critic.fit(&xtr, &ytr, epochs, 0.02, 1e-5);
        let rmse = |x: &[Vec<f32>], y: &[f32]| {
            (x.iter()
                .zip(y)
                .map(|(f, &t)| {
                    let e = (critic.predict(f) - t) as f64;
                    e * e
                })
                .sum::<f64>()
                / x.len().max(1) as f64)
                .sqrt() as f32
        };
        let mean = yte.iter().sum::<f32>() / yte.len() as f32;
        let sd = (yte.iter().map(|&t| ((t - mean) as f64).powi(2)).sum::<f64>()
            / yte.len() as f64)
            .sqrt() as f32;
        Ok(Some(ValueFit {
            train_rmse: rmse(&xtr, &ytr),
            test_rmse: rmse(&xte, &yte),
            spread: sd,
            states: xtr.len() + xte.len(),
            episodes: runs.len(),
        }))
    }

    /// Read the archive of successful trajectories back off disk.
    ///
    /// Keyed by what the episode was played ON, so the archive holds the best
    /// run of every LEVEL rather than the best runs overall. Without the key
    /// a greedy top-N fills with copies of whichever level is easiest and
    /// deletes the only trajectory that ever solved a hard one - which is the
    /// stepping stone this loop exists to keep.
    fn read_archive(path: &str) -> Vec<Solved> {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Vec::new();
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
            return Vec::new();
        };
        // Written by hand rather than derived, because the SDK's serde
        // derive is behind a feature this surface does not enable and an
        // archive format is not worth widening a dependency for.
        let demo = |d: &serde_json::Value| -> Option<Demo> {
            Some(Demo {
                objective: d.get("objective")?.as_str()?.to_string(),
                observation: d.get("observation")?.as_str()?.to_string(),
                options: d
                    .get("options")?
                    .as_array()?
                    .iter()
                    .filter_map(|o| Some(o.as_str()?.to_string()))
                    .collect(),
                action: d.get("action")?.as_u64()? as usize,
            })
        };
        v.as_array()
            .map(|rows| {
                rows.iter()
                    .filter_map(|r| {
                        // An archive written before a world had an identity of
                        // its own names only the kind, and that kind IS the
                        // instance for a level with fixed geometry, which is
                        // all such an archive can hold.
                        let kind = r.get("kind").or_else(|| r.get("label"))?.as_str()?.to_string();
                        let instance = match r.get("instance").and_then(|i| i.as_str()) {
                            Some(i) => i.to_string(),
                            None => kind.clone(),
                        };
                        Some(Solved {
                            kind,
                            instance,
                            score: r.get("score")?.as_f64()? as f32,
                            demos: r.get("demos")?.as_array()?.iter().filter_map(demo).collect(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn write_archive(path: &str, kept: &Archive) {
        if let Some(dir) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        // Written beside and renamed, so an interrupted write cannot leave a
        // half-file where the only record of a solved level used to be.
        let tmp = format!("{path}.partial");
        let rows: Vec<serde_json::Value> = kept
            .by_instance
            .values()
            .map(|s| {
                serde_json::json!({
                    "kind": s.kind,
                    "instance": s.instance,
                    "score": s.score,
                    "demos": s.demos
                        .iter()
                        .map(|d| serde_json::json!({
                            "objective": d.objective,
                            "observation": d.observation,
                            "options": d.options,
                            "action": d.action,
                        }))
                        .collect::<Vec<_>>(),
                })
            })
            .collect();
        if serde_json::to_string(&rows)
            .ok()
            .and_then(|t| std::fs::write(&tmp, t).ok())
            .is_some()
        {
            let _ = std::fs::rename(&tmp, path);
        }
    }

    /// Search for successful trajectories by returning to places it has been.
    ///
    /// The half of a search-and-compress loop that this sample was missing.
    /// Sampling from the policy's own distribution can only find trajectories
    /// a short distance from what it already does - measured here, after one
    /// round nine episodes in ten beat nothing at all - so the archive stops
    /// filling and the compression has nothing new to compress.
    ///
    /// The fix is not more sampling, it is resuming. Keep an archive of the
    /// places the run has been, go back to one, and explore from THERE. The
    /// depth reachable then stops being exponential in the length of an
    /// episode, because a strategy four hundred decisions deep is reached by
    /// four hundred decisions of archive plus a handful of new ones rather
    /// than by four hundred lucky draws in a row. This is Go-Explore, and its
    /// expensive requirement - a simulator that can be restored to any state
    /// - is the one thing already built here.
    ///
    /// Two properties are worth being explicit about.
    ///
    /// **It does not use the policy.** Exploration is random over the options
    /// the game offers, which costs an engine step and no forward pass:
    /// measured on this sample, about 4 ms against 100 ms with a network in
    /// the loop. The policy's job is the other half, compressing what this
    /// finds.
    ///
    /// **It is told nothing about the game.** A cell is where the run is and
    /// what it carries - see [`Env::cell`] - so picking up a key makes every
    /// place reachable with it new, and the search files those and goes on.
    /// Nothing says that keys open doors.
    fn explore(&mut self, spec: &ControlSpec) -> Result<Vec<Solved>> {
        let slots = self.env.slots();
        if slots == 0 || spec.explore == 0 {
            return Ok(Vec::new());
        }
        let mut best: Option<(f32, Vec<Demo>)> = None;
        // cell -> the best trail that reached it this round
        let mut opened: std::collections::HashMap<String, (f32, Vec<Demo>)> =
            std::collections::HashMap::new();
        let (mut steps, mut restores) = (0usize, 0usize);
        // Cells whose slot was taken to hold a better one. See below.
        let mut evicted = 0usize;
        let mut obs = String::new();

        // THE ARCHIVE OUTLIVES THE ROUND.
        //
        // This is the whole mechanism, and it was missing. The cells lived in
        // a local, so every round reset the level, searched from the spawn
        // with an empty archive, found its twenty-odd cells and threw them
        // away. Four rounds of that is the first round four times: measured
        // on E1M1, 27 cells then 26 then 30 then 25, and nothing the search
        // produced ever beat what the policy already had.
        //
        // Go-Explore compounds because the archive grows. A cell reached in
        // one round is somewhere the NEXT round can set off from, so the
        // frontier moves outward instead of being rediscovered. The held
        // states survive a reset - they are snapshots of a level, and the
        // level is the same one - so the only thing that had to change is
        // where the map lives.
        let here = self.env.instance().or_else(|| self.env.label());
        let carried = match self.explored.take() {
            Some((was, cells, slot)) if Some(&was) == here.as_ref() && !cells.is_empty() => {
                Some((was, cells, slot))
            }
            // A different world, or nothing yet. Whatever was held belongs to
            // a level that is no longer standing.
            _ => None,
        };
        let (kind, instance, mut seen, mut next_slot) = match carried {
            Some((was, cells, slot)) => {
                let kind = self.env.label().unwrap_or_else(|| was.clone());
                (kind, was, cells, slot)
            }
            None => {
                self.episode_seed += 1;
                obs = self.env.reset(self.episode_seed);
                let kind = self.env.label().unwrap_or_else(|| "world".into());
                let instance = self.env.instance().unwrap_or_else(|| kind.clone());
                let mut seen: std::collections::HashMap<String, (usize, f32, u32)> =
                    std::collections::HashMap::new();
                let mut next_slot = 1usize;
                // The starting point, so the archive is never empty and the
                // first pick has somewhere to go.
                if let Some(cell) = self.env.cell() {
                    if self.env.hold_at(next_slot) {
                        seen.insert(cell, (next_slot, self.env.progress().unwrap_or(0.0), 0));
                        next_slot += 1;
                    }
                }
                (kind, instance, seen, next_slot)
            }
        };
        let carried_cells = seen.len();
        // Nowhere to return TO. An environment that cannot name a cell or
        // cannot hold one has no archive, and the selection below would index
        // an empty list.
        if seen.is_empty() {
            println!(
                "    explore: {instance} could not hold its starting state, so there is \
                 nothing to return to"
            );
            return Ok(Vec::new());
        }
        let mut refused = 0usize;

        for _ in 0..spec.explore {
            let mut trail: Vec<Demo> = Vec::new();

            // Go back to a cell chosen with probability proportional to
            //
            //     W = 1 / sqrt(C_seen + 1)
            //
            // which is the selection weight Go-Explore reports (Ecoffet et
            // al., "First return, then explore", Extended Data Table 1). A
            // cell seen once is worth about seven times one seen fifty times,
            // so the frontier is favoured without the rest of the archive
            // ever being cut off - which is the detachment the archive exists
            // to prevent. Sampling from the distribution rather than taking
            // the N least-seen matters for the same reason.
            let pick = {
                // The frontier, by two measures at once.
                //
                // `1 / sqrt(seen + 1)` is Go-Explore's own weight and favours
                // what has rarely been set off from, which is what stops the
                // search settling into one corner. On its own it is blind to
                // how much a cell has to offer: a spot at the level's front
                // door with nothing done is drawn exactly as often as one
                // deep in with most of the level cleared, and only the second
                // can lead anywhere new.
                //
                // So the score the cell was reached with multiplies it. A
                // cell twice as far along is worth twice as many attempts,
                // and the +1 keeps a cell that has achieved nothing yet in
                // the draw rather than cutting it off - that is where the
                // search has to start.
                let top = seen
                    .values()
                    .map(|(_, v, _)| *v)
                    .fold(f32::MIN_POSITIVE, f32::max) as f64;
                let weights: Vec<(&String, f64)> = seen
                    .iter()
                    .map(|(c, (_, v, n))| {
                        let worth = 1.0 + (*v as f64 / top).clamp(0.0, 1.0);
                        (c, worth / ((*n as f64) + 1.0).sqrt())
                    })
                    .collect();
                let total: f64 = weights.iter().map(|(_, w)| *w).sum();
                let mut u = self.rng.next_f32() as f64 * total;
                let mut chosen = weights[weights.len() - 1].0;
                for (c, w) in &weights {
                    if u < *w {
                        chosen = c;
                        break;
                    }
                    u -= *w;
                }
                chosen.clone()
            };
            let slot = match seen.get_mut(&pick) {
                Some((slot, _, n)) => {
                    *n += 1;
                    *slot
                }
                None => continue,
            };
            match self.env.resume_from(slot) {
                Some(o) => {
                    obs = o;
                    restores += 1;
                }
                // A refusal is reported, never papered over by starting the
                // level again - a search that quietly restarts is a search
                // that explores the opening a thousand times and says it
                // resumed.
                None => {
                    refused += 1;
                    continue;
                }
            }

            // What was done last, so it can be done again. See below.
            let mut last: Option<String> = None;
            for _ in 0..spec.explore_steps {
                let options = self.env.actions();
                if options.is_empty() {
                    break;
                }
                // KEEP DOING THE SAME THING, 95% of the time.
                //
                // "To help explore in a consistent direction, the probability
                // of repeating the previous action is 95% for Atari and 90%
                // for robotics" - Go-Explore, Methods. Without it, uniform
                // random per step is a random walk: it covers distance like
                // the square root of the steps taken, so a hundred steps go
                // almost nowhere and the archive stops growing. Measured here
                // before this was added, twenty-four times the search budget
                // bought one and a third times the cells.
                //
                // An option list that is rebuilt every step has no stable
                // action INDEX, so "the same action" is matched on the text -
                // "walk forward, 320 units" and "walk forward, 288 units" are
                // the same intention with a different number in it.
                // Sometimes what a competent player would do, mostly what
                // nobody would.
                //
                // Uniformly random is blind, and blind in a level full of
                // things that shoot back is mostly dead: measured, an eighty
                // step random walk scores 0.03 where the policy scores 0.18,
                // so the search spent its whole budget on continuations no
                // trajectory worth keeping would ever contain. Taking the
                // scripted player's choice some of the time makes the walk
                // start from somewhere plausible and wander off it, which is
                // the useful shape - and it costs a branch rather than a
                // forward pass, which is why this half of the loop is cheap
                // enough to run at all.
                //
                // Not always, or the search only ever finds what the teacher
                // finds, and the teacher has never finished a level.
                let guided = self.rng.next_f32() < GUIDED_CHANCE;
                let a = match self.env.demo().filter(|_| guided) {
                    Some(i) if i < options.len() => i,
                    _ => match last.as_deref().filter(|_| self.rng.next_f32() < REPEAT_CHANCE) {
                        Some(prev) => same_again(prev, &options)
                            .unwrap_or_else(|| (self.rng.next_u64() as usize) % options.len()),
                        None => (self.rng.next_u64() as usize) % options.len(),
                    },
                };
                last = Some(options[a].clone());
                trail.push(Demo { objective: self.env.objective(), observation: obs.clone(), options, action: a });
                let (next, _, done) = self.env.step(a);
                obs = next;
                steps += 1;
                if done {
                    break;
                }
                let Some(cell) = self.env.cell() else { continue };
                let scored = self.env.progress().unwrap_or(0.0);
                let fresh = !seen.contains_key(&cell);
                let better = seen.get(&cell).is_some_and(|(_, v, _)| scored > *v);
                if fresh || better {
                    // Where to hold it. A cell already in the archive keeps
                    // its own slot; a new one takes the next free slot, and
                    // when there are none left it takes the slot of whichever
                    // cell the selection rule is least likely to draw.
                    //
                    // Without that last part the archive silently stopped
                    // growing the moment the engine ran out of places to put
                    // things, and a search whose archive cannot grow is a
                    // search that has finished. It went unnoticed because
                    // nothing said so: the cells kept being FOUND and simply
                    // were not kept.
                    let slot = match seen.get(&cell) {
                        Some((s, _, _)) if better => *s,
                        _ if next_slot < slots => {
                            let s = next_slot;
                            next_slot += 1;
                            s
                        }
                        _ => {
                            let top = seen
                                .values()
                                .map(|(_, v, _)| *v)
                                .fold(f32::MIN_POSITIVE, f32::max);
                            let weakest = seen
                                .iter()
                                .filter(|(c, _)| *c != &cell)
                                .map(|(c, (s, v, n))| {
                                    let worth = 1.0 + (v / top).clamp(0.0, 1.0);
                                    (c.clone(), *s, worth / ((*n as f32) + 1.0).sqrt())
                                })
                                .min_by(|a, b| {
                                    a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)
                                });
                            match weakest {
                                Some((c, s, _)) => {
                                    seen.remove(&c);
                                    evicted += 1;
                                    s
                                }
                                None => continue,
                            }
                        }
                    };
                    if self.env.hold_at(slot) {
                        let visits = seen.get(&cell).map(|(_, _, n)| *n).unwrap_or(0);
                        seen.insert(cell.clone(), (slot, scored, visits));
                        // THE WAY THERE, not the run it happened during.
                        //
                        // What a search produces is ground reached, and what
                        // is worth keeping is the best way to reach each
                        // piece of it. Judged as a whole episode instead, an
                        // eighty step random walk is never going to beat a
                        // trained policy's own run - measured on E1M1, the
                        // search scored 0.02 against the policy's 0.18 every
                        // round, so nothing it found was ever kept and the
                        // whole half of the loop did nothing.
                        opened.insert(cell.clone(), (scored, trail.clone()));
                    }
                }
                if best.as_ref().is_none_or(|(v, _)| scored > *v) {
                    best = Some((scored, trail.clone()));
                }
            }
        }
        if refused > 0 {
            println!("    explore: {refused} resumes were REFUSED - the archive is not restorable");
        }
        println!(
            "    explore: {} cells in {instance} ({carried_cells} carried in, \
             {evicted} evicted) over {steps} steps, {restores} resumed from, best {}",
            seen.len(),
            match &best {
                Some((v, _)) => format!("{v:.2}"),
                None => "nothing".to_string(),
            }
        );
        // Hand the archive on. Next round sets off from the frontier this
        // one reached instead of from the level's front door.
        self.explored = Some((instance.clone(), seen, next_slot));
        // One per cell opened, each keyed by the place it reaches, so the
        // archive keeps the best way to each rather than the best single
        // episode. The whole-episode best is in there too, under the world's
        // own name, so a search that genuinely out-plays the policy still
        // counts as that.
        let mut out: Vec<Solved> = opened
            .into_iter()
            .map(|(cell, (score, demos))| Solved {
                kind: instance.clone(),
                instance: format!("{instance}@{cell}"),
                score,
                demos,
            })
            .collect();
        if let Some((v, d)) = best {
            out.push(Solved { kind, instance, score: v, demos: d });
        }
        Ok(out)
    }

    /// Rounds of: play, keep the episodes that actually scored best, and
    /// clone those.
    ///
    /// The cheapest improvement OPERATOR this environment admits, and the one
    /// whose signal is largest. A policy's own episodes on the same worlds
    /// score anywhere from 0.20 to 1.20 here; the gap between the best and
    /// worst ACTION at a single decision is 0.027, against a re-run noise of
    /// about the same. Credit assignment at the episode is working with a
    /// signal some thirty times larger than credit assignment at the
    /// decision, for a tenth of the game steps - no snapshots, no branches,
    /// no cost-to-go estimate to be wrong about.
    ///
    /// It ratchets because the elite set only ever improves: each round adds
    /// its own episodes, re-sorts, and keeps the best of everything seen. The
    /// policy is therefore always being fitted to behaviour better than its
    /// own average, which is the one property that makes a loop like this
    /// climb rather than drift. It is bounded above by the best the policy
    /// can stumble into, so exploration is what eventually limits it - not
    /// the teacher, which is what makes this the first phase here whose
    /// ceiling is not the teacher AND whose signal is not inside its own
    /// noise.
    ///
    /// Scored on [`Env::progress`] rather than on return, so that what an
    /// episode is KEPT for is exactly what the run is judged on.
    fn self_imitate(
        &mut self,
        spec: &ControlSpec,
        log: &mut dyn FnMut(usize, f32),
        step: &mut usize,
    ) -> Result<()> {
        let entered = self.gauge(spec.gauge_episodes, spec.max_steps)?;
        let mut keep_best =
            Keep::starting(entered.unwrap_or(f32::NEG_INFINITY), self.model.head_weights());
        let mut before = entered;
        // Everything that has ever been worth keeping, on this run and on
        // every run before it. See [`Archive`].
        let mut best = Archive::default();
        if let Some(path) = &spec.archive {
            for s in Self::read_archive(path) {
                best.offer(&s.kind, &s.instance, s.score, s.demos);
            }
            if best.len() > 0 {
                let had: Vec<String> = best
                    .by_kind()
                    .iter()
                    .map(|(k, v, n)| match n {
                        1 => format!("{k} {v:.2}"),
                        n => format!("{k} {v:.2} (best of {n} worlds)"),
                    })
                    .collect();
                println!(
                    "    archive: carrying {} solved from before - {}",
                    best.len(),
                    had.join(", ")
                );
            }
        }
        for round in 0..spec.self_imitate {
            let mut fresh: Vec<Solved> = Vec::new();
            for _ in 0..spec.episodes {
                self.episode_seed += 1;
                let mut obs = self.env.reset(self.episode_seed);
                let mut demos = Vec::new();
                for _ in 0..spec.max_steps {
                    let options = self.env.actions();
                    if options.is_empty() {
                        break;
                    }
                    let probs = self.policy(&obs, &options)?;
                    // SAMPLED. The exploration this phase lives on is the
                    // policy's own spread over the options: a greedy rollout
                    // would produce one trajectory per world and there would
                    // be no best of anything to keep.
                    let mut u = self.rng.next_f32();
                    let mut chosen = probs.len() - 1;
                    for (i, &pi) in probs.iter().enumerate() {
                        if u < pi {
                            chosen = i;
                            break;
                        }
                        u -= pi;
                    }
                    demos.push(Demo { objective: self.env.objective(), observation: obs.clone(), options, action: chosen });
                    let (next, _, done) = self.env.step(chosen);
                    obs = next;
                    if done {
                        break;
                    }
                }
                let Some(scored) = self.env.progress() else {
                    println!("    self-imitation: the environment does not score an episode");
                    return Ok(());
                };
                // WHICH world it was played on, and what kind of world
                // that is. The archive keeps a run per world and caps how
                // many worlds of a kind it keeps. Environments with nothing
                // to say fall into one bucket, which is the old behaviour.
                let kind = self.env.label().unwrap_or_else(|| "world".into());
                let instance = self.env.instance().unwrap_or_else(|| kind.clone());
                fresh.push(Solved { kind, instance, score: scored, demos });
            }
            // The search runs first, so what it finds is in the archive
            // before this round's fit reads it.
            fresh.extend(self.explore(spec)?);
            let played: Vec<f32> = fresh.iter().map(|r| r.score).collect();
            let mean = played.iter().sum::<f32>() / played.len().max(1) as f32;
            let top = played.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            // Each world keeps its own best run, so finding a way through
            // a hard one is never undone by finding a better way through an
            // easy one.
            let mut gained = 0usize;
            for s in fresh {
                if best.offer(&s.kind, &s.instance, s.score, s.demos) {
                    gained += 1;
                }
            }
            if let Some(path) = &spec.archive {
                Self::write_archive(path, &best);
            }
            let bar = best.worst();
            // The bar is what the policy manages on its own. Anything in the
            // archive that clears it is worth copying; anything below it is
            // the policy's own average handed back to it.
            let demos: Vec<Demo> = best.demos_above(mean);
            let loss = self.fit_demos(&demos, spec.warmup_epochs, log, step)?;
            let after = self.gauge(spec.gauge_episodes, spec.max_steps)?;
            println!(
                "    self-imitate {:>2}  played {} episodes (mean {mean:.2}, best {top:.2})  \
                 archive holds {} worlds, worst {bar:.2} ({gained} improved)  {} decisions \
                 cloned  loss {loss:.4}{}",
                round + 1,
                spec.episodes,
                best.len(),
                demos.len(),
                match (before, after) {
                    (Some(b), Some(a)) => format!("  fixed block {b:.3} -> {a:.3}"),
                    _ => String::new(),
                }
            );
            keep_best.offer(round + 1, after.unwrap_or(-loss), || self.model.head_weights());
            before = after;
        }
        let (round, w) = keep_best.best();
        if round != spec.self_imitate {
            println!(
                "    keeping {}, which scored {:.3}",
                match round {
                    0 => "the policy the imitation phases produced".to_string(),
                    n => format!("self-imitation round {n}"),
                },
                keep_best.rank
            );
            let w = w.to_vec();
            self.model.set_head_weights(&w);
        }
        Ok(())
    }

    /// How much of the policy's belief sits on the candidate that actually
    /// measured best, over the probes where the choice made a difference.
    ///
    /// Forward only. Read on probes the fit did NOT see, before and after,
    /// this is the number that says whether an outcome-fitted update learned
    /// anything about the decision or only about the decisions it was shown.
    fn outcome_belief(&mut self, probes: &[Probe]) -> Result<Option<(f32, usize)>> {
        let (mut got, mut n) = (0.0f64, 0usize);
        for p in probes {
            let scored: Vec<f32> = p.tried.iter().map(|(_, s)| *s).collect();
            let Some(best) = measured_best(&scored) else {
                continue;
            };
            let q = self.question_for(&p.objective, &p.options);
            let scores = self
                .model
                .score(&p.observation, std::slice::from_ref(&q))
                .map_err(Error::Backend)?;
            let picked: Vec<usize> = p
                .tried
                .iter()
                .map(|&(c, _)| c)
                .filter(|&c| c < scores[0].len())
                .collect();
            let among: Vec<f32> = picked.iter().map(|&c| scores[0][c]).collect();
            let mine = decide::loss::softmax(&among);
            got += mine.get(best).copied().unwrap_or(0.0) as f64;
            n += 1;
        }
        Ok((n > 0).then(|| ((got / n as f64) as f32, n)))
    }

    /// Move the policy toward whichever candidate actually scored better.
    ///
    /// The update every other phase of this run cannot do. Cloning and DAgger
    /// fit the teacher's CHOICE, so their ceiling is the teacher. This fits
    /// the measured OUTCOME, so its ceiling is whatever the candidate set
    /// contains - which is why it is the only phase whose result is not
    /// bounded above by the teacher.
    ///
    /// The reduction is cost-sensitive multiclass classification, which is
    /// what AggreVaTe and LOLS both reduce to - see [`outcome_costs`] for the
    /// cost vector and [`outcome_pull`] for the loss and its gradient. Only
    /// the candidates that were actually measured are moved: an option nobody
    /// tried gets no opinion pushed onto it, which is correct, since nothing
    /// here knows what it was worth.
    ///
    /// `epochs` is passed rather than read from the spec because this fits a
    /// dataset that GROWS - see [`Self::improve`] - and a fixed pass count
    /// over a growing set makes the last round cost several times the first
    /// for no reason.
    fn fit_outcomes(
        &mut self,
        probes: &[Probe],
        epochs: usize,
        spec: &ControlSpec,
        log: &mut dyn FnMut(usize, f32),
        step: &mut usize,
    ) -> Result<(f32, f32, f32, usize)> {
        let head_lr = spec.head_lr;
        if probes.is_empty() {
            return Ok((0.0, 0.0, 0.0, 0));
        }
        let costs: Vec<Vec<f32>> = probes
            .iter()
            .map(|p| outcome_costs(&p.tried.iter().map(|(_, s)| *s).collect::<Vec<_>>()))
            .collect();

        let mut loss = 0.0f32;
        let mut moved = 0.0f64;
        // How far this fit has carried the policy from the one that collected
        // the probes, and whether that is what stopped it. See
        // [`candidate_drift`].
        let mut drift = 0.0f32;
        let mut passes = 0usize;
        let mut spent = false;
        let mut order: Vec<usize> = (0..probes.len()).collect();
        for _ in 0..epochs.max(1) {
            if spent {
                break;
            }
            for i in (1..order.len()).rev() {
                let j = (self.rng.next_u64() % (i as u64 + 1)) as usize;
                order.swap(i, j);
            }
            let mut epoch_loss = 0.0f32;
            moved = 0.0;
            let mut travelled = 0.0f64;
            let mut counted = 0usize;
            for chunk in order.chunks(MINIBATCH) {
                if spent {
                    break;
                }
                self.model.zero_grads();
                for &i in chunk {
                    let p = &probes[i];
                    let c = &costs[i];
                    let q = self.question_for(&p.objective, &p.options);
                    let mut shift = 0.0f32;
                    let mut here = 0.0f32;
                    epoch_loss += self
                        .model
                        .accumulate(&p.observation, &q, |scores| {
                            let mut grad = vec![0.0f32; scores.len()];
                            // A softmax over the CANDIDATE logits alone. The
                            // decision is which of the options that were
                            // actually measured is best; the ones nobody tried
                            // are not evidence either way and must not be
                            // moved by a claim nothing supports.
                            let picked: Vec<usize> = p
                                .tried
                                .iter()
                                .map(|&(c, _)| c)
                                .filter(|&c| c < scores.len())
                                .collect();
                            let among: Vec<f32> = picked.iter().map(|&c| scores[c]).collect();
                            let mine = decide::loss::softmax(&among);
                            let (l, pull) = outcome_pull(&mine, c);
                            for (k, &col) in picked.iter().enumerate() {
                                grad[col] = pull.get(k).copied().unwrap_or(0.0);
                            }
                            // How much of the policy's belief among the
                            // candidates sits on the one that actually
                            // measured best - the number this update exists
                            // to raise.
                            if let Some(k) = measured_best(&p.tried.iter().map(|(_, s)| *s).collect::<Vec<_>>()) {
                                shift = mine.get(k).copied().unwrap_or(0.0);
                            }
                            // Measured on the SAME forward the update is
                            // computed from, so it costs nothing beyond the
                            // pass that was happening anyway.
                            here = candidate_drift(&p.belief, &mine);
                            (l, grad)
                        })
                        .map_err(Error::Backend)?;
                    moved += shift as f64;
                    travelled += here as f64;
                    counted += 1;
                    *step += 1;
                }
                // BEFORE the step this minibatch would take, for the reason
                // the PPO loop puts its own guard there: a check that can
                // only fire after a whole pass is a report, not a guard.
                drift = (travelled / counted.max(1) as f64) as f32;
                if spec.policy.target_kl > 0.0 && drift > spec.policy.target_kl {
                    spent = true;
                    break;
                }
                self.model.adamw_scaled(ENCODER_LR, head_lr, 1.0 / chunk.len() as f32);
            }
            passes += 1;
            loss = epoch_loss / probes.len() as f32;
            log(*step, loss);
        }
        Ok((loss, (moved / probes.len() as f64) as f32, drift, passes))
    }

    /// Rounds of: probe the policy's own decisions, find out what the
    /// alternatives were actually worth, and move toward whichever won.
    ///
    /// Reports what it could have gained beside what it did, because the two
    /// together are the only honest way to read a round. The counterfactual
    /// already measures the room available - "picking the best of what was
    /// offered would gain this much a decision" - so a round that gains
    /// nothing against a measured room of zero is a round with nothing to do,
    /// and a round that gains nothing against a room of 0.012 is a broken
    /// update. Those two look identical in a score.
    fn improve(
        &mut self,
        spec: &ControlSpec,
        log: &mut dyn FnMut(usize, f32),
        step: &mut usize,
    ) -> Result<()> {
        // What the imitation phases produced, and what it scores. Every round
        // below has to beat this to be adopted, or the phase hands it back.
        let entered = self.gauge(spec.gauge_episodes, spec.max_steps)?;
        let mut keep = Keep::starting(entered.unwrap_or(f32::NEG_INFINITY), self.model.head_weights());
        // Carried rather than re-measured. A round's score going IN is the one
        // the round before it left behind - same policy, same worlds, same
        // action stream - so gauging it again spends a block of episodes to
        // re-derive a number that cannot have moved.
        let mut before = entered;
        // Every probe this phase has ever taken, fitted together. AggreVaTe
        // and NRPI both do this - `D <- D u D_i`, then train on all of D -
        // and it is not incidental: the guarantee is a reduction to NO-REGRET
        // online learning, and the learner they name is Follow-The-Leader
        // over the aggregate. Fitting only the newest round is Follow-the-
        // LAST-Leader, which is the textbook algorithm with linear regret.
        //
        // Measured here before it was fixed: six rounds fitted separately
        // walked the fixed block from 0.780 down to 0.428, while the DAgger
        // phase in the same run - same code, same environment, aggregating -
        // climbed 0.633 to 0.770.
        let mut aggregate: Vec<Probe> = Vec::new();
        let mut held: Vec<Probe> = Vec::new();
        let mut unit = 0usize;
        for round in 0..spec.improve {
            let probed = self.probe(
                spec.episodes,
                spec.states,
                spec.alternatives,
                if spec.wide { Candidates::Wide } else { Candidates::Contested },
                spec,
            )?;
            let Some((room, mut probes)) = probed else {
                println!("    improve: the environment cannot go back to a decision");
                return Ok(());
            };
            if probes.is_empty() {
                println!("    improve: no decision could be probed");
                return Ok(());
            }
            // A quarter of each round held back, for the only question that
            // decides whether this phase is worth running at all: does
            // fitting the measured outcome at some decisions change what the
            // policy believes at decisions it was not shown? See
            // [`Self::outcome_belief`]. Shuffled first, because probes arrive
            // in episode order and the last quarter of a run is not a sample
            // of it. Held-out probes are held out for good - a probe that
            // joined the training set on a later round would quietly turn
            // this into a measurement of the fitted set.
            for i in (1..probes.len()).rev() {
                let j = (self.rng.next_u64() % (i as u64 + 1)) as usize;
                probes.swap(i, j);
            }
            held.extend(probes.split_off(probes.len() - probes.len() / 4));
            aggregate.extend(probes);
            if unit == 0 {
                unit = aggregate.len();
            }
            // A constant amount of OPTIMIZATION per round rather than a
            // constant number of passes, exactly as the DAgger phase does:
            // the aggregate grows by one round every round, so a fixed pass
            // count makes the sixth round cost six times the first to revisit
            // data the head is already fitted to.
            let epochs = (spec.warmup_epochs * unit / aggregate.len().max(1)).max(1);
            let unseen_before = self.outcome_belief(&held)?;
            let (loss, on_best, drift, passes) =
                self.fit_outcomes(&aggregate, epochs, spec, log, step)?;
            let unseen_after = self.outcome_belief(&held)?;
            let after = self.gauge(spec.gauge_episodes, spec.max_steps)?;
            println!(
                "    improve {:>2}  {} probed ({} in all, {epochs} passes, {} game steps)  \
                 {:.0}% mattered  room {:.3}/decision  loss {loss:.4}  the best-measured \
                 option now holds {:.2} of the policy's belief{}",
                round + 1,
                room.states,
                aggregate.len(),
                room.steps,
                room.pivotal * 100.0,
                room.regret,
                on_best,
                match (before, after) {
                    (Some(b), Some(a)) => format!("  fixed block {b:.3} -> {a:.3}"),
                    _ => String::new(),
                }
            );
            // The pair, in the order the `fit` command reports its own: what
            // the update did where it was applied, then what it did where it
            // was not. Only the second one can tell them apart.
            if let (Some((b, n)), Some((a, _))) = (unseen_before, unseen_after) {
                println!(
                    "             on {n} probes it did not fit, the best-measured option went \
                     from {b:.2} to {a:.2} of the policy's belief"
                );
            }
            // Said out loud when the trust region is what ended the fit,
            // because "it stopped early" and "it ran out of passes" are
            // different states of the world and only one of them means the
            // budget is the thing to change.
            if passes < epochs {
                println!(
                    "             stopped after {passes} of {epochs} passes: the policy had \
                     moved {drift:.4} from the one that measured the probes, against a budget \
                     of {:.4}",
                    spec.policy.target_kl
                );
            }
            keep.offer(round + 1, after.unwrap_or(-loss), || self.model.head_weights());
            before = after;
        }
        let (round, w) = keep.best();
        if round != spec.improve {
            println!(
                "    keeping {}, which scored {:.3}",
                match round {
                    0 => "the policy the imitation phases produced".to_string(),
                    n => format!("improve round {n}"),
                },
                keep.rank
            );
            let w = w.to_vec();
            self.model.set_head_weights(&w);
        }
        Ok(())
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
        self.measuring(|p| -> Result<()> {
            for i in 0..n {
                p.episode_seed += 1;
                let seed = p.episode_seed;
                println!("\n  episode {} (seed {seed})", i + 1);
                let (_, ret, won) = p.episode(seed, true, max_steps, true)?;
                println!("       = return {ret:+.2}, {}", if won { "WON" } else { "lost" });
                total += ret;
                wins += usize::from(won);
            }
            Ok(())
        })?;
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
#[derive(Clone, Debug)]
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
    /// Where the archive of successful trajectories lives between runs.
    ///
    /// The durable artifact of a search-and-compress loop is the ARCHIVE, not
    /// the weights: the weights are a lossy compression of it that can be
    /// rebuilt, and a trajectory that solved a level is evidence that cannot
    /// be. Held only in memory, it is erased whenever the process ends, so
    /// every generation starts its search from nothing and a level solved in
    /// one generation can be silently lost in the next.
    pub archive: Option<String>,
    /// Exploring episodes run before each round of self-imitation. `0` is off.
    ///
    /// See [`ControlPipeline::explore`]. This is the SEARCH half: it costs no
    /// forward passes and its job is to put trajectories in the archive that
    /// the policy could not have found by sampling itself.
    pub explore: usize,
    /// Steps taken by each exploring episode after it resumes.
    pub explore_steps: usize,
    /// Rounds of playing, keeping the best episodes and cloning those. `0`
    /// is off.
    ///
    /// See [`ControlPipeline::self_imitate`]. The improvement operator whose
    /// signal is the episode rather than the decision, which on this sample
    /// is some thirty times larger and an order of magnitude cheaper to
    /// collect.
    pub self_imitate: usize,
    /// Rounds of outcome-fitted improvement after the imitation phases. `0`
    /// is off.
    ///
    /// The only phase of a run whose ceiling is not the teacher. See
    /// [`ControlPipeline::improve`].
    pub improve: usize,
    /// Decisions probed per improvement round.
    pub states: usize,
    /// Alternatives tried at each probed decision, beside the teacher's own.
    pub alternatives: usize,
    /// How often a probe's roll-out is the TEACHER rather than the policy,
    /// drawn once per probed decision.
    ///
    /// LOLS's mixing parameter, and the one setting in this phase that its
    /// analysis is squarely about. Rolling out with the teacher alone leaves
    /// the learner blind to its own compounding errors and can land
    /// arbitrarily far from locally optimal; rolling out with the learner
    /// alone is the cell that paper marks "RL", the hard problem this phase
    /// exists to avoid. They report 0.5, and that LOLS is not sensitive to
    /// it.
    ///
    /// Measured here, the learner-only end took a DOOM policy's fixed block
    /// from 0.780 to 0.428 over six rounds.
    pub beta: f32,
    /// Decisions past the branch point a candidate is scored over. `0` scores
    /// to the end of the episode.
    ///
    /// Both bounds behind this phase carry the horizon, and this sample's is
    /// 400 against the twenty-odd of the tagging and parsing tasks they were
    /// measured on. A shorter window trades bias for variance: a candidate's
    /// score stops being decided by what happened three hundred decisions
    /// later, at the price of not seeing that far.
    pub credit: usize,
    /// Roll-outs averaged per candidate.
    ///
    /// One roll-out of a long episode is a single draw of a system where any
    /// decision changes everything after it, so its ordering of the
    /// candidates is partly a fact about that path rather than about the
    /// state. Averaging is the only thing that separates the two, and it
    /// costs a level reload per extra roll-out.
    pub repeats: usize,
    /// Draw the alternatives uniformly rather than from what the policy ranks
    /// highest.
    ///
    /// AggreVaTe's sample-complexity result is stated for actions explored
    /// uniformly at random and LOLS evaluates every action at a state. Taking
    /// the policy's own top-ranked instead makes which actions appear in a
    /// cost-sensitive example depend on the policy being trained, which is
    /// the wrong kind of feedback for a phase whose job is to find actions
    /// the policy currently undervalues.
    pub wide: bool,
    /// Rounds of DAgger between the warm start and the policy gradient. `0`
    /// is off.
    ///
    /// Each round runs the STUDENT, asks the teacher what it would have done
    /// at every state the student reached, adds those labels to everything
    /// collected so far and refits. It is the only phase of this run that
    /// puts a label on a state the teacher would never have visited, which is
    /// most of the states a student sees once it is acting on its own. See
    /// [`ControlPipeline::label_student`].
    pub dagger: usize,
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
            policy: PolicyConfig { clip: 0.2, entropy: 0.02, gamma: GAMMA, anchor: 0.0, target_kl: 0.02 },
            seed: 0,
            warmup_episodes: 60,
            warmup_epochs: 12,
            warmup_keep: 1.0,
            improve: 0,
            states: 200,
            alternatives: 2,
            explore: 0,
            explore_steps: 60,
            archive: None,
            self_imitate: 0,
            beta: 0.5,
            credit: 0,
            repeats: 1,
            wide: true,
            dagger: 0,
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
    /// Rounds of outcome-fitted improvement after the imitation phases.
    pub fn improve(mut self, n: usize) -> ControlSpec {
        self.improve = n;
        self
    }

    /// How a probe is shaped: decisions per round, alternatives at each, and
    /// how long a branch departs from the teacher for.
    /// Everything that shapes a probe. Taken together rather than one
    /// setter each, because they are only meaningful as a set: a beta with no
    /// repeats measures a mixture with one draw, and a credit horizon with
    /// the wrong candidate set measures the wrong actions carefully.
    pub fn probing(
        mut self,
        states: usize,
        alternatives: usize,
        beta: f32,
        credit: usize,
        repeats: usize,
        wide: bool,
    ) -> ControlSpec {
        self.states = states;
        self.alternatives = alternatives;
        self.beta = beta;
        self.credit = credit;
        self.repeats = repeats;
        self.wide = wide;
        self
    }

    /// Exploring episodes per round, and how far each goes after resuming.
    pub fn exploring(mut self, episodes: usize, steps: usize) -> ControlSpec {
        self.explore = episodes;
        self.explore_steps = steps;
        self
    }

    /// Where the archive of successful trajectories lives between runs.
    pub fn archive(mut self, path: Option<String>) -> ControlSpec {
        self.archive = path;
        self
    }

    /// Rounds of playing, keeping the best episodes and cloning those.
    pub fn self_imitate(mut self, n: usize) -> ControlSpec {
        self.self_imitate = n;
        self
    }

    /// Rounds of DAgger between the warm start and the policy gradient.
    pub fn dagger(mut self, n: usize) -> ControlSpec {
        self.dagger = n;
        self
    }

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
        self.gamma = spec.policy.gamma;
        // A WARM START IS A START.
        //
        // Cloning the teacher is how a policy gets off the ground when it has
        // nothing: sampling a good action out of a text action space takes
        // longer than any budget here allows. It is not how a policy that
        // already plays gets better, and doing it anyway undoes the run
        // before it.
        //
        // A generation that loads the last one's weights and then clones the
        // teacher again is pulled back toward a player that has never
        // finished a level. The improvement is thrown away at the start of
        // every generation and the whole sequence is capped at the teacher,
        // which is the one thing this loop exists to get past.
        let continuing = self.started_from_head;
        if continuing && spec.warmup_episodes > 0 {
            println!(
                "    no warm start: continuing from weights that already play. \
                 Cloning the teacher again would undo them"
            );
        }
        if spec.warmup_episodes > 0 && !continuing {
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
        }
        // BEFORE the anchor is read, because the policy the anchor holds to
        // should be the best imitation this run can produce and not the first
        // one it happened to fit.
        if spec.dagger > 0 {
            println!(
                "    {} rounds of labelling the states the student reaches, {} episodes each",
                spec.dagger, spec.episodes
            );
            self.dagger(spec, log, &mut step)?;
        }
        if spec.self_imitate > 0 {
            println!(
                "    {} rounds of playing {} episodes and cloning the best of them",
                spec.self_imitate, spec.episodes
            );
            self.self_imitate(spec, log, &mut step)?;
        }
        if spec.improve > 0 {
            println!(
                "    {} rounds of probing what a different action was worth, {} decisions each",
                spec.improve, spec.states
            );
            self.improve(spec, log, &mut step)?;
        }
        if spec.warmup_episodes > 0 {
            // Whatever the imitation phases produced is what the anchor holds
            // to. Read once, here, because every later update moves the head
            // away from it and the point is to remember where it started.
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
        let mut last_rank = f32::NEG_INFINITY;
        let mut best_ranked_on = "";
        // The policy the run entered the loop with is a candidate too - see
        // [`Keep`]. Only when there is a fixed block to rank it on: without
        // one an iteration is ranked on its own rollout, and the policy that
        // has not run one has no comparable number. Then the starting rank is
        // negative infinity and the first iteration takes it, as before.
        let entered = if spec.gauge_episodes > 0 {
            self.gauge(spec.gauge_episodes, spec.max_steps)?
        } else {
            None
        };
        let mut keep =
            Keep::starting(entered.unwrap_or(f32::NEG_INFINITY), self.model.head_weights());
        // The running mean of the iterates, which is a different candidate
        // from the best of them and often a better one. See `mean_of`.
        let mut running: Option<Vec<(String, Vec<f32>)>> = None;
        let mut averaged = 0usize;
        for it in 0..spec.iterations {
            // The weights the rollout is about to be drawn from. Without a
            // fixed block the round's rank IS that rollout's score, and the
            // score belongs to the policy that produced it. See [`Scored`].
            let entering = self.model.head_weights();
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
            let (rank, whose) = ranking(gauged, stats.mean_progress)
                .unwrap_or((stats.mean_return, Scored::Entering));
            let ranked_on = if gauged.is_some() {
                "on the fixed block"
            } else if stats.mean_progress.is_some() {
                "over its own episodes"
            } else {
                "in return"
            };
            // Numbered by the iterate the score belongs to. A rollout drawn
            // before the update scores the policy the round INHERITED, which
            // is the previous round's product.
            let at = match whose {
                Scored::Entering => it,
                Scored::Produced => it + 1,
            };
            if keep.offer(at, rank, || match whose {
                Scored::Entering => entering.clone(),
                Scored::Produced => self.model.head_weights(),
            }) {
                best_ranked_on = ranked_on;
            }
            last_rank = rank;
        }
        // Every rank in that loop belongs to the rollout that PRECEDED its
        // update, so with no fixed block the last update's policy has been
        // measured by nothing at all. One more rollout is what makes it a
        // candidate, and it costs what any other iteration's rollout costs.
        if spec.gauge_episodes == 0 && spec.iterations > 0 {
            let (_, stats) = self.rollout(spec.episodes, spec.max_steps)?;
            self.last = stats;
            let rank = stats.mean_progress.unwrap_or(stats.mean_return);
            let ranked_on = if stats.mean_progress.is_some() {
                "over its own episodes"
            } else {
                "in return"
            };
            println!("    the last iterate scores {rank:.3} {ranked_on}");
            if keep.offer(spec.iterations, rank, || self.model.head_weights()) {
                best_ranked_on = ranked_on;
            }
            last_rank = rank;
        }
        // Both candidates, measured on the same worlds, and the better one
        // kept. Averaging is a claim about the shape of the sequence, not a
        // law, so it is checked rather than assumed.
        if let Some(mean) = &running {
            let live = self.model.head_weights();
            self.model.set_head_weights(mean);
            let m = self.gauge(spec.gauge_episodes, spec.max_steps)?;
            let bw = keep.best().1.to_vec();
            self.model.set_head_weights(&bw);
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
        let (it, w) = keep.best();
        if it != spec.iterations {
            // The SAME number the choice was made on. Reporting the rollout's
            // progress next to a decision taken on the fixed block is two
            // different measurements and one decision, which reads as though
            // the wrong iteration was kept.
            println!(
                "    keeping {}, which scored {:.3} {} - the last one scored {last_rank:.3}",
                match it {
                    0 => "the policy the imitation phases produced".to_string(),
                    n => format!("iteration {n}"),
                },
                keep.rank,
                if it == 0 { "on the fixed block" } else { best_ranked_on }
            );
            let w = w.to_vec();
            self.model.set_head_weights(&w);
        }
        Ok(TrainReport { steps: step, final_loss: last_loss, seconds: 0.0 })
    }

    fn run_eval(&mut self) -> Result<EvalReport> {
        // Greedy, and on seeds no training rollout used: what the policy would
        // actually do, on situations it has not been updated against.
        let (mut total, mut wins, mut steps) = (0.0f32, 0usize, 0usize);
        let n = EVAL_SEEDS.count();
        self.measuring(|p| -> Result<()> {
            for seed in EVAL_SEEDS {
                let (st, ret, won) = p.episode(seed, true, p.max_steps, false)?;
                total += ret;
                wins += usize::from(won);
                steps += st.len();
            }
            Ok(())
        })?;
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
            explored: None,
            started_from_head: self.head.is_some(),
            model,
            env: self.env,
            rng: data::rng::Rng::new(self.seed ^ 0xc0ffee),
            episode_seed: 0,
            reference: None,
            gae_lambda: GAE_LAMBDA,
            gamma: GAMMA,
            last: Rollout::default(),
            critic: Critic::new(cfg_width, CRITIC_HIDDEN, self.seed ^ 0x1c1),
            critic_mse: 0.0,
            max_steps: self.max_steps,
            measurements: 0,
        })
    }
}

#[cfg(test)]
mod repeat_tests {
    use super::same_again;

    /// Repeating an action has to survive the numbers in the sentence
    /// changing, because they change at every step - the room ahead is 320
    /// units and then 288.
    #[test]
    fn the_same_intention_is_found_again_when_its_numbers_have_moved() {
        let options = vec![
            "attack the imp 150 units away, 12 degrees to your left".to_string(),
            "walk forward, 288 units of open floor ahead".to_string(),
            "turn around to see what is behind you".to_string(),
        ];
        assert_eq!(same_again("walk forward, 320 units of open floor ahead", &options), Some(1));
    }

    /// And must not mistake the opposite intention for the same one. Without
    /// the two-word minimum, "turn left and go that way" and "turn right and
    /// go that way" share their first word and a search would wander.
    #[test]
    fn the_opposite_way_is_not_the_same_action() {
        let options = vec![
            "turn right and go that way, 320 units of room".to_string(),
            "push on the wall or door directly in front of you".to_string(),
        ];
        assert_eq!(same_again("turn left and go that way, 288 units of room", &options), None);
    }

    /// When the intention is simply no longer available - the imp is dead,
    /// the door is open - there is nothing to repeat and the caller falls
    /// back to choosing afresh.
    #[test]
    fn an_intention_that_is_gone_is_reported_gone() {
        let options = vec!["walk forward, 96 units of open floor ahead".to_string()];
        assert_eq!(same_again("attack the imp 150 units away, straight ahead", &options), None);
    }
}

#[cfg(test)]
mod keep_tests {
    use super::Keep;

    fn w(tag: f32) -> Vec<(String, Vec<f32>)> {
        vec![("head.weight".into(), vec![tag])]
    }

    /// A phase that cannot improve on what it was handed hands it back.
    ///
    /// This is the one that was wrong, in all three phases at once: the
    /// comparison started at the first round rather than at the policy the
    /// phase inherited, so "keep the best round" meant "keep the least bad
    /// round" whenever no round was actually an improvement - and a run could
    /// only ever leave a phase worse off than it entered it.
    #[test]
    fn rounds_that_are_all_worse_keep_the_policy_the_phase_started_with() {
        let mut k = Keep::starting(0.770, w(0.0));
        k.offer(1, 0.731, || w(1.0));
        k.offer(2, 0.742, || w(2.0));
        k.offer(3, 0.700, || w(3.0));
        let (at, weights) = k.best();
        assert_eq!(at, 0, "a phase with no good round kept one of its bad ones");
        assert_eq!(weights[0].1, vec![0.0]);
    }

    /// And it still takes a round that IS better - the guard must not become a
    /// refusal to learn.
    #[test]
    fn the_best_round_is_kept_when_there_is_one() {
        let mut k = Keep::starting(0.770, w(0.0));
        k.offer(1, 0.731, || w(1.0));
        k.offer(2, 0.812, || w(2.0));
        k.offer(3, 0.790, || w(3.0));
        let (at, weights) = k.best();
        assert_eq!(at, 2);
        assert_eq!(weights[0].1, vec![2.0]);
    }

    /// A tie goes to whatever came first. Every rank here carries noise, so
    /// two scores that came out equal are evidence of nothing, and the policy
    /// that has been moved less is the one to keep.
    #[test]
    fn an_equal_score_is_not_a_reason_to_move() {
        let mut k = Keep::starting(0.770, w(0.0));
        k.offer(1, 0.770, || w(1.0));
        assert_eq!(k.best().0, 0);
    }
}

#[cfg(test)]
mod archive_tests {
    use super::{Archive, Demo, KEEP_PER_KIND};

    fn demo(tag: &str) -> Vec<Demo> {
        vec![Demo {
            objective: "reach the exit".into(),
            observation: tag.into(),
            options: vec!["walk forward".into()],
            action: 0,
        }]
    }

    /// The defect: every maze a scenario generates shared that scenario's
    /// name, so the archive held ONE run for all of them - the run that drew
    /// the easiest maze - and every later round cloned its turns onto layouts
    /// that did not have them.
    #[test]
    fn two_worlds_of_one_kind_each_keep_their_own_run() {
        let mut a = Archive::default();
        a.offer("maze", "maze#1", 0.90, demo("easy"));
        a.offer("maze", "maze#2", 0.30, demo("hard"));
        assert_eq!(a.len(), 2, "one scenario's mazes collapsed onto a single run");
        let seen: Vec<String> = a.demos_above(f32::NEG_INFINITY).iter().map(|d| d.observation.clone()).collect();
        assert!(seen.contains(&"easy".to_string()) && seen.contains(&"hard".to_string()));
    }

    /// And within one world it is still the best run that is kept.
    #[test]
    fn a_better_run_on_the_same_world_replaces_it_and_a_worse_one_does_not() {
        let mut a = Archive::default();
        assert!(a.offer("maze", "maze#1", 0.30, demo("first")));
        assert!(a.offer("maze", "maze#1", 0.70, demo("better")));
        assert!(!a.offer("maze", "maze#1", 0.50, demo("worse")));
        assert_eq!(a.len(), 1);
        assert_eq!(a.demos_above(f32::NEG_INFINITY)[0].observation, "better");
    }

    /// An environment that draws a fresh world every episode would otherwise
    /// archive every episode it ever played, which is cloning its own bad
    /// play. The cap bounds it, and takes the weakest world first.
    #[test]
    fn the_cap_drops_the_weakest_world_of_that_kind() {
        let mut a = Archive::default();
        for i in 0..KEEP_PER_KIND {
            a.offer("maze", &format!("maze#{i}"), 0.50 + i as f32 * 0.01, demo("kept"));
        }
        assert_eq!(a.len(), KEEP_PER_KIND);
        // Better than the weakest: it goes in and the weakest goes out.
        assert!(a.offer("maze", "maze#new", 0.90, demo("strong")));
        assert_eq!(a.len(), KEEP_PER_KIND);
        assert!(!a.by_instance.contains_key("maze#0"));
        // Worse than everything: it does not displace anything.
        assert!(!a.offer("maze", "maze#weak", 0.01, demo("weak")));
        assert_eq!(a.len(), KEEP_PER_KIND);
        assert!(!a.by_instance.contains_key("maze#weak"));
    }

    /// The cap is per KIND, so a scenario that has filled it cannot evict
    /// another scenario's only solved world.
    #[test]
    fn one_kind_filling_up_does_not_evict_another() {
        let mut a = Archive::default();
        a.offer("level", "E1M4", 0.20, demo("the only run through E1M4"));
        for i in 0..KEEP_PER_KIND + 4 {
            a.offer("maze", &format!("maze#{i}"), 0.50 + i as f32 * 0.01, demo("maze"));
        }
        assert!(a.by_instance.contains_key("E1M4"));
        assert_eq!(a.by_kind().len(), 2);
    }

    /// A level with fixed geometry is its own instance, so nothing about the
    /// cap changes what it used to do: one run, replaced only by a better one.
    #[test]
    fn a_fixed_level_still_keeps_exactly_its_best_run() {
        let mut a = Archive::default();
        a.offer("E1M1", "E1M1", 0.41, demo("a"));
        a.offer("E1M1", "E1M1", 0.63, demo("b"));
        a.offer("E1M2", "E1M2", 0.22, demo("c"));
        assert_eq!(a.len(), 2);
        assert_eq!(a.worst(), 0.22);
        assert_eq!(a.by_kind(), vec![("E1M1".to_string(), 0.63, 1), ("E1M2".to_string(), 0.22, 1)]);
    }
}

#[cfg(test)]
mod ranking_tests {
    use super::{ranking, Keep, Scored};

    fn w(tag: f32) -> Vec<(String, Vec<f32>)> {
        vec![("head.weight".into(), vec![tag])]
    }

    #[test]
    fn a_fixed_block_score_belongs_to_the_weights_the_round_produced() {
        assert_eq!(ranking(Some(0.81), Some(0.42)), Some((0.81, Scored::Produced)));
    }

    #[test]
    fn without_a_block_a_rounds_own_score_belongs_to_the_weights_it_entered_with() {
        assert_eq!(ranking(None, Some(0.42)), Some((0.42, Scored::Entering)));
    }

    #[test]
    fn a_round_that_measured_nothing_has_nothing_to_rank() {
        assert_eq!(ranking(None, None), None);
    }

    /// The defect, played out over three rounds.
    ///
    /// Each round rolls out, scores what it rolled out, then updates. Round 2
    /// draws the best rollout and its update then walks the policy off. Paired
    /// the way this loop used to - the score from before the update, the
    /// weights from after it - the run keeps round 2's post-update policy,
    /// which is not the one that scored 0.90 and need not resemble it.
    #[test]
    fn the_weights_kept_are_the_ones_the_score_was_measured_on() {
        let rounds = [(1.0f32, 0.40f32, 10.0f32), (2.0, 0.90, 20.0), (3.0, 0.50, 30.0)];
        let mut k = Keep::starting(f32::NEG_INFINITY, w(0.0));
        for (round, (entering, own, produced)) in rounds.iter().enumerate() {
            let (rank, whose) = ranking(None, Some(*own)).unwrap();
            k.offer(round + 1, rank, || {
                w(match whose {
                    Scored::Entering => *entering,
                    Scored::Produced => *produced,
                })
            });
        }
        assert_eq!(
            k.best().1[0].1,
            vec![2.0],
            "kept the policy that FOLLOWED the best score instead of the one that got it"
        );
    }

    /// And a gauged round still keeps what the gauge measured, which is the
    /// updated policy. The fix must not swap the pairing the other way.
    #[test]
    fn a_gauged_round_keeps_the_policy_the_gauge_ran_on() {
        let mut k = Keep::starting(f32::NEG_INFINITY, w(0.0));
        let (rank, whose) = ranking(Some(0.81), Some(0.42)).unwrap();
        k.offer(1, rank, || {
            w(match whose {
                Scored::Entering => 1.0,
                Scored::Produced => 20.0,
            })
        });
        assert_eq!(k.best().1[0].1, vec![20.0]);
    }
}

#[cfg(test)]
mod measured_best_tests {
    use super::measured_best;

    #[test]
    fn a_decision_where_every_candidate_led_to_the_same_place_has_no_right_answer() {
        assert_eq!(measured_best(&[0.5, 0.5, 0.5]), None);
        // And within the tolerance, which is the case that actually occurs:
        // two trajectories differing only in a facing score the same to
        // within rounding.
        assert_eq!(measured_best(&[0.5000, 0.5002, 0.4999]), None);
    }

    #[test]
    fn the_best_measured_candidate_is_named_when_there_is_one() {
        assert_eq!(measured_best(&[0.40, 0.72, 0.31]), Some(1));
    }
}

#[cfg(test)]
mod drift_tests {
    use super::candidate_drift;

    /// The quantity the outcome fit's trust region fires on: how far the
    /// policy has moved, over the candidates a probe actually measured, from
    /// the policy that measured them.
    ///
    /// This exists because without it the fit had a cap on what any one
    /// decision could pull and no cap at all on how far the whole update
    /// travelled. Measured on a DOOM run, two rounds took the fixed block
    /// from 0.780 to 0.440 while the objective it optimises kept improving -
    /// the update was reaching its own target and taking the rest of the
    /// policy with it.
    #[test]
    fn a_policy_that_has_not_moved_has_not_drifted() {
        let p = [0.6, 0.3, 0.1];
        assert!(candidate_drift(&p, &p) < 1e-6, "an unmoved policy must read zero");
    }

    #[test]
    fn drift_grows_with_the_distance_moved() {
        let old = [0.6, 0.3, 0.1];
        let near = candidate_drift(&old, &[0.55, 0.32, 0.13]);
        let far = candidate_drift(&old, &[0.1, 0.3, 0.6]);
        assert!(near > 0.0, "a moved policy must read above zero: {near}");
        assert!(far > near, "further must read further: {near} then {far}");
    }

    /// It is the forward KL from the collecting policy, which is what makes
    /// `--target-kl` mean the same thing here as it does in the PPO loop.
    #[test]
    fn it_is_the_divergence_from_the_policy_that_collected_the_probe() {
        let old = [0.5, 0.5];
        let new = [0.25, 0.75];
        let want = 0.5 * (0.5f32 / 0.25).ln() + 0.5 * (0.5f32 / 0.75).ln();
        assert!((candidate_drift(&old, &new) - want).abs() < 1e-6);
    }
}

#[cfg(test)]
mod outcome_tests {
    use super::{outcome_costs, outcome_pull};

    /// The candidate that MEASURED better is the one the update aims at,
    /// whichever of them the teacher chose. This is the only thing the
    /// improvement phase does that imitation could not have done.
    #[test]
    fn the_candidate_that_scored_better_is_the_one_aimed_at() {
        // The teacher's own action first, as `Probe::tried` orders them, and
        // it is NOT the best here - the case worth having a test for.
        let costs = outcome_costs(&[0.40, 0.72, 0.31]);
        assert_eq!(costs[1], 0.0, "the winner costs nothing: {costs:?}");
        assert!(costs[2] > costs[0], "the worst-measured must cost the most: {costs:?}");

        // A policy that currently prefers the wrong one is pulled off it.
        let (_, grad) = outcome_pull(&[0.6, 0.3, 0.1], &costs);
        assert!(grad[1] < 0.0, "the better candidate must be pushed up: {grad:?}");
        assert!(grad[0] > 0.0, "the one the policy wrongly prefers must come down: {grad:?}");
    }

    /// A decision where every candidate led to the same place teaches nothing
    /// and must pull nothing - with no threshold deciding what "the same" is.
    /// Measured on this sample that is 40% to 52% of probed decisions, so an
    /// update that gave them any weight would spend most of its magnitude on
    /// them.
    #[test]
    fn a_decision_that_made_no_difference_pulls_nothing() {
        let costs = outcome_costs(&[0.62, 0.62, 0.62]);
        let (l, grad) = outcome_pull(&[0.4, 0.3, 0.3], &costs);
        assert_eq!(l, 0.0);
        for g in &grad {
            assert!(g.abs() < 1e-7, "a decision with no spread pulled: {grad:?}");
        }
    }

    /// And one that made a small difference pulls proportionately less than
    /// one that made a large difference. The costs are in the units the run
    /// is scored in, so this is arithmetic rather than tuning - which is the
    /// point of not putting them through a temperature first.
    #[test]
    fn the_pull_is_proportional_to_what_the_decision_was_worth() {
        let belief = [0.5, 0.3, 0.2];
        let (_, small) = outcome_pull(&belief, &outcome_costs(&[0.620, 0.621, 0.620]));
        let (_, large) = outcome_pull(&belief, &outcome_costs(&[0.40, 0.72, 0.31]));
        let mag = |g: &Vec<f32>| g.iter().map(|x| x.abs()).sum::<f32>();
        assert!(
            mag(&large) > 50.0 * mag(&small),
            "a decision worth 100x as much pulled {:.6} against {:.6}",
            mag(&large),
            mag(&small)
        );
    }

    /// The loss is the policy's own expected cost, so it is bounded by the
    /// costs themselves and can never go negative. The construction this
    /// replaces was a sum of log-probabilities weighted by signed advantages,
    /// which is unbounded below: it drove the worst option's probability
    /// toward zero for ever-increasing reward, took the fixed block from
    /// 0.683 to 0.549, and printed a negative cross-entropy on the way.
    #[test]
    fn the_loss_cannot_be_driven_below_zero() {
        let costs = outcome_costs(&[0.40, 0.72, 0.31]);
        for belief in [[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]] {
            let (l, _) = outcome_pull(&belief, &costs);
            assert!(l >= 0.0, "expected cost went negative at {belief:?}: {l}");
            assert!(l <= costs.iter().copied().fold(0.0, f32::max) + 1e-6);
        }
        // And it is minimised by putting the mass on the winner.
        let (best, _) = outcome_pull(&[0.0, 1.0, 0.0], &costs);
        assert!(best.abs() < 1e-7, "all mass on the winner must cost nothing: {best}");
    }

    /// A policy already on the winner is not pushed anywhere. The gradient
    /// vanishes at the optimum rather than continuing to pay for
    /// ever-smaller probabilities.
    #[test]
    fn a_policy_that_already_agrees_is_left_alone() {
        let costs = outcome_costs(&[0.40, 0.72, 0.31]);
        let (_, grad) = outcome_pull(&[0.0, 1.0, 0.0], &costs);
        for g in &grad {
            assert!(g.abs() < 1e-6, "a policy on the winner was still pulled: {grad:?}");
        }
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

#[cfg(test)]
mod branch_tests {
    use super::branch_plan;

    /// The defect: the teacher's own branch ran from the live state, because
    /// the environment was already standing on the branch point and restoring
    /// onto it looked like an engine call for nothing. It is not for nothing.
    /// Restoring is the thing being trusted, and the check that it landed
    /// where it left - the options on offer are unchanged - ran before every
    /// OTHER roll-out. So the one score every reported number is relative to
    /// was the one score taken from a state the restore path had never been
    /// asked to reproduce and nothing had checked, and a restore that quietly
    /// dropped part of the run would read as the alternatives being worse
    /// than the teacher.
    #[test]
    fn every_candidate_is_measured_from_a_restored_state() {
        let plan = branch_plan(&[3, 0, 7], &[false, true]);
        assert_eq!(plan.len(), 6);
        for t in &plan {
            assert!(t.restore, "a roll-out was measured from an unrestored state: {t:?}");
        }
    }

    /// And the order the candidates were assembled in must not reach the
    /// measurement. Two candidates compared under different conditions is not
    /// a comparison, and the condition that differed was which of them got
    /// the privileged first slot - which is always the teacher's, so the
    /// asymmetry fell the same way at every probed decision rather than
    /// averaging out over them.
    #[test]
    fn the_order_candidates_are_tried_in_does_not_change_what_they_get() {
        let under = |candidates: &[usize], c: usize| -> Vec<(usize, bool, bool)> {
            branch_plan(candidates, &[false, true])
                .into_iter()
                .filter(|t| t.candidate == c)
                .map(|t| (t.repeat, t.reference, t.restore))
                .collect()
        };
        for c in [3, 0, 7] {
            assert_eq!(
                under(&[3, 0, 7], c),
                under(&[7, 3, 0], c),
                "candidate {c} was measured differently for having been listed elsewhere"
            );
        }
    }
}
