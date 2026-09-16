// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Episodes, reward, and the controls that make "it learned" checkable.
//!
//! The whole difficulty of this milestone is that a plastic network's
//! behaviour changes over time whether or not it is learning anything useful.
//! Weights drift, activity wanders, and an episode-to-episode improvement
//! appears in a system that is doing nothing of the kind. So the apparatus
//! here is built around the controls rather than around the learner: every
//! condition runs the same episodes through the same code, and only the one
//! thing under test differs.

use crate::reference::{ImitationReward, Reference};
use crate::Fly;

/// What one episode produced.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Episode {
    /// Net forward displacement of the body over the episode, in the model's
    /// own length units. This is the objective: a fly that walks forward
    /// scores, one that stands still or falls over does not.
    pub distance: f64,
    /// Total spikes, so a condition that simply went quiet is distinguishable
    /// from one that moved less.
    pub spikes: u64,
    /// Proprioceptor spikes, same purpose for the sensory channel.
    pub proprio_spikes: u64,
    /// Total reward earned, on whatever objective was set. For
    /// [`Objective::Imitate`] this is what a creature is actually being
    /// scored on; distance is then only a side observation.
    pub reward: f64,
    /// Ticks the episode actually ran. Under [`Objective::Imitate`] this is
    /// itself a score: an episode ends when the body loses the reference, so
    /// a longer one tracked for longer.
    pub ticks: u32,
    /// The body drifted further from the reference than
    /// [`Objective::Imitate`]'s `terminal_com_dist` allowed.
    pub terminated: bool,
    /// The snippet ran out with the body still tracking it. The good ending.
    pub reached_end: bool,
    /// What the legs did, scored, under [`Objective::Walk`]. `None` under the
    /// other objectives, which do not collect a trace.
    pub gait: Option<crate::gait::Gait>,
    /// Ticks spent off the ground, under [`Objective::Fly`].
    pub airborne: u32,
    /// How far from upright the body ended, in radians: the larger of its
    /// roll and pitch.
    ///
    /// A DIAGNOSTIC and not part of any score, which is the point. Both
    /// locomotion rewards here are built on net travel, and the classic way a
    /// search cheats one is to tip the animal over and let it slide or roll -
    /// which keeps the root height, keeps the legs oscillating, and covers
    /// ground. A number that says "it finished on its side" is what turns that
    /// from an undetectable success into an obvious one.
    pub tipped: f64,
    /// Straight-line distance from where the episode started, in the model's
    /// own length units.
    ///
    /// Distinct from `distance`, which is displacement along world x and can
    /// be earned by a body that is being pushed. This cannot be earned by
    /// jitter: a fly shaking in place travels no net distance however fast its
    /// instantaneous speed reads, which is a confusion this crate has already
    /// measured (the two differ by more than tenfold on a walking run).
    pub net: f64,
}

impl Episode {
    /// What the episode is WORTH on its objective.
    ///
    /// For [`Objective::Walk`] that is net travel multiplied by the gait
    /// score, and the product is the whole design: travel alone is earned by a
    /// single coordinated lunge, rhythm alone is earned by a fly running on
    /// the spot, and only a sustained periodic tripod that actually goes
    /// somewhere earns both. For the others it is the accumulated reward.
    pub fn score(&self) -> f64 {
        match (self.gait, self.airborne) {
            (Some(g), _) => self.net * g.score(),
            (None, a) if a > 0 => self.net * (a as f64 / self.ticks.max(1) as f64),
            _ => self.reward,
        }
    }
}

/// What the creature is being asked to do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Objective {
    /// Net forward displacement of the body.
    ///
    /// Kept because the ceiling was measured under it, and because it is the
    /// honest demonstration of why it is the wrong objective: a single
    /// coordinated lunge scores as well as a gait, and a direct search under
    /// this reward suppressed the cord's recurrent circuitry and drove sensory
    /// input straight to the muscles. That is a reflex, and it is what this
    /// reward asks for.
    Displacement,
    /// Track a recorded fly walking, DeepMimic style.
    ///
    /// What flybody's own walking task uses, and what the imitation-learning
    /// literature settled on precisely because it does not need the reward
    /// engineering the alternative does. The reward is dense - every tick has
    /// a target - where displacement is nearly flat until something moves.
    Imitate {
        /// Where on the recorded data an episode begins.
        start: Start,
        reward: ImitationReward,
        /// End the episode once the body's centre of mass is this far from
        /// the reference's, in the model's own length units.
        ///
        /// Not optional garnish. The reward is a PRODUCT of Gaussian factors,
        /// so it is zero almost everywhere, and without termination an episode
        /// spends nearly all of its ticks collecting nothing and learning from
        /// nothing. flybody's walking task uses 0.33 cm, about 1.3 body
        /// lengths, which is [`RewardConfig::default`]'s value here too.
        terminal_com_dist: f64,
    },
    /// WALK: go somewhere, on a sustained alternating tripod.
    ///
    /// The objective the other two are not. Displacement is earned by one
    /// lunge - measured, a direct search under it suppressed the cord's own
    /// circuitry and wired sensory input straight to the muscles, which is a
    /// reflex and is what that reward asks for. Imitation is worse here and
    /// the measurement is unambiguous: against a reference that walks away
    /// whatever the body does, every motion is velocity error, and a
    /// PARALYSED fly scored 98.7% of the best driven score while driving the
    /// animal harder made the score monotonically worse.
    ///
    /// So this scores the two things a gait is, multiplied: net travel from
    /// where the episode began, times `gait::analyse`'s score over the whole
    /// leg trace. Neither factor alone is a gait and the product cannot be
    /// earned by a corpse, by a lunge, or by running on the spot.
    ///
    /// The PER-TICK reward stays dense (forward progress since the last tick)
    /// because a three-factor plasticity rule needs a signal every tick and a
    /// rhythm is a property of a window, not of a moment. The product is what
    /// a SEARCH sees (`Episode::score`). That split is deliberate and is the
    /// honest way to have both.
    Walk {
        /// End the episode once the body's height drops below this, in the
        /// model's own length units, measured from where it started. A fly
        /// that has fallen over is not walking, and the ticks it spends on the
        /// floor would otherwise dilute the rhythm it is being scored on.
        terminal_fall: f64,
    },
    /// FLY: stay up, and go somewhere while you are up.
    ///
    /// The airborne counterpart of `Walk`, and the same shape of reward for
    /// the same reason: net horizontal travel multiplied by the fraction of
    /// the episode spent off the ground. Travel alone is earned by a ballistic
    /// arc - a fly thrown sideways covers ground the whole way down - and
    /// airtime alone is earned by hovering, or by not having taken off from a
    /// height in the first place. Only staying up AND covering ground earns
    /// both.
    ///
    /// There is no rhythm term because a wingbeat is not the cord's to
    /// produce: the power muscles are stretch-activated and drive a thorax
    /// that resonates at a frequency the thorax chooses, so the wingbeat is
    /// GENERATED and the nervous system modulates it. What the motor neurons
    /// control - and what this objective can therefore reward - is power and
    /// steering, not frequency.
    Fly {
        /// How far the animal may descend from where it started before the
        /// episode ends, in the model's own length units: its altitude.
        ///
        /// An altitude rather than a floor height, because the flight scene
        /// deliberately has NO ground in it - it exists to characterise the
        /// airframe, and a fly that can bounce is measuring contact. So
        /// "landed" is defined by how far it has fallen, and every tick before
        /// that counts as airborne. A fly that holds its height is airborne
        /// for the whole episode.
        altitude: f64,
    },
}

impl Objective {
    /// Track a recorded snippet with flybody's own walking-task settings.
    ///
    /// A constructor rather than a literal, so the termination radius is not a
    /// number every call site has to know and half of them get wrong.
    /// Walk, with a fall threshold that clears the standing pose.
    ///
    /// A fly stands about a tenth of a centimetre up and the floor is at
    /// -0.132, so half a body length below the start is well past stumbling
    /// and well short of tripping the threshold on a normal stride.
    pub fn walk() -> Objective {
        Objective::Walk { terminal_fall: 0.125 }
    }

    /// Fly, with thresholds a quarter of a body length either side of the
    /// start: far enough that a bounce is not a landing and a wobble is not a
    /// take-off.
    /// Fly, with ten centimetres of altitude to lose.
    ///
    /// Forty body lengths, and it is chosen so the measurement is about the
    /// STEADY sink rate rather than the transient. A fly starts from rest and
    /// its thorax has to spin up, so the first centimetre or two is dominated
    /// by acceleration whatever the wings are doing: measured, over two
    /// centimetres a beating fly outlasts a still one by 1.6x, and over ten by
    /// 4.6x, against a steady-state sink rate that differs by 4.7x. A short
    /// altitude does not measure flight, it measures falling.
    pub fn flight() -> Objective {
        Objective::Fly { altitude: 10.0 }
    }

    pub fn imitate(start: Start) -> Objective {
        Objective::Imitate { start, reward: ImitationReward::default(), terminal_com_dist: 0.33 }
    }

    /// The largest reward one tick can earn, for reporting a score as a
    /// fraction of what is achievable.
    ///
    /// Displacement has no such bound - a body can always be flung further -
    /// so it reports `f64::INFINITY` rather than an invented ceiling, and a
    /// caller that divides by it gets 0% instead of a made-up percentage.
    pub fn max_per_tick(&self) -> f64 {
        match self {
            Objective::Displacement | Objective::Walk { .. } | Objective::Fly { .. } => f64::INFINITY,
            Objective::Imitate { reward, .. } => reward.max(),
        }
    }
}

/// Where an imitation episode begins on the reference data.
///
/// DeepMimic's two contributions are early termination and REFERENCE STATE
/// INITIALISATION, and they work together: termination keeps a product reward
/// out of the region where it is flat, and random initialisation is what stops
/// that from confining every episode to the first fraction of a second of the
/// recording. Starting always at frame 0 means a creature only ever sees the
/// part of the gait it can already reach, and the rest of the trajectory is
/// never experienced at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Start {
    /// Frame 0 of one named snippet. Deterministic, which is what a
    /// reproducible control needs.
    Fixed(usize),
    /// A uniformly random snippet and a uniformly random frame within it,
    /// leaving at least `min_frames` of trajectory ahead.
    Random { min_frames: usize },
}

/// How reward becomes a neuromodulator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RewardConfig {
    /// What to reward.
    pub objective: Objective,
    /// Ticks per episode.
    pub ticks: u32,
    /// Descending command held for the episode.
    pub command: f32,
    /// Exponential-moving-average rate for the reward baseline.
    ///
    /// The neuromodulator is a reward PREDICTION ERROR, not a reward: without
    /// a baseline, a constantly-rewarded network potentiates every eligible
    /// synapse without ever distinguishing a good tick from an average one,
    /// which is potentiation dressed as learning.
    pub baseline_rate: f64,
    /// Scales the prediction error into the modulator.
    pub modulator_gain: f32,
}

impl Default for RewardConfig {
    fn default() -> Self {
        RewardConfig {
            objective: Objective::Displacement,
            ticks: 300,
            command: 2.0,
            baseline_rate: 0.01,
            modulator_gain: 50.0,
        }
    }
}

/// Which condition an episode is run under.
///
/// These are not options; they are the control matrix. A result that appears
/// under `Learning` and also under `ShuffledReward` is not learning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Condition {
    /// Plasticity on, reward delivered when it is earned.
    Learning,
    /// Plasticity off. Weights cannot move.
    Frozen,
    /// Plasticity on, but the modulator is delivered at the WRONG time: the
    /// same values in a shuffled order, so the distribution is identical and
    /// only the correlation with behaviour is destroyed. This is the control
    /// that separates learning from potentiation.
    ShuffledReward,
    /// Plasticity on and reward delivered correctly, but the WIRING is a
    /// degree-matched shuffle of the connectome. The structural control: if
    /// this does as well, the published wiring was not what mattered.
    ///
    /// Selected when the fly is built, not here - the graph is fixed at
    /// construction - so this variant exists to label the condition rather
    /// than to change what `episode` does.
    ShuffledConnectome,
}

impl Condition {
    /// Whether weights may move under this condition.
    pub fn plastic(self) -> bool {
        self != Condition::Frozen
    }

    /// Whether the modulator this condition delivers is the one that was
    /// earned.
    pub fn reward_is_honest(self) -> bool {
        matches!(self, Condition::Learning | Condition::ShuffledConnectome)
    }
}

/// Run one episode and return what it produced.
///
/// The fly is reset first, so episodes are independent and an improvement
/// cannot come from a body that happens to have fallen into a better pose.
pub fn episode(fly: &mut Fly, cfg: RewardConfig, condition: Condition, rng: &mut Lcg) -> Result<Episode, String> {
    episode_with(fly, cfg, condition, None, rng)
}

/// The same, with a reference trajectory available for [`Objective::Imitate`].
///
/// Separate entry point rather than an `Option` on `RewardConfig` because a
/// reference is data with a lifetime, and threading it through a `Copy` config
/// would make the config borrow.
pub fn episode_with(
    fly: &mut Fly,
    cfg: RewardConfig,
    condition: Condition,
    reference: Option<&Reference>,
    rng: &mut Lcg,
) -> Result<Episode, String> {
    // An episode too short to hold a rhythm scores zero on a gait objective
    // however well the animal walks, and zero is indistinguishable from
    // failure. Refused rather than returned.
    if let Objective::Walk { .. } = cfg.objective {
        let seconds = cfg.ticks as f64 * crate::CONTROL_PERIOD;
        if seconds < crate::gait::MIN_SECONDS {
            return Err(format!(
                "a walking episode of {} ticks is {seconds:.2} s, and a gait cannot be scored below \
                 {:.2} s - it would score zero whatever the animal did",
                cfg.ticks,
                crate::gait::MIN_SECONDS
            ));
        }
    }
    fly.reset();
    let cmd = vec![cfg.command; fly.descending_count()];
    fly.set_descending(&cmd)?;
    fly.set_plasticity(condition.plastic());

    // Reference-state initialisation: an imitation episode starts ON the
    // trajectory it is asked to follow, at a point chosen by `Start`.
    let mut origin = (0usize, 0usize);
    if let Objective::Imitate { start, .. } = cfg.objective {
        let r = reference.ok_or("Objective::Imitate needs a reference trajectory")?;
        let (nq, nv) = fly.dims();
        r.check_matches(nq, nv)?;
        origin = match start {
            Start::Fixed(snippet) => (snippet, 0),
            Start::Random { min_frames } => {
                // Rejection would be simpler but can loop; instead pick among
                // the snippets that are long enough, and fail loudly if none
                // is, rather than silently falling back to frame 0 of snippet
                // 0 and running an experiment nobody asked for.
                let usable: Vec<usize> = (0..r.snippets()).filter(|&i| r.len(i) > min_frames).collect();
                if usable.is_empty() {
                    return Err(format!(
                        "no snippet has more than {min_frames} frames, so a random start leaving that \
                         much trajectory ahead is impossible"
                    ));
                }
                let snippet = usable[rng.index(usable.len())];
                (snippet, rng.index(r.len(snippet) - min_frames))
            }
        };
        let (snippet, offset) = origin;
        let (q, v) = r
            .frame(snippet, offset)
            .ok_or_else(|| format!("snippet {snippet} has no frame {offset}"))?;
        let q: Vec<f64> = q.iter().map(|x| *x as f64).collect();
        let v: Vec<f64> = v.iter().map(|x| *x as f64).collect();
        fly.set_pose(&q, &v)?;
    }

    let start = fly.qpos();
    let floor = start.get(2).copied().unwrap_or(0.0);
    let mut baseline = 0.0f64;
    let mut ep = Episode::default();
    // One sample per control tick, which is the rate `gait::analyse` expects.
    let mut trace = matches!(cfg.objective, Objective::Walk { .. })
        .then(|| crate::gait::Trace::new(crate::CONTROL_PERIOD));
    // Pre-drawn so that ShuffledReward delivers the SAME distribution as
    // Learning, just uncorrelated with what the fly did.
    let mut deltas: Vec<f32> = Vec::with_capacity(cfg.ticks as usize);

    let mut last_x = start.first().copied().unwrap_or(0.0);
    let mut last_y = start.get(1).copied().unwrap_or(0.0);
    for tick in 0..cfg.ticks as usize {
        let t = fly.step()?;
        ep.spikes += t.total_spikes as u64;
        ep.proprio_spikes += fly.proprioceptor_spikes() as u64;

        let reward = match cfg.objective {
            Objective::Displacement => {
                let x = fly.qpos().first().copied().unwrap_or(0.0);
                let r = x - last_x;
                last_x = x;
                r
            }
            Objective::Walk { terminal_fall } => {
                // The legs, every tick, for the rhythm half of the score.
                if let Some(t) = trace.as_mut() {
                    t.push(fly.leg_swing());
                }
                if fly.qpos().get(2).copied().unwrap_or(0.0) < floor - terminal_fall {
                    ep.terminated = true;
                    break;
                }
                let x = fly.qpos().first().copied().unwrap_or(0.0);
                let r = x - last_x;
                last_x = x;
                r
            }
            Objective::Fly { altitude } => {
                let q = fly.qpos();
                let z = q.get(2).copied().unwrap_or(0.0);
                if z < floor - altitude {
                    ep.terminated = true;
                    break;
                }
                ep.airborne += 1;
                // Horizontal progress only: a fly that gains height is not
                // travelling, and rewarding altitude would pay for a jump.
                let (x, y) = (q.first().copied().unwrap_or(0.0), q.get(1).copied().unwrap_or(0.0));
                let r = ((x - last_x).powi(2) + (y - last_y).powi(2)).sqrt();
                last_x = x;
                last_y = y;
                r
            }
            Objective::Imitate { reward, terminal_com_dist, .. } => {
                // One reference frame per control tick - the dataset is
                // sampled at exactly the control period, so `tick` counts off
                // frames from wherever this episode started.
                let r = reference.expect("checked above");
                let (snippet, offset) = origin;
                match r.frame(snippet, offset + tick) {
                    Some((rq, rv)) => {
                        let qpos = fly.qpos();
                        // Terminate BEFORE scoring, so a tick that has already
                        // lost the reference does not pay for itself. An
                        // episode's return is then the thing being maximised
                        // and surviving longer is how it grows, which is what
                        // makes a product reward learnable at all.
                        if ImitationReward::com_distance(&qpos, rq) > terminal_com_dist {
                            ep.terminated = true;
                            break;
                        }
                        reward.total_over(&qpos, &fly.qvel(), rq, rv, r.moving_dofs())
                    }
                    // The snippet ran out: the creature tracked it to the end,
                    // which is the good ending, not a reason to keep paying.
                    None => {
                        ep.reached_end = true;
                        break;
                    }
                }
            }
        };
        ep.reward += reward;
        ep.ticks += 1;
        baseline += cfg.baseline_rate * (reward - baseline);
        let delta = ((reward - baseline) * cfg.modulator_gain as f64) as f32;
        deltas.push(delta);

        match condition {
            Condition::Learning | Condition::ShuffledConnectome => fly.modulate(delta),
            // Deliver a modulator drawn from what this episode has already
            // produced, at a time unrelated to what just happened.
            Condition::ShuffledReward => {
                let pick = deltas[rng.index(deltas.len())];
                fly.modulate(pick);
            }
            Condition::Frozen => {}
        }
    }

    let end = fly.qpos();
    // Roll and pitch out of the root quaternion, MuJoCo's (w, x, y, z) order.
    // Yaw is deliberately not in it: an animal that has turned is not an
    // animal that has fallen over.
    let (qw, qx, qy, qz) = (
        end.get(3).copied().unwrap_or(1.0),
        end.get(4).copied().unwrap_or(0.0),
        end.get(5).copied().unwrap_or(0.0),
        end.get(6).copied().unwrap_or(0.0),
    );
    let roll = (2.0 * (qw * qx + qy * qz)).atan2(1.0 - 2.0 * (qx * qx + qy * qy));
    let pitch = (2.0 * (qw * qy - qz * qx)).clamp(-1.0, 1.0).asin();
    ep.tipped = roll.abs().max(pitch.abs());
    ep.distance = end.first().copied().unwrap_or(0.0) - start.first().copied().unwrap_or(0.0);
    let (dx, dy) = (
        end.first().copied().unwrap_or(0.0) - start.first().copied().unwrap_or(0.0),
        end.get(1).copied().unwrap_or(0.0) - start.get(1).copied().unwrap_or(0.0),
    );
    ep.net = (dx * dx + dy * dy).sqrt();
    // A trace too short to hold a cycle is refused by `analyse` rather than
    // guessed at, and a `None` gait scores zero - which is the right answer
    // for an episode that fell over in the first tenth of a second.
    ep.gait = trace.as_ref().and_then(crate::gait::analyse);
    Ok(ep)
}

/// A small deterministic PRNG for the shuffled-reward control.
///
/// `data::rng::Lcg` is this workspace's test PRNG, but `crates/data` is a
/// heavier dependency than this crate wants for one index draw, and the
/// shuffled-reward control needs to be reproducible rather than
/// cryptographic.
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Lcg {
        Lcg(seed | 1)
    }
    /// The next raw draw. Public so a search can seed an episode with it and
    /// have every candidate in a generation share that seed.
    pub fn next_u64(&mut self) -> u64 {
        self.next()
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    /// A standard normal, by Box-Muller. The search needs a symmetric
    /// perturbation; a uniform one biases every step toward the corners of the
    /// box it samples.
    pub fn normal(&mut self) -> f32 {
        let u1 = ((self.next() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        let u2 = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }

    fn index(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next() >> 33) as usize % n
    }
}

/// A search over per-cell-type gains, with the wiring held fixed.
///
/// This is the CEILING INSTRUMENT, and it exists to answer a question the
/// control matrix cannot: when the local learning rule fails to produce
/// walking, is the rule weak, or can this structure not do this task with this
/// reward and this body? Without an answer, a negative result is ambiguous and
/// therefore not much of a result.
///
/// It is not a surrogate-gradient method, and the substitution is deliberate.
/// A gradient path would have to differentiate through MuJoCo, which this
/// binding does not expose and which would mean either a differentiable body
/// or a policy-gradient estimator - both larger projects than the question
/// needs. What the question needs is an upper bound on what these parameters
/// can achieve under a stronger optimiser than a local rule, and a direct
/// search gives that.
///
/// The parameters are per-presynaptic-SUPER-CLASS gains rather than per-synapse
/// weights, which is what makes the search tractable: eleven numbers against
/// 5.3 million, at roughly three seconds per evaluation. It is also the
/// standard shape for a connectome-constrained model - structure fixed,
/// a small number of biologically meaningful gains free - rather than a
/// convenience. The cost is real and worth naming: a gain search cannot
/// express anything the cell-type partition cannot, so it is a LOWER bound on
/// what the full weight space could do, and a negative result from it is
/// weaker evidence than a negative result from a per-synapse optimiser.
pub struct GainSearch {
    /// Which gain group each edge belongs to, by its presynaptic neuron.
    edge_group: Vec<u8>,
    groups: Vec<String>,
    /// Signed, scaled weights at unit gain.
    base: Vec<f32>,
}

impl GainSearch {
    /// Build against the network a [`Fly`] with this `wiring` will actually
    /// run.
    ///
    /// The wiring is an argument rather than a scale, and that is the fix for
    /// a real defect: this used to build from `signed_csc`, the UNPRUNED
    /// graph, while every `Fly` runs `network`, which drops every pair below
    /// `Wiring::min_synapses` and renumbers what is left. On the fly's cord
    /// that is 5,305,638 weights against 1,372,588, so `Fly::set_weights`
    /// rejected the vector outright and `examples/ceiling` panicked on its
    /// first evaluation from the day the synapse floor was introduced. Taking
    /// the wiring makes the two agree by construction rather than by
    /// coincidence.
    pub fn new(c: &connectome::Connectome, wiring: crate::Wiring) -> GainSearch {
        let mut groups: Vec<String> = Vec::new();
        let mut of_neuron: Vec<u8> = Vec::with_capacity(c.neurons.len());
        for n in &c.neurons {
            let key = if n.super_class.is_empty() { "<none>" } else { n.super_class.as_str() };
            let idx = match groups.iter().position(|g| g == key) {
                Some(i) => i,
                None => {
                    groups.push(key.to_string());
                    groups.len() - 1
                }
            };
            of_neuron.push(idx as u8);
        }
        let net = c.network(wiring.weight_scale, wiring.size_limit, wiring.min_synapses);
        let edge_group = net.pre.iter().map(|&p| of_neuron[p as usize]).collect();
        GainSearch { edge_group, groups, base: net.w }
    }

    pub fn groups(&self) -> &[String] {
        &self.groups
    }

    /// The weight vector for a given gain setting.
    pub fn weights(&self, gains: &[f32]) -> Vec<f32> {
        self.base
            .iter()
            .zip(&self.edge_group)
            .map(|(w, &g)| w * gains.get(g as usize).copied().unwrap_or(1.0))
            .collect()
    }

    /// A neutral starting point: every gain at 1, which reproduces the
    /// connectome exactly as imported.
    pub fn unit_gains(&self) -> Vec<f32> {
        vec![1.0; self.groups.len()]
    }
}
