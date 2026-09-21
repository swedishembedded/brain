// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Doom as something a policy can be trained in: reward, episodes, a scripted
//! teacher to start from, and a window onto what the agent is thinking.
//!
//! ## The mission is part of the input
//!
//! An episode is run under a [`Mission`], and the mission is BOTH the text
//! prepended to every option and the weighting the reward is computed with.
//! That pairing is the point: "clear the level" and "get to the exit" want
//! different behaviour out of the same state, and because the instruction
//! travels with the observation the same weights can serve both - the policy
//! reads what it is being asked to do. Train over a mix of missions and you
//! get one policy that changes strategy on being told to, rather than three
//! policies.
//!
//! ## The reward is shaped, and the shaping is NOT distance to the exit
//!
//! Kills, items and the exit are what the game scores, and the exit arrives
//! once, hundreds of decisions after the choices that earned it. Something has
//! to fill the gap.
//!
//! The obvious filler is the wrong one. Rewarding progress toward the exit in
//! a straight line is potential-based shaping (Ng, Harada & Russell 1999) and
//! so provably leaves the optimal policy alone - but the guarantee is about
//! the optimum, not about what gets learned on the way to it, and in a
//! building the straight line goes through walls. Measured here before this
//! was changed: a trajectory spends 90 of its 120 decisions shuttling between
//! two spots, walking at the wall the exit is behind, backing off, and walking
//! at it again. Every one of those steps was being paid for. That is the
//! reward function teaching the agent to walk into walls, and no amount of
//! training fixes a reward that is wrong.
//!
//! What fills the gap instead is a **count-based exploration bonus**: the
//! first visit to a patch of floor in an episode pays, and a visit to a patch
//! already worn pays 1/sqrt(n) of it. This is the cheap, network-free member
//! of the intrinsic-motivation family that ICM and RND belong to, and it is
//! the one that fits here - it needs no second model and no gradient of its
//! own. Walking into a wall discovers nothing and earns nothing; finding a
//! corridor earns. The agent is not told where to go, and it is not told that
//! walls are bad. It is paid for finding out, which is the difference between
//! an agent that explores and one that is steered.
//!
//! The exit is still in the observation, because a player can see the level
//! and so should the agent. What has gone is being PAID for pointing at it.
//!
//! Swedish Embedded AB builds reward models and evaluation harnesses that
//! measure what a customer actually wants rather than what is easy to log. If
//! your team needs that, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::sync::{Arc, Mutex};

use brain::decision::Rng;
use brain::Env;

use crate::action::{self, Option_, Tag};
use crate::doom::{Config, Doom};
use crate::frame::{Frame, Map};
use crate::obs::{self, History, State};
use crate::report::{Gauge, Progress};

/// What a decision is paid for.
///
/// The two are not two tunings of one idea. [`Payment::Shaped`] is a weighted
/// sum over things that happened - kills, items, damage, floor newly walked -
/// and [`Payment::Gauge`] is the movement of the single number a run is
/// finally kept or discarded on. Only the second has any guarantee of
/// agreeing with that number; see [`crate::report::Gauge`] for the identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Payment {
    /// DOOM's own events under the mission's weights, plus an exploration
    /// bonus and route shaping.
    Shaped,
    /// What the decision changed about [`crate::report::Score::value`].
    Gauge,
}

impl Payment {
    pub const ALL: [Payment; 2] = [Payment::Shaped, Payment::Gauge];

    pub fn name(self) -> &'static str {
        match self {
            Payment::Shaped => "shaped",
            Payment::Gauge => "gauge",
        }
    }

    pub fn parse(name: &str) -> Option<Payment> {
        Payment::ALL.into_iter().find(|p| p.name() == name)
    }
}


/// What the agent is being told to do this episode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mission {
    /// Kill everything. The exit is worth little.
    Clear,
    /// Reach the exit, fast. Fighting is a cost, not a goal.
    Speedrun,
    /// Come out alive. Damage hurts far more than anything else pays.
    Survive,
}

impl Mission {
    pub const ALL: [Mission; 3] = [Mission::Clear, Mission::Speedrun, Mission::Survive];

    pub fn name(self) -> &'static str {
        match self {
            Mission::Clear => "clear",
            Mission::Speedrun => "speedrun",
            Mission::Survive => "survive",
        }
    }

    pub fn parse(s: &str) -> Option<Mission> {
        Mission::ALL.into_iter().find(|m| m.name() == s)
    }

    /// The instruction the model reads, prepended to every option.
    pub fn instruction(self) -> &'static str {
        match self {
            Mission::Clear => {
                "Your orders: kill every enemy on this level. Finding the exit can wait. \
                 Which of these is the best next move?"
            }
            Mission::Speedrun => {
                "Your orders: reach the level exit as fast as you can. Fight only what \
                 blocks your path. Which of these is the best next move?"
            }
            Mission::Survive => {
                "Your orders: stay alive. Avoid damage, take cover and heal before you \
                 take risks. Which of these is the best next move?"
            }
        }
    }

    fn weights(self) -> Weights {
        match self {
            // Clearing wants to find the monsters, which means covering the
            // level; speedrunning wants to cover it faster and cares little
            // about what it meets; surviving would rather sit still, so its
            // bonus is smallest and its damage term largest.
            Mission::Clear => Weights {
                kill: 1.5,
                item: 0.2,
                hurt: 0.02,
                exit: 3.0,
                explore: 0.10,
                approach: 0.02,
            },
            // Speedrun's exploration bonus is a tenth of the others' and its
            // exit is worth more than twice as much, and those two numbers are
            // not free. PPO optimises the DISCOUNTED return: at gamma 0.99 an
            // exit reward arriving 150 decisions away is worth 0.22 of its
            // face value, while each step's exploration bonus is worth nearly
            // all of its own. At 0.20 a 300-decision wander was worth about 19
            // discounted and the exit about 3.3 - wandering paid six times
            // better than finishing, and a policy cloned from a teacher that
            // finishes six times out of six was trained back down to zero out
            // of four over eight iterations, correctly maximising what it had
            // been given. Here the exit is worth about 8.8 against at most 1.9
            // for covering the whole level.
            // `hurt` is set from arithmetic, not from feel. Nukage does 5
            // points every 32 tics and a decision is 4 to 6 of them, so a
            // tick lands about every five and a half decisions: at 0.02 a
            // decision spent wading cost 0.018 and a decision spent exploring
            // paid 0.02, and wading through a pool to reach new ground was
            // therefore FREE. That is not an agent failing to learn that
            // slime is bad, it is an agent correctly learning that it is not.
            // At 0.06 a decision in nukage costs 0.055 against an exploration
            // bonus of at most 0.02, and one in hellslime costs 0.22.
            Mission::Speedrun => Weights {
                kill: 0.2,
                item: 0.1,
                hurt: 0.06,
                exit: 40.0,
                explore: 0.02,
                // Per 32-unit cell closed along the route. A level whose exit
                // is 5300 units of walking away is about 165 cells, so a full
                // traversal pays about 3.3 against the exit's 40 - enough to
                // be visible to an estimator that cannot see the exit, small
                // enough that finishing is still overwhelmingly what pays.
                approach: 0.02,
            },
            Mission::Survive => Weights {
                kill: 0.3,
                item: 0.4,
                hurt: 0.08,
                exit: 5.0,
                explore: 0.05,
                approach: 0.01,
            },
        }
    }
}

/// How much a cell of progress along the route is worth, and the rules that
/// keep it from being the straight-line shaping that was rejected before.
///
/// PPO's advantage estimator has a half-life of about eleven decisions here
/// (`gamma * lambda` is 0.9405), so a reward paid at the exit reaches a
/// decision a hundred steps earlier with a weight of 0.002 and one two
/// hundred steps earlier with 0.000005. In a finishing episode the exit is
/// 89% of the return, paid on a single step. Everything before the last
/// thirty decisions is therefore learning from the dense terms alone - and
/// the dense term used to be the exploration bonus, which pays for covering
/// NEW floor. So the signal the gradient could see rewarded wandering and the
/// signal it could not see rewarded finishing, and PPO correctly optimised
/// the one it could see: over ten iterations it took a cloned policy from
/// 5 wins in 8 down to 3, and held it there to within 0.06.
///
/// The fix is a potential-based term on the ROUTE distance, which is the
/// distance over walkable ground the player has actually seen. The straight
/// line was rejected for good reason - it goes through walls, so pressing
/// against the wall the exit is behind reduces it and the agent was paid to
/// do that. Route distance does not fall when the player walks into a wall,
/// because the route does not go through the wall. Going out and back nets
/// `(gamma - 1)(phi(a) + phi(b))`, which is nothing, so oscillation still
/// pays nothing.
const CELL: f32 = 32.0;

/// A re-plan is not movement. The route changes its mind when the seen set
/// grows, and the distance can jump by more than a player could walk; paying
/// for that would be paying for the map improving rather than for progress.
/// A decision moves the player at most a couple of hundred units.
const REPLAN_JUMP: i32 = 256;

/// Cells of route closed between two decisions, or nothing when the two
/// distances do not describe the same walk.
fn cells_closed(before: Option<i32>, after: Option<i32>, comparable: bool) -> f32 {
    if !comparable {
        return 0.0;
    }
    let (Some(before), Some(after)) = (before, after) else {
        return 0.0;
    };
    // A re-plan is not movement. The route changes its mind when the seen set
    // grows, and the distance can jump by more than a player could walk.
    if (before - after).abs() > REPLAN_JUMP {
        return 0.0;
    }
    (before - after) as f32 / CELL
}

struct Weights {
    kill: f32,
    item: f32,
    /// Per point of health lost.
    hurt: f32,
    exit: f32,
    /// Paid for the first visit to a patch of floor, and 1/sqrt(n) of it on
    /// the n-th. The only thing filling the gap between one decision and a
    /// level's worth of them.
    explore: f32,
    /// Per 32-unit cell of route closed on the goal. See [`CELL`].
    approach: f32,
}

/// What the engine calls a floor that hurts, for telling damage from the
/// ground apart from damage from a monster.
const BURNING: [&str; 5] = [
    "nukage",
    "hellslime",
    "super hellslime",
    "a damaging floor",
    "the exit floor",
];

/// Paid on death under every mission. Large enough that dying is never the
/// cheap way to end a bad episode, which it is if the only alternative is a
/// slow drip of step cost.
const DEATH: f32 = 5.0;
/// Charged every step, so standing still is never free.
const STEP_COST: f32 = 0.002;
/// How many past positions to keep, and how far the player has to have got
/// from all of them to count as making progress. 16 decisions is about three
/// seconds of game time, and 160 units is five player widths - covered easily
/// by three steps of real walking and never by a cycle.
const CIRCLE_WINDOW: usize = 16;
const CIRCLE_RADIUS: i32 = 160;
/// Side of the patch of floor the exploration bonus counts visits to.
const EXPLORE_CELL: i32 = 128;
/// Episodes the curriculum judges before moving the start, the share of them
/// that must have finished, and how much further back it then goes.
const CURRICULUM_WINDOW: usize = 8;
const CURRICULUM_ADVANCE: f32 = 0.6;
const CURRICULUM_STEP: i32 = 400;
/// Reverse curriculum: start the episode near the exit and move the start
/// further away as the policy succeeds.
///
/// The problem it solves is not difficulty, it is SILENCE. Reaching the exit
/// of E1M1 from the level's own spawn pays once, a thousand decisions later,
/// and across five training runs it never happened - so the exit reward
/// contributed exactly nothing to any gradient the policy ever took. A reward
/// that never fires is not a hard reward, it is an absent one. This is
/// Florensa et al.'s reverse curriculum, the standard answer.
///
/// The start is a DISTANCE ALONG THE ROUTE, not a random walk from the goal.
/// A random walk was tried first and is worse in every way that matters: it
/// wanders, two stages of it are not comparable, and nobody can say how hard
/// a given stage is. "Start 1500 units of walking from the exit" means the
/// same thing every episode and grows cleanly to the whole level - on E1M1
/// the spawn is 5344 units out.
///
/// It is TRAINING state, and it is the only state here that outlives an
/// episode, which is what makes [`Curriculum::set_counting`] necessary.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Curriculum {
    /// Off by default: an environment plays the level as it ships until a run
    /// asks for the curriculum.
    on: bool,
    /// How far from the exit episodes currently start, in map units.
    start_distance: i32,
    /// Whether the episodes arriving now COUNT - see
    /// [`Curriculum::set_counting`].
    counting: bool,
    /// Recent episode outcomes, for deciding when to move the start back.
    recent: std::collections::VecDeque<bool>,
}

impl Default for Curriculum {
    /// Counting, because a plain episode is a training episode: the thing
    /// that has to be declared is a MEASUREMENT, and a default that had to be
    /// switched on would silently do nothing for an environment nobody
    /// remembered to switch it on for.
    fn default() -> Curriculum {
        Curriculum {
            on: false,
            start_distance: 0,
            counting: true,
            recent: std::collections::VecDeque::new(),
        }
    }
}

impl Curriculum {
    /// How far out to start the next episode, or `None` when the curriculum
    /// is off and the level is played as it ships.
    ///
    /// Read whether or not the episode counts: an evaluation has to face the
    /// same task training does, or it is measuring something else.
    fn start(&self) -> Option<i32> {
        self.on.then_some(self.start_distance)
    }

    /// Whether the episodes arriving from now on are TRAINING.
    ///
    /// A measurement - a fixed-block score, a hypothetical roll-out that asks
    /// what a different action would have been worth - is an episode from in
    /// here and indistinguishable from a real one. Left counting, those
    /// episodes feed the window that decides when the start moves back, so
    /// the difficulty the next training episode faces depends on how often
    /// the run stopped to measure itself.
    fn set_counting(&mut self, on: bool) {
        self.counting = on;
    }

    /// Record how an episode ended, and say so when that moved the start
    /// back: `(wins in the window, the new distance)`.
    ///
    /// Advanced on a WINDOW rather than a single success, because one win
    /// from two decisions away is luck; and by a small step, because a start
    /// that jumps past what the policy can do puts it back in the silent
    /// regime the curriculum exists to escape.
    fn note(&mut self, won: bool) -> Option<(usize, i32)> {
        if !self.on || !self.counting {
            return None;
        }
        self.recent.push_back(won);
        if self.recent.len() > CURRICULUM_WINDOW {
            self.recent.pop_front();
        }
        if self.recent.len() < CURRICULUM_WINDOW {
            return None;
        }
        let wins = self.recent.iter().filter(|w| **w).count();
        if (wins as f32 / self.recent.len() as f32) < CURRICULUM_ADVANCE {
            return None;
        }
        self.start_distance += CURRICULUM_STEP;
        self.recent.clear();
        Some((wins, self.start_distance))
    }
}

/// Decisions of going nowhere before the teacher stops repeating itself.
///
/// Six, because the slowest thing here that legitimately leaves the player
/// standing still is a door: one press and thirty tics to rise, which is five
/// to eight decisions. Any rule that was going to work has had its chance by
/// then, and at two the teacher gave up on doors it had only just pressed.
const STUCK_TRY_SOMETHING_ELSE: u32 = 6;
/// Decisions of finding nowhere new before the player starts trying walls.
///
/// Long enough that crossing a room already walked does not set it off, short
/// enough to matter inside one episode.
const STALE_TRY_THE_WALLS: u32 = 12;
/// Decisions to hold one direction for once circling is detected. Long enough
/// to clear the cycle's own diameter at the walking speed one decision buys.
const COMMIT_STEPS: u32 = 8;

/// What the agent just did, for the window and the transcript.
///
/// Written by the environment and read by whatever is drawing; a mutex rather
/// than a channel because a viewer wants the LATEST state, not every state,
/// and a channel that nobody drains grows without bound.
#[derive(Clone, Default)]
pub struct Inspect {
    pub observation: String,
    pub options: Vec<String>,
    /// The policy's distribution, when the caller is running the loop itself
    /// and has one. Empty during training.
    pub probs: Vec<f32>,
    pub chosen: usize,
    pub reward: f32,
    pub total: f32,
    pub step: u32,
    pub episode: u64,
    pub mission: &'static str,
    pub outcome: String,
    pub kills: u32,
    pub total_kills: u32,
    pub health: i32,
    pub items: u32,
    pub secrets: u32,
    pub map: u32,
    /// Tics of game time since the episode started - the game's own clock,
    /// which is not the number of decisions.
    pub game_tic: i64,
    /// Decisions in a row that moved the player nowhere.
    pub stuck: u32,
    /// How far from the exit the episode started, in map units. Zero when the
    /// curriculum is off.
    pub back_steps: u32,
    /// Reward per step for the episode so far, for the chart.
    pub history: Vec<f32>,
    /// The game's framebuffer, one per TIC of the decision just taken, oldest
    /// first. Empty when frame capture is off - it costs a round trip and 85KB
    /// each, which is most of a training step, so it is only paid for when
    /// somebody is looking.
    ///
    /// Per tic rather than per decision because a decision spans four to six
    /// of them: keeping only the last throws away five sixths of the motion,
    /// and a recording made from those looks like the player is teleporting.
    pub frames: Vec<Frame>,
    /// What the agent knows about where it can go and where it has been.
    pub known: Map,
}

/// The monsters an arena episode draws from, weakest first.
///
/// Three of DOOM's easiest, on purpose: the question being asked is whether a
/// policy learns to fight at all, and a Baron of Hell answers it by killing
/// every episode before either player has made a decision worth scoring.
const ARENA_MONSTERS: [&str; 3] = ["FORMER HUMAN", "FORMER HUMAN SERGEANT", "IMP"];

/// Where an arena spawn may go: the six directions the observation reports
/// clearance for. A monster is only placed down one with room, so it starts on
/// open floor and in sight rather than inside a wall.
const ARENA_BEARINGS: [i32; 6] = [0, -45, 45, -90, 90, 180];

pub struct DoomEnv {
    doom: Doom,
    cfg: Config,
    pub mission: Mission,
    /// Fixed for the run, or sampled per episode when training over a mix.
    mission_mix: bool,
    /// The levels episodes are drawn from, one per episode.
    ///
    /// A policy trained on one level can learn that level: where its exit is,
    /// which way to leave the first room. Several levels is what makes that
    /// strategy stop paying and leaves only the ability - and it costs
    /// nothing, because a level is chosen by the same seed that already makes
    /// an episode reproducible.
    maps: Vec<u32>,
    /// Overrides the mission's own `approach` weight, for measuring what that
    /// term is worth by turning it off. See [`CELL`].
    approach_weight: Option<f32>,
    /// What a decision is paid for. See [`Payment`].
    payment: Payment,
    /// How much of the gauge this episode has already been paid for. Reset
    /// with the episode, or the next run starts in debt.
    gauge: Gauge,
    /// The decision budget an episode is given, so that "how far it got" can
    /// be a fraction rather than a count.
    max_steps: u32,
    /// The scenarios episodes are drawn from, one per episode, in place of the
    /// levels. A scenario is a level the engine builds from the episode's own
    /// seed, so "several levels" becomes "a new level every episode" - which
    /// is the only arrangement in which memorising the map pays nothing.
    scenarios: Vec<String>,
    state: State,
    opts: Vec<Option_>,
    /// Where the player was, and for how many steps it has not changed.
    ///
    /// The scripted player walks into walls: it aims at the exit, the exit is
    /// through a door it cannot see, and it pushes forward forever. Noticing
    /// that is what turns the baseline from "broken" into "beatable", and a
    /// baseline that is broken makes the learned number unreadable.
    last_pos: Option<(i32, i32)>,
    stuck: u32,
    /// Decisions since the player last walked somewhere new. See `visit`.
    stale: u32,
    /// The kinds of thing already tried since the player last moved.
    tried: std::collections::HashSet<Tag>,
    /// The last few positions, for noticing that the player is going round in
    /// circles rather than merely standing still. Greedy navigation's
    /// characteristic failure is not being stuck, it is a two-step cycle -
    /// walk toward the goal into a corner, back out of the corner, repeat -
    /// and every position in it is a position the player MOVED to, so `stuck`
    /// cannot see it. Measured before this existed: 120 decisions, 15 distinct
    /// cells, the last 90 of them shuttling between two of them.
    recent: std::collections::VecDeque<(i32, i32)>,
    /// Visits per patch of floor THIS EPISODE. Episodic, not lifetime: the
    /// agent should re-explore a level it has been reset into, and a lifetime
    /// count would stop paying for that after the first few episodes and leave
    /// nothing filling the gap at all.
    visited: std::collections::HashMap<(i32, i32), u32>,
    /// While positive, keep doing the same KIND of thing. Committing for
    /// several steps is what breaks a two-cycle; re-deciding every step from
    /// the same state just re-enters it.
    commit: u32,
    commit_tag: Option<Tag>,
    total: f32,
    /// The part of the return the game itself scores - see
    /// [`DoomEnv::reward`].
    extrinsic: f32,
    /// Health lost to burning floor this episode. Scored separately because
    /// "does it learn that slime is bad" is a question about this number and
    /// nothing else, and it is invisible in a return that mixes it with
    /// everything else the episode did.
    floor_damage: u32,
    steps: u32,
    exited: bool,
    episode: u64,
    pub inspect: Arc<Mutex<Inspect>>,
    /// Fetch the framebuffer with every observation. Off unless something is
    /// drawing it.
    capture_frames: bool,
    /// Fetch a frame for every tic rather than every decision. See
    /// [`Inspect::frames`].
    frames_per_tic: bool,
    /// Monsters to place around the player at the start of every episode.
    ///
    /// Zero plays the level as it ships. Above zero is a SCENARIO, which is
    /// what makes the experiment answerable: measured on E1M1 at 250
    /// decisions, neither the scripted player nor the policy killed anything
    /// in twelve episodes, because an episode is spent getting out of the
    /// spawn area. A metric with no events in it cannot separate two players.
    ///
    /// This is the same move ViZDoom makes - `defend_the_center`,
    /// `deadly_corridor` and `health_gathering` are all hand-made starting
    /// positions - and for the same reason. It is scenario design, not a
    /// cheat: the game, the actions, the reward and the opponent are
    /// unchanged, and both players face the identical arena on a given seed.
    arena: usize,
    /// How far out episodes start and when that moves. See [`Curriculum`].
    curriculum: Curriculum,
    warned_dropped: bool,
    /// How this episode has been going, for the line printed when it ends.
    progress: Progress,
    /// What the player has seen and can no longer see. See
    /// [`crate::memory::Memory`].
    memory: crate::memory::Memory,
    /// The route distance to the goal at the last decision, and which goal it
    /// was measured to. See [`CELL`].
    approach_from: Option<i32>,
    approach_goal: Option<String>,
    /// Set when the game itself failed (the process died, the socket broke).
    /// An environment that silently returns a terminal state on an I/O error
    /// teaches the policy that the error was a legal end to an episode.
    pub fault: Option<String>,
    /// What [`Env::hold`] kept, for [`Env::resume`] to put back.
    /// The client-side half of every held state, by slot.
    slots: std::collections::HashMap<usize, Held>,
}

impl DoomEnv {
    pub fn new(doom: Doom, cfg: Config, mission: Mission, mission_mix: bool) -> DoomEnv {
        let only = vec![cfg.map];
        DoomEnv {
            doom,
            cfg,
            mission,
            mission_mix,
            maps: only,
            approach_weight: None,
            payment: Payment::Shaped,
            gauge: Gauge::new(),
            max_steps: 400,
            scenarios: Vec::new(),
            state: State::parse(EMPTY).expect("the empty state is well formed"),
            opts: Vec::new(),
            last_pos: None,
            stuck: 0,
            stale: 0,
            tried: std::collections::HashSet::new(),
            recent: std::collections::VecDeque::new(),
            visited: std::collections::HashMap::new(),
            commit: 0,
            commit_tag: None,
            total: 0.0,
            extrinsic: 0.0,
            floor_damage: 0,
            steps: 0,
            exited: false,
            episode: 0,
            inspect: Arc::new(Mutex::new(Inspect::default())),
            capture_frames: false,
            frames_per_tic: false,
            arena: 0,
            curriculum: Curriculum::default(),
            warned_dropped: false,
            progress: Progress::new(),
            memory: crate::memory::Memory::new(),
            approach_from: None,
            approach_goal: None,
            fault: None,
            slots: std::collections::HashMap::new(),
        }
    }

    /// Fetch and publish the framebuffer with every step. See
    /// [`Inspect::frame`] for what it costs.
    pub fn capture_frames(&mut self, on: bool) {
        self.capture_frames = on;
    }

    /// Keep a frame for every tic, not just the last of each decision. Costs a
    /// round trip per tic; only worth it when recording.
    pub fn frames_per_tic(&mut self, on: bool) {
        self.frames_per_tic = on;
    }

    /// Draw each episode's level from this list.
    pub fn set_maps(&mut self, maps: Vec<u32>) {
        if !maps.is_empty() {
            self.maps = maps;
        }
    }

    /// Override the mission's route-closing weight. `None` leaves the mission
    /// to decide; `Some(0.0)` is the ablation.
    pub fn set_approach(&mut self, w: Option<f32>) {
        self.approach_weight = w;
    }

    /// Pay decisions for what they moved the gauge, or by the mission's
    /// weights. See [`Payment`].
    pub fn set_payment(&mut self, p: Payment) {
        self.payment = p;
    }

    /// How many decisions an episode is allowed, for scoring how far one got.
    pub fn set_max_steps(&mut self, n: usize) {
        if n > 0 {
            self.max_steps = n as u32;
        }
    }

    /// Play generated scenarios instead of the game's own levels, drawing one
    /// per episode from this list. Overrides `set_maps`.
    pub fn set_scenarios(&mut self, names: Vec<String>) {
        self.scenarios = names;
    }

    pub fn scenarios(&self) -> &[String] {
        &self.scenarios
    }

    /// Turn the reverse curriculum on. See [`Curriculum`].
    pub fn set_curriculum(&mut self, on: bool) {
        self.curriculum.on = on;
    }

    /// Start episodes this many map units of walking from the exit.
    ///
    /// The curriculum's reach is TRAINING state and is not saved with the
    /// head, so a policy trained out to 3000 units is, on a fresh run, asked
    /// to finish from zero - two decisions, every time. Evaluating or
    /// recording a curriculum policy therefore has to say how hard to make it,
    /// and this is where that number comes from.
    pub fn set_start_distance(&mut self, units: i32) {
        self.curriculum.start_distance = units;
    }

    /// Start every episode with `n` monsters around the player. See
    /// [`DoomEnv::arena`].
    pub fn set_arena(&mut self, n: usize) {
        self.arena = n;
    }

    /// Place the episode's monsters, on open floor and in sight.
    ///
    /// Driven by the episode seed, so the arena is a property of the seed and
    /// both players meet the same one - which is what makes the comparison a
    /// comparison.
    fn build_arena(&mut self, seed: u64) -> Result<(), String> {
        let mut rng = Rng::new(seed ^ 0x0a4e_a000);
        for i in 0..self.arena {
            let c = self.state.clearance;
            // Only directions with room, and never further than the room they
            // have - a monster behind a wall is a monster neither player can
            // reach and an episode that scores nothing.
            let open: Vec<(i32, i32)> = ARENA_BEARINGS
                .iter()
                .map(|&b| {
                    let room = match b {
                        0 => c.ahead,
                        -45 => c.ahead_left,
                        45 => c.ahead_right,
                        -90 => c.left,
                        90 => c.right,
                        _ => c.behind,
                    };
                    (b, room)
                })
                .filter(|&(_, room)| room >= 160)
                .collect();
            if open.is_empty() {
                return Err(format!(
                    "no open direction to place arena monster {} of {}",
                    i + 1,
                    self.arena
                ));
            }
            let (bearing, room) = open[(rng.next_u64() % open.len() as u64) as usize];
            let distance = 96 + (rng.next_u64() % (room - 96).max(1) as u64) as i32;
            let kind = ARENA_MONSTERS[(rng.next_u64() % ARENA_MONSTERS.len() as u64) as usize];
            self.doom
                .spawn(kind, distance, bearing)
                .map_err(|e| format!("spawn: {e}"))?;
        }
        // One tic so the spawns are in the world before the first observation
        // describes it.
        let json = self.doom.step("[]", 1).map_err(|e| format!("{e}"))?;
        self.state = State::parse(&json)?;
        self.remember();
        Ok(())
    }

    /// Fold the settled observation into the player's memory, and put back
    /// what it now recalls. Called once per observation, after any last
    /// adjustment to the state, because what is remembered has to be what
    /// the agent was actually shown.
    fn remember(&mut self) {
        self.memory.observe(&self.state);
        // And ask the engine the way back to each of them, so that "go back
        // for the medikit you saw" is a walk round the corner rather than a
        // heading into the wall the corner is made of. A failed call leaves
        // every path `None`, which is the straight-line behaviour and not a
        // wrong route.
        let goals = self.memory.goals();
        let places: Vec<(f64, f64)> = goals.iter().map(|(_, x, y)| (*x, *y)).collect();
        let paths = self.doom.route_to(&places).unwrap_or_default();
        let answered: Vec<(i64, Option<crate::memory::Path>)> = goals
            .iter()
            .enumerate()
            .map(|(i, (id, _, _))| (*id, paths.get(i).copied().flatten()))
            .collect();
        self.memory.routed(&answered);
        self.state.recalled = self.memory.recall(&self.state);
        self.state.wounded = self.memory.wounded();
    }

    /// What the game itself scored this episode, with no exploration bonus.
    pub fn extrinsic(&self) -> f32 {
        self.extrinsic
    }

    /// Health lost to burning floor this episode.
    pub fn floor_damage(&self) -> u32 {
        self.floor_damage
    }

    /// What the agent has done recently - the part of the observation the game
    /// does not report. See [`History`].
    pub fn history(&self) -> History {
        let cell = self.cell();
        History {
            stuck: self.stuck,
            visits_here: self.visited.get(&cell).copied().unwrap_or(0),
            patches: self.visited.len(),
        }
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn options(&self) -> &[Option_] {
        &self.opts
    }

    fn cell(&self) -> (i32, i32) {
        (
            self.state.player.x.unwrap_or(0).div_euclid(EXPLORE_CELL),
            self.state.player.y.unwrap_or(0).div_euclid(EXPLORE_CELL),
        )
    }

    fn visit(&mut self) -> u32 {
        let cell = self.cell();
        let fresh = !self.visited.contains_key(&cell);
        let n = self.visited.entry(cell).or_insert(0);
        *n += 1;
        let n = *n;
        // How long since the player last set foot anywhere new.
        //
        // Being STUCK and being out of IDEAS are different failures and want
        // different things done. Stuck is not moving, and the rotation in
        // `choose` handles it. Out of ideas is moving perfectly well around
        // ground already walked, finding nothing, for ever - which is what
        // happens in a room whose only way on is a switch on its wall. E1M8
        // opens with exactly that room, and a player that shoves at walls
        // only when it cannot move never presses it. That level has never
        // been played past its first room.
        if fresh {
            self.stale = 0;
        } else {
            self.stale = self.stale.saturating_add(1);
        }
        n
    }

    /// Tell the curriculum how the episode that just ended ended.
    fn note_outcome(&mut self, won: bool) {
        if let Some((wins, distance)) = self.curriculum.note(won) {
            println!(
                "doom: curriculum - {wins}/{CURRICULUM_WINDOW} finished, starting {distance} \
                 units from the exit"
            );
        }
    }

    /// Reward for what just happened, under the current mission.
    ///
    /// Returns `(total, extrinsic)`. They are separated because the
    /// exploration bonus is most of the total - measured, about 92% of a
    /// 140-decision episode - so a comparison on total return is mostly a
    /// comparison of who covered more floor. That is a real thing to measure
    /// and it is not what the GAME scores, and conflating them lets a policy
    /// look better or worse than it plays. The extrinsic half is kills, items,
    /// damage and the exit: DOOM's own opinion.
    fn reward(&mut self) -> (f32, f32) {
        let w = self.mission.weights();
        let mut r = -STEP_COST;
        for e in &self.state.events {
            r += match e.kind.as_str() {
                "kill" => w.kill * e.amount as f32,
                "item" | "ammo" | "weapon" => w.item,
                "key" => w.item * 4.0,
                "secret" => w.item * 3.0,
                "hurt" => -w.hurt * e.amount as f32,
                // Healing pays back less than the damage cost, so walking into
                // a fight for a medkit is never profitable by itself.
                "heal" => w.hurt * 0.4 * e.amount as f32,
                "armor" => w.hurt * 0.2 * e.amount as f32,
                "death" => -DEATH,
                "exit" => w.exit,
                _ => 0.0,
            };
        }
        let extrinsic = r;
        for e in &self.state.events {
            // The engine names the cause; a floor is the one with no
            // inflictor, and it names which floor.
            if e.kind == "hurt" && matches!(e.what.as_deref(), Some(w) if BURNING.contains(&w)) {
                self.floor_damage += e.amount.max(0) as u32;
            }
        }

        // The exploration bonus. 1/sqrt(n) rather than first-visit-only so
        // that a patch stays slightly worth revisiting - a strictly one-shot
        // bonus makes a corridor already walked worth exactly nothing, and an
        // agent that has to cross one to reach anything new is being charged
        // for the crossing.
        let n = self.visit();
        r += w.explore / (n as f32).sqrt();
        r += self.approach(self.approach_weight.unwrap_or(w.approach));
        (r, extrinsic)
    }

    /// Potential-based shaping on the route distance to the goal: cells of
    /// route actually closed since the last decision.
    ///
    /// This is `phi(s') - phi(s)` with `phi = -distance` and NO discount on
    /// the potential, and the undiscounted form is the point. Ng, Harada and
    /// Russell's term is `gamma * phi(s') - phi(s)`, which for a negative
    /// potential leaves a residue of `d * (1 - gamma)` when nothing happens -
    /// at gamma 0.99 and a goal 5300 units away that is a decision spent
    /// standing perfectly still earning 1.66 cells' worth of progress. The
    /// difference form pays exactly nothing for standing still, exactly
    /// nothing for going out and coming back, and exactly nothing for walking
    /// into a wall, at any distance. Those three are the properties that
    /// matter here; strict policy invariance under a discount the shaping
    /// does not share is not.
    ///
    /// See [`CELL`] for why this exists at all and why the straight-line
    /// version of it did not work.
    fn approach(&mut self, weight: f32) -> f32 {
        let goal = self
            .state
            .exit
            .as_ref()
            .and_then(|e| e.goal.clone());
        let here = self
            .state
            .exit
            .as_ref()
            .and_then(|e| e.path_distance);

        // Unexplored ground is not a goal. The frontier moves every time it
        // is reached, so closing on it happens over and over and paying for
        // that is paying for the exploration bonus twice under another name.
        let real_goal = goal.as_deref() != Some("unexplored");
        let was = self.approach_from.take();
        let same_goal = self.approach_goal == goal;
        self.approach_goal = goal;
        self.approach_from = if real_goal { here } else { None };

        weight * cells_closed(was, here, same_goal && real_goal)
    }

    /// Fetch the framebuffer, or turn capture off if it cannot be read.
    ///
    /// A frame nobody can draw is a display problem and never a reason to end
    /// an episode - the policy is not reading it.
    fn grab_frame(&mut self) -> Option<Frame> {
        if !self.capture_frames {
            return None;
        }
        match self
            .doom
            .frame()
            .map_err(|e| e.to_string())
            .and_then(|j| Frame::parse(&j))
        {
            Ok(f) => Some(f),
            Err(e) => {
                eprintln!("doom: could not read the framebuffer: {e}");
                self.capture_frames = false;
                None
            }
        }
    }

    fn publish(&mut self, chosen: usize, reward: f32, probs: Vec<f32>, frames: Vec<Frame>) {
        // The map changes slowly - a cell a step - so it is fetched with the
        // frames rather than on its own schedule, and only when drawing.
        let map = if self.capture_frames {
            self.doom.map().ok().and_then(|j| Map::parse(&j).ok())
        } else {
            None
        };
        if let Ok(mut i) = self.inspect.lock() {
            if let Some(m) = map {
                i.known = m;
            }
            if !frames.is_empty() {
                i.frames = frames;
            }
            i.observation = obs::render(&self.state, self.history());
            i.options = self.opts.iter().map(|o| o.text.clone()).collect();
            i.probs = probs;
            i.chosen = chosen;
            i.reward = reward;
            i.total = self.total;
            i.step = self.steps;
            i.episode = self.episode;
            i.mission = self.mission.name();
            i.outcome = self.state.outcome.clone();
            i.kills = self.state.level.kills;
            i.total_kills = self.state.level.total_kills;
            i.items = self.state.level.items;
            i.secrets = self.state.level.secrets;
            i.map = self.state.level.map;
            i.game_tic = self.state.episode_tic;
            i.stuck = self.stuck;
            i.back_steps = self.curriculum.start_distance.max(0) as u32;
            i.health = self.state.player.health;
            if i.step <= 1 {
                i.history.clear();
            }
            i.history.push(reward);
        }
    }

    /// Record something that makes the rest of the run meaningless, and say
    /// so the first time.
    ///
    /// The `Env` trait has no error channel: a step returns a reward and a
    /// done flag, so a broken environment is indistinguishable from an episode
    /// that ended early with nothing gained. Twelve PPO iterations were spent
    /// that way - `return +0.00, steps 8` for eight episodes, over and over,
    /// with the cause recorded in a field nothing in the training path reads.
    fn fail(&mut self, msg: String) {
        if self.fault.is_none() {
            eprintln!("doom: {msg}");
            eprintln!("doom: the rest of this run is measuring nothing.");
            self.fault = Some(msg);
        }
    }

    /// Take an option and record the outcome. Shared by the SDK's rollout
    /// (through [`Env::step`]) and by this sample's own inspected loop, so the
    /// two cannot disagree about what a step is.
    pub fn apply(&mut self, action: usize, probs: Vec<f32>) -> (f32, bool) {
        let Some(opt) = self.opts.get(action).cloned() else {
            self.fail(format!(
                "the policy chose option {action} of {}",
                self.opts.len()
            ));
            return (0.0, true);
        };
        // The teacher's memory advances from what actually happened, not from
        // having been asked. See `DoomEnv::scripted`.
        self.note_executed(opt.tag);
        // While recording, the step is run ONE TIC AT A TIME so every rendered
        // frame can be kept. The engine's key handling is unchanged by the
        // split - `forward` holds its key for a countdown of tics and the turn
        // servo closes its angle per tic, both of which carry across separate
        // step calls - so the game sees the same decision either way. It costs
        // a round trip per tic and is only done when something is recording.
        let mut frames = Vec::new();
        let per_tic = self.frames_per_tic;
        // Events accumulated across the tics of this decision. Each response
        // DRAINS the engine's event log, so stepping tic by tic and keeping
        // only the last state loses every event from the earlier tics - which
        // is most of them, including the kill or the level exit that the
        // decision was about. Measured: a recorded run that finished the level
        // four times out of four and scored -0.07 for it.
        let mut carried: Vec<crate::obs::Event> = Vec::new();
        let json = if per_tic {
            let mut last = String::new();
            for t in 0..opt.tics {
                let cmds = if t == 0 { opt.commands.as_str() } else { "[]" };
                match self.doom.step(cmds, 1) {
                    Ok(j) => last = j,
                    Err(e) => {
                        self.fail(format!("the game stopped answering: {e}"));
                        return (0.0, true);
                    }
                }
                if let Ok(st) = State::parse(&last) {
                    carried.extend(st.events.iter().cloned());
                    if st.done {
                        // The decision ended the episode; running its
                        // remaining tics would step past the end.
                        if let Some(f) = self.grab_frame() {
                            frames.push(f);
                        }
                        break;
                    }
                }
                if let Some(f) = self.grab_frame() {
                    frames.push(f);
                }
            }
            last
        } else {
            match self.doom.step(&opt.commands, opt.tics) {
                Ok(j) => j,
                Err(e) => {
                    self.fail(format!("the game stopped answering: {e}"));
                    return (0.0, true);
                }
            }
        };
        if !per_tic {
            if let Some(f) = self.grab_frame() {
                frames.push(f);
            }
        }
        match State::parse(&json) {
            Ok(s) => self.state = s,
            Err(e) => {
                self.fail(e);
                return (0.0, true);
            }
        }
        if per_tic {
            // The final response's own events are already in `carried`.
            self.state.events = carried;
        }
        self.remember();
        self.steps += 1;
        let pos = (
            self.state.player.x.unwrap_or(0),
            self.state.player.y.unwrap_or(0),
        );
        // 24 map units is about a third of the player's own width, so anything
        // under it over a whole decision is not movement.
        self.stuck = match self.last_pos {
            Some(p) if (p.0 - pos.0).abs() + (p.1 - pos.1).abs() < 24 => self.stuck + 1,
            _ => 0,
        };
        if self.stuck == 0 {
            self.tried.clear();
        }
        self.last_pos = Some(pos);
        self.recent.push_back(pos);
        if self.recent.len() > CIRCLE_WINDOW {
            self.recent.pop_front();
        }
        if self.state.events_dropped > 0 && !self.warned_dropped {
            // Once, not per step: the buffer is sized for a step and this
            // means a step ran long enough to overflow it, which makes the
            // reward for that step an undercount.
            eprintln!(
                "doom: the engine dropped {} events in one step; rewards for that step are \
                 an undercount",
                self.state.events_dropped
            );
            self.warned_dropped = true;
        }
        self.progress.note(&self.state);
        let (shaped, extrinsic) = self.reward();
        // Both halves are computed whichever one is being paid: `reward` also
        // keeps the floor-damage tally the observation reads and the route
        // bookkeeping the shaping term needs, and the extrinsic number is
        // reported either way. Only which of them the policy learns from
        // changes here.
        let closed = self.gauge.credit(self.score(self.max_steps).value());
        let r = match self.payment {
            Payment::Shaped => shaped,
            Payment::Gauge => closed,
        };
        self.total += r;
        self.extrinsic += extrinsic;
        if self.state.outcome == "exited" {
            self.exited = true;
        }
        self.opts = action::options(&self.state);
        self.publish(action, r, probs, frames);
        if self.state.done {
            let won = self.exited;
            self.note_outcome(won);
        }
        (r, self.state.done)
    }

    /// One line saying how the episode that just ended ended, and where it
    /// stopped making progress. See [`crate::report`].
    pub fn report(&self) -> String {
        self.progress.report(&self.state.outcome)
    }

    /// How far the episode that just ended actually got, on a scale that is
    /// still readable when it did not finish. See [`crate::report::Score`].
    pub fn score(&self, allowed: u32) -> crate::report::Score {
        self.progress
            .score(self.state.outcome == "exited", allowed)
    }

    /// What the route makes of where the player is standing, asked only when
    /// an episode ended by going nowhere.
    ///
    /// A stall is the one outcome the observation cannot explain on its own:
    /// the route hands out a bearing whatever happens, and a bearing computed
    /// by falling back to a cell six away looks exactly like one that leads
    /// somewhere. This is the engine being asked directly, at the spot where
    /// it went wrong, while the level is still in the state that produced it -
    /// which is not reproducible afterwards by walking back to the same
    /// coordinates, because the doors and lifts have moved since.
    pub fn stall_detail(&mut self) -> Option<String> {
        if !self.progress.stalled() {
            return None;
        }
        Some(match self.doom.route() {
            Ok(j) => j,
            Err(e) => format!("could not be asked: {e}"),
        })
    }

    pub fn start(&mut self, seed: u64) -> String {
        // Check the run that just ended before anything is reset. Paying a
        // decision what it moved the gauge is only worth doing if what the
        // episode was PAID adds up to what it SCORED, and there are paths -
        // an engine that stopped answering, a state that would not parse -
        // that end an episode without paying for the decision that ended it.
        // Those are exactly the episodes whose return is a lie, and silently
        // learning from them is how a reward becomes untrustworthy without
        // anyone noticing.
        if self.payment == Payment::Gauge && self.steps > 0 {
            let scored = self.gauge.paid();
            if (self.total - scored).abs() > 1e-3 {
                eprintln!(
                    "doom: episode {} was paid {:+.4} for a run that scored {scored:.4}",
                    self.episode, self.total
                );
            }
        }
        if self.mission_mix {
            // One policy, several orders: which one this episode runs under is
            // part of what the policy has to read.
            self.mission = Mission::ALL[(seed as usize) % Mission::ALL.len()];
        }
        self.episode = seed;
        // Which level, from the same seed. Mixed differently from the mission
        // so that "map 2" and "survive" do not always arrive together, which
        // would make either one unlearnable from the other.
        let pick = (seed.wrapping_mul(0x9e37_79b9) >> 16) as usize;
        if self.scenarios.is_empty() {
            self.cfg.scenario = None;
            self.cfg.map = self.maps[pick % self.maps.len()];
        } else {
            self.cfg.scenario = Some(self.scenarios[pick % self.scenarios.len()].clone());
        }
        let start = self.curriculum.start();
        let json = match self.doom.reset(&self.cfg, seed, start) {
            Ok(j) => j,
            Err(e) => {
                self.fail(format!("could not restart the level: {e}"));
                return String::new();
            }
        };
        match State::parse(&json) {
            Ok(s) => self.state = s,
            Err(e) => {
                self.fail(e);
                return String::new();
            }
        }
        self.total = 0.0;
        self.extrinsic = 0.0;
        self.floor_damage = 0;
        self.steps = 0;
        self.exited = false;
        self.last_pos = None;
        self.stuck = 0;
        self.stale = 0;
        self.tried.clear();
        self.recent.clear();
        // A new episode is a new world: nothing seen in the last one is
        // anywhere now, least of all on a level that is built fresh.
        self.memory.clear();
        self.approach_from = None;
        self.approach_goal = None;
        self.remember();
        self.visited.clear();
        self.commit = 0;
        self.commit_tag = None;
        self.progress = Progress::new();
        self.gauge = Gauge::new();
        if self.arena > 0 {
            if let Err(e) = self.build_arena(seed) {
                self.fail(e);
                return String::new();
            }
        }
        self.opts = action::options(&self.state);
        let frames = self.grab_frame().into_iter().collect();
        self.publish(0, 0.0, Vec::new(), frames);
        obs::render(&self.state, self.history())
    }

    /// The scripted player this run is measured against, and warm-started from.
    ///
    /// Fight what is in front of you, take what is under your nose, otherwise
    /// go wherever there is the most room, preferring the way the exit lies.
    /// That last clause is the whole of its navigation and it is deliberately
    /// crude - a greedy hill climb with no memory, so it circles a level
    /// rather than solving one, never retreats from a fight it is losing,
    /// never prioritises the enemy that is actually shooting at it, and plans
    /// nothing beyond the next decision. It is meant to be a floor worth
    /// clearing, not a solution.
    ///
    /// It was WORSE than this before: it aimed at the exit whether or not
    /// there was floor between, and spent whole episodes shoving at the wall
    /// in between. A baseline that is merely broken makes the learned number
    /// unreadable - beating it would prove nothing - so it is worth the twenty
    /// lines to make it a real player.
    /// Whether the last several decisions have gone nowhere in aggregate.
    fn circling(&self) -> bool {
        if self.recent.len() < CIRCLE_WINDOW {
            return false;
        }
        let (cx, cy) = *self.recent.back().expect("non-empty");
        self.recent
            .iter()
            .all(|(x, y)| (x - cx).abs() + (y - cy).abs() < CIRCLE_RADIUS)
    }

    /// The scripted player, with one rule wrapped round it: do not keep doing
    /// something that is not working.
    ///
    /// Every version of this teacher has eventually been caught repeating one
    /// action for hundreds of decisions - pressing at a door the `use` ray
    /// misses, strafing into the wall beside the barrel it is trying to get
    /// round, walking at a route it cannot follow. Each was fixed where it
    /// was found, and the next one appeared somewhere else, because the fault
    /// is not in any of those rules. It is that a rule which fires on a
    /// condition, and does not change the condition, fires again.
    ///
    /// So: once the player has stopped moving, each KIND of thing is tried at
    /// most once until it moves again. What that buys is not cleverness, it
    /// is exhaustion - the teacher works through what it has rather than
    /// hammering the first thing on the list.
    /// What the teacher would do here. A QUESTION, not a move.
    ///
    /// Asking used to change the answer to the next question: it recorded the
    /// option as tried and could start a commitment, whether or not the
    /// action was ever executed. DAgger asks at every state the STUDENT
    /// reaches and executes the student's choice instead, so the teacher was
    /// being told it had tried things that never happened - label noise with
    /// no cause but bookkeeping.
    ///
    /// The bookkeeping now advances from what was actually EXECUTED, in
    /// [`DoomEnv::note_executed`], which is what "have I tried this" and "am
    /// I committed to a direction" were always supposed to mean.
    pub fn scripted(&mut self) -> Option<usize> {
        let saved = (self.tried.clone(), self.commit, self.commit_tag);
        let pick = self.scripted_inner();
        self.tried = saved.0;
        self.commit = saved.1;
        self.commit_tag = saved.2;
        pick
    }

    fn scripted_inner(&mut self) -> Option<usize> {
        // Once everything on offer has been tried, start again rather than
        // run out of ideas. Exhaustion is meant to be a rotation, not a
        // one-shot: a door takes one press and thirty tics to rise, during
        // which the player has not moved, so a rule that never presses twice
        // gives up on the door it only just missed.
        if self.opts.iter().all(|o| self.tried.contains(&o.tag)) {
            self.tried.clear();
        }
        let pick = self.choose();
        if let Some(i) = pick {
            if self.stuck >= STUCK_TRY_SOMETHING_ELSE {
                if let Some(o) = self.opts.get(i) {
                    self.tried.insert(o.tag);
                }
            }
        }
        pick
    }

    /// Advance the teacher's own memory from an action that was EXECUTED.
    ///
    /// Whoever chose it. "I have tried the door" is a fact about the run, not
    /// about who was asked, so a student that walks into a wall makes the
    /// teacher's next suggestion account for it exactly as the teacher's own
    /// step would have.
    fn note_executed(&mut self, tag: Tag) {
        if self.opts.iter().all(|o| self.tried.contains(&o.tag)) {
            self.tried.clear();
        }
        if self.stuck >= STUCK_TRY_SOMETHING_ELSE {
            self.tried.insert(tag);
        }
        // A commitment is kept only while it is being followed.
        if self.commit > 0 {
            if self.commit_tag == Some(tag) {
                self.commit -= 1;
            } else {
                self.commit = 0;
                self.commit_tag = None;
            }
        } else if self.circling() && matches!(tag, Tag::Explore | Tag::Advance) {
            self.commit = COMMIT_STEPS;
            self.commit_tag = Some(tag);
        }
    }

    fn choose(&mut self) -> Option<usize> {
        let stuck = self.stuck;
        let tried = &self.tried;
        let by = |tag: Tag| {
            if stuck >= STUCK_TRY_SOMETHING_ELSE && tried.contains(&tag) {
                return None;
            }
            self.opts.iter().position(|o| o.tag == tag)
        };
        let hurt_badly = self.state.player.health < 40;
        let threat_near = self.state.threat_within(600);
        // Close enough that walking past it means taking hits the whole way.
        let in_my_face = self.state.threat_within(300);
        let underfoot = |d: i32| {
            self.state
                .pickups
                .iter()
                .any(|p| p.visible && p.distance < d)
                // One it saw a moment ago counts too, and at a longer reach:
                // the whole point of remembering it is that going back for it
                // is now a thing that can be decided rather than stumbled on.
                || self
                    .state
                    .recalled
                    .iter()
                    .any(|r| {
                        r.class != crate::memory::Class::Threat
                            && r.distance < d * 3
                            && self.state.worth_taking_kind(&r.kind)
                    })
        };

        // Standing in slime: anywhere else will do, and the exit route is
        // already computed to avoid it.
        //
        // Unless there IS nowhere else. On a level whose whole floor burns,
        // "get out of it" has no answer, and taking the branch anyway sent
        // the player in a straight line to the nearest wall to die there -
        // every episode, at decision 102, with the medkits that would have
        // kept it alive still on the floor. Burning floor is only an
        // emergency while somewhere dry exists; otherwise it is the weather,
        // and the ordinary order of business applies.
        if self.state.player.standing_in_damage && self.state.player.dry_land.is_some() {
            // The way OUT of it, when the engine could see one. Taking the
            // route instead is what killed the scripted player on E1M3: the
            // route led across more nukage, because that is where the run was
            // going, and health does not last that long.
            if let Some(i) = by(Tag::Escape) {
                return Some(i);
            }
            if let Some(i) = by(Tag::Exit) {
                return Some(i);
            }
            let best_move = self
                .opts
                .iter()
                .enumerate()
                .filter(|(_, o)| matches!(o.tag, Tag::Advance | Tag::Explore | Tag::Sidestep))
                .max_by_key(|(_, o)| o.room);
            if let Some((i, _)) = best_move {
                return Some(i);
            }
        }

        // A SHUT DOOR IS NOT A DECISION. The route goes through doors because
        // a player opens them, so the only thing to do at one is open it -
        // every other option leads away from the exit, and the movement
        // options are scored on clearance, which a closed door reads as zero.
        // Without this the teacher demonstrated walking into E1M1's first
        // door for as long as the episode lasted.
        // SOMETHING STANDING IN THE WAY, once it has actually stopped the
        // player. The route is over geometry and cannot see who is standing
        // in it, so a barrel in a doorway is a perfectly good route the
        // player cannot walk - but "a solid thing is on the line to the next
        // waypoint" is true constantly in a level full of furniture, and
        // acting on it every time is a player who sidesteps for ever. Not
        // moving for two decisions is what turns it from a fact into a
        // problem.
        if self.stuck >= 2 {
            if let Some(i) = by(Tag::Clear) {
                return Some(i);
            }
        }

        if let Some(i) = by(Tag::Open) {
            return Some(i);
        }

        // NOR IS A SWITCH. Same argument, one step further off: what is in the
        // way opens from somewhere else, so the only thing that makes progress
        // is walking to the switch and pressing it. Every other option leads
        // away from the exit and the movement options read the shut way as
        // zero clearance.
        if let Some(i) = by(Tag::Switch) {
            return Some(i);
        }

        // STAYING ALIVE COMES FIRST, whatever the orders say.
        //
        // Without this the scripted player stands in the open trading shots
        // until it dies - measured, dead at decision 550 with two thirds of
        // the level crossed and the exit in reach. Backing off or sidestepping
        // while hurt is the cheapest possible tactic and it is the difference
        // between a teacher that finishes levels and one that does not.
        if hurt_badly && threat_near {
            if let Some(i) = by(Tag::Grab) {
                if self
                    .state
                    .pickups
                    .iter()
                    .any(|p| p.visible && p.distance < 400)
                {
                    return Some(i);
                }
            }
            // Sidestep rather than back away where there is room: strafing
            // keeps the enemy in front, so the next decision can still shoot.
            if let Some(i) = by(Tag::Sidestep) {
                return Some(i);
            }
            if let Some(i) = by(Tag::Retreat) {
                return Some(i);
            }
        }

        // A lift on the route beats everything the orders rank, because it
        // is not a preference - it is the only way the route goes on. The
        // floor ahead is higher than a player climbs, and walking at it is
        // holding forward against a wall. Pressing use and waiting is the
        // act, and waiting is the part no other option can express.
        if let Some(i) = by(Tag::Ride) {
            if !in_my_face {
                return Some(i);
            }
        }

        // THE ORDERS DECIDE THE ORDER. Until this was mission-aware the
        // scripted player did the same thing whatever it was told, and under
        // `speedrun` that meant collecting sixteen items over four hundred
        // decisions and never once setting off for the exit - so the warm
        // start cloned a player that had never finished a level, and the
        // policy had no demonstration of finishing to learn from.
        let order: &[Tag] = match self.mission {
            Mission::Clear => &[Tag::Attack, Tag::Grab, Tag::Exit],
            Mission::Speedrun => &[Tag::Exit, Tag::Attack, Tag::Grab],
            Mission::Survive => &[Tag::Grab, Tag::Attack, Tag::Exit],
        };

        for tag in order {
            let Some(i) = by(*tag) else { continue };
            let take = match tag {
                Tag::Attack => threat_near,
                Tag::Grab => {
                    hurt_badly
                        || underfoot(if self.mission == Mission::Speedrun {
                            100
                        } else {
                            200
                        })
                }
                // Only when the engine gave a real route. Without one the
                // option aims down a straight line through walls, and taking
                // it is the behaviour this whole exercise exists to stop
                // demonstrating.
                //
                // And not with something shooting at you from close range.
                // Under `speedrun` the exit is first in the order and a route
                // almost always exists, so the exit won every single decision
                // and the teacher crossed the whole of E1M1 with one kill,
                // taking fire the entire way. A speedrunner shoots what is in
                // their face and then runs; anything further off is not worth
                // the ammunition.
                Tag::Exit => {
                    self.state
                        .exit
                        .as_ref()
                        .is_some_and(|e| e.route_bearing.is_some())
                        && !(by(Tag::Attack).is_some() && in_my_face)
                }
                _ => false,
            };
            if take {
                return Some(i);
            }
        }

        // Somewhere to go. Every movement option carries the clearance it was
        // built from, so this is a choice among measured distances rather than
        // a fixed preference order.
        let mut best: Option<(i32, usize)> = None;
        for (i, o) in self.opts.iter().enumerate() {
            if stuck >= STUCK_TRY_SOMETHING_ELSE && tried.contains(&o.tag) {
                continue;
            }
            let score = match o.tag {
                Tag::Exit => o.room + 400,
                Tag::Advance | Tag::Retreat | Tag::Explore => o.room,
                _ => 0,
            };
            // Backing away is a last resort: it is how a greedy walker
            // oscillates in place, one step forward and one step back forever.
            let score = if o.tag == Tag::Retreat {
                score / 4
            } else {
                score
            };
            if score > 0 && best.is_none_or(|(s, _)| score > s) {
                best = Some((score, i));
            }
        }

        // Going round in circles: commit to one direction long enough to leave
        // the cycle behind.
        if self.commit == 0 && self.circling() {
            let away: Vec<(i32, usize)> = self
                .opts
                .iter()
                .enumerate()
                .filter(|(_, o)| matches!(o.tag, Tag::Explore | Tag::Advance))
                .map(|(i, o)| (o.room, i))
                .collect();
            if let Some(&(_, i)) = away.iter().max_by_key(|(room, _)| *room) {
                self.commit = COMMIT_STEPS;
                self.commit_tag = Some(self.opts[i].tag);
                return Some(i);
            }
        }
        if self.commit > 0 {
            self.commit -= 1;
            if let Some(tag) = self.commit_tag {
                if let Some(i) = by(tag) {
                    return Some(i);
                }
            }
            self.commit = 0;
        }

        // Nothing has room: shove at whatever is in the way, then try each way
        // out in turn. Rotating matters - always shoving is right for a door
        // and useless for a wall.
        // Nowhere new for a long time and no route to anywhere either: the
        // level is not saying where to go and walking is not finding out.
        // Try the walls.
        let out_of_ideas = self.stale >= STALE_TRY_THE_WALLS && self.state.exit.is_none();
        if self.stuck >= 3 || out_of_ideas || best.is_none() {
            let mut ways: Vec<usize> = Vec::new();
            if let Some(i) = by(Tag::Use) {
                ways.push(i);
            }
            ways.extend(
                self.opts
                    .iter()
                    .enumerate()
                    .filter(|(_, o)| o.tag == Tag::Explore)
                    .map(|(i, _)| i),
            );
            if let Some(i) = by(Tag::Retreat) {
                ways.push(i);
            }
            if !ways.is_empty() {
                // Rotate on whichever counter opened this branch: shove, then
                // each way out in turn. Always shoving is right for a door
                // and useless for a wall.
                let turn = if self.stuck >= 3 {
                    self.stuck as usize - 3
                } else {
                    self.stale as usize
                };
                return Some(ways[turn % ways.len()]);
            }
        }
        best.map(|(_, i)| i).or_else(|| by(Tag::Use)).or(Some(0))
    }
}

impl Env for DoomEnv {
    /// How far this episode got, for picking which iteration to keep. See
    /// [`crate::report::Score`] for why return is not enough on its own.
    fn progress(&self) -> Option<f32> {
        Some(self.score(self.max_steps).value())
    }

    fn reset(&mut self, seed: u64) -> String {
        self.start(seed)
    }

    fn actions(&mut self) -> Vec<String> {
        self.opts.iter().map(|o| o.text.clone()).collect()
    }

    fn step(&mut self, action: usize) -> (String, f32, bool) {
        let (r, done) = self.apply(action, Vec::new());
        (obs::render(&self.state, self.history()), r, done)
    }

    fn objective(&self) -> String {
        self.mission.instruction().to_string()
    }

    fn render(&self) -> Option<String> {
        Some(format!(
            "hp {:>3} kills {}/{} items {} {}",
            self.state.player.health,
            self.state.level.kills,
            self.state.level.total_kills,
            self.state.level.items,
            self.state.outcome
        ))
    }

    fn won(&self) -> bool {
        self.exited
    }

    fn demo(&mut self) -> Option<usize> {
        self.scripted()
    }

    /// Which generated scenario this episode is, or which of the game's own
    /// levels. What a measurement is broken down BY, since the nine scenarios
    /// ask for entirely different things and an average over them describes
    /// none of them.
    fn label(&self) -> Option<String> {
        match &self.cfg.scenario {
            Some(name) => Some(name.clone()),
            None => Some(format!("E{}M{}", self.cfg.episode, self.cfg.map)),
        }
    }

    /// WHICH world, not what kind of one.
    ///
    /// A scenario is built by the engine from the episode's own seed, so one
    /// scenario name covers as many distinct layouts as there are seeds - and
    /// a trajectory through one of them is a sequence of turns that fits no
    /// other. The game's own levels have fixed geometry and are their own
    /// instance.
    fn instance(&self) -> Option<String> {
        self.cfg
            .scenario
            .as_ref()
            .map(|name| format!("{name}#{}", self.episode))
    }

    /// The reverse curriculum is the one thing here that outlives an episode,
    /// so it is the one thing a measurement can damage. See [`Curriculum`].
    fn set_counting(&mut self, on: bool) {
        self.curriculum.set_counting(on);
    }

    fn hold(&mut self) -> bool {
        self.hold_at(0)
    }

    fn hold_at(&mut self, slot: usize) -> bool {
        if self.doom.snapshot_at(slot).is_err() {
            return false;
        }
        // The client's own half of the state, kept beside the engine's. Half
        // a run restored is worse than none, because it looks like an answer.
        self.slots.insert(slot, Held {
            state: self.state.clone(),
            opts: self.opts.clone(),
            last_pos: self.last_pos,
            stuck: self.stuck,
            stale: self.stale,
            tried: self.tried.clone(),
            recent: self.recent.clone(),
            visited: self.visited.clone(),
            commit: self.commit,
            commit_tag: self.commit_tag,
            total: self.total,
            extrinsic: self.extrinsic,
            floor_damage: self.floor_damage,
            steps: self.steps,
            exited: self.exited,
            progress: self.progress.clone(),
            memory: self.memory.clone(),
            gauge: self.gauge,
            approach_from: self.approach_from,
            approach_goal: self.approach_goal.clone(),
        });
        true
    }

    fn slots(&self) -> usize {
        // What the engine was built with. See api_snapshot.h.
        4096
    }

    /// Where the run is, coarsely, and what it is carrying.
    ///
    /// The grid is 128 units - four of the engine's route cells, about a
    /// corridor and a half - so that shuffling on the spot is the same place
    /// and walking into the next room is not. Keys are in the name because
    /// carrying one makes everywhere reachable with it somewhere new, which
    /// is how a search discovers that keys open doors without being told.
    fn cell(&self) -> Option<String> {
        let p = &self.state.player;
        let (x, y) = (p.x?, p.y?);
        let mut keys = p.keys.clone();
        keys.sort();
        Some(format!(
            "{}:{}:{}:{}",
            self.cfg.map,
            x.div_euclid(128),
            y.div_euclid(128),
            keys.join("+")
        ))
    }

    fn resume(&mut self) -> Option<String> {
        self.resume_from(0)
    }

    fn resume_from(&mut self, slot: usize) -> Option<String> {
        let held = self.slots.get(&slot).cloned()?;
        if self.doom.restore_from(slot).is_err() {
            return None;
        }
        self.state = held.state;
        self.opts = held.opts;
        self.last_pos = held.last_pos;
        self.stuck = held.stuck;
        self.stale = held.stale;
        self.tried = held.tried;
        self.recent = held.recent;
        self.visited = held.visited;
        self.commit = held.commit;
        self.commit_tag = held.commit_tag;
        self.total = held.total;
        self.extrinsic = held.extrinsic;
        self.floor_damage = held.floor_damage;
        self.steps = held.steps;
        self.exited = held.exited;
        self.progress = held.progress;
        self.memory = held.memory;
        self.gauge = held.gauge;
        self.approach_from = held.approach_from;
        self.approach_goal = held.approach_goal;
        Some(obs::render(&self.state, self.history()))
    }
}

/// Everything about a run that lives on THIS side of the socket.
///
/// A snapshot of the engine puts back the level, the player and the monsters.
/// It does not put back what this side has accumulated about the run - what
/// the agent remembers seeing, which patches of floor it has walked, how far
/// along the route it has ever got - and every one of those is read by the
/// observation or by the score. Restoring half of a run would produce an
/// observation no run ever saw and a score no run ever earned, which is a
/// worse failure than not being able to go back at all, because it looks
/// like an answer.
#[derive(Clone)]
struct Held {
    state: State,
    opts: Vec<Option_>,
    last_pos: Option<(i32, i32)>,
    stuck: u32,
    stale: u32,
    tried: std::collections::HashSet<Tag>,
    recent: std::collections::VecDeque<(i32, i32)>,
    visited: std::collections::HashMap<(i32, i32), u32>,
    commit: u32,
    commit_tag: Option<Tag>,
    total: f32,
    extrinsic: f32,
    floor_damage: u32,
    steps: u32,
    exited: bool,
    progress: Progress,
    memory: crate::memory::Memory,
    gauge: Gauge,
    approach_from: Option<i32>,
    approach_goal: Option<String>,
}

/// A state that parses but describes nothing, so the environment has a valid
/// value before its first reset rather than an Option that every reader has to
/// unwrap.
const EMPTY: &str = r#"{"tic":0,"episodeTic":0,"level":{"episode":0,"map":0,"skill":0,"tic":0,
  "kills":0,"totalKills":0,"items":0,"totalItems":0,"secrets":0,"totalSecrets":0},
  "player":{"id":0,"health":0,"armor":0,"x":null,"y":null,"angle":null,"weapon":null,
  "ammo":null,"keys":[]},"threats":[],"hazards":[],"pickups":[],"clearance":{"ahead":0,"right":0,
  "behind":0,"left":0,"aheadRight":0,"aheadLeft":0},"exit":null,"unexplored":null,"events":[],"done":false,
  "outcome":"alive"}"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// Training difficulty must survive being measured.
    ///
    /// The defect: the curriculum window was fed by every episode that ended,
    /// and an episode that ends is all a fixed-block score or a hypothetical
    /// roll-out looks like from in here. A gauge of eight episodes that went
    /// well therefore moved the start 400 units further out on its own, and a
    /// probe did it again for every branch it rolled out - so how hard the
    /// next training episode was depended on how often the run had stopped to
    /// measure itself, and two runs with identical policies and different
    /// gauge budgets were not training on the same task.
    #[test]
    fn measuring_the_policy_leaves_the_curriculum_where_it_found_it() {
        let mut c = Curriculum { on: true, ..Curriculum::default() };
        // Part-way through a window, which is the state a measurement has to
        // preserve: the outcomes already banked as well as the distance.
        c.note(true);
        let before = c.clone();

        c.set_counting(false);
        for _ in 0..CURRICULUM_WINDOW * 2 {
            assert_eq!(c.note(true), None, "a measured episode reported curriculum progress");
        }
        c.set_counting(true);
        assert_eq!(c, before, "a measurement moved the training curriculum");

        // And the training episodes either side of it still advance it, on
        // their own window and nobody else's.
        for _ in 0..CURRICULUM_WINDOW - 2 {
            assert_eq!(c.note(true), None);
        }
        assert_eq!(c.note(true), Some((CURRICULUM_WINDOW, CURRICULUM_STEP)));
    }

    #[test]
    fn closing_on_the_goal_pays_and_nothing_else_does() {
        // One cell closed is one cell paid.
        assert!((cells_closed(Some(1000), Some(968), true) - 1.0).abs() < 1e-6);

        // Standing still pays nothing, at any distance. The discounted form
        // of potential shaping does not have this property - with a negative
        // potential it leaves a residue proportional to the distance, and at
        // 5300 units that residue is worth more than a cell of real progress.
        for d in [100, 1000, 5300] {
            assert_eq!(cells_closed(Some(d), Some(d), true), 0.0);
        }

        // Out and back nets exactly nothing, which is what stops the reward
        // from paying for the two-step cycle greedy navigation falls into.
        let out = cells_closed(Some(1000), Some(900), true);
        let back = cells_closed(Some(900), Some(1000), true);
        assert_eq!(out + back, 0.0);

        // Walking away costs what walking toward pays.
        assert!(cells_closed(Some(900), Some(1000), true) < 0.0);
    }

    #[test]
    fn a_route_that_changed_its_mind_is_not_progress() {
        // The goal moved: the two distances are to different places.
        assert_eq!(cells_closed(Some(4000), Some(100), false), 0.0);

        // The goal is the same but the distance jumped further than anyone
        // could walk in one decision - the route re-planned over ground that
        // had just been seen, and paying for that is paying for the MAP
        // getting better rather than for the player getting closer.
        assert_eq!(cells_closed(Some(4000), Some(100), true), 0.0);
        assert_eq!(cells_closed(Some(100), Some(4000), true), 0.0);

        // Nothing to compare against on the first decision of an episode.
        assert_eq!(cells_closed(None, Some(1000), true), 0.0);
        assert_eq!(cells_closed(Some(1000), None, true), 0.0);
    }

    #[test]
    fn the_empty_state_parses() {
        // It is a const in this file, so nothing else would catch it rotting
        // when the observation schema changes - and it is constructed before
        // any reset, i.e. on every single run.
        State::parse(EMPTY).expect("EMPTY must stay in sync with the schema");
    }

    #[test]
    fn missions_disagree_about_what_is_worth_doing() {
        // If they did not, the instruction in the observation would be
        // decoration and the policy would have nothing to learn from reading
        // it. A kill is worth more than the exit under Clear and far less
        // under Speedrun.
        let clear = Mission::Clear.weights();
        let speed = Mission::Speedrun.weights();
        assert!(clear.kill > speed.kill);
        assert!(speed.exit > clear.exit);
        assert!(Mission::Survive.weights().hurt > clear.hurt);
        for m in Mission::ALL {
            assert_eq!(Mission::parse(m.name()), Some(m));
        }
    }
}
