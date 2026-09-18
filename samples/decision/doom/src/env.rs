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

use brain::Env;

use crate::action::{self, Option_, Tag};
use crate::doom::{Config, Doom};
use crate::frame::Frame;
use crate::obs::{self, History, State};

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
            Mission::Clear => Weights { kill: 1.5, item: 0.2, hurt: 0.02, exit: 3.0, explore: 0.10 },
            Mission::Speedrun => {
                Weights { kill: 0.2, item: 0.1, hurt: 0.02, exit: 15.0, explore: 0.20 }
            }
            Mission::Survive => {
                Weights { kill: 0.3, item: 0.4, hurt: 0.08, exit: 5.0, explore: 0.05 }
            }
        }
    }
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
}

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
    /// Reward per step for the episode so far, for the chart.
    pub history: Vec<f32>,
    /// The game's own framebuffer, when frame capture is on. Empty otherwise -
    /// fetching it costs a round trip and 85KB per step, which is most of a
    /// training step, so it is only paid for when somebody is looking.
    pub frame: Frame,
}

pub struct DoomEnv {
    doom: Doom,
    cfg: Config,
    pub mission: Mission,
    /// Fixed for the run, or sampled per episode when training over a mix.
    mission_mix: bool,
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
    steps: u32,
    exited: bool,
    episode: u64,
    pub inspect: Arc<Mutex<Inspect>>,
    /// Fetch the framebuffer with every observation. Off unless something is
    /// drawing it.
    capture_frames: bool,
    warned_dropped: bool,
    /// Set when the game itself failed (the process died, the socket broke).
    /// An environment that silently returns a terminal state on an I/O error
    /// teaches the policy that the error was a legal end to an episode.
    pub fault: Option<String>,
}

impl DoomEnv {
    pub fn new(doom: Doom, cfg: Config, mission: Mission, mission_mix: bool) -> DoomEnv {
        DoomEnv {
            doom,
            cfg,
            mission,
            mission_mix,
            state: State::parse(EMPTY).expect("the empty state is well formed"),
            opts: Vec::new(),
            last_pos: None,
            stuck: 0,
            recent: std::collections::VecDeque::new(),
            visited: std::collections::HashMap::new(),
            commit: 0,
            commit_tag: None,
            total: 0.0,
            extrinsic: 0.0,
            steps: 0,
            exited: false,
            episode: 0,
            inspect: Arc::new(Mutex::new(Inspect::default())),
            capture_frames: false,
            warned_dropped: false,
            fault: None,
        }
    }

    /// Fetch and publish the framebuffer with every step. See
    /// [`Inspect::frame`] for what it costs.
    pub fn capture_frames(&mut self, on: bool) {
        self.capture_frames = on;
    }

    /// What the agent has done recently - the part of the observation the game
    /// does not report. See [`History`].
    /// What the game itself scored this episode, with no exploration bonus.
    pub fn extrinsic(&self) -> f32 {
        self.extrinsic
    }

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

    /// How many times this episode has been in the patch of floor the player
    /// now stands on, counting this visit.
    ///
    /// 128 units to a patch: a corridor's width, so moving to a new patch is a
    /// real change of place and pacing about a room is not.
    fn cell(&self) -> (i32, i32) {
        (
            self.state.player.x.unwrap_or(0).div_euclid(EXPLORE_CELL),
            self.state.player.y.unwrap_or(0).div_euclid(EXPLORE_CELL),
        )
    }

    fn visit(&mut self) -> u32 {
        let cell = self.cell();
        let n = self.visited.entry(cell).or_insert(0);
        *n += 1;
        *n
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

        // The exploration bonus. 1/sqrt(n) rather than first-visit-only so
        // that a patch stays slightly worth revisiting - a strictly one-shot
        // bonus makes a corridor already walked worth exactly nothing, and an
        // agent that has to cross one to reach anything new is being charged
        // for the crossing.
        let n = self.visit();
        r += w.explore / (n as f32).sqrt();
        (r, extrinsic)
    }

    fn publish(&mut self, chosen: usize, reward: f32, probs: Vec<f32>) {
        let frame = if self.capture_frames {
            match self.doom.frame().map_err(|e| e.to_string()).and_then(|j| Frame::parse(&j)) {
                Ok(f) => Some(f),
                Err(e) => {
                    // A frame nobody can draw is a display problem, never a
                    // reason to end an episode - the policy is not reading it.
                    eprintln!("doom: could not read the framebuffer: {e}");
                    self.capture_frames = false;
                    None
                }
            }
        } else {
            None
        };
        if let Ok(mut i) = self.inspect.lock() {
            if let Some(f) = frame {
                i.frame = f;
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
            i.health = self.state.player.health;
            if i.step <= 1 {
                i.history.clear();
            }
            i.history.push(reward);
        }
    }

    /// Take an option and record the outcome. Shared by the SDK's rollout
    /// (through [`Env::step`]) and by this sample's own inspected loop, so the
    /// two cannot disagree about what a step is.
    pub fn apply(&mut self, action: usize, probs: Vec<f32>) -> (f32, bool) {
        let Some(opt) = self.opts.get(action).cloned() else {
            self.fault = Some(format!("the policy chose option {action} of {}", self.opts.len()));
            return (0.0, true);
        };
        let json = match self.doom.step(&opt.commands, opt.tics) {
            Ok(j) => j,
            Err(e) => {
                self.fault = Some(format!("the game stopped answering: {e}"));
                return (0.0, true);
            }
        };
        match State::parse(&json) {
            Ok(s) => self.state = s,
            Err(e) => {
                self.fault = Some(e);
                return (0.0, true);
            }
        }
        self.steps += 1;
        let pos = (self.state.player.x.unwrap_or(0), self.state.player.y.unwrap_or(0));
        // 24 map units is about a third of the player's own width, so anything
        // under it over a whole decision is not movement.
        self.stuck = match self.last_pos {
            Some(p) if (p.0 - pos.0).abs() + (p.1 - pos.1).abs() < 24 => self.stuck + 1,
            _ => 0,
        };
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
        let (r, extrinsic) = self.reward();
        self.total += r;
        self.extrinsic += extrinsic;
        if self.state.outcome == "exited" {
            self.exited = true;
        }
        self.opts = action::options(&self.state);
        self.publish(action, r, probs);
        (r, self.state.done)
    }

    pub fn start(&mut self, seed: u64) -> String {
        if self.mission_mix {
            // One policy, several orders: which one this episode runs under is
            // part of what the policy has to read.
            self.mission = Mission::ALL[(seed as usize) % Mission::ALL.len()];
        }
        self.episode = seed;
        let json = match self.doom.reset(&self.cfg, seed) {
            Ok(j) => j,
            Err(e) => {
                self.fault = Some(format!("could not restart the level: {e}"));
                return String::new();
            }
        };
        match State::parse(&json) {
            Ok(s) => self.state = s,
            Err(e) => {
                self.fault = Some(e);
                return String::new();
            }
        }
        self.total = 0.0;
        self.extrinsic = 0.0;
        self.steps = 0;
        self.exited = false;
        self.last_pos = None;
        self.stuck = 0;
        self.recent.clear();
        self.visited.clear();
        self.commit = 0;
        self.commit_tag = None;
        self.opts = action::options(&self.state);
        self.publish(0, 0.0, Vec::new());
        obs::render(&self.state, self.history())
    }

    /// The scripted player this run is measured against, and warm-started from.
    ///
    /// Fight what is in front of you, take what is under your nose, otherwise
    /// go wherever there is the most room, preferring the way the exit lies.
    /// That last clause is the whole of its navigation and it is deliberately
    /// crude - a greedy hill climb with no memory, so it circles a level
    /// rather than solving one, never retreats from a fight it is losing,
    /// never prioritises the enemy that is actually shooting at it, and does
    /// exactly the same thing whatever the orders say. It is meant to be a
    /// floor worth clearing, not a solution.
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
        self.recent.iter().all(|(x, y)| (x - cx).abs() + (y - cy).abs() < CIRCLE_RADIUS)
    }

    pub fn scripted(&mut self) -> Option<usize> {
        let by = |tag: Tag| self.opts.iter().position(|o| o.tag == tag);
        let hurt_badly = self.state.player.health < 40;

        // Something in range and in sight: shoot it.
        if let Some(i) = by(Tag::Attack) {
            if self.state.visible_threats().next().map(|t| t.distance) < Some(600) {
                return Some(i);
            }
        }
        // Something on the floor worth stepping on.
        if let Some(i) = by(Tag::Grab) {
            if hurt_badly || self.state.pickups.iter().any(|p| p.visible && p.distance < 200) {
                return Some(i);
            }
        }

        // Somewhere to go. Every movement option knows the clearance it was
        // built from, so this is a choice among measured distances rather than
        // a fixed preference order.
        let room = |tag: Tag| -> i32 {
            let c = &self.state.clearance;
            match tag {
                Tag::Advance => c.ahead,
                Tag::Retreat => c.behind,
                _ => 0,
            }
        };
        let mut best: Option<(i32, usize)> = None;
        for (i, o) in self.opts.iter().enumerate() {
            let score = match o.tag {
                Tag::Exit => {
                    // The exit is the goal, so it wins any tie it can reach.
                    self.state.exit.as_ref().map_or(0, |e| e.clearance) + 400
                }
                Tag::Advance | Tag::Retreat => room(o.tag),
                // Left and right are both tagged Explore and their clearances
                // are not distinguishable from the tag, so they are scored on
                // the number in their own text - which is the clearance they
                // were built from. Reading it back is ugly; the alternative is
                // a second copy of the layout in two places.
                Tag::Explore => o
                    .text
                    .split_whitespace()
                    .find_map(|w| w.parse::<i32>().ok())
                    .unwrap_or(0),
                _ => 0,
            };
            // Backing away is a last resort: it is how a greedy walker
            // oscillates in place, one step forward and one step back forever.
            let score = if o.tag == Tag::Retreat { score / 4 } else { score };
            if score > 0 && best.is_none_or(|(s, _)| score > s) {
                best = Some((score, i));
            }
        }

        // Going round in circles: stop chasing the exit for a while and
        // commit to one direction long enough to leave the cycle behind.
        if self.commit == 0 && self.circling() {
            let away: Vec<(i32, usize)> = self
                .opts
                .iter()
                .enumerate()
                .filter(|(_, o)| matches!(o.tag, Tag::Explore | Tag::Advance))
                .map(|(i, o)| {
                    (o.text.split_whitespace().find_map(|w| w.parse::<i32>().ok()).unwrap_or(0), i)
                })
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
            // The kind of move committed to is no longer on offer, so the
            // commitment is over rather than silently ignored.
            self.commit = 0;
        }

        // Nothing has room: shove at whatever is in the way, then try each way
        // out in turn. Rotating matters - the first version always shoved, and
        // shoving is right for a door and useless for a wall.
        if self.stuck >= 3 || best.is_none() {
            let mut ways: Vec<usize> = Vec::new();
            if let Some(i) = by(Tag::Use) {
                ways.push(i);
            }
            ways.extend(
                self.opts.iter().enumerate().filter(|(_, o)| o.tag == Tag::Explore).map(|(i, _)| i),
            );
            if let Some(i) = by(Tag::Retreat) {
                ways.push(i);
            }
            if !ways.is_empty() {
                return Some(ways[(self.stuck.max(3) as usize - 3) % ways.len()]);
            }
        }
        best.map(|(_, i)| i).or_else(|| by(Tag::Use)).or(Some(0))
    }
}

impl Env for DoomEnv {
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
}

/// A state that parses but describes nothing, so the environment has a valid
/// value before its first reset rather than an Option that every reader has to
/// unwrap.
const EMPTY: &str = r#"{"tic":0,"episodeTic":0,"level":{"episode":0,"map":0,"skill":0,"tic":0,
  "kills":0,"totalKills":0,"items":0,"totalItems":0,"secrets":0,"totalSecrets":0},
  "player":{"health":0,"armor":0,"x":null,"y":null,"angle":null,"weapon":null,"ammo":null,
  "keys":[]},"threats":[],"hazards":[],"pickups":[],"clearance":{"ahead":0,"right":0,
  "behind":0,"left":0,"aheadRight":0,"aheadLeft":0},"exit":null,"events":[],"done":false,
  "outcome":"alive"}"#;

#[cfg(test)]
mod tests {
    use super::*;

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
