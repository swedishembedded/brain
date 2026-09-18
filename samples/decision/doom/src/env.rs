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
//! ## The reward is shaped, and says so
//!
//! Kills, items and the exit are what the game scores. Progress toward the
//! exit is shaped in on top, because without it "reach the exit" is a signal
//! that arrives once, thousands of steps after the decisions that earned it,
//! and nothing in between can be learned from. Shaping is potential-based
//! (the change in distance, not the distance) so it adds no optimum the game
//! itself does not have.
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
use crate::obs::{self, State};

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
            Mission::Clear => {
                Weights { kill: 1.5, item: 0.2, hurt: 0.02, exit: 3.0, progress: 0.0005 }
            }
            Mission::Speedrun => {
                Weights { kill: 0.2, item: 0.1, hurt: 0.02, exit: 15.0, progress: 0.004 }
            }
            Mission::Survive => {
                Weights { kill: 0.3, item: 0.4, hurt: 0.08, exit: 5.0, progress: 0.001 }
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
    /// Per map unit of ground made toward the exit.
    progress: f32,
}

/// Paid on death under every mission. Large enough that dying is never the
/// cheap way to end a bad episode, which it is if the only alternative is a
/// slow drip of step cost.
const DEATH: f32 = 5.0;
/// Charged every step, so standing still is never free.
const STEP_COST: f32 = 0.002;

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
    last_exit_dist: Option<i32>,
    total: f32,
    steps: u32,
    exited: bool,
    episode: u64,
    pub inspect: Arc<Mutex<Inspect>>,
    /// Fetch the framebuffer with every observation. Off unless something is
    /// drawing it.
    capture_frames: bool,
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
            last_exit_dist: None,
            total: 0.0,
            steps: 0,
            exited: false,
            episode: 0,
            inspect: Arc::new(Mutex::new(Inspect::default())),
            capture_frames: false,
            fault: None,
        }
    }

    /// Fetch and publish the framebuffer with every step. See
    /// [`Inspect::frame`] for what it costs.
    pub fn capture_frames(&mut self, on: bool) {
        self.capture_frames = on;
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn options(&self) -> &[Option_] {
        &self.opts
    }

    pub fn doom_mut(&mut self) -> &mut Doom {
        &mut self.doom
    }

    /// Reward for what just happened, under the current mission.
    fn reward(&mut self, prev_exit: Option<i32>) -> f32 {
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
        // Potential-based shaping on the distance to the exit.
        if let (Some(p), Some(c)) = (prev_exit, self.state.exit.as_ref().map(|e| e.distance)) {
            r += w.progress * (p - c) as f32;
        }
        r
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
            i.observation = obs::render(&self.state);
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
        let prev_exit = self.last_exit_dist;
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
        self.last_exit_dist = self.state.exit.as_ref().map(|e| e.distance);
        let r = self.reward(prev_exit);
        self.total += r;
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
        self.steps = 0;
        self.exited = false;
        self.last_exit_dist = self.state.exit.as_ref().map(|e| e.distance);
        self.opts = action::options(&self.state);
        self.publish(0, 0.0, Vec::new());
        obs::render(&self.state)
    }

    /// The scripted player this run is measured against, and warm-started from.
    ///
    /// Deliberately simple and deliberately NOT good: shoot what is in front
    /// of you, grab what is under your nose, otherwise walk toward the exit and
    /// shove at whatever stops you. It leaves obvious value on the table - it
    /// never retreats, never prioritises the thing that is actually shooting at
    /// it, and never changes behaviour when the orders change - which is what
    /// makes "did the policy beat it" a real question rather than a formality.
    pub fn scripted(&self) -> Option<usize> {
        let by = |tag: Tag| self.opts.iter().position(|o| o.tag == tag);
        let hurt_badly = self.state.player.health < 40;

        if let Some(i) = by(Tag::Attack) {
            // Only engage what is close enough for a pistol to matter.
            if self.state.visible_threats().next().map(|t| t.distance) < Some(600) {
                return Some(i);
            }
        }
        if hurt_badly {
            if let Some(i) = by(Tag::Grab) {
                return Some(i);
            }
        }
        if let Some(i) = by(Tag::Grab) {
            if self.state.pickups.iter().any(|p| p.visible && p.distance < 200) {
                return Some(i);
            }
        }
        by(Tag::Exit)
            .or_else(|| by(Tag::Advance))
            .or_else(|| by(Tag::Explore))
            .or_else(|| by(Tag::Use))
            .or(Some(0))
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
        (obs::render(&self.state), r, done)
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
