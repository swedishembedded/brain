// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Why an episode ended where it did.
//!
//! A score tells you an episode was worth +19.39 and nothing about what went
//! wrong in it. The three questions that actually come up while getting a
//! player through a level are: did it finish, if it died what killed it, and
//! if it neither finished nor died, where did it stop making progress. None of
//! them can be answered from the return, and all three can be answered from
//! the states the episode already streamed past - so this watches them go by
//! and keeps the few numbers that carry the answer.
//!
//! What it keeps is deliberately small. The closest the player ever got to the
//! exit and when, so "it stalled at the start" and "it stalled one room short"
//! are different sentences; damage by cause, so a death is attributable; and
//! the patches of floor covered recently, because the failure that dominates
//! at this horizon is not death, it is a walker oscillating between two cells
//! for three hundred decisions while its return ticks up on the exploration
//! bonus.
//!
//! Swedish Embedded AB builds diagnostics into the control systems it
//! delivers, so that a run that went wrong explains itself instead of being
//! reproduced. If your team needs that, you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::collections::{BTreeMap, VecDeque};

use crate::obs::State;

/// Decisions of history the stall check looks at, and how few distinct patches
/// of floor over that stretch count as going nowhere.
///
/// 60 decisions is about twelve seconds of game time and a good deal further
/// than the player can travel without leaving a 128-unit patch, so a run that
/// is genuinely walking somewhere covers many more than four.
const STALL_WINDOW: usize = 60;
const STALL_PATCHES: usize = 4;
/// Side of a patch of floor, matching the exploration bonus's own cell so the
/// two agree about what "somewhere new" means.
const PATCH: i32 = 128;

/// What happened over one episode, accumulated as it runs.
#[derive(Default)]
pub struct Progress {
    steps: u32,
    /// Route distance to the exit at the start, at its best, and last seen.
    start_path: Option<i32>,
    best_path: Option<i32>,
    best_at: u32,
    last_path: Option<i32>,
    /// Health lost per cause, as the engine attributed it.
    damage: BTreeMap<String, i32>,
    /// The blow that landed at zero health.
    killed_by: Option<String>,
    /// Patches of floor over the last [`STALL_WINDOW`] decisions.
    recent: VecDeque<(i32, i32)>,
    pos: (i32, i32),
    health: i32,
    kills: u32,
    total_kills: u32,
    /// What the route said was in the way, at the end.
    blocked: Option<String>,
    /// What the route was leading to: the exit, or a key it needs first.
    goal: Option<String>,
    /// Progress against the goals the route has already finished with.
    reached: Vec<String>,
}

impl Progress {
    pub fn new() -> Progress {
        Progress::default()
    }

    /// Fold in the state the game returned for one decision.
    pub fn note(&mut self, state: &State) {
        self.steps += 1;
        for e in &state.events {
            match e.kind.as_str() {
                "hurt" => {
                    let who = e.what.clone().unwrap_or_else(|| "something unseen".into());
                    *self.damage.entry(who).or_insert(0) += e.amount;
                }
                "death" => self.killed_by = e.what.clone().or(Some("something unseen".into())),
                _ => {}
            }
        }
        if let Some(exit) = &state.exit {
            // The route re-targets mid-episode: it leads to a key while the
            // way out is locked, and to the exit once the key is held. Those
            // are two different distances and folding them together reads as
            // "closed to 32 units and then fell back to 6944", which is a
            // description of success. So the record starts again whenever the
            // goal changes, and what is reported is progress toward whatever
            // the route is currently leading to.
            if exit.goal != self.goal {
                // Rendered before the goal moves on, so the line that is kept
                // names the goal it was measured against.
                self.reached.extend(self.route_line());
                self.goal = exit.goal.clone();
                self.start_path = None;
                self.best_path = None;
            }
            // Only a real route counts. `pathDistance` is absent when the
            // engine could not route at all, and treating that as progress
            // would make an unreachable exit look like the closest approach.
            if let Some(p) = exit.path_distance {
                self.start_path.get_or_insert(p);
                self.last_path = Some(p);
                if self.best_path.is_none_or(|b| p < b) {
                    self.best_path = Some(p);
                    self.best_at = self.steps;
                }
            }
            self.blocked = exit.blocked_by.as_ref().map(|b| b.kind.clone());
        }
        self.pos = (state.player.x.unwrap_or(0), state.player.y.unwrap_or(0));
        self.health = state.player.health;
        self.kills = state.level.kills;
        self.total_kills = state.level.total_kills;
        self.recent
            .push_back((self.pos.0.div_euclid(PATCH), self.pos.1.div_euclid(PATCH)));
        if self.recent.len() > STALL_WINDOW {
            self.recent.pop_front();
        }
    }

    /// Patches of floor covered over the last [`STALL_WINDOW`] decisions.
    fn patches(&self) -> usize {
        self.recent
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
    }

    /// Whether the player spent the end of the episode going nowhere.
    pub fn stalled(&self) -> bool {
        self.recent.len() >= STALL_WINDOW && self.patches() <= STALL_PATCHES
    }

    /// Damage by cause, worst first, as "74 to an IMP, 30 to nukage".
    fn damage_line(&self) -> String {
        let mut by: Vec<(&String, &i32)> = self.damage.iter().collect();
        by.sort_by_key(|(_, n)| -**n);
        by.iter()
            .take(4)
            .map(|(who, n)| format!("{n} to {who}"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// How far the route got, as a sentence, or nothing when there never was
    /// a route to measure against.
    fn route_line(&self) -> Option<String> {
        let (start, best, last) = (self.start_path?, self.best_path?, self.last_path?);
        let goal = match self.goal.as_deref() {
            Some("switch") => "the switch that opens the way".to_string(),
            Some("unexplored") => "unexplored ground".to_string(),
            Some(k) => format!("the {k} key"),
            None => "the exit".to_string(),
        };
        let mut s = if best >= start {
            format!("got no closer to {goal} than the {start} units it started at")
        } else {
            format!(
                "closed on {goal} from {start} units to {best} at decision {}",
                self.best_at
            )
        };
        if last > best + PATCH {
            s.push_str(&format!(", and ended {last} out"));
        }
        Some(s)
    }

    /// One line saying how this episode ended and where it stopped.
    ///
    /// `outcome` is the game's own word for it - "exited", "dead", "alive" -
    /// with "alive" meaning the decision limit ran out first.
    pub fn report(&self, outcome: &str) -> String {
        let head = match outcome {
            "exited" => format!("finished in {} decisions", self.steps),
            "dead" => {
                let who = self.killed_by.as_deref().unwrap_or("something unseen");
                format!(
                    "died at decision {} to {who}, at ({}, {})",
                    self.steps, self.pos.0, self.pos.1
                )
            }
            _ if self.stalled() => format!(
                "stalled: the last {STALL_WINDOW} of {} decisions covered {} patches of floor \
                 around ({}, {})",
                self.steps,
                self.patches(),
                self.pos.0,
                self.pos.1
            ),
            _ => format!("ran out of decisions after {}, still walking", self.steps),
        };
        let mut s = format!(
            "{head}; {} of {} kills, {} health",
            self.kills, self.total_kills, self.health
        );
        for r in self.reached.iter().chain(self.route_line().iter()) {
            s.push_str(&format!("; {r}"));
        }
        if let Some(b) = &self.blocked {
            s.push_str(&format!("; a {b} in the way"));
        }
        if !self.damage.is_empty() {
            s.push_str(&format!("; damage {}", self.damage_line()));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(x: i32, y: i32, path: Option<i32>, events: &str) -> State {
        let p = match path {
            Some(p) => format!(
                r#"{{"kind":"switch","distance":{p},"bearing":0,"clearance":64,
                     "pathDistance":{p},"routeBearing":0,"routeDistance":32,"routeClearance":64}}"#
            ),
            None => "null".into(),
        };
        let json = format!(
            r#"{{"tic":0,"episodeTic":0,"level":{{"episode":1,"map":2,"skill":2,"tic":0,
              "kills":3,"totalKills":41,"items":0,"totalItems":0,"secrets":0,"totalSecrets":0}},
              "player":{{"id":0,"health":37,"armor":0,"x":{x},"y":{y},"angle":0,"weapon":"pistol",
              "ammo":10,"keys":[]}},"threats":[],"hazards":[],"pickups":[],
              "clearance":{{"ahead":0,"right":0,"behind":0,"left":0,"aheadRight":0,"aheadLeft":0}},
              "exit":{p},"unexplored":null,"events":[{events}],"done":false,"outcome":"alive"}}"#
        );
        State::parse(&json).expect("the fixture is a valid state")
    }

    fn goal_state(x: i32, y: i32, path: i32, goal: Option<&str>) -> State {
        let mut s = state(x, y, Some(path), "");
        s.exit.as_mut().expect("the fixture has an exit").goal = goal.map(str::to_string);
        s
    }

    #[test]
    fn a_death_names_what_killed_the_player_and_what_wore_it_down() {
        // The whole point of the attribution: "died at decision 769" is not a
        // diagnosis, "died to an IMP having lost most of its health to nukage"
        // is - the first says fight better, the second says route elsewhere.
        let mut p = Progress::new();
        p.note(&state(
            0,
            0,
            Some(6944),
            r#"{"tic":1,"type":"hurt","what":"nukage","amount":60}"#,
        ));
        p.note(&state(
            64,
            0,
            Some(6800),
            r#"{"tic":2,"type":"hurt","what":"an IMP","amount":20}"#,
        ));
        p.note(&state(
            64,
            0,
            Some(6800),
            r#"{"tic":3,"type":"hurt","what":"an IMP","amount":20},
               {"tic":3,"type":"death","what":"an IMP","amount":1}"#,
        ));
        let r = p.report("dead");
        assert!(r.contains("died at decision 3 to an IMP"), "{r}");
        assert!(r.contains("at (64, 0)"), "{r}");
        assert!(r.contains("60 to nukage"), "{r}");
        assert!(r.contains("40 to an IMP"), "{r}");
        assert!(
            r.contains("closed on the exit from 6944 units to 6800"),
            "{r}"
        );
    }

    #[test]
    fn an_episode_that_went_nowhere_says_so_and_says_where() {
        // The failure that actually dominates at this horizon. An episode
        // that shuttles between two cells for its whole length still earns a
        // healthy return from the exploration bonus and reads, from the score
        // alone, exactly like one that walked half the level.
        let mut p = Progress::new();
        for i in 0..STALL_WINDOW + 40 {
            let x = if i % 2 == 0 { -1420 } else { -1400 };
            p.note(&state(x, 2060, Some(1248), ""));
        }
        assert!(p.stalled());
        let r = p.report("alive");
        assert!(r.contains("stalled"), "{r}");
        assert!(r.contains("patches of floor"), "{r}");
        assert!(
            r.contains("(-1400, 2060)") || r.contains("(-1420, 2060)"),
            "{r}"
        );

        // And a player that keeps moving is not accused of stalling.
        let mut q = Progress::new();
        for i in 0..STALL_WINDOW + 40 {
            q.note(&state(i as i32 * PATCH, 0, Some(1000 - i as i32), ""));
        }
        assert!(!q.stalled());
        assert!(
            q.report("alive").contains("still walking"),
            "{}",
            q.report("alive")
        );
    }

    #[test]
    fn each_goal_the_route_takes_up_is_measured_on_its_own() {
        // The route leads to a key while the way out is locked and to the
        // exit once it is held. Measured across the change as one number, an
        // episode that walked to the key and then set off for a far-away exit
        // reads as "closed to 32 units and then fell back to 6944" - which
        // describes the one thing that went RIGHT as the failure.
        let mut p = Progress::new();
        for d in [2112, 900, 32] {
            p.note(&goal_state(0, 0, d, Some("red")));
        }
        for d in [6944, 6800] {
            p.note(&goal_state(0, 0, d, None));
        }
        let r = p.report("alive");
        assert!(
            r.contains("closed on the red key from 2112 units to 32"),
            "{r}"
        );
        assert!(
            r.contains("closed on the exit from 6944 units to 6800"),
            "{r}"
        );
    }

    #[test]
    fn an_unreachable_exit_is_not_reported_as_progress() {
        // `pathDistance` is absent when the engine could not route. Folding
        // that in as zero would make every unroutable level look finished.
        let mut p = Progress::new();
        p.note(&state(0, 0, None, ""));
        let r = p.report("alive");
        assert!(!r.contains("closed on"), "{r}");
        assert!(
            r.contains("2 of 41 kills") || r.contains("3 of 41 kills"),
            "{r}"
        );
    }
}
