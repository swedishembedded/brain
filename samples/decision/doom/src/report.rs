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
#[derive(Clone, Default)]
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
    /// Every patch of floor entered this episode, not just the recent ones.
    /// Under fair play this is the closest thing to a measure of how much of
    /// the level the run actually saw.
    covered: std::collections::HashSet<(i32, i32)>,
    health: i32,
    kills: u32,
    total_kills: u32,
    /// What the route said was in the way, at the end.
    blocked: Option<String>,
    /// What the route was leading to: the exit, or a key it needs first.
    goal: Option<String>,
    /// Progress against the goals the route has already finished with.
    reached: Vec<String>,
    /// The largest fraction of the way to a real goal this episode ever
    /// covered. See [`Progress::score`].
    toward_best: f32,
    had_route: bool,
    /// What the run has taken off the level so far. Monotone, so it can be
    /// read at any point in the episode and never goes down.
    haul: Haul,
}

/// What surviving the whole horizon is worth to a run that never found
/// anything to walk toward.
///
/// It has to clear two bars at once. Big enough that discovering the way out
/// cannot cost a run more than it is worth - see [`Score::value`], where this
/// is a floor - and small enough that merely lasting never outranks a run
/// that crossed most of a level and died, which is the ordering a person
/// reading the two runs would give.
const LASTING: f32 = 0.2;


/// What a full clear is worth on top of getting out, split the way DOOM's own
/// intermission screen splits it.
///
/// Kills lead because they are the bulk of a level and the thing most likely
/// to kill you back; items and secrets are equal and smaller because a level
/// can hold three of one and forty of the other. They sum to 1.0, so a
/// finished run scores 1.0 for the exit, up to 1.0 more for the clear, and up
/// to 0.25 for walking out in good health.
const KILLS_WORTH: f32 = 0.5;
const ITEMS_WORTH: f32 = 0.25;
const SECRETS_WORTH: f32 = 0.25;

/// What a run took off a level, each as a fraction of what the level held.
///
/// All three are monotone counters, which is what lets them live in a score
/// that has to be computable on a PREFIX - see [`Gauge`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Haul {
    pub kills: f32,
    pub items: f32,
    pub secrets: f32,
}

impl Haul {
    /// The three weighted into one number in `0..=1`.
    fn value(&self) -> f32 {
        (KILLS_WORTH * self.kills + ITEMS_WORTH * self.items + SECRETS_WORTH * self.secrets)
            .clamp(0.0, 1.0)
    }
}

/// How far an episode got, on a scale that still means something when it did
/// not finish.
///
/// Return does not. An episode that crossed nine tenths of a level and then
/// died scores about what one that pressed against a wall for four hundred
/// decisions scores, because almost all of the return is the one payment at
/// the exit that neither of them collected. A training run choosing its best
/// policy by mean return is therefore choosing between numbers that are
/// mostly noise, and two runs of the same level cannot be told apart at all
/// unless one of them happened to finish.
#[derive(Clone, Copy, Debug, Default)]
pub struct Score {
    pub finished: bool,
    /// Fraction of the way to a goal worth walking to - the exit, or the key
    /// or switch that opens it. `None` when the route only ever led to
    /// unexplored ground, which moves every time it is reached and so cannot
    /// be made progress against.
    pub toward: Option<f32>,
    /// Health left, as a fraction of full. Zero if it died.
    pub alive: f32,
    /// Decisions survived, as a fraction of those allowed.
    pub lasted: f32,
    /// Distinct patches of floor entered.
    pub seen: usize,
    /// What the run took off the level: kills, items and secrets, each as a
    /// fraction of what the level held. This is three quarters of what DOOM
    /// itself calls finishing a level, and a score that leaves it out ranks a
    /// run that sprinted past everything level with one that cleared the map.
    pub haul: Haul,
}

impl Score {
    /// One number, for ranking two runs against each other.
    ///
    /// Finishing beats everything, and finishing in good health beats
    /// finishing on fumes. Short of that, what counts is how far along the
    /// way it got, discounted by how close it came to dying: nine tenths of
    /// the way and dead (0.45) still beats a tenth of the way alive (0.10),
    /// which is the ordering a person reading the two runs would give.
    ///
    /// When there was no goal to walk toward, lasting IS the task - that is
    /// the whole of health-gathering - and it is scaled down so that it can
    /// never outrank having actually got somewhere.
    ///
    /// Lasting is a FLOOR on the score, not an alternative to it. The route
    /// leads to unexplored ground until the exit, a key or a switch has
    /// actually been seen, so a run switches from one to the other the moment
    /// it finds the way out - and at that moment it has covered none of the
    /// way to it. Read as alternatives, discovering the way out therefore
    /// COST a run most of its score: measured on E1M7, the same policy scored
    /// 0.48 stopped at 120 decisions and 0.00 allowed 900, having ended
    /// closer to its goal than it did at 120. Under `--reward gauge` that is
    /// a large negative payment for the one decision that found the exit.
    pub fn value(&self) -> f32 {
        // Health is a TIEBREAKER over ground already covered, not a discount
        // on it. Walking most of the way to the exit and then dying does not
        // un-walk it, and failing to finish is already the difference between
        // this branch and the one above.
        //
        // It used to halve the distance covered on death, and that made the
        // score disagree with what a run is supposed to be judged on. A
        // player that never fired a shot, at full health, outranked the same
        // player killing nine of a level's eighty-five: measured on E1M4,
        // 0.14 against 0.09. Anything learning from that number learns to
        // stand still, which is the opposite of the task.
        let condition = 0.9 + 0.1 * self.alive;
        let haul = self.haul.value();
        if self.finished {
            return 1.0 + haul + 0.25 * self.alive;
        }
        let lasting = LASTING * self.lasted;
        let went = self.toward.map_or(lasting, |f| f.max(lasting)) * condition;
        // Halved, so that everything an unfinished run can show for itself
        // together stays under the 1.0 that walking out of the level is worth
        // on its own. A level cleared but not left is not a level finished.
        0.5 * (went + haul)
    }
}

/// The learning signal the gauge itself defines.
///
/// A run is kept or discarded on [`Score::value`], so that is what a decision
/// should be paid for moving. `Score` can score a PREFIX - `toward_best` is a
/// running maximum, `alive` is health now, `lasted` is decisions so far - so
/// the value exists after every decision, and the difference between two
/// consecutive values is what the decision in between was worth.
///
/// Undiscounted, those differences sum to exactly the final gauge:
///
/// ```text
/// sum_t [ M(h_t+1) - M(h_t) ]  =  M(h_T) - M(h_0)  =  M(h_T)
/// ```
///
/// because an episode begins with nothing covered, nothing survived and
/// nothing finished, so `M(h_0)` is zero. That identity is the point of the
/// type: it is an algebraic guarantee that maximising return maximises the
/// gauge, rather than a hope that separately chosen weights for kills, items
/// and floor covered happen to rank two runs the way the gauge ranks them.
/// They did not. Measured on this sample, about 92% of a shaped episode's
/// return was the exploration bonus, which the gauge does not read at all -
/// so every iteration was trained on one quantity and kept on another.
#[derive(Clone, Copy, Debug, Default)]
pub struct Gauge {
    paid: f32,
}

impl Gauge {
    /// One per episode. Starting at zero is what makes the sum come out at
    /// the gauge rather than at the gauge minus wherever the last run ended.
    pub fn new() -> Gauge {
        Gauge::default()
    }

    /// What the decision that arrived at `now` was worth: everything `now` is
    /// worth that has not already been paid for.
    pub fn credit(&mut self, now: f32) -> f32 {
        let moved = now - self.paid;
        self.paid = now;
        moved
    }

    /// Everything credited so far, which is the gauge as of now.
    pub fn paid(&self) -> f32 {
        self.paid
    }
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

            // Kept across goal changes, unlike start_path and best_path: a
            // run that reached the blue key and then set off for the exit has
            // made progress twice, and resetting would throw the first away.
            // Unexplored ground is not a goal in this sense - it moves every
            // time it is reached, so "closed on it" says nothing about how
            // far through the level the run is.
            if self.goal.as_deref() != Some("unexplored") {
                self.had_route = true;
                if let (Some(start), Some(best)) = (self.start_path, self.best_path) {
                    if start > 0 {
                        let f = ((start - best) as f32 / start as f32).clamp(0.0, 1.0);
                        self.toward_best = self.toward_best.max(f);
                    }
                }
            }
        }
        let frac = |got: u32, total: u32| if total == 0 { 1.0 } else { got as f32 / total as f32 };
        // A level with none of a thing counts as having taken all of it: a
        // map with no secrets must not be unclearable.
        self.haul = Haul {
            kills: frac(state.level.kills, state.level.total_kills),
            items: frac(state.level.items, state.level.total_items),
            secrets: frac(state.level.secrets, state.level.total_secrets),
        };
        self.pos = (state.player.x.unwrap_or(0), state.player.y.unwrap_or(0));
        self.health = state.player.health;
        self.kills = state.level.kills;
        self.total_kills = state.level.total_kills;
        let patch = (self.pos.0.div_euclid(PATCH), self.pos.1.div_euclid(PATCH));
        self.covered.insert(patch);
        self.recent.push_back(patch);
        if self.recent.len() > STALL_WINDOW {
            self.recent.pop_front();
        }
    }

    /// How far this episode got, as a thing two runs can be compared on.
    ///
    /// `allowed` is the decision budget the episode was given, so that
    /// "lasted" means a fraction rather than a count.
    pub fn score(&self, finished: bool, allowed: u32) -> Score {
        Score {
            finished,
            toward: self.had_route.then_some(self.toward_best),
            alive: if self.killed_by.is_some() {
                0.0
            } else {
                (self.health as f32 / 100.0).clamp(0.0, 1.0)
            },
            lasted: if allowed == 0 {
                0.0
            } else {
                (self.steps as f32 / allowed as f32).clamp(0.0, 1.0)
            },
            seen: self.covered.len(),
            haul: self.haul,
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

    /// Whatever a run scored, that is what it was paid - no more, no less.
    ///
    /// This is the property the shaped reward did not have and could not be
    /// given by tuning, because it weighs things the gauge never looks at.
    #[test]
    fn a_run_is_paid_exactly_what_it_scored() {
        // The prefix scores of one run, decision by decision: some ground
        // closed, a stretch where nothing moved, damage that took back part
        // of what was covered, then the exit.
        let run = [0.10, 0.22, 0.22, 0.18, 0.31, 1.25];
        let mut g = Gauge::new();
        let paid: f32 = run.iter().map(|&v| g.credit(v)).sum();
        let scored = *run.last().expect("a run has decisions");
        assert!(
            (paid - scored).abs() < 1e-6,
            "the run scored {scored} and the policy was paid {paid}"
        );
        assert!((g.paid() - scored).abs() < 1e-6);

        // The setback has to COST, or the sum could not come out - a reward
        // that clipped it to zero would pay more than the run was worth and
        // the guarantee would be gone.
        let mut g = Gauge::new();
        g.credit(0.22);
        assert!(g.credit(0.18) < 0.0, "losing ground has to cost what gaining it paid");

        // A fresh gauge per episode: one carried over would charge the next
        // run for where the last one finished.
        assert_eq!(Gauge::new().paid(), 0.0);
    }

    #[test]
    fn a_run_that_did_not_finish_still_has_a_number_on_it() {
        // Two runs of the same level, neither of which finished. One crossed
        // most of it and died; the other barely left the spawn and lived.
        // Return puts them within noise of each other; this has to not.
        let nearly = Score {
            finished: false,
            toward: Some(0.9),
            alive: 0.0,
            lasted: 0.7,
            seen: 60,
            haul: Haul::default(),
        };
        let barely = Score {
            finished: false,
            toward: Some(0.1),
            alive: 1.0,
            lasted: 1.0,
            seen: 6,
            haul: Haul::default(),
        };
        assert!(
            nearly.value() > barely.value(),
            "nine tenths of the way and dead beats a tenth of the way alive: {} vs {}",
            nearly.value(),
            barely.value()
        );
    }

    #[test]
    fn finishing_outranks_any_amount_of_getting_close() {
        let finished = Score {
            finished: true,
            toward: Some(1.0),
            alive: 0.01,
            lasted: 1.0,
            seen: 10,
            haul: Haul::default(),
        };
        let close = Score {
            finished: false,
            toward: Some(0.99),
            alive: 1.0,
            lasted: 0.5,
            seen: 200,
            haul: Haul::default(),
        };
        assert!(finished.value() > close.value());
        // And finishing in one piece beats finishing on fumes.
        let healthy = Score {
            alive: 1.0,
            ..finished
        };
        assert!(healthy.value() > finished.value());
    }

    #[test]
    fn with_nowhere_to_walk_to_lasting_is_the_whole_of_the_task() {
        // health-gathering: no exit, so no route to make progress against.
        // Surviving longer is the only thing that separates two runs.
        let long = Score {
            finished: false,
            toward: None,
            alive: 0.8,
            lasted: 1.0,
            seen: 40,
            haul: Haul::default(),
        };
        let short = Score {
            lasted: 0.3,
            alive: 0.0,
            ..long
        };
        assert!(long.value() > short.value());
        // But it cannot outrank a run that actually went somewhere.
        let went = Score {
            finished: false,
            toward: Some(0.95),
            alive: 0.5,
            lasted: 0.5,
            seen: 40,
            haul: Haul::default(),
        };
        assert!(went.value() > long.value());
    }

    /// Playing must beat standing still, or the number teaches standing
    /// still.
    ///
    /// Two runs that got equally far along the way: one killed a tenth of the
    /// level and died for it, the other never fired a shot and finished the
    /// episode untouched. Health used to HALVE the ground already covered, so
    /// the second outranked the first - measured on E1M4, 0.14 against 0.09,
    /// on the very run where the player went from never firing a shot to nine
    /// kills. Dying does not un-walk the distance walked, and not finishing
    /// is already the difference between this and the branch above.
    #[test]
    fn a_run_that_fought_and_died_beats_one_that_stood_still_unharmed() {
        let idle = Score {
            finished: false,
            toward: Some(0.28),
            alive: 1.0,
            lasted: 1.0,
            seen: 12,
            haul: Haul::default(),
        };
        let fought = Score {
            alive: 0.0,
            lasted: 0.6,
            haul: Haul { kills: 0.11, ..Haul::default() },
            ..idle
        };
        assert!(
            fought.value() > idle.value(),
            "standing still outranked playing: idle {:.3}, fought {:.3}",
            idle.value(),
            fought.value()
        );
    }

    /// And health still breaks a tie between two runs that did the same.
    #[test]
    fn health_still_separates_two_runs_that_achieved_the_same() {
        let hurt = Score {
            finished: false,
            toward: Some(0.5),
            alive: 0.1,
            lasted: 1.0,
            seen: 12,
            haul: Haul::default(),
        };
        let whole = Score { alive: 1.0, ..hurt };
        assert!(whole.value() > hurt.value());
    }

    /// Finding the way out must never lower a run's score.
    ///
    /// The route leads to unexplored ground until the exit, a key or a switch
    /// has actually been seen, and at that moment the score stops being
    /// "how long did it last" and becomes "how far along the way is it" -
    /// which is nothing yet. Measured on E1M7: the same policy scored 0.48
    /// stopped at 120 decisions and 0.00 allowed 900, having ended CLOSER to
    /// its goal. Under `--reward gauge`, which pays a decision the difference
    /// between consecutive scores, the one decision that discovered the way
    /// out was paid a large negative reward for it.
    /// The score has to BE the definition of success, or a loop that climbs
    /// it climbs something else. Doom's own definition is the level finished
    /// with every monster killed, every item taken and every secret found -
    /// which is the intermission screen it shows you - so all four have to be
    /// in the number a run is kept on.
    #[test]
    fn a_full_clear_outranks_a_bare_exit() {
        let bare = Score {
            finished: true,
            toward: Some(1.0),
            alive: 1.0,
            lasted: 0.3,
            haul: Haul::default(),
            seen: 40,
        };
        let full = Score {
            haul: Haul { kills: 1.0, items: 1.0, secrets: 1.0 },
            ..bare
        };
        assert!(
            full.value() > bare.value(),
            "clearing the level scored no better than walking out of it: {:.3} vs {:.3}",
            full.value(),
            bare.value()
        );
        assert!(full.value() > 2.0, "a full clear is the top of the scale: {:.3}", full.value());
    }

    /// And killing things is worth something even to a run that never gets
    /// out, or there is no gradient toward clearing a level at all.
    #[test]
    fn what_a_run_cleared_counts_even_when_it_did_not_finish() {
        let empty = Score {
            finished: false,
            toward: Some(0.3),
            alive: 1.0,
            lasted: 0.5,
            haul: Haul::default(),
            seen: 40,
        };
        let fought = Score {
            haul: Haul { kills: 0.8, items: 0.5, secrets: 0.0 },
            ..empty
        };
        assert!(fought.value() > empty.value());
        // But it never reaches a run that actually got out.
        let out = Score { finished: true, ..empty };
        assert!(out.value() > fought.value(), "not finishing outranked finishing");
    }

    #[test]
    fn discovering_the_way_out_never_costs_a_run_anything() {
        let before = Score {
            finished: false,
            toward: None,
            alive: 1.0,
            lasted: 0.5,
            seen: 40,
            haul: Haul::default(),
        };
        // The very next decision, with the exit now in view and none of the
        // way to it covered. Nothing about the run has got worse.
        let after = Score { toward: Some(0.0), ..before };
        assert!(
            after.value() >= before.value(),
            "discovering the goal cost the run {:.3}, going {:.3} -> {:.3}",
            before.value() - after.value(),
            before.value(),
            after.value()
        );
    }

    #[test]
    fn unexplored_ground_is_not_something_to_make_progress_against() {
        // The frontier moves every time it is reached, so "closed on it from
        // 64 units to 0" happens over and over and says nothing about how far
        // through a level a run is.
        let mut p = Progress::new();
        for _ in 0..4 {
            p.note(&goal_state(0, 0, 64, Some("unexplored")));
            p.note(&goal_state(0, 0, 0, Some("unexplored")));
        }
        let s = p.score(false, 100);
        assert!(
            s.toward.is_none(),
            "there was never a goal worth measuring against"
        );
    }

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
