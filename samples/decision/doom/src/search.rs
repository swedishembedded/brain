// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Discovery, as a program with no model in it.
//!
//! This is the SEARCH half of `SEARCH -> VERIFY -> SELECT -> COMPRESS`, split
//! off from the training pipeline it used to be a phase inside. The split is
//! not tidiness. Policy improvement is a compression mechanism: it makes a
//! better copy of behaviour already present in its data and has no mechanism
//! for finding behaviour that is not. The sample's own campaign measured that
//! directly - cloning converged on a teacher that dies on E1M1 with 6 of 29
//! kills, and no amount of it produced a run that finished a level. Reaching
//! a strategy nobody demonstrated is a different problem, and this is the
//! program that works on it.
//!
//! It costs no forward pass and loads no encoder, so it runs at engine speed:
//! measured on this box, about 19 ms a decision against roughly 100 ms with a
//! network in the loop.
//!
//! ## The contract that makes the result honest
//!
//! > **Search may use snapshots. The artifact may not.**
//!
//! Returning to a promising position by restoring an engine snapshot is what
//! makes this affordable - a replay of three hundred actions costs three
//! hundred steps every time, a snapshot costs one request. But a trajectory
//! assembled out of restores is not a run anybody could play. So what a
//! campaign PRODUCES is an action list from the level's own start, and it is
//! not in the solution set until that list has been replayed from the start
//! and has reproduced the same kills, secrets, exit and tic count. The engine
//! is deterministic - measured, two separate process launches on one seed
//! agree bit for bit - which is what makes that a real gate rather than a
//! hope.
//!
//! ## What it optimises
//!
//! [`crate::report::Bar::UvMax`]: every monster, every secret, then the exit.
//! There is no time term in that score on purpose (see its own docs); time is
//! the archive's tiebreaker instead, so the campaign drives the tic count
//! down everywhere at once without ever paying an agent to end its own run.

use std::time::{Duration, Instant};

use brain::decision::Rng;
use brain::search::{Admission, Allocator, Archive, Cascade, Gain, Niche, Rung, Verdict, Worth};
use brain::Env;

use crate::env::DoomEnv;

/// How close two achievements have to be before the archive calls them equal
/// and lets the tic count decide.
///
/// The UV-Max bar moves in steps of `0.8/total_kills` per monster - about
/// 0.027 on E1M1 and 0.0045 on E1M6 - so a tolerance well below the smallest
/// of those keeps two genuinely different achievements apart while absorbing
/// float noise on the route term.
const SAME_ACHIEVEMENT: f32 = 1e-4;

/// How often an exploring walk repeats what it just did.
///
/// "To help explore in a consistent direction, the probability of repeating
/// the previous action is 95% for Atari and 90% for robotics" - Go-Explore,
/// Methods. Without it, uniform random per step is a random walk: it covers
/// distance like the square root of the steps taken, so a hundred steps go
/// almost nowhere and the archive stops growing.
const REPEAT: f32 = 0.95;

/// One way of producing a candidate.
///
/// Operators differ in only two things - how long they act for, and how often
/// they take the scripted player's choice rather than a random one - and that
/// is deliberate: a table of two numbers is something a campaign can be
/// honest about, where four hand-written walk functions would drift into four
/// slightly different definitions of what a walk is.
///
/// Which of them is worth running is [`Allocator`]'s problem, measured on
/// archive gain per second. It is NOT settled here, and the first campaign
/// showed why it cannot be: over 180 s on E1M1 the three original operators
/// scored 3.75, 3.57 and 3.70 gain/s - indistinguishable, because all three
/// walked 60 steps and 60 steps is not long enough to accomplish anything in
/// a level full of monsters. Measured against that campaign, the scripted
/// player playing straight through killed a monster every 244 decisions while
/// the search managed one every 960: the search was four times LESS
/// productive per step than simply playing. `commit` and `chase` are the
/// response - long stretches of real play resumed from a cell the archive
/// already reached, which is the thing a plain teacher run cannot do.
struct Operator {
    name: &'static str,
    /// Decisions to take after returning to a cell.
    walk: usize,
    /// How often to take the scripted player's choice. Below 1.0 the walk
    /// wanders off what the teacher would do, which is the only way to reach
    /// something no teacher demonstrates; at 1.0 it is the teacher continuing
    /// from somewhere the teacher could never have got to on its own.
    guided: f32,
    /// Set off from the best cell the archive holds rather than from a drawn
    /// one. What makes an operator a LOCAL search on the best thing found so
    /// far instead of a sample of the frontier.
    from_best: bool,
    /// How often to press use against whatever is in front of the player,
    /// ahead of anything else.
    ///
    /// The mechanism that finds SECRETS, and there is no other one. A DOOM
    /// secret is a wall that opens when pushed and looks very nearly like a
    /// wall that does not; nothing in the observation says which, and saying
    /// which would be handing the agent the answer. What a player does is
    /// push on things, so this does too. Measured before it existed, every
    /// campaign on E1M1 found zero secrets - which makes the Max category
    /// unreachable no matter how well the fighting goes.
    press: f32,
}

const OPERATORS: [Operator; 5] = [
    // Blind, and blind in a level full of things that shoot back is mostly
    // dead - but it is the only operator that can produce an action no
    // teacher and no policy would ever pick.
    Operator { name: "wander", walk: 60, guided: 0.0, from_best: false, press: 0.0 },
    // Starts from somewhere plausible and wanders off it.
    Operator { name: "probe", walk: 60, guided: 0.85, from_best: false, press: 0.0 },
    // A long stretch of real play from a drawn cell. Long enough to finish a
    // firefight, clear a room and walk into the next one.
    Operator { name: "commit", walk: 400, guided: 1.0, from_best: false, press: 0.0 },
    // The same, from the furthest-along cell there is: pushing the front of
    // the search forward rather than filling in behind it.
    Operator { name: "chase", walk: 400, guided: 1.0, from_best: true, press: 0.0 },
    // Walk about pressing on everything. Half the decisions are a push, the
    // rest are the teacher moving on, which together is a player running
    // their shoulder along the walls of a room - the only way a secret is
    // ever found by someone who has not been told where it is.
    Operator { name: "frisk", walk: 200, guided: 1.0, from_best: false, press: 0.5 },
];

/// What a search campaign found, and what it cost.
pub struct Found {
    pub cells: usize,
    pub best: f32,
    pub best_tics: u64,
    /// Verified UV-Max runs: replayed from the level's own start and confirmed.
    pub solved: Vec<Solution>,
}

/// A trajectory that was replayed from the level start and reproduced what it
/// claimed. The durable output of a campaign.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Solution {
    /// `E1M1`, or the generated scenario's name.
    pub level: String,
    pub skill: u32,
    /// The option TEXT chosen at each decision, from the level's own start.
    ///
    /// Text and not an index. On an exact replay the indices would be stable,
    /// because the same state rebuilds the same option list - but only for as
    /// long as nothing upstream changes how options are built, and the day
    /// that changes a stored index silently means a different action. Text is
    /// self-checking: a replay that cannot find the sentence it is looking for
    /// says so and fails, instead of quietly playing something else.
    pub actions: Vec<String>,
    /// Game tics the run took, at 35 to the second. What a speedrun is scored
    /// on, and what the archive minimises.
    pub tics: u64,
    pub kills: u32,
    pub total_kills: u32,
    pub secrets: u32,
    pub total_secrets: u32,
    pub score: f32,
}

impl Solution {
    pub fn seconds(&self) -> f64 {
        self.tics as f64 / 35.0
    }

    /// `2:14.83`, the way a runner writes a time.
    pub fn clock(&self) -> String {
        let s = self.seconds();
        format!("{}:{:05.2}", (s / 60.0) as u64, s % 60.0)
    }
}

/// How a cell was reached: the actions taken since its PARENT cell, and which
/// cell that was.
///
/// Stored as a chain rather than as a whole trajectory per cell. A campaign
/// holds thousands of cells and a trajectory runs to thousands of decisions;
/// keeping a full action list in each would be hundreds of megabytes of
/// almost entirely duplicated prefix. Walking the chain rebuilds the full list
/// when one is actually needed, which is only ever at verification.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Trail {
    /// The cell this one was reached from. `None` for the level's own start.
    pub from: Option<Niche>,
    /// Which VERSION of that cell, so a stale chain is caught rather than
    /// reconstructed.
    ///
    /// The archive replaces a cell's contents whenever a better way of
    /// reaching it turns up. These steps were recorded after resuming the
    /// version that was there at the time, and they only continue THAT state:
    /// splice them onto a different prefix and the sequence describes nothing
    /// that ever happened. The replay gate would catch it - that is what the
    /// gate is for - but only after paying for a whole episode to find out,
    /// and a rehydration would burn one of its bounded attempts on a trail
    /// that cannot work.
    #[serde(default)]
    pub from_generation: u32,
    /// What was done after resuming there.
    pub steps: Vec<String>,
}

/// The whole action list from the level's start, by walking the chain back.
///
/// Returns `None` on a chain that does not terminate at the start cell. That
/// is not paranoia: an elite is replaced whenever a better route to its cell
/// turns up, and nothing stops the better route passing THROUGH a cell whose
/// own parent is the one being replaced - which closes a loop. A trail that
/// loops would replay forever, so it is refused and the candidate is dropped.
pub fn full_trail(archive: &Archive<Trail>, at: &Niche) -> Option<Vec<String>> {
    let mut seen = std::collections::HashSet::new();
    let mut chain: Vec<&Trail> = Vec::new();
    let mut here = Some(at.clone());
    let mut want: Option<u32> = None;
    while let Some(n) = here {
        if !seen.insert(n.clone()) {
            return None;
        }
        if chain.len() > MAX_CHAIN {
            return None;
        }
        let e = archive.get(&n)?;
        // The link is only good against the version of this cell that the
        // child was recorded after. See `Trail::from_generation`.
        if want.is_some_and(|g| g != e.generation) {
            return None;
        }
        chain.push(&e.what);
        want = Some(e.what.from_generation);
        here = e.what.from.clone();
    }
    let mut out = Vec::new();
    for t in chain.iter().rev() {
        out.extend(t.steps.iter().cloned());
    }
    Some(out)
}

/// How many cells read back off disk a campaign will pay to make returnable
/// again.
///
/// Rehydrating one means REPLAYING its trail from the level's own start, at
/// engine speed - for a cell deep in a level that is upwards of a thousand
/// decisions, tens of seconds. Unbounded, a reloaded campaign would spend its
/// whole budget rebuilding snapshots it then draws from twice. Bounded, it
/// resumes the most promising handful and re-finds the rest, which it is
/// quite good at: the trails that were not rehydrated are still in the
/// archive and still feed the compression phase.
const REHYDRATIONS: usize = 24;

/// A hard cap on how long a reconstructed trail may be, in LINKS.
///
/// A campaign that runs for hours can chain a very long way, and a trail
/// longer than any episode could execute is not a solution however it was
/// assembled.
const MAX_CHAIN: usize = 20_000;

/// One search campaign against one loaded level.
pub struct Campaign {
    archive: Archive<Trail>,
    alloc: Allocator,
    rng: Rng,
    /// The niche the level's own start sits in - where every chain terminates
    /// and the one cell that may have no parent.
    start: Option<Niche>,
    /// The episode seed every replay has to use. A trail is only a way back
    /// to a cell on the level it was walked on.
    seed: u64,
    steps: usize,
    restores: usize,
    refused: usize,
    /// Cells whose engine snapshot exists IN THIS PROCESS.
    ///
    /// An archive read back off disk carries trails, not snapshots - a
    /// snapshot lives in the engine, and the engine that held it has exited.
    /// So a reloaded cell is a place the search knows how to reach and cannot
    /// yet return to, and the difference has to be tracked or `resume_from`
    /// silently restores whatever the slot held last, which is some other
    /// cell's state. That is the worst possible failure for a search: it
    /// explores the wrong place and reports the right one.
    live: std::collections::HashSet<Niche>,
    /// How many more reloaded cells may be paid for. See [`Campaign::go_to`].
    rehydrations: usize,
    /// Completed categories noticed during a walk, waiting to be replayed.
    ///
    /// Queued rather than verified on the spot because verifying RESTARTS the
    /// level, and doing that in the middle of a walk would throw away the rest
    /// of the walk's budget.
    claims: Vec<Claim>,
}

impl Campaign {
    pub fn new(slots: usize, seed: u64) -> Campaign {
        Campaign {
            archive: Archive::new(slots, SAME_ACHIEVEMENT),
            alloc: Allocator::new(&OPERATORS.map(|o| o.name)),
            rng: Rng::new(seed),
            start: None,
            seed,
            steps: 0,
            restores: 0,
            refused: 0,
            live: std::collections::HashSet::new(),
            rehydrations: REHYDRATIONS,
            claims: Vec::new(),
        }
    }

    /// Read an archive back off disk, if there is one.
    ///
    /// Its cells carry trails and no snapshots; see [`Campaign::live`]. What
    /// this buys is a campaign that COMPOUNDS - without it, every run searches
    /// from the level's front door again and a solution found once can be
    /// quietly lost.
    fn carry_in(&mut self, path: &str) {
        let Ok(text) = std::fs::read_to_string(path) else {
            return;
        };
        match Archive::<Trail>::from_json(&text) {
            Ok(was) => {
                println!(
                    "  carried in {} cells from {path}, best {:.3} - up to {} of them \
                     will be walked back to, the rest are trails only",
                    was.len(),
                    was.best().map(|e| e.worth.reached).unwrap_or(0.0),
                    REHYDRATIONS
                );
                self.archive = was;
            }
            // Said out loud rather than started fresh. An archive that will
            // not parse is a campaign's whole history, and silently replacing
            // it with an empty one is how a week of searching disappears.
            Err(e) => println!("  {path} will not be read ({e}); searching from nothing"),
        }
    }

    fn carry_out(&self, path: &str) {
        if let Some(dir) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        match self.archive.to_json().and_then(|t| {
            std::fs::write(path, t).map_err(|e| format!("{path}: {e}"))
        }) {
            Ok(()) => println!("  archive: {} cells to {path}", self.archive.len()),
            Err(e) => println!("  the archive could NOT be written: {e}"),
        }
    }

    /// Stand the level up and file the spawn, so the first draw has somewhere
    /// to go. Without this the archive is empty and every operator returns
    /// having done nothing.
    fn seed_start(&mut self, env: &mut DoomEnv, seed: u64, allowed: u32) -> Result<(), String> {
        env.reset(seed);
        if let Some(f) = env.fault() {
            return Err(format!("the engine faulted before the search began: {f}"));
        }
        let Some(cell) = env.cell() else {
            return Err("the engine would not say where the player is".into());
        };
        let worth = Worth::new(env.score(allowed).value(), env.cost());
        if !env.hold_at(0) {
            return Err("the engine would not hold the level's starting state".into());
        }
        self.archive.offer(cell.clone(), worth, Trail::default());
        // The archive assigns the slot; the engine is then told to hold that
        // slot. Doing it the other way round - hold first, file second - is
        // how a cell ends up pointing at another cell's snapshot.
        let slot = self.archive.get(&cell).map(|e| e.slot).unwrap_or(0);
        if !env.hold_at(slot) {
            self.archive.remove(&cell);
            return Err("the engine would not hold the level's starting state".into());
        }
        self.start = Some(cell);
        Ok(())
    }

    /// File a state the search has just reached, holding the engine snapshot
    /// in whatever slot the archive gave it.
    ///
    /// A cell the engine will not hold is REMOVED rather than kept: an
    /// archive entry whose snapshot does not exist is one that returns to
    /// some other cell's state and then reports having explored this one.
    fn file(
        &mut self,
        env: &mut DoomEnv,
        cell: Niche,
        worth: Worth,
        trail: Trail,
    ) -> Admission {
        let verdict = self.archive.offer(cell.clone(), worth, trail);
        if verdict == Admission::Rejected {
            return verdict;
        }
        let Some(slot) = self.archive.get(&cell).map(|e| e.slot) else {
            return Admission::Rejected;
        };
        if !env.hold_at(slot) {
            self.archive.remove(&cell);
            return Admission::Rejected;
        }
        self.live.insert(cell);
        verdict
    }

    /// Return to a cell, rebuilding its snapshot first if this process never
    /// held one.
    ///
    /// A refusal is reported and the cell is DROPPED, never papered over by
    /// starting the level again: a search that quietly restarts is one that
    /// explores the opening a thousand times and says it resumed.
    fn go_to(&mut self, env: &mut DoomEnv, at: &Niche, slot: usize, allowed: u32) -> bool {
        if self.live.contains(at) {
            if env.resume_from(slot).is_some() {
                return true;
            }
            self.refused += 1;
            self.live.remove(at);
            self.archive.remove(at);
            return false;
        }
        // Read back off disk. The only way to stand where it stands is to
        // walk there again from the level's own start.
        if self.rehydrations == 0 {
            return false;
        }
        self.rehydrations -= 1;
        let Some(actions) = full_trail(&self.archive, at) else {
            self.archive.remove(at);
            return false;
        };
        if replay(env, self.seed, &actions, allowed).is_err() || !env.hold_at(slot) {
            self.archive.remove(at);
            return false;
        }
        self.live.insert(at.clone());
        true
    }

    /// Run one operator once and report what it bought.
    ///
    /// Every operator has the same shape - go back somewhere, act for a
    /// while, file what was reached - and differs only in WHERE it returns to
    /// and HOW it chooses each action. Keeping that shape in one place is
    /// what stops three operators drifting into three slightly different
    /// definitions of what a walk is.
    fn operate(&mut self, env: &mut DoomEnv, arm: usize, allowed: u32) -> Gain {
        let began = Instant::now();
        let mut gain = Gain::default();
        let op = &OPERATORS[arm];

        let picked = if op.from_best {
            self.archive.best().map(|e| (e.niche.clone(), e.slot, e.generation))
        } else {
            self.archive
                .pick(&mut self.rng)
                .map(|e| (e.niche.clone(), e.slot, e.generation))
        };
        let Some((from, slot, from_gen)) = picked else {
            gain.seconds = began.elapsed().as_secs_f64();
            return gain;
        };
        if !self.go_to(env, &from, slot, allowed) {
            gain.seconds = began.elapsed().as_secs_f64();
            return gain;
        }
        self.restores += 1;

        let mut steps: Vec<String> = Vec::new();
        let mut last: Option<String> = None;
        for _ in 0..op.walk {
            let options = env.actions();
            if options.is_empty() {
                break;
            }
            let chose = self.choose(env, op, &options, last.as_deref());
            last = Some(options[chose].clone());
            steps.push(options[chose].clone());
            let (_, _, done) = env.step(chose);
            self.steps += 1;
            if env.fault().is_some() {
                break;
            }
            if let Some(cell) = env.cell() {
                let worth = Worth::new(env.score(allowed).value(), env.cost());
                let trail = Trail {
                    from: Some(from.clone()),
                    from_generation: from_gen,
                    steps: steps.clone(),
                };
                // What the archive's best was BEFORE this admission, so a
                // cell that advances the frontier is credited against the old
                // frontier rather than against itself.
                let top = self.archive.best().map(|e| e.worth.reached).unwrap_or(0.0);
                match self.file(env, cell, worth, trail) {
                    Admission::Fresh => gain.admitted(true, worth.reached, top),
                    Admission::Improved => gain.admitted(false, worth.reached, top),
                    Admission::Rejected => {}
                }
            }
            if done {
                let ended = env.score(allowed);
                if ended.is_uvmax() {
                    // The prefix that reached the cell this walk resumed at,
                    // plus what the walk itself did. Built from the RESUMED
                    // cell rather than from the one just filed: the archive
                    // may have refused the terminal cell in favour of an
                    // equally good one, and reading the trail back out would
                    // then return a different trajectory than the one that
                    // actually just finished the level.
                    if let Some(mut actions) = full_trail(&self.archive, &from) {
                        actions.extend(steps.iter().cloned());
                        self.claims.push(Claim {
                            actions,
                            claimed: ended,
                            claimed_tics: env.cost(),
                        });
                    }
                }
                break;
            }
        }
        gain.seconds = began.elapsed().as_secs_f64();
        gain
    }

    /// One progress line: what the archive holds and what it has verified.
    ///
    /// Coverage per axis rather than a cell count on its own, because a cell
    /// count cannot tell a search that is opening new regions from one stuck
    /// in a room producing a new square of floor every few steps. The axes
    /// are the niche's own: map, x, y, keys, monsters left, secrets found.
    fn say(&self, level: &str, elapsed: Duration, solved: &[Solution]) {
        let cover = self
            .archive
            .coverage()
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join("/");
        let best = self.archive.best();
        println!(
            "  {level} {:>5.0}s: {} cells (cover {}), best {:.3} in {} tics,              {} steps, {} resumes, {} verified",
            elapsed.as_secs_f64(),
            self.archive.len(),
            cover,
            best.map(|e| e.worth.reached).unwrap_or(0.0),
            best.map(|e| e.worth.cost).unwrap_or(0),
            self.steps,
            self.restores,
            solved.len()
        );
    }

    /// Which option this operator takes next.
    fn choose(
        &mut self,
        env: &mut DoomEnv,
        op: &Operator,
        options: &[String],
        last: Option<&str>,
    ) -> usize {
        if op.press > 0.0 && self.rng.next_f32() < op.press {
            if let Some(i) = env.use_option().filter(|i| *i < options.len()) {
                return i;
            }
        }
        if op.guided > 0.0 && self.rng.next_f32() < op.guided {
            if let Some(i) = env.demo().filter(|i| *i < options.len()) {
                return i;
            }
        }
        // An option list rebuilt every step has no stable index, so "the same
        // action again" is matched on the TEXT up to the number in it:
        // "walk forward, 320 units" and "walk forward, 288 units" are the
        // same intention.
        if let Some(prev) = last.filter(|_| self.rng.next_f32() < REPEAT) {
            if let Some(i) = same_again(prev, options) {
                return i;
            }
        }
        (self.rng.next_u64() as usize) % options.len()
    }
}

/// The option in `options` that means what `prev` meant, if any.
///
/// Matched on the words rather than the whole sentence, because the numbers
/// in an option are recomputed every step: the distance to a monster changes
/// while the intention "attack that monster" does not.
fn same_again(prev: &str, options: &[String]) -> Option<usize> {
    let words = |s: &str| {
        s.split_whitespace()
            .filter(|w| !w.chars().next().is_some_and(|c| c.is_ascii_digit()))
            .map(str::to_string)
            .collect::<Vec<_>>()
    };
    let want = words(prev);
    options.iter().position(|o| words(o) == want)
}

/// Replay an action list from the level's own start and report what actually
/// happened.
///
/// The expensive rung of the cascade, and the only one whose answer is worth
/// anything to a human: everything above it reasons about states reached by
/// restoring snapshots, which is not a run anybody could play. This is a
/// whole episode against the engine, from the level's front door, taking only
/// the actions in the list.
///
/// An option the list names and the game does not offer is a HARD failure,
/// never a skip. The list is a claim about what the level does; a replay that
/// silently walks past a missing option is one that verifies a different
/// trajectory and reports success.
pub fn replay(
    env: &mut DoomEnv,
    seed: u64,
    actions: &[String],
    allowed: u32,
) -> Result<crate::report::Score, String> {
    env.reset(seed);
    if let Some(f) = env.fault() {
        return Err(format!("the engine faulted at the level's start: {f}"));
    }
    for (i, want) in actions.iter().enumerate() {
        let options = env.actions();
        let Some(chose) = options.iter().position(|o| o == want) else {
            return Err(format!(
                "at decision {i} the game did not offer {want:?}; it offered {} options",
                options.len()
            ));
        };
        let (_, _, done) = env.step(chose);
        if let Some(f) = env.fault() {
            return Err(format!("the engine faulted at decision {i}: {f}"));
        }
        if done {
            break;
        }
    }
    Ok(env.score(allowed))
}

/// What a candidate solution claims, for the cascade to check.
pub struct Claim {
    pub actions: Vec<String>,
    /// What the SEARCH believed, reached by restoring snapshots.
    pub claimed: crate::report::Score,
    pub claimed_tics: u64,
}

/// Rung one: is this even a claim worth paying to check?
///
/// Free, and it refuses the overwhelming majority. A campaign files thousands
/// of cells and almost none of them are a completed category; replaying every
/// one of them from the level start would be the entire budget.
struct WorthChecking {
    /// The best verified time so far. A claim no faster than this is not
    /// worth an episode, however complete it is.
    best: u64,
}

impl Rung<Claim> for WorthChecking {
    fn name(&self) -> &str {
        "is a uv-max, and faster than the best verified one"
    }
    fn check(&mut self, c: &Claim) -> Verdict {
        if !c.claimed.is_uvmax() {
            return Verdict::Reject("not a complete category".into());
        }
        if c.claimed_tics >= self.best {
            return Verdict::Reject(format!(
                "{} tics is no better than the verified {}",
                c.claimed_tics, self.best
            ));
        }
        Verdict::Pass
    }
}

/// Run a campaign against the level the environment is configured for.
///
/// `budget` is wall clock, because that is the resource a search actually
/// spends and the only one the allocator can compare its operators on.
pub fn campaign(
    env: &mut DoomEnv,
    seed: u64,
    budget: Duration,
    allowed: u32,
    every: Duration,
    archive: Option<&str>,
) -> Result<Found, String> {
    let mut run = Campaign::new(env.slots(), seed);
    // The archive comes in BEFORE the level is stood up, so the spawn cell is
    // filed against whatever was carried in rather than into an empty map.
    if let Some(path) = archive {
        run.carry_in(path);
    }
    run.seed_start(env, seed, allowed)?;
    let level = env.label().unwrap_or_else(|| "the level".into());
    println!(
        "doom: searching {level} for {:.0}s, {} slots, operators: {}",
        budget.as_secs_f64(),
        env.slots(),
        OPERATORS.map(|o| o.name).join(", ")
    );

    let mut cascade: Cascade<Claim> = Cascade::new();
    cascade.push(Box::new(WorthChecking { best: u64::MAX }));

    let began = Instant::now();
    let mut said = Instant::now();
    let mut solved: Vec<Solution> = Vec::new();
    let mut verified_best = u64::MAX;

    while began.elapsed() < budget {
        let arm = run.alloc.choose();
        let gain = run.operate(env, arm, allowed);
        run.alloc.credit(arm, gain);

        // A completed category is checked the moment it is claimed, not at
        // the end: the archive keeps one elite per niche, so a slower
        // solution found later can replace the record of a faster one that
        // was never written down.
        for claim in std::mem::take(&mut run.claims) {
            // The cascade's cheap rung reads the running best, which moves as
            // solutions are verified. Rebuilt rather than mutated so the bar
            // a claim is judged against is never stale.
            let mut ladder: Cascade<Claim> = Cascade::new();
            ladder.push(Box::new(WorthChecking { best: verified_best }));
            if let Verdict::Reject(_) = ladder.admit(&claim) {
                continue;
            }
            match replay(env, seed, &claim.actions, allowed) {
                Ok(got) if got.is_uvmax() => {
                    let l = env.level_counts();
                    let s = Solution {
                        level: level.clone(),
                        skill: env.skill(),
                        actions: claim.actions.clone(),
                        tics: env.cost(),
                        kills: l.0,
                        total_kills: l.1,
                        secrets: l.2,
                        total_secrets: l.3,
                        score: got.value(),
                    };
                    println!(
                        "doom: VERIFIED a UV-Max on {level}: {}/{} kills, {}/{} secrets, \
                         {} tics ({}), {} decisions, replayed from the level's own start",
                        s.kills, s.total_kills, s.secrets, s.total_secrets, s.tics, s.clock(),
                        s.actions.len()
                    );
                    verified_best = s.tics;
                    solved.push(s);
                }
                // A claim the replay does not reproduce is a DEFECT, not a
                // near miss: the search reached that state by restoring
                // snapshots and the action list is supposed to be another way
                // of reaching the same one. Said out loud, because a campaign
                // whose claims do not replay is producing artifacts nobody
                // can use and every other number it reports is suspect.
                Ok(got) => println!(
                    "doom: a claimed UV-Max did NOT replay - from the start it scored \
                     {:.3} rather than 2.000. The search reached that state by restoring \
                     snapshots; the action list does not reach it.",
                    got.value()
                ),
                Err(e) => println!("doom: a claimed UV-Max could not be replayed: {e}"),
            }
            // The replay left the engine at the end of its own episode, and
            // every snapshot the archive points at is still valid - they are
            // the engine's, not the episode's - so the search carries on.
        }

        if said.elapsed() >= every {
            said = Instant::now();
            run.say(&level, began.elapsed(), &solved);
        }
    }

    run.say(&level, began.elapsed(), &solved);
    if let Some(path) = archive {
        run.carry_out(path);
    }
    for spent in run.alloc.report() {
        println!(
            "    {:<16} {:>5} draws, {:>7.1}s, {:.3} gain/s",
            spent.name, spent.draws, spent.seconds, spent.rate
        );
    }
    if run.refused > 0 {
        println!(
            "    {} resumes were REFUSED - that part of the archive is not restorable",
            run.refused
        );
    }
    let best = run.archive.best();
    Ok(Found {
        cells: run.archive.len(),
        best: best.map(|e| e.worth.reached).unwrap_or(0.0),
        best_tics: best.map(|e| e.worth.cost).unwrap_or(0),
        solved,
    })
}
